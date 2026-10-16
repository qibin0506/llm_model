import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from .model_config import Config
from .norm import RMSNorm

try:
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule

    _SUPPORT_FLA = True
except (ImportError, ModuleNotFoundError):
    _SUPPORT_FLA = False

_SUPPORT_IMPL = ('fla', 'default')


class ShortConvolution(nn.Module):
    def __init__(self, hidden_size: int, kernel_size: int = 4, bias: bool = False):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(
            in_channels=hidden_size, out_channels=hidden_size,
            kernel_size=kernel_size, groups=hidden_size, padding=0, bias=bias
        )
        self._smooth_initialization()

    def _smooth_initialization(self):
        if hasattr(self.conv.weight, "ds_id"):
            import deepspeed
            params = [self.conv.weight] + ([self.conv.bias] if self.conv.bias is not None else [])
            with deepspeed.zero.GatheredParameters(params, modifier_rank=0):
                if self.conv.bias is not None:
                    nn.init.zeros_(self.conv.bias)
                nn.init.normal_(self.conv.weight, mean=0.0, std=0.02)
                with torch.no_grad():
                    if self.conv.weight.ndim == 3:
                        self.conv.weight[:, 0, -1] += 1.0
        else:
            if self.conv.bias is not None:
                nn.init.zeros_(self.conv.bias)
            nn.init.normal_(self.conv.weight, mean=0.0, std=0.02)
            with torch.no_grad():
                self.conv.weight[:, 0, -1] += 1.0

    def forward(self, x: torch.Tensor, conv_state: Optional[torch.Tensor] = None):
        x_t = x.transpose(1, 2)
        if conv_state is not None:
            x_cat = torch.cat([conv_state, x_t], dim=-1)
            new_conv_state = x_cat[:, :, -(self.kernel_size - 1):]
            out = self.conv(x_cat)
            out = out.transpose(1, 2)
            return out, new_conv_state
        else:
            x_padded = F.pad(x_t, (self.kernel_size - 1, 0))
            out = self.conv(x_padded).transpose(1, 2)
            new_conv_state = x_padded[:, :, -(self.kernel_size - 1):]
            return out, new_conv_state


def l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    """与 FLA 库中的 l2norm 实现对齐: x * rsqrt(sum(x^2) + eps)"""
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x * inv_norm


def torch_chunk_gated_delta_rule(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        scale: float,
        chunk_size: int = 64,
        initial_state: Optional[torch.Tensor] = None,
        output_final_state: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    分块 (chunk) 形式的 Gated Delta Rule 纯 PyTorch 实现
    全程使用 float32 计算，输出转回输入 dtype，状态保持 float32。

    Args:
        q, k: (b, t, h, dk)，需已做 l2norm
        v: (b, t, h, dv)
        g: (b, t, h)，log 空间衰减 (<= 0)
        beta: (b, t, h)
        scale: q 的缩放系数
    Returns:
        out: (b, t, h, dv)
        final_state: (b, h, dk, dv) 或 None
    """
    initial_dtype = q.dtype
    q, k, v, beta, g = [x.transpose(1, 2).contiguous().to(torch.float32) for x in (q, k, v, beta, g)]

    batch_size, num_heads, seq_len, k_head_dim = k.shape
    v_head_dim = v.shape[-1]
    pad_size = (chunk_size - seq_len % chunk_size) % chunk_size
    q = F.pad(q, (0, 0, 0, pad_size))
    k = F.pad(k, (0, 0, 0, pad_size))
    v = F.pad(v, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))
    total_seq_len = seq_len + pad_size

    q = q * scale
    v_beta = v * beta.unsqueeze(-1)
    k_beta = k * beta.unsqueeze(-1)

    # (b, h, n_chunks, chunk_size, d)
    q, k, v, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1]) for x in (q, k, v, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), diagonal=0)

    # chunk 内累积衰减
    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
    attn = -((k_beta @ k.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
    # 前代求解 (I + A)^-1 (下三角)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    v = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))

    last_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim, dtype=v.dtype, device=v.device)
        if initial_state is None
        else initial_state.to(v)
    )
    out = torch.zeros_like(v)

    for i in range(0, total_seq_len // chunk_size):
        q_i, k_i, v_i = q[:, :, i], k[:, :, i], v[:, :, i]
        # decay_mask 已经是下三角，上三角为 0
        intra_attn = q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]
        v_prime = k_cumdecay[:, :, i] @ last_state
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_state
        out[:, :, i] = attn_inter + intra_attn @ v_new
        last_state = (
            last_state * g[:, :, i, -1, None, None].exp()
            + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
        )

    if not output_final_state:
        last_state = None
    out = out.reshape(out.shape[0], out.shape[1], -1, out.shape[-1])[:, :, :seq_len]
    out = out.transpose(1, 2).contiguous().to(initial_dtype)
    return out, last_state


def torch_recurrent_gated_delta_rule(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        scale: float,
        initial_state: Optional[torch.Tensor] = None,
        output_final_state: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    逐 token 递推形式的 Gated Delta Rule 纯 PyTorch 实现 (适用于 decode 阶段 t 很小的情况)。
    参数与返回值同 `torch_chunk_gated_delta_rule`。
    """
    initial_dtype = q.dtype
    q, k, v, beta, g = [x.transpose(1, 2).contiguous().to(torch.float32) for x in (q, k, v, beta, g)]

    batch_size, num_heads, seq_len, k_head_dim = k.shape
    v_head_dim = v.shape[-1]
    q = q * scale

    out = torch.zeros(batch_size, num_heads, seq_len, v_head_dim, dtype=v.dtype, device=v.device)
    last_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim, dtype=v.dtype, device=v.device)
        if initial_state is None
        else initial_state.to(v)
    )

    for i in range(seq_len):
        q_t, k_t, v_t = q[:, :, i], k[:, :, i], v[:, :, i]
        g_t = g[:, :, i].exp().unsqueeze(-1).unsqueeze(-1)
        beta_t = beta[:, :, i].unsqueeze(-1)

        last_state = last_state * g_t
        kv_mem = (last_state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - kv_mem) * beta_t
        last_state = last_state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        out[:, :, i] = (last_state * q_t.unsqueeze(-1)).sum(dim=-2)

    if not output_final_state:
        last_state = None
    out = out.transpose(1, 2).contiguous().to(initial_dtype)
    return out, last_state


class GatedDeltaNet(nn.Module):
    def __init__(self, config: Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim or (self.hidden_size // self.num_heads)

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.beta_proj = nn.Linear(self.hidden_size, self.num_heads, bias=False)
        self.alpha_proj = nn.Linear(self.hidden_size, self.num_heads, bias=False)
        self.g_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)

        # dt_bias 初始化为 1，A ~ U(0.01, 16)，使不同 head 拥有不同的衰减速率 (多尺度记忆)
        # A 的下界远离 0，保证 log(A) 不会变成 -inf
        self.dt_bias = nn.Parameter(torch.ones(self.num_heads))
        A = torch.empty(self.num_heads).uniform_(0.01, 16)
        self.A_log = nn.Parameter(torch.log(A))

        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self.g_norm = RMSNorm(self.head_dim, config.norm_eps)

        self.use_short_conv = (config.gated_deltanet_config is not None and config.gated_deltanet_config.use_short_conv)

        if self.use_short_conv:
            kernel_size = config.gated_deltanet_config.conv_kernel_size
            conv_bias = config.gated_deltanet_config.conv_bias
            self.conv_q = ShortConvolution(self.num_heads * self.head_dim, kernel_size, conv_bias)
            self.conv_k = ShortConvolution(self.num_heads * self.head_dim, kernel_size, conv_bias)
            self.conv_v = ShortConvolution(self.num_heads * self.head_dim, kernel_size, conv_bias)
        else:
            self.conv_q = self.conv_k = self.conv_v = None

    def forward(
            self, hidden_states: torch.Tensor, padding_mask: Optional[torch.Tensor] = None,
            state_cache: Optional[dict] = None, use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[dict]]:
        b, t, _ = hidden_states.shape

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        g = self.g_proj(hidden_states)

        # beta 与衰减 g_decay 在 float32 下计算，避免 fp16/bf16 下 exp(A_log) / softplus 精度不足
        beta = torch.sigmoid(self.beta_proj(hidden_states).float())
        alpha_logits = self.alpha_proj(hidden_states).float()
        g_decay = -self.A_log.float().exp() * F.softplus(alpha_logits + self.dt_bias.float())  # (b, t, h)

        if padding_mask is not None:
            mask_t = padding_mask[:, -t:]
            mask_d = mask_t.unsqueeze(-1).to(q.dtype)
            q = q * mask_d
            k = k * mask_d
            v = v * mask_d

        if self.use_short_conv:
            cq_state = state_cache.get('conv_q') if state_cache is not None else None
            ck_state = state_cache.get('conv_k') if state_cache is not None else None
            cv_state = state_cache.get('conv_v') if state_cache is not None else None

            q, new_cq_state = self.conv_q(q, cq_state)
            k, new_ck_state = self.conv_k(k, ck_state)
            v, new_cv_state = self.conv_v(v, cv_state)
        else:
            new_cq_state = new_ck_state = new_cv_state = None

        q = F.silu(q).reshape(b, t, self.num_heads, self.head_dim)
        k = F.silu(k).reshape(b, t, self.num_heads, self.head_dim)
        v = F.silu(v).reshape(b, t, self.num_heads, self.head_dim)
        g = g.reshape(b, t, self.num_heads, self.head_dim)

        if padding_mask is not None:
            mask_h = mask_t.unsqueeze(-1).to(torch.float32)  # (b, t, 1)
            mask_hd = mask_t.unsqueeze(-1).unsqueeze(-1).to(q.dtype)  # (b, t, 1, 1)

            q = q * mask_hd
            k = k * mask_hd
            v = v * mask_hd
            g = g * mask_hd
            # padding 位置: beta=0 不写入状态，g_decay=0 (exp(0)=1) 不衰减状态
            beta = beta * mask_h
            g_decay = g_decay * mask_h

        q = l2norm(q.float(), dim=-1, eps=1e-6).to(v.dtype)
        k = l2norm(k.float(), dim=-1, eps=1e-6).to(v.dtype)

        # 缩放统一通过 kernel 的 scale 参数传入，避免与 FLA 内部默认的 K**-0.5 重复缩放
        scale = self.head_dim ** -0.5
        initial_state = state_cache.get('recurrent_state') if state_cache is not None else None
        output_final_state = use_cache or state_cache is not None

        gd_impl = _get_gated_deltanet_impl(self.config, q.device)
        if gd_impl == 'fla':
            fla_kernel = chunk_gated_delta_rule if t > 1 else fused_recurrent_gated_delta_rule
            out, final_state = fla_kernel(
                q=q, k=k, v=v, g=g_decay, beta=beta.to(q.dtype), scale=scale,
                initial_state=initial_state, output_final_state=output_final_state
            )
        else:
            torch_kernel = torch_chunk_gated_delta_rule if t > 1 else torch_recurrent_gated_delta_rule
            out, final_state = torch_kernel(
                q=q, k=k, v=v, g=g_decay, beta=beta, scale=scale,
                initial_state=initial_state, output_final_state=output_final_state
            )

        out = self.g_norm(out) * F.silu(g)
        out = out.reshape(b, t, -1)
        out = self.o_proj(out)

        new_cache = {
            'recurrent_state': final_state, 'conv_q': new_cq_state, 'conv_k': new_ck_state, 'conv_v': new_cv_state,
        } if output_final_state else None

        return out, new_cache


def _get_gated_deltanet_impl(config: Config, device: Optional[torch.device] = None) -> str:
    """
    选择 Gated DeltaNet 算子实现。FLA 基于 Triton，仅支持 CUDA 张量；
    auto 模式下若张量不在 CUDA 上 (如 CPU / MPS)，自动回退到纯 PyTorch 实现。
    """
    impl = config.gated_deltanet_implementation
    on_cuda = device is None or device.type == 'cuda'

    if impl == 'fla':
        if not _SUPPORT_FLA:
            raise RuntimeError('Please install fla')
        if not on_cuda:
            raise RuntimeError(f'fla kernels require CUDA tensors, but got device={device}')
        return 'fla'

    if impl == 'default':
        return 'default'

    return 'fla' if (_SUPPORT_FLA and on_cuda) else 'default'