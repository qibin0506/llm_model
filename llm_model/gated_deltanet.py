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

        self.dt_bias = nn.Parameter(torch.zeros(self.num_heads))
        self.A_log = nn.Parameter(torch.zeros(self.num_heads))

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

        beta = torch.sigmoid(self.beta_proj(hidden_states))
        alpha_logits = self.alpha_proj(hidden_states)

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

        dt = F.softplus(alpha_logits + self.dt_bias)
        g_decay = -dt * torch.exp(self.A_log)
        g_decay = g_decay.unsqueeze(-1)  # (b, t, h, 1)

        if padding_mask is not None:
            mask_h = mask_t.unsqueeze(-1).to(q.dtype)  # (b, t, 1)
            mask_hd = mask_t.unsqueeze(-1).unsqueeze(-1).to(q.dtype)  # (b, t, 1, 1)

            q = q * mask_hd
            k = k * mask_hd
            v = v * mask_hd
            g = g * mask_hd
            beta = beta * mask_h

            g_decay = g_decay * mask_hd

        q = F.normalize(q, p=2, dim=-1)
        k = F.normalize(k, p=2, dim=-1)

        q = q * (self.head_dim ** -0.5)

        initial_state = state_cache.get('recurrent_state') if state_cache is not None else None

        gd_impl = _get_gated_deltanet_impl(self.config)
        if gd_impl == 'fla':
            if t > 1:
                out, final_state = chunk_gated_delta_rule(
                    q=q, k=k, v=v, g=g_decay.squeeze(-1), beta=beta.squeeze(-1),
                    initial_state=initial_state, output_final_state=True
                )
            else:
                out, final_state = fused_recurrent_gated_delta_rule(
                    q=q, k=k, v=v, g=g_decay.squeeze(-1), beta=beta.squeeze(-1),
                    initial_state=initial_state, output_final_state=True
                )
        else:
            alpha = torch.exp(g_decay)
            if initial_state is None:
                curr_state = torch.zeros(b, self.num_heads, self.head_dim, self.head_dim, device=q.device,
                                         dtype=q.dtype)
            else:
                curr_state = initial_state

            out_list = []
            for i in range(t):
                q_i, k_i, v_i = q[:, i], k[:, i], v[:, i]
                a_i = alpha[:, i].unsqueeze(-1)  # (b, h, 1, 1)
                b_i = beta[:, i].unsqueeze(-1)  # (b, h, 1)

                curr_state = curr_state * a_i
                v_pred = torch.einsum('bhk, bhkv -> bhv', k_i, curr_state)
                v_err = v_i - v_pred
                delta_s = torch.einsum('bhk, bhv -> bhkv', k_i, b_i * v_err)
                curr_state = curr_state + delta_s
                o_i = torch.einsum('bhk, bhkv -> bhv', q_i, curr_state)
                out_list.append(o_i)

            out = torch.stack(out_list, dim=1)
            final_state = curr_state

        out = self.g_norm(out) * F.silu(g)
        out = out.reshape(b, t, -1)
        out = self.o_proj(out)

        new_cache = {
            'recurrent_state': final_state, 'conv_q': new_cq_state, 'conv_k': new_ck_state, 'conv_v': new_cv_state,
        } if (use_cache or state_cache is not None) else None

        return out, new_cache


def _get_gated_deltanet_impl(config: Config) -> str:
    impl = config.gated_deltanet_implementation
    if impl == 'fla' and not _SUPPORT_FLA:
        raise RuntimeError('Please install fla')
    if impl != 'auto':
        if impl in _SUPPORT_IMPL:
            return impl
    return 'fla' if _SUPPORT_FLA else 'default'