import inspect
import torch
from torch import nn
import torch.nn.functional as F
from packaging import version
from .model_config import Config


if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
    _SUPPORT_SDPA = True
    try:
        _sdpa_params = inspect.signature(F.scaled_dot_product_attention).parameters
        _SDPA_SUPPORT_GQA = 'enable_gqa' in _sdpa_params
    except ValueError:
        _SDPA_SUPPORT_GQA = False
else:
    _SUPPORT_SDPA = False
    _SDPA_SUPPORT_GQA = False

_SUPPORT_ATTNS = ('sdpa', 'default')


def sdpa_attention_forward(
        module: nn.Module,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: torch.Tensor,
        num_key_value_groups: int,
        scale: float,
        drop_rate: float
):
    is_gqa = num_key_value_groups > 1
    seq_len = q.shape[-2]
    is_causal = attention_mask is None and seq_len > 1

    with _sdpa_kernel():
        if is_gqa and _SDPA_SUPPORT_GQA:
            attn = F.scaled_dot_product_attention(
                query=q,
                key=k,
                value=v,
                dropout_p=drop_rate if module.training else 0.0,
                attn_mask=attention_mask,
                is_causal=is_causal,
                scale=scale,
                enable_gqa=True
            )
        else:
            if is_gqa:
                k = _repeat_kv(k, num_key_value_groups)
                v = _repeat_kv(v, num_key_value_groups)

            attn = F.scaled_dot_product_attention(
                query=q,
                key=k,
                value=v,
                dropout_p=drop_rate if module.training else 0.0,
                attn_mask=attention_mask,
                is_causal=is_causal,
                scale=scale
            )

    # (batch, num_heads, seq_len, head_size) -> (batch, seq_len, num_heads, head_size)
    attn = attn.transpose(1, 2)
    return attn


def default_attention_forward(
        module: nn.Module,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: torch.Tensor,
        num_key_value_groups: int,
        scale: float,
        drop_rate: float
):
    is_gqa = num_key_value_groups > 1
    if is_gqa:
        k = _repeat_kv(k, num_key_value_groups)
        v = _repeat_kv(v, num_key_value_groups)

    # (batch, num_heads, q_seq_len, k_seq_len)
    attn_scores = (scale * q) @ k.transpose(-1, -2)
    if attention_mask is not None:
        attn_scores = attn_scores + attention_mask

    attn_weights = attn_scores.softmax(-1)
    if module.training:
        attn_weights = F.dropout(attn_weights, p=drop_rate)

    # (batch, num_heads, seq_len, head_size)
    attn = attn_weights @ v
    # (batch, seq_len, num_heads, head_size)
    attn = attn.permute(0, 2, 1, 3)
    return attn


_ATTENTION_INTERFACE = {
    'sdpa': sdpa_attention_forward,
    'default': default_attention_forward
}

def get_attention_interface(config: Config):
    return _ATTENTION_INTERFACE[_get_attn_impl(config)]


def supports_fused_causal_mask(config: Config):
    return _get_attn_impl(config) == 'sdpa'


def _repeat_kv(hidden_states: torch.Tensor, num_key_value_groups: int):
    # (batch, num_key_value_heads, seq_len, head_size) ->
    # (batch, num_key_value_heads, 1, seq_len, head_size) ->
    # (batch, num_key_value_heads, num_key_value_groups=num_heads//num_key_value_heads, seq_len, head_size) ->
    # (batch, num_heads=num_key_value_heads*num_key_value_groups, seq_len, head_size)
    batch, num_key_value_heads, seq_len, head_size = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, num_key_value_groups, seq_len, head_size
    ).reshape(
        batch, num_key_value_heads * num_key_value_groups, seq_len, head_size
    )

    return hidden_states


def _sdpa_kernel(enable_flash: bool = True, enable_math: bool = True, enable_mem_efficient: bool = True):
    if version.parse(torch.__version__).release < version.parse("2.3").release:
        return torch.backends.cuda.sdp_kernel(
            enable_flash=enable_flash, enable_math=enable_math, enable_mem_efficient=enable_mem_efficient
        )

    backends = []
    if enable_flash:
        backends += [torch.nn.attention.SDPBackend.FLASH_ATTENTION]
    if enable_math:
        backends += [torch.nn.attention.SDPBackend.MATH]
    if enable_mem_efficient:
        backends += [torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION]

    return torch.nn.attention.sdpa_kernel(backends)


def _get_attn_impl(config: Config):
    attn_impl = config.attention_implementation
    if attn_impl != 'auto':
        if attn_impl in _SUPPORT_ATTNS:
            return attn_impl

    return 'sdpa' if _SUPPORT_SDPA else 'default'