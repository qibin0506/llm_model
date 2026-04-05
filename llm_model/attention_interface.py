import torch
import inspect
import torch.nn.functional as F
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


def get_attn_impl(config: Config):
    attn_impl = config.attention_implementation
    if attn_impl != 'auto':
        if attn_impl in _SUPPORT_ATTNS:
            return attn_impl

    return 'sdpa' if _SUPPORT_SDPA else 'default'


def support_flash_causal(attn_impl):
    return attn_impl == 'sdpa'
