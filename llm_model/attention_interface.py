import torch
import inspect
import torch.nn.functional as F
from .model_config import Config

if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
    _SUPPORT_SDPA = True
else:
    _SUPPORT_SDPA = False

try:
    from torch.nn.attention.flex_attention import flex_attention
    _SUPPORT_FLEX = True
except ImportError:
    _SUPPORT_FLEX = False

try:
    _sdpa_params = inspect.signature(F.scaled_dot_product_attention).parameters
    _SDPA_SUPPORT_GQA = 'enable_gqa' in _sdpa_params
except ValueError:
    _SDPA_SUPPORT_GQA = False

_SUPPORT_ATTNS = ('sdpa', 'flex', 'default')


def get_attn_impl(config: Config):
    attn_impl = config.attention_implementation
    if attn_impl != 'auto':
        if attn_impl in _SUPPORT_ATTNS:
            return attn_impl

    if _SUPPORT_SDPA:
        return 'sdpa'

    return 'flex' if _SUPPORT_FLEX else 'default'


def support_flash_causal(attn_impl):
    return attn_impl == 'sdpa' or attn_impl == 'flex'
