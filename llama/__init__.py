from . import llama
from . import llama_config

LlamaModel = llama.LlamaModel
LlamaConfig = llama_config.Config

__all__ = [
    'LlamaModel',
    'LlamaConfig'
]