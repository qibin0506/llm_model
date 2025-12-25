from typing import List, Optional, Callable
from dataclasses import dataclass, field
import torch


@dataclass(kw_only=True)
class RoPEConfig:
    """
    RoPE config

    Args:
        rope_type (`str`, default is default):
            the rope type, support `default`, `dynamic`, `yarn`,
        rope_theta (`float`, default is 10000.0)
            the rope theta args
        factor (`float`):
            Used with all rope types except 'default'. The scaling factor to apply to the RoPE embeddings. In
            most scaling types, a `factor` of x will enable the model to handle sequences of length x *
            original maximum pre-trained length.
        attention_factor (`float`):
            Used with 'yarn'. The scaling factor to be applied on the attention
            computation. If unspecified, it defaults to value recommended by the implementation, using the
            `factor` field to infer the suggested value.
        beta_fast (`float`):
            Only used with 'yarn'. Parameter to set the boundary for extrapolation (only) in the linear
            ramp function. If unspecified, it defaults to 32.
        beta_slow (`float`):
            Only used with 'yarn'. Parameter to set the boundary for interpolation (only) in the linear
            ramp function. If unspecified, it defaults to 1.
        mscale (`float`):
            Only used with 'yarn'. Default is None.
        mscale_all_dim (`float`):
            Only used with 'yarn'. Default is None.
    """
    rope_type: str = 'default'
    rope_theta: float = 10000.0
    factor: float = 1.0
    partial_rotary_factor: float = 1.0
    beta_fast: float = 32
    beta_slow: float = 1
    mscale: Optional[float] = None
    mscale_all_dim: Optional[float] = None
    attention_factor: Optional[float] = None


@dataclass(kw_only=True)
class MoEConfig:
    """
    MoE Config
    Args:
        intermediate_size (`int`, *optional*, defaults to None):
            Dimension of the MoE representations.
        n_dense_layer (`int`, default to None)
            n layers use dense for moe
        num_experts_per_tok (`int`, *optional*, defaults to None):
            Number of selected experts, None means dense model.
        n_shared_experts (`int`, *optional*, defaults to None):
            Number of shared experts, None means dense model.
        n_routed_experts (`int`, *optional*, defaults to None):
            Number of routed experts, None means dense model.
        routed_scaling_factor (`float`, *optional*, defaults to 1.0):
            Scaling factor or routed experts.
        seq_aux (`bool`, *optional*, defaults to True):
            Whether to compute the auxiliary loss for each individual sample.
        norm_topk_prob (`bool`, *optional*, defaults to False):
            Whether to normalize the weights of the routed experts.
        z_loss_coef (`float`)
            moe z-loss coef
    """
    intermediate_size: Optional[int] = None
    n_dense_layer: Optional[int] = None
    num_experts_per_tok: Optional[int] = None
    n_shared_experts: Optional[int] = None
    n_routed_experts: Optional[int] = None
    routed_scaling_factor: float = 1.0
    seq_aux: bool = True
    norm_topk_prob: bool = False
    z_loss_coef: float = 1e-4


@dataclass(kw_only=True)
class Config:
    """
    llm model config

    Args:
        vocab_size (`int`, *optional*, default is 21128):
            the vocab size
        hidden_size (`int`, *optional*, default is 4096):
            the hidden size
        intermediate_size (`int`, *optional*, default is 11008):
            the intermediate_size
        num_hidden_layers (`int`, *optional*, default is 32)
            decoder layers count
        num_attention_heads (`int`, *optional*, default is 32):
            the attention head count
        num_key_value_heads (`int`, *optional*, default is num_attention_heads):
            key value heads count,
            if num_key_value_heads=num_attention_heads, will use Multi Head Attention (MHA),
            if num_key_value_heads=1, will use Multi Query Attention (MQA),
            else will use Group Query Attention (GQA)
        max_position_embeddings (`int`, *optional*, default is 2048):
            max position embeddings
        attention_dropout (`float`, *optional*, default is 0.1)
            dropout for attention
        attention_implementation (`str`, default is auto)
            if attention_implementation='auto' use F.scaled_dot_product_attention first
            if attention_implementation='sdpa' will use F.scaled_dot_product_attention
            if attention_implementation='default' will use pure implementation
        rope_config (`RoPEConfig`)
            RoPE configurations
        moe_config (`MoEConfig`)
            MoE config
        use_qk_norm (`bool`)
            add qk norm
        tie_word_embeddings (`bool`)
            tie word embeddings
    """
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    max_position_embeddings: int
    original_max_position_embeddings: Optional[int] = None
    attention_dropout: float = 0.1
    attention_implementation: str = 'auto'
    use_qk_norm: bool = True
    tie_word_embeddings: bool = False
    rope_config: RoPEConfig = field(default_factory=RoPEConfig)
    moe_config: Optional[MoEConfig] = None

    def __post_init__(self):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads


@dataclass(kw_only=True)
class VLMConfig(Config):
    """
    vlm config
    """
    image_tok: int
    image_size: int
    patch_size: int
    tokens_per_image: int
    vision_hidden_size: int
    vision_tower: Callable[[torch.Tensor], torch.Tensor]

