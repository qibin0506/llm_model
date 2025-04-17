from typing import List, Optional, Callable
import torch


class RoPEConfig:
    """
    RoPE config

    Args:
        rope_type (`str`, default is default):
            the rope type, support `default`, `linear`, `dynamic`, `yarn`, `longrope`, `llama3`
        rope_theta (`float`, default is 10000.0)
            the rope theta args
        factor (`float`):
            Used with all rope types except 'default'. The scaling factor to apply to the RoPE embeddings. In
            most scaling types, a `factor` of x will enable the model to handle sequences of length x *
            original maximum pre-trained length.
        attention_factor (`float`):
            Used with 'yarn' and 'longrope'. The scaling factor to be applied on the attention
            computation. If unspecified, it defaults to value recommended by the implementation, using the
            `factor` field to infer the suggested value.
        beta_fast (`float`):
            Only used with 'yarn'. Parameter to set the boundary for extrapolation (only) in the linear
            ramp function. If unspecified, it defaults to 32.
        beta_slow (`float`):
            Only used with 'yarn'. Parameter to set the boundary for interpolation (only) in the linear
            ramp function. If unspecified, it defaults to 1.
        long_factor (`List[float]`):
            Only used with 'longrope'. The scaling factor to be applied to long contexts (<
            `original_max_position_embeddings`). Must be a list of numbers with the same length as the hidden
            size divided by the number of attention heads divided by 2
        short_factor (`List[float]`):
            Only used with 'longrope'. The scaling factor to be applied to short contexts (<
            `original_max_position_embeddings`). Must be a list of numbers with the same length as the hidden
            size divided by the number of attention heads divided by 2
        low_freq_factor (`float`):
            Only used with 'llama3'. Scaling factor applied to low frequency components of the RoPE
        high_freq_factor (`float`):
            Only used with 'llama3'. Scaling factor applied to high frequency components of the RoPE
    """

    def __init__(
            self,
            *,
            rope_type: str = 'default',
            rope_theta: float = 10000.0,
            factor: float = 1.0,
            attention_factor: float = None,
            beta_fast: float = 32,
            beta_slow: float = 1,
            long_factor: List[float] = None,
            short_factor: List[float] = None,
            low_freq_factor: float = None,
            high_freq_factor: float = None,
    ):
        self.rope_type = rope_type
        self.rope_theta = rope_theta
        self.factor = factor
        self.attention_factor = attention_factor
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        self.long_factor = long_factor
        self.short_factor = short_factor
        self.low_freq_factor = low_freq_factor
        self.high_freq_factor = high_freq_factor


class MoEConfig:
    """
        MoE Config
        Args:
            num_experts_per_tok (`int`, *optional*, defaults to None):
                Number of selected experts, None means dense model.
            n_routed_experts (`int`, *optional*, defaults to None):
                Number of routed experts, None means dense model.
            n_shared_experts (`int`, *optional*, defaults to None):
                Number of shared experts, None means dense model.
            scoring_func (`str`, *optional*, defaults to 'softmax'):
                Method of computing expert weights.
            aux_loss_alpha (`float`, *optional*, defaults to 0.001):
                Auxiliary loss weight coefficient.
            seq_aux = (`bool`, *optional*, defaults to True):
                Whether to compute the auxiliary loss for each individual sample.
            norm_topk_prob (`bool`, *optional*, defaults to False):
                Whether to normalize the weights of the routed experts.
    """
    def __init__(
            self,
            num_experts_per_tok: Optional[int] = None,
            n_routed_experts: Optional[int] = None,
            n_shared_experts: Optional[int] = None,
            scoring_func: str = 'softmax',
            aux_loss_alpha: float = 0.001,
            seq_aux: bool = True,
            norm_topk_prob: bool = False,
    ):
        self.num_experts_per_tok = num_experts_per_tok
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.scoring_func = scoring_func
        self.aux_loss_alpha = aux_loss_alpha
        self.seq_aux = seq_aux
        self.norm_topk_prob = norm_topk_prob


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
            moe_intermediate_size (`int`, *optional*, defaults to 1407):
                Dimension of the MoE representations.
            moe_n_dense_layer (`int`, default is 1)
                n layers use dense for moe
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
            rope_config (`RoPEConfig`)
                RoPE configurations
            attention_implementation (`str`, default is auto)
                if attention_implementation='auto' use F.scaled_dot_product_attention first
                if attention_implementation='sdpa' will use F.scaled_dot_product_attention
                if attention_implementation='default' will use pure implementation
        """

    def __init__(
            self,
            *,
            vocab_size: int = 21128,
            hidden_size: int = 4096,
            intermediate_size: int = 11008,
            moe_intermediate_size: int = 1407,
            moe_n_dense_layer = 1,
            num_hidden_layers: int = 32,
            num_attention_heads: int = 32,
            num_key_value_heads: int = 32,
            max_position_embeddings: int = 2048,
            attention_dropout: float = 0.1,
            attention_implementation = 'auto',
            rope_config: RoPEConfig = RoPEConfig(),
            moe_config: Optional[MoEConfig] = None
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.moe_intermediate_size = moe_intermediate_size
        self.moe_n_dense_layer = moe_n_dense_layer
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.max_position_embeddings = max_position_embeddings
        self.attention_dropout = attention_dropout
        self.attention_implementation = attention_implementation
        self.rope_config = rope_config
        self.moe_config = moe_config


class VLMConfig(Config):
    def __init__(
            self,
            *,
            image_tok: int,
            image_size: int,
            patch_size: int,
            tokens_per_image: int,
            vision_hidden_size: int,
            vision_tower: Callable[[torch.Tensor], torch.Tensor],
            vocab_size: int = 21128,
            hidden_size: int = 4096,
            intermediate_size: int = 11008,
            moe_intermediate_size: int = 1407,
            moe_n_dense_layer=1,
            num_hidden_layers: int = 32,
            num_attention_heads: int = 32,
            num_key_value_heads: int = 32,
            max_position_embeddings: int = 2048,
            attention_dropout: float = 0.1,
            attention_implementation='auto',
            rope_config: RoPEConfig = RoPEConfig(),
            moe_config: Optional[MoEConfig] = None
    ):
        super().__init__(
            vocab_size = vocab_size,
            hidden_size = hidden_size,
            intermediate_size = intermediate_size,
            moe_intermediate_size = moe_intermediate_size,
            moe_n_dense_layer = moe_n_dense_layer,
            num_hidden_layers = num_hidden_layers,
            num_attention_heads = num_attention_heads,
            num_key_value_heads = num_key_value_heads,
            max_position_embeddings = max_position_embeddings,
            attention_dropout = attention_dropout,
            attention_implementation = attention_implementation,
            rope_config = rope_config,
            moe_config = moe_config
        )

        self.image_tok = image_tok
        self.image_size = image_size
        self.patch_size = patch_size
        self.tokens_per_image = tokens_per_image
        self.vision_hidden_size = vision_hidden_size
        self.vision_tower = vision_tower

