from typing import List


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


class Config:
    """
        llama config

        Args:
            vocab_size (`int`, *optional*, default is 21128):
                the vocab size
            hidden_size (`int`, *optional*, default is 4096):
                the hidden size
            intermediate_size (`int`, *optional*, default is 11008):
                the intermediate_size
            num_attention_heads (`int`, *optional*, default is 32):
                the attention head count
            num_key_value_heads (`int`, *optional*, default is num_attention_heads):
                key value heads count,
                if num_key_value_heads=num_attention_heads, will use Multi Head Attention (MHA),
                if num_key_value_heads=1, will use Multi Query Attention (MQA),
                else will use Group Query Attention (GQA)
            max_position_embeddings (`int`, *optional*, default is 2048):
                max position embeddings
            rope_config (`RoPEConfig`)
                RoPE configurations
            attention_dropout (`float`, *optional*, default is 0.1)
                dropout for attention
            num_hidden_layers (`int`, *optional*, default is 32)
                decoder layers count
            num_experts (`int`, *optional*, default is 0)
                number of moe experts, 0 means without moe
            slots_per_expert (`int`, *optional*, default is 1)
                number of token slots per expert
        """

    def __init__(
            self,
            *,
            vocab_size: int = 21128,
            hidden_size: int = 4096,
            intermediate_size: int = 11008,
            num_hidden_layers: int = 32,
            num_attention_heads: int = 32,
            num_key_value_heads: int = 32,
            max_position_embeddings: int = 2048,
            rope_config: RoPEConfig = RoPEConfig(),
            attention_dropout: float = 0.1,
            num_experts: int = 0,
            slots_per_expert: int = 1,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.max_position_embeddings = max_position_embeddings
        self.rope_config = rope_config
        self.attention_dropout = attention_dropout
        self.num_experts = num_experts
        self.slots_per_expert = slots_per_expert
