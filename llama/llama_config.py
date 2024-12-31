from dataclasses import dataclass


@dataclass
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
        rope_theta (`float`, *optional*, default is 10000.0)
            the rope theta args
        attention_dropout (`float`, *optional*, default is 0.1)
            dropout for attention
        num_hidden_layers (`int`, *optional*, default is 32)
            decoder layers count
        num_experts (`int`, *optional*, default is 0)
            number of moe experts, 0 means without moe
        slots_per_expert (`int`, *optional*, default is 1)
            number of token slots per expert
    """
    vocab_size: int = 21128
    hidden_size: int = 4096
    intermediate_size: int = 11008
    num_attention_heads: int = 32
    num_key_value_heads: int = 32
    max_position_embeddings: int = 2048
    rope_theta: float = 10000.0
    attention_dropout: float = 0.1
    num_hidden_layers: int = 32
    num_experts: int = 0
    slots_per_expert: int = 1

