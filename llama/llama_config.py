from dataclasses import dataclass


@dataclass
class Config:
    vocab_size: int = 21128
    hidden_size: int = 4096
    intermediate_size: int = 11008
    num_attention_heads: int = 32
    num_key_value_heads: int = num_attention_heads
    max_position_embeddings: int = 2048
    rope_theta: float = 10000.0
    attention_dropout: float = 0.1
    num_hidden_layers: int = 32
