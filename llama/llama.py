import torch
from torch import nn
from typing import Optional, Tuple
from .llama_config import Config
from .rmsnorm import RMSNorm
from .rope import RotaryEmbedding, apply_rotary_pos_emb


class MLP(nn.Module):
    def __init__(self, config: Config):
        super().__init__()

        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.activation = nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.activation(self.gate_proj(x)) * self.up_proj(x))


class Attention(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_size = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.scale = self.head_size ** -0.5

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_size, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_size, self.hidden_size, bias=False)
        self.dropout = nn.Dropout(p=config.attention_dropout)

    def forward(self,
                x: torch.Tensor,
                mask=None,
                position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        batch, seq_len, _ = x.shape

        # (batch, seq_len, num_heads*head_size)
        query_states = self.q_proj(x)
        # (batch, seq_len, num_key_value_heads*head_size)
        key_states = self.k_proj(x)
        # (batch, seq_len, num_key_value_heads*head_size)
        value_states = self.v_proj(x)

        # query_states (batch, seq_len, num_heads, head_size)
        # key_states (batch, seq_len, num_key_value_heads, head_size)
        # value_states (batch, seq_len, num_key_value_heads, head_size)
        query_states, key_states, value_states = map(
            lambda t: t.reshape(batch, seq_len, -1, self.head_size),
            (query_states, key_states, value_states))

        # query_states (batch, num_heads, seq_len, head_size)
        # key_states (batch, num_key_value_heads, seq_len, head_size)
        # value_states (batch, num_key_value_heads, seq_len, head_size)
        query_states, key_states, value_states = map(
            lambda t: t.permute(0, 2, 1, 3),
            (query_states, key_states, value_states))

        cos, sin = position_embeddings
        # query_states (batch, num_heads, seq_len, head_size)
        # key_states (batch, num_key_value_heads, seq_len, head_size)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if self.num_key_value_heads == 1:
            # key_states value_states (batch, 1, seq_len, head_size)
            pass
        else:
            # (batch, num_key_value_heads, 1, seq_len, head_size)
            key_states, value_states = map(lambda t: t[:, :, None, :, :], (key_states, value_states))
            # (batch, num_key_value_heads, num_key_value_groups=num_heads//num_key_value_heads, seq_len, head_size)
            key_states, value_states = map(
                lambda t: t.expand(batch, self.num_key_value_heads, self.num_key_value_groups, seq_len, self.head_size),
                (key_states, value_states))

            # (batch, num_heads=num_key_value_heads*num_key_value_groups, seq_len, head_size)
            key_states, value_states = map(
                lambda t: t.reshape(batch, self.num_key_value_heads * self.num_key_value_groups, seq_len, self.head_size),
                (key_states, value_states))

        # (batch, num_heads, seq_len, seq_len)
        attn_scores = (self.scale * query_states) @ key_states.transpose(-1, -2)
        if mask is not None:
            attn_scores.masked_fill_(mask == torch.tensor(False), -torch.inf)

        attn_weights = self.dropout(attn_scores.softmax(-1))
        # (batch, num_heads, seq_len, head_size)
        attn = attn_weights @ value_states
        # (batch, seq_len, num_heads, head_size)
        attn = attn.permute(0, 2, 1, 3)
        # (batch, seq_len, num_heads*head_size)
        attn = attn.reshape(batch, seq_len, -1)

        # (batch, seq_len, hidden_size)
        out = self.o_proj(attn)

        return out


class DecoderLayer(nn.Module):
    def __init__(self, config: Config):
        super().__init__()

        self.attn_norm = RMSNorm(config.hidden_size)
        self.attn = Attention(config)

        self.mlp_norm = RMSNorm(config.hidden_size)
        self.mlp = MLP(config)

    def forward(self,
                x: torch.Tensor,
                position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        mask = torch.tril(torch.ones(x.shape[1], x.shape[1]), diagonal=0).to(x.device)
        x = self.attn(self.attn_norm(x), mask=mask, position_embeddings=position_embeddings) + x
        x = self.mlp(self.mlp_norm(x)) + x

        return x


class LlamaModel(nn.Module):
    def __init__(self, config: Config):
        super().__init__()

        self.rotary_emb = RotaryEmbedding(config=config)

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [DecoderLayer(config) for _ in range(config.num_hidden_layers)])

        self.head_norm = RMSNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, x):
        # (batch, seq_len, hidden_size)
        x = self.embed_tokens(x)

        position_ids = torch.arange(0, x.shape[1], device=x.device).unsqueeze(0)
        position_embeddings = self.rotary_emb(x, position_ids)

        for layer in self.layers:
            x = layer(x=x, position_embeddings=position_embeddings)

        x = self.head_norm(x)
        head = self.lm_head(x)

        return head


