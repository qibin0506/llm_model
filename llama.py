import torch
from torch import nn
from typing import Optional, Tuple
from .llama_config import Config
from .rmsnorm import RMSNorm
from .rope import RotaryEmbedding, apply_rotary_pos_emb
from .kv_cache import KVCache
from .attention_masks import prepare_decoder_attention_mask


class MLP(nn.Module):
    def __init__(self, config: Config):
        super().__init__()

        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.activation = nn.SiLU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.activation(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class Attention(nn.Module):
    def __init__(self, config: Config, layer_idx: int):
        super().__init__()
        assert config.num_attention_heads % config.num_key_value_heads == 0

        self.layer_idx = layer_idx
        
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
                hidden_states: torch.Tensor,
                position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                attention_mask: Optional[torch.Tensor] = None,
                past_key_values: Optional[KVCache] = None) -> torch.Tensor:
        batch, seq_len, _ = hidden_states.shape

        # (batch, seq_len, num_heads*head_size)
        query_states = self.q_proj(hidden_states)
        # (batch, seq_len, num_key_value_heads*head_size)
        key_states = self.k_proj(hidden_states)
        # (batch, seq_len, num_key_value_heads*head_size)
        value_states = self.v_proj(hidden_states)

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

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        if self.num_key_value_heads == 1:
            # key_states value_states (batch, 1, seq_len, head_size)
            pass
        else:
            # (batch, num_key_value_heads, 1, seq_len, head_size)
            key_states, value_states = map(lambda t: t[:, :, None, :, :], (key_states, value_states))
            # (batch, num_key_value_heads, num_key_value_groups=num_heads//num_key_value_heads, seq_len, head_size)
            key_states, value_states = map(
                lambda t: t.expand(
                    batch, self.num_key_value_heads, self.num_key_value_groups, t.shape[-2], self.head_size),
                (key_states, value_states))

            # (batch, num_heads=num_key_value_heads*num_key_value_groups, seq_len, head_size)
            key_states, value_states = map(
                lambda t: t.reshape(
                    batch, self.num_key_value_heads * self.num_key_value_groups, t.shape[-2], self.head_size),
                (key_states, value_states))

        # (batch, num_heads, q_seq_len, k_seq_len)
        attn_scores = (self.scale * query_states) @ key_states.transpose(-1, -2)
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask

        attn_weights = self.dropout(attn_scores.softmax(-1, dtype=torch.float32))
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
    def __init__(self, config: Config, layer_idx: int):
        super().__init__()

        self.attn_norm = RMSNorm(config.hidden_size)
        self.attn = Attention(config, layer_idx)

        self.mlp_norm = RMSNorm(config.hidden_size)
        self.mlp = MLP(config)

    def forward(self,
                hidden_states: torch.Tensor,
                position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                attention_mask: Optional[torch.Tensor] = None,
                past_key_values: Optional[KVCache] = None) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.attn(
                      hidden_states=self.attn_norm(hidden_states),
                      position_embeddings=position_embeddings,
                      attention_mask=attention_mask,
                      past_key_values=past_key_values
        )

        hidden_states = hidden_states + residual
        hidden_states = self.mlp(self.mlp_norm(hidden_states)) + hidden_states

        return hidden_states


class LlamaModel(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.rotary_emb = RotaryEmbedding(config=config)

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [DecoderLayer(config, idx) for idx in range(config.num_hidden_layers)])

        self.head_norm = RMSNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                past_key_values: Optional[KVCache] = None,
                use_cache: bool = False) -> Tuple[torch.Tensor, Optional[KVCache]]:
        """
        :param input_ids: input tokens
        :param attention_mask: input mask for ignore padding, shape is (batch, seq_len)
            eg: [[true, true, true, false], [true, true, true, true]]
        :param past_key_values:
        :param use_cache:
        :return:
        """
        batch_size, seq_len = input_ids.shape

        if use_cache and past_key_values is None:
            past_key_values = KVCache()

        # (batch, seq_len, hidden_size)
        # for inference with past_key_values inputs_embeds.shape is (1, 1, hidden_size)
        inputs_embeds = self.embed_tokens(input_ids)

        # seq_len
        past_seen_tokens = past_key_values.get_seq_len() if past_key_values is not None else 0
        full_seq_len = past_seen_tokens + seq_len

        position_ids = torch.arange(past_seen_tokens, full_seq_len, device=inputs_embeds.device).unsqueeze(0)
        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)

        if attention_mask is None:
            # (batch_size, past_seen_tokens+seq_len)
            attention_mask = torch.ones(
                (batch_size, full_seq_len), dtype=torch.bool, device=inputs_embeds.device
            )

        # (batch_size, 1, seq_len, past_seen_tokens+seq_len)
        attention_mask = prepare_decoder_attention_mask(
            attention_mask=attention_mask,
            input_shape=(batch_size, seq_len),
            past_key_values_length=past_seen_tokens,
            dtype=inputs_embeds.dtype,
            device=inputs_embeds.device
        )

        hidden_states = inputs_embeds
        for layer in self.layers:
            hidden_states = layer(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                past_key_values=past_key_values)

        hidden_states = self.head_norm(hidden_states)
        head = self.lm_head(hidden_states)

        return head, past_key_values


