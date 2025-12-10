from typing import Optional, Tuple, Dict
from packaging import version
import inspect

import torch
from torch import nn
import torch.nn.functional as F

from .model_config import Config
from .rope import ROPE_INIT_FUNCTIONS, apply_rotary_pos_emb
from .kv_cache import KVCache
from .attention_masks import prepare_decoder_attention_mask
from .sparse_moe import MoE

try:
    _sdpa_params = inspect.signature(F.scaled_dot_product_attention).parameters
    _SDPA_SUPPORT_GQA = 'enable_gqa' in _sdpa_params
except ValueError:
    _SDPA_SUPPORT_GQA = False


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class MLP(nn.Module):
    def __init__(self, config: Config, intermediate_size: Optional[int] = None):
        super().__init__()
        config_intermediate_size = intermediate_size if intermediate_size else config.intermediate_size

        self.gate_proj = nn.Linear(config.hidden_size, config_intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config_intermediate_size, bias=False)
        self.down_proj = nn.Linear(config_intermediate_size, config.hidden_size, bias=False)
        self.activation = nn.SiLU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.activation(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class RotaryEmbedding(nn.Module):
    def __init__(self, config: Optional[Config] = None, device = None):
        super().__init__()

        if config is not None and config.rope_config.rope_type is not None:
            self.rope_type = config.rope_config.rope_type
        else:
            self.rope_type = "default"

        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    def _dynamic_frequency_update(self, position_ids, device):
        """
        dynamic RoPE layers should recompute `inv_freq` in the following situations:
        1 - growing beyond the cached sequence length (allow scaling)
        2 - the current sequence length is in the original scale (avoid losing precision with small sequences)
        """
        seq_len = torch.max(position_ids) + 1
        if seq_len > self.max_seq_len_cached:  # growth
            inv_freq, self.attention_scaling = self.rope_init_fn(
                self.config, device, seq_len=seq_len
            )
            self.register_buffer("inv_freq", inv_freq, persistent=False)
            self.max_seq_len_cached = seq_len

        if seq_len < self.original_max_seq_len and self.max_seq_len_cached > self.original_max_seq_len:  # reset
            self.register_buffer("inv_freq", self.original_inv_freq, persistent=False)
            self.max_seq_len_cached = self.original_max_seq_len

    @torch.no_grad()
    def forward(self, x, position_ids):
        if "dynamic" == self.rope_type:
            self._dynamic_frequency_update(position_ids, device=x.device)

        # Core RoPE block
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 (see https://github.com/huggingface/transformers/pull/29285)
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        # Advanced RoPE types (e.g. yarn) apply a post-processing scaling factor, equivalent to scaling attention
        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class Attention(nn.Module):
    def __init__(self, config: Config, layer_idx: int, use_sdpa_attention: bool):
        super().__init__()
        assert config.num_attention_heads % config.num_key_value_heads == 0

        self.use_sdpa_attention = use_sdpa_attention
        self.use_qk_norm = config.use_qk_norm
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

        if self.use_qk_norm:
            self.q_norm = RMSNorm(self.head_size)
            self.k_norm = RMSNorm(self.head_size)

    def _sdpa_kernel(self, enable_flash: bool = True, enable_math: bool = True, enable_mem_efficient: bool = True):
        if version.parse(torch.__version__).release < version.parse("2.3").release:
            return torch.backends.cuda.sdp_kernel(
                enable_flash=enable_flash, enable_math=enable_math, enable_mem_efficient=enable_mem_efficient
            )

        backends = []
        if enable_flash:
            backends += [torch.nn.attention.SDPBackend.FLASH_ATTENTION]
        if enable_math:
            backends += [torch.nn.attention.SDPBackend.MATH]
        if enable_mem_efficient:
            backends += [torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION]
        return torch.nn.attention.sdpa_kernel(backends)

    def forward(
            self,
            hidden_states: torch.Tensor,
            position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
            attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[KVCache] = None
    ) -> torch.Tensor:
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
        query_states = query_states.reshape(batch, seq_len, -1, self.head_size)
        key_states = key_states.reshape(batch, seq_len, -1, self.head_size)
        value_states = value_states.reshape(batch, seq_len, -1, self.head_size)

        if self.use_qk_norm:
            query_states = self.q_norm(query_states)
            key_states = self.k_norm(key_states)

        # query_states (batch, num_heads, seq_len, head_size)
        # key_states (batch, num_key_value_heads, seq_len, head_size)
        # value_states (batch, num_key_value_heads, seq_len, head_size)
        query_states = query_states.permute(0, 2, 1, 3)
        key_states = key_states.permute(0, 2, 1, 3)
        value_states = value_states.permute(0, 2, 1, 3)

        cos, sin = position_embeddings
        # query_states (batch, num_heads, seq_len, head_size)
        # key_states (batch, num_key_value_heads, seq_len, head_size)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        is_gqa = self.num_key_value_groups > 1
        if self.use_sdpa_attention:
            with self._sdpa_kernel():
                dropout_p = self.dropout.p if self.training else 0.0
                is_causal = attention_mask is None and seq_len > 1

                if is_gqa and _SDPA_SUPPORT_GQA:
                    attn = F.scaled_dot_product_attention(
                        query=query_states,
                        key=key_states,
                        value=value_states,
                        dropout_p=dropout_p,
                        attn_mask=attention_mask,
                        is_causal=is_causal,
                        enable_gqa=True
                    )
                else:
                    if is_gqa:
                        # (batch, num_key_value_heads, seq_len, head_size) ->
                        # (batch, num_key_value_heads, 1, seq_len, head_size) ->
                        # (batch, num_key_value_heads, num_key_value_groups=num_heads//num_key_value_heads, seq_len, head_size) ->
                        # (batch, num_heads=num_key_value_heads*num_key_value_groups, seq_len, head_size)
                        key_states = key_states.repeat_interleave(self.num_key_value_groups, dim=1)
                        value_states = value_states.repeat_interleave(self.num_key_value_groups, dim=1)

                    attn = F.scaled_dot_product_attention(
                        query=query_states,
                        key=key_states,
                        value=value_states,
                        dropout_p=dropout_p,
                        attn_mask=attention_mask,
                        is_causal=is_causal
                    )

            # (batch, num_heads, seq_len, head_size) -> (batch, seq_len, num_heads, head_size)
            attn = attn.transpose(1, 2)
        else:
            if is_gqa:
                # (batch, num_key_value_heads, seq_len, head_size) ->
                # (batch, num_key_value_heads, 1, seq_len, head_size) ->
                # (batch, num_key_value_heads, num_key_value_groups=num_heads//num_key_value_heads, seq_len, head_size) ->
                # (batch, num_heads=num_key_value_heads*num_key_value_groups, seq_len, head_size)
                key_states = key_states.repeat_interleave(self.num_key_value_groups, dim=1)
                value_states = value_states.repeat_interleave(self.num_key_value_groups, dim=1)

            # (batch, num_heads, q_seq_len, k_seq_len)
            attn_scores = (self.scale * query_states) @ key_states.transpose(-1, -2)
            if attention_mask is not None:
                attn_scores = attn_scores + attention_mask

            attn_weights = attn_scores.softmax(-1)
            if self.training:
                attn_weights = self.dropout(attn_weights)

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
    def __init__(self, config: Config, layer_idx: int, use_sdpa_attention: bool):
        super().__init__()

        self.attn_norm = RMSNorm(config.hidden_size)
        self.attn = Attention(config, layer_idx, use_sdpa_attention)

        self.mlp_norm = RMSNorm(config.hidden_size)

        use_moe = (
                config.moe_config
                and config.moe_config.intermediate_size
                and config.moe_config.num_experts_per_tok
                and config.moe_config.n_routed_experts
                and config.moe_config.n_shared_experts
                and layer_idx >= config.moe_config.n_dense_layer
        )

        if use_moe:
            self.mlp = MoE(config=config, layer=MLP)
        else:
            self.mlp = MLP(config)

    def forward(
            self,
            hidden_states: torch.Tensor,
            position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
            attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[KVCache] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        residual = hidden_states
        hidden_states = self.attn(
            hidden_states=self.attn_norm(hidden_states),
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_values=past_key_values
        )

        hidden_states = hidden_states + residual

        if isinstance(self.mlp, MoE):
            mlp_states, aux_loss = self.mlp(self.mlp_norm(hidden_states))
        else:
            mlp_states = self.mlp(self.mlp_norm(hidden_states))
            aux_loss = None

        hidden_states = mlp_states + hidden_states

        return hidden_states, aux_loss


class LlmModel(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        if config.attention_implementation == 'auto':
            self.use_sdpa_attention = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        else:
            self.use_sdpa_attention = config.attention_implementation == 'sdpa'

        self.rotary_emb = RotaryEmbedding(config=config)

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [DecoderLayer(config, idx, self.use_sdpa_attention) for idx in range(config.num_hidden_layers)])

        self.head_norm = RMSNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        std = 0.02
        if isinstance(module, nn.Linear):
            # torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            past_key_values: Optional[KVCache] = None,
            use_cache: bool = False,
            **kwargs,
    ) -> Dict[str, any]:
        """
        Args:
            input_ids (`torch.Tensor`):
                input tokens
            attention_mask (`torch.Tensor`, *optional*, default is None)
                input mask for ignore padding, shape is (batch, seq_len),
                eg [[true, true, true, false], [true, true, true, true]]
            position_ids
                shape is (batch, seq_len)
            past_key_values (`KVCache`, *optional*, default is None):
                inference key value cache, when use_cache == True, will return KVCache on first forward
            use_cache (`bool`, default is False)
                use KVCache or not

        Returns:
            logits
                the model logits output
            past_key_values:
                inference key value cache, when use_cache == True, will return KVCache on first forward
            aux_loss:
                aux loss when use MOE, else None
        """
        batch_size, seq_len = input_ids.shape
        if use_cache and past_key_values is None:
            past_key_values = KVCache()

        # (batch, seq_len, hidden_size)
        # for inference with past_key_values inputs_embeds.shape is (1, 1, hidden_size)
        inputs_embeds = self.get_input_embeddings(input_ids, attention_mask, **kwargs)

        # seq_len
        past_seen_tokens = past_key_values.get_seq_len() if past_key_values is not None else 0
        full_seq_len = past_seen_tokens + seq_len

        if position_ids is None:
            position_ids = torch.arange(past_seen_tokens, full_seq_len, device=inputs_embeds.device).unsqueeze(0)

        if attention_mask is None:
            # (batch_size, past_seen_tokens+seq_len)
            # all true, no paddings for mask
            attention_mask = torch.ones(
                (batch_size, full_seq_len), dtype=torch.bool, device=inputs_embeds.device
            )

        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)

        can_use_flash_causal = False
        if self.use_sdpa_attention and seq_len > 1:
            if attention_mask.all():
                can_use_flash_causal = True

        if can_use_flash_causal:
            causal_mask = None
        else:
            # (bsz, 1, seq_len, full_seq_len)
            causal_mask = prepare_decoder_attention_mask(
                attention_mask=attention_mask,
                input_shape=(batch_size, seq_len),
                past_key_values_length=past_seen_tokens,
                dtype=inputs_embeds.dtype,
                device=inputs_embeds.device
            )

        hidden_states = inputs_embeds
        aux_losses = ()

        for layer in self.layers:
            hidden_states, aux_loss = layer(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=causal_mask,
                past_key_values=past_key_values
            )

            if aux_loss is not None:
                aux_losses += (aux_loss,)

        # (batch, seq_len, hidden_size)
        hidden_states = self.head_norm(hidden_states)

        # (batch, seq_len, vocab_size)
        head = self.lm_head(hidden_states)

        return {
            'logits': head,
            'hidden_states': hidden_states,
            'past_key_values': past_key_values,
            'aux_loss': None if len(aux_losses) == 0 else sum(aux_losses)
        }

    def get_input_embeddings(
            self,
            input_ids: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            **kwargs,
    ):
        return self.embed_tokens(input_ids)