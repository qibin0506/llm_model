from typing import Optional, Tuple, Dict
from functools import partial
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from .model_config import Config
from .attention_interface import get_attention_interface, supports_fused_causal_mask
from .rope import ROPE_INIT_FUNCTIONS, apply_rotary_pos_emb
from .kv_cache import KVCache
from .attention_masks import prepare_decoder_attention_mask
from .sparse_moe import MoE


class BlockAttnRes(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(hidden_size))
        self.norm = RMSNorm(hidden_size)

    def forward(self, blocks: list[torch.Tensor], partial_block: torch.Tensor):
        # [N+1, B, T, D]
        if len(blocks) > 0:
            V = torch.stack(blocks + [partial_block], dim=0)
        else:
            V = partial_block.unsqueeze(0)

        K = self.norm(V)

        # [N+1, B, T]
        logits = torch.einsum('d, n b t d -> n b t', self.weight, K)
        scores = logits.softmax(dim=0)
        # [B, T, D]
        h = torch.einsum('n b t, n b t d -> b t d', scores, V)
        return h


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.square().mean(-1, keepdim=True)
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
        seq_len = torch.max(position_ids).item() + 1
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

        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class Attention(nn.Module):
    def __init__(self, config: Config, layer_idx: int):
        super().__init__()
        assert config.num_attention_heads % config.num_key_value_heads == 0

        self.use_qk_norm = config.use_qk_norm
        self.layer_idx = layer_idx

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_size = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.scale = self.head_size ** -0.5
        self.drop_rate = config.attention_dropout

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_size, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_size, self.hidden_size, bias=False)
        self.attention_interface = get_attention_interface(config)

        if self.use_qk_norm:
            self.q_norm = RMSNorm(self.head_size)
            self.k_norm = RMSNorm(self.head_size)

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

        attn = self.attention_interface(
            module=self,
            q=query_states,
            k=key_states,
            v=value_states,
            attention_mask=attention_mask,
            num_key_value_groups=self.num_key_value_groups,
            scale=self.scale,
            drop_rate=self.drop_rate,
        )

        # (batch, seq_len, num_heads*head_size)
        attn = attn.reshape(batch, seq_len, -1)
        # (batch, seq_len, hidden_size)
        out = self.o_proj(attn)
        return out


class DecoderLayer(nn.Module):
    def __init__(self, config: Config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx

        self.attn_norm = RMSNorm(config.hidden_size)
        self.attn = Attention(config, layer_idx)
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

        self.attn_res_config = config.attn_res_config
        if self.attn_res_config is not None:
            # block_size = num_hidden_layers / attn_res_num_blocks
            self.layers_per_block = config.num_hidden_layers // self.attn_res_config.num_blocks
            self.attn_res_agg = BlockAttnRes(config.hidden_size)
            self.mlp_res_agg = BlockAttnRes(config.hidden_size)

    def forward(
            self,
            hidden_states: torch.Tensor,
            position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
            attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[KVCache] = None,
            blocks: Optional[list[torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[list[torch.Tensor]]]:
        if self.attn_res_config is not None:
            # Block Attention Residuals
            partial_block = hidden_states
            h = self.attn_res_agg(blocks, partial_block)
            if self.layer_idx % self.layers_per_block == 0:
                blocks.append(partial_block)
                partial_block = None

            attn_out = self.attn(
                hidden_states=self.attn_norm(h),
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                past_key_values=past_key_values
            )
            partial_block = partial_block + attn_out if partial_block is not None else attn_out

            h = self.mlp_res_agg(blocks, partial_block)
            if isinstance(self.mlp, MoE):
                mlp_states, aux_loss = self.mlp(self.mlp_norm(h))
            else:
                mlp_states = self.mlp(self.mlp_norm(h))
                aux_loss = None
            partial_block = partial_block + mlp_states

            return partial_block, aux_loss, blocks

        # standard residual
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

        return hidden_states, aux_loss, blocks


class LlmModel(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.gradient_checkpointing = False
        self._checkpoint_func = None

        if config.attn_res_config is not None:
            assert config.num_hidden_layers % config.attn_res_config.num_blocks == 0

        self.rotary_emb = RotaryEmbedding(config=config)

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [DecoderLayer(config, idx) for idx in range(config.num_hidden_layers)])

        self.head_norm = RMSNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.apply(self._init_weights)

        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

    def _init_weights(self, module: nn.Module):
        std = self.config.initializer_range

        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.MultiheadAttention):
            # This uses torch's original init
            module._reset_parameters()

    def gradient_checkpointing_enable(self):
        self.gradient_checkpointing = True

        if self._checkpoint_func is None:
            try:
                import deepspeed
                self._checkpoint_func = deepspeed.checkpointing.checkpoint
            except ImportError:
                self._checkpoint_func = partial(torch_checkpoint, use_reentrant=False)

    def gradient_checkpointing_disable(self):
        self.gradient_checkpointing = False

    def _create_custom_forward(self, module):
        def custom_forward(*inputs):
            # inputs: hidden_states, attention_mask, cos, sin, blocks
            h = inputs[0]
            m = inputs[1]
            c = inputs[2]
            s = inputs[3]
            num_input_blocks = len(inputs) - 4

            if self.config.attn_res_config is not None:
                blocks = list(inputs[4:])
            else:
                blocks = None

            out_h, out_aux, out_blocks = module(
                hidden_states=h,
                position_embeddings=(c, s),
                attention_mask=m,
                past_key_values=None,
                blocks=blocks
            )

            res = [out_h, out_aux]
            if out_blocks is not None and blocks is not None:
                if len(out_blocks) > num_input_blocks:
                    res.append(out_blocks[-1])

            return tuple(res)

        return custom_forward

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

        if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
            use_cache = False

        if use_cache and past_key_values is None:
            past_key_values = KVCache()

        # (batch, seq_len, hidden_size)
        # for inference with past_key_values inputs_embeds.shape is (1, 1, hidden_size)
        inputs_embeds = self.get_input_embeddings(input_ids, attention_mask, **kwargs)

        # seq_len
        past_seen_tokens = past_key_values.get_seq_len() if past_key_values is not None else 0
        full_seq_len = past_seen_tokens + seq_len

        if position_ids is None:
            if attention_mask is not None and attention_mask.shape[-1] == full_seq_len:
                position_ids = (attention_mask.cumsum(dim=-1) - 1).clamp(min=0)
                position_ids = position_ids[:, -seq_len:]
            else:
                position_ids = torch.arange(
                    past_seen_tokens,
                    full_seq_len,
                    device=inputs_embeds.device
                ).unsqueeze(0).expand(batch_size, -1)

        if attention_mask is None:
            # (batch_size, past_seen_tokens+seq_len)
            # all true, no paddings for mask
            attention_mask = torch.ones(
                (batch_size, full_seq_len), dtype=torch.bool, device=inputs_embeds.device
            )

        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)

        fused_causal_mask = False
        if supports_fused_causal_mask(self.config) and seq_len > 1 and past_seen_tokens == 0:
            if attention_mask.dim() == 2 and attention_mask.all():
                fused_causal_mask = True

        if fused_causal_mask:
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
        blocks = [] if self.config.attn_res_config is not None else None

        for layer in self.layers:
            if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                cos, sin = position_embeddings
                custom_forward = self._create_custom_forward(layer)
                if blocks is not None:
                    layer_outputs = self._checkpoint_func(
                        custom_forward,
                        hidden_states, causal_mask, cos, sin, *blocks
                    )
                    hidden_states = layer_outputs[0]
                    aux_loss = layer_outputs[1]
                    if len(layer_outputs) > 2:
                        blocks.append(layer_outputs[2])
                else:
                    layer_outputs = self._checkpoint_func(
                        custom_forward,
                        hidden_states, causal_mask, cos, sin
                    )
                    hidden_states = layer_outputs[0]
                    aux_loss = layer_outputs[1]
            else:
                hidden_states, aux_loss, blocks = layer(
                    hidden_states=hidden_states,
                    position_embeddings=position_embeddings,
                    attention_mask=causal_mask,
                    past_key_values=past_key_values,
                    blocks=blocks
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