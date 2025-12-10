import torch
from typing import Optional


def _expand_mask(
        mask: torch.Tensor,
        dtype: torch.dtype,
        tgt_len: Optional[int] = None
):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    if len(mask.size()) == 3:
        bsz, src_len, _ = mask.size()
        tgt_len = tgt_len if tgt_len is not None else src_len
        expanded_mask = mask[:,None,:,:].expand(bsz, 1, tgt_len, src_len).to(dtype)
    else:
        bsz, src_len = mask.size()
        tgt_len = tgt_len if tgt_len is not None else src_len
        expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    # [true, true, true, false] -> [0, 0, 0, 1]
    inverted_mask = 1.0 - expanded_mask
    # tensor([[[[     0.,      0.,      0., -65504.],
    #           [     0.,      0.,      0., -65504.],
    #           [     0.,      0.,      0., -65504.],
    #           [     0.,      0.,      0., -65504.]]]])
    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


# Copied from transformers.models.bart.modeling_bart._make_causal_mask
def _make_causal_mask(
        input_ids_shape: torch.Size,
        dtype: torch.dtype,
        device:
        torch.device,
        past_key_values_length: int = 0
):
    """
    input_shape (batch_size, seq_length)
    seq_length > 1
    combined_attention_mask = _make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.tensor(torch.finfo(dtype).min, device=device), device=device)
    # (0, 1, 2, 3, 4 ...)
    mask_cond = torch.arange(mask.size(-1), device=device)
    # mask_cond shape (5) expand to (5, 5)
    #  [[0, 1, 2, 3, 4],
    #   [0, 1, 2, 3, 4],
    #   ...
    #   [0, 1, 2, 3, 4]]
    # (mask_cond + 1).view(mask.size(-1), 1) shape (5, 1) expand to (5, 5)
    # [[1, 1, 1, 1, 1],
    #  [2, 2, 2, 2, 2],
    #  ...
    #  [5, 5, 5, 5, 5]]
    # mask_cond < (mask_cond + 1).view(mask.size(-1), 1)
    # [[True, False, False, False, False],
    #  [True, True, False, False, False],
    #  [True, True, True, False, False],
    #  [True, True, True, True, False],
    #  [True, True, True, True, True]]
    # mask shape (seq_len, seq_len)
    #  [[0, -inf, -inf, -inf, -inf],
    #   [0, 0, -inf, -inf, -inf],
    #   ...
    #   [0, 0, 0, 0, 0]]
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


def prepare_decoder_attention_mask(
        attention_mask,
        input_shape,
        past_key_values_length,
        dtype,
        device
):
    """
    attention_mask = torch.ones(
            (batch_size, seq_length_with_past), dtype=torch.bool, device=inputs_embeds.device
        )
    attention_mask = self._prepare_decoder_attention_mask(
        attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
    )
    """
    # create causal mask
    # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
    combined_attention_mask = None
    if input_shape[-1] > 1:
        #tensor([[[[     0., -65504., -65504., -65504.],
        #  [     0.,      0., -65504., -65504.],
        #  [     0.,      0.,      0., -65504.],
        #  [     0.,      0.,      0.,      0.]]]])
        combined_attention_mask = _make_causal_mask(
            input_shape,
            dtype,
            device=device,
            past_key_values_length=past_key_values_length,
        )

    if attention_mask is not None:
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        # -65504 is pad
        # tensor([[[[     0.,      0.,      0., -65504.],
        #           [     0.,      0.,      0., -65504.],
        #           [     0.,      0.,      0., -65504.],
        #           [     0.,      0.,      0., -65504.]]]])
        expanded_attn_mask = _expand_mask(attention_mask, dtype, tgt_len=input_shape[-1]).to(
            device
        )

        # tensor([[[[     0., -65504., -65504.,    -inf],
        #           [     0.,      0., -65504.,    -inf],
        #           [     0.,      0.,      0.,    -inf],
        #           [     0.,      0.,      0., -65504.]]]], dtype=torch.float16)
        combined_attention_mask = (
            expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
        )

    return combined_attention_mask