import torch
from typing import Optional


class KVCache:
    def __init__(self, max_capacity: int = 0):
        # (layer_idx, batch, num_heads, seq_len, head_size)
        self.key_cache = []
        self.value_cache = []

        self.max_capacity = max_capacity
        self.lengths = []

    def get_seq_len(self, layer_idx: Optional[int] = 0):
        if self.max_capacity > 0:
            if layer_idx < len(self.lengths):
                return self.lengths[layer_idx]
            return 0

        empty_cache = (
                len(self.key_cache) == 0
                or len(self.key_cache) <= layer_idx
                or len(self.key_cache[layer_idx]) == 0
        )

        return self.key_cache[layer_idx].shape[-2] if not empty_cache else 0

    def update(self, key_states, value_states, layer_idx):
        if len(self.key_cache) <= layer_idx:
            for _ in range(len(self.key_cache), layer_idx + 1):
                self.key_cache.append(torch.empty(0))
                self.value_cache.append(torch.empty(0))
                self.lengths.append(0)

        input_seq_len = key_states.shape[-2]

        if self.max_capacity > 0:
            current_len = self.lengths[layer_idx]

            if self.key_cache[layer_idx].numel() == 0:
                batch_size, num_heads, _, head_dim = key_states.shape
                # 申请最大显存
                cache_shape = (batch_size, num_heads, self.max_capacity, head_dim)

                self.key_cache[layer_idx] = torch.zeros(cache_shape, dtype=key_states.dtype, device=key_states.device)
                self.value_cache[layer_idx] = torch.zeros(cache_shape, dtype=value_states.dtype, device=value_states.device)

            end_idx = current_len + input_seq_len
            if end_idx > self.max_capacity:
                raise ValueError(f"KVCache capacity exceeded: max {self.max_capacity}, but trying to reach {end_idx}")

            self.key_cache[layer_idx][..., current_len:end_idx, :] = key_states
            self.value_cache[layer_idx][..., current_len:end_idx, :] = value_states

            self.lengths[layer_idx] = end_idx

            return (
                self.key_cache[layer_idx][..., :end_idx, :],
                self.value_cache[layer_idx][..., :end_idx, :]
            )

        if len(self.key_cache[layer_idx]) == 0:
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states
        else:
            self.key_cache[layer_idx] = torch.cat((self.key_cache[layer_idx], key_states), dim=-2)
            self.value_cache[layer_idx] = torch.cat((self.value_cache[layer_idx], value_states), dim=-2)

        return self.key_cache[layer_idx], self.value_cache[layer_idx]