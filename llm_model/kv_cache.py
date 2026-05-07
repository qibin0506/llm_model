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
                or self.key_cache[layer_idx] is None
        )

        return self.key_cache[layer_idx].shape[-2] if not empty_cache else 0

    def update(self, key_states, value_states, layer_idx):
        if len(self.key_cache) <= layer_idx:
            for _ in range(len(self.key_cache), layer_idx + 1):
                self.key_cache.append(None)
                self.value_cache.append(None)
                self.lengths.append(0)

        input_seq_len = key_states.shape[-2]
        current_batch = key_states.shape[0]

        if self.max_capacity > 0:
            current_len = self.lengths[layer_idx]

            if self.key_cache[layer_idx] is None or self.key_cache[layer_idx].shape[0] != current_batch:
                batch_size, num_heads, _, head_dim = key_states.shape

                if self.key_cache[layer_idx] is not None and self.key_cache[layer_idx].shape[0] < current_batch:
                    repeat_factor = current_batch // self.key_cache[layer_idx].shape[0]
                    old_keys = self.key_cache[layer_idx][..., :current_len, :]
                    old_vals = self.value_cache[layer_idx][..., :current_len, :]

                    cache_shape = (batch_size, num_heads, self.max_capacity, head_dim)
                    new_key_cache = torch.empty(cache_shape, dtype=key_states.dtype, device=key_states.device)
                    new_val_cache = torch.empty(cache_shape, dtype=value_states.dtype, device=value_states.device)

                    new_key_cache[..., :current_len, :] = old_keys.repeat_interleave(repeat_factor, dim=0)
                    new_val_cache[..., :current_len, :] = old_vals.repeat_interleave(repeat_factor, dim=0)

                    self.key_cache[layer_idx] = new_key_cache
                    self.value_cache[layer_idx] = new_val_cache
                else:
                    cache_shape = (batch_size, num_heads, self.max_capacity, head_dim)
                    self.key_cache[layer_idx] = torch.empty(cache_shape, dtype=key_states.dtype,
                                                            device=key_states.device)
                    self.value_cache[layer_idx] = torch.empty(cache_shape, dtype=value_states.dtype,
                                                              device=value_states.device)
                    self.lengths[layer_idx] = 0
                    current_len = 0

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

        if self.key_cache[layer_idx] is None:
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states
        else:
            if self.key_cache[layer_idx].shape[0] < current_batch:
                repeat_factor = current_batch // self.key_cache[layer_idx].shape[0]
                self.key_cache[layer_idx] = self.key_cache[layer_idx].repeat_interleave(repeat_factor, dim=0)
                self.value_cache[layer_idx] = self.value_cache[layer_idx].repeat_interleave(repeat_factor, dim=0)
            elif self.key_cache[layer_idx].shape[0] > current_batch:
                self.key_cache[layer_idx] = key_states
                self.value_cache[layer_idx] = value_states
                return self.key_cache[layer_idx], self.value_cache[layer_idx]

            self.key_cache[layer_idx] = torch.cat((self.key_cache[layer_idx], key_states), dim=-2)
            self.value_cache[layer_idx] = torch.cat((self.value_cache[layer_idx], value_states), dim=-2)

        return self.key_cache[layer_idx], self.value_cache[layer_idx]