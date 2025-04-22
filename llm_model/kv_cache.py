import torch
from typing import Optional


class KVCache:
    def __init__(self):
        # (layer_idx, batch, num_heads, seq_len, head_size)
        self.key_cache = []
        self.value_cache = []

    def get_seq_len(self, layer_idx: Optional[int] = 0):
        empty_cache = (
                len(self.key_cache) == 0
                or len(self.key_cache) <= layer_idx
                or len(self.key_cache[layer_idx]) == 0
        )

        return self.key_cache[layer_idx].shape[-2] if not empty_cache else 0

    def update(self, key_states, value_states, layer_idx):
        if len(self.key_cache) <= layer_idx:
            for _ in range(len(self.key_cache), layer_idx):
                self.key_cache.append([])
                self.value_cache.append([])

            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        elif len(self.key_cache[layer_idx]) == 0:
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states
        else:
            self.key_cache[layer_idx] = torch.cat((self.key_cache[layer_idx], key_states), dim=-2)
            self.value_cache[layer_idx] = torch.cat((self.value_cache[layer_idx], value_states), dim=-2)

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

