import torch
from typing import Optional, Dict, Any


class KVCache:
    def __init__(self, max_capacity: int = 0):
        # (layer_idx, batch, num_heads, seq_len, head_size)
        self.key_cache = []
        self.value_cache = []
        self.recurrent_cache: Dict[int, Dict[str, Any]] = {}

        self.max_capacity = max_capacity
        self.lengths = []
        self.seen_tokens = 0

    def reset(self):
        self.key_cache = []
        self.value_cache = []
        self.recurrent_cache.clear()
        self.lengths = []
        self.seen_tokens = 0

    def get_cached_batch_size(self) -> Optional[int]:
        for k in self.key_cache:
            if k is not None:
                return k.shape[0]
        for state_dict in self.recurrent_cache.values():
            if state_dict is not None:
                first_tensor = next((v for v in state_dict.values() if isinstance(v, torch.Tensor)), None)
                if first_tensor is not None:
                    return first_tensor.shape[0]
        return None

    def check_and_reset_on_batch_shrink(self, current_batch: int) -> bool:
        cached_batch = self.get_cached_batch_size()
        if cached_batch is not None and (
            cached_batch > current_batch
            or (current_batch > cached_batch and current_batch % cached_batch != 0)
        ):
            self.reset()
            return True
        return False

    def get_seq_len(self, layer_idx: Optional[int] = None):
        if layer_idx is not None:
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

        return self.seen_tokens

    def _is_first_kv_layer(self, layer_idx: int) -> bool:
        return all(k is None for k in self.key_cache[:layer_idx])

    def update(self, key_states, value_states, layer_idx):
        input_seq_len = key_states.shape[-2]
        current_batch = key_states.shape[0]

        if len(self.key_cache) > layer_idx and self.key_cache[layer_idx] is not None:
            old_batch = self.key_cache[layer_idx].shape[0]
            if old_batch > current_batch or (current_batch > old_batch and current_batch % old_batch != 0):
                for idx in range(len(self.key_cache)):
                    if self.key_cache[idx] is not None and self.key_cache[idx].shape[0] == old_batch:
                        self.key_cache[idx] = None
                        self.value_cache[idx] = None
                        if idx < len(self.lengths):
                            self.lengths[idx] = 0
                self.recurrent_cache.clear()
                self.seen_tokens = 0

        if len(self.key_cache) <= layer_idx:
            for _ in range(len(self.key_cache), layer_idx + 1):
                self.key_cache.append(None)
                self.value_cache.append(None)
                self.lengths.append(0)

        if self.max_capacity > 0:
            current_len = self.lengths[layer_idx]

            if self.key_cache[layer_idx] is None or self.key_cache[layer_idx].shape[0] != current_batch:
                batch_size, num_heads, _, head_dim = key_states.shape

                if (
                    self.key_cache[layer_idx] is not None
                    and self.key_cache[layer_idx].shape[0] < current_batch
                    and current_batch % self.key_cache[layer_idx].shape[0] == 0
                ):
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
            if self._is_first_kv_layer(layer_idx):
                self.seen_tokens = end_idx

            return (
                self.key_cache[layer_idx][..., :end_idx, :],
                self.value_cache[layer_idx][..., :end_idx, :]
            )

        if self.key_cache[layer_idx] is None:
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states
        else:
            if self.key_cache[layer_idx].shape[0] < current_batch and current_batch % self.key_cache[layer_idx].shape[0] == 0:
                repeat_factor = current_batch // self.key_cache[layer_idx].shape[0]
                self.key_cache[layer_idx] = self.key_cache[layer_idx].repeat_interleave(repeat_factor, dim=0)
                self.value_cache[layer_idx] = self.value_cache[layer_idx].repeat_interleave(repeat_factor, dim=0)
            elif self.key_cache[layer_idx].shape[0] != current_batch:
                self.key_cache[layer_idx] = key_states
                self.value_cache[layer_idx] = value_states
                if self._is_first_kv_layer(layer_idx):
                    self.seen_tokens = input_seq_len
                return self.key_cache[layer_idx], self.value_cache[layer_idx]

            self.key_cache[layer_idx] = torch.cat((self.key_cache[layer_idx], key_states), dim=-2)
            self.value_cache[layer_idx] = torch.cat((self.value_cache[layer_idx], value_states), dim=-2)

        if self._is_first_kv_layer(layer_idx):
            self.seen_tokens = self.key_cache[layer_idx].shape[-2]

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_recurrent_state(self, layer_idx: int, current_batch: Optional[int] = None) -> Optional[Dict[str, Any]]:
        state_dict = self.recurrent_cache.get(layer_idx, None)
        if state_dict is None or current_batch is None:
            return state_dict

        first_tensor = next((v for v in state_dict.values() if isinstance(v, torch.Tensor)), None)
        if first_tensor is not None:
            old_batch = first_tensor.shape[0]
            if old_batch < current_batch:
                if current_batch % old_batch == 0:
                    repeat_factor = current_batch // old_batch
                    expanded_dict = {}
                    for k, v in state_dict.items():
                        if isinstance(v, torch.Tensor) and v.shape[0] < current_batch:
                            expanded_dict[k] = v.repeat_interleave(repeat_factor, dim=0)
                        else:
                            expanded_dict[k] = v
                    self.recurrent_cache[layer_idx] = expanded_dict
                    return expanded_dict
                else:
                    self.recurrent_cache.pop(layer_idx, None)
                    return None
            elif old_batch > current_batch:
                self.recurrent_cache.pop(layer_idx, None)
                return None

        return state_dict

    def update_recurrent_state(self, layer_idx: int, state_dict: Dict[str, Any]):
        self.recurrent_cache[layer_idx] = state_dict