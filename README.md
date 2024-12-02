# llama-pytorch
Implement Meta Llama in pytorch

``` python
import torch
from llama import LlamaModel
from llama import LlamaConfig, KVCache


def get_llama_config(vocab_size):
    return LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=256,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=100)


def test_llama_model(use_kv_cache=True):
    llama: LlamaModel = LlamaModel(config=get_llama_config(vocab_size=1000))
    input_ids = torch.ones((1, 3), dtype=torch.long)

    kv_cache: KVCache = None

    llama.eval()
    with torch.no_grad():
        for _ in range(10):
            logits, kv_cache = llama(input_ids, past_key_values=kv_cache, use_cache=use_kv_cache)
            logits = logits[:, -1, :]
            out_token = logits.argmax(dim=-1, keepdim=True)
            print(out_token)

            if use_kv_cache:
                input_ids = out_token
            else:
                input_ids = torch.concat((input_ids, out_token), dim=-1)


if __name__ == '__main__':
    test_llama_model(use_kv_cache=True)
```
