# llm-model-pytorch
Implement llm model in pytorch, support MoE and RoPE

## install
```python
pip3 install project_llm_model
```

## usage
``` python
import torch
from llm_model import LlmModel
from llm_model import ModelConfig, RoPEConfig, MoEConfig


def get_model_config():
    return ModelConfig(
        vocab_size=TrainerTools().tokenizer.vocab_size,
        hidden_size=768,
        intermediate_size=2048,
        moe_intermediate_size=1024,
        moe_n_dense_layer=1,
        num_hidden_layers=24,
        num_attention_heads=12,
        num_key_value_heads=2,
        max_position_embeddings=1024,
        attention_implementation='auto',
        rope_config=RoPEConfig(
            rope_theta=1e6
        ),
        moe_config=MoEConfig(
            num_experts_per_tok=2,
            n_routed_experts=8,
            n_shared_experts=1,
            aux_loss_alpha=0.1,
            seq_aux=True,
            norm_topk_prob=True
        )
    )


def test_model(test_train=True):
    model: LlmModel = LlmModel(config=get_model_config(vocab_size=1000))
    pad_token_id = 0

    if test_train:
        input_ids = torch.tensor([[1, 2, 3], [2, pad_token_id, pad_token_id]], dtype=torch.long)
        # [[true, true, true], [true, false, false]]
        attention_mask = input_ids != pad_token_id
        logits, _ = model(input_ids, attention_mask=attention_mask)
    else:
        input_ids = torch.ones((1, 3), dtype=torch.long)
        kv_cache: KVCache = None
        model.eval()
        with torch.no_grad():
            for _ in range(10):
                logits, kv_cache = model(input_ids, past_key_values=kv_cache, use_cache=True)
                logits = logits[:, -1, :]
                out_token = logits.argmax(dim=-1, keepdim=True)
                print(out_token)

                input_ids = out_token


if __name__ == '__main__':
    test_model(test_train=True)
```
