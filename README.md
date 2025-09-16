# llm-model-pytorch
Implement LLM and VLM model in pytorch, support MoE and RoPE

## 安装
```python
pip3 install project_llm_model
```

## 快速开始
``` python
from llm_model import *
def get_model_config():
    return ModelConfig(...)

def get_vlm_config():
    return VLMConfig(...)

model = LlmModel(get_model_config())
vlm_model = VlmModel(get_vlm_config())

print(model)
print(vlm_model)
```

## LLM ModelConfig 配置说明
|  字段 | 类型 | 解释 |
|  ---- |  ----   | ----  |
| vocab_size | int | 指定使用字典大小 |
| hidden_size | int | 指定模型的hidden size |
| intermediate_size | int | 指定模型中MLP的intermediate size |
| num_hidden_layers | int | 指定使用几个隐藏层 |
| num_attention_heads | int | 指定attention的头的数量 |
| num_key_value_heads | int | 指定attention的key和value头的数量 |
| max_position_embeddings | int | 配置位置编码的max_position_embeddings |
| original_max_position_embeddings | int | 当使用YaRN时，配置原始的max_position_embeddings |
| attention_dropout | float | attention的Dropout rate |
| attention_implementation | str | attention实现方式，取值：auto\sdpa\default |
| rope_config.rope_type | str | RoPE类型，取值：default\yarn\dynamic |
| rope_config.rope_theta | float | 对所有RoPE生效 |
| rope_config.factor | float | 对除default外所有RoPE生效 |
| rope_config.partial_rotary_factor | 对所有RoPE生效 |
| rope_config.beta_fast | float | 仅对YaRN生效 |
| rope_config.beta_slow | float | 仅对YaRN生效 |
| rope_config.mscale | float | 仅对YaRN生效 |
| rope_config.mscale_all_dim | Option[float] | 仅对YaRN生效 |
| rope_config.attention_factor | Option[Option] | 仅对YaRN生效 |
| moe_config.intermediate_size | Optional[int] | 使用MoE模型时指定专家MLP的intermediate size |
| moe_config.n_dense_layer | Optional[int] | 使用MoE模型时指定使用多少MLP层 |
| moe_config.num_experts_per_tok | Option[int] | MoE模型每个token选择的专家数 |
| moe_config.n_shared_experts | Option[int] | MoE模型共享专家总数 |
| moe_config.n_routed_experts | Option[int] | MoE模型被路由的专家总数 |
| moe_config.seq_aux | bool | 是否计算每个单独样本的辅助损失 |
| moe_config.norm_topk_prob | bool | 是否对路由专家的权重进行标准化 |
| use_qk_norm | bool | 是否使用qk norm |


## VLMConfig
VlmConfig 继承自 LLMConfig
|  字段 | 类型 | 解释 |
|  ---- |  ----   | ---- |
| image_tok | int | 指定图像的token id |
| image_size | int | 指定图像的大小 |
| patch_size | int | 指定每个patch的大小 |
| tokens_per_image | int | 指定每个图片占用token个数 |
| vision_hidden_size | int | 指定vision projector的hidden size |
| vision_tower | Callable[[torch.Tensor], torch.Tensor] | 用于指定视觉模型的输出 |



## Demo
``` python
import torch
from llm_model import *

def get_model_config(long_context = False):
    # max_position_embeddings: 512 -> 2048
    max_position_embeddings = 2048 if long_context else 512
    original_max_position_embeddings = 512 if long_context else None
    rope_type = 'yarn' if long_context else 'default'

    return ModelConfig(
        vocab_size=TrainerTools().tokenizer.vocab_size,
        hidden_size=768,
        intermediate_size=2048,
        num_hidden_layers=24,
        num_attention_heads=12,
        num_key_value_heads=4,
        max_position_embeddings=max_position_embeddings,
        original_max_position_embeddings=original_max_position_embeddings,
        attention_implementation='auto',
        rope_config=RoPEConfig(
            rope_type=rope_type,
            rope_theta=1e6
        ),
        moe_config=MoEConfig(
            intermediate_size=1024,
            n_dense_layer=1,
            num_experts_per_tok=2,
            n_shared_experts=1,
            n_routed_experts=8,
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
