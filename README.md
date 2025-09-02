# llm-model-pytorch
Implement LLM and VLM model in pytorch, support MoE and RoPE

## Install
```python
pip3 install project_llm_model
```

## Quick start
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

## LLM Model Config
|  field | type | explanation | 中文解释 | 
|  ---- |  ----   | ----  | ---- |
| vocab_size | int | the size of the vocab | 指定使用字典大小 |
| hidden_size | int | the hidden size of model | 指定模型的hidden size |
| intermediate_size | int | the intermediate size of MLP | 指定模型中MLP的intermediate size |
| moe_intermediate_size | int | the intermediate size of export's MLP when use MoE | 使用MoE模型时指定专家MLP的intermediate size |
| moe_n_dense_layer | int | number of MLPs when use MoE | 使用MoE模型时指定使用多少MLP层 |
| num_hidden_layers | int | number of hidden layers | 指定使用几个隐藏层 |
| num_attention_heads | int | number of heads for attention | 指定attention的头的数量 |
| num_key_value_heads | int | number of heads for attention's key and value | 指定attention的key和value头的数量 |
| max_position_embeddings | int | config the max_position_embeddings of RoPE | 配置位置编码的max_position_embeddings |
| original_max_position_embeddings | int | config the origin_max_position_embeddings of YaRN | 当使用YaRN时，配置原始的max_position_embeddings |
| attention_dropout | float | dropout rate for attention | attention的Dropout rate |
| attention_implementation | str | the implemention of attention, auto or sdpa or default | attention实现方式，取值：auto\sdpa\default |
| rope_config.rope_type | str | the type of RoPE，default or yarn or dynamic | RoPE类型，取值：default\yarn\dynamic |
| rope_config.rope_theta | float | effective for all RoPE| 对所有RoPE生效 |
| rope_config.factor | float | effective for all RoPE except 'default' | 对除default外所有RoPE生效 |
| rope_config.partial_rotary_factor | float | effective for all RoPE | 对所有RoPE生效 |
| rope_config.beta_fast | float | only effective for YaRN | 仅对YaRN生效 |
| rope_config.beta_slow | float | only effective for YaRN | 仅对YaRN生效 |
| rope_config.mscale | float | only effective for YaRN | 仅对YaRN生效 |
| rope_config.mscale_all_dim | Option[float] | only effective for YaRN | 仅对YaRN生效 |
| rope_config.attention_factor | Option[Option] | only effective for YaRN | 仅对YaRN生效 |
| moe_config.num_experts_per_tok | Option[int] | number of selected experts when use MoE | MoE模型每个token选择的专家数 |
| moe_config.n_routed_experts | Option[int] | number of routed experts when use MoE | MoE模型被路由的专家总数 |
| moe_config.n_shared_experts | Option[int] | number of shared experts when use MoE | MoE模型共享专家总数 |
| moe_config.scoring_func | str | only support softmax now | 仅支持softmax |
| moe_config.aux_loss_alpha | float | auxiliary loss weight coefficient | MoE辅助loss系数 |
| moe_config.seq_aux | bool | whether to compute the auxiliary loss for each individual sample | 是否计算每个单独样本的辅助损失 |
| moe_config.norm_topk_prob | bool | whether to normalize the weights of the routed experts | 是否对路由专家的权重进行标准化 |
| use_qk_norm | bool | whether to use qk norm | 是否使用qk norm |


## VLM Config
VlmConfig inherits from LLMConfig
|  field | type | explanation | 中文解释 | 
|  ---- |  ----   | ----  | ---- |
| image_tok | int | specify the token id of image | 指定图像的token id |
| image_size | int | specify the size of image | 指定图像的大小 |
| patch_size | int | specify the patch size of image | 指定每个patch的大小 |
| tokens_per_image | int | specify the number of tokens each image occupies | 指定每个图片占用token个数 |
| vision_hidden_size | int | specify hidden size of vision projector | 指定vision projector的hidden size |
| vision_tower | Callable[[torch.Tensor], torch.Tensor] | specify the output of the vision model | 用于指定视觉模型的输出 |



## Demo
``` python
import torch
from llm_model *

def get_model_config(long_context = False):
    # max_position_embeddings: 512 -> 2048
    max_position_embeddings = 2048 if long_context else 512
    original_max_position_embeddings = 512 if long_context else None
    rope_type = 'yarn' if long_context else 'default'

    return ModelConfig(
        vocab_size=TrainerTools().tokenizer.vocab_size,
        hidden_size=768,
        intermediate_size=2048,
        moe_intermediate_size=1024,
        moe_n_dense_layer=1,
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
