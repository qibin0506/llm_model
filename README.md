# LLM Model PyTorch

这是一个灵活且高效的 PyTorch 实现的大语言模型（LLM）和视觉语言模型（VLM）库。该项目旨在提供一个简洁的代码库，用于训练和推理现代 Transformer 模型，支持 **Mixture of Experts (MoE)**、**RoPE (YaRN/Dynamic)** 以及 **GQA/MQA** 等前沿技术。

## ✨ 核心特性 (Key Features)

基于源代码分析，本项目包含以下核心功能：

*   **先进的注意力机制**:

    *   支持 **GQA (Grouped Query Attention)** 和 **MQA (Multi-Query Attention)**，由 `num_key_value_heads` 参数控制。
    *   原生支持 PyTorch 的 `F.scaled_dot_product_attention` (Flash Attention) 以加速计算。
*   **位置编码 (Positional Embeddings)**:

    *   实现 **Rotary Position Embeddings (RoPE)**。
    *   支持 **YaRN (Yet another RoPE extension)** 用于长上下文外推。
    *   支持 **Dynamic NTK** 缩放。
*   **混合专家模型 (MoE)**:

    *   支持 **Sparse MoE** 架构。
    *   支持 **Shared Experts**（共享专家）与 Routed Experts（路由专家）结合的机制。
    *   包含负载均衡辅助损失 (Auxiliary Loss) 和 Z-Loss。
*   **多模态能力 (VLM)**:

    *   提供 `VlmModel`，支持自定义视觉塔 (Vision Tower)。
    *   实现多模态投影层 (Multi-Modal Projector)，支持 Patch Pooling。
*   **推理优化**:

    *   完整的 **KV Cache** 实现，支持高效的自回归解码。
    *   支持梯度检查点 (Gradient Checkpointing) 以节省训练显存。
 
*   **配套训练和推理框架**:
    *   [https://github.com/qibin0506/llm_trainer](https://github.com/qibin0506/llm_trainer)

## 🛠️ 安装 (Installation)

你可以通过 pip 直接安装：

``` Bash
pip3 install project_llm_model

```

或者从源码安装：

``` Bash
git clone https://github.com/qibin0506/llm_model.git
cd llm_model
python3 setup.py install

```

## 🚀 快速开始 (Quick Start)

### 1. 初始化 LLM 模型

``` Python
import torch
from llm_model import LlmModel, ModelConfig, RoPEConfig

# 配置一个简单的 LLM
config = ModelConfig(
    vocab_size=32000,
    hidden_size=4096,
    intermediate_size=11008,
    num_hidden_layers=32,
    num_attention_heads=32,
    num_key_value_heads=8,  # 使用 GQA (32/8 = 4 groups)
    max_position_embeddings=4096,
    rope_config=RoPEConfig(
        rope_type='yarn',   # 使用 YaRN 支持长文本
        rope_theta=10000.0,
        factor=8.0          # 扩展系数
    )
)

model = LlmModel(config)
print(model)

# 前向传播示例
input_ids = torch.randint(0, 32000, (1, 128))
output = model(input_ids)
print(f"Logits shape: {output['logits'].shape}")

```

### 2. 初始化 VLM 模型

``` Python
from llm_model import VlmModel, VLMConfig

def dummy_vision_tower(images):
    # 模拟视觉编码器输出: (batch, num_patches, vision_hidden)
    return torch.randn(images.shape[0], 256, 1024)

vlm_config = VLMConfig(
    vocab_size=32000,
    hidden_size=4096,
    intermediate_size=11008,
    num_hidden_layers=32,
    num_attention_heads=32,
    num_key_value_heads=32,
    max_position_embeddings=2048,
    # 视觉部分配置
    image_tok=32001,        # 特殊 Image Token ID
    image_size=336,
    patch_size=14,
    tokens_per_image=576,   # 投影后的 token 数量
    vision_hidden_size=1024,
    vision_tower=dummy_vision_tower
)

vlm_model = VlmModel(vlm_config)

```

## ⚙️ 配置说明 (Configuration)

### ModelConfig (LLM)

| **参数**                     | **类型** | **说明**                                          |
| :------------------------- | :----- | :---------------------------------------------- |
| `vocab_size`               | int    | 词表大小                                            |
| `hidden_size`              | int    | 隐藏层维度                                           |
| `intermediate_size`        | int    | MLP 中间层维度                                       |
| `num_hidden_layers`        | int    | Transformer 层数                                  |
| `num_attention_heads`      | int    | 注意力头数 (Query)                                   |
| `num_key_value_heads`      | int    | KV 头数 (用于 GQA/MQA)                              |
| `max_position_embeddings`  | int    | 最大序列长度                                          |
| `attention_implementation` | str    | Attention 实现: `auto`, `sdpa` (Flash), `default` |
| `use_qk_norm`              | bool   | 是否对 Q/K 进行 LayerNorm (推荐 True 以稳定训练)            |

### RoPEConfig (位置编码)

| **参数**                    | **类型** | **默认值**   | **说明**                                 |
| :------------------------ | :----- | :-------- | :------------------------------------- |
| `rope_type`               | str    | `default` | 可选: `default`, `dynamic` (NTK), `yarn` |
| `rope_theta`              | float  | 10000.0   | RoPE 的基频                               |
| `factor`                  | float  | 1.0       | 缩放系数 (用于外推)                            |
| `beta_fast` / `beta_slow` | float  | -         | YaRN 算法专用的插值参数                         |

### MoEConfig (混合专家)

当配置了 `moe_config` 时，模型会将 MLP 层替换为 MoE 层（在 `n_dense_layer` 层之后）。

| **参数**                | **类型** | **说明**                   |
| :-------------------- | :----- | :----------------------- |
| `num_experts_per_tok` | int    | 每个 Token 激活的专家数量 (Top-K) |
| `n_routed_experts`    | int    | 路由专家总数                   |
| `n_shared_experts`    | int    | 共享专家数量 (总是激活)            |
| `intermediate_size`   | int    | 单个专家的维度                  |
| `seq_aux`             | bool   | 是否计算序列级辅助损失              |

## 📂 项目结构

Plaintext

```
llm_model/
├── llm_model.py       # 核心模型实现 (LlmModel, DecoderLayer, Attention)
├── vlm_model.py       # 视觉语言模型扩展 (VlmModel)
├── sparse_moe.py      # MoE 路由与专家实现
├── rope.py            # 旋转位置编码 (RoPE, YaRN, Dynamic)
├── kv_cache.py        # 推理 KV 缓存管理
├── attention_masks.py # 因果与 Padding Mask 处理
└── model_config.py    # 配置类定义

```
