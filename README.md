# LLM Model PyTorch

这是一个灵活、轻量且高效的 PyTorch 大语言模型（LLM）和视觉语言模型（VLM）构建库。该项目旨在提供一个简洁的代码库，用于训练和推理现代 Transformer 模型，支持 **Mixture of Experts (MoE)**、**RoPE (YaRN/Dynamic)**、**GQA/MQA** 以及 **Block Attention Residuals** 等前沿技术。

## ✨ 核心特性 (Key Features)

本项目包含以下核心功能：

*   **先进的注意力机制**:
    *   支持 **GQA (Grouped Query Attention)** 和 **MQA (Multi-Query Attention)**，由 `num_key_value_heads` 参数精确控制。
    *   原生支持 PyTorch 的 `F.scaled_dot_product_attention` (Flash Attention) 以加速前向与反向计算，节省显存。
    *   支持跨层 **Block Attention Residuals (块残差注意力)**。
*   **位置编码 (Positional Embeddings)**:
    *   实现 **Rotary Position Embeddings (RoPE)**，支持部分维度旋转 (`partial_rotary_factor`)。
    *   支持 **YaRN (Yet another RoPE extension)** 及温度缩放 (`mscale`) 用于长上下文无损外推。
    *   支持 **Dynamic NTK** 动态缩放。
*   **混合专家模型 (Sparse MoE)**:
    *   支持 **Shared Experts**（共享专家）与 **Routed Experts**（路由专家）相结合的混合架构。
    *   完善的路由机制：支持 Top-K 路由、权重归一化 (`norm_topk_prob`)、Router Jitter 噪声注入。
    *   内置高级训练损失：支持 **Auxiliary Loss**（负载均衡辅助损失）和 **Z-Loss**（缓解 Logits 数值溢出）。
    *   支持限制专家容量 (`capacity_factor`) 及 Token 丢弃策略 (`drop_tokens`)。
*   **多模态能力 (VLM)**:
    *   提供 `VlmModel`，支持挂载自定义视觉提取塔 (Vision Tower)。
    *   内置 Multi-Modal Projector，支持 Patch Pooling 压缩图像 Token 数量。
*   **推理优化**:
    *   完整的 **KV Cache** 实现，支持预分配与动态扩容，加速自回归解码。
    *   支持 Gradient Checkpointing (梯度检查点) 以极大节省训练显存。
*   **配套训练和推理框架**:
    *   请访问：[https://github.com/qibin0506/llm_trainer](https://github.com/qibin0506/llm_trainer)

## 🛠️ 安装 (Installation)

你可以通过 pip 直接安装：

```bash
pip3 install project_llm_model
```

或者从源码安装：

```bash
git clone https://github.com/qibin0506/llm_model.git
cd llm_model
python3 setup.py install
```

## 🚀 快速开始 (Quick Start)

### 1. 初始化 LLM 模型 (含 MoE 与 YaRN)

```python
import torch
from llm_model import LlmModel, ModelConfig, RoPEConfig, MoEConfig, AttnResConfig

# 配置一个现代架构的 LLM
config = ModelConfig(
    vocab_size=32000,
    hidden_size=4096,
    intermediate_size=11008,
    num_hidden_layers=32,
    num_attention_heads=32,
    num_key_value_heads=8,  # 使用 GQA (32/8 = 4 groups)
    max_position_embeddings=4096,
    use_qk_norm=True,       # 开启 QK Norm 提升训练稳定性
    rope_config=RoPEConfig(
        rope_type='yarn',   # 使用 YaRN 支持长文本外推
        rope_theta=10000.0,
        factor=8.0          # 上下文扩展系数
    ),
    moe_config=MoEConfig(
        n_dense_layer=2,          # 前 2 层保持 Dense
        n_routed_experts=8,       # 8 个路由专家
        num_experts_per_tok=2,    # 激活 Top-2
        n_shared_experts=1,       # 1 个共享专家
        intermediate_size=2048    # MoE 专家维度
    ),
    attn_res_config=AttnResConfig(num_blocks=8) # 开启块残差注意力
)

model = LlmModel(config)
print(model)

# 前向传播示例
input_ids = torch.randint(0, 32000, (1, 128))
output = model(input_ids)
print(f"Logits shape: {output['logits'].shape}") # (1, 128, 32000)
```

### 2. 初始化 VLM 模型 (多模态)

```python
from llm_model import VlmModel, VLMConfig

def dummy_vision_tower(images):
    # 模拟视觉编码器输出: (batch, num_patches, vision_hidden_size)
    return torch.randn(images.shape[0], 256, 1024)

# VLMConfig 继承自 ModelConfig，包含所有 LLM 参数
vlm_config = VLMConfig(
    vocab_size=32000,
    hidden_size=4096,
    intermediate_size=11008,
    num_hidden_layers=32,
    num_attention_heads=32,
    max_position_embeddings=2048,
    
    # --- 视觉模态专属配置 ---
    image_tok=32001,        # 特殊 Image Token 的 ID (<image>)
    image_size=336,         # 输入图像分辨率
    patch_size=14,          # 图像切片大小 (336/14 = 24*24 = 576 patches)
    tokens_per_image=144,   # 经过 AvgPool 投影后送入 LLM 的最终 Token 数量
    vision_hidden_size=1024,# 视觉塔特征维度
    vision_tower=dummy_vision_tower
)

vlm_model = VlmModel(vlm_config)
```

## ⚙️ 配置字典详细说明 (Configuration)

### ModelConfig (核心 LLM 配置)
基础的 Transformer 模型架构配置。包含所有标准 Dense 模型所需的参数。

| **参数** | **类型** | **默认值** | **说明** |
| :--- | :--- | :--- | :--- |
| `vocab_size` | int | - | 模型的词表大小。 |
| `hidden_size` | int | - | 模型的隐藏层特征维度（d_model）。 |
| `intermediate_size` | int | - | 密集前馈神经网络（MLP）的中间层维度。 |
| `num_hidden_layers` | int | - | 解码层（Decoder Layer）总数。 |
| `num_attention_heads` | int | - | QKV 注意力机制的 Query 头数量。 |
| `num_key_value_heads` | Optional[int] | None | KV 头数量。等于 Q头数为 MHA，等于 1 为 MQA，否则为 GQA。None 表示等同于 Q头数。 |
| `max_position_embeddings` | int | - | 模型的最大位置嵌入长度，通常为当前上下文窗口大小。 |
| `original_max_position_embeddings`| Optional[int]| None | 预训练时的原始最大上下文长度，常被 RoPE 扩展算法（如 YaRN）用来计算精确倍率。 |
| `attention_dropout` | float | 0.1 | 注意力权重概率矩阵上的 Dropout 比例。 |
| `attention_implementation` | str | 'auto' | 算子实现：`auto` (自动选择), `sdpa` (强制 Flash Attention), `default` (原生实现)。 |
| `initializer_range` | float | 0.02 | 权重初始化的正态分布标准差参数。 |
| `use_qk_norm` | bool | True | 是否在计算注意力前对 Q 和 K 执行 RMSNorm，可显著提升训练稳定性。 |
| `tie_word_embeddings` | bool | False | 是否将 Input Embedding 与 LM Head 输出权重矩阵绑定（权重共享）。 |
| `rope_config` | RoPEConfig | 实例化 | 旋转位置编码详细配置实例。 |
| `moe_config` | Optional[MoEConfig]| None | MoE（混合专家）配置实例，不传入时默认是纯 Dense 稠密模型。 |
| `attn_res_config` | Optional[...] | None | 块残差注意力的细粒度控制配置。 |

### RoPEConfig (旋转位置编码)
控制 RoPE 及其长上下文扩展策略（NTK/YaRN）。

| **参数** | **类型** | **默认值** | **说明** |
| :--- | :--- | :--- | :--- |
| `rope_type` | str | 'default' | 可选: `default` (标准实现), `dynamic` (动态 NTK), `yarn` (YaRN 缩放)。 |
| `rope_theta` | float | 10000.0 | RoPE 的基础频率（theta）参数，常用 10000、500000 等。 |
| `factor` | float | 1.0 | 线性缩放因子。用于外推扩展序列长度。 |
| `partial_rotary_factor` | float | 1.0 | 旋转位置编码应用的维度比例（如 0.5 表示仅对一半维度旋转）。 |
| `beta_fast` | float | 32.0 | YaRN 专用的外推（extrapolation）线性渐变边界参数。 |
| `beta_slow` | float | 1.0 | YaRN 专用的内插（interpolation）线性渐变边界参数。 |
| `mscale` | Optional[float]| None | YaRN 专用的注意力温度缩放乘数。 |
| `mscale_all_dim` | Optional[float]| None | YaRN 专用，针对全部维度的额外缩放配置。 |
| `attention_factor`| Optional[float]| None | YaRN 专用，应用在注意力计算时的缩放因子，未指定时会自动推断。 |

### MoEConfig (混合专家配置)
配置传入 `ModelConfig.moe_config` 即可将 MLP 转为稀疏混合专家网络。

| **参数** | **类型** | **默认值** | **说明** |
| :--- | :--- | :--- | :--- |
| `intermediate_size` | Optional[int] | None | 单个 MoE 专家的内部隐藏层（FFN）维度大小。 |
| `n_dense_layer` | Optional[int] | None | 前 N 层保持为 Dense（密集）层，从第 N+1 层起才转换为 MoE 层。 |
| `num_experts_per_tok` | Optional[int] | None | 每个 Token 激活的路由专家数量（Top-K 路由）。 |
| `n_shared_experts` | Optional[int] | None | 共享专家数量（在所有 token 上强制激活计算）。 |
| `n_routed_experts` | Optional[int] | None | 参与动态路由决策的独立专家总数。 |
| `routed_scaling_factor` | float | 1.0 | 路由专家输出信号的统一标量放大系数。 |
| `seq_aux` | bool | True | 决定辅助损失计算层级：True 在序列级进行统计，False 整个 batch 打平计算。 |
| `norm_topk_prob` | bool | False | 是否将选出的 Top-K 专家的路由权重进行归一化（使得求和等于 1）。 |
| `aux_loss_coef` | float | 1e-3 | 负载均衡辅助损失（Load Balancing Loss）乘以的系数因子。 |
| `z_loss_coef` | float | 1e-4 | Z-Loss 系数，用于惩罚过大的 Router Logits 避免数值溢出。 |
| `router_jitter_noise` | float | 0.01 | 训练时注入到路由 Logits 中的均匀噪声幅度，提升分发均衡性。 |
| `capacity_factor` | float | 1.25 | 专家容量因子。专家最大容纳量为 `ceil(capacity_factor * tokens / experts)`。 |
| `drop_tokens` | bool | False | 当某专家接收的 Token 数量超过容量上限时，是否直接丢弃超载的 Token。 |

### VLMConfig (视觉语言模型配置)
多模态扩展配置，继承自 `ModelConfig`，用于初始化 `VlmModel`。

| **参数** | **类型** | **说明** |
| :--- | :--- | :--- |
| `image_tok` | int | 特殊 Token ID，代表文本序列中图像被安放的占位符位置（如 `<image>`）。 |
| `image_size` | int | 视觉模型的输入图像分辨率尺寸（如 336 代表 336x336）。 |
| `patch_size` | int | 视觉模型将图像切分为块 (Patch) 的大小（如 14）。 |
| `tokens_per_image` | int | 经过投影和池化后，单张图片实际转换为多少个进入 LLM 的 Token。 |
| `vision_hidden_size` | int | 视觉塔最后一层的隐藏特征维度。 |
| `vision_tower` | Callable | 视觉处理网络本身实例，输入为 Pixel Values，输出视觉表征 Tensor。 |

### AttnResConfig (注意力残差块配置)

| **参数** | **类型** | **默认值** | **说明** |
| :--- | :--- | :--- | :--- |
| `num_blocks` | int | 8 | 划分模型的 block 块总数，模型按照 `num_hidden_layers // num_blocks` 为一组按块执行残差聚合。 |

## 📂 项目结构

```text
llm_model/
├── llm_model.py       # 核心模型实现 (LlmModel, DecoderLayer, Attention)
├── vlm_model.py       # 视觉语言模型扩展 (VlmModel, MultiModalProjector)
├── sparse_moe.py      # MoE 路由计算、损失统筹与专家组实现
├── rope.py            # 旋转位置编码 (包含 YaRN, Dynamic NTK 算法推导)
├── kv_cache.py        # 预分配与动态更新的推理 KV Cache
├── attention_masks.py # Causal Mask 与 Padding 处理逻辑
└── model_config.py    # 包含详细注释的数据类定义 (Config)
```
