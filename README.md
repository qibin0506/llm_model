# 🚀 Modern LLM & VLM Modular Framework

一个基于 PyTorch 构建的高性能、模块化大语言模型（LLM）与视觉语言模型（VLM）研发框架。支持 **Softmax Attention (GQA/MQA)**、**Gated DeltaNet (线性注意力)**、**混合注意力架构 (Hybrid Architecture)**、**Sparse MoE (混合专家系统)**、**Block Attention Residuals (BlockAttnRes)** 以及 **多种 RoPE 上下文扩展算法（YaRN / Dynamic NTK）**。

---

## 目录
- [✨ 核心特性](#-核心特性)
- [📦 环境依赖与安装](#-环境依赖与安装)
- [第 1 章：快速开始与基础 LLM 推理](#第-1-章快速开始与基础-llm-推理)
- [第 2 章：注意力机制与混合架构 (Hybrid Attention)](#第-2-章注意力机制与混合架构-hybrid-attention)
- [第 3 章：稀疏混合专家系统 (Sparse MoE)](#第-3-章稀疏混合专家系统-sparse-moe)
- [第 4 章：位置编码 (RoPE) 与长上下文扩展](#第-4-章位置编码-rope-与长上下文扩展)
- [第 5 章：块级注意力残差 (Block Attention Residuals)](#第-5-章块级注意力残差-block-attention-residuals)
- [第 6 章：多模态视觉语言模型 (VLM)](#第-6-章多模态视觉语言模型-vlm)
- [第 7 章：模型训练与显存优化](#第-7-章模型训练与显存优化)
- [📖 附录：全量配置参数参考手册](#-附录全量配置参数参考手册)

---

## ✨ 核心特性

1. **灵活的注意力机制**：
   - 支持标准 Softmax Attention（支持 GQA/MQA、QK-Norm 以及门控输出 Gated Attention）。
   - 内置 **Gated DeltaNet** 线性注意力机制（结合 1D 短卷积，支持 Flash-Linear-Attention 加速引擎及原生 PyTorch 循环降级实现）。
   - **Hybrid 混合架构**：支持按比例交替堆叠 Gated DeltaNet 与 Softmax Attention（如 3:1 混合）。
2. **高性能 Sparse MoE (Mixture of Experts)**：
   - 支持 Top-K 路由、共享专家 (Shared Experts)、路由噪声 (Jitter Noise)、容量限制 (Capacity Factor) 及 Token 丢弃策略。
   - 包含序列级/全局负载均衡损失 (Auxiliary Loss) 与 Router Z-Loss。
3. **先进的残差设计**：
   - 支持 **Block Attention Residuals (BlockAttnRes)**，打破传统层级残差局限，提高深度网络的梯度流动与表征能力。
4. **长上下文支持 (RoPE Variants)**：
   - 内置 Default、Dynamic NTK 缩放以及 **YaRN** (Yet Another RoPE Extrapolation) 算法。
5. **多模态 VLM 原生集成**：
   - 结合视觉塔 (Vision Tower) 与 2D 平均池化 Projector，支持图像 Token 动态替换与图文混合输入。
6. **高效 KV / Recurrent Cache 管理器**：
   - 统一管理 Transformer 的 KV Cache 和 RNN/线性注意力的 Recurrent State。

---

## 📦 环境依赖与安装

- Python >= 3.9
- PyTorch >= 2.0 (推荐 PyTorch >= 2.3 以获得最佳 SDPA 支持)
- `packaging`

*(可选组件)* 若需要启用 Gated DeltaNet 极速算子加速，请安装 `flash-linear-attention`：
```bash
pip install flash-linear-attention
```

---

## 第 1 章：快速开始与基础 LLM 推理

你可以通过 `ModelConfig` 定义模型维度，并实例化 `LlmModel` 进行文本生成与自回归推理。

### 1.1 构建 Dense 模型并进行 Forward 演示

```python
import torch
from model_config import Config
from llm_model import LlmModel

# 1. 定义模型配置
config = Config(
    vocab_size=32000,
    hidden_size=2048,
    intermediate_size=5632,
    num_hidden_layers=16,
    num_attention_heads=16,
    num_key_value_heads=4, # 支持 GQA
    max_position_embeddings=4096,
)

# 2. 实例化模型
model = LlmModel(config).cuda()

# 3. 前向传播
input_ids = torch.randint(0, 32000, (2, 128), device="cuda") # (batch_size, seq_len)
outputs = model(input_ids)

logits = outputs['logits'] # (2, 128, 32000)
print("Logits shape:", logits.shape)
```

### 1.2 使用 KVCache 进行自回归增量生成

框架提供了统一的 `KVCache` 类，自动管理预填阶段 (Prefill) 和解码阶段 (Decoding) 的状态。

```python
from kv_cache import KVCache

# 初始化 KV Cache
kv_cache = KVCache(max_capacity=2048) # 可选指定最大容量

# Phase 1: Prefill 阶段 (输入 Prompt)
prompt_ids = torch.randint(0, 32000, (1, 32), device="cuda")
outputs = model(prompt_ids, past_key_values=kv_cache, use_cache=True)
next_token = outputs['logits'][:, -1:].argmax(dim=-1)

# Phase 2: Decoding 阶段 (逐 Token 生成)
generated_tokens = [next_token]
for _ in range(10):
    outputs = model(next_token, past_key_values=kv_cache, use_cache=True)
    next_token = outputs['logits'][:, -1:].argmax(dim=-1)
    generated_tokens.append(next_token)

print("Generation completed. KV Cache length:", kv_cache.get_seq_len())
```

---

## 第 2 章：注意力机制与混合架构 (Hybrid Attention)

本框架提供了丰富的注意力机制实现，涵盖标准 Softmax Attention、线性注意力 Gated DeltaNet 以及两者的混合架构。

### 2.1 Softmax Attention 特性

1. **GQA / MQA**：通过设置 `num_key_value_heads` 轻松开启。
2. **QK-Norm**：默认开启（`use_qk_norm=True`），在 Query 和 Key 计算点积前施加 RMSNorm，显著提升超大模型训练稳定性。
3. **Gated Attention**：设置 `hybrid_softmax_gated=True` 可开启门控注意力（产生 Gate 标量控流）。

### 2.2 Gated DeltaNet 线性注意力

Gated DeltaNet 结合了状态空间模型（SSM）的线性复杂度与 Delta Rule 记忆更新规则，并引入 **1D 短卷积 (Short Convolution)** 补充局部时序感知。

```python
from model_config import Config, GatedDeltaNetConfig

# 纯 Gated DeltaNet 模型配置
config = Config(
    vocab_size=32000,
    hidden_size=2048,
    intermediate_size=5632,
    num_hidden_layers=12,
    num_attention_heads=16,
    max_position_embeddings=4096,
    
    attention_type='gated_deltanet', # 设置注意力类型为 gated_deltanet
    gated_deltanet_implementation='auto', # 'fla' (极速算子) 或 'default' (原生 PyTorch)
    gated_deltanet_config=GatedDeltaNetConfig(
        use_short_conv=True,
        conv_kernel_size=4
    )
)
```

### 2.3 Hybrid 混合架构

混合架构在同一模型中交替堆叠 Gated DeltaNet（负责长序列高效处理）和 Softmax Attention（负责精密上下文检索）。

```python
config = Config(
    vocab_size=32000,
    hidden_size=2048,
    intermediate_size=5632,
    num_hidden_layers=16,
    num_attention_heads=16,
    max_position_embeddings=4096,
    
    attention_type='hybrid', # 开启混合模式
    hybrid_ratio='3:1',       # 每 3 层 Gated DeltaNet 堆叠 1 层 Softmax Attention
    
    # 针对 Hybrid 中 Softmax 层的独立参数配置（可选）
    hybrid_softmax_head_dim=128,
    hybrid_softmax_num_heads=16,
    hybrid_softmax_num_kv_heads=4,
    hybrid_softmax_gated=True
)
```

---

## 第 3 章：稀疏混合专家系统 (Sparse MoE)

通过集成 `MoEConfig`，可以将模型中的 MLP 层替换为 Sparse MoE 模块，大幅增加模型参数量同时保持恒定的计算复杂度 (FLOPs)。

### 3.1 MoE 核心配置示例

```python
from model_config import Config, MoEConfig

moe_config = MoEConfig(
    intermediate_size=1408,       # 单个专家的 FFN 中间层维度
    n_routed_experts=64,          # 路由专家总数
    num_experts_per_tok=6,        # Top-K 路由激活的专家数
    n_shared_experts=2,           # 共享专家数 (所有 Token 必选)
    n_dense_layer=2,              # 前 2 层保持 Dense 层，第 3 层起转为 MoE
    routed_scaling_factor=1.0,    # 路由专家输出缩放因子
    norm_topk_prob=True,          # 对 Top-K 权重进行归一化
    aux_loss_coef=1e-3,           # 负载均衡辅助损失系数
    z_loss_coef=1e-4,             # Router Z-Loss 系数 (防止 Logits 溢出)
    capacity_factor=1.25,         # 专家容量因子
    drop_tokens=False             # 溢出时是否丢弃 Token
)

config = Config(
    vocab_size=32000,
    hidden_size=2048,
    intermediate_size=5632,
    num_hidden_layers=16,
    num_attention_heads=16,
    max_position_embeddings=4096,
    moe_config=moe_config
)

model = LlmModel(config).cuda()
```

### 3.2 MoE 训练损失返回

在训练阶段，若开启 MoE，`outputs['aux_loss']` 会自动返回 Router 负载均衡损失与 Z-Loss 的和，需将其叠加到总 Loss 中：

```python
outputs = model(input_ids)
logits = outputs['logits']
aux_loss = outputs['aux_loss'] # 获取 MoE 辅助损失

# 计算交叉熵主损失
main_loss = compute_cross_entropy(logits, labels)

# 汇总训练损失
total_loss = main_loss + aux_loss
total_loss.backward()
```

---

## 第 4 章：位置编码 (RoPE) 与长上下文扩展

框架在 `rope.py` 中实现了三种旋转位置编码策略，可通过 `RoPEConfig` 无缝切换：

### 4.1 位置编码类型

1. **`default`**：标准 RoPE 实现。
2. **`dynamic` (Dynamic NTK)**：推理时根据实际输入序列长度动态调整 base 频率，无需微调即可外推上下文。
3. **`yarn` (YaRN)**：通过高频外推、低频内插与注意力温度缩放，实现超长上下文（如 32k/128k）扩展。

### 4.2 配置 YaRN 示例

```python
from model_config import Config, RoPEConfig

rope_config = RoPEConfig(
    rope_type='yarn',
    rope_theta=10000.0,
    factor=8.0,             # 扩展倍率：将上下文扩展 8 倍
    beta_fast=32,
    beta_slow=1,
    mscale=1.0,
    mscale_all_dim=1.0
)

config = Config(
    vocab_size=32000,
    hidden_size=2048,
    intermediate_size=5632,
    num_hidden_layers=16,
    num_attention_heads=16,
    max_position_embeddings=32768,          # 扩展后的目标长度
    original_max_position_embeddings=4096,  # 原始预训练长度
    rope_config=rope_config
)
```

---

## 第 5 章：块级注意力残差 (Block Attention Residuals)

传统的 Residual Connection 是简单的 $h_{l+1} = h_l + f(h_l)$。本框架实现了 **Block Attention Residuals (BlockAttnRes)**，它将网络划分成若干 Block，在 Block 内部通过 Softmax 动态注意力机制对历史上所有 Block 的表征进行加权聚合。

```python
from model_config import Config, AttnResConfig

config = Config(
    vocab_size=32000,
    hidden_size=2048,
    intermediate_size=5632,
    num_hidden_layers=16,
    num_attention_heads=16,
    max_position_embeddings=4096,
    
    # 启用 AttnRes，将 16 层划分为 4 个 Block (每个 Block 4 层)
    attn_res_config=AttnResConfig(num_blocks=4)
)
```

*(注意：`num_hidden_layers` 必须能够被 `attn_res_config.num_blocks` 整除)*。

---

## 第 6 章：多模态视觉语言模型 (VLM)

`VlmModel` 继承自 `LlmModel`，内置了多模态投射器 (Projector) 和图像 Token 动态替换逻辑。

### 6.1 VLM 模型定义与前向传播

```python
import torch
from torch import nn
from model_config import VLMConfig
from vlm_model import VlmModel

# 1. 模拟定义一个 Vision Tower (如 CLIP / SigLIP)
class DummyVisionTower(nn.Module):
    def __init__(self):
        super().__init__()
        # 输出形状: (batch_size, num_patches, vision_hidden_size)
    def forward(self, pixel_values):
        bsz = pixel_values.shape[0]
        return torch.randn(bsz, 576, 1024, device=pixel_values.device)

vision_tower = DummyVisionTower()

# 2. 配置 VLM
vlm_config = VLMConfig(
    vocab_size=32000,
    hidden_size=2048,
    intermediate_size=5632,
    num_hidden_layers=16,
    num_attention_heads=16,
    max_position_embeddings=4096,
    
    # 多模态专用参数
    image_tok=31999,          # <image> 占位符 Token ID
    image_size=336,           # 图像分辨率 336x336
    patch_size=14,            # Patch 大小 14x14 (共 (336/14)^2 = 576 patches)
    tokens_per_image=144,     # 投射器池化降采样后的最终图像 Token 数量 (12x12)
    vision_hidden_size=1024,  # Vision Tower 维度
    vision_tower=vision_tower
)

# 3. 实例化 VLM
vlm_model = VlmModel(vlm_config).cuda()

# 4. 前向传播
# 假设输入文本中包含 144 个连续的 image_tok 占位符
input_ids = torch.randint(0, 31000, (1, 200), device="cuda")
input_ids[0, 10:154] = vlm_config.image_tok # 插入 144 个图像占位符

pixel_values = torch.randn(1, 3, 336, 336, device="cuda") # (num_images, C, H, W)

outputs = vlm_model(input_ids=input_ids, pixel_values=pixel_values)
print("VLM Output Logits shape:", outputs['logits'].shape)
```

---

## 第 7 章：模型训练与显存优化

### 7.1 梯度检查点 (Gradient Checkpointing)

激活梯度检查点可以大幅降低大模型训练时的显存占用：

```python
model = LlmModel(config)

# 开启梯度检查点
model.gradient_checkpointing_enable()

# 若需要自定义 PyTorch torch.utils.checkpoint 方法：
# model.gradient_checkpointing_enable(custom_checkpoint_func)

# 禁用梯度检查点
# model.gradient_checkpointing_disable()
```

### 7.2 Padding Mask 与因果掩码

在前向传播时传入 `attention_mask`（以 `1` 表示真实 Token，`0` 表示 Padding）：

```python
attention_mask = torch.tensor([
    [1, 1, 1, 1, 0, 0],
    [1, 1, 1, 1, 1, 1]
], device="cuda")

outputs = model(input_ids=input_ids, attention_mask=attention_mask)
```

框架内部会自动将其转换为兼容 SDPA 或原生 Softmax 的四维因果扩展掩码。

---

## 📖 附录：全量配置参数参考手册

### A.1 主配置参数 (`Config`)

| 参数名 | 类型 | 默认值 | 说明 |
| :--- | :--- | :--- | :--- |
| `vocab_size` | `int` | **必填** | 模型词表大小。 |
| `hidden_size` | `int` | **必填** | 模型的隐藏层特征维度 ($d_{model}$)。 |
| `intermediate_size` | `int` | **必填** | 密集前馈神经网络 (MLP) 的中间层扩展维度。 |
| `num_hidden_layers` | `int` | **必填** | Decoder 层的总套数。 |
| `num_attention_heads` | `int` | **必填** | Query (Q) 注意力头数。 |
| `num_key_value_heads` | `Optional[int]` | `None` | Key/Value (KV) 头数。若未设置则等于 `num_attention_heads` (MHA)；若为 1 则为 MQA；介于两者之间为 GQA。 |
| `head_dim` | `Optional[int]` | `None` | 单个头的特征维度。未指定时默认为 `hidden_size // num_attention_heads`。 |
| `max_position_embeddings` | `int` | **必填** | 模型的最大位置上下文窗口长度。 |
| `original_max_position_embeddings` | `Optional[int]` | `None` | 预训练时的原始最大上下文长度 (用于 YaRN 等缩放算法)。 |
| `attention_dropout` | `float` | `0.0` | 注意力权重 Dropout 比例。 |
| `attention_implementation` | `str` | `'auto'` | 注意力算子实现方案：`'auto'`、`'sdpa'` (PyTorch 官方) 或 `'default'` (原生 PyTorch 实现)。 |
| `initializer_range` | `float` | `0.02` | 线性层与嵌入层权重的初始化标准差。 |
| `use_qk_norm` | `bool` | `True` | 是否在 Q, K 点积计算前应用 RMSNorm 归一化。 |
| `norm_eps` | `float` | `1e-6` | RMSNorm 层极小值防止除零 epsilon。 |
| `tie_word_embeddings` | `bool` | `False` | 是否共享输入 Embedding 矩阵与输出 LM Head 的权重。 |
| `attention_qkv_bias` | `bool` | `False` | 注意力 Q, K, V 投影线性层是否启用 Bias。 |
| `attention_out_bias` | `bool` | `False` | 注意力 Out 投影线性层是否启用 Bias。 |
| `mlp_bias` | `bool` | `False` | MLP 层线性变换是否启用 Bias。 |
| `lm_head_bias` | `bool` | `False` | 语言模型输出头 (LM Head) 是否启用 Bias。 |
| `rope_config` | `RoPEConfig` | `RoPEConfig()`| RoPE 旋转位置编码控制配置。 |
| `moe_config` | `Optional[MoEConfig]`| `None` | MoE 混合专家配置。未指定时模型为 Dense 稠密模型。 |
| `attn_res_config` | `Optional[AttnResConfig]`| `None` | Block Attention Residuals 块注意力残差配置。 |
| `attention_type` | `str` | `'softmax'` | 注意力模式类型：`'softmax'`、`'gated_deltanet'` 或 `'hybrid'`。 |
| `gated_deltanet_implementation` | `str` | `'auto'` | Gated DeltaNet 算子实现：`'auto'`、`'fla'` 或 `'default'`。 |
| `hybrid_ratio` | `str` | `"3:1"` | 混合架构层比例，格式为 `"x:y"`，代表 `x` 层 Gated DeltaNet 与 `y` 层 Softmax 交替堆叠。 |
| `gated_deltanet_config` | `Optional[GatedDeltaNetConfig]` | `GatedDeltaNetConfig()` | Gated DeltaNet 的详细参数。 |
| `hybrid_softmax_head_dim` | `Optional[int]` | `None` | Hybrid 模式下 Softmax 注意力层的独立 Head 维度。 |
| `hybrid_softmax_num_heads` | `Optional[int]` | `None` | Hybrid 模式下 Softmax 注意力层的独立 Q 头数。 |
| `hybrid_softmax_num_kv_heads` | `Optional[int]` | `None` | Hybrid 模式下 Softmax 注意力层的独立 KV 头数。 |
| `hybrid_softmax_gated` | `bool` | `False` | Hybrid 模式下 Softmax 注意力层是否开启输出 Sigmoid 门控控流。 |

---

### A.2 位置编码配置 (`RoPEConfig`)

| 参数名 | 类型 | 默认值 | 说明 |
| :--- | :--- | :--- | :--- |
| `rope_type` | `str` | `'default'` | RoPE 算法类型：`'default'`、`'dynamic'` (Dynamic NTK) 或 `'yarn'`。 |
| `rope_theta` | `float` | `10000.0` | RoPE 的基频 (Base Frequency) 参数。 |
| `factor` | `float` | `1.0` | 位置编码的线性扩展缩放系数。 |
| `partial_rotary_factor` | `float` | `1.0` | 旋转位置编码施加的 Head 维度比例 ($1.0$ 为全维度， $0.5$ 为一半维度)。 |
| `beta_fast` | `float` | `32` | 仅在 `yarn` 模式下使用：高频外推渐变边界参数。 |
| `beta_slow` | `float` | `1` | 仅在 `yarn` 模式下使用：低频内插渐变边界参数。 |
| `mscale` | `Optional[float]` | `None` | 仅在 `yarn` 模式下使用：注意力温度缩放乘数。 |
| `mscale_all_dim` | `Optional[float]` | `None` | 仅在 `yarn` 模式下使用：针对全维度的额外缩放乘数。 |
| `attention_factor` | `Optional[float]` | `None` | 仅在 `yarn` 模式下使用：显式指定的注意力计算缩放因子。 |

---

### A.3 混合专家系统配置 (`MoEConfig`)

| 参数名 | 类型 | 默认值 | 说明 |
| :--- | :--- | :--- | :--- |
| `intermediate_size` | `Optional[int]` | `None` | 单个路由专家的 MLP 中间层维度。 |
| `n_dense_layer` | `Optional[int]` | `None` | 前 `N` 层保持 Dense 稠密层，从第 `N+1` 层起才转换为 MoE 层。 |
| `num_experts_per_tok` | `Optional[int]` | `None` | 每个 Token 路由选择激活的专家数量 (Top-K)。 |
| `n_shared_experts` | `Optional[int]` | `None` | 共享专家数量 (所有 Token 无条件激活，不参与 Top-K 竞争)。 |
| `n_routed_experts` | `Optional[int]` | `None` | 参与动态路由选拔的专家总总数。 |
| `routed_scaling_factor` | `float` | `1.0` | 路由专家输出叠加时的标量乘法缩放因子。 |
| `seq_aux` | `bool` | `True` | 辅助损失计算维度：`True` 表示按序列 (Sequence Level) 独立计算并取均值，`False` 表示按大 Batch 统一计算。 |
| `norm_topk_prob` | `bool` | `False` | 是否对选出的 Top-K 专家的 Router Prob 重新归一化至和为 1。 |
| `aux_loss_coef` | `float` | `1e-3` | 负载均衡辅助损失 (Load Balancing Aux Loss) 系数。 |
| `z_loss_coef` | `float` | `1e-4` | Router Z-Loss 系数 (抑制 Router Logits 幅度)。 |
| `router_jitter_noise` | `float` | `0.01` | 训练期间加入到 Router Logits 的均匀分布噪声幅度，促使分流均衡。 |
| `capacity_factor` | `float` | `1.25` | 专家容量上限乘数系数。 |
| `drop_tokens` | `bool` | `False` | 当到达专家的 Token 数超出容量上限时，是否丢弃超额 Token。 |

---

### A.4 Gated DeltaNet 配置 (`GatedDeltaNetConfig`)

| 参数名 | 类型 | 默认值 | 说明 |
| :--- | :--- | :--- | :--- |
| `use_short_conv` | `bool` | `True` | 是否在 Q, K, V 投影后应用 1D 短因果卷积。 |
| `conv_kernel_size` | `int` | `4` | 1D 短因果卷积的 Kernel Size。 |
| `conv_bias` | `bool` | `False` | 1D 短因果卷积层是否启用 Bias 项。 |

---

### A.5 注意力残差配置 (`AttnResConfig`)

| 参数名 | 类型 | 默认值 | 说明 |
| :--- | :--- | :--- | :--- |
| `num_blocks` | `int` | `8` | 划分模型的 Block 总块数，要求 `num_hidden_layers % num_blocks == 0`。 |

---

### A.6 多模态 VLM 配置 (`VLMConfig`)

*(继承自 `Config` 并添加以下特有参数)*

| 参数名 | 类型 | 默认值 | 说明 |
| :--- | :--- | :--- | :--- |
| `image_tok` | `int` | **必填** | 文本序列中代表图像位置的占位符 Token ID。 |
| `image_size` | `int` | **必填** | 输入 Vision Tower 的图像正方形分辨率 (如 `336`)。 |
| `patch_size` | `int` | **必填** | 视觉切片 Patch 分辨率 (如 `14`)。 |
| `tokens_per_image` | `int` | **必填** | 经过 Projector 降采样后，单张图片等价于接入大模型的 Token 数量。 |
| `vision_hidden_size` | `int` | **必填** | Vision Tower 最后一层输出特征的隐藏维度大小。 |
| `vision_tower` | `Callable` | **必填** | 视觉特征提取网络 PyTorch Module 实例。 |
