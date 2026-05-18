from typing import Optional, Callable
from dataclasses import dataclass, field
import torch


@dataclass(kw_only=True)
class RoPEConfig:
    """
    RoPE (Rotary Position Embedding) 配置，用于定义旋转位置编码的类型及相关的缩放参数。

    Args:
        rope_type (`str`, 默认 'default'): RoPE 的类型。支持 `default` (标准实现)、`dynamic` (动态 NTK 缩放)、`yarn` (YaRN 缩放)。
        rope_theta (`float`, 默认 10000.0): RoPE 的基础频率（theta）参数，常用值如 10000、500000、1000000 等。
        factor (`float`, 默认 1.0): 线性缩放因子。用于外推/内插以扩展上下文长度，设定为 x 可以使模型处理“原始预训练长度 * x”倍的序列。
        partial_rotary_factor (`float`, 默认 1.0): 旋转位置编码应用的维度比例。1.0 表示在全部 head_dim 上应用，0.5 表示仅对一半维度进行旋转（类似 GPT-NeoX 的机制）。
        beta_fast (`float`, 默认 32): 仅在 `yarn` 模式下使用。外推（extrapolation）线性渐变函数的边界参数。
        beta_slow (`float`, 默认 1): 仅在 `yarn` 模式下使用。内插（interpolation）线性渐变函数的边界参数。
        mscale (`Optional[float]`, 默认 None): 仅在 `yarn` 模式下使用。用于注意力温度缩放乘数（attention scaling）。
        mscale_all_dim (`Optional[float]`, 默认 None): 仅在 `yarn` 模式下使用。针对全部维度的额外缩放配置。
        attention_factor (`Optional[float]`, 默认 None): 仅在 `yarn` 模式下使用。应用在注意力计算时的缩放因子，未指定时会通过 `factor` 等字段自动推断推荐值。
    """
    rope_type: str = 'default'
    rope_theta: float = 10000.0
    factor: float = 1.0
    partial_rotary_factor: float = 1.0
    beta_fast: float = 32
    beta_slow: float = 1
    mscale: Optional[float] = None
    mscale_all_dim: Optional[float] = None
    attention_factor: Optional[float] = None


@dataclass(kw_only=True)
class MoEConfig:
    """
    MoE (Mixture of Experts) 混合专家配置，用于定义稀疏专家模型的结构、路由机制及损失函数计算细节。

    Args:
        intermediate_size (`Optional[int]`, 默认 None): 单个 MoE 专家的中间隐藏层（FFN）维度大小。
        n_dense_layer (`Optional[int]`, 默认 None): 前 N 层保持为 Dense（密集）层，从第 N 层起才转换为 MoE 层。
        num_experts_per_tok (`Optional[int]`, 默认 None): 每个 token 激活的专家数量（即 Top-K 路由配置）。为 None 时表示标准 Dense 模型。
        n_shared_experts (`Optional[int]`, 默认 None): 共享专家的数量。这些专家会在所有 token 上激活，其参数量等效为 `intermediate_size * n_shared_experts`。
        n_routed_experts (`Optional[int]`, 默认 None): 参与动态路由决策的专家总数。
        routed_scaling_factor (`float`, 默认 1.0): 对经过路由的专家输出所乘以的标量缩放因子。
        seq_aux (`bool`, 默认 True): 决定 Auxiliary Loss（负载均衡辅助损失）的计算层级。True 意味着在序列级别进行统计计算，False 意味着打平为一个大 batch 计算。
        norm_topk_prob (`bool`, 默认 False): 是否将选出的 Top-K 专家的路由权重进行归一化，使得它们相加等于 1。
        aux_loss_coef (`float`, 默认 1e-3): 负载均衡辅助损失（Load Balancing Loss）乘以的系数因子。
        z_loss_coef (`float`, 默认 1e-4): Z-Loss 系数因子。用于鼓励 router logits 接近零以避免数值溢出，提高大规模集群训练时的稳定性。
        router_jitter_noise (`float`, 默认 0.01): 训练期间加入到 router logits 的均匀分布噪声幅度（Uniform [-noise, noise]），用于提升路由分发的均衡性。
        capacity_factor (`float`, 默认 1.25): 专家的容量因子。单个专家的最大容纳量计算为 `ceil(capacity_factor * tokens / experts_per_rank)`。
        drop_tokens (`bool`, 默认 False): 当某专家接收的 Token 数量超过容量上限时，是否直接丢弃超额的 Token (将其设为 padding)。
    """
    intermediate_size: Optional[int] = None
    n_dense_layer: Optional[int] = None
    num_experts_per_tok: Optional[int] = None
    n_shared_experts: Optional[int] = None
    n_routed_experts: Optional[int] = None
    routed_scaling_factor: float = 1.0
    seq_aux: bool = True
    norm_topk_prob: bool = False
    aux_loss_coef: float = 1e-3
    z_loss_coef: float = 1e-4
    router_jitter_noise: float = 0.01
    capacity_factor: float = 1.25
    drop_tokens: bool = False


@dataclass(kw_only=True)
class AttnResConfig:
    """
    Attention Residuals 注意力残差配置

    Args:
        num_blocks (`int`, 默认 8): 划分模型的 block 块总数。模型将按照 `num_hidden_layers // num_blocks` 为一组，按块执行残差聚合。
    """
    num_blocks: int = 8


@dataclass(kw_only=True)
class Config:
    """
    LLM (Large Language Model) 基础模型配置参数。

    Args:
        vocab_size (`int`): 模型的词表大小。
        hidden_size (`int`): 模型的隐藏层特征维度（d_model）。
        intermediate_size (`int`): 密集前馈神经网络（MLP）的中间层维度。
        num_hidden_layers (`int`): Transformer 的解码层（Decoder Layer）总数。
        num_attention_heads (`int`): QKV 注意力机制的 Query 头数量。
        num_key_value_heads (`Optional[int]`, 默认 None): Key 和 Value 头的数量。
            若等于 num_attention_heads 则是 MHA (Multi-Head Attention)；
            若等于 1 则是 MQA (Multi-Query Attention)；
            若大于 1 且小于 num_attention_heads 则是 GQA (Grouped-Query Attention)。
            当未指定时，默认等同于 num_attention_heads。
        max_position_embeddings (`int`): 模型的最大位置嵌入长度，通常为当前上下文窗口大小。
        original_max_position_embeddings (`Optional[int]`, 默认 None): 预训练时的原始最大上下文长度。常被各种 RoPE 扩展算法（如 YaRN）用来计算精确的降频倍率。
        attention_dropout (`float`, 默认 0.1): 注意力权重概率矩阵上的 Dropout 比例。
        attention_implementation (`str`, 默认 'auto'): 注意力算子的实现方案。
            `auto`: 自动判断，若 PyTorch 环境支持则用 F.scaled_dot_product_attention (sdpa)，否则用纯手写实现。
            `sdpa`: 强制使用官方 SDPA（支持 FlashAttention/内存优化等引擎）。
            `default`: 强制使用简单的矩阵相乘及 Softmax 原生实现。
        initializer_range (`float`, 默认 0.02): 权重初始化的正态分布标准差参数。
        use_qk_norm (`bool`, 默认 True): 是否在计算注意力权重之前对 Query 和 Key 执行一层 RMSNorm，通常可提升训练稳定性。
        tie_word_embeddings (`bool`, 默认 False): 是否将输入 Token Embedding 矩阵与输出层 LM Head 权重矩阵进行绑定（权重共享）。
        rope_config (`RoPEConfig`): 旋转位置编码详细配置实例。
        moe_config (`Optional[MoEConfig]`, 默认 None): MoE 的详细配置，不传入时默认这是一个 Dense 稠密模型。
        attn_res_config (`Optional[AttnResConfig]`, 默认 None): 块残差注意力的细粒度控制配置。
    """
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: Optional[int] = None
    max_position_embeddings: int
    original_max_position_embeddings: Optional[int] = None
    attention_dropout: float = 0.1
    attention_implementation: str = 'auto'
    initializer_range: float = 0.02
    use_qk_norm: bool = True
    tie_word_embeddings: bool = False
    rope_config: RoPEConfig = field(default_factory=RoPEConfig)
    moe_config: Optional[MoEConfig] = None
    attn_res_config: Optional[AttnResConfig] = None

    def __post_init__(self):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads


@dataclass(kw_only=True)
class VLMConfig(Config):
    """
    VLM (Vision-Language Model) 多模态配置，继承自 LLM Config 并扩展了处理视觉特征的特定参数。

    Args:
        image_tok (`int`): 特殊 Token 的 ID，代表文本序列中图像序列被安放和替换的占位符位置（例如 `<image>` 的 token_id）。
        image_size (`int`): 视觉模型的输入图像尺寸（通常意味着正方形尺寸，如 336 表示 336x336）。
        patch_size (`int`): 视觉模型将图像切分为块 (Patch) 的大小（如 14 代表 14x14 像素）。
        tokens_per_image (`int`): 经过池化和 Projector 后，单张图片等价于接入到大模型内部的最终 Token 数量。
        vision_hidden_size (`int`): 视觉特征提取塔（Vision Tower）最后一层的输出隐藏维度大小。
        vision_tower (`Callable[[torch.Tensor], torch.Tensor]`): 视觉处理网络本身。入参为处理好的 pixel_values Tensor，输出视觉表征 Tensor。
    """
    image_tok: int
    image_size: int
    patch_size: int
    tokens_per_image: int
    vision_hidden_size: int
    vision_tower: Callable[[torch.Tensor], torch.Tensor]

