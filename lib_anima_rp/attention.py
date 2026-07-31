"""分区 cross-attention。设计文档 §7。

只依赖 torch。宿主模块 backend.nn.anima.SelfCrossAttention 的接口约定：
q_proj / q_norm / k_proj / k_norm / v_proj / v_norm / output_proj / n_heads / head_dim。

核心思路：q 只投影一次（图像 token 数量大，这一步最贵）；每个区域只重算
k/v 与 attention 结果，按逐 token 掩码加权累加；output_proj 因为无 bias，
`Σ mask_i · output_proj(a_i) == output_proj(Σ mask_i · a_i)`，可以从加权和
中外提到最后只做一次。
"""

from dataclasses import dataclass

import torch


@dataclass
class RPState:
    """采样期通过 transformer_options 传递的分区状态。

    可变（非 frozen）：网格尺寸要到实际前向时才知道，届时会把重建好的
    masks 写回本对象。
    """

    masks: torch.Tensor              # (N, TOK)
    region_contexts: torch.Tensor    # (N, L, context_dim)
    base_context: torch.Tensor | None  # (1, L, context_dim)
    base_weights: torch.Tensor       # (N,)


def _to_heads(tensor: torch.Tensor, n_heads: int, head_dim: int) -> torch.Tensor:
    """(b, s, h*d) → (b, s, h, d)，等价于 einops 的 rearrange。"""
    b, s, _ = tensor.shape
    return tensor.view(b, s, n_heads, head_dim)


def regional_attention(module, x, state: RPState, attention_op) -> torch.Tensor:
    """对全部输入行做分区注意力，返回 (b, TOK, x_dim)。

    参数：
        module: 提供 q_proj/q_norm/k_proj/k_norm/v_proj/v_norm/output_proj/
            n_heads/head_dim 属性的 cross-attention 模块（SelfCrossAttention 接口）。
        x: (b, TOK, x_dim) 图像 token。
        state: 分区状态，masks 的 TOK 维必须等于 x.shape[1]。
        attention_op: (q, k, v) -> out，q/k/v 形状 (b, s, h, d)，
            out 形状 (b, s, h*d)。

    q 只投影一次；每个区域只重算 k/v 与 attention；output_proj 因无 bias
    可以从加权和中外提，故只做一次。
    """
    n_heads, head_dim = module.n_heads, module.head_dim
    batch = x.shape[0]

    q = module.q_norm(_to_heads(module.q_proj(x), n_heads, head_dim))

    masks = state.masks.to(device=x.device, dtype=x.dtype)
    contexts = state.region_contexts.to(device=x.device, dtype=x.dtype)

    acc = None
    for index in range(contexts.shape[0]):
        ctx = contexts[index].unsqueeze(0).expand(batch, -1, -1)
        k = module.k_norm(_to_heads(module.k_proj(ctx), n_heads, head_dim))
        v = module.v_norm(_to_heads(module.v_proj(ctx), n_heads, head_dim))
        attended = attention_op(q, k, v)
        weighted = attended * masks[index].view(1, -1, 1)
        acc = weighted if acc is None else acc + weighted

    if state.base_context is not None:
        weights = state.base_weights.to(device=x.device, dtype=x.dtype)
        # 逐 token 的权重图：每个 token 按自己所属区域的 base_weights 混合，
        # 而非对所有区域使用同一个标量比例。
        weight_map = (masks * weights.view(-1, 1)).sum(dim=0).view(1, -1, 1)
        base_ctx = state.base_context.to(device=x.device, dtype=x.dtype)
        if base_ctx.shape[0] != batch:
            base_ctx = base_ctx.expand(batch, -1, -1)
        k = module.k_norm(_to_heads(module.k_proj(base_ctx), n_heads, head_dim))
        v = module.v_norm(_to_heads(module.v_proj(base_ctx), n_heads, head_dim))
        base_attended = attention_op(q, k, v)
        acc = (1.0 - weight_map) * acc + weight_map * base_attended

    return module.output_proj(acc)
