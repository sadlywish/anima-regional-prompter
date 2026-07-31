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
    if state.masks.shape[1] != x.shape[1]:
        # 采样期的 unet function wrapper（scripts/anima_regional_prompter.py 的
        # process_before_every_sampling）负责把 process() 塞入的 (N, 1) 占位
        # masks 重建成真实网格形状。若那个 wrapper 没有运行——例如另一个扩展
        # 覆盖了 model_function_wrapper 却没有链式保留（Forge 内置的
        # sd_forge_spectrum 就是这样，sorting_priority 排在本扩展之后）——
        # masks 就会一直停留在 (N, 1)。(1, 1, 1) 与 (b, TOK, h*d) 的广播乘法
        # 完全合法，结果恒为 0：所有正向行的 cross-attention 输出会静默变成
        # 全零，模型在没有任何文本条件的情况下出图，没有任何异常或警告。
        # 这里响亮报错，把静默错图变成一眼能定位根因的异常。
        raise RuntimeError(
            f"分区掩码形状不匹配：state.masks 形状为 {tuple(state.masks.shape)}，"
            f"但当前图像 token 数为 {x.shape[1]}（masks 的第二维应等于该 token 数）。"
            "这通常发生在采样期重建掩码的 unet function wrapper 没有运行——"
            "很可能是另一个扩展覆盖了 model_function_wrapper 而没有链式保留；"
            "请检查是否同时启用了 Spectrum 之类的扩展。"
        )

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


TRANSFORMER_OPTION_KEY = "anima_rp"
GRID_OPTION_KEY = "anima_rp_grid"

_PATCHED = False
_ORIG_FORWARD = None


def regional_forward(
    module, x, context, rope_emb, transformer_options, state, attention_op, orig_forward
) -> torch.Tensor:
    """按 cond/uncond 分流：正向行走分区注意力，负向行走原始（不分区）路径。

    Forge 采样时会把正向、负向条件拼在同一个 batch 里前向，
    transformer_options 里的 cond_indices / uncond_indices（来自
    backend/sampling/sampling_function.py 的 compute_cond_indices）标出
    哪些行是正向、哪些是负向。负向提示词不分区，直接用传入的 context
    交给原实现处理。

    output_proj 只在两条路径里各作用一次：regional_attention 内部已经
    做过投影，orig_forward 自带投影，这里不再重复投影。
    """
    options = transformer_options or {}
    cond_indices = options.get("cond_indices")
    uncond_indices = options.get("uncond_indices")

    # 两个键都缺失：无法区分正负向，保守地把全部行当正向处理
    if cond_indices is None and uncond_indices is None:
        return regional_attention(module, x, state, attention_op)

    cond_indices = list(cond_indices or [])
    uncond_indices = list(uncond_indices or [])

    if not cond_indices:
        # 纯负向批：一次性委托给原实现，不逐行处理
        return orig_forward(module, x, context, rope_emb, transformer_options)

    if not uncond_indices:
        # 纯正向批：全部行走分区路径
        return regional_attention(module, x, state, attention_op)

    out = torch.empty(
        x.shape[0], x.shape[1], module.output_proj.out_features,
        device=x.device, dtype=x.dtype,
    )
    out[cond_indices] = regional_attention(module, x[cond_indices], state, attention_op)
    out[uncond_indices] = orig_forward(
        module, x[uncond_indices], context[uncond_indices], rope_emb, transformer_options
    )
    return out


def patch_anima_attention() -> None:
    """给 backend.nn.anima.SelfCrossAttention.forward 打补丁，接入分区注意力。

    幂等：重复调用只生效一次。补丁后的 forward 未启用分区（transformer_options
    里没有 TRANSFORMER_OPTION_KEY）、self-attn、或 context 为 None 时，
    只做一次 dict 查找就委托回原始 forward，self-attn、非 ANIMA 模型、
    其他扩展的生成流程完全不受影响。

    Forge 的 import 放在函数体内部：lib_anima_rp/ 顶层不得依赖 Forge，
    只有真正调用本函数时才会触发这些 import，跑单元测试时不会命中。
    """
    global _PATCHED, _ORIG_FORWARD
    if _PATCHED:
        return

    from backend.attention import attention_function
    from backend.nn.anima import SelfCrossAttention

    _ORIG_FORWARD = SelfCrossAttention.forward

    def patched_forward(self, x, context=None, rope_emb=None, transformer_options=None):
        state = (transformer_options or {}).get(TRANSFORMER_OPTION_KEY)
        if state is None or self.is_SelfAttn or context is None:
            return _ORIG_FORWARD(self, x, context, rope_emb, transformer_options)

        def anima_attention_op(q, k, v):
            # 与真实 SelfCrossAttention.torch_attention_op 语义一致：
            # (b, s, h, d) -> (b, h, s, d) 后交给 attention_function；
            # skip_reshape=True 表示输入已是 (b, h, s, d)，默认
            # skip_output_reshape=False 时返回 (b, s, h*d)，正好满足
            # regional_attention 对 attention_op 的契约。
            # transformer_options 顺手透传：真实实现也会传（虽然目前各
            # attention_function 后端都用 **kwargs 吞掉、不会用到它）。
            return attention_function(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                q.shape[2], skip_reshape=True, transformer_options=transformer_options,
            )

        return regional_forward(
            self, x, context, rope_emb, transformer_options, state,
            anima_attention_op, _ORIG_FORWARD,
        )

    SelfCrossAttention.forward = patched_forward
    _PATCHED = True
