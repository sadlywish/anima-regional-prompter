"""分区注意力的数值正确性。设计文档 §11.1。

用一个与 backend/nn/anima.py SelfCrossAttention 同接口的假模块，
不依赖 Forge 即可验证。
"""

import os

import pytest
import torch
from einops import rearrange

from lib_anima_rp.attention import RPState, regional_attention

TOL = 1e-4


class FakeCrossAttention(torch.nn.Module):
    """复刻 SelfCrossAttention 的 cross-attn 路径，权重随机。"""

    def __init__(self, query_dim=32, context_dim=24, n_heads=4, head_dim=8):
        super().__init__()
        inner = n_heads * head_dim
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.q_proj = torch.nn.Linear(query_dim, inner, bias=False)
        self.q_norm = torch.nn.RMSNorm(head_dim, eps=1e-6)
        self.k_proj = torch.nn.Linear(context_dim, inner, bias=False)
        self.k_norm = torch.nn.RMSNorm(head_dim, eps=1e-6)
        self.v_proj = torch.nn.Linear(context_dim, inner, bias=False)
        self.v_norm = torch.nn.Identity()
        self.output_proj = torch.nn.Linear(inner, query_dim, bias=False)

    def reference_forward(self, x, context):
        """与 Anima 原实现等价的参照路径。"""
        q = self.q_proj(x)
        k = self.k_proj(context)
        v = self.v_proj(context)
        q, k, v = (
            rearrange(t, "b s (h d) -> b s h d", h=self.n_heads, d=self.head_dim)
            for t in (q, k, v)
        )
        q = self.q_norm(q)
        k = self.k_norm(k)
        v = self.v_norm(v)
        out = torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        )
        out = out.transpose(1, 2).reshape(x.shape[0], x.shape[1], -1)
        return self.output_proj(out)


def reference_attention_op(q, k, v):
    """(b, s, h, d) → (b, s, h*d)，与 Anima 的 torch_attention_op 语义一致。"""
    out = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    )
    return out.transpose(1, 2).reshape(q.shape[0], q.shape[1], -1)


def make_case(tokens=12, regions=2, ctx_len=5, batch=1, seed=0):
    torch.manual_seed(seed)
    module = FakeCrossAttention()
    x = torch.randn(batch, tokens, 32)
    contexts = torch.randn(regions, ctx_len, 24)
    return module, x, contexts


def test_single_region_equals_plain_attention():
    module, x, contexts = make_case(regions=1)
    masks = torch.ones(1, x.shape[1])
    state = RPState(
        masks=masks,
        region_contexts=contexts,
        base_context=None,
        base_weights=torch.zeros(1),
    )
    actual = regional_attention(module, x, state, reference_attention_op)
    expected = module.reference_forward(x, contexts[0].unsqueeze(0))
    assert torch.allclose(actual, expected, atol=TOL)


def test_region_tokens_match_their_own_context():
    module, x, contexts = make_case(regions=2)
    tokens = x.shape[1]
    masks = torch.zeros(2, tokens)
    masks[0, : tokens // 2] = 1.0
    masks[1, tokens // 2 :] = 1.0
    state = RPState(
        masks=masks,
        region_contexts=contexts,
        base_context=None,
        base_weights=torch.zeros(2),
    )
    actual = regional_attention(module, x, state, reference_attention_op)

    only_a = module.reference_forward(x, contexts[0].unsqueeze(0))
    only_b = module.reference_forward(x, contexts[1].unsqueeze(0))

    assert torch.allclose(actual[:, : tokens // 2], only_a[:, : tokens // 2], atol=TOL)
    assert torch.allclose(actual[:, tokens // 2 :], only_b[:, tokens // 2 :], atol=TOL)


def test_base_weight_zero_is_identical_to_no_base():
    module, x, contexts = make_case(regions=2)
    tokens = x.shape[1]
    masks = torch.zeros(2, tokens)
    masks[0, : tokens // 2] = 1.0
    masks[1, tokens // 2 :] = 1.0
    base = torch.randn(1, 5, 24)

    without = regional_attention(
        module,
        x,
        RPState(masks, contexts, None, torch.zeros(2)),
        reference_attention_op,
    )
    with_zero = regional_attention(
        module,
        x,
        RPState(masks, contexts, base, torch.zeros(2)),
        reference_attention_op,
    )
    assert torch.allclose(without, with_zero, atol=TOL)


def test_base_weight_one_is_pure_base():
    module, x, contexts = make_case(regions=2)
    tokens = x.shape[1]
    masks = torch.zeros(2, tokens)
    masks[0, : tokens // 2] = 1.0
    masks[1, tokens // 2 :] = 1.0
    base = torch.randn(1, 5, 24)

    actual = regional_attention(
        module,
        x,
        RPState(masks, contexts, base, torch.ones(2)),
        reference_attention_op,
    )
    expected = module.reference_forward(x, base)
    assert torch.allclose(actual, expected, atol=TOL)


def test_per_region_base_weights_interpolate():
    module, x, contexts = make_case(regions=2)
    tokens = x.shape[1]
    masks = torch.zeros(2, tokens)
    masks[0, : tokens // 2] = 1.0
    masks[1, tokens // 2 :] = 1.0
    base = torch.randn(1, 5, 24)
    weights = torch.tensor([0.0, 1.0])

    actual = regional_attention(
        module, x, RPState(masks, contexts, base, weights), reference_attention_op
    )
    pure_a = module.reference_forward(x, contexts[0].unsqueeze(0))
    pure_base = module.reference_forward(x, base)

    assert torch.allclose(actual[:, : tokens // 2], pure_a[:, : tokens // 2], atol=TOL)
    assert torch.allclose(actual[:, tokens // 2 :], pure_base[:, tokens // 2 :], atol=TOL)


def test_single_region_full_coverage_with_batch_gt_1():
    """batch>1 时单区域全覆盖，掩码广播必须对齐 token 维而非批次维。

    覆盖空白：test_single_region_equals_plain_attention 用 batch=1、掩码全 1，
    若把 regional_attention 里 masks[i].view(1, -1, 1) 误写成 view(-1, 1, 1)，
    batch=1 时 attended 的批次维（1）会被广播吃掉，恰好与预期值逐元素相同，
    torch.allclose 因此“意外通过”而发现不了这个错误。batch>1 时批次维不再是
    1，这种误写要么导致形状不可广播直接报错，要么产生明显偏差的结果，
    从而真正验证掩码的广播维度写对了。
    """
    module, x, contexts = make_case(regions=1, batch=3)
    masks = torch.ones(1, x.shape[1])
    state = RPState(
        masks=masks,
        region_contexts=contexts,
        base_context=None,
        base_weights=torch.zeros(1),
    )
    actual = regional_attention(module, x, state, reference_attention_op)
    expected = module.reference_forward(x, contexts[0].unsqueeze(0))
    assert torch.allclose(actual, expected, atol=TOL)


def test_batch_rows_are_independent():
    module, x, contexts = make_case(regions=2, batch=3)
    tokens = x.shape[1]
    masks = torch.zeros(2, tokens)
    masks[0, : tokens // 2] = 1.0
    masks[1, tokens // 2 :] = 1.0
    state = RPState(masks, contexts, None, torch.zeros(2))

    batched = regional_attention(module, x, state, reference_attention_op)
    for row in range(3):
        single = regional_attention(
            module, x[row : row + 1], state, reference_attention_op
        )
        assert torch.allclose(batched[row : row + 1], single, atol=TOL)


def test_anima_output_proj_has_no_bias():
    """守护设计文档 §2.1 性质 3 所依赖的前提。

    regional_attention 把 output_proj 提到掩码加权和之外只做一次，
    这只在 output_proj 无 bias 时成立。若上游把 Anima 的 SelfCrossAttention
    改成带 bias，本断言先失败，而不是让分区结果悄悄偏移。

    这里同时检查假模块与真实模块的定义保持一致。真实模块的源码以文本方式
    读取（而非 import backend.nn.anima），因为跑测试时 Forge 不在
    sys.path 上，import 会直接 ImportError 走掉，导致这条断言形同虚设。
    """
    module, _, _ = make_case()
    assert module.output_proj.bias is None

    forge_root_explicit = "FORGE_ROOT" in os.environ
    forge_root = os.environ.get("FORGE_ROOT", r"F:\sd-webui-forge-neo")
    anima_path = os.path.join(forge_root, "backend", "nn", "anima.py")

    if not os.path.isfile(anima_path):
        if forge_root_explicit:
            raise FileNotFoundError(
                f"Forge 参考源码不存在：{anima_path}（FORGE_ROOT 被显式设置）"
            )
        pytest.skip(f"Forge 参考源码不存在：{anima_path}（设置 FORGE_ROOT 以启用）")

    with open(anima_path, encoding="utf-8") as f:
        source = f.read()

    assert "self.output_proj = nn.Linear(inner_dim, query_dim, bias=False)" in source


def test_uncond_rows_use_original_path():
    """混合 batch 中负向行必须走原路径，正向行走分区路径。"""
    from lib_anima_rp.attention import regional_forward

    module, x, contexts = make_case(regions=2, batch=4)
    tokens = x.shape[1]
    masks = torch.zeros(2, tokens)
    masks[0, : tokens // 2] = 1.0
    masks[1, tokens // 2 :] = 1.0
    state = RPState(masks, contexts, None, torch.zeros(2))

    negative = torch.randn(4, 5, 24)
    options = {"cond_indices": [0, 1], "uncond_indices": [2, 3]}

    def orig_forward(mod, xx, ctx, rope_emb=None, transformer_options=None):
        return mod.reference_forward(xx, ctx)

    actual = regional_forward(
        module, x, negative, None, options, state, reference_attention_op, orig_forward
    )

    expected_cond = regional_attention(
        module, x[[0, 1]], state, reference_attention_op
    )
    expected_uncond = module.reference_forward(x[[2, 3]], negative[[2, 3]])

    assert torch.allclose(actual[[0, 1]], expected_cond, atol=TOL)
    assert torch.allclose(actual[[2, 3]], expected_uncond, atol=TOL)


def test_all_uncond_batch_delegates_entirely():
    from lib_anima_rp.attention import regional_forward

    module, x, contexts = make_case(regions=2, batch=2)
    masks = torch.zeros(2, x.shape[1])
    masks[0, :] = 1.0
    state = RPState(masks, contexts, None, torch.zeros(2))
    negative = torch.randn(2, 5, 24)
    options = {"cond_indices": [], "uncond_indices": [0, 1]}

    calls = []

    def orig_forward(mod, xx, ctx, rope_emb=None, transformer_options=None):
        calls.append(xx.shape[0])
        return mod.reference_forward(xx, ctx)

    actual = regional_forward(
        module, x, negative, None, options, state, reference_attention_op, orig_forward
    )
    assert calls == [2], "纯 uncond 批应当一次性委托，而不是逐行处理"
    assert torch.allclose(actual, module.reference_forward(x, negative), atol=TOL)


class TestMaskShapeContract:
    """C1（最终审查 Critical）：masks 的 token 维必须等于 x 的 token 维。

    背景：`process()` 先塞一个 `(N, 1)` 的占位 masks，等采样期的 unet
    function wrapper 拿到真实网格后写回。若那个 wrapper 没有运行（例如被
    另一个不链式保留 model_function_wrapper 的扩展——Forge 内置的
    sd_forge_spectrum 就是这样——覆盖掉），masks 会一直停留在 `(N, 1)`。

    `(1, 1, 1)` 与 `(b, TOK, h*d)` 广播相乘是**合法**的 torch 操作，结果
    恒为 0：所有正向行的 cross-attention 输出精确为 0，模型在没有任何文本
    条件的情况下出图，且没有任何异常或警告。唯一的解法是在 regional_attention
    开头就检查 masks 的 token 维是否与 x 一致，不一致就响亮报错，而不是让
    它静默广播成全零。
    """

    def test_placeholder_mask_shape_raises_with_actionable_message(self):
        module, x, contexts = make_case(regions=3)
        # (N, 1) 占位形状——scripts/anima_regional_prompter.py:197 的
        # torch.zeros(region_count, 1)，唯一能静默广播成全零的形状。
        placeholder_masks = torch.zeros(3, 1)
        state = RPState(
            masks=placeholder_masks,
            region_contexts=contexts,
            base_context=None,
            base_weights=torch.zeros(3),
        )

        with pytest.raises(Exception) as excinfo:
            regional_attention(module, x, state, reference_attention_op)

        message = str(excinfo.value)
        # 必须包含：实际 masks 形状、期望的 token 数、以及指向根因（其他扩展
        # 未链式保留 model_function_wrapper）的可定位说明。
        assert "1" in message and "3" in message  # (3, 1) 的形状信息
        assert str(x.shape[1]) in message  # 期望的 token 数
        assert "wrapper" in message.lower() or "Spectrum" in message

    def test_matching_mask_shape_does_not_raise(self):
        module, x, contexts = make_case(regions=3)
        tokens = x.shape[1]
        masks = torch.zeros(3, tokens)
        masks[0, : tokens // 3] = 1.0
        masks[1, tokens // 3 : 2 * tokens // 3] = 1.0
        masks[2, 2 * tokens // 3 :] = 1.0
        state = RPState(
            masks=masks,
            region_contexts=contexts,
            base_context=None,
            base_weights=torch.zeros(3),
        )

        # 不应抛出：形状匹配是正常路径。
        regional_attention(module, x, state, reference_attention_op)


def test_missing_indices_treats_all_rows_as_cond():
    """transformer_options 里没有 cond_indices 时保守地把所有行当正向处理。"""
    from lib_anima_rp.attention import regional_forward

    module, x, contexts = make_case(regions=1, batch=2)
    masks = torch.ones(1, x.shape[1])
    state = RPState(masks, contexts, None, torch.zeros(1))

    def orig_forward(mod, xx, ctx, rope_emb=None, transformer_options=None):
        raise AssertionError("不应走原路径")

    actual = regional_forward(
        module, x, torch.randn(2, 5, 24), None, {}, state,
        reference_attention_op, orig_forward,
    )
    expected = regional_attention(module, x, state, reference_attention_op)
    assert torch.allclose(actual, expected, atol=TOL)
