"""patch_anima_attention() 的接线测试：幂等性与三个短路条件。

patched_forward 是每次 ANIMA 生成、每个 Block、每一步的必经之路——包括
本扩展未启用的时候。三个短路条件（state 缺失 / self-attn / context 为
None）任何一个写错，所有 ANIMA 用户的出图都会被悄悄改坏，不能只靠人工
审查兜底。

不依赖真实 Forge：用 stub 模块伪造 backend.attention.attention_function 与
backend.nn.anima.SelfCrossAttention，参照 tests/conftest.py 给 RP
regions.py 做 modules stub 的既有模式——只注入不存在的键，测试结束后
立刻清理，不覆盖、不删除真实模块（若某次真的在能 import 真实 backend
的环境里跑，这里直接跳过，不弄脏真实模块）。
"""

import sys
import types

import pytest

STUB_KEYS = ("backend", "backend.nn", "backend.nn.anima", "backend.attention")


def _build_stub_backend():
    """构造 stub 版 backend 包，返回 (SelfCrossAttention 类, attention_function 调用计数字典)。"""
    call_log = {"attention_function_calls": 0}

    def attention_function(q, k, v, heads, skip_reshape=False, transformer_options=None, **kwargs):
        call_log["attention_function_calls"] += 1
        return q

    class SelfCrossAttention:
        """伪造的宿主类：只提供 patched_forward 所需的最小接口。"""

        def __init__(self, is_SelfAttn=False):
            self.is_SelfAttn = is_SelfAttn
            self.forward_calls = 0

        def forward(self, x, context=None, rope_emb=None, transformer_options=None):
            self.forward_calls += 1
            return "orig-forward-result"

    backend_pkg = types.ModuleType("backend")
    backend_nn_pkg = types.ModuleType("backend.nn")
    backend_attention_mod = types.ModuleType("backend.attention")
    backend_nn_anima_mod = types.ModuleType("backend.nn.anima")

    backend_attention_mod.attention_function = attention_function
    backend_nn_anima_mod.SelfCrossAttention = SelfCrossAttention
    backend_pkg.nn = backend_nn_pkg
    backend_pkg.attention = backend_attention_mod
    backend_nn_pkg.anima = backend_nn_anima_mod

    modules = {
        "backend": backend_pkg,
        "backend.nn": backend_nn_pkg,
        "backend.nn.anima": backend_nn_anima_mod,
        "backend.attention": backend_attention_mod,
    }
    return SelfCrossAttention, call_log, modules


@pytest.fixture
def stub_backend():
    """注入 stub 版 backend 包；测试结束后清理，并还原
    lib_anima_rp.attention 的模块级补丁状态（_PATCHED / _ORIG_FORWARD）。

    这两个全局变量会跨测试残留：若不还原，一个测试打了补丁，下一个测试会
    因为 _PATCHED 已是 True 而看不到自己打补丁的效果，这种污染依赖测试
    执行顺序、极难排查，所以必须 try/finally 保证复原。
    """
    if any(key in sys.modules for key in STUB_KEYS):
        pytest.skip("sys.modules 里已存在真实的 backend 包，跳过以免弄脏它")

    import lib_anima_rp.attention as attn_mod

    saved_patched = attn_mod._PATCHED
    saved_orig_forward = attn_mod._ORIG_FORWARD

    SelfCrossAttention, call_log, modules = _build_stub_backend()
    for key, module in modules.items():
        module._anima_rp_stub = True
        sys.modules[key] = module

    try:
        yield SelfCrossAttention, call_log
    finally:
        for key in STUB_KEYS:
            if key in sys.modules:
                del sys.modules[key]
        attn_mod._PATCHED = saved_patched
        attn_mod._ORIG_FORWARD = saved_orig_forward

        # 断言清理彻底：不残留任何 stub
        for key in STUB_KEYS:
            assert key not in sys.modules, f"stub 模块 {key} 未被清理干净"


def test_patch_is_idempotent_and_wraps_only_once(stub_backend):
    """连续调用两次 patch_anima_attention()，只应该包一层。

    只断言 SelfCrossAttention.forward 前后是同一个对象不够强：若幂等保护
    失效，第二次打补丁会把第一次打过补丁的函数当成“原始 forward”存进
    _ORIG_FORWARD，但 SelfCrossAttention.forward 依然只是“换了一层新皮”，
    对象identity 检查照样能通过。真正有区分度的是行为检查：在“未启用分区”
    场景下调用一次 patched forward，断言 stub 的原始 forward 恰好被调用
    1 次、且返回值是原始 forward 的返回值。
    """
    from lib_anima_rp.attention import patch_anima_attention

    SelfCrossAttention, _ = stub_backend

    patch_anima_attention()
    forward_after_first = SelfCrossAttention.forward
    patch_anima_attention()  # 幂等：第二次应当什么都不做
    forward_after_second = SelfCrossAttention.forward

    assert forward_after_first is forward_after_second

    instance = SelfCrossAttention(is_SelfAttn=False)
    result = instance.forward(object(), context=object(), transformer_options={})

    assert instance.forward_calls == 1
    assert result == "orig-forward-result"


def test_missing_state_delegates_to_original(stub_backend, monkeypatch):
    """transformer_options 里没有 TRANSFORMER_OPTION_KEY 时，委托原始 forward。"""
    import lib_anima_rp.attention as attn_mod
    from lib_anima_rp.attention import patch_anima_attention

    SelfCrossAttention, _ = stub_backend
    patch_anima_attention()

    def boom(*args, **kwargs):
        raise AssertionError("不应调用 regional_forward：state 缺失时应当委托原始 forward")

    monkeypatch.setattr(attn_mod, "regional_forward", boom)

    instance = SelfCrossAttention(is_SelfAttn=False)
    result = instance.forward(object(), context=object(), transformer_options={})

    assert instance.forward_calls == 1
    assert result == "orig-forward-result"


def test_self_attn_delegates_to_original(stub_backend, monkeypatch):
    """self.is_SelfAttn 为真时，即便 state 与 context 都不缺，也要委托原始 forward。"""
    import lib_anima_rp.attention as attn_mod
    from lib_anima_rp.attention import patch_anima_attention

    SelfCrossAttention, _ = stub_backend
    patch_anima_attention()

    def boom(*args, **kwargs):
        raise AssertionError("不应调用 regional_forward：self-attn 时应当委托原始 forward")

    monkeypatch.setattr(attn_mod, "regional_forward", boom)

    instance = SelfCrossAttention(is_SelfAttn=True)
    options = {attn_mod.TRANSFORMER_OPTION_KEY: object()}
    result = instance.forward(object(), context=object(), transformer_options=options)

    assert instance.forward_calls == 1
    assert result == "orig-forward-result"


def test_context_none_delegates_to_original(stub_backend, monkeypatch):
    """context 为 None 时，即便 state 存在、非 self-attn，也要委托原始 forward。"""
    import lib_anima_rp.attention as attn_mod
    from lib_anima_rp.attention import patch_anima_attention

    SelfCrossAttention, _ = stub_backend
    patch_anima_attention()

    def boom(*args, **kwargs):
        raise AssertionError("不应调用 regional_forward：context 为 None 时应当委托原始 forward")

    monkeypatch.setattr(attn_mod, "regional_forward", boom)

    instance = SelfCrossAttention(is_SelfAttn=False)
    options = {attn_mod.TRANSFORMER_OPTION_KEY: object()}
    result = instance.forward(object(), context=None, transformer_options=options)

    assert instance.forward_calls == 1
    assert result == "orig-forward-result"


def test_enabled_routes_to_regional_forward(stub_backend, monkeypatch):
    """三个短路条件都不满足（有 state、非 self-attn、有 context）时，走 regional_forward。

    同时验证：原始 forward 完全没被调用（不是先跑一遍原始再跑分区的双重
    派发），且传给 regional_forward 的 orig_forward 参数确实是真正的原始
    forward（能正确委托、且只让 forward_calls 增加到 1）。
    """
    import lib_anima_rp.attention as attn_mod
    from lib_anima_rp.attention import patch_anima_attention

    SelfCrossAttention, _ = stub_backend
    patch_anima_attention()

    calls = []

    def fake_regional_forward(
        module, x, context, rope_emb, transformer_options, state, attention_op, orig_forward
    ):
        calls.append((module, x, context, rope_emb, transformer_options, state, orig_forward))
        return "regional-result"

    monkeypatch.setattr(attn_mod, "regional_forward", fake_regional_forward)

    instance = SelfCrossAttention(is_SelfAttn=False)
    state = object()
    options = {attn_mod.TRANSFORMER_OPTION_KEY: state}
    x, context = object(), object()

    result = instance.forward(x, context=context, rope_emb=None, transformer_options=options)

    assert result == "regional-result"
    assert instance.forward_calls == 0, "走分区路径时不该额外调用一次原始 forward"
    assert len(calls) == 1

    called_module, called_x, called_context, _rope, called_options, called_state, called_orig = calls[0]
    assert called_module is instance
    assert called_x is x
    assert called_context is context
    assert called_options is options
    assert called_state is state

    # orig_forward 必须是真正未包过的原始 forward：调用它应当直接命中
    # stub 的原始实现，而不是再绕回 patched_forward。
    assert called_orig(instance, x, context, None, options) == "orig-forward-result"
    assert instance.forward_calls == 1
