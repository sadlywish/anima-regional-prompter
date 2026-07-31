"""process() 对 build_config() 抛出 ValueError 的兜底处理。

背景（Task 11 真机验证发现）：`build_config()` 在「区域数与提示词段数不匹配」
时会抛 `ValueError`——这是设计上有意的「响亮报错」，而不是静默补齐。但 Forge
的 `modules/scripts.py` 里 `process()` 会用 `try/except Exception` 把脚本抛出
的异常吞掉，只在控制台打印一次 traceback，生成照常进行：调用方（包括走 API
的调用方）拿到的仍是 HTTP 200 和一张没有分区、也没有任何 RP 元数据的图，事后
完全无法从产物本身得知发生过什么。

`AnimaRegionalPrompter.process()` 因此必须自己兜住这个 ValueError：控制台警告
一次、把错误消息写进 `p.extra_generation_params["RP Error"]`（留痕在 PNG info
里）、不重新抛出（反正 Forge 也会吞掉，抛出只会多一份没人看的 traceback）。
同时不应该写 `RP Active`——分区并未生效，写了会误导事后看图的人。

不依赖真实 Forge：
- 参照 tests/test_preview.py 的 `_load_anima_script`，临时 stub 好 Forge 专属
  的 `modules` / `modules.ui_components`，只为让模块顶层 import 通过。
- 参照 tests/test_patch_anima_attention.py 的 `stub_backend`，临时 stub 好
  `backend.attention` / `backend.nn.anima`，让「正常路径」测试里
  `patch_anima_attention()` 打补丁时不需要真实 Forge 后端。
"""

import importlib.util
import os
import sys
import types

import pytest
import torch

from lib_anima_rp.state import RP_ARG_DEFAULTS, RP_ARG_NAMES

SCRIPT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "scripts", "anima_regional_prompter.py"
)

MODULES_STUB_KEYS = ("modules", "modules.scripts", "modules.ui_components")
BACKEND_STUB_KEYS = ("backend", "backend.nn", "backend.nn.anima", "backend.attention")


def _load_anima_module():
    """在临时 stub 好 `modules` / `modules.ui_components` 的前提下加载扩展主模块。

    与 test_preview.py 的 `_load_anima_script` 同一套做法：stub 只为让模块顶层
    import 通过，本文件的测试从未用到 stub 对象本身，加载完成后立刻清理。
    """
    stubs_to_inject = []

    if "modules" not in sys.modules:
        modules_stub = types.ModuleType("modules")
        scripts_stub = types.ModuleType("modules.scripts")

        class _Script:
            infotext_fields = None
            paste_field_names = None
            sorting_priority = 0

        scripts_stub.Script = _Script
        scripts_stub.AlwaysVisible = object()
        modules_stub.scripts = scripts_stub

        ui_components_stub = types.ModuleType("modules.ui_components")

        class _InputAccordion:
            def __init__(self, *args, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        ui_components_stub.InputAccordion = _InputAccordion

        modules_stub._anima_rp_stub = True
        scripts_stub._anima_rp_stub = True
        ui_components_stub._anima_rp_stub = True

        sys.modules["modules"] = modules_stub
        sys.modules["modules.scripts"] = scripts_stub
        sys.modules["modules.ui_components"] = ui_components_stub
        stubs_to_inject = list(MODULES_STUB_KEYS)

    try:
        spec = importlib.util.spec_from_file_location(
            "anima_regional_prompter_error_handling_under_test", SCRIPT_PATH
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        for key in stubs_to_inject:
            if key in sys.modules:
                del sys.modules[key]


@pytest.fixture(scope="module")
def anima_module():
    return _load_anima_module()


def _build_stub_backend():
    """构造最小可用的 stub 版 backend 包，供 patch_anima_attention() 打补丁用。"""

    def attention_function(q, k, v, heads, skip_reshape=False, transformer_options=None, **kwargs):
        return q

    class SelfCrossAttention:
        def __init__(self, is_SelfAttn=False):
            self.is_SelfAttn = is_SelfAttn

        def forward(self, x, context=None, rope_emb=None, transformer_options=None):
            return x

    backend_pkg = types.ModuleType("backend")
    backend_nn_pkg = types.ModuleType("backend.nn")
    backend_attention_mod = types.ModuleType("backend.attention")
    backend_nn_anima_mod = types.ModuleType("backend.nn.anima")

    backend_attention_mod.attention_function = attention_function
    backend_nn_anima_mod.SelfCrossAttention = SelfCrossAttention
    backend_pkg.nn = backend_nn_pkg
    backend_pkg.attention = backend_attention_mod
    backend_nn_pkg.anima = backend_nn_anima_mod

    return {
        "backend": backend_pkg,
        "backend.nn": backend_nn_pkg,
        "backend.nn.anima": backend_nn_anima_mod,
        "backend.attention": backend_attention_mod,
    }


@pytest.fixture
def stub_backend():
    """注入 stub 版 backend 包；测试结束后清理，并还原
    lib_anima_rp.attention 的模块级补丁状态（_PATCHED / _ORIG_FORWARD）。

    做法与 tests/test_patch_anima_attention.py 的同名 fixture 一致：若真实
    backend 包已存在于 sys.modules，直接跳过，避免弄脏它。
    """
    if any(key in sys.modules for key in BACKEND_STUB_KEYS):
        pytest.skip("sys.modules 里已存在真实的 backend 包，跳过以免弄脏它")

    import lib_anima_rp.attention as attn_mod

    saved_patched = attn_mod._PATCHED
    saved_orig_forward = attn_mod._ORIG_FORWARD

    modules = _build_stub_backend()
    for key, module in modules.items():
        module._anima_rp_stub = True
        sys.modules[key] = module

    try:
        yield
    finally:
        for key in BACKEND_STUB_KEYS:
            if key in sys.modules:
                del sys.modules[key]
        attn_mod._PATCHED = saved_patched
        attn_mod._ORIG_FORWARD = saved_orig_forward


class Anima:
    """类名必须恰好是 "Anima"：is_anima() 靠 __class__.__name__ 精确匹配判定。"""


def _make_args(**overrides):
    """按 RP_ARG_NAMES 顺序构造 process() 的位置参数，只覆盖需要的字段。"""
    params = dict(zip(RP_ARG_NAMES, RP_ARG_DEFAULTS))
    params.update(overrides)
    return tuple(params[name] for name in RP_ARG_NAMES)


def _make_fake_p(prompt, *, negative_prompt="", boom_on_encode=False):
    """构造一个假的 `p` 对象，只提供 process() 用到的属性。

    `boom_on_encode=True` 用于错误路径测试：一旦真的走到了条件编码这一步，
    说明 build_config() 的异常没能在 is_anima() 之前被拦下，测试应当失败，
    而不是悄悄用假编码结果掩盖过去。
    """
    if boom_on_encode:
        def get_learned_conditioning(texts):
            raise AssertionError("配置错误时不应该走到条件编码这一步")
    else:
        def get_learned_conditioning(texts):
            return [torch.zeros(4, 8) for _ in texts]

    sd_model = types.SimpleNamespace(
        forge_objects=types.SimpleNamespace(
            unet=types.SimpleNamespace(
                model=types.SimpleNamespace(diffusion_model=Anima())
            )
        ),
        get_learned_conditioning=get_learned_conditioning,
    )
    return types.SimpleNamespace(
        prompt=prompt,
        negative_prompt=negative_prompt,
        extra_generation_params={},
        sd_model=sd_model,
    )


class TestRegionPromptMismatch:
    """ratios="1,1,1"（3 区域）配只有 1 个 BREAK 的提示词（2 段）：真机复现的错误配置。"""

    def test_process_does_not_raise(self, anima_module):
        script = anima_module.AnimaRegionalPrompter()
        p = _make_fake_p("a BREAK b", boom_on_encode=True)
        args = _make_args(active=True, ratios="1,1,1", mmode="Columns")

        script.process(p, *args)  # 不应抛出 ValueError

    def test_rp_error_records_region_count_segment_count_and_ratios(self, anima_module):
        script = anima_module.AnimaRegionalPrompter()
        p = _make_fake_p("a BREAK b", boom_on_encode=True)
        ratios = "1,1,1"
        args = _make_args(active=True, ratios=ratios, mmode="Columns")

        script.process(p, *args)

        assert "RP Error" in p.extra_generation_params
        message = p.extra_generation_params["RP Error"]
        assert repr(ratios) in message  # Divide ratio 原文
        assert "解析出 3 个区域" in message  # 区域数
        assert "切出 2 段" in message  # 提示词段数

    def test_rp_active_not_set_on_error(self, anima_module):
        script = anima_module.AnimaRegionalPrompter()
        p = _make_fake_p("a BREAK b", boom_on_encode=True)
        args = _make_args(active=True, ratios="1,1,1", mmode="Columns")

        script.process(p, *args)

        assert "RP Active" not in p.extra_generation_params


class TestNormalConfig:
    """区域数与提示词段数匹配时，不应留下任何错误痕迹，应正常写入 RP Active。"""

    def test_rp_active_set_and_no_rp_error(self, anima_module, stub_backend):
        script = anima_module.AnimaRegionalPrompter()
        p = _make_fake_p("a BREAK b")  # 2 段
        args = _make_args(active=True, ratios="1,1", mmode="Columns")  # 2 区域，匹配

        script.process(p, *args)

        assert p.extra_generation_params.get("RP Active") is True
        assert "RP Error" not in p.extra_generation_params
