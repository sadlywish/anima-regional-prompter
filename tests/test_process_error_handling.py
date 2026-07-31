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
import logging
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


def _make_fake_p(
    prompt,
    *,
    negative_prompt="",
    boom_on_encode=False,
    all_prompts=None,
    all_negative_prompts=None,
    all_hr_negative_prompts=None,
):
    """构造一个假的 `p` 对象，只提供 process() 用到的属性。

    `boom_on_encode=True` 用于错误路径测试：一旦真的走到了条件编码这一步，
    说明 build_config() 的异常没能在 is_anima() 之前被拦下，测试应当失败，
    而不是悄悄用假编码结果掩盖过去。

    `all_prompts` / `all_negative_prompts` / `all_hr_negative_prompts` 默认不
    设置在返回的 `p` 上（真实 Forge 里前两者恒存在、后者只在 hires 时存在）——
    只有显式传入时才附加对应属性，让 `getattr(p, "...", None)` 的防御式读取
    与真实环境的可选性保持一致。
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
    p = types.SimpleNamespace(
        prompt=prompt,
        negative_prompt=negative_prompt,
        extra_generation_params={},
        sd_model=sd_model,
    )
    if all_prompts is not None:
        p.all_prompts = all_prompts
    if all_negative_prompts is not None:
        p.all_negative_prompts = all_negative_prompts
    if all_hr_negative_prompts is not None:
        p.all_hr_negative_prompts = all_hr_negative_prompts
    return p


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


class TestSegmentationUsesAllPrompts:
    """I2（最终审查 Important）：分段与门禁都必须读 p.all_prompts，而非 p.prompt。

    背景：Forge 的 Styles 只作用于 `p.all_prompts`，不作用于 `p.prompt`
    （modules/processing.py:406）。若关键字（典型是 BREAK）来自某个 Style，
    `p.prompt` 里完全不含它，用 `p.prompt` 做门禁判定会让整个扩展静默不生效；
    即使门禁侥幸通过，用 `p.prompt` 分段也会漏掉 Style 拼进来的内容。
    """

    def test_process_segments_from_all_prompts_not_prompt(self, anima_module, stub_backend):
        script = anima_module.AnimaRegionalPrompter()
        # p.prompt 故意不含任何分隔关键字，模拟关键字完全来自 Style、
        # 只存在于 all_prompts 里的场景；若实现仍读 p.prompt 做门禁，
        # has_any_keyword 会判定为 False，build_config 返回 None，
        # RP Active 不会被设置。
        p = _make_fake_p(
            "plain prompt without any keyword",
            all_prompts=["styled a BREAK styled b"],
        )
        args = _make_args(active=True, ratios="1,1", mmode="Columns")  # 2 区域

        script.process(p, *args)

        assert p.extra_generation_params.get("RP Active") is True
        assert "RP Error" not in p.extra_generation_params
        # 分段结果必须来自 all_prompts（含 Style 拼接的 "styled" 前缀），
        # 而不是空的 p.prompt。
        assert script.config.region_prompts == ("styled a", "styled b")

    def test_process_falls_back_to_prompt_when_all_prompts_absent(self, anima_module, stub_backend):
        """all_prompts 缺失（测试替身、或极端情况下的旧调用方）时仍能正常工作。"""
        script = anima_module.AnimaRegionalPrompter()
        p = _make_fake_p("a BREAK b")  # 不设置 all_prompts
        args = _make_args(active=True, ratios="1,1", mmode="Columns")

        script.process(p, *args)

        assert p.extra_generation_params.get("RP Active") is True
        assert script.config.region_prompts == ("a", "b")


class TestNegativePromptStripping:
    """I1（最终审查 Important）：负面提示词关键字剥离必须写进 Forge 真正读取的字段。

    背景：Forge 在 process() 被调用之前就已经把 `all_negative_prompts` /
    `all_hr_negative_prompts` 从 `negative_prompt` 展开好了
    （modules/processing.py:398-407, 1584-1590），此后再无重新推导点。
    只写 `p.negative_prompt` 会被完全忽略，用户负面提示词里的 BREAK/ADDCOMM
    会原样作为字面文本进入 uncond 编码。
    """

    def test_all_negative_prompts_cleaned_in_every_item(self, anima_module, stub_backend):
        script = anima_module.AnimaRegionalPrompter()
        p = _make_fake_p(
            "a BREAK b",
            negative_prompt="bad BREAK worse",
            all_negative_prompts=["bad BREAK worse", "bad BREAK worse"],
        )
        args = _make_args(active=True, ratios="1,1", mmode="Columns")

        script.process(p, *args)

        assert p.all_negative_prompts == ["bad, worse", "bad, worse"]

    def test_all_hr_negative_prompts_cleaned_when_present(self, anima_module, stub_backend):
        script = anima_module.AnimaRegionalPrompter()
        p = _make_fake_p(
            "a BREAK b",
            negative_prompt="bad BREAK worse",
            all_negative_prompts=["bad BREAK worse"],
            all_hr_negative_prompts=["bad BREAK worse", "bad BREAK worse"],
        )
        args = _make_args(active=True, ratios="1,1", mmode="Columns")

        script.process(p, *args)

        assert p.all_hr_negative_prompts == ["bad, worse", "bad, worse"]

    def test_no_keyword_leaves_all_negative_prompts_untouched(self, anima_module, stub_backend):
        script = anima_module.AnimaRegionalPrompter()
        p = _make_fake_p(
            "a BREAK b",
            negative_prompt="plain negative",
            all_negative_prompts=["plain negative"],
        )
        args = _make_args(active=True, ratios="1,1", mmode="Columns")

        script.process(p, *args)

        assert p.all_negative_prompts == ["plain negative"]


def _make_fake_p_with_conds(prompt, conds_by_text):
    """构造一个假的 `p`，各文本段返回各不相同的 cond，用来验证调试诊断里
    范数/差值的具体数值（而不仅仅是键是否存在）。

    与 `_make_fake_p` 的区别：后者对所有段返回同一份全零张量，无法区分
    "两个区域碰巧都是 0" 与 "两个区域共享了同一份张量" 这两种情况。
    """

    def get_learned_conditioning(texts):
        return [conds_by_text[text] for text in texts]

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
        negative_prompt="",
        extra_generation_params={},
        sd_model=sd_model,
    )


class _FakeUnet:
    """process_before_every_sampling() 用到的最小 unet 替身。

    只提供该方法实际调用到的接口：clone/model_options/
    set_model_unet_function_wrapper/set_transformer_option，外加
    `.wrapper` 属性方便测试事后取出被注册的闭包自行调用。
    """

    def __init__(self, diffusion_model):
        self.model = types.SimpleNamespace(diffusion_model=diffusion_model)
        self.model_options = {}
        self.wrapper = None

    def clone(self):
        return self

    def set_model_unet_function_wrapper(self, wrapper):
        self.wrapper = wrapper

    def set_transformer_option(self, key, value):
        pass


class TestDebugDiagnostics:
    """debug=True 时应写入运行期诊断（真机验证「区域内容错位」bug 用），
    debug=False 时不能有任何额外产出。"""

    def test_debug_false_produces_no_diagnostic_keys(self, anima_module, stub_backend):
        script = anima_module.AnimaRegionalPrompter()
        p = _make_fake_p("a BREAK b BREAK c")  # 3 段
        args = _make_args(active=True, debug=False, ratios="1,1,1", mmode="Columns")

        script.process(p, *args)

        assert not [k for k in p.extra_generation_params if k.startswith("RP Dbg")]

    def test_debug_true_writes_process_time_diagnostics(self, anima_module, stub_backend):
        script = anima_module.AnimaRegionalPrompter()
        conds_by_text = {
            "a": torch.zeros(4, 8),
            "b": torch.ones(4, 8),
            "c": torch.full((4, 8), 2.0),
        }
        p = _make_fake_p_with_conds("a BREAK b BREAK c", conds_by_text)
        args = _make_args(active=True, debug=True, ratios="1,1,1", mmode="Columns")

        script.process(p, *args)

        assert p.extra_generation_params["RP Dbg Segs"] == "n=3 | 0:a / 1:b / 2:c"
        assert p.extra_generation_params["RP Dbg Cond"] == "n=3 | 4x8 / 4x8 / 4x8"
        assert p.extra_generation_params["RP Dbg Stack"] == "3x4x8"
        # 三个区域的 cond 分别是全 0/全 1/全 2，范数各不相同，
        # 若两个区域的范数意外相等就说明它们共享了同一份张量。
        assert p.extra_generation_params["RP Dbg Norm"] == "0:0.00 / 1:5.66 / 2:11.31"
        # 0-1 与 1-2 的差值相等（都是"相差全 1"），0-2 差值加倍，
        # 任何一对意外读到 0.00 都说明对应两个区域内容混淆了。
        # 遍历顺序是 (0,1) (0,2) (1,2)，与 _debug_generation_params 的双重循环一致。
        assert p.extra_generation_params["RP Dbg Diff"] == "0-1:5.66 / 0-2:11.31 / 1-2:5.66"
        # 采样阶段的诊断（grid/mask）此时还没发生，不应提前出现
        assert "RP Dbg Grid" not in p.extra_generation_params
        assert "RP Dbg Mask" not in p.extra_generation_params
        assert "RP Dbg Tok" not in p.extra_generation_params

    def test_debug_true_writes_sampling_time_diagnostics(self, anima_module, stub_backend):
        script = anima_module.AnimaRegionalPrompter()
        conds_by_text = {
            "a": torch.zeros(4, 8),
            "b": torch.ones(4, 8),
            "c": torch.full((4, 8), 2.0),
        }
        p = _make_fake_p_with_conds("a BREAK b BREAK c", conds_by_text)
        args = _make_args(active=True, debug=True, ratios="1,1,1", mmode="Columns")
        script.process(p, *args)

        diffusion_model = Anima()
        diffusion_model.patch_spatial = 1
        diffusion_model.patch_temporal = 1
        p.sd_model.forge_objects.unet = _FakeUnet(diffusion_model)

        script.process_before_every_sampling(p)

        # 手动模拟一次采样期的 unet 前向：宽度 6、Columns 三等分，每区域恰好 2 个 token
        latent = torch.zeros(1, 4, 1, 1, 6)
        wrapper_kwargs = {
            "input": latent,
            "timestep": torch.zeros(1),
            "c": {"transformer_options": {}},
        }

        def model_function(inp, timestep, **kwargs):
            return inp

        p.sd_model.forge_objects.unet.wrapper(model_function, wrapper_kwargs)

        assert p.extra_generation_params["RP Dbg Grid"] == "1x1x6"
        assert p.extra_generation_params["RP Dbg Mask"] == "3x6"
        assert p.extra_generation_params["RP Dbg Tok"] == "0:2 / 1:2 / 2:2"


class TestEmptyRegionWarningUnconditional:
    """I3（最终审查 Important）：区域分不到 token 的警告必须无条件发出，
    不能被关在 debug=True 后面。

    背景：设计文档 §8 要求这条警告无条件——某个区域分不到任何 token 意味着
    用户写的那段提示词对画面完全没有影响，画面却仍然看起来正常，这正是本
    项目反复强调要避免的静默失效。这里用一个极端偏斜的 Divide Ratio 在很窄
    的网格上复现同样的效果，不必真的调大分辨率。
    """

    def test_empty_region_warns_even_when_debug_is_false(self, anima_module, stub_backend, caplog):
        script = anima_module.AnimaRegionalPrompter()
        p = _make_fake_p("a BREAK b")  # 2 段，匹配 2 区域
        # ratios="1,999" 在只有 4 个横向 token 的网格上，第 0 区域
        # （占比 1/1000）分不到任何 token。
        args = _make_args(active=True, debug=False, ratios="1,999", mmode="Columns")
        script.process(p, *args)

        diffusion_model = Anima()
        diffusion_model.patch_spatial = 1
        diffusion_model.patch_temporal = 1
        p.sd_model.forge_objects.unet = _FakeUnet(diffusion_model)
        script.process_before_every_sampling(p)

        latent = torch.zeros(1, 4, 1, 1, 4)
        wrapper_kwargs = {
            "input": latent,
            "timestep": torch.zeros(1),
            "c": {"transformer_options": {}},
        }

        def model_function(inp, timestep, **kwargs):
            return inp

        with caplog.at_level(logging.WARNING, logger="anima-rp"):
            p.sd_model.forge_objects.unet.wrapper(model_function, wrapper_kwargs)

        assert any("分不到 token" in record.getMessage() for record in caplog.records)
        # debug=False 时仍不应该有诊断键（与 TestDebugDiagnostics 的约定一致），
        # 空区域警告与 debug 专属诊断是两回事，不能因为前者无条件化就带出后者。
        assert "RP Dbg Grid" not in p.extra_generation_params

    def test_empty_region_warning_reported_only_once_per_sampling_stage(
        self, anima_module, stub_backend, caplog
    ):
        """同一采样阶段的多个 denoise step 只应该报一次，复用现有的 diag_written。"""
        script = anima_module.AnimaRegionalPrompter()
        p = _make_fake_p("a BREAK b")
        args = _make_args(active=True, debug=False, ratios="1,999", mmode="Columns")
        script.process(p, *args)

        diffusion_model = Anima()
        diffusion_model.patch_spatial = 1
        diffusion_model.patch_temporal = 1
        p.sd_model.forge_objects.unet = _FakeUnet(diffusion_model)
        script.process_before_every_sampling(p)

        latent = torch.zeros(1, 4, 1, 1, 4)

        def model_function(inp, timestep, **kwargs):
            return inp

        with caplog.at_level(logging.WARNING, logger="anima-rp"):
            for _ in range(3):  # 模拟 3 个 denoise step
                wrapper_kwargs = {
                    "input": latent,
                    "timestep": torch.zeros(1),
                    "c": {"transformer_options": {}},
                }
                p.sd_model.forge_objects.unet.wrapper(model_function, wrapper_kwargs)

        empty_region_warnings = [
            record for record in caplog.records if "分不到 token" in record.getMessage()
        ]
        assert len(empty_region_warnings) == 1
