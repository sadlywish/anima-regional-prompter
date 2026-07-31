"""render_preview() 的纯函数测试。

`scripts/anima_regional_prompter.py` 顶层 `import` 了 `modules.scripts` 与
`modules.ui_components`（只在 Forge 运行时才存在的包），本任务无法启动 Forge
来验证 UI 接线，但 `render_preview()` 本身是纯函数、不依赖 Forge，因此这里
仿照 `tests/conftest.py` 里 `rp_regions` fixture 的做法：临时 stub 这两个
Forge 专属模块，只为让模块顶层 import 通过；stub 对象本身从未被
`render_preview()` 用到，加载完成后立刻从 `sys.modules` 移除。
"""

import importlib.util
import os
import re
import sys
import types

import pytest

from lib_anima_rp.layout import COLUMNS, ROWS, parse_layout

SCRIPT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "scripts", "anima_regional_prompter.py"
)


def _load_anima_script():
    """在临时 stub 好 `modules` / `modules.ui_components` 的前提下加载扩展主模块。"""
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

        # 给 stub 打上私有标记属性，便于测试验证清理（与 conftest.py 的做法一致）
        modules_stub._anima_rp_stub = True
        scripts_stub._anima_rp_stub = True
        ui_components_stub._anima_rp_stub = True

        sys.modules["modules"] = modules_stub
        sys.modules["modules.scripts"] = scripts_stub
        sys.modules["modules.ui_components"] = ui_components_stub
        stubs_to_inject = ["modules", "modules.scripts", "modules.ui_components"]

    try:
        spec = importlib.util.spec_from_file_location(
            "anima_regional_prompter_under_test", SCRIPT_PATH
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        for key in stubs_to_inject:
            if key in sys.modules:
                del sys.modules[key]


@pytest.fixture(scope="module")
def render_preview():
    return _load_anima_script().render_preview


def _region_count(html: str) -> int:
    """预览 HTML 里每个区域格子都带 `position:absolute`，据此数区域数。"""
    return len(re.findall(r"position:absolute", html))


class TestRenderPreview:
    def test_default_single_row_two_columns(self, render_preview):
        html = render_preview("1,1", COLUMNS, False)
        assert _region_count(html) == 2

    def test_two_dimensional_layout_matches_parse_layout(self, render_preview):
        # 与 brief Step 3 描述一致：Divide Ratio 改成 "1,2,1,1;2,4,6" 应产出 5 个区域
        ratios, mmode, flipper = "1,2,1,1;2,4,6", COLUMNS, False
        html = render_preview(ratios, mmode, flipper)
        expected = len(parse_layout(ratios, mmode, flipper))
        assert expected == 5
        assert _region_count(html) == expected

    def test_rows_mode_region_count(self, render_preview):
        ratios, mmode, flipper = "1,2,1,1;2,4,6", ROWS, False
        html = render_preview(ratios, mmode, flipper)
        expected = len(parse_layout(ratios, mmode, flipper))
        assert _region_count(html) == expected

    def test_flipper_can_change_region_count(self, render_preview):
        ratios, mmode = "1,2,1,1;2,4,6", COLUMNS
        html_normal = render_preview(ratios, mmode, False)
        html_flipped = render_preview(ratios, mmode, True)
        assert html_normal != html_flipped

    def test_invalid_ratio_returns_error_html_instead_of_raising(self, render_preview):
        html = render_preview("not-a-number", COLUMNS, False)
        assert "color:#e88a9a" in html
        assert _region_count(html) == 0

    def test_invalid_mmode_returns_error_html_instead_of_raising(self, render_preview):
        html = render_preview("1,1", "Diagonal", False)
        assert "color:#e88a9a" in html
        assert _region_count(html) == 0

    def test_empty_ratio_returns_error_html_instead_of_raising(self, render_preview):
        html = render_preview("", COLUMNS, False)
        assert "color:#e88a9a" in html
