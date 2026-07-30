# ANIMA Regional Prompter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 sd-webui-forge-neo 实现一个适配 ANIMA 模型的分区提示词扩展，通过遮罩 cross-attention 实现按比例分区渲染，API 参数契约与 sd-webui-regional-prompter 一致。

**Architecture:** 纯逻辑层（比例解析、掩码构建、提示词切分、参数归一化）与 Forge 完全解耦，可独立 pytest。注意力层 monkeypatch `SelfCrossAttention.forward`，通过 `transformer_options` 旁路传入区域条件，按 `cond_mark` 区分正负条件行。Forge 接线集中在唯一一个 `scripts/` 文件里。

**Tech Stack:** Python 3.13、PyTorch、Gradio、pytest。宿主 sd-webui-forge-neo，模型 ANIMA（`circlestone-labs/Anima`）。

## Global Constraints

- 设计文档：`docs/superpowers/specs/2026-07-31-anima-regional-prompter-design.md`。实现与文档冲突时，先改文档再改代码。
- Python 解释器一律用 `F:\sd-webui-forge-neo\venv\Scripts\python.exe`。
- 所有面向用户的字符串（UI 标签除外）、注释、文档用简体中文。UI 标签沿用 RP 的英文原文以便迁移。
- `lib_anima_rp/` 下除 `attention.py` 外**不得 import torch 以外的第三方库，更不得 import Forge 的任何模块**。`attention.py` 只能 import torch。Forge API 只在 `scripts/anima_regional_prompter.py` 中出现。
- token 展平顺序恒为 `(t h w)`。
- `patch_spatial` / `patch_temporal` 必须运行时从 `diffusion_model` 对象读取，禁止硬编码。
- `title()` 必须返回 `"ANIMA Regional Prompter"`。
- `ui()` 必须返回恰好 20 个组件，顺序与 RP `scripts/rp.py:344` 完全一致。
- RP 参考源码位于 `F:\magic\extensions\sd-webui-regional-prompter`，测试通过环境变量 `RP_REFERENCE_PATH` 定位。
- 提交信息用英文，结尾附带 `Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>` 与 `Claude-Session: https://claude.ai/code/session_01LXo7bEbzrWFNA7K3jC9B2g` 两行。

## File Structure

| 文件 | 职责 |
|------|------|
| `lib_anima_rp/__init__.py` | 空，仅标记包 |
| `lib_anima_rp/layout.py` | `Rect` 数据类；Divide ratio 字符串 → 归一化矩形列表；Base ratio 字符串 → 逐区域权重 |
| `lib_anima_rp/prompts.py` | 主提示词按 `ADDCOMM`/`ADDBASE`/`BREAK` 切段；负面提示词标记剥离 |
| `lib_anima_rp/state.py` | 20 个 API 参数归一化、降级判定、`RegionalConfig` 构建与跨模块校验 |
| `lib_anima_rp/masks.py` | 矩形 + token 网格 → 掩码张量，带缓存 |
| `lib_anima_rp/attention.py` | `SelfCrossAttention.forward` 的 patch 与分区注意力实现 |
| `scripts/anima_regional_prompter.py` | Gradio UI、Script 生命周期、条件编码、unet 接线 |
| `tests/conftest.py` | pytest 夹具：RP 参考源码导入器 |
| `tests/test_layout.py` | 比例解析、`Columns`/`Rows`、`flipper`、错误输入 |
| `tests/test_layout_rp_parity.py` | 与 RP `regions.py` 的布局一致性断言 |
| `tests/test_base_weights.py` | Base ratio 广播规则 |
| `tests/test_prompts.py` | 提示词切分 |
| `tests/test_state.py` | 参数归一化、降级分支、校验 |
| `tests/test_masks.py` | 掩码不变量与展平顺序 |
| `tests/test_attention_math.py` | 分区注意力的数值正确性 |
| `pytest.ini` | pytest 配置 |

依赖方向：`state` → `layout` + `prompts`；`masks` → `layout`；`attention` → 无（只吃张量与普通数据）；`scripts/` → 全部。

---

### Task 1: 项目骨架与测试基础设施

**Files:**
- Create: `pytest.ini`
- Create: `lib_anima_rp/__init__.py`
- Create: `tests/conftest.py`
- Create: `tests/test_smoke.py`

**Interfaces:**
- Consumes: 无
- Produces: pytest 可运行；`rp_regions` fixture 返回 RP 的 `regions` 模块或 skip 整个测试

- [ ] **Step 1: 写 pytest 配置**

创建 `pytest.ini`：

```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_functions = test_*
addopts = -q
```

- [ ] **Step 2: 创建包标记文件**

创建 `lib_anima_rp/__init__.py`，内容为单行注释：

```python
"""ANIMA Regional Prompter 的纯逻辑层。不得 import Forge 的任何模块。"""
```

- [ ] **Step 3: 写 conftest，提供 RP 参考源码 fixture**

创建 `tests/conftest.py`：

```python
import importlib.util
import os
import sys
import types

import pytest

DEFAULT_RP_PATH = r"F:\magic\extensions\sd-webui-regional-prompter"


@pytest.fixture(scope="session")
def rp_regions():
    """导入 RP 自己的 regions 模块作为布局参照。

    RP 的 regions.py 会 `from modules import devices`，该模块只在 WebUI 运行时存在，
    因此这里注入一个 stub。若 RP 源码不可用则 skip，而不是 fail。
    """
    root = os.environ.get("RP_REFERENCE_PATH", DEFAULT_RP_PATH)
    path = os.path.join(root, "scripts", "regions.py")
    if not os.path.isfile(path):
        pytest.skip(f"RP 参考源码不存在：{path}（设置 RP_REFERENCE_PATH 以启用）")

    if "modules" not in sys.modules:
        modules_stub = types.ModuleType("modules")
        devices_stub = types.ModuleType("modules.devices")
        devices_stub.device = "cpu"
        modules_stub.devices = devices_stub
        sys.modules["modules"] = modules_stub
        sys.modules["modules.devices"] = devices_stub

    spec = importlib.util.spec_from_file_location("rp_regions_reference", path)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        pytest.skip(f"RP 参考源码导入失败：{exc}")
    return module
```

- [ ] **Step 4: 写冒烟测试**

创建 `tests/test_smoke.py`：

```python
import lib_anima_rp


def test_package_importable():
    assert lib_anima_rp.__doc__


def test_rp_reference_available(rp_regions):
    assert hasattr(rp_regions, "split_l2")
    assert hasattr(rp_regions, "ratiosdealer")
```

- [ ] **Step 5: 运行测试**

Run: `F:\sd-webui-forge-neo\venv\Scripts\python.exe -m pytest -v`
Expected: 2 passed（若开发机上没有 RP 源码，第二个 skip）

- [ ] **Step 6: 提交**

```bash
git add pytest.ini lib_anima_rp/__init__.py tests/conftest.py tests/test_smoke.py
git commit -m "test: add pytest scaffolding and RP reference fixture"
```

---

### Task 2: 比例解析（一维）

**Files:**
- Create: `lib_anima_rp/layout.py`
- Create: `tests/test_layout.py`

**Interfaces:**
- Consumes: 无
- Produces:
  - `Rect` — `@dataclass(frozen=True)`，字段 `x0: float, y0: float, x1: float, y1: float`，归一化到 `[0, 1]`
  - `parse_layout(ratios: str, mmode: str, flipper: bool = False) -> list[Rect]`
  - `swap_delimiters(s: str) -> str`

- [ ] **Step 1: 写失败测试**

创建 `tests/test_layout.py`：

```python
import pytest

from lib_anima_rp.layout import Rect, parse_layout, swap_delimiters

TOL = 1e-9


def assert_rects_close(actual, expected):
    assert len(actual) == len(expected), f"区域数不符：{len(actual)} != {len(expected)}"
    for i, (a, e) in enumerate(zip(actual, expected)):
        for field in ("x0", "y0", "x1", "y1"):
            assert abs(getattr(a, field) - getattr(e, field)) < TOL, (
                f"区域 {i} 的 {field} 不符：{getattr(a, field)} != {getattr(e, field)}"
            )


def test_swap_delimiters():
    assert swap_delimiters("1,2;3,4") == "1;2,3;4"


def test_one_dimension_columns():
    # 不含 ';' 走一维分支：Columns 下切成左右并列的列
    assert_rects_close(
        parse_layout("1,1", "Columns"),
        [Rect(0.0, 0.0, 0.5, 1.0), Rect(0.5, 0.0, 1.0, 1.0)],
    )


def test_one_dimension_rows():
    # 同一个字符串在 Rows 下切成上下堆叠的行
    assert_rects_close(
        parse_layout("1,1", "Rows"),
        [Rect(0.0, 0.0, 1.0, 0.5), Rect(0.0, 0.5, 1.0, 1.0)],
    )


def test_one_dimension_uneven():
    assert_rects_close(
        parse_layout("1,3", "Columns"),
        [Rect(0.0, 0.0, 0.25, 1.0), Rect(0.25, 0.0, 1.0, 1.0)],
    )


def test_single_region():
    assert_rects_close(parse_layout("1", "Columns"), [Rect(0.0, 0.0, 1.0, 1.0)])


def test_rejects_empty():
    with pytest.raises(ValueError, match="不能为空"):
        parse_layout("", "Columns")


def test_rejects_non_numeric():
    with pytest.raises(ValueError, match="不是有效数字"):
        parse_layout("1,abc", "Columns")


def test_rejects_negative():
    with pytest.raises(ValueError, match="必须为正数"):
        parse_layout("1,-2", "Columns")


def test_rejects_all_zero():
    with pytest.raises(ValueError, match="必须为正数"):
        parse_layout("0,0", "Columns")


def test_rejects_unknown_mmode():
    with pytest.raises(ValueError, match="mmode"):
        parse_layout("1,1", "Diagonal")
```

- [ ] **Step 2: 运行测试确认失败**

Run: `F:\sd-webui-forge-neo\venv\Scripts\python.exe -m pytest tests/test_layout.py -v`
Expected: FAIL，`ModuleNotFoundError: No module named 'lib_anima_rp.layout'`

- [ ] **Step 3: 写最小实现**

创建 `lib_anima_rp/layout.py`：

```python
"""Divide ratio 与 Base ratio 的解析。

语法与 sd-webui-regional-prompter 完全一致，参照其 regions.py:274 makeimgtmp。
详见设计文档 §4。
"""

from dataclasses import dataclass

COLUMNS = "Columns"
ROWS = "Rows"
VALID_MMODES = (COLUMNS, ROWS)

_BAND_DELIM = ";"
_CELL_DELIM = ","


@dataclass(frozen=True)
class Rect:
    """归一化到 [0, 1] 的区域矩形，原点在左上角。"""

    x0: float
    y0: float
    x1: float
    y1: float


def swap_delimiters(s: str) -> str:
    """交换 ',' 与 ';'，等价于 RP 的 changecs()。"""
    return s.replace(",", "\x00").replace(";", ",").replace("\x00", ";")


def _to_positive_floats(parts, context):
    values = []
    for raw in parts:
        text = raw.strip()
        if not text:
            raise ValueError(f"Divide ratio 的 {context} 中存在空值：{parts!r}")
        try:
            value = float(text)
        except ValueError:
            raise ValueError(f"Divide ratio 的 {context} 中 {text!r} 不是有效数字") from None
        if value <= 0:
            raise ValueError(f"Divide ratio 的 {context} 中 {text!r} 必须为正数")
        values.append(value)
    return values


def _spans(sizes):
    """把一组正数尺寸转成归一化的 [start, end] 区间列表。"""
    total = sum(sizes)
    spans = []
    cursor = 0.0
    for size in sizes:
        start = cursor
        cursor += size / total
        spans.append((start, cursor))
    # 消除浮点累积误差，最后一段强制贴边
    spans[-1] = (spans[-1][0], 1.0)
    return spans


def parse_layout(ratios: str, mmode: str, flipper: bool = False) -> list[Rect]:
    """把 Divide ratio 字符串解析成区域矩形列表，band-major 顺序。"""
    if mmode not in VALID_MMODES:
        raise ValueError(f"未知的 mmode：{mmode!r}，只接受 {VALID_MMODES}")

    text = ratios.strip()
    if not text:
        raise ValueError("Divide ratio 不能为空")

    if flipper:
        text = swap_delimiters(text)

    # 一维分支：字面不含 ';'，与 RP regions.py:281 一致，对两种 mmode 都用字面 ';' 检测
    if _BAND_DELIM not in text:
        cells = _to_positive_floats(text.split(_CELL_DELIM), "带内单元")
        bands = [1.0]
        cell_rows = [cells]
    else:
        bands, cell_rows = _split_bands(text, mmode)

    flipped = mmode == ROWS
    band_spans = _spans(bands)
    rects = []
    for band_index, band_span in enumerate(band_spans):
        for cell_span in _spans(cell_rows[band_index]):
            if flipped:
                rects.append(Rect(band_span[0], cell_span[0], band_span[1], cell_span[1]))
            else:
                rects.append(Rect(cell_span[0], band_span[0], cell_span[1], band_span[1]))
    return rects


def _split_bands(text, mmode):
    """二维分支。RP 的 split_l2 在 indflip 时交换两个分隔符，这里照搬。"""
    if mmode == ROWS:
        band_delim, cell_delim = _CELL_DELIM, _BAND_DELIM
    else:
        band_delim, cell_delim = _BAND_DELIM, _CELL_DELIM

    bands = []
    cell_rows = []
    for band_text in text.split(band_delim):
        values = _to_positive_floats(band_text.split(cell_delim), "带")
        bands.append(values[0])
        # RP regions.py:131 —— 只有一个值时该值同时充当带尺寸与唯一单元
        cell_rows.append(values[1:] if len(values) > 1 else values)
    return bands, cell_rows
```

- [ ] **Step 4: 运行测试确认通过**

Run: `F:\sd-webui-forge-neo\venv\Scripts\python.exe -m pytest tests/test_layout.py -v`
Expected: 10 passed

- [ ] **Step 5: 提交**

```bash
git add lib_anima_rp/layout.py tests/test_layout.py
git commit -m "feat(layout): parse one-dimensional divide ratios"
```

---

### Task 3: 比例解析（二维）与 RP 一致性断言

**Files:**
- Modify: `tests/test_layout.py`（追加测试）
- Create: `tests/test_layout_rp_parity.py`

**Interfaces:**
- Consumes: `Rect`、`parse_layout`、`swap_delimiters`（Task 2）；`rp_regions` fixture（Task 1）
- Produces: 无新接口，仅确立 Task 2 实现的二维行为正确

- [ ] **Step 1: 追加二维测试**

在 `tests/test_layout.py` 末尾追加：

```python
def test_two_dimensions_columns():
    # 带 1 尺寸 1、带内 2:1:1；带 2 尺寸 2、带内 4:6
    rects = parse_layout("1,2,1,1;2,4,6", "Columns")
    assert_rects_close(
        rects,
        [
            Rect(0.0, 0.0, 0.5, 1 / 3),
            Rect(0.5, 0.0, 0.75, 1 / 3),
            Rect(0.75, 0.0, 1.0, 1 / 3),
            Rect(0.0, 1 / 3, 0.4, 1.0),
            Rect(0.4, 1 / 3, 1.0, 1.0),
        ],
    )


def test_two_dimensions_rows_is_transpose():
    # Rows 模式下分隔符一并转置，得到 Columns 版本的转置版面
    rects = parse_layout("1;2;1;1,2;4;6", "Rows")
    assert_rects_close(
        rects,
        [
            Rect(0.0, 0.0, 1 / 3, 0.5),
            Rect(0.0, 0.5, 1 / 3, 0.75),
            Rect(0.0, 0.75, 1 / 3, 1.0),
            Rect(1 / 3, 0.0, 1.0, 0.4),
            Rect(1 / 3, 0.4, 1.0, 1.0),
        ],
    )


def test_band_with_single_value_fills_band():
    # 带 2 只给一个数字：尺寸为 3，带内单个单元铺满
    rects = parse_layout("1,1,1;3", "Columns")
    assert_rects_close(
        rects,
        [
            Rect(0.0, 0.0, 0.5, 0.25),
            Rect(0.5, 0.0, 1.0, 0.25),
            Rect(0.0, 0.25, 1.0, 1.0),
        ],
    )


def test_flipper_makes_rows_accept_columns_syntax():
    flipped = parse_layout("1,2,1,1;2,4,6", "Rows", flipper=True)
    direct = parse_layout("1;2;1;1,2;4;6", "Rows")
    assert_rects_close(flipped, direct)


def test_flipper_does_not_affect_one_dimension_shape():
    # "1,1" 翻转后是 "1;1"，含 ';' 故走二维分支：两个带，各自单值铺满
    rects = parse_layout("1,1", "Columns", flipper=True)
    assert_rects_close(
        rects, [Rect(0.0, 0.0, 1.0, 0.5), Rect(0.0, 0.5, 1.0, 1.0)]
    )
```

- [ ] **Step 2: 运行测试**

Run: `F:\sd-webui-forge-neo\venv\Scripts\python.exe -m pytest tests/test_layout.py -v`
Expected: 15 passed。若 `test_two_dimensions_rows_is_transpose` 或 `test_band_with_single_value_fills_band` 失败，说明 Task 2 的 `_split_bands` 有误，在此修正。

- [ ] **Step 3: 写 RP 一致性测试**

创建 `tests/test_layout_rp_parity.py`：

```python
"""与 sd-webui-regional-prompter 的布局一致性断言。

参照实现按 RP regions.py:274 makeimgtmp 的分支复刻，直接调用 RP 自己的
split_l2 / ratiosdealer。设计文档 §11.2。
"""

import pytest

from lib_anima_rp.layout import Rect, parse_layout

CASES = [
    ("1,1", "Columns", False),
    ("1,1", "Rows", False),
    ("1,3", "Columns", False),
    ("1,1,1", "Rows", False),
    ("1,2,1,1;2,4,6", "Columns", False),
    ("1;2;1;1,2;4;6", "Rows", False),
    ("1,1,1;3", "Columns", False),
    ("1,2,1,1;2,4,6", "Rows", True),
    ("2,3;1,1,1;4,5", "Columns", False),
]

TOL = 1e-9


def rp_reference_rects(rp_regions, ratios, mmode, flipper):
    mode = {"Columns": "Horizontal", "Rows": "Vertical"}[mmode]
    if flipper:
        ratios = rp_regions.changecs(ratios)
    indflip = "Ver" in mode
    if ";" not in ratios:
        cells = rp_regions.split_l2(
            ratios, ";", ",", fmap=rp_regions.ffloatd(1), indflip=False
        )
        bands = [1]
    else:
        bands, cells = rp_regions.split_l2(
            ratios, ";", ",", indsingles=True, fmap=rp_regions.ffloatd(1), indflip=indflip
        )
    cells, bands = rp_regions.ratiosdealer(cells, bands)
    out = []
    for band_index, band in enumerate(bands):
        for cell in cells[band_index]:
            if not indflip:
                out.append(Rect(cell[0], band[0], cell[1], band[1]))
            else:
                out.append(Rect(band[0], cell[0], band[1], cell[1]))
    return out


@pytest.mark.parametrize("ratios,mmode,flipper", CASES)
def test_matches_rp_layout(rp_regions, ratios, mmode, flipper):
    expected = rp_reference_rects(rp_regions, ratios, mmode, flipper)
    actual = parse_layout(ratios, mmode, flipper)

    assert len(actual) == len(expected), (
        f"{ratios!r}/{mmode}/flip={flipper} 区域数不符："
        f"我们 {len(actual)}，RP {len(expected)}"
    )
    for i, (a, e) in enumerate(zip(actual, expected)):
        for field in ("x0", "y0", "x1", "y1"):
            assert abs(getattr(a, field) - getattr(e, field)) < TOL, (
                f"{ratios!r}/{mmode}/flip={flipper} 区域 {i} 的 {field} 不符："
                f"我们 {getattr(a, field)}，RP {getattr(e, field)}"
            )
```

- [ ] **Step 4: 运行一致性测试**

Run: `F:\sd-webui-forge-neo\venv\Scripts\python.exe -m pytest tests/test_layout_rp_parity.py -v`
Expected: 9 passed。

若某个用例失败，以 RP 的输出为准修正 `layout.py`——兼容性优先。唯一例外是 RP 明显的浮点误差（末段不贴 1.0），此时放宽该断言并在测试里注明。

- [ ] **Step 5: 提交**

```bash
git add lib_anima_rp/layout.py tests/test_layout.py tests/test_layout_rp_parity.py
git commit -m "feat(layout): support two-dimensional ratios with RP parity assertions"
```

---

### Task 4: Base ratio 权重解析

**Files:**
- Modify: `lib_anima_rp/layout.py`
- Create: `tests/test_base_weights.py`

**Interfaces:**
- Consumes: `parse_layout`（Task 2/3）
- Produces: `parse_base_weights(baseratios: str, ratios: str, mmode: str, flipper: bool = False) -> list[float]`，返回长度等于区域数的权重列表

- [ ] **Step 1: 写失败测试**

创建 `tests/test_base_weights.py`：

```python
import pytest

from lib_anima_rp.layout import parse_base_weights

TOL = 1e-9


def assert_weights(actual, expected):
    assert len(actual) == len(expected)
    for a, e in zip(actual, expected):
        assert abs(a - e) < TOL


def test_single_value_broadcasts_to_all_regions():
    assert_weights(parse_base_weights("0.2", "1,1,1", "Columns"), [0.2, 0.2, 0.2])


def test_per_region_values():
    assert_weights(parse_base_weights("0.1,0.2,0.3", "1,1,1", "Columns"), [0.1, 0.2, 0.3])


def test_too_few_values_repeat_last():
    assert_weights(parse_base_weights("0.1,0.2", "1,1,1,1", "Columns"), [0.1, 0.2, 0.2, 0.2])


def test_too_many_values_truncated():
    assert_weights(parse_base_weights("0.1,0.2,0.3", "1,1", "Columns"), [0.1, 0.2])


def test_band_structure_is_flattened_band_major():
    # 5 个区域：带 1 有 3 个，带 2 有 2 个
    assert_weights(
        parse_base_weights("0.1,0.2,0.3;0.4,0.5", "1,2,1,1;2,4,6", "Columns"),
        [0.1, 0.2, 0.3, 0.4, 0.5],
    )


def test_non_numeric_falls_back_to_zero():
    # RP 用 ffloatd(0)，非数字退化为 0 而不是报错
    assert_weights(parse_base_weights("0.2,x", "1,1", "Columns"), [0.2, 0.0])


def test_empty_string_is_all_zero():
    assert_weights(parse_base_weights("", "1,1", "Columns"), [0.0, 0.0])


def test_flipper_does_not_apply_to_base_ratios():
    # RP rp.py:386 只对 aratios 调 changecs，baseratios 原样使用。
    # 在 baseratios 里 ';' 与 ',' 同义（都只是值分隔符），故两种写法结果相同。
    assert_weights(
        parse_base_weights("0.1;0.2", "1,1", "Columns"),
        parse_base_weights("0.1,0.2", "1,1", "Columns"),
    )


def test_out_of_range_is_clamped():
    assert_weights(parse_base_weights("-1,5", "1,1", "Columns"), [0.0, 1.0])
```

- [ ] **Step 2: 运行测试确认失败**

Run: `F:\sd-webui-forge-neo\venv\Scripts\python.exe -m pytest tests/test_base_weights.py -v`
Expected: FAIL，`ImportError: cannot import name 'parse_base_weights'`

- [ ] **Step 3: 实现**

在 `lib_anima_rp/layout.py` 末尾追加：

```python
def _float_or_zero(text):
    """RP 的 ffloatd(0)：非数字退化为 0 而不是报错。"""
    try:
        return float(text.strip())
    except ValueError:
        return 0.0


def parse_base_weights(
    baseratios: str, ratios: str, mmode: str, flipper: bool = False
) -> list[float]:
    """把 Base ratio 字符串展开成与区域一一对应的权重列表。

    广播规则（RP split_l2 的 basestruct 行为）：值不足时重复最后一个，多余的截断。
    注意 flipper 不作用于 baseratios，这与 RP rp.py:386 一致——它只对 aratios
    调用 changecs。此处保留 flipper 参数仅为让调用方无需区分，内部不使用。
    """
    del flipper  # 有意不使用，见 docstring

    region_count = len(parse_layout(ratios, mmode, flipper=False))

    raw = baseratios.strip()
    if not raw:
        values = [0.0]
    else:
        values = [_float_or_zero(part) for part in raw.replace(";", ",").split(",")]
    if not values:
        values = [0.0]

    if len(values) < region_count:
        values = values + [values[-1]] * (region_count - len(values))
    weights = values[:region_count]
    return [min(1.0, max(0.0, w)) for w in weights]
```

> 注意：`parse_base_weights` 内部调用 `parse_layout(ratios, mmode, flipper=False)`
> 而不是传入外部的 `flipper`。这看似矛盾，但正确——区域**数量**与 flipper 无关
> （交换分隔符只改变版面朝向，不改变区域总数），而 baseratios 本身不受 flipper 影响。
> `test_flipper_does_not_apply_to_base_ratios` 覆盖这一点。

- [ ] **Step 4: 运行测试确认通过**

Run: `F:\sd-webui-forge-neo\venv\Scripts\python.exe -m pytest tests/test_base_weights.py -v`
Expected: 9 passed

- [ ] **Step 5: 全量回归**

Run: `F:\sd-webui-forge-neo\venv\Scripts\python.exe -m pytest -v`
Expected: 全部通过

- [ ] **Step 6: 提交**

```bash
git add lib_anima_rp/layout.py tests/test_base_weights.py
git commit -m "feat(layout): expand base ratios into per-region weights"
```

---

### Task 5: 提示词切分

**Files:**
- Create: `lib_anima_rp/prompts.py`
- Create: `tests/test_prompts.py`

**Interfaces:**
- Consumes: 无
- Produces:
  - `HAS_KEYWORD_RE` — 编译好的正则，用于判断提示词是否含任何分隔关键字
  - `@dataclass(frozen=True) class PromptSegments`，字段 `common: str`（无则空串）、`base: str`（无则空串）、`regions: tuple[str, ...]`
  - `split_prompt(prompt: str, usecom: bool, usebase: bool) -> PromptSegments`
  - `strip_negative_keywords(negative: str) -> tuple[str, bool]`，返回 `(清理后的文本, 是否发生过剥离)`
  - `has_any_keyword(prompt: str) -> bool`

- [ ] **Step 1: 写失败测试**

创建 `tests/test_prompts.py`：

```python
import pytest

from lib_anima_rp.prompts import (
    has_any_keyword,
    split_prompt,
    strip_negative_keywords,
)


def test_plain_break_splits_regions():
    seg = split_prompt("1girl BREAK 1boy", usecom=False, usebase=False)
    assert seg.regions == ("1girl", "1boy")
    assert seg.common == ""
    assert seg.base == ""


def test_addcol_and_addrow_are_aliases_of_break():
    seg = split_prompt("a ADDCOL b ADDROW c", usecom=False, usebase=False)
    assert seg.regions == ("a", "b", "c")


def test_addcomm_extracts_common_when_enabled():
    seg = split_prompt("quality ADDCOMM a BREAK b", usecom=True, usebase=False)
    assert seg.common == "quality"
    assert seg.regions == ("a", "b")


def test_addcomm_is_plain_separator_when_disabled():
    seg = split_prompt("quality ADDCOMM a BREAK b", usecom=False, usebase=False)
    assert seg.common == ""
    assert seg.regions == ("quality", "a", "b")


def test_addbase_extracts_base_when_enabled():
    seg = split_prompt("room ADDBASE a BREAK b", usecom=False, usebase=True)
    assert seg.base == "room"
    assert seg.regions == ("a", "b")


def test_addbase_is_plain_separator_when_disabled():
    seg = split_prompt("room ADDBASE a BREAK b", usecom=False, usebase=False)
    assert seg.base == ""
    assert seg.regions == ("room", "a", "b")


def test_common_and_base_together():
    seg = split_prompt(
        "quality ADDCOMM room ADDBASE a BREAK b", usecom=True, usebase=True
    )
    assert seg.common == "quality"
    assert seg.base == "room"
    assert seg.regions == ("a", "b")


def test_enabled_but_keyword_absent_is_silent():
    seg = split_prompt("a BREAK b", usecom=True, usebase=True)
    assert seg.common == ""
    assert seg.base == ""
    assert seg.regions == ("a", "b")


def test_whitespace_and_trailing_commas_trimmed():
    seg = split_prompt("  a ,  BREAK   b,  ", usecom=False, usebase=False)
    assert seg.regions == ("a", "b")


def test_empty_region_raises():
    with pytest.raises(ValueError, match="第 2 段为空"):
        split_prompt("a BREAK  BREAK c", usecom=False, usebase=False)


def test_keyword_must_be_standalone_word():
    # BREAKFAST 不是分隔符
    seg = split_prompt("BREAKFAST table", usecom=False, usebase=False)
    assert seg.regions == ("BREAKFAST table",)


def test_lowercase_break_is_not_a_separator():
    seg = split_prompt("a break b", usecom=False, usebase=False)
    assert seg.regions == ("a break b",)


def test_has_any_keyword():
    assert has_any_keyword("a BREAK b")
    assert has_any_keyword("a ADDBASE b")
    assert not has_any_keyword("a plain prompt")
    assert not has_any_keyword("BREAKFAST")


def test_strip_negative_keywords():
    cleaned, changed = strip_negative_keywords("worst quality BREAK bad hands")
    assert cleaned == "worst quality, bad hands"
    assert changed is True


def test_strip_negative_keywords_noop():
    cleaned, changed = strip_negative_keywords("worst quality")
    assert cleaned == "worst quality"
    assert changed is False
```

- [ ] **Step 2: 运行测试确认失败**

Run: `F:\sd-webui-forge-neo\venv\Scripts\python.exe -m pytest tests/test_prompts.py -v`
Expected: FAIL，`ModuleNotFoundError: No module named 'lib_anima_rp.prompts'`

- [ ] **Step 3: 实现**

创建 `lib_anima_rp/prompts.py`：

```python
"""主提示词与负面提示词的分段。设计文档 §6。

分隔关键字必须大写且是独立词，避免误伤 BREAKFAST 这类单词。
"""

import re
from dataclasses import dataclass

KEY_COMMON = "ADDCOMM"
KEY_BASE = "ADDBASE"
REGION_KEYS = ("BREAK", "ADDCOL", "ADDROW")

ALL_KEYS = (KEY_COMMON, KEY_BASE) + REGION_KEYS

HAS_KEYWORD_RE = re.compile(r"\b(?:" + "|".join(ALL_KEYS) + r")\b")
_REGION_SPLIT_RE = re.compile(r"\b(?:" + "|".join(REGION_KEYS) + r")\b")
_COMMON_SPLIT_RE = re.compile(r"\b" + KEY_COMMON + r"\b")
_BASE_SPLIT_RE = re.compile(r"\b" + KEY_BASE + r"\b")


@dataclass(frozen=True)
class PromptSegments:
    common: str
    base: str
    regions: tuple[str, ...]


def has_any_keyword(prompt: str) -> bool:
    """提示词是否含任何分隔关键字。不含时整个扩展应静默不生效。"""
    return HAS_KEYWORD_RE.search(prompt) is not None


def _tidy(text: str) -> str:
    return text.strip().strip(",").strip()


def split_prompt(prompt: str, usecom: bool, usebase: bool) -> PromptSegments:
    """按 ADDCOMM / ADDBASE / BREAK 切分主提示词。

    usecom / usebase 关闭时，对应关键字退化为普通的区域分隔符（与 RP 一致）。
    """
    rest = prompt
    common = ""
    base = ""

    if usecom:
        parts = _COMMON_SPLIT_RE.split(rest, maxsplit=1)
        if len(parts) == 2:
            common, rest = _tidy(parts[0]), parts[1]

    if usebase:
        parts = _BASE_SPLIT_RE.split(rest, maxsplit=1)
        if len(parts) == 2:
            base, rest = _tidy(parts[0]), parts[1]

    # 未启用的关键字降级为区域分隔符
    leftover_keys = []
    if not usecom:
        leftover_keys.append(KEY_COMMON)
    if not usebase:
        leftover_keys.append(KEY_BASE)
    if leftover_keys:
        rest = re.sub(r"\b(?:" + "|".join(leftover_keys) + r")\b", "BREAK", rest)

    raw_regions = _REGION_SPLIT_RE.split(rest)
    regions = []
    for index, chunk in enumerate(raw_regions):
        tidied = _tidy(chunk)
        if not tidied:
            raise ValueError(
                f"提示词的第 {index + 1} 段为空。检查是否有连续的 BREAK 或首尾多余的分隔符。"
            )
        regions.append(tidied)

    return PromptSegments(common=common, base=base, regions=tuple(regions))


def strip_negative_keywords(negative: str) -> tuple[str, bool]:
    """剥掉负面提示词里的分隔关键字并以逗号合并。

    本扩展的负面提示词不分区。返回 (清理后的文本, 是否发生过剥离)。
    """
    if not HAS_KEYWORD_RE.search(negative):
        return negative, False
    parts = [_tidy(part) for part in HAS_KEYWORD_RE.split(negative)]
    return ", ".join(part for part in parts if part), True


def compose_region_prompts(segments: PromptSegments) -> tuple[str, ...]:
    """把 common 前缀拼到每个区域提示词前面。"""
    if not segments.common:
        return segments.regions
    return tuple(f"{segments.common}, {region}" for region in segments.regions)


def compose_base_prompt(segments: PromptSegments) -> str:
    """把 common 前缀拼到 base 提示词前面。base 为空时返回空串。"""
    if not segments.base:
        return ""
    if not segments.common:
        return segments.base
    return f"{segments.common}, {segments.base}"
```

- [ ] **Step 4: 运行测试确认通过**

Run: `F:\sd-webui-forge-neo\venv\Scripts\python.exe -m pytest tests/test_prompts.py -v`
Expected: 16 passed

- [ ] **Step 5: 追加 common 拼接测试**

在 `tests/test_prompts.py` 末尾追加：

```python
def test_compose_region_prompts_prefixes_common():
    from lib_anima_rp.prompts import compose_region_prompts

    seg = split_prompt("quality ADDCOMM a BREAK b", usecom=True, usebase=False)
    assert compose_region_prompts(seg) == ("quality, a", "quality, b")


def test_compose_region_prompts_without_common():
    from lib_anima_rp.prompts import compose_region_prompts

    seg = split_prompt("a BREAK b", usecom=False, usebase=False)
    assert compose_region_prompts(seg) == ("a", "b")


def test_compose_base_prompt():
    from lib_anima_rp.prompts import compose_base_prompt

    seg = split_prompt("quality ADDCOMM room ADDBASE a BREAK b", usecom=True, usebase=True)
    assert compose_base_prompt(seg) == "quality, room"


def test_compose_base_prompt_empty_when_no_base():
    from lib_anima_rp.prompts import compose_base_prompt

    seg = split_prompt("a BREAK b", usecom=False, usebase=True)
    assert compose_base_prompt(seg) == ""
```

- [ ] **Step 6: 运行测试**

Run: `F:\sd-webui-forge-neo\venv\Scripts\python.exe -m pytest tests/test_prompts.py -v`
Expected: 20 passed

- [ ] **Step 7: 提交**

```bash
git add lib_anima_rp/prompts.py tests/test_prompts.py
git commit -m "feat(prompts): split prompts on ADDCOMM/ADDBASE/BREAK keywords"
```

---

### Task 6: 参数归一化与配置构建

**Files:**
- Create: `lib_anima_rp/state.py`
- Create: `tests/test_state.py`

**Interfaces:**
- Consumes: `parse_layout`、`parse_base_weights`（Task 2–4）；`split_prompt`、`has_any_keyword`、`compose_region_prompts`、`compose_base_prompt`（Task 5）
- Produces:
  - `RP_ARG_NAMES` — 长度 20 的 tuple，按 RP 顺序给出参数名
  - `@dataclass(frozen=True) class RegionalConfig`，字段：`rects: tuple[Rect, ...]`、`region_prompts: tuple[str, ...]`、`base_prompt: str`、`base_weights: tuple[float, ...]`、`debug: bool`、`degradations: tuple[str, ...]`
  - `normalize_args(args: Sequence) -> dict` — 把 20 个位置参数转成命名 dict，缺失的补 RP 默认值
  - `collect_degradations(params: dict) -> tuple[str, ...]` — 返回降级说明列表
  - `build_config(params: dict, prompt: str) -> RegionalConfig | None` — 返回 `None` 表示应静默不生效

- [ ] **Step 1: 写失败测试**

创建 `tests/test_state.py`：

```python
import pytest

from lib_anima_rp.state import (
    RP_ARG_NAMES,
    build_config,
    collect_degradations,
    normalize_args,
)

RP_DEFAULT_ARGS = [
    True, False, "Matrix", "Columns", "Mask", "Prompt", "1,1", "0.2",
    False, False, False, "Attention", False, "0", "0", "0.4", None,
    "0", "0", False,
]


def test_arg_names_match_rp_order():
    assert len(RP_ARG_NAMES) == 20
    assert RP_ARG_NAMES[0] == "active"
    assert RP_ARG_NAMES[3] == "mmode"
    assert RP_ARG_NAMES[6] == "ratios"
    assert RP_ARG_NAMES[7] == "baseratios"
    assert RP_ARG_NAMES[11] == "calcmode"
    assert RP_ARG_NAMES[19] == "flipper"


def test_normalize_full_args():
    params = normalize_args(RP_DEFAULT_ARGS)
    assert params["active"] is True
    assert params["mmode"] == "Columns"
    assert params["ratios"] == "1,1"


def test_normalize_short_args_fills_defaults():
    params = normalize_args([True])
    assert params["active"] is True
    assert params["mmode"] == "Columns"
    assert params["calcmode"] == "Attention"


def test_no_degradations_on_defaults():
    assert collect_degradations(normalize_args(RP_DEFAULT_ARGS)) == ()


def test_latent_calcmode_degrades():
    args = list(RP_DEFAULT_ARGS)
    args[11] = "Latent"
    assert collect_degradations(normalize_args(args)) == ("calcmode=Latent",)


def test_mask_tab_degrades():
    args = list(RP_DEFAULT_ARGS)
    args[2] = "Mask"
    assert collect_degradations(normalize_args(args)) == ("tab=Mask",)


def test_random_mmode_degrades():
    args = list(RP_DEFAULT_ARGS)
    args[3] = "Random"
    assert collect_degradations(normalize_args(args)) == ("mmode=Random",)


def test_usencom_degrades():
    args = list(RP_DEFAULT_ARGS)
    args[10] = True
    assert collect_degradations(normalize_args(args)) == ("usencom=True",)


def test_multiple_degradations_are_all_reported():
    args = list(RP_DEFAULT_ARGS)
    args[2] = "Prompt"
    args[11] = "Latent"
    assert set(collect_degradations(normalize_args(args))) == {"tab=Prompt", "calcmode=Latent"}


def test_random_mmode_falls_back_to_columns_in_config():
    args = list(RP_DEFAULT_ARGS)
    args[3] = "Random"
    config = build_config(normalize_args(args), "a BREAK b")
    assert len(config.rects) == 2
    assert config.rects[0].x1 == pytest.approx(0.5)


def test_build_config_basic():
    config = build_config(normalize_args(RP_DEFAULT_ARGS), "a BREAK b")
    assert config.region_prompts == ("a", "b")
    assert config.base_prompt == ""
    # 没有 base prompt 时权重恒为 0，baseratios 的取值无意义
    assert config.base_weights == (0.0, 0.0)
    assert len(config.rects) == 2


def test_base_weights_populated_when_base_present():
    args = list(RP_DEFAULT_ARGS)
    args[8] = True  # usebase
    config = build_config(normalize_args(args), "room ADDBASE a BREAK b")
    assert config.base_prompt == "room"
    assert config.base_weights == (0.2, 0.2)


def test_build_config_returns_none_without_keywords():
    assert build_config(normalize_args(RP_DEFAULT_ARGS), "a plain prompt") is None


def test_build_config_returns_none_when_inactive():
    args = list(RP_DEFAULT_ARGS)
    args[0] = False
    assert build_config(normalize_args(args), "a BREAK b") is None


def test_region_count_mismatch_raises_with_detail():
    args = list(RP_DEFAULT_ARGS)
    args[6] = "1,1,1"
    with pytest.raises(ValueError) as excinfo:
        build_config(normalize_args(args), "a BREAK b")
    message = str(excinfo.value)
    assert "3" in message and "2" in message
    assert "1,1,1" in message


def test_base_prompt_and_common_are_composed():
    args = list(RP_DEFAULT_ARGS)
    args[8] = True   # usebase
    args[9] = True   # usecom
    config = build_config(
        normalize_args(args), "quality ADDCOMM room ADDBASE a BREAK b"
    )
    assert config.region_prompts == ("quality, a", "quality, b")
    assert config.base_prompt == "quality, room"


def test_addbase_degrades_to_separator_when_usebase_off():
    # usebase=False：ADDBASE 退化为区域分隔符，得到 3 个区域，与 ratios "1,1" 不符
    with pytest.raises(ValueError, match="不匹配"):
        build_config(normalize_args(RP_DEFAULT_ARGS), "room ADDBASE a BREAK b")
```

- [ ] **Step 2: 运行测试确认失败**

Run: `F:\sd-webui-forge-neo\venv\Scripts\python.exe -m pytest tests/test_state.py -v`
Expected: FAIL，`ModuleNotFoundError: No module named 'lib_anima_rp.state'`

- [ ] **Step 3: 实现**

创建 `lib_anima_rp/state.py`：

```python
"""API 参数归一化、降级判定与配置构建。设计文档 §9。

参数顺序必须与 sd-webui-regional-prompter 的 scripts/rp.py:344 完全一致。
"""

from collections.abc import Sequence
from dataclasses import dataclass

from lib_anima_rp.layout import COLUMNS, ROWS, Rect, parse_base_weights, parse_layout
from lib_anima_rp.prompts import (
    compose_base_prompt,
    compose_region_prompts,
    has_any_keyword,
    split_prompt,
)

RP_ARG_NAMES = (
    "active",
    "debug",
    "rp_selected_tab",
    "mmode",
    "xmode",
    "pmode",
    "ratios",
    "baseratios",
    "usebase",
    "usecom",
    "usencom",
    "calcmode",
    "nchangeand",
    "lnter",
    "lnur",
    "threshold",
    "polymask",
    "lstop",
    "lstop_hr",
    "flipper",
)

RP_ARG_DEFAULTS = (
    False, False, "Matrix", "Columns", "Mask", "Prompt", "1,1", "0.2",
    False, False, False, "Attention", False, "0", "0", "0.4", None,
    "0", "0", False,
)

assert len(RP_ARG_NAMES) == len(RP_ARG_DEFAULTS) == 20


@dataclass(frozen=True)
class RegionalConfig:
    rects: tuple[Rect, ...]
    region_prompts: tuple[str, ...]
    base_prompt: str
    base_weights: tuple[float, ...]
    debug: bool
    degradations: tuple[str, ...]


def normalize_args(args: Sequence) -> dict:
    """把位置参数转成命名 dict，缺失的补 RP 默认值，多余的忽略。"""
    params = dict(zip(RP_ARG_NAMES, RP_ARG_DEFAULTS))
    for name, value in zip(RP_ARG_NAMES, args):
        params[name] = value
    return params


def collect_degradations(params: dict) -> tuple[str, ...]:
    """列出会影响画面结果、但本扩展不支持的参数取值。"""
    notes = []
    if params["rp_selected_tab"] != "Matrix":
        notes.append(f"tab={params['rp_selected_tab']}")
    if params["mmode"] not in (COLUMNS, ROWS):
        notes.append(f"mmode={params['mmode']}")
    if params["calcmode"] != "Attention":
        notes.append(f"calcmode={params['calcmode']}")
    if params["usencom"]:
        notes.append("usencom=True")
    return tuple(notes)


def build_config(params: dict, prompt: str) -> RegionalConfig | None:
    """构建运行期配置。返回 None 表示本次生成应静默不启用分区。"""
    if not params["active"]:
        return None
    if not has_any_keyword(prompt):
        # 提示词没写任何分隔关键字，静默不生效（RP rp.py:359 的行为）
        return None

    degradations = collect_degradations(params)

    mmode = params["mmode"] if params["mmode"] in (COLUMNS, ROWS) else COLUMNS
    ratios = params["ratios"]
    flipper = bool(params["flipper"])

    rects = parse_layout(ratios, mmode, flipper)
    segments = split_prompt(prompt, usecom=bool(params["usecom"]), usebase=bool(params["usebase"]))
    region_prompts = compose_region_prompts(segments)

    if len(rects) != len(region_prompts):
        raise ValueError(
            f"分区数与提示词段数不匹配：Divide ratio {ratios!r}（{mmode}"
            f"{'，已翻转分隔符' if flipper else ''}）解析出 {len(rects)} 个区域，"
            f"但提示词切出 {len(region_prompts)} 段。"
            f"请增删 BREAK，或调整 Divide ratio。"
        )

    base_prompt = compose_base_prompt(segments)
    if base_prompt:
        base_weights = tuple(parse_base_weights(params["baseratios"], ratios, mmode))
    else:
        base_weights = tuple(0.0 for _ in rects)

    return RegionalConfig(
        rects=tuple(rects),
        region_prompts=region_prompts,
        base_prompt=base_prompt,
        base_weights=base_weights,
        debug=bool(params["debug"]),
        degradations=degradations,
    )
```

- [ ] **Step 4: 运行测试确认通过**

Run: `F:\sd-webui-forge-neo\venv\Scripts\python.exe -m pytest tests/test_state.py -v`
Expected: 17 passed

- [ ] **Step 5: 全量回归**

Run: `F:\sd-webui-forge-neo\venv\Scripts\python.exe -m pytest -v`
Expected: 全部通过

- [ ] **Step 6: 提交**

```bash
git add lib_anima_rp/state.py tests/test_state.py
git commit -m "feat(state): normalize RP args and build validated regional config"
```

---

### Task 7: 掩码构建

**Files:**
- Create: `lib_anima_rp/masks.py`
- Create: `tests/test_masks.py`

**Interfaces:**
- Consumes: `Rect`（Task 2）
- Produces:
  - `build_masks(rects: Sequence[Rect], grid: tuple[int, int, int], device=None, dtype=None) -> torch.Tensor`，形状 `(N, T*H*W)`
  - `empty_region_indices(masks: torch.Tensor) -> tuple[int, ...]` — 返回一个 token 都没分到的区域下标
  - `MaskCache` 类，方法 `get(rects, grid, device, dtype) -> torch.Tensor`

- [ ] **Step 1: 写失败测试**

创建 `tests/test_masks.py`：

```python
import torch

from lib_anima_rp.layout import Rect, parse_layout
from lib_anima_rp.masks import MaskCache, build_masks, empty_region_indices


def test_single_region_covers_everything():
    masks = build_masks([Rect(0.0, 0.0, 1.0, 1.0)], (1, 4, 4))
    assert masks.shape == (1, 16)
    assert torch.all(masks == 1.0)


def test_partition_is_exact():
    """每个 token 恰好属于一个区域，Σmask 处处为 1。这是最重要的不变量。"""
    masks = build_masks(parse_layout("1,2,1,1;2,4,6", "Columns"), (1, 16, 16))
    total = masks.sum(dim=0)
    assert torch.allclose(total, torch.ones_like(total))


def test_partition_is_exact_for_awkward_ratios():
    masks = build_masks(parse_layout("1,1,1,1,1,1,1", "Rows"), (1, 15, 15))
    total = masks.sum(dim=0)
    assert torch.allclose(total, torch.ones_like(total))


def test_two_columns_split_left_right():
    masks = build_masks(parse_layout("1,1", "Columns"), (1, 2, 4))
    # (t h w) 展平：第 0 行是 w=0..3，第 1 行是 w=0..3
    assert masks[0].tolist() == [1, 1, 0, 0, 1, 1, 0, 0]
    assert masks[1].tolist() == [0, 0, 1, 1, 0, 0, 1, 1]


def test_two_rows_split_top_bottom():
    masks = build_masks(parse_layout("1,1", "Rows"), (1, 4, 2))
    assert masks[0].tolist() == [1, 1, 1, 1, 0, 0, 0, 0]
    assert masks[1].tolist() == [0, 0, 0, 0, 1, 1, 1, 1]


def test_flatten_order_is_t_h_w():
    masks = build_masks(parse_layout("1,1", "Columns"), (2, 2, 2))
    assert masks.shape == (2, 8)
    # 每个时间片内部都是 [左, 右, 左, 右]
    assert masks[0].tolist() == [1, 0, 1, 0, 1, 0, 1, 0]


def test_region_token_counts_match_ratio():
    masks = build_masks(parse_layout("1,3", "Columns"), (1, 8, 8))
    assert masks[0].sum().item() == 2 * 8
    assert masks[1].sum().item() == 6 * 8


def test_empty_region_is_reported_not_raised():
    # 4 个区域挤在宽度 2 的网格上，必然有区域分不到 token
    masks = build_masks(parse_layout("1,1,1,1", "Columns"), (1, 4, 2))
    empties = empty_region_indices(masks)
    assert len(empties) > 0
    total = masks.sum(dim=0)
    assert torch.allclose(total, torch.ones_like(total))


def test_dtype_and_device_respected():
    masks = build_masks([Rect(0.0, 0.0, 1.0, 1.0)], (1, 2, 2), dtype=torch.float16)
    assert masks.dtype == torch.float16


def test_cache_returns_same_tensor_for_same_key():
    cache = MaskCache()
    rects = parse_layout("1,1", "Columns")
    first = cache.get(rects, (1, 4, 4), torch.device("cpu"), torch.float32)
    second = cache.get(rects, (1, 4, 4), torch.device("cpu"), torch.float32)
    assert first is second


def test_cache_rebuilds_for_different_grid():
    cache = MaskCache()
    rects = parse_layout("1,1", "Columns")
    first = cache.get(rects, (1, 4, 4), torch.device("cpu"), torch.float32)
    second = cache.get(rects, (1, 8, 8), torch.device("cpu"), torch.float32)
    assert first is not second
    assert second.shape == (2, 64)
```

- [ ] **Step 2: 运行测试确认失败**

Run: `F:\sd-webui-forge-neo\venv\Scripts\python.exe -m pytest tests/test_masks.py -v`
Expected: FAIL，`ModuleNotFoundError: No module named 'lib_anima_rp.masks'`

- [ ] **Step 3: 实现**

创建 `lib_anima_rp/masks.py`：

```python
"""区域矩形 → token 掩码张量。设计文档 §8。

token 展平顺序恒为 (t h w)，与 backend/nn/anima.py 的
rearrange(x, "b t h w d -> b (t h w) d") 一致。
"""

from collections.abc import Sequence

import torch

from lib_anima_rp.layout import Rect


def build_masks(
    rects: Sequence[Rect],
    grid: tuple[int, int, int],
    device=None,
    dtype=None,
) -> torch.Tensor:
    """构建形状 (N, T*H*W) 的二值掩码。

    判定规则是 token 中心点落在哪个矩形，因此每个 token 恰好属于一个区域，
    Σmask 处处为 1。若不满足，区域边界会出现亮度漂移。
    """
    if dtype is None:
        dtype = torch.float32
    t_count, h_count, w_count = grid

    # token 中心的归一化坐标
    ys = (torch.arange(h_count, dtype=torch.float32) + 0.5) / h_count
    xs = (torch.arange(w_count, dtype=torch.float32) + 0.5) / w_count
    grid_y = ys.view(h_count, 1).expand(h_count, w_count).reshape(-1)
    grid_x = xs.view(1, w_count).expand(h_count, w_count).reshape(-1)

    # 每个空间位置归属的区域下标；后写入的覆盖先写入的，保证唯一归属
    owner = torch.zeros(h_count * w_count, dtype=torch.long)
    for index, rect in enumerate(rects):
        inside = (
            (grid_x >= rect.x0)
            & (grid_x < rect.x1)
            & (grid_y >= rect.y0)
            & (grid_y < rect.y1)
        )
        owner[inside] = index

    plane = torch.zeros(len(rects), h_count * w_count, dtype=dtype)
    plane.scatter_(0, owner.unsqueeze(0), 1.0)

    masks = plane.repeat(1, t_count)
    if device is not None:
        masks = masks.to(device)
    return masks


def empty_region_indices(masks: torch.Tensor) -> tuple[int, ...]:
    """返回一个 token 都没分到的区域下标。分辨率太低或区域太窄时会发生。"""
    sums = masks.sum(dim=1)
    return tuple(int(i) for i in torch.nonzero(sums == 0).flatten().tolist())


class MaskCache:
    """按 (区域矩形, 网格, device, dtype) 缓存掩码。

    hires 二阶段分辨率变化时 key 改变，自动重建。
    """

    def __init__(self):
        self._cache = {}

    @staticmethod
    def _key(rects, grid, device, dtype):
        rect_key = tuple((r.x0, r.y0, r.x1, r.y1) for r in rects)
        return (rect_key, tuple(grid), str(device), str(dtype))

    def get(self, rects, grid, device, dtype) -> torch.Tensor:
        key = self._key(rects, grid, device, dtype)
        cached = self._cache.get(key)
        if cached is None:
            cached = build_masks(rects, grid, device=device, dtype=dtype)
            self._cache[key] = cached
        return cached

    def clear(self):
        self._cache.clear()
```

- [ ] **Step 4: 运行测试确认通过**

Run: `F:\sd-webui-forge-neo\venv\Scripts\python.exe -m pytest tests/test_masks.py -v`
Expected: 11 passed

`scatter_` 那一步是关键：`owner` 记录每个空间位置的唯一归属区域，散射成 one-hot，天然保证 Σmask ≡ 1。若 `test_partition_is_exact` 失败，检查 `owner` 是否被正确初始化为 0（未被任何矩形覆盖的位置默认归属区域 0，而不是无归属）。

- [ ] **Step 5: 全量回归**

Run: `F:\sd-webui-forge-neo\venv\Scripts\python.exe -m pytest -v`
Expected: 全部通过

- [ ] **Step 6: 提交**

```bash
git add lib_anima_rp/masks.py tests/test_masks.py
git commit -m "feat(masks): build exact-partition token masks with caching"
```

---

### Task 8: 分区注意力

**Files:**
- Create: `lib_anima_rp/attention.py`
- Create: `tests/test_attention_math.py`

**Interfaces:**
- Consumes: 无（只吃张量与普通数据）
- Produces:
  - `RPState` — `@dataclass`（**可变，非 frozen**：采样期需要就地更新 `masks`），字段
    `masks: torch.Tensor` `(N, TOK)`、`region_contexts: torch.Tensor` `(N, L, ctx_dim)`、
    `base_context: torch.Tensor | None` `(1, L, ctx_dim)`、`base_weights: torch.Tensor` `(N,)`
  - `regional_attention(module, x, state: RPState, attention_op) -> torch.Tensor` —
    对**全部**输入行做分区注意力，返回 `(b, TOK, x_dim)`。
    `attention_op(q, k, v)` 接收 `(b, s, h, d)` 三元组，返回 `(b, s, h*d)`。

`regional_forward` 与 patch 在 Task 9 引入。

`module` 必须提供 `q_proj`、`q_norm`、`k_proj`、`k_norm`、`v_proj`、`v_norm`、`output_proj`、`n_heads`、`head_dim` 属性，即 `backend/nn/anima.py` 的 `SelfCrossAttention` 接口。测试用一个同接口的假模块，不依赖 Forge。

- [ ] **Step 1: 写失败测试**

创建 `tests/test_attention_math.py`：

```python
"""分区注意力的数值正确性。设计文档 §11.1。

用一个与 backend/nn/anima.py SelfCrossAttention 同接口的假模块，
不依赖 Forge 即可验证。
"""

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


def test_output_projection_can_be_hoisted():
    """验证设计文档 §2.1 性质 3：output_proj 无 bias，故可从加权和中外提。"""
    module, x, _ = make_case()
    a = torch.randn(1, 12, module.n_heads * module.head_dim)
    b = torch.randn(1, 12, module.n_heads * module.head_dim)
    m1 = torch.rand(12).unsqueeze(0).unsqueeze(-1)
    m2 = 1.0 - m1

    separate = m1 * module.output_proj(a) + m2 * module.output_proj(b)
    hoisted = module.output_proj(m1 * a + m2 * b)
    assert torch.allclose(separate, hoisted, atol=TOL)
```

- [ ] **Step 2: 运行测试确认失败**

Run: `F:\sd-webui-forge-neo\venv\Scripts\python.exe -m pytest tests/test_attention_math.py -v`
Expected: FAIL，`ModuleNotFoundError: No module named 'lib_anima_rp.attention'`

- [ ] **Step 3: 实现分区注意力核心**

创建 `lib_anima_rp/attention.py`：

```python
"""分区 cross-attention。设计文档 §7。

只依赖 torch。宿主模块 backend.nn.anima.SelfCrossAttention 的接口约定：
q_proj / q_norm / k_proj / k_norm / v_proj / v_norm / output_proj / n_heads / head_dim。
"""

from dataclasses import dataclass

import torch


@dataclass
class RPState:
    """采样期通过 transformer_options 传递的分区状态。"""

    masks: torch.Tensor              # (N, TOK)
    region_contexts: torch.Tensor    # (N, L, context_dim)
    base_context: torch.Tensor | None  # (1, L, context_dim)
    base_weights: torch.Tensor       # (N,)


def _to_heads(tensor, n_heads, head_dim):
    """(b, s, h*d) → (b, s, h, d)，等价于 einops 的 rearrange。"""
    b, s, _ = tensor.shape
    return tensor.view(b, s, n_heads, head_dim)


def regional_attention(module, x, state: RPState, attention_op) -> torch.Tensor:
    """对全部输入行做分区注意力，返回 (b, TOK, x_dim)。

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
        weight_map = (masks * weights.view(-1, 1)).sum(dim=0).view(1, -1, 1)
        base_ctx = state.base_context.to(device=x.device, dtype=x.dtype)
        if base_ctx.shape[0] != batch:
            base_ctx = base_ctx.expand(batch, -1, -1)
        k = module.k_norm(_to_heads(module.k_proj(base_ctx), n_heads, head_dim))
        v = module.v_norm(_to_heads(module.v_proj(base_ctx), n_heads, head_dim))
        base_attended = attention_op(q, k, v)
        acc = (1.0 - weight_map) * acc + weight_map * base_attended

    return module.output_proj(acc)
```

- [ ] **Step 4: 运行测试确认通过**

Run: `F:\sd-webui-forge-neo\venv\Scripts\python.exe -m pytest tests/test_attention_math.py -v`
Expected: 7 passed

若 `test_single_region_equals_plain_attention` 失败且差异微小，先确认 `FakeCrossAttention.reference_forward` 与 `regional_attention` 用的是同一个 attention 实现；两边都走 `scaled_dot_product_attention` 时应当在 `1e-4` 内一致。

- [ ] **Step 5: 提交**

```bash
git add lib_anima_rp/attention.py tests/test_attention_math.py
git commit -m "feat(attention): implement masked regional cross-attention"
```

---

### Task 9: forward patch 与 cond/uncond 分流

**Files:**
- Modify: `lib_anima_rp/attention.py`
- Modify: `tests/test_attention_math.py`（追加测试）

**Interfaces:**
- Consumes: `regional_attention`、`RPState`（Task 8）
- Produces:
  - `regional_forward(module, x, context, rope_emb, transformer_options, state, attention_op, orig_forward) -> torch.Tensor`
  - `patch_anima_attention() -> None` — 幂等，对 `backend.nn.anima.SelfCrossAttention.forward` 打补丁
  - `TRANSFORMER_OPTION_KEY = "anima_rp"`、`GRID_OPTION_KEY = "anima_rp_grid"`

- [ ] **Step 1: 写失败测试**

在 `tests/test_attention_math.py` 末尾追加：

```python
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
```

- [ ] **Step 2: 运行测试确认失败**

Run: `F:\sd-webui-forge-neo\venv\Scripts\python.exe -m pytest tests/test_attention_math.py -v`
Expected: 3 failed（`ImportError: cannot import name 'regional_forward'`），7 passed

- [ ] **Step 3: 实现分流与 patch**

在 `lib_anima_rp/attention.py` 末尾追加：

```python
TRANSFORMER_OPTION_KEY = "anima_rp"
GRID_OPTION_KEY = "anima_rp_grid"

_PATCHED = False
_ORIG_FORWARD = None


def regional_forward(
    module, x, context, rope_emb, transformer_options, state, attention_op, orig_forward
) -> torch.Tensor:
    """按 cond_mark 分流：正向行走分区路径，负向行走原路径。"""
    options = transformer_options or {}
    cond_indices = options.get("cond_indices")
    uncond_indices = options.get("uncond_indices")

    # 缺少索引信息时保守处理：全部按正向走分区路径
    if cond_indices is None and uncond_indices is None:
        return regional_attention(module, x, state, attention_op)

    cond_indices = list(cond_indices or [])
    uncond_indices = list(uncond_indices or [])

    if not cond_indices:
        # 纯负向批，一次性委托原实现
        return orig_forward(module, x, context, rope_emb, transformer_options)

    if not uncond_indices:
        return regional_attention(module, x, state, attention_op)

    out = torch.empty(
        x.shape[0], x.shape[1], module.output_proj.out_features,
        device=x.device, dtype=x.dtype,
    )
    out[cond_indices] = regional_attention(
        module, x[cond_indices], state, attention_op
    )
    out[uncond_indices] = orig_forward(
        module, x[uncond_indices], context[uncond_indices], rope_emb, transformer_options
    )
    return out


def patch_anima_attention() -> None:
    """给 backend.nn.anima.SelfCrossAttention.forward 打补丁。幂等。

    未启用时只是一次 dict 查找加委托，self-attn 与非 ANIMA 模型完全不受影响。
    """
    global _PATCHED, _ORIG_FORWARD
    if _PATCHED:
        return

    from backend.attention import attention_function
    from backend.nn.anima import SelfCrossAttention

    _ORIG_FORWARD = SelfCrossAttention.forward

    def anima_attention_op(q, k, v):
        return attention_function(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
            q.shape[2], skip_reshape=True,
        )

    def patched_forward(self, x, context=None, rope_emb=None, transformer_options=None):
        state = (transformer_options or {}).get(TRANSFORMER_OPTION_KEY)
        if state is None or self.is_SelfAttn or context is None:
            return _ORIG_FORWARD(self, x, context, rope_emb, transformer_options)
        return regional_forward(
            self, x, context, rope_emb, transformer_options, state,
            anima_attention_op, _ORIG_FORWARD,
        )

    SelfCrossAttention.forward = patched_forward
    _PATCHED = True
```

- [ ] **Step 4: 运行测试确认通过**

Run: `F:\sd-webui-forge-neo\venv\Scripts\python.exe -m pytest tests/test_attention_math.py -v`
Expected: 10 passed

`patch_anima_attention` 的 import 在函数体内，因此测试文件不导入它时不会触发 Forge 依赖。

- [ ] **Step 5: 全量回归**

Run: `F:\sd-webui-forge-neo\venv\Scripts\python.exe -m pytest -v`
Expected: 全部通过

- [ ] **Step 6: 提交**

```bash
git add lib_anima_rp/attention.py tests/test_attention_math.py
git commit -m "feat(attention): route cond and uncond rows, patch SelfCrossAttention"
```

---

### Task 10: Forge 接线与 UI

**Files:**
- Create: `scripts/anima_regional_prompter.py`

**Interfaces:**
- Consumes: `lib_anima_rp` 全部模块
- Produces: Forge 可加载的扩展；`title()` 返回 `"ANIMA Regional Prompter"`；`ui()` 返回 20 个组件

本任务无法用 pytest 验证，验收靠启动 Forge。步骤按「能加载 → UI 正确 → 不影响其他模型」递进。

- [ ] **Step 1: 写扩展主文件**

创建 `scripts/anima_regional_prompter.py`：

```python
"""ANIMA Regional Prompter —— Forge Neo 扩展入口。

参数顺序与 sd-webui-regional-prompter 的 scripts/rp.py:344 完全一致，
迁移时只需把 alwayson_scripts 的 key 改成 "ANIMA Regional Prompter"。
"""

import logging
import math
import os
import sys

import gradio as gr
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib_anima_rp.attention import (  # noqa: E402
    GRID_OPTION_KEY,
    TRANSFORMER_OPTION_KEY,
    RPState,
    patch_anima_attention,
)
from lib_anima_rp.layout import COLUMNS, ROWS, parse_layout  # noqa: E402
from lib_anima_rp.masks import MaskCache, empty_region_indices  # noqa: E402
from lib_anima_rp.prompts import strip_negative_keywords  # noqa: E402
from lib_anima_rp.state import build_config, normalize_args  # noqa: E402

from modules import scripts  # noqa: E402
from modules.ui_components import InputAccordion  # noqa: E402

logger = logging.getLogger("anima-rp")

MASK_CACHE = MaskCache()

PREVIEW_COLORS = (
    "#6ea8fe", "#f7a072", "#8fd694", "#d4a5e8", "#f6c453",
    "#7fd4d4", "#e88a9a", "#a9b5c9", "#c3e88d", "#e0a3c4",
)


def render_preview(ratios, mmode, flipper):
    """把当前分区渲染成一张 CSS grid 预览图，消除方向语义的记忆负担。"""
    try:
        rects = parse_layout(ratios, mmode, flipper)
    except ValueError as exc:
        return f'<div style="color:#e88a9a;padding:8px">{exc}</div>'

    cells = []
    for index, rect in enumerate(rects):
        color = PREVIEW_COLORS[index % len(PREVIEW_COLORS)]
        cells.append(
            f'<div style="position:absolute;'
            f"left:{rect.x0 * 100:.4f}%;top:{rect.y0 * 100:.4f}%;"
            f"width:{(rect.x1 - rect.x0) * 100:.4f}%;height:{(rect.y1 - rect.y0) * 100:.4f}%;"
            f"background:{color};border:1px solid rgba(0,0,0,.35);box-sizing:border-box;"
            f'display:flex;align-items:center;justify-content:center;'
            f'font:600 13px/1 sans-serif;color:#1a1a1a">{index}</div>'
        )
    return (
        '<div style="position:relative;width:100%;max-width:320px;aspect-ratio:1;'
        'margin:4px 0;border-radius:4px;overflow:hidden">' + "".join(cells) + "</div>"
    )


def is_anima(p) -> bool:
    try:
        model = p.sd_model.forge_objects.unet.model.diffusion_model
    except AttributeError:
        return False
    return model.__class__.__name__ == "Anima"


class AnimaRegionalPrompter(scripts.Script):
    sorting_priority = 20

    def __init__(self):
        super().__init__()
        self.config = None
        self.state = None
        self.warned_non_anima = False

    def title(self):
        return "ANIMA Regional Prompter"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with InputAccordion(False, label=self.title()) as active:
            with gr.Row():
                mmode = gr.Radio(
                    label="Main Splitting", choices=[COLUMNS, ROWS],
                    value=COLUMNS, type="value",
                )
                ratios = gr.Textbox(label="Divide Ratio", lines=1, value="1,1")
            preview = gr.HTML(value=render_preview("1,1", COLUMNS, False))
            with gr.Row():
                baseratios = gr.Textbox(label="Base Ratio", lines=1, value="0.2")
                flipper = gr.Checkbox(value=False, label='flip "," and ";"')
            with gr.Row():
                usebase = gr.Checkbox(value=False, label="Use base prompt")
                usecom = gr.Checkbox(value=False, label="Use common prompt")
                debug = gr.Checkbox(value=False, label="debug")

        # 仅为 API 兼容而存在的占位参数，取值固定为 RP 默认值
        rp_selected_tab = gr.State("Matrix")
        xmode = gr.State("Mask")
        pmode = gr.State("Prompt")
        usencom = gr.State(False)
        calcmode = gr.State("Attention")
        nchangeand = gr.State(False)
        lnter = gr.State("0")
        lnur = gr.State("0")
        threshold = gr.State("0.4")
        polymask = gr.State(None)
        lstop = gr.State("0")
        lstop_hr = gr.State("0")

        for control in (ratios, mmode, flipper):
            control.change(
                fn=render_preview, inputs=[ratios, mmode, flipper], outputs=[preview]
            )

        self.infotext_fields = [
            (active, "RP Active"),
            (mmode, "RP Main splitting"),
            (ratios, "RP Divide ratio"),
            (baseratios, "RP Base ratio"),
            (usebase, "RP Use base"),
            (usecom, "RP Use common"),
            (flipper, "RP Flip"),
        ]
        self.paste_field_names = [name for _, name in self.infotext_fields]

        # 顺序必须与 RP scripts/rp.py:344 一致
        return [
            active, debug, rp_selected_tab, mmode, xmode, pmode, ratios, baseratios,
            usebase, usecom, usencom, calcmode, nchangeand, lnter, lnur, threshold,
            polymask, lstop, lstop_hr, flipper,
        ]

    def process(self, p, *args):
        self.config = None
        self.state = None

        params = normalize_args(args)
        prompt = p.prompt[0] if isinstance(p.prompt, list) else p.prompt

        config = build_config(params, prompt)
        if config is None:
            return

        if not is_anima(p):
            if not self.warned_non_anima:
                logger.warning(
                    "当前模型不是 ANIMA，ANIMA Regional Prompter 不生效。"
                )
                self.warned_non_anima = True
            return

        patch_anima_attention()

        negative = p.negative_prompt[0] if isinstance(p.negative_prompt, list) else p.negative_prompt
        cleaned_negative, changed = strip_negative_keywords(negative)
        if changed:
            logger.warning("负面提示词不分区，已剥离其中的分隔关键字。")
            if isinstance(p.negative_prompt, list):
                p.negative_prompt = [cleaned_negative] * len(p.negative_prompt)
            else:
                p.negative_prompt = cleaned_negative

        texts = list(config.region_prompts)
        if config.base_prompt:
            texts.append(config.base_prompt)

        # 只在 process() 编码一次，hires 二阶段复用同一批 cond，
        # 故设计文档 §5.3 要求的缓存由「编码时机」本身满足，无需额外缓存层。
        conds = p.sd_model.get_learned_conditioning(texts)
        stacked = _pad_and_stack(conds)

        region_count = len(config.region_prompts)
        base_context = stacked[region_count:] if config.base_prompt else None

        self.config = config
        self.state = RPState(
            masks=torch.zeros(region_count, 1),   # 采样期按实际网格重建
            region_contexts=stacked[:region_count],
            base_context=base_context,
            base_weights=torch.tensor(config.base_weights),
        )

        p.extra_generation_params.update({
            "RP Active": True,
            "RP Divide ratio": params["ratios"],
            "RP Main splitting": params["mmode"],
            "RP Base ratio": params["baseratios"],
            "RP Use base": bool(params["usebase"]),
            "RP Use common": bool(params["usecom"]),
            "RP Flip": bool(params["flipper"]),
        })
        if config.degradations:
            joined = ", ".join(config.degradations)
            logger.warning(f"以下参数不受支持，已降级：{joined}")
            p.extra_generation_params["RP Unsupported"] = joined

    def process_before_every_sampling(self, p, *args, **kwargs):
        if self.config is None or self.state is None:
            return

        unet = p.sd_model.forge_objects.unet.clone()
        diffusion_model = unet.model.diffusion_model
        patch_spatial = diffusion_model.patch_spatial
        patch_temporal = diffusion_model.patch_temporal

        config = self.config
        state = self.state
        previous_wrapper = unet.model_options.get("model_function_wrapper")

        def wrapper(model_function, wrapper_kwargs):
            latent = wrapper_kwargs["input"]
            grid = (
                math.ceil(latent.shape[2] / patch_temporal),
                math.ceil(latent.shape[3] / patch_spatial),
                math.ceil(latent.shape[4] / patch_spatial),
            )
            options = wrapper_kwargs["c"]["transformer_options"]
            options[GRID_OPTION_KEY] = grid
            state.masks = MASK_CACHE.get(
                config.rects, grid, latent.device, latent.dtype
            )
            if config.debug:
                empties = empty_region_indices(state.masks)
                if empties:
                    logger.warning(f"以下区域在 {grid} 网格上分不到 token：{empties}")
            if previous_wrapper is not None:
                return previous_wrapper(model_function, wrapper_kwargs)
            return model_function(
                wrapper_kwargs["input"], wrapper_kwargs["timestep"], **wrapper_kwargs["c"]
            )

        unet.set_model_unet_function_wrapper(wrapper)
        unet.set_transformer_option(TRANSFORMER_OPTION_KEY, state)
        p.sd_model.forge_objects.unet = unet


def _pad_and_stack(conds):
    """把长度不一的 cond 对齐后堆成 (N, L, D)。

    补齐方式是重复最后一个向量，与 Forge 的 prompt_parser.stack_conds 一致。
    """
    flattened = [c[0] if c.dim() == 3 else c for c in conds]
    longest = max(c.shape[0] for c in flattened)
    padded = []
    for cond in flattened:
        if cond.shape[0] < longest:
            tail = cond[-1:].repeat(longest - cond.shape[0], 1)
            cond = torch.cat([cond, tail], dim=0)
        padded.append(cond)
    return torch.stack(padded)
```

- [ ] **Step 2: 确认 Forge 能加载扩展**

Run: `F:\sd-webui-forge-neo\venv\Scripts\python.exe -c "import ast,sys; ast.parse(open(r'F:\anima-regional-prompter\scripts\anima_regional_prompter.py',encoding='utf-8').read()); print('语法 OK')"`
Expected: `语法 OK`

然后启动 Forge：

Run: `F:\sd-webui-forge-neo\webui-user.bat`
Expected: 启动日志中无 `Error loading script: anima_regional_prompter.py`

- [ ] **Step 3: 确认 UI 正确渲染**

在浏览器打开 txt2img 页面，展开 `ANIMA Regional Prompter` 面板。

预期：`Main Splitting` 单选、`Divide Ratio` 文本框、一张两列的布局预览、`Base Ratio`、`flip "," and ";"`、三个复选框。把 `Divide Ratio` 改成 `1,2,1,1;2,4,6`，预览应立即变成 2 行 5 区。

- [ ] **Step 4: 确认非 ANIMA 模型不受影响**

切换到任意非 ANIMA 模型，保持面板关闭，生成一张图。
Expected: 正常出图，控制台无 anima-rp 相关报错。

- [ ] **Step 5: 提交**

```bash
git add scripts/anima_regional_prompter.py
git commit -m "feat(scripts): wire regional prompter into Forge Neo with RP-compatible UI"
```

---

### Task 11: 端到端验证与文档收尾

**Files:**
- Modify: `README.md`
- Modify: `docs/superpowers/specs/2026-07-31-anima-regional-prompter-design.md`（仅在发现与实现不符时）

**Interfaces:**
- Consumes: 全部
- Produces: 可交付的扩展

- [ ] **Step 1: 一维分区实图验证**

在 Forge 中选择 `anima_baseV10.safetensors`，展开面板并勾选启用。

- Divide Ratio: `1,1`
- Main Splitting: `Columns`
- 提示词：`1girl, red hair, red dress BREAK 1boy, blue hair, blue suit`
- 分辨率 1024×1024，步数 20

Expected: 左半边红发红裙女性，右半边蓝发蓝西装男性。若左右颜色互相污染，先检查 `test_masks.py::test_two_columns_split_left_right` 是否仍通过，再检查 `state.masks` 在 wrapper 中是否被正确更新。

- [ ] **Step 2: 二维分区实图验证**

- Divide Ratio: `1,1;2,1,1`
- Main Splitting: `Columns`
- 提示词：`sky BREAK grass BREAK 1girl BREAK 1boy`

Expected: 上半整块天空，下半三等分为草地、女孩、男孩。

- [ ] **Step 3: Base prompt 验证**

- 勾选 `Use base prompt`，Base Ratio 填 `0.3`
- 提示词：`a sunlit forest ADDBASE 1girl, red dress BREAK 1boy, blue suit`

Expected: 左右人物区分仍在，但整体背景统一为阳光森林，区域边界的割裂感明显减轻。把 Base Ratio 调到 `0` 应退化成不带 base 的结果。

- [ ] **Step 4: API 调用验证**

创建临时脚本并运行：

```python
import base64
import json
import urllib.request

payload = {
    "prompt": "1girl, red hair, red dress BREAK 1boy, blue hair, blue suit",
    "negative_prompt": "worst quality",
    "width": 1024,
    "height": 1024,
    "steps": 20,
    "alwayson_scripts": {
        "ANIMA Regional Prompter": {
            "args": [True, False, "Matrix", "Columns", "Mask", "Prompt", "1,1", "0.2",
                     False, False, False, "Attention", False, "0", "0", "0.4", None,
                     "0", "0", False]
        }
    },
}

request = urllib.request.Request(
    "http://127.0.0.1:7860/sdapi/v1/txt2img",
    data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(request) as response:
    result = json.load(response)

with open("api_check.png", "wb") as handle:
    handle.write(base64.b64decode(result["images"][0]))
print("已保存 api_check.png")
print(json.loads(result["info"])["infotexts"][0])
```

Run: `F:\sd-webui-forge-neo\venv\Scripts\python.exe api_check.py`
Expected: 生成的图与 Step 1 构图一致；打印的 infotext 含 `RP Active: True`、`RP Divide ratio: 1,1`。

- [ ] **Step 5: 降级路径验证**

把上面 payload 的第 12 个参数（下标 11）改成 `"Latent"` 再跑一次。
Expected: 正常出图（按 Attention 模式），控制台出现「以下参数不受支持，已降级：calcmode=Latent」，infotext 含 `RP Unsupported: calcmode=Latent`。

- [ ] **Step 6: 无 BREAK 静默降级验证**

保持面板启用，提示词改成不含任何关键字的 `1girl, red dress`，生成。
Expected: 正常出图，无报错、无警告，infotext 中不出现 `RP Active`。

- [ ] **Step 7: 更新 README 的实现差异章节**

在 `README.md` 的「与 RP 的已知差异」小节补充实际发现的差异（若 Step 1–6 中出现任何与文档不符之处）。同时把「状态：开发中」改为记录已验证的功能范围。

- [ ] **Step 8: 全量回归**

Run: `F:\sd-webui-forge-neo\venv\Scripts\python.exe -m pytest -v`
Expected: 全部通过

- [ ] **Step 9: 提交**

```bash
git add README.md docs/
git commit -m "docs: record end-to-end verification results"
git push origin main
```

---

## 附录：验收清单

交付前逐项确认：

- [ ] `pytest` 全绿，含 RP 布局一致性断言（非 skip）
- [ ] 一维、二维分区实图符合预期
- [ ] Base prompt 生效且 Base Ratio 为 0 时可退化
- [ ] API 调用与 UI 路径产出一致
- [ ] `calcmode="Latent"` 走降级而非报错
- [ ] 提示词无 `BREAK` 时静默不生效
- [ ] 非 ANIMA 模型完全不受影响
- [ ] 分区数与提示词段数不匹配时报错信息包含两个数字与 Divide ratio 原文
