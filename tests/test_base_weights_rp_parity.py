"""与 sd-webui-regional-prompter 的权重计算一致性断言。

参照实现按 RP regions.py:133-162 的 split_l2(basestruct=...) 分支复刻，
对应 matrixdealer 中的调用（regions.py:379）：

    bratios2 = split_l2(bratios, ";", ",", fmap=ffloatd(0),
                         basestruct=lbreaks, indflip=indflip)

其中 lbreaks 与区域的带/单元结构（aratios2）同形状、值全为 0——
basestruct 只有其形状参与广播，值本身不参与计算。indflip 由
mmode 决定（Rows 时为 True），这是 mmode 自身的行为，与 flipper
开关（rp.py:386，只对 aratios 调 changecs）无关。

**flipper 与结构的关系（修复轮 4）**：rp.py:386 的 `changecs` 发生在
matrixdealer 之前，传入的 aratios 已经是翻转过的，basestruct 的形状
（lbreaks）就是从翻转后的结构推出来的；但 bratios 字符串本身从始至终
没有被 changecs 处理过。所以：flipper 影响 ratios → 影响结构 → 影响
basestruct 形状；不影响 baseratios 字符串怎么切。
"""

import pytest

from lib_anima_rp.layout import (
    COLUMNS,
    ROWS,
    _parse_ratio_structure,
    parse_base_weights,
    parse_layout,
    swap_delimiters,
)

TOL = 1e-9


def _cell_rows_shape(ratios, mmode, flipper):
    """取布局的带/单元结构形状（每带的单元数，不含带尺寸值本身）。

    复用 lib_anima_rp.layout._parse_ratio_structure（parse_layout 内部用的
    也是这一个函数），并按 flipper 对 ratios 做与 parse_layout 完全一致的
    翻转预处理。这里只关心形状，不重新发明一遍分支判断逻辑。

    注意：这意味着本文件对形状的构造和被测实现共享同一个底层函数，
    一个只存在于 _parse_ratio_structure/_split_bands 内部的 bug 可能会
    同时影响两边而不被发现——这类"真正独立"的形状校验由
    tests/test_layout_rp_parity.py 提供（它直接调 RP 自己的 split_l2，
    完全不经过我们的解析代码）。两个文件用到的每个 (ratios, mmode, flipper)
    组合都必须能在 test_layout_rp_parity.py 的 CASES 里找到对应项。
    """
    text = ratios.strip()
    if flipper:
        text = swap_delimiters(text)
    _, cell_rows = _parse_ratio_structure(text, mmode)
    return [len(row) for row in cell_rows]


def rp_reference_weights(rp_regions, baseratios, ratios, mmode, flipper):
    """使用 RP 自己的 split_l2 计算期望的权重列表。

    basestruct 的形状来自翻转后的 ratios（对应 lbreaks），baseratios 字符串
    本身传给 RP 的 split_l2 时不做任何翻转处理——这正是 RP 的真实行为。
    """
    shape = _cell_rows_shape(ratios, mmode, flipper)
    basestruct = [[None] * n for n in shape]
    indflip = mmode == ROWS

    values_per_band = rp_regions.split_l2(
        baseratios,
        ";",
        ",",
        basestruct=basestruct,
        fmap=rp_regions.ffloatd(0),
        indsingles=False,
        indflip=indflip,
    )

    result = []
    for band_values in values_per_band:
        result.extend(band_values)
    return result


# 测试用例：(baseratios, ratios, mmode, flipper)
CASES = [
    # 单值：广播到所有区域
    ("0.2", "1,1,1", COLUMNS, False),
    # 逐区域刚好对齐
    ("0.1,0.2,0.3", "1,1,1", COLUMNS, False),
    # 一维带内不足：重复该带（唯一的带）自己的最后一个值
    ("0.1,0.2", "1,1,1,1", COLUMNS, False),
    # 一维带内超出：截断（只有一个带，没有下一带可溢出）
    ("0.1,0.2,0.3,0.4", "1,1", COLUMNS, False),
    # 值远多于总区域数
    ("0.1,0.2,0.3,0.4,0.5,0.6,0.7", "1,1,1", COLUMNS, False),
    # 含非数字（退化为 0）
    ("0.2,x,0.3", "1,1,1", COLUMNS, False),
    # 二维：带内不足，重复该段自己的最后值
    ("0.1,0.2;0.3,0.4", "1,2,1,1;2,4,6", COLUMNS, False),
    # 二维：带内超出，溢出到下一带
    ("0.1,0.2,0.3,0.4;0.5", "1,2,1,1;2,4,6", COLUMNS, False),
    # 二维：逐带刚好对齐
    ("0.1,0.2,0.3;0.4,0.5", "1,2,1,1;2,4,6", COLUMNS, False),
    # 二维：值远多于总区域数
    ("0.1,0.2,0.3;0.4,0.5;0.6,0.7,0.8,0.9", "1,1,1;2,2", COLUMNS, False),
    # 一维 Rows 模式
    ("0.1,0.2,0.3", "1,1,1", ROWS, False),
    # 二维 Rows 模式
    ("0.1;0.2,0.3;0.4,0.5", "1,2,1;2,3,1", ROWS, False),
    # --- flipper=True：以下 5 个是团队复审报告里"长度契约破裂"的原始用例 ---
    # flip 把 "1,2;3,4" 变成 3 带各 1 单元（原本 2 带各 1/2 单元），区域数 2→3
    ("0.1,0.2", "1,2;3,4", COLUMNS, True),
    # flip 把 "1,2,1,1;2,4,6" 变成 6 带各 1 单元，区域数 5→6
    ("0.1,0.2,0.3,0.4,0.5", "1,2,1,1;2,4,6", COLUMNS, True),
    # flip 把 "2,3;1,1,1;4,5" 变成 5 带各 1 单元，区域数 4→5
    ("0.1,0.2,0.3,0.4", "2,3;1,1,1;4,5", COLUMNS, True),
    # flip 把一维 "1,1" 变成 "1;1"（2 带各 1 单元），区域数 2→1
    ("0.2", "1,1", ROWS, True),
    # flip 把一维 "1,2,3" 变成 "1;2;3"（3 带各 1 单元），区域数 3→2
    ("0.1,0.2", "1,2,3", ROWS, True),
]


@pytest.mark.parametrize("baseratios,ratios,mmode,flipper", CASES)
def test_matches_rp_weights(rp_regions, baseratios, ratios, mmode, flipper):
    """校验 parse_base_weights 与 RP 的权重计算一致。"""
    expected = rp_reference_weights(rp_regions, baseratios, ratios, mmode, flipper)
    actual = parse_base_weights(baseratios, ratios, mmode, flipper)

    # 独立锚点：区域数量由 parse_layout 用同样的 (ratios, mmode, flipper) 独立给出
    # （已被 tests/test_layout_rp_parity.py 对照 RP 验证过），单独就能抓住
    # "权重数量与区域数不符"这类 bug，即便 basestruct 形状构造和被测实现共享了
    # 同一个错误也不会被掩盖。
    region_count = len(parse_layout(ratios, mmode, flipper))
    assert len(actual) == region_count, (
        f"{baseratios!r}/{ratios!r}/{mmode}/flip={flipper} 权重数 {len(actual)} "
        f"与 parse_layout 给出的区域数 {region_count} 不符"
    )

    assert len(actual) == len(expected), (
        f"{baseratios!r}/{ratios!r}/{mmode}/flip={flipper} 区域数不符："
        f"我们 {len(actual)}，RP {len(expected)}"
    )
    for i, (a, e) in enumerate(zip(actual, expected)):
        assert abs(a - e) < TOL, (
            f"{baseratios!r}/{ratios!r}/{mmode}/flip={flipper} 区域 {i} 权重不符："
            f"我们 {a}，RP {e}"
        )


# 长度契约的全组合断言：单独覆盖 flipper 两种取值（上面的 test_matches_rp_weights
# 已经把这条断言作为独立锚点内嵌了一遍，这里额外用固定的单值 baseratios 铺开更多
# (ratios, mmode, flipper) 组合，专注于长度契约本身，不依赖具体取值是否正确）。
LENGTH_CASES = [
    ("1,1,1", COLUMNS, False),
    ("1,1,1,1", COLUMNS, False),
    ("1,1", COLUMNS, False),
    ("1,2,1,1;2,4,6", COLUMNS, False),
    ("1,1,1;2,2", COLUMNS, False),
    ("1,1,1", ROWS, False),
    ("1,2,1;2,3,1", ROWS, False),
    ("1,2;3,4", COLUMNS, True),
    ("1,2,1,1;2,4,6", COLUMNS, True),
    ("2,3;1,1,1;4,5", COLUMNS, True),
    ("1,1", ROWS, True),
    ("1,2,3", ROWS, True),
]


@pytest.mark.parametrize("ratios,mmode,flipper", LENGTH_CASES)
def test_weight_count_matches_region_count(ratios, mmode, flipper):
    region_count = len(parse_layout(ratios, mmode, flipper))
    weight_count = len(parse_base_weights("0.2", ratios, mmode, flipper))
    assert weight_count == region_count, (
        f"{ratios!r}/{mmode}/flip={flipper} 权重数 {weight_count} 与区域数 "
        f"{region_count} 不符"
    )
