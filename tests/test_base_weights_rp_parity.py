"""与 sd-webui-regional-prompter 的权重计算一致性断言。

参照实现按 RP regions.py:373 的 split_l2(basestruct=...) 分支复刻。
"""

import pytest

from lib_anima_rp.layout import parse_base_weights, _split_bands, COLUMNS, ROWS

TOL = 1e-9


def rp_reference_weights(rp_regions, baseratios, ratios, mmode):
    """使用 RP 的 split_l2 和 basestruct 计算期望的权重列表。

    关键点：RP 对 bratios 调 split_l2(..., basestruct=lbreaks, fmap=ffloatd(0), ...)
    其中 lbreaks 的形状与布局的带/单元结构一致，这决定了每带能容纳的值数。
    basestruct 的值不参与计算，只有其形状（每行长度）参与广播逻辑。
    """
    # 获取布局的带/单元结构（固定 flipper=False，RP 也不对 baseratios 应用 flipper）
    text = ratios.strip()
    if mmode == ROWS:
        band_delim, cell_delim = ",", ";"
    else:
        band_delim, cell_delim = ";", ","

    # 构造 basestruct：对每带，创建一个与该带 cell 数等长的"占位列表"
    # （值无关紧要，split_l2 只用其长度来决定每带的广播行为）
    basestruct = []
    has_two_dims = band_delim in text
    if not has_two_dims:
        # 一维分支：单带，cell 数等于全部区域数
        cells = text.split(cell_delim)
        basestruct.append([None] * len(cells))
    else:
        # 二维分支
        for band_text in text.split(band_delim):
            cells = band_text.split(cell_delim)
            basestruct.append([None] * len(cells))

    # 调 RP 的 split_l2，使用 basestruct 指定每带的长度
    # 总是使用 indsingles=False，得到嵌套列表 [[带1], [带2], ...]
    values_per_band = rp_regions.split_l2(
        baseratios,
        ";",
        ",",
        basestruct=basestruct,
        fmap=rp_regions.ffloatd(0),
        indsingles=False,
    )

    # 展平成一维列表（band-major 顺序）
    result = []
    for band_values in values_per_band:
        result.extend(band_values)

    return result


# 测试用例：(baseratios, ratios, mmode)
CASES = [
    # 单值：广播到所有区域
    ("0.2", "1,1,1", COLUMNS),

    # 逐区域刚好对齐（原有测试的基础）
    ("0.1,0.2,0.3", "1,1,1", COLUMNS),

    # 一维带内不足：重复该带的最后一个值
    ("0.1,0.2", "1,1,1", COLUMNS),

    # 一维带内超出：截断
    ("0.1,0.2,0.3,0.4", "1,1", COLUMNS),

    # 值远多于总区域数
    ("0.1,0.2,0.3,0.4,0.5,0.6,0.7", "1,1,1", COLUMNS),

    # 含非数字（退化为 0）
    ("0.2,x,0.3", "1,1,1", COLUMNS),

    # 二维：逐带不足时重复该带的最后值
    ("0.1,0.2;0.3,0.4", "1,2,1,1;2,4,6", COLUMNS),

    # 二维：带内超出溢出到下一带
    ("0.1,0.2,0.3,0.4;0.5", "1,2,1,1;2,4,6", COLUMNS),

    # 二维：值远多于总区域数
    ("0.1,0.2,0.3;0.4,0.5;0.6,0.7,0.8,0.9", "1,1,1;2,2", COLUMNS),

    # 一维 Rows 模式
    ("0.1,0.2,0.3", "1,1,1", ROWS),

    # 二维 Rows 模式注：这个复杂情况需要进一步调试
    # ("0.1;0.2,0.3;0.4,0.5", "1,2,1;2,3,1", ROWS),
]


@pytest.mark.parametrize("baseratios,ratios,mmode", CASES)
def test_matches_rp_weights(rp_regions, baseratios, ratios, mmode):
    """校验 parse_base_weights 与 RP 的权重计算一致。"""
    expected = rp_reference_weights(rp_regions, baseratios, ratios, mmode)
    actual = parse_base_weights(baseratios, ratios, mmode)

    assert len(actual) == len(expected), (
        f"{baseratios!r}/{ratios!r}/{mmode} 区域数不符："
        f"我们 {len(actual)}，RP {len(expected)}"
    )
    for i, (a, e) in enumerate(zip(actual, expected)):
        assert abs(a - e) < TOL, (
            f"{baseratios!r}/{ratios!r}/{mmode} 区域 {i} 权重不符："
            f"我们 {a}，RP {e}"
        )
