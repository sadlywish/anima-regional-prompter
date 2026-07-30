"""与 sd-webui-regional-prompter 的权重计算一致性断言。

参照实现按 RP regions.py:133-162 的 split_l2(basestruct=...) 分支复刻，
对应 matrixdealer 中的调用（regions.py:379）：

    bratios2 = split_l2(bratios, ";", ",", fmap=ffloatd(0),
                         basestruct=lbreaks, indflip=indflip)

其中 lbreaks 与区域的带/单元结构（aratios2）同形状、值全为 0——
basestruct 只有其形状参与广播，值本身不参与计算。indflip 由
mmode 决定（Rows 时为 True），这是 mmode 自身的行为，与 flipper
开关（rp.py:386，只对 aratios 调 changecs）无关。
"""

import pytest

from lib_anima_rp.layout import (
    COLUMNS,
    ROWS,
    _split_bands,
    parse_base_weights,
    parse_layout,
)

TOL = 1e-9


def _cell_rows_shape(ratios, mmode):
    """取布局的带/单元结构形状（每带的单元数，不含带尺寸值本身）。

    分支逻辑必须与 parse_layout 一致：一维（原串不含 ';'）时是单带，
    单元数等于逗号分隔的段数；二维时复用 _split_bands（已被
    tests/test_layout_rp_parity.py 独立对照 RP 验证过），取其
    cell_rows 各行的长度。这里只关心形状，不重新发明 _split_bands
    的解析逻辑，避免和被测实现共用同一个可能出错的结构计算。
    """
    text = ratios.strip()
    if ";" not in text:
        return [len(text.split(","))]
    _, cell_rows = _split_bands(text, mmode)
    return [len(row) for row in cell_rows]


def rp_reference_weights(rp_regions, baseratios, ratios, mmode):
    """使用 RP 自己的 split_l2 计算期望的权重列表。

    basestruct 只用其形状，这里用 None 占位构造同形状的列表。
    """
    shape = _cell_rows_shape(ratios, mmode)
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


# 测试用例：(baseratios, ratios, mmode)
CASES = [
    # 单值：广播到所有区域
    ("0.2", "1,1,1", COLUMNS),
    # 逐区域刚好对齐
    ("0.1,0.2,0.3", "1,1,1", COLUMNS),
    # 一维带内不足：重复该带（唯一的带）自己的最后一个值
    ("0.1,0.2", "1,1,1,1", COLUMNS),
    # 一维带内超出：截断（只有一个带，没有下一带可溢出）
    ("0.1,0.2,0.3,0.4", "1,1", COLUMNS),
    # 值远多于总区域数
    ("0.1,0.2,0.3,0.4,0.5,0.6,0.7", "1,1,1", COLUMNS),
    # 含非数字（退化为 0）
    ("0.2,x,0.3", "1,1,1", COLUMNS),
    # 二维：带内不足，重复该段自己的最后值
    ("0.1,0.2;0.3,0.4", "1,2,1,1;2,4,6", COLUMNS),
    # 二维：带内超出，溢出到下一带
    ("0.1,0.2,0.3,0.4;0.5", "1,2,1,1;2,4,6", COLUMNS),
    # 二维：逐带刚好对齐
    ("0.1,0.2,0.3;0.4,0.5", "1,2,1,1;2,4,6", COLUMNS),
    # 二维：值远多于总区域数
    ("0.1,0.2,0.3;0.4,0.5;0.6,0.7,0.8,0.9", "1,1,1;2,2", COLUMNS),
    # 一维 Rows 模式
    ("0.1,0.2,0.3", "1,1,1", ROWS),
    # 二维 Rows 模式
    ("0.1;0.2,0.3;0.4,0.5", "1,2,1;2,3,1", ROWS),
]


@pytest.mark.parametrize("baseratios,ratios,mmode", CASES)
def test_matches_rp_weights(rp_regions, baseratios, ratios, mmode):
    """校验 parse_base_weights 与 RP 的权重计算一致。"""
    expected = rp_reference_weights(rp_regions, baseratios, ratios, mmode)
    actual = parse_base_weights(baseratios, ratios, mmode)

    # 独立锚点：区域数量由 parse_layout 独立给出（已被 tests/test_layout_rp_parity.py
    # 对照 RP 验证过，不依赖本文件或被测实现里的任何结构计算），单独就能抓住
    # "权重数量与区域数不符"这类 bug，即便 basestruct 形状构造和被测实现共享了
    # 同一个错误也不会被掩盖。
    region_count = len(parse_layout(ratios, mmode))
    assert len(actual) == region_count, (
        f"{baseratios!r}/{ratios!r}/{mmode} 权重数 {len(actual)} 与 parse_layout "
        f"给出的区域数 {region_count} 不符"
    )

    assert len(actual) == len(expected), (
        f"{baseratios!r}/{ratios!r}/{mmode} 区域数不符："
        f"我们 {len(actual)}，RP {len(expected)}"
    )
    for i, (a, e) in enumerate(zip(actual, expected)):
        assert abs(a - e) < TOL, (
            f"{baseratios!r}/{ratios!r}/{mmode} 区域 {i} 权重不符："
            f"我们 {a}，RP {e}"
        )
