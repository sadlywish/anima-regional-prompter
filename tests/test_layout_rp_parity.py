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
    ("1,1,1;3", "Columns", True),
    ("1;2;1;1,2;4;6", "Columns", True),
    # 以下四条是为了让 tests/test_base_weights_rp_parity.py 用到的每个
    # (ratios, mmode) 带/单元形状都能在这里找到一个完全独立的对照点
    # （rp_reference_rects 直接调 RP 的 split_l2，不经过我们的 _split_bands）。
    # 起因：复审在 _split_bands 里注入过一个只在特定形状下才现形的 bug，
    # 而当时这四种形状都不在本文件里，导致独立锚点也没能抓住——它依赖的
    # parse_layout 和被测实现共用同一个 _split_bands，一起被那个 bug 带偏了。
    ("1,1,1", "Columns", False),
    ("1,1,1,1", "Columns", False),
    ("1,1,1;2,2", "Columns", False),
    ("1,2,1;2,3,1", "Rows", False),
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
