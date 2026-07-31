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


def test_edge_clipping_with_non_power_of_two_ratios():
    """验证分母不是 2 的幂时，最后一段仍贴边到 1.0。

    十等分 "1,1,1,1,1,1,1,1,1,1" 会产生浮点累积误差：最后一段末尾不是 1.0，
    而是 0.9999999999... 。_spans 函数中最后一行 `spans[-1] = (spans[-1][0], 1.0)`
    用于消除这个误差。此测试用 == 而非近似比较锁住这个边界，防止该行代码被误删。
    """
    # Columns 模式：十等分横向
    rects = parse_layout("1,1,1,1,1,1,1,1,1,1", "Columns")
    # 第十个 rect 的 x1 必须恰好等于 1.0
    assert rects[9].x1 == 1.0, f"第十个 rect 的 x1={rects[9].x1}，不是 1.0"

    # Rows 模式：十等分纵向
    rects = parse_layout("1,1,1,1,1,1,1,1,1,1", "Rows")
    # 第十个 rect 的 y1 必须恰好等于 1.0
    assert rects[9].y1 == 1.0, f"第十个 rect 的 y1={rects[9].y1}，不是 1.0"
