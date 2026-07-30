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
