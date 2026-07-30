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
    # 7 个区域（显示字符中的单元数）：带 1 有 4 个，带 2 有 3 个
    # 全局流情况（无 ';'）：支持跨带溢出
    assert_weights(
        parse_base_weights("0.1,0.2,0.3,0.4,0.5", "1,2,1,1;2,4,6", "Columns"),
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.5, 0.5],
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
