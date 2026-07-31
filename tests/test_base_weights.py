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
    # 5 个区域：带 1 有 3 个，带 2 有 2 个（带尺寸值本身不计入单元数）
    assert_weights(
        parse_base_weights("0.1,0.2,0.3;0.4,0.5", "1,2,1,1;2,4,6", "Columns"),
        [0.1, 0.2, 0.3, 0.4, 0.5],
    )


def test_band_underfilled_repeats_own_last_value():
    # 反例（对照 RP split_l2 basestruct 分支实跑得出）：
    # 带 1 需要 3 个单元，段 "0.1,0.2" 只有 2 个值，用段自己的最后值(0.2)补齐；
    # 带 2 需要 2 个单元，段 "0.3,0.4" 恰好对齐。
    assert_weights(
        parse_base_weights("0.1,0.2;0.3,0.4", "1,2,1,1;2,4,6", "Columns"),
        [0.1, 0.2, 0.2, 0.3, 0.4],
    )


def test_band_overfilled_spills_into_next_band():
    # 反例（对照 RP split_l2 basestruct 分支实跑得出）：
    # 段 "0.1,0.2,0.3,0.4" 有 4 个值，带 1 只需要 3 个，截断后剩余的 0.4
    # 溢出到带 2（带 2 需要 2 个单元，0.4 重复补齐）。
    assert_weights(
        parse_base_weights("0.1,0.2,0.3,0.4;0.5", "1,2,1,1;2,4,6", "Columns"),
        [0.1, 0.2, 0.3, 0.4, 0.4],
    )


def test_non_numeric_falls_back_to_zero():
    # RP 用 ffloatd(0)，非数字退化为 0 而不是报错
    assert_weights(parse_base_weights("0.2,x", "1,1", "Columns"), [0.2, 0.0])


def test_empty_string_is_all_zero():
    assert_weights(parse_base_weights("", "1,1", "Columns"), [0.0, 0.0])


def test_flipper_does_not_apply_to_base_ratios():
    # RP rp.py:386 只对 aratios 调 changecs，baseratios 完全不受 flipper 开关影响
    # （本函数干脆不接受 flipper 参数，从签名上就杜绝了这个可能性）。
    #
    # 注意：这不意味着 ';' 与 ',' 在 baseratios 里是同义的。经对照 RP 源码
    # （split_l2 的 basestruct 分支）直接实跑验证，二者结果并不相同：
    # "0.1;0.2" 按 ';' 切成两段，每段单独匹配一个带；该带（"1,1"）只有 2 个
    # 单元，第一段 "0.1" 就已用重复自身最后值的方式填满（[0.1, 0.1]），
    # 带已耗尽，第二段 "0.2" 被丢弃。而 "0.1,0.2" 是同一段里的两个值，
    # 直接一一对应两个单元（[0.1, 0.2]）。
    assert_weights(parse_base_weights("0.1;0.2", "1,1", "Columns"), [0.1, 0.1])
    assert_weights(parse_base_weights("0.1,0.2", "1,1", "Columns"), [0.1, 0.2])


def test_out_of_range_is_clamped():
    assert_weights(parse_base_weights("-1,5", "1,1", "Columns"), [0.0, 1.0])


def test_rows_mode_swaps_base_ratios_delimiter_roles():
    # mmode 会影响 baseratios 自身的分隔符角色（对应 RP 的 indflip = "Ver" in mode，
    # regions.py:379 对 bratios 的 split_l2 调用同样传入 indflip）。这与 flipper
    # 开关无关，是 mmode 参数本身的行为，与 aratios 的处理完全对称。
    # Rows 模式下 ',' 反而充当"行分隔符"：整段 "0.1,0.2,0.3" 被视作 3 个独立的
    # 单值段，第一段就已把唯一的带（3 个单元）填满，其余两段被丢弃。
    assert_weights(
        parse_base_weights("0.1,0.2,0.3", "1,1,1", "Rows"),
        [0.1, 0.1, 0.1],
    )


def test_unknown_mmode_raises():
    with pytest.raises(ValueError):
        parse_base_weights("0.2", "1,1", "Diagonal")


def test_invalid_ratios_raises_for_1d():
    # 一维分支此前完全不校验 ratios（只数逗号个数），负数/非数字会被静默接受。
    # parse_layout 对同样的字符串会抛 ValueError，parse_base_weights 必须一致。
    with pytest.raises(ValueError):
        parse_base_weights("0.2", "1,-5,abc", "Columns")


def test_empty_ratios_raises():
    with pytest.raises(ValueError):
        parse_base_weights("0.2", "", "Columns")


def test_ratios_with_only_delimiter_raises():
    # 只有分隔符（等价于两个空值之间夹着一个逗号）在一维分支下此前也不会报错。
    with pytest.raises(ValueError):
        parse_base_weights("0.2", ",", "Columns")


def test_invalid_ratios_raises_for_2d():
    # 二维分支复用 _split_bands，此前就已经会校验；这里补一个回归测试，
    # 确保重构后二维分支的校验行为不受影响。
    with pytest.raises(ValueError):
        parse_base_weights("0.2", "1,2,1,1;2,-4,6", "Columns")


def test_flipper_changes_region_structure_and_weight_alignment():
    # 修复轮 4：flipper=True 会交换 ratios 里的 ';' 与 ','，彻底改变带/单元结构，
    # 区域**数量**也会随之改变——此前 docstring 里"区域数量与 flipper 无关"的说法
    # 是错的。以下 5 个用例取自团队复审报告，取值已对照 RP 真实 split_l2 逐一验证：
    #   flipper 应用于 ratios（决定 basestruct 的形状），
    #   但 baseratios 字符串本身从不翻转（RP 从未对 bratios 调用 changecs）。
    assert_weights(
        parse_base_weights("0.1,0.2", "1,2;3,4", "Columns", flipper=True),
        [0.1, 0.2, 0.2],
    )
    assert_weights(
        parse_base_weights(
            "0.1,0.2,0.3,0.4,0.5", "1,2,1,1;2,4,6", "Columns", flipper=True
        ),
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.5],
    )
    assert_weights(
        parse_base_weights("0.1,0.2,0.3,0.4", "2,3;1,1,1;4,5", "Columns", flipper=True),
        [0.1, 0.2, 0.3, 0.4, 0.4],
    )
    assert_weights(
        parse_base_weights("0.2", "1,1", "Rows", flipper=True),
        [0.2],
    )
    assert_weights(
        parse_base_weights("0.1,0.2", "1,2,3", "Rows", flipper=True),
        [0.1, 0.1],
    )


def test_flipper_defaults_to_false():
    # 新增的 flipper 形参必须向后兼容：不传时行为与此前完全一致。
    assert_weights(
        parse_base_weights("0.1,0.2,0.3", "1,1,1", "Columns"),
        parse_base_weights("0.1,0.2,0.3", "1,1,1", "Columns", flipper=False),
    )


def test_flipper_weight_count_matches_layout_region_count():
    # 长度契约：无论 flipper 取何值，parse_base_weights 的返回长度必须等于
    # parse_layout 用同样的 (ratios, mmode, flipper) 独立算出的区域数。
    from lib_anima_rp.layout import parse_layout

    combos = [
        ("1,2;3,4", "Columns", True),
        ("1,2,1,1;2,4,6", "Columns", True),
        ("2,3;1,1,1;4,5", "Columns", True),
        ("1,1", "Rows", True),
        ("1,2,3", "Rows", True),
        ("1,2,1,1;2,4,6", "Columns", False),
        ("1,1,1", "Rows", False),
    ]
    for ratios, mmode, flipper in combos:
        region_count = len(parse_layout(ratios, mmode, flipper))
        weight_count = len(parse_base_weights("0.2", ratios, mmode, flipper))
        assert weight_count == region_count, (
            f"{ratios!r}/{mmode}/flip={flipper} 权重数 {weight_count} 与区域数 "
            f"{region_count} 不符"
        )
