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


def test_flipper_changes_region_count_and_base_weights_must_match():
    """flipper 会交换 Divide ratio 里 ','/';' 的角色，从而改变带/单元结构，
    连区域总数都可能变化。ratios="1,2;3,4" 在 Columns 模式下 flipper=False
    解析出 2 个区域，flipper=True 解析出 3 个区域。

    这条测试守住的契约是：build_config 必须把同一个 flipper 值同时传给
    parse_layout 和 parse_base_weights。如果漏传给 parse_base_weights，
    它会按未翻转的旧结构（2 个区域）算权重，而 rects 是按翻转后的新结构
    （3 个区域）算的，两者长度就会不一致——下游按索引把权重和区域掩码
    配对时会直接出错或静默算错。
    """
    args = list(RP_DEFAULT_ARGS)
    args[6] = "1,2;3,4"  # ratios
    args[8] = True  # usebase
    args[19] = True  # flipper
    config = build_config(normalize_args(args), "room ADDBASE a BREAK b BREAK c")
    assert len(config.rects) == 3
    assert len(config.base_weights) == len(config.rects)
