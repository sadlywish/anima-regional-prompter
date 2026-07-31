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


def test_breakfast_is_split_and_raises_on_empty_segment():
    # 不做边界判定，与 RP 一致：BREAKFAST 里的 BREAK 会被当成分隔符切出空段，
    # 触发「第 1 段为空」报错。这是有意接受的代价——响亮报错好过静默不生效。
    with pytest.raises(ValueError, match="第 1 段为空"):
        split_prompt("BREAKFAST table", usecom=False, usebase=False)


def test_lowercase_break_is_not_a_separator():
    # 仍然区分大小写：只匹配大写关键字，与 RP 一致
    seg = split_prompt("a break b", usecom=False, usebase=False)
    assert seg.regions == ("a break b",)


def test_has_any_keyword():
    assert has_any_keyword("a BREAK b")
    assert has_any_keyword("a ADDBASE b")
    assert not has_any_keyword("a plain prompt")
    assert has_any_keyword("BREAKFAST")


def test_strip_negative_keywords():
    cleaned, changed = strip_negative_keywords("worst quality BREAK bad hands")
    assert cleaned == "worst quality, bad hands"
    assert changed is True


def test_strip_negative_keywords_noop():
    cleaned, changed = strip_negative_keywords("worst quality")
    assert cleaned == "worst quality"
    assert changed is False


# --- 词边界行为探测：与 RP 完全一致，不做任何边界判定，关键字只要作为子串
# 出现就会被切分——紧贴中文、下划线、标点、英文字母、数字统统能识别。
# 代价是 BREAKFAST 这类英文单词会被误伤，切出空段并触发报错（见上面
# test_breakfast_is_split_and_raises_on_empty_segment）。这是有意的取舍：
# 响亮的报错好过 ADDBASE 紧贴 Danbooru 数字标签（如 BREAK1boy）时的静默失效。
# 这里把实测行为钉住，防止回归。


def test_keyword_with_spaces_on_both_sides_is_separator():
    # 正常情况：关键字两侧有空格
    seg = split_prompt("room ADDBASE 左边", usecom=False, usebase=True)
    assert seg.base == "room"
    assert seg.regions == ("左边",)


def test_keyword_glued_to_chinese_is_recognized():
    # 关键字紧贴中文：房间ADDBASE左边 —— 能正常识别
    seg = split_prompt("房间ADDBASE左边", usecom=False, usebase=True)
    assert seg.base == "房间"
    assert seg.regions == ("左边",)


def test_keyword_glued_to_ascii_letters_is_recognized():
    # 关键字紧贴英文字母：roomADDBASEgirl —— 不做边界判定，同样能识别
    seg = split_prompt("roomADDBASEgirl", usecom=False, usebase=True)
    assert seg.base == "room"
    assert seg.regions == ("girl",)


def test_keyword_glued_to_digit_is_recognized():
    # 关键字紧贴数字：1girlBREAK1boy —— Danbooru 风格标签常以数字开头，
    # 这正是本轮改动要保证能正常切分的场景
    seg = split_prompt("1girlBREAK1boy", usecom=False, usebase=False)
    assert seg.regions == ("1girl", "1boy")


def test_keyword_glued_to_underscore_is_recognized():
    seg = split_prompt("room_ADDBASE_girl", usecom=False, usebase=True)
    assert seg.base == "room_"
    assert seg.regions == ("_girl",)


def test_keyword_glued_to_punctuation_is_recognized():
    seg = split_prompt("room, ADDBASE, girl", usecom=False, usebase=True)
    assert seg.base == "room"
    assert seg.regions == ("girl",)


def test_keyword_glued_to_comma_no_space_is_recognized():
    seg = split_prompt("room,ADDBASE,girl", usecom=False, usebase=True)
    assert seg.base == "room"
    assert seg.regions == ("girl",)


def test_mixed_cjk_english_prompt_with_glued_digit_tags():
    # 中英混排端到端用例：ADDBASE/BREAK 紧贴数字开头的标签（1girl/1boy），
    # 不做边界判定后两处关键字都能正常识别切分。
    seg = split_prompt(
        "温暖的房间, 柔和光线ADDBASE1girl, red dressBREAK1boy, blue suit",
        usecom=False,
        usebase=True,
    )
    assert seg.base == "温暖的房间, 柔和光线"
    assert seg.regions == ("1girl, red dress", "1boy, blue suit")


def test_mixed_cjk_english_prompt_with_common_and_base():
    # 中英混排端到端用例：qualityADDCOMM 紧贴英文字母、房间ADDBASE左边人物
    # 紧贴中文、人物BREAK右边人物紧贴中文——三处现在都能正常识别切分。
    seg = split_prompt(
        "masterpiece, best qualityADDCOMM温暖的房间ADDBASE左边人物BREAK右边人物",
        usecom=True,
        usebase=True,
    )
    assert seg.common == "masterpiece, best quality"
    assert seg.base == "温暖的房间"
    assert seg.regions == ("左边人物", "右边人物")


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
