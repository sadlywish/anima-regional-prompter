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


def test_keyword_must_be_standalone_word():
    # BREAKFAST 不是分隔符
    seg = split_prompt("BREAKFAST table", usecom=False, usebase=False)
    assert seg.regions == ("BREAKFAST table",)


def test_lowercase_break_is_not_a_separator():
    seg = split_prompt("a break b", usecom=False, usebase=False)
    assert seg.regions == ("a break b",)


def test_has_any_keyword():
    assert has_any_keyword("a BREAK b")
    assert has_any_keyword("a ADDBASE b")
    assert not has_any_keyword("a plain prompt")
    assert not has_any_keyword("BREAKFAST")


def test_strip_negative_keywords():
    cleaned, changed = strip_negative_keywords("worst quality BREAK bad hands")
    assert cleaned == "worst quality, bad hands"
    assert changed is True


def test_strip_negative_keywords_noop():
    cleaned, changed = strip_negative_keywords("worst quality")
    assert cleaned == "worst quality"
    assert changed is False


# --- 词边界行为探测：\b 是基于 \w 判定的，中文汉字在 Python re 中算作 \w，
# 因此关键字紧贴中文/英文/数字时不会被识别为分隔符；紧贴标点（逗号等）时
# 标点不是 \w，边界成立，能被正常识别。这里把实测行为钉住，防止回归。


def test_keyword_with_spaces_on_both_sides_is_separator():
    # 正常情况：关键字两侧有空格
    seg = split_prompt("room ADDBASE 左边", usecom=False, usebase=True)
    assert seg.base == "room"
    assert seg.regions == ("左边",)


def test_keyword_glued_to_chinese_is_not_recognized():
    # 关键字紧贴中文：房间ADDBASE左边 —— 汉字在 Python \w 中，两侧都不构成词边界
    seg = split_prompt("房间ADDBASE左边", usecom=False, usebase=True)
    assert seg.base == ""
    assert seg.regions == ("房间ADDBASE左边",)


def test_keyword_glued_to_ascii_letters_is_not_recognized():
    # 关键字紧贴英文字母：roomADDBASEgirl —— 字母也在 \w 中，同样不构成词边界
    seg = split_prompt("roomADDBASEgirl", usecom=False, usebase=True)
    assert seg.base == ""
    assert seg.regions == ("roomADDBASEgirl",)


def test_keyword_glued_to_punctuation_is_recognized():
    # 关键字紧贴标点（逗号，两侧有空格）：标点不是 \w，词边界成立
    seg = split_prompt("room, ADDBASE, girl", usecom=False, usebase=True)
    assert seg.base == "room"
    assert seg.regions == ("girl",)


def test_keyword_glued_to_comma_no_space_is_recognized():
    # 关键字紧贴标点（逗号，无空格）：room,ADDBASE,girl —— 逗号同样不是 \w
    seg = split_prompt("room,ADDBASE,girl", usecom=False, usebase=True)
    assert seg.base == "room"
    assert seg.regions == ("girl",)


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
