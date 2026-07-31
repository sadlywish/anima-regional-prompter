"""主提示词与负面提示词的分段。设计文档 §6。

分隔关键字必须大写，且与 RP 完全一致——不做任何边界判定，直接按子串匹配/切分。
这意味着关键字紧贴中文、下划线、标点、英文字母或数字时都能被正常识别为分隔符，
天然支持「房间ADDBASE左边」这种中英混排写法，也支持「BREAK1boy」这种紧贴
Danbooru 数字标签的常见写法。

代价是 BREAKFAST 之类的英文单词会被误伤：BREAK 会被当成分隔符切出一个空段，
从而触发 `split_prompt` 的空段报错。这是有意接受的取舍——响亮的报错远好过
「关键字紧贴数字标签导致 `has_any_keyword` 判定为假、整个扩展静默不生效」这种
用户毫无察觉的失败模式。

五个关键字（ADDCOMM/ADDBASE/ADDCOL/ADDROW/BREAK）互不为对方前缀，因此交替顺序
不影响匹配结果；将来新增关键字（例如 RP 的 ADDP）时需要重新核对这一前提。
"""

import re
from dataclasses import dataclass

KEY_COMMON = "ADDCOMM"
KEY_BASE = "ADDBASE"
REGION_KEYS = ("BREAK", "ADDCOL", "ADDROW")

ALL_KEYS = (KEY_COMMON, KEY_BASE) + REGION_KEYS

HAS_KEYWORD_RE = re.compile("|".join(ALL_KEYS))
_REGION_SPLIT_RE = re.compile("|".join(REGION_KEYS))
_COMMON_SPLIT_RE = re.compile(KEY_COMMON)
_BASE_SPLIT_RE = re.compile(KEY_BASE)


@dataclass(frozen=True)
class PromptSegments:
    common: str
    base: str
    regions: tuple[str, ...]


def has_any_keyword(prompt: str) -> bool:
    """提示词是否含任何分隔关键字。不含时整个扩展应静默不生效。"""
    return HAS_KEYWORD_RE.search(prompt) is not None


def _tidy(text: str) -> str:
    return text.strip().strip(",").strip()


def split_prompt(prompt: str, usecom: bool, usebase: bool) -> PromptSegments:
    """按 ADDCOMM / ADDBASE / BREAK 切分主提示词。

    usecom / usebase 关闭时，对应关键字退化为普通的区域分隔符（与 RP 一致）。
    """
    rest = prompt
    common = ""
    base = ""

    if usecom:
        parts = _COMMON_SPLIT_RE.split(rest, maxsplit=1)
        if len(parts) == 2:
            common, rest = _tidy(parts[0]), parts[1]

    if usebase:
        parts = _BASE_SPLIT_RE.split(rest, maxsplit=1)
        if len(parts) == 2:
            base, rest = _tidy(parts[0]), parts[1]

    # 未启用的关键字降级为区域分隔符
    leftover_keys = []
    if not usecom:
        leftover_keys.append(KEY_COMMON)
    if not usebase:
        leftover_keys.append(KEY_BASE)
    if leftover_keys:
        rest = re.sub("|".join(leftover_keys), "BREAK", rest)

    raw_regions = _REGION_SPLIT_RE.split(rest)
    regions = []
    for index, chunk in enumerate(raw_regions):
        tidied = _tidy(chunk)
        if not tidied:
            raise ValueError(
                f"提示词的第 {index + 1} 段为空。检查是否有连续的 BREAK 或首尾多余的分隔符。"
            )
        regions.append(tidied)

    return PromptSegments(common=common, base=base, regions=tuple(regions))


def strip_negative_keywords(negative: str) -> tuple[str, bool]:
    """剥掉负面提示词里的分隔关键字并以逗号合并。

    本扩展的负面提示词不分区。返回 (清理后的文本, 是否发生过剥离)。
    """
    if not HAS_KEYWORD_RE.search(negative):
        return negative, False
    parts = [_tidy(part) for part in HAS_KEYWORD_RE.split(negative)]
    return ", ".join(part for part in parts if part), True


def compose_region_prompts(segments: PromptSegments) -> tuple[str, ...]:
    """把 common 前缀拼到每个区域提示词前面。"""
    if not segments.common:
        return segments.regions
    return tuple(f"{segments.common}, {region}" for region in segments.regions)


def compose_base_prompt(segments: PromptSegments) -> str:
    """把 common 前缀拼到 base 提示词前面。base 为空时返回空串。"""
    if not segments.base:
        return ""
    if not segments.common:
        return segments.base
    return f"{segments.common}, {segments.base}"
