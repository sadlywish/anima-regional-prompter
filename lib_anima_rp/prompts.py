"""主提示词与负面提示词的分段。设计文档 §6。

分隔关键字必须大写且是独立词。RP 原版对关键字不做任何边界判定，直接用裸字符串
切分，因此天然支持「房间ADDBASE左边」这种关键字紧贴中文的写法，但代价是
BREAKFAST 之类的英文单词会被误伤（切成空段 + FAST）。本模块在 RP 行为之上只加
一层最小加固：关键字前后不能紧邻 ASCII 字母或数字（不含下划线、不含中文），
既保留了对中文场景的支持，又修掉了误伤英文单词的问题。
"""

import re
from dataclasses import dataclass

KEY_COMMON = "ADDCOMM"
KEY_BASE = "ADDBASE"
REGION_KEYS = ("BREAK", "ADDCOL", "ADDROW")

ALL_KEYS = (KEY_COMMON, KEY_BASE) + REGION_KEYS


def _boundary(inner: str) -> str:
    """给关键字模式包一层边界断言。

    只排除紧邻的 ASCII 字母/数字（`[A-Za-z0-9]`），不用 `\\b`（`\\b` 基于 `\\w`
    判定，会把中文汉字也当成单词字符，导致紧贴中文的关键字识别不到）。下划线、
    中文、标点等一律放行。
    """
    return r"(?<![A-Za-z0-9])(?:" + inner + r")(?![A-Za-z0-9])"


HAS_KEYWORD_RE = re.compile(_boundary("|".join(ALL_KEYS)))
_REGION_SPLIT_RE = re.compile(_boundary("|".join(REGION_KEYS)))
_COMMON_SPLIT_RE = re.compile(_boundary(KEY_COMMON))
_BASE_SPLIT_RE = re.compile(_boundary(KEY_BASE))


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
        rest = re.sub(_boundary("|".join(leftover_keys)), "BREAK", rest)

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
