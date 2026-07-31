"""API 参数归一化、降级判定与配置构建。设计文档 §9。

参数顺序必须与 sd-webui-regional-prompter 的 scripts/rp.py:344 完全一致。
"""

from collections.abc import Sequence
from dataclasses import dataclass

from lib_anima_rp.layout import COLUMNS, ROWS, Rect, parse_base_weights, parse_layout
from lib_anima_rp.prompts import (
    compose_base_prompt,
    compose_region_prompts,
    has_any_keyword,
    split_prompt,
)

RP_ARG_NAMES = (
    "active",
    "debug",
    "rp_selected_tab",
    "mmode",
    "xmode",
    "pmode",
    "ratios",
    "baseratios",
    "usebase",
    "usecom",
    "usencom",
    "calcmode",
    "nchangeand",
    "lnter",
    "lnur",
    "threshold",
    "polymask",
    "lstop",
    "lstop_hr",
    "flipper",
)

RP_ARG_DEFAULTS = (
    False, False, "Matrix", "Columns", "Mask", "Prompt", "1,1", "0.2",
    False, False, False, "Attention", False, "0", "0", "0.4", None,
    "0", "0", False,
)

assert len(RP_ARG_NAMES) == len(RP_ARG_DEFAULTS) == 20


@dataclass(frozen=True)
class RegionalConfig:
    """一次生成所需的、经过校验的运行期配置。"""

    rects: tuple[Rect, ...]
    region_prompts: tuple[str, ...]
    base_prompt: str
    base_weights: tuple[float, ...]
    debug: bool
    degradations: tuple[str, ...]


def normalize_args(args: Sequence) -> dict:
    """把位置参数转成命名 dict，缺失的补 RP 默认值，多余的忽略。"""
    params = dict(zip(RP_ARG_NAMES, RP_ARG_DEFAULTS))
    for name, value in zip(RP_ARG_NAMES, args):
        params[name] = value
    return params


def collect_degradations(params: dict) -> tuple[str, ...]:
    """列出会影响画面结果、但本扩展不支持的参数取值。

    只关心确实会改变渲染结果的参数；其余参数（xmode/pmode/nchangeand/
    lnter/lnur/threshold/polymask/lstop/lstop_hr）本扩展完全不读、不警告。
    降级不抛异常，仍按 Attention/Matrix 继续处理。
    """
    notes = []
    if params["rp_selected_tab"] != "Matrix":
        notes.append(f"tab={params['rp_selected_tab']}")
    if params["mmode"] not in (COLUMNS, ROWS):
        notes.append(f"mmode={params['mmode']}")
    if params["calcmode"] != "Attention":
        notes.append(f"calcmode={params['calcmode']}")
    if params["usencom"]:
        notes.append("usencom=True")
    return tuple(notes)


def build_config(params: dict, prompt: str) -> RegionalConfig | None:
    """构建运行期配置。返回 None 表示本次生成应静默不启用分区。"""
    if not params["active"]:
        return None
    if not has_any_keyword(prompt):
        # 提示词没写任何分隔关键字，静默不生效（RP rp.py:359 的行为）。
        # x/y 图、批量任务里「开关开着但当前提示词没写 BREAK」是常态，
        # 报错会让整批任务失败。
        return None

    degradations = collect_degradations(params)

    # mmode 非法时回退到 Columns，并已在 collect_degradations 里计入降级说明
    mmode = params["mmode"] if params["mmode"] in (COLUMNS, ROWS) else COLUMNS
    ratios = params["ratios"]
    flipper = bool(params["flipper"])

    rects = parse_layout(ratios, mmode, flipper)
    segments = split_prompt(
        prompt, usecom=bool(params["usecom"]), usebase=bool(params["usebase"])
    )
    region_prompts = compose_region_prompts(segments)

    if len(rects) != len(region_prompts):
        raise ValueError(
            f"分区数与提示词段数不匹配：Divide ratio {ratios!r}（{mmode}"
            f"{'，已翻转分隔符' if flipper else ''}）解析出 {len(rects)} 个区域，"
            f"但提示词切出 {len(region_prompts)} 段。"
            f"请增删 BREAK，或调整 Divide ratio。"
        )

    base_prompt = compose_base_prompt(segments)
    if base_prompt:
        # flipper 会改变 parse_layout 解析出的带/单元结构（连区域总数都可能
        # 变化），因此必须把同一个 flipper 值也传给 parse_base_weights，
        # 否则两者算出的区域数会不一致，导致 base_weights 与 rects 长度不等。
        base_weights = tuple(parse_base_weights(params["baseratios"], ratios, mmode, flipper))
    else:
        base_weights = tuple(0.0 for _ in rects)

    return RegionalConfig(
        rects=tuple(rects),
        region_prompts=region_prompts,
        base_prompt=base_prompt,
        base_weights=base_weights,
        debug=bool(params["debug"]),
        degradations=degradations,
    )
