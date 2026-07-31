"""ANIMA Regional Prompter —— Forge Neo 扩展入口。

参数顺序与 sd-webui-regional-prompter 的 scripts/rp.py:344 完全一致，
迁移时只需把 alwayson_scripts 的 key 改成 "ANIMA Regional Prompter"。
"""

import logging
import math

import gradio as gr
import torch

# Forge 在 modules/scripts.py:505 就把每个扩展的根目录加进了 sys.path 并保留到
# 会话结束，所以 lib_anima_rp 可以直接 import，无需自己动 sys.path。
# 内置扩展 sd_forge_radial 用的也是这个方式。
from lib_anima_rp.attention import (
    GRID_OPTION_KEY,
    TRANSFORMER_OPTION_KEY,
    RPState,
    patch_anima_attention,
)
from lib_anima_rp.layout import COLUMNS, ROWS, parse_layout
from lib_anima_rp.masks import MaskCache, empty_region_indices
from lib_anima_rp.prompts import strip_negative_keywords
from lib_anima_rp.state import build_config, normalize_args
from modules import scripts
from modules.ui_components import InputAccordion

logger = logging.getLogger("anima-rp")

MASK_CACHE = MaskCache()

PREVIEW_COLORS = (
    "#6ea8fe", "#f7a072", "#8fd694", "#d4a5e8", "#f6c453",
    "#7fd4d4", "#e88a9a", "#a9b5c9", "#c3e88d", "#e0a3c4",
)


def render_preview(ratios, mmode, flipper):
    """把当前分区渲染成一张 CSS grid 预览图，消除方向语义的记忆负担。"""
    try:
        rects = parse_layout(ratios, mmode, flipper)
    except ValueError as exc:
        return f'<div style="color:#e88a9a;padding:8px">{exc}</div>'

    cells = []
    for index, rect in enumerate(rects):
        color = PREVIEW_COLORS[index % len(PREVIEW_COLORS)]
        cells.append(
            f'<div style="position:absolute;'
            f"left:{rect.x0 * 100:.4f}%;top:{rect.y0 * 100:.4f}%;"
            f"width:{(rect.x1 - rect.x0) * 100:.4f}%;height:{(rect.y1 - rect.y0) * 100:.4f}%;"
            f"background:{color};border:1px solid rgba(0,0,0,.35);box-sizing:border-box;"
            f'display:flex;align-items:center;justify-content:center;'
            f'font:600 13px/1 sans-serif;color:#1a1a1a">{index}</div>'
        )
    return (
        '<div style="position:relative;width:100%;max-width:320px;aspect-ratio:1;'
        'margin:4px 0;border-radius:4px;overflow:hidden">' + "".join(cells) + "</div>"
    )


def is_anima(p) -> bool:
    try:
        model = p.sd_model.forge_objects.unet.model.diffusion_model
    except AttributeError:
        return False
    return model.__class__.__name__ == "Anima"


class AnimaRegionalPrompter(scripts.Script):
    sorting_priority = 20

    def __init__(self):
        super().__init__()
        self.config = None
        self.state = None
        self.warned_non_anima = False

    def title(self):
        return "ANIMA Regional Prompter"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with InputAccordion(False, label=self.title()) as active:
            with gr.Row():
                mmode = gr.Radio(
                    label="Main Splitting", choices=[COLUMNS, ROWS],
                    value=COLUMNS, type="value",
                )
                ratios = gr.Textbox(label="Divide Ratio", lines=1, value="1,1")
            preview = gr.HTML(value=render_preview("1,1", COLUMNS, False))
            with gr.Row():
                baseratios = gr.Textbox(label="Base Ratio", lines=1, value="0.2")
                flipper = gr.Checkbox(value=False, label='flip "," and ";"')
            with gr.Row():
                usebase = gr.Checkbox(value=False, label="Use base prompt")
                usecom = gr.Checkbox(value=False, label="Use common prompt")
                debug = gr.Checkbox(value=False, label="debug")

        # 仅为 API 兼容而存在的占位参数，取值固定为 RP 默认值
        rp_selected_tab = gr.State("Matrix")
        xmode = gr.State("Mask")
        pmode = gr.State("Prompt")
        usencom = gr.State(False)
        calcmode = gr.State("Attention")
        nchangeand = gr.State(False)
        lnter = gr.State("0")
        lnur = gr.State("0")
        threshold = gr.State("0.4")
        polymask = gr.State(None)
        lstop = gr.State("0")
        lstop_hr = gr.State("0")

        for control in (ratios, mmode, flipper):
            control.change(
                fn=render_preview, inputs=[ratios, mmode, flipper], outputs=[preview]
            )

        self.infotext_fields = [
            (active, "RP Active"),
            (mmode, "RP Main splitting"),
            (ratios, "RP Divide ratio"),
            (baseratios, "RP Base ratio"),
            (usebase, "RP Use base"),
            (usecom, "RP Use common"),
            (flipper, "RP Flip"),
        ]
        self.paste_field_names = [name for _, name in self.infotext_fields]

        # 顺序必须与 RP scripts/rp.py:344 一致
        return [
            active, debug, rp_selected_tab, mmode, xmode, pmode, ratios, baseratios,
            usebase, usecom, usencom, calcmode, nchangeand, lnter, lnur, threshold,
            polymask, lstop, lstop_hr, flipper,
        ]

    def process(self, p, *args):
        self.config = None
        self.state = None

        params = normalize_args(args)
        prompt = p.prompt[0] if isinstance(p.prompt, list) else p.prompt

        try:
            config = build_config(params, prompt)
        except ValueError as exc:
            # build_config() 用「响亮报错」而非静默补齐来防止分区悄悄失效，
            # 但 Forge 的 scripts.py process() 会吞掉脚本抛出的异常（只在控制台
            # 打印 traceback，生成照常进行）：调用方（含 API）拿到的仍是一张
            # 没有分区的图，且从 HTTP 响应完全看不出发生过什么。这里主动把
            # 错误落进 extra_generation_params，让"生成继续、但产物留痕"，
            # 而不是让 traceback 消失在没人看的控制台里；同时不写 RP Active，
            # 因为分区并未生效，写了会误导事后看图的人。
            logger.warning(f"ANIMA Regional Prompter 配置错误，本次生成不分区：{exc}")
            p.extra_generation_params["RP Error"] = str(exc)
            return
        if config is None:
            return

        if not is_anima(p):
            if not self.warned_non_anima:
                logger.warning(
                    "当前模型不是 ANIMA，ANIMA Regional Prompter 不生效。"
                )
                self.warned_non_anima = True
            return

        patch_anima_attention()

        negative = p.negative_prompt[0] if isinstance(p.negative_prompt, list) else p.negative_prompt
        cleaned_negative, changed = strip_negative_keywords(negative)
        if changed:
            logger.warning("负面提示词不分区，已剥离其中的分隔关键字。")
            if isinstance(p.negative_prompt, list):
                p.negative_prompt = [cleaned_negative] * len(p.negative_prompt)
            else:
                p.negative_prompt = cleaned_negative

        texts = list(config.region_prompts)
        if config.base_prompt:
            texts.append(config.base_prompt)

        # 只在 process() 编码一次，hires 二阶段复用同一批 cond，
        # 故设计文档 §5.3 要求的缓存由「编码时机」本身满足，无需额外缓存层。
        conds = p.sd_model.get_learned_conditioning(texts)
        stacked = _pad_and_stack(conds)

        region_count = len(config.region_prompts)
        base_context = stacked[region_count:] if config.base_prompt else None

        self.config = config
        self.state = RPState(
            masks=torch.zeros(region_count, 1),   # 采样期按实际网格重建
            region_contexts=stacked[:region_count],
            base_context=base_context,
            base_weights=torch.tensor(config.base_weights),
        )

        p.extra_generation_params.update({
            "RP Active": True,
            "RP Divide ratio": params["ratios"],
            "RP Main splitting": params["mmode"],
            "RP Base ratio": params["baseratios"],
            "RP Use base": bool(params["usebase"]),
            "RP Use common": bool(params["usecom"]),
            "RP Flip": bool(params["flipper"]),
        })
        if config.degradations:
            joined = ", ".join(config.degradations)
            logger.warning(f"以下参数不受支持，已降级：{joined}")
            p.extra_generation_params["RP Unsupported"] = joined

    def process_before_every_sampling(self, p, *args, **kwargs):
        if self.config is None or self.state is None:
            return

        unet = p.sd_model.forge_objects.unet.clone()
        diffusion_model = unet.model.diffusion_model
        patch_spatial = diffusion_model.patch_spatial
        patch_temporal = diffusion_model.patch_temporal

        config = self.config
        state = self.state
        previous_wrapper = unet.model_options.get("model_function_wrapper")

        def wrapper(model_function, wrapper_kwargs):
            latent = wrapper_kwargs["input"]
            grid = (
                math.ceil(latent.shape[2] / patch_temporal),
                math.ceil(latent.shape[3] / patch_spatial),
                math.ceil(latent.shape[4] / patch_spatial),
            )
            options = wrapper_kwargs["c"]["transformer_options"]
            options[GRID_OPTION_KEY] = grid
            state.masks = MASK_CACHE.get(
                config.rects, grid, latent.device, latent.dtype
            )
            if config.debug:
                empties = empty_region_indices(state.masks)
                if empties:
                    logger.warning(f"以下区域在 {grid} 网格上分不到 token：{empties}")
            if previous_wrapper is not None:
                return previous_wrapper(model_function, wrapper_kwargs)
            return model_function(
                wrapper_kwargs["input"], wrapper_kwargs["timestep"], **wrapper_kwargs["c"]
            )

        unet.set_model_unet_function_wrapper(wrapper)
        unet.set_transformer_option(TRANSFORMER_OPTION_KEY, state)
        p.sd_model.forge_objects.unet = unet


def _pad_and_stack(conds):
    """把长度不一的 cond 对齐后堆成 (N, L, D)。

    补齐方式是重复最后一个向量，与 Forge 的 prompt_parser.stack_conds 一致。
    """
    flattened = [c[0] if c.dim() == 3 else c for c in conds]
    longest = max(c.shape[0] for c in flattened)
    padded = []
    for cond in flattened:
        if cond.shape[0] < longest:
            tail = cond[-1:].repeat(longest - cond.shape[0], 1)
            cond = torch.cat([cond, tail], dim=0)
        padded.append(cond)
    return torch.stack(padded)
