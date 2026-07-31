"""ANIMA Regional Prompter 端到端验证。

把「分区有没有生效」从肉眼判断变成数值判据：用强色彩提示词生成，
再测量各区域的像素统计量。核心是**对照组**——同一段提示词在扩展关闭时
再跑一次，若两次的区域色彩分布没有显著差异，说明分区并未真正起作用，
只是模型自己碰巧那么画。

用法：
    python verify_e2e.py [--url http://127.0.0.1:7860] [--model anima_baseV10]
"""

import argparse
import base64
import io
import json
import sys
import urllib.error
import urllib.request

from PIL import Image

SCRIPT_KEY = "ANIMA Regional Prompter"

# RP 的 20 个位置参数，顺序同 sd-webui-regional-prompter 的 scripts/rp.py:344
RP_DEFAULTS = [
    False, False, "Matrix", "Columns", "Mask", "Prompt", "1,1", "0.2",
    False, False, False, "Attention", False, "0", "0", "0.4", None,
    "0", "0", False,
]


def rp_args(**over):
    """按名字覆盖 RP 参数，返回 20 元列表。"""
    names = [
        "active", "debug", "rp_selected_tab", "mmode", "xmode", "pmode",
        "ratios", "baseratios", "usebase", "usecom", "usencom", "calcmode",
        "nchangeand", "lnter", "lnur", "threshold", "polymask", "lstop",
        "lstop_hr", "flipper",
    ]
    args = list(RP_DEFAULTS)
    for key, value in over.items():
        args[names.index(key)] = value
    return args


def post(url, path, payload, timeout=600):
    req = urllib.request.Request(
        url + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def get(url, path, timeout=30):
    with urllib.request.urlopen(url + path, timeout=timeout) as resp:
        return json.load(resp)


def txt2img(url, prompt, args=None, **over):
    payload = {
        "prompt": prompt,
        "negative_prompt": "worst quality, blurry",
        "width": 1024, "height": 1024, "steps": 16, "cfg_scale": 4.0,
        "sampler_name": "Euler", "seed": 12345, "batch_size": 1,
    }
    payload.update(over)
    if args is not None:
        payload["alwayson_scripts"] = {SCRIPT_KEY: {"args": args}}
    result = post(url, "/sdapi/v1/txt2img", payload)
    image = Image.open(io.BytesIO(base64.b64decode(result["images"][0]))).convert("RGB")
    info = json.loads(result["info"])
    return image, info


def half_stats(image, axis="x"):
    """返回两半区域的 (平均R - 平均B)。正值偏红，负值偏蓝。"""
    w, h = image.size
    if axis == "x":
        boxes = [(0, 0, w // 2, h), (w // 2, 0, w, h)]
    else:
        boxes = [(0, 0, w, h // 2), (0, h // 2, w, h)]
    out = []
    for box in boxes:
        crop = image.crop(box)
        pixels = list(crop.getdata())
        n = len(pixels)
        r = sum(p[0] for p in pixels) / n
        b = sum(p[2] for p in pixels) / n
        out.append(r - b)
    return out


def report(name, ok, detail):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    for line in detail.splitlines():
        print(f"       {line}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:7860")
    ap.add_argument("--model", default=None, help="checkpoint 名字片段，默认用当前已加载的")
    ap.add_argument("--outdir", default=".")
    args_cli = ap.parse_args()
    url = args_cli.url.rstrip("/")

    try:
        opts = get(url, "/sdapi/v1/options")
    except (urllib.error.URLError, OSError) as exc:
        print(f"无法连接 Forge API：{exc}")
        return 2

    if args_cli.model:
        models = get(url, "/sdapi/v1/sd-models")
        match = [m for m in models if args_cli.model.lower() in m["title"].lower()]
        if not match:
            print(f"找不到含 {args_cli.model!r} 的 checkpoint")
            return 2
        print(f"切换模型 → {match[0]['title']}")
        post(url, "/sdapi/v1/options", {"sd_model_checkpoint": match[0]["title"]}, timeout=600)
    print(f"当前模型：{get(url, '/sdapi/v1/options')['sd_model_checkpoint']}\n")

    results = []
    red_blue = ("solid red background, red, crimson, monochrome red BREAK "
                "solid blue background, blue, azure, monochrome blue")

    # --- 0. 恒等式检验（唯一不依赖模型行为的正确性判据） ---
    #
    # 所有区域共用同一段提示词时，Σ mask_i · attn(ctx) 必须精确退化成 attn(ctx)，
    # 因为 Σ mask_i ≡ 1 且 output_proj 是线性的。所以输出应当与「不启用扩展」
    # 逐像素完全一致。这一条覆盖掩码构建、context 索引、注意力累加、投影外提
    # 整条链路，且完全不受模型对提示词的理解影响——数值不为 0 就是真 bug。
    import numpy as np

    identity = "1girl, red dress, standing in a garden, detailed background"
    plain, _ = txt2img(url, identity)
    plain_arr = np.asarray(plain).astype(float)
    worst = []
    for n in (1, 2, 3, 4):
        img_n, _ = txt2img(url, " BREAK ".join([identity] * n),
                           rp_args(active=True, ratios=",".join(["1"] * n),
                                   mmode="Columns"))
        mae = float(np.abs(np.asarray(img_n).astype(float) - plain_arr).mean())
        worst.append((n, mae))
    results.append(report(
        "恒等式：N 个相同提示词的分区输出 == 不分区输出",
        all(mae < 0.01 for _, mae in worst),
        "  ".join(f"N={n}: MAE={mae:.2f}" for n, mae in worst),
    ))

    # --- 1. 对照组：扩展关闭，同一段提示词（BREAK 会被当普通文本） ---
    base_img, base_info = txt2img(url, red_blue)
    base_img.save(f"{args_cli.outdir}/e2e_control.png")
    base_l, base_r = half_stats(base_img)
    print(f"对照组（未启用）左半 R-B = {base_l:+.1f}，右半 R-B = {base_r:+.1f}\n")

    # --- 2. 分区左右：左红右蓝 ---
    img, info = txt2img(url, red_blue, rp_args(active=True, ratios="1,1", mmode="Columns"))
    img.save(f"{args_cli.outdir}/e2e_columns.png")
    left, right = half_stats(img)
    sep = left - right
    base_sep = base_l - base_r
    results.append(report(
        "分区生效（Columns 1,1，左红右蓝）",
        left > 0 and right < 0 and sep > max(40.0, base_sep + 30.0),
        f"启用后 左 R-B={left:+.1f}  右 R-B={right:+.1f}  分离度={sep:+.1f}\n"
        f"对照组 分离度={base_sep:+.1f}（启用后应显著更大）",
    ))

    # --- 3. Rows 模式：用颜色对调判定，而不是绝对阈值 ---
    #
    # 纵向的分离强度天然弱于横向（模型对上下结构有很强的构图先验），所以
    # 「分离度 > 某个绝对值」这种判据对纵轴过严。真正与模型无关的判据是：
    # 把两段提示词的颜色对调，分离度必须**翻转符号**——只有分区真的在驱动
    # 版面时才会如此，模型自身的构图偏好不会随提示词顺序改变。
    red_first = ("solid red background, red, crimson, monochrome red BREAK "
                 "solid blue background, blue, azure, monochrome blue")
    blue_first = ("solid blue background, blue, azure, monochrome blue BREAK "
                  "solid red background, red, crimson, monochrome red")
    rows_args = rp_args(active=True, ratios="1,1", mmode="Rows")

    img_a, _ = txt2img(url, red_first, rows_args)
    img_a.save(f"{args_cli.outdir}/e2e_rows_red_top.png")
    top_a, bottom_a = half_stats(img_a, axis="y")
    sep_a = top_a - bottom_a

    img_b, _ = txt2img(url, blue_first, rows_args)
    img_b.save(f"{args_cli.outdir}/e2e_rows_blue_top.png")
    top_b, bottom_b = half_stats(img_b, axis="y")
    sep_b = top_b - bottom_b

    # 同时确认没有轴向对调：Rows 不该产生明显的左右分离
    lr_a = half_stats(img_a, axis="x")
    lateral = abs(lr_a[0] - lr_a[1])

    results.append(report(
        "Rows 模式（颜色对调时分离度翻转符号，且不产生横向分离）",
        sep_a > 15.0 and sep_b < -15.0 and lateral < 20.0,
        f"红上蓝下 分离度={sep_a:+.1f}   蓝上红下 分离度={sep_b:+.1f}（应翻转符号）\n"
        f"横向分离={lateral:.1f}（应接近 0，否则说明轴向对调）",
    ))

    # --- 4. infotext 元数据 ---
    text = info["infotexts"][0]
    want = ["RP Active", "RP Divide ratio", "RP Main splitting"]
    results.append(report(
        "PNG info 元数据可复现",
        all(k in text for k in want),
        "\n".join(line for line in text.splitlines() if "RP " in line) or "(未找到 RP 键)",
    ))

    # --- 5. 降级路径：calcmode=Latent ---
    _, info_deg = txt2img(url, red_blue,
                          rp_args(active=True, ratios="1,1", calcmode="Latent"))
    text_deg = info_deg["infotexts"][0]
    results.append(report(
        "降级路径（calcmode=Latent 应出图并标注）",
        "RP Unsupported" in text_deg and "Latent" in text_deg,
        next((l for l in text_deg.splitlines() if "Unsupported" in l), "(未找到 RP Unsupported)"),
    ))

    # --- 6. 无 BREAK 时静默不生效 ---
    _, info_plain = txt2img(url, "1girl, red dress",
                            rp_args(active=True, ratios="1,1"))
    text_plain = info_plain["infotexts"][0]
    results.append(report(
        "提示词无 BREAK 时静默不生效",
        "RP Active" not in text_plain,
        "infotext 中未出现 RP 键" if "RP Active" not in text_plain
        else "意外出现了 RP 元数据",
    ))

    # --- 7. 区域数不匹配：错误必须留痕在产物里 ---
    #
    # Forge 的 modules/scripts.py:846 会吞掉 process() 抛出的异常并继续生成，
    # 所以我们无法让它中止。退而求其次的正确行为是：不分区、不写 RP Active、
    # 但把错误写进 PNG info 的 RP Error，让用户事后可查。
    _, info_bad = txt2img(url, red_blue, rp_args(active=True, ratios="1,1,1"))
    text_bad = info_bad["infotexts"][0]
    has_error = "RP Error" in text_bad
    no_active = "RP Active" not in text_bad
    detail_line = next((l for l in text_bad.split(", ") if "RP Error" in l), "(未找到 RP Error)")
    results.append(report(
        "区域数不匹配时错误留痕于 PNG info",
        has_error and no_active,
        f"RP Error 存在={has_error}  RP Active 缺席={no_active}\n{detail_line}",
    ))

    print(f"\n{'=' * 52}")
    print(f"通过 {sum(results)}/{len(results)}")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
