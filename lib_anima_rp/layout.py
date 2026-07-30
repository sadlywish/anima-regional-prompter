"""Divide ratio 与 Base ratio 的解析。

语法与 sd-webui-regional-prompter 完全一致，参照其 regions.py:274 makeimgtmp。
详见设计文档 §4。
"""

from dataclasses import dataclass

COLUMNS = "Columns"
ROWS = "Rows"
VALID_MMODES = (COLUMNS, ROWS)

_BAND_DELIM = ";"
_CELL_DELIM = ","


@dataclass(frozen=True)
class Rect:
    """归一化到 [0, 1] 的区域矩形，原点在左上角。"""

    x0: float
    y0: float
    x1: float
    y1: float


def swap_delimiters(s: str) -> str:
    """交换 ',' 与 ';'，等价于 RP 的 changecs()。"""
    return s.replace(",", "\x00").replace(";", ",").replace("\x00", ";")


def _to_positive_floats(parts, context):
    values = []
    for raw in parts:
        text = raw.strip()
        if not text:
            raise ValueError(f"Divide ratio 的 {context} 中存在空值：{parts!r}")
        try:
            value = float(text)
        except ValueError:
            raise ValueError(f"Divide ratio 的 {context} 中 {text!r} 不是有效数字") from None
        if value <= 0:
            raise ValueError(f"Divide ratio 的 {context} 中 {text!r} 必须为正数")
        values.append(value)
    return values


def _spans(sizes):
    """把一组正数尺寸转成归一化的 [start, end] 区间列表。"""
    total = sum(sizes)
    spans = []
    cursor = 0.0
    for size in sizes:
        start = cursor
        cursor += size / total
        spans.append((start, cursor))
    # 消除浮点累积误差，最后一段强制贴边
    spans[-1] = (spans[-1][0], 1.0)
    return spans


def parse_layout(ratios: str, mmode: str, flipper: bool = False) -> list[Rect]:
    """把 Divide ratio 字符串解析成区域矩形列表，band-major 顺序。"""
    if mmode not in VALID_MMODES:
        raise ValueError(f"未知的 mmode：{mmode!r}，只接受 {VALID_MMODES}")

    text = ratios.strip()
    if not text:
        raise ValueError("Divide ratio 不能为空")

    if flipper:
        text = swap_delimiters(text)

    # 一维分支：字面不含 ';'，与 RP regions.py:281 一致，对两种 mmode 都用字面 ';' 检测
    if _BAND_DELIM not in text:
        cells = _to_positive_floats(text.split(_CELL_DELIM), "带内单元")
        bands = [1.0]
        cell_rows = [cells]
    else:
        bands, cell_rows = _split_bands(text, mmode)

    flipped = mmode == ROWS
    band_spans = _spans(bands)
    rects = []
    for band_index, band_span in enumerate(band_spans):
        for cell_span in _spans(cell_rows[band_index]):
            if flipped:
                rects.append(Rect(band_span[0], cell_span[0], band_span[1], cell_span[1]))
            else:
                rects.append(Rect(cell_span[0], band_span[0], cell_span[1], band_span[1]))
    return rects


def _split_bands(text, mmode):
    """二维分支。RP 的 split_l2 在 indflip 时交换两个分隔符，这里照搬。"""
    if mmode == ROWS:
        band_delim, cell_delim = _CELL_DELIM, _BAND_DELIM
    else:
        band_delim, cell_delim = _BAND_DELIM, _CELL_DELIM

    bands = []
    cell_rows = []
    for band_text in text.split(band_delim):
        values = _to_positive_floats(band_text.split(cell_delim), "带")
        bands.append(values[0])
        # RP regions.py:131 —— 只有一个值时该值同时充当带尺寸与唯一单元
        cell_rows.append(values[1:] if len(values) > 1 else values)
    return bands, cell_rows


def _float_or_zero(text):
    """RP 的 ffloatd(0)：非数字退化为 0 而不是报错。"""
    try:
        return float(text.strip())
    except ValueError:
        return 0.0


def parse_base_weights(baseratios: str, ratios: str, mmode: str) -> list[float]:
    """把 Base ratio 字符串展开成与区域一一对应的权重列表。

    广播规则（RP split_l2 的 basestruct 行为）：值不足时重复最后一个，多余的截断。

    **本函数不接受 flipper。** RP 在 rp.py:386 只对 aratios 调用 changecs，
    baseratios 原样使用；而区域**数量**与 flipper 无关（交换分隔符只改变版面朝向）。
    因此这里恒以 flipper=False 解析布局，只为取区域数量。
    """
    region_count = len(parse_layout(ratios, mmode, flipper=False))

    raw = baseratios.strip()
    if not raw:
        values = [0.0]
    else:
        values = [_float_or_zero(part) for part in raw.replace(";", ",").split(",")]
    if not values:
        values = [0.0]

    if len(values) < region_count:
        values = values + [values[-1]] * (region_count - len(values))
    weights = values[:region_count]
    return [min(1.0, max(0.0, w)) for w in weights]
