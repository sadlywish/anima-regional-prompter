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

    广播规则（RP split_l2 的 basestruct 行为）：
    - 如果 baseratios 含 ';'：各分段对应各带，逐带独立处理
    - 如果 baseratios 不含 ';'：全局值流，从左到右分配（支持溢出）
    - 带内不足 → 重复该带的最后值填满
    - 带已用完 → 剩余值全部丢弃

    **本函数不接受 flipper。** RP 在 rp.py:386 只对 aratios 调用 changecs，
    baseratios 原样使用；而区域**数量**与 flipper 无关（交换分隔符只改变版面朝向）。
    因此这里恒以 flipper=False 解析布局。baseratios 总是用固定的 ';' 和 ','，
    不受 mmode 影响。
    """
    # 获取布局的带/单元结构（固定 flipper=False）
    # 注意：这里按 ratios 的原始字符串来计算单元数，不应用 _split_bands 的逻辑
    text = ratios.strip()
    if mmode == ROWS:
        ratio_band_delim, ratio_cell_delim = _CELL_DELIM, _BAND_DELIM
    else:
        ratio_band_delim, ratio_cell_delim = _BAND_DELIM, _CELL_DELIM

    # 构造每带的 cell 数列表（按字符串的逗号/分号分割数）
    cell_counts = []
    if ratio_band_delim not in text:
        # 一维分支：单带
        cell_counts.append(len(text.split(ratio_cell_delim)))
    else:
        # 二维分支
        for band_text in text.split(ratio_band_delim):
            # 直接按分隔符计算，不处理"第一个值是带宽度"
            cells = band_text.split(ratio_cell_delim)
            cell_counts.append(len(cells))

    # 解析 baseratios：总是用固定的 ';' 和 ',' 作为分隔符，不受 mmode 影响
    # RP 的逻辑（split_l2 with basestruct）：
    # - 如果 baseratios 有 ';'：按 ';' 分隔，每段对应一个带（不支持跨带溢出）
    # - 如果 baseratios 无 ';'：全局值流（支持跨带溢出）
    # - 特例：单带情况下，';' 和 ',' 等价
    raw = baseratios.strip()
    if not raw:
        value_stream = [0.0]
        has_band_sep = False
    else:
        has_band_sep = _BAND_DELIM in raw and len(cell_counts) > 1

    if has_band_sep:
        # 有 ';' 且多带：按 ';' 分隔，每段对应一个带
        sub_streams = []
        for part in raw.split(_BAND_DELIM):
            if not part:
                sub_streams.append([0.0])
            else:
                sub_streams.append([_float_or_zero(x) for x in part.split(_CELL_DELIM)])

        result = []
        for band_idx, cell_count in enumerate(cell_counts):
            if band_idx < len(sub_streams):
                stream = sub_streams[band_idx]
            else:
                stream = [0.0]

            band_values = []
            last_value = 0.0
            for cell_idx in range(cell_count):
                if cell_idx < len(stream):
                    value = stream[cell_idx]
                    band_values.append(value)
                    last_value = value
                else:
                    band_values.append(last_value)

            result.extend(band_values)
    else:
        # 无 ';' 或单带：全局值流
        if not raw:
            value_stream = [0.0]
        else:
            # 单带情况：';' 和 ',' 都是值分隔符
            value_stream = [_float_or_zero(x) for x in raw.replace(_BAND_DELIM, _CELL_DELIM).split(_CELL_DELIM)]
            if not value_stream:
                value_stream = [0.0]

        result = []
        stream_idx = 0
        last_value = 0.0

        for band_idx, cell_count in enumerate(cell_counts):
            band_values = []

            for cell_idx in range(cell_count):
                if stream_idx < len(value_stream):
                    value = value_stream[stream_idx]
                    band_values.append(value)
                    last_value = value
                    stream_idx += 1
                else:
                    band_values.append(last_value)

            result.extend(band_values)

    # Clamp 到 [0, 1]
    return [min(1.0, max(0.0, w)) for w in result]
