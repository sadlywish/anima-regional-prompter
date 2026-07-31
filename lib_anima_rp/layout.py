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


def _parse_ratio_structure(ratios: str, mmode: str):
    """把 Divide ratio 字符串解析成 (bands, cell_rows) 结构。

    `parse_layout` 与 `parse_base_weights` 共用这一处逻辑（一维/二维分支判断，
    以及数值校验：空值、非数字、非正数一律 `ValueError`）。二者都只需要区域
    结构本身，若各自重新实现一遍分支判断，校验行为迟早会分叉——`parse_layout`
    早年就出现过这种分叉：只有二维分支复用了 `_split_bands`（因而会校验），
    一维分支自己重写了一遍且完全没做数值校验。抽成这一个函数，两处的校验
    行为由构造保证一致。
    """
    text = ratios.strip()
    if not text:
        raise ValueError("Divide ratio 不能为空")

    # 一维分支：字面不含 ';'，与 RP regions.py:281 一致，对两种 mmode 都用字面 ';' 检测
    if _BAND_DELIM not in text:
        cells = _to_positive_floats(text.split(_CELL_DELIM), "带内单元")
        return [1.0], [cells]
    return _split_bands(text, mmode)


def parse_layout(ratios: str, mmode: str, flipper: bool = False) -> list[Rect]:
    """把 Divide ratio 字符串解析成区域矩形列表，band-major 顺序。"""
    if mmode not in VALID_MMODES:
        raise ValueError(f"未知的 mmode：{mmode!r}，只接受 {VALID_MMODES}")

    text = ratios.strip()
    if flipper:
        text = swap_delimiters(text)

    bands, cell_rows = _parse_ratio_structure(text, mmode)

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


def parse_base_weights(
    baseratios: str, ratios: str, mmode: str, flipper: bool = False
) -> list[float]:
    """把 Base ratio 字符串展开成与区域一一对应的权重列表。

    完整复刻 RP regions.py:133-162 的 split_l2(basestruct=...) 分支，对应
    matrixdealer 里的调用（regions.py:379）：

        bratios2 = split_l2(bratios, ";", ",", fmap=ffloatd(0),
                             basestruct=lbreaks, indflip=indflip)

    其中 lbreaks 与区域的带/单元结构（aratios2）同形状、值全为 0——
    basestruct 只有其**形状**参与广播，值本身不参与计算。

    广播规则（逐段消费，段以 ';'/',' 中充当"行分隔符"的那个切出）：
    - 段内取值数 < 当前带的单元数 → 用该段自己的最后一个值重复填满该带
    - 段内取值数 == 当前带的单元数 → 原样对应
    - 段内取值数 > 当前带的单元数 → 截断到该带长度，**剩余的值溢出到下一带**
      继续消费（同一段可以连续填满多个带）
    - 带索引越界（所有带都已填满）→ 剩余的值全部丢弃
    - 所有输入段处理完后仍有带未填充 → 用最后一次出现的取值重复填满剩余带

    **`flipper` 影响区域结构，但不影响 baseratios 字符串本身怎么切**（修复轮 4，
    纠正此前"区域数量与 flipper 无关"的错误结论）。RP `rp.py:386` 的
    `if flipper: aratios = changecs(aratios)` 发生在 matrixdealer **之前**，
    传进 matrixdealer 的 aratios 已经是交换过 ';'/',' 的版本，`basestruct` 的形状
    （`lbreaks`）就是从这个翻转后的结构推出来的——所以交换分隔符会重新划分带/单元，
    连区域**总数**都可能改变。而 `bratios`（baseratios）字符串自身从未被
    `changecs` 处理过，`split_l2(bratios, ...)` 用的还是原始的 ';'/','。
    因此这里对 `ratios` 应用 flipper 来算区域结构（与 `parse_layout` 保持一致），
    但 `baseratios` 字符串本身照旧不翻转。

    另外，**mmode（Rows/Columns）仍然会影响 baseratios 自身的分隔符角色**，
    这与 flipper 无关：RP 对 bratios 的 split_l2 调用同样传入
    indflip = ("Ver" in mode)，即 mmode 为 Rows 时会交换 ';' 与 ',' 的行/列
    语义——这是 mmode 参数本身的行为，与 aratios 的处理完全对称。
    """
    if mmode not in VALID_MMODES:
        raise ValueError(f"未知的 mmode：{mmode!r}，只接受 {VALID_MMODES}")

    # 1. 计算区域的带/单元结构（形状）。flipper 应用在 ratios 上，与 parse_layout
    #    的处理完全一致（先 strip，flipper 时 swap_delimiters，再解析结构），
    #    因为 RP 的 basestruct 形状本来就是从翻转后的 aratios 推出来的。
    #    与 parse_layout 共用 _parse_ratio_structure，保证对非法 ratios（负数、
    #    非数字、空串、只有分隔符等）的校验行为完全一致。
    text = ratios.strip()
    if flipper:
        text = swap_delimiters(text)
    _, cell_rows = _parse_ratio_structure(text, mmode)
    cell_rows_shape = [len(row) for row in cell_rows]

    # 2. baseratios 自身的行/列分隔符角色由 mmode 决定（对应 RP 的 indflip）
    if mmode == ROWS:
        row_delim, cell_delim = _CELL_DELIM, _BAND_DELIM
    else:
        row_delim, cell_delim = _BAND_DELIM, _CELL_DELIM

    # 3. 完整复刻 split_l2 的 basestruct 分支
    bands: list[list[float]] = []
    band_index = 0
    last_seen = 0.0
    for segment in baseratios.split(row_delim):
        values = [_float_or_zero(part) for part in segment.split(cell_delim)]
        last_seen = values[-1] if values else 0.0
        stop = False
        while not stop:
            if band_index >= len(cell_rows_shape) or (not values and cell_rows_shape):
                stop = True
                continue
            band_len = cell_rows_shape[band_index]
            if band_len >= len(values):
                stop = True
                bands.append(values + [values[-1]] * (band_len - len(values)))
                band_index += 1
            else:
                bands.append(values[:band_len])
                values = values[band_len:]
                band_index += 1

    # 带用完但输入段已耗尽 → 用最后一次出现的取值重复填满剩余带
    while len(bands) < len(cell_rows_shape):
        bands.append([last_seen] * cell_rows_shape[len(bands)])

    weights = [w for band in bands for w in band]
    return [min(1.0, max(0.0, w)) for w in weights]
