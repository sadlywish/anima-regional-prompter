import torch

from lib_anima_rp.layout import Rect, parse_layout
from lib_anima_rp.masks import MaskCache, build_masks, empty_region_indices


def test_single_region_covers_everything():
    masks = build_masks([Rect(0.0, 0.0, 1.0, 1.0)], (1, 4, 4))
    assert masks.shape == (1, 16)
    assert torch.all(masks == 1.0)


def test_partition_is_exact():
    """每个 token 恰好属于一个区域，Σmask 处处为 1。这是最重要的不变量。"""
    masks = build_masks(parse_layout("1,2,1,1;2,4,6", "Columns"), (1, 16, 16))
    total = masks.sum(dim=0)
    assert torch.allclose(total, torch.ones_like(total))


def test_partition_is_exact_for_awkward_ratios():
    masks = build_masks(parse_layout("1,1,1,1,1,1,1", "Rows"), (1, 15, 15))
    total = masks.sum(dim=0)
    assert torch.allclose(total, torch.ones_like(total))


def test_two_columns_split_left_right():
    masks = build_masks(parse_layout("1,1", "Columns"), (1, 2, 4))
    # (t h w) 展平：第 0 行是 w=0..3，第 1 行是 w=0..3
    assert masks[0].tolist() == [1, 1, 0, 0, 1, 1, 0, 0]
    assert masks[1].tolist() == [0, 0, 1, 1, 0, 0, 1, 1]


def test_two_rows_split_top_bottom():
    masks = build_masks(parse_layout("1,1", "Rows"), (1, 4, 2))
    assert masks[0].tolist() == [1, 1, 1, 1, 0, 0, 0, 0]
    assert masks[1].tolist() == [0, 0, 0, 0, 1, 1, 1, 1]


def test_flatten_order_is_t_h_w():
    masks = build_masks(parse_layout("1,1", "Columns"), (2, 2, 2))
    assert masks.shape == (2, 8)
    # 每个时间片内部都是 [左, 右, 左, 右]
    assert masks[0].tolist() == [1, 0, 1, 0, 1, 0, 1, 0]


def test_region_token_counts_match_ratio():
    masks = build_masks(parse_layout("1,3", "Columns"), (1, 8, 8))
    assert masks[0].sum().item() == 2 * 8
    assert masks[1].sum().item() == 6 * 8


def test_empty_region_is_reported_not_raised():
    # 4 个区域挤在宽度 2 的网格上，必然有区域分不到 token
    masks = build_masks(parse_layout("1,1,1,1", "Columns"), (1, 4, 2))
    empties = empty_region_indices(masks)
    assert len(empties) > 0
    total = masks.sum(dim=0)
    assert torch.allclose(total, torch.ones_like(total))


def test_dtype_and_device_respected():
    masks = build_masks([Rect(0.0, 0.0, 1.0, 1.0)], (1, 2, 2), dtype=torch.float16)
    assert masks.dtype == torch.float16


def test_center_sampling_locks_w_axis_boundary():
    """token 归属判据是"中心点"而非"左上角"，用能锁住这条规则的边界来测。

    `1,3` 配 W=5：分区边界在 x=0.25，0.25×5=1.25 不是整数，边界不落在网格线
    上——中心采样 x=0.1,0.3,0.5,0.7,0.9 与角点采样 x=0.0,0.2,0.4,0.6,0.8 在
    这个边界下分组结果不同，能区分两种采样规则（不像 `1,1` 配 W=4 那种边界
    恰好落在网格线上的组合，两种规则结果相同，锁不住任何东西）。
    中心采样下只有 x=0.1 这一列落在 region0，region0 应恰好拿到 1 列。
    """
    masks = build_masks(parse_layout("1,3", "Columns"), (1, 3, 5))
    assert masks[0].sum().item() == 1 * 3
    assert masks[1].sum().item() == 4 * 3
    # 逐位锁死：h=3 行，每行都是 [region0, region1, region1, region1, region1]
    assert masks[0].tolist() == [1, 0, 0, 0, 0] * 3
    assert masks[1].tolist() == [0, 1, 1, 1, 1] * 3


def test_center_sampling_locks_h_axis_boundary():
    """纵向对称用例：`1,3` 配 Rows 模式、H=5，边界同样是 y=0.25，
    0.25×5=1.25 非整数，能在 h 轴上区分中心采样与角点采样。
    中心采样 y=0.1,0.3,0.5,0.7,0.9 下只有 y=0.1 这一行落在 region0（上）。
    """
    masks = build_masks(parse_layout("1,3", "Rows"), (1, 5, 3))
    assert masks[0].sum().item() == 1 * 3
    assert masks[1].sum().item() == 4 * 3
    # 逐位锁死：h=5 行，第 0 行全属于 region0，其余 4 行全属于 region1
    assert masks[0].tolist() == [1, 1, 1] + [0, 0, 0] * 4
    assert masks[1].tolist() == [0, 0, 0] + [1, 1, 1] * 4


def test_cache_returns_same_tensor_for_same_key():
    cache = MaskCache()
    rects = parse_layout("1,1", "Columns")
    first = cache.get(rects, (1, 4, 4), torch.device("cpu"), torch.float32)
    second = cache.get(rects, (1, 4, 4), torch.device("cpu"), torch.float32)
    assert first is second


def test_partition_is_exact_for_overlapping_input_rects():
    """手工构造两个重叠矩形（非 parse_layout 输出，parse_layout 天然精确密铺，
    抓不住这个场景）。重叠区若逐区域独立判断会被计入两个区域，Σ=2；
    owner + scatter_ 的写法后写覆盖先写，保证唯一归属，Σ 仍处处为 1。"""
    rects = [Rect(0.0, 0.0, 0.6, 1.0), Rect(0.4, 0.0, 1.0, 1.0)]
    masks = build_masks(rects, (1, 1, 10))
    total = masks.sum(dim=0)
    assert torch.equal(total, torch.ones_like(total))


def test_partition_is_exact_for_gapped_input_rects():
    """手工构造两个之间留有空隙的矩形。留空区若逐区域独立判断谁都不覆盖，
    Σ=0；owner 默认初始化为区域 0，留空区回落到区域 0，Σ 仍处处为 1。"""
    rects = [Rect(0.0, 0.0, 0.3, 1.0), Rect(0.5, 0.0, 1.0, 1.0)]
    masks = build_masks(rects, (1, 1, 10))
    total = masks.sum(dim=0)
    assert torch.equal(total, torch.ones_like(total))


def test_cache_rebuilds_for_different_grid():
    cache = MaskCache()
    rects = parse_layout("1,1", "Columns")
    first = cache.get(rects, (1, 4, 4), torch.device("cpu"), torch.float32)
    second = cache.get(rects, (1, 8, 8), torch.device("cpu"), torch.float32)
    assert first is not second
    assert second.shape == (2, 64)
