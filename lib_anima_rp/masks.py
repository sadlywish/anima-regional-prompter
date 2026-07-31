"""区域矩形 → token 掩码张量。设计文档 §8。

token 展平顺序恒为 (t h w)，与 backend/nn/anima.py 的
rearrange(x, "b t h w d -> b (t h w) d") 一致。
"""

from collections.abc import Sequence

import torch

from lib_anima_rp.layout import Rect


def build_masks(
    rects: Sequence[Rect],
    grid: tuple[int, int, int],
    device=None,
    dtype=None,
) -> torch.Tensor:
    """构建形状 (N, T*H*W) 的二值掩码。

    判定规则是 token 中心点落在哪个矩形，因此每个 token 恰好属于一个区域，
    Σmask 处处为 1。若不满足，区域边界会出现亮度漂移。
    """
    if dtype is None:
        dtype = torch.float32
    t_count, h_count, w_count = grid

    # token 中心的归一化坐标
    ys = (torch.arange(h_count, dtype=torch.float32) + 0.5) / h_count
    xs = (torch.arange(w_count, dtype=torch.float32) + 0.5) / w_count
    grid_y = ys.view(h_count, 1).expand(h_count, w_count).reshape(-1)
    grid_x = xs.view(1, w_count).expand(h_count, w_count).reshape(-1)

    # 每个空间位置归属的区域下标；后写入的覆盖先写入的，保证唯一归属。
    # 初始化为 0 而非其他值：未被任何矩形覆盖的位置（输入矩形留空时）
    # 默认回落到区域 0，而不是产生越界下标或无归属状态。
    owner = torch.zeros(h_count * w_count, dtype=torch.long)
    for index, rect in enumerate(rects):
        inside = (
            (grid_x >= rect.x0)
            & (grid_x < rect.x1)
            & (grid_y >= rect.y0)
            & (grid_y < rect.y1)
        )
        owner[inside] = index

    plane = torch.zeros(len(rects), h_count * w_count, dtype=dtype)
    plane.scatter_(0, owner.unsqueeze(0), 1.0)

    assert torch.equal(plane.sum(dim=0), torch.ones(h_count * w_count, dtype=dtype)), (
        "掩码不满足 Σmask≡1，存在 token 被重复计数或未被任何区域覆盖"
    )

    masks = plane.repeat(1, t_count)
    if device is not None:
        masks = masks.to(device)
    return masks


def empty_region_indices(masks: torch.Tensor) -> tuple[int, ...]:
    """返回一个 token 都没分到的区域下标。分辨率太低或区域太窄时会发生。"""
    sums = masks.sum(dim=1)
    return tuple(int(i) for i in torch.nonzero(sums == 0).flatten().tolist())


class MaskCache:
    """按 (区域矩形, 网格, device, dtype) 缓存掩码。

    hires 二阶段分辨率变化时 key 改变，自动重建。
    """

    def __init__(self):
        self._cache = {}

    @staticmethod
    def _key(rects, grid, device, dtype):
        rect_key = tuple((r.x0, r.y0, r.x1, r.y1) for r in rects)
        return (rect_key, tuple(grid), str(device), str(dtype))

    def get(self, rects, grid, device, dtype) -> torch.Tensor:
        key = self._key(rects, grid, device, dtype)
        cached = self._cache.get(key)
        if cached is None:
            cached = build_masks(rects, grid, device=device, dtype=dtype)
            self._cache[key] = cached
        return cached

    def clear(self):
        self._cache.clear()
