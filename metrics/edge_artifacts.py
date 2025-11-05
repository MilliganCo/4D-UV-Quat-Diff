# metrics/edge_artifacts.py
from __future__ import annotations

import torch


def block_edge_score(imgs: torch.Tensor, s: int = 2) -> torch.Tensor:
    """Оценка артефактов вдоль границ блоков.

    imgs: [B,C,H,W] в диапазоне [-1,1]. Чем больше, тем хуже.

    """
    B,C,H,W = imgs.shape
    # границы между блоками по X и Y
    gx = (imgs[:, :, s::s, :] - imgs[:, :, s-1:-1:s, :]).abs().mean()
    gy = (imgs[:, :, :, s::s] - imgs[:, :, :, s-1:-1:s]).abs().mean()
    return (gx + gy) / 2

