# ops/fold_unfold_4d.py
from __future__ import annotations

import torch
import torch.nn.functional as F
from einops import rearrange


@torch.no_grad()
def uv_roll(x: torch.Tensor, du: int, dv: int) -> torch.Tensor:
    """Циклический сдвиг по (U,V). x: [B,C,X,Y,U,V]."""
    return x.roll(shifts=(du, dv), dims=(-2, -1))


def image_to_4d(x: torch.Tensor, s: int = 2, du: int = 0, dv: int = 0) -> torch.Tensor:
    """Изображение -> 4D поле.

    x: [B,C,H,W] в [B,C,X,Y,U,V], где X=H//s, Y=W//s, U,V∈{0..s-1}.

    du,dv — сдвиг решетки фолдинга (аугментация).

    """
    B, C, H, W = x.shape
    assert H % s == 0 and W % s == 0
    X, Y = H // s, W // s
    # разбиваем на s×s блоки
    x_blocks = rearrange(x, 'b c (x s1) (y s2) -> b c x y s1 s2', s1=s, s2=s)
    if (du % s) != 0 or (dv % s) != 0:
        x_blocks = x_blocks.roll(shifts=(du % s, dv % s), dims=(-2, -1))
    return x_blocks.contiguous()


def field4d_to_image(x4: torch.Tensor) -> torch.Tensor:
    """4D поле [B,C,X,Y,U,V] -> изображение [B,C,H,W]."""
    B, C, X, Y, U, V = x4.shape
    s1, s2 = U, V
    x = rearrange(x4, 'b c x y u v -> b c (x u) (y v)')
    return x


def uv_grad_l2(x4: torch.Tensor) -> torch.Tensor:
    """||∇_U x||^2 + ||∇_V x||^2 усреднённые.

    x4: [B,C,X,Y,U,V]

    """
    du = x4 - x4.roll(shifts=1, dims=-2)
    dv = x4 - x4.roll(shifts=1, dims=-1)
    return (du.square().mean() + dv.square().mean())

