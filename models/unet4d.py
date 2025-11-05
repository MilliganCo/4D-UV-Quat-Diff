# models/unet4d.py
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from ops.conv_uv_quat import QuaternionUVConv


# вспомогательные обертки для Conv2d по XY и GroupNorm по каналам
class ConvXY(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1):
        super().__init__()
        p = k // 2
        self.conv = nn.Conv2d(in_ch, out_ch, k, stride=s, padding=p)
    def forward(self, x):  # x: [B,C,X,Y,U,V]
        B,C,X,Y,U,V = x.shape
        x_ = x.permute(0,4,5,1,2,3).reshape(B*U*V, C, X, Y)
        y_ = self.conv(x_)
        X2, Y2 = y_.shape[-2:]
        y = y_.reshape(B, U, V, -1, X2, Y2).permute(0,3,4,5,1,2).contiguous()
        return y


class GN(nn.Module):
    def __init__(self, ch: int, groups: int = 32):
        super().__init__()
        self.gn = nn.GroupNorm(num_groups=min(groups, ch), num_channels=ch)
    def forward(self, x):  # [B,C,X,Y,U,V]
        B,C,X,Y,U,V = x.shape
        x_ = x.permute(0,4,5,1,2,3).reshape(B*U*V, C, X, Y)
        y_ = self.gn(x_)
        y = y_.reshape(B, U, V, C, X, Y).permute(0,3,4,5,1,2).contiguous()
        return y


class ResBlock4D(nn.Module):
    def __init__(self, ch_in: int, ch_out: int, time_dim: int, uv_kernel: int = 1, down: bool = False, up: bool = False):
        super().__init__()
        self.down, self.up = down, up
        stride = 2 if down else 1
        self.conv1 = ConvXY(ch_in, ch_out, k=3, s=stride)
        self.norm1 = GN(ch_out)
        self.uv = QuaternionUVConv(ch_out, kernel_size=uv_kernel)
        self.norm2 = GN(ch_out)
        self.act = nn.GELU()
        self.emb = nn.Linear(time_dim, ch_out)
        self.skip = None
        if ch_in != ch_out or down or up:
            # для up используем интерполяцию в forward
            self.skip = nn.Conv2d(ch_in, ch_out, 1)


    def forward(self, x, t_emb):
        B,C,X_orig,Y_orig,U,V = x.shape
        h = x
        if self.up:
            # апсемплинг только по XY
            h_ = h.permute(0,4,5,1,2,3).reshape(B*U*V, C, X_orig, Y_orig)
            h_ = F.interpolate(h_, scale_factor=2, mode='nearest')
            X_new, Y_new = h_.shape[-2:]
            h = h_.reshape(B, U, V, C, X_new, Y_new).permute(0,3,4,5,1,2)
        else:
            X_new, Y_new = X_orig, Y_orig
        h = self.conv1(h)
        h = self.norm1(h)
        # добавим time-эмбеддинг как сдвиг/смещение
        temb = self.emb(t_emb).view(B, -1, 1, 1, 1, 1)
        h = h + temb
        h = self.act(h)
        h = self.uv(h)
        h = self.norm2(h)
        h = self.act(h)


        # skip
        if self.skip is not None:
            x_ = x.permute(0,4,5,1,2,3).reshape(B*U*V, C, X_orig, Y_orig)
            if self.down:
                x_ = F.avg_pool2d(x_, kernel_size=2)
            elif self.up:
                # для апсемплинга нужно интерполировать skip connection
                x_ = F.interpolate(x_, size=(X_new, Y_new), mode='nearest')
            x_ = self.skip(x_)
            x = x_.reshape(B, U, V, -1, h.shape[2], h.shape[3]).permute(0,3,4,5,1,2)
        return x + h


class TimeEmbed(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim*4), nn.SiLU(), nn.Linear(dim*4, dim)
        )
    def forward(self, t):
        # sinusoidal
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(0, half, device=device) / half)
        args = t[:, None] * freqs[None]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0,1))
        return self.mlp(emb)


class UNet4D(nn.Module):
    def __init__(self, in_channels=3, base_channels=64, channel_mults=(1,2,2,2), num_res_blocks=2, uv_kernel=1):
        super().__init__()
        self.time = TimeEmbed(dim=128)
        ch = base_channels
        self.in_proj = ConvXY(in_channels, ch, k=3, s=1)


        # down
        downs = []
        chs = [ch]
        for i, mult in enumerate(channel_mults):
            out_ch = base_channels * mult
            for _ in range(num_res_blocks):
                downs.append(ResBlock4D(ch, out_ch, time_dim=128, uv_kernel=uv_kernel, down=False))
                ch = out_ch
                chs.append(ch)
            # downsample
            downs.append(ResBlock4D(ch, ch, time_dim=128, uv_kernel=uv_kernel, down=True))
            chs.append(ch)
        self.downs = nn.ModuleList(downs)


        # middle
        self.mid = ResBlock4D(ch, ch, time_dim=128, uv_kernel=uv_kernel)


        # up
        ups = []
        for i, mult in list(enumerate(channel_mults))[::-1]:
            out_ch = base_channels * mult
            # несколько блоков с апсемплом в конце
            for _ in range(num_res_blocks):
                ups.append(ResBlock4D(ch, out_ch, time_dim=128, uv_kernel=uv_kernel))
                ch = out_ch
            ups.append(ResBlock4D(ch, ch, time_dim=128, uv_kernel=uv_kernel, up=True))
        self.ups = nn.ModuleList(ups)


        self.out_proj = ConvXY(ch, in_channels, k=3, s=1)


    def forward(self, x4, t):  # x4: [B,C,X,Y,U,V]
        t_emb = self.time(t)
        h = self.in_proj(x4)
        skips = []
        for m in self.downs:
            h = m(h, t_emb)
            skips.append(h)
        h = self.mid(h, t_emb)
        for m in self.ups:
            # простая U-Net без явных конкатенаций, чтобы упростить память
            h = m(h, t_emb)
        out = self.out_proj(h)
        return out

