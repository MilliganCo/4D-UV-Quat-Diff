# tests/smoke.py
import torch
from models.unet4d import UNet4D


B,C,H,W = 2,3,32,32
s=2
X,Y = H//s, W//s
x4 = torch.randn(B,C,X,Y,s,s)
t = torch.randint(0,1000,(B,)).float()
net = UNet4D(in_channels=C, base_channels=32, channel_mults=(1,2), num_res_blocks=1, uv_kernel=1)
y = net(x4, t)
print('ok', y.shape)

