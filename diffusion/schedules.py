# diffusion/schedules.py
from __future__ import annotations

import torch


def get_beta_schedule(T: int, schedule: str = 'cosine'):
    if schedule == 'linear':
        return torch.linspace(1e-4, 2e-2, T)
    elif schedule == 'cosine':
        # Nichol & Dhariwal cosine
        s = 0.008
        t = torch.linspace(0, 1, T+1)
        alphas_cumprod = torch.cos(((t + s) / (1 + s)) * 3.1415926535 / 2) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return betas.clamp(1e-5, 0.999)
    else:
        raise ValueError(schedule)


class Diffusion:
    def __init__(self, T=1000, schedule='cosine', device='cuda'):
        self.T = T
        self.betas = get_beta_schedule(T, schedule).to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)


    def q_sample(self, x0, t, noise):
        a = self.sqrt_alphas_cumprod[t][:, None, None, None, None, None]
        b = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None, None, None]
        return a * x0 + b * noise


    @torch.no_grad()
    def ddim_sample(self, model, x_shape, steps=50, eta=0.0, s=2, shift_avg=True, device='cuda'):
        # старт из N(0,1) в 4D
        B, C, X, Y = x_shape
        U = V = s
        x = torch.randn(B, C, X, Y, U, V, device=device)
        # индексы времени
        ts = torch.linspace(self.T-1, 0, steps, device=device).long()
        for i, t in enumerate(ts):
            t_batch = t.expand(B)
            eps = model(x, t_batch.float())
            alpha_t = self.alphas[t]
            alpha_bar_t = self.alphas_cumprod[t]
            sqrt_alpha_bar = torch.sqrt(alpha_bar_t)
            sqrt_one_minus = torch.sqrt(1 - alpha_bar_t)
            # x0 estimate
            x0 = (x - sqrt_one_minus * eps) / sqrt_alpha_bar
            if i == steps-1:
                x = x0
                break
            t_prev = ts[i+1]
            alpha_bar_prev = self.alphas_cumprod[t_prev]
            # DDIM update
            sigma = 0.0 if eta == 0 else eta * torch.sqrt((1 - alpha_bar_prev)/(1 - alpha_bar_t) * (1 - alpha_bar_t/alpha_bar_prev))
            x = torch.sqrt(alpha_bar_prev) * x0 + torch.sqrt(1 - alpha_bar_prev - sigma**2) * eps + sigma * torch.randn_like(x)
        if shift_avg:
            # усреднение по 4 сдвигам
            from ops.fold_unfold_4d import field4d_to_image
            imgs = []
            for du in (0,1):
                for dv in (0,1):
                    imgs.append(field4d_to_image(x.roll((du,dv), dims=(-2,-1))))
            img = torch.stack(imgs, dim=0).mean(dim=0)
            return img
        else:
            from ops.fold_unfold_4d import field4d_to_image
            return field4d_to_image(x)

