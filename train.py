# train.py
from __future__ import annotations

import os, math, argparse, yaml
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


from datasets.cifar10 import make_loader
from diffusion.schedules import Diffusion
from models.unet4d import UNet4D
from ops.fold_unfold_4d import image_to_4d, uv_grad_l2


class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k,v in model.state_dict().items()}
    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1-self.decay)
    def copy_to(self, model):
        model.load_state_dict(self.shadow, strict=True)





def main(cfg):
    torch.manual_seed(cfg['seed'])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        try:
            torch.cuda.set_per_process_memory_fraction(0.7, torch.cuda.current_device())
        except Exception:
            pass
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision('medium')

    dl = make_loader(cfg['batch_size'], cfg['num_workers'], train=True)


    # X=H/s, Y=W/s
    s = cfg['stride_s']
    X = cfg['image_size'] // s
    Y = cfg['image_size'] // s


    model = UNet4D(in_channels=cfg['channels'], base_channels=cfg['base_channels'],
                   channel_mults=tuple(cfg['channel_mults']), num_res_blocks=cfg['num_res_blocks'],
                   uv_kernel=cfg['uv_kernel']).to(device)


    diff = Diffusion(T=cfg['num_timesteps'], schedule=cfg['beta_schedule'], device=device)


    opt = optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    ema = EMA(model, decay=cfg['ema_decay'])
    scaler = torch.cuda.amp.GradScaler(enabled=cfg['mixed_precision'] and device=='cuda')


    os.makedirs(cfg['output_dir'], exist_ok=True)


    global_step = 0
    for epoch in range(9999999):
        pbar = tqdm(dl, desc=f"epoch {epoch}")
        for imgs in pbar:
            imgs = imgs.to(device)
            # 4D фолдинг с случайным (δU,δV)
            du = torch.randint(0, s, ()).item()
            dv = torch.randint(0, s, ()).item()
            x4 = image_to_4d(imgs, s=s, du=du, dv=dv)


            B = x4.size(0)
            t = torch.randint(0, diff.T, (B,), device=device)
            noise = torch.randn_like(x4)
            x_t = diff.q_sample(x4, t, noise)


            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                eps_pred = model(x_t, t.float())
                loss_eps = (noise - eps_pred).pow(2).mean()
                loss_uv = cfg['lambda_uv'] * uv_grad_l2(x_t)
                loss = loss_eps + loss_uv


            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            ema.update(model)


            global_step += 1
            pbar.set_postfix({'loss': float(loss.detach()), 'loss_eps': float(loss_eps.detach()), 'loss_uv': float(loss_uv.detach())})


            if global_step % 1000 == 0:
                ckpt = {
                    'model': model.state_dict(),
                    'ema': ema.shadow,
                    'opt': opt.state_dict(),
                    'cfg': cfg,
                    'step': global_step,
                }
                torch.save(ckpt, os.path.join(cfg['output_dir'], f'ckpt_{global_step}.pt'))


            if global_step >= 20000:  # быстрый прогон для smoke
                ckpt = {
                    'model': model.state_dict(),
                    'ema': ema.shadow,
                    'cfg': cfg,
                }
                torch.save(ckpt, os.path.join(cfg['output_dir'], 'ckpt_last.pt'))
                return


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=str, default='configs/cifar10.yaml')
    args = ap.parse_args()
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    main(cfg)

