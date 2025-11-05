# sample.py
from __future__ import annotations

import os, argparse, yaml
import torch
from torchvision.utils import save_image, make_grid
from models.unet4d import UNet4D
from diffusion.schedules import Diffusion


@torch.no_grad()
def main(cfg_path: str, ckpt_path: str, out_dir: str, n: int = 64):
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'


    s = cfg['stride_s']
    X = cfg['image_size'] // s
    Y = cfg['image_size'] // s


    model = UNet4D(in_channels=cfg['channels'], base_channels=cfg['base_channels'],
                   channel_mults=tuple(cfg['channel_mults']), num_res_blocks=cfg['num_res_blocks'],
                   uv_kernel=cfg['uv_kernel']).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    if 'ema' in ckpt:
        model.load_state_dict(ckpt['ema'])
    else:
        model.load_state_dict(ckpt['model'])
    model.eval()


    diff = Diffusion(T=cfg['num_timesteps'], schedule=cfg['beta_schedule'], device=device)


    os.makedirs(out_dir, exist_ok=True)
    B = n
    imgs = diff.ddim_sample(model, x_shape=(B, cfg['channels'], X, Y), steps=cfg['sample_steps'], s=s, shift_avg=cfg['shift_averaging'], device=device)
    # денормализация
    imgs = imgs.clamp(-1,1)
    imgs = (imgs + 1)/2
    grid = make_grid(imgs, nrow=int(n**0.5))
    save_image(grid, os.path.join(out_dir, 'samples.png'))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=str, default='configs/cifar10.yaml')
    ap.add_argument('--ckpt', type=str, required=True)
    ap.add_argument('--out', type=str, default='samples')
    ap.add_argument('--n', type=int, default=64)
    args = ap.parse_args()
    main(args.config, args.ckpt, args.out, args.n)

