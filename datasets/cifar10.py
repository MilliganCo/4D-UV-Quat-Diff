# datasets/cifar10.py
from __future__ import annotations

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


class CIFAR10_4D:
    def __init__(self, root='./data', train=True, download=True):
        tfm = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),  # [0,1]
            transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])  # [-1,1]
        ])
        self.ds = datasets.CIFAR10(root=root, train=train, download=download, transform=tfm)
    def __len__(self): return len(self.ds)
    def __getitem__(self, i):
        img, _ = self.ds[i]
        return img





def make_loader(batch_size=64, num_workers=4, train=True):
    ds = CIFAR10_4D(train=train)
    return DataLoader(ds, batch_size=batch_size, shuffle=train, num_workers=num_workers, pin_memory=True)

