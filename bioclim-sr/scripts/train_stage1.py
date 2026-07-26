#!/usr/bin/env python3
"""Stage-1 super-resolution: ERA5-Land (0.1 deg) + DEM -> CHELSA (1/120 deg) tas."""
import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import rasterio
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
        )

    def forward(self, x):
        return x + self.body(x)


class EDSR(nn.Module):
    def __init__(self, in_channels=2, out_channels=1, n_feats=64, n_resblocks=8):
        super().__init__()
        head = [nn.Conv2d(in_channels, n_feats, 3, padding=1)]
        body = [ResidualBlock(n_feats) for _ in range(n_resblocks)]
        body.append(nn.Conv2d(n_feats, n_feats, 3, padding=1))
        tail = [
            nn.Conv2d(n_feats, n_feats, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_feats, out_channels, 3, padding=1),
        ]
        self.head = nn.Sequential(*head)
        self.body = nn.Sequential(*body)
        self.tail = nn.Sequential(*tail)

    def forward(self, x):
        x = self.head(x)
        res = self.body(x)
        res = res + x
        x = self.tail(res)
        return x


class GeoDataset(Dataset):
    def __init__(self, lr_path, dem_path, hr_path, patch_size=128, split='train',
                 val_ratio=0.1, augment=True, seed=42, norm=None):
        self.patch_size = patch_size
        self.split = split
        self.augment = augment and split == 'train'

        with rasterio.open(lr_path) as src:
            self.lr = src.read(1).astype(np.float32)
        with rasterio.open(dem_path) as src:
            self.dem = src.read(1).astype(np.float32)
        with rasterio.open(hr_path) as src:
            self.hr = src.read(1).astype(np.float32)
            self.profile = src.profile

        # Normalize using provided or dataset-wide stats
        self.norm = norm or {}
        self.lr_mean = self.norm.get('lr_mean', float(np.nanmean(self.lr)))
        self.lr_std = self.norm.get('lr_std', float(np.nanstd(self.lr)))
        self.dem_mean = self.norm.get('dem_mean', float(np.nanmean(self.dem)))
        self.dem_std = self.norm.get('dem_std', float(np.nanstd(self.dem)))
        self.hr_mean = self.norm.get('hr_mean', float(np.nanmean(self.hr)))
        self.hr_std = self.norm.get('hr_std', float(np.nanstd(self.hr)))
        self.lr = (self.lr - self.lr_mean) / self.lr_std
        self.dem = (self.dem - self.dem_mean) / self.dem_std
        self.hr = (self.hr - self.hr_mean) / self.hr_std

        # Valid mask: all three finite and HR not NaN
        self.mask = (
            np.isfinite(self.lr) & np.isfinite(self.dem) &
            np.isfinite(self.hr)
        )
        # Compute valid top-left corners for patches
        h, w = self.hr.shape
        ps = patch_size
        self.indices = []
        rng = np.random.default_rng(seed)
        for i in range(0, h - ps + 1, ps // 2):
            for j in range(0, w - ps + 1, ps // 2):
                if self.mask[i:i+ps, j:j+ps].mean() > 0.8:
                    self.indices.append((i, j))
        # Train/val split by patch index (deterministic)
        n = len(self.indices)
        perm = rng.permutation(n)
        n_val = int(n * val_ratio)
        if split == 'train':
            self.indices = [self.indices[k] for k in perm[n_val:]]
        else:
            self.indices = [self.indices[k] for k in perm[:n_val]]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i, j = self.indices[idx]
        ps = self.patch_size
        lr = self.lr[i:i+ps, j:j+ps]
        dem = self.dem[i:i+ps, j:j+ps]
        hr = self.hr[i:i+ps, j:j+ps]
        mask = self.mask[i:i+ps, j:j+ps].astype(np.float32)

        x = np.stack([lr, dem], axis=0)
        y = hr[np.newaxis, ...]

        if self.augment:
            # Random horizontal/vertical flips
            if random.random() > 0.5:
                x = np.flip(x, axis=2).copy()
                y = np.flip(y, axis=2).copy()
                mask = np.flip(mask, axis=1).copy()
            if random.random() > 0.5:
                x = np.flip(x, axis=1).copy()
                y = np.flip(y, axis=1).copy()
                mask = np.flip(mask, axis=0).copy()

        return (
            torch.from_numpy(x).float(),
            torch.from_numpy(y).float(),
            torch.from_numpy(mask)[None, ...],
        )


def masked_l1_loss(pred, target, mask):
    diff = torch.abs(pred - target)
    diff = diff * mask
    return diff.sum() / mask.sum().clamp_min(1.0)


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    ds_train = GeoDataset(args.lr, args.dem, args.hr, patch_size=args.patch_size,
                          split='train', val_ratio=args.val_ratio, augment=True)
    norm = {
        'lr_mean': ds_train.lr_mean, 'lr_std': ds_train.lr_std,
        'dem_mean': ds_train.dem_mean, 'dem_std': ds_train.dem_std,
        'hr_mean': ds_train.hr_mean, 'hr_std': ds_train.hr_std,
    }
    with open(os.path.join(args.out_dir, 'norm.json'), 'w') as f:
        json.dump(norm, f, indent=2)
    ds_val = GeoDataset(args.lr, args.dem, args.hr, patch_size=args.patch_size,
                        split='val', val_ratio=args.val_ratio, augment=False, norm=norm)
    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, pin_memory=True)
    dl_val = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    model = EDSR(in_channels=2, out_channels=1, n_feats=args.n_feats,
                 n_resblocks=args.n_resblocks).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.decay_every, gamma=0.5)

    os.makedirs(args.out_dir, exist_ok=True)
    best_val = float('inf')
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for x, y, mask in tqdm(dl_train, desc=f"Epoch {epoch} train"):
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            pred = model(x)
            loss = masked_l1_loss(pred, y, mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        scheduler.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y, mask in dl_val:
                x, y, mask = x.to(device), y.to(device), mask.to(device)
                pred = model(x)
                val_loss += masked_l1_loss(pred, y, mask).item()
        val_loss /= max(len(dl_val), 1)
        train_loss /= max(len(dl_train), 1)
        print(f"Epoch {epoch}: train={train_loss:.4f} val={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'val_loss': val_loss,
            }, os.path.join(args.out_dir, 'best_model.pt'))
        torch.save(model.state_dict(), os.path.join(args.out_dir, 'last_model.pt'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default='/home/a474r867/beegfs/stage1_conus_2015/era5_tas_2015_1-120_c.tif')
    parser.add_argument('--dem', default='/home/a474r867/beegfs/stage1_conus_2015/dem_1-120.tif')
    parser.add_argument('--hr', default='/home/a474r867/beegfs/stage1_conus_2015/chelsa_tas_2015_1-120.tif')
    parser.add_argument('--out-dir', default='/home/a474r867/beegfs/stage1_conus_2015/models')
    parser.add_argument('--patch-size', type=int, default=128)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--n-resblocks', type=int, default=8)
    parser.add_argument('--n-feats', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr-rate', type=float, default=1e-3)
    parser.add_argument('--decay-every', type=int, default=20)
    parser.add_argument('--val-ratio', type=float, default=0.1)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train(args)


if __name__ == '__main__':
    main()
