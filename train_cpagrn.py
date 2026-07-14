"""
train_cpagrn.py — Training script for CPA-GRN (v4: temporally-aware encoder).

"""

from __future__ import annotations
import os
import sys
import math
import time
import argparse
import logging

import torch
import torch.nn as nn
import numpy as np

from dataset import get_dataloaders
from model_cpagrn import CPAGRN, cpagrn_loss


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir',       type=str,   default='dataset/noaa_dec2021_1min')
    p.add_argument('--obs_len',        type=int,   default=5)
    p.add_argument('--pred_len',       type=int,   default=5)
    p.add_argument('--d_model',        type=int,   default=64)
    p.add_argument('--gru_layers',     type=int,   default=1)
    p.add_argument('--epochs',         type=int,   default=200)
    p.add_argument('--batch_size',     type=int,   default=32)
    p.add_argument('--lr',             type=float, default=1e-3)
    p.add_argument('--clip_grad',      type=float, default=1.0)
    p.add_argument('--gpu_num',        type=int,   default=0)
    p.add_argument('--tag',            type=str,   default='CPAGRN_obs5_pred5')
    p.add_argument('--seed',          type=int,   default=42)
    p.add_argument('--log_every',      type=int,   default=10)
    return p.parse_args()


def get_lr(epoch, args):
    warmup = 10
    if epoch < warmup:
        return args.lr * (epoch + 1) / warmup
    progress = (epoch - warmup) / max(1, args.epochs - warmup)
    return args.lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def run_epoch(loader, model, optimizer, device, args, stats, train: bool):
    model.train(train)
    total_loss = 0.0
    n_batches  = 0

    for obs, pred_gt, mask, _ in loader:
        obs     = obs.to(device)
        pred_gt = pred_gt.to(device)
        mask    = mask.to(device)

        last_obs    = obs[:, :, -1, :2]
        target_disp = pred_gt - last_obs.unsqueeze(2)

        pred_disp = model(obs, mask=mask, stats=stats)

        loss = cpagrn_loss(pred_disp, target_disp, mask)

        if train:
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()

        total_loss += loss.item()
        n_batches  += 1

    return total_loss / max(n_batches, 1)


def main():
    args = get_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_num)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    ckpt_dir = os.path.join('checkpoints', args.tag)
    os.makedirs(ckpt_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s  %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(ckpt_dir, 'train.log')),
            logging.StreamHandler(sys.stdout),
        ]
    )
    log = logging.getLogger()
    log.info(f'Tag: {args.tag}')
    log.info(f'Args: {vars(args)}')

    train_loader, val_loader, _, stats = get_dataloaders(
        data_dir   = args.data_dir,
        obs_len    = args.obs_len,
        pred_len   = args.pred_len,
        batch_size = args.batch_size,
    )
    log.info(f'Train batches: {len(train_loader)} | Val batches: {len(val_loader)}')

    model = CPAGRN(
        feature_size = 4,
        d_model      = args.d_model,
        gru_layers   = args.gru_layers,
        pred_len     = args.pred_len,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f'Parameters: {n_params:,}')

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val   = float('inf')
    best_epoch = 0

    for epoch in range(args.epochs):
        lr = get_lr(epoch, args)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        t0 = time.time()
        train_loss = run_epoch(train_loader, model, optimizer, device, args, stats, train=True)
        val_loss   = run_epoch(val_loader,   model, optimizer, device, args, stats, train=False)
        elapsed    = time.time() - t0

        if (epoch + 1) % args.log_every == 0 or epoch == 0:
            log.info(
                f'Epoch {epoch+1:>3}/{args.epochs} | lr={lr:.2e} | '
                f'train={train_loss:.6f} | val={val_loss:.6f} | t={elapsed:.1f}s'
            )

        if val_loss < best_val:
            best_val   = val_loss
            best_epoch = epoch + 1
            torch.save({
                'epoch':    epoch + 1,
                'model':    model.state_dict(),
                'val_loss': val_loss,
                'args':     vars(args),
                'stats':    stats,
            }, os.path.join(ckpt_dir, 'val_best.pth'))

        torch.save({
            'epoch': epoch + 1,
            'model': model.state_dict(),
        }, os.path.join(ckpt_dir, 'latest.pth'))

    log.info(f'Done. Best val: {best_val:.6f} at epoch {best_epoch}')


if __name__ == '__main__':
    main()