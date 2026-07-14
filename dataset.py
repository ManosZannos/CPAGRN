"""
dataset.py — Fast AISDataset using vectorized numpy operations.

Output per sample:
    obs  : [N, obs_len, 4]   (LON, LAT, SOG, Heading  — z-score)
    pred : [N, pred_len, 2]  (LON, LAT — z-score)
    mask : [N]               True = real vessel
"""

from __future__ import annotations
import os
import glob
import json

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


def denorm(values: np.ndarray, mean: float, std: float) -> np.ndarray:
    return values * std + mean


class AISDataset(Dataset):

    def __init__(
        self,
        csv_dir:     str,
        obs_len:     int = 5,
        pred_len:    int = 5,
        min_vessels: int = 3,
        stride:      int = 1,
    ):
        self.obs_len    = obs_len
        self.pred_len   = pred_len
        self.win_len    = obs_len + pred_len
        self.min_vessels = min_vessels

        self.obs_list  = []   # list of np.ndarray [N, obs_len, 4]
        self.pred_list = []   # list of np.ndarray [N, pred_len, 2]

        files = sorted(glob.glob(os.path.join(csv_dir, '*.csv')))
        if not files:
            raise FileNotFoundError(f'No CSV files in {csv_dir}')

        total_valid = 0
        for fpath in files:
            n = self._process_file(fpath, stride)
            total_valid += n

        split = os.path.basename(csv_dir)
        print(f'  {split}: {total_valid:,} samples from {len(files)} files')

    #Core: process one daily CSV into windows

    def _process_file(self, fpath: str, stride: int) -> int:

        df = pd.read_csv(fpath).dropna()
        if df.empty:
            return 0

        #Build 3D array [T, N, 4] for this file
        # Map frame_id and vessel_id to contiguous indices
        ts_vals  = np.sort(df['frame_id'].unique())
        v_vals   = np.sort(df['vessel_id'].unique())

        T = len(ts_vals)
        N = len(v_vals)

        if T < self.win_len:
            return 0

        ts_idx = {t: i for i, t in enumerate(ts_vals)}
        v_idx  = {v: i for i, v in enumerate(v_vals)}

        # Fill 3D array using vectorized indexing
        arr = np.full((T, N, 4), np.nan, dtype=np.float32)

        ti = df['frame_id'].map(ts_idx).values
        vi = df['vessel_id'].map(v_idx).values
        arr[ti, vi, 0] = df['LON'].values
        arr[ti, vi, 1] = df['LAT'].values
        arr[ti, vi, 2] = df['SOG'].values
        arr[ti, vi, 3] = df['Heading'].values

        #Verify consecutive frame_ids
        ts_diffs = np.diff(ts_vals)

        n_valid = 0
        for start in range(0, T - self.win_len + 1, stride):
            end = start + self.win_len

            # Check: all frame_ids in this window are consecutive
            if np.any(ts_diffs[start:end - 1] != 1):
                continue

            obs_arr  = arr[start : start + self.obs_len]   # [obs_len, N, 4]
            pred_arr = arr[start + self.obs_len : end]      # [pred_len, N, 2 of 4]

            # Vessels present at ALL obs timesteps (no NaN in LAT/LON)
            # obs_arr: [obs_len, N, 4] → check NaN along time and feature
            present = ~np.isnan(obs_arr[:, :, :2]).any(axis=(0, 2))  # [N]

            if present.sum() < self.min_vessels:
                continue

            # Also require complete prediction trajectory for those vessels
            pred_latlon = pred_arr[:, :, :2]                          # [pred_len, N, 2]
            has_pred = ~np.isnan(pred_latlon[:, present, :]).any(axis=(0, 2))  # [N_present]

            if has_pred.sum() < self.min_vessels:
                continue

            # Final valid vessels: present in obs AND pred
            valid_idx = np.where(present)[0][has_pred]

            if len(valid_idx) < self.min_vessels:
                continue

            self.obs_list.append(
                obs_arr[:, valid_idx, :].transpose(1, 0, 2)   # [N, obs_len, 4]
            )
            self.pred_list.append(
                pred_latlon[:, valid_idx, :].transpose(1, 0, 2)  # [N, pred_len, 2]
            )
            n_valid += 1

        return n_valid

    #Dataset interface

    def __len__(self) -> int:
        return len(self.obs_list)

    def __getitem__(self, idx: int):
        return (
            torch.from_numpy(self.obs_list[idx]),   # [N, obs_len, 4]
            torch.from_numpy(self.pred_list[idx]),  # [N, pred_len, 2]
        )

# Collate: pad to max N in batch
def collate_fn(batch):
    obs_list  = [item[0] for item in batch]
    pred_list = [item[1] for item in batch]

    counts   = torch.tensor([o.shape[0] for o in obs_list])
    N_max    = counts.max().item()
    B        = len(batch)
    obs_len  = obs_list[0].shape[1]
    pred_len = pred_list[0].shape[1]

    obs_pad  = torch.zeros(B, N_max, obs_len,  4)
    pred_pad = torch.zeros(B, N_max, pred_len, 2)
    mask     = torch.zeros(B, N_max, dtype=torch.bool)

    for i, (obs, pred) in enumerate(zip(obs_list, pred_list)):
        N = obs.shape[0]
        obs_pad[i,  :N] = obs
        pred_pad[i, :N] = pred
        mask[i,     :N] = True

    return obs_pad, pred_pad, mask, counts

# Convenience function
def get_dataloaders(
    data_dir:     str = 'dataset/noaa_dec2021_1min',
    obs_len:      int = 5,
    pred_len:     int = 5,
    batch_size:   int = 32,
    train_stride: int = 1,
    eval_stride:  int = None,
    num_workers:  int = 0,
):
    
    # train_stride=1  : maximum overlap for training (more samples)
    #eval_stride=None: defaults to obs_len+pred_len (fully non-overlapping) gives independent test samples for honest evaluation
    
    if eval_stride is None:
        eval_stride = obs_len + pred_len

    stats_path = os.path.join(data_dir, 'global_stats.json')
    with open(stats_path) as f:
        stats = json.load(f)

    print('Building datasets...')
    train_ds = AISDataset(os.path.join(data_dir, 'train'), obs_len, pred_len, stride=train_stride)
    val_ds   = AISDataset(os.path.join(data_dir, 'val'),   obs_len, pred_len, stride=eval_stride)
    test_ds  = AISDataset(os.path.join(data_dir, 'test'),  obs_len, pred_len, stride=eval_stride)

    print(f'\nSample counts:')
    print(f'  Train: {len(train_ds):,}')
    print(f'  Val:   {len(val_ds):,}')
    print(f'  Test:  {len(test_ds):,}')

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=num_workers,
    )

    return train_loader, val_loader, test_loader, stats