"""
check_arch_match.py — Συγκρίνει τα keys ενός checkpoint με αυτά που περιμένει
ο ΤΡΕΧΩΝ κώδικας model_cpagrn.py (χωρίς να χρειάζεται να φορτωθεί επιτυχώς
το state_dict). Χρήσιμο για να εντοπίσουμε ΠΟΙΑ checkpoints ταιριάζουν με το
τελικό, locked v4 architecture (neighbor_agg + final_spatial) και ποια είναι
από παλιότερες εκδοχές (π.χ. v1 με ένα μόνο "spatial" module).

Usage:
    python check_arch_match.py --tags CPAGRN_obs10_pred10 CPAGRN_obs10_pred10_s123 CPAGRN_obs10_pred10_s456 CPAGRN_obs10_pred10_d128_s42
"""
import argparse
import torch
from model_cpagrn import CPAGRN

p = argparse.ArgumentParser()
p.add_argument('--tags', nargs='+', required=True)
args = p.parse_args()

current_keys = set(CPAGRN(feature_size=4, d_model=64, gru_layers=1, pred_len=10).state_dict().keys())

for tag in args.tags:
    path = f'checkpoints/{tag}/val_best.pth'
    try:
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
    except FileNotFoundError:
        print(f'{tag:40s} ❌ ΔΕΝ βρέθηκε ({path})')
        continue
    ckpt_keys = set(ckpt['model'].keys())
    n_params_in_ckpt = sum(v.numel() for v in ckpt['model'].values())

    if ckpt_keys == current_keys:
        print(f'{tag:40s} ✅ ΤΑΙΡΙΑΖΕΙ με το τρέχον model_cpagrn.py (v4)  [{n_params_in_ckpt:,} params]')
    else:
        missing = current_keys - ckpt_keys
        extra = ckpt_keys - current_keys
        print(f'{tag:40s} ❌ ΔΕΝ ταιριάζει με το τρέχον v4  [{n_params_in_ckpt:,} params]')
        if extra:
            # Δείξε μόνο τα module-level ονόματα (πριν την πρώτη τελεία) για συντομία
            extra_modules = sorted(set(k.split('.')[0] for k in extra))
            print(f'   {"":40s}   έχει επιπλέον modules: {extra_modules}')
        if missing:
            missing_modules = sorted(set(k.split('.')[0] for k in missing))
            print(f'   {"":40s}   λείπουν modules: {missing_modules}')
