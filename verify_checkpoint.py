"""
verify_checkpoint.py — Επαλήθευση checkpoint ΠΡΙΝ χρησιμοποιηθεί οπουδήποτε.

Τρέξε αυτό ΠΡΙΝ το plot_trajectories.py, για να επιβεβαιώσεις ότι διαλέγεις
το σωστό, μη-corrupted checkpoint. Ελέγχει:
  1. Αριθμό εκπαιδεύσιμων παραμέτρων — συγκρίνει με τις γνωστές, σωστές τιμές
     (Πίνακας 4.6 της εργασίας): CPA-GRN v4 10min = 52.438, No-CPA v4 10min = 52.310
  2. Την ημερομηνία τροποποίησης του αρχείου (για να δεις ποιο rerun είναι πιο πρόσφατο)
  3. Προαιρετικά, τρέχει πλήρη evaluation στο test set και συγκρίνει το ADE/FDE
     με τις ήδη δημοσιευμένες τιμές (0.001137 / 0.001628 για CPA-GRN, 0.001185 /
     0.001800 για No-CPA, 10min task) — αν διαφέρουν σημαντικά, το checkpoint
     ΔΕΝ είναι αυτό που χρησιμοποιήθηκε στα reported αποτελέσματα.

Usage (τρέξε ΚΑΙ στα δύο υποψήφια tags πριν αποφασίσεις ποιο να χρησιμοποιήσεις
στο plot_trajectories.py):

    python verify_checkpoint.py --tag CPAGRN_obs10_pred10 --model full
    python verify_checkpoint.py --tag CPAGRN_nocpa_v4_obs10_pred10_s42 --model nocpa
    python verify_checkpoint.py --tag CPAGRN_nocpa_obs5_pred5_s42 --model nocpa --obs_len 5 --pred_len 5
    python verify_checkpoint.py --tag CPAGRN_nocpa_v4_obs5_pred5_s42_rerun --model nocpa --obs_len 5 --pred_len 5

    # Με πλήρες evaluation στο test set (πιο αργό αλλά πιο σίγουρο):
    python verify_checkpoint.py --tag CPAGRN_obs10_pred10 --model full --full_eval --gpu_num 0
"""
from __future__ import annotations
import os
import argparse
from datetime import datetime

import torch

# Γνωστές, σωστές τιμές από τον Πίνακα 4.6 (αριθμός παραμέτρων ανά μοντέλο/ορίζοντα)
EXPECTED_PARAMS = {
    'full':  {5: 51788, 10: 52438, 20: 53738, 30: 55038},
    'nocpa': {5: 51660, 10: 52310, 20: 53610, 30: 54910},
    'lstm':  {5: 38410, 10: 39060, 20: 40360, 30: 41660},
}

# Γνωστά, ήδη δημοσιευμένα ADE/FDE (10min task, main comparison table) — μόνο
# για cross-check αν τρέξεις --full_eval σε αυτόν τον συγκεκριμένο ορίζοντα.
EXPECTED_METRICS_10MIN = {
    'full':  {'ADE': 0.001137, 'FDE': 0.001628},
    'nocpa': {'ADE': 0.001185, 'FDE': 0.001800},
    'lstm':  {'ADE': 0.001486, 'FDE': 0.002295},
}


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_model(model_kind, ckpt, obs_len, pred_len, device):
    saved = ckpt.get('args', {})
    if model_kind == 'full':
        from model_cpagrn import CPAGRN
        m = CPAGRN(feature_size=4, d_model=saved.get('d_model', 64),
                   gru_layers=saved.get('gru_layers', 1),
                   pred_len=saved.get('pred_len', pred_len)).to(device)
    elif model_kind == 'nocpa':
        from model_cpagrn_nocpa import CPAGRN as CPAGRN_NoCPA
        m = CPAGRN_NoCPA(feature_size=4, d_model=saved.get('d_model', 64),
                          gru_layers=saved.get('gru_layers', 1),
                          pred_len=saved.get('pred_len', pred_len)).to(device)
    elif model_kind == 'lstm':
        from model_lstm import VanillaLSTM
        m = VanillaLSTM(feature_size=4, hidden_size=saved.get('hidden_size', 64),
                         num_layers=saved.get('num_layers', 1),
                         pred_len=saved.get('pred_len', pred_len)).to(device)
    else:
        raise ValueError(f'Άγνωστο model_kind: {model_kind}')
    m.load_state_dict(ckpt['model'])
    m.eval()
    return m


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tag', type=str, required=True)
    p.add_argument('--model', type=str, required=True, choices=['full', 'nocpa', 'lstm'])
    p.add_argument('--obs_len', type=int, default=10)
    p.add_argument('--pred_len', type=int, default=10)
    p.add_argument('--gpu_num', type=int, default=0)
    p.add_argument('--full_eval', action='store_true',
                   help='Τρέχει πλήρες evaluation στο test set (αργό, αλλά επιβεβαιώνει ADE/FDE)')
    p.add_argument('--data_dir', type=str, default='dataset/noaa_dec2021_1min')
    p.add_argument('--batch_size', type=int, default=32)
    args = p.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_num)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ckpt_path = f'checkpoints/{args.tag}/val_best.pth'
    if not os.path.exists(ckpt_path):
        print(f'❌ ΔΕΝ βρέθηκε: {ckpt_path}')
        return

    mtime = datetime.fromtimestamp(os.path.getmtime(ckpt_path))
    size_mb = os.path.getsize(ckpt_path) / (1024 * 1024)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = load_model(args.model, ckpt, args.obs_len, args.pred_len, device)
    n_params = count_params(model)
    expected = EXPECTED_PARAMS[args.model].get(args.pred_len)

    print('=' * 60)
    print(f'Tag:              {args.tag}')
    print(f'Model kind:       {args.model}')
    print(f'Horizon:          obs={args.obs_len} pred={args.pred_len}')
    print(f'Αρχείο:           {ckpt_path}')
    print(f'Μέγεθος:          {size_mb:.2f} MB')
    print(f'Τροποποιήθηκε:    {mtime.strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'Epoch (ckpt):     {ckpt.get("epoch", "?")}')
    print(f'Val loss (ckpt):  {ckpt.get("val_loss", "?")}')
    print('-' * 60)
    print(f'Αριθμός παραμέτρων: {n_params:,}')
    if expected is not None:
        if n_params == expected:
            print(f'✅ ΤΑΙΡΙΑΖΕΙ με την αναμενόμενη τιμή ({expected:,}) — σωστή αρχιτεκτονική.')
        else:
            print(f'❌ ΔΕΝ ταιριάζει! Αναμενόταν {expected:,}, βρέθηκε {n_params:,}.')
            print('   ΠΡΟΣΟΧΗ: πιθανό corrupted checkpoint (λάθος EDGE_DIM ή architecture).')
            print('   ΜΗΝ το χρησιμοποιήσεις πριν καταλάβεις γιατί διαφέρει.')
    else:
        print(f'(Δεν υπάρχει reference τιμή για ορίζοντα {args.pred_len}min — έλεγξε χειροκίνητα)')
    print('=' * 60)

    if args.full_eval:
        print('\nΤρέχει πλήρες evaluation στο test set (ίδια λογική με evaluate_cpagrn.py)...')
        from dataset import get_dataloaders, denorm
        import numpy as np

        _, val_loader, test_loader, file_stats = get_dataloaders(
            args.data_dir, args.obs_len, args.pred_len, args.batch_size,
        )
        loader = test_loader

        # ΚΡΙΣΙΜΟ: χρησιμοποίησε τα stats που είναι αποθηκευμένα ΜΕΣΑ στο ίδιο
        # το checkpoint (αν υπάρχουν), όχι αυτόματα το global_stats.json —
        # ίδια λογική με το επίσημο evaluate_cpagrn.py. Αν το dataset
        # ξαναδημιουργήθηκε ποτέ με ελαφρώς διαφορετικά στατιστικά, η χρήση
        # του λάθος συνόλου στατιστικών προκαλεί ακριβώς τέτοιου μεγέθους
        # απόκλιση χωρίς να σημαίνει ότι το checkpoint είναι εσφαλμένο.
        stats = ckpt.get('stats', None)
        if stats is None:
            print('  (Το checkpoint δεν έχει αποθηκευμένα δικά του stats — '
                  'χρησιμοποιώ το global_stats.json του dataset.)')
            stats = file_stats
        else:
            print('  (Χρησιμοποιώ τα stats που ήταν αποθηκευμένα ΜΕΣΑ στο checkpoint.)')

        lon_mean, lon_std = stats['LON']['mean'], stats['LON']['std']
        lat_mean, lat_std = stats['LAT']['mean'], stats['LAT']['std']

        T = args.pred_len
        ade_per_horizon = [[] for _ in range(T)]
        fde_list = []

        with torch.no_grad():
            for obs, pred_gt, mask, _ in loader:
                obs, pred_gt, mask = obs.to(device), pred_gt.to(device), mask.to(device)

                last_obs    = obs[:, :, -1, :2]
                target_disp = pred_gt - last_obs.unsqueeze(2)
                pred_disp   = model(obs, mask=mask, stats=stats)

                pred_abs   = (pred_disp   + last_obs.unsqueeze(2)).cpu().numpy()
                target_abs = (target_disp + last_obs.unsqueeze(2)).cpu().numpy()
                mask_np    = mask.cpu().numpy()
                B, N       = mask_np.shape

                pred_lon = denorm(pred_abs[..., 0],   lon_mean, lon_std)
                pred_lat = denorm(pred_abs[..., 1],   lat_mean, lat_std)
                true_lon = denorm(target_abs[..., 0], lon_mean, lon_std)
                true_lat = denorm(target_abs[..., 1], lat_mean, lat_std)

                for b in range(B):
                    for n in range(N):
                        if not mask_np[b, n]:
                            continue
                        err = np.sqrt((pred_lat[b,n,:] - true_lat[b,n,:])**2 +
                                      (pred_lon[b,n,:] - true_lon[b,n,:])**2)
                        for t in range(T):
                            ade_per_horizon[t].append(err[t])
                        fde_list.append(err[-1])

        ade_h = [np.mean(h) for h in ade_per_horizon]
        ade = float(np.mean(ade_h))
        fde = float(np.mean(fde_list))
        print(f'\nADE (test): {ade:.6f}°')
        print(f'FDE (test): {fde:.6f}°')

        if args.pred_len == 10 and args.model in EXPECTED_METRICS_10MIN:
            exp = EXPECTED_METRICS_10MIN[args.model]
            print(f'\nΣύγκριση με reported (10min): ADE={exp["ADE"]}° FDE={exp["FDE"]}°')
            if abs(ade - exp['ADE']) < 1e-5 and abs(fde - exp['FDE']) < 1e-5:
                print('✅ Ταιριάζει με το reported αποτέλεσμα — αυτό είναι το σωστό checkpoint.')
            else:
                print('❌ ΔΕΝ ταιριάζει ακριβώς — μπορεί να είναι διαφορετικό run/seed/checkpoint.')


if __name__ == '__main__':
    main()