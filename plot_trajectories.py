"""
plot_trajectories.py — Ποιοτική ανάλυση: εντοπισμός σεναρίου σύγκλισης
(convergence scenario) στο test set και σχεδίαση τροχιών παρατήρησης /
πραγματικής συνέχειας / πρόβλεψης, για No-CPA v4 vs CPA-GRN v4 (10min task).

"""
from __future__ import annotations
import os
import json
import argparse

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from dataset import AISDataset, denorm


# ── CPA γεωμετρία (ίδιος τύπος με model_cpagrn.CPAFeatures, σε numpy) ─────────

def pairwise_dcpa_tcpa(pos: np.ndarray, vel: np.ndarray, eps: float = 1e-6,
                        tcpa_clamp=(-60.0, 60.0), dcpa_clamp=(0.0, 100.0)):
   
    N = pos.shape[0]
    pos_i = pos[:, None, :]
    pos_j = pos[None, :, :]
    vel_i = vel[:, None, :]
    vel_j = vel[None, :, :]

    r = pos_j - pos_i
    v = vel_j - vel_i

    v_sq = (v * v).sum(-1) + eps
    tcpa = np.clip(-(r * v).sum(-1) / v_sq, tcpa_clamp[0], tcpa_clamp[1])
    dcpa = np.clip(np.linalg.norm(r + tcpa[..., None] * v, axis=-1), dcpa_clamp[0], dcpa_clamp[1])
    return dcpa, tcpa


def to_local_nm(obs_np: np.ndarray, stats: dict):
   
    lon_mean, lon_std = stats['LON']['mean'], stats['LON']['std']
    lat_mean, lat_std = stats['LAT']['mean'], stats['LAT']['std']

    lon_last = denorm(obs_np[:, -1, 0], lon_mean, lon_std)
    lat_last = denorm(obs_np[:, -1, 1], lat_mean, lat_std)
    lon_prev = denorm(obs_np[:, -2, 0], lon_mean, lon_std)
    lat_prev = denorm(obs_np[:, -2, 1], lat_mean, lat_std)

    lat_ref_rad = np.radians(lat_last.mean())
    cos_lat = np.cos(lat_ref_rad)

    x_last = lon_last * 60.0 * cos_lat
    y_last = lat_last * 60.0
    x_prev = lon_prev * 60.0 * cos_lat
    y_prev = lat_prev * 60.0

    pos_nm = np.stack([x_last, y_last], axis=-1)          # [N,2]
    vel_nm = pos_nm - np.stack([x_prev, y_prev], axis=-1)  # nm per 1-min step

    return pos_nm, vel_nm


def find_convergence_scenes(test_ds, stats, min_tcpa=0.5, max_tcpa=8.0,
                             min_speed_knots=1.0, max_speed_knots=25.0,
                             scan_n=30):
    
    candidates = []
    for idx in range(len(test_ds)):
        obs = test_ds.obs_list[idx]           # [N, T_obs, 4] numpy, z-score
        N = obs.shape[0]
        if N < 2:
            continue

        pos_nm, vel_nm = to_local_nm(obs, stats)
        speed_kn = np.linalg.norm(vel_nm, axis=-1) * 60.0     # nm/min → knots

        dcpa, tcpa = pairwise_dcpa_tcpa(pos_nm, vel_nm)        # dcpa: nm, tcpa: min
        np.fill_diagonal(dcpa, np.inf)

        speed_ok = ((speed_kn[:, None] >= min_speed_knots) & (speed_kn[:, None] <= max_speed_knots) &
                    (speed_kn[None, :] >= min_speed_knots) & (speed_kn[None, :] <= max_speed_knots))
        tcpa_ok  = (tcpa > min_tcpa) & (tcpa < max_tcpa)
        valid    = speed_ok & tcpa_ok

        dcpa_masked = np.where(valid, dcpa, np.inf)
        if not np.isfinite(dcpa_masked).any():
            continue

        i, j = np.unravel_index(np.argmin(dcpa_masked), dcpa_masked.shape)
        candidates.append({
            'scene_idx': idx, 'i': int(i), 'j': int(j),
            'dcpa': float(dcpa_masked[i, j]), 'tcpa': float(tcpa[i, j]),
            'speed_i_kn': float(speed_kn[i]), 'speed_j_kn': float(speed_kn[j]),
        })

    candidates.sort(key=lambda c: c['dcpa'])
    return candidates[:scan_n]


def count_distractors(pos_nm, i, j, radius_nm=0.5):
    
    N = pos_nm.shape[0]
    others = [k for k in range(N) if k not in (i, j)]
    if not others:
        return 0
    d_to_i = np.linalg.norm(pos_nm[others] - pos_nm[i], axis=-1)
    d_to_j = np.linalg.norm(pos_nm[others] - pos_nm[j], axis=-1)
    close = (d_to_i <= radius_nm) | (d_to_j <= radius_nm)
    return int(close.sum())


def relative_trajectory_ade(pred_lon, pred_lat, true_lon, true_lat, i, j):
    
    rel_pred_lon = pred_lon[j] - pred_lon[i]
    rel_pred_lat = pred_lat[j] - pred_lat[i]
    rel_true_lon = true_lon[j] - true_lon[i]
    rel_true_lat = true_lat[j] - true_lat[i]
    err = np.sqrt((rel_pred_lon - rel_true_lon)**2 + (rel_pred_lat - rel_true_lat)**2)
    return float(err.mean())


def scene_aggregate_ade(result, N):
    
    gt_lon, gt_lat       = result['gt_lonlat']
    nocpa_lon, nocpa_lat = result['nocpa_lonlat']
    cpa_lon, cpa_lat     = result['cpagrn_lonlat']

    err_nocpa = np.sqrt((nocpa_lon - gt_lon)**2 + (nocpa_lat - gt_lat)**2)  # [N,T_pred]
    err_cpa   = np.sqrt((cpa_lon   - gt_lon)**2 + (cpa_lat   - gt_lat)**2)

    agg = {'nocpa': float(err_nocpa.mean()), 'cpa': float(err_cpa.mean())}
    if 'lstm_lonlat' in result:
        lstm_lon, lstm_lat = result['lstm_lonlat']
        err_lstm = np.sqrt((lstm_lon - gt_lon)**2 + (lstm_lat - gt_lat)**2)
        agg['lstm'] = float(err_lstm.mean())
    return agg


# ── Model loaders (ίδιο pattern με measure_inference.py) ────────────────────

def load_lstm(tag, pred_len, device):
    from model_lstm import VanillaLSTM
    ckpt = torch.load(f'checkpoints/{tag}/val_best.pth',
                      map_location=device, weights_only=False)
    saved = ckpt.get('args', {})
    m = VanillaLSTM(feature_size=4,
                    hidden_size=saved.get('hidden_size', 64),
                    num_layers=saved.get('num_layers', 1),
                    pred_len=saved.get('pred_len', pred_len)).to(device)
    m.load_state_dict(ckpt['model'])
    m.eval()
    return m


def load_cpagrn(tag, pred_len, device):
    from model_cpagrn import CPAGRN
    ckpt = torch.load(f'checkpoints/{tag}/val_best.pth',
                      map_location=device, weights_only=False)
    saved = ckpt.get('args', {})
    m = CPAGRN(feature_size=4,
               d_model=saved.get('d_model', 64),
               gru_layers=saved.get('gru_layers', 1),
               pred_len=saved.get('pred_len', pred_len)).to(device)
    m.load_state_dict(ckpt['model'])
    m.eval()
    return m


def load_nocpa(tag, pred_len, device):
    
    from model_cpagrn_nocpa import CPAGRN as CPAGRN_NoCPA
    ckpt = torch.load(f'checkpoints/{tag}/val_best.pth',
                      map_location=device, weights_only=False)
    saved = ckpt.get('args', {})
    m = CPAGRN_NoCPA(feature_size=4,
                      d_model=saved.get('d_model', 64),
                      gru_layers=saved.get('gru_layers', 1),
                      pred_len=saved.get('pred_len', pred_len)).to(device)
    m.load_state_dict(ckpt['model'])
    m.eval()
    return m


def make_identity(T: int, N: int, device):
    identity_spatial  = torch.ones((T, N, N), device=device) * torch.eye(N, device=device)
    identity_temporal = torch.ones((N, T, T), device=device) * torch.eye(T, device=device)
    return [identity_spatial, identity_temporal]


def load_smchn(tag, obs_len, pred_len, device):
    from model_smchn import TrajectoryModel
    ckpt = torch.load(f'checkpoints/{tag}/val_best.pth',
                      map_location=device, weights_only=False)
    saved = ckpt.get('args', {})
    m = TrajectoryModel(
        number_asymmetric_conv_layer = saved.get('number_asymmetric_conv_layer', 2),
        embedding_dims               = saved.get('embedding_dims', 64),
        number_gcn_layers            = saved.get('number_gcn_layers', 1),
        dropout                      = 0.0,
        obs_len                      = saved.get('obs_len', obs_len),
        pred_len                     = saved.get('pred_len', pred_len),
        out_dims                     = 5,
        num_heads                    = saved.get('num_heads', 4),
    ).to(device)
    m.load_state_dict(ckpt['model'])
    m.eval()
    return m


def smchn_predict_abs(model, obs_t, device):
    
    N = obs_t.shape[1]
    T_obs = obs_t.shape[2]

    abs_obs = obs_t[0].permute(1, 0, 2) # [T_obs, N, 4]
    rel_obs = torch.zeros_like(abs_obs)
    rel_obs[1:] = abs_obs[1:] - abs_obs[:-1]

    pos_idx = torch.arange(1, T_obs + 1, device=device, dtype=torch.float32)
    pos_idx = pos_idx.view(T_obs, 1, 1).expand(T_obs, N, 1)

    V_obs = torch.cat([pos_idx, rel_obs], dim=-1).unsqueeze(0)  # [1,T_obs,N,5]
    identity = make_identity(T_obs, N, device)

    V_pred = model(V_obs, identity) # [T_pred, N, 5] Gaussian params over VELOCITY
    mu_vel = V_pred[:, :, :2]
    last_obs_pos = abs_obs[-1, :, :2]
    mu_abs = torch.cumsum(mu_vel, dim=0) + last_obs_pos.unsqueeze(0) # [T_pred,N,2]
    return mu_abs


#Πρόβλεψη + denormalization για ένα σενάριο

def predict_scene(test_ds, scene_idx, nocpa, cpagrn, device, stats, lstm=None, smchn=None):
    
    obs_np  = test_ds.obs_list[scene_idx]    # [N, T_obs, 4]
    pred_np = test_ds.pred_list[scene_idx]   # [N, T_pred, 2]
    N = obs_np.shape[0]

    obs_t  = torch.from_numpy(obs_np).unsqueeze(0).float().to(device)   # [1,N,T_obs,4]
    mask_t = torch.ones(1, N, dtype=torch.bool, device=device)

    last_obs = obs_t[:, :, -1, :2] # [1,N,2]

    with torch.no_grad():
        disp_nocpa  = nocpa(obs_t, mask=mask_t) # [1,N,T_pred,2]
        disp_cpagrn = cpagrn(obs_t, mask=mask_t) # [1,N,T_pred,2]
        abs_nocpa   = (disp_nocpa  + last_obs.unsqueeze(2))[0].cpu().numpy() # [N,T_pred,2]
        abs_cpagrn  = (disp_cpagrn + last_obs.unsqueeze(2))[0].cpu().numpy()

        abs_lstm = None
        if lstm is not None:
            disp_lstm = lstm(obs_t, mask=mask_t)
            abs_lstm  = (disp_lstm + last_obs.unsqueeze(2))[0].cpu().numpy()

        abs_smchn = None
        if smchn is not None:
            mu_abs = smchn_predict_abs(smchn, obs_t, device) # [T_pred,N,2]
            abs_smchn = mu_abs.permute(1, 0, 2).cpu().numpy() # [N,T_pred,2]

    lon_mean, lon_std = stats['LON']['mean'], stats['LON']['std']
    lat_mean, lat_std = stats['LAT']['mean'], stats['LAT']['std']

    def to_degrees(arr_zscore):
        lon = denorm(arr_zscore[..., 0], lon_mean, lon_std)
        lat = denorm(arr_zscore[..., 1], lat_mean, lat_std)
        return lon, lat

    result = {
        'obs_lonlat':    to_degrees(obs_np[..., :2]), # ([N,T_obs], [N,T_obs])
        'gt_lonlat':     to_degrees(pred_np), # ([N,T_pred], [N,T_pred])
        'nocpa_lonlat':  to_degrees(abs_nocpa),
        'cpagrn_lonlat': to_degrees(abs_cpagrn),
    }
    if abs_lstm is not None:
        result['lstm_lonlat'] = to_degrees(abs_lstm)
    if abs_smchn is not None:
        result['smchn_lonlat'] = to_degrees(abs_smchn)
    return result


def ade_degrees(pred_lon, pred_lat, true_lon, true_lat, vessel_idx):
    err = np.sqrt((pred_lon[vessel_idx] - true_lon[vessel_idx])**2 +
                  (pred_lat[vessel_idx] - true_lat[vessel_idx])**2)
    return float(err.mean())


# Plot

def plot_scene(result, i, j, dcpa, tcpa, out_path, include_smchn, include_lstm=True):
    
    obs_lon, obs_lat     = result['obs_lonlat']
    gt_lon,  gt_lat      = result['gt_lonlat']
    nocpa_lon, nocpa_lat = result['nocpa_lonlat']
    cpa_lon,  cpa_lat    = result['cpagrn_lonlat']
    has_lstm = include_lstm and 'lstm_lonlat' in result
    if has_lstm:
        lstm_lon, lstm_lat = result['lstm_lonlat']

    fig, ax = plt.subplots(figsize=(10, 7.5))

    N = obs_lon.shape[0]
    other = [k for k in range(N) if k not in (i, j)]
    ax.scatter(obs_lon[other, -1], obs_lat[other, -1],
               c='lightgray', s=12, zorder=1, label='Άλλα πλοία σκηνής (τελ. θέση)')

    all_lons, all_lats = [], []
    for vessel in (i, j):
        all_lons += [obs_lon[vessel], gt_lon[vessel], nocpa_lon[vessel], cpa_lon[vessel]]
        all_lats += [obs_lat[vessel], gt_lat[vessel], nocpa_lat[vessel], cpa_lat[vessel]]
        if has_lstm:
            all_lons.append(lstm_lon[vessel]); all_lats.append(lstm_lat[vessel])
        if include_smchn and 'smchn_lonlat' in result:
            sm_lon, sm_lat = result['smchn_lonlat']
            all_lons.append(sm_lon[vessel]); all_lats.append(sm_lat[vessel])
    all_lons = np.concatenate(all_lons)
    all_lats = np.concatenate(all_lats)

    lon_min, lon_max = all_lons.min(), all_lons.max()
    lat_min, lat_max = all_lats.min(), all_lats.max()
    lon_pad = max((lon_max - lon_min) * 0.35, 0.003)
    lat_pad = max((lat_max - lat_min) * 0.35, 0.003)

    colors = {'i': 'tab:blue', 'j': 'tab:orange'}
    for vessel, color, label in [(i, colors['i'], 'Πλοίο A'), (j, colors['j'], 'Πλοίο B')]:
        # Παρατήρηση (obs)
        ax.plot(obs_lon[vessel], obs_lat[vessel], '-o', color=color,
                linewidth=2, markersize=4, zorder=3,
                label=f'{label} — παρατήρηση')
        # Πραγματική συνέχεια (ground truth)
        full_gt_lon = np.concatenate([[obs_lon[vessel, -1]], gt_lon[vessel]])
        full_gt_lat = np.concatenate([[obs_lat[vessel, -1]], gt_lat[vessel]])
        ax.plot(full_gt_lon, full_gt_lat, '-', color=color, linewidth=2,
                alpha=0.9, zorder=3, label=f'{label} — πραγματική τροχιά')
        # No-CPA πρόβλεψη (ίδιο attention, ΧΩΡΙΣ TCPA/DCPA — κύρια σύγκριση)
        full_nocpa_lon = np.concatenate([[obs_lon[vessel, -1]], nocpa_lon[vessel]])
        full_nocpa_lat = np.concatenate([[obs_lat[vessel, -1]], nocpa_lat[vessel]])
        ax.plot(full_nocpa_lon, full_nocpa_lat, '--', color=color, linewidth=1.8,
                alpha=0.8, zorder=3, label=f'{label} — No-CPA πρόβλεψη')
        # CPA-GRN πρόβλεψη (πλήρες μοντέλο)
        full_cpa_lon = np.concatenate([[obs_lon[vessel, -1]], cpa_lon[vessel]])
        full_cpa_lat = np.concatenate([[obs_lat[vessel, -1]], cpa_lat[vessel]])
        ax.plot(full_cpa_lon, full_cpa_lat, ':', color=color, linewidth=2.4,
                alpha=0.95, zorder=4, label=f'{label} — CPA-GRN πρόβλεψη')

        # LSTM πρόβλεψη (προαιρετική, αχνή — ΜΟΝΟ οπτική αναφορά "μηδενικής αλληλεπίδρασης")
        if has_lstm:
            full_lstm_lon = np.concatenate([[obs_lon[vessel, -1]], lstm_lon[vessel]])
            full_lstm_lat = np.concatenate([[obs_lat[vessel, -1]], lstm_lat[vessel]])
            ax.plot(full_lstm_lon, full_lstm_lat, '-.', color=color, linewidth=1.1,
                    alpha=0.45, zorder=1, label=f'{label} — LSTM πρόβλεψη (αναφορά)')

        if include_smchn and 'smchn_lonlat' in result:
            sm_lon, sm_lat = result['smchn_lonlat']
            full_sm_lon = np.concatenate([[obs_lon[vessel, -1]], sm_lon[vessel]])
            full_sm_lat = np.concatenate([[obs_lat[vessel, -1]], sm_lat[vessel]])
            ax.plot(full_sm_lon, full_sm_lat, (0, (1, 1)), color=color, linewidth=1.1,
                    alpha=0.45, zorder=1, label=f'{label} — SMCHN πρόβλεψη (αναφορά)')

        # Marker στο τελευταίο σημείο παρατήρησης (σημείο σύγκλισης)
        ax.scatter([obs_lon[vessel, -1]], [obs_lat[vessel, -1]],
                   c=color, s=90, marker='*', zorder=5, edgecolors='black')

    ax.set_xlim(lon_min - lon_pad, lon_max + lon_pad)
    ax.set_ylim(lat_min - lat_pad, lat_max + lat_pad)

    ade_nocpa_i = ade_degrees(nocpa_lon, nocpa_lat, gt_lon, gt_lat, i)
    ade_cpa_i   = ade_degrees(cpa_lon,   cpa_lat,   gt_lon, gt_lat, i)
    ade_nocpa_j = ade_degrees(nocpa_lon, nocpa_lat, gt_lon, gt_lat, j)
    ade_cpa_j   = ade_degrees(cpa_lon,   cpa_lat,   gt_lon, gt_lat, j)
    rade_nocpa  = relative_trajectory_ade(nocpa_lon, nocpa_lat, gt_lon, gt_lat, i, j)
    rade_cpa    = relative_trajectory_ade(cpa_lon,   cpa_lat,   gt_lon, gt_lat, i, j)

    dcpa_m = dcpa * 1852.0 # nm → μέτρα, για να είναι πιο κατανοητό σε μικρές αποστάσεις
    title = (f'Σενάριο σύγκλισης — DCPA={dcpa_m:.1f} m   TCPA={tcpa:.1f} min\n'
             f'ADE Πλοίο A: No-CPA={ade_nocpa_i:.5f}° CPA-GRN={ade_cpa_i:.5f}°   |   '
             f'ADE Πλοίο B: No-CPA={ade_nocpa_j:.5f}° CPA-GRN={ade_cpa_j:.5f}°\n'
             f'Relative-trajectory ADE (A↔B): No-CPA={rade_nocpa:.5f}° CPA-GRN={rade_cpa:.5f}°'
             + ('   [+ No-CPA/LSTM ως αναφορά]' if has_lstm else ''))
    ax.set_title(title, fontsize=9)
    ax.set_xlabel('Γεωγραφικό μήκος (LON, °)')
    ax.set_ylabel('Γεωγραφικό πλάτος (LAT, °)')
    ax.legend(fontsize=7, loc='upper center', bbox_to_anchor=(0.5, -0.14), ncol=4)
    ax.set_aspect('equal', adjustable='datalim')
    fig.tight_layout(rect=[0, 0.10, 1, 0.90])

    fig.savefig(out_path + '.png', dpi=300)
    fig.savefig(out_path + '.pdf')
    plt.close(fig)

    return {
        'ade_nocpa_i': ade_nocpa_i, 'ade_cpa_i': ade_cpa_i,
        'ade_nocpa_j': ade_nocpa_j, 'ade_cpa_j': ade_cpa_j,
        'rade_nocpa': rade_nocpa, 'rade_cpa': rade_cpa,
    }


#Main

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--gpu_num',   type=int, default=0)
    p.add_argument('--obs_len',   type=int, default=10)
    p.add_argument('--pred_len',  type=int, default=10)
    p.add_argument('--seed',      type=int, default=42)
    p.add_argument('--data_dir',  type=str, default='dataset/noaa_dec2021_1min')
    p.add_argument('--out_dir',   type=str, default='figures/convergence')
    p.add_argument('--scan_n',    type=int, default=30,
                   help='Πόσα υποψήφια σενάρια σύγκλισης (κατά DCPA) να αξιολογηθούν συνολικά')
    p.add_argument('--top_n',     type=int, default=3,
                   help='Πόσα από τα καλύτερα (κατά συνδυασμένο score) να αποθηκευτούν ως plots')
    p.add_argument('--min_tcpa',  type=float, default=0.5)
    p.add_argument('--max_tcpa',  type=float, default=8.0)
    p.add_argument('--min_speed_knots', type=float, default=1.0,
                   help='Ελάχιστη φυσικά αποδεκτή ταχύτητα (κόμβοι) και για τα δύο πλοία')
    p.add_argument('--max_speed_knots', type=float, default=25.0,
                   help='Μέγιστη φυσικά αποδεκτή ταχύτητα (κόμβοι) — συνεπές με SOG<=22kn '
                        'της προεπεξεργασίας (§4.1.1.3), με μικρό περιθώριο ανοχής')
    p.add_argument('--max_distractors', type=int, default=None,
                   help='Αν οριστεί, αγνοεί σενάρια με περισσότερα από αυτά τα άλλα πλοία '
                        'κοντά στη δυάδα A-B (βλ. count_distractors) — προτιμά "καθαρές" δυάδες '
                        'όπου το top-k attention πιθανότατα επικεντρώνεται στο ίδιο το ζεύγος.')
    p.add_argument('--distractor_radius_nm', type=float, default=0.5,
                   help='Ακτίνα (nm) γύρω από A/B εντός της οποίας μετράμε distractors.')
    p.add_argument('--rank_by', type=str, default='combined',
                   choices=['pair', 'relative', 'combined'],
                   help="'pair': ADE No-CPA vs CPA-GRN στο ζεύγος A-B. "
                        "'relative': μόνο βελτίωση σχετικής τροχιάς (RADE) — πιο αυστηρή "
                        "απόδειξη μεταξύ-τους αλληλεπίδρασης. "
                        "'combined' (default): μέσος όρος τυποποιημένης κατάταξης και των δύο.")
    p.add_argument('--include_smchn', action='store_true')
    p.add_argument('--no_lstm', action='store_true',
                   help='Αν οριστεί, το LSTM δεν φορτώνεται/σχεδιάζεται καθόλου — μόνο '
                        'No-CPA vs CPA-GRN. Default: το LSTM παραμένει ως αχνή, προαιρετική '
                        'γραμμή αναφοράς, χωρίς να επηρεάζει την επιλογή σεναρίου.')
    p.add_argument('--nocpa_tag', type=str, default=None,
                   help='Override για το tag του No-CPA checkpoint. Default: '
                        'CPAGRN_nocpa_obs{T}_pred{T}_s{seed} (σύμβαση ablation guide).')
    args = p.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_num)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    os.makedirs(args.out_dir, exist_ok=True)

    with open(os.path.join(args.data_dir, 'global_stats.json')) as f:
        stats = json.load(f)

    print('Φόρτωση test dataset...')
    test_ds = AISDataset(os.path.join(args.data_dir, 'test'),
                          args.obs_len, args.pred_len,
                          stride=args.obs_len + args.pred_len)
    print(f'  Test set: {len(test_ds):,} σκηνές\n')

    print(f'Αναζήτηση {args.scan_n} υποψήφιων σεναρίων σύγκλισης '
          f'(γεωμετρικά, με φυσικό φίλτρο ταχύτητας {args.min_speed_knots}-{args.max_speed_knots} κόμβοι)...')
    candidates = find_convergence_scenes(
        test_ds, stats, min_tcpa=args.min_tcpa, max_tcpa=args.max_tcpa,
        min_speed_knots=args.min_speed_knots, max_speed_knots=args.max_speed_knots,
        scan_n=args.scan_n,
    )
    if not candidates:
        raise RuntimeError(
            'Δεν βρέθηκε κανένα σενάριο σύγκλισης με τους δεδομένους '
            'περιορισμούς (--min_tcpa/--max_tcpa/--min_speed). Δοκίμασε να '
            'χαλαρώσεις τα thresholds.'
        )
    print(f'  Βρέθηκαν {len(candidates)} υποψήφια σενάρια.\n')

    nocpa_tag  = args.nocpa_tag or f'CPAGRN_nocpa_obs{args.obs_len}_pred{args.pred_len}_s{args.seed}'
    cpagrn_tag = f'CPAGRN_obs{args.obs_len}_pred{args.pred_len}_s{args.seed}'
    lstm_tag   = f'LSTM_obs{args.obs_len}_pred{args.pred_len}'
    smchn_tag  = f'SMCHN_obs{args.obs_len}_pred{args.pred_len}_s{args.seed}'

    print('Φόρτωση μοντέλων...')
    nocpa  = load_nocpa(nocpa_tag, args.pred_len, device)
    cpagrn = load_cpagrn(cpagrn_tag, args.pred_len, device)
    lstm   = None
    if not args.no_lstm:
        lstm = load_lstm(lstm_tag, args.pred_len, device)
    smchn  = None
    if args.include_smchn:
        try:
            smchn = load_smchn(smchn_tag, args.obs_len, args.pred_len, device)
        except Exception as e:
            print(f'  ⚠ SMCHN δεν φορτώθηκε ({e}) — συνεχίζω χωρίς αυτό.')

    print('\nΑξιολόγηση όλων των υποψηφίων (αυτό μπορεί να πάρει λίγα λεπτά)...\n')
    evaluated = []
    for c in candidates:
        obs = test_ds.obs_list[c['scene_idx']]
        N = obs.shape[0]
        pos_nm, _ = to_local_nm(obs, stats)
        n_distractors = count_distractors(pos_nm, c['i'], c['j'], radius_nm=args.distractor_radius_nm)

        if args.max_distractors is not None and n_distractors > args.max_distractors:
            continue

        result = predict_scene(test_ds, c['scene_idx'], nocpa, cpagrn, device, stats,
                                lstm=lstm, smchn=smchn)

        gt_lon, gt_lat       = result['gt_lonlat']
        nocpa_lon, nocpa_lat = result['nocpa_lonlat']
        cpa_lon, cpa_lat     = result['cpagrn_lonlat']

        ade_nocpa_i = ade_degrees(nocpa_lon, nocpa_lat, gt_lon, gt_lat, c['i'])
        ade_cpa_i   = ade_degrees(cpa_lon,   cpa_lat,   gt_lon, gt_lat, c['i'])
        ade_nocpa_j = ade_degrees(nocpa_lon, nocpa_lat, gt_lon, gt_lat, c['j'])
        ade_cpa_j   = ade_degrees(cpa_lon,   cpa_lat,   gt_lon, gt_lat, c['j'])

        pair_ade_nocpa = (ade_nocpa_i + ade_nocpa_j) / 2
        pair_ade_cpa   = (ade_cpa_i   + ade_cpa_j)   / 2
        pair_advantage = pair_ade_nocpa - pair_ade_cpa   # θετικό = CPA-GRN καλύτερο στο ζεύγος

        rade_nocpa = relative_trajectory_ade(nocpa_lon, nocpa_lat, gt_lon, gt_lat, c['i'], c['j'])
        rade_cpa   = relative_trajectory_ade(cpa_lon,   cpa_lat,   gt_lon, gt_lat, c['i'], c['j'])
        rade_advantage = rade_nocpa - rade_cpa   # θετικό = CPA-GRN καλύτερο στη ΣΧΕΤΙΚΗ τροχιά

        agg = scene_aggregate_ade(result, N)

        evaluated.append({
            **c, 'N': N, 'result': result, 'n_distractors': n_distractors,
            'ade_nocpa_i': ade_nocpa_i, 'ade_cpa_i': ade_cpa_i,
            'ade_nocpa_j': ade_nocpa_j, 'ade_cpa_j': ade_cpa_j,
            'pair_advantage': pair_advantage,
            'rade_nocpa': rade_nocpa, 'rade_cpa': rade_cpa,
            'rade_advantage': rade_advantage,
            'agg': agg,
        })

    if not evaluated:
        raise RuntimeError(
            'Κανένα υποψήφιο δεν πέρασε το --max_distractors φίλτρο. '
            'Χαλάρωσε το όριο ή αύξησε --scan_n.'
        )

    # Τυποποιημένη (rank-based) κατάταξη, ώστε το 'combined' να μη κυριαρχείται
    # από τη μονάδα μέτρησης του ενός εκ των δύο score (ADE vs RADE έχουν
    # διαφορετική κλίμακα ανάλογα με τη σκηνή).
    by_pair = sorted(evaluated, key=lambda x: -x['pair_advantage'])
    by_rade = sorted(evaluated, key=lambda x: -x['rade_advantage'])
    rank_pair = {id(e): r for r, e in enumerate(by_pair)}
    rank_rade = {id(e): r for r, e in enumerate(by_rade)}
    for e in evaluated:
        e['combined_rank_score'] = rank_pair[id(e)] + rank_rade[id(e)]  # χαμηλότερο = καλύτερο

    if args.rank_by == 'pair':
        ranked = sorted(evaluated, key=lambda x: -x['pair_advantage'])
    elif args.rank_by == 'relative':
        ranked = sorted(evaluated, key=lambda x: -x['rade_advantage'])
    else:
        ranked = sorted(evaluated, key=lambda x: x['combined_rank_score'])

    print(f'{"scene":>6} {"N":>4} {"dist":>4} {"DCPA":>7} {"TCPA":>6} | '
          f'{"pairNoCPA":>10} {"pairCPA":>9} {"pairAdv":>8} | '
          f'{"radeNoCPA":>10} {"radeCPA":>9} {"radeAdv":>8}')
    for e in ranked:
        print(f"{e['scene_idx']:>6} {e['N']:>4} {e['n_distractors']:>4} "
              f"{e['dcpa']:>7.4f} {e['tcpa']:>6.2f} | "
              f"{e['ade_nocpa_i']+e['ade_nocpa_j']:>10.5f} "
              f"{e['ade_cpa_i']+e['ade_cpa_j']:>9.5f} "
              f"{e['pair_advantage']:>+8.5f} | "
              f"{e['rade_nocpa']:>10.5f} {e['rade_cpa']:>9.5f} "
              f"{e['rade_advantage']:>+8.5f}")

    best = ranked[:args.top_n]
    n_wins_pair = sum(1 for e in evaluated if e['pair_advantage'] > 0)
    n_wins_rade = sum(1 for e in evaluated if e['rade_advantage'] > 0)
    n_wins_both = sum(1 for e in evaluated if e['pair_advantage'] > 0 and e['rade_advantage'] > 0)
    print(f"\n{n_wins_pair}/{len(evaluated)} σενάρια: CPA-GRN καλύτερο από No-CPA στο ζεύγος (pair ADE).")
    print(f"{n_wins_rade}/{len(evaluated)} σενάρια: CPA-GRN καλύτερο από No-CPA στη ΣΧΕΤΙΚΗ τροχιά (RADE) "
          f"— το πιο άμεσο τεκμήριο μεταξύ-τους αλληλεπίδρασης.")
    print(f"{n_wins_both}/{len(evaluated)} σενάρια δείχνουν ΚΑΙ ΤΑ ΔΥΟ θετικά — αυτά είναι τα πιο "
          f"αξιόπιστα υποψήφια για το qualitative figure.")
    if n_wins_both == 0:
        print('⚠ ΚΑΝΕΝΑ σενάριο δεν δείχνει ταυτόχρονα θετικό pair ADE ΚΑΙ θετικό RADE.')
        print('  Μην διαλέξεις σενάριο μόνο βάσει pair ADE ως "success story" για τη μεταξύ-τους')
        print('  αλληλεπίδραση — θα ήταν το ίδιο πρόβλημα που ήδη εντοπίσατε. Καλύτερα να')
        print('  παρουσιαστεί ως γενικό ποιοτικό παράδειγμα attention, χωρίς ρητή απόδοση στο CPA,')
        print('  ή να χαλαρώσεις τα thresholds/--scan_n για μεγαλύτερη δεξαμενή υποψηφίων.')

    print('\nΠαραγωγή plots για τα top candidates...')
    for rank, e in enumerate(best, 1):
        out_path = os.path.join(args.out_dir, f'convergence_{rank}_scene{e["scene_idx"]}')
        plot_scene(e['result'], e['i'], e['j'], e['dcpa'], e['tcpa'],
                   out_path, include_smchn=args.include_smchn, include_lstm=not args.no_lstm)
        print(f"  [{rank}] scene={e['scene_idx']} distractors={e['n_distractors']} "
              f"pairAdv={e['pair_advantage']:+.5f} radeAdv={e['rade_advantage']:+.5f} → {out_path}.png")

    print(f'\nΈτοιμο. Έλεγξε τον πίνακα παραπάνω πριν κοιτάξεις τα plots — προτίμησε σενάρια με '
          f'ΘΕΤΙΚΟ pairAdv ΚΑΙ ΘΕΤΙΚΟ radeAdv ΚΑΙ λίγους distractors, γιατί αυτά είναι τα μόνα που '
          f'τεκμηριώνουν ειδικά τη συνεισφορά του TCPA/DCPA στη μεταξύ-τους σχέση, όχι απλώς μια '
          f'γενική βελτίωση ADE που θα μπορούσε να προέρχεται από κάτι άλλο.')


if __name__ == '__main__':
    main()