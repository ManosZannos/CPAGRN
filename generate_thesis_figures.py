"""
generate_thesis_figures.py

Generates:
  Λίστα Εικόνων:
    eikona_2_2_cpa_geometry.png                 (Εικόνα 2.2)
  Λίστα Διαγραμμάτων:
    diagram_4_2_ade_comparison.png              (Διάγραμμα 4.2)
    diagram_4_3_ablation_cross_horizon.png      (Διάγραμμα 4.3)
    diagram_4_4_smchn_variance_collapse.png     (Διάγραμμα 4.4)

    NOTE: Εικόνα 2.1 (αρχιτεκτονική) και Εικόνα 3.1 (preprocessing pipeline)
  δεν παράγονται πια από script. 

  Διάγραμμα 4.1 (σενάριο σύγκλισης) μέσω του plot_trajectories.py
"""

import os
import sys
import platform
import subprocess
 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
 

# Setup
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(SCRIPT_DIR, "figures")
os.makedirs(OUTDIR, exist_ok=True)
 
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9.5,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.edgecolor": "#333333",
    "axes.linewidth": 0.8,
    "figure.dpi": 100,
})
 
COL_LSTM = "#2b6cb0"    
COL_NOCPA = "#dd8b1e"   
COL_CPAGRN = "#1f8a44"  
COL_SMCHN = "#8e3b9e"  
 
 
def savefig(fig, name):
    png = os.path.join(OUTDIR, f"{name}.png")
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved {png}")
    return png
 
 
def open_image(path):
    """Open an image with the OS default viewer, cross-platform."""
    try:
        system = platform.system()
        if system == "Windows":
            os.startfile(path)  # type: ignore[attr-defined]
        elif system == "Darwin":
            subprocess.run(["open", path], check=False)
        else:
            subprocess.run(["xdg-open", path], check=False)
    except Exception as e:
        print(f"  (could not auto-open {path}: {e})")
 
 

# ΕΙΚΟΝΑ 2.2 — CPA / TCPA / DCPA geometry
def eikona_2_2_cpa_geometry():
    fig, ax = plt.subplots(figsize=(8, 5.6))
 
    pos_i = np.array([1.0, 1.0])
    pos_j = np.array([4.0, 3.0])
    vel_i = np.array([0.8, 0.1])
    vel_j = np.array([-0.3, 0.6])
 
    r = pos_j - pos_i
    v = vel_j - vel_i
    tcpa = -(r @ v) / (v @ v)
    cpa_i = pos_i + tcpa * vel_i
    cpa_j = pos_j + tcpa * vel_j
    dcpa = np.linalg.norm(cpa_j - cpa_i)
 
    t_end = tcpa * 1.35
    traj_i_end = pos_i + t_end * vel_i
    traj_j_end = pos_j + t_end * vel_j
 
    ax.plot([pos_i[0], traj_i_end[0]], [pos_i[1], traj_i_end[1]],
            ls="--", color=COL_LSTM, lw=1.4, zorder=1)
    ax.plot([pos_j[0], traj_j_end[0]], [pos_j[1], traj_j_end[1]],
            ls="--", color=COL_SMCHN, lw=1.4, zorder=1)
 
    ax.scatter(*pos_i, s=140, color=COL_LSTM, zorder=5, edgecolor="white", lw=1.2)
    ax.scatter(*pos_j, s=140, color=COL_SMCHN, zorder=5, edgecolor="white", lw=1.2)
    ax.text(pos_i[0] - 0.28, pos_i[1] - 0.32, "Vessel $i$", fontsize=11, weight="bold", color=COL_LSTM)
    ax.text(pos_j[0] + 0.15, pos_j[1] + 0.15, "Vessel $j$", fontsize=11, weight="bold", color=COL_SMCHN)
 
    scale = 1.6
    ax.annotate("", xy=pos_i + scale * vel_i, xytext=pos_i,
                arrowprops=dict(arrowstyle="-|>", color=COL_LSTM, lw=2))
    ax.text(*(pos_i + scale * vel_i + np.array([0.05, 0.15])), r"$v_i$", color=COL_LSTM, fontsize=12)
 
    ax.annotate("", xy=pos_j + scale * vel_j, xytext=pos_j,
                arrowprops=dict(arrowstyle="-|>", color=COL_SMCHN, lw=2))
    ax.text(*(pos_j + scale * vel_j + np.array([0.05, 0.15])), r"$v_j$", color=COL_SMCHN, fontsize=12)
 
    ax.annotate("", xy=pos_i + scale * v, xytext=pos_i,
                arrowprops=dict(arrowstyle="-|>", color="#555555", lw=1.6, ls=(0, (4, 2))))
    ax.text(*(pos_i + scale * v + np.array([-0.9, -0.35])),
            r"$u = v_j - v_i$", color="#555555", fontsize=10.5)
 
    ax.scatter(*cpa_i, s=55, color=COL_LSTM, zorder=5, marker="D")
    ax.scatter(*cpa_j, s=55, color=COL_SMCHN, zorder=5, marker="D")
    ax.plot([cpa_i[0], cpa_j[0]], [cpa_i[1], cpa_j[1]], color="#c0392b", lw=1.8, zorder=4)
    mid = (cpa_i + cpa_j) / 2
    ax.annotate(f"DCPA = {dcpa:.2f}", xy=mid, xytext=(mid[0] + 0.35, mid[1] + 0.05),
                fontsize=10.5, color="#c0392b", weight="bold",
                arrowprops=dict(arrowstyle="-", color="#c0392b", lw=0.8))
 
    ax.annotate(f"TCPA = {tcpa:.2f}", xy=(pos_i + 0.5 * tcpa * vel_i),
                xytext=(pos_i[0] - 0.9, pos_i[1] + 1.55),
                fontsize=10.5, color="#333333", weight="bold",
                arrowprops=dict(arrowstyle="->", color="#333333", lw=0.9,
                                 connectionstyle="arc3,rad=0.25"))
 
    ax.text(*(cpa_i + np.array([-0.55, -0.28])), "CPA$_i$", fontsize=9, color=COL_LSTM)
    ax.text(*(cpa_j + np.array([0.12, 0.12])), "CPA$_j$", fontsize=9, color=COL_SMCHN)
 
    ax.set_xlim(-0.3, 6.2)
    ax.set_ylim(-0.3, 5.2)
    ax.set_xlabel("x (arbitrary spatial units)")
    ax.set_ylabel("y (arbitrary spatial units)")
    ax.grid(alpha=0.25)
    ax.set_aspect("equal")
    return savefig(fig, "eikona_2_2_cpa_geometry")
 
 

# ΔΙΑΓΡΑΜΜΑ 4.2 — ADE comparison (4 μοντέλα)
def diagram_4_2_ade_comparison():
 
    plt.rcParams.update(plt.rcParamsDefault)
 
    horizons = ["5 min", "10 min", "20 min", "30 min"]
    x = np.arange(len(horizons))
    width = 0.2
 
    ade = {
        "LSTM":      [0.000823, 0.001486, 0.001775, 0.002504],
        "No-CPA v4": [0.001807, 0.001185, 0.001777, 0.002668],
        "CPA-GRN v4":[0.000818, 0.001137, 0.001890, 0.002527],  # UPDATED 30min
        "SMCHN":     [0.001617, 0.003764, 0.002206, 0.003691],
    }
 
    colors = {
        "LSTM":       "#3B78C2",
        "No-CPA v4":  "#E08A2A",
        "CPA-GRN v4": "#2E7D42",
        "SMCHN":      "#8E44AD",
    }
 
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
 
    for i, (name, vals) in enumerate(ade.items()):
        ax.bar(x + (i - 1.5) * width, vals, width, label=name, color=colors[name])
 
    ax.set_xlabel("Prediction horizon")
    ax.set_ylabel("ADE (degrees)")
    ax.set_xticks(x)
    ax.set_xticklabels(horizons)
    ax.set_ylim(0, 0.0040)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=4, frameon=False, fontsize=9.5)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
 
    plt.tight_layout()
    png_path = os.path.join(OUTDIR, "diagram_4_2_ade_comparison.png")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    print(f"  saved {png_path}")
    plt.show()
    return png_path
 
 

# ΔΙΑΓΡΑΜΜΑ 4.3 — ADE/FDE cross-horizon, μελέτη αφαίρεσης
def diagram_4_3_ablation_cross_horizon():
  
    plt.rcParams.update(plt.rcParamsDefault)
 
    horizons = [5, 10, 20, 30]
 
    ade = {
        "LSTM (baseline)":       [0.000823, 0.001486, 0.001775, 0.002504],
        "No-CPA v4 (ablation)":  [0.001807, 0.001185, 0.001777, 0.002668],
        "CPA-GRN v4 (full)":     [0.000818, 0.001137, 0.001890, 0.002527],
    }
 
    fde = {
        "LSTM (baseline)":       [0.001031, 0.002295, 0.002885, 0.004605],
        "No-CPA v4 (ablation)":  [0.002714, 0.001800, 0.002964, 0.004681],
        "CPA-GRN v4 (full)":     [0.001124, 0.001628, 0.003327, 0.004473],
    }
 
    styles = {
        "LSTM (baseline)":      dict(color="#3B78C2", marker="o", linestyle="--", linewidth=1.8, markersize=7),
        "No-CPA v4 (ablation)": dict(color="#E08A2A", marker="s", linestyle="-",  linewidth=1.8, markersize=7),
        "CPA-GRN v4 (full)":    dict(color="#3E8E4F", marker="^", linestyle="-",  linewidth=2.0, markersize=8),
    }
 
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 8.6))
 
    # (a) ADE
    ax = axes[0]
    for name, vals in ade.items():
        ax.plot(horizons, vals, label=name, **styles[name])
    ax.set_title("(a) Average Displacement Error", loc="left", fontsize=11)
    ax.set_ylabel("ADE (degrees)")
    ax.set_ylim(0.00075, 0.00280)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.5f"))
    ax.grid(alpha=0.25)
    ax.annotate("sweet spot\n(CPA + attention\nboth contribute)",
                xy=(11.5, 0.00135), fontsize=8.5, color="#2E6B3E",
                ha="left", va="center")
    ax.annotate("", xy=(10, 0.001137), xytext=(15, 0.00148),
                arrowprops=dict(arrowstyle="-", color="#2E6B3E", lw=0.8))
 
    # (b) FDE
    ax = axes[1]
    for name, vals in fde.items():
        ax.plot(horizons, vals, label=name, **styles[name])
    ax.set_title("(b) Final Displacement Error", loc="left", fontsize=11)
    ax.set_ylabel("FDE (degrees)")
    ax.set_xlabel("Prediction horizon (minutes)")
    ax.set_ylim(0.0009, 0.0051)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))
    ax.grid(alpha=0.25)
    ax.annotate("sweet spot\n(CPA + attention\nboth contribute)",
                xy=(11.5, 0.00205), fontsize=8.5, color="#2E6B3E",
                ha="left", va="center")
    ax.annotate("", xy=(10, 0.001628), xytext=(15, 0.00225),
                arrowprops=dict(arrowstyle="-", color="#2E6B3E", lw=0.8))
 
    for ax in axes:
        ax.set_xticks(horizons)
        ax.set_xlim(3.5, 31.5)
 
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 0.945), fontsize=9.5)
 
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    png_path = os.path.join(OUTDIR, "diagram_4_3_ablation_cross_horizon.png")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    print(f"  saved {png_path}")
    plt.show()
    return png_path
 

# ΔΙΑΓΡΑΜΜΑ 4.4 — SMCHN training curves: NLL vs deterministic FDE
def diagram_4_4_smchn_variance_collapse():
    epochs = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 107, 120, 140, 160, 180, 200]
    nll_val = [-2.6, -8.5, -9.2, -9.8, -10.1, -10.3, -10.5, -10.6,
               -10.7, -10.75, -10.78, -10.80, -10.82, -10.84, -10.85,
               -10.86, -10.87]
    fde_val = [0.015, 0.009, 0.008, 0.0076, 0.0073, 0.0071, 0.0070,
               0.00690, 0.00685, 0.00682, 0.006799, 0.006699,
               0.006750, 0.006800, 0.006870, 0.006940, 0.007020]
 
    fig, ax1 = plt.subplots(figsize=(7.6, 5.0))
    ax2 = ax1.twinx()
 
    l1, = ax1.plot(epochs, nll_val, color=COL_LSTM, lw=2.0, marker="o", ms=4,
                    label="Validation NLL")
    l2, = ax2.plot(epochs, fde_val, color="#c0392b", lw=2.0, ls="--", marker="s", ms=4,
                    label="Deterministic FDE")
 
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Validation NLL (lower = better)", color=COL_LSTM)
    ax2.set_ylabel("Deterministic FDE (degrees, lower = better)", color="#c0392b")
    ax1.tick_params(axis="y", labelcolor=COL_LSTM)
    ax2.tick_params(axis="y", labelcolor="#c0392b")
 
    ax1.axvline(107, color="#555555", lw=1.3, ls=(0, (3, 2)))
    ax1.annotate("Best checkpoint\n(epoch 107)", xy=(107, -10.80), xytext=(122, -9.3),
                 fontsize=9.3, ha="left",
                 arrowprops=dict(arrowstyle="->", color="#555555", lw=0.9))
 
    ax1.grid(alpha=0.25)
    lines = [l1, l2]
    ax1.legend(lines, [ln.get_label() for ln in lines], loc="center right", frameon=False, fontsize=9.5)
    fig.tight_layout()
    return savefig(fig, "diagram_4_4_smchn_variance_collapse")
 
 
if __name__ == "__main__":
    print("Generating figures into:", OUTDIR)
    png_paths = []
    # Λίστα Εικόνων
    png_paths.append(eikona_2_2_cpa_geometry())
    # Λίστα Διαγραμμάτων (Διάγραμμα 4.1 δεν παράγεται εδώ)
    png_paths.append(diagram_4_4_smchn_variance_collapse())
    png_paths.append(diagram_4_2_ade_comparison())
    png_paths.append(diagram_4_3_ablation_cross_horizon())
    print("Done. Opening images...")
    for p in png_paths:
        open_image(p)