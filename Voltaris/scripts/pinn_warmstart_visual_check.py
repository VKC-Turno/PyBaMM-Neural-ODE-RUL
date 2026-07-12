"""Side-by-side visual comparison: baseline vs warm-started PINN on CALB.

Loads both checkpoints, plots each CALB cell twice (baseline pred vs warm pred),
so we can see whether the warm-start actually eliminated the visual "flat-then-
drop" artefact.
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, "/home/hj/Desktop/PINNs")

import numpy as np
import matplotlib.pyplot as plt
import torch

from Voltaris.sciml.data_combined         import load_combined, feature_normaliser
from Voltaris.sciml.physics               import estimate_k_sei_from_window, physics_trajectory
from Voltaris.sciml.train_joint           import (JointConfig, JointPINN,
                                                    predict_full_trajectory_joint)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
K = 50
OUT = Path("/home/hj/Desktop/PINNs/Voltaris/outputs/sciml_hybrid")

cells, meta = load_combined(include_synth=False)
mean, std   = feature_normaliser(cells)
mean_s      = mean[:-1]; std_s = std[:-1]
n_shared    = len(cells[0].x_health) - 1
n_norm_scale = float(max(c.n_total for c in cells))
cfg = JointConfig(K=K, n_norm_scale=n_norm_scale, p_init=0.5)

def build_and_load(ckpt):
    m = JointPINN(n_cells=len(cells), n_shared_features=n_shared,
                    embed_dim=8, hidden=128, n_layers=5,
                    feat_mean=mean_s, feat_std=std_s, p_init=0.5)
    m.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    return m.to(DEVICE).eval()

base = build_and_load(OUT / "baseline_K50.pt")
warm = build_and_load(OUT / "warmstart_K50.pt")

# CALB cells only
calb = [(i, c) for i, c in enumerate(cells) if meta[c.cell_id]["make"] == "CALB"]

fig, axs = plt.subplots(len(calb), 2, figsize=(11, 3.4*len(calb)))
if len(calb) == 1: axs = axs.reshape(1, 2)

for row, (i, cell) in enumerate(calb):
    n = cell.n_traj.numpy(); s = cell.soh_traj.numpy()
    first_cy = float(n[0]); k_end = first_cy + K
    mask_te = n > k_end

    pred_b = predict_full_trajectory_joint(base, cell, i, cfg, DEVICE).numpy()
    pred_w = predict_full_trajectory_joint(warm, cell, i, cfg, DEVICE).numpy()

    rmse_b = float(np.sqrt(np.mean((pred_b[mask_te]-s[mask_te])**2))) * 100
    rmse_w = float(np.sqrt(np.mean((pred_w[mask_te]-s[mask_te])**2))) * 100

    for col, (pred, tag, rmse) in enumerate([
        (pred_b, "BASELINE", rmse_b),
        (pred_w, "WARM-STARTED", rmse_w),
    ]):
        ax = axs[row, col]
        ymin = float(min(s.min(), pred.min())) * 100 - 1.5
        ymax = float(max(s.max(), pred.max())) * 100 + 1.5
        ax.set_ylim(ymin, ymax)
        ax.axvspan(k_end, n[-1], color="tab:blue", alpha=0.06)
        ax.axvspan(first_cy, k_end, color="tab:orange", alpha=0.12)
        ax.scatter(n, s*100, s=6, color="black", alpha=0.30, label="Measured")
        ax.plot(n, pred*100, color="tab:green", lw=2.0, label=f"{tag} {rmse:.2f} pp")
        ax.axvline(k_end, color="dimgray", ls="--", lw=0.7)
        ax.set_title(f"CALB {cell.cell_id.split('_')[-1]} — {tag} — {rmse:.2f} pp")
        ax.set_xlabel("Cycle"); ax.set_ylabel("SoH [%]")
        ax.grid(alpha=0.3); ax.legend(fontsize=9, loc="lower left")

fig.suptitle("CALB: baseline (softplus(NN)) vs warm-started (NN pre-trained on linear-fade)\n"
              "K=50 training window (orange), held-out (blue)", y=1.001, fontsize=13)
fig.tight_layout()
outfile = OUT / "calb_baseline_vs_warmstart.png"
fig.savefig(outfile, dpi=150, bbox_inches="tight")
print(f"Wrote {outfile}")
