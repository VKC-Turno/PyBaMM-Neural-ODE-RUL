"""Better trajectory plots — one figure per manufacturer, clipped y-axis
so measured + PINN detail is visible even when pure physics diverges.

Loads the pickled trajectories from the first run; no retraining needed.
"""
from __future__ import annotations
import sys, pickle
from pathlib import Path

sys.path.insert(0, "/home/hj/Desktop/PINNs")

import numpy as np
import matplotlib.pyplot as plt

OUT = Path("/home/hj/Desktop/PINNs/Voltaris/outputs/sciml_day2")
with open(OUT / "make_agnostic_trajectories.pkl", "rb") as f:
    all_trajs = pickle.load(f)


def _plot_one(ax, cid, t, make, ymin_pad=1.5, ymax_pad=1.5,
               clip_phys_visually=True):
    """One cell subplot. Clips y-axis around the *measured + PINN* range
    so the physics-off-a-cliff extrapolation doesn't dominate the frame.
    Physics line still shown but may leave the visible region."""
    n, s = t["n"], t["s"]
    first_cy, k_end = t["first_cy"], t["k_end"]
    soh_meas_pct = s * 100
    soh_pred_pct = t["soh_pred"] * 100
    soh_phys_pct = t["soh_phys"] * 100

    # y-limits based on measured + PINN only
    ymin = float(min(soh_meas_pct.min(), soh_pred_pct.min())) - ymin_pad
    ymax = float(max(soh_meas_pct.max(), soh_pred_pct.max())) + ymax_pad
    if clip_phys_visually:
        ax.set_ylim(ymin, ymax)

    ax.axvspan(k_end, n[-1], color="tab:blue", alpha=0.06)
    ax.axvspan(first_cy, k_end, color="tab:orange", alpha=0.12)
    ax.scatter(n, soh_meas_pct, s=4, color="black", alpha=0.30, label="Measured")
    ax.plot(n, soh_phys_pct, color="tab:red", lw=1.4, ls="--",
             label=f"phys {t['rmse_phys']:.2f} pp")
    ax.plot(n, soh_pred_pct, color="tab:green", lw=2.0,
             label=f"PINN {t['rmse_pinn']:.2f} pp")
    ax.axvline(k_end, color="dimgray", ls="--", lw=0.7)

    # Annotate if physics falls off-frame at end
    if soh_phys_pct[-1] < ymin:
        ax.annotate(f"phys → {soh_phys_pct[-1]:.1f}%",
                     xy=(n[-1], ymin + 0.5), fontsize=8, color="tab:red",
                     ha="right")

    passer = "  ✓" if t["rmse_pinn"] < 3.0 else "  ✗"
    ax.set_title(f"{make} cell {cid}{passer}", fontsize=10)
    ax.set_xlabel("Cycle"); ax.set_ylabel("SoH [%]")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower left")


def make_figure(name: str, trajs: dict, ncols: int = 3):
    """Grid of trajectories for one manufacturer, ncols wide."""
    keys = sorted(trajs.keys())
    n = len(keys)
    nrows = (n + ncols - 1) // ncols
    fig, axs = plt.subplots(nrows, ncols, figsize=(5.3*ncols, 3.6*nrows))
    axs = np.array(axs).reshape(-1)
    for ax, cid in zip(axs, keys):
        _plot_one(ax, cid, trajs[cid], name)
    for ax in axs[n:]:
        ax.set_visible(False)

    # Overall stats
    med_pinn = np.median([trajs[k]["rmse_pinn"] for k in keys])
    med_phys = np.median([trajs[k]["rmse_phys"] for k in keys])
    n_pass   = sum(trajs[k]["rmse_pinn"] < 3.0 for k in keys)
    fig.suptitle(f"{name} — Path B joint PINN vs pure physics (K=50)\n"
                  f"PINN median {med_pinn:.2f} pp, phys median {med_phys:.2f} pp   ·   "
                  f"PINN under 3 pp: {n_pass}/{len(keys)}",
                  fontsize=12, y=1.005)
    fig.tight_layout()
    return fig


for name, trajs in [("CALB", all_trajs["CALB"]),
                     ("REPT", all_trajs["REPT"]),
                     ("EVE",  all_trajs["EVE"])]:
    fig = make_figure(name, trajs)
    outfile = OUT / f"make_agnostic_{name.lower()}_grid.png"
    fig.savefig(outfile, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {outfile.name}")

# Also a combined summary bar chart (log-y so CALB physics doesn't dominate)
import pandas as pd
df = pd.read_csv(OUT / "make_agnostic_K50.csv")

fig, ax = plt.subplots(figsize=(12, 4.5))
xoff = {"CALB": 0, "REPT": 10, "EVE": 20}
colors = {"CALB": "tab:red", "REPT": "tab:blue", "EVE": "tab:purple"}
for m in ["CALB", "REPT", "EVE"]:
    d = df[df.make == m].sort_values("cell_id").reset_index(drop=True)
    x = np.arange(len(d)) + xoff[m]
    ax.bar(x - 0.2, d.rmse_phys_pp, 0.4, color=colors[m], alpha=0.4,
            label=f"{m} phys")
    ax.bar(x + 0.2, d.rmse_pinn_pp, 0.4, color=colors[m], alpha=0.9,
            label=f"{m} PINN", edgecolor="black", linewidth=0.5)
ax.axhline(3.0, color="black", ls="--", lw=1.2, label="3 pp target")
ax.set_yscale("log")
ax.set_ylabel("Held-out RMSE [pp SoH]")
ax.set_title("Make-agnostic K=50: PINN cuts CALB error 10×, "
             "matches physics on REPT / EVE")

# X-tick labels
xticks, xlabels = [], []
for m in ["CALB", "REPT", "EVE"]:
    d = df[df.make == m].sort_values("cell_id").reset_index(drop=True)
    for i, cid in enumerate(d.cell_id):
        xticks.append(i + xoff[m]); xlabels.append(f"{m[:1]}{cid}")
ax.set_xticks(xticks); ax.set_xticklabels(xlabels, rotation=45, fontsize=8)
ax.grid(alpha=0.3, axis="y", which="both")
ax.legend(fontsize=8, ncol=4, loc="upper right")
fig.tight_layout()
fig.savefig(OUT / "make_agnostic_bar.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Wrote make_agnostic_bar.png")
