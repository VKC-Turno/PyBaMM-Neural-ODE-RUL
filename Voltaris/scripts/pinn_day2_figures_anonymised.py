"""Regenerate v2 figures with anonymised manufacturer labels (MFR_A/B/C)
to match the existing pushed repo's anonymisation convention.

Mapping:
  CALB → MFR_A
  EVE  → MFR_B
  REPT → MFR_C
"""
from __future__ import annotations
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUT = Path("/home/hj/Desktop/PINNs/Voltaris/outputs/sciml_day2")
with open(OUT / "make_agnostic_trajectories.pkl", "rb") as f:
    all_trajs = pickle.load(f)

MAKE_MAP = {"CALB": "MFR_A", "REPT": "MFR_C", "EVE": "MFR_B"}


def _plot_one(ax, cid, t, make_disp):
    n, s = t["n"], t["s"]
    first_cy, k_end = t["first_cy"], t["k_end"]
    meas = s * 100
    pred = t["soh_pred"] * 100
    phys = t["soh_phys"] * 100

    ymin = float(min(meas.min(), pred.min())) - 1.5
    ymax = float(max(meas.max(), pred.max())) + 1.5
    ax.set_ylim(ymin, ymax)

    ax.axvspan(k_end, n[-1], color="tab:blue", alpha=0.06)
    ax.axvspan(first_cy, k_end, color="tab:orange", alpha=0.12)
    ax.scatter(n, meas, s=4, color="black", alpha=0.30, label="Measured")
    ax.plot(n, phys, color="tab:red", lw=1.4, ls="--",
             label=f"phys {t['rmse_phys']:.2f} pp")
    ax.plot(n, pred, color="tab:green", lw=2.0,
             label=f"PINN {t['rmse_pinn']:.2f} pp")
    ax.axvline(k_end, color="dimgray", ls="--", lw=0.7)

    if phys[-1] < ymin:
        ax.annotate(f"phys → {phys[-1]:.1f}%",
                     xy=(n[-1], ymin + 0.5), fontsize=8, color="tab:red",
                     ha="right")

    passer = "  ✓" if t["rmse_pinn"] < 3.0 else "  ✗"
    ax.set_title(f"{make_disp} cell {cid}{passer}", fontsize=10)
    ax.set_xlabel("Cycle"); ax.set_ylabel("SoH [%]")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower left")


def make_figure(real_name: str, disp_name: str, trajs: dict, ncols: int = 3):
    keys = sorted(trajs.keys())
    n = len(keys)
    nrows = (n + ncols - 1) // ncols
    fig, axs = plt.subplots(nrows, ncols, figsize=(5.3*ncols, 3.6*nrows))
    axs = np.array(axs).reshape(-1)
    for ax, cid in zip(axs, keys):
        _plot_one(ax, cid, trajs[cid], disp_name)
    for ax in axs[n:]:
        ax.set_visible(False)

    med_pinn = np.median([trajs[k]["rmse_pinn"] for k in keys])
    med_phys = np.median([trajs[k]["rmse_phys"] for k in keys])
    n_pass   = sum(trajs[k]["rmse_pinn"] < 3.0 for k in keys)
    fig.suptitle(f"{disp_name} — Path B joint PINN vs pure physics (K=50)\n"
                  f"PINN median {med_pinn:.2f} pp, phys median {med_phys:.2f} pp   ·   "
                  f"PINN under 3 pp: {n_pass}/{len(keys)}",
                  fontsize=12, y=1.005)
    fig.tight_layout()
    return fig


for real_name, trajs in [("CALB", all_trajs["CALB"]),
                          ("REPT", all_trajs["REPT"]),
                          ("EVE",  all_trajs["EVE"])]:
    disp = MAKE_MAP[real_name]
    fig = make_figure(real_name, disp, trajs)
    outfile = OUT / f"anonymised_{disp.lower()}_grid.png"
    fig.savefig(outfile, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {outfile.name}")

# Combined bar chart with anonymised labels
df = pd.read_csv(OUT / "make_agnostic_K50.csv")
df["make_disp"] = df["make"].map(MAKE_MAP)

fig, ax = plt.subplots(figsize=(12, 4.5))
xoff = {"MFR_A": 0, "MFR_C": 10, "MFR_B": 20}
colors = {"MFR_A": "tab:red", "MFR_C": "tab:blue", "MFR_B": "tab:purple"}
order = ["MFR_A", "MFR_C", "MFR_B"]
for m in order:
    d = df[df.make_disp == m].sort_values("cell_id").reset_index(drop=True)
    x = np.arange(len(d)) + xoff[m]
    ax.bar(x - 0.2, d.rmse_phys_pp, 0.4, color=colors[m], alpha=0.4,
            label=f"{m} phys")
    ax.bar(x + 0.2, d.rmse_pinn_pp, 0.4, color=colors[m], alpha=0.9,
            label=f"{m} PINN", edgecolor="black", linewidth=0.5)
ax.axhline(3.0, color="black", ls="--", lw=1.2, label="3 pp target")
ax.set_yscale("log")
ax.set_ylabel("Held-out RMSE [pp SoH]")
ax.set_title("Make-agnostic K=50: PINN cuts MFR_A error 10×, "
             "matches physics on MFR_B / MFR_C")

xticks, xlabels = [], []
for m in order:
    d = df[df.make_disp == m].sort_values("cell_id").reset_index(drop=True)
    for i, cid in enumerate(d.cell_id):
        xticks.append(i + xoff[m]); xlabels.append(f"{m[-1]}{cid}")
ax.set_xticks(xticks); ax.set_xticklabels(xlabels, rotation=45, fontsize=8)
ax.grid(alpha=0.3, axis="y", which="both")
ax.legend(fontsize=8, ncol=4, loc="upper right")
fig.tight_layout()
fig.savefig(OUT / "anonymised_bar.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Wrote anonymised_bar.png")

# Also anonymise CSV
df_anon = df.drop(columns=["make"]).rename(columns={"make_disp": "make"})
df_anon = df_anon[["make", "cell_id", "K", "n_total", "n_test",
                     "rmse_pinn_pp", "rmse_phys_pp"]]
df_anon.to_csv(OUT / "anonymised_make_agnostic_K50.csv", index=False)
print("Wrote anonymised_make_agnostic_K50.csv")
