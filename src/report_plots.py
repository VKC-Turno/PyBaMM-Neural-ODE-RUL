"""
src/report_plots.py
-------------------
Generate the report-quality figures referenced from
data/processed/lab_test_audit_report.md (and its PDF rendering).

Outputs to outputs/results/ — kept alongside the other Phase-1 plots.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "matplotlib"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data_loader import load_test


OUT = Path("outputs/results")
OUT.mkdir(parents=True, exist_ok=True)
CELLS_ALL = [f"{i:04d}" for i in range(1, 9)]
CELLS_COHORT = ["0005", "0006", "0007", "0008"]


# ── Constant-power discharge: voltage vs delivered Ah, one line per cell ──
def constant_power_curves():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for cid in CELLS_ALL:
        try:
            df = load_test("ConstantPower", cell_id=cid).sort_values("time")
        except Exception:
            continue
        disc = df[df["current"] < 0]
        if disc.empty:
            continue
        Q = disc["capacity"].abs().to_numpy(dtype=float)
        V = disc["voltage"].to_numpy(dtype=float)
        I = disc["current"].to_numpy(dtype=float)
        P = (-V * I)
        axes[0].plot(Q, V, lw=1.0, label=cid)
        axes[1].plot(Q, P, lw=1.0, label=cid)
    axes[0].set(xlabel="Capacity delivered [Ah]", ylabel="Voltage [V]",
                title="Constant-power discharge: V(Q)")
    axes[0].axhline(2.5, ls="--", color="red", alpha=0.5, label="cut-off")
    axes[0].legend(fontsize=7, ncol=2)
    axes[1].set(xlabel="Capacity delivered [Ah]", ylabel="Power [W]",
                title="Constant-power discharge: P(Q)")
    axes[1].legend(fontsize=7, ncol=2)
    fig.suptitle("Constant-power discharge (25 °C, 8 cells)")
    fig.tight_layout()
    fig.savefig(OUT / "constantpower_curves.png", dpi=150)
    plt.close(fig)
    print("wrote constantpower_curves.png")


# ── Rate capability: capacity vs C-rate, dots per cell ──
def rate_capability():
    rows = []
    for cid in CELLS_ALL:
        try:
            df = load_test("RateCapability", cell_id=cid)
        except Exception:
            continue
        disc = df[df["current"] < 0]
        per_cy = disc.groupby("cycle").agg(
            Q=("capacity", lambda x: float(abs(x.min()))),
            I=("current", lambda x: float(abs(x.mean()))),
        ).reset_index()
        for _, r in per_cy.iterrows():
            rows.append({"cell_id": cid, "C_rate": r["I"] / 105.0, "Q_Ah": r["Q"]})
    if not rows:
        return
    d = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for cid, g in d.groupby("cell_id"):
        ax.plot(g["C_rate"], g["Q_Ah"], marker="o", lw=1.0, label=cid, alpha=0.8)
    ax.set(xlabel="Discharge C-rate", ylabel="Capacity [Ah]",
           title="Rate capability — capacity vs C-rate (25 °C)")
    ax.legend(title="cell", fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "rate_capability.png", dpi=150)
    plt.close(fig)
    print("wrote rate_capability.png")


# ── Peak-power: voltage transient through the pulse, one line per cohort cell ──
def peak_power():
    fig, ax = plt.subplots(figsize=(7, 4))
    have = False
    for cid in CELLS_COHORT:
        try:
            df = load_test("PeakPower", cell_id=cid).sort_values("time").reset_index(drop=True)
        except Exception:
            continue
        disc = df[df["current"] < 0]
        if disc.empty:
            continue
        # Align to start of pulse
        t0 = float(disc["time"].iloc[0])
        t_rel = disc["time"].to_numpy(dtype=float) - t0
        ax.plot(t_rel, disc["voltage"].to_numpy(), lw=1.0, label=cid)
        have = True
    if not have:
        plt.close(fig)
        return
    ax.set(xlabel="Time since pulse start [s]", ylabel="Voltage [V]",
           title="Peak-power pulse — voltage transient, cohort cells")
    ax.legend(title="cell", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "peakpower_pulse.png", dpi=150)
    plt.close(fig)
    print("wrote peakpower_pulse.png")


# ── HPPC: per-cell R0 distribution (box plot) ──
def hppc_r0_box():
    p = Path("data/processed/dcir_hppc_pulses.parquet")
    if not p.exists():
        return
    d = pd.read_parquet(p)
    d = d[d["direction"] == "discharge"].copy()
    d["R0_mOhm"] = d["R0_Ohm"] * 1000
    cells = sorted(d["cell_id"].unique())
    data = [d.loc[d["cell_id"] == c, "R0_mOhm"].values for c in cells]
    fig, ax = plt.subplots(figsize=(6, 4))
    bp = ax.boxplot(data, labels=cells, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("lightsteelblue")
    ax.axhline(1.8, ls="--", color="red", alpha=0.6, label="EVE LF105 spec ≤ 1.8 mΩ")
    ax.set(xlabel="cell", ylabel=r"R$_0$ [mΩ]",
           title="HPPC + DCIR discharge-pulse R$_0$ per cell (SOC 0.97–1.00)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "hppc_R0_box.png", dpi=150)
    plt.close(fig)
    print("wrote hppc_R0_box.png")


# ── Self-discharge: cohort-level OCV decay vs hours (already exists, but
# regenerate with explicit cohort labels in case of staleness) ──
def selfdischarge_cohort():
    # The Phase-1 script already wrote outputs/results/selfdischarge_decay.png
    # so this is a no-op unless missing.
    p = OUT / "selfdischarge_decay.png"
    if p.exists():
        return
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for cid in CELLS_COHORT:
        try:
            df = load_test("SelfDischarge", cell_id=cid).sort_values("time")
            # Same long-rest extraction logic as src/param_id/sei_selfdisc.py
            df["seg"] = (df["step_name"] != df["step_name"].shift()).cumsum()
            seg_long = df.groupby("seg").agg(
                step_name=("step_name", "first"),
                dur=("time", lambda s: (s.max() - s.min()) / 3600.0),
            ).reset_index()
            long_seg_id = seg_long[(seg_long["step_name"] == "Rest")
                                     & (seg_long["dur"] > 24)].sort_values("dur",
                                     ascending=False).iloc[0]["seg"]
            rest = df[df["seg"] == long_seg_id]
            t_h = (rest["time"].values - rest["time"].values[0]) / 3600.0
            ax.plot(t_h, rest["voltage"].values, lw=1.0, label=cid)
        except Exception:
            continue
    ax.set(xlabel="Hours since top-of-charge", ylabel="OCV [V]",
           title="Self-discharge: OCV decay over the 172.5 h top-of-charge rest")
    ax.legend(title="cell", fontsize=8)
    fig.tight_layout()
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print("wrote selfdischarge_decay.png")


if __name__ == "__main__":
    constant_power_curves()
    rate_capability()
    peak_power()
    hppc_r0_box()
    selfdischarge_cohort()
    print("Done — plots under outputs/results/")
