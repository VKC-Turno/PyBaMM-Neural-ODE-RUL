"""Figure generation for the data-quality review."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent))
from extract import (BOUNDS, MAKE_COLOR, MAKES, OUT_DIR, ExtractionResult,
                       run_extraction)


plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# --------------------------------------------------------------------------- #
def plot_soh_by_make(res: ExtractionResult, out: Path) -> None:
    fig, axes = plt.subplots(1, len(MAKES), figsize=(15, 5.0), sharey=True)
    titles = {"CALB": "CALB batch 2", "REPT": "REPT", "EVE": "EVE"}
    for ax, make in zip(axes, MAKES):
        traces = res.soh_traces.get(make, {})
        color = MAKE_COLOR[make]
        max_cycle = 0
        # per-cell only; median removed per project decision
        for cid, df in traces.items():
            ax.plot(df["cycle"], df["soh"], color=color, alpha=0.55, lw=1.0)
            max_cycle = max(max_cycle, int(df["cycle"].max()))
        # For CALB: draw the batch-1 -> batch-2 seam as a vertical dashed line
        if make == "CALB" and traces:
            seams = []
            for df in traces.values():
                if "batch" not in df.columns:
                    continue
                b = df["batch"].to_numpy()
                # last row of batch=1 = seam boundary
                idx = np.where(b == 1)[0]
                if idx.size:
                    seams.append(int(df["cycle"].iloc[idx[-1]]))
            if seams:
                seam_cy = int(np.median(seams))
                ax.axvline(seam_cy, color="0.35", ls="--", lw=1.0,
                           label=f"batch 1 -> 2 seam (cy {seam_cy})")
        ax.axhline(0.80, color="0.4", ls=":", lw=0.8, label="EoL (SoH 0.80)")
        ax.set_title(f"{titles[make]} - Longterm SoH (n={len(traces)})")
        ax.set_xlabel("Cycle number")
        ax.set_ylim(0.0, 1.05)
        ax.set_xlim(0, max(max_cycle, 1))
        ax.grid(True, alpha=0.25)
        ax.legend(loc="lower left", fontsize=8)
    axes[0].set_ylabel("SoH = discharge_cap_ah / nameplate (max_cap)")
    fig.suptitle("Longterm SoH vs cycle - overlay per cohort",
                 fontsize=13, y=1.02)
    # caption for the CALB context (Athena batch=1 is NOT first life)
    fig.text(0.02, -0.02,
             "CALB: Athena batch=1 does NOT correspond to first-life cycling; "
             "these cells were cycled >=1000 cy in a prior stage not present "
             "in this dataset. True SoH at Athena batch=1 start is therefore "
             "already below 1.0 (nameplate = 72 Ah). EVE / REPT SoH come "
             "from soh/data/canonical/*.parquet.",
             ha="left", va="top", fontsize=8, color="0.25", wrap=True)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_ocv_curves(res: ExtractionResult, out: Path) -> None:
    fig, axes = plt.subplots(1, len(MAKES), figsize=(15, 4.6), sharey=True)
    for ax, make in zip(axes, MAKES):
        curves = res.ocv_curves.get(make, {})
        color = MAKE_COLOR[make]
        for cid, curve in curves.items():
            ax.plot(curve["soc_pct"], curve["v"], color=color,
                    alpha=0.35, lw=0.9)
        ax.set_title(f"{make} - OCV_SOC discharge branch (n={len(curves)})")
        ax.set_xlabel("SoC (% of |Q_max|, discharge to 0)")
        ax.set_xlim(100, 0)  # invert: discharge progresses left-to-right
        ax.set_ylim(2.4, 3.55)
        ax.grid(True, alpha=0.25)
        # LFP plateau band
        ax.axhspan(BOUNDS["V_plateau"][0], BOUNDS["V_plateau"][1],
                   color="0.85", alpha=0.35, label="LFP plateau (3.15-3.35 V)")
        ax.legend(loc="lower left", fontsize=8)
    axes[0].set_ylabel("Terminal voltage (V)")
    fig.suptitle("OCV_SOC discharge branches - inter-cell stability check",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _strip_box(ax, data_by_make: dict[str, np.ndarray], ylabel: str,
               bounds: tuple[float, float] | None = None,
               title: str = "") -> None:
    positions = np.arange(len(data_by_make))
    labels = list(data_by_make.keys())
    values = [np.asarray(v)[np.isfinite(v)] for v in data_by_make.values()]
    ax.boxplot(values, positions=positions, widths=0.55,
               patch_artist=True,
               boxprops=dict(facecolor="0.92", edgecolor="0.35"),
               medianprops=dict(color="0.15", lw=1.6),
               whiskerprops=dict(color="0.35"),
               capprops=dict(color="0.35"),
               flierprops=dict(marker="", ms=0),
               showfliers=False)
    for pos, make in zip(positions, labels):
        vals = values[pos]
        if vals.size == 0:
            continue
        jitter = np.random.default_rng(pos).uniform(-0.13, 0.13, size=vals.size)
        ax.scatter(pos + jitter, vals, color=MAKE_COLOR[make],
                   alpha=0.75, s=22, edgecolors="none")
    if bounds is not None:
        lo, hi = bounds
        if np.isfinite(lo):
            ax.axhline(lo, ls="--", color="0.4", lw=0.9)
        if np.isfinite(hi):
            ax.axhline(hi, ls="--", color="0.4", lw=0.9)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.grid(True, axis="y", alpha=0.25)


def plot_scalar_grid(res: ExtractionResult, out: Path) -> None:
    s = res.scalars
    panels: list[tuple[str, str, tuple | None, str]] = [
        ("V_top", "V_top (V)", BOUNDS["V_top"],
         "OCV discharge - 99th pct V"),
        ("V_bottom", "V_bottom (V)", BOUNDS["V_bottom"],
         "OCV discharge - 1st pct V"),
        ("V_plateau", "V_plateau (V)", BOUNDS["V_plateau"],
         "OCV discharge - LFP plateau (30-70 %)"),
        ("dV_per_sqrt_t", "dV/dsqrt(t) (V/sqrt(s))",
         BOUNDS["dV_per_sqrt_t"], "GITT diffusion slope"),
        ("DCIR_R0_mOhm", "DCIR R0 (mOhm)", BOUNDS["R0_mOhm"],
         "DCIR ohmic resistance"),
        ("HPPC_R0_mOhm", "HPPC R0 (mOhm)", BOUNDS["R0_mOhm"],
         "HPPC ohmic resistance"),
        ("HPPC_R1_mOhm", "HPPC R1 (mOhm)", None,
         "HPPC charge-transfer resistance"),
        ("capacity_Ah", "Capacity (Ah)", None,
         "RPT largest-discharge capacity"),
        ("coulombic_efficiency_pct", "CE (%)", BOUNDS["coulombic_efficiency_pct"],
         "RPT coulombic efficiency"),
        ("fade_pct", "Fade (%)", None, "Longterm SoH fade first-last"),
        ("monotone_frac", "Monotone frac", BOUNDS["monotone_frac"],
         "Longterm SoH monotonicity"),
        ("n_cycles", "n_cycles", None, "Longterm cycles logged"),
    ]
    ncols = 4
    nrows = int(np.ceil(len(panels) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 3.2 * nrows))
    axes = np.array(axes).ravel()
    for ax, (col, ylabel, bounds, title) in zip(axes, panels):
        data_by_make: dict[str, np.ndarray] = {}
        for make in MAKES:
            sub = s[s["make"] == make]
            if col in sub.columns:
                data_by_make[make] = sub[col].to_numpy(dtype=float)
            else:
                data_by_make[make] = np.array([])
        _strip_box(ax, data_by_make, ylabel, bounds, title)
    for ax in axes[len(panels):]:
        ax.axis("off")
    fig.suptitle("Per-cell characterization scalars - "
                 "dashed lines are physical-bound flags",
                 fontsize=13, y=1.005)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_hppc_pulse_resistances(res: ExtractionResult, out: Path) -> None:
    s = res.scalars
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    for ax, col, title, bnd in (
        (axes[0], "HPPC_R0_mOhm", "HPPC R0 (ohmic, ~10 ms after pulse)",
         BOUNDS["R0_mOhm"]),
        (axes[1], "HPPC_R1_mOhm", "HPPC R1 (charge-transfer, tau fit)", None),
    ):
        data = {make: s[s["make"] == make][col].to_numpy(dtype=float)
                if col in s.columns else np.array([])
                for make in MAKES}
        _strip_box(ax, data, f"{col.split('_')[1]} (mOhm)", bnd, title)
    fig.suptitle("HPPC pulse-derived resistances", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
def main() -> ExtractionResult:
    res = run_extraction()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    res.scalars.to_csv(OUT_DIR / "characterization_scalars.csv", index=False)
    plot_soh_by_make(res, OUT_DIR / "soh_vs_cycle_by_make.png")
    plot_ocv_curves(res, OUT_DIR / "ocv_curves_overlay.png")
    plot_scalar_grid(res, OUT_DIR / "characterization_parameter_distributions.png")
    plot_hppc_pulse_resistances(res, OUT_DIR / "hppc_pulse_resistances.png")
    return res


if __name__ == "__main__":
    main()
