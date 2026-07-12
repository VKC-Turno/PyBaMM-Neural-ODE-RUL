"""Long-horizon REPT simulations.

For a hand-picked subset of REPT cells, runs a 150-cycle PyBaMM simulation
using the D_SEI value already calibrated by `rept_sweep.py`, and overlays
it against the full longterm CSV. Useful to check whether the calibrated
fade rate (from the short-window calibration) reproduces the full-test
trajectory, or diverges over the longer horizon.

For each cell:
  - Load char_b1 + the calibrated D_SEI from `REPT_<id>_aging_calibrated.json`
  - Run 150-cycle simulation (cached)
  - Save `REPT_<id>_long_sim_<N>cy.parquet`
  - Save `REPT_<id>_long_overlay.png` (anchored at cycle SKIP+1)

Also writes a combined `REPT_long_overlay_grid.png` (2×3 grid).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, "/home/hj/Desktop/PINNs")
from pybamm_tuning import (
    build_pybamm_parameters, load_characterization, SEI_ONLY_DFN_OPTIONS,
)
from pybamm_tuning.simulation import CyclingProtocol, Simulation


# ─────────────────────────── config ───────────────────────────
DEFAULT_CELLS = [78, 87, 7, 43, 74, 3]
PROTOCOL = CyclingProtocol(c_rate=0.25)
TEMP_K = 298.15
N_CYCLES_LONG = 150   # default; --n-cycles overrides at the CLI
SKIP_FIRST_N = 4

CACHE_DIR = Path("/home/hj/Desktop/PINNs/Voltaris/outputs/pybamm_cache")
OUT_DIR = Path("/home/hj/Desktop/PINNs/Voltaris/outputs/tuned_params")
LONGTERM_DIR = Path("/home/hj/Desktop/PINNs/Data/Longterm")


def _drop_outliers(series: pd.Series, k: float = 3.0, window: int = 5) -> pd.Series:
    """Hampel filter — matches the notebook's outlier rejection so the plots
    here line up with the cohort notebook."""
    if len(series) < window:
        return pd.Series([True] * len(series), index=series.index)
    med = series.rolling(window, center=True, min_periods=1).median()
    mad = (series - med).abs().rolling(window, center=True, min_periods=1).median()
    threshold = k * 1.4826 * mad.clip(lower=1e-9)
    return (series - med).abs() <= threshold


def per_cycle_soh(longterm_csv: Path, nominal_ah: float) -> pd.DataFrame:
    df = pd.read_csv(longterm_csv, usecols=["cycle_no", "step_name", "capacity_ah"])
    dchg = df[df["step_name"].astype(str).str.contains("DChg")]
    if dchg.empty:
        return pd.DataFrame(columns=["cycle_no", "dchg_cap_ah", "soh"])
    out = (dchg.groupby("cycle_no")["capacity_ah"]
               .agg(lambda s: float(s.abs().max()))
               .reset_index()
               .rename(columns={"capacity_ah": "dchg_cap_ah"}))
    out["soh"] = out["dchg_cap_ah"] / nominal_ah
    return out.sort_values("cycle_no").reset_index(drop=True)


def run_long_sim(cell_id: int) -> dict:
    cell_tag = f"REPT_{cell_id}"
    print(f"\n=== {cell_tag} — long sim ({N_CYCLES_LONG} cy) ===")
    t0 = time.time()

    # 1) Read the existing calibration result
    json_path = OUT_DIR / f"{cell_tag}_aging_calibrated.json"
    if not json_path.exists():
        raise FileNotFoundError(f"No prior calibration for {cell_tag} — "
                                 f"run rept_sweep.py first.")
    cal = json.loads(json_path.read_text())
    D_SEI = float(cal["calibrated_value"])
    pre_age_soh = float(cal.get("pre_age_to_soh", 1.0))
    print(f"  using D_SEI={D_SEI:.3e} m²/s, pre_age_to_soh={pre_age_soh:.3f}")

    # 2) Load char_b1 (the snapshot used for calibration)
    char = load_characterization(manufacturer="REPT", cell_id=str(cell_id), batch=1)

    # 3) Run the long sim (cache key includes n_cycles so this is a fresh entry)
    params = build_pybamm_parameters(
        char, base="Prada2013", temperature_K=TEMP_K,
        extra_overrides={"SEI solvent diffusivity [m2.s-1]": D_SEI},
        pre_age_to_soh=pre_age_soh,
    )
    sim = Simulation(params, protocol=PROTOCOL, cache_dir=CACHE_DIR,
                      dfn_options=SEI_ONLY_DFN_OPTIONS)
    sim_df = sim.run(n_cycles=N_CYCLES_LONG)
    long_parq = OUT_DIR / f"{cell_tag}_long_sim_{N_CYCLES_LONG}cy.parquet"
    sim_df.to_parquet(long_parq)
    print(f"  sim wrote {long_parq.name} (fresh={not getattr(sim, 'last_was_cached', False)}, "
          f"wall={time.time()-t0:.1f}s)")

    # 4) Load measured longterm CSV
    csv = LONGTERM_DIR / f"REPT_Longterm_cell_{int(cell_id):04d}.csv"
    meas = per_cycle_soh(csv, nominal_ah=char.nominal_capacity_ah) \
            if csv.exists() else pd.DataFrame()

    # 5) Generate the overlay PNG
    fig, ax = plt.subplots(figsize=(10, 5))
    if not meas.empty:
        m = meas[meas.cycle_no >= SKIP_FIRST_N + 1].copy()
        m["soh_pct"] = m["soh"] * 100.0
        keep = _drop_outliers(m["soh_pct"], k=3.0, window=5)
        clean = m[keep]
        dropped = m[~keep]
        ax.plot(clean.cycle_no, clean["soh_pct"], "o-", lw=0.8, ms=2.5,
                 color="#d62728",
                 label=f"measured (longterm CSV, {len(clean)} pts, {len(dropped)} dropped)")
        if not dropped.empty:
            ax.scatter(dropped.cycle_no, dropped["soh_pct"], s=30,
                        marker="x", color="grey", alpha=0.5, zorder=2)

        # Anchor sim to the first kept measured value
        sim_soh = sim_df["SOH"].values * 100.0
        sim_cy  = sim_df["cycle_n"].values.astype(float)
        first_cy = int(clean["cycle_no"].iloc[0])
        meas_start = float(clean["soh_pct"].iloc[0])
        sim_anchored = sim_soh + (meas_start - sim_soh[0])
        sim_cy_anchored = sim_cy + (first_cy - sim_cy[0])
        ax.plot(sim_cy_anchored, sim_anchored, "s--", lw=1.2, ms=3,
                 color="#1f77b4",
                 label=f"sim (D_SEI={D_SEI:.2e}, anchored at cycle {first_cy})")
    else:
        sim_soh = sim_df["SOH"].values * 100.0
        sim_cy  = sim_df["cycle_n"].values.astype(float)
        ax.plot(sim_cy, sim_soh, "s--", lw=1.2, ms=3,
                 color="#1f77b4", label=f"sim (D_SEI={D_SEI:.2e})")

    ax.axhline(80, ls=":", color="grey", alpha=0.5, label="EOL = 80 %")
    ax.set_xlabel("Cycle")
    ax.set_ylabel("SoH (%)")
    ax.set_title(f"{cell_tag} — long simulation ({N_CYCLES_LONG} cy) vs longterm CSV")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    png_path = OUT_DIR / f"{cell_tag}_long_overlay.png"
    fig.savefig(png_path, dpi=120)
    plt.close(fig)
    print(f"  overlay PNG: {png_path.name}")

    # Stats for return
    sim_end_soh = float(sim_df["SOH"].iloc[-1] * 100.0)
    meas_end_soh = float(meas["soh"].iloc[-1] * 100.0) if not meas.empty else float("nan")
    return {
        "cell": cell_tag,
        "D_SEI": D_SEI,
        "sim_soh_at_cyc150": sim_end_soh,
        "meas_soh_at_cyc150": meas_end_soh,
        "soh_disagree_pp": (sim_end_soh - meas_end_soh) if not np.isnan(meas_end_soh) else float("nan"),
        "wall_time_s": time.time() - t0,
    }


def build_combined_grid(cell_ids: list[int]) -> None:
    """2×3 grid of all 6 long overlays."""
    fig, axs = plt.subplots(2, 3, figsize=(16, 8))
    axs = axs.flatten()
    for i, cid in enumerate(cell_ids):
        ax = axs[i]
        cell_tag = f"REPT_{cid}"
        json_path = OUT_DIR / f"{cell_tag}_aging_calibrated.json"
        long_parq = OUT_DIR / f"{cell_tag}_long_sim_{N_CYCLES_LONG}cy.parquet"
        if not (json_path.exists() and long_parq.exists()):
            ax.text(0.5, 0.5, f"{cell_tag}\nmissing artifacts",
                     transform=ax.transAxes, ha="center", va="center",
                     fontsize=11, color="grey")
            ax.set_axis_off()
            continue

        cal = json.loads(json_path.read_text())
        D_SEI = float(cal["calibrated_value"])
        sim_df = pd.read_parquet(long_parq)
        char = load_characterization(manufacturer="REPT", cell_id=str(cid), batch=1)
        csv = LONGTERM_DIR / f"REPT_Longterm_cell_{int(cid):04d}.csv"
        meas = per_cycle_soh(csv, nominal_ah=char.nominal_capacity_ah) \
                if csv.exists() else pd.DataFrame()

        if not meas.empty:
            m = meas[meas.cycle_no >= SKIP_FIRST_N + 1].copy()
            m["soh_pct"] = m["soh"] * 100.0
            keep = _drop_outliers(m["soh_pct"], k=3.0, window=5)
            clean = m[keep]
            sim_soh = sim_df["SOH"].values * 100.0
            sim_cy  = sim_df["cycle_n"].values.astype(float)
            first_cy = int(clean["cycle_no"].iloc[0])
            meas_start = float(clean["soh_pct"].iloc[0])
            sim_anchored = sim_soh + (meas_start - sim_soh[0])
            sim_cy_anchored = sim_cy + (first_cy - sim_cy[0])
            ax.plot(clean.cycle_no, clean["soh_pct"], "o-", lw=0.7, ms=2,
                     color="#d62728", label="measured")
            ax.plot(sim_cy_anchored, sim_anchored, "s--", lw=1.0, ms=2.5,
                     color="#1f77b4", label="sim (anchored)")

        rel = cal.get("relative_error_pct", float("nan"))
        ax.set_title(f"{cell_tag} — D_SEI={D_SEI:.1e}, err={rel:.1f}%", fontsize=10)
        ax.set_xlabel("Cycle"); ax.set_ylabel("SoH (%)")
        ax.legend(loc="best", fontsize=8)
        ax.grid(alpha=0.3)

    fig.suptitle(f"REPT — long simulations ({N_CYCLES_LONG} cy) vs longterm CSV",
                  fontsize=13, y=1.01)
    fig.tight_layout()
    grid_path = OUT_DIR / "REPT_long_overlay_grid.png"
    fig.savefig(grid_path, dpi=120)
    plt.close(fig)
    print(f"\nCombined grid: {grid_path}")


def main() -> None:
    global N_CYCLES_LONG
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", type=int, nargs="+", default=DEFAULT_CELLS,
                     help="REPT cell IDs to run long sims for.")
    ap.add_argument("--n-cycles", type=int, default=N_CYCLES_LONG,
                     help="Number of cycles to simulate (default: 150).")
    args = ap.parse_args()
    # Override the module constant so run_long_sim + build_combined_grid both see it
    N_CYCLES_LONG = int(args.n_cycles)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    for cid in args.cells:
        try:
            results.append(run_long_sim(cid))
        except Exception as e:
            print(f"  FAIL REPT_{cid}: {type(e).__name__}: {e}")

    build_combined_grid(args.cells)

    print("\n=== Long-sim summary ===")
    print(f"{'cell':<10} {'D_SEI':<11} {'sim@150':<10} {'meas@150':<10} {'Δ pp':<8} {'wall':<6}")
    for r in results:
        print(f"  {r['cell']:<8} {r['D_SEI']:<11.2e} "
              f"{r['sim_soh_at_cyc150']:<10.2f} "
              f"{r['meas_soh_at_cyc150']:<10.2f} "
              f"{r['soh_disagree_pp']:<+8.2f} "
              f"{r['wall_time_s']:<6.1f}s")


if __name__ == "__main__":
    main()
