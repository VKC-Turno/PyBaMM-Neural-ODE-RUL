"""CALB_old clean-cells rxn-lim hold-out experiment.

For each clean long-trajectory CALB_old cell that already fits pure rxn-lim
well, split the measured trajectory at K cycles, calibrate j_SEI on ONLY the
0..K window's slope, simulate the full horizon, and report RMSE on the held-out
K..end window. Answers the question:

    "How few cycles of characterisation before PyBaMM's rxn-lim SEI can
     predict the rest of the used cell's fade trajectory?"

Sweeps K ∈ {50, 100, 200, 400, 800} × 7 rxn-lim-friendly cells = 35 runs.

Reuses helpers from calb_clean_cells_rxnlim_sweep.py so the physics stack is
identical — only the calibration WINDOW changes.
"""
from __future__ import annotations

import json, sys, time, warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, "/home/hj/Desktop/PINNs")
sys.path.insert(0, str(Path(__file__).parent))
from calb_clean_cells_rxnlim_sweep import (  # noqa: E402
    load_measured, load_char_for, run_long_sim, calibrate_rxnlim,
    CACHE_DIR, PROTOCOL, TEMP_K, RXNLIM_OPTS, KEY_J,
)


# ─────────────────────────── config ───────────────────────────
# Only the cells that ALREADY passed the rxn-lim shape filter — no point
# hold-out-testing cells we know need joint SEI+LAM.
CELLS = [6, 7, 10, 14, 19, 20, 25]  # 24 & 30 excluded (batch-transition artifacts)
K_VALUES = [50, 100, 200, 400, 800]

OUT_DIR = Path("/home/hj/Desktop/PINNs/Voltaris/outputs/holdout_sweep")


# ─────────────────────────── helpers ───────────────────────────

def train_slope_pp_per_100cy(meas: pd.DataFrame, k: int) -> float:
    """Measured fade rate over cycles [first_meas_cycle .. first + k]."""
    lo = float(meas.global_cycle.iloc[0])
    hi = lo + k
    win = meas[(meas.global_cycle >= lo) & (meas.global_cycle <= hi)]
    if len(win) < 5:
        return float("nan")
    slope, _ = np.polyfit(win.global_cycle, win.soh_pct, 1)
    return float(slope * 100)


def rmse_window(meas: pd.DataFrame, sim_cy: np.ndarray, sim_anc: np.ndarray,
                lo: float, hi: float) -> tuple[float, int]:
    mask = (meas.global_cycle >= lo) & (meas.global_cycle <= hi)
    n = int(mask.sum())
    if n < 5:
        return float("nan"), n
    pred = np.interp(meas.global_cycle.values[mask], sim_cy, sim_anc)
    rmse = float(np.sqrt(np.mean((pred - meas.soh_pct.values[mask]) ** 2)))
    return rmse, n


# ─────────────────────────── main ───────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"=== hold-out sweep: {len(CELLS)} cells × {len(K_VALUES)} K values "
          f"= {len(CELLS)*len(K_VALUES)} runs ===\n")

    rows: list[dict] = []
    t_top = time.time()

    for cid in CELLS:
        tag = f"CALB_old_{cid}"
        meas = load_measured(cid)
        n_cycles = int(meas.global_cycle.max())
        pre_age = float(meas.soh.iloc[0])
        char, char_src = load_char_for(cid)
        first_cy = float(meas.global_cycle.iloc[0])
        start_y  = float(meas.soh_pct.iloc[0])

        for k in K_VALUES:
            t = time.time()
            k_end = first_cy + k

            # Skip K values that would leave <100 cycles of held-out data.
            held_out_cycles = n_cycles - k_end
            if held_out_cycles < 100:
                print(f"  [{tag} K={k}]  SKIP — only {held_out_cycles:.0f} cy held out")
                continue

            target = train_slope_pp_per_100cy(meas, k)
            print(f"\n--- {tag}  K={k}cy  target={target:+.4f} pp/100cy ---")

            # Calibrate j_SEI against TRAIN window's slope only.
            cal = calibrate_rxnlim(char, target, pre_age)
            print(f"  j_SEI={cal['value']:.3e}, achieved {cal['achieved']:+.4f}, "
                  f"{cal['n_evals']} evals, {cal['note']}")

            # Simulate the FULL horizon so we can score on train + held-out
            sim = run_long_sim(char, cal["value"], n_cycles, pre_age)
            sim_soh = sim.SOH.values * 100.0
            sim_cy  = sim.cycle_n.values.astype(float) + (first_cy - sim.cycle_n.values[0])
            sim_anc = sim_soh + (start_y - sim_soh[0])

            rmse_train, n_train = rmse_window(meas, sim_cy, sim_anc, first_cy, k_end)
            rmse_test,  n_test  = rmse_window(meas, sim_cy, sim_anc, k_end, n_cycles)
            end_meas = float(meas.soh_pct.iloc[-1])
            end_sim  = float(sim_anc[-1])

            rows.append({
                "cell": tag, "cell_id": cid, "char_source": char_src,
                "n_cycles_total": n_cycles,
                "K_train_cy": k,
                "n_train": n_train, "n_test": n_test,
                "target_slope_pp_per_100cy": target,
                "achieved_slope_pp_per_100cy": cal["achieved"],
                "j_SEI": cal["value"], "log10_j_SEI": cal["log10"],
                "rmse_train_pp": rmse_train,
                "rmse_test_pp":  rmse_test,
                "end_meas_pct":  end_meas,
                "end_sim_pct":   end_sim,
                "delta_end_pp":  end_sim - end_meas,
                "n_evals": cal["n_evals"], "note": cal["note"],
                "wall_time_s": time.time() - t,
            })
            print(f"  RMSE: train {rmse_train:.2f} pp (n={n_train})  |  "
                  f"TEST {rmse_test:.2f} pp (n={n_test})  |  "
                  f"end Δ={end_sim - end_meas:+.2f} pp")

    print(f"\n=== Total wall-time: {time.time()-t_top:.1f}s ===")
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "CALB_holdout_sweep_summary.csv", index=False)
    print(f"\nSummary CSV: {OUT_DIR / 'CALB_holdout_sweep_summary.csv'}")

    # ── Headline metric: median RMSE_test vs K ──
    if not df.empty:
        print("\nHeadline (median across cells):")
        print(df.groupby("K_train_cy").agg(
            n_cells=("cell_id", "nunique"),
            median_rmse_test_pp=("rmse_test_pp", "median"),
            n_under_3pp=("rmse_test_pp", lambda s: int((s < 3.0).sum())),
        ).to_string())

        # Plot: RMSE_test vs K, one line per cell + median band
        fig, ax = plt.subplots(figsize=(8, 5))
        for cid, g in df.groupby("cell_id"):
            ax.plot(g.K_train_cy, g.rmse_test_pp, "o-", alpha=0.6,
                    label=f"cell {cid}")
        med = df.groupby("K_train_cy").rmse_test_pp.median()
        ax.plot(med.index, med.values, "k-", lw=2.5, label="median")
        ax.axhline(3.0, color="r", ls="--", alpha=0.5, label="3 pp target")
        ax.set_xlabel("Calibration window K (cycles)")
        ax.set_ylabel("RMSE on held-out cycles [pp SoH]")
        ax.set_title("CALB rxn-lim hold-out: prediction error vs training length")
        ax.legend(fontsize=8, ncol=2)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUT_DIR / "holdout_rmse_vs_K.png", dpi=150)
        print(f"Figure: {OUT_DIR / 'holdout_rmse_vs_K.png'}")


if __name__ == "__main__":
    main()
