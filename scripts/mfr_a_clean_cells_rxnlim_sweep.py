"""MFR_A clean-cells reaction-limited SEI sweep.

For each clean long-trajectory MFR_A cell, calibrates pure rxn-lim SEI
against the whole-trajectory smoothed slope and runs the full measured-horizon
sim. Produces per-cell RMSE + classification, a 3×3 grid of overlays, and a
summary CSV that the routing rule reads from.

Clean cells (from notebook 08 follow-up survey, ≥1000 cy, noise std < 0.8 pp,
no sensor zeros):
    1, 2, 3, 4, 5, 6, 7, 8, 9
Cell 25 is NOT in the char workbook → use cohort-median MFR_A char.
All others use cell-specific char.
"""
from __future__ import annotations

import json, sys, time, warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

REPO_ROOT = Path(__file__).resolve().parents[1]
from pybamm_tuning import build_pybamm_parameters, load_characterization, list_available_cells
from pybamm_tuning.simulation import CyclingProtocol, Simulation


# ─────────────────────────── config ───────────────────────────
CELLS = [1, 2, 3, 4, 5, 6, 7, 8, 9]
PROTOCOL = CyclingProtocol(c_rate=0.25)
TEMP_K   = 298.15
N_CYCLES_CALIBRATION = 10
SKIP_FIRST_N = 1

CACHE_DIR = REPO_ROOT / "outputs/pybamm_cache"
OUT_DIR   = REPO_ROOT / "outputs/tuned_params"
CANON_PQ  = REPO_ROOT / "data/canonical/mfr_a.parquet"

RXNLIM_OPTS = {
    "SEI": "reaction limited",
    "SEI porosity change": "true",
    "lithium plating": "none",
    "loss of active material": "none",
}
KEY_J = "SEI reaction exchange current density [A.m-2]"


def _drop_outliers(series, k=3.0, window=5):
    if len(series) < window:
        return pd.Series([True] * len(series), index=series.index)
    med = series.rolling(window, center=True, min_periods=1).median()
    mad = (series - med).abs().rolling(window, center=True, min_periods=1).median()
    return (series - med).abs() <= k * 1.4826 * mad.clip(lower=1e-9)


def load_measured(cid: int) -> pd.DataFrame:
    df = pd.read_parquet(CANON_PQ)
    c = df[df.cell_id.astype(str).str.zfill(4) == f"{cid:04d}"].sort_values("global_cycle")
    c = c[(c.global_cycle >= SKIP_FIRST_N + 1) & (c.soh > 0.05)].copy()
    c["soh_pct"] = c.soh * 100.0
    keep = _drop_outliers(c["soh_pct"], k=3.0, window=5)
    c = c[keep].copy()
    c["smoothed"] = c.soh_pct.rolling(50, center=True, min_periods=10).median()
    return c


def classify_shape(meas: pd.DataFrame) -> str:
    sm = meas.dropna(subset=["smoothed"])
    if len(sm) < 100:
        return "?"
    def slope(lo, hi):
        s = sm[sm.global_cycle.between(lo, hi)]
        if len(s) < 5: return float("nan")
        return float(np.polyfit(s.global_cycle, s.smoothed, 1)[0] * 100)
    e, m, l = slope(2, 200), slope(201, 800), slope(801, 2000)
    if any(np.isnan([e, m, l])):
        return "?"
    # Three-regime: mid much steeper than early/late
    if m < (e - 0.3) and l > (m + 0.5):
        return "three-regime"
    if abs(e - m) < 0.3 and abs(m - l) < 0.3:
        return "near-linear"
    if l < e - 0.3:
        return "accelerating"
    if l > e + 0.3:
        return "decelerating"
    return "mixed"


def calibrate_rxnlim(char, target, pre_age):
    n_evals = 0
    def f(log10_j):
        nonlocal n_evals
        params = build_pybamm_parameters(
            char, base="Prada2013", temperature_K=TEMP_K,
            extra_overrides={KEY_J: 10**log10_j}, pre_age_to_soh=pre_age)
        sim = Simulation(params, protocol=PROTOCOL, cache_dir=CACHE_DIR,
                          dfn_options=RXNLIM_OPTS)
        df = sim.run(n_cycles=N_CYCLES_CALIBRATION)
        n_evals += 1
        soh = df.SOH.to_numpy(dtype=float) * 100
        cy = df.cycle_n.to_numpy(dtype=float)
        s, _ = np.polyfit(cy[1:], soh[1:], 1)
        return float(s * 100)

    lo, hi = -10.0, -6.0
    sl_lo, sl_hi = f(lo), f(hi)
    if (sl_lo - target) * (sl_hi - target) > 0:
        best = lo if abs(sl_lo - target) < abs(sl_hi - target) else hi
        return {"log10": best, "value": 10**best,
                "achieved": sl_lo if best == lo else sl_hi,
                "n_evals": n_evals, "note": "bracket-edge"}
    for _ in range(12):
        mid = 0.5 * (lo + hi)
        sl_mid = f(mid)
        if abs(sl_mid - target) <= 0.20 * max(abs(target), 0.05):
            return {"log10": mid, "value": 10**mid, "achieved": sl_mid,
                    "n_evals": n_evals, "note": "converged"}
        if (sl_lo - target) * (sl_mid - target) < 0:
            hi, sl_hi = mid, sl_mid
        else:
            lo, sl_lo = mid, sl_mid
    return {"log10": mid, "value": 10**mid, "achieved": sl_mid,
            "n_evals": n_evals, "note": "max_iter"}


def run_long_sim(char, j_sei, n_cycles, pre_age):
    params = build_pybamm_parameters(
        char, base="Prada2013", temperature_K=TEMP_K,
        extra_overrides={KEY_J: j_sei}, pre_age_to_soh=pre_age)
    sim = Simulation(params, protocol=PROTOCOL, cache_dir=CACHE_DIR,
                      dfn_options=RXNLIM_OPTS)
    return sim.run(n_cycles=n_cycles)


def load_char_for(cid: int):
    cells = list_available_cells()
    sub = cells[(cells.cohort == "MFR_A") & (cells.cell_id.astype(str) == str(cid))]
    if not sub.empty:
        return load_characterization(cohort="MFR_A", cell_id=str(cid)), "cell-specific"
    return load_characterization(cohort="MFR_A", aggregate=True), "cohort-median"


# ─────────────────────────── main ───────────────────────────
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"=== MFR_A clean-cells rxn-lim sweep ({len(CELLS)} cells) ===\n")
    results = []
    t_top = time.time()

    for cid in CELLS:
        tag = f"MFR_A_{cid}"
        print(f"\n--- {tag} ---")
        t = time.time()
        meas = load_measured(cid)
        n_cycles = int(meas.global_cycle.max())
        pre_age = float(meas.soh.iloc[0])
        target = float(np.polyfit(meas.global_cycle, meas.soh_pct, 1)[0] * 100)
        shape = classify_shape(meas)
        print(f"  n_cy={n_cycles}, pre_age={pre_age:.3f}, target={target:+.4f} pp/100cy, shape={shape}")

        char, char_src = load_char_for(cid)
        print(f"  char: {char_src} ({char.cell_id})")

        cal = calibrate_rxnlim(char, target, pre_age)
        print(f"  j_SEI={cal['value']:.3e}, achieved {cal['achieved']:+.4f}, {cal['n_evals']} evals, "
              f"{cal['note']}")

        t_sim = time.time()
        sim = run_long_sim(char, cal["value"], n_cycles, pre_age)
        sim.to_parquet(OUT_DIR / f"{tag}_rxnlim_sweep_sim_{n_cycles}cy.parquet")
        print(f"  long sim: {n_cycles}cy in {time.time()-t_sim:.1f}s")

        # Anchor + RMSE
        first_cy = float(meas.global_cycle.iloc[0])
        start_y  = float(meas.soh_pct.iloc[0])
        sim_soh = sim.SOH.values * 100.0
        sim_cy  = sim.cycle_n.values.astype(float) + (first_cy - sim.cycle_n.values[0])
        sim_anc = sim_soh + (start_y - sim_soh[0])
        rmse = float(np.sqrt(np.mean(
            (np.interp(meas.global_cycle.values, sim_cy, sim_anc) - meas.soh_pct.values)**2)))

        # Per-window RMSE
        def win_rmse(lo, hi):
            mask = (meas.global_cycle >= lo) & (meas.global_cycle <= hi)
            if mask.sum() < 5: return float("nan")
            return float(np.sqrt(np.mean((
                np.interp(meas.global_cycle.values[mask], sim_cy, sim_anc) -
                meas.soh_pct.values[mask])**2)))

        end_meas = float(meas.soh_pct.iloc[-1])
        end_sim  = float(sim_anc[-1])

        result = {
            "cell": tag, "cell_id": cid, "shape": shape,
            "char_source": char_src,
            "n_cycles": n_cycles, "pre_age_to_soh": pre_age,
            "soh_start_meas": float(meas.soh_pct.iloc[0]),
            "soh_end_meas":   end_meas,
            "soh_end_sim":    end_sim,
            "delta_end_pp":   end_sim - end_meas,
            "fade_pp_smoothed": float(meas.smoothed.dropna().iloc[0] - meas.smoothed.dropna().iloc[-1]),
            "target_slope_pp_per_100cy": target,
            "j_SEI": cal["value"], "log10_j_SEI": cal["log10"],
            "achieved_slope_pp_per_100cy": cal["achieved"],
            "rel_err_pct": abs(cal["achieved"] - target) / max(abs(target), 1e-9) * 100,
            "rmse_full_pp":  rmse,
            "rmse_early_pp": win_rmse(2, 200),
            "rmse_mid_pp":   win_rmse(201, 800),
            "rmse_late_pp":  win_rmse(801, 2000),
            "n_evals": cal["n_evals"], "note": cal["note"],
            "wall_time_s": time.time() - t,
        }
        results.append(result)
        (OUT_DIR / f"{tag}_rxnlim_sweep_calibrated.json").write_text(
            json.dumps(result, indent=2, default=str))
        print(f"  RMSE: full {rmse:.2f} | early {result['rmse_early_pp']:.2f} | "
              f"mid {result['rmse_mid_pp']:.2f} | late {result['rmse_late_pp']:.2f} pp  "
              f"end Δ={end_sim - end_meas:+.2f} pp")

    print(f"\n=== Total wall-time: {time.time()-t_top:.1f}s ===")
    df = pd.DataFrame(results)
    df.to_csv(OUT_DIR / "MFR_A_clean_rxnlim_sweep_summary.csv", index=False)
    print(f"\nSummary CSV: MFR_A_clean_rxnlim_sweep_summary.csv")

    # ── 3×3 grid plot ──
    fig, axs = plt.subplots(3, 3, figsize=(18, 12))
    for i, cid in enumerate(CELLS):
        ax = axs[i // 3, i % 3]
        tag = f"MFR_A_{cid}"
        meas = load_measured(cid)
        n_cycles = int(meas.global_cycle.max())
        sim = pd.read_parquet(OUT_DIR / f"{tag}_rxnlim_sweep_sim_{n_cycles}cy.parquet")
        cal = json.loads((OUT_DIR / f"{tag}_rxnlim_sweep_calibrated.json").read_text())

        first_cy = float(meas.global_cycle.iloc[0])
        start_y  = float(meas.soh_pct.iloc[0])
        sim_soh = sim.SOH.values * 100.0
        sim_cy  = sim.cycle_n.values.astype(float) + (first_cy - sim.cycle_n.values[0])
        sim_anc = sim_soh + (start_y - sim_soh[0])

        ax.plot(meas.global_cycle, meas.soh_pct, "o", ms=1.2, color="#d62728", alpha=0.6,
                 label="measured")
        ax.plot(sim_cy, sim_anc, "-", lw=1.6, color="#2ca02c",
                 label=f"sim rxn-lim (j={cal['j_SEI']:.1e})")
        ax.axvline(200, ls=":", color="grey", alpha=0.3)
        ax.axvline(800, ls=":", color="grey", alpha=0.3)
        ax.set_title(f"{tag} — {cal['shape']}, RMSE {cal['rmse_full_pp']:.2f} pp",
                      fontsize=10)
        ax.set_xlabel("Cycle"); ax.set_ylabel("SoH (%)")
        ax.grid(alpha=0.3); ax.legend(loc="best", fontsize=8)
    fig.suptitle("MFR_A clean cells — pure rxn-lim SEI vs measured", fontsize=13, y=1.005)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "MFR_A_clean_rxnlim_sweep_grid.png", dpi=110)
    plt.close(fig)
    print(f"\n3x3 grid: MFR_A_clean_rxnlim_sweep_grid.png")


if __name__ == "__main__":
    main()
