"""REPT cohort sweep — calibration target is the **workbook b1→b2 fade rate**.

REPT has the rare luxury of TWO characterization snapshots per cell (batch 1
and batch 2 in `Char_Consolidated_VKC_SoC.xlsx`). The difference in measured
SoH between those two snapshots, divided by `cycles_per_batch`, is a
direct per-cell ground-truth fade rate — no formation artifacts, no DoD
window arithmetic, no longterm-CSV regression noise.

Per-cell pipeline:
    1. Load BOTH batches of char data; b1 = "starting state", b2 = "later state"
    2. target_slope = -(SoH_b1 - SoH_b2) / cycles_per_batch * 100  (pp/100cy)
    3. Build PyBaMM parameters from the b1 snapshot, pre-aged to SoH_b1
    4. Calibrate SEI diffusivity to match target_slope
    5. Validate against the longterm CSV (cycles 1-N):
         - measured_csv_slope = polyfit(cycles, dchg_cap/nominal*100)
         - flag CSV_VS_WORKBOOK_DISAGREE if they differ by >2× (expected:
           early cycles fade faster than the long-run average)
"""
from __future__ import annotations

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
    build_pybamm_parameters, calibrate_sei_diffusivity,
    fit_stoichiometry_from_ocv, load_characterization,
    list_available_cells, SEI_ONLY_DFN_OPTIONS,
)
from pybamm_tuning.simulation import CyclingProtocol, Simulation


# ─────────────────────────── config ───────────────────────────
PROTOCOL = CyclingProtocol(c_rate=0.25)
TEMP_K = 298.15
N_CYCLES_CALIBRATION = 10
N_CYCLES_VALIDATION = 20
CYCLES_PER_BATCH = 600   # convention in pybamm_tuning.compute_actual_fade_rate

CACHE_DIR = Path("/home/hj/Desktop/PINNs/Voltaris/outputs/pybamm_cache")
OUT_DIR = Path("/home/hj/Desktop/PINNs/Voltaris/outputs/tuned_params")
LONGTERM_DIR = Path("/home/hj/Desktop/PINNs/Data/Longterm")


# ─────────────────────────── cohort selection ───────────────────────────
def select_top_n_by_fade(n: int = 10) -> list[dict]:
    """Pick the n REPT cells with the largest |b1-b2 fade|, filtered to
    those with a longterm CSV on disk."""
    cells = list_available_cells()
    rept = cells[cells.manufacturer == "REPT"].copy()
    rept["cell_id"] = rept["cell_id"].astype(str)

    # Cells with BOTH batches
    paired = rept.groupby("cell_id")["batch"].nunique() == 2
    paired_ids = set(paired[paired].index)

    # Intersect with longterm CSV availability
    csv_ids = {p.stem.split("_")[-1].lstrip("0")
                for p in LONGTERM_DIR.glob("REPT_*.csv")}

    rows = []
    for cid in sorted(paired_ids & csv_ids, key=lambda x: int(x)):
        sub = rept[rept["cell_id"] == cid].sort_values("batch")
        if len(sub) < 2:
            continue
        soh_b1 = float(sub["Soh"].iloc[0])
        soh_b2 = float(sub["Soh"].iloc[1])
        fade_pp = soh_b1 - soh_b2
        rows.append({"cell_id": cid, "soh_b1": soh_b1, "soh_b2": soh_b2,
                      "fade_pp": fade_pp})
    df = pd.DataFrame(rows)
    df = df.iloc[df["fade_pp"].abs().sort_values(ascending=False).index]
    return df.head(n).to_dict(orient="records")


# ─────────────────────────── per-cycle helpers ───────────────────────────
def per_cycle_soh(longterm_csv: Path, nominal_ah: float) -> pd.DataFrame:
    df = pd.read_csv(longterm_csv,
                      usecols=["cycle_no", "step_name", "capacity_ah"])
    dchg = df[df["step_name"].astype(str).str.contains("DChg")]
    if dchg.empty:
        return pd.DataFrame(columns=["cycle_no", "dchg_cap_ah", "soh"])
    out = (dchg.groupby("cycle_no")["capacity_ah"]
               .agg(lambda s: float(s.abs().max()))
               .reset_index()
               .rename(columns={"capacity_ah": "dchg_cap_ah"}))
    out["soh"] = out["dchg_cap_ah"] / nominal_ah
    return out.sort_values("cycle_no").reset_index(drop=True)


def csv_slope_pp_per_100cy(per_cycle: pd.DataFrame) -> float:
    if len(per_cycle) < 5:
        return float("nan")
    x = per_cycle["cycle_no"].astype(float).values
    y = per_cycle["soh"].astype(float).values * 100.0
    slope, _ = np.polyfit(x, y, 1)
    return float(slope * 100.0)


# ─────────────────────────── per-cell workflow ───────────────────────────
def run_cell(meta: dict) -> dict:
    cid = meta["cell_id"]
    cell_tag = f"REPT_{cid}"
    print(f"\n=== {cell_tag} ===")
    t0 = time.time()

    # 1) Load both batches
    char_b1 = load_characterization(manufacturer="REPT", cell_id=cid, batch=1)
    char_b2 = load_characterization(manufacturer="REPT", cell_id=cid, batch=2)
    soh_b1, soh_b2 = float(char_b1.soh_pct), float(char_b2.soh_pct)
    fade_pp = soh_b1 - soh_b2

    # 2) Calibration target from workbook pair
    target_slope = -fade_pp / CYCLES_PER_BATCH * 100.0  # pp/100cy
    print(f"  workbook: SoH_b1={soh_b1:.2f} %, SoH_b2={soh_b2:.2f} %, "
          f"fade={fade_pp:+.2f} pp over {CYCLES_PER_BATCH} cy "
          f"→ target slope={target_slope:+.4f} pp/100cy")

    # 3) Sanity gates
    gates = {
        "INVERTED_SLOPE": target_slope > 0,
        "SHORT_LONGTERM": False,    # filled below
        "CSV_VS_WORKBOOK_DISAGREE": False,
        "LOW_SOH_SIGNAL": abs(target_slope) < 0.05,
    }

    if gates["INVERTED_SLOPE"]:
        print(f"  cal:  SKIPPED (INVERTED_SLOPE: b1<b2, cell may have GAINED capacity)")
        classification = "POOR"
        return _write_skip(cell_tag, meta, soh_b1, soh_b2, target_slope, gates,
                            classification, t0)

    # 4) Pre-age to workbook SoH_b1 (clamped to [0.5, 1.0] inside apply_pre_aging)
    pre_age_factor = soh_b1 / 100.0
    if pre_age_factor > 1.0:
        pre_age_factor = 1.0   # cells coming in >nominal start "fresh"

    # 5) Calibration
    cal = calibrate_sei_diffusivity(
        char_b1, target_slope_pp_per_100cy=target_slope,
        protocol=PROTOCOL, temperature_K=TEMP_K,
        n_cycles=N_CYCLES_CALIBRATION,
        log10_bracket=(-24.0, -19.0), rtol=0.20,
        cache_dir=CACHE_DIR,
        pre_age_to_soh=pre_age_factor,
    )
    rel_err = abs(cal.residual_pp_per_100cy / cal.target_slope_pp_per_100cy) * 100.0
    classification = ("GOOD" if rel_err <= 25 else
                       "FAIR" if rel_err <= 50 else "POOR")
    print(f"  cal:  D_SEI={cal.fitted_value:.3e} m²/s, "
          f"rel_err={rel_err:.1f} % → {classification}  "
          f"(fresh={cal.n_fresh_sims}/{cal.n_evaluations})")

    # 6) Validation against longterm CSV
    csv = LONGTERM_DIR / f"REPT_Longterm_cell_{int(cid):04d}.csv"
    per_cycle = per_cycle_soh(csv, nominal_ah=char_b1.nominal_capacity_ah) \
                  if csv.exists() else pd.DataFrame()
    n_cyc = int(per_cycle["cycle_no"].max()) if not per_cycle.empty else 0
    gates["SHORT_LONGTERM"] = n_cyc < 50
    csv_slope = csv_slope_pp_per_100cy(per_cycle)
    # Disagreement: CSV early-cycle fade vs workbook averaged fade.
    # Expect CSV slope to be steeper (more negative) — flag if >3× ratio.
    if not np.isnan(csv_slope) and target_slope < 0:
        ratio = csv_slope / target_slope
        gates["CSV_VS_WORKBOOK_DISAGREE"] = ratio > 3.0 or ratio < 0.33
    print(f"  long: {n_cyc} cycles, CSV slope={csv_slope:+.4f} pp/100cy "
          f"(target was {target_slope:+.4f})")
    if not per_cycle.empty:
        per_cycle.to_csv(OUT_DIR / f"{cell_tag}_longterm_per_cycle.csv", index=False)

    # 7) Run a 20-cy sim with the calibrated D_SEI for the overlay plot
    val_params = build_pybamm_parameters(
        char_b1, base="Prada2013", temperature_K=TEMP_K,
        extra_overrides={"SEI solvent diffusivity [m2.s-1]": cal.fitted_value},
        pre_age_to_soh=pre_age_factor,
    )
    sim = Simulation(val_params, protocol=PROTOCOL, cache_dir=CACHE_DIR,
                      dfn_options=SEI_ONLY_DFN_OPTIONS)
    val_df = sim.run(n_cycles=N_CYCLES_VALIDATION)
    val_df.to_parquet(OUT_DIR / f"{cell_tag}_calibrated_sim_{N_CYCLES_VALIDATION}cy.parquet")
    sim_soh_pct = val_df["SOH"].values * 100.0
    sim_cyc = val_df["cycle_n"].values.astype(float)
    sim_slope = float(np.polyfit(sim_cyc[1:], sim_soh_pct[1:], 1)[0] * 100)

    # 8) Plot: measured CSV (first 20 cy) vs simulated, no anchoring needed
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if not per_cycle.empty:
        early = per_cycle[per_cycle.cycle_no <= N_CYCLES_VALIDATION]
        ax.plot(early.cycle_no, early.soh * 100.0, "o-", lw=1.2,
                 color="#d62728", label="measured (longterm CSV)")
    ax.plot(sim_cyc, sim_soh_pct, "s--", lw=1.2,
             color="#1f77b4",
             label=f"sim (D_SEI={cal.fitted_value:.1e}, target={target_slope:+.3f})")
    ax.axhline(80, ls=":", color="grey", alpha=0.6)
    ax.set_xlabel("Cycle"); ax.set_ylabel("SoH (%)")
    ax.set_title(f"{cell_tag} — calibrated SEI vs measured ({N_CYCLES_VALIDATION} cy)")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{cell_tag}_sim_vs_measured.png", dpi=120)
    plt.close(fig)

    # 9) JSON output
    payload = {
        "cell": cell_tag,
        "cohort": "REPT",
        "soh_b1_pct": soh_b1,
        "soh_b2_pct": soh_b2,
        "fade_b1_to_b2_pp": fade_pp,
        "cycles_per_batch": CYCLES_PER_BATCH,
        "pre_age_to_soh": pre_age_factor,
        "target_slope_pp_per_100cy": target_slope,
        "achieved_slope_pp_per_100cy": cal.achieved_slope_pp_per_100cy,
        "residual_pp_per_100cy": cal.residual_pp_per_100cy,
        "relative_error_pct": rel_err,
        "calibrated_param": cal.parameter_name,
        "calibrated_value": cal.fitted_value,
        "log10_bracket_used": list(cal.log10_bracket_used),
        "n_evaluations": cal.n_evaluations,
        "n_fresh_sims": cal.n_fresh_sims,
        "classification": classification,
        "csv_slope_pp_per_100cy": csv_slope,
        "csv_n_cycles": n_cyc,
        "sim_slope_pp_per_100cy": sim_slope,
        "gate_audit": {k: {"tripped": bool(v)} for k, v in gates.items()},
        "fallback_strategies_invoked": [],
    }
    (OUT_DIR / f"{cell_tag}_aging_calibrated.json").write_text(
        json.dumps(payload, indent=2, default=str))

    # 10) Markdown report
    md = f"""# {cell_tag} — Voltaris parameter-tuning report

## TL;DR
**Classification: {classification}** — SEI solvent diffusivity calibrated to `{cal.fitted_value:.3e} m²/s`
against the workbook b1→b2 fade (rel err **{rel_err:.2f} %**).

| | b1 (start) | b2 (after {CYCLES_PER_BATCH} cy) |
|---|---|---|
| Measured SoH (%) | {soh_b1:.2f} | {soh_b2:.2f} |

- **Workbook fade rate**: `{target_slope:+.4f} pp/100cy`
- **Longterm CSV slope** ({n_cyc} cy): `{csv_slope:+.4f} pp/100cy`  ← independent check
- **Sim slope** (20 cy, calibrated D_SEI): `{sim_slope:+.4f} pp/100cy`
- **Pre-aged to**: SoH = `{pre_age_factor:.3f}` (workbook b1)

## Gates
| Gate | Tripped? |
|---|:---:|
""" + "\n".join(f"| `{g}` | {'✓' if v else '✗'} |" for g, v in gates.items()) + f"""

## Wall-time
- Total: **{time.time() - t0:.1f} s**, fresh PyBaMM sims: **{cal.n_fresh_sims}/{cal.n_evaluations}**
"""
    (OUT_DIR / f"{cell_tag}_calibration_report.md").write_text(md)

    return {
        "cell": cell_tag, "classification": classification,
        "D_SEI": cal.fitted_value, "rel_err": rel_err,
        "csv_slope": csv_slope, "sim_slope": sim_slope, "target_slope": target_slope,
        "n_fresh_sims": cal.n_fresh_sims,
        "wall_time_s": time.time() - t0,
        "gates_tripped": [g for g, v in gates.items() if v],
    }


def _write_skip(cell_tag, meta, soh_b1, soh_b2, target_slope, gates,
                classification, t0):
    """Write minimal JSON + report when calibration is skipped."""
    payload = {
        "cell": cell_tag, "cohort": "REPT",
        "soh_b1_pct": soh_b1, "soh_b2_pct": soh_b2,
        "fade_b1_to_b2_pp": soh_b1 - soh_b2,
        "target_slope_pp_per_100cy": target_slope,
        "classification": classification,
        "calibrated_value": float("nan"),
        "relative_error_pct": float("nan"),
        "n_fresh_sims": 0, "n_evaluations": 0,
        "gate_audit": {k: {"tripped": bool(v)} for k, v in gates.items()},
    }
    (OUT_DIR / f"{cell_tag}_aging_calibrated.json").write_text(
        json.dumps(payload, indent=2, default=str))
    return {"cell": cell_tag, "classification": classification,
            "D_SEI": float("nan"), "rel_err": float("nan"),
            "csv_slope": float("nan"), "sim_slope": float("nan"),
            "target_slope": target_slope, "n_fresh_sims": 0,
            "wall_time_s": time.time() - t0,
            "gates_tripped": [g for g, v in gates.items() if v]}


def main(n_cells: int = 10) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Selecting top {n_cells} REPT cells by |fade_pp| …")
    targets = select_top_n_by_fade(n_cells)
    for t in targets:
        print(f"  {t['cell_id']:>4}  fade={t['fade_pp']:+.2f} pp  "
              f"(SoH {t['soh_b1']:.2f} → {t['soh_b2']:.2f})")

    results = []
    for t in targets:
        try:
            results.append(run_cell(t))
        except Exception as e:
            print(f"  FAIL REPT_{t['cell_id']}: {type(e).__name__}: {e}")
            results.append({"cell": f"REPT_{t['cell_id']}", "error": str(e)})

    print("\n=== Sweep summary ===")
    print(f"{'cell':<10} {'class':<6} {'D_SEI':<11} {'err':<6} "
          f"{'tgt':<8} {'csv':<8} {'sim':<8} {'fresh':<5} gates")
    for r in results:
        if "error" in r:
            print(f"  {r['cell']:<8} ERROR: {r['error']}")
            continue
        print(f"  {r['cell']:<8} {r['classification']:<6} "
              f"{r['D_SEI']:<11.2e} {r['rel_err']:<6.1f} "
              f"{r['target_slope']:<+8.3f} {r['csv_slope']:<+8.3f} "
              f"{r['sim_slope']:<+8.3f} "
              f"{r['n_fresh_sims']:<5} "
              f"{','.join(r['gates_tripped']) or '-'}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-cells", type=int, default=10)
    args = ap.parse_args()
    main(n_cells=args.n_cells)
