"""Validate that switching SEI submodel from 'solvent-diffusion limited' to
'reaction limited' produces the decelerating fade profile observed in
measured cells.

Rationale:
- Solvent-diffusion-limited SEI grows via solvent transport, yielding roughly
  linear-with-time thickness growth. In a cycling context, the SEI barrier
  never limits its own growth strongly. Fade is then dominated by LAM_neg
  hitting a stoichiometry wall (the accelerating knee at SoH ~0.9).

- Reaction-limited SEI grows via the SEI-forming reaction current. As SEI
  thickens, its resistance suppresses the reaction current itself, so growth
  self-limits and fade DECELERATES (Ramadass-canonical form).

- LFP graphite cells are known to be SEI-limited in ageing, so this is the
  more physically appropriate model for our chemistry.

Parameter changes:
- SEI submodel: "solvent-diffusion limited" -> "reaction limited"
- LAM_negative_rate_s: 5.5e-8 -> 1e-8   (drop by ~5x since SEI will now
                                          contribute more; LAM stays as a
                                          minor correction not the driver)
- k_SEI_ms: 1.6e-15 -> 5e-13            (bump up so reaction-limited SEI
                                          produces meaningful fade rate)

Writes: data/synthetic/calibration/deceleration_check_rxnlim.parquet
        data/synthetic/calibration/deceleration_check_rxnlim.png
"""
from __future__ import annotations
import sys, time, json
from pathlib import Path

sys.path.insert(0, "/home/hj/Desktop/PINNs")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


OUT_DIR = Path("/home/hj/Desktop/PINNs/data/synthetic/calibration")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CANDIDATE = {
    "sample_id": "s_calib_rxnlim_v1",
    "k_SEI_ms":                              5.0e-13,      # up from 1.6e-15
    "SEI_partial_molar_volume_m3mol":        9.438e-05,    # unchanged
    "lithium_plating_exchange_current_A_m2": 5.991e-08,    # unchanged
    "LAM_positive_rate_s":                   1.821e-08,    # unchanged
    "LAM_negative_rate_s":                   1.0e-08,      # down from 5.5e-8
    "temperature_K":                         298.15,
    "c_rate":                                0.5,
}

MODEL_OPTIONS = {
    "SEI": "reaction limited",                              # SWITCH from solvent-diffusion
    "SEI porosity change": "true",
    "lithium plating": "irreversible",
    "loss of active material": "stress-driven",
}

BANDS = [
    ("fresh (0.85-1.00)", 0.85, 1.00, 10.0),
    ("mid   (0.45-0.55)", 0.45, 0.55,  3.7),
    ("aged  (0.25-0.35)", 0.25, 0.35,  1.3),
]


def band_rate(sub: pd.DataFrame, lo: float, hi: float) -> float:
    m = (sub.SOH >= lo) & (sub.SOH <= hi)
    if m.sum() < 10: return float("nan")
    slope, _ = np.polyfit(sub.cycle_n[m], sub.SOH[m], 1)
    return -slope * 100 * 1000


def run_one_with_options(sample_row, n_cycles, model_options):
    """Bespoke sim runner that lets us override model options."""
    import pybamm
    from src.simulation.run_sweep import (
        _build_param_with_overrides, _build_experiment,
    )
    from src.simulation.extract_features import per_cycle_features

    model = pybamm.lithium_ion.DFN(options=model_options)
    param = _build_param_with_overrides(sample_row)
    experiment = _build_experiment(float(sample_row["c_rate"]), n_cycles)
    try:
        solver = pybamm.IDAKLUSolver(rtol=1e-6, atol=1e-6)
    except Exception:
        solver = pybamm.CasadiSolver(mode="safe", dt_max=600.0)
    sim = pybamm.Simulation(model, parameter_values=param,
                              experiment=experiment, solver=solver)
    sol = sim.solve()
    features = per_cycle_features(sol, params_used=sample_row)
    if "sample_id" not in features.columns:
        features.insert(0, "sample_id", sample_row["sample_id"])
    return features


def main() -> int:
    print("=== Reaction-limited SEI validation sim ===")
    print(f"  Model options: {json.dumps(MODEL_OPTIONS, indent=2)}")
    print(f"  Candidate params: {json.dumps({k: v for k, v in CANDIDATE.items() if k != 'sample_id'}, indent=2)}")
    print(f"  Target: fresh 10 pp/1000cy -> mid 3.7 -> aged 1.3 (decelerating)")
    print(f"  n_cycles: 3000, expected wall time 10 min", flush=True)

    t0 = time.time()
    try:
        traj = run_one_with_options(CANDIDATE, n_cycles=3000, model_options=MODEL_OPTIONS)
    except Exception as e:
        import traceback
        print(f"\nSim FAILED with {type(e).__name__}: {e}")
        print(traceback.format_exc()[:3000])
        return 1
    elapsed = time.time() - t0

    print(f"\nSim ok, elapsed {elapsed:.1f}s")
    print(f"cycles completed: {int(traj.cycle_n.max())}, SoH: {traj.SOH.iloc[0]:.4f} -> {traj.SOH.iloc[-1]:.4f}")
    traj.to_parquet(OUT_DIR / "deceleration_check_rxnlim.parquet", index=False)

    print("\n=== Simulated fade rate by SoH band ===")
    sim_rates = {}
    for name, lo, hi, tgt in BANDS:
        r = band_rate(traj, lo, hi)
        sim_rates[name] = r
        n_pts = int(((traj.SOH >= lo) & (traj.SOH <= hi)).sum())
        print(f"  {name}: {r:.2f} pp/1000cy   (n_pts={n_pts})")

    print("\n=== Comparison to measured ===")
    for name, lo, hi, tgt in BANDS:
        s = sim_rates[name]
        err = (s - tgt) / tgt * 100 if s == s else float("nan")
        print(f"  {name}: sim={s:.2f}  measured={tgt:.2f}  err={err:+.0f}%")

    # Load solvent-diffusion baseline for side-by-side comparison
    baseline = pd.read_parquet(OUT_DIR / "deceleration_check.parquet")

    # ── Plot ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
    ax1.plot(baseline.cycle_n, baseline.SOH, "b-", lw=1.4,
              label="Solvent-diffusion SEI (baseline)")
    ax1.plot(traj.cycle_n, traj.SOH, "g-", lw=1.4,
              label="Reaction-limited SEI (new)")
    ax1.axhspan(0.85, 1.00, color="tab:red",    alpha=0.10, label="fresh band")
    ax1.axhspan(0.45, 0.55, color="tab:orange", alpha=0.10, label="mid band")
    ax1.axhspan(0.25, 0.35, color="tab:green",  alpha=0.10, label="aged band")
    ax1.axhline(0.80, color="k", ls=":", lw=0.8, label="EOL threshold")
    ax1.set_xlabel("Cycle"); ax1.set_ylabel("SoH")
    ax1.set_title("SoH trajectory: solvent-diffusion vs reaction-limited SEI")
    ax1.legend(fontsize=8); ax1.grid(alpha=0.3)

    labels = ["fresh (SoH ~1.0)", "mid (SoH ~0.5)", "aged (SoH ~0.3)"]
    measured  = [10.0, 3.7, 1.3]
    baseline_rates = [
        band_rate(baseline, 0.85, 1.00),
        band_rate(baseline, 0.45, 0.55),
        band_rate(baseline, 0.25, 0.35),
    ]
    baseline_rates = [0 if pd.isna(r) else r for r in baseline_rates]
    new_rates      = [sim_rates[name] for name, _, _, _ in BANDS]
    new_rates      = [0 if pd.isna(r) else r for r in new_rates]
    x = np.arange(3); w = 0.27
    ax2.bar(x - w, measured,       w, color="tab:blue",   label="Measured",              edgecolor="k")
    ax2.bar(x,     baseline_rates, w, color="tab:orange", label="Solvent-diff (base)",   edgecolor="k")
    ax2.bar(x + w, new_rates,      w, color="tab:green",  label="Reaction-lim (new)",    edgecolor="k")
    ax2.set_ylabel("Fade rate [pp/1000cy]")
    ax2.set_xticks(x); ax2.set_xticklabels(labels)
    ax2.set_title("Fade rate by SoH band")
    ax2.legend(fontsize=9); ax2.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "deceleration_check_rxnlim.png", dpi=120, bbox_inches="tight")
    print(f"\nWrote {OUT_DIR / 'deceleration_check_rxnlim.parquet'}")
    print(f"Wrote {OUT_DIR / 'deceleration_check_rxnlim.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
