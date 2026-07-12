"""Single-shot validation sim to check whether PyBaMM's degradation model
reproduces the measured deceleration profile.

Measured fade profile (real cells):
  fresh band (SoH ~1.0):   ~10 pp/1000cy
  mid band   (SoH ~0.5):   ~3.7 pp/1000cy
  aged band  (SoH ~0.3):   ~1.3 pp/1000cy

Candidate params (from top-fresh matching sim s00048 in existing corpus):
  k_SEI_ms                                = 1.6e-15  m/s
  SEI_partial_molar_volume_m3mol          = 9.4e-5   m3/mol
  lithium_plating_exchange_current_A_m2   = 6.0e-8   A/m2
  LAM_positive_rate_s                     = 1.8e-8   s^-1
  LAM_negative_rate_s                     = 5.5e-8   s^-1 (dominant channel)
  c_rate                                  = 0.5      (aligned with measured protocol)

Writes: data/synthetic/calibration/deceleration_check.parquet
        data/synthetic/calibration/deceleration_check.png
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
    "sample_id": "s_calib_v1",
    "k_SEI_ms":                              1.625e-15,
    "SEI_partial_molar_volume_m3mol":        9.438e-05,
    "lithium_plating_exchange_current_A_m2": 5.991e-08,
    "LAM_positive_rate_s":                   1.821e-08,
    "LAM_negative_rate_s":                   5.464e-08,
    "temperature_K":                         298.15,
    "c_rate":                                0.5,
}

BANDS = [
    ("fresh (0.85-1.00)",  0.85, 1.00, 10.0),
    ("mid   (0.45-0.55)",  0.45, 0.55,  3.7),
    ("aged  (0.25-0.35)",  0.25, 0.35,  1.3),
]


def band_rate(sub: pd.DataFrame, lo: float, hi: float) -> float:
    m = (sub.SOH >= lo) & (sub.SOH <= hi)
    if m.sum() < 10: return float("nan")
    slope, _ = np.polyfit(sub.cycle_n[m], sub.SOH[m], 1)
    return -slope * 100 * 1000


def main() -> int:
    from src.simulation.run_sweep import run_one_simulation

    print("=== Calibration validation sim ===")
    print(f"  Candidate params: {json.dumps({k: v for k, v in CANDIDATE.items() if k != 'sample_id'}, indent=2)}")
    print(f"  Target profile: fresh 10 pp/1000cy → mid 3.7 → aged 1.3 pp/1000cy (decelerating)")
    print(f"  n_cycles: 3000, expected wall time 30-90 min")
    print(flush=True)

    t0 = time.time()
    result = run_one_simulation(CANDIDATE, n_cycles=3000, timeout_s=7200)
    elapsed = time.time() - t0

    print(f"\nSim status: {result['status']}, elapsed {elapsed:.1f}s")
    if result["status"] != "ok":
        print(f"ERROR: {result.get('error', 'unknown')}")
        print(result.get("traceback", "")[:2000])
        return 1

    traj = result["features"]
    print(f"cycles completed: {result['n_cycles_completed']}, "
          f"SoH: {traj.SOH.iloc[0]:.4f} → {traj.SOH.iloc[-1]:.4f}")
    traj.to_parquet(OUT_DIR / "deceleration_check.parquet", index=False)

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

    # ── Plot ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
    ax1.plot(traj.cycle_n, traj.SOH, "b-", lw=1.4)
    ax1.axhspan(0.85, 1.00, color="tab:red",    alpha=0.10, label="fresh band")
    ax1.axhspan(0.45, 0.55, color="tab:orange", alpha=0.10, label="mid band")
    ax1.axhspan(0.25, 0.35, color="tab:green",  alpha=0.10, label="aged band")
    ax1.axhline(0.80, color="k", ls=":", lw=0.8, label="EOL threshold")
    ax1.set_xlabel("Cycle"); ax1.set_ylabel("SoH")
    ax1.set_title("Simulated SoH trajectory (candidate params, 3000 cy @ 0.5C)")
    ax1.legend(fontsize=9); ax1.grid(alpha=0.3)

    labels = ["fresh (SoH ~1.0)", "mid (SoH ~0.5)", "aged (SoH ~0.3)"]
    measured = [tgt for _, _, _, tgt in BANDS]
    simulated = [sim_rates[name] for name, _, _, _ in BANDS]
    x = np.arange(3)
    ax2.bar(x - 0.2, measured,  0.4, color="tab:blue",   label="Measured",              edgecolor="k")
    ax2.bar(x + 0.2, simulated, 0.4, color="tab:orange", label="Simulated (candidate)", edgecolor="k")
    ax2.set_ylabel("Fade rate [pp/1000cy]")
    ax2.set_xticks(x); ax2.set_xticklabels(labels)
    ax2.set_title("Fade rate by SoH band — measured vs simulated")
    ax2.legend(fontsize=10); ax2.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "deceleration_check.png", dpi=120, bbox_inches="tight")
    print(f"\nWrote {OUT_DIR / 'deceleration_check.parquet'}")
    print(f"Wrote {OUT_DIR / 'deceleration_check.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
