"""Minimal Turno-inspired PyBaMM setup — SEI-only degradation, no LAM.

Reduces the Turno notebook setup to the essentials that PyBaMM can solve
without geometry conflicts. Keeps:
  - "SEI: interstitial-diffusion limited"     (their choice, different from ours)
  - "SEI porosity change: true"
  - "lithium plating: irreversible"
  - NO "loss of active material" option        (this is the key — no LAM knee)

And keeps their DEGRADATION parameter overrides (Turno notebook cell 5):
  - SEI resistivity: 400,000 Ω·m
  - SEI partial molar volume: 4.76e-5 m3/mol
  - SEI lithium interstitial diffusivity: 1e-18 m2/s
  - SEI growth activation energy: 38,000 J/mol
  - Lithium plating kinetic rate constant: 1e-9 m/s

Skips the CALB-specific GEOMETRY overrides (Nominal cap, NE thickness,
particle radius, porosity, conductivity, etc.) — those caused solver
init failure and were fitted to their specific CALB cell anyway.

Writes: data/synthetic/calibration/deceleration_check_turno_min.parquet + png
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

MODEL_OPTIONS = {
    "SEI": "interstitial-diffusion limited",
    "SEI porosity change": "true",
    "lithium plating": "irreversible",
    # NO thermal (default isothermal), NO LAM
}

TURNO_DEGRAD_OVERRIDES = {
    "SEI resistivity [Ohm.m]": 400_000.0,
    "SEI partial molar volume [m3.mol-1]": 4.76e-5,
    "SEI lithium interstitial diffusivity [m2.s-1]": 1e-18,
    "SEI growth activation energy [J.mol-1]": 38_000.0,
    "Lithium plating kinetic rate constant [m.s-1]": 1e-9,
}

C_RATE = 0.5
N_CYCLES = 3000

BANDS = [
    ("fresh (0.85-1.00)", 0.85, 1.00, 10.0),
    ("mid   (0.45-0.55)", 0.45, 0.55,  3.7),
    ("aged  (0.25-0.35)", 0.25, 0.35,  1.3),
]


def compute_soh_from_lli(sol, q_nominal_Ah: float):
    F = 96485.3
    cycles = []; Q_LLI = []; Q_SEI = []; Q_plating = []
    for i, c in enumerate(sol.cycles):
        if c is None: continue
        try: end = c.steps[0]["Total lithium lost [mol]"].entries[-1]
        except Exception: end = c["Total lithium lost [mol]"].entries[-1]
        q_lli = abs(end) * F / 3600.0
        Q_LLI.append(q_lli)
        try: Q_SEI.append(c.steps[0]["Loss of capacity to negative SEI [A.h]"].entries[-1])
        except Exception: Q_SEI.append(np.nan)
        try: Q_plating.append(c.steps[0]["Loss of capacity to negative lithium plating [A.h]"].entries[-1])
        except Exception: Q_plating.append(np.nan)
        cycles.append(i + 1)
    df = pd.DataFrame({"cycle_n": cycles, "Q_LLI_Ah": Q_LLI,
                         "Q_SEI_Ah": Q_SEI, "Q_plating_Ah": Q_plating})
    df["SOH"] = 1.0 - df["Q_LLI_Ah"] / q_nominal_Ah
    return df


def band_rate(sub, lo, hi):
    m = (sub.SOH >= lo) & (sub.SOH <= hi)
    if m.sum() < 10: return float("nan")
    slope, _ = np.polyfit(sub.cycle_n[m], sub.SOH[m], 1)
    return -slope * 100 * 1000


def main() -> int:
    import pybamm

    print("=== Minimal Turno-inspired setup validation ===")
    print(f"  Options: {MODEL_OPTIONS}")
    print(f"  Turno degradation overrides: {json.dumps(TURNO_DEGRAD_OVERRIDES, indent=2)}")
    print(f"  N cycles: {N_CYCLES}, C-rate: {C_RATE}", flush=True)

    param = pybamm.ParameterValues("Prada2013")
    donor = pybamm.ParameterValues("OKane2022")
    for k in set(donor.keys()) - set(param.keys()):
        param.update({k: donor[k]}, check_already_exists=False)
    for k, v in TURNO_DEGRAD_OVERRIDES.items():
        if k in param.keys():
            param.update({k: v})
        else:
            param.update({k: v}, check_already_exists=False)

    q_nom = float(param["Nominal cell capacity [A.h]"])
    print(f"  Nominal capacity: {q_nom:.3f} Ah")

    model = pybamm.lithium_ion.DFN(options=MODEL_OPTIONS)
    experiment = pybamm.Experiment([
        (
            f"Discharge at {C_RATE:.4f}C until 2.5 V",
            "Rest for 10 minutes",
            f"Charge at {C_RATE:.4f}C until 3.65 V",
            "Hold at 3.65 V until C/100",
            "Rest for 10 minutes",
        ),
    ] * N_CYCLES)

    try:
        solver = pybamm.IDAKLUSolver(rtol=1e-6, atol=1e-6)
    except Exception:
        solver = pybamm.CasadiSolver(mode="safe", dt_max=600.0)

    print("\n  Building + solving...", flush=True)
    t0 = time.time()
    sim = pybamm.Simulation(model, parameter_values=param,
                              experiment=experiment, solver=solver)
    try:
        sol = sim.solve()
    except Exception as e:
        import traceback
        print(f"\nSim FAILED: {type(e).__name__}: {e}")
        print(traceback.format_exc()[:2500])
        return 1
    elapsed = time.time() - t0

    df = compute_soh_from_lli(sol, q_nom)
    print(f"\nSim ok, elapsed {elapsed:.1f}s, cycles: {int(df.cycle_n.max())}, "
          f"SoH: {df.SOH.iloc[0]:.4f} -> {df.SOH.iloc[-1]:.4f}")
    df.to_parquet(OUT_DIR / "deceleration_check_turno_min.parquet", index=False)

    print("\n=== Simulated fade rate by SoH band ===")
    sim_rates = {}
    for name, lo, hi, tgt in BANDS:
        r = band_rate(df, lo, hi)
        sim_rates[name] = r
        n_pts = int(((df.SOH >= lo) & (df.SOH <= hi)).sum())
        print(f"  {name}: {r:.2f} pp/1000cy   (n_pts={n_pts})")

    print("\n=== Comparison to measured ===")
    for name, lo, hi, tgt in BANDS:
        s = sim_rates[name]
        err = (s - tgt) / tgt * 100 if s == s else float("nan")
        print(f"  {name}: sim={s:.2f}  measured={tgt:.2f}  err={err:+.0f}%")

    # ── Plot alongside all prior baselines ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))

    base_solvent = pd.read_parquet(OUT_DIR / "deceleration_check.parquet")
    base_rxnlim  = pd.read_parquet(OUT_DIR / "deceleration_check_rxnlim.parquet")

    ax1.plot(base_solvent.cycle_n, base_solvent.SOH, "b-", lw=1.2, alpha=0.6,
              label="solvent-diff SEI + LAM (baseline)")
    ax1.plot(base_rxnlim.cycle_n,  base_rxnlim.SOH,  "g-", lw=1.2, alpha=0.6,
              label="reaction-lim SEI + LAM")
    ax1.plot(df.cycle_n,           df.SOH,           "m-", lw=2.0,
              label="Turno-min: interstitial-diff SEI, NO LAM")
    ax1.axhspan(0.85, 1.00, color="tab:red",    alpha=0.10, label="fresh band")
    ax1.axhspan(0.45, 0.55, color="tab:orange", alpha=0.10, label="mid band")
    ax1.axhspan(0.25, 0.35, color="tab:green",  alpha=0.10, label="aged band")
    ax1.axhline(0.80, color="k", ls=":", lw=0.8)
    ax1.set_xlabel("Cycle"); ax1.set_ylabel("SoH")
    ax1.set_title(f"SoH trajectories (3000 cy @ {C_RATE}C)")
    ax1.legend(fontsize=8, loc="lower left"); ax1.grid(alpha=0.3)

    labels = ["fresh", "mid", "aged"]
    measured = [10.0, 3.7, 1.3]
    turno_r = [sim_rates[name] if sim_rates[name] == sim_rates[name] else 0
                 for name, _, _, _ in BANDS]
    x = np.arange(3); w = 0.35
    ax2.bar(x - w/2, measured, w, color="tab:blue",   label="Measured", edgecolor="k")
    ax2.bar(x + w/2, turno_r,  w, color="tab:purple", label="Turno-min",   edgecolor="k")
    ax2.set_ylabel("Fade rate [pp/1000cy]")
    ax2.set_xticks(x); ax2.set_xticklabels(labels)
    ax2.set_title("Turno-min vs measured — fade rate by SoH band")
    ax2.legend(fontsize=10); ax2.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "deceleration_check_turno_min.png", dpi=120, bbox_inches="tight")
    print(f"\nWrote {OUT_DIR / 'deceleration_check_turno_min.parquet'}")
    print(f"Wrote {OUT_DIR / 'deceleration_check_turno_min.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
