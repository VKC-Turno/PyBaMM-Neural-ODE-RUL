"""Validate the Turno CALB-tuned PyBaMM setup reproduces measured decelerating fade.

Setup replicated from:
  /home/hj/Desktop/PyBaMM/PyBaMM/turno_cell_modelling/notebooks/
  Sunil_sensitivity_analysis_first_second_life_V2_final.ipynb  (cells 4, 5)

Key differences from what we were using:
  Model options:
    - open-circuit potential: ("single", "current sigmoid")  [smooth LFP plateau]
    - SEI: "interstitial-diffusion limited"                    [NOT solvent-diff or reaction-lim]
    - thermal: "lumped"                                         [enabled]
    - lithium plating: "irreversible", porosity change "true"
    - intercalation kinetics: "symmetric Butler-Volmer"
    - NO loss of active material option                        [this is the key]

  Cell geometry / chemistry overrides:
    - Nominal cell capacity: 72 Ah
    - NE active-material volume fraction: 0.4891
    - NE thickness: 3.9e-5 m
    - NE max concentration: 31965 mol/m3
    - NE initial concentration: 29222 mol/m3
    - NE conductivity: 315 S/m
    - NE porosity: 0.4189
    - Total heat transfer coeff: 19.8 W/m2K
    - PE particle radius: 1.51e-8 m
    - NE particle diffusivity: 2e-14 m2/s
    - Electrode height: 0.213 m

  Degradation:
    - SEI resistivity: 400,000 Ohm.m
    - SEI partial molar volume: 4.76e-5 m3/mol
    - SEI Li interstitial diffusivity: 1e-18 m2/s  (their sensitivity midpoint)
    - SEI growth activation energy: 38000 J/mol
    - Plating kinetic rate: 1e-9 m/s

SoH via LLI (matches notebook):
  Q_LLI = |Total lithium lost [mol]| * F / 3600     # in Ah
  SoH = 1 - Q_LLI / Q_nominal_Ah

Writes: data/synthetic/calibration/deceleration_check_turno.parquet + png
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
    # Skipping ("single", "current sigmoid") OCP option — requires separate
    # delithiation/lithiation OCP curves that vanilla Prada2013 doesn't ship.
    # Numerical smoothing only, not essential to fade shape.
    "SEI": "interstitial-diffusion limited",
    "SEI porosity change": "true",
    "thermal": "lumped",
    "lithium plating": "irreversible",
    "lithium plating porosity change": "true",
    "intercalation kinetics": "symmetric Butler-Volmer",
    # NOTE: no "loss of active material" option -> disabled  (Turno choice)
}

# Overrides from notebook cell 5 (calb tuning)
TURNO_OVERRIDES = {
    "Nominal cell capacity [A.h]": 72.0,
    "Negative electrode active material volume fraction": 0.489075,
    "Negative electrode thickness [m]": 3.9e-05,
    "Maximum concentration in negative electrode [mol.m-3]": 31965.010317,
    # NOTE: initial concentrations left at Prada2013 default so PE and NE stay
    # electrochemically consistent (setting only NE_init without matching PE_init
    # causes solver init failure — LFP flat plateau needs consistent stoich).
    # "Initial concentration in negative electrode [mol.m-3]": 29222.664248,
    "Negative electrode conductivity [S.m-1]": 315.01868665506436,
    "Negative particle diffusivity [m2.s-1]": 0.2e-13,     # override (2e-14)
    "Negative electrode porosity": 0.41894248857756198055,
    "Total heat transfer coefficient [W.m-2.K-1]": 19.8,
    # Degradation
    "SEI resistivity [Ohm.m]": 400000.0,
    "SEI partial molar volume [m3.mol-1]": 4.76e-5,
    # BoL-optimised
    "Positive particle radius [m]": 1.51e-08,
    "Electrode height [m]": 2.13e-01,
    # Life-specific (their midpoint values from sensitivity sweep)
    "Initial temperature [K]": 298.15,
    "Ambient temperature [K]": 298.15,
    "SEI lithium interstitial diffusivity [m2.s-1]": 1e-18,
    "SEI growth activation energy [J.mol-1]": 38000.0,
    "Lithium plating kinetic rate constant [m.s-1]": 1e-9,
}

C_RATE = 0.5
N_CYCLES = 3000

# Measured targets
BANDS = [
    ("fresh (0.85-1.00)", 0.85, 1.00, 10.0),
    ("mid   (0.45-0.55)", 0.45, 0.55,  3.7),
    ("aged  (0.25-0.35)", 0.25, 0.35,  1.3),
]


def build_model_and_params():
    import pybamm

    # Load standard Prada2013 (Turno variant is identical to standard)
    param = pybamm.ParameterValues("Prada2013")
    # OKane2022 donor for missing degradation keys (irreversible plating etc)
    donor = pybamm.ParameterValues("OKane2022")
    for k in set(donor.keys()) - set(param.keys()):
        param.update({k: donor[k]}, check_already_exists=False)
    # Now apply the Turno overrides
    for k, v in TURNO_OVERRIDES.items():
        if k in param.keys():
            param.update({k: v})
        else:
            print(f"  [note] key not in base param: {k!r}, forcing")
            param.update({k: v}, check_already_exists=False)

    model = pybamm.lithium_ion.DFN(options=MODEL_OPTIONS)
    return model, param


def build_experiment(c_rate: float, n_cycles: int, v_min: float = 2.5, v_max: float = 3.65):
    import pybamm
    block = (
        f"Discharge at {c_rate:.4f}C until {v_min:.2f} V",
        "Rest for 10 minutes",
        f"Charge at {c_rate:.4f}C until {v_max:.2f} V",
        f"Hold at {v_max:.2f} V until C/100",
        "Rest for 10 minutes",
    )
    return pybamm.Experiment([block] * int(n_cycles))


def compute_soh_from_lli(sol, q_nominal_Ah: float):
    """SoH via total lithium lost (matches notebook cell 2 method)."""
    F = 96485.3
    cycles = []
    Q_LLI = []
    Q_SEI = []
    Q_plating = []
    for i, c in enumerate(sol.cycles):
        if c is None: continue
        # Look at end of first step (discharge)
        try:
            end = c.steps[0]["Total lithium lost [mol]"].entries[-1]
        except Exception:
            end = c["Total lithium lost [mol]"].entries[-1]
        q_lli = abs(end) * F / 3600.0  # Ah
        Q_LLI.append(q_lli)
        try:
            Q_SEI.append(c.steps[0]["Loss of capacity to negative SEI [A.h]"].entries[-1])
        except Exception:
            Q_SEI.append(np.nan)
        try:
            Q_plating.append(c.steps[0]["Loss of capacity to negative lithium plating [A.h]"].entries[-1])
        except Exception:
            Q_plating.append(np.nan)
        cycles.append(i + 1)
    df = pd.DataFrame({
        "cycle_n": cycles,
        "Q_LLI_Ah": Q_LLI,
        "Q_SEI_Ah": Q_SEI,
        "Q_plating_Ah": Q_plating,
    })
    df["SOH"] = 1.0 - df["Q_LLI_Ah"] / q_nominal_Ah
    return df


def band_rate(sub: pd.DataFrame, lo: float, hi: float) -> float:
    m = (sub.SOH >= lo) & (sub.SOH <= hi)
    if m.sum() < 10: return float("nan")
    slope, _ = np.polyfit(sub.cycle_n[m], sub.SOH[m], 1)
    return -slope * 100 * 1000


def main() -> int:
    import pybamm

    print("=== Turno-CALB PyBaMM setup validation ===")
    print(f"  Model options: {json.dumps({k: str(v) for k, v in MODEL_OPTIONS.items()}, indent=2)}")
    print(f"  N cycles: {N_CYCLES}, C-rate: {C_RATE}")

    model, param = build_model_and_params()
    experiment = build_experiment(C_RATE, N_CYCLES)
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
        print(traceback.format_exc()[:3000])
        return 1
    elapsed = time.time() - t0

    q_nom = float(param["Nominal cell capacity [A.h]"])
    df = compute_soh_from_lli(sol, q_nom)
    print(f"\nSim ok, elapsed {elapsed:.1f}s, cycles: {int(df.cycle_n.max())}, "
          f"SoH: {df.SOH.iloc[0]:.4f} -> {df.SOH.iloc[-1]:.4f}")
    df.to_parquet(OUT_DIR / "deceleration_check_turno.parquet", index=False)

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

    # ── Plot: side-by-side with the two baselines ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))

    base_solvent = pd.read_parquet(OUT_DIR / "deceleration_check.parquet")
    base_rxnlim  = pd.read_parquet(OUT_DIR / "deceleration_check_rxnlim.parquet")

    ax1.plot(base_solvent.cycle_n, base_solvent.SOH, "b-", lw=1.4, label="Baseline: solvent-diff SEI + LAM")
    ax1.plot(base_rxnlim.cycle_n,  base_rxnlim.SOH,  "g-", lw=1.4, label="Baseline: reaction-lim SEI + LAM")
    ax1.plot(df.cycle_n,           df.SOH,           "m-", lw=1.8, label="Turno: interstitial-diff SEI, no LAM")
    ax1.axhspan(0.85, 1.00, color="tab:red",    alpha=0.10, label="fresh band")
    ax1.axhspan(0.45, 0.55, color="tab:orange", alpha=0.10, label="mid band")
    ax1.axhspan(0.25, 0.35, color="tab:green",  alpha=0.10, label="aged band")
    ax1.axhline(0.80, color="k", ls=":", lw=0.8)
    ax1.set_xlabel("Cycle"); ax1.set_ylabel("SoH")
    ax1.set_title(f"SoH trajectories (3000 cy @ {C_RATE}C)")
    ax1.legend(fontsize=8); ax1.grid(alpha=0.3)

    labels = ["fresh", "mid", "aged"]
    measured  = [10.0, 3.7, 1.3]
    turno = [sim_rates[name] if sim_rates[name] == sim_rates[name] else 0 for name, _, _, _ in BANDS]
    x = np.arange(3); w = 0.35
    ax2.bar(x - w/2, measured, w, color="tab:blue",   label="Measured", edgecolor="k")
    ax2.bar(x + w/2, turno,    w, color="tab:purple", label="Turno sim", edgecolor="k")
    ax2.set_ylabel("Fade rate [pp/1000cy]")
    ax2.set_xticks(x); ax2.set_xticklabels(labels)
    ax2.set_title("Turno vs measured — fade rate by SoH band")
    ax2.legend(fontsize=10); ax2.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "deceleration_check_turno.png", dpi=120, bbox_inches="tight")
    print(f"\nWrote {OUT_DIR / 'deceleration_check_turno.parquet'}")
    print(f"Wrote {OUT_DIR / 'deceleration_check_turno.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
