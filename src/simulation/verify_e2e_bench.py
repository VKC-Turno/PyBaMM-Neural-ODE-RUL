"""
Benchmark a single 150-cycle PyBaMM sim with the smaller mesh to size
the Phase-2 differential-evolution optimizer budget.
"""
from __future__ import annotations

import gc
import sys
import time
from pathlib import Path

sys.path.insert(0, "/home/hj/Desktop/PINNs")

import numpy as np
import pandas as pd
import pybamm
import yaml

from src.simulation._pybamm_setup import build_parameter_values


OUT_DIR = Path("/home/hj/Desktop/PINNs/data/synthetic/verification")

MODEL_OPTIONS = {
    "SEI": "solvent-diffusion limited",
    "SEI porosity change": "true",
    "lithium plating": "irreversible",
}

def _submesh_pts():
    v = pybamm.standard_spatial_vars
    return {v.x_n: 16, v.x_s: 8, v.x_p: 16, v.r_n: 8, v.r_p: 8}


def bol_overrides_from_cell(bol_yaml: Path) -> dict:
    cfg = yaml.safe_load(bol_yaml.read_text())
    over: dict = {}
    st = cfg["stoichiometry"]
    base = pybamm.ParameterValues("Prada2013")
    cn = float(base["Maximum concentration in negative electrode [mol.m-3]"])
    cp = float(base["Maximum concentration in positive electrode [mol.m-3]"])
    over["Initial concentration in negative electrode [mol.m-3]"] = st["x_100"] * cn
    over["Initial concentration in positive electrode [mol.m-3]"] = st["y_100"] * cp
    sei = cfg.get("sei") or {}
    if "k_SEI_max_m_per_s" in sei:
        over["SEI kinetic rate constant [m.s-1]"] = float(sei["k_SEI_max_m_per_s"])
    # Isothermal 25 C
    over["Ambient temperature [K]"] = 298.15
    over["Initial temperature [K]"] = 298.15
    return over


def build_experiment(n_cycles: int, c_rate: float = 0.5) -> pybamm.Experiment:
    block = (
        f"Discharge at {c_rate:.4f}C until 2.5 V",
        "Rest for 10 minutes",
        f"Charge at {c_rate:.4f}C until 3.65 V",
        "Hold at 3.65 V until C/100",
        "Rest for 10 minutes",
    )
    return pybamm.Experiment([block] * int(n_cycles))


def extract_soh_trace(sol) -> np.ndarray:
    """Per-cycle discharge Q (Ah), normalised to first cycle."""
    caps = []
    for cy in sol.cycles:
        disc = None
        for step in cy.steps:
            try:
                Imean = float(np.nanmean(step["Current [A]"].entries))
            except Exception:
                continue
            if Imean < -1e-3:
                disc = step
                break
        if disc is None:
            caps.append(np.nan)
            continue
        Q = disc["Discharge capacity [A.h]"].entries
        caps.append(abs(float(Q[-1] - Q[0])))
    caps = np.array(caps, dtype=float)
    if caps.size == 0 or not np.isfinite(caps[0]) or caps[0] <= 0:
        return caps
    return caps / caps[0]


def bench_single(n_cycles: int = 150, c_rate: float = 0.5) -> None:
    print(f"=== Benchmark: {n_cycles}-cycle DFN sim @ {c_rate}C ===", flush=True)

    overrides = bol_overrides_from_cell(OUT_DIR / "eve_0008_bol_params.yaml")
    # add representative degradation overrides (mid-range starting guess)
    overrides.update({
        "SEI kinetic rate constant [m.s-1]": 1e-13,
        "SEI partial molar volume [m3.mol-1]": 1e-4,
        "SEI solvent diffusivity [m2.s-1]": 2.5e-22,
        "Lithium plating kinetic rate constant [m.s-1]": 1e-11,
    })
    param = build_parameter_values(overrides=overrides)

    model = pybamm.lithium_ion.DFN(options=MODEL_OPTIONS)
    solver = pybamm.IDAKLUSolver(rtol=1e-6, atol=1e-6)
    exp = build_experiment(n_cycles, c_rate=c_rate)

    t0 = time.time()
    var_pts = _submesh_pts()
    sim = pybamm.Simulation(
        model, parameter_values=param, experiment=exp, solver=solver,
        var_pts=var_pts,
    )
    sol = sim.solve()
    dt = time.time() - t0

    soh = extract_soh_trace(sol)
    print(f"  wall={dt:.1f}s  ({dt/n_cycles*1000:.0f} ms/cycle)  "
          f"SoH first={soh[0]:.4f}  last={soh[-1]:.4f}  "
          f"delta pp={(1-soh[-1])*100:.3f}", flush=True)
    del sol, sim, exp
    gc.collect()


if __name__ == "__main__":
    bench_single()
