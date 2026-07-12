"""PyBaMM-driven synthetic trajectory generator for PINN augmentation.

Sweeps a parameter grid:
  - k_SEI values × 5-6 → varied fade rates
  - pre_age_to_soh × 3 → fresh, mid-life, used
  - c_rate × 3 → different cycling protocols (0.25, 0.5, 1.0)
  - Base char donor × 3 → cohort-median CALB / EVE / REPT

Each combo yields a PyBaMM cycling trajectory (≥1000 cycles). Output
schema matches the CALB canonical parquet (cohort, cell_id, cycle_no,
soh, ir_ohm, c_rate, etc.) so the combined data loader can treat
synthetic + real cells uniformly.

Reuses the existing pybamm_tuning.Simulation cache (~/Voltaris/outputs/
pybamm_cache/) so repeated runs with the same fingerprint are free.
"""
from __future__ import annotations
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/hj/Desktop/PINNs")
from pybamm_tuning import build_pybamm_parameters, load_characterization
from pybamm_tuning.simulation import CyclingProtocol, Simulation


CACHE_DIR = Path("/home/hj/Desktop/PINNs/Voltaris/outputs/pybamm_cache")
OUT_DIR   = Path("/home/hj/Desktop/PINNs/Voltaris/outputs/synthetic")
OUT_DIR.mkdir(parents=True, exist_ok=True)

RXNLIM_OPTS = {
    "SEI": "reaction limited",
    "SEI porosity change": "true",
    "lithium plating": "none",
    "loss of active material": "none",
}
KEY_J = "SEI reaction exchange current density [A.m-2]"


@dataclass
class SweepPoint:
    make: str            # donor cohort: "CALB_old", "EVE", "REPT"
    k_SEI: float         # SEI reaction current density
    pre_age: float       # initial SoH (0.5-1.0)
    c_rate: float        # cycling C-rate
    n_cycles: int = 1000

    def synth_id(self) -> str:
        """Unique identifier for this synthetic trajectory."""
        return (f"{self.make}_k{self.k_SEI:.2e}"
                f"_a{self.pre_age:.2f}_c{self.c_rate:.2f}").replace(".", "p")


def default_sweep_grid() -> list[SweepPoint]:
    """Modest grid: 3 makes × 4 k_SEI × 3 pre_age × 3 c_rate = 108 points."""
    grid = []
    makes = ["CALB_old", "EVE", "REPT"]
    k_seis = [1e-7, 3e-7, 1e-6, 3e-6]        # rxn-limited SEI current density range
    pre_ages = [0.95, 0.80, 0.65]             # fresh, mid, used
    c_rates = [0.25, 0.50, 1.00]

    for make in makes:
        for k in k_seis:
            for a in pre_ages:
                for c in c_rates:
                    grid.append(SweepPoint(
                        make=make, k_SEI=k, pre_age=a, c_rate=c,
                        n_cycles=1000,
                    ))
    return grid


def small_sweep_grid() -> list[SweepPoint]:
    """Pilot: 3 makes × 2 k_SEI × 2 pre_age × 2 c_rate = 24 points."""
    grid = []
    for make in ["CALB_old", "EVE", "REPT"]:
        for k in [3e-7, 1e-6]:
            for a in [0.90, 0.70]:
                for c in [0.25, 0.50]:
                    grid.append(SweepPoint(
                        make=make, k_SEI=k, pre_age=a, c_rate=c,
                        n_cycles=800,
                    ))
    return grid


def _run_one(pt: SweepPoint) -> Optional[pd.DataFrame]:
    """Run one PyBaMM sim; return trajectory DataFrame in canonical schema."""
    try:
        char = load_characterization(cohort=pt.make, aggregate=True)
    except Exception as e:
        print(f"  [{pt.synth_id()}] SKIP char: {e}")
        return None

    try:
        params = build_pybamm_parameters(
            char, base="Prada2013", temperature_K=298.15,
            extra_overrides={KEY_J: pt.k_SEI},
            pre_age_to_soh=pt.pre_age,
        )
    except Exception as e:
        print(f"  [{pt.synth_id()}] SKIP params: {e}")
        return None

    protocol = CyclingProtocol(c_rate=pt.c_rate)
    sim = Simulation(params, protocol=protocol, cache_dir=CACHE_DIR,
                       dfn_options=RXNLIM_OPTS)
    try:
        df = sim.run(n_cycles=pt.n_cycles)
    except Exception as e:
        print(f"  [{pt.synth_id()}] SKIP sim: {e}")
        return None

    # Convert to canonical schema
    out = pd.DataFrame(dict(
        cohort="SYNTH",
        cell_id=pt.synth_id(),
        manufacturer=pt.make,
        chemistry="LFP",
        nominal_cap_ah=float(char.q_rpt_ah) if hasattr(char, "q_rpt_ah") else 100.0,
        batch=1,
        cycle_no=df.cycle_n.values,
        global_cycle=df.cycle_n.values,
        dchg_cap_ah=df.SOH.values * (float(char.q_rpt_ah) if hasattr(char, "q_rpt_ah") else 100.0),
        chg_cap_ah=df.SOH.values * (float(char.q_rpt_ah) if hasattr(char, "q_rpt_ah") else 100.0),
        ir_ohm=np.nan,
        c_rate=pt.c_rate,
        d_rate=pt.c_rate,
        dod_low=0.0,
        dod_high=1.0,
        ambient_c=25.0,
        soh=df.SOH.values,
        quality_flag="synthetic",
    ))
    # Metadata columns for down-stream filtering
    out["synth_k_SEI"] = pt.k_SEI
    out["synth_pre_age"] = pt.pre_age
    return out


def generate(grid: list[SweepPoint], out_path: Optional[Path] = None) -> pd.DataFrame:
    """Run all sweep points, return concatenated DataFrame."""
    import time
    dfs = []
    total_t = time.time()
    for i, pt in enumerate(grid):
        t0 = time.time()
        df = _run_one(pt)
        elapsed = time.time() - t0
        if df is None:
            continue
        dfs.append(df)
        print(f"  [{i+1}/{len(grid)}] {pt.synth_id():>55s}  N={len(df):>4}  "
              f"SoH {df.soh.iloc[0]:.3f}->{df.soh.iloc[-1]:.3f}  ({elapsed:.1f}s)")

    if not dfs:
        raise RuntimeError("No trajectories generated")
    combined = pd.concat(dfs, ignore_index=True)
    print(f"\nTotal wall-time: {(time.time()-total_t):.1f}s")
    print(f"Total trajectories: {combined.cell_id.nunique()}")
    print(f"Total rows: {len(combined)}")

    if out_path:
        combined.to_parquet(out_path, index=False)
        print(f"Saved: {out_path}")
    return combined


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--pilot", action="store_true", help="use small 24-point grid")
    p.add_argument("--out", default=str(OUT_DIR / "synthetic_trajectories.parquet"))
    args = p.parse_args()

    grid = small_sweep_grid() if args.pilot else default_sweep_grid()
    print(f"=== Synthetic PyBaMM sweep ({len(grid)} points) ===\n")
    generate(grid, out_path=Path(args.out))
