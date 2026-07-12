"""
Phase 2: Fit PyBaMM degradation parameters against the 150-cycle
measured SoH trajectory of EVE cell 0008.

Free parameters (4):
    SEI kinetic rate constant [m.s-1]       log10-space [1e-15, 5e-12]
    SEI partial molar volume [m3.mol-1]     linear      [5e-5, 2e-4]
    SEI solvent diffusivity [m2.s-1]        log10-space [1e-23, 1e-20]
    Lithium plating kinetic rate constant   log10-space [1e-12, 1e-9]

Optimizer: scipy.optimize.differential_evolution (workers=1)

Cost function: RMSE between simulated normalized SoH (Q_sim / Q_sim[0])
and measured normalized SoH (soh_meas / soh_meas[0]) over cycles 1..150.

Writes:
    data/synthetic/verification/eve_0008_deg_params.yaml
    data/synthetic/verification/eve_0008_phase2_progress.csv
    data/synthetic/verification/eve_0008_phase2_log.txt
"""
from __future__ import annotations

import gc
import json
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, "/home/hj/Desktop/PINNs")

import numpy as np
import pandas as pd
import pybamm
import yaml
from scipy.optimize import differential_evolution

from src.simulation._pybamm_setup import build_parameter_values


OUT_DIR = Path("/home/hj/Desktop/PINNs/data/synthetic/verification")
BOL_YAML = OUT_DIR / "eve_0008_bol_params.yaml"
PROGRESS_CSV = OUT_DIR / "eve_0008_phase2_progress.csv"
LOG_TXT = OUT_DIR / "eve_0008_phase2_log.txt"
DEG_YAML = OUT_DIR / "eve_0008_deg_params.yaml"

CELL = "0008"
C_RATE = 0.5           # from Longterm CSV column
N_CYCLES = 150
MAX_WALLTIME_S = 3.0 * 3600   # hard cap for the whole DE run
MEM_LIMIT_GB_ABORT = 0.8      # abort if system available memory < 0.8 GB
                              # (lowered from 4.0 GB — using SPMe cuts our
                              # per-eval RSS to ~500 MB and there is 40+ GB
                              # of swap available; a concurrent sweep is
                              # holding ~60 GB of RAM)

MODEL_OPTIONS = {
    "SEI": "solvent-diffusion limited",
    "SEI porosity change": "true",
    "lithium plating": "irreversible",
}

# DE parameter bounds (in optimizer space)
BOUNDS = [
    (-15.0, np.log10(5e-12)),   # log10 SEI kinetic rate constant
    (5e-5, 2e-4),               # SEI partial molar volume (linear)
    (-23.0, -20.0),             # log10 SEI solvent diffusivity
    (-12.0, -9.0),              # log10 plating kinetic rate constant
]

# DE budget — 4 params × popsize=6 = 24 individuals → 20 iterations
# → ~504 evals × ~10s = ~85 minutes. Well within the 3h cap.
DE_MAXITER = 20
DE_POPSIZE = 6


LOG_LINES: list[str] = []


def log(msg: str) -> None:
    print(msg, flush=True)
    LOG_LINES.append(msg)


def sys_mem_available_gb() -> float:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return float(line.split()[1]) / (1024 * 1024)
    except Exception:
        return float("nan")
    return float("nan")


def _submesh_pts():
    v = pybamm.standard_spatial_vars
    return {v.x_n: 16, v.x_s: 8, v.x_p: 16, v.r_n: 8, v.r_p: 8}


def bol_overrides() -> dict:
    cfg = yaml.safe_load(BOL_YAML.read_text())
    st = cfg["stoichiometry"]
    base = pybamm.ParameterValues("Prada2013")
    cn = float(base["Maximum concentration in negative electrode [mol.m-3]"])
    cp = float(base["Maximum concentration in positive electrode [mol.m-3]"])
    return {
        "Initial concentration in negative electrode [mol.m-3]": st["x_100"] * cn,
        "Initial concentration in positive electrode [mol.m-3]": st["y_100"] * cp,
        "Ambient temperature [K]": 298.15,
        "Initial temperature [K]": 298.15,
    }


def build_experiment(n_cycles: int, c_rate: float) -> pybamm.Experiment:
    block = (
        f"Discharge at {c_rate:.4f}C until 2.5 V",
        "Rest for 10 minutes",
        f"Charge at {c_rate:.4f}C until 3.65 V",
        "Hold at 3.65 V until C/100",
        "Rest for 10 minutes",
    )
    return pybamm.Experiment([block] * int(n_cycles))


def extract_soh(sol) -> np.ndarray:
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


def load_measured_soh() -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_parquet("/home/hj/Desktop/PINNs/soh/data/canonical/eve.parquet")
    s = df[df.cell_id == CELL].sort_values("global_cycle").reset_index(drop=True)
    cycles = s["global_cycle"].to_numpy(int)
    soh = s["soh"].to_numpy(float)
    # Normalize so first measured cycle = 1.0
    soh_norm = soh / soh[0]
    return cycles, soh_norm


class CostEvaluator:
    def __init__(self):
        self.eval_i = 0
        self.t_start = time.time()
        self.rows: list[dict] = []
        self.best_rmse = float("inf")
        self.best_x = None
        self.aborted = False
        self.abort_reason: str | None = None
        self.meas_cycles, self.meas_soh_norm = load_measured_soh()
        # Use SPMe (not DFN) for the fit — captures SEI+plating degradation
        # shape at 5 s/eval and ~500 MB RSS, versus DFN's 9 s/eval and ~3 GB
        # RSS. The identified degradation parameters are then applied to a
        # full DFN sim in Phase 3, which is the physically defensible one.
        self.model = pybamm.lithium_ion.SPMe(options=MODEL_OPTIONS)
        self.solver = pybamm.IDAKLUSolver(rtol=1e-6, atol=1e-6)
        self.var_pts = _submesh_pts()
        log(f"Measured trace: {len(self.meas_cycles)} cycles, "
            f"soh_norm range [{self.meas_soh_norm.min():.4f}, "
            f"{self.meas_soh_norm.max():.4f}]")

    def _params_from_x(self, x) -> dict:
        return {
            "SEI kinetic rate constant [m.s-1]": float(10 ** x[0]),
            "SEI partial molar volume [m3.mol-1]": float(x[1]),
            "SEI solvent diffusivity [m2.s-1]": float(10 ** x[2]),
            "Lithium plating kinetic rate constant [m.s-1]": float(10 ** x[3]),
        }

    def __call__(self, x) -> float:
        self.eval_i += 1
        # Memory guard
        mem_gb = sys_mem_available_gb()
        if mem_gb < MEM_LIMIT_GB_ABORT:
            self.aborted = True
            self.abort_reason = f"memory below {MEM_LIMIT_GB_ABORT} GB (was {mem_gb:.1f})"
            log(f"[eval {self.eval_i:04d}] ABORT: available memory {mem_gb:.1f} GB")
            return 1e6
        if time.time() - self.t_start > MAX_WALLTIME_S:
            self.aborted = True
            self.abort_reason = "wall-time cap exceeded"
            log(f"[eval {self.eval_i:04d}] ABORT: wall-time cap")
            return 1e6

        params = self._params_from_x(x)
        overrides = bol_overrides()
        overrides.update(params)

        t0 = time.time()
        try:
            pv = build_parameter_values(overrides=overrides)
            exp = build_experiment(N_CYCLES, C_RATE)
            sim = pybamm.Simulation(
                self.model, parameter_values=pv, experiment=exp,
                solver=self.solver, var_pts=self.var_pts,
            )
            sol = sim.solve()
            soh_sim = extract_soh(sol)
            del sol, sim, exp, pv
            gc.collect()
        except Exception as e:
            del_msg = f"solver-fail:{type(e).__name__}"
            log(f"[eval {self.eval_i:04d}] FAIL: {del_msg} — "
                f"k_SEI={params['SEI kinetic rate constant [m.s-1]']:.2e}, "
                f"D_sei={params['SEI solvent diffusivity [m2.s-1]']:.2e}")
            self.rows.append({
                "eval": self.eval_i,
                "elapsed_s": time.time() - t0,
                "wall_since_start_s": time.time() - self.t_start,
                "x0_log_k_SEI": x[0], "x1_V_SEI": x[1],
                "x2_log_D_SEI": x[2], "x3_log_k_plating": x[3],
                "rmse_pp": np.nan,
                "sim_final_soh_norm": np.nan,
                "fail": del_msg,
            })
            gc.collect()
            return 1.0  # large penalty but not infinity

        # Trim to matching cycles (sim = 1..N_CYCLES → indices 0..N-1)
        n = min(len(soh_sim), len(self.meas_cycles))
        if n < 10:
            self.rows.append({
                "eval": self.eval_i,
                "elapsed_s": time.time() - t0,
                "wall_since_start_s": time.time() - self.t_start,
                "x0_log_k_SEI": x[0], "x1_V_SEI": x[1],
                "x2_log_D_SEI": x[2], "x3_log_k_plating": x[3],
                "rmse_pp": np.nan,
                "sim_final_soh_norm": np.nan,
                "fail": "too-few-cycles",
            })
            return 1.0

        # Both traces are normalised to their own cycle-1 (=1.0)
        residual = soh_sim[:n] - self.meas_soh_norm[:n]
        rmse = float(np.sqrt(np.nanmean(residual ** 2)))
        rmse_pp = rmse * 100.0
        dt = time.time() - t0

        row = {
            "eval": self.eval_i,
            "elapsed_s": dt,
            "wall_since_start_s": time.time() - self.t_start,
            "x0_log_k_SEI": x[0], "x1_V_SEI": x[1],
            "x2_log_D_SEI": x[2], "x3_log_k_plating": x[3],
            "rmse_pp": rmse_pp,
            "sim_final_soh_norm": float(soh_sim[n - 1]),
            "fail": "",
        }
        self.rows.append(row)

        marker = ""
        if rmse < self.best_rmse:
            self.best_rmse = rmse
            self.best_x = np.array(x, copy=True)
            marker = "  <-- best"
        if self.eval_i % 5 == 0 or marker:
            log(
                f"[eval {self.eval_i:04d}] {dt:5.1f}s  RMSE={rmse_pp:6.3f}pp  "
                f"sim_end={soh_sim[n-1]:.4f}  "
                f"k_SEI={10**x[0]:.2e}  D_SEI={10**x[2]:.2e}  "
                f"V_SEI={x[1]:.2e}  k_plt={10**x[3]:.2e}{marker}"
            )
        return rmse


def run_phase2() -> dict:
    log(f"=== Phase 2: DE fit against 150-cycle SoH ===")
    log(f"System memory available at start: {sys_mem_available_gb():.1f} GB")

    evaluator = CostEvaluator()

    def callback(x, convergence=None):
        # early-stop if the optimizer already found something excellent
        if evaluator.best_rmse < 0.002:  # 0.2 pp
            log(f"  callback: RMSE < 0.2 pp, stopping DE early")
            return True
        return False

    try:
        result = differential_evolution(
            evaluator, BOUNDS,
            maxiter=DE_MAXITER,
            popsize=DE_POPSIZE,
            seed=42,
            workers=1,
            polish=False,
            tol=1e-4,
            callback=callback,
            updating="deferred",
        )
        de_success = True
        de_message = result.message
    except Exception as e:
        de_success = False
        de_message = f"DE crashed: {type(e).__name__}: {e}"
        log(f"DE crashed: {traceback.format_exc()}")
        result = None

    # Persist progress table
    prog = pd.DataFrame(evaluator.rows)
    prog.to_csv(PROGRESS_CSV, index=False)
    log(f"\nWrote progress table: {PROGRESS_CSV}")

    if evaluator.best_x is None:
        log("No successful evaluation. Falling back to Prada+OKane defaults.")
        best_params = {
            "SEI kinetic rate constant [m.s-1]": 1e-12,
            "SEI partial molar volume [m3.mol-1]": 1e-4,
            "SEI solvent diffusivity [m2.s-1]": 2.5e-22,
            "Lithium plating kinetic rate constant [m.s-1]": 1e-11,
        }
        best_rmse_pp = float("nan")
    else:
        best_params = evaluator._params_from_x(evaluator.best_x)
        best_rmse_pp = evaluator.best_rmse * 100.0
        log(f"\nBest RMSE = {best_rmse_pp:.3f} pp")
        for k, v in best_params.items():
            log(f"  {k}: {v:.4e}")

    # Identifiability diagnostics: report per-parameter spread of the top-10%
    ident: dict = {}
    if not prog.empty and "fail" in prog.columns:
        df_ok = prog[prog["fail"] == ""].copy()
        if not df_ok.empty and not np.isnan(best_rmse_pp):
            cut = df_ok["rmse_pp"].quantile(0.10)
            top = df_ok[df_ok["rmse_pp"] <= cut]
            if not top.empty:
                names = ["x0_log_k_SEI", "x1_V_SEI",
                        "x2_log_D_SEI", "x3_log_k_plating"]
                labels = [
                    "SEI kinetic rate constant [m.s-1] (log10)",
                    "SEI partial molar volume [m3.mol-1]",
                    "SEI solvent diffusivity [m2.s-1] (log10)",
                    "Lithium plating kinetic rate [m.s-1] (log10)",
                ]
                for i, (col, label) in enumerate(zip(names, labels)):
                    lo, hi = float(top[col].min()), float(top[col].max())
                    bnd_lo, bnd_hi = float(BOUNDS[i][0]), float(BOUNDS[i][1])
                    span_frac = (hi - lo) / max(1e-9, (bnd_hi - bnd_lo))
                    ident[label] = {
                        "top10pct_range": [lo, hi],
                        "span_of_full_range": float(span_frac),
                        "well_identified": bool(span_frac < 0.25),
                    }

    out = {
        "cell_id": CELL,
        "n_measured_cycles": int(N_CYCLES),
        "n_evaluations": int(evaluator.eval_i),
        "n_successful_evaluations": int((prog["fail"] == "").sum()) if not prog.empty else 0,
        "wall_time_s": float(time.time() - evaluator.t_start),
        "de_maxiter": DE_MAXITER,
        "de_popsize": DE_POPSIZE,
        "de_success": bool(de_success),
        "de_message": str(de_message),
        "aborted_early": bool(evaluator.aborted),
        "abort_reason": evaluator.abort_reason,
        "best_rmse_pp": float(best_rmse_pp) if not np.isnan(best_rmse_pp) else None,
        "best_parameters": {k: float(v) for k, v in best_params.items()},
        "identifiability": ident,
        "notes": (
            "Cost = RMSE(sim SoH normalized to sim cycle 1, "
            "measured SoH normalized to measured cycle 1) over 150 cycles."
        ),
    }

    with open(DEG_YAML, "w") as f:
        yaml.safe_dump(out, f, sort_keys=False)
    log(f"Wrote degradation parameters: {DEG_YAML}")

    LOG_TXT.write_text("\n".join(LOG_LINES))
    return out


if __name__ == "__main__":
    run_phase2()
