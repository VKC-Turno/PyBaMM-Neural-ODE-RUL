"""
Per-cell end-to-end verification driver.

Runs all three phases of the workflow originally proved out on EVE cell
0008, for any EVE cell:
  Phase 1: BOL identification from OCV/GITT/HPPC/DCIR (+ SelfDischarge if present)
  Phase 2: Fit 4 degradation parameters against measured Longterm SoH
  Phase 3: 5000-cycle DFN long-run using the fitted parameters

Usage
-----
    .venv/bin/python src/simulation/verify_e2e_percell.py --cell 0002 \\
        [--phase1] [--phase2] [--phase3] [--all]

If no phase flag is supplied, all three phases are run in sequence.

Outputs
-------
    data/synthetic/verification/per_cell/<cell>/
        bol_params.yaml
        deg_params.yaml
        longrun.parquet
        validation.png
        log.txt
        phase2_progress.csv     # DE eval progress (for identifiability)
        phase3_summary.yaml     # phase3 top-level metrics
"""
from __future__ import annotations

import argparse
import gc
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.param_id.ocv_fit import fit_one_cell as ocv_fit_one
from src.param_id.dcir_hppc import extract_pulses
from src.param_id.gitt_ds import extract_gitt_step_metrics
from src.param_id.sei_selfdisc import fit_one_cell as sei_fit_one
from src.simulation._pybamm_setup import build_parameter_values


ROOT = Path("/home/hj/Desktop/PINNs")
BASE_OUT = ROOT / "data/synthetic/verification/per_cell"

# --- Simulation config (shared across cells) ---
C_RATE = 0.5
MODEL_OPTIONS = {
    "SEI": "solvent-diffusion limited",
    "SEI porosity change": "true",
    "lithium plating": "irreversible",
}

# Phase 2 DE budget
DE_BOUNDS = [
    (-15.0, np.log10(5e-12)),   # log10 SEI kinetic rate constant
    (5e-5, 2e-4),               # SEI partial molar volume (linear)
    (-23.0, -20.0),             # log10 SEI solvent diffusivity
    (-12.0, -9.0),              # log10 plating kinetic rate constant
]
DE_MAXITER = 20
DE_POPSIZE = 6
PHASE2_WALLTIME_S = 30 * 60      # 30 min per cell hard cap
PHASE2_MEM_LIMIT_GB = 0.8

# Phase 3
CYCLES_PER_BATCH = 1000
MAX_CYCLES = 5000
SOH_STOP = 0.35
PHASE3_WALLTIME_S = 60 * 60


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


# -------------------------------- Phase 1 --------------------------------

def phase1(cell: str, out_dir: Path) -> dict:
    log(f"\n==================== Phase 1: BOL identification (cell {cell}) ====================")
    result: dict = {"cell_id": cell, "notes": []}

    # (a) OCV fit
    log("=== Phase 1a: OCV stoichiometry fit ===")
    ocv = ocv_fit_one(cell)
    log(
        f"  x_100={ocv.x_100:.4f} x_0={ocv.x_0:.4f} "
        f"y_100={ocv.y_100:.4f} y_0={ocv.y_0:.4f}  "
        f"rmse={ocv.rmse_mV:.2f} mV"
    )
    log(f"  Q_dchg={ocv.Q_dchg_Ah:.3f} Ah, Q_n={ocv.Q_n_init_Ah:.3f} Ah, "
        f"Q_p={ocv.Q_p_init_Ah:.3f} Ah")
    result["stoichiometry"] = {
        "x_100": float(ocv.x_100),
        "x_0": float(ocv.x_0),
        "y_100": float(ocv.y_100),
        "y_0": float(ocv.y_0),
        "ocv_rmse_mV": float(ocv.rmse_mV),
        "_source": "src/param_id/ocv_fit.py against Prada2013 half-cells",
    }
    result["capacity"] = {
        "Q_dchg_measured_Ah": float(ocv.Q_dchg_Ah),
        "Q_n_init_Ah": float(ocv.Q_n_init_Ah),
        "Q_p_init_Ah": float(ocv.Q_p_init_Ah),
        "_source": "derived from OCV stoichiometry + measured OCVSOC discharge Q",
    }

    # (b) HPPC RC pulse fits
    log("\n=== Phase 1b: HPPC RC pulse fit ===")
    pulses = extract_pulses(cell, "HPPC", Q_nominal_Ah=float(ocv.Q_dchg_Ah))
    pulse_df = pd.DataFrame([vars(p) for p in pulses])
    if pulse_df.empty:
        log("  NO HPPC pulses extracted — falling back to DCIR")
        pulses = extract_pulses(cell, "DCIR", Q_nominal_Ah=float(ocv.Q_dchg_Ah))
        pulse_df = pd.DataFrame([vars(p) for p in pulses])
    if pulse_df.empty:
        log(f"  No resistance pulses could be extracted for cell {cell}")
        result["resistance"] = {"error": "no pulses found"}
        result["notes"].append("resistance not identified")
    else:
        disc = pulse_df[pulse_df["direction"] == "discharge"]
        if disc.empty:
            disc = pulse_df
        R0 = float(disc["R0_Ohm"].median())
        R1 = float(disc["R1_Ohm"].median())
        tau = float(disc["tau_s"].median())
        C1 = float(disc["C1_F"].median())
        log(f"  n_pulses={len(disc)}  R0={R0*1000:.3f} mOhm  R1={R1*1000:.3f} mOhm  "
            f"tau={tau:.1f} s  C1={C1:.0f} F  "
            f"SOC=[{disc['SOC_est'].min():.3f},{disc['SOC_est'].max():.3f}]")
        result["resistance"] = {
            "R0_Ohm": R0,
            "R1_Ohm": R1,
            "tau_s": tau,
            "C1_F": C1,
            "n_pulses": int(len(disc)),
            "SOC_min": float(disc["SOC_est"].min()),
            "SOC_max": float(disc["SOC_est"].max()),
            "_source": "src/param_id/dcir_hppc.py (RC discharge pulses)",
            "_caveat": "HPPC probes only SOC ~0.97-1.00 for this dataset.",
        }

    # (c) GITT
    log("\n=== Phase 1c: GITT diffusion timescale ===")
    try:
        gitt = extract_gitt_step_metrics(
            cell_id=cell, Q_total_Ah=float(ocv.Q_dchg_Ah),
            diffusion_length_m=None,
        )
        if gitt.empty:
            log("  GITT metrics empty")
            result["diffusion"] = {"note": "no GITT metrics extracted"}
        else:
            dV = float(gitt["dV_dsqrt_t_V_sqrt_s"].median())
            tau_pulse = float(gitt["tau_s"].median())
            r2 = float(gitt["fit_r2"].median())
            log(f"  n_steps={len(gitt)}  dV/dsqrt(t) med={dV:.6f} V/sqrt(s)  "
                f"pulse tau med={tau_pulse:.1f} s  R2 med={r2:.4f}")
            result["diffusion"] = {
                "dV_dsqrt_t_V_per_sqrt_s_median": dV,
                "tau_pulse_s_median": tau_pulse,
                "gitt_fit_r2_median": r2,
                "n_steps": int(len(gitt)),
                "_source": "src/param_id/gitt_ds.py",
                "_caveat": "Full-cell GITT cannot separate D_s_n vs D_s_p; "
                           "Prada2013 defaults retained in Phase 2.",
            }
    except Exception as e:
        log(f"  GITT extraction failed: {type(e).__name__}: {e}")
        result["diffusion"] = {"error": f"{type(e).__name__}: {e}"}

    # (d) Self-discharge
    log("\n=== Phase 1d: Self-discharge SEI ceiling ===")
    try:
        sd = sei_fit_one(cell, Q_nominal_Ah=float(ocv.Q_dchg_Ah))
        log(f"  I_sd={sd.I_sd_uA:.1f} uA  dSOC/dt={sd.dSOC_dt_per_h*100:+.4f} %SOC/h  "
            f"k_SEI_max={sd.k_SEI_max_m_per_s:.3e} m/s")
        result["sei"] = {
            "I_sd_uA": float(sd.I_sd_uA),
            "dSOC_dt_per_h_pct": float(sd.dSOC_dt_per_h * 100),
            "k_SEI_max_m_per_s": float(sd.k_SEI_max_m_per_s),
            "dV_dt_uV_per_s": float(sd.dV_dt_uV_per_s),
            "_source": "src/param_id/sei_selfdisc.py (upper bound)",
            "_caveat": "Bound uses Prada2013 geometric area (0.18 m^2); "
                       "true k_SEI ceiling likely ~30x smaller.",
        }
    except Exception as e:
        log(f"  Self-discharge fit failed: {type(e).__name__}: {e}")
        result["sei"] = {"error": f"{type(e).__name__}: {e}"}
        result["notes"].append("SEI ceiling not identified — using OKane2022 default in Phase 2.")

    out_yaml = out_dir / "bol_params.yaml"
    with open(out_yaml, "w") as f:
        yaml.safe_dump(result, f, sort_keys=False)
    log(f"\nWrote BOL params -> {out_yaml}")
    return result


# -------------------------------- Phase 2 --------------------------------

def load_measured_soh(cell: str) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_parquet(ROOT / "soh/data/canonical/eve.parquet")
    s = df[df.cell_id == cell].sort_values("global_cycle").reset_index(drop=True)
    return s["global_cycle"].to_numpy(int), s["soh"].to_numpy(float)


def _bol_overrides(bol_yaml: Path) -> dict:
    cfg = yaml.safe_load(bol_yaml.read_text())
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


def _build_experiment(n_cycles: int, c_rate: float) -> pybamm.Experiment:
    block = (
        f"Discharge at {c_rate:.4f}C until 2.5 V",
        "Rest for 10 minutes",
        f"Charge at {c_rate:.4f}C until 3.65 V",
        "Hold at 3.65 V until C/100",
        "Rest for 10 minutes",
    )
    return pybamm.Experiment([block] * int(n_cycles))


def _extract_soh_from_sol(sol) -> np.ndarray:
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


class Phase2Evaluator:
    def __init__(self, cell: str, bol_yaml: Path, n_cycles: int,
                 maxiter: int, popsize: int, walltime_s: int,
                 mem_limit_gb: float):
        self.cell = cell
        self.bol_yaml = bol_yaml
        self.n_cycles = n_cycles
        self.maxiter = maxiter
        self.popsize = popsize
        self.walltime_s = walltime_s
        self.mem_limit_gb = mem_limit_gb

        self.eval_i = 0
        self.t_start = time.time()
        self.rows: list[dict] = []
        self.best_rmse = float("inf")
        self.best_x = None
        self.aborted = False
        self.abort_reason: str | None = None
        self.meas_cycles, meas_soh = load_measured_soh(cell)
        self.meas_soh_norm = meas_soh / meas_soh[0]
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
        mem_gb = sys_mem_available_gb()
        if mem_gb < self.mem_limit_gb:
            self.aborted = True
            self.abort_reason = f"memory below {self.mem_limit_gb} GB (was {mem_gb:.1f})"
            log(f"[eval {self.eval_i:04d}] ABORT: mem {mem_gb:.1f} GB")
            return 1e6
        if time.time() - self.t_start > self.walltime_s:
            self.aborted = True
            self.abort_reason = "wall-time cap exceeded"
            log(f"[eval {self.eval_i:04d}] ABORT: wall-time cap")
            return 1e6

        params = self._params_from_x(x)
        overrides = _bol_overrides(self.bol_yaml)
        overrides.update(params)

        t0 = time.time()
        try:
            pv = build_parameter_values(overrides=overrides)
            exp = _build_experiment(self.n_cycles, C_RATE)
            sim = pybamm.Simulation(
                self.model, parameter_values=pv, experiment=exp,
                solver=self.solver, var_pts=self.var_pts,
            )
            sol = sim.solve()
            soh_sim = _extract_soh_from_sol(sol)
            del sol, sim, exp, pv
            gc.collect()
        except Exception as e:
            del_msg = f"solver-fail:{type(e).__name__}"
            log(f"[eval {self.eval_i:04d}] FAIL: {del_msg}")
            self.rows.append({
                "eval": self.eval_i, "elapsed_s": time.time() - t0,
                "wall_since_start_s": time.time() - self.t_start,
                "x0_log_k_SEI": x[0], "x1_V_SEI": x[1],
                "x2_log_D_SEI": x[2], "x3_log_k_plating": x[3],
                "rmse_pp": np.nan, "sim_final_soh_norm": np.nan,
                "fail": del_msg,
            })
            gc.collect()
            return 1.0

        n = min(len(soh_sim), len(self.meas_cycles))
        if n < 10:
            self.rows.append({
                "eval": self.eval_i, "elapsed_s": time.time() - t0,
                "wall_since_start_s": time.time() - self.t_start,
                "x0_log_k_SEI": x[0], "x1_V_SEI": x[1],
                "x2_log_D_SEI": x[2], "x3_log_k_plating": x[3],
                "rmse_pp": np.nan, "sim_final_soh_norm": np.nan,
                "fail": "too-few-cycles",
            })
            return 1.0

        residual = soh_sim[:n] - self.meas_soh_norm[:n]
        rmse = float(np.sqrt(np.nanmean(residual ** 2)))
        rmse_pp = rmse * 100.0
        dt = time.time() - t0

        row = {
            "eval": self.eval_i, "elapsed_s": dt,
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
            log(f"[eval {self.eval_i:04d}] {dt:5.1f}s  RMSE={rmse_pp:6.3f}pp  "
                f"sim_end={soh_sim[n-1]:.4f}  "
                f"k_SEI={10**x[0]:.2e}  D_SEI={10**x[2]:.2e}  "
                f"V_SEI={x[1]:.2e}  k_plt={10**x[3]:.2e}{marker}")
        return rmse


def phase2(cell: str, out_dir: Path, maxiter: int = DE_MAXITER,
            popsize: int = DE_POPSIZE, walltime_s: int = PHASE2_WALLTIME_S) -> dict:
    log(f"\n==================== Phase 2: Degradation fit (cell {cell}) ====================")
    log(f"Memory available: {sys_mem_available_gb():.1f} GB")

    bol_yaml = out_dir / "bol_params.yaml"
    meas_cy, _ = load_measured_soh(cell)
    n_meas = len(meas_cy)
    log(f"Fitting to {n_meas} measured cycles")

    evaluator = Phase2Evaluator(
        cell=cell, bol_yaml=bol_yaml, n_cycles=n_meas,
        maxiter=maxiter, popsize=popsize, walltime_s=walltime_s,
        mem_limit_gb=PHASE2_MEM_LIMIT_GB,
    )

    def callback(x, convergence=None):
        if evaluator.best_rmse < 0.002:
            log("  callback: RMSE < 0.2 pp, stopping DE early")
            return True
        return False

    try:
        result = differential_evolution(
            evaluator, DE_BOUNDS,
            maxiter=maxiter, popsize=popsize, seed=42, workers=1,
            polish=False, tol=1e-4, callback=callback, updating="deferred",
        )
        de_success = True
        de_message = result.message
    except Exception as e:
        de_success = False
        de_message = f"DE crashed: {type(e).__name__}: {e}"
        log(traceback.format_exc()[:800])

    prog = pd.DataFrame(evaluator.rows)
    prog.to_csv(out_dir / "phase2_progress.csv", index=False)

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

    # Identifiability
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
                    bnd_lo, bnd_hi = float(DE_BOUNDS[i][0]), float(DE_BOUNDS[i][1])
                    span_frac = (hi - lo) / max(1e-9, (bnd_hi - bnd_lo))
                    ident[label] = {
                        "top10pct_range": [lo, hi],
                        "span_of_full_range": float(span_frac),
                        "well_identified": bool(span_frac < 0.25),
                    }

    out = {
        "cell_id": cell,
        "n_measured_cycles": int(n_meas),
        "n_evaluations": int(evaluator.eval_i),
        "n_successful_evaluations": int((prog["fail"] == "").sum()) if not prog.empty else 0,
        "wall_time_s": float(time.time() - evaluator.t_start),
        "de_maxiter": maxiter,
        "de_popsize": popsize,
        "de_success": bool(de_success),
        "de_message": str(de_message),
        "aborted_early": bool(evaluator.aborted),
        "abort_reason": evaluator.abort_reason,
        "best_rmse_pp": float(best_rmse_pp) if not np.isnan(best_rmse_pp) else None,
        "best_parameters": {k: float(v) for k, v in best_params.items()},
        "identifiability": ident,
        "notes": (
            "Cost = RMSE(sim SoH normalized to sim cycle 1, "
            "measured SoH normalized to measured cycle 1) over the measured cycles."
        ),
    }
    with open(out_dir / "deg_params.yaml", "w") as f:
        yaml.safe_dump(out, f, sort_keys=False)
    log(f"Wrote degradation parameters: {out_dir / 'deg_params.yaml'}")
    return out


# -------------------------------- Phase 3 --------------------------------

def _p3_build_overrides(bol_yaml: Path, deg_yaml: Path) -> dict:
    bol = yaml.safe_load(bol_yaml.read_text())
    deg = yaml.safe_load(deg_yaml.read_text())
    st = bol["stoichiometry"]
    base = pybamm.ParameterValues("Prada2013")
    cn = float(base["Maximum concentration in negative electrode [mol.m-3]"])
    cp = float(base["Maximum concentration in positive electrode [mol.m-3]"])
    over = {
        "Initial concentration in negative electrode [mol.m-3]": st["x_100"] * cn,
        "Initial concentration in positive electrode [mol.m-3]": st["y_100"] * cp,
        "Ambient temperature [K]": 298.15,
        "Initial temperature [K]": 298.15,
    }
    over.update(deg["best_parameters"])
    return over


def _p3_extract_per_cycle(sol, cycle_offset: int, skip_first: bool) -> pd.DataFrame:
    rows = []
    cycles_iter = sol.cycles[1:] if skip_first else sol.cycles
    for local_n, cycle in enumerate(cycles_iter, start=1):
        disc = None
        for step in cycle.steps:
            try:
                Imean = float(np.nanmean(step["Current [A]"].entries))
            except Exception:
                continue
            if Imean < -1e-3:
                disc = step
                break
        if disc is None:
            continue
        Q = disc["Discharge capacity [A.h]"].entries
        Q_Ah = abs(float(Q[-1] - Q[0]))
        rows.append({"cycle_n": local_n + cycle_offset, "Q_Ah": Q_Ah})
    return pd.DataFrame(rows)


def _p3_cycle_at_soh(df: pd.DataFrame, target: float) -> float | None:
    below = df[df.SOH <= target]
    if below.empty:
        return None
    idx = below.index[0]
    if idx == 0:
        return float(df.cycle_n.iloc[0])
    prev = df.iloc[idx - 1]
    curr = df.iloc[idx]
    if curr.SOH == prev.SOH:
        return float(curr.cycle_n)
    frac = (prev.SOH - target) / (prev.SOH - curr.SOH)
    return float(prev.cycle_n + frac * (curr.cycle_n - prev.cycle_n))


def phase3(cell: str, out_dir: Path) -> dict:
    log(f"\n==================== Phase 3: DFN long-run (cell {cell}) ====================")
    log(f"Memory available: {sys_mem_available_gb():.1f} GB")
    t_total_start = time.time()

    bol_yaml = out_dir / "bol_params.yaml"
    deg_yaml = out_dir / "deg_params.yaml"
    overrides = _p3_build_overrides(bol_yaml, deg_yaml)

    param = build_parameter_values(overrides=overrides)
    model = pybamm.lithium_ion.DFN(options=MODEL_OPTIONS)
    solver = pybamm.IDAKLUSolver(rtol=1e-6, atol=1e-6)
    var_pts = _submesh_pts()

    starting_solution = None
    all_frames = []
    batch_i = 0
    n_batches_max = MAX_CYCLES // CYCLES_PER_BATCH
    meta = {"batch_summaries": [], "aborted": False, "abort_reason": None}

    while batch_i < n_batches_max:
        batch_i += 1
        if time.time() - t_total_start > PHASE3_WALLTIME_S:
            meta["aborted"] = True
            meta["abort_reason"] = "wall-time cap"
            log("HARD-CAP: wall time, stopping")
            break
        mem_gb = sys_mem_available_gb()
        if mem_gb < 1.0:
            meta["aborted"] = True
            meta["abort_reason"] = f"low mem {mem_gb:.1f} GB"
            log(f"HARD-CAP: memory {mem_gb:.1f} GB, stopping")
            break

        log(f"--- Batch {batch_i}/{n_batches_max} ({CYCLES_PER_BATCH} cy) mem={mem_gb:.1f} GB ---")
        t0 = time.time()
        try:
            exp = _build_experiment(CYCLES_PER_BATCH, C_RATE)
            sim = pybamm.Simulation(
                model, parameter_values=param, experiment=exp,
                solver=solver, var_pts=var_pts,
            )
            sol = sim.solve(starting_solution=starting_solution)
        except Exception as e:
            log(f"  Solver error: {type(e).__name__}: {e}")
            log(traceback.format_exc()[:800])
            meta["aborted"] = True
            meta["abort_reason"] = f"solver:{type(e).__name__}"
            gc.collect()
            break

        cycle_offset = sum(b["n_cycles_ok"] for b in meta["batch_summaries"])
        df = _p3_extract_per_cycle(sol, cycle_offset,
                                    skip_first=(starting_solution is not None))
        n_ok = len(df)
        dt = time.time() - t0
        summary = {
            "batch": batch_i, "n_cycles_ok": n_ok, "elapsed_s": dt,
            "Q_Ah_first": float(df["Q_Ah"].iloc[0]) if not df.empty else np.nan,
            "Q_Ah_last": float(df["Q_Ah"].iloc[-1]) if not df.empty else np.nan,
        }
        meta["batch_summaries"].append(summary)
        all_frames.append(df)
        log(f"  batch {batch_i}: {n_ok} cy in {dt:.1f}s  "
            f"Q_Ah {summary['Q_Ah_first']:.3f} -> {summary['Q_Ah_last']:.3f}")

        try:
            starting_solution = sol.last_state
        except AttributeError:
            starting_solution = sol
        del sol, sim, exp
        gc.collect()

        combined = pd.concat(all_frames, ignore_index=True)
        q0 = float(combined["Q_Ah"].iloc[0])
        soh_last = float(combined["Q_Ah"].iloc[-1]) / q0
        log(f"  SoH now = {soh_last:.4f}")
        if soh_last <= SOH_STOP:
            log(f"  Reached SoH_STOP {SOH_STOP} at cy "
                f"{int(combined['cycle_n'].iloc[-1])}, stopping.")
            break

    if not all_frames:
        return {"success": False}

    full = pd.concat(all_frames, ignore_index=True)
    q0 = float(full["Q_Ah"].iloc[0])
    full["SOH"] = full["Q_Ah"] / q0
    meta["q0_Ah"] = q0
    meta["elapsed_total_s"] = time.time() - t_total_start

    parquet_out = out_dir / "longrun.parquet"
    full.to_parquet(parquet_out, index=False)
    log(f"Wrote long-run trajectory: {parquet_out}")

    df_meas_raw = pd.read_parquet(ROOT / "soh/data/canonical/eve.parquet")
    df_meas = df_meas_raw[df_meas_raw.cell_id == cell].sort_values("global_cycle") \
        .reset_index(drop=True)[["global_cycle", "soh"]] \
        .rename(columns={"global_cycle": "cycle_n"})

    eol = _p3_cycle_at_soh(full, 0.80)
    eosl = _p3_cycle_at_soh(full, 0.40)
    _make_validation_plot(cell, full, df_meas, eol, eosl, out_dir / "validation.png")
    log(f"Wrote validation plot: {out_dir / 'validation.png'}")

    d = full["SOH"].diff().dropna()
    monotonic = bool((d <= 1e-4).all())
    n_up_steps = int((d > 1e-4).sum())

    sim_at_cy = np.interp(df_meas["cycle_n"], full["cycle_n"], full["SOH"])
    meas_norm = df_meas["soh"].to_numpy() / df_meas["soh"].to_numpy()[0]
    resid = sim_at_cy - meas_norm
    rmse_pp = float(np.sqrt(np.mean(resid ** 2)) * 100)

    n_meas = len(df_meas)
    sim_delta = float((full["SOH"].iloc[0]
                       - np.interp(df_meas["cycle_n"].iloc[-1],
                                    full["cycle_n"], full["SOH"])) * 100)
    meas_delta = float((df_meas["soh"].iloc[0] - df_meas["soh"].iloc[-1])
                        / df_meas["soh"].iloc[0] * 100)

    log("\nSummary:")
    log(f"  q0 (cy 1 Q, Ah): {meta['q0_Ah']:.3f}")
    log(f"  final SoH: {full['SOH'].iloc[-1]:.4f}")
    log(f"  cycles simulated: {int(full['cycle_n'].iloc[-1])}")
    log(f"  cycle at SoH 0.80 (EoL): {eol:.0f}" if eol else "  cycle at SoH 0.80: not reached")
    log(f"  cycle at SoH 0.40 (EoSL): {eosl:.0f}" if eosl else "  cycle at SoH 0.40: not reached")
    log(f"  Monotonic decreasing: {monotonic} ({n_up_steps} up-steps > 1e-4)")
    log(f"  First-{n_meas}-cycle fit RMSE: {rmse_pp:.3f} pp "
        f"(sim fade {sim_delta:.2f} pp vs meas fade {meas_delta:.2f} pp)")

    result = {
        "success": True,
        "cell_id": cell,
        "q0_Ah": float(meta["q0_Ah"]),
        "n_cycles_simulated": int(full["cycle_n"].iloc[-1]),
        "final_soh": float(full["SOH"].iloc[-1]),
        "cycle_at_soh_0p80": float(eol) if eol else None,
        "cycle_at_soh_0p40": float(eosl) if eosl else None,
        "monotonic_decreasing": monotonic,
        "n_up_steps": n_up_steps,
        "measured_cycles_covered": int(n_meas),
        "measured_window_rmse_pp": rmse_pp,
        "sim_delta_soh_measured_window_pp": sim_delta,
        "meas_delta_soh_pp": meas_delta,
        "batch_summaries": meta["batch_summaries"],
        "elapsed_total_s": meta.get("elapsed_total_s"),
        "aborted": meta["aborted"],
        "abort_reason": meta["abort_reason"],
    }
    with open(out_dir / "phase3_summary.yaml", "w") as f:
        yaml.safe_dump(result, f, sort_keys=False)
    return result


def _make_validation_plot(cell: str, df_sim: pd.DataFrame, df_meas: pd.DataFrame,
                          eol_cycle: float | None, eosl_cycle: float | None,
                          out_path: Path) -> None:
    fig = plt.figure(figsize=(11, 6.5))
    gs = fig.add_gridspec(1, 3, width_ratios=[2.2, 1, 1])
    ax_main = fig.add_subplot(gs[0])
    ax_zoom = fig.add_subplot(gs[1])
    ax_resid = fig.add_subplot(gs[2])

    ax_main.plot(df_sim["cycle_n"], df_sim["SOH"], "b-", lw=1.5,
                 label="Simulated (DFN + SEI + plating)")
    ax_main.scatter(df_meas["cycle_n"], df_meas["soh"] / df_meas["soh"].iloc[0],
                    c="k", s=14, label=f"Measured EVE {cell} (norm)")
    ax_main.axhline(0.80, color="tab:orange", ls="--", lw=0.8, label="EoL 0.80")
    ax_main.axhline(0.40, color="tab:red", ls="--", lw=0.8, label="EoSL 0.40")
    if eol_cycle is not None:
        ax_main.axvline(eol_cycle, color="tab:orange", ls=":", lw=0.7, alpha=0.6)
        ax_main.annotate(f"EoL @ {eol_cycle:.0f}", xy=(eol_cycle, 0.80),
                         xytext=(eol_cycle * 0.6, 0.87), fontsize=9,
                         arrowprops=dict(arrowstyle="->", color="tab:orange", lw=0.6))
    if eosl_cycle is not None:
        ax_main.axvline(eosl_cycle, color="tab:red", ls=":", lw=0.7, alpha=0.6)
        ax_main.annotate(f"EoSL @ {eosl_cycle:.0f}", xy=(eosl_cycle, 0.40),
                         xytext=(eosl_cycle * 0.55, 0.5), fontsize=9,
                         arrowprops=dict(arrowstyle="->", color="tab:red", lw=0.6))
    ax_main.set_xlabel("Cycle")
    ax_main.set_ylabel("SoH")
    ax_main.set_ylim(0, 1.05)
    ax_main.set_title(f"End-to-end workflow validation: EVE cell {cell}\n"
                      "per-cell BOL + fitted deg params -> DFN long run")
    ax_main.grid(alpha=0.3)
    ax_main.legend(loc="lower left", fontsize=9)

    n_meas = len(df_meas)
    zoom_end = max(200, int(df_meas["cycle_n"].max() * 1.1) + 10)
    df_first = df_sim[df_sim["cycle_n"] <= zoom_end]
    ax_zoom.plot(df_first["cycle_n"], df_first["SOH"], "b-", lw=1.4, label="Sim")
    ax_zoom.scatter(df_meas["cycle_n"],
                    df_meas["soh"] / df_meas["soh"].iloc[0],
                    c="k", s=10, label="Meas norm")
    ax_zoom.set_xlabel("Cycle")
    ax_zoom.set_ylabel("SoH (norm)")
    ax_zoom.set_xlim(0, zoom_end)
    ax_zoom.set_title(f"First {n_meas} cy (zoom)", fontsize=10)
    ax_zoom.grid(alpha=0.3)
    ax_zoom.legend(fontsize=8)

    sim_at_cy = np.interp(df_meas["cycle_n"], df_sim["cycle_n"], df_sim["SOH"])
    meas_norm = df_meas["soh"].to_numpy() / df_meas["soh"].to_numpy()[0]
    resid_pp = (sim_at_cy - meas_norm) * 100
    ax_resid.plot(df_meas["cycle_n"], resid_pp, "g-", lw=1.2)
    ax_resid.axhline(0, color="k", lw=0.5)
    ax_resid.set_xlabel("Cycle")
    ax_resid.set_ylabel("Sim - Meas [pp]")
    ax_resid.set_title(f"Residuals ({n_meas} cy)", fontsize=10)
    ax_resid.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# -------------------------------- Main --------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True, help="EVE cell id (e.g. 0002)")
    ap.add_argument("--phase1", action="store_true")
    ap.add_argument("--phase2", action="store_true")
    ap.add_argument("--phase3", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--maxiter", type=int, default=DE_MAXITER)
    ap.add_argument("--popsize", type=int, default=DE_POPSIZE)
    ap.add_argument("--phase2-walltime-s", type=int, default=PHASE2_WALLTIME_S)
    args = ap.parse_args()

    cell = args.cell
    out_dir = BASE_OUT / cell
    out_dir.mkdir(parents=True, exist_ok=True)

    run_all = args.all or not (args.phase1 or args.phase2 or args.phase3)
    t0 = time.time()

    try:
        if args.phase1 or run_all:
            phase1(cell, out_dir)
        if args.phase2 or run_all:
            phase2(cell, out_dir, maxiter=args.maxiter, popsize=args.popsize,
                    walltime_s=args.phase2_walltime_s)
        if args.phase3 or run_all:
            phase3(cell, out_dir)
    finally:
        (out_dir / "log.txt").write_text("\n".join(LOG_LINES))
        log(f"Total wall time: {time.time() - t0:.1f} s")
        (out_dir / "log.txt").write_text("\n".join(LOG_LINES))


if __name__ == "__main__":
    main()
