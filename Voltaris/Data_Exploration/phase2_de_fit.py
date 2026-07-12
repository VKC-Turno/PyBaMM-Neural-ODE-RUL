"""
Phase 2 — per-cell degradation parameter identification via differential
evolution against the measured Longterm SoH trajectory.

Free parameters (5) — brief-signed set:

    k_SEI              [m.s-1]      log10 bounds  [-13, -10]
    V_SEI              [m3.mol-1]   linear bounds [8e-5, 2e-4]
    D_SEI_solvent      [m2.s-1]     log10 bounds  [-22, -18]
    k_plating          [m.s-1]      log10 bounds  [-12, -9]
    k_LAM_negative     [s-1]        log10 bounds  [-10, -7]

Key correctness change vs. src/simulation/verify_e2e_phase2.py:

  1. **Per-cell experiment protocol** from
     configs/cohort_experiment_protocols.yaml (previous code hardcoded a
     0.5C / 2.5–3.65 V / 150-cycle block for EVE_0008).
  2. **Per-cell BOL overrides** from configs/bol_params/{make}_{cell}.yaml
     (previous code hardcoded eve_0008_bol_params.yaml).
  3. **k_LAM_negative added** — driven by memory note that negative-electrode
     LAM dominates PyBaMM cycle fade for this LFP cell; needed for a
     physically meaningful 5-parameter identification.
  4. **Loss over the FULL measured horizon** (min(len(sim), len(meas))) —
     no more 150-cycle truncation.
  5. **RMSE reported in percentage points** (loss = rmse_pp so DE scales
     well; solver-failure penalty is 10 pp, not 100 pp, to avoid distorting
     the landscape).

Model + solver (unchanged from Phase 2 reference):
    SPMe + SEI solvent-diffusion-limited + irreversible plating +
    stress-driven LAM.  IDAKLU rtol=atol=1e-6, var_pts
    {x_n:16, x_s:8, x_p:16, r_n:8, r_p:8}.

Pure module — no I/O side effects on import.  All heavy work happens
inside identify_cell_deg().
"""
from __future__ import annotations

import gc
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml

# Make src/simulation importable regardless of where this module is called
# from (module lives outside /src but re-uses build_parameter_values).
_PROJECT_ROOT = Path("/home/hj/Desktop/PINNs")
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# --------------------------------------------------------------------------- #
# Fitted-parameter definition
# --------------------------------------------------------------------------- #
#
# THETA_SPEC is the single source of truth for parameter order, encoding
# (log10 vs linear), search bounds and PyBaMM key. Everything downstream
# (bounds vector, identifiability report, YAML output) reads from here.
#
THETA_SPEC: tuple[dict, ...] = (
    dict(name="k_SEI",
         pybamm_key="SEI kinetic rate constant [m.s-1]",
         encoding="log10",
         bounds=(-13.0, -10.0)),
    dict(name="V_SEI",
         pybamm_key="SEI partial molar volume [m3.mol-1]",
         encoding="linear",
         bounds=(8e-5, 2e-4)),
    dict(name="D_SEI_solvent",
         pybamm_key="SEI solvent diffusivity [m2.s-1]",
         encoding="log10",
         bounds=(-22.0, -18.0)),
    dict(name="k_plating",
         pybamm_key="Lithium plating kinetic rate constant [m.s-1]",
         encoding="log10",
         bounds=(-12.0, -9.0)),
    dict(name="k_LAM_negative",
         pybamm_key="Negative electrode LAM constant proportional term [s-1]",
         encoding="log10",
         bounds=(-10.0, -7.0)),
)

MODEL_OPTIONS: dict[str, str] = {
    "SEI": "solvent-diffusion limited",
    "SEI porosity change": "true",
    "lithium plating": "irreversible",
    "loss of active material": "stress-driven",
}

CONFIGS_DIR = _PROJECT_ROOT / "configs"
BOL_PARAMS_DIR = CONFIGS_DIR / "bol_params"
COHORT_PROTOCOLS_YAML = CONFIGS_DIR / "cohort_experiment_protocols.yaml"

CANONICAL_SOH_DIR = _PROJECT_ROOT / "soh" / "data" / "canonical"

# Solver / discretisation config (matched to verify_e2e_phase2.py)
IDAKLU_RTOL = 1e-6
IDAKLU_ATOL = 1e-6

# Penalty applied when a single evaluation fails to solve. Big enough to
# push DE away from bad regions, small enough not to dominate the RMSE
# landscape (verify_e2e_phase2.py used 1.0 = 100 pp which distorts the
# fit; we use 10 pp = 0.10 RMSE-fraction if loss is RMSE-fraction, or
# 10.0 if loss is RMSE-pp).
SOLVER_FAIL_PENALTY_PP = 10.0


# --------------------------------------------------------------------------- #
# Cell context loader
# --------------------------------------------------------------------------- #

def _load_cohort_protocols() -> dict:
    """Load configs/cohort_experiment_protocols.yaml — raises if missing."""
    if not COHORT_PROTOCOLS_YAML.exists():
        raise FileNotFoundError(
            f"Cohort protocols YAML not found: {COHORT_PROTOCOLS_YAML}"
        )
    return yaml.safe_load(COHORT_PROTOCOLS_YAML.read_text())


def _load_bol_yaml(make: str, cell: str) -> dict:
    """Load configs/bol_params/{make}_{cell}.yaml — raises if missing."""
    p = BOL_PARAMS_DIR / f"{make}_{cell}.yaml"
    if not p.exists():
        raise FileNotFoundError(f"Per-cell BOL YAML not found: {p}")
    return yaml.safe_load(p.read_text())


def _load_measured_soh(make: str, cell: str) -> tuple[np.ndarray, np.ndarray]:
    """Load measured (cycle_n, soh) trace for one cell from canonical
    parquet.

    For CALB, we prefer the batch-2-corrected `calb_new` parquet if
    available (CC-only capacity per xgb_share protocol correction).
    """
    make_lower = make.lower()
    if make_lower == "calb":
        p_new = CANONICAL_SOH_DIR / "calb_new.parquet"
        p_old = CANONICAL_SOH_DIR / "calb_old.parquet"
        # calb_new has the protocol-corrected SoH (per CALB_new/old memory).
        parquet_path = p_new if p_new.exists() else p_old
    elif make_lower == "eve":
        parquet_path = CANONICAL_SOH_DIR / "eve.parquet"
    elif make_lower == "rept":
        parquet_path = CANONICAL_SOH_DIR / "rept.parquet"
    else:
        raise ValueError(f"Unknown make: {make}")
    if not parquet_path.exists():
        raise FileNotFoundError(f"SoH parquet not found: {parquet_path}")

    df = pd.read_parquet(parquet_path)
    cell_str = str(cell).zfill(4)
    sub = (df[df["cell_id"].astype(str).str.zfill(4) == cell_str]
             .sort_values("global_cycle")
             .reset_index(drop=True))
    if sub.empty:
        raise ValueError(f"No rows for {make} cell {cell_str} in {parquet_path.name}")
    cycles = sub["global_cycle"].to_numpy(dtype=int)
    soh = sub["soh"].to_numpy(dtype=float)
    return cycles, soh


def load_cell_context(make: str, cell: str) -> dict:
    """Return per-cell context needed to run the DE fit.

    Returns
    -------
    dict with keys:
        - bol: dict (Phase 1 BOL YAML contents)
        - experiment_steps: tuple[str, ...] — PyBaMM Experiment step list
        - measured_soh_array: np.ndarray — measured SoH (not renormalised)
        - measured_cycles: np.ndarray — global_cycle values
        - nominal_capacity_Ah: float
        - protocol_id: str — key into cohort_experiment_protocols.yaml
        - cycles_measured: int — number of measured cycles
    """
    cell_str = str(cell).zfill(4)
    bol = _load_bol_yaml(make, cell_str)

    protos = _load_cohort_protocols()
    cell_key = f"{make}_{cell_str}"
    if cell_key not in protos["cell_to_protocol"]:
        raise KeyError(
            f"{cell_key} not in cohort_experiment_protocols cell_to_protocol map"
        )
    proto_id = protos["cell_to_protocol"][cell_key]
    proto = protos["protocols"][proto_id]

    experiment_steps = tuple(proto["experiment_steps"])
    nominal_capacity_Ah = float(proto["nominal_capacity_Ah"])

    cycles, soh = _load_measured_soh(make, cell_str)

    return dict(
        make=make,
        cell=cell_str,
        bol=bol,
        experiment_steps=experiment_steps,
        measured_soh_array=soh,
        measured_cycles=cycles,
        nominal_capacity_Ah=nominal_capacity_Ah,
        protocol_id=proto_id,
        cycles_measured=int(len(cycles)),
    )


# --------------------------------------------------------------------------- #
# PyBaMM ParameterValues construction
# --------------------------------------------------------------------------- #

def build_pybamm_parameters(bol_yaml_data: dict):
    """Construct a Prada2013 + OKane2022 fusion ParameterValues, then
    apply per-cell BOL overrides.

    Overrides applied:
        - stoichiometry x_100/y_100 -> initial concentrations
          (scaled by base-set max concentrations)
        - resistance R0_Ohm         -> Contact resistance [Ohm]
        - diffusion D_s_m2_s        -> Negative particle diffusivity [m2.s-1]
          (GITT probes graphite for LFP; radius already reflected in D_s)
        - Ambient/Initial temperature = 298.15 K (25 °C dataset)

    NOT mapped (documented tradeoffs):
        - R1 (charge-transfer): mapping to exchange current density
          requires geometry-consistent conversion; defer until we have a
          j0-fit routine.
        - Q_n_Ah, Q_p_Ah: Prada2013 is 2.3 Ah; our cells are 72–150 Ah.
          SOH shape (not absolute Ah) is the fit target — see
          src/simulation/_pybamm_setup.py note.
    """
    from src.simulation._pybamm_setup import build_parameter_values
    import pybamm

    base = pybamm.ParameterValues("Prada2013")
    c_n_max = float(base["Maximum concentration in negative electrode [mol.m-3]"])
    c_p_max = float(base["Maximum concentration in positive electrode [mol.m-3]"])

    overrides: dict = {}

    st = bol_yaml_data.get("stoichiometry") or {}
    if "x_100" in st and "y_100" in st:
        overrides["Initial concentration in negative electrode [mol.m-3]"] = \
            float(st["x_100"]) * c_n_max
        overrides["Initial concentration in positive electrode [mol.m-3]"] = \
            float(st["y_100"]) * c_p_max

    res = bol_yaml_data.get("resistance") or {}
    if "R0_Ohm" in res and np.isfinite(res["R0_Ohm"]):
        overrides["Contact resistance [Ohm]"] = float(res["R0_Ohm"])

    diff = bol_yaml_data.get("diffusion") or {}
    if "D_s_m2_s" in diff and np.isfinite(diff["D_s_m2_s"]):
        # BOL D_s is from GITT — dominated by graphite in LFP
        overrides["Negative particle diffusivity [m2.s-1]"] = float(diff["D_s_m2_s"])

    overrides["Ambient temperature [K]"] = 298.15
    overrides["Initial temperature [K]"] = 298.15

    return build_parameter_values(overrides=overrides)


def apply_deg_params(param_values, k_SEI: float, V_SEI: float,
                      D_SEI_solvent: float, k_plating: float,
                      k_LAM_negative: float):
    """Apply the 5 fitted degradation parameters on top of a BOL
    ParameterValues.  All inputs are in **physical** (not log10) units.
    """
    # Copy so the caller's ParameterValues stays clean across evals
    pv = param_values.copy()
    updates = {
        "SEI kinetic rate constant [m.s-1]": float(k_SEI),
        "SEI partial molar volume [m3.mol-1]": float(V_SEI),
        "SEI solvent diffusivity [m2.s-1]": float(D_SEI_solvent),
        "Lithium plating kinetic rate constant [m.s-1]": float(k_plating),
        "Negative electrode LAM constant proportional term [s-1]": float(k_LAM_negative),
    }
    for k, v in updates.items():
        if k in pv.keys():
            pv.update({k: v})
        else:
            # Fall back to check_already_exists=False for LAM key that may
            # not be present until stress-driven-LAM options are set
            pv.update({k: v}, check_already_exists=False)
    return pv


# --------------------------------------------------------------------------- #
# Simulation → SoH trajectory
# --------------------------------------------------------------------------- #

def _submesh_pts():
    import pybamm
    v = pybamm.standard_spatial_vars
    return {v.x_n: 16, v.x_s: 8, v.x_p: 16, v.r_n: 8, v.r_p: 8}


def _extract_soh_from_solution(sol) -> np.ndarray:
    """Per-cycle discharge capacity normalised to cycle 1.

    For each cycle, we scan ALL steps and pick the one that actually
    discharged the most charge (largest |ΔQ| among steps with net-positive
    average current — PyBaMM's convention is I>0 for discharge).

    The older behaviour picked "first step with mean(I) < -1e-3", which was
    (a) the wrong sign — that's the CHARGE step in PyBaMM's convention —
    and (b) fooled by a trivially-small first-cycle charge when the cell
    started near full SoC, so caps[0] came out ~0.02 Ah and everything
    downstream blew up by a factor of 100.
    """
    caps = []
    for cy in sol.cycles:
        best_dQ = 0.0
        for step in cy.steps:
            try:
                Imean = float(np.nanmean(step["Current [A]"].entries))
            except Exception:
                continue
            if Imean <= 1e-3:  # not a net-discharge step (I>0 == discharge)
                continue
            try:
                Q = step["Discharge capacity [A.h]"].entries
                dQ = abs(float(Q[-1] - Q[0]))
            except Exception:
                continue
            if dQ > best_dQ:
                best_dQ = dQ
        caps.append(best_dQ if best_dQ > 0 else np.nan)
    caps = np.array(caps, dtype=float)
    if caps.size == 0 or not np.isfinite(caps[0]) or caps[0] <= 0:
        return np.full(max(caps.size, 1), np.nan)
    # Bug fix (2026-07-10): handle anomalous cycle-1 discharge caps.
    # There are two failure modes, both caused by cycle-1 not matching
    # the steady-state protocol:
    #   (a) caps[0] << steady state (CV-tail extraction fail; prior fix)
    #   (b) caps[0] >> steady state — happens when the ICs place the cell
    #       at 100% SoC but the protocol cycles between 0-80% SoC (etc.).
    #       Cycle-1 discharges the full 100% span (~1.25× steady state);
    #       cycle 2+ only refills the protocol window. Left uncorrected,
    #       normalising by caps[0] gives soh_sim = [1.0, 0.8, ...] which
    #       forces RMSE ≥ 20 pp against a near-flat aged cell — DE then
    #       prefers solver-death (10 pp penalty) over surviving fits.
    if caps.size >= 3:
        ref = float(np.nanmedian(caps[1:min(6, caps.size)]))
        if np.isfinite(ref) and ref > 0:
            ratio = caps[0] / ref
            if ratio < 0.5:
                # Case (a) — first-cycle extraction failure. Invalid.
                return np.full(caps.size, np.nan)
            if ratio > 1.3:
                # Case (b) — cycle-1 conditioning discharge in partial-DoD
                # protocol. Use median as the SoH denominator so the
                # trajectory reflects the actual protocol steady state.
                return caps / ref
    return caps / caps[0]


def simulate_soh_trajectory(param_values, experiment_steps,
                             n_cycles: int,
                             nominal_capacity_Ah: float) -> np.ndarray:
    """Run n_cycles of the given experiment; return per-cycle
    normalised SoH.

    Returns a NaN array of length n_cycles on solver failure — does
    NOT raise.  nominal_capacity_Ah is currently unused (kept in signature
    for future absolute-Ah support).
    """
    import pybamm

    try:
        model = pybamm.lithium_ion.SPMe(options=MODEL_OPTIONS)
        solver = pybamm.IDAKLUSolver(rtol=IDAKLU_RTOL, atol=IDAKLU_ATOL)
        experiment = pybamm.Experiment(
            [tuple(experiment_steps)] * int(n_cycles)
        )
        sim = pybamm.Simulation(
            model,
            parameter_values=param_values,
            experiment=experiment,
            solver=solver,
            var_pts=_submesh_pts(),
        )
        sol = sim.solve()
        soh_sim = _extract_soh_from_solution(sol)
        del sol, sim, experiment
        gc.collect()
        return soh_sim
    except Exception:
        # Silent NaN return — the DE loss layer wraps with penalty
        gc.collect()
        return np.full(n_cycles, np.nan)


# --------------------------------------------------------------------------- #
# DE loss
# --------------------------------------------------------------------------- #

def _theta_from_log(theta_log: np.ndarray) -> dict:
    """Decode DE search-space vector to physical-units dict keyed by
    THETA_SPEC name."""
    out = {}
    for i, spec in enumerate(THETA_SPEC):
        v = float(theta_log[i])
        if spec["encoding"] == "log10":
            v = 10.0 ** v
        out[spec["name"]] = v
    return out


def de_loss(theta_log: np.ndarray, cell_ctx: dict) -> float:
    """DE cost function.  Returns RMSE in **percentage points**.

    Both simulated and measured SoH are renormalised to their own
    cycle-1 value; residual is over min(len(sim), len(measured)).
    Solver failures / degenerate simulations return SOLVER_FAIL_PENALTY_PP.
    """
    theta = _theta_from_log(theta_log)
    pv = build_pybamm_parameters(cell_ctx["bol"])
    pv = apply_deg_params(pv, **theta)

    n_cycles_meas = int(cell_ctx["cycles_measured"])
    soh_sim = simulate_soh_trajectory(
        pv, cell_ctx["experiment_steps"],
        n_cycles=n_cycles_meas,
        nominal_capacity_Ah=cell_ctx["nominal_capacity_Ah"],
    )

    if soh_sim.size == 0 or np.all(np.isnan(soh_sim)):
        return float(SOLVER_FAIL_PENALTY_PP)

    meas = np.asarray(cell_ctx["measured_soh_array"], dtype=float)
    if meas.size == 0 or not np.isfinite(meas[0]) or meas[0] <= 0:
        return float(SOLVER_FAIL_PENALTY_PP)
    meas_norm = meas / meas[0]

    n = min(len(soh_sim), len(meas_norm))
    if n < 10:
        return float(SOLVER_FAIL_PENALTY_PP)

    # Bug fix (2026-07-10): if the sim died before covering >=90% of the
    # measured horizon, DE could reward θ that produce short "clean" runs.
    # Penalise partial-sim survival explicitly.
    coverage = float(n) / float(len(meas_norm))
    if coverage < 0.90:
        return float(SOLVER_FAIL_PENALTY_PP)

    residual = soh_sim[:n] - meas_norm[:n]
    rmse = float(np.sqrt(np.nanmean(residual ** 2)))
    if not np.isfinite(rmse):
        return float(SOLVER_FAIL_PENALTY_PP)
    return rmse * 100.0  # percentage points


# --------------------------------------------------------------------------- #
# Identifiability
# --------------------------------------------------------------------------- #

def _identifiability_from_population(pop_x: np.ndarray,
                                      pop_loss: np.ndarray) -> dict:
    """From the DE final population, compute per-parameter top-10% span
    and flag well_identified when span < 30% of the search-bound width.

    pop_x has shape (n_individuals, 5) in DE search space (log10 for
    kinetic constants, linear for V_SEI).
    """
    out: dict = {}
    if pop_x.size == 0 or pop_loss.size == 0:
        return out
    finite = np.isfinite(pop_loss)
    if not finite.any():
        return out

    x = pop_x[finite]
    losses = pop_loss[finite]
    cutoff = float(np.quantile(losses, 0.10))
    keep = losses <= cutoff
    if not keep.any():
        return out
    top_x = x[keep]

    for i, spec in enumerate(THETA_SPEC):
        col = top_x[:, i]
        lo, hi = float(col.min()), float(col.max())
        bnd_lo, bnd_hi = spec["bounds"]
        width = bnd_hi - bnd_lo
        span_frac = (hi - lo) / max(1e-12, width)
        out[spec["name"]] = dict(
            top10pct_range=[lo, hi],
            span_of_full_range=float(span_frac),
            well_identified=bool(span_frac < 0.30),
            encoding=spec["encoding"],
            pybamm_key=spec["pybamm_key"],
        )
    return out


# --------------------------------------------------------------------------- #
# Full per-cell identification
# --------------------------------------------------------------------------- #

def identify_cell_deg(make: str, cell: str, *,
                      n_evaluations: int = 200,
                      workers: int = -1,
                      n_cycles_override: Optional[int] = None,
                      seed: int = 42,
                      tol: float = 1e-3,
                      verbose: bool = True) -> dict:
    """End-to-end DE identification for one cell.

    Parameters
    ----------
    n_evaluations
        Approximate DE budget.  With popsize=6 and 5 params → 30
        individuals per generation → maxiter = round(n_evaluations/30) - 1.
        Actual eval count = (maxiter + 1) * n_individuals.
    workers
        Passed straight to differential_evolution.  -1 = all cores.
    n_cycles_override
        If set, truncate the measured trace before fitting (useful for a
        smoke-test to keep wall-time under a minute).  When None (default),
        fit against the FULL measured horizon (brief requirement).
    """
    from scipy.optimize import differential_evolution

    make = str(make)
    cell_str = str(cell).zfill(4)
    t_start = time.time()

    if verbose:
        print(f"=== Phase 2 DE fit: {make}_{cell_str} ===", flush=True)

    ctx = load_cell_context(make, cell_str)

    if n_cycles_override is not None:
        n_keep = int(n_cycles_override)
        ctx["measured_soh_array"] = ctx["measured_soh_array"][:n_keep]
        ctx["measured_cycles"] = ctx["measured_cycles"][:n_keep]
        ctx["cycles_measured"] = int(min(n_keep, len(ctx["measured_cycles"])))

    # ---- DE bounds vector (search space) ----
    bounds = [spec["bounds"] for spec in THETA_SPEC]
    n_params = len(bounds)

    # ---- Budget bookkeeping: popsize is per-parameter in scipy ----
    # popsize=6 with 5 params → 30 individuals/generation
    # brief: popsize=30 (total) → 6 per param.
    popsize_per_param = 6
    n_individuals = popsize_per_param * n_params
    # maxiter+1 generations of n_individuals ≈ n_evaluations
    de_maxiter = max(1, int(round(n_evaluations / n_individuals)) - 1)

    if verbose:
        print(f"  protocol: {ctx['protocol_id']}", flush=True)
        print(f"  measured cycles: {ctx['cycles_measured']}", flush=True)
        print(f"  DE: popsize/param={popsize_per_param} → "
              f"{n_individuals} indiv/gen × {de_maxiter+1} gens "
              f"≈ {n_individuals*(de_maxiter+1)} evals", flush=True)

    # DE call
    call_log: list[dict] = []

    def loss_fn(theta_log):
        t0 = time.time()
        val = de_loss(theta_log, ctx)
        call_log.append(dict(
            elapsed_s=time.time() - t0,
            theta_log=[float(v) for v in theta_log],
            loss_pp=float(val),
        ))
        return val

    de_success = False
    de_message = ""
    result = None
    try:
        result = differential_evolution(
            loss_fn, bounds,
            maxiter=de_maxiter,
            popsize=popsize_per_param,
            seed=seed,
            workers=workers,
            polish=False,
            tol=tol,
            updating="deferred",   # required when workers != 1
        )
        de_success = bool(result.success)
        de_message = str(result.message)
    except Exception as e:
        de_message = f"DE crashed: {type(e).__name__}: {e}"
        if verbose:
            print(traceback.format_exc(), flush=True)

    wall_time_s = time.time() - t_start

    # ---- Best theta ----
    if result is not None and getattr(result, "x", None) is not None:
        best_x = np.asarray(result.x, dtype=float)
        best_loss_pp = float(result.fun)
        theta_phys = _theta_from_log(best_x)
    else:
        # Fall back to best from call_log
        if call_log:
            k = int(np.argmin([r["loss_pp"] for r in call_log]))
            best_x = np.asarray(call_log[k]["theta_log"], dtype=float)
            best_loss_pp = float(call_log[k]["loss_pp"])
            theta_phys = _theta_from_log(best_x)
        else:
            best_x = np.full(n_params, np.nan)
            best_loss_pp = float("nan")
            theta_phys = {spec["name"]: float("nan") for spec in THETA_SPEC}

    # ---- Identifiability from DE final population ----
    ident: dict = {}
    if result is not None and hasattr(result, "population") \
            and hasattr(result, "population_energies"):
        pop_x = np.asarray(result.population, dtype=float)
        pop_loss = np.asarray(result.population_energies, dtype=float)
        ident = _identifiability_from_population(pop_x, pop_loss)

    # Merge best-param and identifiability into a serialisable structure
    fitted_params_yaml: dict = {}
    for spec in THETA_SPEC:
        name = spec["name"]
        info = dict(
            value=float(theta_phys[name]),
            pybamm_key=spec["pybamm_key"],
            encoding=spec["encoding"],
            search_bounds=list(spec["bounds"]),
        )
        if name in ident:
            info["identifiability"] = ident[name]
        fitted_params_yaml[name] = info

    out = dict(
        make=make,
        cell=cell_str,
        protocol_id=ctx["protocol_id"],
        experiment_steps=list(ctx["experiment_steps"]),
        nominal_capacity_Ah=ctx["nominal_capacity_Ah"],
        n_cycles_fitted=int(ctx["cycles_measured"]),
        de=dict(
            n_evaluations=int(len(call_log)),
            popsize_per_param=int(popsize_per_param),
            n_individuals=int(n_individuals),
            maxiter=int(de_maxiter),
            tol=float(tol),
            seed=int(seed),
            workers=int(workers),
            success=bool(de_success),
            message=str(de_message),
            wall_time_s=float(wall_time_s),
        ),
        best_rmse_pp=(float(best_loss_pp)
                      if np.isfinite(best_loss_pp) else None),
        fitted_params=fitted_params_yaml,
        model_options=dict(MODEL_OPTIONS),
        _provenance=dict(
            identified_at_utc=datetime.now(timezone.utc).isoformat(),
            source_module="Voltaris/Data_Exploration/phase2_de_fit.py",
            pipeline_version=1,
            base_parameter_set="pybamm.ParameterValues('Prada2013')",
            degradation_donor="pybamm.ParameterValues('OKane2022')",
        ),
    )
    return out


def save_yaml(result: dict, path) -> None:
    """Persist the identify_cell_deg() result to YAML."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(yaml.safe_dump(result, sort_keys=False, allow_unicode=True))


# --------------------------------------------------------------------------- #
# Convenience: single-cell CLI (only if run directly, never at import)
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("make")
    ap.add_argument("cell")
    ap.add_argument("--n-evals", type=int, default=200)
    ap.add_argument("--workers", type=int, default=-1)
    ap.add_argument("--n-cycles-override", type=int, default=None)
    ap.add_argument("--out", default=None,
                     help="Output YAML path (default configs/deg_params/{make}_{cell}.yaml)")
    a = ap.parse_args()
    res = identify_cell_deg(a.make, a.cell,
                             n_evaluations=a.n_evals,
                             workers=a.workers,
                             n_cycles_override=a.n_cycles_override)
    out = a.out or (CONFIGS_DIR / "deg_params" / f"{a.make}_{a.cell}.yaml")
    save_yaml(res, out)
    print(f"Wrote: {out}")
