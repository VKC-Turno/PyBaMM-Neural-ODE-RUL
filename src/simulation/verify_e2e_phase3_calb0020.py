"""
Phase 3: Long-horizon DFN cycling simulation using the per-cell BOL
parameters + fitted degradation parameters for CALB cell 0020.

Configuration
=============
- C-rate: 0.5 C  (inferred from Longterm CSV 'crate' column)
- Temperature: 298.15 K (isothermal, 25 C)
- Cycles: up to 5000 or until SoH < 0.35
- Batching: 1000-cycle chunks chained via starting_solution=

Outputs (under data/synthetic/verification/):
    calb_0020_longrun.parquet
    calb_0020_workflow_validation.png
    calb_0020_phase3_log.txt
"""
from __future__ import annotations

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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.simulation._pybamm_setup import build_parameter_values


OUT_DIR = Path("/home/hj/Desktop/PINNs/data/synthetic/verification")
BOL_YAML = OUT_DIR / "calb_0020_bol_params.yaml"
DEG_YAML = OUT_DIR / "calb_0020_deg_params.yaml"
LONGRUN_PARQUET = OUT_DIR / "calb_0020_longrun.parquet"
VAL_PLOT = OUT_DIR / "calb_0020_workflow_validation.png"
LOG_TXT = OUT_DIR / "calb_0020_phase3_log.txt"

CELL = "0020"
C_RATE = 0.5
CYCLES_PER_BATCH = 1000
MAX_CYCLES = 5000
SOH_STOP = 0.35
MODEL_OPTIONS = {
    "SEI": "solvent-diffusion limited",
    "SEI porosity change": "true",
    "lithium plating": "irreversible",
}
MAX_WALLTIME_S = 60 * 60  # 60 min hard cap for Phase 3


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


def build_overrides() -> dict:
    bol = yaml.safe_load(BOL_YAML.read_text())
    deg = yaml.safe_load(DEG_YAML.read_text())
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


def build_experiment(n_cycles: int, c_rate: float = C_RATE) -> pybamm.Experiment:
    block = (
        f"Discharge at {c_rate:.4f}C until 2.5 V",
        "Rest for 10 minutes",
        f"Charge at {c_rate:.4f}C until 3.65 V",
        "Hold at 3.65 V until C/100",
        "Rest for 10 minutes",
    )
    return pybamm.Experiment([block] * int(n_cycles))


def extract_per_cycle(sol, cycle_offset: int, skip_first: bool) -> pd.DataFrame:
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


def load_measured_soh() -> pd.DataFrame:
    df = pd.read_parquet("/home/hj/Desktop/PINNs/soh/data/canonical/calb_new.parquet")
    s = df[df.cell_id == CELL].sort_values("global_cycle").reset_index(drop=True)
    return s[["global_cycle", "soh"]].rename(columns={"global_cycle": "cycle_n"})


def cycle_at_soh(df: pd.DataFrame, target: float) -> float | None:
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


def run_long_sim() -> tuple[pd.DataFrame | None, dict]:
    t_total_start = time.time()
    log(f"=== Phase 3: DFN long-run ({C_RATE}C, 25 C) ===")
    log(f"Memory available at start: {sys_mem_available_gb():.1f} GB")

    overrides = build_overrides()
    log("Overrides applied:")
    for k, v in overrides.items():
        log(f"  {k}: {v}")

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
        if time.time() - t_total_start > MAX_WALLTIME_S:
            meta["aborted"] = True
            meta["abort_reason"] = "wall-time cap"
            log("HARD-CAP: wall time reached, stopping")
            break
        mem_gb = sys_mem_available_gb()
        if mem_gb < 1.0:
            meta["aborted"] = True
            meta["abort_reason"] = f"low mem {mem_gb:.1f} GB"
            log(f"HARD-CAP: memory {mem_gb:.1f} GB, stopping")
            break

        log(f"\n--- Batch {batch_i}/{n_batches_max} ({CYCLES_PER_BATCH} cy) "
            f"mem={mem_gb:.1f} GB ---")
        t0 = time.time()
        try:
            exp = build_experiment(CYCLES_PER_BATCH)
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
        df = extract_per_cycle(
            sol, cycle_offset,
            skip_first=(starting_solution is not None),
        )
        n_ok = len(df)
        dt = time.time() - t0
        summary = {
            "batch": batch_i,
            "n_cycles_ok": n_ok,
            "elapsed_s": dt,
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
        return None, meta
    full = pd.concat(all_frames, ignore_index=True)
    q0 = float(full["Q_Ah"].iloc[0])
    full["SOH"] = full["Q_Ah"] / q0
    meta["q0_Ah"] = q0
    meta["elapsed_total_s"] = time.time() - t_total_start
    return full, meta


def make_validation_plot(df_sim: pd.DataFrame, df_meas: pd.DataFrame,
                          eol_cycle: float | None,
                          eosl_cycle: float | None) -> None:
    fig = plt.figure(figsize=(11, 6.5))
    gs = fig.add_gridspec(1, 3, width_ratios=[2.2, 1, 1])
    ax_main = fig.add_subplot(gs[0])
    ax_zoom = fig.add_subplot(gs[1])
    ax_resid = fig.add_subplot(gs[2])

    ax_main.plot(df_sim["cycle_n"], df_sim["SOH"], "b-", lw=1.5,
                 label="Simulated (DFN + SEI + plating)")
    ax_main.scatter(df_meas["cycle_n"], df_meas["soh"] / df_meas["soh"].iloc[0],
                    c="k", s=14, label="Measured CALB 0020 (norm)")
    ax_main.axhline(0.80, color="tab:orange", ls="--", lw=0.8, label="EoL 0.80")
    ax_main.axhline(0.40, color="tab:red", ls="--", lw=0.8, label="EoSL 0.40")
    if eol_cycle is not None:
        ax_main.axvline(eol_cycle, color="tab:orange", ls=":", lw=0.7, alpha=0.6)
        ax_main.annotate(f"EoL @ {eol_cycle:.0f}", xy=(eol_cycle, 0.80),
                         xytext=(eol_cycle * 0.6, 0.87),
                         fontsize=9,
                         arrowprops=dict(arrowstyle="->", color="tab:orange",
                                         lw=0.6))
    if eosl_cycle is not None:
        ax_main.axvline(eosl_cycle, color="tab:red", ls=":", lw=0.7, alpha=0.6)
        ax_main.annotate(f"EoSL @ {eosl_cycle:.0f}", xy=(eosl_cycle, 0.40),
                         xytext=(eosl_cycle * 0.55, 0.5),
                         fontsize=9,
                         arrowprops=dict(arrowstyle="->", color="tab:red",
                                         lw=0.6))
    ax_main.set_xlabel("Cycle")
    ax_main.set_ylabel("SoH")
    ax_main.set_ylim(0, 1.05)
    ax_main.set_title("End-to-end workflow validation: CALB cell 0020\n"
                      "per-cell BOL + fitted deg params  ->  DFN long run")
    ax_main.grid(alpha=0.3)
    ax_main.legend(loc="lower left", fontsize=9)

    # Inset: zoom to first 150 cycles
    df_first = df_sim[df_sim["cycle_n"] <= 200]
    ax_zoom.plot(df_first["cycle_n"], df_first["SOH"], "b-", lw=1.4, label="Sim")
    ax_zoom.scatter(df_meas["cycle_n"],
                    df_meas["soh"] / df_meas["soh"].iloc[0],
                    c="k", s=10, label="Meas norm")
    ax_zoom.set_xlabel("Cycle")
    ax_zoom.set_ylabel("SoH (norm)")
    ax_zoom.set_xlim(0, 160)
    ax_zoom.set_title("First 150 cycles", fontsize=10)
    ax_zoom.grid(alpha=0.3)
    ax_zoom.legend(fontsize=8)

    # Residuals (pp)
    sim_at_cy = np.interp(df_meas["cycle_n"], df_sim["cycle_n"], df_sim["SOH"])
    meas_norm = df_meas["soh"].to_numpy() / df_meas["soh"].to_numpy()[0]
    resid_pp = (sim_at_cy - meas_norm) * 100
    ax_resid.plot(df_meas["cycle_n"], resid_pp, "g-", lw=1.2)
    ax_resid.axhline(0, color="k", lw=0.5)
    ax_resid.set_xlabel("Cycle")
    ax_resid.set_ylabel("Sim - Meas [pp]")
    ax_resid.set_title("Residuals (first 150 cy)", fontsize=10)
    ax_resid.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(VAL_PLOT, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main() -> dict:
    df_sim, meta = run_long_sim()

    if df_sim is None or df_sim.empty:
        log("\nNo simulation data.")
        LOG_TXT.write_text("\n".join(LOG_LINES))
        return {"success": False}

    df_sim.to_parquet(LONGRUN_PARQUET, index=False)
    log(f"\nWrote long-run trajectory: {LONGRUN_PARQUET}")

    df_meas = load_measured_soh()
    eol = cycle_at_soh(df_sim, 0.80)
    eosl = cycle_at_soh(df_sim, 0.40)
    make_validation_plot(df_sim, df_meas, eol, eosl)
    log(f"Wrote validation plot: {VAL_PLOT}")

    # Monotonicity
    d = df_sim["SOH"].diff().dropna()
    monotonic = bool((d <= 1e-4).all())
    n_up_steps = int((d > 1e-4).sum())

    # First-150-cycle RMSE
    sim_at_cy = np.interp(df_meas["cycle_n"], df_sim["cycle_n"], df_sim["SOH"])
    meas_norm = df_meas["soh"].to_numpy() / df_meas["soh"].to_numpy()[0]
    resid = sim_at_cy - meas_norm
    rmse_pp_150 = float(np.sqrt(np.mean(resid ** 2)) * 100)

    # Fade delta over 150 cy
    sim_delta_150 = float((df_sim["SOH"].iloc[0]
                           - np.interp(150, df_sim["cycle_n"], df_sim["SOH"])) * 100)
    meas_delta_150 = float((df_meas["soh"].iloc[0] - df_meas["soh"].iloc[-1])
                            / df_meas["soh"].iloc[0] * 100)

    log("\nSummary:")
    log(f"  q0 (cy 1 Q, Ah): {meta['q0_Ah']:.3f}")
    log(f"  final SoH: {df_sim['SOH'].iloc[-1]:.4f}")
    log(f"  cycles simulated: {int(df_sim['cycle_n'].iloc[-1])}")
    log(f"  cycle at SoH 0.80 (EoL): "
        f"{eol:.0f}" if eol else "  cycle at SoH 0.80 (EoL): not reached")
    log(f"  cycle at SoH 0.40 (EoSL): "
        f"{eosl:.0f}" if eosl else "  cycle at SoH 0.40 (EoSL): not reached")
    log(f"  Monotonic decreasing: {monotonic} "
        f"({n_up_steps} up-steps > 1e-4)")
    log(f"  First-150-cycle fit RMSE: {rmse_pp_150:.3f} pp "
        f"(sim fade {sim_delta_150:.2f} pp vs meas fade {meas_delta_150:.2f} pp)")

    result = {
        "success": True,
        "q0_Ah": float(meta.get("q0_Ah", np.nan)),
        "n_cycles_simulated": int(df_sim["cycle_n"].iloc[-1]),
        "final_soh": float(df_sim["SOH"].iloc[-1]),
        "cycle_at_soh_0p80": float(eol) if eol else None,
        "cycle_at_soh_0p40": float(eosl) if eosl else None,
        "monotonic_decreasing": monotonic,
        "n_up_steps": n_up_steps,
        "first_150cy_rmse_pp": rmse_pp_150,
        "sim_delta_soh_first_150cy_pp": sim_delta_150,
        "meas_delta_soh_first_150cy_pp": meas_delta_150,
        "batch_summaries": meta["batch_summaries"],
        "elapsed_total_s": meta.get("elapsed_total_s"),
        "aborted": meta["aborted"],
        "abort_reason": meta["abort_reason"],
    }

    # save summary yaml
    with open(OUT_DIR / "calb_0020_phase3_summary.yaml", "w") as f:
        yaml.safe_dump(result, f, sort_keys=False)
    LOG_TXT.write_text("\n".join(LOG_LINES))
    return result


if __name__ == "__main__":
    main()
