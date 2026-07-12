"""
src/simulation/verify_long_run.py
---------------------------------
Long-horizon PyBaMM verification at 0.25 C / 25 degC to test whether the
identified BOL parameters + a carefully chosen degradation setup can
produce a physically plausible LFP fade curve that reaches SoH = 0.40
(second-life EoSL threshold).

Approach
========
1. Build a DFN model with SEI (solvent-diffusion limited) + SEI porosity
   change + irreversible lithium plating. LAM is intentionally NOT
   activated (prior tests showed LAM_neg drives an unphysical knee at
   SoH ~ 0.9).
2. Overlay the identified BOL stoichiometry / capacities / SEI ceiling
   via `overrides_from_identified_params()`.
3. Run in 5 batches of 1000 cycles, chaining each batch via
   `starting_solution=sol.last_state` so we never hold more than one
   batch's Solution object in memory.
4. Save per-batch parquet + a combined trajectory + a plot annotated
   with the EoSL threshold.

Outputs (under data/synthetic/verification/):
  batch_{1..5}.parquet   per-batch per-cycle features
  full_trajectory.parquet
  full_trajectory.png
  report.md
"""
from __future__ import annotations

import gc
import os
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, "/home/hj/Desktop/PINNs")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pybamm

from src.simulation._pybamm_setup import (
    build_parameter_values,
    overrides_from_identified_params,
)


OUT_DIR = Path("/home/hj/Desktop/PINNs/data/synthetic/verification")
OUT_DIR.mkdir(parents=True, exist_ok=True)

IDENTIFIED_PATH = Path("/home/hj/Desktop/PINNs/configs/identified_params.yaml")

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
MODEL_OPTIONS = {
    "SEI": "solvent-diffusion limited",
    "SEI porosity change": "true",
    "lithium plating": "irreversible",
    # Deliberately NO "loss of active material" — prior tests showed
    # stress-driven LAM produces an unphysical knee at SoH ~= 0.9 and
    # reaction-limited SEI + LAM_neg drives a stoichiometry collapse.
}

DEG_OVERRIDES = {
    # SEI kinetics (identified ceiling from self-discharge; order-of-magnitude)
    "SEI kinetic rate constant [m.s-1]": 2.73e-12,
    "SEI partial molar volume [m3.mol-1]": 1e-4,
    # Prada2013 default is 2.5e-22 m^2/s. If fade is too slow we'd bump
    # this to ~1e-21 in a retry.
    "SEI solvent diffusivity [m2.s-1]": 2.5e-22,
    # Small plating channel — a minor perturbation, not the main driver
    "Lithium plating kinetic rate constant [m.s-1]": 1e-11,
    # Isothermal at 25 C
    "Ambient temperature [K]": 298.15,
    "Initial temperature [K]": 298.15,
}

C_RATE = 0.25
CYCLES_PER_BATCH = 1000
N_BATCHES = 5
SOH_EOSL = 0.40
MAX_TOTAL_WALLTIME_S = 90 * 60  # 90 min hard cap


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def build_experiment(n_cycles: int) -> pybamm.Experiment:
    block = (
        f"Discharge at {C_RATE:.4f}C until 2.5 V",
        "Rest for 10 minutes",
        f"Charge at {C_RATE:.4f}C until 3.65 V",
        "Hold at 3.65 V until C/100",
        "Rest for 10 minutes",
    )
    return pybamm.Experiment([block] * int(n_cycles))


def build_param() -> pybamm.ParameterValues:
    overrides = dict(overrides_from_identified_params(IDENTIFIED_PATH))
    overrides.update(DEG_OVERRIDES)
    return build_parameter_values(overrides=overrides)


def _last_or_nan(step, key: str) -> float:
    try:
        arr = step[key].entries
        return float(arr.flat[-1])
    except Exception:
        return float("nan")


def extract_per_cycle(sol, cycle_offset: int,
                      skip_first: bool = False) -> pd.DataFrame:
    """Return per-cycle Q_Ah + degradation state for the given solution.

    `cycle_offset` shifts the batch-local 1-based index into a global
    cycle number that is continuous across batches.

    When resuming from a `starting_solution`, PyBaMM re-emits the resume
    cycle as `sol.cycles[0]`, which duplicates the last cycle of the
    previous batch. Pass `skip_first=True` on chained batches to drop it.
    """
    rows = []
    cycles_iter = sol.cycles[1:] if skip_first else sol.cycles
    for local_n, cycle in enumerate(cycles_iter, start=1):
        # Find the discharge step
        disc = None
        for step in cycle.steps:
            try:
                I_mean = float(np.nanmean(step["Current [A]"].entries))
            except Exception:
                continue
            if I_mean < -1e-3:
                disc = step
                break
        if disc is None:
            continue

        Q = disc["Discharge capacity [A.h]"].entries
        Q_Ah = abs(float(Q[-1] - Q[0]))
        last_step = cycle.steps[-1]
        sei = _last_or_nan(last_step, "X-averaged negative SEI thickness [m]")
        dead_li = _last_or_nan(last_step, "Loss of capacity to negative lithium plating [A.h]")
        rows.append({
            "cycle_n": local_n + cycle_offset,
            "Q_Ah": Q_Ah,
            "SEI_thickness_m": sei,
            "dead_lithium_Ah": dead_li,
        })
    return pd.DataFrame(rows)


def _peak_rss_mb() -> float:
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except Exception:
        return float("nan")


def _mem_snapshot() -> dict:
    """Read /proc/meminfo for a system-level view."""
    try:
        with open("/proc/meminfo") as f:
            info = {ln.split(":")[0]: ln.split(":")[1].strip()
                    for ln in f.readlines()}
        return {"MemAvailable": info.get("MemAvailable", "?"),
                "MemFree": info.get("MemFree", "?")}
    except Exception:
        return {}


# ---------------------------------------------------------------------
# Batch driver with retry-on-failure
# ---------------------------------------------------------------------
def run_batches() -> tuple[pd.DataFrame | None, dict]:
    t_total_start = time.time()
    peak_self_rss_mb = 0.0

    param = build_param()
    model = pybamm.lithium_ion.DFN(options=MODEL_OPTIONS)
    solver = pybamm.IDAKLUSolver(rtol=1e-6, atol=1e-6)

    starting_solution = None
    batch_summaries = []
    all_frames = []
    batch_size = CYCLES_PER_BATCH

    batch_i = 0
    while batch_i < N_BATCHES:
        batch_i += 1
        if time.time() - t_total_start > MAX_TOTAL_WALLTIME_S:
            print(f"  HARD-CAP: exceeded {MAX_TOTAL_WALLTIME_S}s total, stopping.",
                  flush=True)
            break

        print(f"\n=== Batch {batch_i}/{N_BATCHES} "
              f"({batch_size} cycles at {C_RATE}C) ===", flush=True)
        print(f"  memory: {_mem_snapshot()}", flush=True)

        attempts = 0
        succeeded = False
        while attempts < 2 and not succeeded:
            attempts += 1
            t_batch = time.time()
            try:
                exp = build_experiment(batch_size)
                sim = pybamm.Simulation(
                    model, parameter_values=param,
                    experiment=exp, solver=solver,
                )
                sol = sim.solve(starting_solution=starting_solution)
            except MemoryError as e:
                print(f"  MemoryError on attempt {attempts}: {e}", flush=True)
                gc.collect()
                if attempts >= 2:
                    print("  Two consecutive memory failures — abort.", flush=True)
                    return _finalize(all_frames, batch_summaries,
                                     t_total_start, peak_self_rss_mb, aborted=True)
                batch_size = max(200, batch_size // 2)
                print(f"  Retrying with batch_size={batch_size}", flush=True)
                time.sleep(60)
                continue
            except Exception as e:
                tb = traceback.format_exc()
                print(f"  Solver error on attempt {attempts}: "
                      f"{type(e).__name__}: {e}", flush=True)
                print(tb[:1000], flush=True)
                gc.collect()
                if attempts >= 2:
                    print("  Two consecutive solver failures — abort.", flush=True)
                    return _finalize(all_frames, batch_summaries,
                                     t_total_start, peak_self_rss_mb, aborted=True)
                # Try halving the batch on retry
                batch_size = max(200, batch_size // 2)
                print(f"  Retrying with batch_size={batch_size}", flush=True)
                continue

            # ---- successful solve ----
            solve_s = time.time() - t_batch
            # Cycle offset: use the cumulative number of cycles completed so far
            cycle_offset = sum(bs["n_cycles_ok"] for bs in batch_summaries)
            df = extract_per_cycle(
                sol, cycle_offset,
                skip_first=(starting_solution is not None),
            )
            n_ok = int(df["cycle_n"].max() - cycle_offset) if not df.empty else 0

            # Save per-batch parquet
            df.to_parquet(OUT_DIR / f"batch_{batch_i}.parquet", index=False)

            # Capture starting solution for the next batch (small: last state only)
            try:
                starting_solution = sol.last_state
            except AttributeError:
                starting_solution = sol  # fall back to full solution
            summary = {
                "batch": batch_i,
                "attempts": attempts,
                "batch_size_requested": batch_size,
                "n_cycles_ok": n_ok,
                "elapsed_s": solve_s,
                "Q_Ah_first": float(df["Q_Ah"].iloc[0]) if not df.empty else float("nan"),
                "Q_Ah_last": float(df["Q_Ah"].iloc[-1]) if not df.empty else float("nan"),
            }
            batch_summaries.append(summary)
            all_frames.append(df)
            print(f"  batch {batch_i}: {n_ok} cycles in {solve_s:.1f}s "
                  f"(Q_Ah {summary['Q_Ah_first']:.3f} -> "
                  f"{summary['Q_Ah_last']:.3f})", flush=True)

            # Free memory
            del sol, sim, exp
            gc.collect()
            rss = _peak_rss_mb()
            if rss > peak_self_rss_mb:
                peak_self_rss_mb = rss
            print(f"  after gc: peak self-RSS ~{rss:.0f} MB, "
                  f"sys mem {_mem_snapshot()}", flush=True)
            succeeded = True

            # Early-exit if we've already crossed EoSL
            combined_so_far = pd.concat(all_frames, ignore_index=True)
            q0 = float(combined_so_far["Q_Ah"].iloc[0])
            soh_last = float(combined_so_far["Q_Ah"].iloc[-1]) / q0
            print(f"  cumulative SoH after batch {batch_i}: {soh_last:.4f} "
                  f"(vs EoSL {SOH_EOSL})", flush=True)
            if soh_last <= SOH_EOSL:
                print(f"  EoSL reached at cumulative cycle "
                      f"{int(combined_so_far['cycle_n'].iloc[-1])} — stopping.",
                      flush=True)
                return _finalize(all_frames, batch_summaries,
                                 t_total_start, peak_self_rss_mb, aborted=False)

    return _finalize(all_frames, batch_summaries,
                     t_total_start, peak_self_rss_mb, aborted=False)


def _finalize(all_frames, batch_summaries, t_start, peak_rss_mb, aborted: bool):
    if not all_frames:
        return None, {"batch_summaries": batch_summaries,
                      "elapsed_total_s": time.time() - t_start,
                      "peak_self_rss_mb": peak_rss_mb,
                      "aborted": aborted}
    full = pd.concat(all_frames, ignore_index=True)
    q0 = float(full["Q_Ah"].iloc[0])
    full["SOH"] = full["Q_Ah"] / q0
    full.to_parquet(OUT_DIR / "full_trajectory.parquet", index=False)
    return full, {"batch_summaries": batch_summaries,
                  "q0_Ah": q0,
                  "elapsed_total_s": time.time() - t_start,
                  "peak_self_rss_mb": peak_rss_mb,
                  "aborted": aborted}


# ---------------------------------------------------------------------
# Analysis + plot + report
# ---------------------------------------------------------------------
def fade_rate_per_1000cy(df: pd.DataFrame, lo: int, hi: int) -> float:
    """Linear-fit fade rate in a cycle window (pp per 1000 cycles)."""
    sub = df[(df.cycle_n >= lo) & (df.cycle_n <= hi)]
    if len(sub) < 5:
        return float("nan")
    slope, _ = np.polyfit(sub.cycle_n, sub.SOH, 1)
    return -slope * 100 * 1000


def cycle_at_soh(df: pd.DataFrame, target: float) -> float | None:
    """Linear interpolation of the first cycle to reach `target`."""
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


def make_plot(df: pd.DataFrame, eosl_cycle: float | None):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(df.cycle_n, df.SOH, "b-", lw=1.5, label="Simulated SoH")
    ax.axhline(0.80, color="tab:orange", ls="--", lw=0.9,
               label="EoL (0.80)")
    ax.axhline(SOH_EOSL, color="tab:red", ls="--", lw=0.9,
               label=f"EoSL ({SOH_EOSL:.2f})")
    if eosl_cycle is not None:
        ax.axvline(eosl_cycle, color="tab:red", ls=":", lw=0.8, alpha=0.6)
        ax.annotate(f"EoSL at cycle {eosl_cycle:.0f}",
                    xy=(eosl_cycle, SOH_EOSL),
                    xytext=(eosl_cycle * 0.55, SOH_EOSL + 0.08),
                    fontsize=10,
                    arrowprops=dict(arrowstyle="->", color="tab:red", lw=0.7))
    ax.set_xlabel("Cycle")
    ax.set_ylabel("SoH")
    ax.set_title(
        "PyBaMM long-horizon verification: DFN + SEI (solvent-diff) + "
        "plating, NO LAM\n"
        f"0.25 C, 25 degC, {int(df.cycle_n.iloc[-1])} cycles"
    )
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left", fontsize=9)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "full_trajectory.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def write_report(df: pd.DataFrame | None, meta: dict):
    lines = []
    lines.append("# PyBaMM long-horizon verification report\n")
    lines.append(f"- Date: {time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    lines.append(f"- Script: `src/simulation/verify_long_run.py`")
    lines.append(f"- Total wall time: {meta['elapsed_total_s']/60:.1f} min")
    lines.append(f"- Peak self-process RSS: {meta['peak_self_rss_mb']:.0f} MB")
    lines.append(f"- Aborted early (any batch failed twice)? {meta['aborted']}")
    lines.append("")
    lines.append("## Model configuration")
    lines.append("```")
    for k, v in MODEL_OPTIONS.items():
        lines.append(f"{k}: {v}")
    lines.append("(loss of active material intentionally disabled)")
    lines.append("```")
    lines.append("")
    lines.append("## Degradation parameter overrides")
    lines.append("```")
    for k, v in DEG_OVERRIDES.items():
        lines.append(f"{k}: {v}")
    lines.append("```")
    lines.append("")
    lines.append("## Per-batch summary")
    lines.append("| batch | attempts | cycles_ok | elapsed_s | Q_Ah first -> last |")
    lines.append("|-------|----------|-----------|-----------|---------------------|")
    for bs in meta["batch_summaries"]:
        lines.append(f"| {bs['batch']} | {bs['attempts']} | {bs['n_cycles_ok']} | "
                     f"{bs['elapsed_s']:.1f} | "
                     f"{bs['Q_Ah_first']:.3f} -> {bs['Q_Ah_last']:.3f} |")
    lines.append("")

    if df is None or df.empty:
        lines.append("## Result: NO DATA — no batch completed successfully.")
        (OUT_DIR / "report.md").write_text("\n".join(lines))
        return "\n".join(lines)

    total_cycles = int(df.cycle_n.iloc[-1])
    final_soh = float(df.SOH.iloc[-1])
    eosl_cycle = cycle_at_soh(df, SOH_EOSL)
    eol_cycle = cycle_at_soh(df, 0.80)

    lines.append("## Fade rate per 1000-cycle segment (pp/1000cy)")
    lines.append("| segment (cycles) | fade rate |")
    lines.append("|------------------|-----------|")
    for lo in range(0, total_cycles, 1000):
        hi = min(lo + 1000, total_cycles)
        r = fade_rate_per_1000cy(df, lo + 1, hi)
        lines.append(f"| {lo+1}-{hi} | {r:.2f} |")
    lines.append("")

    lines.append("## Key results")
    lines.append(f"- q0 (cycle-1 discharge capacity, Ah): {meta.get('q0_Ah', float('nan')):.3f}")
    lines.append(f"- Total cycles simulated: {total_cycles}")
    lines.append(f"- Final SoH: {final_soh:.4f}")
    lines.append(f"- Cycle at SoH 0.80 (EoL): "
                 f"{'{:.0f}'.format(eol_cycle) if eol_cycle else 'not reached'}")
    lines.append(f"- Cycle at SoH 0.40 (EoSL): "
                 f"{'{:.0f}'.format(eosl_cycle) if eosl_cycle else 'not reached'}")
    lines.append("")

    # Verdict
    monotonic = bool((df.SOH.diff().dropna() <= 1e-4).all())
    smoothness_ok = monotonic
    reached_eosl = eosl_cycle is not None
    target_low, target_high = 4400 * 0.7, 4400 * 1.3
    on_target = (eosl_cycle is not None and target_low <= eosl_cycle <= target_high)

    lines.append("## Verdict")
    lines.append(f"- Monotonic decrease (no rebound): **{monotonic}**")
    lines.append(f"- Reached SoH 0.40: **{reached_eosl}**")
    if reached_eosl:
        lines.append(f"- Cycle-to-EoSL within +/-30% of the paper's ~4400 target "
                     f"(3080 <= x <= 5720): **{on_target}**")
    if reached_eosl and smoothness_ok:
        lines.append("- **Overall: physically plausible curve — YES.**")
    else:
        lines.append("- **Overall: physically plausible curve — see caveats.**")
    (OUT_DIR / "report.md").write_text("\n".join(lines))
    return "\n".join(lines)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main() -> int:
    print("=== PyBaMM long-horizon verification ===", flush=True)
    print(f"Cycles per batch: {CYCLES_PER_BATCH}, batches: {N_BATCHES}", flush=True)
    print(f"C-rate: {C_RATE}, target EoSL: {SOH_EOSL}", flush=True)
    print(f"Model options: {MODEL_OPTIONS}", flush=True)
    print(f"Deg overrides: {DEG_OVERRIDES}", flush=True)
    print(f"System mem at start: {_mem_snapshot()}", flush=True)

    df, meta = run_batches()

    if df is not None and not df.empty:
        eosl_cycle = cycle_at_soh(df, SOH_EOSL)
        make_plot(df, eosl_cycle)
        print(f"\nWrote {OUT_DIR/'full_trajectory.png'}", flush=True)
        print(f"Wrote {OUT_DIR/'full_trajectory.parquet'}", flush=True)

    report = write_report(df, meta)
    print(f"\nWrote {OUT_DIR/'report.md'}", flush=True)
    print("\n--- report.md ---\n" + report, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
