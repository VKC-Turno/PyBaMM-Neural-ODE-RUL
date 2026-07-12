"""
src/simulation/run_sweep.py
---------------------------
Sobol-sampled degradation sweep using PyBaMM DFN + SEI + plating + LAM
submodels. Each sample becomes one degradation simulation; results are
collected to a single parquet (one row per sample × cycle).

Usage
~~~~~
    .venv/bin/python -m src.simulation.run_sweep --n-samples 4 --n-cycles 20
        # dry run (~1 minute on 8 cores)

    .venv/bin/python -m src.simulation.run_sweep
        # full sweep using configs/sweep_config.yaml (~800 sims × 500 cycles)

Outputs (per `data/synthetic/`):
    trajectories.parquet         all per-cycle features stacked
    sweep_manifest.yaml          configuration snapshot + per-sample status
    ic_curves/sample_<id>/*.npz  per-cycle IC curve arrays
    failed/                      checkpoint files for samples that errored
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


SYNTH_DIR = Path("data/synthetic")
TRAJECTORIES_PATH = SYNTH_DIR / "trajectories.parquet"
MANIFEST_PATH = SYNTH_DIR / "sweep_manifest.yaml"
IC_DIR = SYNTH_DIR / "ic_curves"
FAILED_DIR = SYNTH_DIR / "failed"


# --- Mapping from sweep_config.yaml names to PyBaMM parameter keys ---
# We collapse the 6 sweep dimensions into actual PyBaMM keys; the rest
# (e.g. c_rate, temperature) drive the experiment string, not the
# parameter set.
_PYBAMM_KEY_MAP: dict[str, str] = {
    "k_SEI_ms":                          "SEI kinetic rate constant [m.s-1]",
    "SEI_partial_molar_volume_m3mol":   "SEI partial molar volume [m3.mol-1]",
    # `lithium_plating_exchange_current_A_m2` cannot be mapped directly to
    # OKane2022's `Exchange-current density for plating [A.m-2]` because the
    # latter is a function (of c_e, c_Li, T). Map it to the scalar plating
    # kinetic rate constant instead, scaled into m/s by F·c_Li_typ ≈ 9.6e7.
    "lithium_plating_exchange_current_A_m2": "Lithium plating kinetic rate constant [m.s-1]",
    "LAM_positive_rate_s":              "Positive electrode LAM constant proportional term [s-1]",
    # The graphite (negative) electrode LAM is the DOMINANT fade driver for
    # this LFP/graphite cell. At OKane2022's default 2.78e-7/s it gives ~700
    # cycles to 80%; ~2.78e-8/s lands on the spec's ≥4000-cycle rating.
    "LAM_negative_rate_s":              "Negative electrode LAM constant proportional term [s-1]",
}
_PLATING_RESCALE_TO_K_M_S = 1.0 / (96485.0 * 1000.0)  # divide A/m^2 by F * c_Li_typ


def _sobol_sample(ranges: dict[str, dict], n: int, seed: int = 123
                  ) -> pd.DataFrame:
    """Sobol-sample the configured parameter ranges."""
    from scipy.stats.qmc import Sobol

    keys = list(ranges.keys())
    rng = Sobol(d=len(keys), seed=seed, scramble=True)
    n_pow2 = 1 << max(0, int(np.ceil(np.log2(max(2, n)))))
    u = rng.random(n=n_pow2)[:n]
    rows = []
    for j, k in enumerate(keys):
        r = ranges[k]
        lo, hi = float(r["min"]), float(r["max"])
        if r.get("scale") == "log":
            vals = 10 ** (np.log10(max(lo, 1e-30)) + u[:, j] * (np.log10(hi) - np.log10(max(lo, 1e-30))))
        else:
            vals = lo + u[:, j] * (hi - lo)
        rows.append(vals)
    df = pd.DataFrame(np.array(rows).T, columns=keys)
    df.insert(0, "sample_id", [f"s{i:05d}" for i in range(len(df))])
    return df


def _build_param_with_overrides(sample_row: dict) -> "pybamm.ParameterValues":
    import pybamm
    from src.simulation._pybamm_setup import (
        build_parameter_values,
        overrides_from_identified_params,
    )

    overrides: dict[str, Any] = dict(overrides_from_identified_params())
    for sweep_key, pybamm_key in _PYBAMM_KEY_MAP.items():
        if sweep_key in sample_row and pd.notna(sample_row[sweep_key]):
            val = float(sample_row[sweep_key])
            if sweep_key == "lithium_plating_exchange_current_A_m2":
                val = val * _PLATING_RESCALE_TO_K_M_S
            overrides[pybamm_key] = val

    # Ambient + initial temperature
    if "temperature_K" in sample_row and pd.notna(sample_row["temperature_K"]):
        T = float(sample_row["temperature_K"])
        overrides["Ambient temperature [K]"] = T
        overrides["Initial temperature [K]"] = T

    return build_parameter_values(overrides=overrides)


def _build_experiment(c_rate: float, n_cycles: int) -> "pybamm.Experiment":
    import pybamm
    block = (
        f"Discharge at {c_rate:.4f}C until 2.5 V",
        "Rest for 10 minutes",
        f"Charge at {c_rate:.4f}C until 3.65 V",
        "Hold at 3.65 V until C/100",
        "Rest for 10 minutes",
    )
    return pybamm.Experiment([block] * int(n_cycles))


def run_one_simulation(sample_row: dict, n_cycles: int,
                       timeout_s: int = 600,
                       save_ic_dir: Path | None = None) -> dict:
    """Run a single degradation simulation and return per-cycle features."""
    import pybamm
    from src.simulation._pybamm_setup import build_dfn
    from src.simulation.extract_features import per_cycle_features

    t0 = time.time()
    sample_id = sample_row["sample_id"]
    try:
        model = build_dfn()
        param = _build_param_with_overrides(sample_row)
        experiment = _build_experiment(float(sample_row["c_rate"]), n_cycles)
        # IDAKLU is much more robust than CasadiSolver for DAEs with
        # stiff degradation source terms; the sweep-extreme parameter
        # combinations cause CasadiSolver to repeatedly exhaust its
        # internal step budget and slow down by 100×.
        try:
            solver = pybamm.IDAKLUSolver(rtol=1e-6, atol=1e-6)
        except Exception:
            solver = pybamm.CasadiSolver(mode="safe", dt_max=600.0)
        sim = pybamm.Simulation(model, parameter_values=param,
                                experiment=experiment, solver=solver)
        sol = sim.solve()

        params_used = {k: sample_row[k] for k in sample_row.keys()}
        features = per_cycle_features(sol, params_used=params_used,
                                      save_ic_dir=save_ic_dir)
        if "sample_id" not in features.columns:
            features.insert(0, "sample_id", sample_id)
        return {
            "sample_id": sample_id,
            "status": "ok",
            "n_cycles_completed": int(features["cycle_n"].max() or 0) if not features.empty else 0,
            "elapsed_s": time.time() - t0,
            "features": features,
        }
    except Exception as e:
        tb = traceback.format_exc()
        return {
            "sample_id": sample_id,
            "status": "failed",
            "error": f"{type(e).__name__}: {e}",
            "traceback": tb,
            "elapsed_s": time.time() - t0,
            "features": pd.DataFrame(),
        }


def _run_in_subprocess_pool(rows: list[dict], save_dirs: list[Path | None],
                            n_cycles: int, n_jobs: int, timeout_s: int
                            ) -> list[dict]:
    """
    Dispatch each sample as a `python -m src.simulation._one_sim` subprocess.

    A ThreadPoolExecutor with `n_jobs` workers serialises subprocess.run
    calls (each call uses one CPU core, so n_jobs threads ≈ n_jobs cores).
    A subprocess timeout kills any sim that hangs; on TimeoutExpired we
    fabricate a failure result and continue. Loky / joblib's all-or-nothing
    failure mode is avoided entirely.
    """
    import concurrent.futures
    import multiprocessing as mp
    import pickle
    import subprocess
    import sys as _sys
    import tempfile
    import shutil

    if n_jobs <= 0:
        n_jobs = mp.cpu_count()

    tmp_root = Path(tempfile.mkdtemp(prefix="lfp_sweep_"))
    results: list[dict] = [None] * len(rows)
    completed = 0
    t_start = time.time()

    def _one(idx: int) -> dict:
        row = rows[idx]
        save_ic = save_dirs[idx]
        sid = row["sample_id"]
        in_path = tmp_root / f"{sid}.json"
        out_path = tmp_root / f"{sid}.pkl"
        in_path.write_text(json.dumps(row, default=str))
        argv = [
            _sys.executable, "-m", "src.simulation._one_sim",
            str(in_path), str(out_path),
            "--n-cycles", str(n_cycles),
        ]
        if save_ic is not None:
            argv += ["--save-ic-dir", str(save_ic)]
        t0 = time.time()
        try:
            subprocess.run(argv, timeout=timeout_s, check=False,
                           stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL,
                           cwd=str(Path(__file__).resolve().parents[2]))
        except subprocess.TimeoutExpired:
            return {
                "sample_id": sid,
                "status": "failed",
                "error": f"TimeoutExpired: exceeded {timeout_s}s wall-clock",
                "traceback": "",
                "elapsed_s": time.time() - t0,
                "features": pd.DataFrame(),
            }
        # Read pickled result
        if out_path.exists():
            try:
                with open(out_path, "rb") as f:
                    return pickle.load(f)
            except Exception as e:
                return {
                    "sample_id": sid, "status": "failed",
                    "error": f"unpickle: {type(e).__name__}: {e}",
                    "traceback": "", "elapsed_s": time.time() - t0,
                    "features": pd.DataFrame(),
                }
        return {
            "sample_id": sid, "status": "failed",
            "error": "no output file produced (subprocess likely crashed)",
            "traceback": "", "elapsed_s": time.time() - t0,
            "features": pd.DataFrame(),
        }

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_jobs) as pool:
            futures = {pool.submit(_one, i): i for i in range(len(rows))}
            for fut in concurrent.futures.as_completed(futures):
                idx = futures[fut]
                results[idx] = fut.result()
                completed += 1
                # Light progress log so the user can see liveness
                if completed % max(1, len(rows) // 20) == 0 or completed == len(rows):
                    ok = sum(1 for r in results if r and r.get("status") == "ok")
                    failed = sum(1 for r in results if r and r.get("status") != "ok")
                    print(f"  [{completed:4d}/{len(rows)}] "
                          f"ok={ok} failed={failed} "
                          f"elapsed={time.time()-t_start:.0f}s "
                          f"({results[idx]['sample_id']}: "
                          f"{results[idx]['status']}, "
                          f"{results[idx]['elapsed_s']:.1f}s)",
                          flush=True)
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)

    return results


def run_sweep(n_samples: int, n_cycles: int, n_jobs: int = -1,
              seed: int = 123, save_ic_curves: bool = True,
              min_crate: float | None = None,
              config_path: Path = Path("configs/sweep_config.yaml")
              ) -> dict:
    cfg = yaml.safe_load(config_path.read_text())
    ranges = cfg.get("degradation_parameters", {})
    if min_crate is not None and "c_rate" in ranges:
        ranges = dict(ranges)
        ranges["c_rate"] = dict(ranges["c_rate"])
        ranges["c_rate"]["min"] = float(max(min_crate, ranges["c_rate"]["min"]))
        print(f"  (overriding c_rate min → {ranges['c_rate']['min']})")
    sweep_df = _sobol_sample(ranges, n=n_samples, seed=seed)

    SYNTH_DIR.mkdir(parents=True, exist_ok=True)
    FAILED_DIR.mkdir(parents=True, exist_ok=True)
    if save_ic_curves:
        IC_DIR.mkdir(parents=True, exist_ok=True)

    timeout_s = int(cfg.get("compute", {}).get("timeout_per_sim_s", 600))
    print(f"→ Sweeping {n_samples} samples × {n_cycles} cycles "
          f"(jobs={n_jobs}, seed={seed}, per-sim timeout={timeout_s}s)")
    print(f"  parameter dims: {list(ranges.keys())}")

    rows = sweep_df.to_dict(orient="records")
    save_dirs = ([IC_DIR / r["sample_id"] for r in rows]
                 if save_ic_curves else [None] * len(rows))

    # Each sim runs in its own Python subprocess so a hang or C-level
    # crash inside PyBaMM/IDAKLU only kills *that* sim, not the pool.
    t0 = time.time()
    results = _run_in_subprocess_pool(
        rows, save_dirs, n_cycles=n_cycles, n_jobs=n_jobs,
        timeout_s=timeout_s,
    )
    elapsed = time.time() - t0

    # Aggregate features
    all_features = [r["features"] for r in results if r["status"] == "ok" and not r["features"].empty]
    if all_features:
        traj = pd.concat(all_features, ignore_index=True)
        traj.to_parquet(TRAJECTORIES_PATH, index=False)
        print(f"  wrote {len(traj):,} rows → {TRAJECTORIES_PATH}")
    else:
        traj = pd.DataFrame()
        print("  WARNING: no successful simulations")

    # Write failure traces for forensic review
    n_failed = 0
    for r in results:
        if r["status"] != "ok":
            n_failed += 1
            (FAILED_DIR / f"{r['sample_id']}.txt").write_text(
                f"{r['error']}\n\n{r['traceback']}"
            )

    # Sweep manifest
    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "config_snapshot": cfg,
        "n_samples_requested": int(n_samples),
        "n_samples_succeeded": int(sum(1 for r in results if r["status"] == "ok")),
        "n_samples_failed": n_failed,
        "n_cycles_requested": int(n_cycles),
        "seed": int(seed),
        "elapsed_seconds": elapsed,
        "per_sample": [
            {k: r[k] for k in ("sample_id", "status", "n_cycles_completed",
                               "elapsed_s") if k in r}
            for r in results
        ],
    }
    MANIFEST_PATH.write_text(yaml.safe_dump(manifest, sort_keys=False))
    print(f"  wrote manifest → {MANIFEST_PATH}")

    return {
        "n_total": len(results),
        "n_ok": sum(1 for r in results if r["status"] == "ok"),
        "n_failed": n_failed,
        "elapsed_seconds": elapsed,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="PyBaMM degradation sweep")
    ap.add_argument("--n-samples", type=int, default=None,
                    help="Override number of Sobol samples (default: from sweep_config.yaml)")
    ap.add_argument("--n-cycles", type=int, default=None,
                    help="Override cycles per simulation (default: from sweep_config.yaml)")
    ap.add_argument("--n-jobs", type=int, default=-1,
                    help="Parallel workers (default: -1 = all cores)")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--no-ic-curves", action="store_true",
                    help="Skip per-cycle IC curve saving (speeds dry runs)")
    ap.add_argument("--min-crate", type=float, default=None,
                    help="Override the lower bound on the C-rate sweep range "
                         "(useful for dry runs — low C-rates make each cycle a "
                         "very long physical-time integration).")
    ap.add_argument("--config", type=Path,
                    default=Path("configs/sweep_config.yaml"),
                    help="Sweep config file (default: configs/sweep_config.yaml)")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    n_samples = args.n_samples or int(cfg["sweep"]["n_samples"])
    n_cycles = args.n_cycles or int(cfg["sweep"]["n_cycles_per_sim"])
    seed = args.seed if args.seed is not None else int(cfg["sweep"].get("seed", 123))

    # n_jobs precedence: explicit --n-jobs (if not the -1 default) > config
    # compute.n_jobs > -1. Running at -1 (all cores) with long sims can
    # trigger systemd-oomd, so honour the config's conservative value.
    n_jobs = args.n_jobs
    if n_jobs == -1:
        n_jobs = int(cfg.get("compute", {}).get("n_jobs", -1))

    summary = run_sweep(n_samples=n_samples, n_cycles=n_cycles,
                        n_jobs=n_jobs, seed=seed,
                        save_ic_curves=not args.no_ic_curves,
                        min_crate=args.min_crate,
                        config_path=args.config)
    print(json.dumps(summary, indent=2))
    return 0 if summary["n_ok"] > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
