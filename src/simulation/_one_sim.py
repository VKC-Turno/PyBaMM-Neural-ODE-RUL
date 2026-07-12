"""
src/simulation/_one_sim.py
--------------------------
Run a single PyBaMM degradation simulation in this process.

Usage (intended to be invoked by `run_sweep.py` as a subprocess):
    .venv/bin/python -m src.simulation._one_sim INPUT_JSON OUTPUT_PKL [--n-cycles N]

`INPUT_JSON` must contain a single dict with the full sample row
(sample_id + sweep parameter values). The result — same shape as the
return value of `run_one_simulation` from `run_sweep.py` — is written as
a pickle to `OUTPUT_PKL`. Any unhandled exception is captured in the
result so the parent does not crash on this sim.

Each simulation runs in a fresh Python process, so a SIGKILL from the
parent (on timeout) kills only this single sim without affecting the
pool or sibling sims.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

# Make the project importable when launched as -m src.simulation._one_sim
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.simulation.run_sweep import run_one_simulation  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("input_json", help="path to JSON file with the sample dict")
    ap.add_argument("output_pkl", help="path where the result pickle is written")
    ap.add_argument("--n-cycles", type=int, required=True)
    ap.add_argument("--save-ic-dir", default="", help="optional dir for IC curves")
    args = ap.parse_args()

    sample = json.loads(Path(args.input_json).read_text())
    save_ic = Path(args.save_ic_dir) if args.save_ic_dir else None
    try:
        result = run_one_simulation(sample, args.n_cycles, save_ic_dir=save_ic)
    except BaseException as e:   # truly catch everything, incl. KeyboardInterrupt
        import traceback
        result = {
            "sample_id": sample.get("sample_id", "?"),
            "status": "failed",
            "error": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc(),
            "elapsed_s": -1.0,
        }
        try:
            import pandas as pd
            result["features"] = pd.DataFrame()
        except Exception:
            pass

    Path(args.output_pkl).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_pkl, "wb") as f:
        pickle.dump(result, f)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
