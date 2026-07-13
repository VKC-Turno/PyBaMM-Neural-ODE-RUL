"""
Voltaris/Data_Exploration/phase3_v7_validate.py
================================================

Held-out validation for the v7 encoder-decoder operator on CALB_0029
(single-cell PoC). Uses the first K=50 observed cycles as encoder input,
forecasts cycles [K, K + forecast_len], compares the in-coverage segment
to the ground truth (RMSE in percentage points), and extrapolates the
remainder as a projection to lower SoH.

Outputs
-------
- JSON report:  outputs/results/phase3_v7_heldout_CALB_0029.json
- PDF figure:   paper/figures/heldout_v7_CALB_0029.pdf

CLI
---
    .venv/bin/python -u Voltaris/Data_Exploration/phase3_v7_validate.py \
        --checkpoint outputs/models/pinn_phase3_v7_operator.pt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from phase3_v7_operator import OperatorV7, X_HEALTH_DIM, THETA_DIM  # noqa: E402
from Voltaris.Data_Exploration.phase3_validate import (  # noqa: E402
    _load_longterm_soh,
    _load_theta_norm,
    _load_x_health,
)


DEFAULT_CHECKPOINT = _PROJECT_ROOT / "outputs" / "models" / "pinn_phase3_v7_operator.pt"
DEFAULT_JSON       = _PROJECT_ROOT / "outputs" / "results" / "phase3_v7_heldout_CALB_0029.json"
DEFAULT_PDF        = _PROJECT_ROOT / "paper" / "figures" / "heldout_v7_CALB_0029.pdf"

CELL_ID = "0029"
MAKE = "CALB"
DEFAULT_K = 50
DEFAULT_FORECAST_LEN = 500


# ---------------------------------------------------------------------------
# Checkpoint loader
# ---------------------------------------------------------------------------
def load_v7_operator(path: str | Path) -> tuple[OperatorV7, dict]:
    """Load an OperatorV7 checkpoint and return (model, checkpoint_dict).

    Restores the state_dict (which contains xh_mean/xh_std/th_mean/th_std
    as registered buffers), so the model is ready to run _normalise_xh
    internally on raw physical x_health.
    """
    ckpt = torch.load(str(path), weights_only=False, map_location="cpu")
    cfg = ckpt.get("config", {})
    K = int(cfg.get("K", DEFAULT_K))
    model = OperatorV7(K=K)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


# ---------------------------------------------------------------------------
# Forecast
# ---------------------------------------------------------------------------
def forecast_v7(model: OperatorV7,
                cell_id: str,
                make: str,
                K: int,
                forecast_len: int,
                smooth_window: int = 1,   # 1 = no smoothing (default);
                                          # smoothing tested at w=5 hurts
                                          # RMSE (EVE_0003 0.83 -> 1.20 pp),
                                          # kept as an experimental knob only.
                ) -> dict:
    """Run the v7 operator on ``{make}_{cell_id}`` and return a report dict.

    Steps
    -----
    1. Load per-cycle observed SoH (nameplate-normalised via phase3_validate).
    2. Apply a centred rolling-median filter (window=smooth_window) to the
       first K observed cycles to suppress cycle-to-cycle measurement noise
       before feeding the encoder. Raw obs (unsmoothed) is still used for
       RMSE against the forecast.
    3. Build context_delta = smoothed_obs[:K] - smoothed_obs[0].
    4. Build x_health (v7 uses [T, c_rate, DCIR]).
    5. Load theta_norm; integrate over target_cycles = [K, K+forecast_len].
    """
    obs_n, obs_soh = _load_longterm_soh(cell_id, make)
    if obs_n.size < K:
        raise ValueError(f"{make}_{cell_id}: only {obs_n.size} observed cycles, "
                         f"need at least K={K}")

    context_raw = obs_soh[:K].astype(np.float32)
    if smooth_window and smooth_window > 1:
        context_obs = (pd.Series(context_raw)
                        .rolling(int(smooth_window), center=True, min_periods=1)
                        .median()
                        .to_numpy(dtype=np.float32))
    else:
        context_obs = context_raw
    context_soh_start = float(context_obs[0])
    context_delta = (context_obs - context_soh_start).astype(np.float32)

    # v7 x_health: [T, c_rate, DCIR]. v6 helper returns 5 fields; drop last 2.
    xh_v6 = _load_x_health(cell_id, make, ambient_C=25.0, default_c_rate=0.5)
    x_health = xh_v6[:X_HEALTH_DIM].astype(np.float32)

    theta_norm, has_theta = _load_theta_norm(cell_id, make)
    theta_norm = theta_norm.astype(np.float32)

    # Target cycles: start at cycle K (t0 for the operator; forecast horizon
    # is a further forecast_len cycles). We use the actual observed cycle
    # numbers so grids match if the observed grid is dense-integer.
    start_cy = int(obs_n[K - 1])   # anchor at last context cycle
    target_cycles = np.arange(start_cy, start_cy + forecast_len + 1,
                              dtype=np.float32)

    with torch.no_grad():
        pred = model(
            torch.from_numpy(x_health).unsqueeze(0),
            torch.from_numpy(theta_norm).unsqueeze(0),
            torch.from_numpy(context_delta).unsqueeze(0),
            torch.tensor([context_soh_start], dtype=torch.float32),
            torch.from_numpy(target_cycles),
        ).squeeze(0).cpu().numpy().astype(np.float32)

    # In-coverage segment: pred at cycles that overlap observed cycles > K.
    covered_mask = (obs_n >= start_cy) & (obs_n <= target_cycles[-1])
    obs_n_cov = obs_n[covered_mask].astype(np.float32)
    obs_soh_cov = obs_soh[covered_mask].astype(np.float32)
    if obs_n_cov.size >= 2:
        pred_at_obs = np.interp(obs_n_cov, target_cycles, pred).astype(np.float32)
        diff = pred_at_obs - obs_soh_cov
        rmse_pp = float(np.sqrt(np.mean(diff * diff)) * 100.0)
    else:
        pred_at_obs = np.array([], dtype=np.float32)
        rmse_pp = float("nan")

    return {
        "cell_id": cell_id,
        "make": make,
        "K": int(K),
        "forecast_len": int(forecast_len),
        "context_start_cycle": int(obs_n[0]),
        "context_end_cycle": int(obs_n[K - 1]),
        "context_soh_start": context_soh_start,
        "context_soh_end": float(context_obs[-1]),
        "theta_norm_from_yaml": bool(has_theta),
        "x_health_raw": x_health.tolist(),
        # Grids
        "obs_cycles": obs_n.astype(int).tolist(),
        "obs_soh": obs_soh.tolist(),
        "obs_covered_cycles": obs_n_cov.astype(int).tolist(),
        "obs_covered_soh": obs_soh_cov.tolist(),
        "pred_cycles": target_cycles.astype(int).tolist(),
        "pred_soh": pred.tolist(),
        "pred_at_obs_soh": pred_at_obs.tolist(),
        # Metrics
        "n_covered": int(obs_n_cov.size),
        "rmse_pp_covered": rmse_pp,
        "pred_soh_end": float(pred[-1]),
    }


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def render_pdf(report: dict, out_pdf: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    obs_n = np.asarray(report["obs_cycles"], dtype=float)
    obs_s = np.asarray(report["obs_soh"], dtype=float)
    pred_n = np.asarray(report["pred_cycles"], dtype=float)
    pred_s = np.asarray(report["pred_soh"], dtype=float)
    K = int(report["K"])
    ctx_start_cy = int(report["context_start_cycle"])
    ctx_end_cy = int(report["context_end_cycle"])
    rmse = report["rmse_pp_covered"]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    # Grey shaded context window (cycles ctx_start_cy .. ctx_end_cy)
    ax.axvspan(ctx_start_cy, ctx_end_cy, color="0.85", alpha=0.55,
               label=f"context (first K={K} cycles)")
    # Blue observed
    ax.plot(obs_n, obs_s, color="#1f77b4", lw=1.8, label="observed SoH")
    # Red dashed forecast
    ax.plot(pred_n, pred_s, color="#d62728", lw=1.6, ls="--",
            label="v7 forecast")
    ax.set_xlabel("cycle number")
    ax.set_ylabel("SoH (nameplate-normalised)")
    x_max = float(max(pred_n.max(), obs_n.max()))
    ax.set_xlim(0.0, x_max)
    # Auto y-range with a small pad
    y_vals = np.concatenate([obs_s, pred_s])
    lo = float(np.nanmin(y_vals))
    hi = float(np.nanmax(y_vals))
    pad = 0.02 * max(hi - lo, 1e-3)
    ax.set_ylim(lo - pad, hi + pad)

    rmse_str = f"{rmse:.2f} pp" if isinstance(rmse, float) and not np.isnan(rmse) else "n/a"
    ax.set_title(f"{report['make']}_{report['cell_id']} — "
                 f"v7 held-out forecast (in-coverage RMSE = {rmse_str}, "
                 f"n_covered={report['n_covered']})")
    ax.grid(True, ls=":", alpha=0.5)
    ax.legend(loc="lower left", framealpha=0.9)
    fig.tight_layout()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, format="pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument("--out-json",   type=Path, default=DEFAULT_JSON)
    p.add_argument("--out-pdf",    type=Path, default=DEFAULT_PDF)
    p.add_argument("--k",          type=int, default=DEFAULT_K,
                   help="Context window length (encoder input cycles).")
    p.add_argument("--forecast-len", type=int, default=DEFAULT_FORECAST_LEN,
                   help="Forecast horizon in cycles beyond context.")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv or sys.argv[1:])
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"[v7_validate] checkpoint not found: {ckpt_path}", file=sys.stderr)
        return 2

    print(f"[v7_validate] loading checkpoint {ckpt_path}")
    model, ckpt = load_v7_operator(ckpt_path)
    print(f"[v7_validate] model params={model.n_parameters():,}, K={model.K}")
    print(f"[v7_validate] xh_mean={model.xh_mean.tolist()}, "
          f"xh_std={model.xh_std.tolist()}")

    report = forecast_v7(model, CELL_ID, MAKE, K=args.k,
                         forecast_len=args.forecast_len)
    report["checkpoint"] = str(ckpt_path)
    report["best_val"] = float(ckpt.get("best_val", float("nan")))

    print(f"[v7_validate] {MAKE}_{CELL_ID}: context [{report['context_start_cycle']}"
          f"..{report['context_end_cycle']}]  "
          f"soh_start={report['context_soh_start']:.4f}  "
          f"soh_end_ctx={report['context_soh_end']:.4f}  "
          f"n_covered={report['n_covered']}  "
          f"RMSE={report['rmse_pp_covered']:.3f} pp  "
          f"pred_soh_end={report['pred_soh_end']:.4f}")

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, default=str))
    print(f"[v7_validate] wrote {out_json}")

    render_pdf(report, Path(args.out_pdf))
    print(f"[v7_validate] wrote {args.out_pdf}")

    _ = X_HEALTH_DIM, THETA_DIM  # silence unused-import lint
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
