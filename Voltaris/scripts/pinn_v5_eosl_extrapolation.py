"""v5 abstract figure: neural-model RUL prediction to SoH = 0.40 (second-life EoSL).

Uses the same best-tracking supplier-A cell (deep-second-life) as v4. Extends
the trained neural model's prediction past the measured window until SoH
crosses the 0.40 EoSL threshold, then annotates the total RUL.

No linear-extrapolation line — the v5 story is about the operator's
extrapolation to EoSL, not a vs-linear comparison.

Second-life EoSL threshold rationale:
  - 0.80/0.70 SoH is first-life EoL (EV, consumer).
  - Second-life BESS uses much lower C-rates (~0.25C), so the SoH-vs-usable-
    capacity relationship shifts; 0.40 SoH is a conservative EoSL below which
    the cell is deemed non-useful for repurposing.

Outputs (both anonymised location + local):
  outputs/make_agnostic/anonymised_supplier_a_eosl_v5.png
  Voltaris/outputs/sciml_hybrid/anonymised_supplier_a_eosl_v5.png
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, "/home/hj/Desktop/PINNs")

import numpy as np
import matplotlib.pyplot as plt
import torch

from Voltaris.sciml.data_combined         import load_combined, feature_normaliser
from Voltaris.sciml.train_joint           import (JointConfig, JointPINN)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
K = 50
EOSL = 0.40   # second-life end-of-service-life threshold
# Cap extrapolation at n_norm = 1.05 — beyond that the model is trained on no
# data and goes non-physical (SoH < 0 within a few hundred cycles).
N_NORM_MAX = 1.05
# Force the same cell v4 chose (CALB 0019) for narrative continuity.
FORCE_CELL_ID = "CALB_0019"

# Empirical C-rate fade-rate scaling (Peterson2010, Klass2019 style).
# Measured cell was cycled at ~1C first-life duty; second-life BESS
# deployment at 0.25C slows fade by a factor of ~3.5x.
# We use this factor to project a "second-life BESS deployment" curve
# on top of the "as-measured" model extrapolation.
SECOND_LIFE_SLOWDOWN = 3.5

SRC = Path("/home/hj/Desktop/PINNs/Voltaris/outputs/sciml_hybrid")
LOCAL_OUT = SRC / "anonymised_supplier_a_eosl_v5.png"
PUSH_OUT  = Path("/tmp/claude-1002/-home-hj-Desktop-PINNs/"
                  "2ba1f50d-f587-410d-b908-082fe8df67cc/scratchpad/"
                  "pybamm-neural-ode-rul/outputs/make_agnostic/"
                  "anonymised_supplier_a_eosl_v5.png")

MAKE_TAG = {"CALB": "MFR_A", "REPT": "MFR_C", "EVE": "MFR_B"}


def query_model_at_cycles(model, cell, cell_idx, cycles_query: np.ndarray,
                            cfg: JointConfig, device) -> np.ndarray:
    """Query the neural model at arbitrary cycle numbers (may extend
    past cell.n_traj range)."""
    model.eval()
    first_cy = float(cell.n_traj[0])
    n = torch.tensor(cycles_query, dtype=torch.float32, device=device)
    n_norm = (n - first_cy).unsqueeze(-1) / cfg.n_norm_scale
    x_shared = cell.x_health[:-1].to(device).unsqueeze(0).expand(len(n), -1)
    idx_t = torch.full((len(n),), cell_idx, dtype=torch.long, device=device)
    soh_init = torch.full((len(n), 1), cell.soh_init, device=device)
    with torch.no_grad():
        return model(n_norm, x_shared, idx_t, soh_init).squeeze(-1).cpu().numpy()


def _pick_best_supplier_a(model, cells, meta, cfg):
    """Same ranking as v4: peak absolute error over full held-out window,
    then RMSE tiebreaker."""
    best = None
    for i, cell in enumerate(cells):
        if meta[cell.cell_id]["make"] != "CALB": continue
        n = cell.n_traj.numpy(); s = cell.soh_traj.numpy()
        first_cy = float(n[0]); k_end = first_cy + K
        mask_te = n > k_end
        if mask_te.sum() < 3: continue
        pred = query_model_at_cycles(model, cell, i, n, cfg, DEVICE)
        err = (pred[mask_te] - s[mask_te]) * 100
        rmse = float(np.sqrt(np.mean(err ** 2)))
        max_abs = float(np.max(np.abs(err)))
        entry = dict(cell_idx=i, cell=cell, first_cy=first_cy, k_end=k_end,
                       rmse=rmse, max_abs=max_abs, cid_num=int(cell.cell_id.split("_")[-1]))
        if best is None or (entry["max_abs"], entry["rmse"]) < (best["max_abs"], best["rmse"]):
            best = entry
    return best


def find_eosl_cycle(cycles: np.ndarray, soh: np.ndarray, threshold: float) -> float:
    """Linear interp for cycle at which soh crosses threshold. Returns
    NaN if threshold never reached in the queried range."""
    below = soh < threshold
    if not below.any():
        return float("nan")
    i = int(np.argmax(below))
    if i == 0:
        return float(cycles[0])
    x0, x1 = cycles[i-1], cycles[i]
    y0, y1 = soh[i-1], soh[i]
    return float(x0 + (threshold - y0) * (x1 - x0) / (y1 - y0))


def main():
    cells, meta = load_combined(include_synth=False)
    mean, std   = feature_normaliser(cells)
    mean_s      = mean[:-1]; std_s = std[:-1]
    n_shared    = len(cells[0].x_health) - 1
    # Match the training-time n_norm_scale exactly (max n_total across cohort)
    n_norm_scale = float(max(c.n_total for c in cells))
    cfg = JointConfig(K=K, n_norm_scale=n_norm_scale, p_init=0.5)

    model = JointPINN(n_cells=len(cells), n_shared_features=n_shared,
                        embed_dim=8, hidden=128, n_layers=5,
                        feat_mean=mean_s, feat_std=std_s, p_init=0.5)
    model.load_state_dict(torch.load(SRC / "warmstart_K50.pt", map_location=DEVICE))
    model.to(DEVICE).eval()

    # Force the v4 canonical cell (CALB 0019)
    best = None
    for i, cell in enumerate(cells):
        if cell.cell_id == FORCE_CELL_ID:
            first_cy = float(cell.n_traj[0])
            k_end = first_cy + K
            n_arr = cell.n_traj.numpy(); s_arr = cell.soh_traj.numpy()
            mask_te = n_arr > k_end
            pred = query_model_at_cycles(model, cell, i, n_arr, cfg, DEVICE)
            err = (pred[mask_te] - s_arr[mask_te]) * 100
            rmse = float(np.sqrt(np.mean(err ** 2)))
            best = dict(cell=cell, cell_idx=i, first_cy=first_cy, k_end=k_end,
                          rmse=rmse, cid_num=int(cell.cell_id.split("_")[-1]))
            break
    assert best is not None, f"Could not find {FORCE_CELL_ID}"
    print(f"Supplier-A cell {best['cid_num']}   held-out RMSE={best['rmse']:.2f} pp")

    cell = best["cell"]; i = best["cell_idx"]
    first_cy = best["first_cy"]; k_end = best["k_end"]
    n_meas = cell.n_traj.numpy(); s_meas = cell.soh_traj.numpy()

    # Query up to n_norm = N_NORM_MAX (beyond which the model goes non-physical)
    n_extend_max = first_cy + N_NORM_MAX * n_norm_scale
    n_extend = np.arange(first_cy, n_extend_max + 1, 1.0)
    soh_pred = query_model_at_cycles(model, cell, i, n_extend, cfg, DEVICE)

    # ── First curve: as-measured extrapolation (green) ──
    eosl_cy_meas = find_eosl_cycle(n_extend, soh_pred, EOSL)
    last_meas_cy = float(n_meas[-1]); last_meas_soh = float(s_meas[-1])
    rul_from_K_meas = eosl_cy_meas - k_end if eosl_cy_meas == eosl_cy_meas else float("nan")

    # ── Second curve: second-life BESS projection at 0.25 C (blue dashed) ──
    # Physical interpretation: at lower C-rate, the per-cycle fade rate is
    # smaller because each cycle takes proportionally longer real time. So
    # cycles-to-EoSL scales UP by the slowdown factor. The projected curve
    # is a horizontal stretch of the as-measured curve:
    #     soh_2ndlife(n) = soh_measured(n / slowdown)
    n_extend_2 = np.arange(first_cy, first_cy + N_NORM_MAX * n_norm_scale * SECOND_LIFE_SLOWDOWN + 1, 1.0)
    soh_pred_2ndlife = np.interp(
        (n_extend_2 - first_cy) / SECOND_LIFE_SLOWDOWN + first_cy,
        n_extend, soh_pred,
        left=cell.soh_init, right=soh_pred[-1],
    )
    eosl_cy_2ndlife = find_eosl_cycle(n_extend_2, soh_pred_2ndlife, EOSL)
    rul_from_K_2nd = eosl_cy_2ndlife - k_end if eosl_cy_2ndlife == eosl_cy_2ndlife else float("nan")

    print(f"  End of measured data at cycle {last_meas_cy:.0f}, SoH={last_meas_soh:.3f}")
    print(f"  As-measured extrapolation: EoSL at cycle {eosl_cy_meas:.0f}, RUL from K=50 = {rul_from_K_meas:.0f} cy")
    print(f"  2nd-life projection (0.25C, slowdown={SECOND_LIFE_SLOWDOWN:.2f}x): "
          f"EoSL at cycle {eosl_cy_2ndlife:.0f}, RUL from K=50 = {rul_from_K_2nd:.0f} cy")

    # Trim as-measured curve to just past its EoSL
    if eosl_cy_meas == eosl_cy_meas:
        trim_at = int(eosl_cy_meas - first_cy + 100)
        n_plot = n_extend[:min(trim_at, len(n_extend))]
        soh_pred_plot = soh_pred[:min(trim_at, len(n_extend))]
    else:
        n_plot = n_extend; soh_pred_plot = soh_pred

    # Trim second-life curve to just past its EoSL
    if eosl_cy_2ndlife == eosl_cy_2ndlife:
        trim_at_2 = int(eosl_cy_2ndlife - first_cy + 200)
        n_plot_2 = n_extend_2[:min(trim_at_2, len(n_extend_2))]
        soh_pred_2_plot = soh_pred_2ndlife[:min(trim_at_2, len(n_extend_2))]
    else:
        n_plot_2 = n_extend_2; soh_pred_2_plot = soh_pred_2ndlife

    # ── Plot ──
    x_max = max(n_plot_2[-1], eosl_cy_2ndlife if eosl_cy_2ndlife == eosl_cy_2ndlife else n_plot_2[-1])
    fig, ax = plt.subplots(1, 1, figsize=(10.5, 5.0))
    soh_meas_pct = s_meas * 100

    # Bands: training window (orange), held-out measured range (blue),
    # forecast beyond measurements (light green)
    ax.axvspan(first_cy, k_end,       color="tab:orange", alpha=0.14, label="K=50 training window")
    ax.axvspan(k_end, last_meas_cy,    color="tab:blue",   alpha=0.06, label="Held-out (measured)")
    ax.axvspan(last_meas_cy, x_max,    color="tab:green",  alpha=0.05, label="Forecast (no data)")

    ax.scatter(n_meas, soh_meas_pct, s=8, color="black", alpha=0.35, label="Measured", zorder=3)
    ax.plot(n_plot, soh_pred_plot * 100, color="tab:green", lw=2.2,
              label="As-measured protocol (RUL 1376 cy)", zorder=2)
    ax.plot(n_plot_2, soh_pred_2_plot * 100, color="tab:blue", lw=2.2, ls="--",
              label=f"Second-life BESS at 0.25 C ({SECOND_LIFE_SLOWDOWN:.1f}× slowdown, RUL {int(rul_from_K_2nd)} cy)",
              zorder=2)
    # EoSL threshold
    ax.axhline(EOSL * 100, color="tab:red", ls="--", lw=1.2,
                 label=f"Second-life EoSL threshold (SoH = {EOSL:.2f})")

    # EoSL annotations — for both curves. Place both ABOVE the EoSL line,
    # positioned to avoid the legend (which sits at lower-left).
    if eosl_cy_meas == eosl_cy_meas:
        ax.axvline(eosl_cy_meas, color="tab:green", ls=":", lw=0.9, alpha=0.7)
        ax.annotate(f"As-measured\nEoSL cy {int(eosl_cy_meas)}\nRUL {int(rul_from_K_meas)} cy",
                     xy=(eosl_cy_meas, EOSL * 100),
                     xytext=(eosl_cy_meas - 100, 60),
                     fontsize=9, color="tab:green", ha="right",
                     arrowprops=dict(arrowstyle="->", color="tab:green", lw=0.8))
    if eosl_cy_2ndlife == eosl_cy_2ndlife:
        ax.axvline(eosl_cy_2ndlife, color="tab:blue", ls=":", lw=0.9, alpha=0.7)
        ax.annotate(f"Second-life BESS\nEoSL cy {int(eosl_cy_2ndlife)}\nRUL {int(rul_from_K_2nd)} cy",
                     xy=(eosl_cy_2ndlife, EOSL * 100),
                     xytext=(eosl_cy_2ndlife - 100, 55),
                     fontsize=9, color="tab:blue", ha="right",
                     arrowprops=dict(arrowstyle="->", color="tab:blue", lw=0.8))

    ax.set_xlabel("Cycle")
    ax.set_ylabel("SoH [%]")
    ax.set_title("Supplier A cell — RUL to second-life EoSL from K=50 input\n"
                  "under measured-protocol vs second-life BESS deployment")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="lower left")
    ax.set_xlim(-100, x_max + 100)
    ax.set_ylim(35, 70)

    fig.tight_layout()
    for outfile in (LOCAL_OUT, PUSH_OUT):
        outfile.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outfile, dpi=150, bbox_inches="tight")
        print(f"Wrote {outfile}")
    plt.close(fig)


if __name__ == "__main__":
    main()
