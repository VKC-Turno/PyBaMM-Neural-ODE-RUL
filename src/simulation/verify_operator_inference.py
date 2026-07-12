"""
Operator inference + measured/DFN overlay for one cell.

Given a cell tag (e.g. "eve_0008" or "calb_0020") that has
    data/synthetic/verification/<tag>_bol_params.yaml
    data/synthetic/verification/<tag>_deg_params.yaml
    data/synthetic/verification/<tag>_longrun.parquet
this script:
    1. Loads the theta-conditioned DeepONet checkpoint.
    2. Constructs the fingerprint (dcir_fp, rpt_fp, soh_early K=50, theta_vec,
       protocol) using the per-cell identified BOL + fitted deg params and
       the K=50 measured SoH values (normalised to cy1 = 1.0).
    3. Rolls out the operator on n = 1..5000 (extrapolating past training
       horizon 1500 via the hard-monotonic softplus decrement).
    4. Overlays 3 lines (measured, DFN 5000-cy, operator) → PNG.
    5. Computes per-cell RMSE metrics and returns them.

Handles the training-time feature standardisation by rebuilding the same
`SyntheticTrajectoryDataset` and re-using its stats dict so the operator
sees inputs in the exact distribution it was trained on.
"""
from __future__ import annotations

import sys
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/home/hj/Desktop/PINNs")

from src.operator.model   import ThetaDeepONet, OperatorConfig
from src.operator.dataset import build_dataset, DatasetConfig


OUT_DIR = Path("/home/hj/Desktop/PINNs/data/synthetic/verification")
CKPT    = Path("/home/hj/Desktop/PINNs/outputs/models/theta_deeponet.pt")


@dataclass
class CellCase:
    tag: str                    # "eve_0008" or "calb_0020"
    manufacturer: str           # "EVE" or "CALB"
    cell_id: str                # "0008" or "0020"
    canonical_parquet: Path     # /home/hj/Desktop/PINNs/soh/data/canonical/eve.parquet


def _theta_deg_from_deg_yaml(deg: dict) -> tuple[float, float, float, float]:
    """Extract the four DE-fit params in (k_SEI, V_SEI, k_plating, D_SEI_solvent)
    named-parameter form. Note: the fit uses PyBaMM-style keys.
    """
    p = deg["best_parameters"]
    return (
        float(p["SEI kinetic rate constant [m.s-1]"]),
        float(p["SEI partial molar volume [m3.mol-1]"]),
        float(p["Lithium plating kinetic rate constant [m.s-1]"]),
        float(p["SEI solvent diffusivity [m2.s-1]"]),
    )


def build_fingerprint(cell: CellCase, ds, load_measured_fn) -> dict:
    """Return standardised torch tensors ready for ThetaDeepONet.forward.

    Rationale for each stream (see docstring):
      dcir_fp   : zeros (training corpus had constant zero → operator learned
                   to ignore; no signal to leak)
      rpt_fp    : the standardiser's mean (real cell's Ah scale is 45× the
                   synthetic PyBaMM-per-electrode scale; feeding the mean
                   keeps the branch in its trained regime and matches the
                   fact that these features carried no meaningful variation
                   in the training distribution)
      theta_vec : the fitted degradation params (log10) + identified BOL
                   scalars — this IS the meaningful conditioning
      protocol  : (0.5, 1.0, 298.15, 10) matching Prada2013 sweep protocol
      soh_early : first K=50 measured SoH, normalised to cy1=1.0
    """
    # ----- Standardisation stats -----
    m_dcir, s_dcir = ds.stats["dcir_fp"]
    m_rpt,  s_rpt  = ds.stats["rpt_fp"]
    m_th,   s_th   = ds.stats["theta_vec"]
    m_pr,   s_pr   = ds.stats["protocol"]

    # ----- BOL + deg params -----
    bol_yaml = OUT_DIR / f"{cell.tag}_bol_params.yaml"
    deg_yaml = OUT_DIR / f"{cell.tag}_deg_params.yaml"
    bol = yaml.safe_load(bol_yaml.read_text())
    deg = yaml.safe_load(deg_yaml.read_text())

    # ----- theta_vec (raw, unstandardised) -----
    # First 5: log10(sweep params)
    # Training-time keys: k_SEI, V_SEI, k_plating, LAM_pos, LAM_neg
    # DE-fit gives:      k_SEI, V_SEI, k_plating, D_SEI_solvent  (NO LAM rates)
    # For the LAM rates we use the training-corpus MEAN as a neutral prior —
    # DE didn't identify them for this cell, so the fair "θ tell me what to do"
    # is to feed the median parameterisation.
    k_SEI, V_SEI, k_plate, D_SEI_solv = _theta_deg_from_deg_yaml(deg)

    def logify(x, floor=1e-20):
        return float(np.log10(max(abs(float(x)), floor)))

    # We only have 5 training dims for deg params; substitute mean-of-training
    # for LAM_pos_rate_s and LAM_neg_rate_s (unidentified).
    theta_deg = np.array([
        logify(k_SEI),                    # k_SEI_ms
        logify(V_SEI),                    # SEI_partial_molar_volume
        logify(k_plate),                  # lithium_plating_exchange_current
        m_th[3],                          # LAM_positive_rate_s (standardised mean)
        m_th[4],                          # LAM_negative_rate_s (standardised mean)
    ], dtype=np.float32)

    # BOL identifiers dims 5-9
    stoich  = bol.get("stoichiometry", {})
    cap     = bol.get("capacity",      {})
    res     = bol.get("resistance",    {})
    x_100   = float(stoich.get("x_100", 0.88))
    y_100   = float(stoich.get("y_100", 0.01))
    Q_n     = float(cap.get("Q_n_init_Ah", 138.0))
    R0      = float(res.get("R0_Ohm",  1.7e-3))
    C1      = float(res.get("C1_F",    2.4e4))
    theta_bol = np.array([
        x_100, y_100, Q_n, logify(R0), logify(C1),
    ], dtype=np.float32)

    theta_vec_raw = np.concatenate([theta_deg, theta_bol])

    # ----- rpt_fp: feed the standardiser mean (equivalent to standardised=0) -----
    rpt_fp_raw = m_rpt.copy()

    # ----- dcir_fp: 0 in raw space (matches training) -----
    dcir_fp_raw = np.zeros(9, dtype=np.float32)

    # ----- protocol: match training defaults exactly -----
    protocol_raw = np.array([0.5, 1.0, 298.15, 10.0], dtype=np.float32)

    # ----- soh_early: first K=50 measured SoH normalised to cy1=1.0 -----
    K = ds.cfg.K
    meas_cycles, meas_soh_norm = load_measured_fn(cell)
    if len(meas_soh_norm) < K:
        raise RuntimeError(
            f"{cell.tag}: only {len(meas_soh_norm)} measured cycles, need K={K}")
    soh_early = meas_soh_norm[:K].astype(np.float32)

    # ----- Standardise -----
    dcir_fp_std   = (dcir_fp_raw - m_dcir) / s_dcir
    rpt_fp_std    = (rpt_fp_raw  - m_rpt)  / s_rpt        # = zeros
    theta_vec_std = (theta_vec_raw - m_th) / s_th
    protocol_std  = (protocol_raw - m_pr)  / s_pr

    # Report what the fingerprint looks like
    print(f"[{cell.tag}] fingerprint (standardised):")
    print(f"  theta_vec (log10 deg + BOL): {theta_vec_std}")
    print(f"  protocol                    : {protocol_std}")
    print(f"  soh_early[:5]               : {soh_early[:5]}")
    print(f"  soh_early[-5:]              : {soh_early[-5:]}")

    return dict(
        dcir_fp    = torch.from_numpy(dcir_fp_std.astype(np.float32)).unsqueeze(0),
        rpt_fp     = torch.from_numpy(rpt_fp_std.astype(np.float32)).unsqueeze(0),
        theta_vec  = torch.from_numpy(theta_vec_std.astype(np.float32)).unsqueeze(0),
        protocol   = torch.from_numpy(protocol_std.astype(np.float32)).unsqueeze(0),
        soh_early  = torch.from_numpy(soh_early.astype(np.float32)).unsqueeze(0),
        meas_cycles = meas_cycles,
        meas_soh_norm = meas_soh_norm,
    )


def load_measured_soh(cell: CellCase) -> tuple[np.ndarray, np.ndarray]:
    """Load the canonical SoH trace for one cell and normalise so cy1 = 1.0."""
    df = pd.read_parquet(cell.canonical_parquet)
    s = df[df.cell_id == cell.cell_id].sort_values("global_cycle").reset_index(drop=True)
    cycles = s["global_cycle"].to_numpy(int)
    soh = s["soh"].to_numpy(float)
    soh_norm = soh / soh[0]
    return cycles, soh_norm


def load_operator(device: str = "cpu") -> tuple[ThetaDeepONet, OperatorConfig]:
    ck = torch.load(CKPT, map_location=device, weights_only=False)
    cfg_dict = ck["cfg"]
    cfg = OperatorConfig(**cfg_dict)
    model = ThetaDeepONet(cfg).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    print(f"Loaded {CKPT.name} (val_loss={ck['val_loss']:.5f}, ep={ck['ep']})")
    return model, cfg


def run_operator(model, fp: dict, n_query: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        soh_init = torch.tensor([1.0], dtype=torch.float32)   # (B,)
        n_q_t = torch.from_numpy(n_query.astype(np.float32)).unsqueeze(0)  # (B,N)
        soh_hat = model(
            fp["dcir_fp"], fp["rpt_fp"], fp["soh_early"],
            fp["theta_vec"], fp["protocol"], n_q_t, soh_init,
        )   # (B, N)
    return soh_hat.squeeze(0).cpu().numpy()


def cycle_at_soh(cycles: np.ndarray, soh: np.ndarray, target: float) -> float | None:
    below = np.where(soh <= target)[0]
    if len(below) == 0:
        return None
    idx = below[0]
    if idx == 0:
        return float(cycles[0])
    prev, curr = idx - 1, idx
    if soh[curr] == soh[prev]:
        return float(cycles[curr])
    frac = (soh[prev] - target) / (soh[prev] - soh[curr])
    return float(cycles[prev] + frac * (cycles[curr] - cycles[prev]))


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def process_cell(cell: CellCase, model, cfg, ds) -> dict:
    print(f"\n=== {cell.tag} ===")
    # --- Fingerprint ---
    fp = build_fingerprint(cell, ds, load_measured_soh)
    meas_cycles = fp["meas_cycles"]
    meas_soh    = fp["meas_soh_norm"]

    # --- DFN long-run ---
    dfn = pd.read_parquet(OUT_DIR / f"{cell.tag}_longrun.parquet")
    dfn_cycles = dfn["cycle_n"].to_numpy(int)
    dfn_soh    = dfn["SOH"].to_numpy(float)

    # --- Operator rollout ---
    n_query = np.arange(1, 5001, dtype=np.float32)
    op_soh = run_operator(model, fp, n_query)

    # --- Metrics ---
    # Operator vs DFN over the FULL overlap
    common_cy = np.intersect1d(dfn_cycles, n_query.astype(int))
    dfn_at = np.interp(common_cy, dfn_cycles, dfn_soh)
    op_at  = np.interp(common_cy, n_query, op_soh)
    rmse_op_dfn = rmse(op_at, dfn_at)

    # vs measured (only up to end of measured trace)
    op_at_meas  = np.interp(meas_cycles, n_query, op_soh)
    dfn_at_meas = np.interp(meas_cycles, dfn_cycles, dfn_soh)
    rmse_op_meas  = rmse(op_at_meas,  meas_soh)
    rmse_dfn_meas = rmse(dfn_at_meas, meas_soh)

    # Cycle-at-SoH 0.80
    cy80_op  = cycle_at_soh(n_query.astype(int), op_soh, 0.80)
    cy80_dfn = cycle_at_soh(dfn_cycles,          dfn_soh, 0.80)

    # --- Overlay plot ---
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.plot(meas_cycles, meas_soh, "g.-", ms=3, lw=1.0,
            label=f"Measured {cell.manufacturer} {cell.cell_id} (n={len(meas_cycles)})")
    ax.plot(dfn_cycles, dfn_soh, "b-", lw=1.6,
            label="PyBaMM DFN 5000-cy (per-cell θ)")
    ax.plot(n_query, op_soh, "r-", lw=1.4, alpha=0.9,
            label="θ-DeepONet operator (from fingerprint)")
    ax.axhline(0.80, color="tab:orange", ls="--", lw=0.7, label="EoL 0.80")
    ax.set_xlim(0, 5000)
    ax.set_ylim(0.5, 1.05)
    ax.set_xlabel("Cycle")
    ax.set_ylabel("SoH (normalised to cy1)")
    ax.set_title(
        f"{cell.manufacturer} {cell.cell_id}: measured vs DFN vs operator\n"
        f"RMSE op-DFN={rmse_op_dfn*100:.2f}pp  "
        f"op-meas={rmse_op_meas*100:.2f}pp  "
        f"DFN-meas={rmse_dfn_meas*100:.2f}pp"
    )
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    png = OUT_DIR / f"{cell.tag}_operator_vs_dfn.png"
    fig.savefig(png, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote plot: {png}")

    return dict(
        tag=cell.tag,
        n_measured=int(len(meas_cycles)),
        rmse_op_dfn_pp=float(rmse_op_dfn * 100),
        rmse_op_meas_pp=float(rmse_op_meas * 100),
        rmse_dfn_meas_pp=float(rmse_dfn_meas * 100),
        cy_at_soh_0p80_operator=cy80_op,
        cy_at_soh_0p80_dfn=cy80_dfn,
        dfn_final_soh=float(dfn_soh[-1]),
        op_final_soh=float(op_soh[-1]),
        n_dfn_cycles=int(dfn_cycles[-1]),
        meas_cycles=meas_cycles,
        meas_soh=meas_soh,
        dfn_cycles=dfn_cycles,
        dfn_soh=dfn_soh,
        op_cycles=n_query,
        op_soh=op_soh,
    )


def combined_overlay(results: list[dict]) -> None:
    """Two-panel side-by-side plot (EVE left, CALB right)."""
    fig, axes = plt.subplots(1, len(results), figsize=(13, 5.5), sharey=True)
    if len(results) == 1:
        axes = [axes]
    for ax, r in zip(axes, results):
        tag = r["tag"]
        mfr = "EVE" if tag.startswith("eve") else "CALB"
        cid = tag.split("_")[-1]
        ax.plot(r["meas_cycles"], r["meas_soh"], "g.-", ms=3, lw=1.0,
                label=f"Measured (n={r['n_measured']})")
        ax.plot(r["dfn_cycles"], r["dfn_soh"], "b-", lw=1.6, label="DFN 5000-cy")
        ax.plot(r["op_cycles"], r["op_soh"], "r-", lw=1.4, alpha=0.9,
                label="Operator")
        ax.axhline(0.80, color="tab:orange", ls="--", lw=0.7)
        ax.set_xlim(0, 5000)
        ax.set_ylim(0.5, 1.05)
        ax.set_xlabel("Cycle")
        ax.set_title(
            f"{mfr} {cid}\n"
            f"op-DFN {r['rmse_op_dfn_pp']:.1f}pp  "
            f"op-meas {r['rmse_op_meas_pp']:.1f}pp"
        )
        ax.grid(alpha=0.3)
        ax.legend(loc="lower left", fontsize=9)
    axes[0].set_ylabel("SoH (normalised)")
    fig.suptitle(
        "θ-conditioned DeepONet: EVE-trained corpus vs CALB generalisation",
        fontsize=12,
    )
    fig.tight_layout()
    out = OUT_DIR / "eve_vs_calb_generalisation.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote combined plot: {out}")


def main(cell_tags: list[str]) -> list[dict]:
    model, cfg = load_operator("cpu")
    ds = build_dataset(DatasetConfig(K=cfg.early_K, n_query=30))
    print(f"Built training dataset ({len(ds)} sims) for standardisation stats")

    cells = []
    for tag in cell_tags:
        if tag.startswith("eve"):
            cid = tag.split("_")[-1]
            cells.append(CellCase(
                tag=tag, manufacturer="EVE", cell_id=cid,
                canonical_parquet=Path(
                    "/home/hj/Desktop/PINNs/soh/data/canonical/eve.parquet"),
            ))
        elif tag.startswith("calb"):
            cid = tag.split("_")[-1]
            cells.append(CellCase(
                tag=tag, manufacturer="CALB", cell_id=cid,
                canonical_parquet=Path(
                    "/home/hj/Desktop/PINNs/soh/data/canonical/calb_new.parquet"),
            ))
        else:
            raise ValueError(f"Unknown cell tag: {tag}")

    results = []
    for cell in cells:
        try:
            r = process_cell(cell, model, cfg, ds)
        except FileNotFoundError as e:
            print(f"SKIP {cell.tag}: {e}")
            continue
        results.append(r)
        print(f"[{r['tag']}] RMSE op-DFN={r['rmse_op_dfn_pp']:.2f} pp | "
              f"op-meas={r['rmse_op_meas_pp']:.2f} pp | "
              f"DFN-meas={r['rmse_dfn_meas_pp']:.2f} pp | "
              f"cy@0.80 op={r['cy_at_soh_0p80_operator']} DFN={r['cy_at_soh_0p80_dfn']}")

    if len(results) >= 1:
        combined_overlay(results)
    return results


if __name__ == "__main__":
    tags = sys.argv[1:] if len(sys.argv) > 1 else ["eve_0008", "calb_0020"]
    main(tags)
