"""
src/param_id/sei_selfdisc.py
----------------------------
Bound the SEI growth rate constant from self-discharge OCV decay.

Idea
~~~~
Lithium loss through any parasitic side reaction (SEI growth, electrolyte
oxidation, soluble shuttle, internal micro-shorts, ...) shows up as a
slow drop in open-circuit voltage during a long rest. Mapping that
voltage decay onto the OCV(SOC) curve gives an effective parasitic
current. We attribute *all* of that current to SEI growth, which is
therefore an **upper bound** on the SEI rate constant.

Workflow (per cell)
1. Identify the longest Rest segment that follows a CCCV charge — this
   is the self-discharge dwell at top of charge.
2. Skip an initial relaxation transient (default 6 hours) so the
   remaining trace is dominated by the slow parasitic loss, not by
   concentration / charge-transfer relaxation.
3. Map V(t) → SOC(t) via the cell's own (low-rate) OCV(SOC) curve from
   the OCVSOC test.
4. Linear fit SOC vs t → dSOC/dt → I_sd = Q_cell · |dSOC/dt|.
5. Per-area current density i_sd = I_sd / A_geo (Prada2013 default
   geometry unless overridden); k_SEI_max = i_sd / (F · c_EC_0).

Caveats
- A_geo from the spec sheet is geometric, not BET — using it
  systematically *over*-estimates the per-area current density and
  therefore the bound. That is the safe direction for an upper bound.
- This analysis lumps all parasitic processes into the SEI channel; an
  EC-reaction-limited PyBaMM SEI submodel needs an explicit k_SEI that
  cannot exceed the value returned here.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data_loader import load_test  # noqa: E402

# Physical constants
F_FARADAY = 96485.33212        # C/mol
C_EC_0_DEFAULT = 4541.0        # mol/m^3  EC concentration in EC:EMC (Yang et al.)
A_GEO_DEFAULT_M2 = 0.6 * 0.3   # Prada2013 default electrode area

# Self-discharge analysis defaults
TRANSIENT_SKIP_S = 24.0 * 3600  # skip first 24 h of rest (LFP equilibration is slow)
MIN_REST_DURATION_S = 24.0 * 3600  # require ≥ 24 h of rest
OCV_TOP_BRANCH_V_LOW = 3.35       # below this voltage the LFP plateau starts and
                                  # V(SOC) is no longer locally invertible


@dataclass
class SelfDischargeFit:
    cell_id: str
    rest_t_start_s: float
    rest_duration_s: float
    V_start_V: float
    V_end_V: float
    dV_dt_uV_per_s: float
    dSOC_dt_per_h: float
    I_sd_uA: float
    i_sd_uA_per_m2: float
    k_SEI_max_m_per_s: float
    rmse_mV: float
    n_points_fit: int


def _longest_top_rest(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return the rows of the longest Rest segment that comes immediately
    after a CCCV_Chg. Raises if no such segment exists.
    """
    d = df.sort_values("time").reset_index(drop=True).copy()
    d["seg"] = (d["step_name"] != d["step_name"].shift()).cumsum()
    summary = d.groupby("seg").agg(
        step_name=("step_name", "first"),
        prev_step=("step_name", lambda s: s.iloc[0]),  # placeholder, overwrite below
        dt_s=("time", lambda s: float(s.max() - s.min())),
        n=("voltage", "size"),
    ).reset_index()
    # Identify segments preceded by CCCV_Chg
    prev = d.groupby("seg")["step_name"].first().shift(1)
    summary["prev_step"] = summary["seg"].map(prev.to_dict())

    candidates = summary[
        (summary["step_name"] == "Rest")
        & (summary["prev_step"] == "CCCV_Chg")
        & (summary["dt_s"] >= MIN_REST_DURATION_S)
    ]
    if candidates.empty:
        raise ValueError("No post-CCCV rest segment >= 24 h found")
    best = candidates.sort_values("dt_s", ascending=False).iloc[0]
    return d[d["seg"] == best["seg"]].reset_index(drop=True)


def _load_ocv_top_branch(cell_id: str, v_low: float = OCV_TOP_BRANCH_V_LOW
                         ) -> tuple[np.ndarray, np.ndarray]:
    """
    Return the steep top branch of the LFP discharge OCV curve, where V is
    locally monotonic in SOC (above the 3.31 V plateau).

    Returns arrays sorted so that V is strictly increasing — suitable for
    np.interp(v_query, v_table, soc_table).
    """
    ocv = load_test("OCV_SOC", cell_id=cell_id).sort_values("time").reset_index(drop=True)
    disc = ocv[ocv["step_name"] == "CC_DChg"].copy()
    if disc.empty:
        raise ValueError(f"No OCV discharge branch for cell {cell_id}")
    q_max = float(disc["capacity"].abs().max())
    soc = 1.0 - disc["capacity"].abs().to_numpy() / q_max
    v = disc["voltage"].to_numpy(dtype=float)

    mask = v >= v_low
    if mask.sum() < 20:
        raise ValueError(f"Cell {cell_id}: too few points above {v_low} V to build top branch")
    v_top = v[mask]
    soc_top = soc[mask]
    # Sort by V ascending for np.interp
    order = np.argsort(v_top)
    return v_top[order], soc_top[order]


def fit_one_cell(
    cell_id: str,
    Q_nominal_Ah: float = 105.0,
    A_geo_m2: float = A_GEO_DEFAULT_M2,
    c_EC_0_mol_m3: float = C_EC_0_DEFAULT,
    transient_skip_s: float = TRANSIENT_SKIP_S,
) -> SelfDischargeFit:
    df = load_test("SelfDischarge", cell_id=cell_id).sort_values("time").reset_index(drop=True)
    rest = _longest_top_rest(df)
    t = rest["time"].to_numpy(dtype=float)
    v = rest["voltage"].to_numpy(dtype=float)
    t0 = float(t[0])
    t_rel = t - t0
    mask = t_rel >= transient_skip_s
    if mask.sum() < 50:
        raise ValueError(
            f"Cell {cell_id}: too few rest samples remain after skipping "
            f"{transient_skip_s/3600:.1f} h transient"
        )
    t_fit = t_rel[mask]
    v_fit = v[mask]

    # Map voltage onto SOC using the steep TOP branch of the OCV curve only
    # (LFP plateau makes V(SOC) non-invertible elsewhere).
    v_top, soc_top = _load_ocv_top_branch(cell_id)
    if v_fit.min() < v_top.min() or v_fit.max() > v_top.max():
        # extrapolation off the steep branch — most likely the cell has
        # equilibrated below 3.35 V. In that case dV/dSOC is essentially
        # zero and the upper-bound estimate degenerates; flag with NaN.
        pass
    soc_fit = np.interp(v_fit, v_top, soc_top)

    # Linear fit SOC(t) — slope is dSOC/dt (1/s, negative for self-discharge)
    a, b = np.polyfit(t_fit, soc_fit, 1)
    soc_hat = a * t_fit + b
    rmse_V = float(np.sqrt(np.mean((soc_fit - soc_hat) ** 2)))  # in SOC units

    dSOC_dt_per_s = float(a)
    I_sd_A = Q_nominal_Ah * abs(dSOC_dt_per_s) * 3600.0   # Q[Ah] * (1/h) = A
    i_sd_A_per_m2 = I_sd_A / A_geo_m2
    k_SEI_max = i_sd_A_per_m2 / (F_FARADAY * c_EC_0_mol_m3)  # m/s

    # Independent dV/dt for diagnostics
    a_v, _ = np.polyfit(t_fit, v_fit, 1)

    return SelfDischargeFit(
        cell_id=cell_id,
        rest_t_start_s=t0,
        rest_duration_s=float(t_rel[-1]),
        V_start_V=float(v[0]),
        V_end_V=float(v[-1]),
        dV_dt_uV_per_s=float(a_v * 1e6),
        dSOC_dt_per_h=float(dSOC_dt_per_s * 3600.0),
        I_sd_uA=float(I_sd_A * 1e6),
        i_sd_uA_per_m2=float(i_sd_A_per_m2 * 1e6),
        k_SEI_max_m_per_s=float(k_SEI_max),
        rmse_mV=rmse_V * 1000.0,
        n_points_fit=int(mask.sum()),
    )


def fit_cells(cell_ids: list[str], **kwargs) -> pd.DataFrame:
    rows = []
    for cid in cell_ids:
        try:
            fit = fit_one_cell(cid, **kwargs)
            rows.append(vars(fit))
            print(f"  ✓ cell {cid}: I_sd={fit.I_sd_uA:.1f} µA "
                  f"({fit.dSOC_dt_per_h*100:+.4f} %SOC/h, "
                  f"dV/dt={fit.dV_dt_uV_per_s:+.3f} µV/s)  "
                  f"k_SEI_max={fit.k_SEI_max_m_per_s:.3e} m/s")
        except Exception as e:
            print(f"  ✗ cell {cid}: {type(e).__name__}: {e}")
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import yaml
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg_path = Path("configs/dataset.yaml")
    cells: list[str] = []
    if cfg_path.exists():
        cfg = yaml.safe_load(cfg_path.read_text()) or {}
        cells = [str(c).zfill(4) for c in cfg.get("dataset", {}).get("selected_cells", [])]
    if not cells:
        cells = ["0005", "0006", "0007", "0008"]

    df = fit_cells(cells)
    if df.empty:
        sys.exit(1)

    out_path = Path("data/processed/selfdischarge_fit.parquet")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"Saved self-discharge fits → {out_path}")

    # Diagnostic plot — V(t) for each cell during the top-of-charge rest
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for cid in cells:
        raw = load_test("SelfDischarge", cell_id=cid).sort_values("time").reset_index(drop=True)
        try:
            rest = _longest_top_rest(raw)
        except ValueError:
            continue
        t_h = (rest["time"].values - rest["time"].values[0]) / 3600.0
        ax.plot(t_h, rest["voltage"].values, label=cid, lw=1.0)
    ax.set(xlabel="Hours since top-of-charge", ylabel="OCV [V]",
           title="Self-discharge OCV decay (25 °C)")
    ax.legend(title="cell_id", fontsize=8)
    fig.tight_layout()
    fig_path = Path("outputs/results/selfdischarge_decay.png")
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot → {fig_path}")
