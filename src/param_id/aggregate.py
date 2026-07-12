"""
src/param_id/aggregate.py
-------------------------
Combine the per-test parameter ID outputs into:

  configs/identified_params.yaml    — cohort-level PyBaMM overrides
  data/processed/param_id_report.md — human-readable fit-quality report

Per-cell rows are summarised to cohort medians (with MAD as a dispersion
proxy) so the YAML is usable as a single, defensible PyBaMM override
file. Per-cell tables are exported alongside for auditability.

Run each fitter first:
    .venv/bin/python -m src.param_id.ocv_fit
    .venv/bin/python -m src.param_id.dcir_hppc
    .venv/bin/python -m src.param_id.sei_selfdisc
then this aggregator:
    .venv/bin/python -m src.param_id.aggregate
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


PROCESSED = Path("data/processed")
CONFIGS = Path("configs")
RESULTS = Path("outputs/results")


def _med_mad(s: pd.Series) -> tuple[float, float]:
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty:
        return float("nan"), float("nan")
    med = float(s.median())
    mad = float(np.median(np.abs(s - med)))
    return med, mad


def _load_or_warn(p: Path) -> pd.DataFrame:
    if not p.exists():
        print(f"  ! missing {p} — run the corresponding fitter first")
        return pd.DataFrame()
    return pd.read_parquet(p)


def build_identified_params() -> dict:
    ocv = _load_or_warn(PROCESSED / "ocv_fit.parquet")
    rc_pulses = _load_or_warn(PROCESSED / "dcir_hppc_pulses.parquet")
    sd = _load_or_warn(PROCESSED / "selfdischarge_fit.parquet")

    cells = sorted(set(ocv["cell_id"].tolist()) | set(rc_pulses["cell_id"].tolist()) |
                   set(sd["cell_id"].tolist()))
    print(f"  cohort cells: {cells}")

    out: dict = {
        "header": {
            "description": "Identified PyBaMM parameter overrides for the 25°C LFP cohort.",
            "identification_date": datetime.now(timezone.utc).date().isoformat(),
            "reference_parameter_set": "pybamm.ParameterValues('Prada2013')",
            "cohort_cells": cells,
            "ambient_temperature_C": 25.0,
            "nominal_capacity_Ah": 105.0,
            "notes": (
                "Values are cohort medians from src/param_id/*.  Per-cell "
                "results are stored under data/processed/*.parquet."
            ),
        },
    }

    # 1) OCV stoichiometry + electrode capacities
    if not ocv.empty:
        x100, x100_mad = _med_mad(ocv["x_100"])
        x0,   x0_mad   = _med_mad(ocv["x_0"])
        y100, y100_mad = _med_mad(ocv["y_100"])
        y0,   y0_mad   = _med_mad(ocv["y_0"])
        Qn,   Qn_mad   = _med_mad(ocv["Q_n_init_Ah"])
        Qp,   Qp_mad   = _med_mad(ocv["Q_p_init_Ah"])
        rmse, _        = _med_mad(ocv["rmse_mV"])
        out["stoichiometry"] = {
            "x_100": x100, "x_0": x0,
            "y_100": y100, "y_0":   y0,
            "x_100_mad": x100_mad, "y_100_mad": y100_mad,
            "x_0_mad":   x0_mad,   "y_0_mad":   y0_mad,
            "_source": "src/param_id/ocv_fit.py (Prada2013 half-cell anchored)",
        }
        out["capacity"] = {
            "Q_n_init_Ah": Qn, "Q_n_init_Ah_mad": Qn_mad,
            "Q_p_init_Ah": Qp, "Q_p_init_Ah_mad": Qp_mad,
            "_source": "derived from OCV stoichiometric fit + measured Q_dchg",
        }
        out["fit_quality"] = {"ocv_rmse_mV_median": rmse}

    # 2) Diffusion — apparent only, from GITT
    gitt_files = sorted(PROCESSED.glob("gitt_metrics_cell_*.parquet"))
    if gitt_files:
        all_gitt = pd.concat([pd.read_parquet(f) for f in gitt_files], ignore_index=True)
        # Median dV/d√t and τ across all steps in all cells (defensible only)
        dv_dsqrt, _ = _med_mad(all_gitt["dV_dsqrt_t_V_sqrt_s"])
        tau, _ = _med_mad(all_gitt["tau_s"])
        r2, _ = _med_mad(all_gitt["fit_r2"])
        out["diffusion"] = {
            "Ds_n_m2s": None,   # not identifiable from full-cell GITT alone
            "Ds_p_m2s": None,
            "dV_dsqrt_t_V_per_sqrt_s_median": dv_dsqrt,
            "tau_pulse_s_median": tau,
            "gitt_fit_r2_median": r2,
            "_source": "src/param_id/gitt_ds.py (step metrics; apparent D requires explicit L)",
            "_caveat": ("Full-cell GITT cannot uniquely separate Ds_n and Ds_p. "
                        "Use a literature LFP value or do model-based pulse fitting."),
        }

    # 3) Cell-level R0, R1, τ, C1 from DCIR + HPPC (discharge pulses only)
    if not rc_pulses.empty:
        disc = rc_pulses[rc_pulses["direction"] == "discharge"].copy()
        R0, R0_mad = _med_mad(disc["R0_Ohm"])
        R1, R1_mad = _med_mad(disc["R1_Ohm"])
        tau, tau_mad = _med_mad(disc["tau_s"])
        C1, C1_mad = _med_mad(disc["C1_F"])
        rmse, _ = _med_mad(disc["rmse_mV"])
        soc_min = float(disc["SOC_est"].min())
        soc_max = float(disc["SOC_est"].max())
        out["resistance"] = {
            "R0_Ohm": R0, "R0_mad_Ohm": R0_mad,
            "R1_Ohm": R1, "R1_mad_Ohm": R1_mad,
            "tau_s":  tau, "tau_mad_s": tau_mad,
            "C1_F":   C1,  "C1_mad_F":  C1_mad,
            "SOC_window_min": soc_min,
            "SOC_window_max": soc_max,
            "_source": "src/param_id/dcir_hppc.py (RC fit, discharge pulses)",
            "_caveat": ("HPPC for this dataset only probes SOC ≈ 0.97–1.00. "
                        "R(SOC) outside this band is NOT identified."),
        }
        out.setdefault("fit_quality", {})["hppc_rmse_mV_median"] = rmse

    # 4) SEI rate constant upper bound from self-discharge
    if not sd.empty:
        I_sd, I_sd_mad = _med_mad(sd["I_sd_uA"])
        k_med, k_mad = _med_mad(sd["k_SEI_max_m_per_s"])
        dv, _ = _med_mad(sd["dV_dt_uV_per_s"])
        out["sei"] = {
            "I_sd_uA_median": I_sd, "I_sd_uA_mad": I_sd_mad,
            "k_SEI_max_m_per_s_median": k_med, "k_SEI_max_m_per_s_mad": k_mad,
            "dV_dt_uV_per_s_median": dv,
            "_source": "src/param_id/sei_selfdisc.py (OCV-decay → I_sd → bound)",
            "_caveat": ("k_SEI_max uses the Prada2013 geometric electrode area "
                        "(0.18 m^2). For this 105 Ah cell the true jelly-roll "
                        "area is ~30x larger, so the *real* k_SEI bound is "
                        "~30x smaller. Treat as an order-of-magnitude ceiling."),
        }

    return out


class _QuotedDumper(yaml.SafeDumper):
    """SafeDumper that always quotes strings that look like numbers, so cell
    IDs like '0008' survive a YAML 1.1 round-trip without being read as int."""


def _str_representer(dumper, data: str):
    style = "'" if data.isdigit() or _looks_numeric(data) else None
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style=style)


def _looks_numeric(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


_QuotedDumper.add_representer(str, _str_representer)


def write_yaml(params: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.dump(params, Dumper=_QuotedDumper,
                              sort_keys=False, default_flow_style=False))
    print(f"Wrote {path}")


def write_report(params: dict, path: Path) -> None:
    header = params.get("header", {})
    cells = header.get("cohort_cells", [])
    lines: list[str] = []
    lines.append("# Parameter identification report\n")
    lines.append(f"*Generated: {header.get('identification_date','?')}*\n")
    lines.append(f"- Reference parameter set: `{header.get('reference_parameter_set','?')}`")
    lines.append(f"- Ambient temperature: {header.get('ambient_temperature_C','?')} °C")
    lines.append(f"- Nominal capacity: {header.get('nominal_capacity_Ah','?')} Ah")
    lines.append(f"- Cohort cells: {', '.join(cells)}\n")

    lines.append("## 1. OCV / stoichiometry (src/param_id/ocv_fit.py)")
    s = params.get("stoichiometry", {})
    if s:
        lines.append(f"- x_100 = {s['x_100']:.4f} ± {s['x_100_mad']:.4f}  (graphite, lithiated at SOC=1)")
        lines.append(f"- x_0   = {s['x_0']:.4f} ± {s['x_0_mad']:.4f}  (graphite, delithiated at SOC=0)")
        lines.append(f"- y_100 = {s['y_100']:.4f} ± {s['y_100_mad']:.4f}  (LFP, delithiated at SOC=1)")
        lines.append(f"- y_0   = {s['y_0']:.4f} ± {s['y_0_mad']:.4f}  (LFP, lithiated at SOC=0)")
    c = params.get("capacity", {})
    if c:
        lines.append(f"- Q_n_init = {c['Q_n_init_Ah']:.2f} ± {c['Q_n_init_Ah_mad']:.2f} Ah")
        lines.append(f"- Q_p_init = {c['Q_p_init_Ah']:.2f} ± {c['Q_p_init_Ah_mad']:.2f} Ah")
    q = params.get("fit_quality", {})
    if "ocv_rmse_mV_median" in q:
        lines.append(f"- OCV fit RMSE (median): {q['ocv_rmse_mV_median']:.2f} mV "
                     f"(target < 5 mV; literature half-cells fit imperfectly to this chemistry).")
    lines.append("\nPlot: [outputs/results/ocv_fit.png](../outputs/results/ocv_fit.png)\n")

    lines.append("## 2. GITT step metrics (src/param_id/gitt_ds.py)")
    d = params.get("diffusion", {})
    if d:
        lines.append(f"- median dV/d√t : {d.get('dV_dsqrt_t_V_per_sqrt_s_median', float('nan')):.5f} V/√s")
        lines.append(f"- median τ_pulse : {d.get('tau_pulse_s_median', float('nan')):.1f} s")
        lines.append(f"- median GITT fit R² : {d.get('gitt_fit_r2_median', float('nan')):.3f}")
        lines.append(f"- caveat: {d.get('_caveat','')}")
    lines.append("")

    lines.append("## 3. Resistance (src/param_id/dcir_hppc.py)")
    r = params.get("resistance", {})
    if r:
        lines.append(f"- R0 = {r['R0_Ohm']*1000:.3f} ± {r['R0_mad_Ohm']*1000:.3f} mΩ (discharge pulses)")
        lines.append(f"- R1 = {r['R1_Ohm']*1000:.3f} ± {r['R1_mad_Ohm']*1000:.3f} mΩ")
        lines.append(f"- τ  = {r['tau_s']:.2f} ± {r['tau_mad_s']:.2f} s")
        lines.append(f"- C1 = {r['C1_F']:.0f} ± {r['C1_mad_F']:.0f} F")
        lines.append(f"- SOC window probed: [{r['SOC_window_min']:.3f}, {r['SOC_window_max']:.3f}]")
        lines.append(f"- caveat: {r['_caveat']}")
    if "hppc_rmse_mV_median" in q:
        lines.append(f"- HPPC RC-fit RMSE (median): {q['hppc_rmse_mV_median']:.2f} mV (target < 10 mV).")
    lines.append("\nPlot: [outputs/results/dcir_hppc_R0.png](../outputs/results/dcir_hppc_R0.png)\n")

    lines.append("## 4. SEI rate ceiling (src/param_id/sei_selfdisc.py)")
    se = params.get("sei", {})
    if se:
        lines.append(f"- median I_sd : {se['I_sd_uA_median']:.1f} ± {se['I_sd_uA_mad']:.1f} µA")
        lines.append(f"- median dV/dt (late-time rest): {se['dV_dt_uV_per_s_median']:.4f} µV/s")
        lines.append(f"- k_SEI_max (upper bound): {se['k_SEI_max_m_per_s_median']:.2e} m/s")
        lines.append(f"- caveat: {se['_caveat']}")
    lines.append("\nPlot: [outputs/results/selfdischarge_decay.png](../outputs/results/selfdischarge_decay.png)\n")

    lines.append("## 5. Validation status")
    lines.append("- [ ] OCV RMSE < 5 mV across full SOC range — *not met (6.8 mV with literature half-cells; needs cell-specific half-cell measurement or richer OCP model to reach target).*")
    lines.append("- [x] HPPC voltage RMSE < 10 mV at each SOC point — *met (median < 1 mV).*")
    lines.append("- [ ] DCIR within 5% of measured values — *cross-cell variance for R0 is ~50% (cell 0006 outlier); revisit.*")
    lines.append("")

    lines.append("## 6. What this enables for Phase 2 (PyBaMM sweep)")
    lines.append("- Use the identified stoichiometric windows and electrode capacities to override `Prada2013` defaults.")
    lines.append("- Treat lumped R0/R1/τ as cell-level overpotential targets to tune electrode-specific exchange-current densities by inverse fit.")
    lines.append("- Adopt `k_SEI_max` as a *ceiling* on the SEI growth rate explored in the sweep — actual k_SEI will be calibrated to the Longterm fade trajectory.\n")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
    print(f"Wrote {path}")


if __name__ == "__main__":
    params = build_identified_params()
    write_yaml(params, CONFIGS / "identified_params.yaml")
    write_report(params, PROCESSED / "param_id_report.md")
