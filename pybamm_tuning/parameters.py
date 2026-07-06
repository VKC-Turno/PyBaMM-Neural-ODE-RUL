"""
Build a PyBaMM ParameterValues object from a Characterization.

The key step is mapping measured quantities to PyBaMM parameter keys:

| Char data           | PyBaMM key                                                  |
|---------------------|-------------------------------------------------------------|
| Q_RPT_ah            | "Nominal cell capacity [A.h]" + concentration scaling       |
| R₀ (DCIR or HPPC)   | "Contact resistance [Ohm]"                                  |
| OCV(SoC) curve      | Stoichiometric limits via ocv_fit.fit_stoichiometry_from_ocv |
| Cycling temperature | "Ambient temperature [K]" + "Initial temperature [K]"       |

Aging parameters (k_SEI, plating, LAM) are left at the base set's defaults
unless the caller passes `aging_overrides` (typically from a sweep median
or from calibration.calibrate_k_sei).
"""
from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pybamm

from .characterization import Characterization
from .ocv_fit import StoichiometryResult, fit_stoichiometry_from_ocv


# PyBaMM key registry — single source of truth for downstream code.
PARAM_KEYS = {
    "capacity":            "Nominal cell capacity [A.h]",
    "electrode_width":     "Electrode width [m]",
    "electrode_height":    "Electrode height [m]",
    "contact_R":           "Contact resistance [Ohm]",
    "T_amb":               "Ambient temperature [K]",
    "T_init":              "Initial temperature [K]",
    "c_n_init":            "Initial concentration in negative electrode [mol.m-3]",
    "c_p_init":            "Initial concentration in positive electrode [mol.m-3]",
    "c_n_max":             "Maximum concentration in negative electrode [mol.m-3]",
    "c_p_max":             "Maximum concentration in positive electrode [mol.m-3]",
    # Aging-related (set via aging_overrides):
    "k_SEI":               "SEI kinetic vs solvent diffusion limited model [m.s-1]",  # if present
    "k_SEI_alt":           "SEI kinetic rate constant [m.s-1]",
    "SEI_mv":              "SEI partial molar volume [m3.mol-1]",
    "LAM_pos":             "Positive electrode LAM constant proportional term [s-1]",
    "LAM_neg":             "Negative electrode LAM constant proportional term [s-1]",
}


def apply_q_rpt_to_capacity(
    overrides: dict, char: Characterization,
    *, scale_geometry: bool = True, base: str = "Prada2013",
) -> dict:
    """
    Set nominal capacity from the measured Q_RPT (per-cell if module).

    PyBaMM's `Nominal cell capacity [A.h]` is just a label — to actually have
    the cell *deliver* the target capacity we have to scale the electrode
    geometry. By default we scale `Electrode width [m]` linearly, which keeps
    current density / temperature / aging kinetics per-area identical but
    multiplies total deliverable Ah by the ratio of target to base capacity.
    Set scale_geometry=False to use the label-only behaviour (compatible with
    the old `src/simulation` infrastructure).
    """
    target_ah = char.per_cell_q_rpt_ah() if char.is_module else char.q_rpt_ah
    overrides[PARAM_KEYS["capacity"]] = float(target_ah)
    if not scale_geometry:
        return overrides
    base_param = pybamm.ParameterValues(base)
    base_ah = float(base_param[PARAM_KEYS["capacity"]])
    base_width = float(base_param[PARAM_KEYS["electrode_width"]])
    if base_ah > 0:
        scale = target_ah / base_ah
        overrides[PARAM_KEYS["electrode_width"]] = base_width * scale
        overrides["_capacity_scale_factor"] = scale
    return overrides


def apply_r0_to_contact_resistance(
    overrides: dict, char: Characterization, *, at_soc: float = 0.5,
    prefer: str = "dcir",
) -> dict:
    """
    Set PyBaMM's contact resistance from measured R₀.

    For a module, uses the per-cell collapsed R₀ (÷ N_series/N_parallel).
    Uses :meth:`Characterization.r0_at_soc`, which applies a per-anchor
    sanity envelope (drops R₀ < 0.1 mΩ or > 5 mΩ).

    If neither DCIR nor HPPC produces a usable anchor, emits a UserWarning
    and leaves the override untouched (the PyBaMM default contact resistance
    will be used). This used to fail silently — the calibration would carry
    on with contact_R = 0 Ω, leading to optimistic R0 fits.
    """
    if char.is_module:
        # Per-cell collapse, then run through r0_at_soc for filtering
        if prefer == "dcir":
            grid, r0 = char.per_cell_dcir_r0_mohm()
        else:
            grid, r0 = char.per_cell_hppc_r0_mohm()
        good = ((r0 >= Characterization.R0_SANITY_MIN_mOhm) &
                (r0 <= Characterization.R0_SANITY_MAX_mOhm) & np.isfinite(r0)) \
                if r0.size else np.array([], dtype=bool)
        r0_at = (float(np.interp(at_soc, grid[good][np.argsort(grid[good])],
                                  r0[good][np.argsort(grid[good])]))
                  if good.any() else None)
    else:
        r0_at = char.r0_at_soc(at_soc, prefer=prefer)

    if r0_at is None:
        warnings.warn(
            f"No usable R₀ anchor found for {char.cell_id} (prefer={prefer!r}): "
            f"DCIR has {char.dcir_r0_mohm.size} anchors, HPPC has "
            f"{char.hppc_r0_mohm.size} anchors, but none survive the sanity "
            f"envelope [{Characterization.R0_SANITY_MIN_mOhm}, "
            f"{Characterization.R0_SANITY_MAX_mOhm}] mΩ. Leaving "
            f"'{PARAM_KEYS['contact_R']}' at PyBaMM default.",
            UserWarning, stacklevel=2,
        )
        return overrides

    overrides[PARAM_KEYS["contact_R"]] = r0_at * 1e-3  # mΩ → Ω
    return overrides


def apply_stoichiometry(
    overrides: dict, fit: StoichiometryResult, base: str = "Prada2013",
) -> dict:
    """Translate fitted stoichiometric limits into initial concentrations."""
    base_param = pybamm.ParameterValues(base)
    c_n_max = float(base_param[PARAM_KEYS["c_n_max"]])
    c_p_max = float(base_param[PARAM_KEYS["c_p_max"]])
    # Initial concentrations correspond to SoC = 1 (= x_100, y_100)
    overrides[PARAM_KEYS["c_n_init"]] = fit.x_100 * c_n_max
    overrides[PARAM_KEYS["c_p_init"]] = fit.y_100 * c_p_max
    return overrides


def apply_pre_aging(overrides: dict, soh_factor: float,
                     base: str = "Prada2013") -> dict:
    """Pre-age the PyBaMM cell by reducing the cyclable Li inventory.

    The default workflow initializes the cell at "fresh" — the negative
    electrode starts at ``x_100 * c_n_max``. For cells that enter the
    longterm experiment already aged (workbook q_rpt < nominal), this
    overstates the cyclable Li at the SIMULATION start. ``soh_factor`` < 1
    scales the negative initial concentration down by the same fraction,
    which lowers the deliverable capacity and shifts the start of the sim's
    SoH trajectory to match the measured cell.

    Notes
    -----
    - This is a first-order pre-aging model: it represents lost cyclable Li
      but doesn't redistribute Li across the SEI volume (which is fine for
      slope-matching calibration).
    - Apply AFTER :func:`apply_stoichiometry`; this multiplies the value
      that step put in.
    - ``soh_factor`` is clamped to [0.5, 1.0] — outside that range either
      the workbook/measured disagreement is too large to trust, or there's
      no pre-aging to apply.
    """
    if soh_factor is None or not (0.5 <= soh_factor <= 1.0):
        return overrides
    c_n_init = overrides.get(PARAM_KEYS["c_n_init"])
    if c_n_init is None:
        base_param = pybamm.ParameterValues(base)
        c_n_init = float(base_param[PARAM_KEYS["c_n_init"]])
    overrides[PARAM_KEYS["c_n_init"]] = float(c_n_init) * float(soh_factor)
    return overrides


def apply_aging_overrides(overrides: dict, aging_overrides: dict) -> dict:
    """
    Merge in aging-specific overrides. Accepts both 'short' keys
    (k_SEI_ms, SEI_partial_molar_volume_m3mol, etc.) and direct PyBaMM keys.
    """
    short_map = {
        "k_SEI_ms":                              PARAM_KEYS["k_SEI_alt"],
        "SEI_partial_molar_volume_m3mol":        PARAM_KEYS["SEI_mv"],
        "LAM_positive_rate_s":                   PARAM_KEYS["LAM_pos"],
        "LAM_negative_rate_s":                   PARAM_KEYS["LAM_neg"],
    }
    for k, v in aging_overrides.items():
        if k in short_map:
            overrides[short_map[k]] = float(v)
        else:
            overrides[k] = v
    return overrides


def build_pybamm_parameters(
    char: Characterization,
    *,
    base: str = "Prada2013",
    degradation_donor: str = "OKane2022",
    aging_overrides: Optional[dict] = None,
    fit_stoichiometry: bool = True,
    contact_resistance_at_soc: float = 0.5,
    temperature_K: Optional[float] = None,
    extra_overrides: Optional[dict] = None,
    pre_age_to_soh: Optional[float] = None,
) -> pybamm.ParameterValues:
    """
    End-to-end: take a Characterization, return a PyBaMM ParameterValues
    ready for simulation.

    Order of operations (later steps override earlier ones):
        1. Start from base (Prada2013) + degradation donor (OKane2022)
        2. Apply Q_RPT → nominal cell capacity
        3. Apply DCIR R₀ → contact resistance
        4. Apply OCV-fit → initial concentrations (optional)
        5. Apply pre-aging factor (scales cyclable Li down, optional)
        6. Apply aging overrides (k_SEI etc., optional)
        7. Set temperature (optional)
        8. Apply extra_overrides (highest priority)

    ``pre_age_to_soh`` (∈ [0.5, 1.0]) scales the negative electrode initial
    concentration to start the simulation at a reduced cyclable-Li
    inventory — use this when the cell entered the longterm test already
    aged (workbook q_rpt < nominal).
    """
    # Reuse the established build_parameter_values from src/simulation
    import sys
    from pathlib import Path
    here = Path(__file__).resolve().parent
    project_root = here.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from src.simulation._pybamm_setup import build_parameter_values

    overrides: dict = {}
    apply_q_rpt_to_capacity(overrides, char)
    apply_r0_to_contact_resistance(overrides, char,
                                    at_soc=contact_resistance_at_soc)

    if fit_stoichiometry and char.ocv_soc_grid.size >= 4:
        if char.is_module:
            soc, v = char.per_cell_ocv()
        else:
            soc, v = char.ocv_soc_grid, char.ocv_v_curve
        fit = fit_stoichiometry_from_ocv(soc, v, base=base)
        apply_stoichiometry(overrides, fit, base=base)
        overrides["_stoichiometry_fit"] = fit  # tagged with leading underscore
        overrides["_stoichiometry_fit_rmse_mV"] = fit.rmse_mV

    if pre_age_to_soh is not None:
        apply_pre_aging(overrides, pre_age_to_soh, base=base)
        overrides["_pre_age_to_soh"] = float(pre_age_to_soh)

    if aging_overrides:
        apply_aging_overrides(overrides, aging_overrides)

    if temperature_K is not None:
        overrides[PARAM_KEYS["T_amb"]] = float(temperature_K)
        overrides[PARAM_KEYS["T_init"]] = float(temperature_K)

    if extra_overrides:
        overrides.update(extra_overrides)

    # Filter out tagged-only diagnostic keys before sending to PyBaMM
    pybamm_safe = {k: v for k, v in overrides.items() if not k.startswith("_")}

    return build_parameter_values(base=base, degradation_donor=degradation_donor,
                                   overrides=pybamm_safe)


def summarise_overrides(char: Characterization,
                          aging_overrides: Optional[dict] = None) -> dict:
    """
    Return a human-readable dict of what build_pybamm_parameters would inject,
    without actually constructing PyBaMM objects. Useful for logging / notebooks.
    """
    summary: dict = {
        "source":              char.cell_id,
        "is_module":           char.is_module,
        "Q_RPT_per_cell_Ah":   char.per_cell_q_rpt_ah() if char.is_module else char.q_rpt_ah,
        "R0_at_50pct_SoC_mOhm": char.r0_at_soc(0.5),
        "SoH_pct":             char.soh_pct,
    }
    if char.ocv_soc_grid.size >= 4:
        if char.is_module:
            soc, v = char.per_cell_ocv()
        else:
            soc, v = char.ocv_soc_grid, char.ocv_v_curve
        summary["OCV_top_V"] = float(v[-1])
        summary["OCV_bot_V"] = float(v[0])
    if aging_overrides:
        summary["aging_overrides"] = dict(aging_overrides)
    return summary
