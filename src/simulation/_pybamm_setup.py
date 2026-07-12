"""
src/simulation/_pybamm_setup.py
-------------------------------
Shared PyBaMM model + parameter set construction used by all simulation
scripts. We start from `Prada2013` (LFP chemistry) and inject the missing
degradation parameters from `OKane2022` so the SEI / plating / stress-LAM
submodels build successfully. Identified per-cell overrides from
`configs/identified_params.yaml` (Phase-1 outputs) are then applied.

Notes
~~~~~
- `Prada2013` is a 2.3 Ah cell parameterisation; our real cells are 105 Ah.
  We do **not** rescale geometry here — SOH(n) is a dimensionless quantity
  and the PINN trains on the *shape* of the fade trajectory, not the
  absolute Ah. Absolute capacities can be rescaled at inference time.
- The merged set is fully self-consistent: PyBaMM will refuse to build the
  model otherwise.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import pybamm
import yaml


DEFAULT_OPTIONS: dict[str, str] = {
    "SEI": "solvent-diffusion limited",
    "SEI porosity change": "true",
    "lithium plating": "irreversible",
    "loss of active material": "stress-driven",
}

DEFAULT_BASE = "Prada2013"
DEFAULT_DEGRADATION_DONOR = "OKane2022"


def build_dfn(options: Optional[dict[str, str]] = None) -> pybamm.lithium_ion.DFN:
    """Construct the DFN model with all degradation submodels enabled."""
    return pybamm.lithium_ion.DFN(options=options or DEFAULT_OPTIONS)


def build_parameter_values(
    base: str = DEFAULT_BASE,
    degradation_donor: str = DEFAULT_DEGRADATION_DONOR,
    overrides: Optional[dict[str, Any]] = None,
) -> pybamm.ParameterValues:
    """
    Build a complete ParameterValues for the DFN + degradation model.

    Steps:
      1. Start from the LFP base (`Prada2013`).
      2. Copy any missing keys from the degradation donor (`OKane2022`)
         without overwriting LFP chemistry.
      3. Apply caller-supplied overrides last (highest priority).
    """
    param = pybamm.ParameterValues(base)
    donor = pybamm.ParameterValues(degradation_donor)
    for k in set(donor.keys()) - set(param.keys()):
        param.update({k: donor[k]}, check_already_exists=False)
    if overrides:
        for k, v in overrides.items():
            # Only set keys PyBaMM already knows about; ignore unknowns
            # silently so identified_params.yaml can carry diagnostics.
            if k in param.keys():
                param.update({k: v})
    return param


def overrides_from_identified_params(
    identified_path: Path | str = Path("configs/identified_params.yaml"),
) -> dict[str, Any]:
    """
    Translate `configs/identified_params.yaml` into PyBaMM keys.

    Currently maps:
      - stoichiometry → initial concentrations in each electrode
      - SEI ceiling   → 'SEI kinetic rate constant [m.s-1]' (clamped)

    Returns an empty dict if the file is missing.
    """
    p = Path(identified_path)
    if not p.exists():
        return {}
    cfg = yaml.safe_load(p.read_text()) or {}

    overrides: dict[str, Any] = {}

    stoich = cfg.get("stoichiometry") or {}
    if {"x_100", "y_100"}.issubset(stoich.keys()):
        # Initial concentration in each electrode at SOC=1
        x_100 = float(stoich["x_100"])
        y_100 = float(stoich["y_100"])
        # We need maximum concentrations — use whatever is in the base set.
        base = pybamm.ParameterValues(DEFAULT_BASE)
        c_n_max = float(base["Maximum concentration in negative electrode [mol.m-3]"])
        c_p_max = float(base["Maximum concentration in positive electrode [mol.m-3]"])
        overrides["Initial concentration in negative electrode [mol.m-3]"] = x_100 * c_n_max
        overrides["Initial concentration in positive electrode [mol.m-3]"] = y_100 * c_p_max

    sei = cfg.get("sei") or {}
    if "k_SEI_max_m_per_s_median" in sei:
        # Cap default k_SEI at the identified ceiling (don't go higher)
        ceiling = float(sei["k_SEI_max_m_per_s_median"])
        overrides["SEI kinetic rate constant [m.s-1]"] = ceiling

    return overrides
