Create a comprehensive summary mapping PyBaMM parameters to the experimental tests used to derive, estimate, or optimize them. The goal is to identify the minimum set of laboratory tests required to parameterize a battery model while minimizing testing time and cost.

For each parameter, include:

* Whether it is:

  * directly measured,
  * derived from experimental data,
  * optimized/fitted through model calibration, or
  * obtained from literature/default values.
* The specific test(s) required to estimate or validate it.
* Whether the parameter can be obtained using:

  * standard electrochemical tests,
  * non-destructive characterization methods, or
  * destructive methods such as cell teardown and material characterization.
* The importance of the parameter for this research:

  * essential / high priority,
  * useful but optional,
  * negligible for the current scope.
* Any dependencies between parameters and tests.

Also identify:

* The absolutely necessary parameters required to build a reliable PyBaMM model for this research.
* Which parameters can realistically be inferred or optimized from routine lab experiments.
* Which parameters require advanced characterization, manufacturer data, or destructive analysis.
* The minimum sufficient experimental workflow needed to achieve an acceptable model fidelity.

---

# PyBaMM (Prada2013) parameter → experimental test mapping

**Scope.** Large-format LFP / graphite cells (105 Ah, 25 °C isothermal). Reference parameter set: `pybamm.ParameterValues("Prada2013")` — the only LFP set bundled with PyBaMM. Cell teardown is **not** available; we therefore rely on (a) the existing characterisation suite in [Data/](Data/), (b) manufacturer spec sheet for geometry, and (c) literature defaults for unidentifiable parameters.

**Source-type legend**

| Symbol | Meaning |
|---|---|
| **M** | Directly **m**easured from a test |
| **D** | **D**erived from raw measurements (algebra / counting) |
| **F** | **F**itted via model calibration (non-linear optimisation) |
| **L** | **L**iterature / default — not identifiable from our tests |
| **N/D** | Available **n**on-**d**estructively (test or spec sheet) |
| **DST** | Requires **d**e**st**ructive teardown or advanced characterisation |

**Priority legend**

| Symbol | Meaning (for the LFP RUL PINN goal) |
|---|---|
| ★★★ | Essential — model fails or is qualitatively wrong without it |
| ★★ | Useful — improves fidelity, but a reasonable literature default suffices |
| ★ | Negligible at 25 °C isothermal, low-rate scope |

---

## 1. Cell-level operating envelope (4 params)

| PyBaMM key | Type | Test(s) | Method | Priority | Notes / dependency |
|---|---|---|---|---|---|
| `Nominal cell capacity [A.h]` | M | RPT / OCVSOC / spec sheet | N/D | ★★★ | 105 Ah for this dataset. Used as reference for SOC counting everywhere. |
| `Current function [A]` | — | (scenario input) | — | ★★★ | Defined by the simulation protocol, not identified. |
| `Lower voltage cut-off [V]` | M | spec sheet / RPT discharge | N/D | ★★★ | 2.5 V on these cells (cf. [data/processed/cell_selection_report.md](data/processed/cell_selection_report.md)). |
| `Upper voltage cut-off [V]` | M | spec sheet / CCCV | N/D | ★★★ | 3.65 V (CV target). |

## 2. Open-circuit thermodynamics — the most diagnostic group (10 params)

| PyBaMM key | Type | Test(s) | Method | Priority | Notes |
|---|---|---|---|---|---|
| `Positive electrode OCP [V]` (function of y) | L → F | OCVSOC (anchored to literature half-cell) | N/D | ★★★ | Prada2013's LFP half-cell function used directly. Could be replaced by a half-cell measurement (DST). |
| `Negative electrode OCP [V]` (function of x) | L → F | OCVSOC | N/D | ★★★ | Same — Prada2013's graphite OCP. |
| `Maximum concentration in positive electrode [mol.m-3]` | L | (manufacturer / literature) | DST in principle | ★★ | Material constant; the OCV fit is invariant to its *value* once it's combined with active-material volume. |
| `Maximum concentration in negative electrode [mol.m-3]` | L | (literature) | DST | ★★ | Same. |
| `Initial concentration in positive electrode [mol.m-3]` | D | OCVSOC + stoich fit | N/D | ★★★ | `c_p_init = y_100 · c_p_max`. Comes out of [src/param_id/ocv_fit.py](src/param_id/ocv_fit.py). |
| `Initial concentration in negative electrode [mol.m-3]` | D | OCVSOC + stoich fit | N/D | ★★★ | `c_n_init = x_100 · c_n_max`. |
| `Open-circuit voltage at 0% SOC [V]` | M | OCVSOC | N/D | ★★ | Reported by the fitter; useful as a sanity check. |
| `Open-circuit voltage at 100% SOC [V]` | M | OCVSOC | N/D | ★★ | Same. |
| `Negative electrode OCP entropic change [V.K-1]` | L | dV/dT at controlled temperature | N/D (advanced) | ★ | Isothermal scope at 25 °C → set to 0. |
| `Positive electrode OCP entropic change [V.K-1]` | L | dV/dT | N/D (advanced) | ★ | Same. |

**Dependencies.** The stoichiometric windows {x_0, x_100, y_0, y_100} are **identified jointly** with electrode capacities Q_n and Q_p from a single OCVSOC discharge — they are not independently observable from full-cell voltage alone.

## 3. Cell geometry & electrode architecture (10 params)

| PyBaMM key | Type | Test(s) | Method | Priority | Notes |
|---|---|---|---|---|---|
| `Electrode height [m]` | M | spec sheet / CT scan | N/D | ★★ | Used only to compute electrode area. |
| `Electrode width [m]` | M | spec sheet / CT scan | N/D | ★★ | Same. |
| `Negative electrode thickness [m]` | M | DST (micrometer post-disassembly) | DST | ★ | Affects diffusion length but only as part of `Ds / L²`. Lit. default acceptable. |
| `Positive electrode thickness [m]` | M | DST | DST | ★ | Same. |
| `Separator thickness [m]` | M | DST / spec sheet | DST or N/D | ★ | Affects internal resistance modestly; lit. default acceptable. |
| `Negative electrode active material volume fraction` | L → F | DST (porosimetry) **or** fit to RPT capacity at known geometry | DST | ★★ | Can be tuned in a soft inverse problem so that geometry × concentrations reproduce the measured 105 Ah. |
| `Positive electrode active material volume fraction` | L → F | DST or capacity-matching | DST | ★★ | Same. |
| `Negative electrode porosity` | L | DST (Hg porosimetry / FIB-SEM) | DST | ★ | Lit. default acceptable at low rates. |
| `Positive electrode porosity` | L | DST | DST | ★ | Same. |
| `Separator porosity` | L | spec sheet / DST | DST or N/D | ★ | Same. |

**Underdetermination.** Without teardown, the combination (`area × thickness × active-material fraction × c_max`) can only be identified **as a product** (= cell capacity). Any single factor is arbitrary. We pin geometry + c_max to Prada2013 defaults and let the stoichiometric windows absorb the rest.

## 4. Solid-phase transport (4 params)

| PyBaMM key | Type | Test(s) | Method | Priority | Notes |
|---|---|---|---|---|---|
| `Negative particle radius [m]` | L | DST (SEM) | DST | ★★ | Coupled with `Ds` only via `Ds/r²`. We adopt Prada2013 default (5 µm). |
| `Positive particle radius [m]` | L | DST (SEM) | DST | ★★ | Default 50 nm for LFP. |
| `Negative particle diffusivity [m².s⁻¹]` | F (apparent) | GITT (with assumed L) | N/D | ★★ | [src/param_id/gitt_ds.py](src/param_id/gitt_ds.py) returns an *apparent* Dₛ only when an L is supplied — explicit by design (full-cell GITT cannot separate the two electrodes). |
| `Positive particle diffusivity [m².s⁻¹]` | F (apparent) | GITT (with assumed L) | N/D | ★★ | Same caveat. Default 5.9 × 10⁻¹⁸ m²/s is unrealistically small for LFP — likely needs review. |

**Critical scientific caveat (already documented in [src/param_id/gitt_ds.py](src/param_id/gitt_ds.py:9)).** Full-cell GITT contains contributions from *both* electrodes + ohmic + kinetic overpotentials. Splitting Ds_n and Ds_p uniquely from full-cell GITT alone is **not possible** without either half-cell data (DST) or a model-based pulse fit (e.g. SPMe). We report `D_app` with the explicit L assumption and flag it.

## 5. Reaction kinetics & double layer (6 params)

| PyBaMM key | Type | Test(s) | Method | Priority | Notes |
|---|---|---|---|---|---|
| `Negative electrode exchange-current density [A.m⁻²]` | F | HPPC RC fit (R1) + geometry | N/D | ★★ | R₁ from [src/param_id/dcir_hppc.py](src/param_id/dcir_hppc.py) provides the cell-level charge-transfer resistance; separating electrodes requires extra information (EIS, half-cell). |
| `Positive electrode exchange-current density [A.m⁻²]` | F | HPPC RC fit | N/D | ★★ | Same. |
| `Negative electrode charge transfer coefficient` | L | (Tafel fit, EIS) | N/D (advanced) | ★ | Almost always fixed at 0.5; identification is poorly conditioned. |
| `Positive electrode charge transfer coefficient` | L | (Tafel fit, EIS) | N/D (advanced) | ★ | Same. |
| `Negative electrode double-layer capacity [F.m⁻²]` | L | EIS (Nyquist semi-circle) | N/D | ★ | C₁ from HPPC pulse RC = cell-level value (~22,000 F here); per-area split requires EIS + electrode area. |
| `Positive electrode double-layer capacity [F.m⁻²]` | L | EIS | N/D | ★ | Same. |

## 6. Electrolyte transport (5 params)

| PyBaMM key | Type | Test(s) | Method | Priority | Notes |
|---|---|---|---|---|---|
| `Electrolyte conductivity [S.m⁻¹]` (fn of c) | L | (composition-dependent) | DST or N/D (advanced) | ★ | Strong literature consensus for 1 M LiPF₆ in EC:DMC/EC:EMC blends; Prada2013 default acceptable. |
| `Electrolyte diffusivity [m².s⁻¹]` | L | NMR / restricted diffusion | DST or N/D (advanced) | ★ | Same. |
| `Cation transference number` | L | Galvanostatic polarisation (Bruce–Vincent) | N/D (advanced) | ★ | Same. |
| `Initial concentration in electrolyte [mol.m⁻³]` | L | spec sheet | N/D | ★ | Set to ~1200 mol/m³ for 1 M LiPF₆. |
| `Thermodynamic factor` | L | (electrolyte literature) | DST | ★ | Default 1.0 acceptable at low rates. |

## 7. Effective transport in porous electrodes (3 params)

| PyBaMM key | Type | Test(s) | Method | Priority | Notes |
|---|---|---|---|---|---|
| `Negative electrode Bruggeman coefficient (electrode)` | L | Bruggeman or tortuosity fit | DST | ★ | Default 1.5. |
| `Positive electrode Bruggeman coefficient (electrode)` | L | DST | DST | ★ | Same. |
| `… (electrolyte) coefficients` (3) | L | DST | DST | ★ | Same. |

## 8. Electronic conductivity (2 params)

| PyBaMM key | Type | Test(s) | Method | Priority | Notes |
|---|---|---|---|---|---|
| `Negative electrode conductivity [S.m⁻¹]` | L | 4-probe on coated electrode | DST | ★ | Default 215 S/m for graphite; barely rate-limiting at C/3. |
| `Positive electrode conductivity [S.m⁻¹]` | L | 4-probe | DST | ★★ | LFP is intrinsically resistive (~0.3 S/m default); matters for high-C rate behaviour but at our 0.5 C HPPC step it is dominated by interfacial resistance. |

## 9. Thermal (4 params)

| PyBaMM key | Type | Test(s) | Method | Priority | Notes |
|---|---|---|---|---|---|
| `Ambient temperature [K]` | M | thermocouple | N/D | ★★★ | Set to 298.15 K (25 °C). |
| `Initial temperature [K]` | M | thermocouple | N/D | ★★★ | Same. |
| `Reference temperature [K]` | — | (model setting) | — | ★★ | Used when Arrhenius scaling is active; isothermal = irrelevant. |
| `Contact resistance [Ohm]` | M | DCIR | N/D | ★★ | Can fold into R₀ from [src/param_id/dcir_hppc.py](src/param_id/dcir_hppc.py) (≈1–5 mΩ in this dataset). |

## 10. Battery aggregation (2 params)

| PyBaMM key | Type | Test(s) | Method | Priority | Notes |
|---|---|---|---|---|---|
| `Number of cells connected in series to make a battery` | — | scenario | — | ★ | Single cell here → 1. |
| `Number of electrodes connected in parallel to make a cell` | — | spec sheet | N/D | ★ | Jelly-roll / stack count; for full-cell DFN it is folded into electrode area. |

## 11. Degradation parameters — **not in Prada2013, but required for RUL**

PyBaMM exposes these through additional submodels (`pybamm.lithium_ion.DFN` with `sei="ec reaction limited"`, etc.). For this project they are the *primary* identifiable surface from cycling data.

| Parameter | Type | Test(s) | Method | Priority | Notes |
|---|---|---|---|---|---|
| SEI growth rate constant `k_SEI` | F (upper bound) | SelfDischarge + Longterm capacity fade | N/D | ★★★ | Self-discharge gives an upper bound on the parasitic current → upper bound on k_SEI. Long-term fade slope gives the operational identifiability. |
| SEI molar volume / partial volume | L | DST (electrolyte breakdown chemistry) | DST | ★★ | Default acceptable. |
| EC initial concentration (for `ec reaction limited`) | L | spec sheet | N/D | ★★ | ~4500 mol/m³. |
| Loss of active material rate (LAM_n, LAM_p) | F | RPT capacity fade + OCV-shift analysis | N/D | ★★ | Can be inferred from IC peak displacement on RPT discharge curves. |
| Lithium plating kinetics | L | low-T cycling | N/D (advanced) | ★ | Negligible at 25 °C with the modest C-rates used here. |

---

## 12. The minimum sufficient experimental workflow

To parameterise a PyBaMM Prada2013-style DFN model **for the LFP RUL PINN goal** using only **routine, non-destructive lab tests**:

### Tier A — essential (must-have)
1. **OCVSOC at C/20 (or slower)** — yields stoichiometric windows {x_0, x_100, y_0, y_100} via [src/param_id/ocv_fit.py](src/param_id/ocv_fit.py), and hence initial electrode concentrations and capacity utilisation. ★★★
2. **RPT (reference performance test) every N cycles** — yields the SOH(n) trajectory that the Neural ODE will learn against. ★★★
3. **HPPC** — yields cell-level R₀(SOC), R₁(SOC), τ(SOC), C₁(SOC) via [src/param_id/dcir_hppc.py](src/param_id/dcir_hppc.py). ★★★
4. **Long-term cycling at fixed protocol** — yields the empirical fade curve that constrains SEI/LAM rates. ★★★
5. **Self-discharge (≥ 24 h OCV decay at full charge)** — bounds SEI parasitic current → k_SEI upper bound. ★★★

### Tier B — strongly recommended (substantially reduces literature dependence)
6. **GITT** — apparent diffusivity (with explicit L assumption); also gives a second, slower-time-scale resistance signature distinct from HPPC. ★★
7. **DCIR** — independent confirmation of R₀ at a different reference SOC. ★★
8. **Rate Capability (multiple C-rates discharge)** — exposes mass-transport limits that constrain `Ds`, electrolyte transport, and porosity. ★★
9. **Manufacturer spec sheet** — geometry (length × width × thickness), separator thickness, nominal capacity, voltage limits, electrolyte composition. ★★★ (but cheap)

### Tier C — optional / out of scope for this work
10. **Multi-temperature OCV/HPPC** — for entropic ∂U/∂T and Arrhenius activation energies. ★ (isothermal scope).
11. **EIS (galvanostatic / potentiostatic)** — separates ohmic, SEI, charge-transfer resistances; gives double-layer capacitance per area. ★.
12. **Cell teardown + SEM/porosimetry + half-cell measurements** — would directly furnish electrode-specific Dₛ, particle radius, porosity, Bruggeman, true active-material fractions. Eliminates several literature defaults but is destructive. ★ for the present work.

### What this gets you
With Tier A + Tier B + spec sheet, you can identify:

- The four stoichiometric limits and their associated electrode capacities (OCVSOC + ocv_fit)
- Lumped cell-level R₀, R₁, τ, C₁ over the SOC range probed by HPPC (here: SOC 0.97–1.00 — a **dataset-specific limitation** worth widening in any future protocol)
- An *apparent* Dₛ with an explicit L assumption (GITT)
- An *upper bound* on the SEI growth rate constant (SelfDischarge)
- An *operational* SEI / LAM rate that reproduces the measured Longterm capacity fade (model-based fit, e.g. in PyBaMM with the `ec reaction limited` SEI submodel)

What **remains as literature defaults** (and is therefore the place a future investment in teardown would pay off): electrode thicknesses, porosities, Bruggeman exponents, particle radii, electrolyte transport properties, transfer coefficients, double-layer capacitances per area.

## 13. Mapping to this repository

| Test folder | What it parameterises here | Identifier script |
|---|---|---|
| [Data/OCVSOC](Data/OCVSOC/) | Stoichiometry, initial concentrations | [src/param_id/ocv_fit.py](src/param_id/ocv_fit.py) |
| [Data/GITT](Data/GITT/) | Apparent solid diffusivity (with L) | [src/param_id/gitt_ds.py](src/param_id/gitt_ds.py) |
| [Data/DCIR](Data/DCIR/) | R₀ at mid-SOC | [src/param_id/dcir_hppc.py](src/param_id/dcir_hppc.py) |
| [Data/HPPC](Data/HPPC/) | R₀, R₁, τ, C₁ near top of charge | [src/param_id/dcir_hppc.py](src/param_id/dcir_hppc.py) |
| [Data/SelfDischarge](Data/SelfDischarge/) | k_SEI upper bound | `src/param_id/sei_selfdisc.py` *(to be written)* |
| [Data/RPT](Data/RPT/) | SOH(n) target, LAM signature | (used in PINN training, not in param ID) |
| [Data/Longterm](Data/Longterm/) | Operational SEI/LAM rate calibration | (PyBaMM sweep & PINN fine-tuning) |
| [Data/RateCapability](Data/RateCapability/) | Cross-validation only (not used for ID yet) | — |
| [Data/PeakPower](Data/PeakPower/) | Cross-validation only | — |
| [Data/ConstantPower](Data/ConstantPower/) | Cross-validation only | — |

## 14. Bottom line for this project

- **Cannot avoid the literature** for: electrolyte transport, particle radius, Bruggeman/porosity, electronic conductivities, charge-transfer coefficients, electrode thicknesses. None of these is identifiable from the present test suite, but at 25 °C and ≤ 0.5 C the residual error they contribute is small relative to the SEI/LAM rates that drive RUL.
- **Identifiable from existing tests**: stoichiometric windows, initial Li inventory, lumped R₀/R₁/τ/C₁, apparent Dₛ (with L), SEI rate (with explicit upper bound from self-discharge plus operational fit to Longterm).
- **Highest-value future investment if budget allows**: (i) widen HPPC SOC sweep so R(SOC) is identifiable across the whole range, (ii) one teardown to fix porosity / volume fractions / particle radii, (iii) multi-temperature RPT to enable Arrhenius scaling outside 25 °C.
