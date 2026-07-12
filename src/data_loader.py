"""
src/data_loader.py
------------------
Unified data loader for all LFP characterisation tests.

Data resolution order (first found wins):
  1. /home/hj/Desktop/PINNs/data/raw/<TestName>/   ← symlinks created by setup_project.py
  2. /home/hj/Desktop/PINNs/Data/<TestName>/        ← your original folder, direct fallback
  3. /home/hj/Desktop/PINNs/Data/OCVSOC/            ← handles the OCVSOC spelling variant

Supports: .csv, .xlsx, .parquet, .txt (tab-separated)
Automatically detects and converts mA→A and mAh→Ah.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional
import re

# ── Project root (absolute — no ambiguity) ────────────────────────────────
PROJECT_ROOT = Path("/home/hj/Desktop/PINNs")
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# Ambient conditions (your stated test condition)
AMBIENT_TEMPERATURE_C = 25.0

# Canonical test names and their possible folder name variants
TEST_FOLDER_VARIANTS: dict[str, list[str]] = {
    "OCV_SOC":        ["OCV_SOC", "OCVSOC", "OCV-SOC", "ocvsoc"],
    "GITT":           ["GITT", "gitt"],
    "DCIR":           ["DCIR", "dcir"],
    "HPPC":           ["HPPC", "hppc"],
    "RPT":            ["RPT", "rpt"],
    "PeakPower":      ["PeakPower", "Peak_Power", "peakpower"],
    "RateCapability": ["RateCapability", "Rate_Capability", "ratecapability"],
    "ConstantPower":  ["ConstantPower", "Constant_Power", "constantpower"],
    "Longterm":       ["Longterm", "LongTerm", "Long_Term", "longterm"],
    "SelfDischarge":  ["SelfDischarge", "Self_Discharge", "selfdischarge"],
}

# Search roots in priority order
_SEARCH_ROOTS = [
    PROJECT_ROOT / "data" / "raw",
    PROJECT_ROOT / "Data",
]

# ── Column normalisation ───────────────────────────────────────────────────
COLUMN_MAPS: dict[str, list[str]] = {
    "voltage":     ["V", "Voltage", "voltage_V", "Ewe/V", "U/V",
                    "Voltage(V)", "Ecell/V", "voltage", "Vt/V",
                    # EVE raw schema
                    "volt_v"],
    "current":     ["I", "Current", "current_A", "I/mA", "Current(A)",
                    "<I>/mA", "current", "Current(mA)",
                    # EVE raw schema
                    "current_a"],
    "capacity":    ["Q", "Capacity", "capacity_Ah", "Q/mAh",
                    "Capacity(Ah)", "Q charge/discharge/mA.h",
                    "capacity", "Capacity(mAh)", "Discharge_Capacity(Ah)",
                    # EVE raw schema
                    "capacity_ah"],
    "time":        ["t", "Time", "time_s", "time/s", "Time(s)",
                    "time", "Test_Time(s)", "Step_Time(s)"],
    "soc":         ["SOC", "soc", "State_of_Charge", "SOC(%)", "soc_pct"],
    "temperature": ["T", "Temp", "temperature_C", "T/degC",
                    "Temperature(C)", "Temp(C)", "Temperature(°C)"],
    "cycle":       ["cycle", "Cycle", "cycle_number", "Cycle_Index",
                    "Cycle Number", "cycle_index",
                    # EVE raw schema
                    "cycle_no"],
}


def _find_test_dir(test_name: str) -> Path:
    """Locate test data folder, trying all name variants and search roots."""
    variants = TEST_FOLDER_VARIANTS.get(test_name, [test_name])
    tried = []
    for root in _SEARCH_ROOTS:
        for variant in variants:
            candidate = root / variant
            tried.append(candidate)
            if candidate.exists():
                return candidate
    raise FileNotFoundError(
        f"Could not find data for test '{test_name}'.\nLooked in:\n" +
        "\n".join(f"  {p}" for p in tried)
    )


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename raw column names to standardised names."""
    rename: dict[str, str] = {}
    used_targets: set[str] = set()
    for std_name, aliases in COLUMN_MAPS.items():
        for alias in aliases:
            if alias in df.columns and std_name not in used_targets:
                rename[alias] = std_name
                used_targets.add(std_name)
                break
    return df.rename(columns=rename)


def _fix_units(df: pd.DataFrame) -> pd.DataFrame:
    """Detect and fix mA→A and mAh→Ah conversions by magnitude."""
    df = df.copy()
    if "current" in df.columns:
        max_I = df["current"].abs().max()
        if max_I > 500:          # almost certainly mA
            df["current"] = df["current"] / 1000
    if "capacity" in df.columns:
        max_Q = df["capacity"].abs().max()
        # Note: large-format prismatic cells can legitimately exceed 50 Ah.
        # We only auto-convert when magnitudes look like mAh (e.g. 1000–5000).
        if max_Q > 500:          # almost certainly mAh
            df["capacity"] = df["capacity"] / 1000
    return df


def _parse_crate_value(x) -> Optional[float]:
    """
    Parse strings like "0.25C" or "0.333D" into floats.
    Returns None when not parseable.
    """
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    if isinstance(x, (int, float)) and not isinstance(x, bool):
        return float(x)
    s = str(x).strip()
    m = re.match(r"^([0-9]*\\.?[0-9]+)\\s*[cCdD]$", s)
    return float(m.group(1)) if m else None


def _standardise_cell_id(file_stem: str, df: pd.DataFrame) -> str:
    """
    Produce a consistent cell_id across tests.

    Preference order:
      1) 'cell_no' column (EVE schema) -> zero-padded 4-digit string
      2) filename stem containing 'cell_####'
      3) fallback to file stem
    """
    if "cell_no" in df.columns and df["cell_no"].notna().any():
        try:
            v = df["cell_no"].dropna().iloc[0]
            return f"{int(v):04d}"
        except Exception:
            pass

    m = re.search(r"cell_(\\d+)", file_stem)
    if m:
        return m.group(1).zfill(4)
    return file_stem


def _ensure_time_seconds(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure a monotonic 'time' column in seconds exists.

    If raw data provides 'absolute_time' (EVE schema), we sort by it and
    compute elapsed seconds. This is required for pulse-based analyses (GITT/HPPC/DCIR).
    """
    if "time" in df.columns:
        return df
    if "absolute_time" not in df.columns:
        return df

    t = pd.to_datetime(df["absolute_time"], errors="coerce")
    if t.isna().any():
        raise ValueError("Could not parse some 'absolute_time' values into datetimes")

    df = df.copy()
    df["_abs_t"] = t
    df = df.sort_values("_abs_t").reset_index(drop=True)
    t0 = df["_abs_t"].iloc[0]
    df["time"] = (df["_abs_t"] - t0).dt.total_seconds().astype(float)
    df = df.drop(columns=["_abs_t"])
    return df


def _read_file(f: Path) -> pd.DataFrame:
    """Read a single data file regardless of format."""
    suffix = f.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(f)
    elif suffix in (".xlsx", ".xls"):
        return pd.read_excel(f)
    elif suffix == ".txt":
        return pd.read_csv(f, sep="\t")
    else:
        # Try comma first, then semicolon (common in European lab software)
        try:
            return pd.read_csv(f)
        except Exception:
            return pd.read_csv(f, sep=";")


# ── Public API ─────────────────────────────────────────────────────────────

def load_test(test_name: str,
              cell_id: Optional[str] = None) -> pd.DataFrame:
    """
    Load all files for a characterisation test.

    Args:
        test_name : canonical name, e.g. 'GITT', 'RPT', 'OCV_SOC'
        cell_id   : filename stem to load only one cell (None = all)

    Returns:
        Normalised DataFrame with added 'cell_id' column.
    """
    test_dir = _find_test_dir(test_name)
    glob_patterns = ["*.csv", "*.parquet", "*.xlsx", "*.xls", "*.txt"]
    files: list[Path] = []
    for pat in glob_patterns:
        files.extend(sorted(test_dir.glob(pat)))

    if not files:
        raise ValueError(f"No data files found in {test_dir}")

    requested_id: Optional[str] = None
    if cell_id is not None:
        requested_id = str(cell_id)
        # Common convention in this repo: zero-padded 4-digit cell IDs ("0005")
        if requested_id.isdigit() and len(requested_id) < 4:
            requested_id = requested_id.zfill(4)

    frames = []
    for f in files:
        # Fast-path: if the filename already encodes the cell id, skip reading other cells.
        if requested_id is not None:
            m = re.search(r"cell_(\d+)", f.stem)
            if m and m.group(1).zfill(4) != requested_id:
                continue

        df = _read_file(f)
        df = _normalise_columns(df)
        df = _fix_units(df)
        df = _ensure_time_seconds(df)

        # Coerce common numeric columns
        for col in ["voltage", "current", "capacity", "time", "cycle"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Provide ambient temperature if missing
        if "temperature" not in df.columns:
            df["temperature"] = AMBIENT_TEMPERATURE_C

        # Parse C-rate if present in raw schema (optional)
        if "crate" in df.columns and "c_rate" not in df.columns:
            df["c_rate"] = df["crate"].map(_parse_crate_value)

        cid = _standardise_cell_id(f.stem, df)
        if requested_id and cid != requested_id and cid != str(cell_id):
            continue
        df["cell_id"] = cid
        frames.append(df)

    if not frames:
        raise ValueError(
            f"No files matched cell_id='{cell_id}' in {test_dir}"
        )

    return pd.concat(frames, ignore_index=True)


def list_cells(test_name: str) -> list[str]:
    """Return standardised cell IDs available for a given test."""
    test_dir = _find_test_dir(test_name)
    ids: list[str] = []
    for pattern in ["*.csv", "*.parquet", "*.xlsx", "*.txt"]:
        for f in test_dir.glob(pattern):
            m = re.search(r"cell_(\d+)", f.stem)
            if m:
                ids.append(m.group(1).zfill(4))
            else:
                ids.append(f.stem)
    return sorted(set(ids))


def load_rpt_capacity_fade(cell_id: Optional[str] = None) -> pd.DataFrame:
    """
    Extract per-cycle discharge capacity from RPT data.
    Returns: cell_id, cycle_n, Q_Ah, SOH, temperature_C (if available)
    """
    rpt = load_test("RPT", cell_id)
    discharge = rpt[rpt["current"] < 0]
    if "cycle" not in discharge.columns:
        raise ValueError("RPT data must include a 'cycle' column (e.g. cycle_no)")

    # EVE schema uses signed capacity; during discharge it trends negative.
    # Use the most negative value as the discharge capacity magnitude.
    per_cycle = (
        discharge
        .groupby(["cell_id", "cycle"])["capacity"]
        .min()
        .reset_index()
        .rename(columns={"capacity": "Q_discharge_signed_Ah", "cycle": "cycle_n"})
    )
    per_cycle["Q_Ah"] = -per_cycle["Q_discharge_signed_Ah"].astype(float)
    q0 = per_cycle.groupby("cell_id")["Q_Ah"].transform("first")
    per_cycle["SOH"] = per_cycle["Q_Ah"] / q0

    if "temperature" in rpt.columns:
        temp = (
            rpt.groupby(["cell_id", "cycle"])["temperature"]
            .mean().reset_index()
            .rename(columns={"cycle": "cycle_n", "temperature": "temperature_C"})
        )
        per_cycle = per_cycle.merge(temp, on=["cell_id", "cycle_n"], how="left")
    return per_cycle


def load_ocv_curve(cell_id: Optional[str] = None) -> pd.DataFrame:
    """
    Returns: cell_id, soc, voltage — sorted by SOC ascending.

    If SOC is not explicitly present, approximate SOC by normalising capacity
    against the per-file max capacity (EVE schema: 'max_cap' column when present).
    This is adequate for OCV curve fitting but should be treated as an approximation.
    """
    df = load_test("OCV_SOC", cell_id)
    if "soc" not in df.columns:
        if "capacity" not in df.columns:
            raise ValueError("OCV data missing both 'soc' and 'capacity' columns")
        if "max_cap" in df.columns and df["max_cap"].notna().any():
            qmax = float(df["max_cap"].dropna().iloc[0])
        else:
            # fallback: infer from observed capacity span
            qmax = float(np.nanmax(np.abs(df["capacity"].values)))
        if qmax <= 0:
            raise ValueError("Could not infer max capacity for SOC normalisation")
        # Use discharge capacity direction for SOC estimate (capacity may be signed)
        df = df.copy()
        df["soc"] = (1.0 - (np.abs(df["capacity"].astype(float)) / qmax)).clip(0.0, 1.0)
    return df.sort_values(["cell_id", "soc"]).reset_index(drop=True)[
        [c for c in ["cell_id", "soc", "voltage"] if c in df.columns]
    ]


def load_gitt_pulses(cell_id: Optional[str] = None) -> pd.DataFrame:
    """Returns GITT data with a pulse_id column marking each current pulse."""
    df = load_test("GITT", cell_id)
    df["pulse_id"] = (
        (df["current"].abs() > 0.001).astype(int)
        .diff().fillna(0).cumsum().astype(int)
    )
    return df


def load_hppc(cell_id: Optional[str] = None) -> pd.DataFrame:
    return load_test("HPPC", cell_id)


def load_dcir(cell_id: Optional[str] = None) -> pd.DataFrame:
    return load_test("DCIR", cell_id)


def load_longterm_capacity_fade(cell_id: Optional[str] = None) -> pd.DataFrame:
    """
    Extract per-cycle discharge capacity from long-term cycling data.

    Returns: cell_id, cycle_n, Q_Ah, SOH, temperature_C

    Notes:
    - Uses signed capacity convention from EVE exports (discharge capacity trends negative).
    - SOH is normalised to the first longterm cycle in the file (not necessarily BOL).
    """
    lt = load_test("Longterm", cell_id)
    discharge = lt[lt["current"] < 0]
    if "cycle" not in discharge.columns:
        raise ValueError("Longterm data must include a 'cycle' column (e.g. cycle_no)")

    per_cycle = (
        discharge
        .groupby(["cell_id", "cycle"])["capacity"]
        .min()
        .reset_index()
        .rename(columns={"capacity": "Q_discharge_signed_Ah", "cycle": "cycle_n"})
    )
    per_cycle["Q_Ah"] = -per_cycle["Q_discharge_signed_Ah"].astype(float)
    q0 = per_cycle.groupby("cell_id")["Q_Ah"].transform("first")
    per_cycle["SOH"] = per_cycle["Q_Ah"] / q0
    per_cycle["temperature_C"] = AMBIENT_TEMPERATURE_C
    return per_cycle


def save_processed(df: pd.DataFrame, name: str) -> Path:
    """Save a processed DataFrame to data/processed/<name>.parquet"""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = PROCESSED_DIR / f"{name}.parquet"
    df.to_parquet(out, index=False)
    print(f"Saved {len(df):,} rows → {out}")
    return out


# ── Quick sanity check ─────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Data loader sanity check")
    print(f"Project root : {PROJECT_ROOT}")
    print()
    for test in TEST_FOLDER_VARIANTS:
        try:
            d = _find_test_dir(test)
            cells = list_cells(test)
            print(f"  ✓ {test:<18} → {d.relative_to(PROJECT_ROOT)}  "
                  f"({len(cells)} cell files)")
        except FileNotFoundError:
            print(f"  ✗ {test:<18}  NOT FOUND")
