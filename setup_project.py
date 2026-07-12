#!/usr/bin/env python3
"""
setup_project.py
----------------
Run this ONCE from your terminal to scaffold the full project structure:

    cd /home/hj/Desktop/PINNs
    python3 setup_project.py

Your existing Data/ folder is never touched.
"""
from pathlib import Path

ROOT = Path("/home/hj/Desktop/PINNs")

DIRS = [
    # Agent instruction files land here
    "agents",
    # All YAML configs
    "configs",
    # Source packages
    "src/param_id",
    "src/simulation",
    "src/pinn",
    "src/inference",
    # Mirrors your existing Data/ with standardised names
    # Claude Code agents will read from Data/ directly — these are outputs only
    "data/processed",
    "data/synthetic",
    # Model checkpoints + plots
    "outputs/models",
    "outputs/results",
    "outputs/logs",
    # Lightweight local experiment tracking (run folders)
    "outputs/experiments",
    # Unit tests
    "tests",
]

INIT_PACKAGES = [
    "src",
    "src/param_id",
    "src/simulation",
    "src/pinn",
    "src/inference",
]

# ── Symlinks: map standardised names → your actual Data/ subfolders ─────────
# Left  = what agents expect under data/raw/
# Right = your actual folder under /home/hj/Desktop/PINNs/Data/
DATA_SYMLINKS = {
    "OCV_SOC":        "OCVSOC",        # note: your folder is OCVSOC not OCV_SOC
    "GITT":           "GITT",
    "DCIR":           "DCIR",
    "HPPC":           "HPPC",
    "RPT":            "RPT",
    "PeakPower":      "PeakPower",
    "RateCapability": "RateCapability",
    "ConstantPower":  "ConstantPower",
    "Longterm":       "Longterm",
    "SelfDischarge":  "SelfDischarge",
}


def main():
    print(f"\nScaffolding project at: {ROOT}\n")

    # 1. Create directories
    for d in DIRS:
        p = ROOT / d
        p.mkdir(parents=True, exist_ok=True)
        print(f"  mkdir  {d}/")

    # 2. Add __init__.py to src packages
    for pkg in INIT_PACKAGES:
        init = ROOT / pkg / "__init__.py"
        if not init.exists():
            init.write_text("")

    # 3. Create data/raw/ symlinks → Data/<subfolder>
    raw_dir = ROOT / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    print("\nCreating data/raw/ → Data/ symlinks:")
    for std_name, actual_name in DATA_SYMLINKS.items():
        src  = ROOT / "Data" / actual_name   # your real data
        link = raw_dir / std_name             # what agents import from
        if not src.exists():
            print(f"  SKIP   data/raw/{std_name} → Data/{actual_name}  (source not found)")
            continue
        if link.exists() or link.is_symlink():
            print(f"  EXISTS data/raw/{std_name}")
            continue
        link.symlink_to(src)
        print(f"  link   data/raw/{std_name} → Data/{actual_name}")

    print("""
Done. Next steps:
─────────────────────────────────────────────────────
1.  Install dependencies:
      pip install -r requirements.txt

2.  Open this folder in VS Code — Claude Code will read CLAUDE.md
    automatically as the project context file.

3.  Start Phase 1 (parameter identification):
      Open a new Claude Code task and paste the contents of:
      agents/AGENT_PARAM_ID.md

4.  Start Phase 2 in parallel (simulation sweep):
      Open another Claude Code task and paste:
      agents/AGENT_SIMULATION.md

5.  After both complete, run Phase 3:
      agents/AGENT_PINN.md
─────────────────────────────────────────────────────
""")


if __name__ == "__main__":
    main()
