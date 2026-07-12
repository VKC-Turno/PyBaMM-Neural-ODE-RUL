#!/usr/bin/env bash
# Phase 2 DE fit dispatch — held-out cells, FIRST 100 cycles only.
# Option B: honest "field-calibration" story — evaluates the operator's
# ability to project beyond the calibration horizon.
#
# 3 cells x workers=1 each = 3 cores concurrent (safe under n_jobs<=5 cap).
# Expected wall-time ~15 min/cell (sim only runs 100 cycles per eval).

set -euo pipefail
cd /home/hj/Desktop/PINNs

PY=/home/hj/Desktop/PINNs/.venv/bin/python
SCRIPT=/home/hj/Desktop/PINNs/Voltaris/Data_Exploration/phase2_de_fit.py
LOG_DIR=/home/hj/Desktop/PINNs/outputs/logs/de_heldout_early100
OUT_DIR=/home/hj/Desktop/PINNs/configs/deg_params
mkdir -p "$LOG_DIR"

STAMP=$(date -u +%Y%m%dT%H%M%SZ)

dispatch() {
  local MAKE="$1" CELL="$2"
  local LOG="$LOG_DIR/${MAKE}_${CELL}.${STAMP}.log"
  # Distinct suffix so the early-100 fits do not clobber any future
  # full-curve YAMLs at configs/deg_params/{make}_{cell}.yaml.
  local OUT="$OUT_DIR/${MAKE}_${CELL}_early100.yaml"
  echo "[$MAKE $CELL] -> $LOG"
  nohup "$PY" "$SCRIPT" "$MAKE" "$CELL" \
      --n-evals 200 \
      --workers 1 \
      --n-cycles-override 100 \
      --out "$OUT" \
      >"$LOG" 2>&1 &
  echo "  pid=$! yaml=$OUT"
}

dispatch CALB 0029
dispatch EVE  0003
dispatch REPT 0031

echo
echo "Dispatched 3 DE fits (early-100).  Monitor with:"
echo "  tail -F $LOG_DIR/*.${STAMP}.log"
echo
echo "Completion check (all 3 YAMLs written):"
echo "  ls $OUT_DIR/CALB_0029_early100.yaml $OUT_DIR/EVE_0003_early100.yaml $OUT_DIR/REPT_0031_early100.yaml"
