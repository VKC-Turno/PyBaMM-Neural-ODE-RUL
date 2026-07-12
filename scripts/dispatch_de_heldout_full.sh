#!/usr/bin/env bash
# Phase 2 DE fit dispatch — held-out cells, FULL Longterm SoH curve.
# Option A: best numbers, weakest story (over-fits per-cell horizon).
#
# 3 cells x workers=1 each = 3 cores concurrent (safe under n_jobs<=5 cap).
# Expected wall-time ~30 min/cell.  Detach with nohup; monitor via tail -F.

set -euo pipefail
cd /home/hj/Desktop/PINNs

PY=/home/hj/Desktop/PINNs/.venv/bin/python
SCRIPT=/home/hj/Desktop/PINNs/Voltaris/Data_Exploration/phase2_de_fit.py
LOG_DIR=/home/hj/Desktop/PINNs/outputs/logs/de_heldout_full
OUT_DIR=/home/hj/Desktop/PINNs/configs/deg_params
mkdir -p "$LOG_DIR"

STAMP=$(date -u +%Y%m%dT%H%M%SZ)

dispatch() {
  local MAKE="$1" CELL="$2"
  local LOG="$LOG_DIR/${MAKE}_${CELL}.${STAMP}.log"
  echo "[$MAKE $CELL] -> $LOG"
  nohup "$PY" "$SCRIPT" "$MAKE" "$CELL" \
      --n-evals 200 \
      --workers 1 \
      >"$LOG" 2>&1 &
  echo "  pid=$! yaml=$OUT_DIR/${MAKE}_${CELL}.yaml"
}

dispatch CALB 0029
dispatch EVE  0003
dispatch REPT 0031

echo
echo "Dispatched 3 DE fits (full-curve).  Monitor with:"
echo "  tail -F $LOG_DIR/*.${STAMP}.log"
echo
echo "Completion check (all 3 YAMLs written):"
echo "  ls $OUT_DIR/CALB_0029.yaml $OUT_DIR/EVE_0003.yaml $OUT_DIR/REPT_0031.yaml"
