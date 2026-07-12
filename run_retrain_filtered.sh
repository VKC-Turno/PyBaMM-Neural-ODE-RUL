#!/usr/bin/env bash
set -u
cd /home/hj/Desktop/PINNs

mkdir -p outputs/logs
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
STATUS=outputs/logs/retrain_${STAMP}_status.log
PRETRAIN_LOG=outputs/logs/retrain_${STAMP}_pretrain.log
FINETUNE_LOG=outputs/logs/retrain_${STAMP}_finetune.log
HEARTBEAT=outputs/logs/retrain_${STAMP}_heartbeat

status() {
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $1" | tee -a "$STATUS"
    touch "$HEARTBEAT"
}
(while true; do touch "$HEARTBEAT" 2>/dev/null || exit 0; sleep 30; done) &
HB_PID=$!
trap 'kill $HB_PID 2>/dev/null || true' EXIT

status "RETRAIN START stamp=${STAMP}  filter: max-rate-per-cycle=5e-3"

status "PHASE pretrain: launching"
T0=$(date +%s)
.venv/bin/python -u -m src.pinn.train pretrain --max-rate-per-cycle 5e-3 > "$PRETRAIN_LOG" 2>&1
RC=$?
DT=$(( $(date +%s) - T0 ))
[ $RC -ne 0 ] && { status "PHASE pretrain: FAILED rc=$RC (${DT}s)"; exit 1; }
status "PHASE pretrain: DONE (${DT}s)"

status "PHASE finetune: launching"
T0=$(date +%s)
.venv/bin/python -u -m src.pinn.train finetune > "$FINETUNE_LOG" 2>&1
RC=$?
DT=$(( $(date +%s) - T0 ))
[ $RC -ne 0 ] && { status "PHASE finetune: FAILED rc=$RC (${DT}s)"; exit 2; }
status "PHASE finetune: DONE (${DT}s)"

status "RETRAIN COMPLETE"
