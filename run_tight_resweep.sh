#!/usr/bin/env bash
# run_tight_resweep.sh
# ----------------------------------------------------------------------
# Queued behind the currently-running filter-retrain (PGID 887624).
#
# Sequence:
#   1. Wait until any retrain_*_status.log shows "RETRAIN COMPLETE" or "FAILED"
#      AND no python -m src.pinn.train process is still running.
#   2. Archive the filter-retrain checkpoints as `*_filtered.pt`.
#   3. Archive the existing trajectories.parquet as trajectories_aggressive_800.parquet.
#   4. Run the tight PyBaMM sweep (configs/sweep_config_tight.yaml).
#   5. Retrain pretrain (no extra filter — tight data is already physical) +
#      finetune.
#   6. Run inference anchored to lab data and write a spec-validation report.
# ----------------------------------------------------------------------
set -u
cd "$(dirname "$(readlink -f "$0")")"

mkdir -p outputs/logs data/synthetic/archive
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
STATUS=outputs/logs/tight_${STAMP}_status.log
HEARTBEAT=outputs/logs/tight_${STAMP}_heartbeat

status() {
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $1" | tee -a "$STATUS"
    touch "$HEARTBEAT"
}

(while true; do touch "$HEARTBEAT" 2>/dev/null || exit 0; sleep 30; done) &
HB_PID=$!
trap 'kill $HB_PID 2>/dev/null || true' EXIT

PY=.venv/bin/python

status "TIGHT-RESWEEP QUEUED stamp=$STAMP — waiting for filter-retrain to finish"

# --- 1. Wait for previous retrain to finish ------------------------------
while true; do
    PREV_STATUS=$(ls -t outputs/logs/retrain_*_status.log 2>/dev/null | head -1)
    if [ -n "$PREV_STATUS" ]; then
        last=$(tail -1 "$PREV_STATUS")
        if echo "$last" | grep -qE "RETRAIN COMPLETE|FAILED"; then
            if ! pgrep -f "src.pinn.train" >/dev/null 2>&1; then
                status "Previous retrain finished: $last"
                break
            fi
        fi
    fi
    sleep 60
done

# --- 2. Archive filter-retrain checkpoints -------------------------------
status "PHASE archive: stash filter-retrain checkpoints"
[ -f outputs/models/pinn_pretrained.pt ] && \
    cp -p outputs/models/pinn_pretrained.pt outputs/models/pinn_pretrained_filtered_${STAMP}.pt
[ -f outputs/models/pinn_finetuned.pt ] && \
    cp -p outputs/models/pinn_finetuned.pt outputs/models/pinn_finetuned_filtered_${STAMP}.pt

# --- 3. Archive the aggressive sweep -------------------------------------
if [ -f data/synthetic/trajectories.parquet ]; then
    mv data/synthetic/trajectories.parquet \
       data/synthetic/archive/trajectories_aggressive_800_${STAMP}.parquet
    status "Archived old trajectories → data/synthetic/archive/trajectories_aggressive_800_${STAMP}.parquet"
fi
if [ -d data/synthetic/ic_curves ] && [ "$(ls data/synthetic/ic_curves | wc -l)" -gt 0 ]; then
    mv data/synthetic/ic_curves data/synthetic/archive/ic_curves_aggressive_${STAMP}
    mkdir -p data/synthetic/ic_curves
fi
[ -f data/synthetic/sweep_manifest.yaml ] && \
    mv data/synthetic/sweep_manifest.yaml \
       data/synthetic/archive/sweep_manifest_aggressive_${STAMP}.yaml

# --- 4. Run tight sweep --------------------------------------------------
TIGHT_SWEEP_LOG=outputs/logs/tight_${STAMP}_sweep.log
status "PHASE sweep: launching tight sweep (configs/sweep_config_tight.yaml)"
T0=$(date +%s)
$PY -u -m src.simulation.run_sweep --config configs/sweep_config_tight.yaml --n-jobs 3 > "$TIGHT_SWEEP_LOG" 2>&1
RC=$?
DT=$(( $(date +%s) - T0 ))
[ $RC -ne 0 ] && { status "PHASE sweep: FAILED rc=$RC (${DT}s) — see $TIGHT_SWEEP_LOG"; exit 1; }
status "PHASE sweep: DONE (${DT}s)"

# --- 5. Pretrain + finetune ----------------------------------------------
TIGHT_PRE_LOG=outputs/logs/tight_${STAMP}_pretrain.log
TIGHT_FT_LOG=outputs/logs/tight_${STAMP}_finetune.log

status "PHASE pretrain: launching (no extra filter — tight data is physical)"
T0=$(date +%s)
$PY -u -m src.pinn.train pretrain > "$TIGHT_PRE_LOG" 2>&1
RC=$?
DT=$(( $(date +%s) - T0 ))
[ $RC -ne 0 ] && { status "PHASE pretrain: FAILED rc=$RC (${DT}s) — see $TIGHT_PRE_LOG"; exit 2; }
status "PHASE pretrain: DONE (${DT}s)"

status "PHASE finetune: launching"
T0=$(date +%s)
$PY -u -m src.pinn.train finetune > "$TIGHT_FT_LOG" 2>&1
RC=$?
DT=$(( $(date +%s) - T0 ))
[ $RC -ne 0 ] && { status "PHASE finetune: FAILED rc=$RC (${DT}s) — see $TIGHT_FT_LOG"; exit 3; }
status "PHASE finetune: DONE (${DT}s)"

# --- 6. Inference + spec comparison --------------------------------------
TIGHT_INF_LOG=outputs/logs/tight_${STAMP}_inference.log
status "PHASE inference: anchored cell 0005 + spec compare"
$PY -u -m src.inference.predict_rul --cell-id 0005 --anchor-to-lab --n-mc-samples 50 \
    --out outputs/results/rul_report_tight.json > "$TIGHT_INF_LOG" 2>&1
RC=$?
[ $RC -ne 0 ] && { status "PHASE inference: FAILED rc=$RC"; exit 4; }

# Extract the model's cycle-to-80%
$PY <<'PYEOF' >> "$STATUS" 2>&1
import json, numpy as np
out = json.loads(open('outputs/results/rul_report_tight.json').read())
n = np.asarray(out['n_axis'])
traj = np.clip(np.asarray(out['soh_trajectory_mean']), 0, 1)
above_80 = traj > 0.80
n_80 = float(n[np.where(~above_80)[0][0]]) if (above_80.any() and not above_80.all()) else float('nan')
print(f"[INFO] tight model: anchor cycle {out['cycle_now']:.0f} SOH {out['soh_now']:.4f}")
print(f"[INFO] tight model: cycles to SOH=0.80 = {n_80:.0f}  (spec ≥ 4000)")
print(f"[INFO] tight model: RUL to EOL={out['eol_threshold']} = {out['rul_mean']:.0f} cycles [{out['rul_p5']:.0f},{out['rul_p95']:.0f}]")
PYEOF

status "TIGHT-RESWEEP COMPLETE"
