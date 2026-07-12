#!/usr/bin/env bash
# run_full_pipeline.sh
# ----------------------------------------------------------------------
# Chain the three long-running phases of the LFP RUL PINN pipeline:
#   1. PyBaMM degradation sweep   (configs/sweep_config.yaml defaults)
#   2. PINN pre-train on synthetic (configs/pinn_config.yaml phase1)
#   3. PINN fine-tune on real      (configs/pinn_config.yaml phase2)
#
# Each phase's full output goes to its own log under outputs/logs/.
# A short status line per phase is appended to STATUS_FILE so a watcher
# can tail it without parsing the noisy per-phase logs.
#
# Stops early if any phase exits non-zero.
# ----------------------------------------------------------------------
set -u
cd "$(dirname "$(readlink -f "$0")")"

# Optional overrides — used for crash-recovery runs at lower load
N_JOBS="${N_JOBS:-}"            # empty → run_sweep.py default (-1, all cores)
MIN_CRATE="${MIN_CRATE:-}"      # empty → no override
SWEEP_EXTRA_ARGS="${SWEEP_EXTRA_ARGS:-}"

mkdir -p outputs/logs
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
STATUS_FILE="outputs/logs/full_run_${STAMP}_status.log"
SWEEP_LOG="outputs/logs/full_run_${STAMP}_sweep.log"
PRETRAIN_LOG="outputs/logs/full_run_${STAMP}_pretrain.log"
FINETUNE_LOG="outputs/logs/full_run_${STAMP}_finetune.log"
HEARTBEAT="outputs/logs/full_run_${STAMP}_heartbeat"

PY=.venv/bin/python

status() {
    local msg="$1"
    local line="[$(date -u +%Y-%m-%dT%H:%M:%SZ)] ${msg}"
    echo "${line}" | tee -a "${STATUS_FILE}"
    # Touch heartbeat on every status line so monitors can use mtime
    touch "${HEARTBEAT}"
}

# Background heartbeat ticker: touch the heartbeat file every 30s so a
# monitor watching its mtime can detect liveness without parsing logs.
(
    while true; do touch "${HEARTBEAT}" 2>/dev/null || exit 0; sleep 30; done
) &
HB_PID=$!
trap 'kill ${HB_PID} 2>/dev/null || true' EXIT

status "FULL PIPELINE START stamp=${STAMP}"
status "  sweep log    : ${SWEEP_LOG}"
status "  pretrain log : ${PRETRAIN_LOG}"
status "  finetune log : ${FINETUNE_LOG}"
status "  heartbeat    : ${HEARTBEAT}"
status "  overrides    : N_JOBS=${N_JOBS:-default} MIN_CRATE=${MIN_CRATE:-default} EXTRA=${SWEEP_EXTRA_ARGS:-none}"

SWEEP_CMD_ARGS=""
[ -n "${N_JOBS}" ]    && SWEEP_CMD_ARGS="${SWEEP_CMD_ARGS} --n-jobs ${N_JOBS}"
[ -n "${MIN_CRATE}" ] && SWEEP_CMD_ARGS="${SWEEP_CMD_ARGS} --min-crate ${MIN_CRATE}"
[ -n "${SWEEP_EXTRA_ARGS}" ] && SWEEP_CMD_ARGS="${SWEEP_CMD_ARGS} ${SWEEP_EXTRA_ARGS}"

# ── 1. SWEEP ──────────────────────────────────────────────────────────
status "PHASE 1/3 sweep: launching"
T0=$(date +%s)
${PY} -u -m src.simulation.run_sweep ${SWEEP_CMD_ARGS} >"${SWEEP_LOG}" 2>&1
RC=$?
DT=$(( $(date +%s) - T0 ))
if [ ${RC} -ne 0 ]; then
    status "PHASE 1/3 sweep: FAILED (rc=${RC}, ${DT}s) — see ${SWEEP_LOG}"
    exit 1
fi
status "PHASE 1/3 sweep: DONE (${DT}s)"

# ── 2. PRETRAIN ───────────────────────────────────────────────────────
status "PHASE 2/3 pretrain: launching"
T0=$(date +%s)
${PY} -u -m src.pinn.train pretrain >"${PRETRAIN_LOG}" 2>&1
RC=$?
DT=$(( $(date +%s) - T0 ))
if [ ${RC} -ne 0 ]; then
    status "PHASE 2/3 pretrain: FAILED (rc=${RC}, ${DT}s) — see ${PRETRAIN_LOG}"
    exit 2
fi
status "PHASE 2/3 pretrain: DONE (${DT}s)"

# ── 3. FINETUNE ───────────────────────────────────────────────────────
status "PHASE 3/3 finetune: launching"
T0=$(date +%s)
${PY} -u -m src.pinn.train finetune >"${FINETUNE_LOG}" 2>&1
RC=$?
DT=$(( $(date +%s) - T0 ))
if [ ${RC} -ne 0 ]; then
    status "PHASE 3/3 finetune: FAILED (rc=${RC}, ${DT}s) — see ${FINETUNE_LOG}"
    exit 3
fi
status "PHASE 3/3 finetune: DONE (${DT}s)"

status "FULL PIPELINE COMPLETE"
