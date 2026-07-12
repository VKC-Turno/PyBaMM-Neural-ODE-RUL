#!/bin/bash
# Chain Phase 3 and final report after Phase 2 completes.
set -e
cd /home/hj/Desktop/PINNs
echo "=== Waiting for Phase 2 python process to end ==="
while pgrep -f "verify_e2e_phase2" > /dev/null; do sleep 30; done
echo "=== Phase 2 done, starting Phase 3 ==="
.venv/bin/python src/simulation/verify_e2e_phase3.py \
    > data/synthetic/verification/eve_0008_phase3_run.log 2>&1
echo "=== Writing final report ==="
.venv/bin/python src/simulation/verify_e2e_report.py
echo "=== Done ==="
