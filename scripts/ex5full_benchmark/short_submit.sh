#!/bin/bash
# ex5 shortcuts, run from anywhere on triton:
#   bash scripts/ex5full_benchmark/short_submit.sh smoke   # 1 scene (nerf_synthetic/ship, lmbda 0.001 as in run_shell_blender.py), full 30k config
#   bash scripts/ex5full_benchmark/short_submit.sh all     # all 27 jobs (5 datasets, lmbda from run_shell_*.py)
#   bash scripts/ex5full_benchmark/short_submit.sh collect [smoke]   # summarize finished runs
set -euo pipefail
cd "$(dirname "$0")/../.."   # repo root: submit.py / train.py use paths relative to it

EXP=ex5full_benchmark
MODE=${1:-}

case "$MODE" in
    smoke)
        # Same flags as the real run, only one small scene, separate output dir and logs.
        python3 scripts/$EXP/submit.py --tag smoke \
            --datasets nerf_synthetic --scenes ship \
            --time 1:00:00 --mem 64G --run
        echo "When done: cat outputs/${EXP}_smoke/nerf_synthetic/ship/0.001/metrics.json"
        echo "Log:       scripts/$EXP/sbatch_logs/HACplus_${EXP}_smoke/"
        ;;
    all)
        python3 scripts/$EXP/submit.py --run
        echo "Logs: scripts/$EXP/sbatch_logs/HACplus_$EXP/   (finished runs are skipped on resubmit)"
        ;;
    collect)
        if [ "${2:-}" = smoke ]; then
            python3 scripts/$EXP/collect.py outputs/${EXP}_smoke
        else
            python3 scripts/$EXP/collect.py
        fi
        ;;
    *)
        echo "usage: bash scripts/$EXP/short_submit.sh {smoke|all|collect [smoke]}"
        exit 1
        ;;
esac
