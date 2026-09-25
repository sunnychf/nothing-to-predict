#!/bin/bash
# Time-MoE-200M: ladder and direction at H=128 (greedy multi-horizon decode; no frequency input exists).
set -x
source "$(dirname "${BASH_SOURCE[0]}")/env_timemoe.sh"
cd "$ROOT/code" || exit 1
$PY pilot_a_timemoe.py --stage all --n 512 --seeds 3 --n-dir 128 --horizon-long 128 --batch 64
echo "=== run_timemoe done rc=$? ==="
