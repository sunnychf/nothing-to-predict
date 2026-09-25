#!/bin/bash
# FinCast: ladder, declared-frequency test, direction at H=128, direction under the three frequency labels.
set -x
source "$(dirname "${BASH_SOURCE[0]}")/env_fincast.sh"
cd "$ROOT/code" || exit 1
$PY pilot_a_fincast.py --stage all --n 512 --seeds 3 --freq 0 --n-freq 128 --horizon-long 128 --batch 32
$PY diag_direction_fincast.py
$PY diag_freq_direction_fincast.py
echo "=== run_fincast done rc=$? ==="
