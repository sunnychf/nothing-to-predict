#!/bin/bash
# Moirai-1.1-R-small: ladder (S=100), the patch-size (declaration) sweep, direction at H=128.
set -x
source "$(dirname "${BASH_SOURCE[0]}")/env_moirai.sh"
cd "$ROOT/code" || exit 1
$PY pilot_a_moirai.py --stage all --n 512 --seeds 3 --num-samples 100 --n-freq 128 --horizon-long 128 --patch auto --batch 16
$PY diag_direction_moirai.py
echo "=== run_moirai done rc=$? ==="
