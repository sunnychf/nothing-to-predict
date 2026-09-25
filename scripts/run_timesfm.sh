#!/bin/bash
# TimesFM-2.0-500m: ladder, declared-frequency test, direction at H=128 (one load serves every horizon).
set -x
source "$(dirname "${BASH_SOURCE[0]}")/env_timesfm.sh"
cd "$ROOT/code" || exit 1
$PY pilot_a_timesfm.py --stage all --n 512 --seeds 3 --freq 0 --n-freq 128 --horizon-long 128 --batch 64
echo "=== run_timesfm done rc=$? ==="
