#!/bin/bash
# Sundial-base-128M: smoke test, ladder + direction (S=100 sample trajectories, point forecast the sample
# mean), the correction ladder, the real-data anchor (four passes) and the context sweep.
# Needs results/real_windows.npz first.
set -x
source "$(dirname "${BASH_SOURCE[0]}")/env_sundial.sh"
cd "$ROOT/code" || exit 1
$PY smoke_sundial.py
$PY pilot_a_sundial.py --stage all
$PY remedy_ladder.py --model sundial --batch 32
for flag in "" "--mirror" "--mirror-sur"; do $PY real_probe.py --model sundial --batch 32 $flag; done
$PY diag_context.py --model sundial --lengths 128,256,512,1024,2048 --batch 32
echo "=== run_sundial done rc=$? ==="
