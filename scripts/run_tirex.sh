#!/bin/bash
# TiRex: ladder + direction, the correction ladder, the real-data anchor (raw windows + surrogate copies,
# mirrors of the raw windows, mirrors of the copies) and the context-length sweep.
# Needs results/real_windows.npz (see "Data" in the README) first.
set -x
source "$(dirname "${BASH_SOURCE[0]}")/env_tirex.sh"
cd "$ROOT/code" || exit 1
$PY smoke_tirex.py
$PY pilot_a_tirex.py --stage all
$PY remedy_ladder.py --model tirex
for flag in "" "--mirror" "--mirror-sur"; do $PY real_probe.py --model tirex $flag; done
$PY diag_context.py --model tirex --lengths 128,256,512,1024,2048
echo "=== run_tirex done rc=$? ==="
