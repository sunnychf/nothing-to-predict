#!/bin/bash
# TimesFM-2.5-200M: smoke test, ladder + direction with the released defaults, the direction stage with the
# built-in flip-invariance averaging off, the correction ladder, the real-data anchor (four passes) and the
# context sweep. Needs results/real_windows.npz first.
set -x
source "$(dirname "${BASH_SOURCE[0]}")/env_tsfm25.sh"
cd "$ROOT/code" || exit 1
$PY smoke_timesfm25.py
$PY pilot_a_timesfm25.py --stage all
$PY pilot_a_timesfm25.py --stage direction --flip off
$PY remedy_ladder.py --model timesfm25
for flag in "" "--mirror" "--mirror-sur"; do $PY real_probe.py --model timesfm25 $flag; done
$PY diag_context.py --model timesfm25 --lengths 128,256,512,1024,2048
echo "=== run_tsfm25 done rc=$? ==="
