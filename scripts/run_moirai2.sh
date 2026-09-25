#!/bin/bash
# Moirai-2.0-R-small (decoder-only successor of Moirai-1.1), in the `moirai` env (uni2ts 2.0 ships moirai2):
# smoke test, ladder + direction, the correction ladder, the real-data anchor (four passes) and the context sweep.
# Needs results/real_windows.npz first.
set -x
source "$(dirname "${BASH_SOURCE[0]}")/env_moirai.sh"
cd "$ROOT/code" || exit 1
$PY smoke_moirai2.py
$PY pilot_a_moirai2.py --stage all
$PY remedy_ladder.py --model moirai2
for flag in "" "--mirror" "--mirror-sur"; do $PY real_probe.py --model moirai2 $flag; done
$PY diag_context.py --model moirai2 --lengths 128,256,512,1024,2048
echo "=== run_moirai2 done rc=$? ==="
