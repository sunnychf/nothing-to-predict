#!/bin/bash
# The symmetry-averaged correction (Section 5) and the real-data anchor (Section 6) for the five
# 2024-generation models, each in its own environment: (1) the ladder with antithetic contexts and
# K=4 sign copies (remedy_ladder.py), (2) the 1280 real windows and their K=4 surrogates
# (real_probe.py), (3) the mirror of every raw window (--mirror), (4) the mirror of every surrogate
# copy (--mirror-sur). Chronos last: it is the slow one (about 1.2 s per window at S=100, H=128).
# Needs results/real_windows.npz first (data/real/download_real_data.sh, then code/real_data.py).
set -x
D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for pair in "env_timesfm.sh timesfm" "env_timemoe.sh timemoe" "env_fincast.sh fincast" "env_moirai.sh moirai" "env_chronos.sh chronos"; do
  set -- $pair
  ( source "$D/$1" && cd "$ROOT/code" && $PY remedy_ladder.py --model $2 ); echo "=== $2 ladder rc=$? ==="
  for flag in "" "--mirror" "--mirror-sur"; do
    ( source "$D/$1" && cd "$ROOT/code" && $PY real_probe.py --model $2 $flag ); echo "=== $2 real $flag rc=$? ==="
  done
done
echo "=== run_remedy done ==="
