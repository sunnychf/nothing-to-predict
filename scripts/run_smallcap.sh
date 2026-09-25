#!/bin/bash
# The small-cap anchor (Appendix, tab:smallcap): every model forecasts the size-decile windows (raw + K sign-randomised
# copies), their multiplicative mirrors and the mirrors of the copies, each in its own env. Needs results/smallcap_windows.npz
# (data/real/download_real_data.sh, then code/smallcap_data.py --data data/real --out results/smallcap_windows.npz --n-sur 4).
set -x
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
for pair in "env_timesfm.sh timesfm" "env_timemoe.sh timemoe" "env_sundial.sh sundial" "env_fincast.sh fincast" "env_moirai.sh moirai2" "env_moirai.sh moirai" "env_tirex.sh tirex" "env_tsfm25.sh timesfm25" "env_chronos.sh chronosbolt" "env_chronos.sh chronos2" "env_chronos.sh chronos"; do
  set -- $pair
  for flag in "" "--mirror" "--mirror-sur"; do
    ( source "$ROOT/scripts/$1" && export REAL_WINDOWS="$ROOT/results/smallcap_windows.npz" && cd "$ROOT/code" && $PY real_probe.py --model $2 --prefix smallcap --batch 32 $flag ); echo "=== $2 smallcap $flag rc=$? ==="
  done
done
echo "=== run_smallcap done ==="
