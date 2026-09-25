#!/bin/bash
# The intraday anchor: every model forecasts the transaction-price windows (raw + K sign-randomised copies),
# their multiplicative mirrors and the mirrors of the copies. Needs results/intraday_windows.npz first
# (data/intraday/download_intraday.sh, code/itch_trades.py, code/intraday_data.py). Each model runs in its own env.
set -x
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
for pair in "env_timesfm.sh timesfm" "env_timemoe.sh timemoe" "env_sundial.sh sundial" "env_fincast.sh fincast" "env_moirai.sh moirai2" "env_moirai.sh moirai" "env_tirex.sh tirex" "env_tsfm25.sh timesfm25" "env_chronos.sh chronosbolt" "env_chronos.sh chronos2" "env_chronos.sh chronos"; do
  set -- $pair
  for flag in "" "--mirror" "--mirror-sur"; do
    ( source "$ROOT/scripts/$1" && export REAL_WINDOWS="$ROOT/results/intraday_windows.npz" && cd "$ROOT/code" && $PY real_probe.py --model $2 --prefix intraday --batch 32 $flag )
    echo "=== $2 intraday $flag rc=$? ==="
  done
done
echo "=== run_intraday done ==="
