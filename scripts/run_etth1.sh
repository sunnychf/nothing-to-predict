#!/bin/bash
# The ETTh1 positive control (Appendix, tab:etth1): every model forecasts the 924 ETTh1 windows and their additive
# mirrors, each in its own env. Needs data/etth1/ETTh1.csv (data/etth1/download_etth1.sh). Minutes per model.
set -x
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
for pair in "env_timesfm.sh timesfm" "env_timemoe.sh timemoe" "env_sundial.sh sundial" "env_fincast.sh fincast" "env_moirai.sh moirai2" "env_moirai.sh moirai" "env_tirex.sh tirex" "env_tsfm25.sh timesfm25" "env_chronos.sh chronosbolt" "env_chronos.sh chronos2" "env_chronos.sh chronos"; do
  set -- $pair
  ( source "$ROOT/scripts/$1" && export SPEC_OUT="$ROOT/results" && cd "$ROOT/code" && $PY etth1_probe.py --model $2 --csv "$ROOT/data/etth1/ETTh1.csv" --batch 32 ); echo "=== $2 etth1 rc=$? ==="
done
echo "=== run_etth1 done ==="
