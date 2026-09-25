#!/bin/bash
# Level / shape decoupling (diag_level_all.py): the same 128 increments at five raw levels, every model in its env (Table 7).
set -x
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
for pair in "env_timesfm.sh timesfm" "env_timemoe.sh timemoe" "env_sundial.sh sundial" "env_fincast.sh fincast" "env_moirai.sh moirai2" "env_moirai.sh moirai" "env_tirex.sh tirex" "env_tsfm25.sh timesfm25" "env_chronos.sh chronosbolt" "env_chronos.sh chronos2" "env_chronos.sh chronos"; do
  set -- $pair
  ( source "$ROOT/scripts/$1" && cd "$ROOT/code" && $PY diag_level_all.py --model $2 ); echo "=== $2 level rc=$? ==="
done
echo "=== run_level_all done ==="
