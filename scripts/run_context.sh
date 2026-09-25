#!/bin/bash
# Context-length sweep (Section 6.3): the same 128 zero-drift random walks forecast from nested
# suffixes of length 128 .. 2048; does the prior shrink as the model sees more evidence of no drift?
# Chronos-T5 reads at most 512 points; FinCast's API fixes the context at load time and is run at
# the lengths its loader accepts.
set -x
D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for pair in "env_chronos.sh chronos 128,256,512" "env_chronos.sh chronosbolt 128,256,512,1024,2048" "env_chronos.sh chronos2 128,256,512,1024,2048" \
            "env_timesfm.sh timesfm 128,256,512,1024,2048" "env_timemoe.sh timemoe 128,256,512,1024,2048" "env_moirai.sh moirai 128,256,512,1024,2048" \
            "env_fincast.sh fincast 128,256,512" "env_tirex.sh tirex 128,256,512,1024,2048" \
            "env_moirai.sh moirai2 128,256,512,1024,2048" "env_tsfm25.sh timesfm25 128,256,512,1024,2048"; do
  set -- $pair
  ( source "$D/$1" && cd "$ROOT/code" && $PY diag_context.py --model $2 --lengths $3 ); echo "=== $2 context rc=$? ==="
done
echo "=== run_context done ==="
