#!/bin/bash
# Chronos-Bolt-small and Chronos-2: ladder + direction, then the correction ladder and the real-data
# anchor (raw windows + surrogate copies, mirrors of the raw windows, mirrors of the copies).
# Needs results/real_windows.npz (scripts/run_analysis.sh step 2, or code/real_data.py) first.
set -x
source "$(dirname "${BASH_SOURCE[0]}")/env_chronos.sh"
cd "$ROOT/code" || exit 1
for pair in "bolt chronosbolt" "chronos2 chronos2"; do
  set -- $pair
  $PY pilot_a_chronosx.py --model $1 --stage all
  $PY remedy_ladder.py --model $2
  for flag in "" "--mirror" "--mirror-sur"; do $PY real_probe.py --model $2 $flag; done
done
echo "=== run_chronosx done rc=$? ==="
