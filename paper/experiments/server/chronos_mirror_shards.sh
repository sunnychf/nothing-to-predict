#!/bin/bash
# Chronos-T5 falling mirror pass in three shards (16 threads each) while the arch queue is paused;
# resumes the arch processes and merges the parts when all three shards are written.
cd $HOME/tsfm_rev/code
export TSFM_DEVICE=cpu HF_ENDPOINT=https://huggingface.co HF_HOME=$HOME/tsfm_rev/hf OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
export PYTHONPATH=$HOME/tsfm_rev/code
W=$HOME/tsfm_rev/data/falling_windows.npz; O=$HOME/tsfm_rev/results_falling; PY=$HOME/tsfm_rev/env/bin/python
ARCH_PIDS="$*"
for r in "0 617" "617 1234" "1234 1850"; do
  set -- $r
  $PY shard_mirror.py --model chronos --windows $W --out-dir $O --lo $1 --hi $2 > $HOME/tsfm_rev/logs/falling_chronos_mirror_part$1.log 2>&1 &
done
wait
$PY shard_mirror.py --model chronos --windows $W --out-dir $O --merge >> $HOME/tsfm_rev/logs/falling_chronos_mirror_merge.log 2>&1
[ -n "$ARCH_PIDS" ] && kill -CONT $ARCH_PIDS
echo "END $(date)" >> $HOME/tsfm_rev/logs/falling_chronos_mirror_merge.log
