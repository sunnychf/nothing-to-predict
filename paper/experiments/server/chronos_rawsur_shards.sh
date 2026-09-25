#!/bin/bash
# Chronos-T5 falling raw and sign-randomised passes in six shards (8 threads each); merges when all are written.
cd $HOME/tsfm_rev/code
export TSFM_DEVICE=cpu HF_ENDPOINT=https://huggingface.co HF_HOME=$HOME/tsfm_rev/hf OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export PYTHONPATH=$HOME/tsfm_rev/code
W=$HOME/tsfm_rev/data/falling_windows.npz; O=$HOME/tsfm_rev/results_falling; PY=$HOME/tsfm_rev/env/bin/python
for part in raw sur; do
  for r in "0 617" "617 1234" "1234 1850"; do
    set -- $r
    $PY shard_probe.py --model chronos --windows $W --out-dir $O --part $part --lo $1 --hi $2 > $HOME/tsfm_rev/logs/falling_chronos_${part}_part$1.log 2>&1 &
  done
done
wait
$PY shard_probe.py --model chronos --windows $W --out-dir $O --merge >> $HOME/tsfm_rev/logs/falling_chronos_rawsur_merge.log 2>&1
echo "END $(date)" >> $HOME/tsfm_rev/logs/falling_chronos_rawsur_merge.log
