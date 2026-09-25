#!/bin/bash
# usage: run_falling.sh <python> <threads> <model> [<model> ...]   (raw + one sign copy, then the mirror pass)
PYB=$1; TH=$2; shift 2
cd $HOME/tsfm_rev/code
export TSFM_DEVICE=cpu TSFM_BACKEND=cpu HF_ENDPOINT=https://huggingface.co HF_HOME=$HOME/tsfm_rev/hf OMP_NUM_THREADS=$TH MKL_NUM_THREADS=$TH
export REAL_WINDOWS=$HOME/tsfm_rev/data/falling_windows.npz REAL_OUT=$HOME/tsfm_rev/results_falling
export PYTHONPATH=$HOME/tsfm_rev/code:$HOME/tsfm_rev/FinCast-fts/src
for m in "$@"; do
  [ -f $REAL_OUT/falling_${m}.npz ] || $PYB real_probe.py --model $m --prefix falling --batch 32 > $HOME/tsfm_rev/logs/falling_${m}.log 2>&1
  [ -f $REAL_OUT/falling_${m}_mirror.npz ] || $PYB real_probe.py --model $m --prefix falling --batch 32 --mirror > $HOME/tsfm_rev/logs/falling_${m}_mirror.log 2>&1
done
echo END >> $HOME/tsfm_rev/logs/falling_runner_$1.log
