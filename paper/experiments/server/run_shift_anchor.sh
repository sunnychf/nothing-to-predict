#!/bin/bash
# Calendar-shift check on the paper's anchors (experiments/PREREGISTRATION_shift_anchors.md): raw h = 128 forecasts of
# the shifted windows (shift_anchor_windows.npz, shift_size_windows.npz) with the paper's adapters on one GPU, chosen by UUID.
# usage: bash run_shift_anchor.sh <GPU-UUID> <tag> <model> [<model> ...]      (resumable: finished parts are skipped)
G=$1; TAG=$2; shift 2
W=<workdir>
export TMPDIR=$W/tmp; mkdir -p $TMPDIR $W/logs $W/out
export HF_ENDPOINT=https://huggingface.co HF_HUB_ENABLE_HF_TRANSFER=0 PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=$G PYTHONPATH=<repo>/code
cd <repo>/code
for m in "$@"; do
  case $m in
    chronosbolt|chronos2) PY=<envs>/tsfmfin/bin/python ;;
    tirex) PY=<envs>/tirex/bin/python ;;
    moirai|moirai2) PY=<envs>/moirai/bin/python ;;
    timesfm) PY=<envs>/timesfm/bin/python ;;
    *) echo "unknown model $m" >> $W/logs/runner_$TAG.log; continue ;;
  esac
  for pair in "anchor 6260" "size 1192"; do
    set -- $pair
    f=$W/out/shards/shift_$1_${m}_raw_part0_$2.npz
    if [ ! -f $f ]; then
      echo "$(date '+%F %T') start $m $1" >> $W/logs/runner_$TAG.log
      $PY $W/shard_probe.py --model $m --windows $W/data/shift_$1_windows.npz --out-dir $W/out --prefix shift_$1 --part raw --lo 0 --hi $2 --batch 32 > $W/logs/shift_$1_$m.log 2>&1
      echo "$(date '+%F %T') done $m $1 rc=$?" >> $W/logs/runner_$TAG.log
    fi
  done
done
echo "END $(date '+%F %T')" >> $W/logs/runner_$TAG.log
