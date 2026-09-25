#!/bin/bash
cd $HOME/tsfm_rev
for run in "$@"; do
  until [ -f runs/$run/train.json ]; do sleep 20; done
  TSFM_DEVICE=cpu HF_ENDPOINT=https://huggingface.co HF_HOME=$HOME/tsfm_rev/hf env/bin/python code/eval_bolt.py --tag $run --ckpt runs/$run/model.pt --threads 32 > logs/eval_$run.log 2>&1
done
