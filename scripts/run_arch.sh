#!/bin/bash
# Controlled architecture comparison (Appendix, tab:arch): two architectures trained from scratch on one synthetic
# corpus (arch_corpus.py), 2 architectures x 2 corpora x 3 seeds, ~8 min each on one 48 GB card (12 runs; run in
# parallel by giving each its own CUDA_VISIBLE_DEVICES). Needs torch; any of the model envs works.
set -x
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/env_timemoe.sh"
export SPEC_OUT="$ROOT/results"
for seed in 0 1 2; do for corpus in up sym; do for arch in encoder decoder; do
  ( cd "$ROOT/code" && $PY arch_train.py --arch $arch --corpus $corpus --seed $seed ); echo "=== $arch $corpus s$seed rc=$? ==="
done; done; done
