#!/bin/bash
# Validate the Time-MoE decoding loop on the model's own ETTh1 zero-shot benchmark (validate_timemoe.py; needs data/etth1/ETTh1.csv,
# fetched by data/etth1/download_etth1.sh). Hours to a day on one card: the loop has no KV cache and the 720-step horizon decodes
# 3072-point contexts in 13 calls per window (heads 1/8/32/64: eleven of 64, two of 8). One horizon per process (--horizons 720 --tag _h720) runs the four in parallel, and
# --official-only skips the second decoding (per-window normalisation); make_paper_tables.py merges validate_timemoe_etth1*.json.
set -x
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
( source "$ROOT/scripts/env_timemoe.sh" && cd "$ROOT/code" && $PY validate_timemoe.py --csv "$ROOT/data/etth1/ETTh1.csv" --batch 32 ); echo "=== validate rc=$? ==="
