#!/bin/bash
# Chronos-T5-small: the ladder and its controls, the spectral decomposition, the declared-frequency
# record, the random-init reference, and the diagnostics of Sections 4.1 and 6 (dispersion, N2
# intervals, direction controls, price-level sweep, the five-size scale sweep). Every stage appends
# to results/pilot_a.jsonl or its own file and skips configurations already present, so the script
# can be re-run after an interruption. Runs the paper's settings (n=512 contexts x 3 seeds per rung,
# S=100 sample paths, H=16 main horizon, H=128/256 long horizons).
set -x
source "$(dirname "${BASH_SOURCE[0]}")/env_chronos.sh"
cd "$ROOT/code" || exit 1
M=amazon/chronos-t5-small
$PY test_nulls.py     || exit 1        # the generators must pass their tests before producing a number
$PY test_decompose.py || exit 1        # so must the spectral separator
$PY pilot_a.py --stage floor    --model $M --init pretrained --num-samples 100 --n-floor 256
$PY pilot_a.py --stage ladder   --model $M --init pretrained --num-samples 100 --n 512 --seeds 3
$PY pilot_a.py --stage spectral --model $M --init pretrained --num-samples 100 --n-spec 128
$PY pilot_a.py --stage freq     --model $M --init pretrained
$PY pilot_a.py --stage ladder   --model $M --init random     --num-samples 100 --n 256 --seeds 1   # architecture reference
$PY diag_dispersion.py $M
SPEC_H=128 $PY spectral_deep.py $M     # departures_{N1,N2,N4}_H128.npz + spectral_deep_H128.json
SPEC_H=256 $PY spectral_deep.py $M
$PY diag_n2_interval.py $M
$PY diag_coverage_chronos.py $M               # 80 percent interval coverage on the ladder (Table 7)
$PY diag_direction_control.py pos neg zero mirror    # departures_ctrl_*_H128.npz, diag_direction_control.json
$PY diag_direction_level.py                          # seven price levels
$PY diag_scale_chronos.py                            # tiny .. large (base and large are ~0.8 / 3.6 GB downloads)
echo "=== run_chronos done rc=$? ==="
