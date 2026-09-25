#!/bin/bash
# CPU side, in the `tsfmfin` env or any Python 3.11 with numpy, pandas, matplotlib (torch only for test_decompose):
#  1. the unit tests and the numerical verification of the analytical claims (Appendix: verify_theory)
#  2. the real-data windows from the downloaded sources (identical to the paper's, checked by fingerprint)
#  3. every table and figure of the paper from the result files
# Steps 2-3 run on the shipped result files without any GPU work.
set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-python}"
cd "$ROOT"
$PY code/test_nulls.py
if $PY -c "import torch" 2>/dev/null; then $PY code/test_decompose.py; else echo "test_decompose.py skipped (needs torch; it runs in scripts/run_chronos.sh)"; fi
$PY code/verify_theory.py
[ -s data/real/eurofxref-hist.csv ] || bash data/real/download_real_data.sh
$PY code/real_data.py --data data/real --out results/real_windows.npz --k 0 --n-sur 4 --expect results/real_windows_fingerprint.json
$PY code/n4_bayes.py results                                       # results/n4_optima.json: N4's exact optimum (conditional mean) against its optimal linear forecast
$PY code/make_paper_tables.py results paper/tables                 # Tables 2-4, 6-10 (timemoe_check, ladder, ladder_cx, direction, scale, context, level_all, coverage) + facts.json
$PY code/real_summary.py   --res results --tables paper/tables     # Table 11 (+ real.tex, not in the paper) + results/real_summary.json
$PY code/remedy_summary.py --res results --tables paper/tables     # Tables 1, 12-15 (+ remedy.tex, not in the paper) + results/remedy_summary.json
if [ -s results/intraday_windows.npz ]; then $PY code/intraday_summary.py --res results --tables paper/tables; else echo "intraday_summary.py skipped: results/intraday_windows.npz not built (data/intraday/download_intraday.sh, then itch_trades.py + intraday_data.py; the 3.5 GB ITCH day is not shipped)"; fi
if [ -s results/etth1_windows.npz ]; then $PY code/etth1_summary.py --res results --tables paper/tables; fi           # Table 16 + results/etth1_summary.json (from run_etth1.sh)
if [ -s results/smallcap_windows.npz ]; then $PY code/smallcap_summary.py --res results --tables paper/tables; else echo "smallcap_summary.py skipped: build results/smallcap_windows.npz with code/smallcap_data.py after data/real/download_real_data.sh"; fi   # Table 18
if ls results/arch_*_s*.json >/dev/null 2>&1; then $PY code/arch_summary.py --res results --tables paper/tables; fi   # Table 5 + results/arch_summary.json (from run_arch.sh)
$PY code/make_paper_figures.py --res results --out paper/figures   # Figures 1-7 + figures_facts.json
echo "=== run_analysis done ==="
