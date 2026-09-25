#!/bin/bash
# Fetch the Nasdaq TotalView-ITCH 5.0 sample day used by the intraday anchor and extract the
# transaction prices (code/itch_trades.py). The ITCH file (3.5 GB) is not redistributed with the
# code: it is Nasdaq's sample data, published for developers at emi.nasdaq.com; each user fetches
# it from there. The extracted executions and the windows built from them are derived from it and
# are not shipped either; results/intraday_windows_fingerprint.json lets a rebuild be checked:
#   python code/itch_trades.py data/intraday/12302019.NASDAQ_ITCH50.gz --symbols ALL \
#          --out data/intraday/itch_trades_2019-12-30.npz
#   python code/intraday_data.py --trades data/intraday/itch_trades_2019-12-30.npz \
#          --out results/intraday_windows.npz --expect results/intraday_windows_fingerprint.json
# Usage: bash data/intraday/download_intraday.sh      (needs curl; 3.5 GB; resumable)
set -euo pipefail
cd "$(dirname "$0")"
URL="https://emi.nasdaq.com/ITCH/Nasdaq%20ITCH/12302019.NASDAQ_ITCH50.gz"
F=12302019.NASDAQ_ITCH50.gz
if [ -s "$F" ]; then echo "have $F (resuming if incomplete)"; fi
curl -fL -C - --retry 20 --retry-delay 5 -o "$F" "$URL"
if command -v md5sum >/dev/null 2>&1; then SUM="md5sum"; else SUM="md5 -r"; fi
echo "md5 of the paper's snapshot: $(cat CHECKSUMS.md5 2>/dev/null || echo 'see CHECKSUMS.md5')"
echo "md5 of this download:        $($SUM "$F")"
