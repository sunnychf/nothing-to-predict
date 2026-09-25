#!/bin/bash
# Fetch the two public sources of the real-data anchor into this directory.
#
# The files are not redistributed with the code: ECB statistical data may be reproduced with the
# source acknowledged, and the Kenneth R. French data library is provided for research use with
# acknowledgement of the source, so each user downloads them from the providers. Both providers
# extend their files over time. The paper used the snapshot of 2026-09-16 (ECB file ending
# 2026-09-15, Ken French files ending 2026-07-31), whose checksums are in CHECKSUMS.sha256;
# code/real_data.py cuts a later download at those last dates (--end-fx / --end-eq defaults), and
#   python code/real_data.py --data data/real --out results/real_windows.npz --k 0 --n-sur 4 \
#          --expect results/real_windows_fingerprint.json
# then reports whether the rebuilt windows are the paper's (exact / float-noise / mismatch).
#
# Usage: bash data/real/download_real_data.sh        (needs curl and unzip; ~6 MB of downloads)
set -euo pipefail
cd "$(dirname "$0")"
ECB=https://www.ecb.europa.eu/stats/eurofxref/eurofxref-hist.zip
KF=https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp
get() {                                  # get URL FILE: skip files already present
  if [ -s "$2" ]; then echo "have $2"; else echo "fetching $2"; curl -fL --retry 5 --retry-delay 5 -o "$2" "$1"; fi
}
get "$ECB" eurofxref-hist.zip
get "$KF/49_Industry_Portfolios_daily_CSV.zip" 49_Industry_Portfolios_daily_CSV.zip
get "$KF/F-F_Research_Data_Factors_daily_CSV.zip" F-F_Research_Data_Factors_daily_CSV.zip
get "$KF/Portfolios_Formed_on_ME_Daily_CSV.zip" Portfolios_Formed_on_ME_Daily_CSV.zip      # the small-cap anchor (downloaded 2026-09-19)
for z in eurofxref-hist.zip 49_Industry_Portfolios_daily_CSV.zip F-F_Research_Data_Factors_daily_CSV.zip Portfolios_Formed_on_ME_Daily_CSV.zip; do
  unzip -o -q "$z"                       # eurofxref-hist.csv, 49_Industry_Portfolios_Daily.csv, F-F_Research_Data_Factors_daily.csv
done
ls -l eurofxref-hist.csv 49_Industry_Portfolios_Daily.csv F-F_Research_Data_Factors_daily.csv Portfolios_Formed_on_ME_Daily.csv
if command -v sha256sum >/dev/null 2>&1; then SUM="sha256sum"; else SUM="shasum -a 256"; fi
if $SUM -c CHECKSUMS.sha256 >/dev/null 2>&1; then
  echo "checksums: identical to the paper's snapshot of 2026-09-16"
else
  echo "checksums: differ from the paper's snapshot of 2026-09-16 (expected once the providers have extended the files);"
  echo "  real_data.py cuts at the snapshot's last dates; confirm the windows with --expect results/real_windows_fingerprint.json"
  $SUM -c CHECKSUMS.sha256 || true
fi
