#!/bin/bash
# Fetch ETTh1 (Zhou et al., 2021; github.com/zhouhaoyi/ETDataset) for the Time-MoE harness validation
# (code/validate_timemoe.py). Not redistributed with the code; the snapshot's sha256 is in CHECKSUMS.sha256.
set -euo pipefail
cd "$(dirname "$0")"
[ -s ETTh1.csv ] || curl -fL --retry 5 -o ETTh1.csv "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv"
if command -v sha256sum >/dev/null 2>&1; then sha256sum -c CHECKSUMS.sha256; else shasum -a 256 -c CHECKSUMS.sha256; fi
