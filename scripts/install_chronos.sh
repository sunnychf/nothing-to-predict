#!/bin/bash
# conda env `tsfmfin`: Chronos-T5 (tiny/mini/small/base/large), Chronos-Bolt-small, Chronos-2, and the
# CPU-side analysis (numpy, pandas, scipy, matplotlib). Versions are the ones the paper's runs used
# (torch 2.5.1+cu121, chronos-forecasting 2.3.2, transformers 5.17.0, numpy 2.4.6). Idempotent.
set -ex
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA="${CONDA:-$HOME/anaconda3/bin/conda}"
CONDA_ENVS="${CONDA_ENVS:-$HOME/anaconda3/envs}"
PY="$CONDA_ENVS/tsfmfin/bin/python"
PIP="$CONDA_ENVS/tsfmfin/bin/pip install --no-cache-dir"
[ -x "$PY" ] || $CONDA create -p "$CONDA_ENVS/tsfmfin" python=3.11 -y
$PY -c 'import torch' 2>/dev/null || $PIP torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
$PY -c 'import chronos' 2>/dev/null || $PIP chronos-forecasting==2.3.2 transformers==5.17.0
$PIP numpy==2.4.6 pandas scipy matplotlib
$PY - <<'PYEOF'
import torch, numpy, scipy, transformers, chronos
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "gpus", torch.cuda.device_count())
print("numpy", numpy.__version__, "scipy", scipy.__version__, "transformers", transformers.__version__, "chronos", getattr(chronos, "__version__", "?"))
PYEOF
# weights (amazon/chronos-t5-*, amazon/chronos-bolt-small, amazon/chronos-2) are fetched from the Hugging
# Face hub on first use by the drivers; nothing to download here.
echo "=== install_chronos done ==="
