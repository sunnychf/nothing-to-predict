#!/bin/bash
# conda env `tirex`: TiRex (NX-AI, 2025) through the tirex-ts package. A CUDA torch is installed first
# (torch 2.5.1+cu121 in the paper's run) and tirex-ts without its [cuda] extra, so the sLSTM layers run
# in the package's PyTorch implementation (the custom kernels would need an nvcc matching the torch build).
set -ex
CONDA="${CONDA:-$HOME/anaconda3/bin/conda}"
CONDA_ENVS="${CONDA_ENVS:-$HOME/anaconda3/envs}"
PY="$CONDA_ENVS/tirex/bin/python"
PIP="$CONDA_ENVS/tirex/bin/pip install --no-cache-dir"
[ -x "$PY" ] || $CONDA create -p "$CONDA_ENVS/tirex" python=3.11 -y
$PY -c 'import torch; assert torch.cuda.is_available()' 2>/dev/null || $PIP torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
$PY -c 'import tirex' 2>/dev/null || $PIP tirex-ts
$PY -c 'import torch; assert torch.cuda.is_available(), "tirex-ts replaced torch with a build this machine cannot run"'
$PIP numpy pandas
$PY - <<'PYEOF'
from huggingface_hub import snapshot_download
print("weights:", snapshot_download("NX-AI/TiRex"))
import torch, tirex
from tirex import load_model
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "tirex import OK")
PYEOF
echo "=== install_tirex done ==="
