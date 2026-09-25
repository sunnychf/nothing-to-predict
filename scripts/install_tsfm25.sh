#!/bin/bash
# conda env `tsfm25`: TimesFM-2.5-200M through timesfm 3.x (3.0.2 in the paper's run). A CUDA torch is installed
# FIRST (2.5.1+cu121) so that `timesfm[torch]` keeps it rather than pulling a build the machine cannot run.
set -ex
CONDA="${CONDA:-$HOME/anaconda3/bin/conda}"
CONDA_ENVS="${CONDA_ENVS:-$HOME/anaconda3/envs}"
PY="$CONDA_ENVS/tsfm25/bin/python"
PIP="$CONDA_ENVS/tsfm25/bin/pip install --no-cache-dir"
[ -x "$PY" ] || $CONDA create -p "$CONDA_ENVS/tsfm25" python=3.11 -y
$PY -c 'import torch; assert torch.cuda.is_available()' 2>/dev/null || $PIP torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
$PY -c 'import timesfm' 2>/dev/null || $PIP "timesfm[torch]"
$PY -c 'import torch; assert torch.cuda.is_available(), "timesfm replaced torch with a build this machine cannot run"'
$PIP numpy pandas
$PY - <<'PYEOF'
from huggingface_hub import snapshot_download
print("weights:", snapshot_download("google/timesfm-2.5-200m-pytorch"))
import torch, timesfm
timesfm.TimesFM_2p5_200M_torch; timesfm.ForecastConfig
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "timesfm 2.5 API OK")
PYEOF
echo "=== install_tsfm25 done ==="
