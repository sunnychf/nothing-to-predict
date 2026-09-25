#!/bin/bash
# conda env `timesfm`: TimesFM-2.0-500m through timesfm==1.3.0, the last package version that still
# exposes the frequency indicator (2.5 removed it). A CUDA torch is installed FIRST: `timesfm[torch]`
# would otherwise pull the newest torch, which may need a newer driver than the machine has.
set -ex
CONDA="${CONDA:-$HOME/anaconda3/bin/conda}"
CONDA_ENVS="${CONDA_ENVS:-$HOME/anaconda3/envs}"
PY="$CONDA_ENVS/timesfm/bin/python"
PIP="$CONDA_ENVS/timesfm/bin/pip install --no-cache-dir"
[ -x "$PY" ] || $CONDA create -p "$CONDA_ENVS/timesfm" python=3.11 -y
$PY -c 'import torch; assert torch.cuda.is_available()' 2>/dev/null || $PIP torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
$PY -c 'import timesfm' 2>/dev/null || $PIP "timesfm[torch]==1.3.0"
$PY -c 'import torch; assert torch.cuda.is_available(), "the timesfm install replaced torch with a build this machine cannot run"'
$PY - <<'PYEOF'
from huggingface_hub import snapshot_download
print("weights:", snapshot_download("google/timesfm-2.0-500m-pytorch"))
import torch, timesfm
timesfm.TimesFm; timesfm.TimesFmHparams; timesfm.TimesFmCheckpoint
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "timesfm API OK")
PYEOF
echo "=== install_timesfm done ==="
