#!/bin/bash
# conda env `moirai`: Moirai-1.1-R-small through uni2ts (torch 2.4.1+cu121, gluonts 0.14.4, python 3.10
# in the paper's runs). uni2ts may resolve a CPU-only torch; a CUDA build of the same version is
# installed afterwards if so. Idempotent.
set -ex
CONDA="${CONDA:-$HOME/anaconda3/bin/conda}"
CONDA_ENVS="${CONDA_ENVS:-$HOME/anaconda3/envs}"
PY="$CONDA_ENVS/moirai/bin/python"
PIP="$CONDA_ENVS/moirai/bin/pip install --no-cache-dir"
[ -x "$PY" ] || $CONDA create -p "$CONDA_ENVS/moirai" python=3.10 -y
$PY -c 'import uni2ts' 2>/dev/null || $PIP uni2ts
$PY -c 'import torch; assert torch.cuda.is_available()' 2>/dev/null || {
  TV=$($PY -c 'import torch; print(torch.__version__.split("+")[0])')
  $PIP "torch==$TV" --index-url https://download.pytorch.org/whl/cu121
}
$PY - <<'PYEOF'
from huggingface_hub import snapshot_download
print("weights:", snapshot_download("Salesforce/moirai-1.1-R-small"))
import torch, uni2ts, gluonts
from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "gluonts", gluonts.__version__, "moirai import OK")
PYEOF
echo "=== install_moirai done ==="
