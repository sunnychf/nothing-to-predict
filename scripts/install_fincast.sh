#!/bin/bash
# conda env `fincast_v1`: FinCast (Zhu et al. 2025), from the authors' repository, with its own torch
# (2.5.0+cu124) and the 3.97 GB v1 checkpoint. Time-MoE runs in this env too (transformers 4.53).
# Idempotent; the checkpoint download resumes (curl -C -) because large transfers may drop.
set -ex
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA="${CONDA:-$HOME/anaconda3/bin/conda}"
CONDA_ENVS="${CONDA_ENVS:-$HOME/anaconda3/envs}"
PY="$CONDA_ENVS/fincast_v1/bin/python"
PIP="$CONDA_ENVS/fincast_v1/bin/pip install --no-cache-dir"
REPO="$ROOT/FinCast-fts"
HUB="${HF_ENDPOINT:-https://huggingface.co}"
[ -d "$REPO" ] || git clone --depth 1 https://github.com/vincent05r/FinCast-fts.git "$REPO"
[ -x "$PY" ] || $CONDA create -p "$CONDA_ENVS/fincast_v1" python=3.11.11 -y
cd "$REPO"
$PY -c 'import ffm' 2>/dev/null || {
  $PIP -r requirement_v2.txt
  $PIP -e .
  $PIP torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu124
}
W="$ROOT/weights/fincast_v1.pth"; EXPECT=3966703063
mkdir -p "$(dirname "$W")"
for i in $(seq 1 40); do
  have=$(stat -c %s "$W" 2>/dev/null || stat -f %z "$W" 2>/dev/null || echo 0)
  [ "$have" -ge "$EXPECT" ] && break
  curl -L -C - --retry 3 --max-time 1800 -sS -o "$W" "$HUB/Vincent05R/FinCast/resolve/main/v1.pth" || true
  sleep 5
done
have=$(stat -c %s "$W" 2>/dev/null || stat -f %z "$W" 2>/dev/null || echo 0)
[ "$have" -ge "$EXPECT" ] || { echo "checkpoint incomplete: $have / $EXPECT bytes"; exit 1; }
$PY - <<'PYEOF'
import torch, numpy, transformers
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "numpy", numpy.__version__, "transformers", transformers.__version__)
import ffm
from tools.inference_utils import get_model_api, get_forecasts_f, freq_reader_inference
print("FinCast import OK; freq('D') =", freq_reader_inference("D"))
PYEOF
echo "=== install_fincast done ==="
