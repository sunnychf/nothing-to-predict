# Sourced by every scripts/env_*.sh. Sets the repository root, the Python path, the result
# locations the scripts read from their environment, and the GPU (the card with the most free
# memory, selected by UUID so that a renumbering of the cards cannot change the choice).
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export ROOT
export PYTHONPATH="$ROOT/code${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PILOT_A_OUT="$ROOT/results/pilot_a.jsonl"      # ladder / controls / declared-frequency rows (append-only JSONL)
export SPEC_OUT="$ROOT/results"                        # spectral, dispersion, direction and level diagnostics
export REAL_WINDOWS="$ROOT/results/real_windows.npz"   # built by code/real_data.py
export REAL_OUT="$ROOT/results"                        # real_<model>*.npz, remedy_<model>.npz
CONDA_ENVS="${CONDA_ENVS:-$HOME/anaconda3/envs}"       # where scripts/install_*.sh created the environments; point it at a disk with room (each env is 5-7 GB)
# export HF_ENDPOINT=https://<mirror>                  # only if huggingface.co is reachable through a mirror alone
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ] && command -v nvidia-smi >/dev/null 2>&1; then
  _best=$(nvidia-smi --query-gpu=uuid,memory.used,memory.total --format=csv,noheader,nounits \
    | awk -F', ' '{free=$3-$2; if (free>max) {max=free; u=$1}} END {print u, max}')
  export CUDA_VISIBLE_DEVICES=$(echo "$_best" | cut -d' ' -f1)
  echo "[env] GPU $CUDA_VISIBLE_DEVICES ($(echo "$_best" | cut -d' ' -f2) MiB free)"
fi
