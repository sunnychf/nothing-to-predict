# FinCast: conda env `fincast_v1` (scripts/install_fincast.sh). FinCast's package is imported as
# `ffm`, `tools`, `data_tools` from the clone's src/; the weights file is the 3.97 GB v1.pth.
source "$(dirname "${BASH_SOURCE[0]}")/env_common.sh"
export PY="$CONDA_ENVS/fincast_v1/bin/python"
export PYTHONPATH="$ROOT/FinCast-fts/src:$PYTHONPATH"
export FINCAST_WEIGHTS="$ROOT/weights/fincast_v1.pth"
