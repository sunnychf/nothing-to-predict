# TimesFM-2.5 (timesfm 3.x): conda env `tsfm25` (scripts/install_tsfm25.sh). The released ForecastConfig defaults
# (flip-invariance averaging on) are what the drivers use; --flip off in pilot_a_timesfm25.py switches it off.
source "$(dirname "${BASH_SOURCE[0]}")/env_common.sh"
export PY="$CONDA_ENVS/tsfm25/bin/python"
