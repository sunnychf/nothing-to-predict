# Time-MoE-200M runs in the FinCast env (transformers 4.53, torch 2.5.0+cu124; no separate install).
# The checkpoint Maple728/TimeMoE-200M (~850 MB) is fetched into the Hugging Face cache on first use;
# set HF_HOME to a disk with room if the default cache location is full.
source "$(dirname "${BASH_SOURCE[0]}")/env_common.sh"
export PY="$CONDA_ENVS/fincast_v1/bin/python"
