# Sundial-base-128M runs in the FinCast env (transformers 4.53, torch 2.5.0+cu124; no separate install),
# like Time-MoE. The checkpoint thuml/sundial-base-128m (~0.5 GB) and its model code are fetched into
# the Hugging Face cache on first use (trust_remote_code); set HF_HOME to a disk with room if the
# default cache location is full.
source "$(dirname "${BASH_SOURCE[0]}")/env_common.sh"
export PY="$CONDA_ENVS/fincast_v1/bin/python"
