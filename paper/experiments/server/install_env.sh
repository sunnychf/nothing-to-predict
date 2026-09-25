#!/bin/bash
set -e
export PIP_NO_CACHE_DIR=1
~/miniconda3/bin/conda create -y -p ~/tsfm_rev/env python=3.11 > ~/tsfm_rev/logs/install.log 2>&1
P=~/tsfm_rev/env/bin/pip
$P install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu >> ~/tsfm_rev/logs/install.log 2>&1
$P install --no-cache-dir "chronos-forecasting==2.3.2" datasets pandas pyarrow scipy huggingface_hub >> ~/tsfm_rev/logs/install.log 2>&1
~/tsfm_rev/env/bin/python -c "import torch, chronos, transformers; print(\"OK\", torch.__version__, chronos.__version__ if hasattr(chronos,\"__version__\") else \"?\", transformers.__version__, torch.get_num_threads())" >> ~/tsfm_rev/logs/install.log 2>&1
echo DONE >> ~/tsfm_rev/logs/install.log
