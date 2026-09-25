#!/bin/bash
# CPU builds of the paper's model environments (the GPU server install scripts with the CPU torch wheel).
export PIP_NO_CACHE_DIR=1 HF_ENDPOINT=https://huggingface.co HF_HOME=$HOME/tsfm_rev/hf
C=$HOME/miniconda3/bin/conda; E=$HOME/tsfm_rev/envs; L=$HOME/tsfm_rev/logs
CPU=https://download.pytorch.org/whl/cpu
mk() { [ -x "$E/$1/bin/python" ] || $C create -y -p "$E/$1" python=$2 > $L/inst_$1.log 2>&1; }
( mk tirex 3.11 && $E/tirex/bin/pip install torch==2.5.1 --index-url $CPU >> $L/inst_tirex.log 2>&1 && $E/tirex/bin/pip install tirex-ts numpy pandas huggingface_hub >> $L/inst_tirex.log 2>&1 \
  && $E/tirex/bin/python -c "from huggingface_hub import snapshot_download as s; print(s(\"NX-AI/TiRex\")); import tirex, torch; print(\"OK\", torch.__version__)" >> $L/inst_tirex.log 2>&1; echo END >> $L/inst_tirex.log ) &
( mk tsfm25 3.11 && $E/tsfm25/bin/pip install torch==2.5.1 --index-url $CPU >> $L/inst_tsfm25.log 2>&1 && $E/tsfm25/bin/pip install "timesfm[torch]" numpy pandas huggingface_hub >> $L/inst_tsfm25.log 2>&1 \
  && $E/tsfm25/bin/python -c "from huggingface_hub import snapshot_download as s; print(s(\"google/timesfm-2.5-200m-pytorch\")); import timesfm, torch; timesfm.TimesFM_2p5_200M_torch; print(\"OK\", torch.__version__)" >> $L/inst_tsfm25.log 2>&1; echo END >> $L/inst_tsfm25.log ) &
( mk moirai 3.10 && $E/moirai/bin/pip install uni2ts >> $L/inst_moirai.log 2>&1 && TV=$($E/moirai/bin/python -c "import torch; print(torch.__version__.split(\"+\")[0])") && $E/moirai/bin/pip install "torch==$TV" --index-url $CPU >> $L/inst_moirai.log 2>&1 \
  && $E/moirai/bin/python -c "from huggingface_hub import snapshot_download as s; print(s(\"Salesforce/moirai-1.1-R-small\")); print(s(\"Salesforce/moirai-2.0-R-small\")); import uni2ts, torch; print(\"OK\", torch.__version__)" >> $L/inst_moirai.log 2>&1; echo END >> $L/inst_moirai.log ) &
( mk timesfm 3.11 && $E/timesfm/bin/pip install torch==2.5.1 --index-url $CPU >> $L/inst_timesfm.log 2>&1 && $E/timesfm/bin/pip install "timesfm[torch]==1.3.0" >> $L/inst_timesfm.log 2>&1 \
  && $E/timesfm/bin/python -c "from huggingface_hub import snapshot_download as s; print(s(\"google/timesfm-2.0-500m-pytorch\")); import timesfm, torch; print(\"OK\", torch.__version__)" >> $L/inst_timesfm.log 2>&1; echo END >> $L/inst_timesfm.log ) &
( cd $HOME/tsfm_rev && [ -d FinCast-fts ] || git clone --depth 1 https://github.com/vincent05r/FinCast-fts.git > $L/inst_fincast.log 2>&1; mk fincast_v1 3.11.11 && cd $HOME/tsfm_rev/FinCast-fts && $E/fincast_v1/bin/pip install -r requirement_v2.txt >> $L/inst_fincast.log 2>&1; $E/fincast_v1/bin/pip install -e . >> $L/inst_fincast.log 2>&1; \
  $E/fincast_v1/bin/pip install torch==2.5.0 --index-url $CPU >> $L/inst_fincast.log 2>&1; $E/fincast_v1/bin/pip install "transformers==4.53.0" huggingface_hub >> $L/inst_fincast.log 2>&1; \
  mkdir -p $HOME/tsfm_rev/weights; W=$HOME/tsfm_rev/weights/fincast_v1.pth; for i in $(seq 1 40); do have=$(stat -c %s "$W" 2>/dev/null || echo 0); [ "$have" -ge 3966703063 ] && break; curl -L -C - --retry 3 --max-time 1800 -sS -o "$W" "$HF_ENDPOINT/Vincent05R/FinCast/resolve/main/v1.pth" >> $L/inst_fincast.log 2>&1 || true; sleep 5; done; \
  $E/fincast_v1/bin/python -c "from huggingface_hub import snapshot_download as s; print(s(\"Maple728/TimeMoE-200M\")); print(s(\"thuml/sundial-base-128m\")); import torch, transformers; print(\"OK\", torch.__version__, transformers.__version__)" >> $L/inst_fincast.log 2>&1; ls -la $W >> $L/inst_fincast.log; echo END >> $L/inst_fincast.log ) &
wait
echo ALLDONE > $L/inst_all.done
