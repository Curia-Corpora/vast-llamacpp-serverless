#!/bin/bash
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive

# ---- Model settings (override via template env vars; defaults = this model) ----
# NOTE: these MUST stay `export`ed - the model-download step below runs as a child
# python3 process and reads them from the environment (os.environ[...]). Without
# `export` they are shell-only and the download dies with KeyError: 'MODEL_REPO'.
export MODEL_REPO="${MODEL_REPO:-ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF}"
export MODEL_PATTERNS="${MODEL_PATTERNS:-IQ3_S/*,mmproj-Qwen3.8-Flash-Next-BF16.gguf}"   # comma-separated download globs
export MODEL_FILE="${MODEL_FILE:-IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf}"  # first shard if split
export MMPROJ_FILE="${MMPROJ_FILE-mmproj-Qwen3.8-Flash-Next-BF16.gguf}"  # set to "" for text-only models
export EXTRA_ARGS="${EXTRA_ARGS--lm mmap --lazy-mode on}"                 # set to "" if the model needs none
export KV_TYPE="${KV_TYPE:-q8_0}"                                         # f16 for full-precision KV
export SLOTS="${SLOTS:-5}"                      # concurrent sequences (llama-server -np)
export CTX_PER_SLOT="${CTX_PER_SLOT:-262144}"   # context each slot gets
TOTAL_CTX=$((SLOTS * CTX_PER_SLOT))             # -c is the TOTAL across slots
# The value "none" also disables MMPROJ_FILE / EXTRA_ARGS (empty values can get dropped upstream)
if [ "$MMPROJ_FILE" = "none" ]; then MMPROJ_FILE=""; fi
if [ "$EXTRA_ARGS" = "none" ]; then EXTRA_ARGS=""; fi
# --------------------------------------------------------------------------------

mkdir -p /workspace/models
apt-get update && apt-get install -y git cmake ninja-build build-essential curl python3-pip
pip3 install -U huggingface_hub

# Build llama.cpp for Blackwell (sm_120); cached in /workspace after first run
if [ ! -x /workspace/llama.cpp/build/bin/llama-server ]; then
  git clone --depth 1 https://github.com/ggml-org/llama.cpp /workspace/llama.cpp
  cd /workspace/llama.cpp
  cmake -B build -G Ninja -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120
  cmake --build build -j --target llama-server
fi

# Download the model files
python3 - <<'PY'
import os, sys
required = ["MODEL_REPO", "MODEL_PATTERNS", "MODEL_FILE"]
missing = [k for k in required if not os.environ.get(k)]
if missing:
    sys.exit(f"FATAL: required env vars not exported into this shell: {', '.join(missing)}")
from huggingface_hub import snapshot_download
print(f"downloading {os.environ['MODEL_REPO']} :: {os.environ['MODEL_PATTERNS']}", flush=True)
snapshot_download(
    repo_id=os.environ["MODEL_REPO"],
    allow_patterns=os.environ["MODEL_PATTERNS"].split(","),
    local_dir="/workspace/models",
    token=os.environ.get("HF_TOKEN") or None,
)
PY

# Truncate the log so the PyWorker only sees this run
: > /workspace/llama-server.log

MMPROJ_ARG=""
[ -n "$MMPROJ_FILE" ] && MMPROJ_ARG="--mmproj /workspace/models/$MMPROJ_FILE"

# shellcheck disable=SC2086  # EXTRA_ARGS / MMPROJ_ARG are intentionally word-split
nohup /workspace/llama.cpp/build/bin/llama-server \
  -m "/workspace/models/$MODEL_FILE" $MMPROJ_ARG \
  -ngl 99 --jinja -fa on -ctk "$KV_TYPE" -ctv "$KV_TYPE" \
  -np "$SLOTS" -c "$TOTAL_CTX" $EXTRA_ARGS \
  --host 0.0.0.0 --port 8080 >> /workspace/llama-server.log 2>&1 &

# Wait until the model is actually loaded (can take minutes)
until curl -sf http://127.0.0.1:8080/health >/dev/null; do sleep 5; done
echo "llama-server ready"

# --- Serverless: start the Vast PyWorker (clones $PYWORKER_REPO, installs requirements.txt, runs worker.py) ---
# Override with PYWORKER_BOOTSTRAP_CMD if Vast changes its bootstrap script.
BOOTSTRAP="${PYWORKER_BOOTSTRAP_CMD:-curl -L https://raw.githubusercontent.com/vast-ai/pyworker/main/start_server.sh | bash}"
bash -c "$BOOTSTRAP"
