# Vast.ai Serverless: llama.cpp (any GGUF model) on 1x RTX PRO 6000 Blackwell

Serves any GGUF model supported by llama.cpp through `llama-server`, fronted by a custom Vast PyWorker. The model is chosen by environment variables (section 9). The defaults and the capacity numbers in sections 6-7 are for the example model [ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF) (IQ3_S, with vision).

```
client -> Vast router -> PyWorker :3000 -> llama-server 127.0.0.1:8080
```

Status: written from the model card and Vast's PyWorker/session docs. Not run end to end. Section 8 lists what to verify.

---

## 1. Template settings

| Setting | Value |
|---|---|
| Launch mode | Interactive shell server, SSH |
| Docker image | `nvidia/cuda:12.8.1-devel-ubuntu22.04` (Blackwell/sm_120 needs CUDA 12.8+) |
| Container disk | 200 GB |

## 2. Docker options

```
-p 3000:3000 -e HF_TOKEN=your_token -e MODEL_SERVER_URL=http://127.0.0.1 -e MODEL_SERVER_PORT=8080 -e MODEL_NAME=Qwen3.8-Flash-Next -e SLOTS=5 -e PYWORKER_REPO=https://github.com/<you>/<your-pyworker-repo>
```

- Port 3000 is the PyWorker (what Vast talks to). llama-server on 8080 is internal only.
- `SLOTS` is read by both `worker.py` (`max_sessions`) and the on-start script (`-np`).
- The model itself is chosen by optional environment variables (section 4). If you set none, you get this model (IQ3_S + vision). To use another model, add the variables from section 9.

## 3. Search filters

```
direct_port_count>=2 compute_cap>=1200 cuda_max_good>=12.8 gpu_ram>=90000 gpu_total_ram>=90000 cpu_ram>=64000 num_gpus=1 disk_space>=200
```

---

## 4. On-start script

This is the same script as `scripts/onstart.sh`, which `scripts/setup.sh` uses; keep the two in sync. It is model-agnostic. Every model-specific value is an environment variable whose default is this model.

```bash
#!/bin/bash
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive

# ---- Model settings (override via template env vars; defaults = this model) ----
MODEL_REPO="${MODEL_REPO:-ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF}"
MODEL_PATTERNS="${MODEL_PATTERNS:-IQ3_S/*,mmproj-Qwen3.8-Flash-Next-BF16.gguf}"   # comma-separated download globs
MODEL_FILE="${MODEL_FILE:-IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf}"  # first shard if split
MMPROJ_FILE="${MMPROJ_FILE-mmproj-Qwen3.8-Flash-Next-BF16.gguf}"  # set to "" for text-only models
EXTRA_ARGS="${EXTRA_ARGS--lm mmap --lazy-mode on}"                 # set to "" if the model needs none
KV_TYPE="${KV_TYPE:-q8_0}"                                         # f16 for full-precision KV
SLOTS="${SLOTS:-5}"                       # concurrent sequences (llama-server -np)
CTX_PER_SLOT="${CTX_PER_SLOT:-262144}"    # context each slot gets
TOTAL_CTX=$((SLOTS * CTX_PER_SLOT))       # -c is the TOTAL across slots
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
import os
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id=os.environ["MODEL_REPO"],
    allow_patterns=os.environ["MODEL_PATTERNS"].split(","),
    local_dir="/workspace/models",
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
```

Notes:
- `MMPROJ_FILE` and `EXTRA_ARGS` use `${VAR-default}` (no colon), so setting them to an empty string disables them instead of falling back to this model's defaults.
- `MODEL_PATTERNS` and `EXTRA_ARGS` values with spaces need quotes in the Docker options, e.g. `-e EXTRA_ARGS="-lm mmap --lazy-mode on"`.
- `-lm mmap --lazy-mode on` comes from this model's card. It keeps the 28.8 GB n-gram shard on disk so only shard 1 (54.8 GB) sits in VRAM. If llama-server rejects these flags or says `unknown model architecture (qwen4exp)`, mainline llama.cpp does not support this model yet and you need the build the card's authors used.
- The shard filename `...-IQ3_S-00001-of-00002.gguf` is inferred from the card's IQ3_XXS example. Confirm it in the repo's `IQ3_S/` folder.
- `KV_TYPE=q8_0` halves KV memory (V-cache quantization requires flash attention, `-fa on`). For 5 slots you can use `f16`.
- If your build has `--kv-unified`, add it to `EXTRA_ARGS` so slots share one cache pool instead of each reserving a full context.

---

## 5. PyWorker repo

Put `worker.py` and `requirements.txt` at the root of a **public** Git repo and point `PYWORKER_REPO` at it.

### requirements.txt

```
# vastai is provided by the Vast start-server script; add extra deps here if needed
```

### worker.py

```python
"""Vast.ai Serverless PyWorker for llama.cpp's llama-server (OpenAI-compatible API).

Expects the on-start script to:
  * run llama-server on 127.0.0.1:8080
  * write its output to /workspace/llama-server.log (truncated at each start)
"""
import os
import random

from vastai import (
    Worker,
    WorkerConfig,
    HandlerConfig,
    BenchmarkConfig,
    LogActionConfig,
)

# --- Model server -------------------------------------------------------------

MODEL_SERVER_URL = os.environ.get("MODEL_SERVER_URL", "http://127.0.0.1")
MODEL_SERVER_PORT = int(os.environ.get("MODEL_SERVER_PORT", "8080"))
MODEL_LOG_FILE = os.environ.get("MODEL_LOG_FILE", "/workspace/llama-server.log")
MODEL_NAME = os.environ.get("MODEL_NAME", "default")

# Matching is prefix-based and case-sensitive (per Vast docs), so these are line starts.
# llama-server prints these only after the model has finished loading.
# Check with: grep -n "listening" /workspace/llama-server.log  and adjust if your build differs.
LOAD_MSGS = [
    "main: server is listening",
    "main: starting the main loop",
    "srv  update_slots: all slots are idle",
]
ERROR_MSGS = [
    "main: exiting due to model loading error",
    "llama_model_load: error loading model",
    "llama_model_load_from_file_impl: failed to load model",
    "error: failed to load model",
    "CUDA error",
    "terminate called",
]

# --- Benchmark payloads -------------------------------------------------------

_WORDS = (
    "river stone engine garden window signal market paper silver forest "
    "bridge copper violet harbor lantern meadow orbit pencil quartz ribbon "
    "shadow timber umbrella valley winter yellow zephyr anchor beacon candle"
).split()


def benchmark_generator() -> dict:
    prompt = " ".join(random.choices(_WORDS, k=200))
    return {
        "model": MODEL_NAME,
        "messages": [
            {"role": "user", "content": f"Summarize the following words in one paragraph: {prompt}"}
        ],
        "max_tokens": 256,
        "temperature": 0.7,
        "stream": False,
    }


# --- Workload -----------------------------------------------------------------

def _chat_text_len(messages) -> int:
    total = 0
    for m in messages or []:
        c = m.get("content", "")
        if isinstance(c, str):
            total += len(c)
        elif isinstance(c, list):  # multimodal parts
            for part in c:
                if part.get("type") == "text":
                    total += len(part.get("text", ""))
                else:
                    total += 4000  # rough cost of an image
    return total


def chat_workload(payload: dict) -> float:
    prompt_tokens = _chat_text_len(payload.get("messages")) / 4.0
    return prompt_tokens + float(payload.get("max_tokens", 512))


def completion_workload(payload: dict) -> float:
    prompt = payload.get("prompt", "")
    prompt_tokens = (len(prompt) if isinstance(prompt, str) else 0) / 4.0
    return prompt_tokens + float(payload.get("max_tokens", 512))


# --- Worker config ------------------------------------------------------------

worker_config = WorkerConfig(
    model_server_url=MODEL_SERVER_URL,
    model_server_port=MODEL_SERVER_PORT,
    model_log_file=MODEL_LOG_FILE,
    model_healthcheck_url="/health",   # llama-server's health endpoint
    max_sessions=int(os.environ.get("SLOTS", "5")),  # keep equal to llama-server -np
    handlers=[
        HandlerConfig(
            route="/v1/chat/completions",
            allow_parallel_requests=True,   # llama-server batches across --parallel slots
            max_queue_time=120.0,
            workload_calculator=chat_workload,
            benchmark_config=BenchmarkConfig(
                generator=benchmark_generator,
                runs=4,
                concurrency=2,
            ),
        ),
        HandlerConfig(
            route="/v1/completions",
            allow_parallel_requests=True,
            max_queue_time=120.0,
            workload_calculator=completion_workload,
        ),
    ],
    log_action_config=LogActionConfig(
        on_load=LOAD_MSGS,
        on_error=ERROR_MSGS,
        on_info=["srv  load_model:", "load_tensors:"],
    ),
)

Worker(worker_config).run()
```

Sessions (`endpoint.session()`) need no extra code: the PyWorker serves `/session/create`, `/session/end`, `/session/get` and `/session/health` itself.

---

## 6. Capacity (estimates, for the default Qwen3.8-Flash-Next model only)

Qwen3.8-Flash-Next has 48 layers, of which only 12 use full attention (the rest are Gated DeltaNet), with 2 KV heads and a 262,144-token native context.

| | |
|---|---|
| KV per token, fp16 | ~12-24 KB (head dim 128 vs 256, unconfirmed) |
| KV for one 256k sequence | ~3-6 GB fp16, ~1.6-3 GB q8_0 |
| VRAM free for cache (IQ3_S, 96 GB card) | ~36 GB (shard 1 + mmproj + ~4 GB buffers) |
| 5 x 256k | fits (16-31 GB fp16) |
| 10 x 256k | borderline; use q8_0 KV or IQ3_XXS (~43 GB free) |

To go to 10 slots: set `-e SLOTS=10` (and `CTX_PER_SLOT` if you want a different per-slot context). llama.cpp logs the real KV size at load (`llama_kv_cache: ... size = ... MiB`); trust that over these estimates.

**Prefill is the real limit.** The model card measured ~108-367 t/s prompt processing. A cold 256k prompt can take 12-40+ minutes, and the Vast SDK gives session requests a 600 s worker timeout. Use sessions so a long prefix is prefilled once and reused from llama-server's prompt cache.

---

## 7. Vision

Already enabled by `--mmproj`. Send standard OpenAI image parts to `/v1/chat/completions`:

```bash
IMG=$(base64 -w0 photo.jpg)
curl localhost:8080/v1/chat/completions -H "Content-Type: application/json" -d @- <<EOF
{"messages":[{"role":"user","content":[
  {"type":"text","text":"Describe this image."},
  {"type":"image_url","image_url":{"url":"data:image/jpeg;base64,$IMG"}}
]}],"max_tokens":256}
EOF
```

- Image tokens count toward context and KV. If your build has `--image-max-tokens`, set it to cap cost.
- The projector and vision encoder buffers come out of the same free VRAM as the KV cache.
- The benchmark is text-only, so test an image request by hand.
- Large base64 bodies may hit a request-size limit (unverified).

---

## 8. Verify before relying on it

1. **Test on a normal on-demand instance first** (same image and GPU). Run the on-start script by hand, then check `nvidia-smi` and `curl localhost:8080/v1/models`.
2. **Readiness log lines.** Vast matches `on_load` by exact line prefix. Run `grep -n "listening\|idle" /workspace/llama-server.log` and edit `LOAD_MSGS` in `worker.py` if your build prints something different. If none match, the worker stays on "loading" forever.
3. **Long-context quality.** I could not confirm llama.cpp implements the model's sparse attention (QSA) like Qwen's reference code. Test retrieval at 200k+ tokens.
4. **`model_healthcheck_url`** appears in Vast's sessions example only. Remove it from `worker.py` if your PyWorker version rejects it.
5. **Stuck on loading?** Check `/workspace/llama-server.log` and the PyWorker's own log first.

---

## 9. Swapping in a different llama.cpp model

`worker.py` needs no changes: it only relies on llama-server's OpenAI-compatible API, and `MODEL_NAME` (env var) is mostly ignored by llama-server. Set these template environment variables for the new model:

| Variable | This model (default) | What to set for another model |
|---|---|---|
| `MODEL_REPO` | `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF` | the Hugging Face repo |
| `MODEL_PATTERNS` | `IQ3_S/*,mmproj-Qwen3.8-Flash-Next-BF16.gguf` | globs for just the files you want (avoid downloading every quant) |
| `MODEL_FILE` | `IQ3_S/...-00001-of-00002.gguf` | path to the `.gguf`, or the first shard if split |
| `MMPROJ_FILE` | `mmproj-Qwen3.8-Flash-Next-BF16.gguf` | the projector filename for vision models, or `""` for text-only |
| `EXTRA_ARGS` | `-lm mmap --lazy-mode on` | `""`, or any model-specific flags (e.g. `--chat-template ...`) |
| `KV_TYPE` | `q8_0` | `q8_0` or `f16` |
| `SLOTS` | `5` | concurrent sequences you want |
| `CTX_PER_SLOT` | `262144` | context per sequence the model and your VRAM can hold |
| `MODEL_NAME` | `default` in `worker.py` (the Docker options example sets `Qwen3.8-Flash-Next`) | any label; used in benchmark payloads |

Example (hypothetical dense text-only model in a single file, 4 slots of 32k):
```
-e MODEL_REPO=<org>/<repo> -e MODEL_PATTERNS="<file>.gguf" -e MODEL_FILE="<file>.gguf" -e MMPROJ_FILE="" -e EXTRA_ARGS="" -e SLOTS=4 -e CTX_PER_SLOT=32768 -e MODEL_NAME=<label>
```

Things that do **not** carry over between models:
- **Memory math.** Section 6 is specific to this hybrid architecture (KV for only 12 of 48 layers). A conventional dense model needs far more KV per token, so use much smaller `SLOTS` and `CTX_PER_SLOT`. Check the `llama_kv_cache: ... size` line at load and `nvidia-smi`.
- **Architecture support.** llama.cpp must support the architecture. The build clones latest master, which usually covers new models, but a brand-new architecture may need a fork.
- **Chat template.** `--jinja` uses the template embedded in the GGUF. Some models need `--chat-template` or `--chat-template-file` in `EXTRA_ARGS`.
- **Vision.** Only models with a separate projector file use `MMPROJ_FILE`. Some GGUFs bundle vision differently.
- **Readiness log lines.** `LOAD_MSGS` / `ERROR_MSGS` in `worker.py` are llama.cpp's own and don't depend on the model, but verify them once per llama.cpp build (section 8).
- **GPU generation.** The build targets sm_120 (Blackwell). For a different GPU family, change `CMAKE_CUDA_ARCHITECTURES` (and the Docker image/filters).


## Sources

- Model card: https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF
- Vast custom PyWorkers: https://docs.vast.ai/guides/serverless/creating-new-pyworkers
- Vast sessions: https://docs.vast.ai/guides/serverless/sessions
