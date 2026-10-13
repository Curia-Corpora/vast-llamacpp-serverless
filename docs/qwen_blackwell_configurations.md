# Vast.ai Serverless Configuration Guide: Qwen 3.8 Flash-Next

This document provides the full, ready-to-copy configurations for deploying **Qwen 3.8 Flash-Next** on a single **NVIDIA RTX PRO 6000 Blackwell (96GB)** instance using Vast.ai Serverless.

---

## 🛠️ Option A: NVFP4 Format (Recommended for vLLM)
This configuration targets **primitive-ai/Qwen3.8-Flash-Next-NVFP4** using a modern vLLM image. It leverages Blackwell's native FP4 hardware tensor cores and dynamically offloads the heavy 51GB N-gram (PLE) lookup table to system host RAM.

### 1. Template Settings
* **Launch Mode:** `Interactive shell server, SSH`
* **Docker Image:** `vllm/vllm-openai:latest` (or `vllm/vllm-openai:qwen38-flash-next`)
* **Container Disk Size:** `150 GB` (The NVFP4 weights are heavily compressed to ~40GB-45GB, so 150GB provides plenty of workspace buffer).

### 2. GPU & Instance Filters
Copy this string directly into the Vast.ai template filter bar. This ensures you target a true Blackwell node with massive system host RAM.
```text
direct_port_count>=2 compute_cap>=1200 cuda_max_good>=12.9 gpu_ram>=90000 cpu_ram>=131072 num_gpus==1
```
*Note: `compute_cap>=1200` specifically restricts matches to Blackwell architecture, and `cpu_ram>=131072` ensures the system has 128GB+ host memory for the N-gram table.*

### 3. Environment Variables
```bash
-p 3000:3000 -e MODEL_NAME="primitive-ai/Qwen3.8-Flash-Next-NVFP4" -e VLLM_ARGS="--reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder --trust-remote-code --download-dir /workspace/models --host 127.0.0.1 --port 18000" -e VLLM_PLE_CPU_OFFLOAD="1" -e AUTO_PARALLEL="false" -e SERVERLESS="true" -e BACKEND="openai" -e MODEL_LOG="/var/log/portal/vllm.log" -e MODEL_SERVER_URL="http://127.0.0.1:18000" -e MODEL_HEALTH_ENDPOINT="http://127.0.0.1:18000/health" -e HF_TOKEN="your_huggingface_token_here"
```
*CRITICAL: `--enable-expert-parallel` must be stripped out. Forcing expert parallelism on a single GPU crashes NVFP4 MoE compilation.*

### 4. On-start Script
```bash
export HF_TOKEN="${HF_TOKEN:-1}"
entrypoint.sh &
```

---

## 🔬 Option B: ISTA-DASLab GSQ-RCO GGUF (Custom Strata Engine)
This configuration bypasses vLLM to host **ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF** natively using the custom StrataEngine.

### 1. Template Settings
* **Launch Mode:** `Interactive shell server, SSH`
* **Docker Image:** `pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel` (Requires a clean CUDA development image)
* **Container Disk Size:** `200 GB`

### 2. GPU & Instance Filters
```text
direct_port_count>=2 compute_cap>=1200 cuda_max_good>=12.4 gpu_total_ram>=90000 gpu_ram>=90000 cpu_ram>=131072 num_gpus in ['1']
```

### 3. Environment Variables
```bash
-p 3000:3000 -e MODEL_NAME="ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF" -e SERVERLESS="true" -e BACKEND="openai" -e MODEL_SERVER_URL="http://127.0.0.1:3000" -e HF_TOKEN="your_token"
```

### 4. On-start Script
Because this format uses an experimental structural quantization layout, the On-start script manually installs the required architecture files and initiates an OpenAI-compatible web endpoint.
```bash
#!/bin/bash
export HF_TOKEN="${HF_TOKEN:-1}"

# Ensure system has core compiler and downloading utilities
apt-get update && apt-get install -y git git-lfs curl build-essential

# Upgrade pip and install standard API helper libraries
/opt/conda/bin/python3 -m pip install --upgrade pip
/opt/conda/bin/python3 -m pip install fastapi uvicorn requests pydantic huggingface_hub

# Clone the custom Strata Engine repository
git clone https://github.com /workspace/strata
cd /workspace/strata

# Compile the engine's custom C++/CUDA backend wrappers
/opt/conda/bin/python3 setup.py --setup --build --yes

# Download the core text model AND the mandatory vision projector adapter
mkdir -p /workspace/models
/opt/conda/bin/python3 -c "
from huggingface_hub import hf_hub_download

# 1. Fetch the primary 3-bit text engine file
hf_hub_download(repo_id='ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF', filename='Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S.gguf', local_dir='/workspace/models')

# 2. Fetch the multimodal vision adapter file (Omits the heavy JSON layout ledger logs)
hf_hub_download(repo_id='ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF', filename='mmproj-Qwen3.8-Flash-Next-BF16.gguf', local_dir='/workspace/models')
"

# Launch the engine server with explicit multimodal flags enabled over port 3000
(/opt/conda/bin/python3 setup.py --family qwen --model IQ3_S --port 3000 --host 0.0.0.0 --vision yes > /workspace/strata.log 2>&1 &)

# Allow the vision server a short window to allocate VRAM tables
sleep 5

# Tell Vast's serverless proxy layer to latch onto port 3000
export SERVERLESS="true"
export BACKEND="openai"
export MODEL_SERVER_URL="http://127.0.0.1:3000"
bootstrap_script="https://githubusercontent.com"
curl -sSL "$bootstrap_script" | bash

```
