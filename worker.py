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
