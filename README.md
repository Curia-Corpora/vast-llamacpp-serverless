# llama.cpp PyWorker for Vast.ai Serverless

A custom [Vast.ai Serverless](https://docs.vast.ai/guides/serverless/overview) PyWorker that fronts llama.cpp's `llama-server`. It lets you run any GGUF model that llama.cpp supports as a Serverless endpoint, with OpenAI-style chat/completions routes, vision support (when a projector file is provided), and worker sessions.

The default configuration targets one RTX PRO 6000 Blackwell (96 GB) running [ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF) (IQ3_S). The model is chosen by environment variables, so other models can be swapped in without code changes.

> **Status:** written from the model card and Vast's PyWorker and session docs, and not yet run end to end. See [Before you rely on it](#before-you-rely-on-it).

## How it works

```
client -> Vast router -> PyWorker (:3000) -> llama-server (127.0.0.1:8080)
```

1. The template's on-start script builds llama.cpp, downloads the model, and starts `llama-server` with its output going to `/workspace/llama-server.log`.
2. Vast's start-server script clones this repo (via `PYWORKER_REPO`), installs `requirements.txt`, and runs `worker.py`.
3. `worker.py` tails the llama-server log for a readiness line, runs a short benchmark, and then joins the ready pool.
4. Requests to `/v1/chat/completions` and `/v1/completions` are proxied to llama-server. Sessions (`/session/*`) are served by the PyWorker itself.

## Repo layout

| Path | Purpose |
|---|---|
| `worker.py` | The PyWorker: routes, workload estimation, benchmark, readiness log matching |
| `requirements.txt` | Installed by Vast's start-server script (`vastai-sdk`, which provides `Worker`/`WorkerConfig`) |
| `scripts/setup.sh` | Creates the template, endpoint and workergroup with the `vastai` CLI, driven by `.env` |
| `scripts/onstart.sh` | The template's on-start script (build llama.cpp, download the model, start llama-server, start the PyWorker) |
| `.env.example` | Every setting `setup.sh` understands, with defaults |
| `docs/vast-llamacpp-serverless-config.md` | Full template configuration: Docker image and options, search filters, on-start script, capacity notes, vision, model swapping |

(Adjust the doc path if yours differs.)

## Automated setup (vastai CLI)

```bash
pip install vastai
cp .env.example .env        # fill in VASTAI_KEY and HF_TOKEN (.env is gitignored; never commit it)
./scripts/setup.sh --dry-run   # print the vastai commands first
./scripts/setup.sh             # create template, endpoint and workergroup
./scripts/setup.sh status
```

Only `VASTAI_KEY` and `HF_TOKEN` are required; every other setting in `.env.example` has a default. The script saves the created IDs in `.vast-state` and reuses them on re-runs (`--force` creates new resources instead). By default `HF_TOKEN` is stored as a Vast account environment variable rather than embedded in the template. `TEST_WORKERS` defaults to 1 because each test worker rents a GPU while benchmarking.

The CLI syntax was taken from Vast's docs and the script was tested only against a stub CLI. Use `--dry-run` first, and see the notes in [Before you rely on it](#before-you-rely-on-it).

## Quick start (manual)

1. Make this repo **public** (Vast clones it without credentials).
2. Create a Serverless template on Vast using the values in [`docs/vast-llamacpp-serverless-config.md`](docs/vast-llamacpp-serverless-config.md):
   - Docker image `nvidia/cuda:12.8.1-devel-ubuntu22.04` (Blackwell needs CUDA 12.8+)
   - the Docker options, search filters, and on-start script from the doc
   - `-e PYWORKER_REPO=https://github.com/<you>/<this-repo>`
3. Use `scripts/onstart.sh` as the on-start script. It ends by running Vast's PyWorker bootstrap (`start_server.sh`), which is unverified against current Vast templates; compare it with a stock template's on-start.
4. Create an Endpoint and Workergroup from the template.

## Configuration

`worker.py` reads these environment variables (all optional):

| Variable | Default | Meaning |
|---|---|---|
| `MODEL_SERVER_URL` | `http://127.0.0.1` | Where llama-server listens |
| `MODEL_SERVER_PORT` | `8080` | llama-server port |
| `MODEL_LOG_FILE` | `/workspace/llama-server.log` | Log file the PyWorker tails for readiness and errors |
| `MODEL_NAME` | `default` | Label used in benchmark payloads (llama-server mostly ignores it) |
| `SLOTS` | `5` | `max_sessions`; keep equal to llama-server's `-np` |

Model selection, context size, and KV settings are controlled in the on-start script through further variables (`MODEL_REPO`, `MODEL_FILE`, `MMPROJ_FILE`, `CTX_PER_SLOT`, and others). The doc describes them all in section 9.

## Using the endpoint

Normal requests use the Vast Serverless SDK:

```python
import asyncio
from vastai import Serverless

async def main():
    async with Serverless() as client:
        endpoint = await client.get_endpoint(name="my-endpoint")
        result = await endpoint.request("/v1/chat/completions", {
            "model": "default",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 256,
        })
        print(result["response"]["choices"][0]["message"]["content"])

asyncio.run(main())
```

For multi-turn conversations or long prompts, open a [session](https://docs.vast.ai/guides/serverless/sessions) so every turn lands on the same worker and can reuse llama-server's prompt cache:

```python
async with await endpoint.session(cost=512, lifetime=300) as session:
    result = await session.request("/v1/chat/completions", payload)
```

Always close sessions (the `async with` form does this). An open session holds one of the worker's `SLOTS`.

Images are sent as standard OpenAI `image_url` content parts, provided the template sets a vision projector (`MMPROJ_FILE`).

## Before you rely on it

- **Readiness lines.** Vast matches `on_load` log lines by exact prefix. The values in `LOAD_MSGS` in `worker.py` are llama.cpp's usual output but are unverified for your build. Check with `grep -n "listening\|idle" /workspace/llama-server.log`. If none match, the worker stays on "loading".
- **Test on a normal instance first.** Run the on-start script by hand on an on-demand instance with the same GPU and image, and confirm `curl localhost:8080/v1/models` works before using Serverless.
- **`vastai` package.** `requirements.txt` lists `vastai-sdk` (the package Vast's docs name for the SDK). If the worker log shows an import error for `Worker`/`WorkerConfig`, try `vastai` instead.
- **CLI output parsing.** `setup.sh` reads the template hash and IDs from the CLI's `--raw` JSON. If it can't find them it prints the output; put `TEMPLATE_HASH=...` in `.env` and re-run.
- **`model_healthcheck_url`.** It appears only in Vast's sessions example. Remove it from `worker.py` if your PyWorker version rejects it.
- **Long contexts.** Cold prefill of very long prompts can take many minutes, and session requests have a 600 s worker timeout. Use sessions so a long prefix is processed once.
- **Model support.** llama.cpp must support the model's architecture. A very new architecture may need a fork.

## References

- [Creating custom PyWorkers](https://docs.vast.ai/guides/serverless/creating-new-pyworkers)
- [Worker sessions](https://docs.vast.ai/guides/serverless/sessions)
- [Model card](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF)
