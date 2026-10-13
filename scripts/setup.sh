#!/usr/bin/env bash
# Create a Vast.ai Serverless template, endpoint and workergroup for the llama.cpp PyWorker.
#
# Usage: ./scripts/setup.sh [create|update|status] [--dry-run] [--force]
#   create     (default) create the template, endpoint and workergroup
#   update     re-upload the on-start script and template env to the saved template
#              (use this after editing scripts/onstart.sh — create will not re-send it)
#   status     show endpoints and workergroups
#   --dry-run  print the vastai commands without running them (API key redacted)
#   --force    ignore IDs saved in .vast-state and create new resources
#
# Reads .env from the repo root (override with ENV_FILE=...). Only VASTAI_KEY and HF_TOKEN are
# required; everything else has a default. See .env.example for all options.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${ENV_FILE:-$ROOT/.env}"
STATE_FILE="$ROOT/.vast-state"

CMD="create"; DRY_RUN=0; FORCE=0
for arg in "$@"; do
  case "$arg" in
    create|update|status) CMD="$arg" ;;
    --dry-run) DRY_RUN=1 ;;
    --force) FORCE=1 ;;
    -h|--help) sed -n '2,13p' "$0"; exit 0 ;;
    *) echo "Unknown argument: $arg" >&2; exit 2 ;;
  esac
done

die() { echo "ERROR: $*" >&2; exit 1; }

[ -f "$ENV_FILE" ] || die "Missing $ENV_FILE (copy .env.example to .env and fill it in)"
set -a; . "$ENV_FILE"; set +a

: "${VASTAI_KEY:?VASTAI_KEY is not set in $ENV_FILE}"
command -v vastai >/dev/null 2>&1 || die "vastai CLI not found (pip install vastai)"
command -v python3 >/dev/null 2>&1 || die "python3 is required to parse CLI output"

# ---- Defaults (override in .env) --------------------------------------------------------
: "${NAME_PREFIX:=llamacpp}"
: "${TEMPLATE_NAME:=$NAME_PREFIX-template}"
: "${ENDPOINT_NAME:=$NAME_PREFIX-endpoint}"
: "${IMAGE:=nvidia/cuda:12.8.1-devel-ubuntu22.04}"
: "${DISK_GB:=200}"
: "${SEARCH_PARAMS:=direct_port_count>=2 compute_cap>=1200 cuda_max_good>=12.8 gpu_ram>=90000 gpu_total_ram>=90000 cpu_ram>=128000 num_gpus=1 disk_space>=200}"
: "${SLOTS:=5}"
: "${HF_TOKEN_MODE:=account}"      # account = store as a Vast account env var; template = embed in template
# Endpoint / workergroup scaling
: "${MIN_LOAD:=10}"
: "${TARGET_UTIL:=0.9}"
: "${COLD_MULT:=1}"
: "${COLD_WORKERS:=0}"
: "${MAX_WORKERS:=1}"
: "${TEST_WORKERS:=1}"             # each test worker rents a GPU while benchmarking
: "${GPU_RAM:=96}"                 # estimated GPU RAM (GiB) the model needs

# PyWorker repo: from .env, else derived from this repo's git remote
if [ -z "${PYWORKER_REPO:-}" ]; then
  remote="$(git -C "$ROOT" remote get-url origin 2>/dev/null || true)"
  case "$remote" in
    git@github.com:*) remote="https://github.com/${remote#git@github.com:}" ;;
  esac
  PYWORKER_REPO="${remote%.git}"
  [ -n "$PYWORKER_REPO" ] || die "Set PYWORKER_REPO in $ENV_FILE (public Git URL of this repo)"
  echo "PYWORKER_REPO not set; using git remote: $PYWORKER_REPO"
fi

# ---- Helpers ----------------------------------------------------------------------------
redact() { sed -e "s/$VASTAI_KEY/<VASTAI_KEY>/g" ${HF_TOKEN:+-e "s/$HF_TOKEN/<HF_TOKEN>/g"}; }

# Run a vastai subcommand with the API key and JSON output; echo its output.
vast() {
  if [ "$DRY_RUN" = 1 ]; then
    { printf 'vastai'; printf ' %q' "$@"; printf ' --api-key KEY --raw\n'; } | redact >&2
    echo '{"dry_run": true}'
    return 0
  fi
  vastai "$@" --api-key "$VASTAI_KEY" --raw
}

# json_find KEY... : print the first value found (recursively) for any of the keys on stdin JSON.
json_find() {
  python3 -c '
import json, re, sys
raw = sys.stdin.read()
keys = sys.argv[1:]
def find(o):
    if isinstance(o, dict):
        for k in keys:
            if k in o and o[k] not in (None, "", {}, []) and not isinstance(o[k], (dict, list, bool)):
                return o[k]
        for v in o.values():
            r = find(v)
            if r is not None: return r
    elif isinstance(o, list):
        for v in o:
            r = find(v)
            if r is not None: return r
try:
    r = find(json.loads(raw))
except Exception:
    r = None
if r is None and "hash_id" in keys:
    m = re.search(r"\b[0-9a-f]{32}\b", raw)
    r = m.group(0) if m else None
if r is not None: print(r)
' "$@"
}

save_state() { [ "$DRY_RUN" = 1 ] || echo "$1=$2" >> "$STATE_FILE"; }

TEMPLATE_ENV="-p 3000:3000"
add_env() {  # add_env NAME VALUE  (single quotes are used for quoting, so they are not allowed in values)
  case "$2" in *"'"*) die "Value for $1 contains a single quote, which is not supported" ;; esac
  TEMPLATE_ENV+=" -e $1='$2'"
}

# Build the template's Docker-options env string from .env.
build_template_env() {
  add_env SLOTS "$SLOTS"
  add_env PYWORKER_REPO "$PYWORKER_REPO"
  [ "$HF_TOKEN_MODE" = template ] && add_env HF_TOKEN "$HF_TOKEN"
  add_env MODEL_SERVER_URL "http://127.0.0.1"
  add_env MODEL_SERVER_PORT "8080"
  # Pass only the variables you set in .env; the on-start script has defaults for the rest.
  for v in MODEL_REPO MODEL_PATTERNS MODEL_FILE MMPROJ_FILE EXTRA_ARGS KV_TYPE CTX_PER_SLOT \
           MODEL_NAME PYWORKER_REF PYWORKER_BOOTSTRAP_CMD; do
    if [ -n "${!v+x}" ]; then
      val="${!v}"
      case "$v" in MMPROJ_FILE|EXTRA_ARGS) [ -n "$val" ] || val="none" ;; esac
      if [ -n "$val" ]; then add_env "$v" "$val"; fi
    fi
  done
}

# ---- status -----------------------------------------------------------------------------
if [ "$CMD" = "status" ]; then
  [ -f "$STATE_FILE" ] && { echo "== saved state =="; cat "$STATE_FILE"; echo; }
  echo "== endpoints =="; vast show endpoints
  echo "== workergroups =="; vast show workergroups
  exit 0
fi

# ---- update -----------------------------------------------------------------------------
# Re-send the on-start script and env to the template already in .vast-state.
# `create` will not do this: it reuses the saved hash and skips template creation.
if [ "$CMD" = "update" ]; then
  [ -f "$ROOT/scripts/onstart.sh" ] || die "scripts/onstart.sh not found"
  [ -f "$STATE_FILE" ] && . "$STATE_FILE"
  [ -n "${TEMPLATE_HASH:-}" ] || die "No TEMPLATE_HASH in $STATE_FILE - run './scripts/setup.sh create --force' instead"
  build_template_env
  echo "Updating template $TEMPLATE_HASH (on-start script + env)"
  out="$(vast update template "$TEMPLATE_HASH" \
      --name "$TEMPLATE_NAME" --image "$IMAGE" --env "$TEMPLATE_ENV" \
      --onstart-cmd "$(cat "$ROOT/scripts/onstart.sh")" \
      --search_params "$SEARCH_PARAMS" --disk_space "$DISK_GB" --ssh --direct)"
  # Vast issues a NEW hash_id on every update (same numeric id) - any workergroup
  # must be created against the new hash, so track it in state.
  if [ "$DRY_RUN" = 1 ]; then
    echo "      (dry run)"
  else
    NEW_HASH="$(printf '%s' "$out" | json_find hash_id || true)"
    if [ -n "$NEW_HASH" ] && [ "$NEW_HASH" != "$TEMPLATE_HASH" ]; then
      grep -v '^TEMPLATE_HASH=' "$STATE_FILE" > "$STATE_FILE.tmp" 2>/dev/null || : > "$STATE_FILE.tmp"
      echo "TEMPLATE_HASH=$NEW_HASH" >> "$STATE_FILE.tmp"
      mv "$STATE_FILE.tmp" "$STATE_FILE"
      echo "      template hash: $TEMPLATE_HASH -> $NEW_HASH (state updated; use the new hash for workergroups)"
    else
      echo "      template hash: ${NEW_HASH:-$TEMPLATE_HASH}"
    fi
  fi
  cat <<UPDATE_DONE

Template updated. Existing workers keep the old on-start script until they recycle.
Next:
  ./scripts/setup.sh status              # confirm the endpoint/workergroup still exist
  vastai show instances                  # workers being started
UPDATE_DONE
  exit 0
fi

# ---- create -----------------------------------------------------------------------------
[ -f "$ROOT/scripts/onstart.sh" ] || die "scripts/onstart.sh not found"
[ "$FORCE" = 1 ] && rm -f "$STATE_FILE"
[ -f "$STATE_FILE" ] && . "$STATE_FILE"   # TEMPLATE_HASH / ENDPOINT_ID / WORKERGROUP_ID from earlier runs

# 1. HF token
case "$HF_TOKEN_MODE" in
  account)
    : "${HF_TOKEN:?HF_TOKEN is not set in $ENV_FILE}"
    echo "[1/4] Storing HF_TOKEN as a Vast account environment variable"
    vast create env-var HF_TOKEN "$HF_TOKEN" >/dev/null \
      || echo "      (could not create it; it may already exist. Check Account Settings > Environment Variables)"
    ;;
  template)
    : "${HF_TOKEN:?HF_TOKEN is not set in $ENV_FILE}"
    echo "[1/4] HF_TOKEN will be embedded in the template (visible to anyone with access to it)"
    ;;
  *) die "HF_TOKEN_MODE must be 'account' or 'template'" ;;
esac

# 2. Template
if [ -n "${TEMPLATE_HASH:-}" ]; then
  echo "[2/4] Reusing template $TEMPLATE_HASH (ran './scripts/setup.sh update' if you edited scripts/onstart.sh)"
else
  echo "[2/4] Creating template '$TEMPLATE_NAME'"
  build_template_env
  out="$(vast create template \
      --name "$TEMPLATE_NAME" --image "$IMAGE" --env "$TEMPLATE_ENV" \
      --onstart-cmd "$(cat "$ROOT/scripts/onstart.sh")" \
      --search_params "$SEARCH_PARAMS" --disk_space "$DISK_GB" --ssh --direct)"
  if [ "$DRY_RUN" = 1 ]; then TEMPLATE_HASH="<template_hash>"
  else
    TEMPLATE_HASH="$(printf '%s' "$out" | json_find hash_id hash || true)"
    [ -n "$TEMPLATE_HASH" ] || die "Could not find the template hash in the CLI output:
$out
Put the hash in .env as TEMPLATE_HASH=... and re-run to continue."
    save_state TEMPLATE_HASH "$TEMPLATE_HASH"
  fi
  echo "      template hash: $TEMPLATE_HASH"
fi

# 3. Endpoint
if [ -n "${ENDPOINT_ID:-}" ]; then
  echo "[3/4] Reusing endpoint $ENDPOINT_ID"
else
  echo "[3/4] Creating endpoint '$ENDPOINT_NAME'"
  out="$(vast create endpoint --endpoint_name "$ENDPOINT_NAME" --min_load "$MIN_LOAD" \
      --target_util "$TARGET_UTIL" --cold_mult "$COLD_MULT" --cold_workers "$COLD_WORKERS" \
      --max_workers "$MAX_WORKERS")"
  if [ "$DRY_RUN" = 1 ]; then ENDPOINT_ID="<endpoint_id>"
  else
    ENDPOINT_ID="$(printf '%s' "$out" | json_find result id endpoint_id || true)"
    [ -n "$ENDPOINT_ID" ] || die "Could not find the endpoint id in the CLI output:
$out"
    save_state ENDPOINT_ID "$ENDPOINT_ID"
  fi
  echo "      endpoint id: $ENDPOINT_ID"
fi

# 4. Workergroup
if [ -n "${WORKERGROUP_ID:-}" ]; then
  echo "[4/4] Reusing workergroup $WORKERGROUP_ID"
else
  echo "[4/4] Creating workergroup (test workers: $TEST_WORKERS)"
  out="$(vast create workergroup --endpoint_name "$ENDPOINT_NAME" --template_hash "$TEMPLATE_HASH" \
      --gpu_ram "$GPU_RAM")"
  if [ "$DRY_RUN" = 1 ]; then WORKERGROUP_ID="<workergroup_id>"
  else
    WORKERGROUP_ID="$(printf '%s' "$out" | json_find id result autogroup_id || true)"
    [ -n "$WORKERGROUP_ID" ] || echo "      (created, but could not read the id from: $out)"
    [ -z "$WORKERGROUP_ID" ] || save_state WORKERGROUP_ID "$WORKERGROUP_ID"
  fi
  echo "      workergroup id: ${WORKERGROUP_ID:-unknown}"
fi

DRY_NOTE=""; [ "$DRY_RUN" = 1 ] && DRY_NOTE=" (dry run: nothing was created)"
cat <<NEXT_END_1

Done${DRY_NOTE}. Next:
  ./scripts/setup.sh status            # endpoints and workergroups
  vastai show instances                # workers being started
The first worker builds llama.cpp and downloads the model, which takes a while.
Test workers rent GPUs while they benchmark, so watch your balance.
NEXT_END_1
