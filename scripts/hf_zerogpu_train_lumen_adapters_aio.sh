#!/usr/bin/env bash
set -Eeuo pipefail

IFS=$'\n\t'

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${LUMEN_ZERO_GPU_PYTHON:-python3}"
VENV="${LUMEN_ZERO_GPU_VENV:-$ROOT/.venv-hf-zerogpu}"
USE_ACTIVE_PYTHON="${LUMEN_ZERO_GPU_USE_ACTIVE_PYTHON:-0}"
SKIP_INSTALL="${LUMEN_ZERO_GPU_SKIP_INSTALL:-0}"

RUN_ID="${LUMEN_ZERO_GPU_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_ROOT="${LUMEN_ZERO_GPU_RUN_ROOT:-$ROOT/.local/hf_zerogpu_runs/$RUN_ID}"
DATASET_SOURCE="${LUMEN_ZERO_GPU_DATASET_SOURCE:-$ROOT/generated/agent_manifest/fine_tuning}"
SPACE_REPO="${LUMEN_ZERO_GPU_SPACE_REPO:-ales27pm/lumen-zerogpu-adapter-trainer}"
DATASET_REPO="${LUMEN_ZERO_GPU_DATASET_REPO:-ales27pm/lumen-zerogpu-training-datasets}"
ADAPTER_REPO="${LUMEN_ZERO_GPU_ADAPTER_REPO:-ales27pm/lumen-qwen3-bootstrap-adapters-gguf}"
AGENTS="${LUMEN_ZERO_GPU_AGENTS:-cortex,executor,mouth,mimicry,rem,fleet}"
BASE_MODEL="${LUMEN_ZERO_GPU_BASE_MODEL:-}"
GPU_SIZE="${LUMEN_ZERO_GPU_SIZE:-large}"
GPU_DURATION_SECONDS="${LUMEN_ZERO_GPU_DURATION_SECONDS:-3600}"
SEED="${LUMEN_ZERO_GPU_SEED:-42}"
TRIGGER="${LUMEN_ZERO_GPU_TRIGGER:-1}"
DRY_RUN="${LUMEN_ZERO_GPU_DRY_RUN:-0}"
PRIVATE_SPACE="${LUMEN_ZERO_GPU_PRIVATE_SPACE:-1}"
PRIVATE_DATASET="${LUMEN_ZERO_GPU_PRIVATE_DATASET:-1}"
PRIVATE_ADAPTERS="${LUMEN_ZERO_GPU_PRIVATE_ADAPTERS:-1}"

log() {
  printf '[lumen-zerogpu] %s\n' "$*"
}

die() {
  printf '[lumen-zerogpu] ERROR: %s\n' "$*" >&2
  exit 1
}

if [[ ! -d "$DATASET_SOURCE" && -d "$ROOT/generated/fine_tuning" ]]; then
  DATASET_SOURCE="$ROOT/generated/fine_tuning"
fi
[[ -d "$DATASET_SOURCE" ]] || die "missing fine-tuning dataset source: $DATASET_SOURCE"
[[ -f "$DATASET_SOURCE/adapter_runtime_manifest.json" ]] || die "dataset source is missing adapter_runtime_manifest.json: $DATASET_SOURCE"

if [[ "$USE_ACTIVE_PYTHON" == "1" ]]; then
  TRAIN_PY="$PYTHON_BIN"
else
  "$PYTHON_BIN" -m venv "$VENV"
  TRAIN_PY="$VENV/bin/python"
fi

if [[ "$SKIP_INSTALL" != "1" ]]; then
  log "installing/updating local HF automation dependencies"
  "$TRAIN_PY" -m pip install -U pip setuptools wheel
  "$TRAIN_PY" -m pip install -U huggingface_hub gradio_client
fi

ARGS=(
  --root "$ROOT"
  --run-id "$RUN_ID"
  --run-root "$RUN_ROOT"
  --dataset-source "$DATASET_SOURCE"
  --space-repo "$SPACE_REPO"
  --dataset-repo "$DATASET_REPO"
  --adapter-repo "$ADAPTER_REPO"
  --agents "$AGENTS"
  --gpu-size "$GPU_SIZE"
  --gpu-duration-seconds "$GPU_DURATION_SECONDS"
  --seed "$SEED"
)

if [[ -n "$BASE_MODEL" ]]; then
  ARGS+=(--base-model "$BASE_MODEL")
fi
if [[ "$TRIGGER" == "1" ]]; then
  ARGS+=(--trigger)
fi
if [[ "$DRY_RUN" == "1" ]]; then
  ARGS+=(--dry-run)
fi
if [[ "$PRIVATE_SPACE" == "1" ]]; then
  ARGS+=(--private-space)
fi
if [[ "$PRIVATE_DATASET" == "1" ]]; then
  ARGS+=(--private-dataset)
fi
if [[ "$PRIVATE_ADAPTERS" == "1" ]]; then
  ARGS+=(--private-adapters)
fi

log "run id: $RUN_ID"
log "run root: $RUN_ROOT"
log "space repo: $SPACE_REPO"
log "dataset repo: $DATASET_REPO"
log "adapter repo: $ADAPTER_REPO"
log "agents: $AGENTS"

"$TRAIN_PY" "$ROOT/tools/hf_zerogpu/build_lumen_zerogpu_space.py" "${ARGS[@]}"
