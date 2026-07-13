#!/usr/bin/env bash
set -Eeuo pipefail

IFS=$'\n\t'

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${LUMEN_ZERO_GPU_PYTHON:-python3}"
VENV="${LUMEN_ZERO_GPU_VENV:-$ROOT/.venv-hf-zerogpu}"
USE_ACTIVE_PYTHON="${LUMEN_ZERO_GPU_USE_ACTIVE_PYTHON:-0}"
SKIP_INSTALL="${LUMEN_ZERO_GPU_SKIP_INSTALL:-0}"

: "${LUMEN_ZERO_GPU_EXPERIMENT_VARIANT:?Select an explicit experiment variant}"
: "${LUMEN_ZERO_GPU_CONTAINER_IMAGE_DIGEST:?Declare the intended training container image sha256 digest for manual verification}"
: "${LUMEN_ZERO_GPU_ADMIN_TOKEN:?Set a dedicated administrative token for the training endpoint}"
: "${LUMEN_ZERO_GPU_HUB_TOKEN:?Set a fine-grained Hub token scoped to the required Space, dataset, and adapter repositories}"
EXPERIMENT_VARIANT="$LUMEN_ZERO_GPU_EXPERIMENT_VARIANT"
CONTAINER_IMAGE_DIGEST="$LUMEN_ZERO_GPU_CONTAINER_IMAGE_DIGEST"
RUN_ID_BASE="${LUMEN_ZERO_GPU_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
if [[ "$RUN_ID_BASE" == *"-${EXPERIMENT_VARIANT}" ]]; then
  RUN_ID="$RUN_ID_BASE"
else
  RUN_ID="${RUN_ID_BASE}-${EXPERIMENT_VARIANT}"
fi
RUN_ROOT_BASE="${LUMEN_ZERO_GPU_RUN_ROOT_BASE:-$ROOT/.local/hf_zerogpu_runs}"
RUN_ROOT="${LUMEN_ZERO_GPU_RUN_ROOT:-$RUN_ROOT_BASE/$RUN_ID}"
DATASET_SOURCE="${LUMEN_ZERO_GPU_DATASET_SOURCE:-$ROOT/generated/agent_manifest/fine_tuning}"
SPACE_REPO="${LUMEN_ZERO_GPU_SPACE_REPO:-ales27pm/lumen-zerogpu-adapter-trainer}"
DATASET_REPO="${LUMEN_ZERO_GPU_DATASET_REPO:-ales27pm/lumen-zerogpu-training-datasets}"
ADAPTER_REPO="${LUMEN_ZERO_GPU_ADAPTER_REPO:-ales27pm/lumen-qwen3-bootstrap-adapters-gguf}"
AGENTS="${LUMEN_ZERO_GPU_AGENTS:-cortex,executor,mouth,mimicry,rem,fleet}"
AGENT_BATCH_SIZE="${LUMEN_ZERO_GPU_AGENT_BATCH_SIZE:-1}"
BASE_MODEL="${LUMEN_ZERO_GPU_BASE_MODEL:-}"
GPU_SIZE="${LUMEN_ZERO_GPU_SIZE:-large}"
GPU_DURATION_SECONDS="${LUMEN_ZERO_GPU_DURATION_SECONDS:-1200}"
TRIGGER_TIMEOUT_SECONDS="${LUMEN_ZERO_GPU_TRIGGER_TIMEOUT_SECONDS:-1800}"
SEED="${LUMEN_ZERO_GPU_SEED:-42}"
TRIGGER="${LUMEN_ZERO_GPU_TRIGGER:-1}"
DRY_RUN="${LUMEN_ZERO_GPU_DRY_RUN:-0}"
PUBLIC_SPACE="${LUMEN_ZERO_GPU_PUBLIC_SPACE:-0}"
PRIVATE_DATASET="${LUMEN_ZERO_GPU_PRIVATE_DATASET:-1}"
PRIVATE_ADAPTERS="${LUMEN_ZERO_GPU_PRIVATE_ADAPTERS:-1}"
DESTRUCTIVE_RESET="${LUMEN_ZERO_GPU_DESTRUCTIVE_RESET:-0}"
RESUME="${LUMEN_ZERO_GPU_RESUME:-0}"

log() {
  printf '[lumen-zerogpu] %s\n' "$*"
}

die() {
  printf '[lumen-zerogpu] ERROR: %s\n' "$*" >&2
  exit 1
}

join_by_comma() {
  local IFS=","
  printf '%s' "$*"
}

sanitize_batch_label() {
  printf '%s' "$1" | tr ',' '-' | tr -c '[:alnum:]_-' '-'
}

run_training_batch() {
  local batch_index="$1"
  local batch_count="$2"
  local batch_agents="$3"
  local batch_run_id="$RUN_ID"
  local batch_run_root="$RUN_ROOT"
  if (( batch_count > 1 )); then
    local batch_label
    batch_label="$(sanitize_batch_label "$batch_agents")"
    batch_run_id="${RUN_ID}-b$(printf '%02d' "$batch_index")-${batch_label}"
    batch_run_root="$RUN_ROOT/$batch_run_id"
  fi

  local args=(
    --root "$ROOT"
    --run-id "$batch_run_id"
    --run-root "$batch_run_root"
    --dataset-source "$DATASET_SOURCE"
    --space-repo "$SPACE_REPO"
    --dataset-repo "$DATASET_REPO"
    --adapter-repo "$ADAPTER_REPO"
    --agents "$batch_agents"
    --gpu-size "$GPU_SIZE"
    --gpu-duration-seconds "$GPU_DURATION_SECONDS"
    --trigger-timeout-seconds "$TRIGGER_TIMEOUT_SECONDS"
    --seed "$SEED"
    --experiment-variant "$EXPERIMENT_VARIANT"
    --container-image-digest "$CONTAINER_IMAGE_DIGEST"
  )

  if [[ -n "$BASE_MODEL" ]]; then
    args+=(--base-model "$BASE_MODEL")
  fi
  if [[ "$TRIGGER" == "1" ]]; then
    args+=(--trigger)
  fi
  if [[ "$DRY_RUN" == "1" ]]; then
    args+=(--dry-run)
  fi
  if [[ "$PUBLIC_SPACE" == "1" ]]; then
    args+=(--public-space)
  fi
  if [[ "$PRIVATE_DATASET" == "1" ]]; then
    args+=(--private-dataset)
  fi
  if [[ "$PRIVATE_ADAPTERS" == "1" ]]; then
    args+=(--private-adapters)
  fi
  if [[ "$DESTRUCTIVE_RESET" == "1" ]]; then
    args+=(--destructive-reset)
  fi
  if [[ "$RESUME" == "1" ]]; then
    args+=(--resume)
  fi

  log "batch $batch_index/$batch_count run id: $batch_run_id"
  log "batch $batch_index/$batch_count run root: $batch_run_root"
  log "batch $batch_index/$batch_count agents: $batch_agents"

  "$TRAIN_PY" "$ROOT/tools/hf_zerogpu/build_lumen_zerogpu_space.py" "${args[@]}"
}

if [[ "$RESUME" != "1" ]]; then
  if [[ ! -d "$DATASET_SOURCE" && -d "$ROOT/generated/fine_tuning" ]]; then
    DATASET_SOURCE="$ROOT/generated/fine_tuning"
  fi
  [[ -d "$DATASET_SOURCE" ]] || die "missing fine-tuning dataset source: $DATASET_SOURCE"
  [[ -f "$DATASET_SOURCE/adapter_runtime_manifest.json" ]] || die "dataset source is missing adapter_runtime_manifest.json: $DATASET_SOURCE"
fi
case "$EXPERIMENT_VARIANT" in
  internal_only|internal_plus_public_baseline|internal_plus_public_optimized) ;;
  *) die "unsupported experiment variant: $EXPERIMENT_VARIANT" ;;
esac
[[ "$CONTAINER_IMAGE_DIGEST" =~ ^sha256:[0-9a-f]{64}$ ]] || die "LUMEN_ZERO_GPU_CONTAINER_IMAGE_DIGEST must be sha256:<64 lowercase hex characters>"
(( ${#LUMEN_ZERO_GPU_ADMIN_TOKEN} >= 32 )) || die "LUMEN_ZERO_GPU_ADMIN_TOKEN must contain at least 32 characters"
[[ "$RESUME" != "1" || "$DESTRUCTIVE_RESET" != "1" ]] || die "resume and destructive reset are mutually exclusive"
[[ "$AGENT_BATCH_SIZE" =~ ^[0-9]+$ ]] || die "LUMEN_ZERO_GPU_AGENT_BATCH_SIZE must be a positive integer"
(( AGENT_BATCH_SIZE > 0 )) || die "LUMEN_ZERO_GPU_AGENT_BATCH_SIZE must be greater than zero"
[[ "$TRIGGER_TIMEOUT_SECONDS" =~ ^[0-9]+$ ]] || die "LUMEN_ZERO_GPU_TRIGGER_TIMEOUT_SECONDS must be a positive integer"
(( TRIGGER_TIMEOUT_SECONDS > 0 )) || die "LUMEN_ZERO_GPU_TRIGGER_TIMEOUT_SECONDS must be greater than zero"

if [[ "$USE_ACTIVE_PYTHON" == "1" ]]; then
  TRAIN_PY="$PYTHON_BIN"
else
  "$PYTHON_BIN" -m venv "$VENV"
  TRAIN_PY="$VENV/bin/python"
fi

if [[ "$SKIP_INSTALL" != "1" ]]; then
  log "installing/updating local HF automation dependencies"
  "$TRAIN_PY" -m pip install \
    pip==26.1.1 setuptools==80.9.0 wheel==0.46.3 \
    huggingface_hub==1.23.0 gradio_client==2.5.0
fi

log "run id: $RUN_ID"
log "run root: $RUN_ROOT"
log "space repo: $SPACE_REPO"
log "dataset repo: $DATASET_REPO"
log "adapter repo: $ADAPTER_REPO"
log "agents: $AGENTS"
log "agent batch size: $AGENT_BATCH_SIZE"
log "experiment variant: $EXPERIMENT_VARIANT"
log "trigger timeout seconds: $TRIGGER_TIMEOUT_SECONDS"

IFS=',' read -r -a AGENT_ARRAY <<< "$AGENTS"
TOTAL_AGENTS="${#AGENT_ARRAY[@]}"
(( TOTAL_AGENTS > 0 )) || die "no agents selected"
BATCH_COUNT=$(( (TOTAL_AGENTS + AGENT_BATCH_SIZE - 1) / AGENT_BATCH_SIZE ))

for (( batch_index = 1, start = 0; start < TOTAL_AGENTS; batch_index++, start += AGENT_BATCH_SIZE )); do
  batch_agents=()
  for (( offset = 0; offset < AGENT_BATCH_SIZE && start + offset < TOTAL_AGENTS; offset++ )); do
    agent="${AGENT_ARRAY[start + offset]}"
    agent="${agent//[[:space:]]/}"
    [[ -n "$agent" ]] && batch_agents+=("$agent")
  done
  (( ${#batch_agents[@]} > 0 )) || continue
  run_training_batch "$batch_index" "$BATCH_COUNT" "$(join_by_comma "${batch_agents[@]}")"
done
