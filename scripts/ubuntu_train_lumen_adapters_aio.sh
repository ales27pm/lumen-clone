#!/usr/bin/env bash
set -Eeuo pipefail
umask 077

IFS=$'\n\t'

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
: "${LUMEN_AIO_EXPERIMENT_VARIANT:?Select one exact controlled experiment variant}"
: "${LUMEN_AIO_CONTAINER_IMAGE_DIGEST:?Pass the observed local Docker image ID}"

EXPERIMENT_VARIANT="$LUMEN_AIO_EXPERIMENT_VARIANT"
CONTAINER_IMAGE_DIGEST="$LUMEN_AIO_CONTAINER_IMAGE_DIGEST"
RUN_ID_BASE="${LUMEN_AIO_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
if [[ "$RUN_ID_BASE" == *"-${EXPERIMENT_VARIANT}" ]]; then
  RUN_ID="$RUN_ID_BASE"
else
  RUN_ID="${RUN_ID_BASE}-${EXPERIMENT_VARIANT}"
fi
DEFAULT_RUN_PARENT="$ROOT/.local/ubuntu_finetune_runs"
RUN_ROOT="${LUMEN_AIO_RUN_ROOT:-$DEFAULT_RUN_PARENT/$RUN_ID}"
ALLOWED_RUN_PARENT="${LUMEN_AIO_ALLOWED_RUN_PARENT:-$DEFAULT_RUN_PARENT}"
DATASET_SOURCE="${LUMEN_AIO_DATASET_SOURCE:-$ROOT/generated/fine_tuning}"
AGENTS_CSV="${LUMEN_AIO_AGENTS:-cortex,executor,mouth,mimicry,rem,fleet}"
BASE_MODEL_OVERRIDE="${LUMEN_AIO_BASE_MODEL:-}"
SEED="${LUMEN_AIO_SEED:-42}"
VENV="${LUMEN_AIO_VENV:-$ROOT/.venv-unsloth}"
PYTHON_BIN="${LUMEN_AIO_PYTHON_BIN:-${LUMEN_AIO_PYTHON:-python3.10}}"
TORCH_INDEX_URL="${LUMEN_AIO_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
ASSISTANT_ONLY_LOSS="${LUMEN_AIO_ASSISTANT_ONLY_LOSS:-1}"
RESUME="${LUMEN_AIO_RESUME:-0}"
SKIP_INSTALL="${LUMEN_AIO_SKIP_INSTALL:-0}"
USE_ACTIVE_PYTHON="${LUMEN_AIO_USE_ACTIVE_PYTHON:-0}"
REQUIRE_CUDA="${LUMEN_AIO_REQUIRE_CUDA:-1}"
FULL_PIPELINE="${LUMEN_AIO_FULL_PIPELINE:-1}"
RUN_PREFERENCE="${LUMEN_AIO_RUN_PREFERENCE:-$FULL_PIPELINE}"
EVALUATE="${LUMEN_AIO_EVALUATE:-$FULL_PIPELINE}"
EVAL_MAX_EXAMPLES="${LUMEN_AIO_EVAL_MAX_EXAMPLES:-}"
CONVERT_GGUF="${LUMEN_AIO_CONVERT_GGUF:-1}"
UPLOAD="${LUMEN_AIO_UPLOAD:-0}"
OVERWRITE="${LUMEN_AIO_OVERWRITE:-0}"
PREPARE_ONLY="${LUMEN_AIO_PREPARE_ONLY:-0}"
PREFLIGHT_ONLY="${LUMEN_AIO_PREFLIGHT_ONLY:-0}"
STATIC_PREFLIGHT="${LUMEN_AIO_STATIC_PREFLIGHT:-0}"

log() {
  printf '[lumen-aio] %s\n' "$*"
}

die() {
  printf '[lumen-aio] ERROR: %s\n' "$*" >&2
  exit 1
}

require_bool() {
  local name="$1"
  local value="$2"
  [[ "$value" == "0" || "$value" == "1" ]] || die "$name must be 0 or 1 (got: $value)"
}

for pair in \
  "LUMEN_AIO_ASSISTANT_ONLY_LOSS:$ASSISTANT_ONLY_LOSS" \
  "LUMEN_AIO_RESUME:$RESUME" \
  "LUMEN_AIO_SKIP_INSTALL:$SKIP_INSTALL" \
  "LUMEN_AIO_USE_ACTIVE_PYTHON:$USE_ACTIVE_PYTHON" \
  "LUMEN_AIO_REQUIRE_CUDA:$REQUIRE_CUDA" \
  "LUMEN_AIO_FULL_PIPELINE:$FULL_PIPELINE" \
  "LUMEN_AIO_RUN_PREFERENCE:$RUN_PREFERENCE" \
  "LUMEN_AIO_EVALUATE:$EVALUATE" \
  "LUMEN_AIO_CONVERT_GGUF:$CONVERT_GGUF" \
  "LUMEN_AIO_UPLOAD:$UPLOAD" \
  "LUMEN_AIO_OVERWRITE:$OVERWRITE" \
  "LUMEN_AIO_PREPARE_ONLY:$PREPARE_ONLY" \
  "LUMEN_AIO_PREFLIGHT_ONLY:$PREFLIGHT_ONLY" \
  "LUMEN_AIO_STATIC_PREFLIGHT:$STATIC_PREFLIGHT"; do
  require_bool "${pair%%:*}" "${pair#*:}"
done

case "$EXPERIMENT_VARIANT" in
  internal_only|internal_plus_public_baseline|internal_plus_public_optimized) ;;
  *) die "unsupported experiment variant: $EXPERIMENT_VARIANT" ;;
esac
[[ "$CONTAINER_IMAGE_DIGEST" =~ ^sha256:[0-9a-f]{64}$ ]] \
  || die "LUMEN_AIO_CONTAINER_IMAGE_DIGEST must be sha256:<64 lowercase hex characters>"
[[ "$SEED" =~ ^[0-9]+$ ]] || die "LUMEN_AIO_SEED must be a non-negative integer"
[[ "$RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,255}$ ]] || die "unsafe run ID: $RUN_ID"
[[ ! -L "$RUN_ROOT" ]] || die "run root must not be a symlink: $RUN_ROOT"
[[ "$AGENTS_CSV" =~ ^[a-z]+(,[a-z]+)*$ ]] || die "agents must be a lowercase comma-separated list"
IFS=',' read -r -a AGENTS <<< "$AGENTS_CSV"
if [[ -n "$EVAL_MAX_EXAMPLES" ]]; then
  [[ "$EVAL_MAX_EXAMPLES" =~ ^[1-9][0-9]*$ ]] || die "LUMEN_AIO_EVAL_MAX_EXAMPLES must be positive"
fi
[[ "$OVERWRITE" == "0" || "$RESUME" == "0" ]] || die "overwrite and resume are mutually exclusive"
[[ "$UPLOAD" == "0" ]] \
  || die "direct inner-loop upload is disabled; use ubuntu_train_lumen_full_pipeline.sh --upload"
[[ "$EVALUATE" == "0" || "$RUN_PREFERENCE" == "1" ]] || die "evaluation requires the finalized preference adapters"

if [[ ! -d "$DATASET_SOURCE" && -d "$ROOT/generated/agent_manifest/fine_tuning" ]]; then
  DATASET_SOURCE="$ROOT/generated/agent_manifest/fine_tuning"
fi
[[ -d "$DATASET_SOURCE" ]] || die "missing fine-tuning dataset source: $DATASET_SOURCE"
[[ -f "$DATASET_SOURCE/adapter_runtime_manifest.json" ]] \
  || die "dataset source is missing adapter_runtime_manifest.json: $DATASET_SOURCE"

STATIC_ARGS=(
  --root "$ROOT"
  --dataset-source "$DATASET_SOURCE"
  --agents "$AGENTS_CSV"
  --variant "$EXPERIMENT_VARIANT"
  --seed "$SEED"
  --base-model "$BASE_MODEL_OVERRIDE"
  --container-digest "$CONTAINER_IMAGE_DIGEST"
)

# This validation intentionally runs before mkdir, rm, venv creation, package
# installation, network access, or GPU access.
(
  cd "$ROOT"
  python3 -m tools.fine_tuning.unsloth.ubuntu_pipeline static-preflight \
    "${STATIC_ARGS[@]}" \
    --run-root "$RUN_ROOT" \
    --allowed-run-parent "$ALLOWED_RUN_PARENT" \
    --run-id "$RUN_ID"
)
if [[ "$STATIC_PREFLIGHT" == "1" ]]; then
  log "static preflight complete; no environment or run files were changed"
  exit 0
fi

if [[ "$USE_ACTIVE_PYTHON" == "1" ]]; then
  TRAIN_PY="$PYTHON_BIN"
else
  command -v "$PYTHON_BIN" >/dev/null 2>&1 || die "Python 3.10 not found: $PYTHON_BIN"
  "$PYTHON_BIN" - <<'PY'
import sys
if sys.version_info[:2] != (3, 10):
    raise SystemExit(f"Python 3.10 is required, got {sys.version.split()[0]}")
PY
  "$PYTHON_BIN" -m venv "$VENV"
  TRAIN_PY="$VENV/bin/python"
fi
[[ -x "$TRAIN_PY" ]] || die "training Python is not executable: $TRAIN_PY"
"$TRAIN_PY" - <<'PY'
import sys
if sys.version_info[:2] != (3, 10):
    raise SystemExit(f"Python 3.10 is required, got {sys.version.split()[0]}")
PY

if [[ "$SKIP_INSTALL" != "1" ]]; then
  log "installing the exact controlled Python environment"
  "$TRAIN_PY" -m pip install pip==26.1.1 setuptools==80.9.0 wheel==0.46.3
  "$TRAIN_PY" -m pip install \
    torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 \
    --index-url "$TORCH_INDEX_URL"
  "$TRAIN_PY" -m pip install -r "$ROOT/tools/hf_zerogpu/space_template/requirements.txt"
  "$TRAIN_PY" -m pip check
else
  log "using the prebuilt Python environment: $TRAIN_PY"
fi

if [[ "$REQUIRE_CUDA" == "1" ]]; then
  command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi not found"
  "$TRAIN_PY" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available in PyTorch")
if torch.version.cuda != "12.8":
    raise SystemExit(f"CUDA 12.8 is required, got {torch.version.cuda or '<none>'}")
print(f"CUDA ready: {torch.cuda.get_device_name(0)}")
PY
fi

(
  cd "$ROOT"
  "$TRAIN_PY" -m tools.fine_tuning.unsloth.ubuntu_pipeline runtime-preflight \
    "${STATIC_ARGS[@]}"
)
if [[ "$PREFLIGHT_ONLY" == "1" ]]; then
  log "runtime preflight complete; no run directory was created"
  exit 0
fi

command -v flock >/dev/null 2>&1 || die "flock is required (install util-linux)"
mkdir -p "$ALLOWED_RUN_PARENT"
LOCK_DIR="$ALLOWED_RUN_PARENT/.lumen-training.lock"
if ! mkdir -m 700 "$LOCK_DIR" 2>/dev/null; then
  [[ -d "$LOCK_DIR" && ! -L "$LOCK_DIR" ]] \
    || die "training lock path is not a regular directory: $LOCK_DIR"
fi
exec 9<"$LOCK_DIR"
flock -n 9 || die "another Lumen training pipeline holds $LOCK_DIR"

log "repo root: $ROOT"
log "run root: $RUN_ROOT"
log "dataset source: $DATASET_SOURCE"
log "agents: $AGENTS_CSV"
log "experiment variant: $EXPERIMENT_VARIANT"

if [[ "$RESUME" == "1" ]]; then
  [[ -d "$RUN_ROOT" ]] || die "resume run root does not exist: $RUN_ROOT"
  (
    cd "$ROOT"
    "$TRAIN_PY" -m tools.fine_tuning.unsloth.ubuntu_pipeline validate-prepared-runtime \
      --root "$ROOT" \
      --run-root "$RUN_ROOT" \
      --agents "$AGENTS_CSV" \
      --variant "$EXPERIMENT_VARIANT" \
      --container-digest "$CONTAINER_IMAGE_DIGEST"
  )
  log "resume mode skips verified complete phases and restarts incomplete phases"
else
  if [[ -e "$RUN_ROOT" ]]; then
    [[ "$OVERWRITE" == "1" ]] || die "run root already exists: $RUN_ROOT"
    [[ -d "$RUN_ROOT" && ! -L "$RUN_ROOT" ]] \
      || die "refusing to overwrite a non-directory or symlink run root: $RUN_ROOT"
    (
      cd "$ROOT"
      "$TRAIN_PY" -m tools.fine_tuning.unsloth.ubuntu_pipeline verify-owned-run \
        --run-root "$RUN_ROOT" --variant "$EXPERIMENT_VARIANT"
    )
    rm -rf -- "$RUN_ROOT"
  fi
  (
    cd "$ROOT"
    "$TRAIN_PY" -m tools.fine_tuning.unsloth.ubuntu_pipeline prepare \
      "${STATIC_ARGS[@]}" \
      --run-root "$RUN_ROOT"
  )
  (
    cd "$ROOT"
    "$TRAIN_PY" -m tools.fine_tuning.unsloth.ubuntu_pipeline validate-prepared-runtime \
      --root "$ROOT" \
      --run-root "$RUN_ROOT" \
      --agents "$AGENTS_CSV" \
      --variant "$EXPERIMENT_VARIANT" \
      --container-digest "$CONTAINER_IMAGE_DIGEST"
  )
fi

if [[ "$PREPARE_ONLY" == "1" ]]; then
  log "prepared run manifest: $RUN_ROOT/aio_run_manifest.json"
  exit 0
fi

CONVERTER_REPO=""
CONVERTER=""
GGUF_READER=""
if [[ "$CONVERT_GGUF" == "1" ]]; then
  LLAMA_CPP_REVISION="$(
    cd "$ROOT"
    "$TRAIN_PY" - <<'PY'
from tools.fine_tuning.unsloth.training_lineage import DEFAULT_LLAMA_CPP_REVISION

print(DEFAULT_LLAMA_CPP_REVISION)
PY
  )"
  [[ "$LLAMA_CPP_REVISION" =~ ^[0-9a-f]{40}$ ]] \
    || die "training_lineage supplied an invalid llama.cpp revision"
  CONVERTER_REPO="$RUN_ROOT/llama.cpp"
  CONVERTER="$CONVERTER_REPO/convert_lora_to_gguf.py"
  GGUF_READER="$CONVERTER_REPO/gguf-py/gguf/scripts/gguf_dump.py"
  [[ ! -L "$CONVERTER_REPO" ]] \
    || die "converter checkout path is unsafe: $CONVERTER_REPO"
  if [[ -e "$CONVERTER_REPO" && ! -d "$CONVERTER_REPO" ]]; then
    die "converter checkout path is not a directory: $CONVERTER_REPO"
  fi
  if [[ ! -d "$CONVERTER_REPO" ]]; then
    log "fetching the pinned llama.cpp LoRA converter"
    git init "$CONVERTER_REPO"
    git -C "$CONVERTER_REPO" remote add origin https://github.com/ggml-org/llama.cpp
    git -C "$CONVERTER_REPO" fetch --depth 1 origin "$LLAMA_CPP_REVISION"
    git -C "$CONVERTER_REPO" checkout --detach FETCH_HEAD
  fi
  [[ -f "$CONVERTER" && ! -L "$CONVERTER" ]] \
    || die "missing regular convert_lora_to_gguf.py"
  [[ -f "$GGUF_READER" && ! -L "$GGUF_READER" ]] \
    || die "missing regular pinned llama.cpp GGUF reader"
  [[ "$(git -C "$CONVERTER_REPO" rev-parse HEAD)" == "$LLAMA_CPP_REVISION" ]] \
    || die "llama.cpp converter revision does not match the controlled environment"
  [[ -z "$(git -C "$CONVERTER_REPO" status --porcelain=v1 --untracked-files=all)" ]] \
    || die "llama.cpp converter checkout is dirty"
  [[ "$(git -C "$CONVERTER_REPO" hash-object "$CONVERTER")" == \
      "$(git -C "$CONVERTER_REPO" rev-parse "HEAD:convert_lora_to_gguf.py")" ]] \
    || die "llama.cpp converter file drifted from the pinned revision"
  log "preflighting the pinned llama.cpp LoRA converter"
  "$TRAIN_PY" "$CONVERTER" --help >/dev/null
  log "preflighting the pinned llama.cpp GGUF reader"
  "$TRAIN_PY" "$GGUF_READER" --help >/dev/null
fi

verify_phase() {
  local agent="$1"
  local phase="$2"
  (
    cd "$ROOT"
    "$TRAIN_PY" -m tools.fine_tuning.unsloth.ubuntu_pipeline verify-phase \
      --run-root "$RUN_ROOT" --agent "$agent" --phase "$phase"
  )
}

clean_phase() {
  local agent="$1"
  local phase="$2"
  (
    cd "$ROOT"
    "$TRAIN_PY" -m tools.fine_tuning.unsloth.ubuntu_pipeline clean-phase \
      --run-root "$RUN_ROOT" --agent "$agent" --phase "$phase"
  )
}

TRAIN_ARGS=()
if [[ "$ASSISTANT_ONLY_LOSS" == "1" ]]; then
  TRAIN_ARGS+=(--assistant-only-loss)
fi

for agent in "${AGENTS[@]}"; do
  [[ -n "$agent" ]] || continue
  if [[ "$RESUME" == "1" ]] && verify_phase "$agent" sft >/dev/null 2>&1; then
    log "verified existing SFT phase: $agent"
  else
    if [[ "$RESUME" == "1" ]]; then
      clean_phase "$agent" sft
    fi
    log "training SFT adapter: $agent"
    (
      cd "$ROOT"
      "$TRAIN_PY" -m tools.fine_tuning.unsloth.train_sft \
        --config "$RUN_ROOT/configs/$agent.json" \
        --seed "$SEED" \
        "${TRAIN_ARGS[@]}"
    ) 2>&1 | tee "$RUN_ROOT/logs/train_sft_$agent.log"
    verify_phase "$agent" sft
  fi

  if [[ "$RUN_PREFERENCE" == "1" ]]; then
    if [[ "$RESUME" == "1" ]] && verify_phase "$agent" preference >/dev/null 2>&1; then
      log "verified existing preference phase: $agent"
    else
      if [[ "$RESUME" == "1" ]]; then
        clean_phase "$agent" preference
      fi
      log "training preference adapter: $agent"
      (
        cd "$ROOT"
        "$TRAIN_PY" -m tools.fine_tuning.unsloth.train_dpo \
          --config "$RUN_ROOT/configs/$agent.json" \
          --sft-adapter-dir "$RUN_ROOT/models/lora_qwen3_bootstrap/$agent" \
          --sft-finalized-variant-manifest \
            "$RUN_ROOT/training/$agent/finalized_variant_manifest.json"
      ) 2>&1 | tee "$RUN_ROOT/logs/train_preference_$agent.log"
      verify_phase "$agent" preference
    fi
  fi
done

if [[ "$EVALUATE" == "1" ]]; then
  for agent in "${AGENTS[@]}"; do
    [[ -n "$agent" ]] || continue
    log "preparing frozen evaluation lineage: $agent"
    (
      cd "$ROOT"
      "$TRAIN_PY" -m tools.fine_tuning.unsloth.ubuntu_pipeline write-final-config \
        --run-root "$RUN_ROOT" --agent "$agent"
    )
    eval_args=(
      --config "$RUN_ROOT/configs/$agent.final.json"
      --eval-jsonl "$RUN_ROOT/generated/fine_tuning/$agent/eval.jsonl"
      --behavior-manifest "$RUN_ROOT/generated/agent_manifest/AgentBehaviorManifest.json"
      --output-dir "$RUN_ROOT/evaluation/$agent"
    )
    if [[ -n "$EVAL_MAX_EXAMPLES" ]]; then
      eval_args+=(--max-examples "$EVAL_MAX_EXAMPLES")
    fi
    if [[ "$RESUME" == "1" ]] && compgen -G "$RUN_ROOT/evaluation/$agent/*" >/dev/null; then
      log "replacing prior evaluation outputs: $agent"
      eval_args+=(--overwrite)
    fi
    log "running frozen evaluation: $agent"
    (
      cd "$ROOT"
      "$TRAIN_PY" -m tools.fine_tuning.unsloth.evaluate_adapter "${eval_args[@]}"
    ) 2>&1 | tee "$RUN_ROOT/logs/evaluate_$agent.log"
  done
fi

if [[ "$CONVERT_GGUF" == "1" ]]; then
  for agent in "${AGENTS[@]}"; do
    [[ -n "$agent" ]] || continue
    outfile="$RUN_ROOT/models/lora_qwen3_gguf/lumen-$agent-lora.gguf"
    if [[ "$RESUME" == "1" ]] && (
      cd "$ROOT"
      "$TRAIN_PY" -m tools.fine_tuning.unsloth.ubuntu_pipeline verify-gguf \
        --run-root "$RUN_ROOT" --agent "$agent"
    ) >/dev/null 2>&1; then
      log "verified existing adapter GGUF: $agent"
      continue
    fi
    base_model="$("$TRAIN_PY" - "$RUN_ROOT/configs/$agent.json" <<'PY'
import hashlib
import json
import os
import sys
from huggingface_hub import snapshot_download

cfg = json.loads(open(sys.argv[1], encoding="utf-8").read())
snapshot = snapshot_download(repo_id=cfg["base_model_name"], revision=cfg["baseModelRevision"])

def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

for filename, expected in (
    ("model.safetensors.index.json", cfg["baseModelIndexDigest"]),
    ("tokenizer.json", cfg["baseModelTokenizerDigest"]),
):
    if sha256(f"{snapshot}/{filename}") != expected:
        raise SystemExit(f"Pinned base-model artifact mismatch: {filename}")
shards = sorted(cfg["baseModelWeightShards"], key=lambda item: item["filename"])
contract = {"schemaVersion": "lumen.base-model-weight-shards/1.0.0", "shards": shards}
if hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()).hexdigest() != cfg["baseModelArtifactDigest"]:
    raise SystemExit("Base-model shard contract mismatch")
index = json.load(open(f"{snapshot}/model.safetensors.index.json", encoding="utf-8"))
referenced = sorted(set(index.get("weight_map", {}).values()))
if referenced != [item["filename"] for item in shards] or referenced != cfg["baseModelIndexReferencedShardNames"]:
    raise SystemExit("Base-model index shard set mismatch")
for item in shards:
    path = f"{snapshot}/{item['filename']}"
    if os.path.getsize(path) != item["size"] or sha256(path) != item["sha256"]:
        raise SystemExit(f"Pinned base-model shard mismatch: {item['filename']}")
print(snapshot)
PY
)"
    if [[ "$RUN_PREFERENCE" == "1" ]]; then
      adapter_dir="$RUN_ROOT/models/lora_qwen3_dpo/$agent"
    else
      adapter_dir="$RUN_ROOT/models/lora_qwen3_bootstrap/$agent"
    fi
    rm -f -- "$outfile"
    log "converting finalized adapter to GGUF: $agent"
    "$TRAIN_PY" "$CONVERTER" "$adapter_dir" \
      --outfile "$outfile" \
      --base "$base_model" \
      2>&1 | tee "$RUN_ROOT/logs/convert_$agent.log"
    (
      cd "$ROOT"
      "$TRAIN_PY" -m tools.fine_tuning.unsloth.ubuntu_pipeline \
        verify-gguf-file --run-root "$RUN_ROOT" --path "$outfile"
    )
  done
fi

summary_args=(
  --run-root "$RUN_ROOT"
  --agents "$AGENTS_CSV"
  --variant "$EXPERIMENT_VARIANT"
)
if [[ "$RUN_PREFERENCE" == "1" ]]; then
  summary_args+=(--preference)
fi
if [[ "$CONVERT_GGUF" == "1" ]]; then
  summary_args+=(--require-gguf)
fi
if [[ "$EVALUATE" == "1" ]]; then
  summary_args+=(--require-evaluation)
fi
(
  cd "$ROOT"
  "$TRAIN_PY" -m tools.fine_tuning.unsloth.ubuntu_pipeline write-summary \
    "${summary_args[@]}"
)

log "done"
log "summary: $RUN_ROOT/aio_summary.json"
