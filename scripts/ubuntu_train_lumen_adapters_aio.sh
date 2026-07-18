#!/usr/bin/env bash
set -Eeuo pipefail
umask 077

IFS=$'\n\t'

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$ROOT"
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
AGENTS_CSV="${LUMEN_AIO_AGENTS:-fleet,executor,mouth,rem,mimicry,cortex}"
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
CONVERT_GGUF="${LUMEN_AIO_CONVERT_GGUF:-$FULL_PIPELINE}"
UPLOAD="${LUMEN_AIO_UPLOAD:-0}"
OVERWRITE="${LUMEN_AIO_OVERWRITE:-0}"
LAUNCH_MODE="${LUMEN_AIO_LAUNCH_MODE:-}"
PREPARE_ONLY="${LUMEN_AIO_PREPARE_ONLY:-0}"
PREFLIGHT_ONLY="${LUMEN_AIO_PREFLIGHT_ONLY:-0}"
STATIC_PREFLIGHT="${LUMEN_AIO_STATIC_PREFLIGHT:-0}"
PRECREATED_BIND_ROOT="${LUMEN_AIO_PRECREATED_BIND_ROOT:-0}"
EXPECTED_RUN_ROOT_IDENTITY="${LUMEN_AIO_EXPECTED_RUN_ROOT_IDENTITY:-}"
LOCK_DIR="${LUMEN_AIO_LOCK_DIR:-$ALLOWED_RUN_PARENT/.lumen-training.lock}"
EXPECTED_LOCK_IDENTITY="${LUMEN_AIO_EXPECTED_LOCK_IDENTITY:-}"
PREPARATION_OWNER_FILENAME=".lumen-preparation-owner.json"

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
require_bool "LUMEN_AIO_PRECREATED_BIND_ROOT" "$PRECREATED_BIND_ROOT"

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
[[ "$EVALUATE" != "0" || -z "$EVAL_MAX_EXAMPLES" ]] \
  || die "disabled evaluation cannot retain a smoke cohort size"
[[ "$OVERWRITE" == "0" || "$RESUME" == "0" ]] || die "overwrite and resume are mutually exclusive"
if [[ -z "$LAUNCH_MODE" ]]; then
  LAUNCH_MODE=fresh
  if [[ "$OVERWRITE" == "1" ]]; then
    LAUNCH_MODE=overwrite
  elif [[ "$RESUME" == "1" ]]; then
    LAUNCH_MODE=resume
  fi
fi
case "$LAUNCH_MODE:$OVERWRITE:$RESUME" in
  fresh:0:0|resume:0:1|overwrite:1:0) ;;
  *) die "launch mode does not match overwrite/resume controls" ;;
esac
[[ "$UPLOAD" == "0" ]] \
  || die "direct inner-loop upload is disabled; use ubuntu_train_lumen_full_pipeline.sh --upload"
[[ "$EVALUATE" == "0" || "$RUN_PREFERENCE" == "1" ]] || die "evaluation requires the finalized preference adapters"
[[ "$CONVERT_GGUF" == "0" || "$RUN_PREFERENCE" == "1" ]] \
  || die "GGUF conversion receipts require finalized preference adapters"
if [[ "$PRECREATED_BIND_ROOT" == "1" ]]; then
  [[ "$EXPECTED_RUN_ROOT_IDENTITY" =~ ^[0-9]+:[0-9]+:[0-9]+:[0-9]+:0700$ ]] \
    || die "precreated bind-root mode requires an exact expected root identity"
  [[ "$LOCK_DIR" == /* ]] || die "precreated bind-root mode requires an absolute lock path"
  [[ "$EXPECTED_LOCK_IDENTITY" =~ ^[0-9]+:[0-9]+:[0-9]+:[0-9]+:0700$ ]] \
    || die "precreated bind-root mode requires an exact lock identity"
else
  [[ -z "$EXPECTED_RUN_ROOT_IDENTITY" ]] \
    || die "a run-root identity is valid only in precreated bind-root mode"
  [[ -z "$EXPECTED_LOCK_IDENTITY" ]] \
    || die "a lock identity is valid only in precreated bind-root mode"
fi

if [[ "$EVALUATE" == "0" ]]; then
  EVALUATION_SCOPE="none"
elif [[ -n "$EVAL_MAX_EXAMPLES" ]]; then
  EVALUATION_SCOPE="smoke"
else
  EVALUATION_SCOPE="full"
fi
EXECUTION_PLAN_ARGS=(--evaluation-scope "$EVALUATION_SCOPE")
if [[ "$EVALUATION_SCOPE" == "smoke" ]]; then
  EXECUTION_PLAN_ARGS+=(--evaluation-max-examples "$EVAL_MAX_EXAMPLES")
fi
if [[ "$CONVERT_GGUF" == "1" ]]; then
  EXECUTION_PLAN_ARGS+=(--gguf-requested)
else
  EXECUTION_PLAN_ARGS+=(--no-gguf-requested)
fi

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
PRECREATED_ARGS=()
if [[ "$PRECREATED_BIND_ROOT" == "1" ]]; then
  PRECREATED_ARGS+=(--precreated-bind-root)
  (
    cd "$ROOT"
    python3 -m tools.fine_tuning.unsloth.ubuntu_pipeline verify-bind-root \
      --run-root "$RUN_ROOT" \
      --allowed-run-parent "$ALLOWED_RUN_PARENT" \
      --expected-identity "$EXPECTED_RUN_ROOT_IDENTITY" \
      --mounted-bind >/dev/null
    python3 -m tools.fine_tuning.unsloth.ubuntu_pipeline verify-bind-root \
      --run-root "$LOCK_DIR" \
      --allowed-run-parent "$(dirname "$LOCK_DIR")" \
      --expected-identity "$EXPECTED_LOCK_IDENTITY" \
      --mounted-bind >/dev/null
  )
fi

# This validation intentionally runs before mkdir, rm, venv creation, package
# installation, network access, or GPU access.
(
  cd "$ROOT"
  python3 -m tools.fine_tuning.unsloth.ubuntu_pipeline static-preflight \
    "${STATIC_ARGS[@]}" \
    "${EXECUTION_PLAN_ARGS[@]}" \
    --run-root "$RUN_ROOT" \
    --allowed-run-parent "$ALLOWED_RUN_PARENT" \
    --run-id "$RUN_ID" \
    "${PRECREATED_ARGS[@]}"
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
if [[ "$PRECREATED_BIND_ROOT" != "1" ]]; then
  mkdir -p "$ALLOWED_RUN_PARENT"
  if ! mkdir -m 700 "$LOCK_DIR" 2>/dev/null; then
    [[ -d "$LOCK_DIR" && ! -L "$LOCK_DIR" ]] \
      || die "training lock path is not a regular directory: $LOCK_DIR"
  fi
fi
[[ -d "$LOCK_DIR" && ! -L "$LOCK_DIR" ]] \
  || die "training lock path is not a regular directory: $LOCK_DIR"
[[ "$(stat -c '%u:%g:%a' "$LOCK_DIR")" == "$EUID:$(id -g):700" ]] \
  || die "training lock path must be process-owned mode 0700: $LOCK_DIR"
exec 9<"$LOCK_DIR"
flock -n 9 || die "another Lumen training pipeline holds $LOCK_DIR"

log "repo root: $ROOT"
log "run root: $RUN_ROOT"
log "dataset source: $DATASET_SOURCE"
log "agents: $AGENTS_CSV"
log "experiment variant: $EXPERIMENT_VARIANT"

prepare_run_inputs() {
  (
    cd "$ROOT"
    "$TRAIN_PY" -m tools.fine_tuning.unsloth.ubuntu_pipeline prepare \
      "${STATIC_ARGS[@]}" \
      --run-root "$RUN_ROOT" \
      "${PRECREATED_ARGS[@]}" \
      "${EXECUTION_PLAN_ARGS[@]}"
  )
}

recover_incomplete_preparation() {
  (
    cd "$ROOT"
    "$TRAIN_PY" -m tools.fine_tuning.unsloth.ubuntu_pipeline \
      recover-incomplete-preparation \
      "${STATIC_ARGS[@]}" \
      --run-root "$RUN_ROOT" \
      --allowed-run-parent "$ALLOWED_RUN_PARENT" \
      "${PRECREATED_ARGS[@]}" \
      "${EXECUTION_PLAN_ARGS[@]}"
  )
}

validate_prepared_run() {
  (
    cd "$ROOT"
    "$TRAIN_PY" -m tools.fine_tuning.unsloth.ubuntu_pipeline validate-prepared-runtime \
      --root "$ROOT" \
      --run-root "$RUN_ROOT" \
      --agents "$AGENTS_CSV" \
      --variant "$EXPERIMENT_VARIANT" \
      --container-digest "$CONTAINER_IMAGE_DIGEST" \
      "${EXECUTION_PLAN_ARGS[@]}"
  )
}

run_root_is_empty() {
  [[ ! -e "$RUN_ROOT" ]] && return 1
  ! find "$RUN_ROOT" -mindepth 1 -maxdepth 1 -print -quit | grep -q .
}

reset_owned_precreated_run() {
  (
    cd "$ROOT"
    "$TRAIN_PY" -m tools.fine_tuning.unsloth.ubuntu_pipeline \
      reset-owned-run-root \
      --run-root "$RUN_ROOT" \
      --variant "$EXPERIMENT_VARIANT"
  )
}

if [[ "$RESUME" == "1" ]]; then
  [[ -d "$RUN_ROOT" ]] || die "resume run root does not exist: $RUN_ROOT"
  if [[ ! -f "$RUN_ROOT/aio_run_manifest.json" ]]; then
    if [[ -f "$RUN_ROOT/$PREPARATION_OWNER_FILENAME" ]]; then
      log "recovering an interrupted, cryptographically owned preparation"
      recover_incomplete_preparation
    elif [[ "$PRECREATED_BIND_ROOT" == "1" ]] && run_root_is_empty; then
      log "using the empty precreated bind root reserved by the outer launcher"
    else
      die "resume root lacks both a prepared manifest and a valid preparation owner"
    fi
    prepare_run_inputs
  fi
  validate_prepared_run
  log "resume mode skips verified complete phases and restarts incomplete phases"
else
  if [[ -e "$RUN_ROOT" ]]; then
    if [[ "$PRECREATED_BIND_ROOT" != "1" ]]; then
      [[ "$OVERWRITE" == "1" ]] || die "run root already exists: $RUN_ROOT"
    elif [[ "$OVERWRITE" != "1" ]] && ! run_root_is_empty; then
      die "precreated bind root contains unexpected existing state: $RUN_ROOT"
    fi
    [[ -d "$RUN_ROOT" && ! -L "$RUN_ROOT" ]] \
      || die "refusing to overwrite a non-directory or symlink run root: $RUN_ROOT"
    if [[ -f "$RUN_ROOT/aio_run_manifest.json" ]]; then
      [[ "$OVERWRITE" == "1" ]] \
        || die "prepared precreated bind root requires explicit overwrite"
      if [[ "$PRECREATED_BIND_ROOT" == "1" ]]; then
        reset_owned_precreated_run
      else
        (
          cd "$ROOT"
          "$TRAIN_PY" -m tools.fine_tuning.unsloth.ubuntu_pipeline verify-owned-run \
            --run-root "$RUN_ROOT" --variant "$EXPERIMENT_VARIANT"
        )
        rm -rf -- "$RUN_ROOT"
      fi
    elif [[ -f "$RUN_ROOT/$PREPARATION_OWNER_FILENAME" ]]; then
      [[ "$OVERWRITE" == "1" || "$PRECREATED_BIND_ROOT" == "1" ]] \
        || die "incomplete preparation requires explicit overwrite"
      recover_incomplete_preparation
    elif [[ "$PRECREATED_BIND_ROOT" == "1" ]] && run_root_is_empty; then
      :
    else
      die "refusing to overwrite a run root without verified ownership"
    fi
  fi
  prepare_run_inputs
  validate_prepared_run
fi

if [[ "$PREPARE_ONLY" == "1" ]]; then
  log "prepared run manifest: $RUN_ROOT/aio_run_manifest.json"
  exit 0
fi

log "running one tokenizer-only preflight across all requested agents"
(
  cd "$ROOT"
  "$TRAIN_PY" -m tools.fine_tuning.unsloth.ubuntu_pipeline \
    global-tokenizer-preflight \
    --run-root "$RUN_ROOT" \
    --agents "$AGENTS_CSV"
)

CONVERTER_REPO=""
CONVERTER_STAGING=""
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
  CONVERTER_STAGING="$RUN_ROOT/.llama.cpp.staging"
  CONVERTER="$CONVERTER_REPO/convert_lora_to_gguf.py"
  GGUF_READER="$CONVERTER_REPO/gguf-py/gguf/scripts/gguf_dump.py"

  verify_converter_checkout() {
    local checkout="$1"
    local converter="$checkout/convert_lora_to_gguf.py"
    local reader="$checkout/gguf-py/gguf/scripts/gguf_dump.py"
    [[ -d "$checkout" && ! -L "$checkout" ]] || return 1
    [[ "$(stat -c '%u:%a' "$checkout" 2>/dev/null)" == "$EUID:700" ]] \
      || return 1
    [[ -f "$converter" && ! -L "$converter" ]] || return 1
    [[ -f "$reader" && ! -L "$reader" ]] || return 1
    [[ "$(git -C "$checkout" rev-parse HEAD 2>/dev/null)" == "$LLAMA_CPP_REVISION" ]] \
      || return 1
    [[ "$(git -C "$checkout" config --get remote.origin.url 2>/dev/null)" == \
      "https://github.com/ggml-org/llama.cpp" ]] || return 1
    [[ -z "$(git -C "$checkout" status --porcelain=v1 --untracked-files=all 2>/dev/null)" ]] \
      || return 1
    [[ "$(git -C "$checkout" hash-object "$converter" 2>/dev/null)" == \
      "$(git -C "$checkout" rev-parse "HEAD:convert_lora_to_gguf.py" 2>/dev/null)" ]] \
      || return 1
    [[ "$(git -C "$checkout" hash-object "$reader" 2>/dev/null)" == \
      "$(git -C "$checkout" rev-parse \
        "HEAD:gguf-py/gguf/scripts/gguf_dump.py" 2>/dev/null)" ]] || return 1
  }

  remove_derived_converter_checkout() {
    local checkout="$1"
    [[ "$checkout" == "$CONVERTER_REPO" || "$checkout" == "$CONVERTER_STAGING" ]] \
      || die "refusing to remove an unowned converter path: $checkout"
    [[ ! -L "$checkout" ]] \
      || die "refusing to remove a symlink converter path: $checkout"
    if [[ -e "$checkout" ]]; then
      [[ -d "$checkout" ]] \
        || die "converter path is not a regular directory: $checkout"
      [[ "$(stat -c '%u:%a' "$checkout")" == "$EUID:700" ]] \
        || die "refusing to remove a converter directory that is not private and owned: $checkout"
      "$TRAIN_PY" - "$checkout" <<'PY'
import os
import shutil
import stat
import sys
from pathlib import Path

target = Path(sys.argv[1])
target_stat = os.lstat(target)
if (
    stat.S_ISLNK(target_stat.st_mode)
    or not stat.S_ISDIR(target_stat.st_mode)
    or target_stat.st_uid != os.geteuid()
    or stat.S_IMODE(target_stat.st_mode) != 0o700
    or target_stat.st_dev != os.lstat(target.parent).st_dev
):
    raise SystemExit("converter cleanup target is not a private owned directory")
for directory, child_directories, filenames in os.walk(target, followlinks=False):
    directory_stat = os.lstat(directory)
    if directory_stat.st_dev != target_stat.st_dev:
        raise SystemExit("converter cleanup target contains a nested mount")
    for name in child_directories:
        child_stat = os.lstat(Path(directory) / name)
        if not stat.S_ISLNK(child_stat.st_mode) and (
            not stat.S_ISDIR(child_stat.st_mode)
            or child_stat.st_dev != target_stat.st_dev
        ):
            raise SystemExit("converter cleanup target contains an unsafe directory")
    for name in filenames:
        child_stat = os.lstat(Path(directory) / name)
        if not (
            stat.S_ISREG(child_stat.st_mode)
            or stat.S_ISLNK(child_stat.st_mode)
        ):
            raise SystemExit("converter cleanup target contains a special file")
        if not stat.S_ISLNK(child_stat.st_mode) and child_stat.st_dev != target_stat.st_dev:
            raise SystemExit("converter cleanup target contains a nested mount")
shutil.rmtree(target)
parent_descriptor = os.open(target.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
try:
    os.fsync(parent_descriptor)
finally:
    os.close(parent_descriptor)
PY
    fi
  }

  promote_converter_checkout() {
    "$TRAIN_PY" - "$CONVERTER_STAGING" "$CONVERTER_REPO" <<'PY'
import os
import stat
import sys
from pathlib import Path

staging = Path(sys.argv[1])
destination = Path(sys.argv[2])
if staging.parent != destination.parent or destination.exists() or destination.is_symlink():
    raise SystemExit("unsafe llama.cpp checkout promotion paths")
mode = os.lstat(staging).st_mode
if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
    raise SystemExit("llama.cpp staging checkout is not a regular directory")
os.replace(staging, destination)
descriptor = os.open(destination.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
try:
    os.fsync(descriptor)
finally:
    os.close(descriptor)
PY
  }

  if ! verify_converter_checkout "$CONVERTER_REPO"; then
    if [[ -e "$CONVERTER_REPO" || -L "$CONVERTER_REPO" ]]; then
      log "discarding an invalid interrupted llama.cpp checkout"
      remove_derived_converter_checkout "$CONVERTER_REPO"
    fi
    if verify_converter_checkout "$CONVERTER_STAGING"; then
      log "recovering the verified staged llama.cpp checkout"
      promote_converter_checkout
    else
      if [[ -e "$CONVERTER_STAGING" || -L "$CONVERTER_STAGING" ]]; then
        log "discarding an incomplete staged llama.cpp checkout"
        remove_derived_converter_checkout "$CONVERTER_STAGING"
      fi
      log "fetching the pinned llama.cpp LoRA converter into staging"
      mkdir -m 700 "$CONVERTER_STAGING"
      git init "$CONVERTER_STAGING"
      git -C "$CONVERTER_STAGING" remote add origin https://github.com/ggml-org/llama.cpp
      git -C "$CONVERTER_STAGING" fetch --depth 1 origin "$LLAMA_CPP_REVISION"
      git -C "$CONVERTER_STAGING" checkout --detach FETCH_HEAD
      verify_converter_checkout "$CONVERTER_STAGING" \
        || die "staged llama.cpp checkout failed pinned-source verification"
      promote_converter_checkout
    fi
  elif [[ -e "$CONVERTER_STAGING" || -L "$CONVERTER_STAGING" ]]; then
    log "removing stale llama.cpp staging after verified checkout recovery"
    remove_derived_converter_checkout "$CONVERTER_STAGING"
  fi
  verify_converter_checkout "$CONVERTER_REPO" \
    || die "llama.cpp converter checkout failed final pinned-source verification"
  [[ -f "$CONVERTER" && ! -L "$CONVERTER" ]] \
    || die "missing regular convert_lora_to_gguf.py"
  [[ -f "$GGUF_READER" && ! -L "$GGUF_READER" ]] \
    || die "missing regular pinned llama.cpp GGUF reader"
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

verify_evaluation() {
  local agent="$1"
  (
    cd "$ROOT"
    "$TRAIN_PY" -m tools.fine_tuning.unsloth.ubuntu_pipeline verify-evaluation \
      --run-root "$RUN_ROOT" --agent "$agent"
  )
}

clean_agent_evaluation() {
  local agent="$1"
  local target="$RUN_ROOT/evaluation/$agent"
  case ",$AGENTS_CSV," in
    *",$agent,"*) ;;
    *) die "refusing to clean evaluation for an unowned agent: $agent" ;;
  esac
  [[ ! -L "$target" ]] \
    || die "refusing to clean a symlink evaluation directory: $target"
  if [[ -e "$target" ]]; then
    [[ -d "$target" ]] \
      || die "evaluation path is not a regular directory: $target"
    [[ "$(stat -c '%u:%a' "$target")" == "$EUID:700" ]] \
      || die "refusing to clean an evaluation directory that is not private and owned: $target"
    if command -v mountpoint >/dev/null 2>&1 && mountpoint -q "$target"; then
      die "refusing to clean a mounted evaluation directory: $target"
    fi
    rm -rf -- "$target"
  fi
}

GGUF_STAGING_ROOT="$RUN_ROOT/.gguf-staging"

clean_agent_gguf_staging() {
  local agent="$1"
  local staging_dir="$GGUF_STAGING_ROOT/$agent"
  case ",$AGENTS_CSV," in
    *",$agent,"*) ;;
    *) die "refusing to clean GGUF staging for an unowned agent: $agent" ;;
  esac
  [[ ! -L "$GGUF_STAGING_ROOT" ]] \
    || die "GGUF staging root must not be a symlink: $GGUF_STAGING_ROOT"
  if [[ -e "$GGUF_STAGING_ROOT" ]]; then
    [[ -d "$GGUF_STAGING_ROOT" ]] \
      || die "GGUF staging root is not a directory: $GGUF_STAGING_ROOT"
    [[ "$(stat -c '%u:%a' "$GGUF_STAGING_ROOT")" == "$EUID:700" ]] \
      || die "GGUF staging root must be private and process-owned: $GGUF_STAGING_ROOT"
  fi
  [[ ! -L "$staging_dir" ]] \
    || die "per-agent GGUF staging must not be a symlink: $staging_dir"
  if [[ -e "$staging_dir" ]]; then
    [[ -d "$staging_dir" ]] \
      || die "per-agent GGUF staging is not a directory: $staging_dir"
    [[ "$(stat -c '%u:%a' "$staging_dir")" == "$EUID:700" ]] \
      || die "refusing to clean GGUF staging that is not private and owned: $staging_dir"
    if command -v mountpoint >/dev/null 2>&1 && mountpoint -q "$staging_dir"; then
      die "refusing to clean a mounted GGUF staging directory: $staging_dir"
    fi
    rm -rf -- "$staging_dir"
  fi
  if [[ -d "$GGUF_STAGING_ROOT" ]] && \
    ! find "$GGUF_STAGING_ROOT" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
    rmdir "$GGUF_STAGING_ROOT"
  fi
}

remove_invalid_agent_gguf() {
  local agent="$1"
  local models_root="$RUN_ROOT/models"
  local gguf_root="$models_root/lora_qwen3_gguf"
  local receipt_root="$models_root/lora_qwen3_gguf_receipts"
  local path="$RUN_ROOT/models/lora_qwen3_gguf/lumen-$agent-lora.gguf"
  local receipt="$receipt_root/lumen-$agent-lora.conversion.json"
  case ",$AGENTS_CSV," in
    *",$agent,"*) ;;
    *) die "refusing to remove GGUF for an unowned agent: $agent" ;;
  esac
  [[ -d "$models_root" && ! -L "$models_root" ]] \
    || die "models root is unsafe: $models_root"
  [[ -d "$gguf_root" && ! -L "$gguf_root" ]] \
    || die "GGUF artifact root is unsafe: $gguf_root"
  [[ -d "$receipt_root" && ! -L "$receipt_root" ]] \
    || die "GGUF conversion-receipt root is unsafe: $receipt_root"
  local candidate
  for candidate in "$path" "$receipt"; do
    [[ ! -L "$candidate" ]] \
      || die "refusing to remove a symlink GGUF lineage file: $candidate"
    if [[ -e "$candidate" ]]; then
      [[ -f "$candidate" ]] \
        || die "GGUF lineage path is not a regular file: $candidate"
      [[ "$(stat -c '%u' "$candidate")" == "$EUID" ]] \
        || die "refusing to remove GGUF lineage not owned by the process user: $candidate"
      rm -f -- "$candidate"
    fi
  done
}

clean_agent_posttraining_artifacts() {
  local agent="$1"
  if [[ -e "$RUN_ROOT/evaluation/$agent" || -L "$RUN_ROOT/evaluation/$agent" ]]; then
    clean_agent_evaluation "$agent"
  fi
  remove_invalid_agent_gguf "$agent"
  if [[ -e "$GGUF_STAGING_ROOT/$agent" || -L "$GGUF_STAGING_ROOT/$agent" ]]; then
    clean_agent_gguf_staging "$agent"
  fi
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

install_staged_agent_gguf() {
  local agent="$1"
  local staged="$GGUF_STAGING_ROOT/$agent/lumen-$agent-lora.gguf"
  (
    cd "$ROOT"
    "$TRAIN_PY" -m tools.fine_tuning.unsloth.ubuntu_pipeline \
      install-gguf-file \
      --run-root "$RUN_ROOT" \
      --agent "$agent" \
      --staging-path "$staged"
  )
}

evaluate_agent() {
  local agent="$1"
  log "preparing frozen evaluation lineage: $agent"
  (
    cd "$ROOT"
    "$TRAIN_PY" -m tools.fine_tuning.unsloth.ubuntu_pipeline write-final-config \
      --run-root "$RUN_ROOT" --agent "$agent"
  )
  local -a eval_args=(
    --config "$RUN_ROOT/configs/$agent.final.json"
    --eval-jsonl "$RUN_ROOT/generated/fine_tuning/$agent/eval.jsonl"
    --behavior-manifest "$RUN_ROOT/generated/agent_manifest/AgentBehaviorManifest.json"
    --output-dir "$RUN_ROOT/evaluation/$agent"
  )
  if [[ -n "$EVAL_MAX_EXAMPLES" ]]; then
    eval_args+=(--max-examples "$EVAL_MAX_EXAMPLES")
  fi
  if [[ "$RESUME" == "1" ]] && verify_evaluation "$agent" >/dev/null 2>&1; then
    log "verified existing frozen evaluation: $agent"
    return
  fi
  if [[ "$RESUME" == "1" ]] && \
    [[ -e "$RUN_ROOT/evaluation/$agent" || -L "$RUN_ROOT/evaluation/$agent" ]]; then
    log "replacing invalid or partial evaluation outputs: $agent"
    clean_agent_evaluation "$agent"
  fi
  log "running frozen evaluation: $agent"
  printf '[lumen-aio] evaluation attempt started: %s agent=%s resume=%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$agent" "$RESUME" \
    | tee -a "$RUN_ROOT/logs/evaluate_$agent.log"
  (
    cd "$ROOT"
    "$TRAIN_PY" -m tools.fine_tuning.unsloth.evaluate_adapter "${eval_args[@]}"
  ) 2>&1 | tee -a "$RUN_ROOT/logs/evaluate_$agent.log"
  verify_evaluation "$agent"
}

convert_agent_gguf() {
  local agent="$1"
  local outfile="$RUN_ROOT/models/lora_qwen3_gguf/lumen-$agent-lora.gguf"
  local staged_dir="$GGUF_STAGING_ROOT/$agent"
  local staged_outfile="$staged_dir/lumen-$agent-lora.gguf"
  local base_model
  local adapter_dir
  if [[ "$RESUME" == "1" ]] && (
    cd "$ROOT"
    "$TRAIN_PY" -m tools.fine_tuning.unsloth.ubuntu_pipeline verify-gguf-file \
      --run-root "$RUN_ROOT" --path "$outfile"
  ) >/dev/null 2>&1; then
    log "verified existing adapter GGUF: $agent"
    if [[ -e "$staged_dir" || -L "$staged_dir" ]]; then
      clean_agent_gguf_staging "$agent"
    fi
    return
  fi
  remove_invalid_agent_gguf "$agent"
  if [[ "$RESUME" == "1" ]] && \
    [[ -e "$staged_dir" || -L "$staged_dir" ]]; then
    if install_staged_agent_gguf "$agent" >/dev/null 2>&1; then
      log "recovered and verified staged adapter GGUF: $agent"
      return
    fi
    log "discarding invalid or partial staged adapter GGUF: $agent"
    clean_agent_gguf_staging "$agent"
  elif [[ -e "$staged_dir" || -L "$staged_dir" ]]; then
    clean_agent_gguf_staging "$agent"
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
  [[ ! -L "$GGUF_STAGING_ROOT" ]] \
    || die "GGUF staging root must not be a symlink: $GGUF_STAGING_ROOT"
  if [[ ! -d "$GGUF_STAGING_ROOT" ]]; then
    mkdir -m 700 "$GGUF_STAGING_ROOT"
  fi
  [[ "$(stat -c '%a' "$GGUF_STAGING_ROOT")" == "700" ]] \
    || die "GGUF staging root must remain private: $GGUF_STAGING_ROOT"
  [[ ! -e "$staged_dir" && ! -L "$staged_dir" ]] \
    || die "GGUF per-agent staging unexpectedly exists: $staged_dir"
  mkdir -m 700 "$staged_dir"
  log "converting finalized adapter to staged GGUF: $agent"
  printf '[lumen-aio] GGUF conversion attempt started: %s agent=%s resume=%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$agent" "$RESUME" \
    | tee -a "$RUN_ROOT/logs/convert_$agent.log"
  "$TRAIN_PY" "$CONVERTER" "$adapter_dir" \
    --outfile "$staged_outfile" \
    --base "$base_model" \
    2>&1 | tee -a "$RUN_ROOT/logs/convert_$agent.log"
  (
    cd "$ROOT"
    "$TRAIN_PY" -m tools.fine_tuning.unsloth.ubuntu_pipeline \
      write-gguf-conversion-receipt \
      --run-root "$RUN_ROOT" \
      --agent "$agent" \
      --staging-path "$staged_outfile"
  )
  install_staged_agent_gguf "$agent"
}

TRAIN_ARGS=()
if [[ "$ASSISTANT_ONLY_LOSS" == "1" ]]; then
  TRAIN_ARGS+=(--assistant-only-loss)
fi

# Deliberately qualify and export one agent before spending GPU time on the
# next. A quality failure therefore stops the fleet at the earliest boundary.
for agent in "${AGENTS[@]}"; do
  [[ -n "$agent" ]] || continue
  if [[ "$RESUME" == "1" ]] && verify_phase "$agent" sft >/dev/null 2>&1; then
    log "verified existing SFT phase: $agent"
  else
    clean_agent_posttraining_artifacts "$agent"
    sft_train_args=("${TRAIN_ARGS[@]}")
    if [[ "$RESUME" == "1" ]]; then
      if compgen -G "$RUN_ROOT/training/$agent/checkpoint-*" >/dev/null; then
        sft_train_args+=(--resume-from-checkpoint)
        log "resuming the latest bound SFT checkpoint: $agent"
      else
        clean_phase "$agent" sft
      fi
    fi
    log "training SFT adapter: $agent"
    printf '[lumen-aio] SFT attempt started: %s agent=%s resume=%s\n' \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$agent" \
      "$([[ ${#sft_train_args[@]} -gt ${#TRAIN_ARGS[@]} ]] && printf 1 || printf 0)" \
      | tee -a "$RUN_ROOT/logs/train_sft_$agent.log"
    (
      cd "$ROOT"
      "$TRAIN_PY" -m tools.fine_tuning.unsloth.train_sft \
        --config "$RUN_ROOT/configs/$agent.json" \
        --seed "$SEED" \
        "${sft_train_args[@]}"
    ) 2>&1 | tee -a "$RUN_ROOT/logs/train_sft_$agent.log"
    verify_phase "$agent" sft
  fi

  if [[ "$RUN_PREFERENCE" == "1" ]]; then
    if [[ "$RESUME" == "1" ]] && verify_phase "$agent" preference >/dev/null 2>&1; then
      log "verified existing preference phase: $agent"
    else
      clean_agent_posttraining_artifacts "$agent"
      preference_train_args=()
      if [[ "$RESUME" == "1" ]]; then
        if compgen -G "$RUN_ROOT/training/$agent/dpo/checkpoint-*" >/dev/null; then
          preference_train_args+=(--resume-from-checkpoint)
          log "resuming the latest bound preference checkpoint: $agent"
        else
          clean_phase "$agent" preference
        fi
      fi
      log "training preference adapter: $agent"
      printf '[lumen-aio] preference attempt started: %s agent=%s resume=%s\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$agent" \
        "$([[ ${#preference_train_args[@]} -gt 0 ]] && printf 1 || printf 0)" \
        | tee -a "$RUN_ROOT/logs/train_preference_$agent.log"
      (
        cd "$ROOT"
        "$TRAIN_PY" -m tools.fine_tuning.unsloth.train_dpo \
          --config "$RUN_ROOT/configs/$agent.json" \
          --sft-adapter-dir "$RUN_ROOT/models/lora_qwen3_bootstrap/$agent" \
          --sft-finalized-variant-manifest \
            "$RUN_ROOT/training/$agent/finalized_variant_manifest.json" \
          "${preference_train_args[@]}"
      ) 2>&1 | tee -a "$RUN_ROOT/logs/train_preference_$agent.log"
      verify_phase "$agent" preference
    fi
  fi

  if [[ "$EVALUATE" == "1" ]]; then
    evaluate_agent "$agent"
  fi
  if [[ "$CONVERT_GGUF" == "1" ]]; then
    convert_agent_gguf "$agent"
  fi
done

if [[ "$CONVERT_GGUF" == "1" ]]; then
  if [[ -e "$GGUF_STAGING_ROOT" || -L "$GGUF_STAGING_ROOT" ]]; then
    [[ -d "$GGUF_STAGING_ROOT" && ! -L "$GGUF_STAGING_ROOT" ]] \
      || die "GGUF staging root is unsafe after conversion: $GGUF_STAGING_ROOT"
    ! find "$GGUF_STAGING_ROOT" -mindepth 1 -maxdepth 1 -print -quit | grep -q . \
      || die "GGUF staging root retained unexpected entries: $GGUF_STAGING_ROOT"
    rmdir "$GGUF_STAGING_ROOT"
  fi
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
