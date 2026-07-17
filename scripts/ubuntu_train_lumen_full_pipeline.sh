#!/usr/bin/env bash
set -Eeuo pipefail
umask 077

IFS=$'\n\t'

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
DOCKERFILE="$ROOT/tools/fine_tuning/unsloth/Dockerfile.ubuntu-cu128"
SAFE_REPO_OUTPUT_ROOT="$ROOT/.local/ubuntu_finetune_runs"

IMAGE_TAG="${LUMEN_UBUNTU_IMAGE_TAG:-lumen-training:cu128-py310}"
OUTPUT_ROOT="${LUMEN_UBUNTU_OUTPUT_ROOT:-$SAFE_REPO_OUTPUT_ROOT}"
HF_CACHE="${LUMEN_UBUNTU_HF_CACHE:-${HF_HOME:-$HOME/.cache/huggingface}}"
TOKEN_FILE="${LUMEN_UBUNTU_HF_TOKEN_FILE:-}"
RUN_ID="${LUMEN_UBUNTU_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-$(printf '%04x%04x' "$RANDOM" "$RANDOM")}"
AGENTS_CSV="${LUMEN_UBUNTU_AGENTS:-cortex,executor,mouth,mimicry,rem,fleet}"
VARIANT_SELECTOR="${LUMEN_UBUNTU_VARIANT:-optimized}"
SHM_SIZE="${LUMEN_UBUNTU_SHM_SIZE:-8g}"
BUILD_IMAGE="${LUMEN_UBUNTU_BUILD_IMAGE:-1}"
PULL_BASE="${LUMEN_UBUNTU_PULL_BASE:-1}"
UPLOAD="${LUMEN_UBUNTU_UPLOAD:-0}"
HF_PRIVATE="${LUMEN_UBUNTU_HF_PRIVATE:-1}"
OVERWRITE="${LUMEN_UBUNTU_OVERWRITE:-0}"
RESUME="${LUMEN_UBUNTU_RESUME:-0}"
PREPARE_ONLY="${LUMEN_UBUNTU_PREPARE_ONLY:-0}"
CONVERT_GGUF="${LUMEN_UBUNTU_CONVERT_GGUF:-1}"
EVALUATE="${LUMEN_UBUNTU_EVALUATE:-1}"
EVAL_MAX_EXAMPLES="${LUMEN_UBUNTU_EVAL_MAX_EXAMPLES:-}"
RUNTIME_HOME="/home/lumen-runtime"

log() {
  printf '[lumen-ubuntu] %s\n' "$*"
}

die() {
  printf '[lumen-ubuntu] ERROR: %s\n' "$*" >&2
  exit 1
}

usage() {
  cat <<'EOF'
Run Lumen's pinned CUDA 12.8 / Python 3.10 training pipeline in Docker.

Usage:
  bash scripts/ubuntu_train_lumen_full_pipeline.sh [options]

Options:
  --variant NAME       optimized (default), baseline, internal,
                       baseline-and-optimized, all,
                       or an exact controlled variant name
  --output-dir DIR     Parent directory for immutable per-variant run directories
  --hf-cache DIR       Persistent Hugging Face cache (default: $HF_HOME or ~/.cache/huggingface)
  --image-tag TAG      Local Docker image tag (default: lumen-training:cu128-py310)
  --run-id ID          Safe run identifier; the variant suffix is added automatically
  --agents CSV         Comma-separated agents (default: all six Lumen agents)
  --upload             Upload verified outputs after training (off by default)
  --public             Make an explicitly requested upload public (private by default)
  --token-file FILE    Mount an HF token only into the isolated upload container
  --overwrite          Replace pre-existing per-variant run directories
  --resume             Resume an existing run by skipping verified training phases
  --prepare-only       Validate and prepare run inputs without training
  --no-evaluate        Skip frozen post-training inference and scoring
  --eval-smoke N       Evaluate a deterministic semantic cohort of N frozen cases per agent
  --no-gguf            Skip adapter GGUF conversion
  --no-build           Reuse --image-tag instead of building it
  --no-pull            Do not refresh the pinned CUDA base tag during docker build
  -h, --help           Show this help

Selectors:
  optimized   internal_plus_public_optimized
  baseline    internal_plus_public_baseline
  internal    internal_only
  baseline-and-optimized
              sequential baseline + optimized batch (24 jobs; fail-fast)
  all         internal + baseline + optimized (36 SFT/preference jobs for all agents)

Environment variables use the same LUMEN_UBUNTU_* names shown by the defaults
above. Training never receives an HF token; upload credentials come from
--token-file or the unmounted $HF_HOME/token file.
EOF
}

require_bool() {
  local name="$1"
  local value="$2"
  [[ "$value" == "0" || "$value" == "1" ]] || die "$name must be 0 or 1 (got: $value)"
}

canonical_variant() {
  case "$1" in
    internal|internal_only) printf '%s\n' 'internal_only' ;;
    baseline|internal_plus_public_baseline) printf '%s\n' 'internal_plus_public_baseline' ;;
    optimized|internal_plus_public_optimized) printf '%s\n' 'internal_plus_public_optimized' ;;
    *) return 1 ;;
  esac
}

while (($#)); do
  case "$1" in
    --variant)
      (($# >= 2)) || die "--variant requires a value"
      VARIANT_SELECTOR="$2"
      shift 2
      ;;
    --output-dir)
      (($# >= 2)) || die "--output-dir requires a value"
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --hf-cache)
      (($# >= 2)) || die "--hf-cache requires a value"
      HF_CACHE="$2"
      shift 2
      ;;
    --image-tag)
      (($# >= 2)) || die "--image-tag requires a value"
      IMAGE_TAG="$2"
      shift 2
      ;;
    --run-id)
      (($# >= 2)) || die "--run-id requires a value"
      RUN_ID="$2"
      shift 2
      ;;
    --agents)
      (($# >= 2)) || die "--agents requires a value"
      AGENTS_CSV="$2"
      shift 2
      ;;
    --token-file)
      (($# >= 2)) || die "--token-file requires a value"
      TOKEN_FILE="$2"
      shift 2
      ;;
    --upload)
      UPLOAD=1
      shift
      ;;
    --public)
      HF_PRIVATE=0
      shift
      ;;
    --overwrite)
      OVERWRITE=1
      shift
      ;;
    --resume)
      RESUME=1
      shift
      ;;
    --prepare-only)
      PREPARE_ONLY=1
      shift
      ;;
    --no-gguf)
      CONVERT_GGUF=0
      shift
      ;;
    --no-evaluate)
      EVALUATE=0
      shift
      ;;
    --eval-smoke)
      (($# >= 2)) || die "--eval-smoke requires a positive integer"
      EVAL_MAX_EXAMPLES="$2"
      EVALUATE=1
      shift 2
      ;;
    --no-build)
      BUILD_IMAGE=0
      shift
      ;;
    --no-pull)
      PULL_BASE=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown option: $1 (use --help)"
      ;;
  esac
done

for pair in \
  "LUMEN_UBUNTU_BUILD_IMAGE:$BUILD_IMAGE" \
  "LUMEN_UBUNTU_PULL_BASE:$PULL_BASE" \
  "LUMEN_UBUNTU_UPLOAD:$UPLOAD" \
  "LUMEN_UBUNTU_HF_PRIVATE:$HF_PRIVATE" \
  "LUMEN_UBUNTU_OVERWRITE:$OVERWRITE" \
  "LUMEN_UBUNTU_RESUME:$RESUME" \
  "LUMEN_UBUNTU_PREPARE_ONLY:$PREPARE_ONLY" \
  "LUMEN_UBUNTU_CONVERT_GGUF:$CONVERT_GGUF" \
  "LUMEN_UBUNTU_EVALUATE:$EVALUATE"; do
  require_bool "${pair%%:*}" "${pair#*:}"
done

[[ "$(uname -s)" == "Linux" ]] || die "this launcher must run on Linux; copy the repository to the Ubuntu GPU host first"
[[ -f "$DOCKERFILE" ]] || die "missing training Dockerfile: $DOCKERFILE"
RUNTIME_UID="$(id -u)"
RUNTIME_GID="$(id -g)"
[[ "$RUNTIME_UID" =~ ^[1-9][0-9]*$ ]] \
  || die "run this launcher as a regular non-root user (do not use sudo)"
[[ "$RUNTIME_GID" =~ ^[1-9][0-9]*$ ]] \
  || die "the invoking user's primary group must be non-root"
[[ "$RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]] || die "run ID must contain only letters, digits, dot, underscore, and hyphen"
[[ "$AGENTS_CSV" =~ ^[a-z]+(,[a-z]+)*$ ]] || die "agents must be a comma-separated lowercase list without spaces"
[[ "$OVERWRITE" == "0" || "$RESUME" == "0" ]] || die "--overwrite and --resume are mutually exclusive"
[[ "$UPLOAD" == "0" || "$PREPARE_ONLY" == "0" ]] || die "--upload cannot be combined with --prepare-only"
[[ "$UPLOAD" == "0" || "$BUILD_IMAGE" == "1" ]] \
  || die "credential-scoped upload requires an image built by this invocation; remove --no-build"
if [[ -n "$EVAL_MAX_EXAMPLES" ]]; then
  [[ "$EVAL_MAX_EXAMPLES" =~ ^[1-9][0-9]*$ ]] || die "--eval-smoke requires a positive integer"
fi

declare -A seen_agents=()
IFS=',' read -r -a requested_agents <<< "$AGENTS_CSV"
for agent in "${requested_agents[@]}"; do
  case "$agent" in
    cortex|executor|mouth|mimicry|rem|fleet) ;;
    *) die "unsupported agent: $agent" ;;
  esac
  [[ -z "${seen_agents[$agent]:-}" ]] || die "duplicate agent: $agent"
  seen_agents[$agent]=1
done

declare -a variants=()
case "$VARIANT_SELECTOR" in
  baseline-and-optimized)
    variants=(internal_plus_public_baseline internal_plus_public_optimized)
    ;;
  all)
    variants=(internal_only internal_plus_public_baseline internal_plus_public_optimized)
    ;;
  *)
    variant="$(canonical_variant "$VARIANT_SELECTOR")" || die "unsupported variant selector: $VARIANT_SELECTOR"
    variants=("$variant")
    ;;
esac

command -v docker >/dev/null 2>&1 || die "docker not found; install Docker Engine before running this script"
command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi not found; install a compatible NVIDIA driver before running this script"
docker info >/dev/null 2>&1 || die "Docker daemon is unavailable for the current user"

path_contains() {
  local parent="$1"
  local child="$2"
  [[ "$child" == "$parent" || "$child" == "$parent/"* ]]
}

OUTPUT_ROOT="$(realpath -m -- "$OUTPUT_ROOT")"
HF_CACHE="$(realpath -m -- "$HF_CACHE")"
SAFE_REPO_OUTPUT_ROOT="$(realpath -m -- "$SAFE_REPO_OUTPUT_ROOT")"
[[ "$OUTPUT_ROOT" != "/" ]] || die "refusing to use the filesystem root as --output-dir"
[[ "$HF_CACHE" != "/" ]] || die "refusing to use the filesystem root as --hf-cache"

if path_contains "$OUTPUT_ROOT" "$ROOT"; then
  die "--output-dir must not equal or contain the repository"
fi
if path_contains "$ROOT" "$OUTPUT_ROOT" \
  && ! path_contains "$SAFE_REPO_OUTPUT_ROOT" "$OUTPUT_ROOT"; then
  die "an in-repository --output-dir must stay under $SAFE_REPO_OUTPUT_ROOT"
fi
if path_contains "$HF_CACHE" "$ROOT" || path_contains "$ROOT" "$HF_CACHE"; then
  die "--hf-cache must not overlap the repository"
fi
if path_contains "$OUTPUT_ROOT" "$HF_CACHE" || path_contains "$HF_CACHE" "$OUTPUT_ROOT"; then
  die "--output-dir and --hf-cache must not overlap"
fi

mkdir -p -- "$OUTPUT_ROOT" "$HF_CACHE/hub" "$HF_CACHE/xet" "$HF_CACHE/assets"
resolved_output_root="$(realpath -e -- "$OUTPUT_ROOT")"
resolved_hf_cache="$(realpath -e -- "$HF_CACHE")"
[[ "$resolved_output_root" == "$OUTPUT_ROOT" && "$resolved_hf_cache" == "$HF_CACHE" ]] \
  || die "output or cache path changed while being prepared"
for cache_child in "$HF_CACHE/hub" "$HF_CACHE/xet" "$HF_CACHE/assets"; do
  [[ -d "$cache_child" && ! -L "$cache_child" ]] \
    || die "Hugging Face cache child must be a regular directory: $cache_child"
done
HF_HUB_CACHE_HOST="$(cd "$HF_CACHE/hub" && pwd -P)"
HF_XET_CACHE_HOST="$(cd "$HF_CACHE/xet" && pwd -P)"
HF_ASSETS_CACHE_HOST="$(cd "$HF_CACHE/assets" && pwd -P)"
for cache_path in "$HF_HUB_CACHE_HOST" "$HF_XET_CACHE_HOST" "$HF_ASSETS_CACHE_HOST"; do
  if path_contains "$ROOT" "$cache_path" \
    || path_contains "$cache_path" "$ROOT" \
    || path_contains "$OUTPUT_ROOT" "$cache_path" \
    || path_contains "$cache_path" "$OUTPUT_ROOT"; then
    die "resolved Hugging Face cache mount overlaps the repository or output root: $cache_path"
  fi
done
declare -a cache_mounts=(
  -v "$HF_HUB_CACHE_HOST:/cache/huggingface/hub:rw"
  -v "$HF_XET_CACHE_HOST:/cache/huggingface/xet:rw"
  -v "$HF_ASSETS_CACHE_HOST:/cache/huggingface/assets:rw"
)

if [[ "$UPLOAD" == "1" && -z "$TOKEN_FILE" && -f "$HF_CACHE/token" ]]; then
  TOKEN_FILE="$HF_CACHE/token"
fi
declare -a upload_token_mount=()
if [[ -n "$TOKEN_FILE" ]]; then
  [[ -f "$TOKEN_FILE" && -r "$TOKEN_FILE" ]] || die "HF token file is not readable: $TOKEN_FILE"
  [[ ! -L "$TOKEN_FILE" ]] || die "HF token file must not be a symlink: $TOKEN_FILE"
  TOKEN_FILE="$(realpath -e -- "$TOKEN_FILE")" || die "unable to resolve HF token file"
  [[ "$(stat -c '%u' "$TOKEN_FILE")" == "$(id -u)" ]] \
    || die "HF token file must be owned by the current user"
  token_mode="$(stat -c '%a' "$TOKEN_FILE")" || die "unable to inspect HF token permissions"
  (( (8#$token_mode & 077) == 0 )) \
    || die "HF token file must not be group/world accessible (run: chmod 600 '$TOKEN_FILE')"
  if path_contains "$ROOT" "$TOKEN_FILE" \
    || path_contains "$OUTPUT_ROOT" "$TOKEN_FILE" \
    || path_contains "$HF_HUB_CACHE_HOST" "$TOKEN_FILE" \
    || path_contains "$HF_XET_CACHE_HOST" "$TOKEN_FILE" \
    || path_contains "$HF_ASSETS_CACHE_HOST" "$TOKEN_FILE"; then
    die "HF token file is inside a path visible to the training container"
  fi
  upload_token_mount=(-v "$TOKEN_FILE:/run/secrets/hf_token:ro")
fi

if [[ "$UPLOAD" == "1" && -z "$TOKEN_FILE" ]]; then
  die "upload requires --token-file or an existing login at $HF_CACHE/token"
fi
if [[ "$UPLOAD" != "1" && "$HF_PRIVATE" != "1" ]]; then
  die "--public is only valid together with --upload"
fi

for variant in "${variants[@]}"; do
  host_run_root="$OUTPUT_ROOT/$RUN_ID-$variant"
  [[ ! -L "$host_run_root" ]] || die "run directory must not be a symlink: $host_run_root"
  if [[ "$RESUME" != "1" && -e "$host_run_root" && "$OVERWRITE" != "1" ]]; then
    die "run directory already exists: $host_run_root (choose another --run-id or pass --overwrite)"
  fi
done

if [[ "$BUILD_IMAGE" == "1" ]]; then
  build_args=(
    docker build
    --file "$DOCKERFILE"
    --tag "$IMAGE_TAG"
    --build-arg "LUMEN_RUNTIME_UID=$RUNTIME_UID"
    --build-arg "LUMEN_RUNTIME_GID=$RUNTIME_GID"
  )
  if [[ "$PULL_BASE" == "1" ]]; then
    build_args+=(--pull)
  fi
  build_args+=("$ROOT")
  log "building pinned training image: $IMAGE_TAG"
  "${build_args[@]}"
else
  log "reusing local training image: $IMAGE_TAG"
fi

IMAGE_DIGEST="$(docker image inspect --format '{{.Id}}' "$IMAGE_TAG" 2>/dev/null)" || die "local image not found: $IMAGE_TAG"
[[ "$IMAGE_DIGEST" =~ ^sha256:[0-9a-f]{64}$ ]] || die "Docker returned an invalid local image ID for $IMAGE_TAG: $IMAGE_DIGEST"

log "checking container runtime identity"
docker run --rm \
  --user "$RUNTIME_UID:$RUNTIME_GID" \
  -e "HOME=$RUNTIME_HOME" \
  --entrypoint /opt/lumen-venv/bin/python \
  "$IMAGE_DIGEST" \
  -c 'import getpass, grp, os, pwd, tempfile; from pathlib import Path; uid = os.getuid(); gid = os.getgid(); assert pwd.getpwuid(uid).pw_uid == uid; assert grp.getgrgid(gid).gr_gid == gid; assert getpass.getuser(); home = Path.home(); assert home == Path(os.environ["HOME"]) and home.is_dir(); tempfile.TemporaryFile(dir=home).close()' \
  || die "training image lacks the invoking user's passwd/group mapping or writable home; rebuild without --no-build"

log "checking NVIDIA Container Toolkit access"
docker run --rm --gpus all --user "$RUNTIME_UID:$RUNTIME_GID" \
  --entrypoint /bin/bash "$IMAGE_DIGEST" -c 'exec nvidia-smi' >/dev/null \
  || die "Docker cannot access the NVIDIA GPU; install/configure NVIDIA Container Toolkit"

log "image ID: $IMAGE_DIGEST"
log "runtime identity: $RUNTIME_UID:$RUNTIME_GID ($RUNTIME_HOME)"
log "output root: $OUTPUT_ROOT"
log "HF cache: $HF_CACHE"
log "agents: $AGENTS_CSV"
log "variants: ${variants[*]}"
log "upload: $UPLOAD (private: $HF_PRIVATE)"
log "evaluation: $EVALUATE${EVAL_MAX_EXAMPLES:+ (smoke cases per agent: $EVAL_MAX_EXAMPLES)}"
log "resume: $RESUME"

for variant in "${variants[@]}"; do
  host_run_root="$OUTPUT_ROOT/$RUN_ID-$variant"
  container_run_root="/outputs/$RUN_ID-$variant"
  variant_resume=0
  if [[ "$RESUME" == "1" && -d "$host_run_root" ]]; then
    variant_resume=1
  elif [[ "$RESUME" == "1" ]]; then
    log "no prior run for $variant; preparing it as a fresh variant"
  fi
  log "starting full pipeline for $variant"
  docker_args=(
    docker run --rm --init --gpus all --shm-size "$SHM_SIZE"
    --entrypoint /bin/bash
    --user "$RUNTIME_UID:$RUNTIME_GID"
    -v "$ROOT:/workspace:ro"
    -v "$OUTPUT_ROOT:/outputs:rw"
    "${cache_mounts[@]}"
    -e "HOME=$RUNTIME_HOME"
    -e HF_HOME=/cache/huggingface
    -e HUGGINGFACE_HUB_CACHE=/cache/huggingface/hub
    -e HF_HUB_ENABLE_HF_TRANSFER=1
    -e HF_HUB_DISABLE_IMPLICIT_TOKEN=1
    -e PYTHONDONTWRITEBYTECODE=1
    -e PYTHONPATH=/workspace
    -e TOKENIZERS_PARALLELISM=false
    -e PYTORCH_ALLOC_CONF=expandable_segments:True
    -e "LUMEN_AIO_EXPERIMENT_VARIANT=$variant"
    -e "LUMEN_AIO_CONTAINER_IMAGE_DIGEST=$IMAGE_DIGEST"
    -e "LUMEN_AIO_RUN_ID=$RUN_ID"
    -e "LUMEN_AIO_RUN_ROOT=$container_run_root"
    -e LUMEN_AIO_ALLOWED_RUN_PARENT=/outputs
    -e "LUMEN_AIO_AGENTS=$AGENTS_CSV"
    -e LUMEN_AIO_FULL_PIPELINE=1
    -e LUMEN_AIO_RUN_PREFERENCE=1
    -e LUMEN_AIO_PYTHON_BIN=/opt/lumen-venv/bin/python
    -e LUMEN_AIO_PYTHON=/opt/lumen-venv/bin/python
    -e LUMEN_AIO_VENV=/opt/lumen-venv
    -e LUMEN_AIO_USE_ACTIVE_PYTHON=1
    -e LUMEN_AIO_SKIP_INSTALL=1
    -e LUMEN_AIO_REQUIRE_CUDA=1
    -e "LUMEN_AIO_CONVERT_GGUF=$CONVERT_GGUF"
    -e LUMEN_AIO_UPLOAD=0
    -e "LUMEN_AIO_OVERWRITE=$OVERWRITE"
    -e "LUMEN_AIO_RESUME=$variant_resume"
    -e "LUMEN_AIO_PREPARE_ONLY=$PREPARE_ONLY"
    -e "LUMEN_AIO_EVALUATE=$EVALUATE"
    -e "LUMEN_AIO_EVAL_MAX_EXAMPLES=$EVAL_MAX_EXAMPLES"
    "$IMAGE_DIGEST"
    /workspace/scripts/ubuntu_train_lumen_adapters_aio.sh
  )
  "${docker_args[@]}"
  if [[ "$UPLOAD" == "1" ]]; then
    upload_flags=()
    if [[ "$HF_PRIVATE" != "1" ]]; then
      upload_flags+=(--public)
    fi
    if [[ "$CONVERT_GGUF" == "1" ]]; then
      upload_flags+=(--include-gguf)
    fi
    log "uploading verified outputs in an isolated credential-scoped container"
    docker run --rm --init \
      --entrypoint /bin/bash \
      --user "$RUNTIME_UID:$RUNTIME_GID" \
      -v "$ROOT:/workspace:ro" \
      -v "$host_run_root:$container_run_root:rw" \
      "${upload_token_mount[@]}" \
      -e "HOME=$RUNTIME_HOME" \
      -e PYTHONDONTWRITEBYTECODE=1 \
      -e PYTHONPATH=/workspace \
      "$IMAGE_DIGEST" \
      -Eeuo pipefail -c \
      'exec /opt/lumen-venv/bin/python -m tools.fine_tuning.unsloth.ubuntu_pipeline "$@"' \
      lumen-upload upload \
      --run-root "$container_run_root" \
      --agents "$AGENTS_CSV" \
      --run-id "$RUN_ID-$variant" \
      --token-file /run/secrets/hf_token \
      "${upload_flags[@]}"
  fi
  log "completed $variant: $host_run_root"
done

log "all requested variants completed"
