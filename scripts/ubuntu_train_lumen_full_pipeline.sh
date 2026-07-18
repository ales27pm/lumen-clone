#!/usr/bin/env bash
set -Eeuo pipefail
umask 077

IFS=$'\n\t'

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
DOCKERFILE_RELATIVE="tools/fine_tuning/unsloth/Dockerfile.ubuntu-cu128"
DOCKERFILE="$ROOT/$DOCKERFILE_RELATIVE"
SOURCE_ATTESTOR="$ROOT/tools/fine_tuning/unsloth/ubuntu_source_integrity.py"
SAFE_REPO_OUTPUT_ROOT="$ROOT/.local/ubuntu_finetune_runs"

IMAGE_TAG="${LUMEN_UBUNTU_IMAGE_TAG:-lumen-training:cu128-py310}"
OUTPUT_ROOT="${LUMEN_UBUNTU_OUTPUT_ROOT:-$SAFE_REPO_OUTPUT_ROOT}"
HF_CACHE="${LUMEN_UBUNTU_HF_CACHE:-${HF_HOME:-$HOME/.cache/huggingface}}"
TOKEN_FILE="${LUMEN_UBUNTU_HF_TOKEN_FILE:-}"
RUN_ID="${LUMEN_UBUNTU_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-$(printf '%04x%04x' "$RANDOM" "$RANDOM")}"
AGENTS_CSV="${LUMEN_UBUNTU_AGENTS:-fleet,executor,mouth,rem,mimicry,cortex}"
VARIANT_SELECTOR="${LUMEN_UBUNTU_VARIANT:-optimized}"
SHM_SIZE="${LUMEN_UBUNTU_SHM_SIZE:-8g}"
BUILD_IMAGE="${LUMEN_UBUNTU_BUILD_IMAGE:-1}"
PULL_BASE="${LUMEN_UBUNTU_PULL_BASE:-1}"
UPLOAD="${LUMEN_UBUNTU_UPLOAD:-0}"
ALLOW_DIAGNOSTIC_UPLOAD="${LUMEN_UBUNTU_ALLOW_DIAGNOSTIC_UPLOAD:-0}"
HF_PRIVATE="${LUMEN_UBUNTU_HF_PRIVATE:-1}"
OVERWRITE="${LUMEN_UBUNTU_OVERWRITE:-0}"
RESUME="${LUMEN_UBUNTU_RESUME:-0}"
PREPARE_ONLY="${LUMEN_UBUNTU_PREPARE_ONLY:-0}"
CONVERT_GGUF="${LUMEN_UBUNTU_CONVERT_GGUF:-1}"
EVALUATE="${LUMEN_UBUNTU_EVALUATE:-1}"
EVAL_MAX_EXAMPLES="${LUMEN_UBUNTU_EVAL_MAX_EXAMPLES:-}"
NO_EVALUATE_OPTION_SEEN=0
EVAL_SMOKE_OPTION_SEEN=0
RUNTIME_HOME="/home/lumen-runtime"
IMAGE_SOURCE_ROOT="/opt/lumen/source"
IMAGE_SOURCE_ATTESTATION="/opt/lumen/ubuntu-source-integrity.json"

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
  --allow-diagnostic-upload
                       Permit an explicit smoke/no-evaluation upload under diagnostic-runs/
  --public             Make an explicitly requested upload public (private by default)
  --token-file FILE    Mount an HF token only into the isolated upload container
  --overwrite          Replace pre-existing per-variant run directories
  --resume             Resume an existing run by skipping verified training phases
  --prepare-only       Validate and prepare run inputs without training
  --no-evaluate        Skip frozen post-training inference and scoring
  --eval-smoke N       Evaluate a deterministic semantic cohort of N frozen cases per agent
  --no-gguf            Skip adapter GGUF conversion
  --no-build           Reuse --image-tag instead of building it
  --no-pull            Do not re-fetch the pinned CUDA base digest during docker build
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
    --allow-diagnostic-upload)
      ALLOW_DIAGNOSTIC_UPLOAD=1
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
      NO_EVALUATE_OPTION_SEEN=1
      shift
      ;;
    --eval-smoke)
      (($# >= 2)) || die "--eval-smoke requires a positive integer"
      EVAL_MAX_EXAMPLES="$2"
      EVALUATE=1
      EVAL_SMOKE_OPTION_SEEN=1
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
  "LUMEN_UBUNTU_ALLOW_DIAGNOSTIC_UPLOAD:$ALLOW_DIAGNOSTIC_UPLOAD" \
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
[[ -f "$SOURCE_ATTESTOR" ]] || die "missing source-integrity helper: $SOURCE_ATTESTOR"
RUNTIME_UID="$(id -u)"
RUNTIME_GID="$(id -g)"
[[ "$RUNTIME_UID" =~ ^[1-9][0-9]*$ ]] \
  || die "run this launcher as a regular non-root user (do not use sudo)"
[[ "$RUNTIME_GID" =~ ^[1-9][0-9]*$ ]] \
  || die "the invoking user's primary group must be non-root"
PRIVATE_UPLOAD_TMPFS="/tmp:rw,noexec,nosuid,nodev,mode=700,uid=$RUNTIME_UID,gid=$RUNTIME_GID"
[[ "$RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]] || die "run ID must contain only letters, digits, dot, underscore, and hyphen"
[[ "$AGENTS_CSV" =~ ^[a-z]+(,[a-z]+)*$ ]] || die "agents must be a comma-separated lowercase list without spaces"
[[ "$OVERWRITE" == "0" || "$RESUME" == "0" ]] || die "--overwrite and --resume are mutually exclusive"
[[ "$UPLOAD" == "0" || "$PREPARE_ONLY" == "0" ]] || die "--upload cannot be combined with --prepare-only"
[[ "$UPLOAD" == "0" || "$BUILD_IMAGE" == "1" ]] \
  || die "credential-scoped upload requires an image built by this invocation; remove --no-build"
if [[ -n "$EVAL_MAX_EXAMPLES" ]]; then
  [[ "$EVAL_MAX_EXAMPLES" =~ ^[1-9][0-9]*$ ]] || die "--eval-smoke requires a positive integer"
fi
[[ "$NO_EVALUATE_OPTION_SEEN" == "0" || "$EVAL_SMOKE_OPTION_SEEN" == "0" ]] \
  || die "--no-evaluate and --eval-smoke are mutually exclusive"
[[ "$EVALUATE" != "0" || -z "$EVAL_MAX_EXAMPLES" ]] \
  || die "disabled evaluation cannot retain a smoke cohort size"
[[ "$ALLOW_DIAGNOSTIC_UPLOAD" == "0" || "$UPLOAD" == "1" ]] \
  || die "--allow-diagnostic-upload is only valid together with --upload"
if [[ "$UPLOAD" == "1" && ( "$EVALUATE" == "0" || -n "$EVAL_MAX_EXAMPLES" ) \
  && "$ALLOW_DIAGNOSTIC_UPLOAD" != "1" ]]; then
  die "smoke or unevaluated publication requires --allow-diagnostic-upload"
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

command -v git >/dev/null 2>&1 || die "git not found"
command -v python3 >/dev/null 2>&1 || die "python3 not found"
command -v docker >/dev/null 2>&1 || die "docker not found; install Docker Engine before running this script"
command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi not found; install a compatible NVIDIA driver before running this script"
docker info >/dev/null 2>&1 || die "Docker daemon is unavailable for the current user"

read_source_attestation_fields() {
  python3 -I "$SOURCE_ATTESTOR" attest-host --root "$ROOT" \
    | python3 -I -c 'import json, sys; value = json.load(sys.stdin); print(value["baseCommit"]); print(value["workingTreeDigest"]); print(value["ubuntuOrchestrationCodeSHA256"]); print(value["sourceIntegritySHA256"])'
}

mapfile -t SOURCE_ATTESTATION_FIELDS < <(read_source_attestation_fields) \
  || die "unable to attest the clean Ubuntu pipeline source"
[[ "${#SOURCE_ATTESTATION_FIELDS[@]}" == "4" ]] \
  || die "source-integrity helper returned an incomplete attestation"
SOURCE_BASE_COMMIT="${SOURCE_ATTESTATION_FIELDS[0]}"
SOURCE_WORKING_TREE_DIGEST="${SOURCE_ATTESTATION_FIELDS[1]}"
SOURCE_ORCHESTRATION_DIGEST="${SOURCE_ATTESTATION_FIELDS[2]}"
SOURCE_INTEGRITY_DIGEST="${SOURCE_ATTESTATION_FIELDS[3]}"

verify_clean_source_unchanged() {
  local -a current=()
  mapfile -t current < <(read_source_attestation_fields) \
    || die "Ubuntu pipeline source is no longer a clean checkout"
  [[ "${#current[@]}" == "4" \
      && "${current[0]}" == "$SOURCE_BASE_COMMIT" \
      && "${current[1]}" == "$SOURCE_WORKING_TREE_DIGEST" \
      && "${current[2]}" == "$SOURCE_ORCHESTRATION_DIGEST" \
      && "${current[3]}" == "$SOURCE_INTEGRITY_DIGEST" ]] \
    || die "Ubuntu pipeline source changed after its initial attestation"
}

archive_attested_build_context() {
  local -a git_environment=(
    env
    -u GIT_ALTERNATE_OBJECT_DIRECTORIES
    -u GIT_CEILING_DIRECTORIES
    -u GIT_COMMON_DIR
    -u GIT_CONFIG
    -u GIT_CONFIG_COUNT
    -u GIT_CONFIG_GLOBAL
    -u GIT_CONFIG_NOSYSTEM
    -u GIT_CONFIG_PARAMETERS
    -u GIT_CONFIG_SYSTEM
    -u GIT_DIR
    -u GIT_DISCOVERY_ACROSS_FILESYSTEM
    -u GIT_EXEC_PATH
    -u GIT_INDEX_FILE
    -u GIT_NAMESPACE
    -u GIT_OBJECT_DIRECTORY
    -u GIT_WORK_TREE
  )
  local -a context_paths=(
    scripts/ubuntu_train_lumen_full_pipeline.sh
    scripts/ubuntu_train_lumen_adapters_aio.sh
    tools/fine_tuning/unsloth
    tools/lumen_manifest_crawler/lumen_manifest_crawler
    tools/hf_zerogpu/space_template
    generated/fine_tuning
    generated/agent_manifest/AgentBehaviorManifest.json
  )
  "${git_environment[@]}" \
    GIT_CONFIG_GLOBAL=/dev/null \
    GIT_CONFIG_NOSYSTEM=1 \
    GIT_NO_REPLACE_OBJECTS=1 \
    GIT_OPTIONAL_LOCKS=0 \
    LC_ALL=C \
    git \
      -c core.fsmonitor=false \
      -c core.untrackedCache=false \
      -C "$ROOT" \
      archive \
      --format=tar \
      "$SOURCE_BASE_COMMIT" \
      -- \
      "${context_paths[@]}"
}

path_contains() {
  local parent="$1"
  local child="$2"
  [[ "$child" == "$parent" || "$child" == "$parent/"* ]]
}

training_container_name() {
  local host_run_root="$1"
  python3 -I -c \
    'import hashlib, sys; print("lumen-ubuntu-" + hashlib.sha256(sys.argv[1].encode("utf-8")).hexdigest()[:24])' \
    "$host_run_root"
}

training_launch_contract_digest() {
  python3 -I -c \
    'import hashlib, sys; print(hashlib.sha256(b"\0".join(value.encode("utf-8") for value in sys.argv[1:])).hexdigest())' \
    "$@"
}

build_training_environment_contract() {
  local output_name="$1"
  local launch_mode="$2"
  local overwrite_mode="$3"
  local resume_mode="$4"
  # shellcheck disable=SC2034  # Assignment is through this Bash nameref.
  local -n output="$output_name"
  [[ "$launch_mode" == "fresh" || "$launch_mode" == "resume" \
      || "$launch_mode" == "overwrite" ]] \
    || die "invalid training launch mode: $launch_mode"
  [[ "$overwrite_mode" == "0" || "$overwrite_mode" == "1" ]] \
    || die "invalid overwrite mode for training environment"
  [[ "$resume_mode" == "0" || "$resume_mode" == "1" ]] \
    || die "invalid resume mode for training environment"
  case "$launch_mode:$overwrite_mode:$resume_mode" in
    fresh:0:0|resume:0:1|overwrite:1:0) ;;
    *) die "training launch mode does not match resume/overwrite controls" ;;
  esac
  # shellcheck disable=SC2034  # Assignment is through the Bash nameref above.
  output=(
    "HOME=$RUNTIME_HOME"
    HF_HOME=/cache/huggingface
    HUGGINGFACE_HUB_CACHE=/cache/huggingface/hub
    HF_HUB_ENABLE_HF_TRANSFER=1
    HF_HUB_DISABLE_IMPLICIT_TOKEN=1
    PYTHONDONTWRITEBYTECODE=1
    PYTHONUNBUFFERED=1
    "LUMEN_UBUNTU_SOURCE_ATTESTATION_PATH=$IMAGE_SOURCE_ATTESTATION"
    TOKENIZERS_PARALLELISM=false
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    "LUMEN_AIO_EXPERIMENT_VARIANT=$variant"
    "LUMEN_AIO_CONTAINER_IMAGE_DIGEST=$IMAGE_DIGEST"
    "LUMEN_AIO_RUN_ID=$RUN_ID"
    "LUMEN_AIO_RUN_ROOT=$container_run_root"
    LUMEN_AIO_ALLOWED_RUN_PARENT=/outputs
    LUMEN_AIO_PRECREATED_BIND_ROOT=1
    "LUMEN_AIO_EXPECTED_RUN_ROOT_IDENTITY=$host_run_root_identity"
    "LUMEN_AIO_LOCK_DIR=$CONTAINER_LOCK_DIR"
    "LUMEN_AIO_EXPECTED_LOCK_IDENTITY=$HOST_LOCK_IDENTITY"
    "LUMEN_AIO_AGENTS=$AGENTS_CSV"
    LUMEN_AIO_FULL_PIPELINE=1
    LUMEN_AIO_RUN_PREFERENCE=1
    LUMEN_AIO_PYTHON_BIN=/opt/lumen-venv/bin/python
    LUMEN_AIO_PYTHON=/opt/lumen-venv/bin/python
    LUMEN_AIO_VENV=/opt/lumen-venv
    LUMEN_AIO_USE_ACTIVE_PYTHON=1
    LUMEN_AIO_SKIP_INSTALL=1
    LUMEN_AIO_REQUIRE_CUDA=1
    "LUMEN_AIO_CONVERT_GGUF=$CONVERT_GGUF"
    LUMEN_AIO_UPLOAD=0
    "LUMEN_AIO_OVERWRITE=$overwrite_mode"
    "LUMEN_AIO_RESUME=$resume_mode"
    "LUMEN_AIO_LAUNCH_MODE=$launch_mode"
    "LUMEN_AIO_PREPARE_ONLY=$PREPARE_ONLY"
    "LUMEN_AIO_EVALUATE=$EVALUATE"
    "LUMEN_AIO_EVAL_MAX_EXAMPLES=$EVAL_MAX_EXAMPLES"
  )
}

training_launch_contract_for_environment() {
  local launch_mode="$1"
  shift
  training_launch_contract_digest \
    "$IMAGE_DIGEST" \
    "$SOURCE_INTEGRITY_DIGEST" \
    "$SOURCE_ORCHESTRATION_DIGEST" \
    "$host_run_root" \
    "$container_run_root" \
    "$host_run_root_identity" \
    "$HOST_LOCK_DIR" \
    "$CONTAINER_LOCK_DIR" \
    "$HOST_LOCK_IDENTITY" \
    "$RUNTIME_UID:$RUNTIME_GID" \
    "$SHM_SIZE" \
    "$RUN_ID" \
    "$variant" \
    "$AGENTS_CSV" \
    "$CONVERT_GGUF" \
    "$PREPARE_ONLY" \
    "$EVALUATE" \
    "$EVAL_MAX_EXAMPLES" \
    "$HF_HUB_CACHE_HOST" \
    "$HF_XET_CACHE_HOST" \
    "$HF_ASSETS_CACHE_HOST" \
    "$launch_mode" \
    "$@"
}

verify_exact_training_environment() {
  local container_name="$1"
  shift
  local environment_json
  environment_json="$(inspect_training_container_value \
    "$container_name" '{{json .Config.Env}}')" \
    || die "unable to inspect retained training container environment: $container_name"
  python3 -I - "$environment_json" "$@" <<'PY'
import json
import sys

observed_raw = json.loads(sys.argv[1])
expected_raw = sys.argv[2:]
if not isinstance(observed_raw, list) or any(not isinstance(item, str) for item in observed_raw):
    raise SystemExit("training container has malformed environment evidence")

def parse(items, label):
    result = {}
    for item in items:
        key, separator, value = item.partition("=")
        if not separator or not key or key in result:
            raise SystemExit(f"training container has malformed or duplicate {label} environment")
        result[key] = value
    return result

observed = parse(observed_raw, "observed")
expected = parse(expected_raw, "expected")
for key, value in expected.items():
    if observed.get(key) != value:
        raise SystemExit(f"training container environment drifted: {key}")
unexpected_lumen = sorted(
    key for key in observed if key.startswith("LUMEN_AIO_") and key not in expected
)
if unexpected_lumen:
    raise SystemExit(
        "training container has unexpected state-affecting environment: "
        + ",".join(unexpected_lumen)
    )
PY
}

inspect_training_container_value() {
  local container_name="$1"
  local format="$2"
  docker container inspect --format "$format" "$container_name"
}

initialize_host_bind_root() {
  local host_path="$1"
  local create_if_missing="$2"
  local expected_creation="${3:-either}"
  local -a create_args=()
  local record
  if [[ "$create_if_missing" == "1" ]]; then
    create_args+=(--create-if-missing)
  fi
  record="$({
    cd "$ROOT"
    python3 -m tools.fine_tuning.unsloth.ubuntu_pipeline initialize-bind-root \
      --run-root "$host_path" \
      --allowed-run-parent "$OUTPUT_ROOT" \
      "${create_args[@]}"
  })" || die "unable to initialize exact bind root: $host_path"
  python3 -I -c \
    'import json, sys
value=json.loads(sys.argv[1])
expected=sys.argv[2]
created=value.get("created")
if type(created) is not bool:
    raise SystemExit("bind-root helper returned invalid creation evidence")
if expected == "new" and not created:
    raise SystemExit("exact bind root appeared during exclusive reservation")
if expected == "existing" and created:
    raise SystemExit("exact bind root disappeared during reservation")
if expected not in {"new", "existing", "either"}:
    raise SystemExit("invalid bind-root creation expectation")
print(value["rootIdentity"])' \
    "$record" "$expected_creation"
}

verify_host_bind_root_identity() {
  local host_path="$1"
  local expected_identity="$2"
  (
    cd "$ROOT"
    python3 -m tools.fine_tuning.unsloth.ubuntu_pipeline verify-bind-root \
      --run-root "$host_path" \
      --allowed-run-parent "$OUTPUT_ROOT" \
      --expected-identity "$expected_identity" >/dev/null
  ) || die "exact bind-root identity changed: $host_path"
}

verify_exact_training_mounts() {
  local container_name="$1"
  local host_run_root="$2"
  local container_run_root="$3"
  local mounts_json
  mounts_json="$(inspect_training_container_value "$container_name" '{{json .Mounts}}')" \
    || die "unable to inspect retained training container mounts: $container_name"
  python3 -I - \
    "$mounts_json" \
    "$host_run_root" "$container_run_root" \
    "$HOST_LOCK_DIR" "$CONTAINER_LOCK_DIR" \
    "$HF_HUB_CACHE_HOST" "$HF_XET_CACHE_HOST" "$HF_ASSETS_CACHE_HOST" <<'PY'
import json
import sys

mounts = json.loads(sys.argv[1])
expected = {
    sys.argv[3]: (sys.argv[2], True),
    sys.argv[5]: (sys.argv[4], False),
    "/cache/huggingface/hub": (sys.argv[6], True),
    "/cache/huggingface/xet": (sys.argv[7], True),
    "/cache/huggingface/assets": (sys.argv[8], True),
}
if not isinstance(mounts, list) or len(mounts) != len(expected):
    raise SystemExit("training container has an unexpected mount set")
observed = {}
for item in mounts:
    if not isinstance(item, dict) or not isinstance(item.get("Destination"), str):
        raise SystemExit("training container has malformed mount evidence")
    destination = item["Destination"]
    if destination in observed:
        raise SystemExit("training container has duplicate mount destinations")
    observed[destination] = item
if "/outputs" in observed:
    raise SystemExit("training container exposes the broad output root")
if set(observed) != set(expected):
    raise SystemExit("training container mount destinations drifted")
for destination, (source, writable) in expected.items():
    item = observed[destination]
    if (
        item.get("Type") != "bind"
        or item.get("Source") != source
        or item.get("RW") is not writable
        or item.get("Mode") != ("rw" if writable else "ro")
        or item.get("Propagation") != "rprivate"
    ):
        raise SystemExit(f"training container mount contract drifted: {destination}")
PY
}

verify_training_postcondition() {
  local host_run_root="$1"
  local container_run_root="$2"
  local expected_variant="$3"
  local expected_root_identity="$4"
  local -a execution_plan_args=()
  local evaluation_scope
  [[ -d "$host_run_root" && ! -L "$host_run_root" ]] \
    || die "training container did not leave a regular run root: $host_run_root"
  verify_host_bind_root_identity "$host_run_root" "$expected_root_identity"
  if [[ "$EVALUATE" != "1" ]]; then
    evaluation_scope=none
  elif [[ -n "$EVAL_MAX_EXAMPLES" ]]; then
    evaluation_scope=smoke
  else
    evaluation_scope=full
  fi
  execution_plan_args+=(--evaluation-scope "$evaluation_scope")
  if [[ -n "$EVAL_MAX_EXAMPLES" ]]; then
    execution_plan_args+=(--evaluation-max-examples "$EVAL_MAX_EXAMPLES")
  fi
  if [[ "$CONVERT_GGUF" == "1" ]]; then
    execution_plan_args+=(--gguf-requested)
  else
    execution_plan_args+=(--no-gguf-requested)
  fi
  if [[ "$PREPARE_ONLY" == "1" ]]; then
    execution_plan_args+=(--prepare-only)
  fi
  verify_clean_source_unchanged
  log "independently verifying the exited training container postcondition"
  docker run --rm --init \
    --read-only \
    --network none \
    --cap-drop ALL \
    --security-opt no-new-privileges \
    --tmpfs /tmp:rw,noexec,nosuid,nodev,size=64m \
    --entrypoint /opt/lumen-venv/bin/python \
    --user "$RUNTIME_UID:$RUNTIME_GID" \
    -v "$host_run_root:$container_run_root:ro" \
    -e HOME=/tmp \
    -e XDG_CACHE_HOME=/tmp/cache \
    -e PYTHONDONTWRITEBYTECODE=1 \
    -e "LUMEN_UBUNTU_SOURCE_ATTESTATION_PATH=$IMAGE_SOURCE_ATTESTATION" \
    "$IMAGE_DIGEST" \
    -I "$IMAGE_SOURCE_ROOT/tools/fine_tuning/unsloth/ubuntu_postcondition.py" \
    verify-container-postcondition \
    --root "$IMAGE_SOURCE_ROOT" \
    --run-root "$container_run_root" \
    --agents "$AGENTS_CSV" \
    --variant "$expected_variant" \
    --container-digest "$IMAGE_DIGEST" \
    "${execution_plan_args[@]}"
}

verify_training_container_identity() {
  local container_name="$1"
  local expected_contract_digest="$2"
  local expected_run_root="$3"
  local expected_container_run_root="$4"
  local expected_root_identity="$5"
  local expected_launch_mode="$6"
  shift 6
  local actual_image actual_source actual_contract actual_run_root
  local actual_container_run_root actual_root_identity actual_lock_dir
  local actual_lock_identity
  local actual_entrypoint actual_command actual_user actual_auto_remove actual_init
  local actual_launch_mode
  actual_image="$(inspect_training_container_value "$container_name" '{{.Image}}')" \
    || die "unable to inspect retained training container: $container_name"
  actual_source="$(inspect_training_container_value "$container_name" '{{index .Config.Labels "ai.lumen.source-integrity-sha256"}}')" \
    || die "unable to inspect retained training container source label: $container_name"
  actual_contract="$(inspect_training_container_value "$container_name" '{{index .Config.Labels "ai.lumen.launch-contract-sha256"}}')" \
    || die "unable to inspect retained training container launch label: $container_name"
  actual_run_root="$(inspect_training_container_value "$container_name" '{{index .Config.Labels "ai.lumen.host-run-root"}}')" \
    || die "unable to inspect retained training container run-root label: $container_name"
  actual_container_run_root="$(inspect_training_container_value "$container_name" '{{index .Config.Labels "ai.lumen.container-run-root"}}')" \
    || die "unable to inspect retained training container container-root label: $container_name"
  actual_root_identity="$(inspect_training_container_value "$container_name" '{{index .Config.Labels "ai.lumen.host-run-root-identity"}}')" \
    || die "unable to inspect retained training container root-identity label: $container_name"
  actual_lock_dir="$(inspect_training_container_value "$container_name" '{{index .Config.Labels "ai.lumen.host-lock-dir"}}')" \
    || die "unable to inspect retained training container lock-root label: $container_name"
  actual_lock_identity="$(inspect_training_container_value "$container_name" '{{index .Config.Labels "ai.lumen.host-lock-identity"}}')" \
    || die "unable to inspect retained training container lock-identity label: $container_name"
  actual_launch_mode="$(inspect_training_container_value "$container_name" '{{index .Config.Labels "ai.lumen.launch-mode"}}')" \
    || die "unable to inspect retained training container launch-mode label: $container_name"
  actual_entrypoint="$(inspect_training_container_value "$container_name" '{{json .Config.Entrypoint}}')" \
    || die "unable to inspect retained training container entrypoint: $container_name"
  actual_command="$(inspect_training_container_value "$container_name" '{{json .Config.Cmd}}')" \
    || die "unable to inspect retained training container command: $container_name"
  actual_user="$(inspect_training_container_value "$container_name" '{{.Config.User}}')" \
    || die "unable to inspect retained training container user: $container_name"
  actual_auto_remove="$(inspect_training_container_value "$container_name" '{{.HostConfig.AutoRemove}}')" \
    || die "unable to inspect retained training container removal policy: $container_name"
  actual_init="$(inspect_training_container_value "$container_name" '{{.HostConfig.Init}}')" \
    || die "unable to inspect retained training container init policy: $container_name"
  [[ "$actual_image" == "$IMAGE_DIGEST" \
      && "$actual_source" == "$SOURCE_INTEGRITY_DIGEST" \
      && "$actual_contract" == "$expected_contract_digest" \
      && "$actual_run_root" == "$expected_run_root" \
      && "$actual_container_run_root" == "$expected_container_run_root" \
      && "$actual_root_identity" == "$expected_root_identity" \
      && "$actual_lock_dir" == "$HOST_LOCK_DIR" \
      && "$actual_lock_identity" == "$HOST_LOCK_IDENTITY" \
      && "$actual_launch_mode" == "$expected_launch_mode" \
      && "$actual_entrypoint" == '["/bin/bash"]' \
      && "$actual_command" == "[\"$IMAGE_SOURCE_ROOT/scripts/ubuntu_train_lumen_adapters_aio.sh\"]" \
      && "$actual_user" == "$RUNTIME_UID:$RUNTIME_GID" \
      && "$actual_auto_remove" == "false" \
      && "$actual_init" == "true" ]] \
    || die "retained training container does not match the exact attested launch contract: $container_name"
  verify_host_bind_root_identity "$expected_run_root" "$expected_root_identity"
  verify_host_bind_root_identity "$HOST_LOCK_DIR" "$HOST_LOCK_IDENTITY"
  verify_exact_training_mounts \
    "$container_name" "$expected_run_root" "$expected_container_run_root"
  verify_exact_training_environment "$container_name" "$@"
}

capture_training_container_evidence() {
  local container_name="$1"
  local evidence_dir="$2"
  local disposition="$3"
  local container_id short_id inspect_path log_path
  container_id="$(inspect_training_container_value "$container_name" '{{.Id}}')" \
    || return 1
  [[ "$container_id" =~ ^[0-9a-f]{64}$ ]] || return 1
  short_id="${container_id:0:12}"
  inspect_path="$evidence_dir/$disposition-$short_id.docker-inspect.json"
  log_path="$evidence_dir/$disposition-$short_id.docker.log"
  docker container inspect "$container_name" >"$inspect_path" || return 1
  docker logs --timestamps "$container_name" >"$log_path" 2>&1 || true
  log "container evidence: $inspect_path"
  log "container log: $log_path"
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
HOST_LOCK_DIR="$OUTPUT_ROOT/.lumen-training.lock"
CONTAINER_LOCK_DIR="/run/lumen-training-lock"
HOST_LOCK_IDENTITY="$(initialize_host_bind_root "$HOST_LOCK_DIR" 1 either)" \
  || die "unable to initialize the cross-container training lock"
[[ "$HOST_LOCK_IDENTITY" =~ ^[0-9]+:[0-9]+:[0-9]+:[0-9]+:0700$ ]] \
  || die "cross-container training lock has an invalid identity"

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

verify_clean_source_unchanged
if [[ "$BUILD_IMAGE" == "1" ]]; then
  build_args=(
    docker build
    --file "$DOCKERFILE_RELATIVE"
    --tag "$IMAGE_TAG"
    --build-arg "LUMEN_RUNTIME_UID=$RUNTIME_UID"
    --build-arg "LUMEN_RUNTIME_GID=$RUNTIME_GID"
    --build-arg "LUMEN_SOURCE_BASE_COMMIT=$SOURCE_BASE_COMMIT"
    --build-arg "LUMEN_WORKING_TREE_DIGEST=$SOURCE_WORKING_TREE_DIGEST"
    --build-arg "LUMEN_UBUNTU_ORCHESTRATION_SHA256=$SOURCE_ORCHESTRATION_DIGEST"
  )
  if [[ "$PULL_BASE" == "1" ]]; then
    build_args+=(--pull)
  fi
  log "building pinned training image: $IMAGE_TAG"
  archive_attested_build_context | "${build_args[@]}" -
else
  log "reusing local training image: $IMAGE_TAG"
fi

IMAGE_DIGEST="$(docker image inspect --format '{{.Id}}' "$IMAGE_TAG" 2>/dev/null)" || die "local image not found: $IMAGE_TAG"
[[ "$IMAGE_DIGEST" =~ ^sha256:[0-9a-f]{64}$ ]] || die "Docker returned an invalid local image ID for $IMAGE_TAG: $IMAGE_DIGEST"

verify_clean_source_unchanged
log "verifying the image-baked Ubuntu execution closure"
docker run --rm \
  --network none \
  --read-only \
  --cap-drop ALL \
  --security-opt no-new-privileges \
  --entrypoint /opt/lumen-venv/bin/python \
  "$IMAGE_DIGEST" \
  -I "$IMAGE_SOURCE_ROOT/tools/fine_tuning/unsloth/ubuntu_source_integrity.py" \
  verify-image \
  --root "$IMAGE_SOURCE_ROOT" \
  --record "$IMAGE_SOURCE_ATTESTATION" \
  --base-commit "$SOURCE_BASE_COMMIT" \
  --working-tree-digest "$SOURCE_WORKING_TREE_DIGEST" \
  --orchestration-digest "$SOURCE_ORCHESTRATION_DIGEST" \
  --source-integrity-digest "$SOURCE_INTEGRITY_DIGEST" >/dev/null \
  || die "training image source attestation does not match the clean host checkout"

log "checking container runtime identity"
docker run --rm \
  --network none \
  --tmpfs "$PRIVATE_UPLOAD_TMPFS" \
  --user "$RUNTIME_UID:$RUNTIME_GID" \
  -e "HOME=$RUNTIME_HOME" \
  --entrypoint /opt/lumen-venv/bin/python \
  "$IMAGE_DIGEST" \
  -c 'import getpass, grp, os, pwd, stat, tempfile; from pathlib import Path; uid = os.getuid(); gid = os.getgid(); assert pwd.getpwuid(uid).pw_uid == uid; assert grp.getgrgid(gid).gr_gid == gid; assert getpass.getuser(); home = Path.home(); assert home == Path(os.environ["HOME"]) and home.is_dir(); tempfile.TemporaryFile(dir=home).close(); scratch = os.stat("/tmp", follow_symlinks=False); assert stat.S_ISDIR(scratch.st_mode) and scratch.st_uid == uid and scratch.st_gid == gid and stat.S_IMODE(scratch.st_mode) == 0o700; tempfile.TemporaryFile(dir="/tmp").close()' \
  || die "training image lacks the invoking user's passwd/group mapping or writable home; rebuild without --no-build"

log "checking NVIDIA Container Toolkit access"
docker run --rm --gpus all --user "$RUNTIME_UID:$RUNTIME_GID" \
  --network none \
  --entrypoint /bin/bash "$IMAGE_DIGEST" -c 'exec nvidia-smi' >/dev/null \
  || die "Docker cannot access the NVIDIA GPU; install/configure NVIDIA Container Toolkit"

log "image ID: $IMAGE_DIGEST"
log "runtime identity: $RUNTIME_UID:$RUNTIME_GID ($RUNTIME_HOME)"
log "output root: $OUTPUT_ROOT"
log "HF cache: $HF_CACHE"
log "agents: $AGENTS_CSV"
log "variants: ${variants[*]}"
log "upload: $UPLOAD (private: $HF_PRIVATE)"
log "diagnostic upload override: $ALLOW_DIAGNOSTIC_UPLOAD"
log "evaluation: $EVALUATE${EVAL_MAX_EXAMPLES:+ (smoke cases per agent: $EVAL_MAX_EXAMPLES)}"
log "resume: $RESUME"

for variant in "${variants[@]}"; do
  host_run_root="$OUTPUT_ROOT/$RUN_ID-$variant"
  container_run_root="/outputs/$RUN_ID-$variant"
  host_run_root_preexisted=0
  if [[ -d "$host_run_root" ]]; then
    host_run_root_preexisted=1
  fi
  if [[ "$host_run_root_preexisted" == "1" \
    && "$RESUME" != "1" && "$OVERWRITE" != "1" ]]; then
    die "run directory appeared before exact bind reservation: $host_run_root"
  fi
  expected_bind_creation=new
  if [[ "$host_run_root_preexisted" == "1" ]]; then
    expected_bind_creation=existing
  fi
  host_run_root_identity="$(initialize_host_bind_root \
    "$host_run_root" 1 "$expected_bind_creation")" \
    || die "unable to reserve the exact per-variant bind root"
  [[ "$host_run_root_identity" =~ ^[0-9]+:[0-9]+:[0-9]+:[0-9]+:0700$ ]] \
    || die "per-variant bind root has an invalid identity"
  verify_host_bind_root_identity "$HOST_LOCK_DIR" "$HOST_LOCK_IDENTITY"
  variant_resume=0
  if [[ "$RESUME" == "1" && "$host_run_root_preexisted" == "1" ]]; then
    variant_resume=1
  elif [[ "$RESUME" == "1" ]]; then
    log "no prior run for $variant; preparing it as a fresh variant"
  fi
  variant_launch_mode=fresh
  if [[ "$OVERWRITE" == "1" ]]; then
    variant_launch_mode=overwrite
  elif [[ "$variant_resume" == "1" ]]; then
    variant_launch_mode=resume
  fi
  verify_clean_source_unchanged
  log "starting full pipeline for $variant"
  container_evidence_root="$OUTPUT_ROOT/.lumen-container-evidence"
  container_evidence_dir="$container_evidence_root/$RUN_ID-$variant"
  [[ ! -L "$container_evidence_root" && ! -L "$container_evidence_dir" ]] \
    || die "container evidence paths must not be symlinks"
  mkdir -p -- "$container_evidence_root" "$container_evidence_dir"
  chmod 700 -- "$container_evidence_root" "$container_evidence_dir"
  resolved_container_evidence_dir="$(realpath -e -- "$container_evidence_dir")" \
    || die "unable to resolve container evidence directory"
  path_contains "$OUTPUT_ROOT" "$resolved_container_evidence_dir" \
    || die "container evidence directory escaped the output root"

  training_container="$(training_container_name "$host_run_root")" \
    || die "unable to derive the stable training container name"
  declare -a training_environment_contract=()
  declare -a fresh_environment_contract=()
  declare -a resume_environment_contract=()
  declare -a overwrite_environment_contract=()
  build_training_environment_contract \
    training_environment_contract "$variant_launch_mode" "$OVERWRITE" "$variant_resume"
  build_training_environment_contract fresh_environment_contract fresh 0 0
  build_training_environment_contract resume_environment_contract resume 0 1
  build_training_environment_contract overwrite_environment_contract overwrite 1 0
  fresh_launch_contract_digest="$(training_launch_contract_for_environment \
    fresh "${fresh_environment_contract[@]}")" \
    || die "unable to derive the fresh training launch contract"
  resume_launch_contract_digest="$(training_launch_contract_for_environment \
    resume "${resume_environment_contract[@]}")" \
    || die "unable to derive the resume training launch contract"
  overwrite_launch_contract_digest="$(training_launch_contract_for_environment \
    overwrite "${overwrite_environment_contract[@]}")" \
    || die "unable to derive the overwrite training launch contract"
  case "$variant_launch_mode" in
    fresh) launch_contract_digest="$fresh_launch_contract_digest" ;;
    resume) launch_contract_digest="$resume_launch_contract_digest" ;;
    overwrite) launch_contract_digest="$overwrite_launch_contract_digest" ;;
  esac
  [[ "$launch_contract_digest" =~ ^[0-9a-f]{64}$ ]] \
    || die "unable to derive the exact training launch contract"
  for candidate_digest in \
    "$fresh_launch_contract_digest" \
    "$resume_launch_contract_digest" \
    "$overwrite_launch_contract_digest"; do
    [[ "$candidate_digest" =~ ^[0-9a-f]{64}$ ]] \
      || die "invalid training launch contract digest"
  done

  training_create_args=(
    docker create --name "$training_container" --init --gpus all --shm-size "$SHM_SIZE"
    --restart no
    --label ai.lumen.purpose=ubuntu-training
    --label "ai.lumen.host-run-root=$host_run_root"
    --label "ai.lumen.container-run-root=$container_run_root"
    --label "ai.lumen.host-run-root-identity=$host_run_root_identity"
    --label "ai.lumen.host-lock-dir=$HOST_LOCK_DIR"
    --label "ai.lumen.host-lock-identity=$HOST_LOCK_IDENTITY"
    --label "ai.lumen.source-integrity-sha256=$SOURCE_INTEGRITY_DIGEST"
    --label "ai.lumen.launch-contract-sha256=$launch_contract_digest"
    --label "ai.lumen.launch-mode=$variant_launch_mode"
    --label "ai.lumen.run-id=$RUN_ID"
    --label "ai.lumen.variant=$variant"
    --entrypoint /bin/bash
    --user "$RUNTIME_UID:$RUNTIME_GID"
    -v "$host_run_root:$container_run_root:rw"
    -v "$HOST_LOCK_DIR:$CONTAINER_LOCK_DIR:ro"
    "${cache_mounts[@]}"
  )
  for environment_entry in "${training_environment_contract[@]}"; do
    training_create_args+=( -e "$environment_entry" )
  done
  training_create_args+=(
    "$IMAGE_DIGEST"
    "$IMAGE_SOURCE_ROOT/scripts/ubuntu_train_lumen_adapters_aio.sh"
  )
  training_container_complete=0
  training_container_needs_create=0
  training_container_needs_start=0
  if docker container inspect "$training_container" >/dev/null 2>&1; then
    training_container_status="$(inspect_training_container_value \
      "$training_container" '{{.State.Status}}')" \
      || die "unable to inspect retained training container state"
    case "$training_container_status" in
      created|running|restarting|exited|paused|dead)
        retained_launch_mode="$(inspect_training_container_value \
          "$training_container" '{{index .Config.Labels "ai.lumen.launch-mode"}}')" \
          || die "unable to inspect retained training container launch mode"
        case "$retained_launch_mode" in
          fresh)
            retained_contract_digest="$fresh_launch_contract_digest"
            retained_environment_contract=("${fresh_environment_contract[@]}")
            ;;
          resume)
            retained_contract_digest="$resume_launch_contract_digest"
            retained_environment_contract=("${resume_environment_contract[@]}")
            ;;
          overwrite)
            retained_contract_digest="$overwrite_launch_contract_digest"
            retained_environment_contract=("${overwrite_environment_contract[@]}")
            ;;
          *)
            die "retained training container has an invalid launch mode: $training_container"
            ;;
        esac
        verify_training_container_identity \
          "$training_container" "$retained_contract_digest" "$host_run_root" \
          "$container_run_root" "$host_run_root_identity" "$retained_launch_mode" \
          "${retained_environment_contract[@]}"
        if [[ "$training_container_status" == "created" ]]; then
          if [[ "$variant_launch_mode" == "$retained_launch_mode" ]]; then
            :
          elif [[ "$variant_launch_mode" == "resume" \
            && "$retained_launch_mode" == "fresh" ]]; then
            log "resume will start the authenticated, never-started fresh container"
          elif [[ "$variant_launch_mode" == "resume" \
            && "$retained_launch_mode" == "overwrite" ]]; then
            die "a never-started overwrite container requires --overwrite to preserve its explicit destructive authorization: $training_container"
          else
            die "created training container cannot transition launch modes: $training_container"
          fi
        elif [[ "$training_container_status" == "running" \
          || "$training_container_status" == "restarting" ]]; then
          [[ "$variant_launch_mode" != "overwrite" ]] \
            || die "overwrite cannot attach to a running training container: $training_container"
          [[ "$variant_launch_mode" == "$retained_launch_mode" \
            || "$variant_launch_mode" == "resume" ]] \
            || die "running training container cannot transition launch modes: $training_container"
        fi
        ;;
      *)
        die "training container has an unsupported state '$training_container_status': $training_container"
        ;;
    esac
    case "$training_container_status" in
      running|restarting)
        log "reattaching to durable training container: $training_container"
        ;;
      created)
        log "starting previously created durable training container: $training_container"
        training_container_needs_start=1
        ;;
      exited)
        prior_exit_code="$(inspect_training_container_value \
          "$training_container" '{{.State.ExitCode}}')" \
          || die "unable to inspect retained training container exit status"
        [[ "$prior_exit_code" =~ ^[0-9]+$ ]] \
          || die "retained training container has an invalid exit status"
        if [[ "$prior_exit_code" == "0" ]]; then
          if verify_training_postcondition \
            "$host_run_root" "$container_run_root" "$variant" \
            "$host_run_root_identity"; then
            capture_training_container_evidence \
              "$training_container" "$container_evidence_dir" success \
              || die "unable to preserve successful container evidence"
            docker rm "$training_container" >/dev/null \
              || die "unable to remove completed training container: $training_container"
            training_container_complete=1
            log "recovered completed durable training container: $training_container"
          elif [[ "$RESUME" == "1" || "$OVERWRITE" == "1" ]]; then
            capture_training_container_evidence \
              "$training_container" "$container_evidence_dir" postcondition-failure \
              || die "unable to preserve failed postcondition evidence"
            docker rm "$training_container" >/dev/null \
              || die "unable to remove postcondition-failed container: $training_container"
            training_container_needs_create=1
            log "exited-zero container failed its postcondition; explicit recovery will recreate it"
          else
            capture_training_container_evidence \
              "$training_container" "$container_evidence_dir" postcondition-failure \
              || die "unable to preserve failed postcondition evidence"
            die "retained exited-zero container failed independent postcondition verification and remains available for inspection: $training_container (rerun with --resume or --overwrite)"
          fi
        elif [[ "$RESUME" == "1" || "$OVERWRITE" == "1" ]]; then
          capture_training_container_evidence \
            "$training_container" "$container_evidence_dir" prior-failure \
            || die "unable to preserve prior failed container evidence"
          docker rm "$training_container" >/dev/null \
            || die "unable to remove prior failed training container: $training_container"
          training_container_needs_create=1
          log "prior container exited $prior_exit_code; explicit recovery will recreate it"
        else
          capture_training_container_evidence \
            "$training_container" "$container_evidence_dir" failure \
            || die "unable to preserve failed container evidence"
          die "training container previously exited $prior_exit_code and remains available for inspection: $training_container (rerun with --resume)"
        fi
        ;;
      paused|dead)
        capture_training_container_evidence \
          "$training_container" "$container_evidence_dir" "$training_container_status" \
          || die "unable to preserve abnormal container evidence"
        die "training container is $training_container_status and remains available for inspection: $training_container"
        ;;
      *)
        die "training container has an unsupported state '$training_container_status': $training_container"
        ;;
    esac
  else
    training_container_needs_create=1
  fi

  if [[ "$training_container_complete" != "1" ]]; then
    if [[ "$training_container_needs_create" == "1" ]]; then
      created_container_id="$("${training_create_args[@]}")" \
        || die "unable to create durable training container: $training_container"
      [[ "$created_container_id" =~ ^[0-9a-f]{64}$ ]] \
        || die "Docker returned an invalid training container ID"
      verify_training_container_identity \
        "$training_container" "$launch_contract_digest" "$host_run_root" \
        "$container_run_root" "$host_run_root_identity" "$variant_launch_mode" \
        "${training_environment_contract[@]}"
      training_container_needs_start=1
      log "created durable training container: $training_container"
    fi
    if [[ "$training_container_needs_start" == "1" ]]; then
      verify_host_bind_root_identity "$host_run_root" "$host_run_root_identity"
      verify_host_bind_root_identity "$HOST_LOCK_DIR" "$HOST_LOCK_IDENTITY"
      docker start "$training_container" >/dev/null \
        || die "unable to start durable training container: $training_container"
    fi

    live_container_id="$(inspect_training_container_value \
      "$training_container" '{{.Id}}')" \
      || die "unable to inspect running training container"
    [[ "$live_container_id" =~ ^[0-9a-f]{64}$ ]] \
      || die "running training container has an invalid ID"
    live_log_path="$container_evidence_dir/live-${live_container_id:0:12}.docker.log"
    log "following $training_container; it remains running if this launcher disconnects"
    set +e
    docker logs --timestamps --follow "$training_container" 2>&1 \
      | tee "$live_log_path"
    docker_logs_status=${PIPESTATUS[0]}
    training_exit_code="$(docker wait "$training_container" 2>/dev/null)"
    docker_wait_status=$?
    set -e
    if [[ "$docker_logs_status" != "0" ]]; then
      log "warning: live Docker log streaming ended with status $docker_logs_status; preserved logs will be reconstructed from the container"
    fi
    if [[ "$docker_wait_status" != "0" || ! "$training_exit_code" =~ ^[0-9]+$ ]]; then
      capture_training_container_evidence \
        "$training_container" "$container_evidence_dir" wait-failure \
        || true
      die "lost the Docker wait channel; durable training container was retained: $training_container"
    fi
    if [[ "$training_exit_code" != "0" ]]; then
      training_oom_killed="$(inspect_training_container_value \
        "$training_container" '{{.State.OOMKilled}}')" || training_oom_killed=unknown
      training_state_error="$(inspect_training_container_value \
        "$training_container" '{{.State.Error}}')" || training_state_error=unknown
      capture_training_container_evidence \
        "$training_container" "$container_evidence_dir" failure \
        || die "training failed and its container was retained, but evidence capture failed: $training_container"
      die "training container exited $training_exit_code (OOMKilled=$training_oom_killed, stateError=$training_state_error) and was retained as $training_container; rerun with --resume"
    fi
    verify_training_postcondition \
      "$host_run_root" "$container_run_root" "$variant" \
      "$host_run_root_identity" \
      || die "exited-zero training container failed independent postcondition verification and was retained: $training_container"
    capture_training_container_evidence \
      "$training_container" "$container_evidence_dir" success \
      || die "training succeeded but container evidence capture failed: $training_container"
    docker rm "$training_container" >/dev/null \
      || die "training succeeded but its completed container could not be removed: $training_container"
  fi
  if [[ "$UPLOAD" == "1" ]]; then
    upload_flags=()
    if [[ "$HF_PRIVATE" != "1" ]]; then
      upload_flags+=(--public)
    fi
    if [[ "$CONVERT_GGUF" == "1" ]]; then
      upload_flags+=(--include-gguf)
    fi
    if [[ "$ALLOW_DIAGNOSTIC_UPLOAD" == "1" ]]; then
      upload_flags+=(--allow-diagnostic-upload)
    fi
    verify_clean_source_unchanged
    receipt_staging="$OUTPUT_ROOT/.lumen-upload-receipt-$RUN_ID-$variant"
    receipt_path="$receipt_staging/upload_receipts.json"
    final_receipt_path="$host_run_root/upload_receipts.json"
    [[ ! -L "$receipt_staging" && ! -L "$final_receipt_path" ]] \
      || die "upload receipt paths must not be symlinks"
    receipt_mount_args=()
    if [[ -e "$final_receipt_path" ]]; then
      [[ -f "$final_receipt_path" ]] \
        || die "upload receipt is not a regular file: $final_receipt_path"
      if [[ -e "$receipt_staging" ]]; then
        [[ -d "$receipt_staging" \
          && "$(stat -c '%u:%a' "$receipt_staging")" == "$RUNTIME_UID:700" ]] \
          || die "upload receipt staging is not private and process-owned"
        ! find "$receipt_staging" -mindepth 1 -maxdepth 1 -print -quit | grep -q . \
          || die "completed upload retained unexpected transaction state"
        rmdir -- "$receipt_staging"
      fi
      receipt_container_path="$container_run_root/upload_receipts.json"
      log "re-verifying the existing upload receipt against its immutable remote commit"
    else
      if [[ -e "$receipt_staging" ]]; then
        [[ -d "$receipt_staging" \
          && "$(stat -c '%u:%a' "$receipt_staging")" == "$RUNTIME_UID:700" ]] \
          || die "upload receipt staging is not private and process-owned"
      else
        mkdir -m 700 -- "$receipt_staging"
      fi
      receipt_mount_args+=( -v "$receipt_staging:/receipts:rw" )
      receipt_container_path=/receipts/upload_receipts.json
    fi
    log "uploading verified outputs in an isolated credential-scoped container"
    docker run --rm --init \
      --read-only \
      --cap-drop ALL \
      --security-opt no-new-privileges \
      --tmpfs "$PRIVATE_UPLOAD_TMPFS" \
      --entrypoint /opt/lumen-venv/bin/python \
      --user "$RUNTIME_UID:$RUNTIME_GID" \
      -v "$host_run_root:$container_run_root:ro" \
      "${receipt_mount_args[@]}" \
      "${upload_token_mount[@]}" \
      -e "HOME=$RUNTIME_HOME" \
      -e HF_HOME=/tmp/huggingface \
      -e XDG_CACHE_HOME=/tmp/cache \
      -e PYTHONDONTWRITEBYTECODE=1 \
      -e "LUMEN_UBUNTU_SOURCE_ATTESTATION_PATH=$IMAGE_SOURCE_ATTESTATION" \
      "$IMAGE_DIGEST" \
      -I "$IMAGE_SOURCE_ROOT/tools/fine_tuning/unsloth/ubuntu_uploader.py" \
      upload \
      --run-root "$container_run_root" \
      --agents "$AGENTS_CSV" \
      --run-id "$RUN_ID-$variant" \
      --token-file /run/secrets/hf_token \
      --receipt-path "$receipt_container_path" \
      "${upload_flags[@]}"
    if [[ ! -e "$final_receipt_path" ]]; then
      [[ -f "$receipt_path" && ! -L "$receipt_path" ]] \
        || die "credential-scoped upload did not produce a regular receipt"
      mv -- "$receipt_path" "$final_receipt_path"
      rmdir -- "$receipt_staging"
    fi
  fi
  log "completed $variant: $host_run_root"
done

log "all requested variants completed"
