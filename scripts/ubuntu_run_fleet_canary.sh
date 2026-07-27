#!/usr/bin/env bash
set -Eeuo pipefail
umask 077

IFS=$'\n\t'

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
PIPELINE="$ROOT/scripts/ubuntu_train_lumen_full_pipeline.sh"
SOURCE_ATTESTOR="$ROOT/tools/fine_tuning/unsloth/ubuntu_source_integrity.py"

WORKSPACE_ROOT="${LUMEN_FLEET_CANARY_WORKSPACE_ROOT:-}"
EXPECTED_COMMIT="${LUMEN_FLEET_CANARY_EXPECTED_COMMIT:-}"
RUN_ID="${LUMEN_FLEET_CANARY_RUN_ID:-fleetcanary-$(date -u +%Y%m%dT%H%M%SZ)}"
OLLAMA_CONTAINER="${LUMEN_FLEET_CANARY_OLLAMA_CONTAINER:-mongars-ollama-1}"
HOST_STATE_ROOT="/var/tmp/lumen-fleet-canary-state"
MIN_DOCKER_FREE_BYTES=42949672960
MIN_WORKSPACE_FREE_BYTES=53687091200
MIN_FREE_VRAM_MIB=7168
GPU_WAIT_SECONDS="${LUMEN_FLEET_CANARY_GPU_WAIT_SECONDS:-120}"
HEALTH_WAIT_SECONDS="${LUMEN_FLEET_CANARY_HEALTH_WAIT_SECONDS:-180}"
CONFIRM_STOP_OLLAMA=0
PREFLIGHT_ONLY=0

STATE_DIR=""
OUTPUT_ROOT=""
HF_CACHE=""
EXPECTED_RUN_ROOT=""
RESTORE_MARKER=""
ORIGINAL_CONTAINER_ID=""
ORIGINAL_RESTART_POLICY=""
RESTORE_REQUIRED=0

log() {
  printf '[lumen-fleet-canary] %s\n' "$*"
}

error() {
  printf '[lumen-fleet-canary] ERROR: %s\n' "$*" >&2
}

die() {
  error "$*"
  exit 1
}

usage() {
  cat <<'EOF'
Run the strict, fresh Fleet-only Ubuntu qualification canary while safely
quiescing and restoring the local Ollama service.

Usage:
  bash scripts/ubuntu_run_fleet_canary.sh \
    --workspace-root /absolute/path/to/private-workspace \
    --expected-commit <40-character-git-commit> \
    --confirm-stop-ollama \
    [--run-id ID]

Options:
  --workspace-root DIR
                       Existing or creatable private workspace. The wrapper
                       uses DIR/output and DIR/hf-cache.
  --expected-commit SHA
                       Require the clean checkout HEAD and source attestation
                       to match this exact 40-character commit.
  --run-id ID          Stable run identifier (default: fleetcanary-<UTC>).
  --ollama-container NAME
                       Ollama container to manage (default:
                       mongars-ollama-1).
  --confirm-stop-ollama
                       Explicitly authorize stopping the healthy Ollama
                       container and restoring the same container afterward.
  --preflight-only     Check source, paths, disk, Docker, GPU, and container
                       state without stopping Ollama or launching training.
  -h, --help           Show this help.

The training contract is fixed: build the pinned image, optimized variant,
Fleet only, fresh/non-resumed SFT+DPO, full 15-case evaluation, no GGUF,
no upload, no overwrite, and no smoke/disabled evaluation.

The wrapper never prunes Docker data or changes unrelated images, volumes, or
containers. It does not change source files or generated artifacts. Its
host-global lock and durable Ollama restore marker live under
/var/tmp/lumen-fleet-canary-state.
EOF
}

require_uint() {
  local name="$1"
  local value="$2"
  [[ "$value" =~ ^[0-9]+$ ]] || die "$name must be a non-negative integer (got: $value)"
}

path_contains() {
  local parent="$1"
  local child="$2"
  [[ "$child" == "$parent" || "$child" == "$parent/"* ]]
}

canonical_existing_directory() {
  python3 -I - "$1" <<'PY'
import os
import pathlib
import stat
import sys

path = pathlib.Path(sys.argv[1])
metadata = path.lstat()
if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
    raise SystemExit(f"not a regular directory: {path}")
print(path.resolve(strict=True))
PY
}

canonical_directory_target() {
  python3 -I - "$1" <<'PY'
import pathlib
import stat
import sys

path = pathlib.Path(sys.argv[1])
if path.exists() or path.is_symlink():
    metadata = path.lstat()
    if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
        raise SystemExit(f"not a regular directory: {path}")
print(path.resolve(strict=False))
PY
}

ensure_private_directory() {
  python3 -I - "$1" "$(id -u)" <<'PY'
import os
import pathlib
import stat
import sys

path = pathlib.Path(sys.argv[1])
expected_uid = int(sys.argv[2])
path.mkdir(mode=0o700, parents=True, exist_ok=True)
metadata = path.lstat()
if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
    raise SystemExit(f"private path is not a regular directory: {path}")
if metadata.st_uid != expected_uid:
    raise SystemExit(f"private path is not owned by the invoking user: {path}")
os.chmod(path, 0o700, follow_symlinks=False)
metadata = path.lstat()
if stat.S_IMODE(metadata.st_mode) != 0o700:
    raise SystemExit(f"private path mode is not 0700: {path}")
print(path.resolve(strict=True))
PY
}

prepare_host_state_directory() {
  python3 -I - "$1" "$(id -u)" <<'PY'
import pathlib
import stat
import sys

path = pathlib.Path(sys.argv[1])
expected_uid = int(sys.argv[2])
try:
    path.mkdir(mode=0o700)
except FileExistsError:
    pass
metadata = path.lstat()
if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
    raise SystemExit(f"host state path is not a regular directory: {path}")
if metadata.st_uid != expected_uid:
    raise SystemExit(f"host state path is not owned by the invoking user: {path}")
if stat.S_IMODE(metadata.st_mode) != 0o700:
    raise SystemExit(f"host state path mode is not 0700: {path}")
print(path.resolve(strict=True))
PY
}

prepare_host_lock_file() {
  python3 -I - "$1" "$(id -u)" <<'PY'
import os
import pathlib
import stat
import sys

path = pathlib.Path(sys.argv[1])
expected_uid = int(sys.argv[2])
flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
descriptor = os.open(path, flags, 0o600)
try:
    metadata = os.fstat(descriptor)
    if not stat.S_ISREG(metadata.st_mode):
        raise SystemExit(f"host lock path is not a regular file: {path}")
    if metadata.st_uid != expected_uid:
        raise SystemExit(f"host lock path is not owned by the invoking user: {path}")
    if stat.S_IMODE(metadata.st_mode) != 0o600:
        raise SystemExit(f"host lock path mode is not 0600: {path}")
finally:
    os.close(descriptor)
print(path.resolve(strict=True))
PY
}

available_bytes() {
  python3 -I - "$1" <<'PY'
import os
import sys

stats = os.statvfs(sys.argv[1])
print(stats.f_bavail * stats.f_frsize)
PY
}

gpu_snapshot() {
  nvidia-smi \
    --query-gpu=memory.total,memory.free \
    --format=csv,noheader,nounits \
    | python3 -I -c '
import re
import sys

rows = [row.strip() for row in sys.stdin if row.strip()]
if len(rows) != 1:
    raise SystemExit("strict Fleet canary requires exactly one visible NVIDIA GPU")
parts = [part.strip() for part in rows[0].split(",")]
if len(parts) != 2 or any(re.fullmatch(r"[0-9]+", part) is None for part in parts):
    raise SystemExit("nvidia-smi returned malformed GPU memory evidence")
total, free = map(int, parts)
if total <= 0 or free < 0 or free > total:
    raise SystemExit("nvidia-smi returned invalid GPU memory evidence")
print(total, free)
'
}

write_restore_marker() {
  local created_at
  created_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  python3 -I - \
    "$RESTORE_MARKER" \
    "$OLLAMA_CONTAINER" \
    "$ORIGINAL_CONTAINER_ID" \
    "$ORIGINAL_RESTART_POLICY" \
    "$RUN_ID" \
    "$EXPECTED_COMMIT" \
    "$EXPECTED_RUN_ROOT" \
    "$created_at" <<'PY'
import json
import os
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
payload = {
    "schemaVersion": "lumen.ollama-restore-required/1.0.0",
    "containerName": sys.argv[2],
    "containerID": sys.argv[3],
    "restartPolicy": sys.argv[4],
    "runID": sys.argv[5],
    "sourceCommit": sys.argv[6],
    "expectedRunRoot": sys.argv[7],
    "createdAtUTC": sys.argv[8],
}
flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
descriptor = os.open(path, flags, 0o600)
try:
    data = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    written = 0
    while written < len(data):
        count = os.write(descriptor, data[written:])
        if count <= 0:
            raise OSError("short write while recording Ollama restoration state")
        written += count
    os.fsync(descriptor)
finally:
    os.close(descriptor)
directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
try:
    os.fsync(directory)
finally:
    os.close(directory)
PY
}

remove_restore_marker() {
  python3 -I - "$RESTORE_MARKER" "$(id -u)" <<'PY'
import os
import pathlib
import stat
import sys

path = pathlib.Path(sys.argv[1])
expected_uid = int(sys.argv[2])
metadata = path.lstat()
if not stat.S_ISREG(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
    raise SystemExit(f"restore marker is not a regular file: {path}")
if metadata.st_uid != expected_uid or stat.S_IMODE(metadata.st_mode) != 0o600:
    raise SystemExit(f"restore marker ownership or mode drifted: {path}")
path.unlink()
directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
try:
    os.fsync(directory)
finally:
    os.close(directory)
PY
}

wait_for_ollama_health() {
  local deadline snapshot running health
  deadline=$((SECONDS + HEALTH_WAIT_SECONDS))
  while true; do
    snapshot="$(docker inspect \
      --format '{{.State.Running}}|{{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}' \
      "$ORIGINAL_CONTAINER_ID" 2>/dev/null)" || return 1
    IFS='|' read -r running health <<< "$snapshot"
    if [[ "$running" == "true" && "$health" == "healthy" ]]; then
      return 0
    fi
    if [[ "$health" == "unhealthy" || $SECONDS -ge $deadline ]]; then
      return 1
    fi
    sleep 2
  done
}

running_lumen_training_containers() {
  docker ps \
    --filter label=ai.lumen.purpose=ubuntu-training \
    --format '{{.ID}}'
}

cleanup() {
  local original_status="$?"
  local final_status="$original_status"
  local restore_failed=0
  local running_training current_id current_policy current_running

  trap - EXIT INT TERM HUP
  set +e

  if [[ "$RESTORE_REQUIRED" == "1" ]]; then
    running_training="$(running_lumen_training_containers 2>/dev/null)"
    if [[ "$?" != "0" ]]; then
      error "Docker became unavailable; leaving restore marker: $RESTORE_MARKER"
      restore_failed=1
    elif [[ -n "$running_training" ]]; then
      error "a durable Lumen training container is still running; refusing to restart Ollama into GPU contention"
      error "leave $RESTORE_MARKER in place and reattach with the canonical --resume workflow"
      restore_failed=1
    else
      current_id="$(docker inspect --format '{{.Id}}' "$ORIGINAL_CONTAINER_ID" 2>/dev/null)"
      current_policy="$(docker inspect \
        --format '{{.HostConfig.RestartPolicy.Name}}' \
        "$ORIGINAL_CONTAINER_ID" 2>/dev/null)"
      current_running="$(docker inspect \
        --format '{{.State.Running}}' \
        "$ORIGINAL_CONTAINER_ID" 2>/dev/null)"
      if [[ "$current_id" != "$ORIGINAL_CONTAINER_ID" \
        || "$current_policy" != "$ORIGINAL_RESTART_POLICY" \
        || ( "$current_running" != "true" && "$current_running" != "false" ) ]]; then
        error "the original Ollama container identity or restart policy drifted; refusing to start a replacement"
        restore_failed=1
      else
        if [[ "$current_running" != "true" ]]; then
          log "restoring Ollama container: $ORIGINAL_CONTAINER_ID"
          docker start "$ORIGINAL_CONTAINER_ID" >/dev/null || restore_failed=1
        fi
        if [[ "$restore_failed" == "0" ]] && ! wait_for_ollama_health; then
          error "Ollama did not return to running+healthy within ${HEALTH_WAIT_SECONDS}s"
          restore_failed=1
        fi
        if [[ "$restore_failed" == "0" ]]; then
          remove_restore_marker || restore_failed=1
        fi
      fi
    fi

    if [[ "$restore_failed" == "0" ]]; then
      RESTORE_REQUIRED=0
      log "Ollama restored and healthy"
    else
      error "manual recovery is required; marker retained at $RESTORE_MARKER"
      if [[ "$original_status" == "0" ]]; then
        final_status=70
      else
        error "the canonical pipeline also exited with status $original_status"
        final_status=71
      fi
    fi
  fi

  exit "$final_status"
}

handle_signal() {
  local status="$1"
  error "received a termination signal; preserving run state"
  exit "$status"
}

while (($#)); do
  case "$1" in
    --workspace-root)
      (($# >= 2)) || die "--workspace-root requires a value"
      WORKSPACE_ROOT="$2"
      shift 2
      ;;
    --expected-commit)
      (($# >= 2)) || die "--expected-commit requires a value"
      EXPECTED_COMMIT="$2"
      shift 2
      ;;
    --run-id)
      (($# >= 2)) || die "--run-id requires a value"
      RUN_ID="$2"
      shift 2
      ;;
    --ollama-container)
      (($# >= 2)) || die "--ollama-container requires a value"
      OLLAMA_CONTAINER="$2"
      shift 2
      ;;
    --confirm-stop-ollama)
      CONFIRM_STOP_OLLAMA=1
      shift
      ;;
    --preflight-only)
      PREFLIGHT_ONLY=1
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

[[ "$(uname -s)" == "Linux" ]] \
  || die "this strict canary wrapper must run on the Ubuntu GPU host"
[[ -n "$WORKSPACE_ROOT" && "$WORKSPACE_ROOT" == /* ]] \
  || die "--workspace-root must be an absolute path"
[[ "$WORKSPACE_ROOT" != *$'\n'* ]] \
  || die "--workspace-root must not contain a newline"
[[ "$EXPECTED_COMMIT" =~ ^[0-9a-f]{40}$ ]] \
  || die "--expected-commit must be a lowercase 40-character Git commit"
[[ "$RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]] \
  || die "run ID must contain only letters, digits, dot, underscore, and hyphen"
[[ "$OLLAMA_CONTAINER" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$ ]] \
  || die "Ollama container name is invalid"

require_uint LUMEN_FLEET_CANARY_GPU_WAIT_SECONDS "$GPU_WAIT_SECONDS"
require_uint LUMEN_FLEET_CANARY_HEALTH_WAIT_SECONDS "$HEALTH_WAIT_SECONDS"

for command in git python3 docker nvidia-smi flock grep; do
  command -v "$command" >/dev/null 2>&1 || die "required command not found: $command"
done
[[ -f "$PIPELINE" && ! -L "$PIPELINE" ]] || die "missing canonical launcher: $PIPELINE"
[[ -f "$SOURCE_ATTESTOR" && ! -L "$SOURCE_ATTESTOR" ]] \
  || die "missing source attestor: $SOURCE_ATTESTOR"
[[ "$(id -u)" != "0" ]] || die "run as a regular non-root Docker user"

REPO_TOP="$(git -C "$ROOT" rev-parse --show-toplevel 2>/dev/null)" \
  || die "repository root is not a Git checkout"
REPO_TOP="$(canonical_existing_directory "$REPO_TOP")" \
  || die "unable to resolve repository root"
[[ "$REPO_TOP" == "$ROOT" ]] || die "launcher root does not match the Git worktree"

CURRENT_COMMIT="$(git -C "$ROOT" rev-parse HEAD 2>/dev/null)" \
  || die "unable to resolve checkout HEAD"
[[ "$CURRENT_COMMIT" == "$EXPECTED_COMMIT" ]] \
  || die "checkout HEAD $CURRENT_COMMIT does not match --expected-commit $EXPECTED_COMMIT"
[[ -z "$(git -C "$ROOT" status --porcelain=v1 --untracked-files=all)" ]] \
  || die "strict canary requires a completely clean checkout"

ATTESTATION_JSON="$(python3 -I "$SOURCE_ATTESTOR" attest-host --root "$ROOT")" \
  || die "source-integrity attestation failed"
ATTESTED_COMMIT="$(printf '%s' "$ATTESTATION_JSON" | python3 -I -c '
import json
import sys
value = json.load(sys.stdin)
commit = value.get("baseCommit")
if not isinstance(commit, str):
    raise SystemExit("source attestation omitted baseCommit")
print(commit)
')" || die "source-integrity attestation is malformed"
[[ "$ATTESTED_COMMIT" == "$EXPECTED_COMMIT" ]] \
  || die "source attestation commit does not match --expected-commit"

PROSPECTIVE_WORKSPACE_ROOT="$(canonical_directory_target "$WORKSPACE_ROOT")" \
  || die "unable to resolve prospective workspace root"
if path_contains "$PROSPECTIVE_WORKSPACE_ROOT" "$ROOT" \
  || path_contains "$ROOT" "$PROSPECTIVE_WORKSPACE_ROOT"; then
  die "workspace root overlaps the source checkout: $PROSPECTIVE_WORKSPACE_ROOT"
fi
if path_contains "$PROSPECTIVE_WORKSPACE_ROOT" "$HOST_STATE_ROOT" \
  || path_contains "$HOST_STATE_ROOT" "$PROSPECTIVE_WORKSPACE_ROOT"; then
  die "workspace root overlaps persistent host state: $PROSPECTIVE_WORKSPACE_ROOT"
fi

STATE_DIR="$(prepare_host_state_directory "$HOST_STATE_ROOT")" \
  || die "unable to prepare persistent host-global operator state"
HOST_LOCK_FILE="$(prepare_host_lock_file "$STATE_DIR/fleet-canary.lock")" \
  || die "host-global Fleet canary lock is missing or unsafe"
exec 8<>"$HOST_LOCK_FILE" \
  || die "unable to open host-global Fleet canary lock: $HOST_LOCK_FILE"
LUMEN_FLEET_CANARY_ACTIVE_LOCK_PATH="$HOST_LOCK_FILE" flock -n 8 \
  || die "another strict Fleet canary wrapper holds $HOST_LOCK_FILE"

RESTORE_MARKER="$STATE_DIR/ollama-restore-required"
[[ ! -e "$RESTORE_MARKER" && ! -L "$RESTORE_MARKER" ]] \
  || die "a prior Ollama restore marker requires manual recovery: $RESTORE_MARKER"

if [[ -e "$WORKSPACE_ROOT" || -L "$WORKSPACE_ROOT" ]]; then
  [[ -d "$WORKSPACE_ROOT" && ! -L "$WORKSPACE_ROOT" ]] \
    || die "workspace root must be a regular directory"
else
  mkdir -m 700 -p -- "$WORKSPACE_ROOT" \
    || die "unable to create workspace root: $WORKSPACE_ROOT"
fi
WORKSPACE_ROOT="$(canonical_existing_directory "$WORKSPACE_ROOT")" \
  || die "unable to resolve workspace root"
[[ "$WORKSPACE_ROOT" == "$PROSPECTIVE_WORKSPACE_ROOT" ]] \
  || die "workspace root identity changed while preparing the canary"

OUTPUT_ROOT="$(ensure_private_directory "$WORKSPACE_ROOT/output")" \
  || die "unable to prepare private output root"
HF_CACHE="$(ensure_private_directory "$WORKSPACE_ROOT/hf-cache")" \
  || die "unable to prepare private Hugging Face cache"

for managed_path in "$OUTPUT_ROOT" "$HF_CACHE"; do
  if path_contains "$managed_path" "$ROOT" || path_contains "$ROOT" "$managed_path"; then
    die "managed canary path overlaps the source checkout: $managed_path"
  fi
done
if path_contains "$OUTPUT_ROOT" "$HF_CACHE" || path_contains "$HF_CACHE" "$OUTPUT_ROOT"; then
  die "output and Hugging Face cache paths must not overlap"
fi

EXPECTED_RUN_ROOT="$OUTPUT_ROOT/$RUN_ID-internal_plus_public_optimized"
[[ ! -e "$EXPECTED_RUN_ROOT" && ! -L "$EXPECTED_RUN_ROOT" ]] \
  || die "fresh strict run already exists: $EXPECTED_RUN_ROOT"

docker info >/dev/null 2>&1 || die "Docker daemon is unavailable for the current user"
DOCKER_ROOT="$(docker info --format '{{.DockerRootDir}}')" \
  || die "unable to resolve Docker storage root"
[[ -n "$DOCKER_ROOT" && "$DOCKER_ROOT" == /* && -d "$DOCKER_ROOT" ]] \
  || die "Docker returned an invalid storage root: $DOCKER_ROOT"

DOCKER_FREE_BYTES="$(available_bytes "$DOCKER_ROOT")" \
  || die "unable to measure Docker storage availability"
WORKSPACE_FREE_BYTES="$(available_bytes "$WORKSPACE_ROOT")" \
  || die "unable to measure workspace availability"
(( DOCKER_FREE_BYTES >= MIN_DOCKER_FREE_BYTES )) \
  || die "Docker storage has $DOCKER_FREE_BYTES free bytes; need at least $MIN_DOCKER_FREE_BYTES"
(( WORKSPACE_FREE_BYTES >= MIN_WORKSPACE_FREE_BYTES )) \
  || die "workspace has $WORKSPACE_FREE_BYTES free bytes; need at least $MIN_WORKSPACE_FREE_BYTES"

RUNNING_TRAINING="$(running_lumen_training_containers)" \
  || die "unable to inspect running Lumen training containers"
[[ -z "$RUNNING_TRAINING" ]] \
  || die "a Lumen training container is already running: $RUNNING_TRAINING"

OLLAMA_SNAPSHOT="$(docker inspect \
  --format '{{.Id}}|{{.State.Status}}|{{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}|{{.HostConfig.RestartPolicy.Name}}' \
  "$OLLAMA_CONTAINER" 2>/dev/null)" \
  || die "required Ollama container does not exist: $OLLAMA_CONTAINER"
IFS='|' read -r ORIGINAL_CONTAINER_ID OLLAMA_STATUS OLLAMA_HEALTH ORIGINAL_RESTART_POLICY \
  <<< "$OLLAMA_SNAPSHOT"
[[ "$ORIGINAL_CONTAINER_ID" =~ ^[0-9a-f]{64}$ ]] \
  || die "Docker returned an invalid Ollama container ID"
[[ "$OLLAMA_STATUS" == "running" && "$OLLAMA_HEALTH" == "healthy" ]] \
  || die "Ollama must be running and healthy before managed quiescence (status=$OLLAMA_STATUS health=$OLLAMA_HEALTH)"
[[ "$ORIGINAL_RESTART_POLICY" == "unless-stopped" ]] \
  || die "unexpected Ollama restart policy: $ORIGINAL_RESTART_POLICY"

GPU_SNAPSHOT="$(gpu_snapshot)" \
  || die "unable to read exact GPU memory state"
IFS=' ' read -r GPU_TOTAL_MIB GPU_FREE_MIB <<< "$GPU_SNAPSHOT" \
  || die "unable to parse exact GPU memory state"
(( GPU_TOTAL_MIB >= MIN_FREE_VRAM_MIB )) \
  || die "visible GPU has only $GPU_TOTAL_MIB MiB total; threshold is $MIN_FREE_VRAM_MIB MiB"

log "source commit: $EXPECTED_COMMIT"
log "workspace: $WORKSPACE_ROOT"
log "output root: $OUTPUT_ROOT"
log "HF cache: $HF_CACHE"
log "Docker free bytes: $DOCKER_FREE_BYTES"
log "workspace free bytes: $WORKSPACE_FREE_BYTES"
log "GPU memory before Ollama quiescence: total=${GPU_TOTAL_MIB}MiB free=${GPU_FREE_MIB}MiB"
log "Ollama: $OLLAMA_CONTAINER id=$ORIGINAL_CONTAINER_ID status=$OLLAMA_STATUS health=$OLLAMA_HEALTH policy=$ORIGINAL_RESTART_POLICY"

if [[ "$PREFLIGHT_ONLY" == "1" ]]; then
  log "preflight complete; Ollama was not stopped and training was not launched"
  exit 0
fi
[[ "$CONFIRM_STOP_OLLAMA" == "1" ]] \
  || die "actual launch requires --confirm-stop-ollama"

trap cleanup EXIT
trap 'handle_signal 129' HUP
trap 'handle_signal 130' INT
trap 'handle_signal 143' TERM

write_restore_marker || die "unable to durably record Ollama restoration state"
RESTORE_REQUIRED=1
log "stopping Ollama container before GPU allocation"
docker stop --time 30 "$ORIGINAL_CONTAINER_ID" >/dev/null \
  || die "failed to stop the original Ollama container"

STOPPED_STATE="$(docker inspect --format '{{.State.Running}}' "$ORIGINAL_CONTAINER_ID")" \
  || die "unable to verify the stopped Ollama container"
[[ "$STOPPED_STATE" == "false" ]] \
  || die "Ollama container remained running after docker stop"

deadline=$((SECONDS + GPU_WAIT_SECONDS))
while true; do
  GPU_SNAPSHOT="$(gpu_snapshot)" \
    || die "unable to read GPU memory after Ollama stop"
  IFS=' ' read -r GPU_TOTAL_MIB GPU_FREE_MIB <<< "$GPU_SNAPSHOT" \
    || die "unable to parse GPU memory after Ollama stop"
  if (( GPU_FREE_MIB >= MIN_FREE_VRAM_MIB )); then
    break
  fi
  if (( SECONDS >= deadline )); then
    die "GPU retained only ${GPU_FREE_MIB}MiB free after Ollama stop; need ${MIN_FREE_VRAM_MIB}MiB"
  fi
  sleep 2
done

GPU_COMPUTE_PROCESSES="$(nvidia-smi \
  --query-compute-apps=process_name \
  --format=csv,noheader,nounits)" \
  || die "unable to inspect GPU compute processes after Ollama stop"
if printf '%s\n' "$GPU_COMPUTE_PROCESSES" \
  | grep -Eq '(^|/)(llama-server|ollama|ollama_llama_server)$'; then
  die "an Ollama GPU process remained after container stop"
fi
log "GPU ready: total=${GPU_TOTAL_MIB}MiB free=${GPU_FREE_MIB}MiB"

EXPORTED_ENVIRONMENT_NAMES="$(compgen -e)" \
  || die "unable to enumerate inherited environment variables"
declare -a SANITIZED_ENV=(env)
while IFS= read -r environment_name; do
  case "$environment_name" in
    LUMEN_UBUNTU_*) SANITIZED_ENV+=( -u "$environment_name" ) ;;
  esac
done <<< "$EXPORTED_ENVIRONMENT_NAMES"
SANITIZED_ENV+=(
  LUMEN_UBUNTU_IMAGE_TAG=lumen-training:cu128-py310
  LUMEN_UBUNTU_OUTPUT_ROOT="$OUTPUT_ROOT"
  LUMEN_UBUNTU_HF_CACHE="$HF_CACHE"
  LUMEN_UBUNTU_HF_TOKEN_FILE=
  LUMEN_UBUNTU_RUN_ID="$RUN_ID"
  LUMEN_UBUNTU_EXPECTED_SOURCE_COMMIT="$EXPECTED_COMMIT"
  LUMEN_UBUNTU_HOST_RESERVATION_FD=8
  LUMEN_UBUNTU_AGENTS=fleet
  LUMEN_UBUNTU_VARIANT=optimized
  LUMEN_UBUNTU_SHM_SIZE=8g
  LUMEN_UBUNTU_BUILD_IMAGE=1
  LUMEN_UBUNTU_PULL_BASE=1
  LUMEN_UBUNTU_UPLOAD=0
  LUMEN_UBUNTU_ALLOW_DIAGNOSTIC_UPLOAD=0
  LUMEN_UBUNTU_HF_PRIVATE=1
  LUMEN_UBUNTU_OVERWRITE=0
  LUMEN_UBUNTU_RESUME=0
  LUMEN_UBUNTU_PREPARE_ONLY=0
  LUMEN_UBUNTU_CONVERT_GGUF=0
  LUMEN_UBUNTU_EVALUATE=1
  LUMEN_UBUNTU_EVAL_MAX_EXAMPLES=
)

log "launching strict fresh Fleet canary: $RUN_ID"
set +e
"${SANITIZED_ENV[@]}" \
  bash "$PIPELINE" \
  --variant optimized \
  --agents fleet \
  --no-gguf \
  --run-id "$RUN_ID" \
  --output-dir "$OUTPUT_ROOT" \
  --hf-cache "$HF_CACHE" \
  --expected-source-commit "$EXPECTED_COMMIT"
PIPELINE_STATUS="$?"
set -e

if [[ "$PIPELINE_STATUS" == "0" ]]; then
  log "canonical pipeline and independent postcondition passed: $EXPECTED_RUN_ROOT"
else
  error "canonical pipeline exited $PIPELINE_STATUS; preserving its run evidence at $EXPECTED_RUN_ROOT"
fi
exit "$PIPELINE_STATUS"
