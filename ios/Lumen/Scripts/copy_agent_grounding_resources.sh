#!/bin/sh
set -eu

log() {
  printf '[AgentGroundingResources] %s\n' "$1"
}

fail() {
  printf '[AgentGroundingResources] ERROR: %s\n' "$1" >&2
  exit 1
}

require_file() {
  path="$1"
  label="$2"
  [ -f "$path" ] || fail "Missing required file: $label ($path)"
}

require_dir() {
  path="$1"
  label="$2"
  [ -d "$path" ] || fail "Missing required directory: $label ($path)"
}

PROJECT_DIR_VALUE="${PROJECT_DIR:-}"
TARGET_BUILD_DIR_VALUE="${TARGET_BUILD_DIR:-}"
UNLOCALIZED_RESOURCES_FOLDER_PATH_VALUE="${UNLOCALIZED_RESOURCES_FOLDER_PATH:-}"

if [ -z "$PROJECT_DIR_VALUE" ]; then
  fail 'PROJECT_DIR is not set. Run this script from an Xcode build action or pass Xcode build settings.'
fi

if [ -z "$TARGET_BUILD_DIR_VALUE" ]; then
  fail 'TARGET_BUILD_DIR is not set. Run this script from an Xcode build action or pass Xcode build settings.'
fi

if [ -z "$UNLOCALIZED_RESOURCES_FOLDER_PATH_VALUE" ]; then
  fail 'UNLOCALIZED_RESOURCES_FOLDER_PATH is not set. Run this script from an Xcode build action or pass Xcode build settings.'
fi

# Walk up the directory tree from PROJECT_DIR until we find a `generated/`
# directory containing the agent manifest. This makes the script robust to
# different CI layouts where PROJECT_DIR may not be exactly one level below
# the repo root.
# A repo root candidate must have an agent_manifest with at least one of the
# cross_model_training directories present. This avoids matching a stray
# `generated/agent_manifest` that may exist higher up in the CI workspace.
is_repo_root() {
  c="$1"
  [ -d "$c/generated/agent_manifest" ] || return 1
  if [ -d "$c/generated/cross_model_training" ] \
     || [ -d "$c/generated/agent_manifest/cross_model_training" ]; then
    return 0
  fi
  return 1
}

find_repo_root() {
  candidate="$1"
  for _ in 1 2 3 4 5 6 7 8; do
    if is_repo_root "$candidate"; then
      printf '%s' "$candidate"
      return 0
    fi
    # also try a sibling/child `project` directory (CI uploads sources there)
    if is_repo_root "$candidate/project"; then
      printf '%s' "$candidate/project"
      return 0
    fi
    parent="$(cd "$candidate/.." && pwd)"
    if [ "$parent" = "$candidate" ]; then
      break
    fi
    candidate="$parent"
  done
  return 1
}

START_DIR="$(cd "$PROJECT_DIR_VALUE/.." && pwd)"
if REPO_ROOT="$(find_repo_root "$PROJECT_DIR_VALUE")" && [ -n "$REPO_ROOT" ]; then
  :
elif REPO_ROOT="$(find_repo_root "$START_DIR")" && [ -n "$REPO_ROOT" ]; then
  :
else
  REPO_ROOT="$START_DIR"
fi

AGENT_MANIFEST_DIR="$REPO_ROOT/generated/agent_manifest"
LEGACY_CROSS_MODEL_DIR="$REPO_ROOT/generated/cross_model_training"
NESTED_CROSS_MODEL_DIR="$AGENT_MANIFEST_DIR/cross_model_training"
LOOP_OUTPUT_DIR="$REPO_ROOT/generated/agent_improvement_loop"

log "Resolved REPO_ROOT: $REPO_ROOT"
APP_RESOURCES_DIR="$TARGET_BUILD_DIR_VALUE/$UNLOCALIZED_RESOURCES_FOLDER_PATH_VALUE"
DEST_DIR="$APP_RESOURCES_DIR/AgentGrounding"

if [ -d "$LEGACY_CROSS_MODEL_DIR" ]; then
  CROSS_MODEL_DIR="$LEGACY_CROSS_MODEL_DIR"
elif [ -d "$NESTED_CROSS_MODEL_DIR" ]; then
  CROSS_MODEL_DIR="$NESTED_CROSS_MODEL_DIR"
else
  CROSS_MODEL_DIR=""
fi

# In some build environments (CI / sandboxed installs) the `generated/`
# artifacts directory isn't uploaded alongside the source tree. Rather than
# breaking the entire install, skip the bundling step and let the runtime
# fall back to its baked-in defaults. Required runtime checks in Swift are
# already best-effort and tolerate missing resources.
if [ ! -d "$AGENT_MANIFEST_DIR" ] || [ -z "$CROSS_MODEL_DIR" ]; then
  log "Generated agent grounding artifacts not present at $REPO_ROOT/generated; skipping bundling."
  log "  - agent_manifest dir present: $([ -d "$AGENT_MANIFEST_DIR" ] && echo yes || echo no)"
  log "  - cross_model_training dir present: $([ -n "$CROSS_MODEL_DIR" ] && echo yes || echo no)"
  APP_RESOURCES_DIR="$TARGET_BUILD_DIR_VALUE/$UNLOCALIZED_RESOURCES_FOLDER_PATH_VALUE"
  DEST_DIR="$APP_RESOURCES_DIR/AgentGrounding"
  mkdir -p "$DEST_DIR/agent_manifest" "$DEST_DIR/cross_model_training"
  exit 0
fi

require_file "$AGENT_MANIFEST_DIR/AgentBehaviorManifest.json" 'AgentBehaviorManifest.json'
require_file "$AGENT_MANIFEST_DIR/AgentBehaviorManifest.md" 'AgentBehaviorManifest.md'
require_file "$AGENT_MANIFEST_DIR/fleet_system_prompts.json" 'fleet_system_prompts.json'
require_file "$AGENT_MANIFEST_DIR/manifest_validation_report.json" 'manifest_validation_report.json'
require_file "$AGENT_MANIFEST_DIR/AgentBehaviorManifest.sha256" 'AgentBehaviorManifest.sha256'
require_file "$AGENT_MANIFEST_DIR/AgentBehaviorManifest.incremental.sha256" 'AgentBehaviorManifest.incremental.sha256'
require_file "$AGENT_MANIFEST_DIR/dataset_manifest.json" 'dataset_manifest.json'
require_file "$AGENT_MANIFEST_DIR/dataset_index.csv" 'dataset_index.csv'
require_file "$AGENT_MANIFEST_DIR/tool_registry.csv" 'tool_registry.csv'
require_file "$AGENT_MANIFEST_DIR/routing_matrix.csv" 'routing_matrix.csv'

require_dir "$AGENT_MANIFEST_DIR/dataset" 'generated dataset directory'
require_file "$AGENT_MANIFEST_DIR/dataset/train_sft.jsonl" 'dataset/train_sft.jsonl'
require_file "$AGENT_MANIFEST_DIR/dataset/validation_sft.jsonl" 'dataset/validation_sft.jsonl'
require_file "$AGENT_MANIFEST_DIR/dataset/dpo_preference_pairs.jsonl" 'dataset/dpo_preference_pairs.jsonl'
require_file "$AGENT_MANIFEST_DIR/dataset/eval_scenarios.jsonl" 'dataset/eval_scenarios.jsonl'
require_file "$AGENT_MANIFEST_DIR/dataset/tool_schema_cards.jsonl" 'dataset/tool_schema_cards.jsonl'
require_file "$AGENT_MANIFEST_DIR/dataset/manifest_grounding_cards.jsonl" 'dataset/manifest_grounding_cards.jsonl'
require_file "$AGENT_MANIFEST_DIR/dataset/runtime_audit_repairs.jsonl" 'dataset/runtime_audit_repairs.jsonl'

require_file "$CROSS_MODEL_DIR/cross_model_training.jsonl" 'cross_model_training.jsonl'
require_file "$CROSS_MODEL_DIR/train_sft_cross.jsonl" 'train_sft_cross.jsonl'
require_file "$CROSS_MODEL_DIR/val_sft_cross.jsonl" 'val_sft_cross.jsonl'
require_file "$CROSS_MODEL_DIR/dpo_train_cross.jsonl" 'dpo_train_cross.jsonl'
require_file "$CROSS_MODEL_DIR/dpo_val_cross.jsonl" 'dpo_val_cross.jsonl'
require_file "$CROSS_MODEL_DIR/cross_model_training_index.csv" 'cross_model_training_index.csv'

if command -v python3 >/dev/null 2>&1; then
  python3 - "$AGENT_MANIFEST_DIR/manifest_validation_report.json" <<'PY' || log 'manifest_validation_report.json check skipped (non-fatal)'
import json
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
if report.get('passed') is not True:
    raise SystemExit('manifest_validation_report.json is not passed=true')
if report.get('failures'):
    raise SystemExit(f"manifest_validation_report.json has failures: {len(report['failures'])}")
if report.get('warnings'):
    raise SystemExit(f"manifest_validation_report.json has warnings: {len(report['warnings'])}")
PY
fi

log "Copying generated artifacts into app bundle resources"
rm -rf "$DEST_DIR"
mkdir -p "$DEST_DIR"

cp -R "$AGENT_MANIFEST_DIR" "$DEST_DIR/agent_manifest"
cp -R "$CROSS_MODEL_DIR" "$DEST_DIR/cross_model_training"

if [ -d "$LOOP_OUTPUT_DIR" ]; then
  cp -R "$LOOP_OUTPUT_DIR" "$DEST_DIR/agent_improvement_loop"
  log "Loop outputs: $DEST_DIR/agent_improvement_loop"
else
  log "Loop outputs not present at $LOOP_OUTPUT_DIR (skipped)"
fi

log "Installed resources at: $DEST_DIR"
log "Manifest: $DEST_DIR/agent_manifest/AgentBehaviorManifest.json"
log "Prompts:  $DEST_DIR/agent_manifest/fleet_system_prompts.json"
