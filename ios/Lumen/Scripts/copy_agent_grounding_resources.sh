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
CONFIGURATION_VALUE="${CONFIGURATION:-}"
if [ -n "${AGENT_GROUNDING_RESOURCE_MODE:-}" ]; then
  AGENT_GROUNDING_RESOURCE_MODE_VALUE="$AGENT_GROUNDING_RESOURCE_MODE"
elif [ "$CONFIGURATION_VALUE" = "Release" ]; then
  AGENT_GROUNDING_RESOURCE_MODE_VALUE="minimal"
else
  AGENT_GROUNDING_RESOURCE_MODE_VALUE="full"
fi

case "$AGENT_GROUNDING_RESOURCE_MODE_VALUE" in
  full|minimal|skip)
    ;;
  *)
    fail "Unsupported AGENT_GROUNDING_RESOURCE_MODE=$AGENT_GROUNDING_RESOURCE_MODE_VALUE (expected full, minimal, or skip)"
    ;;
esac

# Only an explicit Debug build using the developer-oriented full/skip modes may
# fall back to an empty bundle for diagnostics. Release, minimal, and invocations
# without a known build configuration fail closed.
ALLOW_DEBUG_RUNTIME_FALLBACK=0
if [ "$CONFIGURATION_VALUE" = "Debug" ] \
   && [ "$AGENT_GROUNDING_RESOURCE_MODE_VALUE" != "minimal" ]; then
  ALLOW_DEBUG_RUNTIME_FALLBACK=1
fi

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
# A repo root candidate must have the canonical runtime manifest. Release
# builds intentionally do not depend on developer/training corpora.
is_repo_root() {
  c="$1"
  [ -f "$c/generated/agent_manifest/AgentBehaviorManifest.json" ]
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
IOS_MANIFEST_MIRROR="$REPO_ROOT/ios/Lumen/AgentBehaviorManifest.json"

log "Resolved REPO_ROOT: $REPO_ROOT"
APP_RESOURCES_DIR="$TARGET_BUILD_DIR_VALUE/$UNLOCALIZED_RESOURCES_FOLDER_PATH_VALUE"
DEST_DIR="$APP_RESOURCES_DIR/AgentGrounding"

prepare_empty_diagnostic_bundle() {
  rm -rf "$DEST_DIR"
  mkdir -p "$DEST_DIR/agent_manifest" "$DEST_DIR/cross_model_training"
}

resource_issue() {
  code="$1"
  detail="$2"
  if [ "$ALLOW_DEBUG_RUNTIME_FALLBACK" -eq 1 ]; then
    log "DIAGNOSTIC code=$code severity=warning action=runtime-fallback"
    log "DIAGNOSTIC detail=$detail"
    prepare_empty_diagnostic_bundle
    exit 0
  fi
  fail "[$code] $detail"
}

require_runtime_file() {
  path="$1"
  label="$2"
  if [ ! -f "$path" ]; then
    resource_issue "missing_required_resource" "Missing required runtime grounding file: $label ($path)"
  fi
}

validate_manifest_report() {
  report_path="$1"

  # Prefer Python's JSON decoder when available. Some Xcode build-service
  # environments return a non-canonical representation for `plutil` boolean
  # extraction even though the same report validates outside the build phase.
  # Keep plutil as the dependency-free fallback for hosts without Python.
  if ! command -v python3 >/dev/null 2>&1 && command -v plutil >/dev/null 2>&1; then
    if ! passed_value="$(plutil -extract passed raw -expect bool -o - "$report_path" 2>&1)"; then
      resource_issue "manifest_validation_report_invalid" "manifest_validation_report.json must be valid JSON with a boolean passed field: $passed_value"
    fi
    if [ "$passed_value" != "true" ]; then
      resource_issue "manifest_validation_report_not_passed" "manifest_validation_report.json must declare passed=true"
    fi

    if ! failure_count="$(plutil -extract failures raw -expect array -o - "$report_path" 2>&1)"; then
      resource_issue "manifest_validation_report_invalid" "manifest_validation_report.json must contain a failures array: $failure_count"
    fi
    if [ "$failure_count" != "0" ]; then
      resource_issue "manifest_validation_report_failures" "manifest_validation_report.json contains $failure_count failure(s)"
    fi

    if ! warning_count="$(plutil -extract warnings raw -expect array -o - "$report_path" 2>&1)"; then
      resource_issue "manifest_validation_report_invalid" "manifest_validation_report.json must contain a warnings array: $warning_count"
    fi
    if [ "$warning_count" != "0" ]; then
      resource_issue "manifest_validation_report_warnings" "manifest_validation_report.json contains $warning_count warning(s)"
    fi
    return 0
  fi

  if command -v python3 >/dev/null 2>&1; then
    if validation_error="$(python3 - "$report_path" 2>&1 <<'PY'
import json
import sys
from pathlib import Path

try:
    report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
except (OSError, UnicodeError, json.JSONDecodeError) as error:
    raise SystemExit(f"invalid JSON: {error}")

if not isinstance(report, dict):
    raise SystemExit("invalid: top-level value must be an object")
if report.get("passed") is not True:
    raise SystemExit("not_passed: passed must be the boolean true")
for key in ("failures", "warnings"):
    value = report.get(key)
    if not isinstance(value, list):
        raise SystemExit(f"invalid: {key} must be an array")
    if value:
        raise SystemExit(f"{key}: contains {len(value)} item(s)")
PY
    )"; then
      return 0
    fi
    case "$validation_error" in
      not_passed:*)
        resource_issue "manifest_validation_report_not_passed" "manifest_validation_report.json must declare passed=true: $validation_error"
        ;;
      failures:*)
        resource_issue "manifest_validation_report_failures" "manifest_validation_report.json has failures: $validation_error"
        ;;
      warnings:*)
        resource_issue "manifest_validation_report_warnings" "manifest_validation_report.json has warnings: $validation_error"
        ;;
      *)
        resource_issue "manifest_validation_report_invalid" "manifest_validation_report.json failed validation: $validation_error"
        ;;
    esac
  fi

  resource_issue "manifest_validation_tool_unavailable" "Neither plutil nor python3 is available to validate manifest_validation_report.json"
}

compute_sha256() {
  input_path="$1"
  if command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$input_path" | awk '{print $1}'
  elif command -v openssl >/dev/null 2>&1; then
    openssl dgst -sha256 "$input_path" | awk '{print $NF}'
  elif command -v python3 >/dev/null 2>&1; then
    python3 - "$input_path" <<'PY'
import hashlib
import sys
from pathlib import Path

print(hashlib.sha256(Path(sys.argv[1]).read_bytes()).hexdigest())
PY
  else
    return 1
  fi
}

verify_manifest_hash() {
  manifest_path="$1"
  hash_path="$2"
  context="$3"

  expected_hash="$(tr -d '\r\n' < "$hash_path")"
  if [ "${#expected_hash}" -ne 64 ]; then
    resource_issue "manifest_hash_sidecar_invalid" "$context AgentBehaviorManifest.sha256 must contain exactly one 64-character lowercase SHA-256 digest"
  fi
  case "$expected_hash" in
    *[!0-9a-f]*)
      resource_issue "manifest_hash_sidecar_invalid" "$context AgentBehaviorManifest.sha256 contains non-hexadecimal or uppercase characters"
      ;;
  esac

  if ! actual_hash="$(compute_sha256 "$manifest_path")"; then
    resource_issue "manifest_hash_tool_unavailable" "No SHA-256 utility is available to verify $context AgentBehaviorManifest.json"
  fi

  if [ "$actual_hash" != "$expected_hash" ]; then
    resource_issue "manifest_hash_mismatch" "$context AgentBehaviorManifest.json does not match AgentBehaviorManifest.sha256 (expected $expected_hash, actual $actual_hash)"
  fi
}

verify_source_integrity_entry() {
  relative_path="$1"
  declared_hash="$2"
  entry_label="$3"

  case "$relative_path" in
    ""|/*|*//*|*\\*)
      resource_issue "manifest_source_integrity_path_traversal" "$entry_label has an unsafe repository-relative path: $relative_path"
      ;;
  esac
  case "/$relative_path/" in
    */../*|*/./*)
      resource_issue "manifest_source_integrity_path_traversal" "$entry_label escapes or ambiguously traverses REPO_ROOT: $relative_path"
      ;;
  esac

  if [ "${#declared_hash}" -ne 64 ]; then
    resource_issue "manifest_source_integrity_invalid" "$entry_label sha256 must contain exactly 64 lowercase hexadecimal characters"
  fi
  case "$declared_hash" in
    *[!0-9a-f]*)
      resource_issue "manifest_source_integrity_invalid" "$entry_label sha256 contains non-hexadecimal or uppercase characters"
      ;;
  esac

  source_path="$REPO_ROOT/$relative_path"
  if [ -L "$source_path" ]; then
    resource_issue "manifest_source_integrity_path_traversal" "$entry_label resolves through a source-file symlink, which is not allowed: $relative_path"
  fi
  if [ ! -f "$source_path" ]; then
    resource_issue "manifest_source_integrity_missing" "$entry_label source file is missing or not a regular file: $relative_path"
  fi

  source_parent="$(dirname "$source_path")"
  if ! resolved_parent="$(cd "$source_parent" 2>/dev/null && pwd -P)"; then
    resource_issue "manifest_source_integrity_missing" "$entry_label source parent cannot be resolved: $relative_path"
  fi
  resolved_repo_root="$(cd "$REPO_ROOT" && pwd -P)"
  case "$resolved_parent" in
    "$resolved_repo_root"|"$resolved_repo_root"/*)
      ;;
    *)
      resource_issue "manifest_source_integrity_path_traversal" "$entry_label resolves outside REPO_ROOT: $relative_path"
      ;;
  esac

  if ! source_hash="$(compute_sha256 "$source_path")"; then
    resource_issue "manifest_hash_tool_unavailable" "No SHA-256 utility is available to verify $entry_label"
  fi
  if [ "$source_hash" != "$declared_hash" ]; then
    resource_issue "manifest_source_integrity_mismatch" "$entry_label hash mismatch for $relative_path (expected $declared_hash, actual $source_hash)"
  fi
}

verify_manifest_source_integrity() {
  manifest_path="$1"

  if command -v python3 >/dev/null 2>&1; then
    if source_integrity_error="$(python3 - "$manifest_path" "$REPO_ROOT" 2>&1 <<'PY'
import hashlib
import json
import re
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
repo_root = Path(sys.argv[2]).resolve(strict=True)

try:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
except (OSError, UnicodeError, json.JSONDecodeError) as error:
    raise SystemExit(f"invalid: manifest JSON cannot be decoded: {error}")

source_integrity = manifest.get("sourceIntegrity") if isinstance(manifest, dict) else None
files = source_integrity.get("files") if isinstance(source_integrity, dict) else None
if not isinstance(files, list) or not files:
    raise SystemExit("invalid: sourceIntegrity.files must be a non-empty array")

issues: list[tuple[str, str]] = []
seen_paths: set[str] = set()
for index, entry in enumerate(files):
    label = f"sourceIntegrity.files[{index}]"
    if not isinstance(entry, dict):
        issues.append(("invalid", f"{label} must be an object"))
        continue

    relative_path = entry.get("path")
    declared_hash = entry.get("sha256")
    if not isinstance(relative_path, str) or not relative_path:
        issues.append(("invalid", f"{label}.path must be a non-empty string"))
        continue
    if relative_path in seen_paths:
        issues.append(("invalid", f"{label}.path is duplicated: {relative_path!r}"))
        continue
    seen_paths.add(relative_path)

    components = relative_path.split("/")
    if (
        relative_path.startswith("/")
        or "\\" in relative_path
        or any(component in {"", ".", ".."} for component in components)
    ):
        issues.append(("traversal", f"{label}.path is unsafe: {relative_path!r}"))
        continue
    if not isinstance(declared_hash, str) or re.fullmatch(r"[0-9a-f]{64}", declared_hash) is None:
        issues.append(("invalid", f"{label}.sha256 is not a lowercase SHA-256 digest"))
        continue

    source_path = repo_root.joinpath(*components)
    try:
        resolved_source = source_path.resolve(strict=True)
    except FileNotFoundError:
        issues.append(("missing", f"{label} source file is missing: {relative_path}"))
        continue
    except (OSError, RuntimeError) as error:
        issues.append(("traversal", f"{label} cannot be resolved safely: {relative_path!r}: {error}"))
        continue
    if resolved_source != repo_root and repo_root not in resolved_source.parents:
        issues.append(("traversal", f"{label} resolves outside REPO_ROOT: {relative_path!r}"))
        continue
    if not resolved_source.is_file():
        issues.append(("missing", f"{label} is not a regular file: {relative_path}"))
        continue

    actual_hash = hashlib.sha256(resolved_source.read_bytes()).hexdigest()
    if actual_hash != declared_hash:
        issues.append(
            (
                "mismatch",
                f"{label} hash mismatch for {relative_path} "
                f"(expected {declared_hash}, actual {actual_hash})",
            )
        )

if issues:
    priority = ("traversal", "invalid", "missing", "mismatch")
    primary_kind = next(kind for kind in priority if any(issue[0] == kind for issue in issues))
    detail = "; ".join(message for _, message in issues)
    raise SystemExit(f"{primary_kind}: {len(issues)} source-integrity issue(s): {detail}")
PY
    )"; then
      return 0
    fi
    case "$source_integrity_error" in
      traversal:*)
        resource_issue "manifest_source_integrity_path_traversal" "$source_integrity_error"
        ;;
      missing:*)
        resource_issue "manifest_source_integrity_missing" "$source_integrity_error"
        ;;
      mismatch:*)
        resource_issue "manifest_source_integrity_mismatch" "$source_integrity_error"
        ;;
      *)
        resource_issue "manifest_source_integrity_invalid" "$source_integrity_error"
        ;;
    esac
  fi

  if command -v plutil >/dev/null 2>&1; then
    if ! source_file_count="$(plutil -extract sourceIntegrity.files raw -expect array -o - "$manifest_path" 2>&1)"; then
      resource_issue "manifest_source_integrity_invalid" "sourceIntegrity.files must be a non-empty array: $source_file_count"
    fi
    case "$source_file_count" in
      ""|*[!0-9]*)
        resource_issue "manifest_source_integrity_invalid" "sourceIntegrity.files count is invalid: $source_file_count"
        ;;
    esac
    if [ "$source_file_count" -eq 0 ]; then
      resource_issue "manifest_source_integrity_invalid" "sourceIntegrity.files must contain at least one source file"
    fi

    source_file_index=0
    while [ "$source_file_index" -lt "$source_file_count" ]; do
      source_entry="sourceIntegrity.files.$source_file_index"
      if ! relative_path="$(plutil -extract "$source_entry.path" raw -expect string -o - "$manifest_path" 2>&1)"; then
        resource_issue "manifest_source_integrity_invalid" "$source_entry.path must be a string: $relative_path"
      fi
      if ! declared_hash="$(plutil -extract "$source_entry.sha256" raw -expect string -o - "$manifest_path" 2>&1)"; then
        resource_issue "manifest_source_integrity_invalid" "$source_entry.sha256 must be a string: $declared_hash"
      fi
      verify_source_integrity_entry "$relative_path" "$declared_hash" "$source_entry"
      source_file_index=$((source_file_index + 1))
    done
    return 0
  fi

  resource_issue "manifest_source_integrity_tool_unavailable" "Neither python3 nor plutil is available to validate manifest source integrity"
}

verify_runtime_grounding_sources() {
  require_runtime_file "$AGENT_MANIFEST_DIR/AgentBehaviorManifest.json" "AgentBehaviorManifest.json"
  require_runtime_file "$AGENT_MANIFEST_DIR/AgentBehaviorManifest.sha256" "AgentBehaviorManifest.sha256"
  require_runtime_file "$AGENT_MANIFEST_DIR/AgentBehaviorManifest.md" "AgentBehaviorManifest.md"
  require_runtime_file "$AGENT_MANIFEST_DIR/fleet_system_prompts.json" "fleet_system_prompts.json"
  require_runtime_file "$AGENT_MANIFEST_DIR/manifest_validation_report.json" "manifest_validation_report.json"
  require_runtime_file "$AGENT_MANIFEST_DIR/runtime_grounding_bundle.json" "runtime_grounding_bundle.json"
  require_runtime_file "$AGENT_MANIFEST_DIR/runtime_grounding_prompt.md" "runtime_grounding_prompt.md"
  require_runtime_file "$IOS_MANIFEST_MIRROR" "iOS AgentBehaviorManifest.json mirror"

  if ! cmp -s "$AGENT_MANIFEST_DIR/AgentBehaviorManifest.json" "$IOS_MANIFEST_MIRROR"; then
    resource_issue "manifest_mirror_diverged" "Canonical generated AgentBehaviorManifest.json and ios/Lumen/AgentBehaviorManifest.json differ; run scripts/sync-agent-manifest-resource.sh before bundling"
  fi

  verify_manifest_hash \
    "$AGENT_MANIFEST_DIR/AgentBehaviorManifest.json" \
    "$AGENT_MANIFEST_DIR/AgentBehaviorManifest.sha256" \
    "Canonical"
  verify_manifest_source_integrity "$AGENT_MANIFEST_DIR/AgentBehaviorManifest.json"
  validate_manifest_report "$AGENT_MANIFEST_DIR/manifest_validation_report.json"
}

verify_installed_manifest_and_hash() {
  installed_manifest="$DEST_DIR/agent_manifest/AgentBehaviorManifest.json"
  installed_hash="$DEST_DIR/agent_manifest/AgentBehaviorManifest.sha256"
  require_runtime_file "$installed_manifest" "bundled AgentBehaviorManifest.json"
  require_runtime_file "$installed_hash" "bundled AgentBehaviorManifest.sha256"

  if ! cmp -s "$AGENT_MANIFEST_DIR/AgentBehaviorManifest.json" "$installed_manifest" \
     || ! cmp -s "$AGENT_MANIFEST_DIR/AgentBehaviorManifest.sha256" "$installed_hash"; then
    resource_issue "installed_manifest_copy_mismatch" "Bundled manifest or SHA sidecar differs from its validated canonical source"
  fi
  verify_manifest_hash "$installed_manifest" "$installed_hash" "Bundled"
}

copy_required_file() {
  source_path="$1"
  relative_path="$2"
  require_file "$source_path" "$relative_path"
  mkdir -p "$(dirname "$DEST_DIR/$relative_path")"
  cp "$source_path" "$DEST_DIR/$relative_path"
}

verify_cross_model_mirrors() {
  if [ ! -d "$LEGACY_CROSS_MODEL_DIR" ] \
     || [ ! -d "$NESTED_CROSS_MODEL_DIR" ]; then
    return 0
  fi

  for filename in \
    cross_model_training.jsonl \
    cross_model_training_index.csv \
    dpo_train_cross.jsonl \
    dpo_val_cross.jsonl \
    orchestration_evals.jsonl \
    train_sft_cross.jsonl \
    val_sft_cross.jsonl
  do
    require_file "$LEGACY_CROSS_MODEL_DIR/$filename" "primary cross-model mirror/$filename"
    require_file "$NESTED_CROSS_MODEL_DIR/$filename" "nested cross-model mirror/$filename"
    if ! cmp -s \
      "$LEGACY_CROSS_MODEL_DIR/$filename" \
      "$NESTED_CROSS_MODEL_DIR/$filename"; then
      fail "Cross-model resource mirrors diverged for $filename; regenerate both copies before bundling."
    fi
  done
}

if [ -d "$LEGACY_CROSS_MODEL_DIR" ]; then
  CROSS_MODEL_DIR="$LEGACY_CROSS_MODEL_DIR"
elif [ -d "$NESTED_CROSS_MODEL_DIR" ]; then
  CROSS_MODEL_DIR="$NESTED_CROSS_MODEL_DIR"
else
  CROSS_MODEL_DIR=""
fi

if [ "$AGENT_GROUNDING_RESOURCE_MODE_VALUE" = "skip" ]; then
  resource_issue "resource_mode_skip" "AGENT_GROUNDING_RESOURCE_MODE=skip is restricted to explicit Debug diagnostics"
fi

if [ ! -d "$AGENT_MANIFEST_DIR" ]; then
  resource_issue "missing_required_directory" "Generated agent manifest directory is missing: $AGENT_MANIFEST_DIR"
fi
if [ "$AGENT_GROUNDING_RESOURCE_MODE_VALUE" = "full" ] && [ -z "$CROSS_MODEL_DIR" ]; then
  resource_issue "missing_required_directory" "Full grounding mode requires generated cross-model training resources"
fi

verify_runtime_grounding_sources

if [ "$AGENT_GROUNDING_RESOURCE_MODE_VALUE" = "minimal" ]; then
  log "AGENT_GROUNDING_RESOURCE_MODE=minimal; copying only runtime-required grounding resources."
  rm -rf "$DEST_DIR"
  mkdir -p "$DEST_DIR/agent_manifest" "$DEST_DIR/cross_model_training"

  copy_required_file "$AGENT_MANIFEST_DIR/AgentBehaviorManifest.json" "agent_manifest/AgentBehaviorManifest.json"
  copy_required_file "$AGENT_MANIFEST_DIR/AgentBehaviorManifest.sha256" "agent_manifest/AgentBehaviorManifest.sha256"
  copy_required_file "$AGENT_MANIFEST_DIR/AgentBehaviorManifest.md" "agent_manifest/AgentBehaviorManifest.md"
  copy_required_file "$AGENT_MANIFEST_DIR/fleet_system_prompts.json" "agent_manifest/fleet_system_prompts.json"
  copy_required_file "$AGENT_MANIFEST_DIR/manifest_validation_report.json" "agent_manifest/manifest_validation_report.json"
  copy_required_file "$AGENT_MANIFEST_DIR/runtime_grounding_bundle.json" "agent_manifest/runtime_grounding_bundle.json"
  copy_required_file "$AGENT_MANIFEST_DIR/runtime_grounding_prompt.md" "agent_manifest/runtime_grounding_prompt.md"

  verify_installed_manifest_and_hash
  log "Installed minimal resources at: $DEST_DIR"
  exit 0
fi

verify_cross_model_mirrors

require_file "$AGENT_MANIFEST_DIR/AgentBehaviorManifest.json" 'AgentBehaviorManifest.json'
require_file "$AGENT_MANIFEST_DIR/AgentBehaviorManifest.md" 'AgentBehaviorManifest.md'
require_file "$AGENT_MANIFEST_DIR/fleet_system_prompts.json" 'fleet_system_prompts.json'
require_file "$AGENT_MANIFEST_DIR/manifest_validation_report.json" 'manifest_validation_report.json'
require_file "$AGENT_MANIFEST_DIR/AgentBehaviorManifest.sha256" 'AgentBehaviorManifest.sha256'
require_file "$AGENT_MANIFEST_DIR/AgentBehaviorManifest.incremental.sha256" 'AgentBehaviorManifest.incremental.sha256'
require_file "$AGENT_MANIFEST_DIR/runtime_grounding_bundle.json" 'runtime_grounding_bundle.json'
require_file "$AGENT_MANIFEST_DIR/runtime_grounding_prompt.md" 'runtime_grounding_prompt.md'
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
# runtime_audit_repairs.jsonl is optional when the generator emits no repair records.
require_file "$AGENT_MANIFEST_DIR/dataset/codebase_home_corpus.jsonl" 'dataset/codebase_home_corpus.jsonl'
require_file "$AGENT_MANIFEST_DIR/dataset/codebase_home_sft.jsonl" 'dataset/codebase_home_sft.jsonl'

require_file "$CROSS_MODEL_DIR/cross_model_training.jsonl" 'cross_model_training.jsonl'
require_file "$CROSS_MODEL_DIR/train_sft_cross.jsonl" 'train_sft_cross.jsonl'
require_file "$CROSS_MODEL_DIR/val_sft_cross.jsonl" 'val_sft_cross.jsonl'
require_file "$CROSS_MODEL_DIR/dpo_train_cross.jsonl" 'dpo_train_cross.jsonl'
require_file "$CROSS_MODEL_DIR/dpo_val_cross.jsonl" 'dpo_val_cross.jsonl'
require_file "$CROSS_MODEL_DIR/cross_model_training_index.csv" 'cross_model_training_index.csv'

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

verify_installed_manifest_and_hash
log "Installed resources at: $DEST_DIR"
log "Manifest: $DEST_DIR/agent_manifest/AgentBehaviorManifest.json"
log "Prompts:  $DEST_DIR/agent_manifest/fleet_system_prompts.json"
