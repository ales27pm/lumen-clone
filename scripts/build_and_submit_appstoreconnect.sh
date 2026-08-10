#!/usr/bin/env bash
set -euo pipefail

FIND_BIN="find"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG_DIR="$REPO_ROOT/.local"
CONFIG_FILE="$CONFIG_DIR/appstoreconnect-upload.env"
STABLE_ARCHIVE_SCRIPT="$REPO_ROOT/scripts/archive_lumen_stable.sh"
PROJECT_FILE="$REPO_ROOT/ios/Lumen.xcodeproj/project.pbxproj"
RELEASE_SCOPE_PATHS=(ios scripts generated/agent_manifest)
LOCAL_CONFIG_ENV_KEYS=(
  ASC_PROJECT_PATH
  ASC_SCHEME
  ASC_CONFIGURATION
  ASC_TEAM_ID
  ASC_EXPORT_METHOD
  ASC_AUTH_MODE
  ASC_API_KEY
  ASC_API_ISSUER
  ASC_API_KEY_DIR
  ASC_APPLE_ID
  ASC_PROVIDER
  ASC_UPLOAD_AFTER_BUILD
  LUMEN_NO_UPLOAD
)
NO_UPLOAD_REQUESTED=0
RELEASE_GIT_COMMIT=""
RELEASE_GIT_SHA=""
RELEASE_BUILD_SCHEME=""
DIAGNOSTIC_RELEASE=0

bold() { printf "\033[1m%s\033[0m\n" "$1"; }
info() { printf "\n➡️  %s\n" "$1"; }
warn() { printf "\n⚠️  %s\n" "$1"; }
fail() { printf "\n❌ %s\n" "$1"; exit 1; }

confirm() {
  local prompt="$1"
  local response
  read -r -p "$prompt [y/N]: " response
  [[ "$response" =~ ^[Yy]$ ]]
}

read_required() {
  local prompt="$1"
  local var=""
  while [[ -z "${var// }" ]]; do
    read -r -p "$prompt" var
  done
  printf "%s" "$var"
}

read_secret_required() {
  local prompt="$1"
  local var=""
  while [[ -z "${var// }" ]]; do
    read -r -s -p "$prompt" var
    printf "\n"
  done
  printf "%s" "$var"
}

read_with_default() {
  local prompt="$1"
  local default_value="$2"
  local value=""
  if [[ -n "$default_value" ]]; then
    read -r -p "$prompt [$default_value]: " value
    printf "%s" "${value:-$default_value}"
  else
    read_required "$prompt: "
  fi
}

read_yes_no_with_default() {
  local prompt="$1"
  local default_value="$2"
  local response=""
  local label="y/N"
  [[ "$default_value" =~ ^[Yy]$ ]] && label="Y/n"
  read -r -p "$prompt [$label]: " response
  response="${response:-$default_value}"
  [[ "$response" =~ ^[Yy]$ ]]
}

load_local_config() {
  local override_names=()
  local override_values=()
  local name
  local index

  # Local config supplies defaults only. Explicit process-environment values,
  # including an explicitly empty value, always win over the sourced file.
  for name in "${LOCAL_CONFIG_ENV_KEYS[@]}"; do
    if printenv "$name" >/dev/null 2>&1; then
      override_names+=("$name")
      override_values+=("${!name}")
    fi
  done

  if [[ "${LUMEN_RESET_ASC_CONFIG:-0}" == "1" ]]; then
    rm -f "$CONFIG_FILE"
    warn "Local App Store Connect config reset: $CONFIG_FILE"
    return 0
  fi

  if [[ -f "$CONFIG_FILE" ]]; then
    # shellcheck disable=SC1090
    source "$CONFIG_FILE"
  fi

  for ((index = 0; index < ${#override_names[@]}; index += 1)); do
    name="${override_names[$index]}"
    printf -v "$name" '%s' "${override_values[$index]}"
    export "$name"
  done
}

configure_no_upload_mode() {
  case "${LUMEN_NO_UPLOAD:-0}" in
    0|"") NO_UPLOAD_REQUESTED=0 ;;
    1) NO_UPLOAD_REQUESTED=1 ;;
    *) fail "LUMEN_NO_UPLOAD must be 0 or 1; received ${LUMEN_NO_UPLOAD}." ;;
  esac
}

is_release_configuration() {
  case "$1" in
    Release|AppStore|App\ Store) return 0 ;;
    *) return 1 ;;
  esac
}

release_scope_changes() {
  git -C "$REPO_ROOT" status --porcelain=v1 --untracked-files=all -- "${RELEASE_SCOPE_PATHS[@]}"
}

prepare_release_provenance() {
  local configuration="$1"
  local allow_dirty_diagnostic="${LUMEN_ALLOW_DIRTY_RELEASE_DIAGNOSTIC:-0}"
  local requested_git_sha="${LUMEN_GIT_SHA:-}"
  local changes=""

  case "$allow_dirty_diagnostic" in
    0|1) ;;
    *) fail "LUMEN_ALLOW_DIRTY_RELEASE_DIAGNOSTIC must be 0 or 1; received $allow_dirty_diagnostic." ;;
  esac

  RELEASE_GIT_COMMIT="$(git -C "$REPO_ROOT" rev-parse --verify HEAD^{commit} 2>/dev/null)" \
    || fail "Cannot resolve a Git commit for release provenance."
  [[ "$RELEASE_GIT_COMMIT" =~ ^[0-9a-f]{40}([0-9a-f]{24})?$ ]] \
    || fail "Release provenance requires a full Git object ID; resolved $RELEASE_GIT_COMMIT."
  if [[ -n "$requested_git_sha" && "$requested_git_sha" != "$RELEASE_GIT_COMMIT" ]]; then
    fail "LUMEN_GIT_SHA=$requested_git_sha does not match the full release commit $RELEASE_GIT_COMMIT."
  fi

  RELEASE_GIT_SHA="$RELEASE_GIT_COMMIT"
  RELEASE_BUILD_SCHEME="$SCHEME"
  DIAGNOSTIC_RELEASE=0

  if is_release_configuration "$configuration"; then
    changes="$(release_scope_changes)" \
      || fail "Could not inspect release-scope Git state."
    if [[ -n "$changes" ]]; then
      if [[ "$allow_dirty_diagnostic" != "1" ]]; then
        printf '%s\n' "$changes" >&2
        fail "Release archive refused because tracked or untracked release inputs are dirty. Commit/stash them, or use LUMEN_ALLOW_DIRTY_RELEASE_DIAGNOSTIC=1 only for non-distributable diagnostics."
      fi
      RELEASE_BUILD_SCHEME="${SCHEME}-DIAGNOSTIC-DIRTY-NOT-FOR-DISTRIBUTION"
      DIAGNOSTIC_RELEASE=1
      NO_UPLOAD_REQUESTED=1
      warn "DIRTY DIAGNOSTIC ONLY: forcing no-upload and stamping the artifact not-for-distribution."
    fi
  fi

  info "Release source commit: $RELEASE_GIT_COMMIT"
}

assert_release_provenance_still_valid() {
  local current_commit
  local changes=""

  current_commit="$(git -C "$REPO_ROOT" rev-parse --verify HEAD^{commit} 2>/dev/null)" \
    || fail "Could not re-resolve Git HEAD after export."
  [[ "$current_commit" == "$RELEASE_GIT_COMMIT" ]] \
    || fail "Git HEAD changed during build/export: expected $RELEASE_GIT_COMMIT, found $current_commit."
  if is_release_configuration "$CONFIGURATION" && [[ "$DIAGNOSTIC_RELEASE" != "1" ]]; then
    changes="$(release_scope_changes)" \
      || fail "Could not re-check release-scope Git state after export."
    if [[ -n "$changes" ]]; then
      printf '%s\n' "$changes" >&2
      fail "Release-scope inputs changed during build/export; the IPA is not valid production evidence."
    fi
  fi
}

save_local_config() {
  mkdir -p "$CONFIG_DIR"
  local tmp_file="$CONFIG_FILE.tmp"
  umask 077
  {
    printf '# Local Lumen App Store Connect upload config. Generated by scripts/build_and_submit_appstoreconnect.sh.\n'
    printf '# This file is intentionally ignored by Git. Do not commit API keys or private credentials.\n'
    printf 'ASC_PROJECT_PATH=%q\n' "$PROJECT_PATH"
    printf 'ASC_SCHEME=%q\n' "$SCHEME"
    printf 'ASC_CONFIGURATION=%q\n' "$CONFIGURATION"
    printf 'ASC_TEAM_ID=%q\n' "$TEAM_ID"
    printf 'ASC_EXPORT_METHOD=%q\n' "$EXPORT_METHOD"
    printf 'ASC_AUTH_MODE=%q\n' "$AUTH_MODE"
    printf 'ASC_API_KEY=%q\n' "$API_KEY"
    printf 'ASC_API_ISSUER=%q\n' "$API_ISSUER"
    printf 'ASC_API_KEY_DIR=%q\n' "$API_KEY_DIR"
    printf 'ASC_APPLE_ID=%q\n' "$APPLE_ID"
    printf 'ASC_PROVIDER=%q\n' "$ASC_PROVIDER"
    printf 'ASC_UPLOAD_AFTER_BUILD=%q\n' "${UPLOAD_AFTER_BUILD:-}"
  } > "$tmp_file"
  mv "$tmp_file" "$CONFIG_FILE"
  chmod 600 "$CONFIG_FILE"
}

install_xcode_cli_tools() {
  if xcode-select -p >/dev/null 2>&1; then
    return 0
  fi

  info "Xcode Command Line Tools are not installed."
  if ! confirm "Install Xcode Command Line Tools now via xcode-select --install?"; then
    fail "Xcode Command Line Tools are required."
  fi

  xcode-select --install || true
  warn "Complete the GUI installer, then re-run this script."
  exit 1
}

ensure_find() {
  if command -v find >/dev/null 2>&1; then
    FIND_BIN="$(command -v find)"
    export FIND_BIN
    return 0
  fi

  warn "find command is missing. Attempting to install GNU findutils via Homebrew."
  if ! command -v brew >/dev/null 2>&1; then
    fail "Homebrew is not installed; install Homebrew or restore /usr/bin/find."
  fi

  brew install findutils
  if command -v find >/dev/null 2>&1; then
    FIND_BIN="$(command -v find)"
    export FIND_BIN
    return 0
  fi
  if [[ -x /opt/homebrew/opt/findutils/libexec/gnubin/find ]]; then
    FIND_BIN="/opt/homebrew/opt/findutils/libexec/gnubin/find"
  elif [[ -x /usr/local/opt/findutils/libexec/gnubin/find ]]; then
    FIND_BIN="/usr/local/opt/findutils/libexec/gnubin/find"
  else
    fail "find installation did not expose expected binary under /opt/homebrew or /usr/local gnubin paths."
  fi
  export FIND_BIN
}

ensure_xcodebuild_and_xcrun() {
  install_xcode_cli_tools

  if ! command -v xcodebuild >/dev/null 2>&1 || ! command -v xcrun >/dev/null 2>&1; then
    fail "xcodebuild/xcrun unavailable. Install Xcode and select it with: sudo xcode-select --switch /Applications/Xcode.app/Contents/Developer"
  fi

  xcodebuild -version >/dev/null 2>&1 || fail "xcodebuild is installed but not usable. Open Xcode once and accept licenses."
}

ensure_upload_tool() {
  if ! xcrun altool --help >/dev/null 2>&1; then
    fail "xcrun altool is unavailable. Install/upgrade Xcode and select it with: sudo xcode-select --switch /Applications/Xcode.app/Contents/Developer"
  fi
}

print_xcode_log_diagnostics() {
  local log_path="$1"
  [[ -f "$log_path" ]] || return 0

  warn "xcodebuild failed. Full log: $log_path"

  info "Most relevant diagnostics from xcodebuild log"
  grep -nEi \
    '(^|[^A-Za-z])(error:|fatal error:|warning:|failed|failure|Command SwiftCompile failed|Command CompileSwift failed|Command CodeSign failed|no such module|cannot find|cannot convert|ambiguous|missing|undefined|duplicate|provisioning|codesign|SwiftDriver|CompileSwift)' \
    "$log_path" | tail -n 120 || true

  info "Last 220 lines of xcodebuild log"
  tail -n 220 "$log_path" || true
}

run_logged() {
  local log_path="$1"
  shift
  mkdir -p "$(dirname "$log_path")"

  set +e
  "$@" 2>&1 | tee "$log_path"
  local status=${PIPESTATUS[0]}
  set -e

  if [[ $status -ne 0 ]]; then
    print_xcode_log_diagnostics "$log_path"
    return "$status"
  fi

  return 0
}

normalize_export_method() {
  case "$1" in
    app-store|app-store-connect) printf "app-store-connect" ;;
    ad-hoc|release-testing) printf "release-testing" ;;
    development|debugging) printf "debugging" ;;
    enterprise) printf "enterprise" ;;
    validation) printf "validation" ;;
    *) return 1 ;;
  esac
}

is_distribution_export() {
  case "$1" in
    app-store-connect|release-testing|enterprise|validation) return 0 ;;
    *) return 1 ;;
  esac
}

next_local_release_build_number() {
  python3 - "$PROJECT_FILE" <<'PY'
from __future__ import annotations

import datetime
import re
import sys
from pathlib import Path

project_file = Path(sys.argv[1])
text = project_file.read_text(encoding="utf-8")
versions = [int(value) for value in re.findall(r"\bCURRENT_PROJECT_VERSION = ([0-9]+);", text)]
if not versions:
    raise SystemExit(f"error: no numeric CURRENT_PROJECT_VERSION found in {project_file}")
timestamp = int(datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d%H%M%S"))
print(max(timestamp, max(versions) + 1))
PY
}

validate_release_build_number() {
  local candidate="$1"
  python3 - "$PROJECT_FILE" "$candidate" <<'PY'
from __future__ import annotations

import re
import sys
from pathlib import Path

project_file = Path(sys.argv[1])
candidate = sys.argv[2]
if not candidate.isdigit():
    raise SystemExit("error: release build number must contain decimal digits only")
text = project_file.read_text(encoding="utf-8")
versions = [int(value) for value in re.findall(r"\bCURRENT_PROJECT_VERSION = ([0-9]+);", text)]
if not versions:
    raise SystemExit(f"error: no numeric CURRENT_PROJECT_VERSION found in {project_file}")
current = max(versions)
if int(candidate) <= current:
    raise SystemExit(
        f"error: release build number {candidate} must be greater than the checked-in build number {current}"
    )
PY
}

write_export_options_plist() {
  local plist_path="$1"
  local export_method="$2"
  local team_id="$3"
  local signing_certificate="$4"

  cat > "$plist_path" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>method</key>
  <string>${export_method}</string>
  <key>signingStyle</key>
  <string>automatic</string>
  <key>signingCertificate</key>
  <string>${signing_certificate}</string>
PLIST

  if [[ -n "$team_id" ]]; then
    cat >> "$plist_path" <<PLIST
  <key>teamID</key>
  <string>${team_id}</string>
PLIST
  fi

  cat >> "$plist_path" <<PLIST
  <key>stripSwiftSymbols</key>
  <true/>
  <key>uploadSymbols</key>
  <true/>
</dict>
</plist>
PLIST
}

UPLOAD_SUCCESS_PATTERN='^[[:space:]]*(UPLOAD SUCCEEDED with no errors[.]?|No errors uploading archive at .+[.])[[:space:]]*$'
UPLOAD_ERROR_PATTERN='(^|[^[:alnum:]_])(error|failed to upload|upload failed|unable to upload|entity_error|state_error|validation_error|validation errors?|asset validation failed|must be higher than|already been used|duplicate build)([^[:alnum:]_]|$)'
DELIVERY_UUID_PATTERN='^[[:space:]]*Delivery UUID:[[:space:]]*[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}[[:space:]]*$'

upload_log_has_success_marker() {
  local log_path="$1"
  LC_ALL=C grep -Eq "$UPLOAD_SUCCESS_PATTERN" "$log_path"
}

upload_log_has_error_pattern() {
  local log_path="$1"
  LC_ALL=C grep -Eiq "$UPLOAD_ERROR_PATTERN" "$log_path"
}

delivery_uuid_from_upload_log() {
  local log_path="$1"
  local delivery_lines
  local delivery_line_count

  delivery_lines="$(LC_ALL=C grep -Ei "$DELIVERY_UUID_PATTERN" "$log_path" || true)"
  delivery_line_count="$(printf '%s\n' "$delivery_lines" | awk 'NF { count += 1 } END { print count + 0 }')"
  [[ "$delivery_line_count" -eq 1 ]] || return 1
  printf '%s\n' "$delivery_lines" | awk 'NF { print $NF; exit }'
}

validate_upload_result() {
  local upload_status="$1"
  local log_path="$2"
  local delivery_uuid

  if [[ ! "$upload_status" =~ ^[0-9]+$ ]] || [[ "$upload_status" -ne 0 ]]; then
    printf 'Upload result rejected: uploader exit status was %s.\n' "$upload_status" >&2
    return 1
  fi
  if [[ ! -f "$log_path" ]]; then
    printf 'Upload result rejected: upload log is missing: %s\n' "$log_path" >&2
    return 1
  fi
  if upload_log_has_error_pattern "$log_path"; then
    printf 'Upload result rejected: upload log contains an error pattern.\n' >&2
    return 1
  fi
  if ! upload_log_has_success_marker "$log_path"; then
    printf 'Upload result rejected: unambiguous success marker is missing.\n' >&2
    return 1
  fi
  if ! delivery_uuid="$(delivery_uuid_from_upload_log "$log_path")"; then
    printf 'Upload result rejected: expected exactly one Delivery UUID.\n' >&2
    return 1
  fi

  printf '%s\n' "$delivery_uuid"
}

verify_artifact_provenance_metadata() {
  local artifact_path="$1"
  local expected_git_sha="$2"
  local expected_build_scheme="$3"

  python3 - "$artifact_path" "$expected_git_sha" "$expected_build_scheme" <<'PY'
from __future__ import annotations

import plistlib
import sys
import zipfile
from pathlib import Path

artifact = Path(sys.argv[1])
expected_git_sha = sys.argv[2]
expected_build_scheme = sys.argv[3]
if artifact.suffix == ".ipa":
    with zipfile.ZipFile(artifact) as ipa:
        members = sorted(
            name
            for name in ipa.namelist()
            if name.startswith("Payload/") and name.endswith(".app/Info.plist")
        )
        if len(members) != 1:
            raise SystemExit(
                f"error: expected exactly one app Info.plist in {artifact}; found {len(members)}"
            )
        info = plistlib.loads(ipa.read(members[0]))
else:
    plists = sorted((artifact / "Products" / "Applications").glob("*.app/Info.plist"))
    if len(plists) != 1:
        raise SystemExit(
            f"error: expected exactly one archived app Info.plist; found {len(plists)}"
        )
    with plists[0].open("rb") as handle:
        info = plistlib.load(handle)

actual_git_sha = info.get("LumenGitSHA")
actual_build_scheme = info.get("LumenBuildScheme")
if actual_git_sha != expected_git_sha:
    raise SystemExit(
        f"error: artifact LumenGitSHA={actual_git_sha!r}; expected {expected_git_sha!r}"
    )
if actual_build_scheme != expected_build_scheme:
    raise SystemExit(
        f"error: artifact LumenBuildScheme={actual_build_scheme!r}; "
        f"expected {expected_build_scheme!r}"
    )
print(f"Artifact provenance verified: {actual_git_sha} [{actual_build_scheme}]")
PY
}

write_upload_result_fixture() {
  local fixture_path="$1"
  local fixture_text="$2"
  printf '%s\n' "$fixture_text" > "$fixture_path"
}

assert_upload_result_accepted() {
  local fixture_name="$1"
  local upload_status="$2"
  local fixture_text="$3"
  local expected_uuid="$4"
  local fixture_path="$UPLOAD_RESULT_SELF_CHECK_DIR/$fixture_name.log"
  local actual_uuid

  write_upload_result_fixture "$fixture_path" "$fixture_text"
  if ! actual_uuid="$(validate_upload_result "$upload_status" "$fixture_path" 2>/dev/null)"; then
    printf 'Upload-result self-check failed: expected acceptance for %s.\n' "$fixture_name" >&2
    return 1
  fi
  if [[ "$actual_uuid" != "$expected_uuid" ]]; then
    printf 'Upload-result self-check failed: wrong Delivery UUID for %s.\n' "$fixture_name" >&2
    return 1
  fi
}

assert_upload_result_rejected() {
  local fixture_name="$1"
  local upload_status="$2"
  local fixture_text="$3"
  local fixture_path="$UPLOAD_RESULT_SELF_CHECK_DIR/$fixture_name.log"

  write_upload_result_fixture "$fixture_path" "$fixture_text"
  if validate_upload_result "$upload_status" "$fixture_path" >/dev/null 2>&1; then
    printf 'Upload-result self-check failed: expected rejection for %s.\n' "$fixture_name" >&2
    return 1
  fi
}

run_upload_result_parser_self_check() {
  local delivery_uuid='58259589-1b30-4ff4-b5e4-ec40cc8f1caf'
  local second_delivery_uuid='11111111-2222-4333-8444-555555555555'
  UPLOAD_RESULT_SELF_CHECK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/lumen-upload-result.XXXXXX")"
  trap 'rm -rf -- "$UPLOAD_RESULT_SELF_CHECK_DIR"' EXIT

  assert_upload_result_accepted \
    canonical-success \
    0 \
    $'UPLOAD SUCCEEDED with no errors\nDelivery UUID: '"$delivery_uuid"$'\nNo errors uploading archive at '\''Lumen.ipa'\''.' \
    "$delivery_uuid"
  assert_upload_result_accepted \
    alternate-success \
    0 \
    $'Delivery UUID: '"$delivery_uuid"$'\nNo errors uploading archive at '\''Lumen.ipa'\''.' \
    "$delivery_uuid"
  assert_upload_result_rejected nonzero-status 1 $'UPLOAD SUCCEEDED with no errors\nDelivery UUID: '"$delivery_uuid"
  assert_upload_result_rejected missing-marker 0 $'Delivery UUID: '"$delivery_uuid"
  assert_upload_result_rejected missing-delivery-uuid 0 'UPLOAD SUCCEEDED with no errors'
  assert_upload_result_rejected embedded-marker 0 $'status: UPLOAD SUCCEEDED with no errors (cached)\nDelivery UUID: '"$delivery_uuid"
  assert_upload_result_rejected conflicting-error 0 $'UPLOAD SUCCEEDED with no errors\nDelivery UUID: '"$delivery_uuid"$'\nERROR: Asset validation failed.'
  assert_upload_result_rejected multiple-delivery-uuids 0 $'UPLOAD SUCCEEDED with no errors\nDelivery UUID: '"$delivery_uuid"$'\nDelivery UUID: '"$second_delivery_uuid"

  printf 'Upload-result parser self-check passed.\n'
}

run_config_precedence_self_check() {
  local fixture_dir

  fixture_dir="$(mktemp -d "${TMPDIR:-/tmp}/lumen-asc-config.XXXXXX")"
  ASC_CONFIG_SELF_CHECK_DIR="$fixture_dir"
  trap 'rm -rf -- "$ASC_CONFIG_SELF_CHECK_DIR"' EXIT
  CONFIG_FILE="$fixture_dir/appstoreconnect-upload.env"
  {
    printf 'ASC_PROJECT_PATH=%q\n' 'local-project.xcodeproj'
    printf 'ASC_SCHEME=%q\n' 'LocalScheme'
    printf 'ASC_PROVIDER=%q\n' 'local-provider'
    printf 'ASC_UPLOAD_AFTER_BUILD=%q\n' 'Y'
    printf 'LUMEN_NO_UPLOAD=%q\n' '0'
  } > "$CONFIG_FILE"

  unset ASC_PROJECT_PATH LUMEN_RESET_ASC_CONFIG
  export ASC_SCHEME='ProcessScheme'
  export ASC_PROVIDER=''
  export ASC_UPLOAD_AFTER_BUILD='N'
  export LUMEN_NO_UPLOAD='1'
  load_local_config

  [[ "$ASC_PROJECT_PATH" == 'local-project.xcodeproj' ]] \
    || fail "Config-precedence self-check did not load an unset local default."
  [[ "$ASC_SCHEME" == 'ProcessScheme' ]] \
    || fail "Config-precedence self-check overwrote an explicit process value."
  [[ "$ASC_PROVIDER" == '' ]] \
    || fail "Config-precedence self-check overwrote an explicit empty process value."
  [[ "$ASC_UPLOAD_AFTER_BUILD" == 'N' ]] \
    || fail "Config-precedence self-check overwrote explicit no-upload with local config."
  [[ "$LUMEN_NO_UPLOAD" == '1' ]] \
    || fail "Config-precedence self-check allowed local config to override LUMEN_NO_UPLOAD=1."

  configure_no_upload_mode
  [[ "$NO_UPLOAD_REQUESTED" == '1' ]] \
    || fail "Config-precedence self-check did not activate the no-upload safety latch."
  printf 'Local-config precedence and no-upload self-check passed.\n'
}

if [[ "${1:-}" == "--self-check-upload-result-parser" ]]; then
  run_upload_result_parser_self_check
  exit 0
fi

if [[ "${1:-}" == "--self-check-config-precedence" ]]; then
  run_config_precedence_self_check
  exit 0
fi

if [[ "${1:-}" == "--print-next-build-number" ]]; then
  next_local_release_build_number
  exit 0
fi

if [[ "${1:-}" == "--validate-release-build-number" ]]; then
  [[ -n "${2:-}" ]] || fail "--validate-release-build-number requires a candidate value."
  validate_release_build_number "$2" || exit $?
  printf 'Release build number %s passes local monotonicity validation.\n' "$2"
  exit 0
fi

[[ "$(uname -s)" == "Darwin" ]] || fail "This script must run on macOS."
cd "$REPO_ROOT"
load_local_config
configure_no_upload_mode
ensure_xcodebuild_and_xcrun
ensure_find

bold "Lumen iOS build + App Store Connect upload"

DEFAULT_PROJECT_PATH="${ASC_PROJECT_PATH:-ios/Lumen.xcodeproj}"
DEFAULT_SCHEME="${ASC_SCHEME:-Lumen}"
DEFAULT_CONFIGURATION="${ASC_CONFIGURATION:-Release}"
DEFAULT_EXPORT_METHOD="${ASC_EXPORT_METHOD:-app-store-connect}"
DEFAULT_TEAM_ID="${ASC_TEAM_ID:-${DEVELOPMENT_TEAM:-52T7P32J34}}"
DEFAULT_AUTH_MODE="${ASC_AUTH_MODE:-1}"

PROJECT_PATH="$(read_with_default 'Project/workspace path' "$DEFAULT_PROJECT_PATH")"
[[ -e "$PROJECT_PATH" ]] || fail "Path not found: $PROJECT_PATH"
[[ "$PROJECT_PATH" == *.xcworkspace || "$PROJECT_PATH" == *.xcodeproj ]] || fail "Path must be .xcworkspace or .xcodeproj"
[[ -f "$STABLE_ARCHIVE_SCRIPT" ]] || fail "Stable archive script not found: $STABLE_ARCHIVE_SCRIPT"

SCHEME="$(read_with_default 'Scheme' "$DEFAULT_SCHEME")"
CONFIGURATION="$(read_with_default 'Configuration' "$DEFAULT_CONFIGURATION")"
DEFAULT_BUILD_NUMBER="${LUMEN_IOS_CURRENT_PROJECT_VERSION:-$(next_local_release_build_number)}"
BUILD_NUMBER="$(read_with_default 'Release build number (CFBundleVersion)' "$DEFAULT_BUILD_NUMBER")"
validate_release_build_number "$BUILD_NUMBER" || fail "Choose a new build number before archiving."
TEAM_ID="$(read_with_default 'Apple Developer Team ID' "$DEFAULT_TEAM_ID")"

EXPORT_METHOD_INPUT="$(read_with_default 'Export method' "$DEFAULT_EXPORT_METHOD")"
EXPORT_METHOD="$(normalize_export_method "$EXPORT_METHOD_INPUT")" || fail "Unsupported export method: $EXPORT_METHOD_INPUT"
if is_distribution_export "$EXPORT_METHOD" && [[ "$CONFIGURATION" != "Release" ]]; then
  fail "Distribution export method '$EXPORT_METHOD' requires the Release configuration; received '$CONFIGURATION'."
fi
prepare_release_provenance "$CONFIGURATION"
if [[ "$DIAGNOSTIC_RELEASE" == "1" ]] && is_distribution_export "$EXPORT_METHOD"; then
  fail "Dirty diagnostic archives cannot use distribution export method '$EXPORT_METHOD'; choose 'debugging'."
fi

TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
SCHEME_SAFE="${SCHEME//[^A-Za-z0-9_.-]/_}"
ARCHIVE_PATH="build/${SCHEME_SAFE}-${TIMESTAMP}.xcarchive"
EXPORT_DIR="build/export-${SCHEME_SAFE}-${TIMESTAMP}"
EXPORT_OPTIONS_PLIST="build/export-options-${SCHEME_SAFE}-${TIMESTAMP}.plist"
LOG_DIR="build/logs"
EXPORT_LOG="$LOG_DIR/export-${SCHEME_SAFE}-${TIMESTAMP}.log"
UPLOAD_LOG="$LOG_DIR/upload-${SCHEME_SAFE}-${TIMESTAMP}.log"

mkdir -p build "$LOG_DIR"

bold "Choose auth mode"
echo "1) App Store Connect API key (recommended)"
echo "2) Apple ID + app-specific password"
AUTH_MODE="$(read_with_default 'Select [1/2]' "$DEFAULT_AUTH_MODE")"

API_KEY=""
API_ISSUER=""
API_KEY_DIR=""
API_KEY_PATH=""
APPLE_ID=""
APP_SPECIFIC_PASSWORD=""
ASC_PROVIDER="${ASC_PROVIDER:-}"
UPLOAD_AFTER_BUILD="${ASC_UPLOAD_AFTER_BUILD:-}"
if [[ "$NO_UPLOAD_REQUESTED" == "1" ]]; then
  UPLOAD_AFTER_BUILD="N"
fi

case "$AUTH_MODE" in
  1)
    API_KEY="$(read_with_default 'API key ID' "${ASC_API_KEY:-}")"
    API_ISSUER="$(read_with_default 'API issuer ID (UUID)' "${ASC_API_ISSUER:-}")"
    API_KEY_DIR="$(read_with_default 'Directory containing AuthKey_<KEYID>.p8' "${ASC_API_KEY_DIR:-}")"
    [[ -d "$API_KEY_DIR" ]] || fail "API key directory not found: $API_KEY_DIR"
    API_KEY_PATH="$API_KEY_DIR/AuthKey_${API_KEY}.p8"
    [[ -f "$API_KEY_PATH" ]] || fail "Missing API key file: $API_KEY_PATH"
    ASC_PROVIDER="$(read_with_default 'Optional provider short name, or blank to skip' "$ASC_PROVIDER")"
    ;;
  2)
    APPLE_ID="$(read_with_default 'Apple ID (email)' "${ASC_APPLE_ID:-}")"
    APP_SPECIFIC_PASSWORD="$(read_secret_required 'App-specific password: ')"
    ASC_PROVIDER="$(read_with_default 'Optional provider short name, or blank to skip' "$ASC_PROVIDER")"
    warn "Apple ID auth can upload the IPA, but automatic profile creation still depends on Xcode being signed into your Apple Developer account."
    ;;
  *)
    fail "Invalid auth mode."
    ;;
esac

save_local_config
info "Saved local defaults to $CONFIG_FILE"

SIGNING_CERTIFICATE="Apple Development"
if is_distribution_export "$EXPORT_METHOD"; then
  SIGNING_CERTIFICATE="Apple Distribution"
fi

write_export_options_plist "$EXPORT_OPTIONS_PLIST" "$EXPORT_METHOD" "$TEAM_ID" "$SIGNING_CERTIFICATE"

XCODE_AUTH_ARGS=()
if [[ "$AUTH_MODE" == "1" ]]; then
  XCODE_AUTH_ARGS+=(
    -authenticationKeyPath "$API_KEY_PATH"
    -authenticationKeyID "$API_KEY"
    -authenticationKeyIssuerID "$API_ISSUER"
  )
fi

info "Archive via stable linker-safe archive script"
ARCHIVE_ENV=(
  "LUMEN_ARCHIVE_PATH=$REPO_ROOT/$ARCHIVE_PATH"
  "LUMEN_IOS_PROJECT_PATH=$PROJECT_PATH"
  "LUMEN_IOS_SCHEME=$SCHEME"
  "LUMEN_IOS_CONFIGURATION=$CONFIGURATION"
  "LUMEN_IOS_CURRENT_PROJECT_VERSION=$BUILD_NUMBER"
  "LUMEN_GIT_SHA=$RELEASE_GIT_SHA"
  "LUMEN_BUILD_SCHEME=$RELEASE_BUILD_SCHEME"
  "LUMEN_ALLOW_DIRTY_RELEASE_DIAGNOSTIC=${LUMEN_ALLOW_DIRTY_RELEASE_DIAGNOSTIC:-0}"
  "LUMEN_IOS_ALLOW_PROVISIONING_UPDATES=1"
  "LUMEN_IOS_CODE_SIGN_STYLE=Automatic"
  "LUMEN_IOS_DEVELOPMENT_TEAM=$TEAM_ID"
  "LUMEN_IOS_CLEAR_PROVISIONING_PROFILE_SPECIFIER=1"
)
if [[ "$AUTH_MODE" == "1" ]]; then
  ARCHIVE_ENV+=(
    "LUMEN_IOS_AUTHENTICATION_KEY_PATH=$API_KEY_PATH"
    "LUMEN_IOS_AUTHENTICATION_KEY_ID=$API_KEY"
    "LUMEN_IOS_AUTHENTICATION_KEY_ISSUER_ID=$API_ISSUER"
  )
fi

env "${ARCHIVE_ENV[@]}" bash "$STABLE_ARCHIVE_SCRIPT"

info "Verify archived app Info.plist"
ARCHIVE_INFO_CHECK=(python3 "$REPO_ROOT/scripts/check_built_app_info_plist.py" \
  "$ARCHIVE_PATH" \
  --expected-bundle-version "$BUILD_NUMBER" \
  --expected-build-configuration "$CONFIGURATION")
if is_distribution_export "$EXPORT_METHOD"; then
  ARCHIVE_INFO_CHECK+=(--require-dsym-archive "$ARCHIVE_PATH")
fi
"${ARCHIVE_INFO_CHECK[@]}"

info "Verify archived full-commit provenance"
verify_artifact_provenance_metadata \
  "$ARCHIVE_PATH" \
  "$RELEASE_GIT_SHA" \
  "$RELEASE_BUILD_SCHEME"

info "Verify archived app signed entitlements"
python3 "$REPO_ROOT/scripts/validate_ios_signing_capabilities.py" \
  --signing-stage archive \
  --signed-app-path "$ARCHIVE_PATH"

info "Export IPA"
run_logged "$EXPORT_LOG" \
  xcodebuild \
    -exportArchive \
    -archivePath "$ARCHIVE_PATH" \
    -exportPath "$EXPORT_DIR" \
    -exportOptionsPlist "$EXPORT_OPTIONS_PLIST" \
    -onlyUsePackageVersionsFromResolvedFile \
    -allowProvisioningUpdates \
    "${XCODE_AUTH_ARGS[@]}"

if is_distribution_export "$EXPORT_METHOD"; then
  PACKAGING_LOG="$EXPORT_DIR/Packaging.log"
  [[ -f "$PACKAGING_LOG" ]] \
    || fail "Distribution export did not produce the required Packaging.log: $PACKAGING_LOG"
  if grep -Fq "Upload Symbols Failed" "$PACKAGING_LOG"; then
    fail "Distribution export reported 'Upload Symbols Failed' in $PACKAGING_LOG"
  fi
  info "Distribution export packaged symbols without an Upload Symbols Failed diagnostic"
fi

IPA_PATH="$("$FIND_BIN" "$EXPORT_DIR" -maxdepth 1 -type f -name '*.ipa' -print -quit)"
[[ -n "$IPA_PATH" ]] || fail "No IPA found in $EXPORT_DIR"

info "Verify exported IPA Info.plist"
IPA_INFO_CHECK=(python3 "$REPO_ROOT/scripts/check_built_app_info_plist.py" \
  "$IPA_PATH" \
  --expected-bundle-version "$BUILD_NUMBER" \
  --expected-build-configuration "$CONFIGURATION")
if is_distribution_export "$EXPORT_METHOD"; then
  IPA_INFO_CHECK+=(--require-dsym-archive "$ARCHIVE_PATH")
fi
"${IPA_INFO_CHECK[@]}"

info "Verify exported full-commit provenance"
verify_artifact_provenance_metadata \
  "$IPA_PATH" \
  "$RELEASE_GIT_SHA" \
  "$RELEASE_BUILD_SCHEME"

info "Verify exported IPA signed entitlements"
python3 "$REPO_ROOT/scripts/validate_ios_signing_capabilities.py" \
  --signing-stage app-store \
  --signed-app-path "$IPA_PATH"

assert_release_provenance_still_valid

bold "Built IPA: $IPA_PATH"
if [[ "$NO_UPLOAD_REQUESTED" == "1" ]]; then
  warn "Upload prohibited by LUMEN_NO_UPLOAD=1 or diagnostic provenance. IPA is ready at: $IPA_PATH"
  exit 0
fi
if [[ -z "$UPLOAD_AFTER_BUILD" ]]; then
  if read_yes_no_with_default "Upload this IPA to App Store Connect now?" "Y"; then
    UPLOAD_AFTER_BUILD="Y"
  else
    UPLOAD_AFTER_BUILD="N"
  fi
  save_local_config
fi

if [[ ! "$UPLOAD_AFTER_BUILD" =~ ^[Yy]$ ]]; then
  warn "Upload skipped. IPA is ready at: $IPA_PATH"
  exit 0
fi

[[ "$NO_UPLOAD_REQUESTED" != "1" ]] \
  || fail "Internal safety latch prevented upload after no-upload mode was requested."
ensure_upload_tool
info "Upload via altool"
UPLOAD_CMD=(xcrun altool --upload-app --type ios --file "$IPA_PATH")
[[ -n "$ASC_PROVIDER" ]] && UPLOAD_CMD+=(--asc-provider "$ASC_PROVIDER")
if [[ "$AUTH_MODE" == "1" ]]; then
  export API_PRIVATE_KEYS_DIR="$API_KEY_DIR"
  UPLOAD_CMD+=(--apiKey "$API_KEY" --apiIssuer "$API_ISSUER")
else
  export APP_SPECIFIC_PASSWORD
  UPLOAD_CMD+=(--username "$APPLE_ID" --password @env:APP_SPECIFIC_PASSWORD)
fi

set +e
"${UPLOAD_CMD[@]}" 2>&1 | tee "$UPLOAD_LOG"
upload_status=${PIPESTATUS[0]}
set -e

if ! delivery_uuid="$(validate_upload_result "$upload_status" "$UPLOAD_LOG")"; then
  unset APP_SPECIFIC_PASSWORD || true
  fail "Upload failed. Full upload log: $UPLOAD_LOG"
fi
unset APP_SPECIFIC_PASSWORD || true
bold "✅ Upload complete. Delivery UUID: $delivery_uuid"
info "Check App Store Connect for processing status."
