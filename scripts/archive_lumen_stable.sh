#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_FILE="$REPO_ROOT/ios/Lumen.xcodeproj/project.pbxproj"
PACKAGE_RESOLVED_PATH="$REPO_ROOT/ios/Lumen.xcodeproj/project.xcworkspace/xcshareddata/swiftpm/Package.resolved"
EXPECTED_MSAL_VERSION="1.9.0"
EXPECTED_MSAL_REVISION="be848ee7fa9516cec47ae6de47cf1087d51bc774"
MSAL_DSYM_CACHE_DIR="${LUMEN_MSAL_DSYM_CACHE_DIR:-$REPO_ROOT/build/ReleaseSymbols/MSAL/$EXPECTED_MSAL_VERSION}"
MSAL_DSYM_SOURCE_ZIP="${LUMEN_MSAL_DSYM_SOURCE_ZIP:-}"
MSAL_DSYM_OFFLINE="${LUMEN_MSAL_DSYM_OFFLINE:-0}"
if [[ "$MSAL_DSYM_CACHE_DIR" != /* ]]; then
  MSAL_DSYM_CACHE_DIR="$REPO_ROOT/$MSAL_DSYM_CACHE_DIR"
fi
PROJECT_PATH_INPUT="${LUMEN_IOS_PROJECT_PATH:-ios/Lumen.xcodeproj}"
PROJECT_PATH="$PROJECT_PATH_INPUT"
if [[ "$PROJECT_PATH" != /* ]]; then
  PROJECT_PATH="$REPO_ROOT/$PROJECT_PATH"
fi
SCHEME="${LUMEN_IOS_SCHEME:-Lumen}"
CONFIGURATION="${LUMEN_IOS_CONFIGURATION:-Release}"
SWIFT_OPTIMIZATION_LEVEL_VALUE="${LUMEN_SWIFT_OPTIMIZATION_LEVEL:--Osize}"
CURRENT_PROJECT_VERSION_VALUE="${LUMEN_IOS_CURRENT_PROJECT_VERSION:-}"
MARKETING_VERSION_VALUE="${LUMEN_IOS_MARKETING_VERSION:-}"
DERIVED_DATA_ROOT="${LUMEN_DERIVED_DATA_ROOT:-$HOME/Library/Developer/Xcode/DerivedData}"
DERIVED_DATA_PATH="${LUMEN_DERIVED_DATA_PATH:-$REPO_ROOT/build/DerivedData/Archive}"
CLONED_SOURCE_PACKAGES_DIR="${LUMEN_CLONED_SOURCE_PACKAGES_DIR:-$REPO_ROOT/build/SourcePackages}"
PACKAGE_CACHE_PATH="${LUMEN_PACKAGE_CACHE_PATH:-$REPO_ROOT/build/SwiftPMCache}"
TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
ARCHIVE_PATH="${LUMEN_ARCHIVE_PATH:-$REPO_ROOT/build/Lumen-$TIMESTAMP.xcarchive}"
LOG_DIR="$REPO_ROOT/build/logs"
LOG_PATH="$LOG_DIR/archive-stable-$TIMESTAMP.log"
SANITIZED_ENTITLEMENTS_PATH="$REPO_ROOT/build/LumenArchive.entitlements"
SIGNING_XCCONFIG_PATH="$REPO_ROOT/build/LumenArchiveSigningOverrides.xcconfig"
LOCK_DIR="$REPO_ROOT/build/archive_lumen_stable.lock"
LOCK_PID_FILE="$LOCK_DIR/pid"
ALLOW_PROVISIONING_UPDATES="${LUMEN_IOS_ALLOW_PROVISIONING_UPDATES:-0}"
CODE_SIGN_STYLE_VALUE="${LUMEN_IOS_CODE_SIGN_STYLE:-}"
CODE_SIGN_IDENTITY_VALUE="${LUMEN_IOS_CODE_SIGN_IDENTITY:-}"
DEVELOPMENT_TEAM_VALUE="${LUMEN_IOS_DEVELOPMENT_TEAM:-}"
PROVISIONING_PROFILE_SPECIFIER_VALUE="${LUMEN_IOS_PROVISIONING_PROFILE_SPECIFIER:-}"
CLEAR_PROVISIONING_PROFILE_SPECIFIER="${LUMEN_IOS_CLEAR_PROVISIONING_PROFILE_SPECIFIER:-0}"
AUTHENTICATION_KEY_PATH="${LUMEN_IOS_AUTHENTICATION_KEY_PATH:-}"
AUTHENTICATION_KEY_ID="${LUMEN_IOS_AUTHENTICATION_KEY_ID:-}"
AUTHENTICATION_KEY_ISSUER_ID="${LUMEN_IOS_AUTHENTICATION_KEY_ISSUER_ID:-}"
REQUESTED_LUMEN_GIT_SHA_VALUE="${LUMEN_GIT_SHA:-}"
ALLOW_DIRTY_RELEASE_DIAGNOSTIC="${LUMEN_ALLOW_DIRTY_RELEASE_DIAGNOSTIC:-0}"
LUMEN_GIT_SHA_VALUE=""
LUMEN_BUILD_SCHEME_VALUE=""
RELEASE_GIT_COMMIT=""
DIAGNOSTIC_RELEASE=0
AGENT_GROUNDING_RESOURCE_MODE_VALUE="${LUMEN_AGENT_GROUNDING_RESOURCE_MODE:-minimal}"
RELEASE_SCOPE_PATHS=(ios scripts generated/agent_manifest)

bold() { printf "\033[1m%s\033[0m\n" "$1"; }
info() { printf "\n➡️  %s\n" "$1"; }
warn() { printf "\n⚠️  %s\n" "$1"; }
fail() { printf "\n❌ %s\n" "$1"; exit 1; }

is_release_configuration() {
  case "$1" in
    Release|AppStore|App\ Store) return 0 ;;
    *) return 1 ;;
  esac
}

release_scope_changes_at() {
  local repo_root="$1"
  git -C "$repo_root" status --porcelain=v1 --untracked-files=all -- "${RELEASE_SCOPE_PATHS[@]}"
}

compute_release_provenance() {
  local repo_root="$1"
  local configuration="$2"
  local requested_git_sha="$3"
  local allow_dirty_diagnostic="$4"
  local full_commit
  local changes=""

  case "$allow_dirty_diagnostic" in
    0|1) ;;
    *)
      printf 'LUMEN_ALLOW_DIRTY_RELEASE_DIAGNOSTIC must be 0 or 1; received %s.\n' \
        "$allow_dirty_diagnostic" >&2
      return 1
      ;;
  esac

  if ! full_commit="$(git -C "$repo_root" rev-parse --verify HEAD^{commit} 2>/dev/null)"; then
    printf 'Cannot resolve a Git commit for release provenance in %s.\n' "$repo_root" >&2
    return 1
  fi
  if [[ ! "$full_commit" =~ ^[0-9a-f]{40}([0-9a-f]{24})?$ ]]; then
    printf 'Release provenance requires a full Git object ID; resolved %s.\n' "$full_commit" >&2
    return 1
  fi

  COMPUTED_RELEASE_GIT_COMMIT="$full_commit"
  COMPUTED_LUMEN_GIT_SHA="$full_commit"
  COMPUTED_RELEASE_PROVENANCE="production-clean:$full_commit"
  COMPUTED_DIAGNOSTIC_RELEASE=0

  if is_release_configuration "$configuration"; then
    if ! changes="$(release_scope_changes_at "$repo_root")"; then
      printf 'Could not inspect release-scope Git state in %s.\n' "$repo_root" >&2
      return 1
    fi
    if [[ -n "$changes" ]]; then
      if [[ "$allow_dirty_diagnostic" != "1" ]]; then
        printf 'Release archive refused: tracked or untracked release inputs are dirty:\n%s\n' \
          "$changes" >&2
        printf 'Commit/stash those inputs, or use LUMEN_ALLOW_DIRTY_RELEASE_DIAGNOSTIC=1 only for a non-distributable diagnostic artifact.\n' >&2
        return 1
      fi
      COMPUTED_RELEASE_PROVENANCE="diagnostic-dirty-not-for-distribution:$full_commit"
      COMPUTED_DIAGNOSTIC_RELEASE=1
    fi
  fi

  if [[ -n "$requested_git_sha" && "$requested_git_sha" != "$full_commit" ]]; then
    printf 'LUMEN_GIT_SHA=%s does not match the full release commit %s.\n' \
      "$requested_git_sha" "$full_commit" >&2
    return 1
  fi
}

prepare_release_provenance() {
  compute_release_provenance \
    "$REPO_ROOT" \
    "$CONFIGURATION" \
    "$REQUESTED_LUMEN_GIT_SHA_VALUE" \
    "$ALLOW_DIRTY_RELEASE_DIAGNOSTIC" \
    || fail "Release provenance preflight failed before archive side effects."

  RELEASE_GIT_COMMIT="$COMPUTED_RELEASE_GIT_COMMIT"
  LUMEN_GIT_SHA_VALUE="$COMPUTED_LUMEN_GIT_SHA"
  DIAGNOSTIC_RELEASE="$COMPUTED_DIAGNOSTIC_RELEASE"

  if [[ "$DIAGNOSTIC_RELEASE" == "1" ]]; then
    LUMEN_BUILD_SCHEME_VALUE="${SCHEME}-DIAGNOSTIC-DIRTY-NOT-FOR-DISTRIBUTION"
    warn "DIRTY DIAGNOSTIC ONLY: this archive is stamped not-for-distribution and is not production provenance."
  else
    LUMEN_BUILD_SCHEME_VALUE="$SCHEME"
    info "Release provenance pinned to full commit: $RELEASE_GIT_COMMIT"
  fi
}

assert_release_provenance_still_valid() {
  local current_commit
  local changes=""

  current_commit="$(git -C "$REPO_ROOT" rev-parse --verify HEAD^{commit} 2>/dev/null)" \
    || fail "Could not re-resolve Git HEAD after archive."
  [[ "$current_commit" == "$RELEASE_GIT_COMMIT" ]] \
    || fail "Git HEAD changed during archive: expected $RELEASE_GIT_COMMIT, found $current_commit."

  if is_release_configuration "$CONFIGURATION" && [[ "$DIAGNOSTIC_RELEASE" != "1" ]]; then
    changes="$(release_scope_changes_at "$REPO_ROOT")" \
      || fail "Could not re-check release-scope Git state after archive."
    if [[ -n "$changes" ]]; then
      printf '%s\n' "$changes" >&2
      fail "Release-scope inputs changed during archive; the artifact is not valid production evidence."
    fi
  fi
}

validate_reviewed_package_resolution() {
  [[ -f "$PACKAGE_RESOLVED_PATH" ]] \
    || fail "Missing tracked Swift package lockfile: $PACKAGE_RESOLVED_PATH"

  python3 - "$PACKAGE_RESOLVED_PATH" "$EXPECTED_MSAL_VERSION" "$EXPECTED_MSAL_REVISION" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
expected_version = sys.argv[2]
expected_revision = sys.argv[3]
payload = json.loads(path.read_text(encoding="utf-8"))
pins = [
    pin
    for pin in payload.get("pins", [])
    if pin.get("identity") == "microsoft-authentication-library-for-objc"
]
if len(pins) != 1:
    raise SystemExit(f"error: {path} must contain exactly one MSAL pin; found {len(pins)}")
state = pins[0].get("state") or {}
actual_version = state.get("version")
actual_revision = state.get("revision")
if actual_version != expected_version or actual_revision != expected_revision:
    raise SystemExit(
        f"error: reviewed MSAL resolution mismatch in {path}: "
        f"version={actual_version!r} revision={actual_revision!r}; "
        f"expected version={expected_version!r} revision={expected_revision!r}"
    )
print(f"Reviewed MSAL resolution verified: {actual_version} ({actual_revision})")
PY
}

verify_archive_provenance_metadata() {
  local archive_path="$1"
  local expected_git_sha="$2"
  local expected_build_scheme="$3"

  python3 - "$archive_path" "$expected_git_sha" "$expected_build_scheme" <<'PY'
from __future__ import annotations

import plistlib
import sys
from pathlib import Path

archive = Path(sys.argv[1])
expected_git_sha = sys.argv[2]
expected_build_scheme = sys.argv[3]
plists = sorted((archive / "Products" / "Applications").glob("*.app/Info.plist"))
if len(plists) != 1:
    raise SystemExit(f"error: expected exactly one archived app Info.plist; found {len(plists)}")
with plists[0].open("rb") as handle:
    info = plistlib.load(handle)
actual_git_sha = info.get("LumenGitSHA")
actual_build_scheme = info.get("LumenBuildScheme")
if actual_git_sha != expected_git_sha:
    raise SystemExit(
        f"error: archived LumenGitSHA={actual_git_sha!r}; expected {expected_git_sha!r}"
    )
if actual_build_scheme != expected_build_scheme:
    raise SystemExit(
        f"error: archived LumenBuildScheme={actual_build_scheme!r}; "
        f"expected {expected_build_scheme!r}"
    )
print(f"Archive provenance verified: {actual_git_sha} [{actual_build_scheme}]")
PY
}

validate_archive_signing_arguments() {
  local argument
  for argument in "$@"; do
    if [[ "$argument" == "CODE_SIGN_IDENTITY=" ]]; then
      printf 'Archive command contains a literal empty CODE_SIGN_IDENTITY override.\n' >&2
      return 1
    fi
  done
}

run_signing_argument_self_check() {
  if validate_archive_signing_arguments xcodebuild CODE_SIGN_IDENTITY= >/dev/null 2>&1; then
    fail "Signing-argument self-check accepted an empty CODE_SIGN_IDENTITY override."
  fi
  validate_archive_signing_arguments xcodebuild CODE_SIGN_STYLE=Automatic
  printf 'Archive signing-argument self-check passed.\n'
}

signing_identity_output_has_valid_match() {
  local expected_name="$1"

  awk -v expected_name="$expected_name" '
    /^[[:space:]]*[0-9]+\)[[:space:]]+[[:xdigit:]]+[[:space:]]+"[^"]+"[[:space:]]*$/ {
      hash_length = length($2)
      if ((hash_length == 40 || hash_length == 64) && index($0, expected_name) > 0) {
        found = 1
      }
    }
    END { exit found ? 0 : 1 }
  '
}

run_signing_identity_output_self_check() {
  local valid_development='  1) 0123456789ABCDEF0123456789ABCDEF01234567 "Apple Development: Example (TEAM123456)"'
  local valid_distribution='  2) 89ABCDEF0123456789ABCDEF0123456789ABCDEF "Apple Distribution: Example (TEAM123456)"'
  local revoked_distribution='  3) FEDCBA9876543210FEDCBA9876543210FEDCBA98 "Apple Distribution: Example (TEAM123456)" (CSSMERR_TP_CERT_REVOKED)'
  local expired_distribution='  4) ABCDEF0123456789ABCDEF0123456789ABCDEF01 "Apple Distribution: Example (TEAM123456)" (CSSMERR_TP_CERT_EXPIRED)'
  local missing_private_key='  5) 76543210FEDCBA9876543210FEDCBA9876543210 "Apple Distribution: Example (TEAM123456)" (Missing private key)'
  local malformed_identity='  6) not-a-certificate-hash "Apple Distribution: Example (TEAM123456)"'
  local short_hash_identity='  7) ABCDEF "Apple Distribution: Example (TEAM123456)"'
  local summary='     2 valid identities found'

  printf '%s\n' "$valid_development" \
    | signing_identity_output_has_valid_match "Apple Development" \
    || fail "Signing identity output self-check rejected a valid development identity."
  printf '%s\n' "$valid_distribution" \
    | signing_identity_output_has_valid_match "Apple Distribution" \
    || fail "Signing identity output self-check rejected a valid distribution identity."

  local rejected_fixture
  for rejected_fixture in \
    "$revoked_distribution" \
    "$expired_distribution" \
    "$missing_private_key" \
    "$malformed_identity" \
    "$short_hash_identity" \
    "$summary"; do
    if printf '%s\n' "$rejected_fixture" \
      | signing_identity_output_has_valid_match "Apple Distribution"; then
      fail "Signing identity output self-check accepted an invalid distribution identity."
    fi
  done

  if printf '%s\n' "$valid_development" "$revoked_distribution" "$summary" \
    | signing_identity_output_has_valid_match "Apple Distribution"; then
    fail "Signing identity output self-check accepted a revoked match from mixed output."
  fi
  printf '%s\n' "$valid_development" "$valid_distribution" "$revoked_distribution" "$summary" \
    | signing_identity_output_has_valid_match "Apple Distribution" \
    || fail "Signing identity output self-check ignored a valid match when an invalid duplicate was present."

  printf 'Signing identity output self-check passed.\n'
}

run_release_provenance_self_check() {
  local fixture_root
  local full_commit

  fixture_root="$(mktemp -d "${TMPDIR:-/tmp}/lumen-release-provenance.XXXXXX")"
  RELEASE_PROVENANCE_SELF_CHECK_DIR="$fixture_root"
  trap 'rm -rf -- "$RELEASE_PROVENANCE_SELF_CHECK_DIR"' EXIT
  mkdir -p \
    "$fixture_root/ios" \
    "$fixture_root/scripts" \
    "$fixture_root/generated/agent_manifest" \
    "$fixture_root/docs"
  printf 'app\n' > "$fixture_root/ios/app.swift"
  printf 'archive\n' > "$fixture_root/scripts/archive.sh"
  printf '{}\n' > "$fixture_root/generated/agent_manifest/AgentBehaviorManifest.json"
  git -C "$fixture_root" init -q
  git -C "$fixture_root" add ios scripts generated/agent_manifest
  git -C "$fixture_root" \
    -c user.name='Lumen Release Self Check' \
    -c user.email='lumen-release-self-check@example.invalid' \
    commit -qm 'fixture'
  full_commit="$(git -C "$fixture_root" rev-parse --verify HEAD^{commit})"

  compute_release_provenance "$fixture_root" Release "" 0
  [[ "$COMPUTED_LUMEN_GIT_SHA" == "$full_commit" ]] \
    || fail "Release-provenance self-check did not preserve the full clean commit."
  [[ "$COMPUTED_RELEASE_PROVENANCE" == "production-clean:$full_commit" ]] \
    || fail "Release-provenance self-check produced the wrong production marker."

  printf 'dirty\n' >> "$fixture_root/ios/app.swift"
  if compute_release_provenance "$fixture_root" Release "" 0 >/dev/null 2>&1; then
    fail "Release-provenance self-check accepted a dirty tracked release input."
  fi
  compute_release_provenance "$fixture_root" Release "$full_commit" 1
  [[ "$COMPUTED_DIAGNOSTIC_RELEASE" == "1" ]] \
    || fail "Release-provenance self-check did not classify the dirty override as diagnostic."
  [[ "$COMPUTED_RELEASE_PROVENANCE" == "diagnostic-dirty-not-for-distribution:$full_commit" ]] \
    || fail "Release-provenance self-check produced the wrong diagnostic marker."

  git -C "$fixture_root" restore -- ios/app.swift
  : > "$fixture_root/scripts/untracked-release-input.sh"
  if compute_release_provenance "$fixture_root" Release "" 0 >/dev/null 2>&1; then
    fail "Release-provenance self-check accepted an untracked release input."
  fi
  rm -f -- "$fixture_root/scripts/untracked-release-input.sh"

  : > "$fixture_root/docs/unrelated-note.md"
  compute_release_provenance "$fixture_root" Release "$full_commit" 0
  [[ "$COMPUTED_DIAGNOSTIC_RELEASE" == "0" ]] \
    || fail "Release-provenance self-check treated an out-of-scope document as a release input."

  printf 'Release provenance self-check passed.\n'
}

if [[ "${1:-}" == "--self-check-signing-arguments" ]]; then
  run_signing_argument_self_check
  exit 0
fi

if [[ "${1:-}" == "--self-check-signing-identity-output" ]]; then
  run_signing_identity_output_self_check
  exit 0
fi

if [[ "${1:-}" == "--self-check-release-provenance" ]]; then
  run_release_provenance_self_check
  exit 0
fi

if [[ "${1:-}" == "--self-check-reviewed-package-resolution" ]]; then
  validate_reviewed_package_resolution
  exit 0
fi

if [[ "${1:-}" == "--preflight-release-provenance" ]]; then
  prepare_release_provenance
  printf 'LUMEN_GIT_SHA=%s\n' "$LUMEN_GIT_SHA_VALUE"
  printf 'LUMEN_BUILD_SCHEME=%s\n' "$LUMEN_BUILD_SCHEME_VALUE"
  exit 0
fi

acquire_archive_lock() {
  mkdir -p "$REPO_ROOT/build"

  if mkdir "$LOCK_DIR" 2>/dev/null; then
    printf '%s\n' "$$" > "$LOCK_PID_FILE"
    trap 'rm -rf "$LOCK_DIR"' EXIT
    return 0
  fi

  local existing_pid=""
  if [[ -f "$LOCK_PID_FILE" ]]; then
    existing_pid="$(cat "$LOCK_PID_FILE" 2>/dev/null || true)"
  fi

  if [[ "$existing_pid" =~ ^[0-9]+$ ]] && kill -0 "$existing_pid" 2>/dev/null; then
    fail "Another Lumen archive is already running (pid $existing_pid). Wait for it to finish before starting a new archive."
  fi

  warn "Removing stale archive lock at $LOCK_DIR"
  rm -rf "$LOCK_DIR"
  if mkdir "$LOCK_DIR" 2>/dev/null; then
    printf '%s\n' "$$" > "$LOCK_PID_FILE"
    trap 'rm -rf "$LOCK_DIR"' EXIT
    return 0
  fi

  fail "Could not acquire archive lock at $LOCK_DIR"
}

print_xcode_log_diagnostics() {
  local log_path="$1"
  [[ -f "$log_path" ]] || return 0

  warn "xcodebuild failed. Full log: $log_path"
  info "Most relevant diagnostics"
  grep -nEi \
    '(^|[^A-Za-z])(error:|fatal error:|warning:|failed|failure|Command SwiftCompile failed|Command CompileSwift failed|Command CodeSign failed|no such module|cannot find|cannot convert|ambiguous|missing|undefined|duplicate|provisioning|codesign|SwiftDriver|CompileSwift|Ld |ld:|clang: error)' \
    "$log_path" | tail -n 160 || true

  info "Last 220 lines"
  tail -n 220 "$log_path" || true
}

run_logged() {
  local log_path="$1"
  shift
  mkdir -p "$(dirname "$log_path")"

  # Temporarily disable `set -e` so we can capture the exit code of the
  # first command in the pipeline (`$@`) while still streaming logs via `tee`.
  # If more pipes are added here, update the PIPESTATUS index intentionally.
  set +e
  "$@" 2>&1 | tee "$log_path"
  local -a cmd_status=("${PIPESTATUS[@]}")
  local status=${cmd_status[0]}
  set -e

  if [[ $status -ne 0 ]]; then
    print_xcode_log_diagnostics "$log_path"
    return "$status"
  fi
}

build_project_selector_args() {
  local project_path="$1"
  if [[ "$project_path" == *.xcworkspace ]]; then
    printf '%s\0%s\0' "-workspace" "$project_path"
  else
    printf '%s\0%s\0' "-project" "$project_path"
  fi
}

clean_lumen_derived_data() {
  [[ -n "$DERIVED_DATA_PATH" && "$DERIVED_DATA_PATH" != "/" ]] \
    || fail "Invalid DERIVED_DATA_PATH for cleanup: $DERIVED_DATA_PATH"

  rm -rf "$DERIVED_DATA_PATH"

  if [[ "${LUMEN_CLEAN_GLOBAL_DERIVED_DATA:-0}" != "1" ]]; then
    return 0
  fi

  [[ -n "$HOME" && "$HOME" != "/" ]] || fail "Invalid HOME for cleanup: $HOME"
  [[ -n "$DERIVED_DATA_ROOT" ]] || fail "DERIVED_DATA_ROOT is empty; refusing cleanup."
  [[ "$DERIVED_DATA_ROOT" == "$HOME/Library/Developer/Xcode/DerivedData" ]] \
    || fail "Refusing to clean unexpected DERIVED_DATA_ROOT: $DERIVED_DATA_ROOT"

  shopt -s nullglob
  local targets=("$DERIVED_DATA_ROOT"/Lumen-*)
  shopt -u nullglob
  if (( ${#targets[@]} > 0 )); then
    rm -rf "${targets[@]}"
  fi
}

reset_swiftpm_cache() {
  [[ -n "$HOME" && "$HOME" != "/" ]] || fail "Invalid HOME for SwiftPM cleanup: $HOME"
  [[ -n "$REPO_ROOT" && "$REPO_ROOT" != "/" ]] || fail "Invalid REPO_ROOT for cleanup: $REPO_ROOT"

  local swiftpm_cache="$HOME/Library/Caches/org.swift.swiftpm"
  if [[ -d "$swiftpm_cache" ]]; then
    rm -rf "$swiftpm_cache"
  else
    info "No SwiftPM cache directory found at $swiftpm_cache."
  fi
  rm -rf "$REPO_ROOT/.build"
}

reset_package_resolution_state() {
  rm -rf "$CLONED_SOURCE_PACKAGES_DIR" "$PACKAGE_CACHE_PATH"
}

resolve_package_dependencies() {
  local attempts="${LUMEN_PACKAGE_RESOLVE_ATTEMPTS:-2}"
  local attempt=1
  local log_path

  while (( attempt <= attempts )); do
    if (( attempt == 1 )); then
      log_path="$LOG_DIR/resolve-packages-$TIMESTAMP.log"
    else
      log_path="$LOG_DIR/resolve-packages-$TIMESTAMP-retry-$attempt.log"
      warn "Retrying Swift package resolution after clearing package checkouts/cache"
      reset_package_resolution_state
    fi

    mkdir -p "$DERIVED_DATA_PATH" "$CLONED_SOURCE_PACKAGES_DIR" "$PACKAGE_CACHE_PATH"
    if run_logged "$log_path" \
      xcodebuild \
        "${PROJECT_SELECTOR[@]}" \
        -scheme "$SCHEME" \
        -derivedDataPath "$DERIVED_DATA_PATH" \
        -clonedSourcePackagesDirPath "$CLONED_SOURCE_PACKAGES_DIR" \
        -packageCachePath "$PACKAGE_CACHE_PATH" \
        -onlyUsePackageVersionsFromResolvedFile \
        -resolvePackageDependencies; then
      return 0
    fi

    if (( attempt >= attempts )); then
      return 1
    fi

    attempt=$((attempt + 1))
  done
}

expected_signing_identity_patterns() {
  if [[ -n "$EFFECTIVE_CODE_SIGN_IDENTITY_VALUE" ]]; then
    printf '%s\n' "$EFFECTIVE_CODE_SIGN_IDENTITY_VALUE"
    return 0
  fi

  case "$CONFIGURATION" in
    Release|AppStore|App\ Store)
      printf '%s\n' "iOS Distribution"
      printf '%s\n' "Apple Distribution"
      ;;
    *)
      printf '%s\n' "iPhone Developer"
      printf '%s\n' "Apple Development"
      ;;
  esac
}

has_matching_signing_identity() {
  local pattern="$1"
  security find-identity -v -p codesigning 2>/dev/null \
    | signing_identity_output_has_valid_match "$pattern"
}

preflight_signing_identity() {
  if [[ "${LUMEN_IOS_SKIP_SIGNING_IDENTITY_PREFLIGHT:-0}" == "1" ]]; then
    warn "Skipping signing identity preflight because LUMEN_IOS_SKIP_SIGNING_IDENTITY_PREFLIGHT=1."
    return 0
  fi
  command -v security >/dev/null 2>&1 || fail "security command not found; cannot inspect code signing identities."

  local patterns=()
  local pattern
  while IFS= read -r pattern; do
    [[ -n "$pattern" ]] && patterns+=("$pattern")
  done < <(expected_signing_identity_patterns)

  for pattern in "${patterns[@]}"; do
    if has_matching_signing_identity "$pattern"; then
      info "Found local code signing identity matching: $pattern"
      return 0
    fi
  done

  local expected
  expected="$(IFS=', '; printf '%s' "${patterns[*]}")"
  if [[ "$ALLOW_PROVISIONING_UPDATES" == "1" && ${#XCODE_AUTH_ARGS[@]} -gt 0 ]]; then
    warn "No local code signing identity matched '$expected'; continuing because authenticated automatic provisioning is enabled."
    return 0
  fi

  fail "No local code signing identity matched '$expected'. Install the Apple distribution certificate/private key for team '${DEVELOPMENT_TEAM_VALUE:-project default}', or rerun with App Store Connect API auth and LUMEN_IOS_ALLOW_PROVISIONING_UPDATES=1."
}

[[ "$(uname -s)" == "Darwin" ]] || fail "This script must run on macOS."
command -v xcodebuild >/dev/null 2>&1 || fail "xcodebuild not found. Select Xcode with: sudo xcode-select --switch /Applications/Xcode.app/Contents/Developer"
command -v python3 >/dev/null 2>&1 || fail "python3 not found. Install Xcode Command Line Tools or Python 3."
[[ -f "$PROJECT_FILE" ]] || fail "Missing project file: $PROJECT_FILE"
[[ -e "$PROJECT_PATH" ]] || fail "Project/workspace path not found: $PROJECT_PATH"
[[ "$PROJECT_PATH" == *.xcworkspace || "$PROJECT_PATH" == *.xcodeproj ]] || fail "Project path must be .xcworkspace or .xcodeproj: $PROJECT_PATH"
case "$CONFIGURATION" in
  Release|AppStore|App\ Store)
    case "$SWIFT_OPTIMIZATION_LEVEL_VALUE" in
      -O|-Osize) ;;
      *) fail "Release archives require Swift optimization (-O or -Osize); refusing SWIFT_OPTIMIZATION_LEVEL=$SWIFT_OPTIMIZATION_LEVEL_VALUE." ;;
    esac
    ;;
esac
case "$AGENT_GROUNDING_RESOURCE_MODE_VALUE" in
  full|minimal|skip) ;;
  *) fail "Unsupported LUMEN_AGENT_GROUNDING_RESOURCE_MODE=$AGENT_GROUNDING_RESOURCE_MODE_VALUE (expected full, minimal, or skip)." ;;
esac
case "$MSAL_DSYM_OFFLINE" in
  0|1) ;;
  *) fail "LUMEN_MSAL_DSYM_OFFLINE must be 0 or 1; received $MSAL_DSYM_OFFLINE." ;;
esac
if [[ -n "$MSAL_DSYM_SOURCE_ZIP" ]]; then
  [[ -f "$MSAL_DSYM_SOURCE_ZIP" ]] \
    || fail "LUMEN_MSAL_DSYM_SOURCE_ZIP does not exist: $MSAL_DSYM_SOURCE_ZIP"
  MSAL_DSYM_SOURCE_ZIP="$(cd "$(dirname "$MSAL_DSYM_SOURCE_ZIP")" && pwd -P)/$(basename "$MSAL_DSYM_SOURCE_ZIP")"
fi

cd "$REPO_ROOT"
prepare_release_provenance
validate_reviewed_package_resolution
mkdir -p "$LOG_DIR" "$REPO_ROOT/build"
acquire_archive_lock

PROJECT_SELECTOR=()
while IFS= read -r -d '' arg; do
  PROJECT_SELECTOR+=("$arg")
done < <(build_project_selector_args "$PROJECT_PATH")

XCODE_AUTH_ARGS=()
if [[ -n "$AUTHENTICATION_KEY_PATH$AUTHENTICATION_KEY_ID$AUTHENTICATION_KEY_ISSUER_ID" ]]; then
  [[ -n "$AUTHENTICATION_KEY_PATH" ]] || fail "LUMEN_IOS_AUTHENTICATION_KEY_PATH is required when using App Store Connect API key auth."
  [[ -n "$AUTHENTICATION_KEY_ID" ]] || fail "LUMEN_IOS_AUTHENTICATION_KEY_ID is required when using App Store Connect API key auth."
  [[ -n "$AUTHENTICATION_KEY_ISSUER_ID" ]] || fail "LUMEN_IOS_AUTHENTICATION_KEY_ISSUER_ID is required when using App Store Connect API key auth."
  [[ -f "$AUTHENTICATION_KEY_PATH" ]] || fail "Authentication key not found: $AUTHENTICATION_KEY_PATH"
  XCODE_AUTH_ARGS+=(
    -authenticationKeyPath "$AUTHENTICATION_KEY_PATH"
    -authenticationKeyID "$AUTHENTICATION_KEY_ID"
    -authenticationKeyIssuerID "$AUTHENTICATION_KEY_ISSUER_ID"
  )
fi

PROVISIONING_ARGS=()
if [[ "$ALLOW_PROVISIONING_UPDATES" == "1" ]]; then
  PROVISIONING_ARGS+=(-allowProvisioningUpdates)
fi

SIGNING_BUILD_SETTINGS=()
SIGNING_XCCONFIG_ARGS=()
if [[ -n "$CODE_SIGN_STYLE_VALUE" ]]; then
  SIGNING_BUILD_SETTINGS+=("CODE_SIGN_STYLE=$CODE_SIGN_STYLE_VALUE")
fi
EFFECTIVE_CODE_SIGN_IDENTITY_VALUE="$CODE_SIGN_IDENTITY_VALUE"
if [[ -n "$CODE_SIGN_IDENTITY_VALUE" ]]; then
  info "Using codesigning identity override: $CODE_SIGN_IDENTITY_VALUE"
elif [[ "$CODE_SIGN_STYLE_VALUE" == "Automatic" ]]; then
  # Automatic signing archives with an Apple Development identity. Pinning it
  # here overrides any legacy conditional distribution identity in the project
  # without disabling signing; App Store distribution signing happens later
  # when the archive is exported.
  EFFECTIVE_CODE_SIGN_IDENTITY_VALUE="Apple Development"
  info "Using Apple Development identity for automatic archive signing"
fi
if [[ -n "$EFFECTIVE_CODE_SIGN_IDENTITY_VALUE" ]]; then
  cat > "$SIGNING_XCCONFIG_PATH" <<XCCONFIG
CODE_SIGN_IDENTITY = $EFFECTIVE_CODE_SIGN_IDENTITY_VALUE
CODE_SIGN_IDENTITY[sdk=iphoneos*] = $EFFECTIVE_CODE_SIGN_IDENTITY_VALUE
XCCONFIG
  SIGNING_XCCONFIG_ARGS=(-xcconfig "$SIGNING_XCCONFIG_PATH")
fi
if [[ -n "$DEVELOPMENT_TEAM_VALUE" ]]; then
  SIGNING_BUILD_SETTINGS+=("DEVELOPMENT_TEAM=$DEVELOPMENT_TEAM_VALUE")
fi
if [[ -n "$CURRENT_PROJECT_VERSION_VALUE" ]]; then
  SIGNING_BUILD_SETTINGS+=("CURRENT_PROJECT_VERSION=$CURRENT_PROJECT_VERSION_VALUE")
fi
if [[ -n "$MARKETING_VERSION_VALUE" ]]; then
  SIGNING_BUILD_SETTINGS+=("MARKETING_VERSION=$MARKETING_VERSION_VALUE")
fi
if [[ "$CLEAR_PROVISIONING_PROFILE_SPECIFIER" == "1" ]]; then
  SIGNING_BUILD_SETTINGS+=("PROVISIONING_PROFILE_SPECIFIER=")
elif [[ -n "$PROVISIONING_PROFILE_SPECIFIER_VALUE" ]]; then
  SIGNING_BUILD_SETTINGS+=("PROVISIONING_PROFILE_SPECIFIER=$PROVISIONING_PROFILE_SPECIFIER_VALUE")
fi

bold "Lumen stable iOS archive"
info "Validating and sanitizing iOS signing capabilities"
python3 "$REPO_ROOT/scripts/validate_ios_signing_capabilities.py" \
  --project-file "$PROJECT_FILE" \
  --entitlements "$REPO_ROOT/ios/Lumen/Lumen.entitlements" \
  --sanitized-entitlements-output "$SANITIZED_ENTITLEMENTS_PATH" \
  --allow-sanitized-output
SIGNING_BUILD_SETTINGS+=("CODE_SIGN_ENTITLEMENTS=$SANITIZED_ENTITLEMENTS_PATH")

info "Verifying durable archive/linker build settings"
python3 "$REPO_ROOT/scripts/apply_ios_archive_linker_fix.py" "$PROJECT_FILE" --check

if [[ "${LUMEN_CLEAN_DERIVED_DATA:-1}" == "1" ]]; then
  info "Cleaning Lumen DerivedData"
  clean_lumen_derived_data
fi

if [[ "${LUMEN_RESET_SWIFTPM_CACHE:-0}" == "1" ]]; then
  info "Cleaning SwiftPM cache"
  reset_swiftpm_cache
fi

info "Checking local code signing identity"
preflight_signing_identity

info "Resolving Swift package dependencies"
resolve_package_dependencies

info "Archiving with linker-safe Swift settings and freshly compiled asset catalogs"
mkdir -p "$DERIVED_DATA_PATH" "$CLONED_SOURCE_PACKAGES_DIR" "$PACKAGE_CACHE_PATH"
ARCHIVE_COMMAND=(
  xcodebuild
  "${PROJECT_SELECTOR[@]}"
)
if (( ${#SIGNING_XCCONFIG_ARGS[@]} > 0 )); then
  ARCHIVE_COMMAND+=("${SIGNING_XCCONFIG_ARGS[@]}")
fi
ARCHIVE_COMMAND+=(
  -scheme "$SCHEME"
  -derivedDataPath "$DERIVED_DATA_PATH"
  -clonedSourcePackagesDirPath "$CLONED_SOURCE_PACKAGES_DIR"
  -packageCachePath "$PACKAGE_CACHE_PATH"
  -onlyUsePackageVersionsFromResolvedFile
  -configuration "$CONFIGURATION"
  -destination "generic/platform=iOS"
  -archivePath "$ARCHIVE_PATH"
)
if (( ${#PROVISIONING_ARGS[@]} > 0 )); then
  ARCHIVE_COMMAND+=("${PROVISIONING_ARGS[@]}")
fi
if (( ${#XCODE_AUTH_ARGS[@]} > 0 )); then
  ARCHIVE_COMMAND+=("${XCODE_AUTH_ARGS[@]}")
fi
if (( ${#SIGNING_BUILD_SETTINGS[@]} > 0 )); then
  ARCHIVE_COMMAND+=("${SIGNING_BUILD_SETTINGS[@]}")
fi
ARCHIVE_COMMAND+=(
  COMPILER_INDEX_STORE_ENABLE=NO
  ALWAYS_EMBED_SWIFT_STANDARD_LIBRARIES=YES
  ENABLE_ON_DEMAND_RESOURCES=NO
  DEAD_CODE_STRIPPING=NO
  SWIFT_COMPILATION_MODE=singlefile
  SWIFT_WHOLE_MODULE_OPTIMIZATION=NO
  "SWIFT_OPTIMIZATION_LEVEL=$SWIFT_OPTIMIZATION_LEVEL_VALUE"
  "LUMEN_GIT_SHA=$LUMEN_GIT_SHA_VALUE"
  "LUMEN_BUILD_SCHEME=$LUMEN_BUILD_SCHEME_VALUE"
  "AGENT_GROUNDING_RESOURCE_MODE=$AGENT_GROUNDING_RESOURCE_MODE_VALUE"
  clean
  archive
)
validate_archive_signing_arguments "${ARCHIVE_COMMAND[@]}" \
  || fail "Refusing to run an archive command that disables signing identity selection."
run_logged "$LOG_PATH" "${ARCHIVE_COMMAND[@]}"

if is_release_configuration "$CONFIGURATION"; then
  info "Installing the official hash-pinned MSAL release dSYM"
  MSAL_DSYM_INSTALL=(
    python3 "$REPO_ROOT/scripts/install_msal_release_dsym.py"
    --archive "$ARCHIVE_PATH"
    --cache-dir "$MSAL_DSYM_CACHE_DIR"
  )
  if [[ -n "$MSAL_DSYM_SOURCE_ZIP" ]]; then
    MSAL_DSYM_INSTALL+=(--source-zip "$MSAL_DSYM_SOURCE_ZIP")
  fi
  if [[ "$MSAL_DSYM_OFFLINE" == "1" ]]; then
    MSAL_DSYM_INSTALL+=(--offline)
  fi
  "${MSAL_DSYM_INSTALL[@]}"
fi

info "Verifying archived app Info.plist"
INFO_PLIST_CHECK=(python3 "$REPO_ROOT/scripts/check_built_app_info_plist.py" "$ARCHIVE_PATH" --expected-build-configuration "$CONFIGURATION")
if [[ -n "$CURRENT_PROJECT_VERSION_VALUE" ]]; then
  INFO_PLIST_CHECK+=(--expected-bundle-version "$CURRENT_PROJECT_VERSION_VALUE")
fi
if is_release_configuration "$CONFIGURATION"; then
  INFO_PLIST_CHECK+=(--require-dsym-archive "$ARCHIVE_PATH")
fi
"${INFO_PLIST_CHECK[@]}"

info "Verifying archived full-commit provenance"
verify_archive_provenance_metadata \
  "$ARCHIVE_PATH" \
  "$LUMEN_GIT_SHA_VALUE" \
  "$LUMEN_BUILD_SCHEME_VALUE"

info "Verifying archived app signed entitlements"
python3 "$REPO_ROOT/scripts/validate_ios_signing_capabilities.py" \
  --project-file "$PROJECT_FILE" \
  --entitlements "$REPO_ROOT/ios/Lumen/Lumen.entitlements" \
  --app-store-entitlements "$REPO_ROOT/ios/Lumen/LumenAppStore.entitlements" \
  --signing-stage archive \
  --signed-app-path "$ARCHIVE_PATH"

assert_release_provenance_still_valid

bold "✅ Archive created: $ARCHIVE_PATH"
