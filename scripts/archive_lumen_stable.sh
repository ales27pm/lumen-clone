#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_FILE="$REPO_ROOT/ios/Lumen.xcodeproj/project.pbxproj"
PROJECT_PATH_INPUT="${LUMEN_IOS_PROJECT_PATH:-ios/Lumen.xcodeproj}"
PROJECT_PATH="$PROJECT_PATH_INPUT"
if [[ "$PROJECT_PATH" != /* ]]; then
  PROJECT_PATH="$REPO_ROOT/$PROJECT_PATH"
fi
SCHEME="${LUMEN_IOS_SCHEME:-Lumen}"
CONFIGURATION="${LUMEN_IOS_CONFIGURATION:-Release}"
DERIVED_DATA_ROOT="${LUMEN_DERIVED_DATA_ROOT:-$HOME/Library/Developer/Xcode/DerivedData}"
TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
ARCHIVE_PATH="${LUMEN_ARCHIVE_PATH:-$REPO_ROOT/build/Lumen-$TIMESTAMP.xcarchive}"
LOG_DIR="$REPO_ROOT/build/logs"
LOG_PATH="$LOG_DIR/archive-stable-$TIMESTAMP.log"
SANITIZED_ENTITLEMENTS_PATH="$REPO_ROOT/build/LumenArchive.entitlements"
ALLOW_PROVISIONING_UPDATES="${LUMEN_IOS_ALLOW_PROVISIONING_UPDATES:-0}"
CODE_SIGN_STYLE_VALUE="${LUMEN_IOS_CODE_SIGN_STYLE:-}"
DEVELOPMENT_TEAM_VALUE="${LUMEN_IOS_DEVELOPMENT_TEAM:-}"
PROVISIONING_PROFILE_SPECIFIER_VALUE="${LUMEN_IOS_PROVISIONING_PROFILE_SPECIFIER:-}"
CLEAR_PROVISIONING_PROFILE_SPECIFIER="${LUMEN_IOS_CLEAR_PROVISIONING_PROFILE_SPECIFIER:-0}"
AUTHENTICATION_KEY_PATH="${LUMEN_IOS_AUTHENTICATION_KEY_PATH:-}"
AUTHENTICATION_KEY_ID="${LUMEN_IOS_AUTHENTICATION_KEY_ID:-}"
AUTHENTICATION_KEY_ISSUER_ID="${LUMEN_IOS_AUTHENTICATION_KEY_ISSUER_ID:-}"

bold() { printf "\033[1m%s\033[0m\n" "$1"; }
info() { printf "\n➡️  %s\n" "$1"; }
warn() { printf "\n⚠️  %s\n" "$1"; }
fail() { printf "\n❌ %s\n" "$1"; exit 1; }

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
  [[ -n "$HOME" && "$HOME" != "/" ]] || fail "Invalid HOME for cleanup: $HOME"
  [[ -n "$DERIVED_DATA_ROOT" ]] || fail "DERIVED_DATA_ROOT is empty; refusing cleanup."
  [[ "$DERIVED_DATA_ROOT" == "$HOME/Library/Developer/Xcode/DerivedData" ]] \
    || fail "Refusing to clean unexpected DERIVED_DATA_ROOT: $DERIVED_DATA_ROOT"

  shopt -s nullglob
  local targets=("$DERIVED_DATA_ROOT"/Lumen-*)
  shopt -u nullglob
  if (( ${#targets[@]} == 0 )); then
    info "No Lumen DerivedData directories found."
    return 0
  fi
  rm -rf "${targets[@]}"
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

[[ "$(uname -s)" == "Darwin" ]] || fail "This script must run on macOS."
command -v xcodebuild >/dev/null 2>&1 || fail "xcodebuild not found. Select Xcode with: sudo xcode-select --switch /Applications/Xcode.app/Contents/Developer"
command -v python3 >/dev/null 2>&1 || fail "python3 not found. Install Xcode Command Line Tools or Python 3."
[[ -f "$PROJECT_FILE" ]] || fail "Missing project file: $PROJECT_FILE"
[[ -e "$PROJECT_PATH" ]] || fail "Project/workspace path not found: $PROJECT_PATH"
[[ "$PROJECT_PATH" == *.xcworkspace || "$PROJECT_PATH" == *.xcodeproj ]] || fail "Project path must be .xcworkspace or .xcodeproj: $PROJECT_PATH"

cd "$REPO_ROOT"
mkdir -p "$LOG_DIR" "$REPO_ROOT/build"

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
if [[ -n "$CODE_SIGN_STYLE_VALUE" ]]; then
  SIGNING_BUILD_SETTINGS+=("CODE_SIGN_STYLE=$CODE_SIGN_STYLE_VALUE")
fi
if [[ -n "$DEVELOPMENT_TEAM_VALUE" ]]; then
  SIGNING_BUILD_SETTINGS+=("DEVELOPMENT_TEAM=$DEVELOPMENT_TEAM_VALUE")
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

info "Applying durable archive/linker build settings"
python3 "$REPO_ROOT/scripts/apply_ios_archive_linker_fix.py" "$PROJECT_FILE" --no-backup

if [[ "${LUMEN_CLEAN_DERIVED_DATA:-1}" == "1" ]]; then
  info "Cleaning Lumen DerivedData"
  clean_lumen_derived_data
fi

if [[ "${LUMEN_RESET_SWIFTPM_CACHE:-0}" == "1" ]]; then
  info "Cleaning SwiftPM cache"
  reset_swiftpm_cache
fi

info "Resolving Swift package dependencies"
run_logged "$LOG_DIR/resolve-packages-$TIMESTAMP.log" \
  xcodebuild \
    "${PROJECT_SELECTOR[@]}" \
    -scheme "$SCHEME" \
    -resolvePackageDependencies

info "Archiving with linker-safe Swift settings"
run_logged "$LOG_PATH" \
  xcodebuild \
    "${PROJECT_SELECTOR[@]}" \
    -scheme "$SCHEME" \
    -configuration "$CONFIGURATION" \
    -destination "generic/platform=iOS" \
    -archivePath "$ARCHIVE_PATH" \
    "${PROVISIONING_ARGS[@]}" \
    "${XCODE_AUTH_ARGS[@]}" \
    "${SIGNING_BUILD_SETTINGS[@]}" \
    COMPILER_INDEX_STORE_ENABLE=NO \
    ALWAYS_EMBED_SWIFT_STANDARD_LIBRARIES=YES \
    DEAD_CODE_STRIPPING=NO \
    SWIFT_COMPILATION_MODE=singlefile \
    SWIFT_WHOLE_MODULE_OPTIMIZATION=NO \
    SWIFT_OPTIMIZATION_LEVEL=-Osize \
    clean archive

bold "✅ Archive created: $ARCHIVE_PATH"
