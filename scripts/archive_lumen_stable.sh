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
SWIFT_OPTIMIZATION_LEVEL_VALUE="${LUMEN_SWIFT_OPTIMIZATION_LEVEL:--Onone}"
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
ACTOOL_SHIM_DIR="$REPO_ROOT/build/actool-shim"
ACTOOL_SHIM_PATH="$ACTOOL_SHIM_DIR/actool"
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

bold() { printf "\033[1m%s\033[0m\n" "$1"; }
info() { printf "\n➡️  %s\n" "$1"; }
warn() { printf "\n⚠️  %s\n" "$1"; }
fail() { printf "\n❌ %s\n" "$1"; exit 1; }

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
        -resolvePackageDependencies; then
      return 0
    fi

    if (( attempt >= attempts )); then
      return 1
    fi

    attempt=$((attempt + 1))
  done
}

[[ "$(uname -s)" == "Darwin" ]] || fail "This script must run on macOS."
command -v xcodebuild >/dev/null 2>&1 || fail "xcodebuild not found. Select Xcode with: sudo xcode-select --switch /Applications/Xcode.app/Contents/Developer"
command -v python3 >/dev/null 2>&1 || fail "python3 not found. Install Xcode Command Line Tools or Python 3."
[[ -f "$PROJECT_FILE" ]] || fail "Missing project file: $PROJECT_FILE"
[[ -e "$PROJECT_PATH" ]] || fail "Project/workspace path not found: $PROJECT_PATH"
[[ "$PROJECT_PATH" == *.xcworkspace || "$PROJECT_PATH" == *.xcodeproj ]] || fail "Project path must be .xcworkspace or .xcodeproj: $PROJECT_PATH"

cd "$REPO_ROOT"
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
if [[ -z "$EFFECTIVE_CODE_SIGN_IDENTITY_VALUE" && "$CODE_SIGN_STYLE_VALUE" == "Automatic" ]]; then
  EFFECTIVE_CODE_SIGN_IDENTITY_VALUE="Apple Development"
fi
if [[ -n "$CODE_SIGN_IDENTITY_VALUE" ]]; then
  info "Using codesigning identity override: $CODE_SIGN_IDENTITY_VALUE"
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
resolve_package_dependencies

info "Archiving with linker-safe Swift settings"
mkdir -p "$DERIVED_DATA_PATH" "$CLONED_SOURCE_PACKAGES_DIR" "$PACKAGE_CACHE_PATH" "$ACTOOL_SHIM_DIR"
ln -sf "$REPO_ROOT/scripts/lumen_actool_cached_assets.sh" "$ACTOOL_SHIM_PATH"
ARCHIVE_COMMAND=(
  xcodebuild
  "${PROJECT_SELECTOR[@]}"
  "${SIGNING_XCCONFIG_ARGS[@]}"
  -scheme "$SCHEME"
  -derivedDataPath "$DERIVED_DATA_PATH"
  -clonedSourcePackagesDirPath "$CLONED_SOURCE_PACKAGES_DIR"
  -packageCachePath "$PACKAGE_CACHE_PATH"
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
  ASSETCATALOG_COMPILER_ENABLE_ON_DEMAND_RESOURCES=NO
  ASSETCATALOG_COMPILER_SKIP_APP_STORE_DEPLOYMENT=YES
  ASSETCATALOG_COMPILER_COMPRESS_PNGS=NO
  ASSETCATALOG_COMPILER_OPTIMIZATION=time
  ASSETCATALOG_COMPILER_STANDALONE_ICON_BEHAVIOR=none
  "ASSETCATALOG_EXEC=$ACTOOL_SHIM_PATH"
  DEAD_CODE_STRIPPING=NO
  SWIFT_COMPILATION_MODE=singlefile
  SWIFT_WHOLE_MODULE_OPTIMIZATION=NO
  "SWIFT_OPTIMIZATION_LEVEL=$SWIFT_OPTIMIZATION_LEVEL_VALUE"
  clean
  archive
)
run_logged "$LOG_PATH" "${ARCHIVE_COMMAND[@]}"

bold "✅ Archive created: $ARCHIVE_PATH"
