#!/usr/bin/env bash
set -euo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SUPPORT="$SCRIPT_DIR/eas_native_ios_support.py"
MODE="${1:-build}"
DISTRIBUTION="${LUMEN_EAS_DISTRIBUTION:-store}"
case "$DISTRIBUTION" in
  store) CONFIGURATION=Release; SIGNING_STAGE=app-store; OPTIMIZATION=-Osize ;;
  internal) CONFIGURATION=Debug; SIGNING_STAGE=archive; OPTIMIZATION=-Onone ;;
  *) printf 'error: expected LUMEN_EAS_DISTRIBUTION=store or internal.\n' >&2; exit 2 ;;
esac
case "$MODE" in
  build|--preflight|--collect-recovery) ;;
  *) printf 'Usage: bash scripts/eas-native-ios.sh [--preflight|--collect-recovery]\n' >&2; exit 2 ;;
esac
if [[ "$MODE" == "--collect-recovery" ]]; then
  python3 "$SUPPORT" collect-recovery \
    --ipa "$REPO_ROOT/build/eas-export/Lumen.ipa" \
    --archive "$REPO_ROOT/build/eas-archive/Lumen.xcarchive" \
    --output "$REPO_ROOT/build/eas-evidence"
  exit 0
fi
[[ "$(uname -s)" == "Darwin" ]] || { printf 'error: native EAS iOS work requires macOS.\n' >&2; exit 1; }
for required in git python3 xcodebuild xcrun; do
  command -v "$required" >/dev/null || { printf 'error: missing Apple executor tool: %s\n' "$required" >&2; exit 1; }
done
cd "$REPO_ROOT"
EVIDENCE="$REPO_ROOT/build/eas-evidence"
PRIVATE="$REPO_ROOT/build/eas-private"
ARCHIVE="$REPO_ROOT/build/eas-archive/Lumen.xcarchive"
EXPORT="$REPO_ROOT/build/eas-export"
IPA="$EXPORT/Lumen.ipa"
mkdir -p "$EVIDENCE" "$PRIVATE"

run_logged() {
  local log="$1"
  shift
  set +e
  "$@" 2>&1 | tee "$log"
  local statuses=("${PIPESTATUS[@]}")
  set -e
  [[ ${statuses[0]} -eq 0 && ${statuses[1]} -eq 0 ]]
}

expected_revision="${EAS_BUILD_GIT_COMMIT_HASH:-${LUMEN_GIT_SHA:-}}"
candidate_args=(candidate --root "$REPO_ROOT" --distribution "$DISTRIBUTION" --output "$EVIDENCE/candidate.json")
if [[ -n "$expected_revision" ]]; then
  candidate_args+=(--expected-revision "$expected_revision")
fi
python3 "$SUPPORT" "${candidate_args[@]}"
field() { python3 "$SUPPORT" field --input "$1" --key "$2"; }
DEPLOYMENT_TARGET="$(field "$EVIDENCE/candidate.json" deploymentTarget)"
xcodebuild -version > "$EVIDENCE/xcode-version.txt"
xcrun swiftc --version > "$EVIDENCE/swift-version.txt"
SDK_VERSION="$(xcrun --sdk iphoneos --show-sdk-version)"
SDK_PATH="$(xcrun --sdk iphoneos --show-sdk-path)"
printf '%s\n' "$SDK_VERSION" > "$EVIDENCE/iphoneos-sdk-version.txt"
python3 "$SUPPORT" toolchain --xcode "$EVIDENCE/xcode-version.txt" --sdk "$SDK_VERSION" --deployment-target "$DEPLOYMENT_TARGET"

SOURCE_SHA="$(field "$EVIDENCE/candidate.json" sourceRevision)"
BUILD_NUMBER="$(field "$EVIDENCE/candidate.json" buildNumber)"
MARKETING_VERSION="$(field "$EVIDENCE/candidate.json" marketingVersion)"
cat > "$PRIVATE/CoreAIImport.swift" <<'SWIFT'
import CoreAI
import SwiftData
func requireCoreAI(_ model: AIModel, _ function: InferenceFunction, _ tensor: NDArray) {}
SWIFT
run_logged "$EVIDENCE/coreai-sdk-typecheck.log" \
  xcrun --sdk iphoneos swiftc -typecheck -sdk "$SDK_PATH" \
  -target "arm64-apple-ios$DEPLOYMENT_TARGET" \
  -module-cache-path "$PRIVATE/module-cache" "$PRIVATE/CoreAIImport.swift"
env "LUMEN_IOS_CONFIGURATION=$CONFIGURATION" "LUMEN_IOS_DISTRIBUTION=$DISTRIBUTION" \
  bash "$SCRIPT_DIR/archive_lumen_stable.sh" --preflight-release-provenance > "$EVIDENCE/source-preflight.log"
printf 'Native source and CoreAI SDK preflight passed for %s.\n' "$SOURCE_SHA"
if [[ "$MODE" == "--preflight" ]]; then
  exit 0
fi

command -v security >/dev/null || { printf 'error: macOS signing tools unavailable.\n' >&2; exit 1; }
[[ ! -e "$ARCHIVE" && ! -e "$EXPORT" ]] || { printf 'error: prior archive/export exists; inspect and reuse that artifact before starting another build.\n' >&2; exit 1; }
python3 "$SUPPORT" signing --root "$REPO_ROOT" --distribution "$DISTRIBUTION" --output "$PRIVATE/signing.json"
python3 "$SUPPORT" export-options --distribution "$DISTRIBUTION" --input "$PRIVATE/signing.json" --output "$PRIVATE/ExportOptions.plist"
python3 "$SUPPORT" signing-xcconfig --distribution "$DISTRIBUTION" --input "$PRIVATE/signing.json" --output "$PRIVATE/Signing.xcconfig"

run_logged "$EVIDENCE/archive.log" env \
  "LUMEN_ARCHIVE_PATH=$ARCHIVE" \
  LUMEN_IOS_PROJECT_PATH=ios/Lumen.xcodeproj \
  LUMEN_IOS_SCHEME=Lumen "LUMEN_IOS_CONFIGURATION=$CONFIGURATION" \
  "LUMEN_IOS_DISTRIBUTION=$DISTRIBUTION" "LUMEN_SWIFT_OPTIMIZATION_LEVEL=$OPTIMIZATION" \
  "LUMEN_IOS_CURRENT_PROJECT_VERSION=$BUILD_NUMBER" \
  "LUMEN_IOS_MARKETING_VERSION=$MARKETING_VERSION" \
  "LUMEN_GIT_SHA=$SOURCE_SHA" LUMEN_BUILD_SCHEME=Lumen \
  LUMEN_ALLOW_DIRTY_RELEASE_DIAGNOSTIC=0 \
  "XCODE_XCCONFIG_FILE=$PRIVATE/Signing.xcconfig" \
  LUMEN_IOS_CODE_SIGN_STYLE= LUMEN_IOS_CODE_SIGN_IDENTITY= \
  LUMEN_IOS_DEVELOPMENT_TEAM= LUMEN_IOS_PROVISIONING_PROFILE_SPECIFIER= \
  LUMEN_IOS_CLEAR_PROVISIONING_PROFILE_SPECIFIER=0 \
  LUMEN_IOS_ALLOW_PROVISIONING_UPDATES=0 \
  LUMEN_IOS_AUTHENTICATION_KEY_PATH= LUMEN_IOS_AUTHENTICATION_KEY_ID= LUMEN_IOS_AUTHENTICATION_KEY_ISSUER_ID= \
  LUMEN_CLEAN_DERIVED_DATA=0 LUMEN_CLEAN_GLOBAL_DERIVED_DATA=0 \
  LUMEN_RESET_SWIFTPM_CACHE=0 LUMEN_PACKAGE_RESOLVE_ATTEMPTS=1 \
  LUMEN_AGENT_GROUNDING_RESOURCE_MODE=minimal \
  bash "$SCRIPT_DIR/archive_lumen_stable.sh"

run_logged "$EVIDENCE/export.log" xcodebuild -exportArchive \
  -archivePath "$ARCHIVE" -exportPath "$EXPORT" \
  -exportOptionsPlist "$PRIVATE/ExportOptions.plist"
[[ -f "$EXPORT/Packaging.log" && -f "$IPA" ]] || { printf 'error: export did not produce its IPA and Packaging.log.\n' >&2; exit 1; }
python3 - "$EXPORT/Packaging.log" <<'PY'
import sys
from pathlib import Path
if "Upload Symbols Failed" in Path(sys.argv[1]).read_text(encoding="utf-8"):
    raise SystemExit("error: distribution export failed symbol packaging")
PY
cp "$EXPORT/Packaging.log" "$EVIDENCE/Packaging.log"
run_logged "$EVIDENCE/ipa-validation.log" python3 "$SCRIPT_DIR/check_built_app_info_plist.py" "$IPA" \
  --expected-bundle-identifier com.27pm.lumenclone \
  --expected-bundle-version "$BUILD_NUMBER" --expected-marketing-version "$MARKETING_VERSION" \
  --expected-build-configuration "$CONFIGURATION" --require-dsym-archive "$ARCHIVE"
run_logged "$EVIDENCE/ipa-signing-validation.log" python3 "$SCRIPT_DIR/validate_ios_signing_capabilities.py" \
  --project-file "$REPO_ROOT/ios/Lumen.xcodeproj/project.pbxproj" \
  --entitlements "$REPO_ROOT/ios/Lumen/Lumen.entitlements" \
  --app-store-entitlements "$REPO_ROOT/ios/Lumen/LumenAppStore.entitlements" \
  --signing-stage "$SIGNING_STAGE" --signed-app-path "$IPA"
if [[ "$DISTRIBUTION" == internal ]]; then
  run_logged "$EVIDENCE/ipa-internal-signing-validation.log" python3 "$SUPPORT" internal-signing \
    --distribution internal --input "$PRIVATE/signing.json" --ipa "$IPA"
fi
python3 "$SUPPORT" candidate --root "$REPO_ROOT" --distribution "$DISTRIBUTION" --expected-revision "$SOURCE_SHA" --output "$EVIDENCE/candidate-after.json"
cmp "$EVIDENCE/candidate.json" "$EVIDENCE/candidate-after.json"
python3 "$SUPPORT" artifact-receipt --root "$REPO_ROOT" --input "$EVIDENCE/candidate.json" \
  --ipa "$IPA" --archive "$ARCHIVE" --output "$EVIDENCE/artifact.json"
if [[ "$DISTRIBUTION" == store ]]; then
  printf 'Verified IPA ready for separate EAS submission: version %s (%s), source %s.\n' "$MARKETING_VERSION" "$BUILD_NUMBER" "$SOURCE_SHA"
else
  printf 'Verified internal Debug IPA ready for registered devices: version %s (%s), source %s. No store submission.\n' "$MARKETING_VERSION" "$BUILD_NUMBER" "$SOURCE_SHA"
fi
