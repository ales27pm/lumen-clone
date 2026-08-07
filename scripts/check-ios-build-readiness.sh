#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PROJECT="ios/Lumen.xcodeproj"
PBX="$PROJECT/project.pbxproj"
READINESS_RG_EXCLUDES=(
  --glob '!*.min.js'
  --glob '!generated/**'
  --glob '!ios/Lumen/AgentBehaviorManifest.json'
  --glob '!*.gguf'
  --glob '!*.mlmodel'
  --glob '!*.mlpackage/**'
  --glob '!*.mlmodelc/**'
)

echo "== Xcode availability =="
if command -v xcodebuild >/dev/null 2>&1; then
  xcodebuild -version
  echo "== Project schemes =="
  (cd ios && xcodebuild -list -project Lumen.xcodeproj)
else
  echo "xcodebuild unavailable; static readiness checks only."
fi

echo "== Project membership format =="
test -f "$PBX"
if rg -q "PBXFileSystemSynchronizedRootGroup" "$PBX"; then
  echo "Project uses file-system synchronized root groups for Lumen and LumenTests."
else
  echo "warning: project does not use synchronized root groups; manual source membership audit required." >&2
fi

echo "== Swift files present =="
find ios/Lumen -name "*.swift" -print | sort >/tmp/lumen_app_swift_files.txt
find ios/LumenTests -name "*.swift" -print | sort >/tmp/lumen_test_swift_files.txt
wc -l /tmp/lumen_app_swift_files.txt /tmp/lumen_test_swift_files.txt

echo "== iOS signing capability checks =="
python3 scripts/validate_ios_signing_capabilities.py

echo "== Generated JSONL artifact checks =="
python3 scripts/check-generated-jsonl-artifacts.py

echo "== Static privacy/build-hardening checks =="
if rg "${READINESS_RG_EXCLUDES[@]}" -n -e "TODO" -e "FIXME" -e "XXX" -e "stub" -e "not implemented" -e "not-implemented" -e "unimplemented" ios/Lumen; then
  echo "Found production unfinished markers in ios/Lumen. Remove them from Release source or guard them behind DEBUG." >&2
  exit 1
fi
if rg "${READINESS_RG_EXCLUDES[@]}" -n -e "placeholder" ios/LumenTests docs; then
  echo "Found placeholder markers in tests/docs. Review above; some may be literal test/prompt text or historical notes." >&2
fi
if ! rg "${READINESS_RG_EXCLUDES[@]}" -n "import AppIntents|AppIntent|AppShortcutsProvider" ios/Lumen/AppIntents >/dev/null; then
  echo "warning: no AppIntents references found under ios/Lumen/AppIntents." >&2
fi
rg "${READINESS_RG_EXCLUDES[@]}" -n "FoundationModels|@available|canImport\(FoundationModels\)" ios/Lumen >/dev/null || true
if ! rg "${READINESS_RG_EXCLUDES[@]}" -n "NSMicrophoneUsageDescription|NSSpeechRecognitionUsageDescription|NSCalendars|NSContactsUsageDescription|NSLocationWhenInUseUsageDescription|BGTaskSchedulerPermittedIdentifiers" ios >/dev/null; then
  echo "warning: expected usage string or BGTask identifiers were not found in static scan." >&2
fi
if rg "${READINESS_RG_EXCLUDES[@]}" -n "OSLog|Logger" ios/Lumen/AppIntents ios/Lumen/Voice ios/Lumen/Diagnostics; then
  echo "Found logging APIs in privacy-sensitive additions; review output above." >&2
else
  true
fi

echo "== Built app artifact checks =="
if [[ -z "${LUMEN_BUILT_APP_PATH:-}" ]]; then
  if [[ "${LUMEN_REQUIRE_BUILT_APP:-0}" == "1" ]]; then
    echo "error: LUMEN_BUILT_APP_PATH is required for final signed Info.plist metadata validation." >&2
    exit 1
  fi
  echo "No explicit built app artifact provided; skipping signed Info.plist metadata validation. Set LUMEN_BUILT_APP_PATH or LUMEN_REQUIRE_BUILT_APP=1 to make this mandatory."
else
  case "$LUMEN_BUILT_APP_PATH" in
    *.xcarchive) signing_stage="archive" ;;
    *.ipa) signing_stage="app-store" ;;
    *.app)
      signing_stage="${LUMEN_BUILT_APP_SIGNING_STAGE:-}"
      case "$signing_stage" in
        archive|app-store) ;;
        *)
          echo "error: Set LUMEN_BUILT_APP_SIGNING_STAGE=archive|app-store for an ambiguous .app path." >&2
          exit 1
          ;;
      esac
      ;;
    *)
      echo "error: Unsupported LUMEN_BUILT_APP_PATH signing artifact: $LUMEN_BUILT_APP_PATH" >&2
      exit 1
      ;;
  esac
  python3 scripts/check_built_app_info_plist.py "$LUMEN_BUILT_APP_PATH"
  python3 scripts/validate_ios_signing_capabilities.py \
    --signing-stage "$signing_stage" \
    --signed-app-path "$LUMEN_BUILT_APP_PATH"
fi

echo "Build readiness static checks completed. Run xcodebuild on macOS for compile validation."
