#!/usr/bin/env bash
set -euo pipefail

if [[ ! -d .git ]]; then
  echo "Error: run this script from the repository root." >&2
  exit 1
fi

DESTINATION="${DESTINATION:-platform=iOS Simulator,name=iPhone 16}"
PROJECT="ios/Lumen.xcodeproj"
SCHEME="Lumen"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "Error: macOS is required to run iOS xcodebuild validation." >&2
  exit 1
fi

if ! command -v xcodebuild >/dev/null 2>&1; then
  echo "Error: xcodebuild is not available in PATH." >&2
  exit 1
fi

echo "== Conflict marker scan =="
if rg "<<<<<<<|=======|>>>>>>>" ios; then
  echo "Error: conflict markers found in ios sources." >&2
  exit 1
fi

echo "== try! scan =="
if rg "try!" ios/Lumen; then
  echo "Error: try! found in app source." >&2
  exit 1
fi

echo "== Task.detached scan =="
task_detached_hits="$(rg -n "Task\\.detached" ios/Lumen ios/LumenTests || true)"
if [[ -n "${task_detached_hits}" ]]; then
  echo "${task_detached_hits}"
  unexpected="$(
    printf "%s\n" "${task_detached_hits}" \
      | grep -v "^ios/Lumen/Services/RemCycleService.swift:24:.*Task\.detached(priority: \.utility)" \
      | grep -v "^ios/Lumen/Services/RemCycleService.swift:27:.*Task\.detached(priority: \.utility)" \
      | grep -v "^ios/Lumen/Views/SettingsView.swift:336:.*Task\.detached(priority: \.utility)" \
      || true
  )"
  if [[ -n "${unexpected}" ]]; then
    echo "${unexpected}"
    echo "Error: unexpected Task.detached usage found outside allowlist." >&2
    exit 1
  fi
fi

echo "== xcodebuild build-for-testing =="
xcodebuild build-for-testing -project "$PROJECT" -scheme "$SCHEME" -destination "$DESTINATION"

echo "== xcodebuild test =="
xcodebuild test -project "$PROJECT" -scheme "$SCHEME" -destination "$DESTINATION"
