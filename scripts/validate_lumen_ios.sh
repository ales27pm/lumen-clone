#!/usr/bin/env bash
set -euo pipefail

if [[ ! -d .git ]]; then
  echo "Error: run this script from the repository root." >&2
  exit 1
fi

DESTINATION="${DESTINATION:-generic/platform=iOS Simulator}"
PROJECT="ios/Lumen.xcodeproj"
SCHEME="Lumen"
RUN_SIMULATOR_TESTS="${RUN_SIMULATOR_TESTS:-0}"
TEST_TIMEOUT_SECONDS="${TEST_TIMEOUT_SECONDS:-900}"
STRICT_TASK_DETACHED_SCAN="${STRICT_TASK_DETACHED_SCAN:-0}"
AGENT_GROUNDING_RESOURCE_MODE="${AGENT_GROUNDING_RESOURCE_MODE:-minimal}"

run_with_timeout() {
  local timeout_seconds="$1"
  shift

  /usr/bin/python3 - "$timeout_seconds" "$@" <<'PY'
import subprocess
import sys

timeout_seconds = int(sys.argv[1])
command = sys.argv[2:]

try:
    completed = subprocess.run(command, timeout=timeout_seconds)
except subprocess.TimeoutExpired:
    print(
        f"error: command timed out after {timeout_seconds}s: "
        + " ".join(command),
        file=sys.stderr,
    )
    sys.exit(124)

sys.exit(completed.returncode)
PY
}

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
  if [[ "$STRICT_TASK_DETACHED_SCAN" == "1" ]]; then
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
  else
    echo "Task.detached scan is advisory. Set STRICT_TASK_DETACHED_SCAN=1 to enforce the legacy allowlist."
  fi
fi

echo "== xcodebuild build-for-testing =="
xcodebuild build-for-testing \
  -project "$PROJECT" \
  -scheme "$SCHEME" \
  -destination "$DESTINATION" \
  -parallel-testing-enabled NO \
  -jobs 2 \
  "AGENT_GROUNDING_RESOURCE_MODE=$AGENT_GROUNDING_RESOURCE_MODE"

if [[ "$RUN_SIMULATOR_TESTS" == "1" ]]; then
  echo "== xcodebuild test =="
  echo "Simulator execution is bounded by TEST_TIMEOUT_SECONDS=${TEST_TIMEOUT_SECONDS}."
  run_with_timeout "$TEST_TIMEOUT_SECONDS" \
    xcodebuild test \
      -project "$PROJECT" \
      -scheme "$SCHEME" \
      -destination "$DESTINATION" \
      -parallel-testing-enabled NO \
      -jobs 2 \
      "AGENT_GROUNDING_RESOURCE_MODE=$AGENT_GROUNDING_RESOURCE_MODE"
else
  echo "== xcodebuild test skipped =="
  echo "Set RUN_SIMULATOR_TESTS=1 to run bounded simulator XCTest execution."
fi
