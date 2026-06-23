#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PROJECT="${PROJECT:-ios/Lumen.xcodeproj}"
SCHEME="${SCHEME:-Lumen}"
SIM_NAME="${SIM_NAME:-Lumen Focused Test iPhone}"
SIM_RUNTIME="${SIM_RUNTIME:-com.apple.CoreSimulator.SimRuntime.iOS-26-3}"
SIM_DEVICE_TYPE="${SIM_DEVICE_TYPE:-com.apple.CoreSimulator.SimDeviceType.iPhone-17}"
ONLY_TESTING="${ONLY_TESTING:-LumenTests/AgentGroundingRegressionTests}"
DERIVED_DATA_PATH="${DERIVED_DATA_PATH:-}"
REPAIR_CORE_SIMULATOR=0
EXTRA_XCODEBUILD_ARGS=()

usage() {
  cat <<'EOF'
Usage: scripts/run_focused_simulator_tests.sh [--repair-core-simulator] [--only-testing TEST_ID] [--] [xcodebuild args...]

Environment:
  SIM_UDID             Use an existing simulator UDID.
  SIM_NAME             Simulator name to reuse/create. Default: Lumen Focused Test iPhone
  SIM_RUNTIME          Runtime identifier. Default: com.apple.CoreSimulator.SimRuntime.iOS-26-3
  SIM_DEVICE_TYPE      Device type identifier. Default: com.apple.CoreSimulator.SimDeviceType.iPhone-17
  ONLY_TESTING         xcodebuild -only-testing value. Default: LumenTests/AgentGroundingRegressionTests
  DERIVED_DATA_PATH    Optional -derivedDataPath value.

The repair mode kills stale CoreSimulator and simulator runtime processes before booting.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repair-core-simulator)
      REPAIR_CORE_SIMULATOR=1
      shift
      ;;
    --only-testing)
      ONLY_TESTING="${2:?missing value for --only-testing}"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    --)
      shift
      EXTRA_XCODEBUILD_ARGS+=("$@")
      break
      ;;
    *)
      EXTRA_XCODEBUILD_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ "$REPAIR_CORE_SIMULATOR" == "1" ]]; then
  echo "== Repairing CoreSimulator service state =="
  pkill -9 -x Simulator 2>/dev/null || true
  pkill -9 -x launchd_sim 2>/dev/null || true
  pkill -9 -f '/Library/Developer/CoreSimulator/Volumes/' 2>/dev/null || true
  pkill -9 -f 'com.apple.CoreSimulator.CoreSimulatorService' 2>/dev/null || true
  sleep 5
fi

if [[ -n "${SIM_UDID:-}" ]]; then
  UDID="$SIM_UDID"
else
  UDID="$(SIM_NAME="$SIM_NAME" SIM_RUNTIME="$SIM_RUNTIME" /usr/bin/python3 <<'PY'
import json
import os
import subprocess

runtime = os.environ["SIM_RUNTIME"]
name = os.environ["SIM_NAME"]
devices = json.loads(subprocess.check_output(["xcrun", "simctl", "list", "devices", "--json"], text=True))
for device in devices.get("devices", {}).get(runtime, []):
    if device.get("name") == name and device.get("isAvailable", True):
        print(device["udid"])
        break
PY
)"
  if [[ -z "$UDID" ]]; then
    echo "== Creating simulator: $SIM_NAME =="
    UDID="$(xcrun simctl create "$SIM_NAME" "$SIM_DEVICE_TYPE" "$SIM_RUNTIME")"
  fi
fi

echo "== Simulator =="
xcrun simctl list devices | rg "$UDID|$SIM_NAME" || true

echo "== Booting simulator =="
echo "== Shutting down other booted simulators =="
while IFS= read -r other_udid; do
  [[ -z "$other_udid" || "$other_udid" == "$UDID" ]] && continue
  xcrun simctl shutdown "$other_udid" 2>/dev/null || true
done < <(xcrun simctl list devices --json | /usr/bin/python3 -c 'import json,sys
devices=json.load(sys.stdin).get("devices", {})
for runtime_devices in devices.values():
    for device in runtime_devices:
        if device.get("state") == "Booted":
            print(device.get("udid", ""))
')

xcrun simctl boot "$UDID" 2>/dev/null || true

# On some machines bootstatus can remain stuck at "Waiting on BackBoard" or
# "Waiting on System App" until the Simulator UI session is attached.
pkill -x Simulator 2>/dev/null || true
open -na Simulator --args -CurrentDeviceUDID "$UDID"
xcrun simctl bootstatus "$UDID" -b

echo "== Running focused tests =="
XCODEBUILD_ARGS=(
  test
  -project "$PROJECT"
  -scheme "$SCHEME"
  -destination "platform=iOS Simulator,id=$UDID"
  -only-testing:"$ONLY_TESTING"
  -enableCodeCoverage NO
  -parallel-testing-enabled NO
)

if [[ -n "$DERIVED_DATA_PATH" ]]; then
  XCODEBUILD_ARGS+=(-derivedDataPath "$DERIVED_DATA_PATH")
fi

if (( ${#EXTRA_XCODEBUILD_ARGS[@]} > 0 )); then
  XCODEBUILD_ARGS+=("${EXTRA_XCODEBUILD_ARGS[@]}")
fi

printf 'xcodebuild'
printf ' %q' "${XCODEBUILD_ARGS[@]}"
printf '\n'

xcodebuild "${XCODEBUILD_ARGS[@]}"
