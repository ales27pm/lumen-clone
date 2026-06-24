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
DERIVED_DATA_PATH="${DERIVED_DATA_PATH:-$ROOT/build/DerivedData-FocusedSimulatorTests}"
PREBOOT_SIMULATOR="${PREBOOT_SIMULATOR:-1}"
TEST_TIMEOUT_SECONDS="${TEST_TIMEOUT_SECONDS:-900}"
SIM_BOOT_TIMEOUT_SECONDS="${SIM_BOOT_TIMEOUT_SECONDS:-1200}"
SIM_BOOTSTATUS_POLL_SECONDS="${SIM_BOOTSTATUS_POLL_SECONDS:-60}"
SIM_READY_PROBE_TIMEOUT_SECONDS="${SIM_READY_PROBE_TIMEOUT_SECONDS:-20}"
SIMCTL_LIST_TIMEOUT_SECONDS="${SIMCTL_LIST_TIMEOUT_SECONDS:-30}"
AGENT_GROUNDING_RESOURCE_MODE="${AGENT_GROUNDING_RESOURCE_MODE:-minimal}"
APP_BUNDLE_ID="${APP_BUNDLE_ID:-com.27pm.lumenclone}"
UNINSTALL_BEFORE_TEST="${UNINSTALL_BEFORE_TEST:-1}"
USE_DISPOSABLE_SIMULATOR="${USE_DISPOSABLE_SIMULATOR:-0}"
DELETE_CREATED_SIMULATOR="${DELETE_CREATED_SIMULATOR:-1}"
SHUTDOWN_OTHER_SIMULATORS="${SHUTDOWN_OTHER_SIMULATORS:-0}"
ATTACH_SIMULATOR_UI="${ATTACH_SIMULATOR_UI:-1}"
PREWARM_ONLY="${PREWARM_ONLY:-0}"
PRINT_MOBILEINSTALLATION_ON_FAILURE="${PRINT_MOBILEINSTALLATION_ON_FAILURE:-0}"
REPAIR_CORE_SIMULATOR=0
EXTRA_XCODEBUILD_ARGS=()

usage() {
  cat <<'EOF'
Usage: scripts/run_focused_simulator_tests.sh [--repair-core-simulator] [--only-testing TEST_ID] [--] [xcodebuild args...]

Environment:
  SIM_UDID             Use an existing simulator UDID. Disables disposable simulator creation.
  SIM_NAME             Simulator name to reuse/create. Default: Lumen Focused Test iPhone
  SIM_RUNTIME          Runtime identifier. Default: com.apple.CoreSimulator.SimRuntime.iOS-26-3
  SIM_DEVICE_TYPE      Device type identifier. Default: com.apple.CoreSimulator.SimDeviceType.iPhone-17
  ONLY_TESTING         xcodebuild -only-testing value. Default: LumenTests/AgentGroundingRegressionTests
  DERIVED_DATA_PATH    DerivedData path. Default: build/DerivedData-FocusedSimulatorTests
  PREBOOT_SIMULATOR    Set to 1 to boot before testing. Default: 1.
  TEST_TIMEOUT_SECONDS Timeout for simulator test execution. Default: 900.
  SIM_BOOT_TIMEOUT_SECONDS
                       Timeout for simulator bootstatus. Default: 1200.
  SIM_BOOTSTATUS_POLL_SECONDS
                       Poll interval for bootstatus before probing readiness. Default: 60.
  SIM_READY_PROBE_TIMEOUT_SECONDS
                       Timeout for alternate SpringBoard/backboardd readiness probe. Default: 20.
  SIMCTL_LIST_TIMEOUT_SECONDS
                       Timeout for informational simctl list calls. Default: 30.
  AGENT_GROUNDING_RESOURCE_MODE
                       full, minimal, or skip for AgentGrounding resources. Default: minimal.
  APP_BUNDLE_ID        App bundle to uninstall before tests. Default: com.27pm.lumenclone.
  UNINSTALL_BEFORE_TEST
                       Set to 0 to keep the existing simulator app install. Default: 1.
  USE_DISPOSABLE_SIMULATOR
                       Set to 1 to create a fresh device for this run. Default: 0.
  DELETE_CREATED_SIMULATOR
                       Set to 0 to leave disposable simulators behind after the run. Default: 1.
  SHUTDOWN_OTHER_SIMULATORS
                       Set to 1 to shut down other booted devices before testing. Default: 0.
  ATTACH_SIMULATOR_UI  Open Simulator.app before bootstatus. Default: 1.
  PREWARM_ONLY         Boot and warm the simulator, then exit before building. Default: 0.
  PRINT_MOBILEINSTALLATION_ON_FAILURE
                       Print MobileInstallation logs for non-timeout test failures. Default: 0.

The repair mode kills stale CoreSimulator and simulator runtime processes before booting.
EOF
}

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

print_mobileinstallation_diagnostics() {
  local udid="$1"
  local log_file="$HOME/Library/Logs/CoreSimulator/$udid/MobileInstallation/mobile_installation.log.0"

  if [[ ! -f "$log_file" ]]; then
    echo "No MobileInstallation log found for simulator $udid."
    return 0
  fi

  echo "== MobileInstallation diagnostics =="
  echo "Log: $log_file"
  rg "Data Migration Failed|Installing <MIInstallable|Install Successful|Install successful|Preflight/Patch|Info.plist missing" "$log_file" \
    | tail -n 60 \
    || true
}

attach_simulator_ui() {
  local udid="$1"

  if [[ "$ATTACH_SIMULATOR_UI" != "1" ]]; then
    return 0
  fi

  echo "== Attaching Simulator.app UI session =="
  open -na Simulator --args -CurrentDeviceUDID "$udid" 2>/dev/null || true
}

probe_simulator_ready() {
  local udid="$1"

  /usr/bin/python3 - "$udid" "$SIM_READY_PROBE_TIMEOUT_SECONDS" <<'PY'
import json
import subprocess
import sys

udid = sys.argv[1]
timeout_seconds = int(sys.argv[2])

try:
    listed = subprocess.run(
        ["xcrun", "simctl", "list", "devices", "--json"],
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )
except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
    raise SystemExit(1)

device_state = None
for devices in json.loads(listed.stdout).get("devices", {}).values():
    for device in devices:
        if device.get("udid") == udid:
            device_state = device.get("state")
            break
    if device_state:
        break

if device_state != "Booted":
    raise SystemExit(1)

for service in ("system/com.apple.SpringBoard", "system/com.apple.backboardd"):
    try:
        service_state = subprocess.run(
            ["xcrun", "simctl", "spawn", udid, "launchctl", "print", service],
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        raise SystemExit(1)
    if "state = running" not in service_state.stdout:
        raise SystemExit(1)

raise SystemExit(0)
PY
}

wait_for_simulator_ready() {
  local udid="$1"
  local deadline=$((SECONDS + SIM_BOOT_TIMEOUT_SECONDS))
  local poll_seconds
  local boot_status

  while (( SECONDS < deadline )); do
    poll_seconds="$SIM_BOOTSTATUS_POLL_SECONDS"
    if (( SECONDS + poll_seconds > deadline )); then
      poll_seconds=$((deadline - SECONDS))
    fi
    if (( poll_seconds <= 0 )); then
      break
    fi

    set +e
    run_with_timeout "$poll_seconds" xcrun simctl bootstatus "$udid" -b
    boot_status=$?
    set -e
    if [[ "$boot_status" == "0" ]]; then
      return 0
    fi

    if probe_simulator_ready "$udid"; then
      echo "== Simulator readiness probe succeeded =="
      echo "bootstatus did not reach terminal readiness, but the device is Booted and SpringBoard/backboardd are running."
      return 0
    fi
  done

  return 124
}

latest_xctestrun() {
  local products_dir="$DERIVED_DATA_PATH/Build/Products"
  [[ -d "$products_dir" ]] || return 1
  find "$products_dir" -name '*.xctestrun' ! -name '*.focused.xctestrun' -type f -print0 \
    | xargs -0 stat -f '%m %N' 2>/dev/null \
    | sort -nr \
    | head -1 \
    | cut -d' ' -f2-
}

create_simulator() {
  local udid=""
  local device_type
  local fallback_types=(
    "$SIM_DEVICE_TYPE"
    "com.apple.CoreSimulator.SimDeviceType.iPhone-16"
    "com.apple.CoreSimulator.SimDeviceType.iPhone-15"
    "com.apple.CoreSimulator.SimDeviceType.iPhone-14"
  )
  local tried_types=" "

  for device_type in "${fallback_types[@]}"; do
    [[ -z "$device_type" ]] && continue
    if [[ "$tried_types" == *" $device_type "* ]]; then
      continue
    fi
    tried_types+="$device_type "
    if udid="$(xcrun simctl create "$SIM_NAME" "$device_type" "$SIM_RUNTIME" 2>/dev/null)"; then
      printf '%s\n' "$udid"
      return 0
    fi
  done

  return 1
}

prepare_focused_xctestrun() {
  local source_path="$1"
  local destination_path="$2"
  local target_name="$3"

  /usr/bin/python3 - "$source_path" "$destination_path" "$target_name" <<'PY'
import plistlib
import sys
from pathlib import Path

source = Path(sys.argv[1])
destination = Path(sys.argv[2])
target_name = sys.argv[3]

with source.open("rb") as handle:
    data = plistlib.load(handle)

kept = 0
for configuration in data.get("TestConfigurations", []):
    targets = configuration.get("TestTargets", [])
    focused_targets = [
        target for target in targets
        if target.get("BlueprintName") == target_name
        or target.get("ProductModuleName") == target_name
    ]
    if not focused_targets:
        continue
    for target in focused_targets:
        target["ParallelizationEnabled"] = False
        if isinstance(target.get("DependentProductPaths"), list):
            target["DependentProductPaths"] = [
                value for value in target["DependentProductPaths"]
                if "UITests" not in str(value)
                and "XCTRunner" not in str(value)
                and "-Runner.app" not in str(value)
            ]
        if isinstance(target.get("BundleIdentifiersForCrashReportEmphasis"), list):
            target["BundleIdentifiersForCrashReportEmphasis"] = [
                value for value in target["BundleIdentifiersForCrashReportEmphasis"]
                if "uitest" not in str(value).lower()
            ]
    configuration["TestTargets"] = focused_targets
    kept += len(focused_targets)

if kept == 0:
    raise SystemExit(f"{source} does not contain test target {target_name}")

destination.parent.mkdir(parents=True, exist_ok=True)
with destination.open("wb") as handle:
    plistlib.dump(data, handle, sort_keys=False)
PY
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

CREATED_SIMULATOR_UDID=""
cleanup_created_simulator() {
  if [[ -n "$CREATED_SIMULATOR_UDID" && "$DELETE_CREATED_SIMULATOR" == "1" ]]; then
    echo "== Deleting disposable simulator =="
    run_with_timeout "$SIMCTL_LIST_TIMEOUT_SECONDS" xcrun simctl shutdown "$CREATED_SIMULATOR_UDID" 2>/dev/null || true
    run_with_timeout "$SIMCTL_LIST_TIMEOUT_SECONDS" xcrun simctl delete "$CREATED_SIMULATOR_UDID" 2>/dev/null || true
  fi
}
trap cleanup_created_simulator EXIT

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
elif [[ "$USE_DISPOSABLE_SIMULATOR" == "1" ]]; then
  SIM_NAME="${SIM_NAME} $(date +%Y%m%d%H%M%S)"
  echo "== Creating disposable simulator: $SIM_NAME =="
  if ! UDID="$(create_simulator)"; then
    echo "Error: failed to create simulator using runtime $SIM_RUNTIME." >&2
    echo "Override SIM_RUNTIME or SIM_DEVICE_TYPE if this Xcode install has different identifiers." >&2
    exit 1
  fi
  CREATED_SIMULATOR_UDID="$UDID"
else
  UDID="$(SIM_NAME="$SIM_NAME" SIM_RUNTIME="$SIM_RUNTIME" /usr/bin/python3 <<'PY'
import json
import os
import subprocess
import sys

runtime = os.environ["SIM_RUNTIME"]
name = os.environ["SIM_NAME"]
try:
    completed = subprocess.run(
        ["xcrun", "simctl", "list", "devices", "--json"],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
except subprocess.TimeoutExpired:
    print("simctl list devices timed out while looking for reusable simulator", file=sys.stderr)
    raise SystemExit(124)

devices = json.loads(completed.stdout)
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
if ! run_with_timeout "$SIMCTL_LIST_TIMEOUT_SECONDS" xcrun simctl list devices | rg "$UDID|$SIM_NAME"; then
  echo "simctl list devices did not return a matching row within ${SIMCTL_LIST_TIMEOUT_SECONDS}s; continuing with UDID $UDID."
fi

if [[ "$SHUTDOWN_OTHER_SIMULATORS" == "1" ]]; then
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
fi

if [[ "$PREBOOT_SIMULATOR" == "1" ]]; then
  echo "== Booting simulator =="
  xcrun simctl boot "$UDID" 2>/dev/null || true
  attach_simulator_ui "$UDID"
  if ! wait_for_simulator_ready "$UDID"; then
    print_mobileinstallation_diagnostics "$UDID"
    exit 124
  fi
else
  echo "== Simulator preboot skipped =="
  echo "xcodebuild will own simulator boot and teardown for this run."
fi

if [[ "$PREWARM_ONLY" == "1" ]]; then
  echo "== Simulator prewarm complete =="
  echo "UDID: $UDID"
  exit 0
fi

COMMON_XCODEBUILD_ARGS=(
  -project "$PROJECT"
  -scheme "$SCHEME"
  -destination "platform=iOS Simulator,id=$UDID"
  -enableCodeCoverage NO
  -parallel-testing-enabled NO
  -parallel-testing-worker-count 1
  -maximum-concurrent-test-simulator-destinations 1
  -jobs 2
  -derivedDataPath "$DERIVED_DATA_PATH"
  "AGENT_GROUNDING_RESOURCE_MODE=$AGENT_GROUNDING_RESOURCE_MODE"
)

if (( ${#EXTRA_XCODEBUILD_ARGS[@]} > 0 )); then
  COMMON_XCODEBUILD_ARGS+=("${EXTRA_XCODEBUILD_ARGS[@]}")
fi

BUILD_ARGS=(build-for-testing "${COMMON_XCODEBUILD_ARGS[@]}")

echo "== Building focused test bundle =="
printf 'xcodebuild'
printf ' %q' "${BUILD_ARGS[@]}"
printf '\n'
xcodebuild "${BUILD_ARGS[@]}"

XCTESTRUN_PATH="$(latest_xctestrun)"
if [[ -z "$XCTESTRUN_PATH" ]]; then
  echo "Error: build-for-testing did not produce an .xctestrun file under $DERIVED_DATA_PATH/Build/Products." >&2
  exit 1
fi

ONLY_TESTING_TARGET="${ONLY_TESTING%%/*}"
FOCUSED_XCTESTRUN_PATH="$DERIVED_DATA_PATH/Build/Products/${SCHEME}-${ONLY_TESTING_TARGET}.focused.xctestrun"

echo "== Preparing focused xctestrun =="
echo "Source: $XCTESTRUN_PATH"
echo "Target: $ONLY_TESTING_TARGET"
prepare_focused_xctestrun "$XCTESTRUN_PATH" "$FOCUSED_XCTESTRUN_PATH" "$ONLY_TESTING_TARGET"

if [[ "$UNINSTALL_BEFORE_TEST" == "1" && -z "$CREATED_SIMULATOR_UDID" ]]; then
  echo "== Removing stale simulator app install =="
  xcrun simctl uninstall "$UDID" "$APP_BUNDLE_ID" 2>/dev/null || true
elif [[ -n "$CREATED_SIMULATOR_UDID" ]]; then
  echo "== Stale app uninstall skipped for disposable simulator =="
fi

TEST_ARGS=(
  test-without-building
  -xctestrun "$FOCUSED_XCTESTRUN_PATH"
  -destination "platform=iOS Simulator,id=$UDID"
  -only-testing:"$ONLY_TESTING"
  -parallel-testing-enabled NO
  -parallel-testing-worker-count 1
  -maximum-concurrent-test-simulator-destinations 1
)

echo "== Running focused tests =="
echo "Simulator execution is bounded by TEST_TIMEOUT_SECONDS=${TEST_TIMEOUT_SECONDS}."
printf 'xcodebuild'
printf ' %q' "${TEST_ARGS[@]}"
printf '\n'
set +e
run_with_timeout "$TEST_TIMEOUT_SECONDS" xcodebuild "${TEST_ARGS[@]}"
test_status=$?
set -e
if [[ "$test_status" != "0" ]]; then
  if [[ "$test_status" == "124" || "$PRINT_MOBILEINSTALLATION_ON_FAILURE" == "1" ]]; then
    print_mobileinstallation_diagnostics "$UDID"
  fi
  exit "$test_status"
fi
