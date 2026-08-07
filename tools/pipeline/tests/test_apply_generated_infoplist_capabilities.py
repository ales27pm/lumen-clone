import os
import plistlib
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "ios" / "Lumen" / "Scripts" / "apply_generated_infoplist_capabilities.sh"


def test_apply_generated_infoplist_capabilities_quotes_spaced_strings(tmp_path: Path):
    if not Path("/usr/libexec/PlistBuddy").exists():
        pytest.skip("PlistBuddy is only available on macOS")

    build_dir = tmp_path / "build"
    build_dir.mkdir()
    plist = build_dir / "Info.plist"
    with plist.open("wb") as handle:
        plistlib.dump({}, handle)

    env = os.environ.copy()
    env.update({
        "TARGET_BUILD_DIR": str(build_dir),
        "INFOPLIST_PATH": "Info.plist",
        "INFOPLIST_KEY_BGTaskSchedulerPermittedIdentifiers": "com.27pm.lumenclone.refresh",
        "PRODUCT_BUNDLE_IDENTIFIER": "com.27pm.lumenclone",
        "PRODUCT_MODULE_NAME": "Lumen",
        "CURRENT_PROJECT_VERSION": "42",
        "CONFIGURATION": "Debug",
        "LUMEN_BUILD_SCHEME": "Lumen",
        "LUMEN_GIT_SHA": "abc123",
    })

    result = subprocess.run(
        ["sh", str(SCRIPT)],
        text=True,
        capture_output=True,
        check=False,
        env=env,
    )

    assert result.returncode == 0, result.stderr
    with plist.open("rb") as handle:
        info = plistlib.load(handle)

    assert info["NSAlarmKitUsageDescription"] == (
        "Lumen uses AlarmKit to schedule prominent alarms and countdowns when you ask."
    )
    assert info["LumenBuildScheme"] == "Lumen"
    assert info["CFBundleURLTypes"][0]["CFBundleURLSchemes"] == ["msauth.com.27pm.lumenclone"]
    assert info["LSApplicationQueriesSchemes"] == ["msauth", "msauthv2", "msauthv3"]
