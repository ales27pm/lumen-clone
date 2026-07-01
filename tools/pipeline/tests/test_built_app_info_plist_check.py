import plistlib
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "check_built_app_info_plist.py"


def _write_app(tmp_path: Path, info: dict) -> Path:
    app = tmp_path / "Lumen.app"
    app.mkdir()
    with (app / "Info.plist").open("wb") as handle:
        plistlib.dump(info, handle)
    return app


def test_built_app_info_plist_check_requires_alarmkit_usage_description(tmp_path: Path):
    app = _write_app(
        tmp_path,
        {
            "CFBundleIdentifier": "com.27pm.lumenclone",
            "CFBundleVersion": "42",
        },
    )

    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(app)],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    assert "missing NSAlarmKitUsageDescription" in result.stderr


def test_built_app_info_plist_check_accepts_signed_bundle_metadata(tmp_path: Path):
    app = _write_app(
        tmp_path,
        {
            "CFBundleIdentifier": "com.27pm.lumenclone",
            "CFBundleVersion": "42",
            "NSAlarmKitUsageDescription": "Lumen uses AlarmKit to schedule alarms.",
            "LumenBuildSourceIdentifier": "42",
            "LumenBuildConfiguration": "Debug",
            "LumenBuildScheme": "Lumen",
            "LumenGitSHA": "abc123",
        },
    )

    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(app)],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert "NSAlarmKitUsageDescription" in result.stdout
    assert "LumenGitSHA=abc123" in result.stdout
