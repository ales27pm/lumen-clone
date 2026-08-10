import os
import plistlib
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "ios" / "Lumen" / "Scripts" / "apply_generated_infoplist_capabilities.sh"
SOURCE_ATTESTATION = (
    ROOT
    / "tools"
    / "lumen_manifest_crawler"
    / "lumen_manifest_crawler"
    / "source_integrity.py"
)


def _source_attestation() -> tuple[str, bool]:
    completed = subprocess.run(
        [sys.executable, str(SOURCE_ATTESTATION), "--root", str(ROOT)],
        text=True,
        capture_output=True,
        check=True,
    )
    digest, dirty = completed.stdout.strip().split()
    return digest, dirty == "true"


def _environment(build_dir: Path, *, configuration: str) -> dict[str, str]:
    env = os.environ.copy()
    env.update({
        "TARGET_BUILD_DIR": str(build_dir),
        "INFOPLIST_PATH": "Info.plist",
        "INFOPLIST_KEY_BGTaskSchedulerPermittedIdentifiers": "com.27pm.lumenclone.refresh",
        "PRODUCT_BUNDLE_IDENTIFIER": "com.27pm.lumenclone",
        "PRODUCT_MODULE_NAME": "Lumen",
        "CURRENT_PROJECT_VERSION": "42",
        "CONFIGURATION": configuration,
        "LUMEN_BUILD_SCHEME": "Lumen",
        "LUMEN_GIT_SHA": "abc123" if configuration == "Debug" else "HEAD",
        "SRCROOT": str(ROOT / "ios"),
        "PROJECT_DIR": str(ROOT / "ios"),
    })
    return env


def _run(build_dir: Path, *, configuration: str, revision: str | None = None):
    env = _environment(build_dir, configuration=configuration)
    if revision is not None:
        env["LUMEN_GIT_SHA"] = revision
    return subprocess.run(
        ["sh", str(SCRIPT)],
        text=True,
        capture_output=True,
        check=False,
        env=env,
    )


def test_apply_generated_infoplist_capabilities_gates_debug_file_sharing(tmp_path: Path):
    if not Path("/usr/libexec/PlistBuddy").exists():
        pytest.skip("PlistBuddy is only available on macOS")

    build_dir = tmp_path / "build"
    build_dir.mkdir()
    plist = build_dir / "Info.plist"
    with plist.open("wb") as handle:
        plistlib.dump({}, handle)

    result = _run(build_dir, configuration="Debug")

    assert result.returncode == 0, result.stderr
    with plist.open("rb") as handle:
        info = plistlib.load(handle)

    assert info["NSAlarmKitUsageDescription"] == (
        "Lumen uses AlarmKit to schedule prominent alarms and countdowns when you ask."
    )
    assert info["LumenBuildScheme"] == "Lumen"
    assert info["LumenGitSHA"] == "abc123"
    expected_digest, expected_dirty = _source_attestation()
    assert info["LumenWorkingTreeDigest"] == expected_digest
    assert info["LumenSourceDirtyState"] is expected_dirty
    assert info["UIFileSharingEnabled"] is True
    assert info["CFBundleURLTypes"][0]["CFBundleURLSchemes"] == ["msauth.com.27pm.lumenclone"]
    assert info["LSApplicationQueriesSchemes"] == ["msauth", "msauthv2", "msauthv3"]


def test_apply_generated_infoplist_capabilities_disables_release_file_sharing_and_stamps_full_revision(tmp_path: Path):
    if not Path("/usr/libexec/PlistBuddy").exists():
        pytest.skip("PlistBuddy is only available on macOS")

    build_dir = tmp_path / "build"
    build_dir.mkdir()
    plist = build_dir / "Info.plist"
    with plist.open("wb") as handle:
        plistlib.dump({"UIFileSharingEnabled": True}, handle)

    short_revision = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    result = _run(
        build_dir,
        configuration="Release",
        revision=short_revision,
    )

    assert result.returncode == 0, result.stderr
    with plist.open("rb") as handle:
        info = plistlib.load(handle)
    expected_revision = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    assert info["UIFileSharingEnabled"] is False
    assert info["LumenGitSHA"] == expected_revision
    assert len(info["LumenGitSHA"]) in {40, 64}
    expected_digest, expected_dirty = _source_attestation()
    assert info["LumenWorkingTreeDigest"] == expected_digest
    assert info["LumenSourceDirtyState"] is expected_dirty


def test_apply_generated_infoplist_capabilities_fails_missing_release_plist(tmp_path: Path):
    result = _run(tmp_path, configuration="Release")

    assert result.returncode == 1
    assert "Release build is missing generated Info.plist" in result.stderr


def test_apply_generated_infoplist_capabilities_fails_unresolvable_release_revision(tmp_path: Path):
    build_dir = tmp_path / "build"
    build_dir.mkdir()
    plist = build_dir / "Info.plist"
    with plist.open("wb") as handle:
        plistlib.dump({}, handle)

    result = _run(
        build_dir,
        configuration="Release",
        revision="definitely-not-a-git-revision",
    )

    assert result.returncode == 1
    assert "requires a resolvable full Git source revision" in result.stderr
    with plist.open("rb") as handle:
        info = plistlib.load(handle)
    assert "LumenGitSHA" not in info


def test_apply_generated_infoplist_capabilities_allows_missing_plist_only_for_debug(tmp_path: Path):
    result = _run(tmp_path, configuration="Debug")

    assert result.returncode == 0
    assert "Debug build has no generated Info.plist" in result.stderr
