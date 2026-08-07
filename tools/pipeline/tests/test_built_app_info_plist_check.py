import plistlib
import subprocess
import sys
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "check_built_app_info_plist.py"


def _write_app(tmp_path: Path, info: dict) -> Path:
    app = tmp_path / "Lumen.app"
    app.mkdir()
    with (app / "Info.plist").open("wb") as handle:
        plistlib.dump(info, handle)
    return app


def _valid_info() -> dict:
    return {
        "CFBundleIdentifier": "com.27pm.lumenclone",
        "CFBundleVersion": "42",
        "NSAlarmKitUsageDescription": "Lumen uses AlarmKit to schedule alarms.",
        "LumenBuildSourceIdentifier": "42",
        "LumenBuildConfiguration": "Debug",
        "LumenBuildScheme": "Lumen",
        "LumenGitSHA": "abc123",
        "CFBundleURLTypes": [
            {
                "CFBundleTypeRole": "Editor",
                "CFBundleURLName": "com.27pm.lumenclone",
                "CFBundleURLSchemes": ["msauth.com.27pm.lumenclone"],
            }
        ],
        "LSApplicationQueriesSchemes": ["msauth", "msauthv2", "msauthv3"],
    }


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
    app = _write_app(tmp_path, _valid_info())

    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(app)],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert "NSAlarmKitUsageDescription" in result.stdout
    assert "LumenGitSHA=abc123" in result.stdout
    assert "MSAL callback URL scheme=msauth.com.27pm.lumenclone" in result.stdout


def test_built_app_info_plist_check_rejects_missing_msal_callback_scheme(tmp_path: Path):
    info = _valid_info()
    info["CFBundleURLTypes"] = [
        {"CFBundleURLSchemes": ["msauth.com.example.wrong"]}
    ]
    app = _write_app(tmp_path, info)

    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(app)],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    assert "missing MSAL callback URL scheme 'msauth.com.27pm.lumenclone'" in result.stderr


def test_built_app_info_plist_check_rejects_missing_msal_query_scheme(tmp_path: Path):
    info = _valid_info()
    info["LSApplicationQueriesSchemes"] = ["msauth", "msauthv3"]
    app = _write_app(tmp_path, info)

    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(app)],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    assert "missing required MSAL query schemes: msauthv2" in result.stderr


def test_built_app_info_plist_check_requires_expected_release_identity(tmp_path: Path):
    info = _valid_info()
    info["LumenBuildConfiguration"] = "Release"
    app = _write_app(tmp_path, info)

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            str(app),
            "--expected-bundle-version",
            "42",
            "--expected-build-configuration",
            "Release",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0


def test_built_app_info_plist_check_rejects_stale_release_build_number(tmp_path: Path):
    info = _valid_info()
    info["LumenBuildConfiguration"] = "Release"
    app = _write_app(tmp_path, info)

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            str(app),
            "--expected-bundle-version",
            "43",
            "--expected-build-configuration",
            "Release",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    assert "CFBundleVersion='42'; expected '43'" in result.stderr


def test_built_app_info_plist_check_rejects_wrong_release_configuration(tmp_path: Path):
    app = _write_app(tmp_path, _valid_info())

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            str(app),
            "--expected-build-configuration",
            "Release",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    assert "LumenBuildConfiguration='Debug'; expected 'Release'" in result.stderr


def test_built_app_info_plist_check_rejects_mismatched_source_identifier(tmp_path: Path):
    info = _valid_info()
    info["LumenBuildSourceIdentifier"] = "41"
    app = _write_app(tmp_path, info)

    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(app)],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    assert "LumenBuildSourceIdentifier='41'; expected CFBundleVersion '42'" in result.stderr


def test_built_app_info_plist_check_rejects_unknown_build_metadata(tmp_path: Path):
    info = _valid_info()
    info["LumenGitSHA"] = "unknown"
    app = _write_app(tmp_path, info)

    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(app)],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    assert "missing required build metadata LumenGitSHA" in result.stderr


def test_built_app_info_plist_check_accepts_xcarchive(tmp_path: Path):
    app = tmp_path / "Lumen.xcarchive" / "Products" / "Applications" / "Lumen.app"
    app.mkdir(parents=True)
    with (app / "Info.plist").open("wb") as handle:
        plistlib.dump(_valid_info(), handle)

    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(tmp_path / "Lumen.xcarchive")],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert "LumenGitSHA=abc123" in result.stdout


def test_built_app_info_plist_check_accepts_ipa(tmp_path: Path):
    ipa = tmp_path / "Lumen.ipa"
    plist_bytes = plistlib.dumps(_valid_info())
    with zipfile.ZipFile(ipa, "w") as archive:
        archive.writestr("Payload/Lumen.app/Info.plist", plist_bytes)

    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(ipa)],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert "LumenBuildScheme=Lumen" in result.stdout
