import hashlib
import plistlib
import struct
import subprocess
import sys
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "check_built_app_info_plist.py"
FULL_REVISION = "a" * 40
MANIFEST = b'{"schemaVersion":1}\n'


def _macho_executable(*, linked_msal: bool) -> bytes:
    load_path = b"@rpath/MSAL.framework/MSAL\0"
    command = b""
    if linked_msal:
        command_size = 24 + len(load_path)
        command_size += (-command_size) % 8
        command = struct.pack("<IIIIII", 0xC, command_size, 24, 0, 0, 0)
        command += load_path
        command += b"\0" * (command_size - len(command))
    header = struct.pack(
        "<IiiIIIII",
        0xFEEDFACF,
        0x0100000C,
        0,
        2,
        1 if command else 0,
        len(command),
        0,
        0,
    )
    return header + command


def _valid_info(*, configuration: str = "Release") -> dict:
    return {
        "CFBundleIdentifier": "com.27pm.lumenclone",
        "CFBundleVersion": "42",
        "CFBundleShortVersionString": "1.0.0",
        "CFBundleExecutable": "Lumen",
        "NSAlarmKitUsageDescription": "Lumen uses AlarmKit to schedule alarms.",
        "LumenBuildSourceIdentifier": "42",
        "LumenBuildConfiguration": configuration,
        "LumenBuildScheme": "Lumen",
        "LumenGitSHA": FULL_REVISION if configuration != "Debug" else "abc123",
        "UIFileSharingEnabled": configuration == "Debug",
        "CFBundleURLTypes": [
            {
                "CFBundleTypeRole": "Editor",
                "CFBundleURLName": "com.27pm.lumenclone",
                "CFBundleURLSchemes": ["msauth.com.27pm.lumenclone"],
            }
        ],
        "LSApplicationQueriesSchemes": ["msauth", "msauthv2", "msauthv3"],
    }


def _write_valid_app(
    app: Path,
    info: dict | None = None,
    *,
    manifest: bytes = MANIFEST,
    sidecar_digest: str | None = None,
    root_manifest: bytes | None = None,
    msal_version: str = "1.9.0",
    linked_msal: bool = True,
) -> Path:
    app.mkdir(parents=True)
    with (app / "Info.plist").open("wb") as handle:
        plistlib.dump(info or _valid_info(), handle)
    with (app / "PrivacyInfo.xcprivacy").open("wb") as handle:
        plistlib.dump({"NSPrivacyTracking": False}, handle)

    grounding = app / "AgentGrounding" / "agent_manifest"
    grounding.mkdir(parents=True)
    (grounding / "AgentBehaviorManifest.json").write_bytes(manifest)
    digest = sidecar_digest or hashlib.sha256(manifest).hexdigest()
    (grounding / "AgentBehaviorManifest.sha256").write_text(
        f"{digest}\n", encoding="ascii"
    )
    (app / "AgentBehaviorManifest.json").write_bytes(
        manifest if root_manifest is None else root_manifest
    )

    framework = app / "Frameworks" / "MSAL.framework"
    framework.mkdir(parents=True)
    with (framework / "Info.plist").open("wb") as handle:
        plistlib.dump(
            {
                "CFBundleIdentifier": "com.microsoft.MSAL",
                "CFBundleShortVersionString": msal_version,
            },
            handle,
        )
    (framework / "MSAL").write_bytes(b"MSAL-framework-binary")
    (app / "Lumen").write_bytes(_macho_executable(linked_msal=linked_msal))
    return app


def _run(app_or_archive: Path, *, configuration: str = "Release", extra: list[str] | None = None):
    command = [
        sys.executable,
        str(SCRIPT),
        str(app_or_archive),
        "--expected-bundle-identifier",
        "com.27pm.lumenclone",
        "--expected-bundle-version",
        "42",
        "--expected-marketing-version",
        "1.0.0",
        "--expected-build-configuration",
        configuration,
        "--expected-msal-version",
        "1.9.0",
    ]
    if configuration != "Debug":
        command.extend(["--expected-source-revision", FULL_REVISION])
    if extra:
        command.extend(extra)
    return subprocess.run(command, text=True, capture_output=True, check=False)


def test_built_app_info_plist_check_accepts_complete_release_bundle(tmp_path: Path):
    app = _write_valid_app(tmp_path / "Lumen.app")

    result = _run(app)

    assert result.returncode == 0, result.stderr
    assert "full source revision=" + FULL_REVISION in result.stdout
    assert "UIFileSharingEnabled=false" in result.stdout
    assert "AgentBehaviorManifest SHA-256=" in result.stdout
    assert "embedded and linked MSAL version=1.9.0" in result.stdout


def test_built_app_info_plist_check_requires_alarmkit_usage_description(tmp_path: Path):
    info = _valid_info()
    info.pop("NSAlarmKitUsageDescription")
    app = _write_valid_app(tmp_path / "Lumen.app", info)

    result = _run(app)

    assert result.returncode == 1
    assert "missing NSAlarmKitUsageDescription" in result.stderr


def test_built_app_info_plist_check_rejects_wrong_release_identity(tmp_path: Path):
    info = _valid_info()
    info["CFBundleIdentifier"] = "com.example.impostor"
    app = _write_valid_app(tmp_path / "Lumen.app", info)

    result = _run(app)

    assert result.returncode == 1
    assert "CFBundleIdentifier='com.example.impostor'; expected 'com.27pm.lumenclone'" in result.stderr


def test_built_app_info_plist_check_rejects_stale_marketing_version(tmp_path: Path):
    info = _valid_info()
    info["CFBundleShortVersionString"] = "0.9.0"
    app = _write_valid_app(tmp_path / "Lumen.app", info)

    result = _run(app)

    assert result.returncode == 1
    assert "CFBundleShortVersionString='0.9.0'; expected '1.0.0'" in result.stderr


def test_built_app_info_plist_check_rejects_short_release_revision(tmp_path: Path):
    info = _valid_info()
    info["LumenGitSHA"] = "abc123"
    app = _write_valid_app(tmp_path / "Lumen.app", info)

    result = _run(app)

    assert result.returncode == 1
    assert "must be a full Git revision" in result.stderr


def test_built_app_info_plist_check_rejects_release_file_sharing(tmp_path: Path):
    info = _valid_info()
    info["UIFileSharingEnabled"] = True
    app = _write_valid_app(tmp_path / "Lumen.app", info)

    result = _run(app)

    assert result.returncode == 1
    assert "UIFileSharingEnabled must be false" in result.stderr


def test_built_app_info_plist_check_allows_debug_file_sharing_only_when_explicit(tmp_path: Path):
    app = _write_valid_app(tmp_path / "Lumen.app", _valid_info(configuration="Debug"))

    release_result = _run(app)
    debug_result = _run(app, configuration="Debug")

    assert release_result.returncode == 1
    assert "LumenBuildConfiguration='Debug'; expected 'Release'" in release_result.stderr
    assert debug_result.returncode == 0, debug_result.stderr


def test_built_app_info_plist_check_rejects_missing_app_privacy_manifest(tmp_path: Path):
    app = _write_valid_app(tmp_path / "Lumen.app")
    (app / "PrivacyInfo.xcprivacy").unlink()

    result = _run(app)

    assert result.returncode == 1
    assert "missing required bundled file PrivacyInfo.xcprivacy" in result.stderr


def test_built_app_info_plist_check_rejects_manifest_sidecar_mismatch(tmp_path: Path):
    app = _write_valid_app(tmp_path / "Lumen.app", sidecar_digest="0" * 64)

    result = _run(app)

    assert result.returncode == 1
    assert "does not match sidecar" in result.stderr


def test_built_app_info_plist_check_rejects_root_manifest_drift(tmp_path: Path):
    app = _write_valid_app(
        tmp_path / "Lumen.app",
        root_manifest=b'{"schemaVersion":2}\n',
    )

    result = _run(app)

    assert result.returncode == 1
    assert "does not match AgentGrounding/agent_manifest/AgentBehaviorManifest.json" in result.stderr


def test_built_app_info_plist_check_rejects_wrong_or_unlinked_msal(tmp_path: Path):
    wrong_version = _write_valid_app(
        tmp_path / "WrongVersion.app", msal_version="1.8.0"
    )
    unlinked = _write_valid_app(tmp_path / "Unlinked.app", linked_msal=False)

    version_result = _run(wrong_version)
    linkage_result = _run(unlinked)

    assert version_result.returncode == 1
    assert "embedded MSAL version='1.8.0'; expected pinned version '1.9.0'" in version_result.stderr
    assert linkage_result.returncode == 1
    assert "Mach-O slice(s) without a load command" in linkage_result.stderr


def test_built_app_info_plist_check_accepts_xcarchive(tmp_path: Path):
    archive = tmp_path / "Lumen.xcarchive"
    _write_valid_app(archive / "Products" / "Applications" / "Lumen.app")

    result = _run(archive)

    assert result.returncode == 0, result.stderr


def test_built_app_info_plist_check_accepts_ipa_with_complete_release_payload(tmp_path: Path):
    app = _write_valid_app(tmp_path / "payload-source" / "Lumen.app")
    ipa = tmp_path / "Lumen.ipa"
    with zipfile.ZipFile(ipa, "w") as archive:
        for path in sorted(app.rglob("*")):
            if path.is_file():
                archive.write(path, "Payload/Lumen.app/" + path.relative_to(app).as_posix())

    result = _run(ipa)

    assert result.returncode == 0, result.stderr
    assert "bundled app privacy manifest=PrivacyInfo.xcprivacy" in result.stdout
