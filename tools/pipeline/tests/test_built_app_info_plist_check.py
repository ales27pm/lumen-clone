import hashlib
import plistlib
import struct
import subprocess
import sys
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "check_built_app_info_plist.py"
ARCHIVE_SCRIPT = ROOT / "scripts" / "archive_lumen_stable.sh"
SUBMIT_SCRIPT = ROOT / "scripts" / "build_and_submit_appstoreconnect.sh"
FULL_REVISION = "a" * 40
MANIFEST = b'{"schemaVersion":1}\n'
APP_UUID = bytes.fromhex("00112233445566778899aabbccddeeff")
MSAL_UUID = bytes.fromhex("102132435465768798a9babcbddceeff")
SECOND_UUID = bytes.fromhex("ffeeddccbbaa99887766554433221100")


def _macho_executable(
    *,
    linked_msal: bool,
    file_type: int = 2,
    uuid_bytes: bytes | None = APP_UUID,
    duplicate_uuid: bool = False,
) -> bytes:
    load_path = b"@rpath/MSAL.framework/MSAL\0"
    commands: list[bytes] = []
    if uuid_bytes is not None:
        uuid_command = struct.pack("<II16s", 0x1B, 24, uuid_bytes)
        commands.append(uuid_command)
        if duplicate_uuid:
            commands.append(uuid_command)
    if linked_msal:
        command_size = 24 + len(load_path)
        command_size += (-command_size) % 8
        command = struct.pack("<IIIIII", 0xC, command_size, 24, 0, 0, 0)
        command += load_path
        command += b"\0" * (command_size - len(command))
        commands.append(command)
    command_bytes = b"".join(commands)
    header = struct.pack(
        "<IiiIIIII",
        0xFEEDFACF,
        0x0100000C,
        0,
        file_type,
        len(commands),
        len(command_bytes),
        0,
        0,
    )
    return header + command_bytes


def _fat_macho(slices: list[bytes]) -> bytes:
    table_size = 8 + 20 * len(slices)
    offsets: list[int] = []
    cursor = table_size
    for value in slices:
        offsets.append(cursor)
        cursor += len(value)
    header = struct.pack(">II", 0xCAFEBABE, len(slices))
    table = b"".join(
        struct.pack(">iiIII", 0x0100000C, index, offsets[index], len(value), 0)
        for index, value in enumerate(slices)
    )
    return header + table + b"".join(slices)


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
    app_uuid: bytes | None = APP_UUID,
    msal_uuid: bytes | None = MSAL_UUID,
    msal_binary: bytes | None = None,
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
                "CFBundleExecutable": "MSAL",
            },
            handle,
        )
    (framework / "MSAL").write_bytes(
        msal_binary
        if msal_binary is not None
        else _macho_executable(
            linked_msal=False,
            file_type=6,
            uuid_bytes=msal_uuid,
        )
    )
    (app / "Lumen").write_bytes(
        _macho_executable(linked_msal=linked_msal, uuid_bytes=app_uuid)
    )
    return app


def _write_dsym(archive: Path, name: str, binary: bytes) -> Path:
    dwarf = archive / "dSYMs" / f"{name}.dSYM" / "Contents" / "Resources" / "DWARF" / name
    dwarf.parent.mkdir(parents=True)
    dwarf.write_bytes(binary)
    return dwarf


def _write_required_dsyms(
    archive: Path,
    *,
    app_binary: bytes | None = None,
    msal_binary: bytes | None = None,
) -> None:
    _write_dsym(
        archive,
        "Lumen",
        app_binary
        if app_binary is not None
        else _macho_executable(
            linked_msal=False,
            file_type=0xA,
            uuid_bytes=APP_UUID,
        ),
    )
    _write_dsym(
        archive,
        "MSAL",
        msal_binary
        if msal_binary is not None
        else _macho_executable(
            linked_msal=False,
            file_type=0xA,
            uuid_bytes=MSAL_UUID,
        ),
    )


def _write_framework(app: Path, name: str, binary: bytes) -> Path:
    framework = app / "Frameworks" / f"{name}.framework"
    framework.mkdir(parents=True)
    with (framework / "Info.plist").open("wb") as handle:
        plistlib.dump(
            {
                "CFBundleIdentifier": f"com.example.{name}",
                "CFBundleExecutable": name,
            },
            handle,
        )
    (framework / name).write_bytes(binary)
    return framework


def _write_ipa(app: Path, ipa: Path) -> Path:
    with zipfile.ZipFile(ipa, "w") as archive:
        for path in sorted(app.rglob("*")):
            if path.is_file():
                archive.write(path, "Payload/Lumen.app/" + path.relative_to(app).as_posix())
    return ipa


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


def test_built_app_info_plist_check_requires_complete_dsym_uuid_coverage(
    tmp_path: Path,
):
    archive = tmp_path / "Lumen.xcarchive"
    _write_valid_app(archive / "Products" / "Applications" / "Lumen.app")
    _write_required_dsyms(archive)

    result = _run(
        archive,
        extra=["--require-dsym-archive", str(archive)],
    )

    assert result.returncode == 0, result.stderr
    assert "dSYM UUID coverage binaries=2 UUIDs=2" in result.stdout


def test_built_app_info_plist_check_rejects_missing_app_dsym(tmp_path: Path):
    archive = tmp_path / "Lumen.xcarchive"
    _write_valid_app(archive / "Products" / "Applications" / "Lumen.app")
    _write_dsym(
        archive,
        "MSAL",
        _macho_executable(
            linked_msal=False,
            file_type=0xA,
            uuid_bytes=MSAL_UUID,
        ),
    )

    result = _run(archive, extra=["--require-dsym-archive", str(archive)])

    assert result.returncode == 1
    assert "Lumen UUID" in result.stderr


def test_built_app_info_plist_check_rejects_missing_framework_dsym(tmp_path: Path):
    archive = tmp_path / "Lumen.xcarchive"
    _write_valid_app(archive / "Products" / "Applications" / "Lumen.app")
    _write_dsym(
        archive,
        "Lumen",
        _macho_executable(
            linked_msal=False,
            file_type=0xA,
            uuid_bytes=APP_UUID,
        ),
    )

    result = _run(archive, extra=["--require-dsym-archive", str(archive)])

    assert result.returncode == 1
    assert "Frameworks/MSAL.framework/MSAL UUID" in result.stderr


def test_built_app_info_plist_check_rejects_wrong_dsym_uuid(tmp_path: Path):
    archive = tmp_path / "Lumen.xcarchive"
    _write_valid_app(archive / "Products" / "Applications" / "Lumen.app")
    _write_required_dsyms(
        archive,
        msal_binary=_macho_executable(
            linked_msal=False,
            file_type=0xA,
            uuid_bytes=SECOND_UUID,
        ),
    )

    result = _run(archive, extra=["--require-dsym-archive", str(archive)])

    assert result.returncode == 1
    assert "Frameworks/MSAL.framework/MSAL UUID" in result.stderr


def test_built_app_info_plist_check_rejects_uncovered_fat_binary_slice(
    tmp_path: Path,
):
    archive = tmp_path / "Lumen.xcarchive"
    app = _write_valid_app(archive / "Products" / "Applications" / "Lumen.app")
    (app / "Lumen").write_bytes(
        _fat_macho(
            [
                _macho_executable(linked_msal=True, uuid_bytes=APP_UUID),
                _macho_executable(linked_msal=True, uuid_bytes=SECOND_UUID),
            ]
        )
    )
    _write_required_dsyms(archive)

    result = _run(archive, extra=["--require-dsym-archive", str(archive)])

    assert result.returncode == 1
    assert "Lumen UUID FFEEDDCC-BBAA-9988-7766-554433221100" in result.stderr


def test_built_app_info_plist_check_rejects_exported_ipa_uuid_drift(
    tmp_path: Path,
):
    archive = tmp_path / "Lumen.xcarchive"
    _write_valid_app(archive / "Products" / "Applications" / "Lumen.app")
    _write_required_dsyms(archive)
    exported_app = _write_valid_app(
        tmp_path / "export" / "Lumen.app",
        msal_uuid=SECOND_UUID,
    )
    ipa = _write_ipa(exported_app, tmp_path / "Lumen.ipa")

    result = _run(ipa, extra=["--require-dsym-archive", str(archive)])

    assert result.returncode == 1
    assert "Frameworks/MSAL.framework/MSAL UUID FFEEDDCC-BBAA-9988-7766-554433221100" in result.stderr


def test_built_app_info_plist_check_skips_static_and_object_frameworks(
    tmp_path: Path,
):
    archive = tmp_path / "Lumen.xcarchive"
    app = _write_valid_app(archive / "Products" / "Applications" / "Lumen.app")
    _write_framework(app, "StaticKit", b"!<arch>\nstatic-object-data")
    _write_framework(
        app,
        "ObjectKit",
        _macho_executable(
            linked_msal=False,
            file_type=0x1,
            uuid_bytes=None,
        ),
    )
    _write_required_dsyms(archive)

    result = _run(archive, extra=["--require-dsym-archive", str(archive)])

    assert result.returncode == 0, result.stderr
    assert "static framework symbol coverage inherited by app" in result.stdout
    assert "object framework symbol coverage inherited by app" in result.stdout


def test_built_app_info_plist_check_rejects_mixed_framework_slice_types(
    tmp_path: Path,
):
    archive = tmp_path / "Lumen.xcarchive"
    app = _write_valid_app(archive / "Products" / "Applications" / "Lumen.app")
    _write_framework(
        app,
        "MixedKit",
        _fat_macho(
            [
                _macho_executable(
                    linked_msal=False,
                    file_type=0x1,
                    uuid_bytes=APP_UUID,
                ),
                _macho_executable(
                    linked_msal=False,
                    file_type=0x6,
                    uuid_bytes=SECOND_UUID,
                ),
            ]
        ),
    )
    _write_required_dsyms(archive)

    result = _run(archive, extra=["--require-dsym-archive", str(archive)])

    assert result.returncode == 1
    assert "must contain only MH_DYLIB slices" in result.stderr


def test_built_app_info_plist_check_rejects_missing_or_duplicate_lc_uuid(
    tmp_path: Path,
):
    missing_archive = tmp_path / "Missing.xcarchive"
    missing_app = _write_valid_app(
        missing_archive / "Products" / "Applications" / "Lumen.app",
        app_uuid=None,
    )
    assert missing_app.is_dir()
    _write_required_dsyms(missing_archive)

    duplicate_archive = tmp_path / "Duplicate.xcarchive"
    duplicate_app = _write_valid_app(
        duplicate_archive / "Products" / "Applications" / "Lumen.app"
    )
    (duplicate_app / "Lumen").write_bytes(
        _macho_executable(
            linked_msal=True,
            uuid_bytes=APP_UUID,
            duplicate_uuid=True,
        )
    )
    _write_required_dsyms(duplicate_archive)

    missing_result = _run(
        missing_archive,
        extra=["--require-dsym-archive", str(missing_archive)],
    )
    duplicate_result = _run(
        duplicate_archive,
        extra=["--require-dsym-archive", str(duplicate_archive)],
    )

    assert missing_result.returncode == 1
    assert "slice 0 is missing LC_UUID" in missing_result.stderr
    assert duplicate_result.returncode == 1
    assert "duplicate LC_UUID commands" in duplicate_result.stderr


def test_built_app_info_plist_check_rejects_wrong_dsym_type_and_duplicate_owner(
    tmp_path: Path,
):
    wrong_type_archive = tmp_path / "WrongType.xcarchive"
    _write_valid_app(
        wrong_type_archive / "Products" / "Applications" / "Lumen.app"
    )
    _write_required_dsyms(
        wrong_type_archive,
        app_binary=_macho_executable(
            linked_msal=False,
            file_type=0x2,
            uuid_bytes=APP_UUID,
        ),
    )

    duplicate_archive = tmp_path / "DuplicateOwner.xcarchive"
    _write_valid_app(
        duplicate_archive / "Products" / "Applications" / "Lumen.app"
    )
    _write_required_dsyms(duplicate_archive)
    _write_dsym(
        duplicate_archive,
        "Other",
        _macho_executable(
            linked_msal=False,
            file_type=0xA,
            uuid_bytes=APP_UUID,
        ),
    )

    wrong_type_result = _run(
        wrong_type_archive,
        extra=["--require-dsym-archive", str(wrong_type_archive)],
    )
    duplicate_result = _run(
        duplicate_archive,
        extra=["--require-dsym-archive", str(duplicate_archive)],
    )

    assert wrong_type_result.returncode == 1
    assert "must contain only MH_DSYM slices" in wrong_type_result.stderr
    assert duplicate_result.returncode == 1
    assert "ambiguously owned" in duplicate_result.stderr


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


def test_release_scripts_require_symbols_before_success_or_upload():
    archive_script = ARCHIVE_SCRIPT.read_text(encoding="utf-8")
    submit_script = SUBMIT_SCRIPT.read_text(encoding="utf-8")

    source_zip_normalization_index = archive_script.index(
        'MSAL_DSYM_SOURCE_ZIP="$(cd "$(dirname "$MSAL_DSYM_SOURCE_ZIP")"'
    )
    repository_chdir_index = archive_script.index('cd "$REPO_ROOT"')
    install_index = archive_script.index("install_msal_release_dsym.py")
    archive_symbol_gate_index = archive_script.index(
        'INFO_PLIST_CHECK+=(--require-dsym-archive "$ARCHIVE_PATH")'
    )
    archive_success_index = archive_script.index('bold "✅ Archive created: $ARCHIVE_PATH"')
    assert source_zip_normalization_index < repository_chdir_index < install_index
    assert install_index < archive_symbol_gate_index < archive_success_index

    archived_check_index = submit_script.index("ARCHIVE_INFO_CHECK=(")
    archived_symbol_gate_index = submit_script.index(
        'ARCHIVE_INFO_CHECK+=(--require-dsym-archive "$ARCHIVE_PATH")'
    )
    export_index = submit_script.index('info "Export IPA"')
    packaging_gate_index = submit_script.index(
        'grep -Fq "Upload Symbols Failed" "$PACKAGING_LOG"'
    )
    ipa_check_index = submit_script.index("IPA_INFO_CHECK=(")
    ipa_symbol_gate_index = submit_script.index(
        'IPA_INFO_CHECK+=(--require-dsym-archive "$ARCHIVE_PATH")'
    )
    built_ipa_index = submit_script.index('bold "Built IPA: $IPA_PATH"')
    upload_index = submit_script.index('ensure_upload_tool', built_ipa_index)

    assert archived_check_index < archived_symbol_gate_index < export_index
    assert export_index < packaging_gate_index < ipa_check_index
    assert ipa_check_index < ipa_symbol_gate_index < built_ipa_index < upload_index
