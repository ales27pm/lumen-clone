#!/usr/bin/env python3
"""Fail-closed verification for a built Lumen app, archive, or IPA."""

from __future__ import annotations

import argparse
import hashlib
import json
import plistlib
import re
import struct
import subprocess
import sys
import uuid as uuidlib
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import NoReturn


DEFAULT_BUNDLE_IDENTIFIER = "com.27pm.lumenclone"
REQUIRED_ALARM_KEY = "NSAlarmKitUsageDescription"
REQUIRED_MSAL_QUERY_SCHEMES = {"msauth", "msauthv2", "msauthv3"}
PRIVACY_MANIFEST_PATH = "PrivacyInfo.xcprivacy"
ROOT_AGENT_MANIFEST_PATH = "AgentBehaviorManifest.json"
AGENT_MANIFEST_PATH = "AgentGrounding/agent_manifest/AgentBehaviorManifest.json"
AGENT_MANIFEST_SHA_PATH = "AgentGrounding/agent_manifest/AgentBehaviorManifest.sha256"
MSAL_FRAMEWORK_INFO_PATH = "Frameworks/MSAL.framework/Info.plist"
MSAL_FRAMEWORK_BINARY_PATH = "Frameworks/MSAL.framework/MSAL"
MSAL_BUNDLE_IDENTIFIER = "com.microsoft.MSAL"
MSAL_LOAD_PATH = "@rpath/MSAL.framework/MSAL"
FULL_GIT_REVISION = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
SHA256_DIGEST = re.compile(r"^[0-9a-f]{64}$")
MACHO_MAGICS = {
    b"\xfe\xed\xfa\xce": (">", 28),
    b"\xce\xfa\xed\xfe": ("<", 28),
    b"\xfe\xed\xfa\xcf": (">", 32),
    b"\xcf\xfa\xed\xfe": ("<", 32),
}
FAT_MAGICS = {
    b"\xca\xfe\xba\xbe": (">", False),
    b"\xbe\xba\xfe\xca": ("<", False),
    b"\xca\xfe\xba\xbf": (">", True),
    b"\xbf\xba\xfe\xca": ("<", True),
}
DYLIB_LOAD_COMMANDS = {0xC, 0x18, 0x1F, 0x23}
LC_UUID = 0x1B
MH_OBJECT = 0x1
MH_EXECUTE = 0x2
MH_DYLIB = 0x6
MH_DSYM = 0xA
STATIC_ARCHIVE_MAGIC = b"!<arch>\n"


def _fail(message: str) -> NoReturn:
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(1)


class DirectoryBundleReader:
    def __init__(self, app_path: Path):
        self.app_path = app_path
        self.source = str(app_path)

    def read_bytes(self, relative_path: str) -> bytes:
        path = self.app_path / relative_path
        if not path.is_file():
            _fail(f"{self.source} is missing required bundled file {relative_path}")
        try:
            return path.read_bytes()
        except OSError as error:
            _fail(f"could not read {relative_path} from {self.source}: {error}")

    def framework_binaries(self) -> list[tuple[str, bytes]]:
        frameworks_root = self.app_path / "Frameworks"
        if not frameworks_root.is_dir():
            return []
        binaries: list[tuple[str, bytes]] = []
        for framework in sorted(frameworks_root.rglob("*.framework")):
            if not framework.is_dir():
                continue
            relative_root = framework.relative_to(self.app_path).as_posix()
            info = _read_plist_bytes(
                self.read_bytes(f"{relative_root}/Info.plist"),
                f"{self.source}/{relative_root}/Info.plist",
            )
            executable = _framework_executable(info, f"{self.source}/{relative_root}")
            relative_binary = f"{relative_root}/{executable}"
            binaries.append((relative_binary, self.read_bytes(relative_binary)))
        return binaries


class IPABundleReader:
    def __init__(self, path: Path, archive: zipfile.ZipFile):
        self.path = path
        self.archive = archive
        self.source = str(path)
        app_roots: set[str] = set()
        names = archive.namelist()
        if len(names) != len(set(names)):
            _fail(f"{path} contains duplicate archive member names")
        for name in names:
            pure = PurePosixPath(name)
            if pure.is_absolute() or ".." in pure.parts:
                _fail(f"{path} contains unsafe archive member {name!r}")
            if len(pure.parts) >= 2 and pure.parts[0] == "Payload" and pure.parts[1].endswith(".app"):
                app_roots.add("/".join(pure.parts[:2]))
        if len(app_roots) != 1:
            _fail(f"expected exactly one Payload/*.app in {path}; found {len(app_roots)}")
        self.prefix = next(iter(app_roots)) + "/"
        self.members = set(names)

    def read_bytes(self, relative_path: str) -> bytes:
        member = self.prefix + relative_path
        if member not in self.members:
            _fail(f"{self.source} is missing required bundled file {relative_path}")
        try:
            return self.archive.read(member)
        except (KeyError, OSError, RuntimeError) as error:
            _fail(f"could not read {relative_path} from {self.source}: {error}")

    def framework_binaries(self) -> list[tuple[str, bytes]]:
        binaries: list[tuple[str, bytes]] = []
        info_paths: list[PurePosixPath] = []
        for member in self.members:
            if not member.startswith(self.prefix):
                continue
            relative = PurePosixPath(member[len(self.prefix) :])
            if relative.name == "Info.plist" and relative.parent.name.endswith(
                ".framework"
            ):
                info_paths.append(relative)
        for info_path in sorted(info_paths, key=str):
            relative_root = info_path.parent.as_posix()
            info = _read_plist_bytes(
                self.read_bytes(info_path.as_posix()),
                f"{self.source}/{info_path.as_posix()}",
            )
            executable = _framework_executable(info, f"{self.source}/{relative_root}")
            relative_binary = f"{relative_root}/{executable}"
            binaries.append((relative_binary, self.read_bytes(relative_binary)))
        return binaries


def _framework_executable(info: dict, source: str) -> str:
    executable = str(info.get("CFBundleExecutable") or "").strip()
    if not executable or "/" in executable or executable in {".", ".."}:
        _fail(f"{source} has invalid or missing CFBundleExecutable")
    return executable


def _find_app(path: Path) -> Path:
    if path.suffix == ".app" and path.is_dir():
        return path
    if path.suffix == ".xcarchive":
        apps = sorted((path / "Products" / "Applications").glob("*.app"))
        if len(apps) == 1:
            return apps[0]
        _fail(f"expected exactly one Products/Applications/*.app in {path}; found {len(apps)}")
    _fail(f"could not locate .app bundle in {path}")


def _read_plist_bytes(value: bytes, source: str) -> dict:
    try:
        parsed = plistlib.loads(value)
    except (plistlib.InvalidFileException, ValueError, TypeError) as error:
        _fail(f"invalid plist in {source}: {error}")
    if not isinstance(parsed, dict):
        _fail(f"expected plist dictionary in {source}")
    return parsed


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _current_source_revision(root: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "--verify", "HEAD^{commit}"],
        text=True,
        capture_output=True,
        check=False,
    )
    revision = result.stdout.strip().lower()
    if result.returncode != 0 or not FULL_GIT_REVISION.fullmatch(revision):
        detail = (result.stderr or result.stdout).strip()
        _fail(f"could not resolve the expected full source revision from {root}: {detail}")
    return revision


def _project_setting(root: Path, key: str) -> str:
    project_path = root / "ios" / "Lumen.xcodeproj" / "project.pbxproj"
    try:
        text = project_path.read_text(encoding="utf-8")
    except OSError as error:
        _fail(f"could not read {project_path}: {error}")
    values = {
        value.strip().strip('"')
        for value in re.findall(rf"\b{re.escape(key)}\s*=\s*([^;]+);", text)
        if value.strip() and "$(" not in value
    }
    if len(values) != 1:
        _fail(
            f"could not infer one exact expected {key} from {project_path}; "
            "pass the corresponding --expected option"
        )
    return next(iter(values))


def _pinned_msal_version(root: Path) -> str:
    resolved_path = (
        root
        / "ios"
        / "Lumen.xcodeproj"
        / "project.xcworkspace"
        / "xcshareddata"
        / "swiftpm"
        / "Package.resolved"
    )
    try:
        resolved = json.loads(resolved_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        _fail(f"could not read MSAL package pin from {resolved_path}: {error}")
    for pin in resolved.get("pins", []):
        if not isinstance(pin, dict):
            continue
        if pin.get("identity") != "microsoft-authentication-library-for-objc":
            continue
        state = pin.get("state")
        version = state.get("version") if isinstance(state, dict) else None
        if isinstance(version, str) and version.strip():
            return version.strip()
    _fail(f"{resolved_path} has no exact MSAL package version pin")


def _validate_info(
    info: dict,
    source: str,
    *,
    expected_bundle_identifier: str,
    expected_bundle_version: str,
    expected_marketing_version: str,
    expected_build_configuration: str,
    expected_source_revision: str | None,
) -> str:
    alarm_value = str(info.get(REQUIRED_ALARM_KEY) or "").strip()
    if not alarm_value:
        _fail(f"{source} Info.plist missing {REQUIRED_ALARM_KEY}")

    exact_values = {
        "CFBundleIdentifier": expected_bundle_identifier,
        "CFBundleVersion": expected_bundle_version,
        "CFBundleShortVersionString": expected_marketing_version,
        "LumenBuildConfiguration": expected_build_configuration,
    }
    for key, expected in exact_values.items():
        actual = str(info.get(key) or "").strip()
        if actual != expected:
            _fail(f"{source} Info.plist {key}={actual!r}; expected {expected!r}")

    bundle_version = str(info["CFBundleVersion"])
    source_identifier = str(info.get("LumenBuildSourceIdentifier") or "").strip()
    if source_identifier != bundle_version:
        _fail(
            f"{source} Info.plist LumenBuildSourceIdentifier={source_identifier!r}; "
            f"expected CFBundleVersion {bundle_version!r}"
        )

    build_scheme = str(info.get("LumenBuildScheme") or "").strip()
    if not build_scheme or build_scheme.casefold() == "unknown":
        _fail(f"{source} Info.plist missing required build metadata LumenBuildScheme")

    source_revision = str(info.get("LumenGitSHA") or "").strip().lower()
    if not source_revision or source_revision == "unknown":
        _fail(f"{source} Info.plist missing required build metadata LumenGitSHA")
    if expected_build_configuration != "Debug":
        if not FULL_GIT_REVISION.fullmatch(source_revision):
            _fail(
                f"{source} Info.plist LumenGitSHA must be a full Git revision for "
                f"{expected_build_configuration}; got {source_revision!r}"
            )
        if expected_source_revision is None:
            _fail("internal error: non-Debug validation has no expected source revision")
    if expected_source_revision is not None and source_revision != expected_source_revision:
        _fail(
            f"{source} Info.plist LumenGitSHA={source_revision!r}; "
            f"expected {expected_source_revision!r}"
        )

    file_sharing = info.get("UIFileSharingEnabled")
    if not isinstance(file_sharing, bool):
        _fail(f"{source} Info.plist UIFileSharingEnabled must be an explicit boolean")
    if expected_build_configuration != "Debug" and file_sharing is not False:
        _fail(f"{source} Info.plist UIFileSharingEnabled must be false for non-Debug artifacts")

    expected_callback_scheme = f"msauth.{expected_bundle_identifier}"
    url_types = info.get("CFBundleURLTypes")
    if not isinstance(url_types, list):
        _fail(f"{source} Info.plist missing CFBundleURLTypes array")
    registered_url_schemes = {
        scheme
        for url_type in url_types
        if isinstance(url_type, dict)
        for scheme in url_type.get("CFBundleURLSchemes", [])
        if isinstance(scheme, str)
    }
    if expected_callback_scheme not in registered_url_schemes:
        _fail(
            f"{source} Info.plist missing MSAL callback URL scheme "
            f"{expected_callback_scheme!r}"
        )

    query_schemes = info.get("LSApplicationQueriesSchemes")
    if not isinstance(query_schemes, list):
        _fail(f"{source} Info.plist missing LSApplicationQueriesSchemes array")
    missing_query_schemes = REQUIRED_MSAL_QUERY_SCHEMES - {
        value for value in query_schemes if isinstance(value, str)
    }
    if missing_query_schemes:
        _fail(
            f"{source} Info.plist missing required MSAL query schemes: "
            f"{', '.join(sorted(missing_query_schemes))}"
        )

    executable = str(info.get("CFBundleExecutable") or "").strip()
    if not executable or "/" in executable or executable in {".", ".."}:
        _fail(f"{source} Info.plist has invalid or missing CFBundleExecutable")

    print(
        "ok: release identity "
        f"bundle={expected_bundle_identifier} marketing={expected_marketing_version} "
        f"build={expected_bundle_version} configuration={expected_build_configuration}"
    )
    print(f"ok: full source revision={source_revision}")
    print(f"ok: UIFileSharingEnabled={str(file_sharing).lower()}")
    print(f"ok: MSAL callback URL scheme={expected_callback_scheme}")
    return executable


def _validate_privacy_manifest(reader: DirectoryBundleReader | IPABundleReader) -> None:
    privacy = _read_plist_bytes(
        reader.read_bytes(PRIVACY_MANIFEST_PATH),
        f"{reader.source}/{PRIVACY_MANIFEST_PATH}",
    )
    if not privacy:
        _fail(f"{reader.source}/{PRIVACY_MANIFEST_PATH} must not be empty")
    print(f"ok: bundled app privacy manifest={PRIVACY_MANIFEST_PATH}")


def _validate_agent_manifest(reader: DirectoryBundleReader | IPABundleReader) -> None:
    manifest = reader.read_bytes(AGENT_MANIFEST_PATH)
    root_manifest = reader.read_bytes(ROOT_AGENT_MANIFEST_PATH)
    if root_manifest != manifest:
        _fail(
            f"{reader.source} bundled {ROOT_AGENT_MANIFEST_PATH} does not match "
            f"{AGENT_MANIFEST_PATH}"
        )
    try:
        parsed = json.loads(manifest)
    except (UnicodeError, json.JSONDecodeError) as error:
        _fail(f"{reader.source}/{AGENT_MANIFEST_PATH} is invalid JSON: {error}")
    if not isinstance(parsed, dict):
        _fail(f"{reader.source}/{AGENT_MANIFEST_PATH} must contain a JSON object")

    try:
        sidecar_text = reader.read_bytes(AGENT_MANIFEST_SHA_PATH).decode(
            "ascii", errors="strict"
        )
    except UnicodeDecodeError as error:
        _fail(f"{reader.source}/{AGENT_MANIFEST_SHA_PATH} is not ASCII: {error}")
    sidecar_lines = sidecar_text.splitlines()
    expected_digest = sidecar_lines[0] if len(sidecar_lines) == 1 else ""
    if not SHA256_DIGEST.fullmatch(expected_digest):
        _fail(
            f"{reader.source}/{AGENT_MANIFEST_SHA_PATH} must contain exactly one "
            "lowercase SHA-256 digest"
        )
    actual_digest = hashlib.sha256(manifest).hexdigest()
    if actual_digest != expected_digest:
        _fail(
            f"{reader.source}/{AGENT_MANIFEST_PATH} SHA-256 {actual_digest} does not "
            f"match sidecar {expected_digest}"
        )
    print(f"ok: AgentBehaviorManifest SHA-256={actual_digest}")


@dataclass(frozen=True)
class MachOSlice:
    file_type: int
    uuid: str | None
    dylib_loads: frozenset[str]


def _thin_macho_slice(value: bytes, source: str) -> MachOSlice:
    magic = value[:4]
    metadata = MACHO_MAGICS.get(magic)
    if metadata is None:
        _fail(f"{source} is not a supported thin Mach-O image")
    endian, header_size = metadata
    if len(value) < header_size:
        _fail(f"{source} has a truncated Mach-O header")
    (file_type,) = struct.unpack_from(f"{endian}I", value, 12)
    ncmds, sizeofcmds = struct.unpack_from(f"{endian}II", value, 16)
    commands_end = header_size + sizeofcmds
    if ncmds > 100_000 or commands_end > len(value):
        _fail(f"{source} has invalid Mach-O load-command bounds")

    loads: set[str] = set()
    uuids: list[str] = []
    offset = header_size
    for _ in range(ncmds):
        if offset + 8 > commands_end:
            _fail(f"{source} has a truncated Mach-O load command")
        command, command_size = struct.unpack_from(f"{endian}II", value, offset)
        if command_size < 8 or offset + command_size > commands_end:
            _fail(f"{source} has an invalid Mach-O load command size")
        command_without_required_bit = command & 0x7FFFFFFF
        if command_without_required_bit == LC_UUID:
            if command_size != 24:
                _fail(f"{source} has an invalid LC_UUID command size")
            uuids.append(str(uuidlib.UUID(bytes=value[offset + 8 : offset + 24])).upper())
        elif command_without_required_bit in DYLIB_LOAD_COMMANDS:
            if command_size < 24:
                _fail(f"{source} has a truncated Mach-O dylib command")
            (name_offset,) = struct.unpack_from(f"{endian}I", value, offset + 8)
            if name_offset < 24 or name_offset >= command_size:
                _fail(f"{source} has an invalid Mach-O dylib name offset")
            name_start = offset + name_offset
            name_end = value.find(b"\0", name_start, offset + command_size)
            if name_end < 0:
                name_end = offset + command_size
            try:
                loads.add(value[name_start:name_end].decode("utf-8", errors="strict"))
            except UnicodeDecodeError as error:
                _fail(f"{source} has a non-UTF-8 Mach-O dylib path: {error}")
        offset += command_size
    if offset != commands_end:
        _fail(f"{source} Mach-O load commands do not match sizeofcmds")
    if len(uuids) > 1:
        _fail(f"{source} Mach-O slice has duplicate LC_UUID commands")
    return MachOSlice(
        file_type=file_type,
        uuid=uuids[0] if uuids else None,
        dylib_loads=frozenset(loads),
    )


def _macho_slices(value: bytes, source: str) -> list[MachOSlice]:
    if len(value) < 4:
        _fail(f"{source} is empty or truncated")
    if value[:4] in MACHO_MAGICS:
        return [_thin_macho_slice(value, source)]

    fat_metadata = FAT_MAGICS.get(value[:4])
    if fat_metadata is None:
        _fail(f"{source} is not a supported Mach-O image")
    endian, is_64_bit = fat_metadata
    if len(value) < 8:
        _fail(f"{source} has a truncated universal Mach-O header")
    (architecture_count,) = struct.unpack_from(f"{endian}I", value, 4)
    architecture_size = 32 if is_64_bit else 20
    if architecture_count == 0 or architecture_count > 128:
        _fail(f"{source} has an invalid universal Mach-O architecture count")
    architecture_table_end = 8 + architecture_count * architecture_size
    if architecture_table_end > len(value):
        _fail(f"{source} has a truncated universal Mach-O architecture table")

    slices: list[MachOSlice] = []
    slice_ranges: list[tuple[int, int]] = []
    for index in range(architecture_count):
        architecture_offset = 8 + index * architecture_size
        if is_64_bit:
            slice_offset, slice_size = struct.unpack_from(
                f"{endian}QQ", value, architecture_offset + 8
            )
        else:
            slice_offset, slice_size = struct.unpack_from(
                f"{endian}II", value, architecture_offset + 8
            )
        slice_end = slice_offset + slice_size
        if (
            slice_size == 0
            or slice_offset < architecture_table_end
            or slice_end > len(value)
        ):
            _fail(f"{source} has invalid universal Mach-O slice bounds")
        if any(slice_offset < end and start < slice_end for start, end in slice_ranges):
            _fail(f"{source} has overlapping universal Mach-O slices")
        slice_ranges.append((slice_offset, slice_end))
        slices.append(
            _thin_macho_slice(
                value[slice_offset:slice_end],
                f"{source} slice {index}",
            )
        )
    return slices


def _macho_dylib_loads_by_slice(value: bytes, source: str) -> list[frozenset[str]]:
    return [slice_metadata.dylib_loads for slice_metadata in _macho_slices(value, source)]


def _required_uuid_set(slices: list[MachOSlice], source: str) -> set[str]:
    uuids: set[str] = set()
    for index, slice_metadata in enumerate(slices):
        if slice_metadata.uuid is None:
            _fail(f"{source} slice {index} is missing LC_UUID")
        if slice_metadata.uuid in uuids:
            _fail(f"{source} contains duplicate UUID {slice_metadata.uuid}")
        uuids.add(slice_metadata.uuid)
    return uuids


def _dsym_uuid_index(archive_path: Path) -> dict[str, Path]:
    if archive_path.suffix != ".xcarchive" or not archive_path.is_dir():
        _fail(f"dSYM coverage requires an existing .xcarchive: {archive_path}")
    dwarf_root = archive_path / "dSYMs"
    dwarf_files = sorted(
        path
        for path in dwarf_root.glob("*.dSYM/Contents/Resources/DWARF/*")
        if path.is_file()
    )
    if not dwarf_files:
        _fail(f"{archive_path} contains no dSYM DWARF binaries")

    index: dict[str, Path] = {}
    for dwarf_path in dwarf_files:
        try:
            value = dwarf_path.read_bytes()
        except OSError as error:
            _fail(f"could not read dSYM DWARF binary {dwarf_path}: {error}")
        slices = _macho_slices(value, str(dwarf_path))
        if {slice_metadata.file_type for slice_metadata in slices} != {MH_DSYM}:
            _fail(f"{dwarf_path} must contain only MH_DSYM slices")
        for uuid in _required_uuid_set(slices, str(dwarf_path)):
            previous = index.get(uuid)
            if previous is not None and previous != dwarf_path:
                _fail(
                    f"dSYM UUID {uuid} is ambiguously owned by {previous} and {dwarf_path}"
                )
            index[uuid] = dwarf_path
    return index


def _validate_dsym_coverage(
    reader: DirectoryBundleReader | IPABundleReader,
    executable: str,
    archive_path: Path,
) -> None:
    symbol_index = _dsym_uuid_index(archive_path)
    required: list[tuple[str, set[str]]] = []

    app_source = f"{reader.source}/{executable}"
    app_slices = _macho_slices(reader.read_bytes(executable), app_source)
    if {slice_metadata.file_type for slice_metadata in app_slices} != {MH_EXECUTE}:
        _fail(f"{app_source} must contain only MH_EXECUTE slices")
    required.append((executable, _required_uuid_set(app_slices, app_source)))

    for relative_path, value in reader.framework_binaries():
        framework_source = f"{reader.source}/{relative_path}"
        if value.startswith(STATIC_ARCHIVE_MAGIC):
            print(f"ok: static framework symbol coverage inherited by app={relative_path}")
            continue
        slices = _macho_slices(value, framework_source)
        file_types = {slice_metadata.file_type for slice_metadata in slices}
        if file_types == {MH_OBJECT}:
            print(f"ok: object framework symbol coverage inherited by app={relative_path}")
            continue
        if file_types != {MH_DYLIB}:
            _fail(
                f"{framework_source} must contain only MH_DYLIB slices, only "
                "MH_OBJECT slices, or a static archive"
            )
        required.append(
            (relative_path, _required_uuid_set(slices, framework_source))
        )

    missing: list[str] = []
    required_uuid_count = 0
    for relative_path, uuids in required:
        required_uuid_count += len(uuids)
        for uuid in sorted(uuids):
            if uuid not in symbol_index:
                missing.append(f"{relative_path} UUID {uuid}")
    if missing:
        _fail(
            f"{reader.source} is missing matching archive dSYM coverage for: "
            + "; ".join(missing)
        )
    print(
        "ok: dSYM UUID coverage "
        f"binaries={len(required)} UUIDs={required_uuid_count} archive={archive_path}"
    )


def _validate_msal(
    reader: DirectoryBundleReader | IPABundleReader,
    executable: str,
    expected_msal_version: str,
) -> None:
    framework_info = _read_plist_bytes(
        reader.read_bytes(MSAL_FRAMEWORK_INFO_PATH),
        f"{reader.source}/{MSAL_FRAMEWORK_INFO_PATH}",
    )
    framework_bundle_id = str(framework_info.get("CFBundleIdentifier") or "").strip()
    if framework_bundle_id != MSAL_BUNDLE_IDENTIFIER:
        _fail(
            f"{reader.source}/{MSAL_FRAMEWORK_INFO_PATH} CFBundleIdentifier="
            f"{framework_bundle_id!r}; expected {MSAL_BUNDLE_IDENTIFIER!r}"
        )
    framework_version = str(
        framework_info.get("CFBundleShortVersionString") or ""
    ).strip()
    if framework_version != expected_msal_version:
        _fail(
            f"{reader.source} embedded MSAL version={framework_version!r}; "
            f"expected pinned version {expected_msal_version!r}"
        )
    if not reader.read_bytes(MSAL_FRAMEWORK_BINARY_PATH):
        _fail(f"{reader.source}/{MSAL_FRAMEWORK_BINARY_PATH} is empty")

    app_executable = reader.read_bytes(executable)
    dylib_loads_by_slice = _macho_dylib_loads_by_slice(
        app_executable, f"{reader.source}/{executable}"
    )
    missing_slice_count = sum(
        MSAL_LOAD_PATH not in dylib_loads for dylib_loads in dylib_loads_by_slice
    )
    if missing_slice_count:
        _fail(
            f"{reader.source}/{executable} has {missing_slice_count} Mach-O slice(s) "
            f"without a load command for {MSAL_LOAD_PATH}"
        )
    print(f"ok: embedded and linked MSAL version={framework_version}")


def _validate_bundle(
    reader: DirectoryBundleReader | IPABundleReader,
    *,
    expected_bundle_identifier: str,
    expected_bundle_version: str,
    expected_marketing_version: str,
    expected_build_configuration: str,
    expected_source_revision: str | None,
    expected_msal_version: str,
    require_dsym_archive: Path | None,
) -> None:
    info = _read_plist_bytes(reader.read_bytes("Info.plist"), f"{reader.source}/Info.plist")
    executable = _validate_info(
        info,
        reader.source,
        expected_bundle_identifier=expected_bundle_identifier,
        expected_bundle_version=expected_bundle_version,
        expected_marketing_version=expected_marketing_version,
        expected_build_configuration=expected_build_configuration,
        expected_source_revision=expected_source_revision,
    )
    _validate_privacy_manifest(reader)
    _validate_agent_manifest(reader)
    _validate_msal(reader, executable, expected_msal_version)
    if require_dsym_archive is not None:
        _validate_dsym_coverage(reader, executable, require_dsym_archive)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="Path to a .app, .xcarchive, or .ipa")
    parser.add_argument(
        "--expected-bundle-identifier",
        default=DEFAULT_BUNDLE_IDENTIFIER,
        help=f"Required CFBundleIdentifier (default: {DEFAULT_BUNDLE_IDENTIFIER}).",
    )
    parser.add_argument(
        "--expected-bundle-version",
        help="Required CFBundleVersion; defaults to CURRENT_PROJECT_VERSION in the project.",
    )
    parser.add_argument(
        "--expected-marketing-version",
        help="Required CFBundleShortVersionString; defaults to MARKETING_VERSION in the project.",
    )
    parser.add_argument(
        "--expected-build-configuration",
        default="Release",
        help="Required LumenBuildConfiguration (default: Release; pass Debug explicitly for developer builds).",
    )
    parser.add_argument(
        "--expected-source-revision",
        help="Required full Git source revision; defaults to the current repository HEAD for non-Debug builds.",
    )
    parser.add_argument(
        "--expected-msal-version",
        help="Required embedded MSAL version; defaults to the exact Package.resolved pin.",
    )
    parser.add_argument(
        "--require-dsym-archive",
        type=Path,
        help=(
            "Require every app/framework Mach-O UUID in the checked artifact to be "
            "covered by a matching dSYM in this .xcarchive."
        ),
    )
    args = parser.parse_args()

    path = args.path.resolve()
    if not path.exists():
        _fail(f"path does not exist: {path}")

    root = _repository_root()
    expected_bundle_version = args.expected_bundle_version or _project_setting(
        root, "CURRENT_PROJECT_VERSION"
    )
    expected_marketing_version = args.expected_marketing_version or _project_setting(
        root, "MARKETING_VERSION"
    )
    expected_source_revision = (
        args.expected_source_revision.lower()
        if args.expected_source_revision
        else (
            None
            if args.expected_build_configuration == "Debug"
            else _current_source_revision(root)
        )
    )
    if (
        args.expected_build_configuration != "Debug"
        and expected_source_revision is not None
        and not FULL_GIT_REVISION.fullmatch(expected_source_revision)
    ):
        _fail(
            "--expected-source-revision must be a full 40- or 64-character "
            "lowercase Git object ID for non-Debug validation"
        )
    expected_msal_version = args.expected_msal_version or _pinned_msal_version(root)

    validation_kwargs = {
        "expected_bundle_identifier": args.expected_bundle_identifier,
        "expected_bundle_version": expected_bundle_version,
        "expected_marketing_version": expected_marketing_version,
        "expected_build_configuration": args.expected_build_configuration,
        "expected_source_revision": expected_source_revision,
        "expected_msal_version": expected_msal_version,
        "require_dsym_archive": (
            args.require_dsym_archive.resolve()
            if args.require_dsym_archive is not None
            else None
        ),
    }
    if path.suffix == ".ipa":
        try:
            with zipfile.ZipFile(path) as archive:
                _validate_bundle(IPABundleReader(path, archive), **validation_kwargs)
        except zipfile.BadZipFile as error:
            _fail(f"invalid IPA zip archive {path}: {error}")
    else:
        _validate_bundle(DirectoryBundleReader(_find_app(path)), **validation_kwargs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
