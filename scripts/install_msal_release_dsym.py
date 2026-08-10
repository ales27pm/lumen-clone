#!/usr/bin/env python3
"""Install the hash-pinned official MSAL dSYM into a Lumen xcarchive."""

from __future__ import annotations

import argparse
import hashlib
import os
import plistlib
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath


@dataclass(frozen=True)
class ReleaseAsset:
    version: str
    url: str
    zip_sha256: str
    zip_size: int
    dwarf_sha256: str
    uuid: str
    architecture: str


MSAL_RELEASE_ASSET = ReleaseAsset(
    version="1.9.0",
    url=(
        "https://github.com/AzureAD/"
        "microsoft-authentication-library-for-objc/releases/download/"
        "1.9.0/MSAL-iOS.framework.dSYM.zip"
    ),
    zip_sha256="ecbb4f3c1e8f7e943cd7bf304b2cbe053bfc9998d41848480d6438218cfb6e12",
    zip_size=3_300_871,
    dwarf_sha256="eb1d565cbf3b9f0b7cc6eadeb126bcabe94279b460f556ac104890e8695ce492",
    uuid="67EC8882-4D2F-33F6-88ED-8B8CEAED65B3",
    architecture="arm64",
)

MAX_ZIP_ENTRIES = 128
MAX_EXPANDED_SIZE = 64 * 1024 * 1024
SUPPORTED_COMPRESSIONS = {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}
UUID_LINE = re.compile(
    r"^UUID: ([0-9A-Fa-f]{8}(?:-[0-9A-Fa-f]{4}){3}-[0-9A-Fa-f]{12}) "
    r"\(([^)]+)\) .+$"
)


class InstallerError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise InstallerError(f"could not read {path}: {error}") from error
    return digest.hexdigest()


def _verify_zip(path: Path, asset: ReleaseAsset) -> None:
    if not path.is_file():
        raise InstallerError(f"MSAL dSYM ZIP does not exist: {path}")
    try:
        size = path.stat().st_size
    except OSError as error:
        raise InstallerError(f"could not inspect {path}: {error}") from error
    if size != asset.zip_size:
        raise InstallerError(
            f"MSAL dSYM ZIP size {size} does not match pinned size {asset.zip_size}: {path}"
        )
    digest = _sha256(path)
    if digest != asset.zip_sha256:
        raise InstallerError(
            f"MSAL dSYM ZIP SHA-256 {digest} does not match pinned digest "
            f"{asset.zip_sha256}: {path}"
        )


def _download_verified_asset(
    cache_path: Path,
    asset: ReleaseAsset,
) -> Path:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for attempt in range(2):
        temporary_path: Path | None = None
        try:
            request = urllib.request.Request(
                asset.url,
                headers={"User-Agent": "Lumen-release-symbol-installer/1"},
            )
            with urllib.request.urlopen(request, timeout=60) as response:
                with tempfile.NamedTemporaryFile(
                    mode="wb",
                    prefix=f".{cache_path.name}.",
                    dir=cache_path.parent,
                    delete=False,
                ) as handle:
                    temporary_path = Path(handle.name)
                    total = 0
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        total += len(chunk)
                        if total > asset.zip_size:
                            raise InstallerError(
                                "MSAL dSYM download exceeded the pinned asset size"
                            )
                        handle.write(chunk)
                    handle.flush()
                    os.fsync(handle.fileno())
            _verify_zip(temporary_path, asset)
            os.replace(temporary_path, cache_path)
            return cache_path
        except (OSError, urllib.error.URLError, InstallerError) as error:
            last_error = error
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
            if attempt == 1:
                break
    raise InstallerError(f"could not download verified MSAL dSYM asset: {last_error}")


def _select_zip(
    *,
    cache_dir: Path,
    source_zip: Path | None,
    offline: bool,
    asset: ReleaseAsset,
) -> Path:
    if source_zip is not None:
        path = source_zip.resolve()
        _verify_zip(path, asset)
        return path

    cache_path = cache_dir.resolve() / (
        f"MSAL-iOS.framework.dSYM-{asset.zip_sha256[:16]}.zip"
    )
    if cache_path.is_file():
        try:
            _verify_zip(cache_path, asset)
            return cache_path
        except InstallerError:
            if offline:
                raise
    if offline:
        raise InstallerError(
            f"offline mode requires a valid pinned MSAL dSYM cache at {cache_path}"
        )
    return _download_verified_asset(cache_path, asset)


def _safe_zip_parts(name: str) -> tuple[str, ...]:
    if not name or "\0" in name or "\\" in name:
        raise InstallerError(f"MSAL dSYM ZIP contains unsafe member name {name!r}")
    if name.startswith("/"):
        raise InstallerError(f"MSAL dSYM ZIP contains absolute member {name!r}")
    raw_parts = name.split("/")
    if raw_parts[-1] == "":
        raw_parts = raw_parts[:-1]
    if not raw_parts or any(part in {"", ".", ".."} for part in raw_parts):
        raise InstallerError(f"MSAL dSYM ZIP contains unsafe member name {name!r}")
    pure = PurePosixPath(*raw_parts)
    if pure.is_absolute() or pure.as_posix() != "/".join(raw_parts):
        raise InstallerError(f"MSAL dSYM ZIP contains non-canonical member {name!r}")
    return tuple(raw_parts)


def _extract_dsym(zip_path: Path, staging_root: Path) -> Path:
    try:
        archive = zipfile.ZipFile(zip_path)
    except (OSError, zipfile.BadZipFile) as error:
        raise InstallerError(f"invalid MSAL dSYM ZIP {zip_path}: {error}") from error

    with archive:
        infos = archive.infolist()
        if not infos or len(infos) > MAX_ZIP_ENTRIES:
            raise InstallerError(
                f"MSAL dSYM ZIP entry count {len(infos)} is outside the allowed range"
            )
        seen: set[str] = set()
        seen_casefolded: set[str] = set()
        roots: set[tuple[str, ...]] = set()
        parsed: list[tuple[zipfile.ZipInfo, tuple[str, ...]]] = []
        expanded_size = 0

        for info in infos:
            parts = _safe_zip_parts(info.filename)
            canonical = "/".join(parts)
            if canonical in seen or canonical.casefold() in seen_casefolded:
                raise InstallerError(
                    f"MSAL dSYM ZIP contains duplicate or case-colliding member {info.filename!r}"
                )
            seen.add(canonical)
            seen_casefolded.add(canonical.casefold())
            if info.flag_bits & 0x1:
                raise InstallerError(f"MSAL dSYM ZIP contains encrypted member {info.filename!r}")
            if info.compress_type not in SUPPORTED_COMPRESSIONS:
                raise InstallerError(
                    f"MSAL dSYM ZIP uses unsupported compression for {info.filename!r}"
                )
            unix_mode = info.external_attr >> 16
            file_type = stat.S_IFMT(unix_mode)
            if file_type not in {0, stat.S_IFREG, stat.S_IFDIR}:
                raise InstallerError(
                    f"MSAL dSYM ZIP contains symlink or special member {info.filename!r}"
                )
            expanded_size += info.file_size
            if expanded_size > MAX_EXPANDED_SIZE:
                raise InstallerError("MSAL dSYM ZIP expanded size exceeds the safety limit")
            indices = [
                index
                for index, part in enumerate(parts)
                if part == "MSAL.framework.dSYM"
            ]
            if len(indices) > 1:
                raise InstallerError(
                    f"MSAL dSYM ZIP member has ambiguous bundle roots: {info.filename!r}"
                )
            if indices:
                roots.add(parts[: indices[0] + 1])
            elif not info.is_dir():
                raise InstallerError(
                    f"MSAL dSYM ZIP contains file outside the dSYM bundle: {info.filename!r}"
                )
            parsed.append((info, parts))

        if len(roots) != 1:
            raise InstallerError(
                f"MSAL dSYM ZIP must contain exactly one MSAL.framework.dSYM root; found {len(roots)}"
            )
        root_parts = next(iter(roots))
        destination = staging_root / "MSAL.framework.dSYM"

        for info, parts in parsed:
            if parts[: len(root_parts)] != root_parts:
                continue
            relative_parts = parts[len(root_parts) :]
            target = destination.joinpath(*relative_parts)
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True, mode=0o755)
                continue
            target.parent.mkdir(parents=True, exist_ok=True, mode=0o755)
            try:
                with archive.open(info, "r") as source, target.open("xb") as output:
                    shutil.copyfileobj(source, output, length=1024 * 1024)
                target.chmod(0o644)
            except (OSError, RuntimeError, zipfile.BadZipFile) as error:
                raise InstallerError(
                    f"could not extract MSAL dSYM member {info.filename!r}: {error}"
                ) from error
        return destination


def _read_plist(path: Path) -> dict:
    try:
        with path.open("rb") as handle:
            value = plistlib.load(handle)
    except (OSError, plistlib.InvalidFileException, ValueError, TypeError) as error:
        raise InstallerError(f"could not read plist {path}: {error}") from error
    if not isinstance(value, dict):
        raise InstallerError(f"expected plist dictionary in {path}")
    return value


def _dwarfdump_uuids(path: Path) -> set[tuple[str, str]]:
    try:
        result = subprocess.run(
            ["/usr/bin/dwarfdump", "--uuid", str(path)],
            text=True,
            capture_output=True,
            check=False,
        )
    except OSError as error:
        raise InstallerError(f"could not execute dwarfdump for {path}: {error}") from error
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise InstallerError(f"dwarfdump failed for {path}: {detail}")
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    parsed: set[tuple[str, str]] = set()
    for line in lines:
        match = UUID_LINE.fullmatch(line)
        if match is None:
            raise InstallerError(f"unexpected dwarfdump output for {path}: {line!r}")
        parsed.add((match.group(1).upper(), match.group(2)))
    if not parsed or len(parsed) != len(lines):
        raise InstallerError(f"dwarfdump returned missing or duplicate UUIDs for {path}")
    return parsed


def _locate_archive_app(archive_path: Path) -> Path:
    if archive_path.suffix != ".xcarchive" or not archive_path.is_dir():
        raise InstallerError(f"expected an existing .xcarchive: {archive_path}")
    apps = sorted((archive_path / "Products" / "Applications").glob("*.app"))
    if len(apps) != 1:
        raise InstallerError(
            f"expected exactly one archived Products/Applications/*.app; found {len(apps)}"
        )
    return apps[0]


def _validate_msal_dsym(
    dsym_path: Path,
    embedded_binary: Path,
    asset: ReleaseAsset,
) -> None:
    info = _read_plist(dsym_path / "Contents" / "Info.plist")
    exact_values = {
        "CFBundlePackageType": "dSYM",
        "CFBundleIdentifier": "com.apple.xcode.dsym.com.microsoft.MSAL",
        "CFBundleShortVersionString": asset.version,
    }
    for key, expected in exact_values.items():
        actual = str(info.get(key) or "").strip()
        if actual != expected:
            raise InstallerError(
                f"MSAL dSYM {key}={actual!r}; expected pinned value {expected!r}"
            )
    dwarf = dsym_path / "Contents" / "Resources" / "DWARF" / "MSAL"
    digest = _sha256(dwarf)
    if digest != asset.dwarf_sha256:
        raise InstallerError(
            f"MSAL dSYM DWARF SHA-256 {digest} does not match pinned digest "
            f"{asset.dwarf_sha256}"
        )
    embedded_uuids = _dwarfdump_uuids(embedded_binary)
    dsym_uuids = _dwarfdump_uuids(dwarf)
    expected = {(asset.uuid, asset.architecture)}
    if embedded_uuids != expected:
        raise InstallerError(
            f"embedded MSAL UUIDs {sorted(embedded_uuids)} do not match pinned {sorted(expected)}"
        )
    if dsym_uuids != embedded_uuids:
        raise InstallerError(
            f"MSAL dSYM UUIDs {sorted(dsym_uuids)} do not match embedded "
            f"MSAL UUIDs {sorted(embedded_uuids)}"
        )


def install_msal_release_dsym(
    archive_path: Path,
    *,
    cache_dir: Path,
    source_zip: Path | None = None,
    offline: bool = False,
    asset: ReleaseAsset = MSAL_RELEASE_ASSET,
) -> Path:
    archive_path = archive_path.resolve()
    app = _locate_archive_app(archive_path)
    embedded_framework = app / "Frameworks" / "MSAL.framework"
    embedded_binary = embedded_framework / "MSAL"
    if not embedded_binary.is_file():
        raise InstallerError(f"archived app is missing embedded MSAL binary: {embedded_binary}")
    framework_info = _read_plist(embedded_framework / "Info.plist")
    embedded_version = str(
        framework_info.get("CFBundleShortVersionString") or ""
    ).strip()
    if embedded_version != asset.version:
        raise InstallerError(
            f"embedded MSAL version {embedded_version!r} does not match pinned "
            f"release-symbol version {asset.version!r}"
        )

    dsym_root = archive_path / "dSYMs"
    dsym_root.mkdir(parents=True, exist_ok=True, mode=0o755)
    destination = dsym_root / "MSAL.framework.dSYM"
    if destination.exists():
        _validate_msal_dsym(destination, embedded_binary, asset)
        print(
            "ok: existing official MSAL dSYM is valid "
            f"version={asset.version} UUID={asset.uuid}"
        )
        return destination

    zip_path = _select_zip(
        cache_dir=cache_dir,
        source_zip=source_zip,
        offline=offline,
        asset=asset,
    )
    staging_root = Path(tempfile.mkdtemp(prefix=".msal-dsym-stage.", dir=dsym_root))
    try:
        staged_dsym = _extract_dsym(zip_path, staging_root)
        _validate_msal_dsym(staged_dsym, embedded_binary, asset)
        if destination.exists():
            raise InstallerError(
                f"refusing to overwrite concurrently installed MSAL dSYM: {destination}"
            )
        os.replace(staged_dsym, destination)
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)

    print(
        "ok: installed official hash-pinned MSAL dSYM "
        f"version={asset.version} UUID={asset.uuid} archive={archive_path}"
    )
    return destination


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=root / "build" / "ReleaseSymbols" / "MSAL" / MSAL_RELEASE_ASSET.version,
    )
    parser.add_argument(
        "--source-zip",
        type=Path,
        help="Use this local ZIP after enforcing the same official size and SHA-256 pins.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Forbid network access; require a valid explicit ZIP, cache, or installed dSYM.",
    )
    args = parser.parse_args()
    try:
        install_msal_release_dsym(
            args.archive,
            cache_dir=args.cache_dir,
            source_zip=args.source_zip,
            offline=args.offline,
        )
    except InstallerError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
