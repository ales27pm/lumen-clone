import hashlib
import importlib.util
import io
import plistlib
import shutil
import stat
import struct
import sys
import zipfile
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "install_msal_release_dsym.py"
UUID = "00112233-4455-6677-8899-AABBCCDDEEFF"
UUID_BYTES = bytes.fromhex("00112233445566778899aabbccddeeff")


def _load_module():
    spec = importlib.util.spec_from_file_location("install_msal_release_dsym", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _macho(file_type: int) -> bytes:
    command = struct.pack("<II16s", 0x1B, 24, UUID_BYTES)
    header = struct.pack(
        "<IiiIIIII",
        0xFEEDFACF,
        0x0100000C,
        0,
        file_type,
        1,
        len(command),
        0,
        0,
    )
    return header + command


def _write_archive(tmp_path: Path) -> tuple[Path, Path]:
    archive = tmp_path / "Lumen.xcarchive"
    framework = (
        archive
        / "Products"
        / "Applications"
        / "Lumen.app"
        / "Frameworks"
        / "MSAL.framework"
    )
    framework.mkdir(parents=True)
    with (framework / "Info.plist").open("wb") as handle:
        plistlib.dump({"CFBundleShortVersionString": "1.9.0"}, handle)
    (framework / "MSAL").write_bytes(_macho(0x6))
    return archive, framework / "MSAL"


def _dsym_info() -> bytes:
    return plistlib.dumps(
        {
            "CFBundlePackageType": "dSYM",
            "CFBundleIdentifier": "com.apple.xcode.dsym.com.microsoft.MSAL",
            "CFBundleShortVersionString": "1.9.0",
        }
    )


def _write_zip(
    path: Path,
    *,
    prefix: str = "Users/runner/work/1/b/iOS.xcarchive/dSYMs",
    extra_entries: list[tuple[zipfile.ZipInfo | str, bytes]] | None = None,
) -> bytes:
    dwarf = _macho(0xA)
    root = f"{prefix}/MSAL.framework.dSYM"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(f"{root}/Contents/Info.plist", _dsym_info())
        archive.writestr(f"{root}/Contents/Resources/DWARF/MSAL", dwarf)
        for name, value in extra_entries or []:
            archive.writestr(name, value)
    return dwarf


def _asset(module, zip_path: Path, dwarf: bytes):
    return module.ReleaseAsset(
        version="1.9.0",
        url="https://example.invalid/MSAL.zip",
        zip_sha256=hashlib.sha256(zip_path.read_bytes()).hexdigest(),
        zip_size=zip_path.stat().st_size,
        dwarf_sha256=hashlib.sha256(dwarf).hexdigest(),
        uuid=UUID,
        architecture="arm64",
    )


def _stub_matching_dwarfdump(monkeypatch, module):
    monkeypatch.setattr(
        module,
        "_dwarfdump_uuids",
        lambda _path: {(UUID, "arm64")},
    )


def test_installer_pins_the_official_msal_release_asset():
    module = _load_module()

    assert module.MSAL_RELEASE_ASSET.version == "1.9.0"
    assert module.MSAL_RELEASE_ASSET.zip_sha256 == (
        "ecbb4f3c1e8f7e943cd7bf304b2cbe053bfc9998d41848480d6438218cfb6e12"
    )
    assert module.MSAL_RELEASE_ASSET.dwarf_sha256 == (
        "eb1d565cbf3b9f0b7cc6eadeb126bcabe94279b460f556ac104890e8695ce492"
    )
    assert module.MSAL_RELEASE_ASSET.uuid == "67EC8882-4D2F-33F6-88ED-8B8CEAED65B3"
    assert module.MSAL_RELEASE_ASSET.url.startswith("https://github.com/AzureAD/")


def test_installer_places_a_verified_matching_dsym_atomically(
    tmp_path: Path,
    monkeypatch,
):
    module = _load_module()
    archive, _ = _write_archive(tmp_path)
    source_zip = tmp_path / "MSAL.zip"
    dwarf = _write_zip(source_zip)
    asset = _asset(module, source_zip, dwarf)
    _stub_matching_dwarfdump(monkeypatch, module)

    installed = module.install_msal_release_dsym(
        archive,
        cache_dir=tmp_path / "cache",
        source_zip=source_zip,
        asset=asset,
    )

    assert installed == archive / "dSYMs" / "MSAL.framework.dSYM"
    assert (installed / "Contents" / "Resources" / "DWARF" / "MSAL").read_bytes() == dwarf
    assert not list((archive / "dSYMs").glob(".msal-dsym-stage.*"))


def test_installer_uses_verified_cache_offline_without_network(
    tmp_path: Path,
    monkeypatch,
):
    module = _load_module()
    archive, _ = _write_archive(tmp_path)
    source_zip = tmp_path / "source.zip"
    dwarf = _write_zip(source_zip)
    asset = _asset(module, source_zip, dwarf)
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    cached = cache_dir / f"MSAL-iOS.framework.dSYM-{asset.zip_sha256[:16]}.zip"
    shutil.copyfile(source_zip, cached)
    _stub_matching_dwarfdump(monkeypatch, module)
    monkeypatch.setattr(
        module.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: pytest.fail("offline install attempted network access"),
    )

    module.install_msal_release_dsym(
        archive,
        cache_dir=cache_dir,
        offline=True,
        asset=asset,
    )

    assert (archive / "dSYMs" / "MSAL.framework.dSYM").is_dir()


def test_installer_rejects_bad_zip_hash_before_archive_mutation(
    tmp_path: Path,
    monkeypatch,
):
    module = _load_module()
    archive, _ = _write_archive(tmp_path)
    source_zip = tmp_path / "MSAL.zip"
    dwarf = _write_zip(source_zip)
    asset = _asset(module, source_zip, dwarf)
    bad_asset = module.ReleaseAsset(
        **{**asset.__dict__, "zip_sha256": "0" * 64}
    )
    _stub_matching_dwarfdump(monkeypatch, module)

    with pytest.raises(module.InstallerError, match="does not match pinned digest"):
        module.install_msal_release_dsym(
            archive,
            cache_dir=tmp_path / "cache",
            source_zip=source_zip,
            asset=bad_asset,
        )

    assert not (archive / "dSYMs" / "MSAL.framework.dSYM").exists()


def test_installer_offline_missing_cache_never_attempts_network(
    tmp_path: Path,
    monkeypatch,
):
    module = _load_module()
    archive, _ = _write_archive(tmp_path)
    source_zip = tmp_path / "asset-shape.zip"
    dwarf = _write_zip(source_zip)
    asset = _asset(module, source_zip, dwarf)
    _stub_matching_dwarfdump(monkeypatch, module)
    monkeypatch.setattr(
        module.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: pytest.fail("offline install attempted network access"),
    )

    with pytest.raises(module.InstallerError, match="offline mode requires"):
        module.install_msal_release_dsym(
            archive,
            cache_dir=tmp_path / "missing-cache",
            offline=True,
            asset=asset,
        )

    assert not (archive / "dSYMs" / "MSAL.framework.dSYM").exists()


def test_installer_rejects_wrong_dwarf_hash_before_installation(
    tmp_path: Path,
    monkeypatch,
):
    module = _load_module()
    archive, _ = _write_archive(tmp_path)
    source_zip = tmp_path / "MSAL.zip"
    dwarf = _write_zip(source_zip)
    asset = _asset(module, source_zip, dwarf)
    bad_asset = module.ReleaseAsset(
        **{**asset.__dict__, "dwarf_sha256": "0" * 64}
    )
    _stub_matching_dwarfdump(monkeypatch, module)

    with pytest.raises(module.InstallerError, match="DWARF SHA-256"):
        module.install_msal_release_dsym(
            archive,
            cache_dir=tmp_path / "cache",
            source_zip=source_zip,
            asset=bad_asset,
        )

    assert not (archive / "dSYMs" / "MSAL.framework.dSYM").exists()


@pytest.mark.parametrize(
    "extra_entries,error",
    [
        ([('../escape', b'escape')], "unsafe member"),
        ([('/absolute', b'absolute')], "absolute member"),
        ([('bad\\member', b'backslash')], "unsafe member"),
        ([('outside.txt', b'outside')], "outside the dSYM bundle"),
        (
            [
                (
                    'Users/runner/work/1/b/iOS.xcarchive/dSYMs/'
                    'MSAL.framework.dSYM/Contents/extra',
                    b'one',
                ),
                (
                    'Users/runner/work/1/b/iOS.xcarchive/dSYMs/'
                    'MSAL.framework.dSYM/Contents/EXTRA',
                    b'two',
                ),
            ],
            "case-colliding",
        ),
    ],
)
def test_installer_rejects_unsafe_zip_layouts(
    tmp_path: Path,
    monkeypatch,
    extra_entries,
    error: str,
):
    module = _load_module()
    archive, _ = _write_archive(tmp_path)
    source_zip = tmp_path / "MSAL.zip"
    dwarf = _write_zip(source_zip, extra_entries=extra_entries)
    asset = _asset(module, source_zip, dwarf)
    _stub_matching_dwarfdump(monkeypatch, module)

    with pytest.raises(module.InstallerError, match=error):
        module.install_msal_release_dsym(
            archive,
            cache_dir=tmp_path / "cache",
            source_zip=source_zip,
            asset=asset,
        )

    assert not (archive / "dSYMs" / "MSAL.framework.dSYM").exists()


def test_installer_rejects_symlinks_and_multiple_dsym_roots(
    tmp_path: Path,
    monkeypatch,
):
    module = _load_module()
    _stub_matching_dwarfdump(monkeypatch, module)

    symlink_archive, _ = _write_archive(tmp_path / "symlink")
    symlink_zip = tmp_path / "symlink.zip"
    symlink_info = zipfile.ZipInfo(
        "Users/runner/work/1/b/iOS.xcarchive/dSYMs/"
        "MSAL.framework.dSYM/Contents/link"
    )
    symlink_info.create_system = 3
    symlink_info.external_attr = (stat.S_IFLNK | 0o777) << 16
    symlink_dwarf = _write_zip(
        symlink_zip,
        extra_entries=[(symlink_info, b"target")],
    )
    symlink_asset = _asset(module, symlink_zip, symlink_dwarf)

    with pytest.raises(module.InstallerError, match="symlink or special"):
        module.install_msal_release_dsym(
            symlink_archive,
            cache_dir=tmp_path / "cache-a",
            source_zip=symlink_zip,
            asset=symlink_asset,
        )


def test_installer_rejects_entry_count_and_expanded_size_limits(
    tmp_path: Path,
    monkeypatch,
):
    module = _load_module()
    _stub_matching_dwarfdump(monkeypatch, module)

    count_archive, _ = _write_archive(tmp_path / "count")
    count_zip = tmp_path / "count.zip"
    count_dwarf = _write_zip(count_zip)
    count_asset = _asset(module, count_zip, count_dwarf)
    monkeypatch.setattr(module, "MAX_ZIP_ENTRIES", 1)
    with pytest.raises(module.InstallerError, match="entry count"):
        module.install_msal_release_dsym(
            count_archive,
            cache_dir=tmp_path / "cache-count",
            source_zip=count_zip,
            asset=count_asset,
        )

    size_archive, _ = _write_archive(tmp_path / "size")
    size_zip = tmp_path / "size.zip"
    size_dwarf = _write_zip(size_zip)
    size_asset = _asset(module, size_zip, size_dwarf)
    monkeypatch.setattr(module, "MAX_ZIP_ENTRIES", 128)
    monkeypatch.setattr(module, "MAX_EXPANDED_SIZE", 1)
    with pytest.raises(module.InstallerError, match="expanded size"):
        module.install_msal_release_dsym(
            size_archive,
            cache_dir=tmp_path / "cache-size",
            source_zip=size_zip,
            asset=size_asset,
        )


def test_download_is_published_only_after_full_hash_verification(
    tmp_path: Path,
    monkeypatch,
):
    module = _load_module()
    source_zip = tmp_path / "source.zip"
    dwarf = _write_zip(source_zip)
    asset = _asset(module, source_zip, dwarf)
    cache_path = tmp_path / "cache" / "asset.zip"
    payload = source_zip.read_bytes()
    monkeypatch.setattr(
        module.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: io.BytesIO(payload),
    )

    result = module._download_verified_asset(cache_path, asset)

    assert result == cache_path
    assert cache_path.read_bytes() == payload
    assert not list(cache_path.parent.glob(f".{cache_path.name}.*"))

    cache_path.unlink()
    monkeypatch.setattr(
        module.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: io.BytesIO(payload + b"tampered"),
    )
    with pytest.raises(module.InstallerError, match="could not download verified"):
        module._download_verified_asset(cache_path, asset)
    assert not cache_path.exists()

    multiple_archive, _ = _write_archive(tmp_path / "multiple")
    multiple_zip = tmp_path / "multiple.zip"
    multiple_dwarf = _write_zip(
        multiple_zip,
        extra_entries=[
            (
                "second/MSAL.framework.dSYM/Contents/Resources/DWARF/MSAL",
                _macho(0xA),
            )
        ],
    )
    multiple_asset = _asset(module, multiple_zip, multiple_dwarf)

    with pytest.raises(module.InstallerError, match="exactly one"):
        module.install_msal_release_dsym(
            multiple_archive,
            cache_dir=tmp_path / "cache-b",
            source_zip=multiple_zip,
            asset=multiple_asset,
        )


def test_installer_rejects_uuid_mismatch_and_preserves_existing_destination(
    tmp_path: Path,
    monkeypatch,
):
    module = _load_module()
    archive, embedded = _write_archive(tmp_path)
    source_zip = tmp_path / "MSAL.zip"
    dwarf = _write_zip(source_zip)
    asset = _asset(module, source_zip, dwarf)

    monkeypatch.setattr(
        module,
        "_dwarfdump_uuids",
        lambda path: (
            {(UUID, "arm64")}
            if path == embedded
            else {("FFEEDDCC-BBAA-9988-7766-554433221100", "arm64")}
        ),
    )
    with pytest.raises(module.InstallerError, match="do not match embedded"):
        module.install_msal_release_dsym(
            archive,
            cache_dir=tmp_path / "cache",
            source_zip=source_zip,
            asset=asset,
        )

    destination = archive / "dSYMs" / "MSAL.framework.dSYM"
    destination.mkdir(parents=True)
    marker = destination / "preserve-me"
    marker.write_text("stale", encoding="utf-8")
    with pytest.raises(module.InstallerError):
        module.install_msal_release_dsym(
            archive,
            cache_dir=tmp_path / "cache",
            source_zip=source_zip,
            asset=asset,
        )
    assert marker.read_text(encoding="utf-8") == "stale"


def test_installer_accepts_an_existing_exact_destination_idempotently(
    tmp_path: Path,
    monkeypatch,
):
    module = _load_module()
    archive, _ = _write_archive(tmp_path)
    source_zip = tmp_path / "MSAL.zip"
    dwarf = _write_zip(source_zip)
    asset = _asset(module, source_zip, dwarf)
    _stub_matching_dwarfdump(monkeypatch, module)

    first = module.install_msal_release_dsym(
        archive,
        cache_dir=tmp_path / "cache",
        source_zip=source_zip,
        asset=asset,
    )
    second = module.install_msal_release_dsym(
        archive,
        cache_dir=tmp_path / "missing-cache",
        offline=True,
        asset=asset,
    )

    assert first == second


def test_dwarfdump_parser_fails_closed_on_malformed_output(tmp_path: Path, monkeypatch):
    module = _load_module()

    class Result:
        returncode = 0
        stdout = "warning then UUID"
        stderr = ""

    monkeypatch.setattr(module.subprocess, "run", lambda *_args, **_kwargs: Result())

    with pytest.raises(module.InstallerError, match="unexpected dwarfdump output"):
        module._dwarfdump_uuids(tmp_path / "binary")
