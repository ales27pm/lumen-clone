#!/usr/bin/env python3
"""Validate privacy/build metadata in a final built Lumen app bundle."""

from __future__ import annotations

import argparse
import plistlib
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import NoReturn


REQUIRED_ALARM_KEY = "NSAlarmKitUsageDescription"


def _fail(message: str) -> NoReturn:
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(1)


def _find_app(path: Path) -> Path:
    if path.suffix == ".app" and path.is_dir():
        return path
    if path.suffix == ".xcarchive":
        apps_dir = path / "Products" / "Applications"
        apps = sorted(apps_dir.glob("*.app"))
        if apps:
            return apps[0]
    _fail(f"could not locate .app bundle in {path}")


def _read_info_plist(app_path: Path) -> dict:
    plist_path = app_path / "Info.plist"
    if not plist_path.is_file():
        _fail(f"missing Info.plist in {app_path}")
    with plist_path.open("rb") as handle:
        return plistlib.load(handle)


def _validate_info(info: dict, source: str) -> None:
    alarm_value = str(info.get(REQUIRED_ALARM_KEY) or "").strip()
    if not alarm_value:
        _fail(f"{source} Info.plist missing {REQUIRED_ALARM_KEY}")

    bundle_id = str(info.get("CFBundleIdentifier") or "").strip()
    bundle_version = str(info.get("CFBundleVersion") or "").strip()
    if not bundle_id:
        _fail(f"{source} Info.plist missing CFBundleIdentifier")
    if not bundle_version:
        _fail(f"{source} Info.plist missing CFBundleVersion")

    print(f"ok: {source} contains {REQUIRED_ALARM_KEY}={alarm_value!r}")
    print(f"ok: bundleIdentifier={bundle_id} CFBundleVersion={bundle_version}")
    for key in ("LumenBuildSourceIdentifier", "LumenBuildConfiguration", "LumenBuildScheme", "LumenGitSHA"):
        value = str(info.get(key) or "").strip()
        if not value or value.casefold() == "unknown":
            _fail(f"{source} Info.plist missing required build metadata {key}")
        print(f"ok: {key}={value}")


def _validate_ipa(path: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="lumen-ipa-plist-") as tmp:
        tmp_path = Path(tmp)
        with zipfile.ZipFile(path) as archive:
            app_info_members = [
                name
                for name in archive.namelist()
                if name.startswith("Payload/") and name.endswith(".app/Info.plist")
            ]
            if not app_info_members:
                _fail(f"could not locate Payload/*.app/Info.plist in {path}")
            member = app_info_members[0]
            archive.extract(member, tmp_path)
        with (tmp_path / member).open("rb") as handle:
            info = plistlib.load(handle)
    _validate_info(info, str(path))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="Path to a .app, .xcarchive, or .ipa")
    args = parser.parse_args()

    path = args.path.resolve()
    if not path.exists():
        _fail(f"path does not exist: {path}")
    if path.suffix == ".ipa":
        _validate_ipa(path)
        return 0
    app = _find_app(path)
    _validate_info(_read_info_plist(app), str(app))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
