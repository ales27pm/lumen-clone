#!/usr/bin/env python3
"""Validate iOS signing capabilities expected by App Store profiles.

This static guard catches capabilities that require special provisioning-profile
approval before Xcode reaches the archive signing phase.
"""
from __future__ import annotations

import argparse
import plistlib
import shutil
import subprocess
import sys
import tempfile
import zipfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

ALLOWED_CARPLAY_ENTITLEMENTS = {
    "com.apple.developer.carplay-voice-based-conversation",
}

APP_STORE_ONLY_SANITIZED_ENTITLEMENTS = {
    "com.apple.developer.kernel.increased-debugging-memory-limit": (
        "The increased debugging memory limit entitlement is only allowed for "
        "TestFlight Internal Only distribution."
    ),
    "com.apple.security.hardened-process.checked-allocations.soft-mode": (
        "Checked allocations soft mode is a development-only hardened runtime "
        "setting and should not be signed into App Store archives."
    ),
}

DISALLOWED_PROJECT_SETTINGS = {
    "INFOPLIST_KEY_UIApplicationSupportsCarPlay": (
        "Use the CPTemplateApplicationSceneSessionRoleApplication scene manifest "
        "for CarPlay voice support instead of the legacy UIApplicationSupportsCarPlay key."
    ),
    "com.apple.developer.carplay": (
        "CarPlay must not be enabled through stale Xcode SystemCapabilities entries. "
        "Use the checked-in voice-based conversation entitlement instead."
    ),
}

DISALLOWED_APP_SOURCE_TOKENS: dict[str, str] = {}

APP_SOURCE_SUFFIXES = {".swift", ".plist", ".entitlements", ".fragment"}

STANDARD_SIGNED_ENTITLEMENTS = {
    "application-identifier",
    "aps-environment",
    "beta-reports-active",
    "com.apple.application-identifier",
    "com.apple.developer.team-identifier",
    "com.apple.security.application-groups",
    "get-task-allow",
}


def repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[1]


def read_plist(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as handle:
            value = plistlib.load(handle)
    except FileNotFoundError:
        raise SystemExit(f"error: missing entitlements file: {path}")
    except plistlib.InvalidFileException as exc:
        raise SystemExit(f"error: invalid plist in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SystemExit(f"error: expected plist dictionary in {path}")
    return value


def read_plist_bytes(value: bytes, source: str) -> dict[str, Any]:
    try:
        parsed = plistlib.loads(value)
    except plistlib.InvalidFileException as exc:
        raise SystemExit(f"error: invalid plist from {source}: {exc}") from exc
    if not isinstance(parsed, dict):
        raise SystemExit(f"error: expected plist dictionary from {source}")
    return parsed


def sanitized_entitlements(entitlements: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Return entitlements with App-Store-profile-incompatible keys removed."""
    sanitized = dict(entitlements)
    removed: list[str] = []
    for key in list(sanitized):
        if sanitized_entitlement_message(key):
            sanitized.pop(key)
            removed.append(key)
    return sanitized, removed


def write_plist(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        plistlib.dump(value, handle, sort_keys=False)


def sanitize_entitlements_file(source: Path, destination: Path) -> list[str]:
    entitlements = read_plist(source)
    sanitized, removed = sanitized_entitlements(entitlements)
    write_plist(destination, sanitized)
    return removed


def disallowed_entitlement_message(key: str) -> str | None:
    if key.startswith("com.apple.developer.carplay") and key not in ALLOWED_CARPLAY_ENTITLEMENTS:
        return (
            "Only the CarPlay Voice Based Conversation entitlement is allowed. "
            "Remove stale or unsupported CarPlay entitlement keys."
        )
    return None


def sanitized_entitlement_message(key: str) -> str | None:
    if key in APP_STORE_ONLY_SANITIZED_ENTITLEMENTS:
        return APP_STORE_ONLY_SANITIZED_ENTITLEMENTS[key]
    return disallowed_entitlement_message(key)


def validate_entitlements(path: Path) -> list[str]:
    entitlements = read_plist(path)
    failures: list[str] = []
    for key in entitlements:
        if message := disallowed_entitlement_message(key):
            failures.append(f"{path}: disallowed entitlement '{key}'. {message}")
    return failures


def format_plist_value(value: Any) -> str:
    if isinstance(value, list):
        return "[" + ", ".join(format_plist_value(item) for item in value) + "]"
    return repr(value)


def value_matches_expected(expected: Any, actual: Any) -> bool:
    if expected == actual:
        return True
    if isinstance(expected, str) and "$(" in expected and isinstance(actual, str):
        return True
    if isinstance(expected, list) and isinstance(actual, list) and len(expected) == len(actual):
        return all(value_matches_expected(expected_item, actual_item) for expected_item, actual_item in zip(expected, actual))
    return False


def validate_app_store_entitlements_match(development_path: Path, app_store_path: Path) -> list[str]:
    development = read_plist(development_path)
    expected_app_store, removed = sanitized_entitlements(development)
    actual_app_store = read_plist(app_store_path)
    failures: list[str] = []
    if actual_app_store != expected_app_store:
        expected_keys = set(expected_app_store)
        actual_keys = set(actual_app_store)
        for key in sorted(expected_keys - actual_keys):
            failures.append(
                f"{app_store_path}: missing App Store entitlement '{key}' from sanitized {development_path}"
            )
        for key in sorted(actual_keys - expected_keys):
            failures.append(
                f"{app_store_path}: extra App Store entitlement '{key}' not present in sanitized {development_path}"
            )
        for key in sorted(expected_keys & actual_keys):
            expected_value = expected_app_store[key]
            actual_value = actual_app_store[key]
            if actual_value != expected_value:
                failures.append(
                    f"{app_store_path}: App Store entitlement '{key}' is {format_plist_value(actual_value)}; "
                    f"expected sanitized value {format_plist_value(expected_value)} from {development_path}"
                )
    if removed:
        print(
            "App Store entitlements match development entitlements after removing "
            f"{', '.join(sorted(removed))}.",
            file=sys.stderr,
        )
    return failures


def entitlements_to_validate(primary: Path, root: Path) -> list[Path]:
    """Return the primary entitlements plus every checked-in app entitlements file."""
    candidates = [primary, *(root / "ios" / "Lumen").glob("*.entitlements")]
    unique: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(candidate)
    return unique


def validate_project_settings(path: Path) -> list[str]:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise SystemExit(f"error: missing Xcode project file: {path}")
    failures: list[str] = []
    lines = text.splitlines()
    for token, message in DISALLOWED_PROJECT_SETTINGS.items():
        for index, line in enumerate(lines, start=1):
            if token in line:
                failures.append(f"{path}:{index}: disallowed project setting '{token}'. {message}")
    return failures


def app_source_files(root: Path) -> list[Path]:
    app_root = root / "ios" / "Lumen"
    return sorted(
        path
        for path in app_root.rglob("*")
        if path.is_file() and path.suffix in APP_SOURCE_SUFFIXES
    )


def validate_app_sources(root: Path) -> list[str]:
    failures: list[str] = []
    for path in app_source_files(root):
        text = path.read_text(encoding="utf-8")
        for token, message in DISALLOWED_APP_SOURCE_TOKENS.items():
            for index, line in enumerate(text.splitlines(), start=1):
                if token in line:
                    failures.append(f"{path}:{index}: disallowed CarPlay reference '{token}'. {message}")
    return failures


def _find_app(path: Path) -> Path:
    if path.suffix == ".app" and path.is_dir():
        return path
    if path.suffix == ".xcarchive":
        apps_dir = path / "Products" / "Applications"
        apps = sorted(apps_dir.glob("*.app"))
        if apps:
            return apps[0]
    raise SystemExit(f"error: could not locate .app bundle in {path}")


def _safe_extract_zip_member(archive: zipfile.ZipFile, member: str, destination: Path) -> None:
    target = (destination / member).resolve()
    root = destination.resolve()
    if root not in target.parents and target != root:
        raise SystemExit(f"error: refusing unsafe IPA member path: {member}")
    archive.extract(member, destination)


@contextmanager
def resolved_signed_app_path(path: Path) -> Iterator[Path]:
    if path.suffix == ".ipa":
        with tempfile.TemporaryDirectory(prefix="lumen-ipa-entitlements-") as tmp:
            tmp_path = Path(tmp)
            with zipfile.ZipFile(path) as archive:
                app_roots = sorted(
                    {
                        "/".join(member.split("/")[:2])
                        for member in archive.namelist()
                        if member.startswith("Payload/") and ".app/" in member
                    }
                )
                if not app_roots:
                    raise SystemExit(f"error: could not locate Payload/*.app in {path}")
                app_root = app_roots[0]
                for member in archive.namelist():
                    if member == app_root or member.startswith(f"{app_root}/"):
                        _safe_extract_zip_member(archive, member, tmp_path)
            yield tmp_path / app_root
            return
    yield _find_app(path)


def read_signed_entitlements(app_path: Path) -> dict[str, Any]:
    codesign = shutil.which("codesign")
    if sys.platform != "darwin" or not codesign:
        raise SystemExit("error: signed entitlement validation requires macOS codesign")
    result = subprocess.run(
        [codesign, "-d", "--entitlements", ":-", str(app_path)],
        text=False,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace").strip()
        raise SystemExit(f"error: codesign could not read entitlements from {app_path}: {stderr}")
    stdout = result.stdout.strip()
    if not stdout:
        raise SystemExit(f"error: codesign returned no entitlements for {app_path}")
    return read_plist_bytes(stdout, str(app_path))


def validate_signed_entitlements(
    signed_entitlements: dict[str, Any],
    expected_entitlements_path: Path,
    source: str,
) -> list[str]:
    expected_entitlements = read_plist(expected_entitlements_path)
    failures: list[str] = []
    for key in signed_entitlements:
        if message := sanitized_entitlement_message(key):
            failures.append(f"{source}: signed entitlement '{key}' must not be present. {message}")
        elif message := disallowed_entitlement_message(key):
            failures.append(f"{source}: disallowed signed entitlement '{key}'. {message}")

    for key, expected_value in expected_entitlements.items():
        if key not in signed_entitlements:
            failures.append(
                f"{source}: missing expected signed entitlement '{key}' from {expected_entitlements_path}"
            )
            continue
        actual_value = signed_entitlements[key]
        if not value_matches_expected(expected_value, actual_value):
            failures.append(
                f"{source}: signed entitlement '{key}' is {format_plist_value(actual_value)}; "
                f"expected {format_plist_value(expected_value)} from {expected_entitlements_path}"
            )

    extra_project_keys = sorted(
        key
        for key in signed_entitlements
        if key not in expected_entitlements
        and key not in STANDARD_SIGNED_ENTITLEMENTS
        and not key.startswith("com.apple.developer.associated-domains")
    )
    if extra_project_keys:
        print(
            f"warning: {source} has additional signed entitlements not tracked in "
            f"{expected_entitlements_path}: {', '.join(extra_project_keys)}",
            file=sys.stderr,
        )
    return failures


def validate_signed_app(path: Path, expected_entitlements_path: Path) -> list[str]:
    with resolved_signed_app_path(path) as app_path:
        entitlements = read_signed_entitlements(app_path)
        return validate_signed_entitlements(entitlements, expected_entitlements_path, str(app_path))


def main(argv: list[str] | None = None) -> int:
    root = repo_root_from_script()
    default_development_entitlements = root / "ios" / "Lumen" / "Lumen.entitlements"
    default_app_store_entitlements = root / "ios" / "Lumen" / "LumenAppStore.entitlements"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-file",
        type=Path,
        default=root / "ios" / "Lumen.xcodeproj" / "project.pbxproj",
        help="Path to project.pbxproj (default: ios/Lumen.xcodeproj/project.pbxproj)",
    )
    parser.add_argument(
        "--entitlements",
        type=Path,
        default=default_development_entitlements,
        help="Path to app entitlements plist (default: ios/Lumen/Lumen.entitlements)",
    )
    parser.add_argument(
        "--app-store-entitlements",
        type=Path,
        default=default_app_store_entitlements,
        help="Path to App Store entitlements plist (default: ios/Lumen/LumenAppStore.entitlements)",
    )
    parser.add_argument(
        "--skip-app-store-profile-comparison",
        action="store_true",
        help="Skip comparison between development entitlements and App Store entitlements.",
    )
    parser.add_argument(
        "--signed-app-path",
        type=Path,
        action="append",
        default=[],
        help="Path to a signed .app, .xcarchive, or .ipa whose codesigned entitlements should be validated.",
    )
    parser.add_argument(
        "--sanitized-entitlements-output",
        type=Path,
        help=(
            "Write a copy of --entitlements with App Store profile-incompatible "
            "entitlements removed. Validation still fails unless --allow-sanitized-output is set."
        ),
    )
    parser.add_argument(
        "--allow-sanitized-output",
        action="store_true",
        help="Allow disallowed entitlements when they are removed into --sanitized-entitlements-output.",
    )
    args = parser.parse_args(argv)

    entitlement_failures = [
        failure
        for entitlement_path in entitlements_to_validate(args.entitlements, root)
        for failure in validate_entitlements(entitlement_path)
    ]
    removed: list[str] = []
    if args.sanitized_entitlements_output:
        removed = sanitize_entitlements_file(args.entitlements, args.sanitized_entitlements_output)
        if removed:
            print(
                "Wrote sanitized entitlements without disallowed keys "
                f"{', '.join(removed)}: {args.sanitized_entitlements_output}",
                file=sys.stderr,
            )

    failures = [
        *validate_project_settings(args.project_file),
        *validate_app_sources(root),
        *(
            []
            if args.skip_app_store_profile_comparison
            else validate_app_store_entitlements_match(args.entitlements, args.app_store_entitlements)
        ),
        *([] if args.allow_sanitized_output and args.sanitized_entitlements_output else entitlement_failures),
    ]
    for signed_app_path in args.signed_app_path:
        if not signed_app_path.exists():
            failures.append(f"{signed_app_path}: signed app artifact does not exist")
            continue
        failures.extend(validate_signed_app(signed_app_path, args.app_store_entitlements))
    if failures:
        print("iOS signing capability validation failed:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1

    print("iOS signing capability validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
