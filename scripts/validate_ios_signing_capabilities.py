#!/usr/bin/env python3
"""Validate iOS signing capabilities expected by App Store profiles.

This static guard catches capabilities that require special provisioning-profile
approval before Xcode reaches the archive signing phase.
"""
from __future__ import annotations

import argparse
import plistlib
import sys
from pathlib import Path
from typing import Any

CARPLAY_REMOVED_MESSAGE = (
    "CarPlay support has been removed. Remove this entitlement and archive "
    "with a provisioning profile that does not require CarPlay capabilities."
)

# Lumen no longer ships CarPlay features. Reject the generic CarPlay entitlement
# and any category-specific CarPlay entitlement so archive signing does not
# require CarPlay-enabled provisioning profiles.
DISALLOWED_ENTITLEMENT_PREFIXES = {
    "com.apple.developer.carplay": CARPLAY_REMOVED_MESSAGE,
}

DISALLOWED_PROJECT_SETTINGS = {
    "INFOPLIST_KEY_UIApplicationSupportsCarPlay": (
        "CarPlay support has been removed. Remove UIApplicationSupportsCarPlay "
        "so Xcode/App Store signing does not expect CarPlay capability support."
    ),
    "CPTemplateApplicationSceneSessionRoleApplication": (
        "CarPlay scene sessions must not be declared in generated Info.plist settings."
    ),
    "CarPlaySceneDelegate": (
        "CarPlay scene delegate references must be removed from project settings."
    ),
    "com.apple.developer.carplay": (
        "CarPlay must not be enabled in Xcode SystemCapabilities."
    ),
}

DISALLOWED_APP_SOURCE_TOKENS = {
    "import CarPlay": "Remove CarPlay framework imports from Swift sources.",
    "CPTemplateApplicationSceneSessionRoleApplication": (
        "Remove CarPlay scene manifest entries from app plist fragments."
    ),
    "CarPlaySceneDelegate": "Remove CarPlay scene delegate code and references.",
}

APP_SOURCE_SUFFIXES = {".swift", ".plist", ".entitlements", ".fragment"}


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


def sanitized_entitlements(entitlements: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Return entitlements with App-Store-profile-incompatible keys removed."""
    sanitized = dict(entitlements)
    removed: list[str] = []
    for key in list(sanitized):
        if disallowed_entitlement_message(key):
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
    for prefix, message in DISALLOWED_ENTITLEMENT_PREFIXES.items():
        if key.startswith(prefix):
            return message
    return None


def validate_entitlements(path: Path) -> list[str]:
    entitlements = read_plist(path)
    failures: list[str] = []
    for key in entitlements:
        if message := disallowed_entitlement_message(key):
            failures.append(f"{path}: disallowed entitlement '{key}'. {message}")
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


def main(argv: list[str] | None = None) -> int:
    root = repo_root_from_script()
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
        default=root / "ios" / "Lumen" / "Lumen.entitlements",
        help="Path to app entitlements plist (default: ios/Lumen/Lumen.entitlements)",
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
        *([] if args.allow_sanitized_output and args.sanitized_entitlements_output else entitlement_failures),
    ]
    if failures:
        print("iOS signing capability validation failed:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1

    print("iOS signing capability validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
