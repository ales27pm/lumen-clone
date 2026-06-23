#!/usr/bin/env python3
"""Validate MSAL iOS release configuration before CI/TestFlight handoff.

Checks:
- MSAL client ID is present and matches expected value.
- MSAL redirect URI uses expected format and value.
- Lumen bundle identifier aligns with redirect URI host segment.
"""
from __future__ import annotations

import plistlib
import re
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PLIST = ROOT / "ios" / "Lumen" / "MicrosoftGraphConfig.plist"
PBXPROJ = ROOT / "ios" / "Lumen.xcodeproj" / "project.pbxproj"

EXPECTED_CLIENT_ID = "51aa8fd9-16b2-4f8e-8b97-b8618ceb6c40"


def fail(message: str) -> None:
    print(f"❌ {message}")
    sys.exit(1)


def find_matching_brace(text: str, open_index: int) -> int:
    depth = 0
    in_string = False
    escaped = False
    for index in range(open_index, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return index
    raise ValueError("Could not find matching brace")


def discover_target_config_ids(text: str, target_name: str) -> list[tuple[str, str]]:
    config_list_pattern = re.compile(
        rf'(?P<config_list>[A-Fa-f0-9]{{24}})\s+/\*\s+Build configuration list for PBXNativeTarget "{re.escape(target_name)}"\s+\*/\s*=\s*\{{',
        re.IGNORECASE,
    )
    config_list_match = config_list_pattern.search(text)
    if not config_list_match:
        fail(f"Missing configuration list for target {target_name}")

    list_open = config_list_match.end() - 1
    list_close = find_matching_brace(text, list_open)
    list_block = text[list_open : list_close + 1]

    build_configs_match = re.search(r"buildConfigurations = \((?P<body>.*?)\);", list_block, re.DOTALL)
    if not build_configs_match:
        fail(f"Missing buildConfigurations for target {target_name}")

    configs = re.findall(
        r"\b([A-Fa-f0-9]{24})\b\s*/\*\s*([^*]+)\s*\*/",
        build_configs_match.group("body"),
        re.MULTILINE,
    )
    if not configs:
        fail(f"No build configurations found for target {target_name}")
    return [(name.strip(), config_id) for config_id, name in configs]


def get_config_block(text: str, config_id: str) -> str:
    marker = f"\t\t{config_id} /* "
    config_start = text.find(marker)
    if config_start < 0:
        fail(f"Missing build configuration {config_id}")
    open_index = text.find("{", config_start)
    close_index = find_matching_brace(text, open_index)
    return text[open_index : close_index + 1]


def extract_setting(block: str, key: str) -> str | None:
    match = re.search(rf"^\s*{re.escape(key)}\s*=\s*([^;]+);", block, re.MULTILINE)
    if not match:
        return None
    return match.group(1).strip().strip('"')


def main() -> None:
    if not CONFIG_PLIST.exists():
        fail(f"Missing config plist: {CONFIG_PLIST}")
    if not PBXPROJ.exists():
        fail(f"Missing Xcode project: {PBXPROJ}")

    with CONFIG_PLIST.open("rb") as handle:
        config = plistlib.load(handle)

    client_id = (config.get("MSALClientID") or "").strip()
    if not client_id:
        fail("MSALClientID is missing/empty in MicrosoftGraphConfig.plist")
    if client_id != EXPECTED_CLIENT_ID:
        fail(f"MSALClientID mismatch. Expected {EXPECTED_CLIENT_ID}, found {client_id}")

    redirect_uri = (config.get("MSALRedirectURI") or "").strip()
    if not redirect_uri:
        fail("MSALRedirectURI is missing/empty in MicrosoftGraphConfig.plist")

    redirect_pattern = re.compile(r"^msauth\.([A-Za-z0-9\.-]+)://auth$")
    match = redirect_pattern.match(redirect_uri)
    if not match:
        fail(f"MSALRedirectURI must match format msauth.<bundle-id>://auth, found: {redirect_uri}")
    redirect_bundle_id = match.group(1)

    pbxproj_text = PBXPROJ.read_text(encoding="utf-8")
    app_configs: dict[str, str] = {}
    app_config_blocks: dict[str, str] = {}
    for config_name, config_id in discover_target_config_ids(pbxproj_text, "Lumen"):
        block = get_config_block(pbxproj_text, config_id)
        bundle_id = extract_setting(block, "PRODUCT_BUNDLE_IDENTIFIER")
        if not bundle_id:
            fail(f"Missing PRODUCT_BUNDLE_IDENTIFIER for Lumen {config_name}")
        app_configs[config_name] = bundle_id
        app_config_blocks[config_name] = block

    if "Release" not in app_configs:
        fail("Missing Lumen Release build configuration")

    release_bundle_id = app_configs["Release"]
    if set(app_configs.values()) != {release_bundle_id}:
        fail(
            "Lumen build configurations use conflicting bundle identifiers: "
            + ", ".join(f"{name}={bundle_id}" for name, bundle_id in sorted(app_configs.items()))
        )

    expected_redirect_uri = f"msauth.{release_bundle_id}://auth"
    if redirect_uri != expected_redirect_uri:
        fail(f"MSALRedirectURI mismatch. Expected {expected_redirect_uri}, found {redirect_uri}")

    if f"msauth.{release_bundle_id}" not in app_config_blocks["Release"]:
        fail(
            "Release generated Info.plist URL scheme does not align with app bundle identifier. "
            f"Expected msauth.{release_bundle_id}"
        )

    if redirect_bundle_id != release_bundle_id:
        fail(
            "Redirect URI bundle identifier does not align with app bundle identifier. "
            f"Redirect uses {redirect_bundle_id}, expected {release_bundle_id}"
        )

    print("✅ MSAL iOS release configuration validation passed")
    print(f"   - MSALClientID: {client_id}")
    print(f"   - MSALRedirectURI: {redirect_uri}")
    print(f"   - App bundle identifier: {release_bundle_id}")


if __name__ == "__main__":
    main()
