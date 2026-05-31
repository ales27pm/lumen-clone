#!/usr/bin/env python3
"""Patch generated Info.plist build settings for MSAL iOS broker callback support.

Lumen uses GENERATE_INFOPLIST_FILE=YES, so there is no checked-in Info.plist to edit.
The app target stores generated plist entries as INFOPLIST_KEY_* build settings in
`ios/Lumen.xcodeproj/project.pbxproj`.

This script inserts the required MSAL URL scheme and broker query schemes into the
Lumen target Debug/Release build settings idempotently.
"""
from __future__ import annotations

from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "ios" / "Lumen.xcodeproj" / "project.pbxproj"

URL_TYPES = '''\t\t\t\tINFOPLIST_KEY_CFBundleURLTypes = (
\t\t\t\t\t{
\t\t\t\t\t\tCFBundleURLName = "com.27pm.lumen";
\t\t\t\t\t\tCFBundleURLSchemes = (
\t\t\t\t\t\t\t"msauth.com.27pm.lumen",
\t\t\t\t\t\t);
\t\t\t\t\t},
\t\t\t\t);
'''

QUERY_SCHEMES = '''\t\t\t\tINFOPLIST_KEY_LSApplicationQueriesSchemes = (
\t\t\t\t\tmsauth,
\t\t\t\t\tmsauthv2,
\t\t\t\t\tmsauthv3,
\t\t\t\t);
'''

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
    raise RuntimeError("Could not find matching brace")


def replace_or_insert_setting(block: str, key: str, value: str) -> str:
    needle = f"\t\t\t\t{key} = "
    existing = block.find(needle)
    if existing >= 0:
        value_start = existing
        cursor = existing + len(needle)
        depth = 0
        in_string = False
        escaped = False
        while cursor < len(block):
            char = block[cursor]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
            else:
                if char == '"':
                    in_string = True
                elif char in "({":
                    depth += 1
                elif char in ")}":
                    depth -= 1
                elif char == ";" and depth == 0:
                    remainder_start = cursor + 1
                    while remainder_start < len(block) and block[remainder_start] in "\r\n":
                        remainder_start += 1
                    return block[:value_start] + value + block[remainder_start:]
            cursor += 1
        raise RuntimeError(f"Could not replace existing {key}")

    build_settings_open = block.find("buildSettings = {")
    if build_settings_open < 0:
        raise RuntimeError(f"Missing buildSettings block for {key}")
    insertion_point = block.find("\n", build_settings_open)
    if insertion_point < 0:
        raise RuntimeError(f"Could not find insertion point for {key}")
    return block[: insertion_point + 1] + value + block[insertion_point + 1 :]


def discover_target_config_ids(text: str, target_name: str) -> list[str]:
    config_list_pattern = re.compile(
        rf'(?P<config_list>[A-Fa-f0-9]{{24}})\s+/\*\s+Build configuration list for PBXNativeTarget "{re.escape(target_name)}"\s+\*/\s*=\s*\{{',
        re.IGNORECASE,
    )
    config_list_match = config_list_pattern.search(text)
    if not config_list_match:
        raise RuntimeError(f"Missing configuration list for target {target_name}")

    config_list_id = config_list_match.group("config_list")
    list_open = config_list_match.end() - 1
    list_close = find_matching_brace(text, list_open)
    list_block = text[list_open : list_close + 1]

    build_configs_match = re.search(r"buildConfigurations = \((?P<body>.*?)\);", list_block, re.DOTALL)
    if not build_configs_match:
        raise RuntimeError(f"Missing buildConfigurations for target {target_name}")
    config_ids = re.findall(
        r"\b([A-Fa-f0-9]{24})\b\s*/\*\s*[^*]+\*/",
        build_configs_match.group("body"),
        re.MULTILINE,
    )
    if not config_ids:
        raise RuntimeError(f"No build configurations found for target {target_name}")
    return config_ids


def patch_config(text: str, config_id: str) -> str:
    marker = f"\t\t{config_id} /* "
    config_start = text.find(marker)
    if config_start < 0:
        raise RuntimeError(f"Missing build configuration {config_id}")
    open_index = text.find("{", config_start)
    close_index = find_matching_brace(text, open_index)
    block = text[open_index : close_index + 1]
    block = replace_or_insert_setting(block, "INFOPLIST_KEY_CFBundleURLTypes", URL_TYPES)
    block = replace_or_insert_setting(block, "INFOPLIST_KEY_LSApplicationQueriesSchemes", QUERY_SCHEMES)
    return text[:open_index] + block + text[close_index + 1 :]


def main() -> None:
    text = PROJECT.read_text(encoding="utf-8")
    original = text
    for config_id in discover_target_config_ids(text, "Lumen"):
        text = patch_config(text, config_id)

    if text != original:
        PROJECT.write_text(text, encoding="utf-8")
        print(f"Patched {PROJECT}")
    else:
        print(f"Already patched: {PROJECT}")

    for required in [
        'INFOPLIST_KEY_CFBundleURLTypes',
        '"msauth.com.27pm.lumen"',
        'INFOPLIST_KEY_LSApplicationQueriesSchemes',
        'msauthv2',
        'msauthv3',
    ]:
        if required not in text:
            raise SystemExit(f"Patch failed; missing {required}")


if __name__ == "__main__":
    main()
