#!/usr/bin/env python3
"""Validate MSAL iOS release configuration before CI/TestFlight handoff.

Checks:
- MSAL client ID is present and matches expected value.
- MSAL redirect URI uses expected format and value.
- Lumen bundle identifier aligns with redirect URI host segment.
"""
from __future__ import annotations

from pathlib import Path
import plistlib
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PLIST = ROOT / "ios" / "Lumen" / "MicrosoftGraphConfig.plist"
PBXPROJ = ROOT / "ios" / "Lumen.xcodeproj" / "project.pbxproj"

EXPECTED_BUNDLE_ID = "com.27pm.lumen"
EXPECTED_CLIENT_ID = "51aa8fd9-16b2-4f8e-8b97-b8618ceb6c40"
EXPECTED_REDIRECT_URI = f"msauth.{EXPECTED_BUNDLE_ID}://auth"


def fail(message: str) -> None:
    print(f"❌ {message}")
    sys.exit(1)


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

    if redirect_uri != EXPECTED_REDIRECT_URI:
        fail(f"MSALRedirectURI mismatch. Expected {EXPECTED_REDIRECT_URI}, found {redirect_uri}")

    pbxproj_text = PBXPROJ.read_text(encoding="utf-8")
    bundle_ids = set(re.findall(r"PRODUCT_BUNDLE_IDENTIFIER = ([^;]+);", pbxproj_text))

    if EXPECTED_BUNDLE_ID not in bundle_ids:
        fail(
            "Expected app bundle identifier not found in project.pbxproj: "
            f"{EXPECTED_BUNDLE_ID}"
        )

    if redirect_bundle_id != EXPECTED_BUNDLE_ID:
        fail(
            "Redirect URI bundle identifier does not align with app bundle identifier. "
            f"Redirect uses {redirect_bundle_id}, expected {EXPECTED_BUNDLE_ID}"
        )

    print("✅ MSAL iOS release configuration validation passed")
    print(f"   - MSALClientID: {client_id}")
    print(f"   - MSALRedirectURI: {redirect_uri}")
    print(f"   - App bundle identifier: {EXPECTED_BUNDLE_ID}")


if __name__ == "__main__":
    main()
