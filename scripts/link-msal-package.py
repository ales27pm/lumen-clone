#!/usr/bin/env python3
"""Deterministically link the Microsoft MSAL Swift Package into the Lumen Xcode target.

Why this exists:
- The Outlook integration imports `MSAL` behind `#if canImport(MSAL)`.
- If the package is only documented but not linked in project.pbxproj, TestFlight builds show
  the "MSAL package missing" fallback screen.
- This script edits the Xcode project in a stable, idempotent way so the Lumen app target
  resolves and links the `MSAL` product from microsoft-authentication-library-for-objc.

Run from repository root:
    python3 scripts/link-msal-package.py

Then commit ios/Lumen.xcodeproj/project.pbxproj.
"""

from __future__ import annotations

import pathlib
import sys

PROJECT = pathlib.Path("ios/Lumen.xcodeproj/project.pbxproj")

MSAL_BUILD_FILE_ID = "A27B0C0D0E0F000000000002"
MSAL_PACKAGE_REF_ID = "A27B0C0D0E0F000000000003"
MSAL_PRODUCT_DEP_ID = "A27B0C0D0E0F000000000004"
SWIFT_LLAMA_BUILD_FILE_ID = "9A1B2C3D4E5F678901234570"
SWIFT_LLAMA_PRODUCT_DEP_ID = "9A1B2C3D4E5F678901234571"
SWIFT_LLAMA_PACKAGE_REF_ID = "9A1B2C3D4E5F678901234572"

MSAL_REPO_URL = "https://github.com/AzureAD/microsoft-authentication-library-for-objc.git"


def replace_once(text: str, old: str, new: str) -> str:
    if old not in text:
        raise RuntimeError(f"Expected project fragment not found:\n{old}")
    return text.replace(old, new, 1)


def ensure_line_after(
    text: str,
    anchor: str,
    line: str,
    *,
    already_present: str | None = None,
) -> str:
    if line in text:
        return text
    if already_present and already_present in text:
        return text
    return replace_once(text, anchor, anchor + line)


def main() -> int:
    if not PROJECT.exists():
        print(f"error: {PROJECT} not found", file=sys.stderr)
        return 1

    text = PROJECT.read_text(encoding="utf-8")
    original = text

    # PBXBuildFile entry: links the Swift package product into the app Frameworks phase.
    text = ensure_line_after(
        text,
        f"\t\t{SWIFT_LLAMA_BUILD_FILE_ID} /* SwiftLlama in Frameworks */ = {{isa = PBXBuildFile; productRef = {SWIFT_LLAMA_PRODUCT_DEP_ID} /* SwiftLlama */; }};\n",
        f"\t\t{MSAL_BUILD_FILE_ID} /* MSAL in Frameworks */ = {{isa = PBXBuildFile; productRef = {MSAL_PRODUCT_DEP_ID} /* MSAL */; }};\n",
        already_present="/* MSAL in Frameworks */ = {isa = PBXBuildFile; productRef = ",
    )

    # Lumen app Frameworks build phase.
    text = ensure_line_after(
        text,
        f"\t\t\t\t{SWIFT_LLAMA_BUILD_FILE_ID} /* SwiftLlama in Frameworks */,\n",
        f"\t\t\t\t{MSAL_BUILD_FILE_ID} /* MSAL in Frameworks */,\n",
        already_present="/* MSAL in Frameworks */,",
    )

    # Lumen app target package product dependencies.
    text = ensure_line_after(
        text,
        f"\t\t\t\t{SWIFT_LLAMA_PRODUCT_DEP_ID} /* SwiftLlama */,\n",
        f"\t\t\t\t{MSAL_PRODUCT_DEP_ID} /* MSAL */,\n",
        already_present="/* MSAL */,",
    )

    # Project-level package reference list.
    text = ensure_line_after(
        text,
        f"\t\t\t\t{SWIFT_LLAMA_PACKAGE_REF_ID} /* XCRemoteSwiftPackageReference \"swift-llama-cpp\" */,\n",
        f"\t\t\t\t{MSAL_PACKAGE_REF_ID} /* XCRemoteSwiftPackageReference \"microsoft-authentication-library-for-objc\" */,\n",
        already_present='/* XCRemoteSwiftPackageReference "microsoft-authentication-library-for-objc" */,',
    )

    # XCRemoteSwiftPackageReference section.
    msal_remote_ref = f"""\t\t{MSAL_PACKAGE_REF_ID} /* XCRemoteSwiftPackageReference \"microsoft-authentication-library-for-objc\" */ = {{
\t\t\tisa = XCRemoteSwiftPackageReference;
\t\t\trepositoryURL = \"{MSAL_REPO_URL}\";
\t\t\trequirement = {{
\t\t\t\tkind = upToNextMajorVersion;
\t\t\t\tminimumVersion = 1.7.0;
\t\t\t}};
\t\t}};
"""
    if (
        msal_remote_ref not in text
        and 'repositoryURL = "https://github.com/AzureAD/microsoft-authentication-library-for-objc.git";'
        not in text
    ):
        text = replace_once(
            text,
            "\t\t};\n/* End XCRemoteSwiftPackageReference section */",
            "\t\t};\n" + msal_remote_ref + "/* End XCRemoteSwiftPackageReference section */",
        )

    # XCSwiftPackageProductDependency section.
    msal_product_dep = f"""\t\t{MSAL_PRODUCT_DEP_ID} /* MSAL */ = {{
\t\t\tisa = XCSwiftPackageProductDependency;
\t\t\tpackage = {MSAL_PACKAGE_REF_ID} /* XCRemoteSwiftPackageReference \"microsoft-authentication-library-for-objc\" */;
\t\t\tproductName = MSAL;
\t\t}};
"""
    if msal_product_dep not in text and "productName = MSAL;" not in text:
        text = replace_once(
            text,
            "\t\t};\n/* End XCSwiftPackageProductDependency section */",
            "\t\t};\n" + msal_product_dep + "/* End XCSwiftPackageProductDependency section */",
        )

    required = [
        MSAL_REPO_URL,
        f"{MSAL_BUILD_FILE_ID} /* MSAL in Frameworks */",
        f"{MSAL_PRODUCT_DEP_ID} /* MSAL */",
        "productName = MSAL;",
    ]
    missing = [fragment for fragment in required if fragment not in text]
    if missing:
        print("error: MSAL linkage validation failed; missing fragments:", file=sys.stderr)
        for fragment in missing:
            print(f"  - {fragment}", file=sys.stderr)
        return 1

    if text == original:
        print("MSAL package already linked in ios/Lumen.xcodeproj/project.pbxproj")
        return 0

    PROJECT.write_text(text, encoding="utf-8")
    print("Linked MSAL package into ios/Lumen.xcodeproj/project.pbxproj")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
