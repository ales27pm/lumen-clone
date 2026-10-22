#!/usr/bin/env python3
"""Private, fail-closed source/signing/receipt helpers for the native EAS lane."""

from __future__ import annotations

import argparse
import hashlib
import json
import plistlib
import re
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import validate_ios_signing_capabilities as signing


PROJECT = Path("ios/Lumen.xcodeproj/project.pbxproj")
RESOLVED = Path("ios/Lumen.xcodeproj/project.xcworkspace/xcshareddata/swiftpm/Package.resolved")
RELEASE_PATHS = ("ios", "scripts", "generated/agent_manifest", ".eas", "eas.json", "app.json", "package.json", "package-lock.json")
BUNDLE = "com.27pm.lumenclone"
TEAM = "52T7P32J34"
EAS_PROJECT = "1b2304fb-7f12-401b-813c-9a331d8b265d"
FULL_REVISION = re.compile(r"[0-9a-f]{40}(?:[0-9a-f]{24})?\Z")


def run(*args: str, cwd: Path | None = None) -> str:
    result = subprocess.run(args, cwd=cwd, text=True, capture_output=True, check=False)
    if result.returncode:
        # Signing-tool stderr can include private credential material.
        raise ValueError(f"{Path(args[0]).name} failed (exit {result.returncode})")
    return result.stdout


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    path.chmod(0o600)


def exact_setting(text: str, name: str) -> str:
    values = set(re.findall(rf"(?m)^\s*{re.escape(name)} = \"?([^;\"\n]+)\"?;", text))
    if len(values) != 1:
        raise ValueError(f"project must define one consistent {name}")
    return values.pop()


def validate_lock(project: str, resolved: dict[str, Any]) -> None:
    pins = resolved.get("pins")
    if not isinstance(pins, list) or not pins:
        raise ValueError("Swift package lock has no pins")
    by_identity: dict[str, dict[str, Any]] = {}
    for pin in pins:
        identity, state = pin.get("identity"), pin.get("state", {})
        if not isinstance(identity, str) or identity in by_identity or not FULL_REVISION.fullmatch(state.get("revision", "")):
            raise ValueError("Swift package lock has duplicate or invalid immutable pins")
        by_identity[identity] = pin
    references = re.findall(r'repositoryURL\s*=\s*"([^"]+)";\s*requirement\s*=\s*\{([^}]+)\}', project)
    if len(references) < 4:
        raise ValueError("expected native package references, including CoreAI and tokenizers")
    for url, requirement in references:
        identity = url.rstrip("/").rsplit("/", 1)[-1].removesuffix(".git").lower()
        pin = by_identity.get(identity)
        if pin is None:
            raise ValueError(f"Swift package lock is missing {identity}")
        if pin.get("location", "").removesuffix(".git").rstrip("/") != url.removesuffix(".git").rstrip("/"):
            raise ValueError(f"Swift package location mismatch: {identity}")
        version = re.search(r"\bversion\s*=\s*\"?([^;\"\s]+)", requirement)
        revision = re.search(r"\brevision\s*=\s*\"?([^;\"\s]+)", requirement)
        expected = revision or version
        key = "revision" if revision else "version"
        if expected is None or pin["state"].get(key) != expected.group(1):
            raise ValueError(f"Swift package lock does not match the reviewed {identity} requirement")


def candidate(root: Path, expected_revision: str | None = None) -> dict[str, Any]:
    revision = run("git", "rev-parse", "--verify", "HEAD^{commit}", cwd=root).strip()
    if not FULL_REVISION.fullmatch(revision) or (expected_revision and revision != expected_revision):
        raise ValueError("candidate revision does not match the requested full commit")
    if run("git", "status", "--porcelain=v1", "--untracked-files=all", "--", *RELEASE_PATHS, cwd=root).strip():
        raise ValueError("native release inputs are dirty or untracked; commit the reviewed candidate")
    text = (root / PROJECT).read_text(encoding="utf-8")
    identifiers = set(re.findall(r"PRODUCT_BUNDLE_IDENTIFIER = ([^;]+);", text))
    if BUNDLE not in identifiers or exact_setting(text, "DEVELOPMENT_TEAM") != TEAM:
        raise ValueError("native app/team identity differs from the established Lumen release")
    deployment = exact_setting(text, "IPHONEOS_DEPLOYMENT_TARGET")
    if deployment != "27.1":
        raise ValueError("native EAS lane requires the authorized iOS 27.1 deployment target")
    validate_lock(text, json.loads((root / RESOLVED).read_text(encoding="utf-8")))
    app = json.loads((root / "app.json").read_text(encoding="utf-8"))["expo"]
    if app.get("extra", {}).get("eas", {}).get("projectId") != EAS_PROJECT:
        raise ValueError("candidate is linked to a different EAS project")
    sources = run("git", "ls-files", "-z", "--", *RELEASE_PATHS, cwd=root).split("\0")
    inventory = {name: digest(root / name) for name in sources if name and (root / name).is_file()}
    build = exact_setting(text, "CURRENT_PROJECT_VERSION")
    if not re.fullmatch(r"[0-9]+(?:\.[0-9]+){0,2}", build):
        raise ValueError("invalid committed build number")
    return {
        "schemaVersion": 1, "sourceRevision": revision, "bundleIdentifier": BUNDLE,
        "teamIdentifier": TEAM, "easProjectId": EAS_PROJECT, "scheme": "Lumen",
        "configuration": "Release", "marketingVersion": exact_setting(text, "MARKETING_VERSION"),
        "buildNumber": build, "deploymentTarget": deployment, "sourceSHA256": inventory,
    }


def validate_toolchain(xcode: str, sdk: str, deployment_target: str) -> None:
    match = re.search(r"\bXcode (\d+)(?:\.(\d+))?", xcode)
    if match is None or int(match.group(1)) < 27:
        raise ValueError("native CoreAI build requires Xcode 27 or newer")
    if not re.fullmatch(r"\d+(?:\.\d+){0,2}", sdk.strip()) or int(sdk.split(".")[0]) < 27:
        raise ValueError("native CoreAI build requires an iOS 27 SDK or newer")
    if not re.fullmatch(r"\d+(?:\.\d+){0,2}", deployment_target):
        raise ValueError("candidate deployment target is not a numeric iOS version")
    def numeric_version(value: str) -> tuple[int, ...]:
        parts = tuple(int(part) for part in value.strip().split("."))
        return parts + (0,) * (3 - len(parts))
    if numeric_version(sdk) < numeric_version(deployment_target):
        raise ValueError(f"iOS SDK {sdk.strip()} cannot archive the required iOS {deployment_target} deployment target")


def valid_identities(output: str) -> set[str]:
    identities = set()
    for line in output.splitlines():
        match = re.fullmatch(r'\s*\d+\)\s+([A-Fa-f0-9]{40})\s+"(Apple Distribution(?::[^\"]*)?)"\s*', line)
        if match:
            identities.add(match.group(1).upper())
    return identities


def profile_choice(
    profiles: list[tuple[Path, dict[str, Any]]], identities: set[str], expected_entitlements: dict[str, Any],
    *, now: datetime | None = None,
) -> dict[str, str]:
    now = now or datetime.now(timezone.utc)
    matches: dict[tuple[str, str, str], dict[str, str]] = {}
    for path, profile in profiles:
        entitlements = profile.get("Entitlements", {})
        expires = profile.get("ExpirationDate")
        if isinstance(expires, datetime) and expires.tzinfo is None:
            expires = expires.replace(tzinfo=timezone.utc)
        app_id = entitlements.get("application-identifier", "")
        if not isinstance(app_id, str) or not app_id.endswith("." + BUNDLE):
            continue
        prefix = app_id[: -len(BUNDLE) - 1]
        if (
            profile.get("TeamIdentifier") != [TEAM]
            or entitlements.get("com.apple.developer.team-identifier") != TEAM
            or prefix not in profile.get("ApplicationIdentifierPrefix", [])
            or not isinstance(expires, datetime) or expires <= now
            or "ProvisionedDevices" in profile or profile.get("ProvisionsAllDevices") is True
            or entitlements.get("get-task-allow") is not False
            or entitlements.get("beta-reports-active") is not True
            or not re.fullmatch(r"[A-Fa-f0-9-]{36}", str(profile.get("UUID", "")))
        ):
            continue
        expected = signing.resolve_expected_build_settings(expected_entitlements, {
            "AppIdentifierPrefix": prefix + ".", "TeamIdentifierPrefix": TEAM + ".",
            "DEVELOPMENT_TEAM": TEAM, "PRODUCT_BUNDLE_IDENTIFIER": BUNDLE,
        })
        if any(
            signing._profile_controls_entitlement(key) and
            (key not in entitlements or not signing._profile_value_authorizes(entitlements[key], value))
            for key, value in expected.items()
        ):
            continue
        certificates = {
            hashlib.sha1(bytes(certificate)).hexdigest().upper()
            for certificate in profile.get("DeveloperCertificates", [])
            if isinstance(certificate, (bytes, bytearray)) and certificate
        } & identities
        for fingerprint in certificates:
            canonical = hashlib.sha256(plistlib.dumps(profile, sort_keys=True)).hexdigest()
            matches[(profile["UUID"], fingerprint, canonical)] = {
                "profileUUID": profile["UUID"], "certificateSHA1": fingerprint,
                "profilePath": str(path), "teamIdentifier": TEAM, "bundleIdentifier": BUNDLE,
            }
    if len(matches) != 1:
        raise ValueError(f"expected one authorized installed App Store profile/certificate pair; found {len(matches)}")
    return next(iter(matches.values()))


def installed_signing(root: Path) -> dict[str, str]:
    profiles: list[tuple[Path, dict[str, Any]]] = []
    for directory in (
        Path.home() / "Library/MobileDevice/Provisioning Profiles",
        Path.home() / "Library/Developer/Xcode/UserData/Provisioning Profiles",
    ):
        for path in sorted(directory.glob("*.mobileprovision")):
            decoded = subprocess.run(["security", "cms", "-D", "-i", str(path)], capture_output=True, check=False)
            if decoded.returncode:
                continue
            try:
                value = plistlib.loads(decoded.stdout)
            except plistlib.InvalidFileException:
                continue
            if isinstance(value, dict):
                profiles.append((path, value))
    identities = valid_identities(run("security", "find-identity", "-v", "-p", "codesigning"))
    with (root / "ios/Lumen/LumenAppStore.entitlements").open("rb") as handle:
        return profile_choice(profiles, identities, plistlib.load(handle))


def export_options(signing_context: dict[str, str]) -> dict[str, Any]:
    return {
        "method": "app-store-connect", "destination": "export", "signingStyle": "manual",
        "teamID": TEAM, "signingCertificate": signing_context["certificateSHA1"],
        "provisioningProfiles": {BUNDLE: signing_context["profileUUID"]},
        "manageAppVersionAndBuildNumber": False, "stripSwiftSymbols": True, "uploadSymbols": True,
    }


def signing_xcconfig(signing_context: dict[str, str]) -> str:
    """Expand credentials for Lumen only, leaving SwiftPM resource targets empty."""
    profile, certificate = signing_context["profileUUID"], signing_context["certificateSHA1"]
    if (
        not re.fullmatch(r"[A-Fa-f0-9-]{36}", profile)
        or not re.fullmatch(r"[A-F0-9]{40}", certificate)
        or signing_context.get("teamIdentifier") != TEAM
        or signing_context.get("bundleIdentifier") != BUNDLE
    ):
        raise ValueError("invalid runtime signing context for the scoped Xcode configuration")
    return "\n".join((
        "// Private runtime signing overrides. No project file is modified.",
        f"LUMEN_PROFILE_Lumen = {profile}",
        f"LUMEN_CERTIFICATE_Lumen = {certificate}",
        f"LUMEN_TEAM_Lumen = {TEAM}",
        "PROVISIONING_PROFILE_SPECIFIER = $(LUMEN_PROFILE_$(TARGET_NAME))",
        "PROVISIONING_PROFILE_SPECIFIER[sdk=iphoneos*] = $(LUMEN_PROFILE_$(TARGET_NAME))",
        "CODE_SIGN_IDENTITY = $(LUMEN_CERTIFICATE_$(TARGET_NAME))",
        "CODE_SIGN_IDENTITY[sdk=iphoneos*] = $(LUMEN_CERTIFICATE_$(TARGET_NAME))",
        "DEVELOPMENT_TEAM = $(LUMEN_TEAM_$(TARGET_NAME))",
        "",
    ))


def collect_recovery(ipa: Path, archive: Path, evidence: Path) -> None:
    """Retain failed-gate outputs as unverified evidence, never a success receipt."""
    symbols = sorted(path for path in (archive / "dSYMs").rglob("*") if path.is_file())
    if not ipa.is_file() and not symbols:
        return
    recovery = evidence / "recovery"
    recovery.mkdir(parents=True, exist_ok=True)
    files: dict[str, dict[str, Any]] = {}
    if ipa.is_file():
        target = recovery / "Lumen.unverified.ipa"
        shutil.copyfile(ipa, target)
        target.chmod(0o600)
        files[target.name] = {"sha256": digest(target), "bytes": target.stat().st_size}
    if symbols:
        target = recovery / "Lumen.original-dSYMs.unverified.zip"
        with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as zipped:
            for path in symbols:
                zipped.write(path, path.relative_to(archive).as_posix())
        target.chmod(0o600)
        files[target.name] = {"sha256": digest(target), "bytes": target.stat().st_size}
    source_revision = None
    candidate_path = evidence / "candidate.json"
    if candidate_path.is_file():
        try:
            candidate_revision = json.loads(candidate_path.read_text(encoding="utf-8")).get("sourceRevision", "")
            if isinstance(candidate_revision, str) and FULL_REVISION.fullmatch(candidate_revision):
                source_revision = candidate_revision
        except (ValueError, OSError):
            pass
    write_json(recovery / "recovery.json", {
        "schemaVersion": 1, "status": "unverified-recovery-not-for-submission", "files": files,
        "candidateSourceRevision": source_revision,
        "dSYMFiles": {path.relative_to(archive).as_posix(): digest(path) for path in symbols},
        "evidenceBoundary": "Partial outputs retained after failed release gates; independently validate before reuse.",
    })


def verify_info(info: dict[str, Any], source: dict[str, Any]) -> None:
    expected = {
        "CFBundleIdentifier": source["bundleIdentifier"], "CFBundleVersion": source["buildNumber"],
        "CFBundleShortVersionString": source["marketingVersion"], "LumenGitSHA": source["sourceRevision"],
        "LumenBuildScheme": source["scheme"], "LumenBuildConfiguration": source["configuration"],
        "MinimumOSVersion": source["deploymentTarget"],
    }
    if any(info.get(key) != value for key, value in expected.items()):
        raise ValueError("exported artifact identity/provenance does not match the frozen candidate")


def artifact_receipt(root: Path, source: dict[str, Any], ipa: Path, archive: Path, output: Path) -> None:
    with zipfile.ZipFile(ipa) as package:
        names = [name for name in package.namelist() if re.fullmatch(r"Payload/[^/]+\.app/Info\.plist", name)]
        if len(names) != 1:
            raise ValueError("expected exactly one app Info.plist in the verified IPA")
        verify_info(plistlib.loads(package.read(names[0])), source)
    apps = list((archive / "Products/Applications").glob("*.app/Info.plist"))
    if len(apps) != 1:
        raise ValueError("expected exactly one app in the verified archive")
    verify_info(plistlib.loads(apps[0].read_bytes()), source)
    symbols = sorted(path for path in (archive / "dSYMs").rglob("*") if path.is_file())
    if not symbols:
        raise ValueError("verified archive has no original dSYM files")
    symbol_zip = output.parent / "Lumen.dSYMs.zip"
    with zipfile.ZipFile(symbol_zip, "w", compression=zipfile.ZIP_DEFLATED) as zipped:
        for path in symbols:
            zipped.write(path, path.relative_to(archive).as_posix())
    write_json(output, {
        "schemaVersion": 1, "status": "verified-ipa-not-submitted", "sourceRevision": source["sourceRevision"],
        "bundleIdentifier": source["bundleIdentifier"], "marketingVersion": source["marketingVersion"],
        "buildNumber": source["buildNumber"], "minimumOSVersion": source["deploymentTarget"],
        "ipaSHA256": digest(ipa), "ipaBytes": ipa.stat().st_size, "symbolsSHA256": digest(symbol_zip),
        "dSYMFiles": {path.relative_to(archive).as_posix(): digest(path) for path in symbols},
        "evidenceBoundary": "Signed archive/export validation; no TestFlight processing or physical-device proof.",
    })


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("candidate", "field", "toolchain", "signing", "export-options", "signing-xcconfig", "artifact-receipt", "collect-recovery"))
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--key")
    parser.add_argument("--expected-revision")
    parser.add_argument("--xcode", type=Path)
    parser.add_argument("--sdk")
    parser.add_argument("--deployment-target")
    parser.add_argument("--ipa", type=Path)
    parser.add_argument("--archive", type=Path)
    args = parser.parse_args()
    try:
        if args.operation == "candidate":
            write_json(args.output, candidate(args.root, args.expected_revision))
        elif args.operation == "field":
            value = json.loads(args.input.read_text())[args.key]
            if not isinstance(value, str) or "\n" in value:
                raise ValueError("requested context field is not a single string")
            print(value)
        elif args.operation == "toolchain":
            validate_toolchain(args.xcode.read_text(), args.sdk, args.deployment_target)
        elif args.operation == "signing":
            write_json(args.output, installed_signing(args.root))
        elif args.operation == "export-options":
            args.output.write_bytes(plistlib.dumps(export_options(json.loads(args.input.read_text()))))
            args.output.chmod(0o600)
        elif args.operation == "signing-xcconfig":
            args.output.write_text(signing_xcconfig(json.loads(args.input.read_text())), encoding="utf-8")
            args.output.chmod(0o600)
        elif args.operation == "artifact-receipt":
            artifact_receipt(args.root, json.loads(args.input.read_text()), args.ipa, args.archive, args.output)
        elif args.operation == "collect-recovery":
            collect_recovery(args.ipa, args.archive, args.output)
    except (ValueError, KeyError, OSError, TypeError, json.JSONDecodeError, plistlib.InvalidFileException, zipfile.BadZipFile) as error:
        print(f"error: native EAS preflight/verification failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
