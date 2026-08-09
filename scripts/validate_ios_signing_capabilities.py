#!/usr/bin/env python3
"""Validate iOS signing capabilities expected by App Store profiles.

This static guard catches capabilities that require special provisioning-profile
approval before Xcode reaches the archive signing phase.
"""
from __future__ import annotations

import argparse
import plistlib
import re
import shutil
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping

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
    "beta-reports-active",
    "com.apple.application-identifier",
    "com.apple.developer.team-identifier",
    "get-task-allow",
}

BUILD_SETTING_REFERENCE = re.compile(r"\$\(([^)]+)\)|\$\{([^}]+)\}")

SIGNING_STAGES = {"archive", "app-store"}
APPLE_DEVELOPMENT_AUTHORITY = "Apple Development"
APPLE_DISTRIBUTION_AUTHORITY = "Apple Distribution"


@dataclass(frozen=True)
class SignedArtifactEvidence:
    bundle_identifier: str
    signature_identifier: str
    signature_team_identifier: str
    signature_marker: str
    authorities: tuple[str, ...]
    leaf_signing_certificate: bytes
    signed_entitlements: dict[str, Any]
    profile: dict[str, Any]


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


def resolve_expected_build_settings(
    expected: Any,
    build_settings: Mapping[str, str],
) -> Any:
    """Resolve entitlement build-setting references to exact signed values."""
    if isinstance(expected, str):
        unresolved: set[str] = set()

        def replacement(match: re.Match[str]) -> str:
            key = match.group(1) or match.group(2)
            value = build_settings.get(key)
            if value is None:
                unresolved.add(key)
                return match.group(0)
            return value

        resolved = BUILD_SETTING_REFERENCE.sub(replacement, expected)
        if unresolved:
            raise ValueError(
                "unresolved build setting(s): " + ", ".join(sorted(unresolved))
            )
        return resolved
    if isinstance(expected, list):
        return [
            resolve_expected_build_settings(item, build_settings)
            for item in expected
        ]
    if isinstance(expected, dict):
        return {
            key: resolve_expected_build_settings(value, build_settings)
            for key, value in expected.items()
        }
    return expected


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


@contextmanager
def resolved_signed_app_path(path: Path) -> Iterator[Path]:
    if path.suffix == ".ipa":
        with tempfile.TemporaryDirectory(prefix="lumen-ipa-entitlements-") as tmp:
            tmp_path = Path(tmp)
            ditto = shutil.which("ditto")
            if sys.platform != "darwin" or not ditto:
                raise SystemExit("error: signed IPA validation requires macOS ditto")
            extraction = subprocess.run(
                [ditto, "-x", "-k", str(path), str(tmp_path)],
                text=True,
                capture_output=True,
                check=False,
            )
            if extraction.returncode != 0:
                detail = (extraction.stderr or extraction.stdout).strip()
                raise SystemExit(f"error: could not extract signed IPA {path}: {detail}")
            apps = sorted((tmp_path / "Payload").glob("*.app"))
            if len(apps) != 1:
                raise SystemExit(
                    f"error: expected exactly one Payload/*.app in {path}; found {len(apps)}"
                )
            yield apps[0]
            return
    yield _find_app(path)


def read_signed_entitlements(app_path: Path) -> dict[str, Any]:
    codesign = shutil.which("codesign")
    if sys.platform != "darwin" or not codesign:
        raise SystemExit("error: signed entitlement validation requires macOS codesign")
    verify_result = subprocess.run(
        [codesign, "--verify", "--deep", "--strict", str(app_path)],
        text=True,
        capture_output=True,
        check=False,
    )
    if verify_result.returncode != 0:
        detail = (verify_result.stderr or verify_result.stdout).strip()
        raise SystemExit(f"error: codesign verification failed for {app_path}: {detail}")
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


def read_code_signature_details(app_path: Path) -> dict[str, list[str]]:
    codesign = shutil.which("codesign")
    if sys.platform != "darwin" or not codesign:
        raise SystemExit("error: signed artifact validation requires macOS codesign")
    result = subprocess.run(
        [codesign, "-d", "--verbose=4", str(app_path)],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise SystemExit(f"error: codesign could not describe {app_path}: {detail}")

    details: dict[str, list[str]] = {}
    for line in f"{result.stdout}\n{result.stderr}".splitlines():
        if "=" not in line:
            continue
        key, value = line.strip().split("=", 1)
        details.setdefault(key, []).append(value)
    return details


def read_leaf_signing_certificate(app_path: Path) -> bytes:
    codesign = shutil.which("codesign")
    if sys.platform != "darwin" or not codesign:
        raise SystemExit("error: signing-certificate validation requires macOS codesign")

    with tempfile.TemporaryDirectory(prefix="lumen-codesign-certificates-") as tmp:
        certificate_prefix = Path(tmp) / "certificate"
        result = subprocess.run(
            [
                codesign,
                "-d",
                f"--extract-certificates={certificate_prefix}",
                str(app_path),
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip()
            raise SystemExit(
                f"error: codesign could not extract signing certificates from {app_path}: {detail}"
            )
        leaf_path = Path(f"{certificate_prefix}0")
        if not leaf_path.is_file():
            raise SystemExit(
                f"error: codesign returned no leaf signing certificate for {app_path}"
            )
        leaf_certificate = leaf_path.read_bytes()
        if not leaf_certificate:
            raise SystemExit(
                f"error: codesign returned an empty leaf signing certificate for {app_path}"
            )
        return leaf_certificate


def read_embedded_profile(app_path: Path) -> dict[str, Any]:
    profile_path = app_path / "embedded.mobileprovision"
    if not profile_path.is_file():
        raise SystemExit(f"error: signed app is missing embedded.mobileprovision: {app_path}")
    security = shutil.which("security")
    if sys.platform != "darwin" or not security:
        raise SystemExit("error: provisioning profile validation requires macOS security")
    result = subprocess.run(
        [security, "cms", "-D", "-i", str(profile_path)],
        text=False,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise SystemExit(f"error: could not decode provisioning profile {profile_path}: {detail}")
    return read_plist_bytes(result.stdout, str(profile_path))


def read_signed_artifact_evidence(app_path: Path) -> SignedArtifactEvidence:
    signed_entitlements = read_signed_entitlements(app_path)
    details = read_code_signature_details(app_path)
    info = read_plist(app_path / "Info.plist")
    bundle_identifier = info.get("CFBundleIdentifier")
    if not isinstance(bundle_identifier, str) or not bundle_identifier:
        raise SystemExit(f"error: signed app has no CFBundleIdentifier: {app_path}")

    def first_detail(key: str) -> str:
        values = details.get(key, [])
        return values[0] if values else ""

    return SignedArtifactEvidence(
        bundle_identifier=bundle_identifier,
        signature_identifier=first_detail("Identifier"),
        signature_team_identifier=first_detail("TeamIdentifier"),
        signature_marker=first_detail("Signature"),
        authorities=tuple(details.get("Authority", [])),
        leaf_signing_certificate=read_leaf_signing_certificate(app_path),
        signed_entitlements=signed_entitlements,
        profile=read_embedded_profile(app_path),
    )


def _authority_kind(authorities: tuple[str, ...]) -> str | None:
    for authority in authorities:
        if authority == APPLE_DISTRIBUTION_AUTHORITY or authority.startswith(
            f"{APPLE_DISTRIBUTION_AUTHORITY}:"
        ):
            return "distribution"
        if authority == APPLE_DEVELOPMENT_AUTHORITY or authority.startswith(
            f"{APPLE_DEVELOPMENT_AUTHORITY}:"
        ):
            return "development"
    return None


def _string_values(value: Any) -> set[str]:
    if isinstance(value, str) and value:
        return {value}
    if isinstance(value, list):
        return {item for item in value if isinstance(item, str) and item}
    return set()


def _profile_value_authorizes(profile_value: Any, signed_value: Any) -> bool:
    if profile_value == "*":
        return True
    if isinstance(profile_value, str) and isinstance(signed_value, str):
        if profile_value.endswith("*"):
            return signed_value.startswith(profile_value[:-1])
        return profile_value == signed_value
    if isinstance(profile_value, list):
        signed_values = signed_value if isinstance(signed_value, list) else [signed_value]
        return all(
            any(_profile_value_authorizes(candidate, item) for candidate in profile_value)
            for item in signed_values
        )
    if isinstance(profile_value, dict) and isinstance(signed_value, dict):
        return all(
            key in profile_value and _profile_value_authorizes(profile_value[key], value)
            for key, value in signed_value.items()
        )
    return profile_value == signed_value


def _profile_controls_entitlement(key: str) -> bool:
    return (
        key
        in {
            "application-identifier",
            "com.apple.application-identifier",
            "aps-environment",
            "keychain-access-groups",
            "com.apple.security.application-groups",
        }
        or key.startswith("com.apple.developer.")
        or key.startswith("com.apple.security.")
    )


def _utc_datetime(value: Any) -> datetime | None:
    if not isinstance(value, datetime):
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def validate_signing_evidence(
    evidence: SignedArtifactEvidence,
    signing_stage: str,
    source: str,
) -> list[str]:
    if signing_stage not in SIGNING_STAGES:
        raise ValueError(f"unsupported signing stage: {signing_stage}")

    failures: list[str] = []
    authority_kind = _authority_kind(evidence.authorities)
    if evidence.signature_marker.lower() == "adhoc":
        failures.append(f"{source}: ad-hoc code signatures are not release signing evidence")
    if authority_kind is None:
        failures.append(
            f"{source}: signature authority must be Apple Development or Apple Distribution"
        )
    elif signing_stage == "app-store" and authority_kind != "distribution":
        failures.append(f"{source}: App Store artifact must use an Apple Distribution signature")

    if evidence.signature_identifier != evidence.bundle_identifier:
        failures.append(
            f"{source}: codesign identifier {evidence.signature_identifier!r} does not match "
            f"CFBundleIdentifier {evidence.bundle_identifier!r}"
        )

    profile = evidence.profile
    profile_entitlements = profile.get("Entitlements")
    if not isinstance(profile_entitlements, dict):
        failures.append(f"{source}: embedded profile has no Entitlements dictionary")
        profile_entitlements = {}

    expiration_date = _utc_datetime(profile.get("ExpirationDate"))
    if expiration_date is None:
        failures.append(f"{source}: embedded profile has no valid ExpirationDate")
    elif expiration_date <= datetime.now(timezone.utc):
        failures.append(
            f"{source}: embedded profile expired at {expiration_date.isoformat()}"
        )

    developer_certificates = profile.get("DeveloperCertificates")
    if not isinstance(developer_certificates, list) or not developer_certificates:
        failures.append(f"{source}: embedded profile has no DeveloperCertificates")
    else:
        valid_profile_certificates = [
            bytes(certificate)
            for certificate in developer_certificates
            if isinstance(certificate, (bytes, bytearray)) and certificate
        ]
        if len(valid_profile_certificates) != len(developer_certificates):
            failures.append(
                f"{source}: embedded profile contains an invalid DeveloperCertificates entry"
            )
        if not evidence.leaf_signing_certificate:
            failures.append(f"{source}: code signature has no extractable leaf certificate")
        elif evidence.leaf_signing_certificate not in valid_profile_certificates:
            failures.append(
                f"{source}: leaf signing certificate is not authorized by embedded profile"
            )

    signature_team = evidence.signature_team_identifier
    profile_teams = _string_values(profile.get("TeamIdentifier"))
    if not signature_team:
        failures.append(f"{source}: code signature has no TeamIdentifier")
    elif signature_team not in profile_teams:
        failures.append(
            f"{source}: code-signing team {signature_team!r} is not authorized by embedded profile"
        )
    for label, entitlements in (
        ("signed entitlements", evidence.signed_entitlements),
        ("profile entitlements", profile_entitlements),
    ):
        entitlement_team = entitlements.get("com.apple.developer.team-identifier")
        if entitlement_team != signature_team:
            failures.append(
                f"{source}: {label} team {entitlement_team!r} does not match "
                f"code-signing team {signature_team!r}"
            )

    signed_app_identifier = evidence.signed_entitlements.get("application-identifier")
    profile_app_identifier = profile_entitlements.get("application-identifier")
    if not isinstance(signed_app_identifier, str) or not signed_app_identifier:
        failures.append(f"{source}: signed entitlements have no application-identifier")
    if not isinstance(profile_app_identifier, str) or not profile_app_identifier:
        failures.append(f"{source}: embedded profile has no application-identifier entitlement")
    if signed_app_identifier != profile_app_identifier:
        failures.append(
            f"{source}: signed application-identifier does not match embedded profile"
        )
    identifier_suffix = f".{evidence.bundle_identifier}"
    if isinstance(signed_app_identifier, str) and not signed_app_identifier.endswith(identifier_suffix):
        failures.append(
            f"{source}: signed application-identifier does not match bundle "
            f"{evidence.bundle_identifier!r}"
        )
    if isinstance(profile_app_identifier, str) and profile_app_identifier.endswith(identifier_suffix):
        app_identifier_prefix = profile_app_identifier[: -len(identifier_suffix)]
        profile_prefixes = _string_values(profile.get("ApplicationIdentifierPrefix"))
        if app_identifier_prefix not in profile_prefixes:
            failures.append(
                f"{source}: application-identifier prefix is not authorized by embedded profile"
            )

    signed_task_allow = evidence.signed_entitlements.get("get-task-allow")
    profile_task_allow = profile_entitlements.get("get-task-allow")
    if not isinstance(signed_task_allow, bool):
        failures.append(f"{source}: signed get-task-allow must be an explicit boolean")
    if not isinstance(profile_task_allow, bool):
        failures.append(f"{source}: profile get-task-allow must be an explicit boolean")
    if isinstance(signed_task_allow, bool) and signed_task_allow != profile_task_allow:
        failures.append(f"{source}: signed get-task-allow does not match embedded profile")

    if signing_stage == "app-store":
        if signed_task_allow is not False or profile_task_allow is not False:
            failures.append(f"{source}: App Store artifact requires get-task-allow=false")
        if evidence.signed_entitlements.get("beta-reports-active") is not True:
            failures.append(
                f"{source}: App Store signed entitlements require beta-reports-active=true"
            )
        if profile_entitlements.get("beta-reports-active") is not True:
            failures.append(f"{source}: App Store profile requires beta-reports-active=true")
        if "ProvisionedDevices" in profile:
            failures.append(f"{source}: App Store profile must not contain ProvisionedDevices")
        if profile.get("ProvisionsAllDevices") is True:
            failures.append(f"{source}: enterprise ProvisionsAllDevices profile is not App Store signing")

    for key, signed_value in evidence.signed_entitlements.items():
        if not _profile_controls_entitlement(key):
            continue
        if key not in profile_entitlements:
            failures.append(
                f"{source}: embedded profile does not authorize signed entitlement {key!r}"
            )
            continue
        if not _profile_value_authorizes(profile_entitlements[key], signed_value):
            failures.append(
                f"{source}: embedded profile value for {key!r} does not authorize signed value"
            )

    return failures


def validate_signed_entitlements(
    signed_entitlements: dict[str, Any],
    expected_entitlements_path: Path,
    source: str,
    *,
    build_settings: Mapping[str, str] | None = None,
) -> list[str]:
    expected_entitlements = read_plist(expected_entitlements_path)
    resolved_build_settings = build_settings or {}
    failures: list[str] = []
    for key in signed_entitlements:
        if message := sanitized_entitlement_message(key):
            failures.append(f"{source}: signed entitlement '{key}' must not be present. {message}")
        elif message := disallowed_entitlement_message(key):
            failures.append(f"{source}: disallowed signed entitlement '{key}'. {message}")

    for key, unresolved_expected_value in expected_entitlements.items():
        try:
            expected_value = resolve_expected_build_settings(
                unresolved_expected_value,
                resolved_build_settings,
            )
        except ValueError as error:
            failures.append(
                f"{source}: cannot resolve expected signed entitlement '{key}' "
                f"from {expected_entitlements_path}: {error}"
            )
            continue
        if key not in signed_entitlements:
            failures.append(
                f"{source}: missing expected signed entitlement '{key}' from {expected_entitlements_path}"
            )
            continue
        actual_value = signed_entitlements[key]
        if expected_value != actual_value:
            failures.append(
                f"{source}: signed entitlement '{key}' is {format_plist_value(actual_value)}; "
                f"expected {format_plist_value(expected_value)} from {expected_entitlements_path}"
            )

    extra_project_keys = sorted(
        key
        for key in signed_entitlements
        if key not in expected_entitlements
        and key not in STANDARD_SIGNED_ENTITLEMENTS
    )
    if extra_project_keys:
        failures.append(
            f"{source}: unexpected signed entitlements not allowlisted or tracked in "
            f"{expected_entitlements_path}: {', '.join(extra_project_keys)}"
        )
    return failures


def entitlement_build_settings(evidence: SignedArtifactEvidence) -> dict[str, str]:
    """Derive the exact values Xcode substituted into signed entitlements."""
    suffix = f".{evidence.bundle_identifier}"
    signed_app_identifier = evidence.signed_entitlements.get("application-identifier")
    app_identifier_prefix = ""
    if isinstance(signed_app_identifier, str) and signed_app_identifier.endswith(suffix):
        app_identifier_prefix = signed_app_identifier[: -len(suffix)]
    if not app_identifier_prefix:
        profile_prefixes = sorted(
            _string_values(evidence.profile.get("ApplicationIdentifierPrefix"))
        )
        if len(profile_prefixes) == 1:
            app_identifier_prefix = profile_prefixes[0].rstrip(".")

    prefix_value = f"{app_identifier_prefix}." if app_identifier_prefix else ""
    team_identifier = evidence.signature_team_identifier
    settings: dict[str, str] = {}
    if prefix_value:
        settings["AppIdentifierPrefix"] = prefix_value
        settings["TeamIdentifierPrefix"] = prefix_value
    if team_identifier:
        settings["DEVELOPMENT_TEAM"] = team_identifier
        settings["DevelopmentTeam"] = team_identifier
        settings["TeamIdentifier"] = team_identifier
    return settings


def validate_signed_app(
    path: Path,
    expected_entitlements_path: Path,
    signing_stage: str,
) -> list[str]:
    with resolved_signed_app_path(path) as app_path:
        evidence = read_signed_artifact_evidence(app_path)
        source = str(app_path)
        return [
            *validate_signed_entitlements(
                evidence.signed_entitlements,
                expected_entitlements_path,
                source,
                build_settings=entitlement_build_settings(evidence),
            ),
            *validate_signing_evidence(evidence, signing_stage, source),
        ]


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
        "--signing-stage",
        choices=("archive", "app-store"),
        help=(
            "Required when --signed-app-path is used. Archive accepts a verified Apple Development "
            "or Apple Distribution signature; app-store requires final Store distribution signing."
        ),
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
    if args.signed_app_path and args.signing_stage is None:
        parser.error("--signing-stage archive|app-store is required with --signed-app-path")
    if args.signing_stage is not None and not args.signed_app_path:
        parser.error("--signing-stage requires --signed-app-path")

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
        failures.extend(
            validate_signed_app(
                signed_app_path,
                args.app_store_entitlements,
                args.signing_stage,
            )
        )
    if failures:
        print("iOS signing capability validation failed:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1

    print("iOS signing capability validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
