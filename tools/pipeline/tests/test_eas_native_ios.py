"""Fail-closed release orchestration tests; fake executors are not native build proof."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import plistlib
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location("eas_native_ios_support", SCRIPTS / "eas_native_ios_support.py")
assert SPEC and SPEC.loader
support = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(support)
UUID = "00112233-4455-6677-8899-AABBCCDDEEFF"
CERTIFICATE = b"public certificate DER fixture"
CERTIFICATE_SHA1 = hashlib.sha1(CERTIFICATE).hexdigest().upper()


def profile() -> dict:
    return {
        "UUID": UUID,
        "Name": "fixture App Store profile",
        "TeamIdentifier": [support.TEAM],
        "ApplicationIdentifierPrefix": [support.TEAM],
        "ExpirationDate": datetime(2100, 1, 1),
        "DeveloperCertificates": [CERTIFICATE],
        "Entitlements": {
            "application-identifier": f"{support.TEAM}.{support.BUNDLE}",
            "com.apple.developer.team-identifier": support.TEAM,
            "get-task-allow": False,
            "beta-reports-active": True,
            "com.apple.developer.healthkit": True,
            "keychain-access-groups": [f"{support.TEAM}.*"],
        },
    }


EXPECTED = {
    "com.apple.developer.healthkit": True,
    "keychain-access-groups": ["$(AppIdentifierPrefix)com.microsoft.adalcache"],
}


def choose(*profiles: dict, identities: set[str] | None = None) -> dict:
    return support.profile_choice(
        [(Path(f"{i}.mobileprovision"), value) for i, value in enumerate(profiles)],
        identities if identities is not None else {CERTIFICATE_SHA1},
        EXPECTED,
        now=datetime(2026, 10, 21, tzinfo=timezone.utc),
    )


def test_profile_selection_uses_actual_certificate_and_resolves_entitlement_prefix():
    value = choose(profile())
    assert value["profileUUID"] == UUID
    assert value["certificateSHA1"] == CERTIFICATE_SHA1
    assert value["bundleIdentifier"] == support.BUNDLE
    # Installed duplicate copies of the exact same profile are one credential pair.
    assert choose(profile(), profile())["profileUUID"] == UUID


@pytest.mark.parametrize("mutation", (
    "team", "bundle", "prefix", "expired", "adhoc", "enterprise", "development",
    "no_beta", "unavailable_certificate", "missing_capability", "unauthorized_keychain", "invalid_uuid",
))
def test_profile_selection_rejects_incompatible_or_unavailable_credentials(mutation):
    value = profile()
    entitlements = value["Entitlements"]
    if mutation == "team":
        value["TeamIdentifier"] = ["OTHERTEAM"]
    elif mutation == "bundle":
        entitlements["application-identifier"] += ".other"
    elif mutation == "prefix":
        value["ApplicationIdentifierPrefix"] = ["OTHERPREFIX"]
    elif mutation == "expired":
        value["ExpirationDate"] = datetime(2020, 1, 1)
    elif mutation == "adhoc":
        value["ProvisionedDevices"] = ["private device fixture"]
    elif mutation == "enterprise":
        value["ProvisionsAllDevices"] = True
    elif mutation == "development":
        entitlements["get-task-allow"] = True
    elif mutation == "no_beta":
        del entitlements["beta-reports-active"]
    elif mutation == "unavailable_certificate":
        value["DeveloperCertificates"] = [b"another certificate"]
    elif mutation == "missing_capability":
        del entitlements["com.apple.developer.healthkit"]
    elif mutation == "unauthorized_keychain":
        entitlements["keychain-access-groups"] = ["OTHERPREFIX.*"]
    elif mutation == "invalid_uuid":
        value["UUID"] = "invalid profile UUID"
    with pytest.raises(ValueError, match="expected one authorized"):
        choose(value)


def test_profile_selection_fails_closed_on_ambiguity():
    other = profile()
    other["UUID"] = "11223344-5566-7788-99AA-BBCCDDEEFF00"
    with pytest.raises(ValueError, match="found 2"):
        choose(profile(), other)
    changed = profile()
    changed["Name"] = "different profile with duplicate UUID"
    with pytest.raises(ValueError, match="found 2"):
        choose(profile(), changed)


def test_revoked_or_development_identity_output_is_not_accepted():
    distribution = f' 1) {CERTIFICATE_SHA1} "Apple Distribution: Fixture (TEAM)"'
    revoked = distribution + " (CSSMERR_TP_CERT_REVOKED)"
    development = distribution.replace("Apple Distribution", "Apple Development")
    assert support.valid_identities(distribution) == {CERTIFICATE_SHA1}
    assert support.valid_identities(revoked + "\n" + development + "\n0 valid identities found") == set()
    with pytest.raises(ValueError, match="found 0"):
        choose(profile(), identities=set())


def test_export_options_preserve_version_and_use_runtime_credentials():
    options = support.export_options(choose(profile()))
    assert options["method"] == "app-store-connect"
    assert options["destination"] == "export"
    assert options["signingStyle"] == "manual"
    assert options["signingCertificate"] == CERTIFICATE_SHA1
    assert options["provisioningProfiles"] == {support.BUNDLE: UUID}
    assert options["manageAppVersionAndBuildNumber"] is False


@pytest.mark.parametrize("xcode,sdk", (("Xcode 26.3", "26.3"), ("Xcode 27.1", "26.9"), ("unavailable", "27.1"), ("Xcode 27.1", "unexpected sdk")))
def test_toolchain_preflight_rejects_unsupported_or_unparseable_sdk(xcode, sdk):
    with pytest.raises(ValueError):
        support.validate_toolchain(xcode, sdk)


def test_toolchain_does_not_lower_the_27_2_runtime_requirement_to_the_build_sdk():
    support.validate_toolchain("Xcode 27.1\nBuild version 27A9275", "27.1")


def test_package_lock_matches_all_four_direct_reviewed_dependencies():
    text = (ROOT / support.PROJECT).read_text()
    resolved = json.loads((ROOT / support.RESOLVED).read_text())
    support.validate_lock(text, resolved)
    broken = copy.deepcopy(resolved)
    broken["pins"] = [p for p in broken["pins"] if p["identity"] != "coreai-models"]
    with pytest.raises(ValueError, match="missing coreai-models"):
        support.validate_lock(text, broken)
    broken = copy.deepcopy(resolved)
    pin = next(p for p in broken["pins"] if p["identity"] == "coreai-models")
    pin["state"]["revision"] = "0" * 40
    with pytest.raises(ValueError, match="reviewed coreai-models"):
        support.validate_lock(text, broken)
    broken["pins"].append(copy.deepcopy(broken["pins"][0]))
    with pytest.raises(ValueError, match="duplicate"):
        support.validate_lock(text, broken)


@pytest.fixture
def candidate_repo(tmp_path):
    repo = tmp_path / "candidate"
    repo.mkdir()
    paths = [support.PROJECT, support.RESOLVED, Path("scripts/eas_native_ios_support.py"), Path("scripts/eas-native-ios.sh"), Path("scripts/archive_lumen_stable.sh")]
    for relative in paths:
        target = repo / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    shutil.copyfile(SCRIPTS / "validate_ios_signing_capabilities.py", repo / "scripts/validate_ios_signing_capabilities.py")
    (repo / ".gitignore").write_text("build/\n**/__pycache__/\n")
    (repo / "app.json").write_text(json.dumps({"expo": {"extra": {"eas": {"projectId": support.EAS_PROJECT}}}}))
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(["git", "-C", str(repo), "-c", "user.name=Release Test", "-c", "user.email=release@example.invalid", "commit", "-qm", "fixture"], check=True)
    return repo


def test_candidate_requires_clean_committed_release_inputs_and_exact_revision(candidate_repo):
    result = support.candidate(candidate_repo)
    assert result["deploymentTarget"] == "27.2"
    assert support.PROJECT.as_posix() in result["sourceSHA256"]
    assert support.RESOLVED.as_posix() in result["sourceSHA256"]
    with pytest.raises(ValueError, match="requested full commit"):
        support.candidate(candidate_repo, "0" * 40)
    (candidate_repo / "scripts/untracked.sh").write_text("private fixture\n")
    with pytest.raises(ValueError, match="dirty or untracked"):
        support.candidate(candidate_repo)


@pytest.mark.parametrize("field", ("CFBundleIdentifier", "CFBundleVersion", "CFBundleShortVersionString", "LumenGitSHA", "LumenBuildScheme", "LumenBuildConfiguration", "MinimumOSVersion"))
def test_exported_info_cannot_be_attributed_to_another_candidate(candidate_repo, field):
    source = support.candidate(candidate_repo)
    info = {
        "CFBundleIdentifier": source["bundleIdentifier"], "CFBundleVersion": source["buildNumber"],
        "CFBundleShortVersionString": source["marketingVersion"], "LumenGitSHA": source["sourceRevision"],
        "LumenBuildScheme": "Lumen", "LumenBuildConfiguration": "Release", "MinimumOSVersion": "27.2",
    }
    support.verify_info(info, source)
    info[field] = "wrong candidate"
    with pytest.raises(ValueError, match="frozen candidate"):
        support.verify_info(info, source)


def test_artifact_receipt_binds_the_exact_ipa_and_original_symbol_bytes(candidate_repo, tmp_path):
    source = support.candidate(candidate_repo)
    info = {
        "CFBundleIdentifier": source["bundleIdentifier"], "CFBundleVersion": source["buildNumber"],
        "CFBundleShortVersionString": source["marketingVersion"], "LumenGitSHA": source["sourceRevision"],
        "LumenBuildScheme": "Lumen", "LumenBuildConfiguration": "Release", "MinimumOSVersion": "27.2",
    }
    ipa = tmp_path / "Lumen.ipa"
    with zipfile.ZipFile(ipa, "w") as package:
        package.writestr("Payload/Lumen.app/Info.plist", plistlib.dumps(info))
    archive = tmp_path / "Lumen.xcarchive"
    archive_info = archive / "Products/Applications/Lumen.app/Info.plist"
    archive_info.parent.mkdir(parents=True)
    archive_info.write_bytes(plistlib.dumps(info))
    dwarf = archive / "dSYMs/Lumen.app.dSYM/Contents/Resources/DWARF/Lumen"
    dwarf.parent.mkdir(parents=True)
    dwarf.write_bytes(b"original symbol fixture, not a real Mach-O")
    output = tmp_path / "evidence/artifact.json"
    output.parent.mkdir()
    support.artifact_receipt(candidate_repo, source, ipa, archive, output)
    receipt = json.loads(output.read_text())
    assert receipt["ipaSHA256"] == hashlib.sha256(ipa.read_bytes()).hexdigest()
    assert receipt["symbolsSHA256"] == hashlib.sha256((output.parent / "Lumen.dSYMs.zip").read_bytes()).hexdigest()
    assert receipt["status"] == "verified-ipa-not-submitted"
    with zipfile.ZipFile(output.parent / "Lumen.dSYMs.zip") as symbols:
        assert symbols.read(dwarf.relative_to(archive).as_posix()) == dwarf.read_bytes()
    assert "profileUUID" not in receipt and "certificateSHA1" not in receipt


def _executable(path, text):
    path.write_text(text)
    path.chmod(0o755)


@pytest.mark.parametrize("typecheck_exit", (0, 43))
def test_preflight_runs_no_archive_and_preserves_failed_compiler_exit(candidate_repo, tmp_path, typecheck_exit):
    tools = tmp_path / "tools"
    tools.mkdir()
    _executable(tools / "uname", "#!/bin/sh\nprintf 'Darwin\\n'\n")
    _executable(tools / "xcodebuild", "#!/bin/sh\n[ \"$1\" = -version ] || exit 97\nprintf 'Xcode 27.1\\nBuild version fixture\\n'\n")
    _executable(tools / "xcrun", """#!/bin/sh
if [ "$1" = swiftc ] && [ "$2" = --version ]; then printf 'Swift fixture\\n'; exit 0; fi
if [ "$3" = --show-sdk-version ]; then printf '27.1\\n'; exit 0; fi
if [ "$3" = --show-sdk-path ]; then printf '/fixture-sdk\\n'; exit 0; fi
if [ "$3" = swiftc ] && [ "$4" = -typecheck ]; then exit "$FIXTURE_TYPECHECK_EXIT"; fi
exit 98
""")
    env = {**os.environ, "PATH": str(tools) + os.pathsep + os.environ["PATH"], "FIXTURE_TYPECHECK_EXIT": str(typecheck_exit)}
    env.pop("EAS_BUILD_GIT_COMMIT_HASH", None)
    env.pop("LUMEN_GIT_SHA", None)
    result = subprocess.run(["bash", "scripts/eas-native-ios.sh", "--preflight"], cwd=candidate_repo, env=env, capture_output=True, text=True, check=False)
    assert (result.returncode == 0) is (typecheck_exit == 0), result.stdout + result.stderr
    assert not (candidate_repo / "build/eas-archive").exists()
    assert not (candidate_repo / "build/eas-private/signing.json").exists()
    assert (candidate_repo / "build/eas-evidence/coreai-sdk-typecheck.log").exists()
    assert (candidate_repo / "build/eas-evidence/source-preflight.log").exists() is (typecheck_exit == 0)


def test_linux_executor_fails_before_generating_or_mutating_outputs(tmp_path):
    if sys.platform != "linux":
        pytest.skip("Linux boundary check")
    folder = tmp_path / "project/scripts"
    folder.mkdir(parents=True)
    shutil.copyfile(SCRIPTS / "eas-native-ios.sh", folder / "eas-native-ios.sh")
    result = subprocess.run(["bash", str(folder / "eas-native-ios.sh"), "--preflight"], capture_output=True, text=True, check=False)
    assert result.returncode == 1
    assert "requires macOS" in result.stderr
    assert not (folder.parent / "build").exists()
