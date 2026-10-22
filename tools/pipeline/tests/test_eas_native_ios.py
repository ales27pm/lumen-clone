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


def test_development_profile_is_native_internal_debug_with_a_separate_workflow():
    config = json.loads((ROOT / "eas.json").read_text())
    development = config["build"]["development"]
    assert development["distribution"] == "internal"
    assert development["developmentClient"] is False
    assert development["credentialsSource"] == "remote"
    assert development["env"]["LUMEN_EAS_DISTRIBUTION"] == "internal"
    assert development["env"]["EXPO_NO_CAPABILITY_SYNC"] == "1"
    assert development["ios"]["buildConfiguration"] == "Debug"
    assert development["ios"]["simulator"] is False
    assert development["ios"]["config"] == "ios-development.yml"
    assert "development" not in config["submit"]
    assert config["build"]["testflight"]["distribution"] == "store"
    assert config["build"]["testflight"]["ios"]["buildConfiguration"] == "Release"
    workflow = (ROOT / ".eas/build/ios-development.yml").read_text()
    assert "eas/resolve_apple_team_id_from_credentials" in workflow
    assert "bash scripts/eas-native-ios.sh --preflight" in workflow
    assert "--collect-recovery" in workflow
    assert "eas/submit" not in workflow


def test_internal_archive_preserves_clean_provenance_and_distribution_symbol_checks(candidate_repo):
    env = {**os.environ, "LUMEN_IOS_CONFIGURATION": "Debug", "LUMEN_IOS_DISTRIBUTION": "internal"}
    command = ["bash", "scripts/archive_lumen_stable.sh", "--preflight-release-provenance"]
    result = subprocess.run(command, cwd=candidate_repo, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    (candidate_repo / "scripts/untracked.sh").write_text("fixture")
    result = subprocess.run(command, cwd=candidate_repo, env=env, capture_output=True, text=True)
    assert result.returncode != 0
    assert "dirty" in result.stderr
    source = (ROOT / "scripts/archive_lumen_stable.sh").read_text()
    assert source.count("if requires_distribution_symbols; then") == 2
    assert 'SOURCE_ENTITLEMENTS_PATH="$REPO_ROOT/ios/Lumen/LumenAppStore.entitlements"' in source


@pytest.mark.parametrize("distribution,configuration", (("internal", "Release"), ("store", "Debug"), ("guess", "Release")))
def test_archive_preflight_rejects_mismatched_distribution_and_configuration(candidate_repo, distribution, configuration):
    env = {**os.environ, "LUMEN_IOS_CONFIGURATION": configuration, "LUMEN_IOS_DISTRIBUTION": distribution}
    result = subprocess.run(["bash", "scripts/archive_lumen_stable.sh", "--preflight-release-provenance"], cwd=candidate_repo, env=env, capture_output=True, text=True)
    assert result.returncode != 0
    assert not (candidate_repo / "build").exists()


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


def choose(*profiles: dict, identities: set[str] | None = None, distribution: str = "store") -> dict:
    return support.profile_choice(
        [(Path(f"{i}.mobileprovision"), value) for i, value in enumerate(profiles)],
        identities if identities is not None else {CERTIFICATE_SHA1},
        EXPECTED,
        now=datetime(2026, 10, 21, tzinfo=timezone.utc),
        distribution=distribution,
    )


def internal_profile() -> dict:
    value = profile()
    value["ProvisionedDevices"] = ["F" * 40]
    value["Entitlements"].pop("beta-reports-active")
    return value


def test_internal_profile_selection_and_export_are_distinct_from_store():
    value = internal_profile()
    context = choose(value, distribution="internal")
    assert context["distribution"] == "internal"
    options = support.export_options(context, "internal")
    assert options["method"] == "release-testing"
    assert options["uploadSymbols"] is False
    assert options["destination"] == "export"
    assert options["manageAppVersionAndBuildNumber"] is False
    with pytest.raises(ValueError, match="expected one authorized"):
        choose(value)
    with pytest.raises(ValueError, match="expected one authorized"):
        choose(profile(), distribution="internal")
    with pytest.raises(ValueError, match="distribution"):
        support.export_options(context)
    with pytest.raises(ValueError, match="invalid runtime signing context"):
        support.signing_xcconfig(context)


@pytest.mark.parametrize("mutation", ("empty", "not_list", "invalid", "duplicate", "enterprise", "development", "beta", "certificate", "capability"))
def test_internal_profile_rejects_ineligible_signing_and_device_lists(mutation):
    value = internal_profile()
    if mutation == "empty":
        value["ProvisionedDevices"] = []
    elif mutation == "not_list":
        value["ProvisionedDevices"] = "F" * 40
    elif mutation == "invalid":
        value["ProvisionedDevices"] = [{}]
    elif mutation == "duplicate":
        value["ProvisionedDevices"] *= 2
    elif mutation == "enterprise":
        value["ProvisionsAllDevices"] = True
    elif mutation == "development":
        value["Entitlements"]["get-task-allow"] = True
    elif mutation == "beta":
        value["Entitlements"]["beta-reports-active"] = True
    elif mutation == "certificate":
        value["DeveloperCertificates"] = [b"unavailable certificate"]
    elif mutation == "capability":
        del value["Entitlements"]["com.apple.developer.healthkit"]
    with pytest.raises(ValueError, match="expected one authorized"):
        choose(value, distribution="internal")


def test_internal_debug_settings_are_target_scoped_and_keep_other_targets_inherited():
    settings = support.signing_xcconfig(choose(internal_profile(), distribution="internal"), "internal")
    assert "LUMEN_SIGNING_STYLE_Lumen = Manual" in settings
    assert "CODE_SIGN_STYLE = $(LUMEN_SIGNING_STYLE_$(TARGET_NAME):default=$(inherited))" in settings
    assert "LUMEN_DSYM_Lumen = dwarf-with-dsym" in settings
    assert "DEBUG_INFORMATION_FORMAT = $(LUMEN_DSYM_$(TARGET_NAME):default=$(inherited))" in settings
    assert "ENABLE_DEBUG_DYLIB = $(LUMEN_DEBUG_DYLIB_$(TARGET_NAME):default=$(inherited))" in settings
    assert "LUMEN_DEBUG_DYLIB_Lumen = NO" in settings
    assert "DEBUG_INFORMATION_FORMAT" not in support.signing_xcconfig(choose(profile()))


def internal_evidence():
    return support.signing.SignedArtifactEvidence(
        bundle_identifier=support.BUNDLE, signature_identifier=support.BUNDLE,
        signature_team_identifier=support.TEAM, signature_marker="signed",
        authorities=("Apple Distribution: Fixture",), leaf_signing_certificate=CERTIFICATE,
        signed_entitlements={"get-task-allow": False}, profile=internal_profile(),
    )


@pytest.mark.parametrize("mutation", (None, "no_devices", "signed_debuggable", "profile_debuggable", "development", "profile", "certificate", "enterprise", "signed_beta", "profile_beta", "team"))
def test_final_internal_signature_requires_the_selected_ad_hoc_profile(mutation):
    from dataclasses import replace
    evidence = internal_evidence()
    context = choose(internal_profile(), distribution="internal")
    if mutation == "no_devices":
        evidence.profile.pop("ProvisionedDevices")
    elif mutation == "signed_debuggable":
        evidence.signed_entitlements["get-task-allow"] = True
    elif mutation == "profile_debuggable":
        evidence.profile["Entitlements"]["get-task-allow"] = True
    elif mutation == "development":
        evidence = replace(evidence, authorities=("Apple Development: Fixture",))
    elif mutation == "profile":
        evidence.profile["UUID"] = "different-profile"
    elif mutation == "certificate":
        evidence = replace(evidence, leaf_signing_certificate=b"different certificate")
    elif mutation == "enterprise":
        evidence.profile["ProvisionsAllDevices"] = True
    elif mutation == "signed_beta":
        evidence.signed_entitlements["beta-reports-active"] = True
    elif mutation == "profile_beta":
        evidence.profile["Entitlements"]["beta-reports-active"] = True
    elif mutation == "team":
        evidence = replace(evidence, signature_team_identifier="OTHERTEAM")
    if mutation is None:
        support.validate_internal_evidence(evidence, context)
    else:
        with pytest.raises(ValueError):
            support.validate_internal_evidence(evidence, context)


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


def test_scoped_xcconfig_rejects_a_signing_context_that_could_inject_settings():
    value = choose(profile())
    value["profileUUID"] += "\nCODE_SIGNING_ALLOWED = NO"
    with pytest.raises(ValueError, match="invalid runtime signing context"):
        support.signing_xcconfig(value)


@pytest.mark.parametrize("xcode,sdk", (("Xcode 26.3", "26.3"), ("Xcode 27.1", "26.9"), ("unavailable", "27.1"), ("Xcode 27.1", "unexpected sdk")))
def test_toolchain_preflight_rejects_unsupported_or_unparseable_sdk(xcode, sdk):
    with pytest.raises(ValueError):
        support.validate_toolchain(xcode, sdk, "27.1")


@pytest.mark.parametrize("target,sdk", (("27.1", "27.0"), ("27.1", "27.0.99"), ("27.2", "27.1")))
def test_toolchain_rejects_sdk_older_than_the_actual_candidate_target(target, sdk):
    with pytest.raises(ValueError, match="cannot archive the required iOS"):
        support.validate_toolchain("Xcode 27.1\nBuild version 27A9275", sdk, target)


@pytest.mark.parametrize("sdk", ("27.1", "27.1.1", "27.2", "27.10", "28.0"))
def test_toolchain_accepts_numeric_sdk_at_or_above_candidate_target(sdk):
    support.validate_toolchain("Xcode 27.1\nBuild version fixture", sdk, "27.1")


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
    assert result["deploymentTarget"] == "27.1"
    assert (result["configuration"], result["distribution"]) == ("Release", "store")
    internal = support.candidate(candidate_repo, distribution="internal")
    assert (internal["configuration"], internal["distribution"]) == ("Debug", "internal")
    with pytest.raises(ValueError, match="unsupported native distribution"):
        support.candidate(candidate_repo, distribution="guess")
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
        "LumenBuildScheme": "Lumen", "LumenBuildConfiguration": "Release", "MinimumOSVersion": "27.1",
    }
    support.verify_info(info, source)
    info[field] = "wrong candidate"
    with pytest.raises(ValueError, match="frozen candidate"):
        support.verify_info(info, source)


@pytest.mark.parametrize("distribution", ("store", "internal"))
def test_artifact_receipt_binds_the_exact_ipa_and_original_symbol_bytes(candidate_repo, tmp_path, distribution):
    source = support.candidate(candidate_repo, distribution=distribution)
    info = {
        "CFBundleIdentifier": source["bundleIdentifier"], "CFBundleVersion": source["buildNumber"],
        "CFBundleShortVersionString": source["marketingVersion"], "LumenGitSHA": source["sourceRevision"],
        "LumenBuildScheme": "Lumen", "LumenBuildConfiguration": source["configuration"], "MinimumOSVersion": "27.1",
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
    assert receipt["configuration"] == source["configuration"]
    assert receipt["distribution"] == distribution
    with zipfile.ZipFile(output.parent / "Lumen.dSYMs.zip") as symbols:
        assert symbols.read(dwarf.relative_to(archive).as_posix()) == dwarf.read_bytes()
    assert "profileUUID" not in receipt and "certificateSHA1" not in receipt


def _executable(path, text):
    path.write_text(text)
    path.chmod(0o755)


def _executor_tools(tmp_path):
    tools = tmp_path / "tools"
    tools.mkdir()
    _executable(tools / "uname", "#!/bin/sh\nprintf 'Darwin\\n'\n")
    _executable(tools / "xcodebuild", "#!/bin/sh\n[ \"$1\" = -version ] || exit 97\nprintf 'Xcode 27.1\\nBuild version fixture\\n'\n")
    _executable(tools / "xcrun", """#!/bin/sh
if [ "$1" = swiftc ] && [ "$2" = --version ]; then printf 'Swift fixture\\n'; exit 0; fi
if [ "$3" = --show-sdk-version ]; then printf '%s\\n' "$FIXTURE_SDK_VERSION"; exit 0; fi
if [ "$3" = --show-sdk-path ]; then printf '/fixture-sdk\\n'; exit 0; fi
if [ "$3" = swiftc ] && [ "$4" = -typecheck ]; then exit "$FIXTURE_TYPECHECK_EXIT"; fi
exit 98
""")
    return tools


@pytest.mark.parametrize("distribution", ("store", "internal"))
@pytest.mark.parametrize("sdk,typecheck_exit", (("27.1", 0), ("27.1", 43), ("27.0", 0)))
def test_preflight_runs_no_archive_and_preserves_sdk_or_compiler_failure(candidate_repo, tmp_path, sdk, typecheck_exit, distribution):
    tools = _executor_tools(tmp_path)
    env = {**os.environ, "PATH": str(tools) + os.pathsep + os.environ["PATH"], "FIXTURE_TYPECHECK_EXIT": str(typecheck_exit), "FIXTURE_SDK_VERSION": sdk}
    env.pop("EAS_BUILD_GIT_COMMIT_HASH", None)
    env.pop("LUMEN_GIT_SHA", None)
    env["LUMEN_EAS_DISTRIBUTION"] = distribution
    result = subprocess.run(["bash", "scripts/eas-native-ios.sh", "--preflight"], cwd=candidate_repo, env=env, capture_output=True, text=True, check=False)
    assert (result.returncode == 0) is (sdk == "27.1" and typecheck_exit == 0), result.stdout + result.stderr
    assert not (candidate_repo / "build/eas-archive").exists()
    assert not (candidate_repo / "build/eas-private/signing.json").exists()
    assert (candidate_repo / "build/eas-evidence/coreai-sdk-typecheck.log").exists() is (sdk == "27.1")
    assert (candidate_repo / "build/eas-evidence/source-preflight.log").exists() is (sdk == "27.1" and typecheck_exit == 0)
    source = json.loads((candidate_repo / "build/eas-evidence/candidate.json").read_text())
    assert source["distribution"] == distribution
    assert source["configuration"] == ("Debug" if distribution == "internal" else "Release")


@pytest.mark.parametrize("distribution", ("store", "internal"))
def test_archive_receives_private_scoped_config_and_no_global_profile(candidate_repo, tmp_path, distribution):
    tools = _executor_tools(tmp_path)
    _executable(tools / "security", "#!/bin/sh\nexit 99\n")
    _executable(tools / "python3", f"""#!{sys.executable}
import json, os, pathlib, sys
if len(sys.argv) > 2 and sys.argv[2] == 'signing':
    pathlib.Path(sys.argv[sys.argv.index('--output') + 1]).write_text(os.environ['FIXTURE_SIGNING_JSON'])
else:
    os.execv({sys.executable!r}, [{sys.executable!r}, *sys.argv[1:]])
""")
    (candidate_repo / "scripts/archive_lumen_stable.sh").write_text("""#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == "--preflight-release-provenance" ]]; then exit 0; fi
python3 - <<'PY'
import json, os
from pathlib import Path
keys = ['XCODE_XCCONFIG_FILE', 'LUMEN_IOS_CODE_SIGN_STYLE', 'LUMEN_IOS_CODE_SIGN_IDENTITY', 'LUMEN_IOS_DEVELOPMENT_TEAM', 'LUMEN_IOS_PROVISIONING_PROFILE_SPECIFIER', 'LUMEN_IOS_CONFIGURATION', 'LUMEN_IOS_DISTRIBUTION', 'LUMEN_SWIFT_OPTIMIZATION_LEVEL']
Path('build/eas-evidence/archive-environment.json').write_text(json.dumps({key:os.environ.get(key) for key in keys}))
PY
exit 67
""")
    subprocess.run(["git", "-C", str(candidate_repo), "add", "scripts/archive_lumen_stable.sh"], check=True)
    subprocess.run(["git", "-C", str(candidate_repo), "-c", "user.name=Release Test", "-c", "user.email=release@example.invalid", "commit", "-qm", "archive executor fixture"], check=True)
    signing_context = choose(internal_profile(), distribution="internal") if distribution == "internal" else choose(profile())
    env = {
        **os.environ, "PATH": str(tools) + os.pathsep + os.environ["PATH"],
        "FIXTURE_TYPECHECK_EXIT": "0", "FIXTURE_SDK_VERSION": "27.1", "FIXTURE_SIGNING_JSON": json.dumps(signing_context),
        "LUMEN_EAS_DISTRIBUTION": distribution,
    }
    env.pop("EAS_BUILD_GIT_COMMIT_HASH", None)
    env.pop("LUMEN_GIT_SHA", None)
    before = subprocess.check_output(["git", "-C", str(candidate_repo), "rev-parse", "HEAD"], text=True)
    result = subprocess.run(["bash", "scripts/eas-native-ios.sh"], cwd=candidate_repo, env=env, capture_output=True, text=True, check=False)
    assert result.returncode != 0  # The deliberately failed archive cannot become a successful release.
    signing_env = json.loads((candidate_repo / "build/eas-evidence/archive-environment.json").read_text())
    assert all(signing_env[key] == "" for key in ("LUMEN_IOS_CODE_SIGN_STYLE", "LUMEN_IOS_CODE_SIGN_IDENTITY", "LUMEN_IOS_DEVELOPMENT_TEAM", "LUMEN_IOS_PROVISIONING_PROFILE_SPECIFIER"))
    assert signing_env["LUMEN_IOS_DISTRIBUTION"] == distribution
    assert signing_env["LUMEN_IOS_CONFIGURATION"] == ("Debug" if distribution == "internal" else "Release")
    assert signing_env["LUMEN_SWIFT_OPTIMIZATION_LEVEL"] == ("-Onone" if distribution == "internal" else "-Osize")
    scoped = Path(signing_env["XCODE_XCCONFIG_FILE"])
    assert scoped == candidate_repo / "build/eas-private/Signing.xcconfig"
    assert scoped.read_text() == support.signing_xcconfig(signing_context, distribution)
    assert not (candidate_repo / "build/eas-evidence/artifact.json").exists()
    assert subprocess.check_output(["git", "-C", str(candidate_repo), "status", "--porcelain"], text=True) == ""
    assert subprocess.check_output(["git", "-C", str(candidate_repo), "rev-parse", "HEAD"], text=True) == before


def test_failed_gate_recovery_keeps_ipa_and_original_symbols_without_success_receipt(candidate_repo):
    evidence = candidate_repo / "build/eas-evidence"
    source = support.candidate(candidate_repo)
    support.write_json(evidence / "candidate.json", source)
    ipa = candidate_repo / "build/eas-export/Lumen.ipa"
    ipa.parent.mkdir(parents=True)
    ipa.write_bytes(b"unverified exported IPA fixture")
    dwarf = candidate_repo / "build/eas-archive/Lumen.xcarchive/dSYMs/Lumen.app.dSYM/Contents/Resources/DWARF/Lumen"
    dwarf.parent.mkdir(parents=True)
    dwarf.write_bytes(b"original symbol fixture")
    result = subprocess.run(["bash", "scripts/eas-native-ios.sh", "--collect-recovery"], cwd=candidate_repo, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert not (evidence / "artifact.json").exists()
    receipt = json.loads((evidence / "recovery/recovery.json").read_text())
    assert receipt["status"] == "unverified-recovery-not-for-submission"
    assert receipt["candidateSourceRevision"] == source["sourceRevision"]
    recovered = evidence / "recovery/Lumen.unverified.ipa"
    assert recovered.read_bytes() == ipa.read_bytes()
    assert receipt["files"][recovered.name]["sha256"] == hashlib.sha256(ipa.read_bytes()).hexdigest()
    symbols = evidence / "recovery/Lumen.original-dSYMs.unverified.zip"
    assert receipt["files"][symbols.name]["sha256"] == hashlib.sha256(symbols.read_bytes()).hexdigest()
    with zipfile.ZipFile(symbols) as archived:
        member = "dSYMs/Lumen.app.dSYM/Contents/Resources/DWARF/Lumen"
        assert archived.read(member) == dwarf.read_bytes()
        assert receipt["dSYMFiles"][member] == hashlib.sha256(dwarf.read_bytes()).hexdigest()


def test_transport_failure_recovery_does_not_drop_an_already_verified_ipa(tmp_path):
    ipa = tmp_path / "export/Lumen.ipa"
    ipa.parent.mkdir()
    ipa.write_bytes(b"already verified artifact fixture")
    evidence = tmp_path / "evidence"
    support.write_json(evidence / "artifact.json", {
        "status": "verified-ipa-not-submitted", "ipaSHA256": hashlib.sha256(ipa.read_bytes()).hexdigest(),
    })
    original_receipt = (evidence / "artifact.json").read_bytes()
    support.collect_recovery(ipa, tmp_path / "archive/Lumen.xcarchive", evidence)
    assert (evidence / "recovery/Lumen.unverified.ipa").read_bytes() == ipa.read_bytes()
    assert (evidence / "artifact.json").read_bytes() == original_receipt


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
