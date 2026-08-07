import importlib.util
import plistlib
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "validate_ios_signing_capabilities.py"
ARCHIVE_SCRIPT = ROOT / "scripts" / "archive_lumen_stable.sh"
SUBMIT_SCRIPT = ROOT / "scripts" / "build_and_submit_appstoreconnect.sh"
READINESS_SCRIPT = ROOT / "scripts" / "check-ios-build-readiness.sh"


def _write_plist(path: Path, value: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        plistlib.dump(value, handle)
    return path


def _write_project(tmp_path: Path, text: str = "// project") -> Path:
    project = tmp_path / "project.pbxproj"
    project.write_text(text, encoding="utf-8")
    return project


def _run_validator(tmp_path: Path, development: dict, app_store: dict) -> subprocess.CompletedProcess[str]:
    development_path = _write_plist(tmp_path / "Lumen.entitlements", development)
    app_store_path = _write_plist(tmp_path / "LumenAppStore.entitlements", app_store)
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--project-file",
            str(_write_project(tmp_path)),
            "--entitlements",
            str(development_path),
            "--app-store-entitlements",
            str(app_store_path),
        ],
        text=True,
        capture_output=True,
        check=False,
    )


def _load_validator_module():
    spec = importlib.util.spec_from_file_location("validate_ios_signing_capabilities", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _signing_evidence(
    module,
    *,
    authority: str = "Apple Development: Test (TEAM123456)",
    get_task_allow: bool = True,
    beta_reports_active: bool | None = None,
    provisioned_devices: bool = True,
    signature_marker: str = "",
    team: str = "TEAM123456",
    bundle_identifier: str = "com.example.lumen",
    leaf_signing_certificate: bytes = b"leaf-signing-certificate",
):
    app_identifier = f"{team}.{bundle_identifier}"
    signed_entitlements = {
        "application-identifier": app_identifier,
        "com.apple.developer.team-identifier": team,
        "com.apple.developer.healthkit": True,
        "get-task-allow": get_task_allow,
    }
    profile_entitlements = dict(signed_entitlements)
    if beta_reports_active is not None:
        signed_entitlements["beta-reports-active"] = beta_reports_active
        profile_entitlements["beta-reports-active"] = beta_reports_active
    profile = {
        "ApplicationIdentifierPrefix": [team],
        "TeamIdentifier": [team],
        "ExpirationDate": datetime(2100, 1, 1),
        "DeveloperCertificates": [leaf_signing_certificate],
        "Entitlements": profile_entitlements,
    }
    if provisioned_devices:
        profile["ProvisionedDevices"] = ["DEVICE-UDID"]
    return module.SignedArtifactEvidence(
        bundle_identifier=bundle_identifier,
        signature_identifier=bundle_identifier,
        signature_team_identifier=team,
        signature_marker=signature_marker,
        authorities=(authority,) if authority else (),
        leaf_signing_certificate=leaf_signing_certificate,
        signed_entitlements=signed_entitlements,
        profile=profile,
    )


def test_ios_signing_capabilities_accepts_sanitized_app_store_profile(tmp_path: Path):
    development = {
        "com.apple.developer.healthkit": True,
        "com.apple.developer.kernel.increased-debugging-memory-limit": True,
        "com.apple.security.hardened-process.checked-allocations.soft-mode": True,
    }
    app_store = {"com.apple.developer.healthkit": True}

    result = _run_validator(tmp_path, development, app_store)

    assert result.returncode == 0
    assert "App Store entitlements match development entitlements after removing" in result.stderr


def test_ios_signing_capabilities_rejects_app_store_profile_drift(tmp_path: Path):
    development = {
        "com.apple.developer.healthkit": True,
        "com.apple.developer.kernel.increased-debugging-memory-limit": True,
    }
    app_store = {
        "com.apple.developer.healthkit": True,
        "com.apple.developer.kernel.increased-debugging-memory-limit": True,
    }

    result = _run_validator(tmp_path, development, app_store)

    assert result.returncode == 1
    assert "extra App Store entitlement 'com.apple.developer.kernel.increased-debugging-memory-limit'" in result.stderr


def test_signed_entitlement_validation_rejects_development_only_keys(tmp_path: Path):
    module = _load_validator_module()
    expected_path = _write_plist(tmp_path / "LumenAppStore.entitlements", {"com.apple.developer.healthkit": True})

    failures = module.validate_signed_entitlements(
        {
            "com.apple.developer.healthkit": True,
            "com.apple.developer.kernel.increased-debugging-memory-limit": True,
        },
        expected_path,
        "SignedLumen.app",
    )

    assert any("must not be present" in failure for failure in failures)


def test_signed_entitlement_validation_allows_codesign_resolved_build_settings(tmp_path: Path):
    module = _load_validator_module()
    expected_path = _write_plist(
        tmp_path / "LumenAppStore.entitlements",
        {"keychain-access-groups": ["$(AppIdentifierPrefix)com.microsoft.adalcache"]},
    )

    failures = module.validate_signed_entitlements(
        {"keychain-access-groups": ["ABCDE12345.com.microsoft.adalcache"]},
        expected_path,
        "SignedLumen.app",
    )

    assert failures == []


@pytest.mark.parametrize(
    "authority,get_task_allow,provisioned_devices",
    [
        ("Apple Development: Test (TEAM123456)", True, True),
        ("Apple Distribution: Test (TEAM123456)", False, False),
    ],
)
def test_archive_stage_accepts_real_apple_development_or_distribution_signing(
    authority: str,
    get_task_allow: bool,
    provisioned_devices: bool,
):
    module = _load_validator_module()
    evidence = _signing_evidence(
        module,
        authority=authority,
        get_task_allow=get_task_allow,
        provisioned_devices=provisioned_devices,
    )

    failures = module.validate_signing_evidence(
        evidence,
        "archive",
        "SignedLumen.app",
    )

    assert failures == []


def test_app_store_stage_accepts_store_distribution_signing():
    module = _load_validator_module()
    evidence = _signing_evidence(
        module,
        authority="Apple Distribution: Test (TEAM123456)",
        get_task_allow=False,
        beta_reports_active=True,
        provisioned_devices=False,
    )

    failures = module.validate_signing_evidence(
        evidence,
        "app-store",
        "SignedLumen.app",
    )

    assert failures == []


def test_app_store_stage_rejects_expired_profile():
    module = _load_validator_module()
    evidence = _signing_evidence(
        module,
        authority="Apple Distribution: Test (TEAM123456)",
        get_task_allow=False,
        beta_reports_active=True,
        provisioned_devices=False,
    )
    evidence.profile["ExpirationDate"] = datetime(2001, 1, 1)

    failures = module.validate_signing_evidence(
        evidence,
        "app-store",
        "SignedLumen.app",
    )

    assert any("embedded profile expired" in failure for failure in failures)


def test_app_store_stage_rejects_missing_profile_expiration_date():
    module = _load_validator_module()
    evidence = _signing_evidence(
        module,
        authority="Apple Distribution: Test (TEAM123456)",
        get_task_allow=False,
        beta_reports_active=True,
        provisioned_devices=False,
    )
    evidence.profile.pop("ExpirationDate")

    failures = module.validate_signing_evidence(
        evidence,
        "app-store",
        "SignedLumen.app",
    )

    assert any("no valid ExpirationDate" in failure for failure in failures)


def test_app_store_stage_rejects_missing_profile_developer_certificates():
    module = _load_validator_module()
    evidence = _signing_evidence(
        module,
        authority="Apple Distribution: Test (TEAM123456)",
        get_task_allow=False,
        beta_reports_active=True,
        provisioned_devices=False,
    )
    evidence.profile.pop("DeveloperCertificates")

    failures = module.validate_signing_evidence(
        evidence,
        "app-store",
        "SignedLumen.app",
    )

    assert any("no DeveloperCertificates" in failure for failure in failures)


def test_app_store_stage_rejects_leaf_certificate_not_in_profile():
    module = _load_validator_module()
    evidence = _signing_evidence(
        module,
        authority="Apple Distribution: Test (TEAM123456)",
        get_task_allow=False,
        beta_reports_active=True,
        provisioned_devices=False,
    )
    evidence.profile["DeveloperCertificates"] = [b"different-certificate"]

    failures = module.validate_signing_evidence(
        evidence,
        "app-store",
        "SignedLumen.app",
    )

    assert any(
        "leaf signing certificate is not authorized" in failure
        for failure in failures
    )


@pytest.mark.parametrize(
    "evidence_kwargs,expected_message",
    [
        (
            {
                "authority": "Apple Development: Test (TEAM123456)",
                "get_task_allow": False,
                "beta_reports_active": True,
                "provisioned_devices": False,
            },
            "must use an Apple Distribution signature",
        ),
        (
            {
                "authority": "Apple Distribution: Test (TEAM123456)",
                "get_task_allow": True,
                "beta_reports_active": True,
                "provisioned_devices": False,
            },
            "requires get-task-allow=false",
        ),
        (
            {
                "authority": "Apple Distribution: Test (TEAM123456)",
                "get_task_allow": False,
                "beta_reports_active": None,
                "provisioned_devices": False,
            },
            "requires beta-reports-active=true",
        ),
        (
            {
                "authority": "Apple Distribution: Test (TEAM123456)",
                "get_task_allow": False,
                "beta_reports_active": True,
                "provisioned_devices": True,
            },
            "must not contain ProvisionedDevices",
        ),
    ],
)
def test_app_store_stage_rejects_non_store_signing_semantics(
    evidence_kwargs: dict,
    expected_message: str,
):
    module = _load_validator_module()
    evidence = _signing_evidence(module, **evidence_kwargs)

    failures = module.validate_signing_evidence(
        evidence,
        "app-store",
        "SignedLumen.app",
    )

    assert any(expected_message in failure for failure in failures)


def test_signing_evidence_rejects_ad_hoc_signature():
    module = _load_validator_module()
    evidence = _signing_evidence(module, authority="", signature_marker="adhoc")

    failures = module.validate_signing_evidence(evidence, "archive", "SignedLumen.app")

    assert any("ad-hoc code signatures" in failure for failure in failures)
    assert any("signature authority must be Apple Development or Apple Distribution" in failure for failure in failures)


def test_signed_entitlement_reader_rejects_unsigned_app(tmp_path: Path, monkeypatch):
    module = _load_validator_module()
    app_path = tmp_path / "Unsigned.app"
    app_path.mkdir()
    monkeypatch.setattr(module.sys, "platform", "darwin")
    monkeypatch.setattr(module.shutil, "which", lambda command: f"/usr/bin/{command}")
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args=args[0],
            returncode=1,
            stdout="",
            stderr="code object is not signed at all",
        ),
    )

    with pytest.raises(SystemExit, match="codesign verification failed"):
        module.read_signed_entitlements(app_path)


def test_leaf_signing_certificate_reader_extracts_codesign_leaf(
    tmp_path: Path,
    monkeypatch,
):
    module = _load_validator_module()
    app_path = tmp_path / "Signed.app"
    app_path.mkdir()
    monkeypatch.setattr(module.sys, "platform", "darwin")
    monkeypatch.setattr(module.shutil, "which", lambda command: f"/usr/bin/{command}")

    def fake_run(arguments, **_kwargs):
        extract_argument = next(
            argument
            for argument in arguments
            if argument.startswith("--extract-certificates=")
        )
        certificate_prefix = Path(extract_argument.split("=", 1)[1])
        Path(f"{certificate_prefix}0").write_bytes(b"leaf-certificate-der")
        return subprocess.CompletedProcess(arguments, 0, "", "")

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    assert module.read_leaf_signing_certificate(app_path) == b"leaf-certificate-der"


def test_archive_stage_rejects_profile_team_bundle_and_capability_drift():
    module = _load_validator_module()
    evidence = _signing_evidence(module)
    evidence.profile["TeamIdentifier"] = ["OTHERTEAM1"]
    evidence.profile["Entitlements"]["application-identifier"] = (
        "OTHERTEAM1.com.example.other"
    )
    evidence.profile["Entitlements"]["com.apple.developer.healthkit"] = False

    failures = module.validate_signing_evidence(evidence, "archive", "SignedLumen.app")

    assert any("is not authorized by embedded profile" in failure for failure in failures)
    assert any("application-identifier does not match embedded profile" in failure for failure in failures)
    assert any("does not authorize signed value" in failure for failure in failures)


def test_cli_requires_signing_stage_with_signed_app_path(tmp_path: Path):
    development_path = _write_plist(tmp_path / "Lumen.entitlements", {})
    app_store_path = _write_plist(tmp_path / "LumenAppStore.entitlements", {})
    signed_app_path = tmp_path / "Signed.app"
    signed_app_path.mkdir()

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--project-file",
            str(_write_project(tmp_path)),
            "--entitlements",
            str(development_path),
            "--app-store-entitlements",
            str(app_store_path),
            "--signed-app-path",
            str(signed_app_path),
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert "--signing-stage archive|app-store is required" in result.stderr


def test_release_callers_select_explicit_signing_stages():
    archive_script = ARCHIVE_SCRIPT.read_text(encoding="utf-8")
    submit_script = SUBMIT_SCRIPT.read_text(encoding="utf-8")
    readiness_script = READINESS_SCRIPT.read_text(encoding="utf-8")

    archive_stage_call = '--signing-stage archive \\\n  --signed-app-path "$ARCHIVE_PATH"'
    app_store_stage_call = '--signing-stage app-store \\\n  --signed-app-path "$IPA_PATH"'
    assert archive_stage_call in archive_script
    assert archive_stage_call in submit_script
    assert app_store_stage_call in submit_script
    assert '*.xcarchive) signing_stage="archive"' in readiness_script
    assert '*.ipa) signing_stage="app-store"' in readiness_script


def test_archive_command_rejects_literal_empty_identity_override():
    result = subprocess.run(
        ["bash", str(ARCHIVE_SCRIPT), "--self-check-signing-arguments"],
        text=True,
        capture_output=True,
        check=False,
    )
    archive_script = ARCHIVE_SCRIPT.read_text(encoding="utf-8")

    assert result.returncode == 0, result.stderr
    assert "Archive signing-argument self-check passed." in result.stdout
    assert 'validate_archive_signing_arguments "${ARCHIVE_COMMAND[@]}"' in archive_script
