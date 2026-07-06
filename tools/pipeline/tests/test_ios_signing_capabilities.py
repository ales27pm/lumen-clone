import importlib.util
import plistlib
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "validate_ios_signing_capabilities.py"


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
    spec.loader.exec_module(module)
    return module


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
