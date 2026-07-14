from __future__ import annotations

import json
from typing import Any

import pytest

from tools.fine_tuning.unsloth import train_sft, training_lineage


def _environment() -> dict[str, Any]:
    distribution_payload = {
        "name": "synthetic-runtime",
        "version": "1.0.0",
        "directURL": None,
        "installer": "test",
        "recordSHA256": "1" * 64,
        "installedFileCount": 1,
        "installedContentSHA256": "2" * 64,
    }
    distribution = {
        **distribution_payload,
        "distributionSHA256": training_lineage.canonical_sha256(
            distribution_payload
        ),
    }
    payload = {
        "schemaVersion": "lumen.resolved-training-environment/1.0.0",
        "recordPolicy": training_lineage.RESOLVED_TRAINING_ENVIRONMENT_RECORD_POLICY,
        "distributions": [distribution],
    }
    return {
        **payload,
        "resolvedTrainingEnvironmentSHA256": training_lineage.canonical_sha256(
            payload
        ),
    }


def _scan(environment: dict[str, Any]) -> dict[str, Any]:
    return {
        "schemaVersion": "lumen.resolved-training-environment-cache/1.0.0",
        "resolvedTrainingEnvironmentSHA256": environment[
            "resolvedTrainingEnvironmentSHA256"
        ],
        "durationMilliseconds": 9,
        "distributionCount": 1,
        "installedFileCount": 1,
        "totalHashedBytes": 128,
    }


def test_space_trainer_uses_authenticated_startup_environment_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = _environment()
    scan = _scan(environment)
    key = b"k" * 32
    attestation = training_lineage.sign_resolved_training_environment_cache(
        environment,
        scan,
        key=key,
        startup_id="a" * 32,
    )
    cfg = {
        "resolvedTrainingEnvironment": environment,
        "resolvedTrainingEnvironmentSHA256": environment[
            "resolvedTrainingEnvironmentSHA256"
        ],
        "resolvedTrainingEnvironmentScanAudit": scan,
        "resolvedTrainingEnvironmentCacheAttestation": (
            training_lineage.sign_resolved_training_environment_cache(
                environment,
                {**scan, "durationMilliseconds": 3},
                key=b"o" * 32,
                startup_id="b" * 32,
            )
        ),
    }
    monkeypatch.setenv(
        "LUMEN_ZERO_GPU_RESOLVED_ENVIRONMENT_CACHE_HMAC_KEY",
        key.hex(),
    )
    monkeypatch.setenv(
        "LUMEN_ZERO_GPU_RESOLVED_ENVIRONMENT_CACHE_ATTESTATION",
        json.dumps(attestation),
    )
    monkeypatch.setattr(
        train_sft,
        "build_resolved_training_environment_snapshot",
        lambda: pytest.fail("authenticated Space trainers must not rescan packages"),
    )

    resolved, digest, observed_scan = train_sft._resolved_environment_runtime_lineage(
        cfg,
        deployed_space=True,
    )

    assert resolved == environment
    assert digest == environment["resolvedTrainingEnvironmentSHA256"]
    assert observed_scan == scan


def test_direct_trainer_rescans_and_rejects_invalid_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = _environment()
    scan = _scan(environment)
    calls = 0

    def rebuild() -> tuple[dict[str, Any], dict[str, Any]]:
        nonlocal calls
        calls += 1
        return environment, scan

    monkeypatch.delenv(
        "LUMEN_ZERO_GPU_RESOLVED_ENVIRONMENT_CACHE_HMAC_KEY",
        raising=False,
    )
    monkeypatch.setattr(
        train_sft,
        "build_resolved_training_environment_snapshot",
        rebuild,
    )
    resolved, _, _ = train_sft._resolved_environment_runtime_lineage(
        {"resolvedTrainingEnvironment": environment},
        deployed_space=True,
    )
    assert resolved == environment
    assert calls == 1

    monkeypatch.setenv(
        "LUMEN_ZERO_GPU_RESOLVED_ENVIRONMENT_CACHE_HMAC_KEY",
        (b"x" * 32).hex(),
    )
    with pytest.raises(RuntimeError, match="cache verification failed"):
        train_sft._resolved_environment_runtime_lineage(
            {
                "resolvedTrainingEnvironment": environment,
                "resolvedTrainingEnvironmentCacheAttestation": (
                    training_lineage.sign_resolved_training_environment_cache(
                        environment,
                        scan,
                        key=b"k" * 32,
                        startup_id="b" * 32,
                    )
                ),
            },
            deployed_space=True,
        )
