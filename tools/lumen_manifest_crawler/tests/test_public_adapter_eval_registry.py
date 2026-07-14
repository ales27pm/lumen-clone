from __future__ import annotations

import copy
import hashlib
import json

import pytest

from lumen_manifest_crawler.dataset.public_adapter_eval_registry import (
    build_public_adapter_eval_fingerprint_bundle,
    canonical_sha256,
    load_public_adapter_eval_fingerprint_bundle,
    load_public_adapter_eval_registry,
    public_adapter_eval_source_descriptors,
    validate_public_adapter_eval_fingerprint_bundle,
    validate_public_adapter_eval_registry,
)


EXPECTED_REVISION = "61fc0608cfd831fcfbbaa676ebdfef0ed963eeda"


def test_bfcl_registry_is_pinned_eval_only_and_covers_selected_categories() -> None:
    registry = load_public_adapter_eval_registry()

    assert registry["revision"] == EXPECTED_REVISION
    assert registry["license"] == "Apache-2.0"
    assert registry["purpose"] == "evaluation_only"
    assert registry["trainingEligible"] is False
    assert registry["trainingTargets"] == []
    assert {artifact["category"] for artifact in registry["artifacts"]} == {
        "simple",
        "multiple",
        "parallel",
        "irrelevance",
    }
    assert all(artifact["trainingEligible"] is False for artifact in registry["artifacts"])
    assert all(artifact["trainingTargets"] == [] for artifact in registry["artifacts"])


def test_registry_schema_accepts_forward_compatible_same_major_versions() -> None:
    registry = load_public_adapter_eval_registry()
    future = copy.deepcopy(registry)
    future["schema"] = "lumen.public-adapter-evaluation-sources/1.27.4"
    future["futureOptionalField"] = {"recognizedByFutureConsumers": True}

    validate_public_adapter_eval_registry(future)

    incompatible = copy.deepcopy(registry)
    incompatible["schema"] = "lumen.public-adapter-evaluation-sources/2.0.0"
    with pytest.raises(ValueError, match="unsupported"):
        validate_public_adapter_eval_registry(incompatible)


def test_registry_rejects_any_training_target() -> None:
    registry = load_public_adapter_eval_registry()
    registry["artifacts"][0]["trainingEligible"] = True
    registry["artifacts"][0]["trainingTargets"] = ["executor"]

    with pytest.raises(ValueError, match="cannot declare training targets"):
        validate_public_adapter_eval_registry(registry)


def test_fingerprint_bundle_is_deterministic_hash_only_and_has_no_training_target() -> None:
    first = build_public_adapter_eval_fingerprint_bundle()
    second = build_public_adapter_eval_fingerprint_bundle()

    assert first == second
    assert first["hashOnly"] is True
    assert first["rawEvaluationTextIncluded"] is False
    assert first["trainingEligible"] is False
    assert first["trainingTargets"] == []
    assert first["rowCount"] == 1_040
    assert len(first["bundleSHA256"]) == 64
    assert all(artifact["trainingTargets"] == [] for artifact in first["artifacts"])
    assert all("question" not in artifact and "answer" not in artifact for artifact in first["artifacts"])
    assert {artifact["id"]: artifact["rowCount"] for artifact in first["artifacts"]} == {
        "bfcl-v3-irrelevance": 240,
        "bfcl-v3-multiple": 200,
        "bfcl-v3-parallel": 200,
        "bfcl-v3-simple": 400,
    }
    assert all(
        0 < len(row["tokenShingleSketch"]) <= 64
        for artifact in first["artifacts"]
        for row in artifact["rows"]
    )


def test_default_fingerprint_bundle_callers_cannot_mutate_cached_evidence() -> None:
    first = build_public_adapter_eval_fingerprint_bundle()
    original_digest = first["artifacts"][0]["rows"][0]["normalizedRowSHA256"]
    first["artifacts"][0]["rows"][0]["normalizedRowSHA256"] = "0" * 64

    second = build_public_adapter_eval_fingerprint_bundle()
    assert second["artifacts"][0]["rows"][0]["normalizedRowSHA256"] == original_digest


def test_committed_bundle_contains_only_hash_contract_fields() -> None:
    bundle = load_public_adapter_eval_fingerprint_bundle()
    encoded = json.dumps(bundle, sort_keys=True)

    assert '"question"' not in encoded
    assert '"function"' not in encoded
    assert '"prompt"' not in encoded
    assert '"answer"' not in encoded
    assert '"text"' not in encoded
    for artifact in bundle["artifacts"]:
        for row in artifact["rows"]:
            assert set(row) == {
                "rowOrdinal",
                "normalizedRowSHA256",
                "tokenCount",
                "tokenShingleCount",
                "tokenShingleSketch",
            }


def test_bundle_validator_rejects_raw_row_fields_even_when_hashes_are_recomputed() -> None:
    registry = load_public_adapter_eval_registry()
    bundle = load_public_adapter_eval_fingerprint_bundle(registry=registry)
    contaminated = copy.deepcopy(bundle)
    artifact = contaminated["artifacts"][0]
    artifact["rows"][0]["question"] = "raw benchmark text"
    artifact["rowFingerprintAggregateSHA256"] = canonical_sha256(artifact["rows"])
    artifact_without_hash = dict(artifact)
    artifact_without_hash.pop("declarationSHA256")
    artifact["declarationSHA256"] = canonical_sha256(artifact_without_hash)
    bundle_without_hash = dict(contaminated)
    bundle_without_hash.pop("bundleSHA256")
    contaminated["bundleSHA256"] = canonical_sha256(bundle_without_hash)

    with pytest.raises(ValueError, match="non-hash fields"):
        validate_public_adapter_eval_fingerprint_bundle(contaminated, registry=registry)


def test_local_artifact_verification_binds_bundle_to_declared_bytes(tmp_path) -> None:
    payload = b'{"id":"simple_0"}\n'
    registry = load_public_adapter_eval_registry()
    registry["artifacts"] = [copy.deepcopy(registry["artifacts"][0])]
    registry["artifacts"][0]["artifactBytes"] = len(payload)
    registry["artifacts"][0]["artifactSHA256"] = hashlib.sha256(payload).hexdigest()
    artifact_path = tmp_path / registry["artifacts"][0]["path"]
    artifact_path.write_bytes(payload)

    bundle = build_public_adapter_eval_fingerprint_bundle(registry, artifact_root=tmp_path)
    assert bundle["artifacts"][0]["artifactSHA256"] == hashlib.sha256(payload).hexdigest()
    assert bundle["rowCount"] == 1
    assert len(bundle["artifacts"][0]["rows"][0]["normalizedRowSHA256"]) == 64
    assert bundle["artifacts"][0]["rows"][0]["tokenShingleSketch"]

    artifact_path.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="byte count mismatch|hash mismatch"):
        build_public_adapter_eval_fingerprint_bundle(registry, artifact_root=tmp_path)


def test_normalized_row_hash_is_stable_across_case_and_whitespace(tmp_path) -> None:
    registry = load_public_adapter_eval_registry()
    registry["artifacts"] = [copy.deepcopy(registry["artifacts"][0])]
    artifact_path = tmp_path / registry["artifacts"][0]["path"]

    first_payload = b'{"question":[{"role":"USER","content":"Call   Weather"}],"id":"X"}\n'
    artifact_path.write_bytes(first_payload)
    registry["artifacts"][0]["artifactBytes"] = len(first_payload)
    registry["artifacts"][0]["artifactSHA256"] = hashlib.sha256(first_payload).hexdigest()
    first = build_public_adapter_eval_fingerprint_bundle(registry, artifact_root=tmp_path)

    second_payload = b'{"id":"x","question":[{"content":"call weather","role":"user"}]}\n'
    artifact_path.write_bytes(second_payload)
    registry["artifacts"][0]["artifactBytes"] = len(second_payload)
    registry["artifacts"][0]["artifactSHA256"] = hashlib.sha256(second_payload).hexdigest()
    second = build_public_adapter_eval_fingerprint_bundle(registry, artifact_root=tmp_path)

    first_row = first["artifacts"][0]["rows"][0]
    second_row = second["artifacts"][0]["rows"][0]
    assert first_row["normalizedRowSHA256"] == second_row["normalizedRowSHA256"]
    assert first_row["tokenShingleSketch"] == second_row["tokenShingleSketch"]


def test_source_descriptors_are_eval_only_and_json_serializable() -> None:
    descriptors = public_adapter_eval_source_descriptors()

    assert [item["id"] for item in descriptors] == sorted(item["id"] for item in descriptors)
    assert all(item["trainingEligible"] is False for item in descriptors)
    assert all(item["trainingTargets"] == [] for item in descriptors)
    assert json.loads(json.dumps(descriptors)) == descriptors
