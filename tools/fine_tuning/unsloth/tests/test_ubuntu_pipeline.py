from __future__ import annotations

import ast
import hashlib
import io
import json
import os
import shutil
import stat
import subprocess
import sys
import tarfile
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, Mapping

import pytest

from lumen_manifest_crawler.dataset import adapter_evaluation
from lumen_manifest_crawler.dataset.chat_template_contract import (
    chat_template_contract,
)
from lumen_manifest_crawler.dataset.optimization_policy import (
    expected_optimization_step_policy,
)
from tools.fine_tuning.unsloth import (
    evaluate_adapter,
    runtime_binding_smoke_gate,
    train_dpo,
    ubuntu_pipeline,
    ubuntu_source_integrity,
)


REPO_ROOT = Path(__file__).resolve().parents[4]
FAKE_IMAGE_DIGEST = "sha256:" + ("a" * 64)
OPTIMIZED_VARIANT = "internal_plus_public_optimized"
VALID_GGUF_TEST_PAYLOAD = (b"\x00" * 64) + b"LUMEN_VALID_GGUF_TEST"


def _test_execution_plan(
    *,
    evaluation_scope: str = "full",
    evaluation_max_examples: int | None = None,
    gguf_requested: bool = True,
) -> dict:
    return ubuntu_pipeline.execution_plan(
        evaluation_scope=evaluation_scope,
        evaluation_max_examples=evaluation_max_examples,
        gguf_requested=gguf_requested,
    )


def _source_integrity_fixture() -> dict:
    orchestration = {
        "schemaVersion": "lumen.ubuntu-orchestration-code/1.0.0",
        "files": [
            {"path": path, "size": 1, "sha256": "4" * 64}
            for path in sorted(
                ubuntu_source_integrity.REQUIRED_ORCHESTRATION_PATHS
            )
        ],
    }
    record = {
        "schema": "lumen.ubuntu-source-integrity/1.0.0",
        "baseCommit": "5" * 40,
        "workingTreeDigest": "1" * 64,
        "dirtyState": False,
        "ubuntuOrchestrationCodeSHA256": ubuntu_pipeline.canonical_sha256(
            orchestration
        ),
        "orchestrationManifest": orchestration,
    }
    record["sourceIntegritySHA256"] = ubuntu_pipeline.canonical_sha256(record)
    return ubuntu_pipeline.source_integrity_fields(record)


def _base_model_tokenizer_lineage_fixture() -> dict[str, Any]:
    return {
        "baseModelID": adapter_evaluation.DEFAULT_BASE_MODEL_ID,
        "baseModelRevision": adapter_evaluation.DEFAULT_BASE_MODEL_REVISION,
        "baseModelTokenizerFiles": [
            dict(item)
            for item in adapter_evaluation.DEFAULT_BASE_MODEL_TOKENIZER_FILES
        ],
        "baseModelTokenizerDigest": (
            adapter_evaluation.DEFAULT_BASE_MODEL_TOKENIZER_DIGEST
        ),
        "baseModelTokenizerClosureSHA256": (
            adapter_evaluation.DEFAULT_BASE_MODEL_TOKENIZER_CLOSURE_SHA256
        ),
    }


def _runtime_snapshot_lineage_fixture() -> dict[str, Any]:
    tokenizer_snapshot_path = "/tmp/lumen-test-tokenizer-snapshot"
    runtime_snapshot_path = "/tmp/lumen-test-runtime-snapshot"
    return {
        "baseModelTokenizerSnapshotPath": tokenizer_snapshot_path,
        "baseModelTokenizerSnapshotVerification": {
            "schemaVersion": "test.private-tokenizer-snapshot/1.0.0",
            "snapshotPath": tokenizer_snapshot_path,
            "snapshotVerificationSHA256": "a" * 64,
        },
        "baseModelGenerationConfigFile": {
            "path": "generation_config.json",
            "sizeBytes": 1,
            "sha256": "b" * 64,
            "huggingFaceBlobID": "c" * 40,
        },
        "baseModelRuntimeSnapshotPath": runtime_snapshot_path,
        "baseModelRuntimeSnapshotVerification": {
            "schemaVersion": "test.private-runtime-snapshot/1.0.0",
            "snapshotPath": runtime_snapshot_path,
            "snapshotVerificationSHA256": "d" * 64,
        },
    }


def _summary_base_model_lineage_fixture() -> dict[str, Any]:
    return {
        **_base_model_tokenizer_lineage_fixture(),
        **_runtime_snapshot_lineage_fixture(),
        "runManifestSHA256": "e" * 64,
    }


def _publication_base_model_lineage_fixture() -> dict[str, Any]:
    lineage = _summary_base_model_lineage_fixture()
    lineage.pop("baseModelTokenizerFiles")
    return lineage


def _phase_runtime_evidence_fixture(
    digest_character: str = "a",
) -> dict[str, str]:
    assert digest_character in "0123456789abcdef"
    evidence = {
        field: digest_character * 64
        for field in ubuntu_pipeline.PHASE_RUNTIME_EVIDENCE_FIELDS
    }
    evidence.update(
        {
            "runtimeModelBindingSHA256": "a" * 64,
            "runtimeTokenizerBindingSHA256": "a" * 64,
            "baseModelTokenizerSnapshotVerificationSHA256": "a" * 64,
            "baseModelRuntimeSnapshotVerificationSHA256": "a" * 64,
        }
    )
    return evidence


def _write_phase_report_fixture(
    run_root: Path,
    agent: str,
    *,
    preference: bool,
    digest_character: str,
) -> dict[str, str]:
    path = (
        run_root / "training" / agent / "dpo" / "dpo_report.json"
        if preference
        else run_root / "training" / agent / "training_report.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    ubuntu_pipeline.write_object(
        path,
        {"fixture": "preference" if preference else "sft"},
    )
    return {
        **_phase_runtime_evidence_fixture(digest_character),
        "report": str(path),
        "trainingReportFileSHA256": ubuntu_pipeline.file_sha256(path),
    }


def _write_runtime_binding_smoke_summary_fixture(
    run_root: Path,
    agents: tuple[str, ...] = ("cortex",),
) -> dict[str, Any]:
    report_path = run_root / "training" / "runtime_binding_smoke.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    gate_digest = "c" * 64
    ubuntu_pipeline.write_object(
        report_path,
        {
            "schemaVersion": "lumen.runtime-binding-smoke-gate/1.0.0",
            "status": "passed",
            "runtimeBindingSmokeGateSHA256": gate_digest,
        },
    )
    return {
        "runtimeBindingSmokeReport": str(report_path),
        "runtimeBindingSmokeReportFileSHA256": ubuntu_pipeline.file_sha256(
            report_path
        ),
        "runtimeBindingSmokeGateSHA256": gate_digest,
        "runtimeBindingSmokeContractEvidence": [
            {
                "runtimeLoadContractSHA256": "d" * 64,
                "agents": list(agents),
                "representativeAgent": agents[0],
                "runtimeBindingSmokeSHA256": "e" * 64,
                "runtimeModelBindingSHA256": "a" * 64,
                "runtimeTokenizerBindingSHA256": "a" * 64,
            }
        ],
        "runtimeBindingSmokeBindingsByAgent": {
            agent: {
                "runtimeModelBindingSHA256": "a" * 64,
                "runtimeTokenizerBindingSHA256": "a" * 64,
            }
            for agent in agents
        },
    }


def _mock_private_base_model_snapshot_preparation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep prepare-run tests tiny while preserving exact path propagation."""

    from tools.fine_tuning.unsloth import training_lineage

    original_validate_variant = ubuntu_pipeline.validate_variant
    original_verify_manifest = ubuntu_pipeline._verify_manifest_integrity
    cache_snapshot = tmp_path / "tiny-hf-cache-snapshot"
    cache_snapshot.mkdir()

    def validate_variant_with_current_tokenizer_contract(
        *args: Any,
        **kwargs: Any,
    ) -> tuple[dict[str, Any], dict[str, Any], Path]:
        config, manifest, variant_root = original_validate_variant(*args, **kwargs)
        current = _base_model_tokenizer_lineage_fixture()
        return (
            {**config, **current},
            {
                **manifest,
                "baseModelTokenizerFiles": current["baseModelTokenizerFiles"],
                "baseModelTokenizerClosureSHA256": current[
                    "baseModelTokenizerClosureSHA256"
                ],
            },
            variant_root,
        )

    def verify_manifest_with_current_tokenizer_contract(path: Path) -> dict[str, Any]:
        manifest = original_verify_manifest(path)
        if (
            path.name == "variant_manifest.json"
            and "generated/fine_tuning" in path.as_posix()
        ):
            current = _base_model_tokenizer_lineage_fixture()
            return {
                **manifest,
                "baseModelTokenizerFiles": current["baseModelTokenizerFiles"],
                "baseModelTokenizerClosureSHA256": current[
                    "baseModelTokenizerClosureSHA256"
                ],
            }
        return manifest

    def create_tokenizer_snapshot(*, snapshot_dir: Path, config: Mapping[str, Any]) -> None:
        del config
        snapshot_dir.mkdir(mode=0o700)

    def tokenizer_verification(
        snapshot_dir: Path,
        **kwargs: Any,
    ) -> dict[str, Any]:
        payload = {
            "schemaVersion": "test.private-tokenizer-snapshot/1.0.0",
            "baseModelID": kwargs["base_model_id"],
            "baseModelRevision": kwargs["base_model_revision"],
            "baseModelTokenizerDigest": kwargs["tokenizer_digest"],
            "baseModelTokenizerFiles": kwargs["tokenizer_files"],
            "baseModelTokenizerClosureSHA256": kwargs[
                "tokenizer_closure_sha256"
            ],
            "snapshotPath": str(snapshot_dir.resolve()),
        }
        return {
            **payload,
            "snapshotVerificationSHA256": ubuntu_pipeline.canonical_sha256(
                payload
            ),
        }

    def runtime_verification(
        snapshot_dir: Path,
        **kwargs: Any,
    ) -> dict[str, Any]:
        payload = {
            "schemaVersion": "test.private-runtime-snapshot/1.0.0",
            "baseModelID": kwargs["base_model_id"],
            "baseModelRevision": kwargs["base_model_revision"],
            "baseModelIndexDigest": kwargs["model_index_digest"],
            "baseModelIndexReferencedShardNames": kwargs[
                "index_referenced_shard_names"
            ],
            "baseModelIndexShardBindingSHA256": kwargs[
                "index_shard_binding_sha256"
            ],
            "baseModelArtifactDigest": kwargs["model_artifact_digest"],
            "baseModelWeightShards": kwargs["weight_shards"],
            "baseModelGenerationConfigFile": kwargs[
                "generation_config_file"
            ],
            "baseModelTokenizerDigest": kwargs["tokenizer_digest"],
            "baseModelTokenizerFiles": kwargs["tokenizer_files"],
            "baseModelTokenizerClosureSHA256": kwargs[
                "tokenizer_closure_sha256"
            ],
            "snapshotPath": str(snapshot_dir.resolve()),
        }
        return {
            **payload,
            "snapshotVerificationSHA256": ubuntu_pipeline.canonical_sha256(
                payload
            ),
        }

    def create_runtime_snapshot(
        *,
        destination: Path,
        **kwargs: Any,
    ) -> dict[str, Any]:
        destination.mkdir(mode=0o700)
        return runtime_verification(destination, **kwargs)

    monkeypatch.setattr(
        ubuntu_pipeline,
        "_create_global_tokenizer_snapshot",
        create_tokenizer_snapshot,
    )
    monkeypatch.setattr(
        ubuntu_pipeline,
        "validate_variant",
        validate_variant_with_current_tokenizer_contract,
    )
    monkeypatch.setattr(
        ubuntu_pipeline,
        "_verify_manifest_integrity",
        verify_manifest_with_current_tokenizer_contract,
    )
    monkeypatch.setattr(
        training_lineage,
        "verify_private_base_model_tokenizer_snapshot",
        tokenizer_verification,
    )
    monkeypatch.setattr(
        training_lineage,
        "verify_private_base_model_conversion_snapshot",
        runtime_verification,
    )
    monkeypatch.setattr(
        training_lineage,
        "create_private_base_model_runtime_snapshot",
        create_runtime_snapshot,
    )
    monkeypatch.setattr(
        training_lineage,
        "private_base_model_runtime_snapshot_required_bytes",
        lambda **_kwargs: 1,
    )
    huggingface_hub = ModuleType("huggingface_hub")
    huggingface_hub.snapshot_download = lambda **_kwargs: str(cache_snapshot)
    monkeypatch.setitem(sys.modules, "huggingface_hub", huggingface_hub)


def _gguf_bytes(
    *,
    version: int = 3,
    tensor_count: int = 1,
    metadata_kv_count: int = 5,
    payload: bytes = VALID_GGUF_TEST_PAYLOAD,
) -> bytes:
    return b"".join(
        (
            b"GGUF",
            version.to_bytes(4, byteorder="little", signed=False),
            tensor_count.to_bytes(8, byteorder="little", signed=False),
            metadata_kv_count.to_bytes(8, byteorder="little", signed=False),
            payload,
        )
    )


def _valid_gguf_semantic_metadata(
    *,
    chat_template: str | None = None,
) -> dict[str, dict[str, object]]:
    values: list[tuple[str, str, object]] = [
        ("general.architecture", "STRING", "qwen3"),
        ("general.type", "STRING", "adapter"),
        ("adapter.type", "STRING", "lora"),
        ("general.base_model.count", "UINT32", 1),
        (
            "general.base_model.0.repo_url",
            "STRING",
            "https://huggingface.co/Qwen/Qwen3-1.7B",
        ),
    ]
    if chat_template is not None:
        values.append(("tokenizer.chat_template", "STRING", chat_template))
    return {
        key: {
            "index": index,
            "type": value_type,
            "offset": index * 16,
            "value": value,
        }
        for index, (key, value_type, value) in enumerate(values)
    }


def _write_fake_gguf_reader(
    root: Path,
    *,
    semantic_metadata: dict[str, dict[str, object]] | None = None,
) -> Path:
    configured_metadata = (
        _valid_gguf_semantic_metadata()
        if semantic_metadata is None
        else semantic_metadata
    )
    reader = root / "fake_gguf_dump.py"
    reader.write_text(
        f"""from __future__ import annotations

import json
import sys
from pathlib import Path

configured_metadata = json.loads({json.dumps(configured_metadata)!r})
model = Path(sys.argv[1]).resolve()
data = model.read_bytes()
if not data.endswith(b"LUMEN_VALID_GGUF_TEST"):
    print("structural GGUF parse failed", file=sys.stderr)
    raise SystemExit(7)
tensor_count = int.from_bytes(data[8:16], "little")
metadata_count = int.from_bytes(data[16:24], "little")
metadata = {{
    "GGUF.version": {{}},
    "GGUF.tensor_count": {{}},
    "GGUF.kv_count": {{}},
}}
if len(configured_metadata) > metadata_count:
    print("configured metadata exceeds fixed header", file=sys.stderr)
    raise SystemExit(8)
metadata.update(configured_metadata)
metadata.update({{
    f"metadata.fixture.{{index}}": {{}}
    for index in range(metadata_count - len(configured_metadata))
}})
tensors = {{f"tensor.{{index}}": {{}} for index in range(tensor_count)}}
print(json.dumps({{
    "filename": str(model),
    "endian": "LITTLE",
    "metadata": metadata,
    "tensors": tensors,
}}))
""",
        encoding="utf-8",
    )
    return reader


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity", "1e400"])
def test_pipeline_json_readers_reject_nonfinite_numbers(
    tmp_path: Path,
    constant: str,
) -> None:
    object_path = tmp_path / "object.json"
    object_path.write_text(f'{{"value":{constant}}}', encoding="utf-8")
    with pytest.raises(RuntimeError, match="Unable to read JSON object"):
        ubuntu_pipeline.read_object(object_path)

    jsonl_path = tmp_path / "records.jsonl"
    jsonl_path.write_text(f'{{"value":{constant}}}\n', encoding="utf-8")
    with pytest.raises(RuntimeError, match="Invalid JSON"):
        ubuntu_pipeline.read_jsonl(jsonl_path)


def test_pipeline_json_readers_reject_duplicate_object_keys(
    tmp_path: Path,
) -> None:
    duplicate = '{"value":1,"value":2}'
    object_path = tmp_path / "object.json"
    object_path.write_text(duplicate, encoding="utf-8")
    with pytest.raises(RuntimeError, match="Unable to read JSON object"):
        ubuntu_pipeline.read_object(object_path)

    jsonl_path = tmp_path / "records.jsonl"
    jsonl_path.write_text(duplicate + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Invalid JSON"):
        ubuntu_pipeline.read_jsonl(jsonl_path)


def test_evaluation_verifier_binds_fleet_neutral_repetition_penalty(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_path = _write_evaluation_evidence(
        tmp_path,
        monkeypatch=monkeypatch,
        agent="fleet",
        status="quality_gate_failed",
        quality_gate_passed=False,
    )
    manifest = ubuntu_pipeline.read_object(run_path)
    assert manifest["generation"]["repetitionPenalty"] == 1.0
    final_phase = _evaluation_final_phase(manifest)
    verified = ubuntu_pipeline._verify_evaluation_outputs(
        tmp_path,
        "fleet",
        final_phase=final_phase,
        require_passing_status=False,
    )
    assert verified["status"] == "quality_gate_failed"

    manifest["generation"]["repetitionPenalty"] = 1.1
    manifest.pop("runManifestSHA256", None)
    manifest["runManifestSHA256"] = ubuntu_pipeline.canonical_sha256(manifest)
    ubuntu_pipeline.write_object(run_path, manifest)
    with pytest.raises(RuntimeError):
        ubuntu_pipeline._verify_evaluation_outputs(
            tmp_path,
            "fleet",
            final_phase=final_phase,
            require_passing_status=False,
        )


@pytest.mark.parametrize(
    "field",
    [
        "agent",
        "config",
        "configSHA256",
        "datasetDir",
        "variantManifestSHA256",
        "sftAdapterDir",
        "sftFinalizedVariantManifest",
        "sftCheckpointLineagePath",
        "sftTokenLengthPreflight",
        "preferenceTrainer",
        "preferenceAdapterDir",
        "preferenceCheckpointLineagePath",
        "preferenceTokenLengthPreflight",
        "preferenceFinalizedVariantManifest",
        "adapterGGUF",
    ],
)
def test_prepared_agent_entry_binds_every_owned_path_and_value(
    tmp_path: Path,
    field: str,
) -> None:
    run_root = tmp_path / "run"
    agent = "cortex"
    paths = ubuntu_pipeline._expected_agent_paths(run_root, agent)
    variant_root = (
        run_root
        / "generated"
        / "fine_tuning"
        / agent
        / "experiments"
        / OPTIMIZED_VARIANT
    )
    config_sha256 = "a" * 64
    variant_manifest_sha256 = "b" * 64
    entry = {
        "agent": agent,
        "config": str(paths["config"]),
        "configSHA256": config_sha256,
        "datasetDir": str(variant_root),
        "variantManifestSHA256": variant_manifest_sha256,
        "sftAdapterDir": str(paths["adapter_output_dir"]),
        "sftFinalizedVariantManifest": str(
            paths["output_dir"] / "finalized_variant_manifest.json"
        ),
        "sftCheckpointLineagePath": str(paths["sftCheckpointLineagePath"]),
        "sftTokenLengthPreflight": str(paths["sftTokenLengthPreflightPath"]),
        "preferenceTrainer": "dpo",
        "preferenceAdapterDir": str(paths["dpo_output_dir"]),
        "preferenceCheckpointLineagePath": str(
            paths["preferenceCheckpointLineagePath"]
        ),
        "preferenceTokenLengthPreflight": str(
            paths["preferenceTokenLengthPreflightPath"]
        ),
        "preferenceFinalizedVariantManifest": str(
            paths["output_dir"] / "dpo" / "finalized_variant_manifest.json"
        ),
        "adapterGGUF": str(paths["adapter_gguf_output_path"]),
    }
    arguments = {
        "run_root": run_root,
        "agent": agent,
        "config_sha256": config_sha256,
        "variant_root": variant_root,
        "variant_manifest_sha256": variant_manifest_sha256,
        "preference_trainer": "dpo",
    }
    ubuntu_pipeline._verify_prepared_agent_entry(entry, **arguments)

    entry[field] = None
    with pytest.raises(RuntimeError, match="ownership entry drifted"):
        ubuntu_pipeline._verify_prepared_agent_entry(entry, **arguments)


def _copy_cortex_variant_source(tmp_path: Path) -> Path:
    source_root = REPO_ROOT / "generated" / "fine_tuning"
    copied_root = tmp_path / "fine_tuning"
    copied_agent_root = copied_root / "cortex"
    copied_variant_root = copied_agent_root / "experiments" / OPTIMIZED_VARIANT
    copied_variant_root.mkdir(parents=True)
    for filename in ("unsloth_config.json", "eval.jsonl"):
        shutil.copy2(source_root / "cortex" / filename, copied_agent_root / filename)
    source_variant_root = (
        source_root / "cortex" / "experiments" / OPTIMIZED_VARIANT
    )
    for filename in (
        *ubuntu_pipeline.DATASET_FILES,
        "variant_manifest.json",
        "contamination_report.json",
    ):
        shutil.copy2(source_variant_root / filename, copied_variant_root / filename)
    return copied_root


def _rehash_contamination_report(path: Path) -> dict:
    report = ubuntu_pipeline.read_object(path)
    report.pop("reportSHA256", None)
    report["reportSHA256"] = ubuntu_pipeline.canonical_sha256(report)
    ubuntu_pipeline.write_object(path, report)
    return report


def _rehash_variant_manifest(path: Path) -> dict:
    manifest = ubuntu_pipeline.read_object(path)
    manifest.pop("variantManifestSHA256", None)
    manifest["variantManifestSHA256"] = ubuntu_pipeline.canonical_sha256(
        manifest
    )
    ubuntu_pipeline.write_object(path, manifest)
    return manifest


def _validate_copied_cortex_variant(source_root: Path) -> None:
    ubuntu_pipeline.validate_variant(
        source_root,
        agent="cortex",
        variant=OPTIMIZED_VARIANT,
        seed=42,
        base_model_override="",
    )


def test_validate_variant_rejects_rehashed_wrong_schema(tmp_path: Path) -> None:
    source_root = _copy_cortex_variant_source(tmp_path)
    manifest_path = (
        source_root
        / "cortex"
        / "experiments"
        / OPTIMIZED_VARIANT
        / "variant_manifest.json"
    )
    manifest = ubuntu_pipeline.read_object(manifest_path)
    manifest["schemaVersion"] = "lumen.adapter-experiment-variant/1.2.0"
    ubuntu_pipeline.write_object(manifest_path, manifest)
    _rehash_variant_manifest(manifest_path)

    with pytest.raises(RuntimeError, match="schema is unsupported"):
        _validate_copied_cortex_variant(source_root)


@pytest.mark.parametrize("seed", [True, 42.0])
def test_validate_variant_rejects_non_integer_requested_seed(
    tmp_path: Path,
    seed: object,
) -> None:
    source_root = _copy_cortex_variant_source(tmp_path)

    with pytest.raises(RuntimeError, match="would break the controlled variant"):
        ubuntu_pipeline.validate_variant(
            source_root,
            agent="cortex",
            variant=OPTIMIZED_VARIANT,
            seed=seed,  # type: ignore[arg-type]
            base_model_override="",
        )


def _executor_optimization_config(
    *,
    sft_count: int,
    dpo_count: int,
) -> dict[str, Any]:
    policy = expected_optimization_step_policy(
        agent="executor",
        sft_train_record_count=sft_count,
        dpo_train_record_count=dpo_count,
    )
    return {
        "agent": "executor",
        "batch_size": 2,
        "gradient_accumulation_steps": 8,
        "learning_rate": 0.00002,
        "num_train_epochs": policy["sft"]["selectedEpochs"],
        "dpo_num_train_epochs": policy["dpo"]["selectedEpochs"],
        "optimizationStepPolicy": policy,
    }


def test_fleet_variant_optimizer_policy_matches_shared_epoch_contract() -> None:
    expected = expected_optimization_step_policy(
        agent="fleet",
        sft_train_record_count=512,
        dpo_train_record_count=348,
    )
    observed = ubuntu_pipeline._expected_variant_optimization_policy(
        agent="fleet",
        sft_train_record_count=512,
        dpo_train_record_count=348,
    )

    assert observed == expected
    assert observed["sft"]["baseEpochs"] == 4
    assert observed["dpo"]["baseEpochs"] == 1


def test_variant_optimizer_overlay_is_exact_and_type_safe() -> None:
    base = _executor_optimization_config(sft_count=128, dpo_count=32)
    controlled = _executor_optimization_config(sft_count=96, dpo_count=16)
    effective = ubuntu_pipeline._variant_effective_training_config(
        agent="executor",
        base_config=base,
        controlled_config=controlled,
        train_sft_record_count=96,
        train_dpo_record_count=16,
    )
    assert effective == controlled
    assert ubuntu_pipeline.canonical_sha256(
        ubuntu_pipeline._variant_invariant_training_config(
            base,
            agent="executor",
        )
    ) == ubuntu_pipeline.canonical_sha256(
        ubuntu_pipeline._variant_invariant_training_config(
            controlled,
            agent="executor",
            sft_train_record_count=96,
            dpo_train_record_count=16,
        )
    )

    nonvariant_drift = json.loads(json.dumps(controlled))
    nonvariant_drift["learning_rate"] = 0.9
    with pytest.raises(RuntimeError, match="non-variant field"):
        ubuntu_pipeline._variant_effective_training_config(
            agent="executor",
            base_config=base,
            controlled_config=nonvariant_drift,
            train_sft_record_count=96,
            train_dpo_record_count=16,
        )

    policy_drift = json.loads(json.dumps(controlled))
    policy_drift["optimizationStepPolicy"]["sft"][
        "minimumEffectiveSteps"
    ] += 1
    with pytest.raises(RuntimeError, match="does not match its training lanes"):
        ubuntu_pipeline._variant_effective_training_config(
            agent="executor",
            base_config=base,
            controlled_config=policy_drift,
            train_sft_record_count=96,
            train_dpo_record_count=16,
        )

    bool_epoch = json.loads(json.dumps(controlled))
    bool_epoch["num_train_epochs"] = True
    with pytest.raises(RuntimeError, match="does not match its training lanes"):
        ubuntu_pipeline._variant_effective_training_config(
            agent="executor",
            base_config=base,
            controlled_config=bool_epoch,
            train_sft_record_count=96,
            train_dpo_record_count=16,
        )


@pytest.mark.parametrize("field,value", [("batch_size", True), ("gradient_accumulation_steps", 8.0)])
def test_variant_optimizer_invariant_rejects_bool_and_float_batch_state(
    field: str,
    value: Any,
) -> None:
    config = _executor_optimization_config(sft_count=96, dpo_count=16)
    config[field] = value
    with pytest.raises(RuntimeError, match="invariant training config is invalid"):
        ubuntu_pipeline._variant_invariant_training_config(
            config,
            agent="executor",
            sft_train_record_count=96,
            dpo_train_record_count=16,
        )


def test_training_attestation_rejects_missing_null_controlled_field() -> None:
    config = {
        "runExecutionPlan": _test_execution_plan(
            evaluation_scope="smoke",
            evaluation_max_examples=1,
            gguf_requested=False,
        )
    }
    manifest = {
        "controlledTrainingConfig": {"spaceConfigurationSHA256": None}
    }

    with pytest.raises(
        RuntimeError,
        match="lacks controlled fields: spaceConfigurationSHA256",
    ):
        ubuntu_pipeline._training_attestation(config, manifest)


@pytest.mark.parametrize("controlled_seed", [True, 1.0])
def test_training_attestation_rejects_python_numeric_equality(
    controlled_seed: object,
) -> None:
    controlled = {"seed": controlled_seed}
    config = {
        "runExecutionPlan": _test_execution_plan(
            evaluation_scope="smoke",
            evaluation_max_examples=1,
            gguf_requested=False,
        ),
        "seed": 1,
    }
    manifest = {
        "controlledTrainingConfig": controlled,
        "trainingConfigSHA256": ubuntu_pipeline.canonical_sha256(controlled),
    }

    with pytest.raises(RuntimeError, match="drifted from the controlled variant"):
        ubuntu_pipeline._training_attestation(config, manifest)


def test_training_attestation_rejects_declared_exact_hash_drift() -> None:
    controlled = {"seed": 42}
    config = {
        "runExecutionPlan": _test_execution_plan(
            evaluation_scope="smoke",
            evaluation_max_examples=1,
            gguf_requested=False,
        ),
        "seed": 42,
    }
    manifest = {
        "controlledTrainingConfig": controlled,
        "trainingConfigSHA256": "0" * 64,
    }

    with pytest.raises(RuntimeError, match="drifted from the controlled variant"):
        ubuntu_pipeline._training_attestation(config, manifest)


def _evaluation_record(eval_id: str = "eval-one", *, agent: str = "cortex") -> dict:
    metrics = (
        [
            {
                "type": "cortex_route_contract",
                "mode": "actionable",
                "expectedToolID": "files.read",
                "expectedIntent": "files",
            }
        ]
        if agent == "cortex"
        else [
            {"type": "json_field_equals", "path": "status", "expected": "ready"}
        ]
    )
    return {
        "schemaVersion": adapter_evaluation.EVALUATION_SCHEMA_VERSION,
        "evalID": eval_id,
        "messages": [
            {"role": "system", "content": "Follow the agent contract."},
            {"role": "user", "content": "Return the structured result."},
        ],
        "metrics": metrics,
        "outputMode": "text" if agent == "mouth" else "json",
        "metadata": {
            "agent": agent,
            "evalType": "unit",
            "mustPass": True,
            "critical": True,
        },
        "weight": 1.0,
    }


def _write_evaluation_evidence(
    run_root: Path,
    *,
    monkeypatch: pytest.MonkeyPatch,
    agent: str = "cortex",
    status: str = "quality_gate_passed",
    quality_gate_passed: bool = True,
    completion: str | None = None,
    retry_from_completion: str | None = None,
    full_case_count: int = 1,
    generated_case_count: int | None = None,
    evaluation_records: list[dict] | None = None,
    attested_source: bool = False,
) -> Path:
    variant = "internal_plus_public_optimized"
    (run_root / "models" / "lora_qwen3_gguf").mkdir(
        parents=True,
        exist_ok=True,
    )
    (run_root / "models" / "lora_qwen3_gguf_receipts").mkdir(
        parents=True,
        exist_ok=True,
    )
    evaluation_path = run_root / "generated" / "fine_tuning" / agent / "eval.jsonl"
    evaluation_path.parent.mkdir(parents=True, exist_ok=True)
    source_records = (
        evaluation_records
        if evaluation_records is not None
        else [
            _evaluation_record(f"eval-{index}", agent=agent)
            for index in range(1, full_case_count + 1)
        ]
    )
    evaluation_path.write_text(
        "".join(json.dumps(record) + "\n" for record in source_records),
        encoding="utf-8",
    )
    evaluation_module = evaluate_adapter._load_evaluation_module()
    monkeypatch.setattr(
        evaluation_module,
        "_valid_variant_manifest",
        lambda *_args, **_kwargs: True,
    )
    records, evaluation_sha256 = evaluate_adapter.load_evaluation_records(
        evaluation_path,
        agent=agent,
        evaluation_module=evaluation_module,
    )
    fixture_tool = {
        "id": "files.read",
        "displayName": "Read File",
        "description": "Read an imported local file. Args: none.",
        "requiresApproval": False,
        "defaultIntent": "files",
        "allowedIntents": ["files"],
        "arguments": [],
    }
    fixture_tool_contracts = {fixture_tool["id"]: fixture_tool}
    if completion is None:
        completion = json.dumps(
            {
                "selectedToolID": "files.read",
                "intent": "files",
                "reasoningSummary": "Manifest row files.read has no required values.",
                "actionStep": {
                    "type": "tool_call",
                    "toolID": "files.read",
                    "mustPersistBeforeFinal": True,
                },
                "requiresApproval": False,
                "nextModel": "executor",
            },
            separators=(",", ":"),
        )
    generated_count = (
        generated_case_count
        if generated_case_count is not None
        else len(records)
    )
    evaluation_scope = "full" if generated_count == len(records) else "smoke"
    evaluation_plan = _test_execution_plan(
        evaluation_scope=evaluation_scope,
        evaluation_max_examples=(
            generated_count if evaluation_scope == "smoke" else None
        ),
        gguf_requested=False,
    )
    selected_records = (
        list(records)
        if generated_count == len(records)
        else evaluate_adapter.select_evaluation_records(
            records,
            max_examples=generated_count,
        )
    )
    output, output_kind, format_error = evaluate_adapter.normalize_candidate_output(
        agent,
        completion,
        output_mode=records[0]["outputMode"],
        evaluation_module=evaluation_module,
        tool_contracts=fixture_tool_contracts,
    )
    retry_output = None
    retry_output_kind = None
    retry_format_error = None
    if retry_from_completion is not None:
        (
            retry_output,
            retry_output_kind,
            retry_format_error,
        ) = evaluate_adapter.normalize_candidate_output(
            agent,
            retry_from_completion,
            output_mode=records[0]["outputMode"],
            evaluation_module=evaluation_module,
            tool_contracts=fixture_tool_contracts,
        )
        if retry_format_error is None:
            raise AssertionError("Retry fixture requires an invalid first completion")
    candidate_rows = []
    for record in selected_records:
        output_mode = record["outputMode"]
        prompt_messages = evaluate_adapter._structured_output_messages(
            agent,
            record["messages"],
            output_mode=output_mode,
            tool_contracts=fixture_tool_contracts,
        )
        generation_attempts = []
        if retry_from_completion is not None:
            generation_attempts.append(
                evaluate_adapter._generation_attempt_record(
                    attempt_index=1,
                    prompt_kind="frozen_evaluation",
                    messages=prompt_messages,
                    completion=retry_from_completion,
                    output_kind=retry_output_kind,
                    format_error=retry_format_error,
                    input_token_count=4,
                    generated_token_count=3,
                    generation_token_budget=8,
                )
            )
            prompt_messages = evaluate_adapter._strict_json_retry_messages(
                agent,
                prompt_messages,
                validation_error=retry_format_error,
                failed_candidate=retry_output,
                tool_contracts=fixture_tool_contracts,
            )
        generation_attempts.append(
            evaluate_adapter._generation_attempt_record(
                attempt_index=len(generation_attempts) + 1,
                prompt_kind=(
                    "strict_json_retry"
                    if generation_attempts
                    else "frozen_evaluation"
                ),
                messages=prompt_messages,
                completion=completion,
                output_kind=output_kind,
                format_error=format_error,
                input_token_count=4,
                generated_token_count=3,
                generation_token_budget=8,
            )
        )
        candidate_row = {
            "schemaVersion": evaluate_adapter.CANDIDATE_OUTPUT_SCHEMA_VERSION,
            "evalID": record["evalID"],
            "agent": agent,
            "outputMode": output_mode,
            "output": output,
            "outputKind": output_kind,
            "formatError": format_error,
            "inputTokenCount": 4,
            "generatedTokenCount": 3,
            "selectedAttemptIndex": len(generation_attempts),
            "generationAttempts": generation_attempts,
        }
        candidate_row["candidateRecordSHA256"] = (
            evaluate_adapter._canonical_sha256(candidate_row)
        )
        candidate_rows.append(candidate_row)
    evaluation_dir = run_root / "evaluation" / agent
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = evaluation_dir / "candidate_outputs.jsonl"
    candidate_path.write_bytes(evaluate_adapter._jsonl_bytes(candidate_rows))

    behavior_path = (
        run_root / "generated" / "agent_manifest" / "AgentBehaviorManifest.json"
    )
    behavior_path.parent.mkdir(parents=True, exist_ok=True)
    behavior_manifest = {
        "tools": [fixture_tool],
        "routingMatrix": [
            {"intent": "files", "allowedTools": ["files.read"]},
        ],
        "fleet": {"slots": [{"id": "cortex"}]},
    }
    ubuntu_pipeline.write_object(behavior_path, behavior_manifest)
    tool_contracts, allowed_slots, behavior_sha256 = (
        evaluate_adapter.load_behavior_contract(behavior_path)
    )
    adapter_dir = run_root / "models" / "lora_qwen3_dpo" / agent
    adapter_dir.mkdir(parents=True, exist_ok=True)
    finalized_path = (
        run_root
        / "training"
        / agent
        / "dpo"
        / "finalized_variant_manifest.json"
    )
    finalized_path.parent.mkdir(parents=True, exist_ok=True)
    source_variant_sha256 = "a" * 64
    runtime_source = {field: "c" * 40 for field in ubuntu_pipeline.RUNTIME_SOURCE_FIELDS}
    source_fields: dict = {}
    if attested_source:
        source_fields = _source_integrity_fixture()
        runtime_source = {
            "runtimeSourceKind": "git",
            "runtimeSourceRevision": "5" * 40,
            "expectedRuntimeSourceRevision": "5" * 40,
            "observedRepositoryRevision": "5" * 40,
            "observedRuntimeRevision": "5" * 40,
            "runtimeSourceBindingStatus": "verified_clean_snapshot",
            "runtimeSourceBindingMethod": (
                "git_clean_worktree_plus_ubuntu_orchestration_manifest"
            ),
        }
    finalized = {
        "agent": agent,
        "variant": variant,
        "sourceVariantManifestSHA256": source_variant_sha256,
        "frozenEvaluationSHA256": evaluation_sha256,
        **_base_model_tokenizer_lineage_fixture(),
        "trainingEnvironmentSHA256": "7" * 64,
        "resolvedTrainingEnvironment": {"schemaVersion": "test"},
        "resolvedTrainingEnvironmentSHA256": "8" * 64,
        "observedAccelerator": {"backend": "cuda", "deviceCount": 1},
        "zeroGPUSize": None,
        "zeroGPUDurationSeconds": None,
        "artifact": {
            "status": "trained",
            "trainingPhase": "sft_dpo",
            "preferenceTrainer": "dpo",
            "adapterSHA256": "b" * 64,
        },
        **runtime_source,
    }
    finalized["variantManifestSHA256"] = evaluate_adapter._canonical_sha256(
        finalized
    )
    ubuntu_pipeline.write_object(finalized_path, finalized)
    config_path = run_root / "configs" / f"{agent}.final.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    base_config = {
        "agent": agent,
        "variant": variant,
        "variantManifestSHA256": source_variant_sha256,
        "base_model_name": adapter_evaluation.DEFAULT_BASE_MODEL_ID,
        **_base_model_tokenizer_lineage_fixture(),
        **_runtime_snapshot_lineage_fixture(),
        "baseModelIndexDigest": "2" * 64,
        "baseModelIndexReferencedShardNames": ["model.safetensors"],
        "baseModelIndexShardBindingSHA256": "3" * 64,
        "baseModelArtifactDigest": "4" * 64,
        "baseModelWeightShards": [
            {"filename": "model.safetensors", "sha256": "5" * 64, "size": 1}
        ],
        "chatTemplateContract": chat_template_contract(),
        "max_seq_length": 64,
        "max_prompt_length": 32,
        "batch_size": 1,
        "gradient_accumulation_steps": 1,
        "warmup_steps": 0,
        "dpo_learning_rate": 5e-6,
        "dpo_num_train_epochs": 1,
        "dpo_beta": 0.1,
        "dpo_rpo_alpha": 1.0 if agent == "fleet" else None,
        "gradient_checkpointing": True,
        "use_logits_to_keep": True,
        "precompute_ref_log_probs": True,
        "precompute_ref_batch_size": 1,
        "bf16": False,
        "fp16": True,
        "seed": 42,
        "output_dir": str((run_root / "training" / agent).resolve()),
        "adapter_output_dir": str(
            (run_root / "models" / "lora_qwen3_bootstrap" / agent).resolve()
        ),
        "preference_trainer": "dpo",
        "merge_adapters_by_default": False,
        "release_bake_enabled_by_default": False,
        "trainingCodeManifestsByPhase": {"dpo": {"phase": "dpo"}},
        "trainingCodeSHA256ByPhase": {"dpo": "d" * 64},
        "runExecutionPlan": evaluation_plan,
        "variantAttestation": {
            "trainingEnvironmentSHA256": None,
            "executionPlanSHA256": evaluation_plan["executionPlanSHA256"],
        },
        "adapterExport": {
            "adapterArtifact": "bootstrap",
            "adapterDirectory": "bootstrap",
            "adapterGGUFArtifact": "pending",
        },
        **source_fields,
    }
    base_config_path = run_root / "configs" / f"{agent}.json"
    ubuntu_pipeline.write_object(base_config_path, base_config)
    config = ubuntu_pipeline._final_evaluation_config_payload(
        run_root,
        agent,
        base_config=base_config,
        finalized=finalized,
        preference={
            "adapterSHA256": "b" * 64,
            "parentSFTAdapterSHA256": "9" * 64,
            "finalizedVariantManifestSHA256": finalized[
                "variantManifestSHA256"
            ],
            "phase": "dpo",
            "tokenLengthPreflightSHA256": "e" * 64,
            "tokenLengthStatistics": {
                "promptTokens": {"min": 1, "p50": 1, "p95": 1, "max": 1}
            },
        },
        behavior_file_sha=ubuntu_pipeline.file_sha256(behavior_path),
    )
    ubuntu_pipeline.write_object(config_path, config)
    candidate_outputs = {
        row["evalID"]: row["output"] for row in candidate_rows
    }
    report = evaluation_module.score_evaluation_suite(
        selected_records,
        candidate_outputs,
        frozen_evaluation_records=records,
        tool_contracts=tool_contracts,
        allowed_slots=allowed_slots,
        agent=agent,
        variant=variant,
        controlled_lineage=evaluation_module._variant_controlled_lineage(finalized),
        variant_manifest=finalized,
        artifact_sha256="b" * 64,
    )
    report_path = evaluation_dir / "evaluation_report.json"
    ubuntu_pipeline.write_object(report_path, report)
    evaluator_path = Path(evaluate_adapter.__file__).resolve()
    generation = {
        "doSample": False,
        "numBeams": 1,
        "repetitionPenalty": (
            evaluate_adapter._generation_repetition_penalty(agent)
        ),
        "thinkingEnabled": False,
        "maxNewTokens": 8,
        "maxSequenceLength": 64,
        "seed": 42,
        "outputModeContract": evaluate_adapter._evaluation_output_mode_contract(
            selected_records,
            agent=agent,
            tool_contracts=fixture_tool_contracts,
        ),
    }
    run_manifest = {
        "schemaVersion": evaluate_adapter.EVALUATION_RUN_SCHEMA_VERSION,
        "status": status,
        "agent": agent,
        "variant": variant,
        "configPath": str(config_path.resolve()),
        "configSHA256": ubuntu_pipeline.file_sha256(config_path),
        "chatTemplateContract": config["chatTemplateContract"],
        "baseModelTokenizerDigest": config["baseModelTokenizerDigest"],
        "baseModelTokenizerFiles": config["baseModelTokenizerFiles"],
        "baseModelTokenizerClosureSHA256": config[
            "baseModelTokenizerClosureSHA256"
        ],
        **_runtime_snapshot_lineage_fixture(),
        "runtimeModelBinding": {"fixture": "runtime-model"},
        "runtimeTokenizerBinding": {"fixture": "runtime-tokenizer"},
        "adapterDirectory": str(adapter_dir.resolve()),
        "adapterSHA256": "b" * 64,
        "finalizedVariantManifestPath": str(finalized_path.resolve()),
        "finalizedVariantManifestSHA256": finalized["variantManifestSHA256"],
        "evaluatorCodePath": str(evaluator_path),
        "evaluatorCodeSHA256": ubuntu_pipeline.file_sha256(evaluator_path),
        "evaluationJSONLPath": str(evaluation_path.resolve()),
        "evaluationSHA256": evaluation_sha256,
        "behaviorManifestPath": str(behavior_path.resolve()),
        "behaviorManifestSHA256": behavior_sha256,
        "candidateOutputsPath": str(candidate_path.resolve()),
        "candidateOutputsFileSHA256": ubuntu_pipeline.file_sha256(candidate_path),
        "candidateOutputsSHA256": report["candidateOutputsSHA256"],
        "evaluationReportPath": str(report_path.resolve()),
        "evaluationReportFileSHA256": ubuntu_pipeline.file_sha256(report_path),
        "evaluationReportSHA256": report["reportSHA256"],
        "fullCaseCount": len(records),
        "generatedCaseCount": len(selected_records),
        "completeEvaluation": len(selected_records) == len(records),
        "executionPlanSHA256": evaluation_plan["executionPlanSHA256"],
        "evaluationScope": evaluation_plan["evaluationScope"],
        "evaluationMaxExamples": evaluation_plan["evaluationMaxExamples"],
        "initialFormatFailureCount": sum(
            row["generationAttempts"][0]["formatError"] is not None
            for row in candidate_rows
        ),
        "formatRecoveryCount": sum(
            row["generationAttempts"][0]["formatError"] is not None
            and row["generationAttempts"][-1]["formatError"] is None
            for row in candidate_rows
        ),
        "formatFailureCount": sum(
            row["generationAttempts"][-1]["formatError"] is not None
            for row in candidate_rows
        ),
        "criticalFailureCount": report["criticalFailureCount"],
        "qualityGatePassed": quality_gate_passed,
        **source_fields,
        "generation": generation,
    }
    run_manifest["runManifestSHA256"] = ubuntu_pipeline.canonical_sha256(
        run_manifest
    )
    run_path = evaluation_dir / "evaluation_run_manifest.json"
    ubuntu_pipeline.write_object(run_path, run_manifest)
    prepared_behavior_sha256 = ubuntu_pipeline.file_sha256(behavior_path)
    prepared_config_sha256 = ubuntu_pipeline.file_sha256(base_config_path)
    monkeypatch.setattr(
        ubuntu_pipeline,
        "_verified_private_tokenizer_snapshot_binding",
        lambda cfg: dict(cfg["baseModelTokenizerSnapshotVerification"]),
    )
    monkeypatch.setattr(
        ubuntu_pipeline,
        "_verified_private_base_model_runtime_snapshot_binding",
        lambda cfg: dict(cfg["baseModelRuntimeSnapshotVerification"]),
    )
    monkeypatch.setattr(
        ubuntu_pipeline,
        "_verified_runtime_model_binding",
        lambda value, **_kwargs: dict(value),
    )
    monkeypatch.setattr(
        ubuntu_pipeline,
        "_verified_runtime_tokenizer_binding",
        lambda value, **_kwargs: dict(value),
    )
    monkeypatch.setattr(
        ubuntu_pipeline,
        "_verified_run_manifest",
        lambda *_args: {
            "variant": variant,
            "behaviorManifest": str(behavior_path.resolve()),
            "behaviorManifestFileSHA256": prepared_behavior_sha256,
            "executionPlan": evaluation_plan,
            **_summary_base_model_lineage_fixture(),
            **source_fields,
            "agents": [
                {
                    "agent": agent,
                    "config": str(base_config_path.resolve()),
                    "configSHA256": prepared_config_sha256,
                }
            ],
        },
    )
    return run_path


def _evaluation_final_phase(evaluation_run: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "adapterSHA256": "b" * 64,
        "finalizedVariantManifestSHA256": evaluation_run[
            "finalizedVariantManifestSHA256"
        ],
        "parentSFTAdapterSHA256": "9" * 64,
        "phase": "dpo",
        "tokenLengthPreflightSHA256": "e" * 64,
        "tokenLengthStatistics": {
            "promptTokens": {"min": 1, "p50": 1, "p95": 1, "max": 1}
        },
    }


def _rehash_evaluation_run(run_path: Path) -> dict:
    manifest = ubuntu_pipeline.read_object(run_path)
    manifest.pop("runManifestSHA256", None)
    manifest["runManifestSHA256"] = ubuntu_pipeline.canonical_sha256(manifest)
    ubuntu_pipeline.write_object(run_path, manifest)
    return manifest


def _rewrite_report_and_enclosing_hashes(
    run_path: Path,
    mutation,
) -> dict:
    manifest = ubuntu_pipeline.read_object(run_path)
    report_path = Path(manifest["evaluationReportPath"])
    report = ubuntu_pipeline.read_object(report_path)
    mutation(report, manifest)
    report.pop("reportSHA256", None)
    report["reportSHA256"] = ubuntu_pipeline.canonical_sha256(report)
    ubuntu_pipeline.write_object(report_path, report)
    manifest["evaluationReportFileSHA256"] = ubuntu_pipeline.file_sha256(
        report_path
    )
    manifest["evaluationReportSHA256"] = report["reportSHA256"]
    manifest.pop("runManifestSHA256", None)
    manifest["runManifestSHA256"] = ubuntu_pipeline.canonical_sha256(manifest)
    ubuntu_pipeline.write_object(run_path, manifest)
    return manifest


def _write_completed_summary_evidence(
    run_root: Path,
    *,
    monkeypatch: pytest.MonkeyPatch,
    evaluation_scope: str = "full",
) -> Path:
    agent = "cortex"
    variant = "internal_plus_public_optimized"
    sft = {
        "phase": "sft",
        "adapterSHA256": "a" * 64,
        **_phase_runtime_evidence_fixture("1"),
    }
    final_phase = {
        "phase": "dpo",
        "adapterSHA256": "b" * 64,
        "parentSFTAdapterSHA256": "a" * 64,
        **_phase_runtime_evidence_fixture("2"),
        "finalizedVariantManifestSHA256": "f" * 64,
    }
    if evaluation_scope == "full":
        plan = _test_execution_plan()
        evaluation = {
            "status": "quality_gate_passed",
            "qualityGatePassed": True,
            "runtimeModelBinding": {"runtimeModelBindingSHA256": "a" * 64},
            "runtimeTokenizerBinding": {
                "runtimeTokenizerBindingSHA256": "a" * 64
            },
        }
        status = "complete"
        evaluation_status = "quality_gate_passed"
        qualification = "quality_gate_passed"
        promotion_eligible = True
    elif evaluation_scope == "smoke":
        plan = _test_execution_plan(
            evaluation_scope="smoke",
            evaluation_max_examples=1,
        )
        evaluation = {
            "status": "smoke_complete",
            "qualityGatePassed": False,
            "runtimeModelBinding": {"runtimeModelBindingSHA256": "a" * 64},
            "runtimeTokenizerBinding": {
                "runtimeTokenizerBindingSHA256": "a" * 64
            },
        }
        status = "smoke_complete"
        evaluation_status = "smoke_complete"
        qualification = "diagnostic_only"
        promotion_eligible = False
    elif evaluation_scope == "none":
        plan = _test_execution_plan(evaluation_scope="none")
        evaluation = None
        status = "training_complete_without_full_evaluation"
        evaluation_status = "not_run"
        qualification = "diagnostic_only"
        promotion_eligible = False
    else:  # pragma: no cover - helper callers are closed above.
        raise AssertionError(evaluation_scope)
    source_fields = {
        "workingTreeDigest": "1" * 64,
        "ubuntuOrchestrationCodeSHA256": "2" * 64,
        "ubuntuSourceIntegritySHA256": "3" * 64,
        "ubuntuSourceIntegrity": {"testFixture": True},
    }
    smoke_evidence = _write_runtime_binding_smoke_summary_fixture(run_root)
    gguf = (
        run_root
        / "models"
        / "lora_qwen3_gguf"
        / "lumen-cortex-lora.gguf"
    )
    gguf.parent.mkdir(parents=True, exist_ok=True)
    gguf.write_bytes(_gguf_bytes())
    conversion_receipt = (
        run_root
        / "models"
        / "lora_qwen3_gguf_receipts"
        / "lumen-cortex-lora.conversion.json"
    )
    conversion_receipt.parent.mkdir(parents=True, exist_ok=True)
    ubuntu_pipeline.write_object(conversion_receipt, {"fixture": True})
    reader_script = _write_fake_gguf_reader(run_root)
    gguf_verification = ubuntu_pipeline.verify_gguf_artifact(
        gguf,
        reader_script=reader_script,
    )
    evaluation_report = (
        run_root / "evaluation" / agent / "evaluation_report.json"
    )
    if evaluation is not None:
        evaluation_report.parent.mkdir(parents=True, exist_ok=True)
        evaluation_report.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        ubuntu_pipeline,
        "_verified_run_manifest",
        lambda *_args: {
            "variant": variant,
            "executionPlan": plan,
            "agents": [{"agent": agent}],
            **_summary_base_model_lineage_fixture(),
            **source_fields,
        },
    )
    monkeypatch.setattr(ubuntu_pipeline, "verify_sft", lambda *_args: sft)
    monkeypatch.setattr(
        ubuntu_pipeline,
        "_verified_runtime_binding_smoke_summary_evidence",
        lambda *_args: smoke_evidence,
    )
    monkeypatch.setattr(
        ubuntu_pipeline,
        "verify_preference",
        lambda *_args: final_phase,
    )
    monkeypatch.setattr(
        ubuntu_pipeline,
        "_verify_evaluation_outputs",
        lambda *_args, **_kwargs: evaluation,
    )
    monkeypatch.setattr(
        ubuntu_pipeline,
        "_verified_pinned_gguf_reader_script",
        lambda *_args: reader_script,
    )
    gguf_verification.update(
        {
            "adapterGGUFConversionReceipt": str(conversion_receipt),
            "adapterGGUFConversionReceiptSHA256": "4" * 64,
            "adapterGGUFConversionQualification": (
                ubuntu_pipeline.GGUF_CONVERSION_QUALIFICATION
            ),
            "adapterGGUFTensorEquivalenceStatus": (
                ubuntu_pipeline.GGUF_TENSOR_EQUIVALENCE_STATUS
            ),
            "adapterGGUFRuntimeModelBindingSHA256": "a" * 64,
            "adapterGGUFRuntimeTokenizerBindingSHA256": "a" * 64,
        }
    )
    def verify_fixture_gguf(_run_root: Path, path: Path) -> dict:
        current = ubuntu_pipeline.verify_gguf_artifact(
            path,
            reader_script=reader_script,
        )
        current.update(
            {
                field: gguf_verification[field]
                for field in ubuntu_pipeline.GGUF_CONVERSION_SUMMARY_FIELDS
            }
        )
        return current

    monkeypatch.setattr(
        ubuntu_pipeline,
        "verify_gguf_file",
        verify_fixture_gguf,
    )
    summary = {
        "schema": ubuntu_pipeline.SUMMARY_SCHEMA_VERSION,
        "status": status,
        "trainingScope": "sft_preference",
        "evaluationStatus": evaluation_status,
        "evaluationScope": evaluation_scope,
        "ggufStatus": "verified",
        "ggufConversionStatus": ubuntu_pipeline.GGUF_CONVERSION_QUALIFICATION,
        "ggufTensorEquivalenceStatus": (
            ubuntu_pipeline.GGUF_TENSOR_EQUIVALENCE_STATUS
        ),
        "qualification": qualification,
        "promotionEligible": promotion_eligible,
        "executionPlanSHA256": plan["executionPlanSHA256"],
        "variant": variant,
        "runRoot": str(run_root),
        "preferenceTraining": True,
        **smoke_evidence,
        **_summary_base_model_lineage_fixture(),
        **source_fields,
        "agents": {
            agent: {
                "sft": sft,
                "finalPhase": final_phase,
                "adapterGGUF": str(gguf),
                "adapterGGUFExists": True,
                "adapterGGUFSHA256": ubuntu_pipeline.file_sha256(gguf),
                "adapterGGUFSizeBytes": gguf.stat().st_size,
                **{
                    field: gguf_verification[field]
                    for field in ubuntu_pipeline.ADAPTER_GGUF_SEMANTIC_FIELDS
                },
                **{
                    field: gguf_verification[field]
                    for field in ubuntu_pipeline.GGUF_CONVERSION_SUMMARY_FIELDS
                },
                "evaluationReport": str(evaluation_report),
                "evaluationReportExists": evaluation is not None,
                "evaluation": evaluation,
            }
        },
    }
    summary["summarySHA256"] = ubuntu_pipeline.canonical_sha256(summary)
    summary_path = run_root / "aio_summary.json"
    ubuntu_pipeline.write_object(summary_path, summary)
    return summary_path


def test_docker_context_includes_the_dependency_lineage_build_preflight() -> None:
    docker_root = REPO_ROOT / "tools/fine_tuning/unsloth"
    dockerfile = (docker_root / "Dockerfile.ubuntu-cu128").read_text(
        encoding="utf-8"
    )
    dockerignore = (
        (docker_root / "Dockerfile.ubuntu-cu128.dockerignore")
        .read_text(encoding="utf-8")
        .splitlines()
    )

    assert dockerfile.splitlines()[0] == (
        "FROM nvidia/cuda:12.8.1-devel-ubuntu22.04@"
        "sha256:a99a1860ba8e2916e5c3e73b72ec4c4301653a84586e05bfc9a2aa2d58027e97"
    )
    assert (
        "COPY tools/fine_tuning/unsloth/training_lineage.py "
        "/tmp/lumen-training-lineage.py"
    ) in dockerfile
    assert "lineage.verify_training_dependency_lock(" in dockerfile
    assert "setpriv --reuid=nobody --regid=nogroup --init-groups python" in dockerfile
    assert "lineage.build_resolved_training_environment_snapshot()" in dockerfile
    assert "lineage.verify_resolved_training_environment(environment)" in dockerfile
    assert dockerignore[0] == "**"
    assert {
        "!scripts/ubuntu_run_fleet_canary.sh",
        "!scripts/ubuntu_train_lumen_full_pipeline.sh",
        "!scripts/ubuntu_train_lumen_adapters_aio.sh",
        "!lumen_manifest_crawler/__init__.py",
        "!tools/fine_tuning/unsloth/**",
        "!tools/lumen_manifest_crawler/lumen_manifest_crawler/**",
        "!tools/hf_zerogpu/space_template/**",
        "!generated/fine_tuning/**",
        "!generated/agent_manifest/AgentBehaviorManifest.json",
    }.issubset(dockerignore)
    assert (
        "COPY tools/hf_zerogpu/space_template/requirements.txt "
        "/tmp/lumen-requirements.txt"
    ) in dockerfile
    copy_sources = {
        line.split()[1]
        for line in dockerfile.splitlines()
        if line.startswith("COPY ")
    }
    assert {
        "tools/fine_tuning/unsloth/training_lineage.py",
        "tools/hf_zerogpu/space_template/requirements.txt",
        "scripts/ubuntu_run_fleet_canary.sh",
        "scripts/ubuntu_train_lumen_full_pipeline.sh",
        "scripts/ubuntu_train_lumen_adapters_aio.sh",
        "lumen_manifest_crawler/__init__.py",
        "tools/fine_tuning/unsloth",
        "tools/lumen_manifest_crawler/lumen_manifest_crawler",
        "tools/hf_zerogpu/space_template",
        "generated/fine_tuning",
        "generated/agent_manifest/AgentBehaviorManifest.json",
    }.issubset(copy_sources)
    assert (
        "COPY lumen_manifest_crawler/__init__.py "
        "/opt/lumen/source/lumen_manifest_crawler/__init__.py"
    ) in dockerfile
    assert (
        "COPY lumen_manifest_crawler /opt/lumen/source/lumen_manifest_crawler"
        not in dockerfile
    )
    assert "lumen_manifest_crawler/__main__.py" not in dockerfile
    assert "!lumen_manifest_crawler/**" not in dockerignore
    assert "cd /opt/lumen/source" in dockerfile
    assert "env -u PYTHONPATH python - <<'PY'" in dockerfile
    assert "python -I - <<'PY'" in dockerfile
    assert "sys.path.insert(0, str(source_root))" in dockerfile
    assert "import lumen_manifest_crawler" in dockerfile
    assert "from lumen_manifest_crawler.dataset import chat_template_contract" in dockerfile
    assert (
        "from tools.fine_tuning.unsloth import "
        "runtime_binding_smoke_gate, ubuntu_pipeline"
    ) in dockerfile
    assert "Path(lumen_manifest_crawler.__file__).resolve()" in dockerfile
    assert "Path(chat_template_contract.__file__).resolve().is_relative_to(" in dockerfile
    assert "Path(ubuntu_pipeline.__file__).resolve().is_relative_to(source_root)" in dockerfile


def test_inner_launcher_uses_the_verified_repo_root_shim_without_pythonpath() -> None:
    launcher = (
        REPO_ROOT / "scripts/ubuntu_train_lumen_adapters_aio.sh"
    ).read_text(encoding="utf-8")

    assert 'ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"' in launcher
    assert 'cd "$ROOT"' in launcher
    assert "PYTHONPATH=" not in launcher


def test_credential_uploader_uses_only_the_verified_image_copy() -> None:
    launcher = (
        REPO_ROOT / "scripts/ubuntu_train_lumen_full_pipeline.sh"
    ).read_text(encoding="utf-8")

    assert '$ROOT:/workspace' not in launcher
    assert "PYTHONPATH=" not in launcher
    assert (
        '"$IMAGE_SOURCE_ROOT/tools/fine_tuning/unsloth/ubuntu_uploader.py"'
        in launcher
    )
    assert '-v "$host_run_root:$container_run_root:ro"' in launcher
    assert '-v "$receipt_staging:/receipts:rw"' in launcher
    assert "--read-only" in launcher
    assert "--cap-drop ALL" in launcher
    assert "--security-opt no-new-privileges" in launcher
    assert (
        'PRIVATE_UPLOAD_TMPFS="/tmp:rw,noexec,nosuid,nodev,mode=700,'
        'uid=$RUNTIME_UID,gid=$RUNTIME_GID"'
    ) in launcher
    assert launcher.count('--tmpfs "$PRIVATE_UPLOAD_TMPFS"') == 2
    assert 'tempfile.TemporaryFile(dir="/tmp").close()' in launcher
    assert "scratch.st_uid == uid" in launcher
    assert "scratch.st_gid == gid" in launcher
    assert "stat.S_IMODE(scratch.st_mode) == 0o700" in launcher
    assert "--source-integrity-digest" in launcher


def test_host_launcher_binds_an_expected_commit_to_source_attestation() -> None:
    launcher = (
        REPO_ROOT / "scripts/ubuntu_train_lumen_full_pipeline.sh"
    ).read_text(encoding="utf-8")

    assert (
        'EXPECTED_SOURCE_COMMIT="${LUMEN_UBUNTU_EXPECTED_SOURCE_COMMIT:-}"'
        in launcher
    )
    assert "--expected-source-commit)" in launcher
    assert (
        '[[ -z "$EXPECTED_SOURCE_COMMIT" '
        '|| "$SOURCE_BASE_COMMIT" == "$EXPECTED_SOURCE_COMMIT" ]]'
    ) in launcher
    assert 'SOURCE_ATTESTATION_OUTPUT="$(read_source_attestation_fields)"' in launcher
    assert (
        'mapfile -t SOURCE_ATTESTATION_FIELDS <<< "$SOURCE_ATTESTATION_OUTPUT"'
        in launcher
    )
    assert "mapfile -t SOURCE_ATTESTATION_FIELDS < <" not in launcher


def test_host_launchers_share_one_persistent_training_reservation() -> None:
    wrapper = (REPO_ROOT / "scripts/ubuntu_run_fleet_canary.sh").read_text(
        encoding="utf-8"
    )
    launcher = (
        REPO_ROOT / "scripts/ubuntu_train_lumen_full_pipeline.sh"
    ).read_text(encoding="utf-8")

    state_root = 'HOST_STATE_ROOT="/var/tmp/lumen-fleet-canary-state"'
    assert state_root in wrapper
    assert state_root in launcher
    assert "LUMEN_UBUNTU_HOST_RESERVATION_FD=8" in wrapper
    assert (
        'HOST_RESERVATION_FD="${LUMEN_UBUNTU_HOST_RESERVATION_FD:-}"'
        in launcher
    )
    assert 'stat -Lc \'%d:%i:%u:%a:%F\' "/proc/$$/fd/8"' in launcher
    assert 'exec 8<>"$HOST_RESERVATION_LOCK"' in launcher
    assert "flock -n 8" in wrapper
    assert "flock -n 8" in launcher
    assert "wrapper-inherited host reservation is missing" in launcher
    assert "marker-bound resume requires --expected-source-commit" in launcher


def test_host_reservation_rejects_competing_durable_training_containers() -> None:
    launcher = (
        REPO_ROOT / "scripts/ubuntu_train_lumen_full_pipeline.sh"
    ).read_text(encoding="utf-8")

    reservation = launcher[
        launcher.index("verify_running_training_reservation()") :
        launcher.index("training_launch_contract_digest()")
    ]
    assert "docker ps" in reservation
    assert "--no-trunc" in reservation
    assert "label=ai.lumen.purpose=ubuntu-training" in reservation
    assert '[[ "$RESUME" == "1" ]]' in reservation
    assert "permits only an exact --resume" in reservation
    assert "more than one durable Lumen training container is running" in reservation
    assert 'docker container inspect "$active_id"' in reservation
    assert 'labels.get("ai.lumen.host-run-root") != expected_run_root' in reservation
    reservation_call = launcher.index("verify_running_training_reservation\n")
    assert (
        reservation_call
        < launcher.index('mkdir -p -- "$OUTPUT_ROOT"')
        < launcher.index("build_args=(", reservation_call)
    )


def test_marker_bound_resume_requires_stopped_ollama_and_existing_run() -> None:
    launcher = (
        REPO_ROOT / "scripts/ubuntu_train_lumen_full_pipeline.sh"
    ).read_text(encoding="utf-8")

    marker_verifier = launcher[
        launcher.index("verify_host_restore_marker()") :
        launcher.index("training_container_name()")
    ]
    assert (
        "{{.Id}}|{{.Name}}|{{.State.Status}}|{{.State.Running}}|"
        "{{.HostConfig.RestartPolicy.Name}}"
    ) in marker_verifier
    assert 'current_container_name" == "/$marker_container_name"' in marker_verifier
    assert 'current_status" == "exited"' in marker_verifier
    assert 'current_running" == "false"' in marker_verifier
    assert (
        "marker-bound resume requires the existing exact run root" in launcher
    )
    assert (
        "Ollama restore marker or stopped-container validation failed" in launcher
    )
    assert "EXACT_EXISTING_RESUME_REQUIRED=1" in launcher
    assert "exact recovery run root disappeared before bind reservation" in launcher


def test_long_running_gpu_container_is_durable_and_recoverable() -> None:
    launcher = (
        REPO_ROOT / "scripts/ubuntu_train_lumen_full_pipeline.sh"
    ).read_text(encoding="utf-8")
    training_block = launcher[
        launcher.index("training_create_args=(") : launcher.index(
            'if [[ "$UPLOAD" == "1" ]]',
            launcher.index("training_create_args=("),
        )
    ]

    assert "docker create --name \"$training_container\"" in training_block
    assert "docker run --rm" not in training_block
    assert 'docker start "$training_container"' in training_block
    assert 'docker logs --timestamps --follow "$training_container"' in training_block
    assert 'docker wait "$training_container"' in training_block
    assert '--label "ai.lumen.source-integrity-sha256=$SOURCE_INTEGRITY_DIGEST"' in training_block
    assert '--label "ai.lumen.launch-contract-sha256=$launch_contract_digest"' in training_block
    assert '--label "ai.lumen.host-run-root-identity=$host_run_root_identity"' in training_block
    assert '--label "ai.lumen.host-lock-identity=$HOST_LOCK_IDENTITY"' in training_block
    assert '--label "ai.lumen.container-run-root=$container_run_root"' in training_block
    assert '-v "$OUTPUT_ROOT:/outputs:rw"' not in launcher
    assert '-v "$host_run_root:$container_run_root:rw"' in training_block
    assert '-v "$HOST_LOCK_DIR:$CONTAINER_LOCK_DIR:ro"' in training_block
    assert "verify_exact_training_mounts" in launcher
    assert "expected_bind_creation=new" in launcher
    assert '"$host_run_root" 1 "$expected_bind_creation"' in launcher
    assert "'{{json .Mounts}}'" in launcher
    assert 'if "/outputs" in observed' in launcher
    assert "bash -l" not in launcher
    assert "/etc/profile" not in launcher
    assert "verify_training_container_identity" in training_block
    assert "verify_training_postcondition" in training_block
    assert "ubuntu_postcondition.py" in launcher
    assert "--network none" in launcher
    assert "OOMKilled=" in training_block
    assert "and was retained as $training_container; rerun with --resume" in training_block
    assert 'docker rm "$training_container"' in training_block
    assert training_block.rindex("capture_training_container_evidence") < (
        training_block.rindex('docker rm "$training_container"')
    )


def test_precreated_bind_preparation_commits_owner_first_and_manifest_last() -> None:
    pipeline = (
        REPO_ROOT / "tools/fine_tuning/unsloth/ubuntu_pipeline.py"
    ).read_text(encoding="utf-8")
    prepare = pipeline[
        pipeline.index("def prepare_run(") : pipeline.index(
            "def _verified_run_manifest(", pipeline.index("def prepare_run(")
        )
    ]
    owner_commit = prepare.index("_initialize_preparation_root(")
    first_snapshot = prepare.index("_copy_private_regular_tree(")
    final_manifest = prepare.index(
        'write_object(run_root / "aio_run_manifest.json", run_manifest)'
    )
    owner_removal = prepare.index("_durably_remove_preparation_owner(run_root)")

    assert owner_commit < first_snapshot < final_manifest < owner_removal
    assert '"runRootInitializationMode"' in prepare
    assert "precreated_bind_root=precreated_bind_root" in prepare

    inner = (
        REPO_ROOT / "scripts/ubuntu_train_lumen_adapters_aio.sh"
    ).read_text(encoding="utf-8")
    assert "LUMEN_AIO_PRECREATED_BIND_ROOT" in inner
    assert "LUMEN_AIO_EXPECTED_RUN_ROOT_IDENTITY" in inner
    assert "reset-owned-run-root" in inner
    assert 'rm -rf -- "$RUN_ROOT"' in inner  # retained only for direct/non-bind use


def test_docker_mount_identity_validator_rejects_the_old_broad_output_mount() -> None:
    launcher = (
        REPO_ROOT / "scripts/ubuntu_train_lumen_full_pipeline.sh"
    ).read_text(encoding="utf-8")
    marker = "mounts = json.loads(sys.argv[1])"
    start = launcher.index("import json\nimport sys\n\n" + marker)
    end = launcher.index("\nPY\n}", start)
    validator = launcher[start:end]
    host_run = "/host/outputs/run-one"
    container_run = "/outputs/run-one"
    host_lock = "/host/outputs/.lumen-training.lock"
    container_lock = "/run/lumen-training-lock"
    hub = "/host/cache/hub"
    xet = "/host/cache/xet"
    assets = "/host/cache/assets"

    def mount(source: str, destination: str, writable: bool) -> dict[str, object]:
        return {
            "Type": "bind",
            "Source": source,
            "Destination": destination,
            "Mode": "rw" if writable else "ro",
            "RW": writable,
            "Propagation": "rprivate",
        }

    exact = [
        mount(host_run, container_run, True),
        mount(host_lock, container_lock, False),
        mount(hub, "/cache/huggingface/hub", True),
        mount(xet, "/cache/huggingface/xet", True),
        mount(assets, "/cache/huggingface/assets", True),
    ]
    arguments = (
        host_run,
        container_run,
        host_lock,
        container_lock,
        hub,
        xet,
        assets,
    )
    accepted = subprocess.run(
        [sys.executable, "-", json.dumps(exact), *arguments],
        input=validator,
        text=True,
        capture_output=True,
        check=False,
    )
    assert accepted.returncode == 0, accepted.stderr

    broad = list(exact)
    broad[0] = mount("/host/outputs", "/outputs", True)
    rejected = subprocess.run(
        [sys.executable, "-", json.dumps(broad), *arguments],
        input=validator,
        text=True,
        capture_output=True,
        check=False,
    )
    assert rejected.returncode != 0
    assert "broad output root" in rejected.stderr


def test_docker_build_uses_only_the_attested_commit_archive() -> None:
    launcher = (
        REPO_ROOT / "scripts/ubuntu_train_lumen_full_pipeline.sh"
    ).read_text(encoding="utf-8")
    archive_function = launcher[
        launcher.index("archive_attested_build_context()") : launcher.index(
            "path_contains()"
        )
    ]
    build_start = launcher.index('if [[ "$BUILD_IMAGE" == "1" ]]')
    build_block = launcher[
        build_start : launcher.index("IMAGE_DIGEST=", build_start)
    ]

    assert "set -Eeuo pipefail" in launcher
    assert "archive" in archive_function
    assert "--format=tar" in archive_function
    assert '"$SOURCE_BASE_COMMIT"' in archive_function
    assert "GIT_CONFIG_GLOBAL=/dev/null" in archive_function
    assert "GIT_NO_REPLACE_OBJECTS=1" in archive_function
    assert "-u GIT_DIR" in archive_function
    assert "-u GIT_WORK_TREE" in archive_function
    assert "-u GIT_INDEX_FILE" in archive_function
    assert {
        "scripts/ubuntu_run_fleet_canary.sh",
        "scripts/ubuntu_train_lumen_full_pipeline.sh",
        "scripts/ubuntu_train_lumen_adapters_aio.sh",
        "lumen_manifest_crawler/__init__.py",
        "tools/fine_tuning/unsloth",
        "tools/lumen_manifest_crawler/lumen_manifest_crawler",
        "tools/hf_zerogpu/space_template",
        "generated/fine_tuning",
        "generated/agent_manifest/AgentBehaviorManifest.json",
    }.issubset(set(archive_function.split()))
    assert '--file "$DOCKERFILE_RELATIVE"' in build_block
    assert '--file "$DOCKERFILE"' not in build_block
    assert 'build_args+=("$ROOT")' not in build_block
    assert 'archive_attested_build_context | "${build_args[@]}" -' in build_block


def test_attested_build_archive_ignores_live_checkout_bytes_and_fails_closed(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    files = {
        "scripts/ubuntu_run_fleet_canary.sh": b"trusted canary wrapper\n",
        "scripts/ubuntu_train_lumen_full_pipeline.sh": b"trusted launcher\n",
        "scripts/ubuntu_train_lumen_adapters_aio.sh": b"trusted inner launcher\n",
        "lumen_manifest_crawler/__init__.py": b"trusted crawler shim\n",
        "tools/fine_tuning/unsloth/Dockerfile.ubuntu-cu128": b"trusted dockerfile\n",
        "tools/lumen_manifest_crawler/lumen_manifest_crawler/__init__.py": (
            b"trusted crawler\n"
        ),
        "tools/hf_zerogpu/space_template/app.py": b"trusted app\n",
        "generated/fine_tuning/manifest.json": b"{}\n",
        "generated/agent_manifest/AgentBehaviorManifest.json": b"{}\n",
    }
    for relative, payload in files.items():
        path = repository / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
    subprocess.run(
        ["git", "config", "user.name", "Lumen Test"],
        cwd=repository,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "lumen@example.invalid"],
        cwd=repository,
        check=True,
    )
    subprocess.run(["git", "add", "."], cwd=repository, check=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", "trusted"],
        cwd=repository,
        check=True,
    )
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        text=True,
    ).strip()
    dockerfile = repository / "tools/fine_tuning/unsloth/Dockerfile.ubuntu-cu128"
    dockerfile.write_bytes(b"transient untrusted dockerfile\n")
    crawler_shim = repository / "lumen_manifest_crawler/__init__.py"
    crawler_shim.write_bytes(b"transient untrusted crawler shim\n")

    launcher = (
        REPO_ROOT / "scripts/ubuntu_train_lumen_full_pipeline.sh"
    ).read_text(encoding="utf-8")
    archive_function = launcher[
        launcher.index("archive_attested_build_context()") : launcher.index(
            "path_contains()"
        )
    ]
    script = (
        f"set -Eeuo pipefail\n{archive_function}\n"
        "archive_attested_build_context\n"
    )
    environment = {
        **os.environ,
        "ROOT": str(repository),
        "SOURCE_BASE_COMMIT": revision,
        "GIT_DIR": str(tmp_path / "attacker-git-dir"),
        "GIT_WORK_TREE": str(tmp_path / "attacker-worktree"),
        "GIT_INDEX_FILE": str(tmp_path / "attacker-index"),
    }
    archived = subprocess.run(
        ["bash", "-c", script],
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    ).stdout
    with tarfile.open(fileobj=io.BytesIO(archived), mode="r:") as archive:
        archived_dockerfile = archive.extractfile(
            "tools/fine_tuning/unsloth/Dockerfile.ubuntu-cu128"
        )
        assert archived_dockerfile is not None
        assert archived_dockerfile.read() == b"trusted dockerfile\n"
        archived_crawler_shim = archive.extractfile(
            "lumen_manifest_crawler/__init__.py"
        )
        assert archived_crawler_shim is not None
        assert archived_crawler_shim.read() == b"trusted crawler shim\n"
        assert "lumen_manifest_crawler/__main__.py" not in archive.getnames()

    failed = subprocess.run(
        ["bash", "-c", script],
        env={**environment, "SOURCE_BASE_COMMIT": "f" * 40},
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert failed.returncode != 0


def test_ubuntu_image_maps_the_invoking_non_root_identity() -> None:
    dockerfile = (
        REPO_ROOT / "tools/fine_tuning/unsloth/Dockerfile.ubuntu-cu128"
    ).read_text(encoding="utf-8")
    launcher = (
        REPO_ROOT / "scripts/ubuntu_train_lumen_full_pipeline.sh"
    ).read_text(encoding="utf-8")

    assert "ARG LUMEN_RUNTIME_UID=1000" in dockerfile
    assert "ARG LUMEN_RUNTIME_GID=1000" in dockerfile
    assert 'getent passwd "${LUMEN_RUNTIME_UID}"' in dockerfile
    assert 'getent group "${LUMEN_RUNTIME_GID}"' in dockerfile
    assert "pwd.getpwuid(uid).pw_uid == uid" in dockerfile
    assert "grp.getgrgid(gid).gr_gid == gid" in dockerfile
    assert (
        "UNSLOTH_COMPILE_LOCATION=/home/lumen-runtime/.cache/unsloth_compiled_cache"
        in dockerfile
    )
    assert 'os.environ["UNSLOTH_COMPILE_LOCATION"]' in dockerfile
    assert "USER ${LUMEN_RUNTIME_UID}:${LUMEN_RUNTIME_GID}" in dockerfile
    assert '--build-arg "LUMEN_RUNTIME_UID=$RUNTIME_UID"' in launcher
    assert '--build-arg "LUMEN_RUNTIME_GID=$RUNTIME_GID"' in launcher
    assert "pwd.getpwuid(uid).pw_uid == uid" in launcher
    assert 'HOME=$RUNTIME_HOME' in launcher


def test_ubuntu_launcher_rejects_an_image_without_the_runtime_identity(
    tmp_path: Path,
) -> None:
    if sys.platform != "linux":
        pytest.skip("launcher harness requires the Ubuntu/GNU host utilities")
    bash_major = int(
        subprocess.check_output(
            ["bash", "-c", 'printf "%s" "${BASH_VERSINFO[0]}"'],
            text=True,
        )
    )
    if bash_major < 4:
        pytest.skip("Ubuntu launcher requires Bash 4 associative arrays")

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    docker_log = tmp_path / "docker.log"
    repository_head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        text=True,
    ).strip()
    fake_commands = {
        "uname": """#!/bin/sh
printf 'Linux\\n'
""",
        "id": """#!/bin/sh
case "${1:-}" in
  -u) printf '12345\\n' ;;
  -g) printf '23456\\n' ;;
  *) exit 2 ;;
esac
""",
        "nvidia-smi": """#!/bin/sh
exit 0
""",
        "python3": f"""#!/bin/sh
case "$*" in
  *'ubuntu_source_integrity.py attest-host'*)
    printf '%s\n' '{{"baseCommit":"{repository_head}","workingTreeDigest":"{'b' * 64}","ubuntuOrchestrationCodeSHA256":"{'c' * 64}","sourceIntegritySHA256":"{'d' * 64}"}}'
    exit 0
    ;;
esac
exec {sys.executable} "$@"
""",
        "docker": f"""#!/bin/sh
set -eu
printf '%s\\n' "$*" >> "$FAKE_DOCKER_LOG"
if [ "${{1:-}}" = build ]; then
  cat >/dev/null
  exit 0
fi
if [ "${{1:-}}" = image ] && [ "${{2:-}}" = inspect ]; then
  printf '{FAKE_IMAGE_DIGEST}\\n'
  exit 0
fi
if [ "${{1:-}}" = ps ]; then
  exit 0
fi
case "$*" in
  *'verify-image'*) exit 0 ;;
  *'/opt/lumen-venv/bin/python'*) exit 23 ;;
esac
exit 0
""",
    }
    for name, source in fake_commands.items():
        path = fake_bin / name
        path.write_text(source, encoding="utf-8")
        path.chmod(0o755)

    result = subprocess.run(
        [
            "bash",
            "scripts/ubuntu_train_lumen_full_pipeline.sh",
            "--prepare-only",
            "--no-pull",
            "--run-id",
            "identity-probe",
            "--output-dir",
            str(tmp_path / "outputs"),
            "--hf-cache",
            str(tmp_path / "hf-cache"),
        ],
        cwd=REPO_ROOT,
        env={
            **os.environ,
            "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
            "FAKE_DOCKER_LOG": str(docker_log),
        },
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "training image lacks the invoking user's passwd/group mapping" in (
        result.stderr
    )
    logged = docker_log.read_text(encoding="utf-8")
    assert "--build-arg LUMEN_RUNTIME_UID=12345" in logged
    assert "--build-arg LUMEN_RUNTIME_GID=23456" in logged
    assert "--user 12345:23456" in logged
    assert "/opt/lumen-venv/bin/python" in logged
    assert "--gpus all" not in logged


@pytest.mark.parametrize("training_exit_code", (0, 137))
def test_ubuntu_launcher_detaches_and_collects_container_evidence(
    tmp_path: Path,
    training_exit_code: int,
) -> None:
    if sys.platform != "linux":
        pytest.skip("launcher harness requires the Ubuntu/GNU host utilities")

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    docker_log = tmp_path / "docker.log"
    docker_state = tmp_path / "docker.state"
    docker_contract = tmp_path / "docker.contract"
    docker_environment = tmp_path / "docker.environment"
    docker_launch_mode = tmp_path / "docker.launch-mode"
    docker_run_root = tmp_path / "docker.run-root"
    repository_head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        text=True,
    ).strip()
    container_id = "1" * 64
    source_integrity = "d" * 64
    fake_commands = {
        "uname": """#!/bin/sh
printf 'Linux\\n'
""",
        "id": """#!/bin/sh
case "${1:-}" in
  -u) printf '12345\\n' ;;
  -g) printf '23456\\n' ;;
  *) exit 2 ;;
esac
""",
        "nvidia-smi": """#!/bin/sh
exit 0
""",
        "python3": f"""#!/bin/sh
case "$*" in
  *'ubuntu_source_integrity.py attest-host'*)
    printf '%s\\n' '{{"baseCommit":"{repository_head}","workingTreeDigest":"{'b' * 64}","ubuntuOrchestrationCodeSHA256":"{'c' * 64}","sourceIntegritySHA256":"{source_integrity}"}}'
    exit 0
    ;;
esac
exec {sys.executable} "$@"
""",
        "docker": f"""#!/bin/sh
set -eu
printf '%s\\n' "$*" >> "$FAKE_DOCKER_LOG"
case "${{1:-}}" in
  info) exit 0 ;;
  ps) exit 0 ;;
  build)
    cat >/dev/null
    exit 0
    ;;
  image)
    if [ "${{2:-}}" = inspect ]; then
      printf '{FAKE_IMAGE_DIGEST}\\n'
      exit 0
    fi
    ;;
  run) exit 0 ;;
  create)
    : > "$FAKE_DOCKER_ENVIRONMENT"
    capture_environment=0
    for value in "$@"; do
      if [ "$capture_environment" = 1 ]; then
        printf '%s\\n' "$value" >> "$FAKE_DOCKER_ENVIRONMENT"
        capture_environment=0
        continue
      fi
      case "$value" in
        -e) capture_environment=1 ;;
        ai.lumen.launch-contract-sha256=*)
          printf '%s\\n' "${{value#*=}}" > "$FAKE_DOCKER_CONTRACT"
          ;;
        ai.lumen.launch-mode=*)
          printf '%s\\n' "${{value#*=}}" > "$FAKE_DOCKER_LAUNCH_MODE"
          ;;
        ai.lumen.host-run-root=*)
          printf '%s\\n' "${{value#*=}}" > "$FAKE_DOCKER_RUN_ROOT"
          ;;
      esac
    done
    printf 'created\\n' > "$FAKE_DOCKER_STATE"
    printf '{container_id}\\n'
    if [ "${{FAKE_DISCONNECT_ON_CREATE:-0}}" = 1 ]; then
      kill -TERM "$PPID"
      exit 143
    fi
    exit 0
    ;;
  start)
    printf 'running\\n' > "$FAKE_DOCKER_STATE"
    exit 0
    ;;
  logs)
    printf 'durable training output\\n'
    exit 0
    ;;
  wait)
    printf 'exited\\n' > "$FAKE_DOCKER_STATE"
    if [ "$FAKE_TRAINING_EXIT" = 0 ]; then
      mkdir -p "$(cat "$FAKE_DOCKER_RUN_ROOT")"
    fi
    printf '%s\\n' "$FAKE_TRAINING_EXIT"
    exit 0
    ;;
  rm)
    rm -f "$FAKE_DOCKER_STATE"
    exit 0
    ;;
  container)
    if [ "${{2:-}}" != inspect ] || [ ! -f "$FAKE_DOCKER_STATE" ]; then
      exit 1
    fi
    if [ "${{3:-}}" != --format ]; then
      printf '[{{"Id":"{container_id}"}}]\\n'
      exit 0
    fi
    format="${{4:-}}"
    case "$format" in
      *source-integrity-sha256*) printf '{source_integrity}\\n' ;;
      *launch-contract-sha256*) cat "$FAKE_DOCKER_CONTRACT" ;;
      *ai.lumen.launch-mode*) cat "$FAKE_DOCKER_LAUNCH_MODE" ;;
      *container-run-root*) printf '/outputs/%s\\n' "$(basename "$(cat "$FAKE_DOCKER_RUN_ROOT")")" ;;
      *host-run-root-identity*) stat -c '%d:%i:%u:%g:0%a' "$(cat "$FAKE_DOCKER_RUN_ROOT")" ;;
      *host-lock-identity*) stat -c '%d:%i:%u:%g:0%a' "$(dirname "$(cat "$FAKE_DOCKER_RUN_ROOT")")/.lumen-training.lock" ;;
      *host-lock-dir*) printf '%s/.lumen-training.lock\\n' "$(dirname "$(cat "$FAKE_DOCKER_RUN_ROOT")")" ;;
      *host-run-root*) cat "$FAKE_DOCKER_RUN_ROOT" ;;
      *'.Mounts'*)
        run_root="$(cat "$FAKE_DOCKER_RUN_ROOT")"
        printf '[{{"Type":"bind","Source":"%s","Destination":"/outputs/%s","Mode":"rw","RW":true,"Propagation":"rprivate"}},{{"Type":"bind","Source":"%s/.lumen-training.lock","Destination":"/run/lumen-training-lock","Mode":"ro","RW":false,"Propagation":"rprivate"}},{{"Type":"bind","Source":"%s/hub","Destination":"/cache/huggingface/hub","Mode":"rw","RW":true,"Propagation":"rprivate"}},{{"Type":"bind","Source":"%s/xet","Destination":"/cache/huggingface/xet","Mode":"rw","RW":true,"Propagation":"rprivate"}},{{"Type":"bind","Source":"%s/assets","Destination":"/cache/huggingface/assets","Mode":"rw","RW":true,"Propagation":"rprivate"}}]\\n' "$run_root" "$(basename "$run_root")" "$(dirname "$run_root")" "$FAKE_HF_CACHE" "$FAKE_HF_CACHE" "$FAKE_HF_CACHE"
        ;;
      *Config.Entrypoint*) printf '["/bin/bash"]\\n' ;;
      *Config.Cmd*) printf '["/opt/lumen/source/scripts/ubuntu_train_lumen_adapters_aio.sh"]\\n' ;;
      *Config.Env*) {sys.executable} -c 'import json,sys; print(json.dumps(open(sys.argv[1], encoding="utf-8").read().splitlines()))' "$FAKE_DOCKER_ENVIRONMENT" ;;
      *Config.User*) printf '12345:23456\\n' ;;
      *HostConfig.AutoRemove*) printf 'false\\n' ;;
      *HostConfig.Init*) printf 'true\\n' ;;
      *.State.Status*) cat "$FAKE_DOCKER_STATE" ;;
      *.State.ExitCode*) printf '%s\\n' "$FAKE_TRAINING_EXIT" ;;
      *.State.OOMKilled*) printf 'false\\n' ;;
      *.State.Error*) printf '\\n' ;;
      *.Image*) printf '{FAKE_IMAGE_DIGEST}\\n' ;;
      *.Id*) printf '{container_id}\\n' ;;
      *) exit 2 ;;
    esac
    exit 0
    ;;
esac
exit 2
""",
    }
    for name, source in fake_commands.items():
        path = fake_bin / name
        path.write_text(source, encoding="utf-8")
        path.chmod(0o755)

    output_root = tmp_path / "outputs"
    result = subprocess.run(
        [
            "bash",
            "scripts/ubuntu_train_lumen_full_pipeline.sh",
            "--prepare-only",
            "--no-pull",
            "--run-id",
            "durable-probe",
            "--output-dir",
            str(output_root),
            "--hf-cache",
            str(tmp_path / "hf-cache"),
        ],
        cwd=REPO_ROOT,
        env={
            **os.environ,
            "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
            "FAKE_DOCKER_LOG": str(docker_log),
            "FAKE_DOCKER_STATE": str(docker_state),
            "FAKE_DOCKER_CONTRACT": str(docker_contract),
            "FAKE_DOCKER_ENVIRONMENT": str(docker_environment),
            "FAKE_DOCKER_LAUNCH_MODE": str(docker_launch_mode),
            "FAKE_DOCKER_RUN_ROOT": str(docker_run_root),
            "FAKE_HF_CACHE": str(tmp_path / "hf-cache"),
            "FAKE_TRAINING_EXIT": str(training_exit_code),
        },
        text=True,
        capture_output=True,
        check=False,
    )

    assert (result.returncode == 0) is (training_exit_code == 0)
    commands = docker_log.read_text(encoding="utf-8").splitlines()
    create_command = next(line for line in commands if line.startswith("create "))
    assert "--rm" not in create_command
    assert "--name lumen-ubuntu-" in create_command
    assert f"-v {output_root}:/outputs:rw" not in create_command
    assert (
        f"-v {output_root}/durable-probe-{OPTIMIZED_VARIANT}:"
        f"/outputs/durable-probe-{OPTIMIZED_VARIANT}:rw"
    ) in create_command
    assert (
        f"-v {output_root}/.lumen-training.lock:"
        "/run/lumen-training-lock:ro"
    ) in create_command
    assert any("{{json .Mounts}}" in line for line in commands)
    assert any(line.startswith("start lumen-ubuntu-") for line in commands)
    assert any(
        line.startswith("logs --timestamps --follow lumen-ubuntu-")
        for line in commands
    )
    assert any(line.startswith("wait lumen-ubuntu-") for line in commands)
    evidence = output_root / ".lumen-container-evidence"
    assert "it remains running if this launcher disconnects" in result.stdout
    if training_exit_code == 0:
        assert any(line.startswith("rm lumen-ubuntu-") for line in commands)
        assert not docker_state.exists()
        assert list(evidence.rglob("success-*.docker-inspect.json"))
        assert list(evidence.rglob("success-*.docker.log"))
    else:
        assert not any(line.startswith("rm lumen-ubuntu-") for line in commands)
        assert docker_state.exists()
        assert list(evidence.rglob("failure-*.docker-inspect.json"))
        assert list(evidence.rglob("failure-*.docker.log"))
        assert "exited 137" in result.stderr
        assert "was retained as lumen-ubuntu-" in result.stderr


def _retained_container_launcher_harness(tmp_path: Path) -> dict[str, Any]:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    paths = {
        "log": tmp_path / "docker.log",
        "state": tmp_path / "docker.state",
        "contract": tmp_path / "docker.contract",
        "environment": tmp_path / "docker.environment",
        "launch_mode": tmp_path / "docker.launch-mode",
        "run_root": tmp_path / "docker.run-root",
        "name": tmp_path / "docker.name",
    }
    repository_head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        text=True,
    ).strip()
    container_id = "7" * 64
    source_integrity = "d" * 64
    commands = {
        "uname": """#!/bin/sh
printf 'Linux\\n'
""",
        "id": """#!/bin/sh
case "${1:-}" in
  -u) printf '12345\\n' ;;
  -g) printf '23456\\n' ;;
  *) exit 2 ;;
esac
""",
        "nvidia-smi": """#!/bin/sh
exit 0
""",
        "python3": f"""#!/bin/sh
case "$*" in
  *'ubuntu_source_integrity.py attest-host'*)
    printf '%s\\n' '{{"baseCommit":"{repository_head}","workingTreeDigest":"{'b' * 64}","ubuntuOrchestrationCodeSHA256":"{'c' * 64}","sourceIntegritySHA256":"{source_integrity}"}}'
    exit 0
    ;;
esac
exec {sys.executable} "$@"
""",
        "docker": f"""#!/bin/sh
set -eu
printf '%s\\n' "$*" >> "$FAKE_DOCKER_LOG"
case "${{1:-}}" in
  info) exit 0 ;;
  ps)
    if [ -f "$FAKE_DOCKER_STATE" ] \
      && [ "$(cat "$FAKE_DOCKER_STATE")" = running ]; then
      printf '{container_id}\\n'
    fi
    exit 0
    ;;
  build) cat >/dev/null; exit 0 ;;
  image)
    [ "${{2:-}}" = inspect ] || exit 2
    printf '{FAKE_IMAGE_DIGEST}\\n'
    exit 0
    ;;
  run) exit 0 ;;
  create)
    printf '%s\\n' "${{3:-}}" > "$FAKE_DOCKER_NAME"
    : > "$FAKE_DOCKER_ENVIRONMENT"
    capture_environment=0
    for value in "$@"; do
      if [ "$capture_environment" = 1 ]; then
        printf '%s\\n' "$value" >> "$FAKE_DOCKER_ENVIRONMENT"
        capture_environment=0
        continue
      fi
      case "$value" in
        -e) capture_environment=1 ;;
        ai.lumen.launch-contract-sha256=*) printf '%s\\n' "${{value#*=}}" > "$FAKE_DOCKER_CONTRACT" ;;
        ai.lumen.launch-mode=*) printf '%s\\n' "${{value#*=}}" > "$FAKE_DOCKER_LAUNCH_MODE" ;;
        ai.lumen.host-run-root=*) printf '%s\\n' "${{value#*=}}" > "$FAKE_DOCKER_RUN_ROOT" ;;
      esac
    done
    printf 'created\\n' > "$FAKE_DOCKER_STATE"
    printf '{container_id}\\n'
    if [ "${{FAKE_DISCONNECT_ON_CREATE:-0}}" = 1 ]; then
      kill -TERM "$PPID"
      exit 143
    fi
    exit 0
    ;;
  start) printf 'running\\n' > "$FAKE_DOCKER_STATE"; exit 0 ;;
  logs)
    printf 'retained training output\\n'
    if [ "${{FAKE_DISCONNECT_ON_FOLLOW:-0}}" = 1 ]; then
      case "$*" in
        *--follow*) kill -TERM "$PPID"; exit 143 ;;
      esac
    fi
    exit 0
    ;;
  wait)
    printf 'exited\\n' > "$FAKE_DOCKER_STATE"
    printf '0\\n'
    exit 0
    ;;
  rm) rm -f "$FAKE_DOCKER_STATE"; exit 0 ;;
  container)
    if [ "${{2:-}}" != inspect ] || [ ! -f "$FAKE_DOCKER_STATE" ]; then
      exit 1
    fi
    if [ "${{3:-}}" != --format ]; then
      printf '[{{"Id":"{container_id}","Name":"/%s","State":{{"Status":"%s"}},"Config":{{"Labels":{{"ai.lumen.purpose":"ubuntu-training","ai.lumen.host-run-root":"%s"}}}}}}]\\n' \
        "$(cat "$FAKE_DOCKER_NAME")" \
        "$(cat "$FAKE_DOCKER_STATE")" \
        "$(cat "$FAKE_DOCKER_RUN_ROOT")"
      exit 0
    fi
    format="${{4:-}}"
    case "$format" in
      *source-integrity-sha256*) printf '{source_integrity}\\n' ;;
      *launch-contract-sha256*) cat "$FAKE_DOCKER_CONTRACT" ;;
      *ai.lumen.launch-mode*) cat "$FAKE_DOCKER_LAUNCH_MODE" ;;
      *container-run-root*) printf '/outputs/%s\\n' "$(basename "$(cat "$FAKE_DOCKER_RUN_ROOT")")" ;;
      *host-run-root-identity*) stat -c '%d:%i:%u:%g:0%a' "$(cat "$FAKE_DOCKER_RUN_ROOT")" ;;
      *host-lock-identity*) stat -c '%d:%i:%u:%g:0%a' "$(dirname "$(cat "$FAKE_DOCKER_RUN_ROOT")")/.lumen-training.lock" ;;
      *host-lock-dir*) printf '%s/.lumen-training.lock\\n' "$(dirname "$(cat "$FAKE_DOCKER_RUN_ROOT")")" ;;
      *host-run-root*) cat "$FAKE_DOCKER_RUN_ROOT" ;;
      *'.Mounts'*)
        run_root="$(cat "$FAKE_DOCKER_RUN_ROOT")"
        printf '[{{"Type":"bind","Source":"%s","Destination":"/outputs/%s","Mode":"rw","RW":true,"Propagation":"rprivate"}},{{"Type":"bind","Source":"%s/.lumen-training.lock","Destination":"/run/lumen-training-lock","Mode":"ro","RW":false,"Propagation":"rprivate"}},{{"Type":"bind","Source":"%s/hub","Destination":"/cache/huggingface/hub","Mode":"rw","RW":true,"Propagation":"rprivate"}},{{"Type":"bind","Source":"%s/xet","Destination":"/cache/huggingface/xet","Mode":"rw","RW":true,"Propagation":"rprivate"}},{{"Type":"bind","Source":"%s/assets","Destination":"/cache/huggingface/assets","Mode":"rw","RW":true,"Propagation":"rprivate"}}]\\n' "$run_root" "$(basename "$run_root")" "$(dirname "$run_root")" "$FAKE_HF_CACHE" "$FAKE_HF_CACHE" "$FAKE_HF_CACHE"
        ;;
      *Config.Entrypoint*) printf '["/bin/bash"]\\n' ;;
      *Config.Cmd*) printf '["/opt/lumen/source/scripts/ubuntu_train_lumen_adapters_aio.sh"]\\n' ;;
      *Config.Env*) {sys.executable} -c 'import json,sys; print(json.dumps(open(sys.argv[1], encoding="utf-8").read().splitlines()))' "$FAKE_DOCKER_ENVIRONMENT" ;;
      *Config.User*) printf '12345:23456\\n' ;;
      *HostConfig.AutoRemove*) printf 'false\\n' ;;
      *HostConfig.Init*) printf 'true\\n' ;;
      *.State.Status*) cat "$FAKE_DOCKER_STATE" ;;
      *.State.ExitCode*) printf '0\\n' ;;
      *.State.OOMKilled*) printf 'false\\n' ;;
      *.State.Error*) printf '\\n' ;;
      *.Image*) printf '{FAKE_IMAGE_DIGEST}\\n' ;;
      *.Id*) printf '{container_id}\\n' ;;
      *) exit 2 ;;
    esac
    exit 0
    ;;
esac
exit 2
""",
    }
    for name, source in commands.items():
        executable = fake_bin / name
        executable.write_text(source, encoding="utf-8")
        executable.chmod(0o755)
    return {
        "paths": paths,
        "env": {
            **os.environ,
            "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
            "FAKE_DOCKER_LOG": str(paths["log"]),
            "FAKE_DOCKER_STATE": str(paths["state"]),
            "FAKE_DOCKER_CONTRACT": str(paths["contract"]),
            "FAKE_DOCKER_ENVIRONMENT": str(paths["environment"]),
            "FAKE_DOCKER_LAUNCH_MODE": str(paths["launch_mode"]),
            "FAKE_DOCKER_RUN_ROOT": str(paths["run_root"]),
            "FAKE_DOCKER_NAME": str(paths["name"]),
            "FAKE_HF_CACHE": str(tmp_path / "hf-cache"),
        },
        "output_root": tmp_path / "outputs",
        "hf_cache": tmp_path / "hf-cache",
    }


def test_running_unrelated_durable_container_blocks_a_fresh_launch(
    tmp_path: Path,
) -> None:
    if sys.platform != "linux":
        pytest.skip("launcher harness requires the Ubuntu/GNU host utilities")
    harness = _retained_container_launcher_harness(tmp_path)
    harness["paths"]["state"].write_text("running\n", encoding="utf-8")
    harness["paths"]["name"].write_text(
        f"lumen-ubuntu-{'a' * 24}\n",
        encoding="utf-8",
    )
    unrelated_run_root = tmp_path / "unrelated-run"
    unrelated_run_root.mkdir()
    harness["paths"]["run_root"].write_text(
        f"{unrelated_run_root}\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            "bash",
            "scripts/ubuntu_train_lumen_full_pipeline.sh",
            "--prepare-only",
            "--no-pull",
            "--run-id",
            "blocked-fresh",
            "--output-dir",
            str(harness["output_root"]),
            "--hf-cache",
            str(harness["hf_cache"]),
        ],
        cwd=REPO_ROOT,
        env=harness["env"],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "permits only an exact --resume" in result.stderr
    commands = harness["paths"]["log"].read_text(encoding="utf-8").splitlines()
    assert not any(line.startswith("build ") for line in commands)
    assert not any(line.startswith("create ") for line in commands)
    assert not any(line.startswith("start ") for line in commands)


@pytest.mark.parametrize("initial_mode", ("fresh", "overwrite"))
def test_running_retained_container_can_be_safely_reattached_by_resume(
    tmp_path: Path,
    initial_mode: str,
) -> None:
    if sys.platform != "linux":
        pytest.skip("launcher harness requires the Ubuntu/GNU host utilities")
    harness = _retained_container_launcher_harness(tmp_path)
    base_args = [
        "bash",
        "scripts/ubuntu_train_lumen_full_pipeline.sh",
        "--prepare-only",
        "--no-pull",
        "--run-id",
        "retained-reattach",
        "--output-dir",
        str(harness["output_root"]),
        "--hf-cache",
        str(harness["hf_cache"]),
    ]
    first_args = [*base_args, *( ["--overwrite"] if initial_mode == "overwrite" else [] )]
    first = subprocess.run(
        first_args,
        cwd=REPO_ROOT,
        env={**harness["env"], "FAKE_DISCONNECT_ON_FOLLOW": "1"},
        text=True,
        capture_output=True,
        check=False,
    )
    assert first.returncode != 0
    assert harness["paths"]["state"].read_text(encoding="utf-8").strip() == "running"
    assert harness["paths"]["launch_mode"].read_text(encoding="utf-8").strip() == initial_mode

    second = subprocess.run(
        [*base_args, "--resume"],
        cwd=REPO_ROOT,
        env={**harness["env"], "FAKE_DISCONNECT_ON_FOLLOW": "0"},
        text=True,
        capture_output=True,
        check=False,
    )
    assert second.returncode == 0, second.stderr
    assert "reattaching to durable training container" in second.stdout
    commands = harness["paths"]["log"].read_text(encoding="utf-8").splitlines()
    assert sum(line.startswith("create ") for line in commands) == 1


def test_fresh_created_retained_container_is_started_by_explicit_resume(
    tmp_path: Path,
) -> None:
    if sys.platform != "linux":
        pytest.skip("launcher harness requires the Ubuntu/GNU host utilities")
    harness = _retained_container_launcher_harness(tmp_path)
    base_args = [
        "bash",
        "scripts/ubuntu_train_lumen_full_pipeline.sh",
        "--prepare-only",
        "--no-pull",
        "--run-id",
        "retained-created",
        "--output-dir",
        str(harness["output_root"]),
        "--hf-cache",
        str(harness["hf_cache"]),
    ]
    first = subprocess.run(
        base_args,
        cwd=REPO_ROOT,
        env={**harness["env"], "FAKE_DISCONNECT_ON_CREATE": "1"},
        text=True,
        capture_output=True,
        check=False,
    )
    assert first.returncode != 0
    assert harness["paths"]["state"].read_text(encoding="utf-8").strip() == "created"

    second = subprocess.run(
        [*base_args, "--resume"],
        cwd=REPO_ROOT,
        env={**harness["env"], "FAKE_DISCONNECT_ON_CREATE": "0"},
        text=True,
        capture_output=True,
        check=False,
    )
    assert second.returncode == 0, second.stderr
    assert "never-started fresh container" in second.stdout
    commands = harness["paths"]["log"].read_text(encoding="utf-8").splitlines()
    assert sum(line.startswith("start ") for line in commands) == 1
    assert sum(line.startswith("create ") for line in commands) == 1


def test_overwrite_created_container_requires_repeated_overwrite_authorization(
    tmp_path: Path,
) -> None:
    if sys.platform != "linux":
        pytest.skip("launcher harness requires the Ubuntu/GNU host utilities")
    harness = _retained_container_launcher_harness(tmp_path)
    base_args = [
        "bash",
        "scripts/ubuntu_train_lumen_full_pipeline.sh",
        "--prepare-only",
        "--no-pull",
        "--run-id",
        "retained-overwrite-created",
        "--output-dir",
        str(harness["output_root"]),
        "--hf-cache",
        str(harness["hf_cache"]),
    ]
    first = subprocess.run(
        [*base_args, "--overwrite"],
        cwd=REPO_ROOT,
        env={**harness["env"], "FAKE_DISCONNECT_ON_CREATE": "1"},
        text=True,
        capture_output=True,
        check=False,
    )
    assert first.returncode != 0
    assert harness["paths"]["state"].read_text(encoding="utf-8").strip() == "created"

    resume = subprocess.run(
        [*base_args, "--resume"],
        cwd=REPO_ROOT,
        env={**harness["env"], "FAKE_DISCONNECT_ON_CREATE": "0"},
        text=True,
        capture_output=True,
        check=False,
    )
    assert resume.returncode != 0
    assert "requires --overwrite" in resume.stderr
    commands = harness["paths"]["log"].read_text(encoding="utf-8").splitlines()
    assert not any(line.startswith("start ") for line in commands)

    overwrite = subprocess.run(
        [*base_args, "--overwrite"],
        cwd=REPO_ROOT,
        env={**harness["env"], "FAKE_DISCONNECT_ON_CREATE": "0"},
        text=True,
        capture_output=True,
        check=False,
    )
    assert overwrite.returncode == 0, overwrite.stderr
    commands = harness["paths"]["log"].read_text(encoding="utf-8").splitlines()
    assert sum(line.startswith("create ") for line in commands) == 1
    assert sum(line.startswith("start ") for line in commands) == 1


def test_running_retained_container_rejects_duplicate_state_environment(
    tmp_path: Path,
) -> None:
    if sys.platform != "linux":
        pytest.skip("launcher harness requires the Ubuntu/GNU host utilities")
    harness = _retained_container_launcher_harness(tmp_path)
    base_args = [
        "bash",
        "scripts/ubuntu_train_lumen_full_pipeline.sh",
        "--prepare-only",
        "--no-pull",
        "--run-id",
        "retained-env-drift",
        "--output-dir",
        str(harness["output_root"]),
        "--hf-cache",
        str(harness["hf_cache"]),
    ]
    first = subprocess.run(
        base_args,
        cwd=REPO_ROOT,
        env={**harness["env"], "FAKE_DISCONNECT_ON_FOLLOW": "1"},
        text=True,
        capture_output=True,
        check=False,
    )
    assert first.returncode != 0
    with harness["paths"]["environment"].open("a", encoding="utf-8") as handle:
        handle.write("LUMEN_AIO_RESUME=1\n")

    second = subprocess.run(
        [*base_args, "--resume"],
        cwd=REPO_ROOT,
        env={**harness["env"], "FAKE_DISCONNECT_ON_FOLLOW": "0"},
        text=True,
        capture_output=True,
        check=False,
    )
    assert second.returncode != 0
    assert "malformed or duplicate observed environment" in second.stderr
    assert harness["paths"]["state"].read_text(encoding="utf-8").strip() == "running"


def test_unsloth_import_guard_fails_closed_if_transformers_loaded_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tools.fine_tuning.unsloth.train_sft import (
        _require_unsloth_before_transformers,
    )

    monkeypatch.delitem(sys.modules, "unsloth", raising=False)
    monkeypatch.setitem(sys.modules, "transformers", ModuleType("transformers"))
    with pytest.raises(RuntimeError, match="before Transformers"):
        _require_unsloth_before_transformers()

    monkeypatch.setitem(sys.modules, "unsloth", ModuleType("unsloth"))
    _require_unsloth_before_transformers()


def test_ubuntu_trainers_import_unsloth_before_transformers_seeding() -> None:
    for filename in ("train_sft.py", "train_dpo.py"):
        source = (
            REPO_ROOT / "tools/fine_tuning/unsloth" / filename
        ).read_text(encoding="utf-8")
        main_source = source[source.index("def main() -> None:") :]
        assert main_source.index("from unsloth import FastLanguageModel") < (
            main_source.index("_seed_everything(seed)")
        )
        if filename == "train_dpo.py":
            assert main_source.index(
                "_latch_expandable_cuda_allocator()"
            ) < main_source.index("from unsloth import FastLanguageModel")

    evaluator = (
        REPO_ROOT / "tools/fine_tuning/unsloth/evaluate_adapter.py"
    ).read_text(encoding="utf-8")
    loader = evaluator[evaluator.index("def load_inference_model(") :]
    assert loader.index("from unsloth import FastLanguageModel") < loader.index(
        "_seed_everything(int(cfg[\"seed\"]))"
    )

    launcher = (
        REPO_ROOT / "scripts/ubuntu_train_lumen_full_pipeline.sh"
    ).read_text(encoding="utf-8")
    assert "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" in launcher
    assert "PYTORCH_ALLOC_CONF" not in launcher


@pytest.mark.parametrize(
    "filename",
    (
        "train_sft.py",
        "train_dpo.py",
        "evaluate_adapter.py",
        "export_gguf.py",
    ),
)
def test_controlled_unsloth_loads_disable_model_name_redirects(
    filename: str,
) -> None:
    source = (
        REPO_ROOT / "tools/fine_tuning/unsloth" / filename
    ).read_text(encoding="utf-8")
    tree = ast.parse(source, filename=filename)
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "from_pretrained"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id in {"FastLanguageModel", "fast_language_model"}
    ]

    assert calls, f"{filename} must load its controlled base model with Unsloth"
    for call in calls:
        exact_name = next(
            (
                keyword.value
                for keyword in call.keywords
                if keyword.arg == "use_exact_model_name"
            ),
            None,
        )
        assert isinstance(exact_name, ast.Constant)
        assert exact_name.value is True


def test_agent_and_run_root_validation_fails_closed(tmp_path: Path) -> None:
    assert ubuntu_pipeline.parse_agents("cortex,executor") == ("cortex", "executor")
    with pytest.raises(RuntimeError, match="duplicates"):
        ubuntu_pipeline.parse_agents("cortex,cortex")
    with pytest.raises(RuntimeError, match="Unsupported agents"):
        ubuntu_pipeline.parse_agents("cortex,unknown")
    with pytest.raises(RuntimeError, match="must be a child"):
        ubuntu_pipeline.validate_run_root(tmp_path, allowed_parent=tmp_path)
    child = tmp_path / "run-one"
    assert ubuntu_pipeline.validate_run_root(child, allowed_parent=tmp_path) == child


def test_exact_bind_root_is_private_durable_and_identity_bound(tmp_path: Path) -> None:
    output_root = tmp_path / "outputs"
    output_root.mkdir(mode=0o700)
    run_root = output_root / "pilot-internal_plus_public_optimized"

    initialized = ubuntu_pipeline.initialize_bind_root(
        run_root,
        allowed_parent=output_root,
        create_if_missing=True,
    )
    identity = initialized["rootIdentity"]
    inode = run_root.stat(follow_symlinks=False).st_ino

    assert initialized["created"] is True
    assert run_root.stat().st_mode & 0o777 == 0o700
    assert ubuntu_pipeline.verify_bind_root(
        run_root,
        allowed_parent=output_root,
        expected_identity=identity,
    )["status"] == "bind_root_identity_verified"

    displaced = output_root / "displaced"
    run_root.rename(displaced)
    run_root.mkdir(mode=0o700)
    with pytest.raises(RuntimeError, match="device/inode/ownership/mode changed"):
        ubuntu_pipeline.verify_bind_root(
            run_root,
            allowed_parent=output_root,
            expected_identity=identity,
        )
    assert displaced.stat().st_ino == inode


def test_precreated_bind_root_rejects_unowned_existing_contents(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "outputs"
    output_root.mkdir(mode=0o700)
    run_root = output_root / "malicious-internal_plus_public_optimized"
    ubuntu_pipeline.initialize_bind_root(
        run_root,
        allowed_parent=output_root,
        create_if_missing=True,
    )
    malicious = run_root / "unowned-payload"
    malicious.write_text("do not delete me", encoding="utf-8")

    with pytest.raises(RuntimeError, match="unexpected state without an ownership record"):
        ubuntu_pipeline.static_preflight(
            root=REPO_ROOT,
            dataset_source=REPO_ROOT / "generated" / "fine_tuning",
            agents=("cortex",),
            variant=OPTIMIZED_VARIANT,
            seed=42,
            base_model_override="",
            container_digest=FAKE_IMAGE_DIGEST,
            run_root=run_root,
            allowed_parent=output_root,
            expected_run_id=run_root.name,
            evaluation_scope="smoke",
            evaluation_max_examples=7,
            gguf_requested=False,
            precreated_bind_root=True,
        )
    assert malicious.read_text(encoding="utf-8") == "do not delete me"


def test_precreated_partial_preparation_recovery_keeps_bind_inode_and_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_root = tmp_path / "outputs"
    output_root.mkdir(mode=0o700)
    lock_root = output_root / ".lumen-training.lock"
    lock_root.mkdir(mode=0o700)
    run_root = output_root / "partial-internal_plus_public_optimized"
    ubuntu_pipeline.initialize_bind_root(
        run_root,
        allowed_parent=output_root,
        create_if_missing=True,
    )
    original_inode = run_root.stat(follow_symlinks=False).st_ino
    source_integrity = _source_integrity_fixture()["ubuntuSourceIntegrity"]
    monkeypatch.setattr(
        ubuntu_pipeline,
        "current_source_integrity",
        lambda *_args, **_kwargs: source_integrity,
    )
    plan = _test_execution_plan(
        evaluation_scope="smoke",
        evaluation_max_examples=7,
        gguf_requested=False,
    )
    owner = ubuntu_pipeline._preparation_owner_record(
        root=REPO_ROOT,
        dataset_source=REPO_ROOT / "generated" / "fine_tuning",
        run_root=run_root,
        agents=("cortex",),
        variant=OPTIMIZED_VARIANT,
        seed=42,
        base_model_override="",
        container_digest=FAKE_IMAGE_DIGEST,
        prepared_execution_plan=plan,
        source_integrity=source_integrity,
        precreated_bind_root=True,
    )
    ubuntu_pipeline._initialize_preparation_root(
        run_root,
        owner,
        precreated_bind_root=True,
    )
    partial = run_root / "generated" / "fine_tuning"
    partial.mkdir(parents=True, mode=0o700)
    (partial / "partial.json").write_text("{}\n", encoding="utf-8")

    result = ubuntu_pipeline.recover_incomplete_preparation(
        root=REPO_ROOT,
        dataset_source=REPO_ROOT / "generated" / "fine_tuning",
        run_root=run_root,
        allowed_parent=output_root,
        agents=("cortex",),
        variant=OPTIMIZED_VARIANT,
        seed=42,
        base_model_override="",
        container_digest=FAKE_IMAGE_DIGEST,
        evaluation_scope="smoke",
        evaluation_max_examples=7,
        gguf_requested=False,
        precreated_bind_root=True,
    )

    assert result["status"] == "incomplete_preparation_cleared"
    assert run_root.stat(follow_symlinks=False).st_ino == original_inode
    assert not list(run_root.iterdir())
    assert lock_root.is_dir()


def test_owned_bind_reset_verifies_before_clearing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_root = tmp_path / "owned-bind-root"
    run_root.mkdir(mode=0o700)
    payload = run_root / "payload"
    payload.write_text("artifact", encoding="utf-8")
    original_inode = run_root.stat(follow_symlinks=False).st_ino

    monkeypatch.setattr(
        ubuntu_pipeline,
        "verify_owned_run",
        lambda *_args, **_kwargs: {
            "runManifestSHA256": "a" * 64,
            "runRootInitializationMode": "precreated_bind_root",
        },
    )
    ubuntu_pipeline.reset_owned_run_root(run_root, variant=OPTIMIZED_VARIANT)
    assert run_root.stat(follow_symlinks=False).st_ino == original_inode
    assert not list(run_root.iterdir())

    payload.write_text("must-survive", encoding="utf-8")

    def reject_ownership(*_args: object, **_kwargs: object) -> dict[str, object]:
        raise RuntimeError("ownership rejected")

    monkeypatch.setattr(ubuntu_pipeline, "verify_owned_run", reject_ownership)
    with pytest.raises(RuntimeError, match="ownership rejected"):
        ubuntu_pipeline.reset_owned_run_root(run_root, variant=OPTIMIZED_VARIANT)
    assert payload.read_text(encoding="utf-8") == "must-survive"


def test_current_optimized_artifacts_pass_static_preflight(tmp_path: Path) -> None:
    result = ubuntu_pipeline.static_preflight(
        root=REPO_ROOT,
        dataset_source=REPO_ROOT / "generated" / "fine_tuning",
        agents=ubuntu_pipeline.AGENTS,
        variant="internal_plus_public_optimized",
        seed=42,
        base_model_override="",
        container_digest=FAKE_IMAGE_DIGEST,
        run_root=tmp_path / "run-one",
        allowed_parent=tmp_path,
    )

    assert result["status"] == "static_ready"
    assert result["trainingReady"] is False
    assert result["executionPlan"] == _test_execution_plan()
    assert [entry["agent"] for entry in result["agents"]] == list(
        ubuntu_pipeline.AGENTS
    )
    assert all(
        entry["evaluationPromptPreflight"]["caseCount"] > 0
        and len(
            entry["evaluationPromptPreflight"][
                "evaluationPromptPreflightSHA256"
            ]
        )
        == 64
        for entry in result["agents"]
    )
    assert not (tmp_path / "run-one").exists()


def test_static_preflight_rejects_evaluator_prompt_contract_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_loader = evaluate_adapter.load_evaluation_records

    def load_drifted_prompts(*args: object, **kwargs: object) -> tuple[list, str]:
        records, evaluation_sha256 = original_loader(*args, **kwargs)
        drifted = json.loads(json.dumps(records))
        system_message = drifted[0]["messages"][0]
        system_message["content"] = system_message["content"].replace(
            "Response format contract:",
            "Response-format contract:",
            1,
        )
        return drifted, evaluation_sha256

    monkeypatch.setattr(
        evaluate_adapter,
        "load_evaluation_records",
        load_drifted_prompts,
    )

    with pytest.raises(
        ValueError,
        match=r"fleet/.*drifted structured-output contract",
    ):
        ubuntu_pipeline.static_preflight(
            root=REPO_ROOT,
            dataset_source=REPO_ROOT / "generated" / "fine_tuning",
            agents=("fleet",),
            variant=OPTIMIZED_VARIANT,
            seed=42,
            base_model_override="",
            container_digest=FAKE_IMAGE_DIGEST,
            run_root=tmp_path / "run-prompt-drift",
            allowed_parent=tmp_path,
        )
    assert not (tmp_path / "run-prompt-drift").exists()


def test_static_preflight_rejects_smoke_size_covering_a_frozen_suite(
    tmp_path: Path,
) -> None:
    dataset_source = REPO_ROOT / "generated" / "fine_tuning"
    frozen_case_count = len(
        ubuntu_pipeline.read_jsonl(dataset_source / "mimicry" / "eval.jsonl")
    )

    with pytest.raises(RuntimeError, match="must be smaller.*mimicry"):
        ubuntu_pipeline.static_preflight(
            root=REPO_ROOT,
            dataset_source=dataset_source,
            agents=("mimicry",),
            variant=OPTIMIZED_VARIANT,
            seed=42,
            base_model_override="",
            container_digest=FAKE_IMAGE_DIGEST,
            run_root=tmp_path / "run-smoke",
            allowed_parent=tmp_path,
            evaluation_scope="smoke",
            evaluation_max_examples=frozen_case_count,
            gguf_requested=False,
        )
    assert not (tmp_path / "run-smoke").exists()


def test_variant_validation_rejects_a_missing_contamination_report(
    tmp_path: Path,
) -> None:
    source_root = _copy_cortex_variant_source(tmp_path)
    report_path = (
        source_root
        / "cortex"
        / "experiments"
        / OPTIMIZED_VARIANT
        / "contamination_report.json"
    )
    report_path.unlink()

    with pytest.raises(RuntimeError, match="Missing regular controlled contamination"):
        _validate_copied_cortex_variant(source_root)


def test_variant_validation_rejects_an_invalid_contamination_report(
    tmp_path: Path,
) -> None:
    source_root = _copy_cortex_variant_source(tmp_path)
    report_path = (
        source_root
        / "cortex"
        / "experiments"
        / OPTIMIZED_VARIANT
        / "contamination_report.json"
    )
    ubuntu_pipeline.write_object(report_path, {"contaminated": False})

    with pytest.raises(RuntimeError, match="integrity check failed"):
        _validate_copied_cortex_variant(source_root)


def test_variant_validation_rejects_contamination_manifest_mismatch(
    tmp_path: Path,
) -> None:
    source_root = _copy_cortex_variant_source(tmp_path)
    report_path = (
        source_root
        / "cortex"
        / "experiments"
        / OPTIMIZED_VARIANT
        / "contamination_report.json"
    )
    report = ubuntu_pipeline.read_object(report_path)
    report["publicEvaluationRowCount"] += 1
    ubuntu_pipeline.write_object(report_path, report)
    _rehash_contamination_report(report_path)

    with pytest.raises(RuntimeError, match="not bound to its variant manifest"):
        _validate_copied_cortex_variant(source_root)


@pytest.mark.parametrize(
    ("binding", "error_pattern"),
    (
        ("training_count", "training-dataset binding mismatch"),
        ("evaluation_count", "evaluation-dataset binding mismatch"),
        ("evaluation_hash", "evaluation-dataset binding mismatch"),
    ),
)
def test_variant_validation_rejects_contamination_dataset_binding_drift(
    tmp_path: Path,
    binding: str,
    error_pattern: str,
) -> None:
    source_root = _copy_cortex_variant_source(tmp_path)
    variant_root = (
        source_root / "cortex" / "experiments" / OPTIMIZED_VARIANT
    )
    report_path = variant_root / "contamination_report.json"
    report = ubuntu_pipeline.read_object(report_path)
    if binding == "training_count":
        report["trainingRecordCount"] += 1
    elif binding == "evaluation_count":
        report["evaluationRecordCount"] += 1
    else:
        report["evaluationRecordsSHA256"] = "f" * 64
    ubuntu_pipeline.write_object(report_path, report)
    report = _rehash_contamination_report(report_path)

    manifest_path = variant_root / "variant_manifest.json"
    manifest = ubuntu_pipeline.read_object(manifest_path)
    manifest["contamination"]["reportSHA256"] = report["reportSHA256"]
    if binding == "evaluation_hash":
        manifest["frozenEvaluationSHA256"] = report[
            "evaluationRecordsSHA256"
        ]
        manifest["contamination"]["evaluationRecordsSHA256"] = report[
            "evaluationRecordsSHA256"
        ]
    ubuntu_pipeline.write_object(manifest_path, manifest)
    _rehash_variant_manifest(manifest_path)

    with pytest.raises(RuntimeError, match=error_pattern):
        _validate_copied_cortex_variant(source_root)


def test_variant_validation_rejects_a_valid_positive_contamination_report(
    tmp_path: Path,
) -> None:
    source_root = _copy_cortex_variant_source(tmp_path)
    variant_root = (
        source_root / "cortex" / "experiments" / OPTIMIZED_VARIANT
    )
    report_path = variant_root / "contamination_report.json"
    report = ubuntu_pipeline.read_object(report_path)
    report["matches"] = [
        {
            "trainingRecordID": "record-" + ("a" * 24),
            "evaluationRecordID": "record-" + ("b" * 24),
            "matchKind": "exact_record",
            "similarity": 1.0,
        }
    ]
    report["matchCount"] = 1
    report["contaminated"] = True
    ubuntu_pipeline.write_object(report_path, report)
    report = _rehash_contamination_report(report_path)

    manifest_path = variant_root / "variant_manifest.json"
    manifest = ubuntu_pipeline.read_object(manifest_path)
    manifest["contamination"].update(
        {
            "contaminated": True,
            "matchCount": 1,
            "reportSHA256": report["reportSHA256"],
        }
    )
    ubuntu_pipeline.write_object(manifest_path, manifest)
    _rehash_variant_manifest(manifest_path)

    with pytest.raises(RuntimeError, match="variant is contaminated"):
        _validate_copied_cortex_variant(source_root)


def test_shell_static_preflight_has_no_run_side_effects(tmp_path: Path) -> None:
    run_id = "static-run"
    variant = "internal_plus_public_optimized"
    run_root = tmp_path / f"{run_id}-{variant}"
    environment = {
        **os.environ,
        "LUMEN_AIO_EXPERIMENT_VARIANT": variant,
        "LUMEN_AIO_CONTAINER_IMAGE_DIGEST": FAKE_IMAGE_DIGEST,
        "LUMEN_AIO_RUN_ID": run_id,
        "LUMEN_AIO_RUN_ROOT": str(run_root),
        "LUMEN_AIO_ALLOWED_RUN_PARENT": str(tmp_path),
        "LUMEN_AIO_STATIC_PREFLIGHT": "1",
    }

    result = subprocess.run(
        ["bash", "scripts/ubuntu_train_lumen_adapters_aio.sh"],
        cwd=REPO_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert '"status": "static_ready"' in result.stdout
    assert not run_root.exists()


def test_ubuntu_inner_launcher_processes_the_final_declared_agent() -> None:
    launcher = (
        REPO_ROOT / "scripts/ubuntu_train_lumen_adapters_aio.sh"
    ).read_text(encoding="utf-8")

    assert 'IFS=\',\' read -r -a AGENTS <<< "$AGENTS_CSV"' in launcher
    assert launcher.count('for agent in "${AGENTS[@]}"; do') == 1
    assert 'evaluate_agent "$agent"' in launcher
    assert 'convert_agent_gguf "$agent"' in launcher
    assert "done < <(printf '%s' \"$AGENTS_CSV\" | tr ',' '\\n')" not in launcher


def test_ubuntu_launchers_default_to_the_same_risk_first_agent_order() -> None:
    expected = "fleet,executor,mouth,rem,mimicry,cortex"
    outer = (
        REPO_ROOT / "scripts/ubuntu_train_lumen_full_pipeline.sh"
    ).read_text(encoding="utf-8")
    inner = (
        REPO_ROOT / "scripts/ubuntu_train_lumen_adapters_aio.sh"
    ).read_text(encoding="utf-8")

    assert f"${{LUMEN_UBUNTU_AGENTS:-{expected}}}" in outer
    assert f"${{LUMEN_AIO_AGENTS:-{expected}}}" in inner


def test_converter_source_preflight_runs_before_the_first_sft_phase() -> None:
    launcher = (
        REPO_ROOT / "scripts/ubuntu_train_lumen_adapters_aio.sh"
    ).read_text(encoding="utf-8")
    first_sft_loop = launcher.index('for agent in "${AGENTS[@]}"; do')

    assert launcher.index('git init "$CONVERTER_STAGING"') < first_sft_loop
    assert launcher.index('promote_converter_checkout') < first_sft_loop
    assert launcher.index(
        'git -C "$checkout" status --porcelain=v1 --untracked-files=all'
    ) < first_sft_loop
    assert launcher.index(
        '"$TRAIN_PY" "$CONVERTER" --help >/dev/null'
    ) < first_sft_loop
    assert launcher.index(
        '"$TRAIN_PY" "$GGUF_READER" --help >/dev/null'
    ) < first_sft_loop
    assert launcher.count('git init "$CONVERTER_STAGING"') == 1
    assert 'git init "$CONVERTER_REPO"' not in launcher


def test_converter_pin_is_derived_from_training_lineage() -> None:
    from tools.fine_tuning.unsloth import training_lineage

    launcher = (
        REPO_ROOT / "scripts/ubuntu_train_lumen_adapters_aio.sh"
    ).read_text(encoding="utf-8")

    assert (
        "from tools.fine_tuning.unsloth.training_lineage import "
        "DEFAULT_LLAMA_CPP_REVISION"
    ) in launcher
    assert "print(DEFAULT_LLAMA_CPP_REVISION)" in launcher
    assert training_lineage.DEFAULT_LLAMA_CPP_REVISION not in launcher


def test_prepare_binds_the_same_resolved_environment_into_config_and_attestation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tools.fine_tuning.unsloth import train_sft

    environment_sha = "e" * 64
    lineage = {
        "resolvedTrainingEnvironment": {
            "schemaVersion": "lumen.resolved-training-environment/1.0.0"
        },
        "resolvedTrainingEnvironmentSHA256": "a" * 64,
        "resolvedTrainingEnvironmentScanAudit": {"distributionCount": 1},
        "spaceConfigurationSHA256": None,
        "zeroGPUSize": None,
        "zeroGPUDurationSeconds": None,
        "observedAccelerator": {"backend": "cuda", "deviceCount": 1},
    }
    environment = {"trainingEnvironmentSHA256": environment_sha}
    monkeypatch.setattr(
        ubuntu_pipeline,
        "_runtime_lineage",
        lambda **_kwargs: (lineage, environment),
    )
    source_integrity = _source_integrity_fixture()["ubuntuSourceIntegrity"]
    monkeypatch.setattr(
        ubuntu_pipeline,
        "current_source_integrity",
        lambda *_args, **_kwargs: source_integrity,
    )
    monkeypatch.setattr(
        train_sft,
        "_training_environment",
        lambda *_args, **_kwargs: environment,
    )
    _mock_private_base_model_snapshot_preparation(tmp_path, monkeypatch)

    run_root = tmp_path / "prepared"
    ubuntu_pipeline.prepare_run(
        root=REPO_ROOT,
        dataset_source=REPO_ROOT / "generated" / "fine_tuning",
        run_root=run_root,
        agents=("cortex",),
        variant="internal_plus_public_optimized",
        seed=42,
        base_model_override="",
        container_digest=FAKE_IMAGE_DIGEST,
        evaluation_scope="smoke",
        evaluation_max_examples=7,
        gguf_requested=False,
    )
    config = json.loads(
        (run_root / "configs" / "cortex.json").read_text(encoding="utf-8")
    )
    run_manifest = ubuntu_pipeline.read_object(run_root / "aio_run_manifest.json")
    expected_plan = _test_execution_plan(
        evaluation_scope="smoke",
        evaluation_max_examples=7,
        gguf_requested=False,
    )

    assert config["trainingEnvironmentSHA256"] == environment_sha
    assert config["variantAttestation"]["trainingEnvironmentSHA256"] == environment_sha
    assert config["resolvedTrainingEnvironment"] == lineage[
        "resolvedTrainingEnvironment"
    ]
    assert "resolvedTrainingEnvironmentCacheAttestation" in config
    assert config["resolvedTrainingEnvironmentCacheAttestation"] is None
    runtime_load_contract = runtime_binding_smoke_gate._runtime_load_contract(config)
    assert (
        runtime_load_contract["preparedConfigInputs"][
            "resolvedTrainingEnvironmentCacheAttestation"
        ]
        is None
    )
    assert config["runExecutionPlan"] == expected_plan
    assert run_manifest["executionPlan"] == expected_plan
    assert not (run_root / ubuntu_pipeline.PREPARATION_OWNER_FILENAME).exists()
    assert (
        run_root / "generated" / "agent_manifest" / "AgentBehaviorManifest.json"
    ).is_file()
    for field in ubuntu_pipeline.UBUNTU_SOURCE_INTEGRITY_FIELDS:
        assert config[field] == run_manifest[field]
        assert config["variantAttestation"][field] == run_manifest[field]
    ubuntu_pipeline.verify_embedded_source_integrity(run_manifest)

    manifest_path = run_root / "aio_run_manifest.json"
    drifted_manifest = dict(run_manifest)
    drifted_manifest["executionPlan"] = _test_execution_plan()
    drifted_manifest.pop("runManifestSHA256", None)
    drifted_manifest["runManifestSHA256"] = ubuntu_pipeline.canonical_sha256(
        drifted_manifest
    )
    ubuntu_pipeline.write_object(manifest_path, drifted_manifest)
    with pytest.raises(RuntimeError, match="execution plan drifted from the config"):
        ubuntu_pipeline._verified_run_manifest(run_root)
    ubuntu_pipeline.write_object(manifest_path, run_manifest)

    config["workingTreeDigest"] = "0" * 64
    with pytest.raises(RuntimeError, match="digest fields drifted"):
        ubuntu_pipeline.verify_embedded_source_integrity(config)

    with pytest.raises(RuntimeError, match="Resume request"):
        ubuntu_pipeline.validate_prepared_runtime(
            root=REPO_ROOT,
            run_root=run_root,
            agents=("cortex",),
            variant=OPTIMIZED_VARIANT,
            container_digest=FAKE_IMAGE_DIGEST,
            evaluation_scope="full",
            evaluation_max_examples=None,
            gguf_requested=True,
        )


def test_local_runtime_lineage_clears_remote_cache_authorization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tools.fine_tuning.unsloth import train_sft

    observed_config: dict[str, Any] = {}
    lineage = {
        "resolvedTrainingEnvironment": {"schemaVersion": "test"},
        "resolvedTrainingEnvironmentSHA256": "a" * 64,
        "resolvedTrainingEnvironmentScanAudit": {"distributionCount": 1},
        "spaceConfigurationSHA256": None,
        "zeroGPUSize": None,
        "zeroGPUDurationSeconds": None,
        "observedAccelerator": {"backend": "cuda", "deviceCount": 1},
    }

    def capture_runtime_lineage(
        config: Mapping[str, Any],
        *,
        phase: str,
    ) -> dict[str, Any]:
        assert phase == "sft"
        observed_config.update(config)
        return lineage

    monkeypatch.setattr(
        train_sft,
        "_training_runtime_lineage",
        capture_runtime_lineage,
    )
    monkeypatch.setattr(
        train_sft,
        "_training_environment",
        lambda *_args, **_kwargs: {"trainingEnvironmentSHA256": "b" * 64},
    )

    ubuntu_pipeline._runtime_lineage(
        root=REPO_ROOT,
        source_config={
            "resolvedTrainingEnvironmentCacheAttestation": {
                "cacheHMACSHA256": "c" * 64
            }
        },
        container_digest=FAKE_IMAGE_DIGEST,
        source_integrity=_source_integrity_fixture()["ubuntuSourceIntegrity"],
    )

    assert "resolvedTrainingEnvironmentCacheAttestation" in observed_config
    assert observed_config["resolvedTrainingEnvironmentCacheAttestation"] is None


def test_prepared_snapshot_copy_normalizes_read_only_image_modes(
    tmp_path: Path,
) -> None:
    source = tmp_path / "read-only-source"
    nested = source / "agent" / "experiments"
    nested.mkdir(parents=True)
    source_file = nested / "train_sft.jsonl"
    source_file.write_text('{"messages": []}\n', encoding="utf-8")
    for directory in (nested, nested.parent, source):
        directory.chmod(0o555)
    source_file.chmod(0o444)

    destination = tmp_path / "private-snapshot"
    try:
        ubuntu_pipeline._copy_private_regular_tree(source, destination)

        assert destination.stat().st_mode & 0o777 == 0o700
        assert (destination / "agent").stat().st_mode & 0o777 == 0o700
        assert (
            destination / "agent" / "experiments"
        ).stat().st_mode & 0o777 == 0o700
        copied_file = destination / "agent" / "experiments" / source_file.name
        assert copied_file.stat().st_mode & 0o777 == 0o600
        assert copied_file.read_bytes() == source_file.read_bytes()

        # Exercise the launcher's explicit-overwrite deletion path. The
        # partial-copy test below separately exercises Python recovery.
        subprocess.run(
            ["rm", "-rf", "--", str(destination)],
            check=True,
        )
        assert not destination.exists()
    finally:
        for directory in (nested, nested.parent, source):
            directory.chmod(0o700)


def test_run_manifest_reverifies_shared_private_snapshots_once_per_command(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tools.fine_tuning.unsloth import train_sft, training_lineage

    runtime_lineage = {
        "resolvedTrainingEnvironment": {"schemaVersion": "test"},
        "resolvedTrainingEnvironmentSHA256": "a" * 64,
        "resolvedTrainingEnvironmentScanAudit": {"distributionCount": 1},
        "spaceConfigurationSHA256": None,
        "zeroGPUSize": None,
        "zeroGPUDurationSeconds": None,
        "observedAccelerator": {"backend": "cuda", "deviceCount": 1},
    }
    environment = {
        "trainingEnvironmentSHA256": "e" * 64,
        "observedAccelerator": runtime_lineage["observedAccelerator"],
    }
    monkeypatch.setattr(
        ubuntu_pipeline,
        "_runtime_lineage",
        lambda **_kwargs: (runtime_lineage, environment),
    )
    monkeypatch.setattr(
        ubuntu_pipeline,
        "current_source_integrity",
        lambda *_args, **_kwargs: _source_integrity_fixture()[
            "ubuntuSourceIntegrity"
        ],
    )
    monkeypatch.setattr(
        train_sft,
        "_training_environment",
        lambda *_args, **_kwargs: environment,
    )
    _mock_private_base_model_snapshot_preparation(tmp_path, monkeypatch)
    run_root = tmp_path / "prepared-shared-snapshot"
    ubuntu_pipeline.prepare_run(
        root=REPO_ROOT,
        dataset_source=REPO_ROOT / "generated" / "fine_tuning",
        run_root=run_root,
        agents=("cortex", "executor"),
        variant=OPTIMIZED_VARIANT,
        seed=42,
        base_model_override="",
        container_digest=FAKE_IMAGE_DIGEST,
        evaluation_scope="smoke",
        evaluation_max_examples=7,
        gguf_requested=False,
    )
    config = ubuntu_pipeline.read_object(run_root / "configs" / "cortex.json")
    calls = {"tokenizer": 0, "runtime": 0}

    def verify_tokenizer(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        calls["tokenizer"] += 1
        return dict(config["baseModelTokenizerSnapshotVerification"])

    def verify_runtime(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        calls["runtime"] += 1
        return dict(config["baseModelRuntimeSnapshotVerification"])

    monkeypatch.setattr(
        training_lineage,
        "verify_private_base_model_tokenizer_snapshot",
        verify_tokenizer,
    )
    monkeypatch.setattr(
        training_lineage,
        "verify_private_base_model_conversion_snapshot",
        verify_runtime,
    )
    monkeypatch.setattr(
        ubuntu_pipeline,
        "_validated_global_tokenizer_resume_state",
        lambda **_kwargs: {},
    )

    ubuntu_pipeline._verified_run_manifest(run_root)

    assert calls == {"tokenizer": 1, "runtime": 1}

    calls.update(tokenizer=0, runtime=0)
    prepared_runtime = ubuntu_pipeline.validate_prepared_runtime(
        root=REPO_ROOT,
        run_root=run_root,
        agents=("cortex", "executor"),
        variant=OPTIMIZED_VARIANT,
        container_digest=FAKE_IMAGE_DIGEST,
        evaluation_scope="smoke",
        evaluation_max_examples=7,
        gguf_requested=False,
        observe_runtime=False,
    )

    assert calls == {"tokenizer": 1, "runtime": 1}
    assert set(prepared_runtime["evaluationPromptPreflights"]) == {
        "cortex",
        "executor",
    }
    assert all(
        evidence["caseCount"] > 0
        for evidence in prepared_runtime["evaluationPromptPreflights"].values()
    )

    original_evaluation_loader = evaluate_adapter.load_evaluation_records

    def load_drifted_prepared_prompts(
        *args: object,
        **kwargs: object,
    ) -> tuple[list, str]:
        records, evaluation_sha256 = original_evaluation_loader(*args, **kwargs)
        if kwargs.get("agent") != "executor":
            return records, evaluation_sha256
        drifted = json.loads(json.dumps(records))
        system_message = drifted[0]["messages"][0]
        system_message["content"] = system_message["content"].replace(
            "Response format contract:",
            "Response-format contract:",
            1,
        )
        return drifted, evaluation_sha256

    with monkeypatch.context() as prompt_drift:
        prompt_drift.setattr(
            evaluate_adapter,
            "load_evaluation_records",
            load_drifted_prepared_prompts,
        )
        with pytest.raises(
            ValueError,
            match=r"executor/.*drifted structured-output contract",
        ):
            ubuntu_pipeline.validate_prepared_runtime(
                root=REPO_ROOT,
                run_root=run_root,
                agents=("cortex", "executor"),
                variant=OPTIMIZED_VARIANT,
                container_digest=FAKE_IMAGE_DIGEST,
                evaluation_scope="smoke",
                evaluation_max_examples=7,
                gguf_requested=False,
                observe_runtime=False,
            )

    executor_config_path = run_root / "configs" / "executor.json"
    executor_config = ubuntu_pipeline.read_object(executor_config_path)
    executor_config["baseModelArtifactDigest"] = "f" * 64
    ubuntu_pipeline.write_object(executor_config_path, executor_config)
    run_manifest = ubuntu_pipeline.read_object(
        run_root / "aio_run_manifest.json"
    )
    executor_entry = next(
        item for item in run_manifest["agents"] if item["agent"] == "executor"
    )
    executor_entry["configSHA256"] = ubuntu_pipeline.file_sha256(
        executor_config_path
    )
    run_manifest.pop("runManifestSHA256")
    run_manifest["runManifestSHA256"] = ubuntu_pipeline.canonical_sha256(
        run_manifest
    )
    ubuntu_pipeline.write_object(
        run_root / "aio_run_manifest.json",
        run_manifest,
    )
    calls.update(tokenizer=0, runtime=0)

    with pytest.raises(
        RuntimeError,
        match=(
            "base-model runtime snapshot contract drifted.*"
            "baseModelArtifactDigest"
        ),
    ):
        ubuntu_pipeline._verified_run_manifest(run_root)

    assert calls == {"tokenizer": 1, "runtime": 1}


def test_partial_prepared_snapshot_remains_removable_after_source_rejection(
    tmp_path: Path,
) -> None:
    source = tmp_path / "partially-unsafe-source"
    completed = source / "a-completed"
    completed.mkdir(parents=True)
    completed_file = completed / "train.jsonl"
    completed_file.write_text("{}\n", encoding="utf-8")
    unsafe = source / "z-unsafe-link"
    unsafe.symlink_to(completed, target_is_directory=True)
    completed_file.chmod(0o444)
    completed.chmod(0o555)
    source.chmod(0o555)

    destination = tmp_path / "partial-private-snapshot"
    try:
        with pytest.raises(RuntimeError, match="contains a symlink"):
            ubuntu_pipeline._copy_private_regular_tree(source, destination)

        assert destination.stat().st_mode & 0o777 == 0o700
        assert (destination / completed.name).stat().st_mode & 0o777 == 0o700
        assert (
            destination / completed.name / completed_file.name
        ).stat().st_mode & 0o777 == 0o600
        shutil.rmtree(destination)
        assert not destination.exists()
    finally:
        source.chmod(0o700)
        completed.chmod(0o700)


def test_incomplete_preparation_recovery_never_deletes_training_progress(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tools.fine_tuning.unsloth import train_sft

    lineage = {
        "resolvedTrainingEnvironment": {
            "schemaVersion": "lumen.resolved-training-environment/1.0.0"
        },
        "resolvedTrainingEnvironmentSHA256": "a" * 64,
        "resolvedTrainingEnvironmentScanAudit": {"distributionCount": 1},
        "spaceConfigurationSHA256": None,
        "zeroGPUSize": None,
        "zeroGPUDurationSeconds": None,
        "observedAccelerator": {"backend": "cuda", "deviceCount": 1},
    }
    environment = {"trainingEnvironmentSHA256": "e" * 64}
    source_integrity = _source_integrity_fixture()["ubuntuSourceIntegrity"]
    monkeypatch.setattr(
        ubuntu_pipeline,
        "_runtime_lineage",
        lambda **_kwargs: (lineage, environment),
    )
    monkeypatch.setattr(
        ubuntu_pipeline,
        "current_source_integrity",
        lambda *_args, **_kwargs: source_integrity,
    )
    monkeypatch.setattr(
        train_sft,
        "_training_environment",
        lambda *_args, **_kwargs: environment,
    )
    _mock_private_base_model_snapshot_preparation(tmp_path, monkeypatch)
    # Simulate a kill after the manifest commit but before the durable owner
    # unlink, then a later loss of the manifest after training had begun.
    monkeypatch.setattr(
        ubuntu_pipeline,
        "_durably_remove_preparation_owner",
        lambda _run_root: None,
    )

    run_root = tmp_path / "prepared-with-progress"
    ubuntu_pipeline.prepare_run(
        root=REPO_ROOT,
        dataset_source=REPO_ROOT / "generated" / "fine_tuning",
        run_root=run_root,
        agents=("cortex",),
        variant=OPTIMIZED_VARIANT,
        seed=42,
        base_model_override="",
        container_digest=FAKE_IMAGE_DIGEST,
        evaluation_scope="smoke",
        evaluation_max_examples=7,
        gguf_requested=False,
    )
    checkpoint = run_root / "training" / "cortex" / "checkpoint-1"
    checkpoint.mkdir(parents=True)
    irreplaceable = b"signed-training-progress"
    (checkpoint / "adapter_model.safetensors").write_bytes(irreplaceable)
    (run_root / "aio_run_manifest.json").unlink()

    with pytest.raises(RuntimeError, match="training progress"):
        ubuntu_pipeline.recover_incomplete_preparation(
            root=REPO_ROOT,
            dataset_source=REPO_ROOT / "generated" / "fine_tuning",
            run_root=run_root,
            allowed_parent=tmp_path,
            agents=("cortex",),
            variant=OPTIMIZED_VARIANT,
            seed=42,
            base_model_override="",
            container_digest=FAKE_IMAGE_DIGEST,
            evaluation_scope="smoke",
            evaluation_max_examples=7,
            gguf_requested=False,
        )

    assert run_root.is_dir()
    assert (checkpoint / "adapter_model.safetensors").read_bytes() == irreplaceable


def test_final_config_switches_to_verified_preference_lineage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    config = {
        "agent": "cortex",
        "preference_trainer": "dpo",
        "dpo_learning_rate": 5e-6,
        "dpo_num_train_epochs": 1,
        "dpo_beta": 0.1,
        "dpo_rpo_alpha": None,
        "max_seq_length": 64,
        "max_prompt_length": 32,
        "batch_size": 1,
        "gradient_accumulation_steps": 1,
        "warmup_steps": 0,
        "gradient_checkpointing": True,
        "use_logits_to_keep": True,
        "precompute_ref_log_probs": True,
        "precompute_ref_batch_size": 1,
        "bf16": False,
        "fp16": True,
        "trainingCodeManifestsByPhase": {"dpo": {"phase": "dpo"}},
        "trainingCodeSHA256ByPhase": {"dpo": "d" * 64},
        "variantAttestation": {
            "trainingEnvironmentSHA256": "a" * 64,
        },
        "adapterExport": {
            "adapterArtifact": "old-sft",
            "adapterDirectory": "old-sft",
            "adapterGGUFArtifact": "old-gguf",
        },
    }
    (config_dir / "cortex.json").write_text(json.dumps(config), encoding="utf-8")
    behavior = (
        tmp_path
        / "generated"
        / "agent_manifest"
        / "AgentBehaviorManifest.json"
    )
    behavior.parent.mkdir(parents=True)
    behavior.write_text("{}\n", encoding="utf-8")
    run_manifest = {
        "behaviorManifestFileSHA256": ubuntu_pipeline.file_sha256(behavior),
    }
    run_manifest["runManifestSHA256"] = ubuntu_pipeline.canonical_sha256(
        run_manifest
    )
    ubuntu_pipeline.write_object(tmp_path / "aio_run_manifest.json", run_manifest)
    finalized = {
        "trainingEnvironmentSHA256": "e" * 64,
        "resolvedTrainingEnvironment": {"schemaVersion": "test"},
        "resolvedTrainingEnvironmentSHA256": "f" * 64,
        "observedAccelerator": {"backend": "cuda"},
        "zeroGPUSize": None,
        "zeroGPUDurationSeconds": None,
        "artifact": {"adapterSHA256": "b" * 64},
        **{field: "c" * 40 for field in ubuntu_pipeline.RUNTIME_SOURCE_FIELDS},
    }
    monkeypatch.setattr(
        ubuntu_pipeline,
        "verify_preference",
        lambda *_args: {
            "adapterSHA256": "b" * 64,
            "parentSFTAdapterSHA256": "a" * 64,
            "tokenLengthPreflightSHA256": "f" * 64,
            "tokenLengthStatistics": {
                "promptTokens": {"min": 1, "p50": 2, "p95": 3, "max": 4}
            },
        },
    )
    monkeypatch.setattr(
        ubuntu_pipeline,
        "_verify_manifest_integrity",
        lambda *_args: finalized,
    )

    result = ubuntu_pipeline.write_final_config(tmp_path, "cortex")
    written = json.loads(Path(result["config"]).read_text(encoding="utf-8"))

    assert written["adapter_training_phase"] == "sft_dpo"
    assert written["parent_sft_adapter_sha256"] == "a" * 64
    assert written["preferenceTokenLengthPreflightSHA256"] == "f" * 64
    assert written["adapter_output_dir"].endswith("models/lora_qwen3_dpo/cortex")
    assert written["trainingCodeSHA256"] == "d" * 64
    assert written["variantAttestation"]["trainingEnvironmentSHA256"] == "e" * 64
    assert written["adapterExport"]["adapterArtifact"].endswith(
        "models/lora_qwen3_dpo/cortex"
    )


def test_summary_rejects_failed_full_evaluation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_path = _write_evaluation_evidence(
        tmp_path,
        monkeypatch=monkeypatch,
        status="quality_gate_failed",
        quality_gate_passed=False,
        attested_source=True,
        completion=(
            '{"selectedToolID":"files.read","intent":"files",'
            '"reasoningSummary":"Manifest row files.read is selected for intent '
            'files without actionStep.","requiresApproval":false,'
            '"nextModel":"executor"}'
        ),
    )
    evaluation_run = ubuntu_pipeline.read_object(run_path)
    smoke_evidence = _write_runtime_binding_smoke_summary_fixture(tmp_path)
    monkeypatch.setattr(
        ubuntu_pipeline,
        "_verified_runtime_binding_smoke_summary_evidence",
        lambda *_args: smoke_evidence,
    )
    monkeypatch.setattr(
        ubuntu_pipeline,
        "verify_sft",
        lambda *_args: {
            "adapterSHA256": "a" * 64,
            **_phase_runtime_evidence_fixture("1"),
        },
    )
    monkeypatch.setattr(
        ubuntu_pipeline,
        "verify_preference",
        lambda *_args: {
            "adapterSHA256": "b" * 64,
            "finalizedVariantManifestSHA256": evaluation_run[
                "finalizedVariantManifestSHA256"
            ],
            "parentSFTAdapterSHA256": "9" * 64,
            "phase": "dpo",
            **_phase_runtime_evidence_fixture("2"),
            "tokenLengthPreflightSHA256": "e" * 64,
            "tokenLengthStatistics": {
                "promptTokens": {"min": 1, "p50": 1, "p95": 1, "max": 1}
            },
        },
    )

    with pytest.raises(RuntimeError, match="did not pass"):
        ubuntu_pipeline.write_summary(
            run_root=tmp_path,
            agents=("cortex",),
            variant="internal_plus_public_optimized",
            preference=True,
            require_gguf=False,
            require_evaluation=True,
        )


def test_full_quality_summary_without_gguf_is_complete_and_upload_qualified(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = "cortex"
    plan = _test_execution_plan(gguf_requested=False)
    source_fields = _source_integrity_fixture()
    (tmp_path / "models" / "lora_qwen3_gguf").mkdir(parents=True)
    (tmp_path / "models" / "lora_qwen3_gguf_receipts").mkdir(parents=True)
    evaluation_dir = tmp_path / "evaluation" / agent
    evaluation_dir.mkdir(parents=True)
    (evaluation_dir / "evaluation_report.json").write_text(
        "{}\n",
        encoding="utf-8",
    )
    (evaluation_dir / "evaluation_run_manifest.json").write_text(
        "{}\n",
        encoding="utf-8",
    )
    sft = {
        "phase": "sft",
        "adapterSHA256": "a" * 64,
        **_phase_runtime_evidence_fixture("1"),
    }
    final_phase = {
        "phase": "dpo",
        "adapterSHA256": "b" * 64,
        "parentSFTAdapterSHA256": "a" * 64,
        **_phase_runtime_evidence_fixture("2"),
    }
    evaluation = {
        "status": "quality_gate_passed",
        "qualityGatePassed": True,
        "runtimeModelBinding": {"runtimeModelBindingSHA256": "a" * 64},
        "runtimeTokenizerBinding": {
            "runtimeTokenizerBindingSHA256": "a" * 64
        },
    }
    smoke_evidence = _write_runtime_binding_smoke_summary_fixture(tmp_path)
    monkeypatch.setattr(
        ubuntu_pipeline,
        "_verified_run_manifest",
        lambda *_args: {
                "variant": OPTIMIZED_VARIANT,
                "executionPlan": plan,
                "agents": [{"agent": agent}],
                **_summary_base_model_lineage_fixture(),
            **source_fields,
        },
    )
    monkeypatch.setattr(ubuntu_pipeline, "verify_sft", lambda *_args: sft)
    monkeypatch.setattr(
        ubuntu_pipeline,
        "_verified_runtime_binding_smoke_summary_evidence",
        lambda *_args: smoke_evidence,
    )
    monkeypatch.setattr(
        ubuntu_pipeline,
        "verify_preference",
        lambda *_args: final_phase,
    )
    monkeypatch.setattr(
        ubuntu_pipeline,
        "_verify_evaluation_outputs",
        lambda *_args, **_kwargs: evaluation,
    )

    summary = ubuntu_pipeline.write_summary(
        run_root=tmp_path,
        agents=(agent,),
        variant=OPTIMIZED_VARIANT,
        preference=True,
        require_gguf=False,
        require_evaluation=True,
    )

    assert summary["status"] == "complete_without_gguf"
    assert summary["evaluationStatus"] == "quality_gate_passed"
    assert summary["evaluationScope"] == "full"
    assert summary["ggufStatus"] == "skipped_by_operator"
    assert summary["qualification"] == "quality_gate_passed"
    assert summary["promotionEligible"] is True
    assert ubuntu_pipeline._verified_completed_summary(
        tmp_path,
        (agent,),
    ) == summary
    assert ubuntu_pipeline._upload_publication_contract(
        summary,
        allow_diagnostic_upload=False,
    )["remoteNamespace"] == "runs"


def test_sft_only_summary_is_distinct_reverifiable_and_diagnostic_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = "cortex"
    plan = _test_execution_plan(evaluation_scope="none", gguf_requested=False)
    source_fields = _source_integrity_fixture()
    sft = {
        "phase": "sft",
        "adapterSHA256": "a" * 64,
        **_phase_runtime_evidence_fixture("1"),
    }
    smoke_evidence = _write_runtime_binding_smoke_summary_fixture(tmp_path)
    monkeypatch.setattr(
        ubuntu_pipeline,
        "_verified_run_manifest",
        lambda *_args: {
            "variant": OPTIMIZED_VARIANT,
            "executionPlan": plan,
            "agents": [{"agent": agent}],
            **_summary_base_model_lineage_fixture(),
            **source_fields,
        },
    )
    monkeypatch.setattr(ubuntu_pipeline, "verify_sft", lambda *_args: sft)
    monkeypatch.setattr(
        ubuntu_pipeline,
        "_verified_runtime_binding_smoke_summary_evidence",
        lambda *_args: smoke_evidence,
    )

    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("SFT-only verification crossed a later-phase boundary")

    monkeypatch.setattr(ubuntu_pipeline, "verify_preference", forbidden)
    monkeypatch.setattr(ubuntu_pipeline, "_verify_evaluation_outputs", forbidden)
    monkeypatch.setattr(ubuntu_pipeline, "_verify_gguf_inventory", forbidden)

    summary = ubuntu_pipeline.write_summary(
        run_root=tmp_path,
        agents=(agent,),
        variant=OPTIMIZED_VARIANT,
        preference=False,
        require_gguf=False,
        require_evaluation=False,
    )

    assert summary["status"] == "sft_only_diagnostic_complete"
    assert summary["trainingScope"] == "sft_only"
    assert summary["preferenceTraining"] is False
    assert summary["evaluationStatus"] == "not_run"
    assert summary["ggufStatus"] == "not_applicable_sft_only"
    assert summary["qualification"] == "diagnostic_only"
    assert summary["promotionEligible"] is False
    assert summary["agents"] == {agent: {"sft": sft, "finalPhase": sft}}
    assert ubuntu_pipeline._verified_completed_summary(
        tmp_path,
        (agent,),
    ) == summary
    with pytest.raises(RuntimeError, match="--allow-diagnostic-upload"):
        ubuntu_pipeline._upload_publication_contract(
            summary,
            allow_diagnostic_upload=False,
        )
    publication = ubuntu_pipeline._upload_publication_contract(
        summary,
        allow_diagnostic_upload=True,
    )
    assert publication["remoteNamespace"] == "diagnostic-sft-runs"
    assert publication["phaseRuntimeEvidenceByAgent"] == {
        agent: {"sft": _phase_runtime_evidence_fixture("1")}
    }


@pytest.mark.parametrize(
    "plan",
    (
        _test_execution_plan(evaluation_scope="full", gguf_requested=False),
        _test_execution_plan(evaluation_scope="none", gguf_requested=True),
    ),
)
def test_sft_only_summary_rejects_later_phase_execution_plans(plan: dict) -> None:
    with pytest.raises(RuntimeError, match="SFT-only diagnostic summaries"):
        ubuntu_pipeline._derived_summary_state(
            plan=plan,
            evaluation_statuses=(),
            agent_count=1,
            gguf_count=0,
            preference_training=False,
        )


@pytest.mark.parametrize("field", ubuntu_pipeline.PHASE_RUNTIME_EVIDENCE_FIELDS)
def test_compact_phase_runtime_evidence_rejects_each_mutated_digest(
    field: str,
) -> None:
    evidence = _phase_runtime_evidence_fixture("1")
    evidence[field] = "not-a-digest"

    with pytest.raises(RuntimeError, match="lacks exact digests"):
        ubuntu_pipeline._compact_phase_runtime_evidence(evidence)


@pytest.mark.parametrize(
    ("model_field", "tokenizer_field"),
    (
        ("runtimeModelBindingSHA256", "runtimeTokenizerBindingSHA256"),
        (
            "adapterGGUFRuntimeModelBindingSHA256",
            "adapterGGUFRuntimeTokenizerBindingSHA256",
        ),
    ),
)
@pytest.mark.parametrize("mutated", ("model", "tokenizer"))
def test_runtime_evidence_must_match_pretraining_smoke_bindings(
    tmp_path: Path,
    model_field: str,
    tokenizer_field: str,
    mutated: str,
) -> None:
    smoke_evidence = _write_runtime_binding_smoke_summary_fixture(tmp_path)
    source = {
        model_field: "a" * 64,
        tokenizer_field: "a" * 64,
    }
    source[model_field if mutated == "model" else tokenizer_field] = "0" * 64

    with pytest.raises(RuntimeError, match="pre-training smoke gate"):
        ubuntu_pipeline._require_runtime_bindings_match_smoke(
            agent="cortex",
            source=source,
            smoke_evidence=smoke_evidence,
            label="test evidence",
            model_field=model_field,
            tokenizer_field=tokenizer_field,
        )


def test_phase_runtime_evidence_returns_only_reconstructed_exact_hashes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer_verification = {"snapshotVerificationSHA256": "1" * 64}
    runtime_verification = {"snapshotVerificationSHA256": "2" * 64}
    config = {
        "baseModelTokenizerDigest": "3" * 64,
        "baseModelTokenizerFiles": [],
        "baseModelTokenizerClosureSHA256": "4" * 64,
        "baseModelGenerationConfigFile": {"path": "generation_config.json"},
        "baseModelTokenizerSnapshotPath": "/private/tokenizer",
        "baseModelRuntimeSnapshotPath": "/private/runtime",
    }
    report = {
        **config,
        "baseModelTokenizerSnapshotVerification": tokenizer_verification,
        "baseModelRuntimeSnapshotVerification": runtime_verification,
        "runtimeModelBinding": {"runtimeModelBindingSHA256": "5" * 64},
        "runtimeTokenizerBinding": {"runtimeTokenizerBindingSHA256": "6" * 64},
        "peftBaseModelIdentity": {"peftBaseModelIdentitySHA256": "7" * 64},
        "adapterTokenizerBinding": {"adapterTokenizerBindingSHA256": "8" * 64},
    }
    monkeypatch.setattr(
        ubuntu_pipeline,
        "_verified_private_tokenizer_snapshot_binding",
        lambda *_args: tokenizer_verification,
    )
    monkeypatch.setattr(
        ubuntu_pipeline,
        "_verified_private_base_model_runtime_snapshot_binding",
        lambda *_args: runtime_verification,
    )
    monkeypatch.setattr(
        ubuntu_pipeline,
        "_verified_runtime_model_binding",
        lambda value, **_kwargs: value,
    )
    monkeypatch.setattr(
        ubuntu_pipeline,
        "_verified_runtime_tokenizer_binding",
        lambda value, **_kwargs: value,
    )
    monkeypatch.setattr(
        ubuntu_pipeline,
        "_verified_peft_base_model_evidence",
        lambda value, **_kwargs: value,
    )
    monkeypatch.setattr(
        ubuntu_pipeline,
        "_verified_adapter_tokenizer_evidence",
        lambda value, **_kwargs: value,
    )

    assert ubuntu_pipeline._verify_phase_runtime_evidence(
        config=config,
        report=report,
        adapter_dir=tmp_path,
    ) == {
        "runtimeModelBindingSHA256": "5" * 64,
        "runtimeTokenizerBindingSHA256": "6" * 64,
        "peftBaseModelIdentitySHA256": "7" * 64,
        "adapterTokenizerBindingSHA256": "8" * 64,
        "baseModelTokenizerSnapshotVerificationSHA256": "1" * 64,
        "baseModelRuntimeSnapshotVerificationSHA256": "2" * 64,
    }


def test_runtime_model_binding_reconstructs_nested_materialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_snapshot = tmp_path / "runtime-snapshot"
    runtime_snapshot.mkdir()
    runtime_model_config = {
        "model_type": "qwen3",
        "num_hidden_layers": 28,
        "vocab_size": 151_936,
        "attention_bias": False,
        "tie_word_embeddings": True,
        "max_position_embeddings": 4096,
    }
    (runtime_snapshot / "config.json").write_text(
        json.dumps(runtime_model_config),
        encoding="utf-8",
    )
    snapshot_verification = {"snapshotVerificationSHA256": "9" * 64}
    config = {
        "baseModelID": "Qwen/Qwen3-0.6B",
        "baseModelRevision": "1" * 40,
        "baseModelIndexDigest": "2" * 64,
        "baseModelIndexShardBindingSHA256": "3" * 64,
        "baseModelArtifactDigest": "4" * 64,
        "baseModelTokenizerClosureSHA256": "5" * 64,
        "baseModelGenerationConfigFile": {
            "path": "generation_config.json",
            "sha256": "6" * 64,
        },
        "baseModelRuntimeSnapshotPath": str(runtime_snapshot),
        "max_seq_length": 2048,
        "bf16": True,
        "fp16": False,
    }
    source_generation = {"max_length": 20, "do_sample": False}

    from tools.fine_tuning.unsloth import train_sft

    class FakeGenerationConfig:
        @classmethod
        def from_pretrained(
            cls,
            _path: str,
            *,
            local_files_only: bool,
        ) -> SimpleNamespace:
            assert local_files_only is True
            return SimpleNamespace(to_dict=lambda: dict(source_generation))

    transformers = ModuleType("transformers")
    transformers.GenerationConfig = FakeGenerationConfig  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setattr(
        ubuntu_pipeline,
        "_verified_private_base_model_runtime_snapshot_binding",
        lambda _config: snapshot_verification,
    )
    calls: list[tuple[dict[str, Any], Mapping[str, Any], Mapping[str, Any]]] = []

    from tools.fine_tuning.unsloth import runtime_binding_smoke_gate

    def verify_materialization(
        value: Mapping[str, Any],
        observed_config: Mapping[str, Any],
        bound_config_payload: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if value != {"controlled": True}:
            raise RuntimeError("nested materialization drifted")
        assert bound_config_payload is not None
        calls.append((dict(value), observed_config, bound_config_payload))
        return dict(value)

    monkeypatch.setattr(
        runtime_binding_smoke_gate,
        "verify_runtime_load_materialization_evidence",
        verify_materialization,
    )
    runtime_generation = {**source_generation, "max_length": 4096}
    unsigned = {
        "schemaVersion": train_sft.RUNTIME_MODEL_BINDING_SCHEMA,
        "baseModelID": config["baseModelID"],
        "baseModelRevision": config["baseModelRevision"],
        "baseModelIndexDigest": config["baseModelIndexDigest"],
        "baseModelIndexShardBindingSHA256": config[
            "baseModelIndexShardBindingSHA256"
        ],
        "baseModelArtifactDigest": config["baseModelArtifactDigest"],
        "baseModelTokenizerClosureSHA256": config[
            "baseModelTokenizerClosureSHA256"
        ],
        "baseModelGenerationConfigFile": config["baseModelGenerationConfigFile"],
        "runtimeSnapshotVerificationSHA256": snapshot_verification[
            "snapshotVerificationSHA256"
        ],
        "runtimeSnapshotPath": str(runtime_snapshot),
        "modelConfigSHA256": "7" * 64,
        "modelConfigVerificationStatus": (
            "attested_runtime_observation_not_independently_reconstructed"
        ),
        "sourceGenerationConfigSHA256": ubuntu_pipeline.canonical_sha256(
            source_generation
        ),
        "generationConfigSHA256": ubuntu_pipeline.canonical_sha256(
            runtime_generation
        ),
        "generationConfigSource": "verified_private_generation_config_file",
        "allowedGenerationConfigTransformations": {
            "maxLength": {
                "source": (
                    "verified_runtime_model.config.max_position_embeddings"
                ),
                "sourceValue": 4096,
                "originalValue": 20,
                "runtimeValue": 4096,
            }
        },
        "runtimeLoadMaterialization": {"controlled": True},
        "localFilesOnly": True,
    }
    binding = {
        **unsigned,
        "runtimeModelBindingSHA256": ubuntu_pipeline.canonical_sha256(unsigned),
    }

    assert ubuntu_pipeline._verified_runtime_model_binding(
        binding,
        config=config,
        snapshot_verification=snapshot_verification,
    ) == binding
    assert calls == [({"controlled": True}, config, runtime_model_config)]

    mutated_unsigned = {
        **unsigned,
        "runtimeLoadMaterialization": {"controlled": False},
    }
    mutated = {
        **mutated_unsigned,
        "runtimeModelBindingSHA256": ubuntu_pipeline.canonical_sha256(
            mutated_unsigned
        ),
    }
    with pytest.raises(RuntimeError, match="nested materialization drifted"):
        ubuntu_pipeline._verified_runtime_model_binding(
            mutated,
            config=config,
            snapshot_verification=snapshot_verification,
        )


@pytest.mark.parametrize(
    ("plan", "statuses", "agent_count", "gguf_count", "error"),
    (
        (
            _test_execution_plan(gguf_requested=False),
            ["quality_gate_passed"],
            2,
            0,
            "partial evaluation evidence",
        ),
        (
            _test_execution_plan(evaluation_scope="smoke", evaluation_max_examples=1, gguf_requested=False),
            ["quality_gate_passed", "smoke_complete"],
            2,
            0,
            "does not match the execution plan",
        ),
        (
            _test_execution_plan(),
            ["quality_gate_passed", "quality_gate_passed"],
            2,
            1,
            "GGUF inventory does not match",
        ),
        (
            _test_execution_plan(),
            ["quality_gate_passed", "quality_gate_passed"],
            2,
            0,
            "GGUF inventory does not match",
        ),
    ),
)
def test_summary_state_rejects_partial_mixed_or_plan_inconsistent_evidence(
    plan: dict,
    statuses: list[str],
    agent_count: int,
    gguf_count: int,
    error: str,
) -> None:
    with pytest.raises(RuntimeError, match=error):
        ubuntu_pipeline._derived_summary_state(
            plan=plan,
            evaluation_statuses=statuses,
            agent_count=agent_count,
            gguf_count=gguf_count,
        )


@pytest.mark.parametrize(
    (
        "evaluation_scope",
        "evaluation_max_examples",
        "evaluation_statuses",
        "gguf_requested",
        "expected_status",
        "expected_evaluation_status",
        "expected_gguf_status",
    ),
    (
        (
            "full",
            None,
            ["quality_gate_passed"],
            True,
            "complete",
            "quality_gate_passed",
            "verified",
        ),
        (
            "full",
            None,
            ["quality_gate_passed"],
            False,
            "complete_without_gguf",
            "quality_gate_passed",
            "skipped_by_operator",
        ),
        (
            "smoke",
            1,
            ["smoke_complete"],
            True,
            "smoke_complete",
            "smoke_complete",
            "verified",
        ),
        (
            "smoke",
            1,
            ["smoke_complete"],
            False,
            "smoke_complete",
            "smoke_complete",
            "skipped_by_operator",
        ),
        (
            "none",
            None,
            [],
            True,
            "training_complete_without_full_evaluation",
            "not_run",
            "verified",
        ),
        (
            "none",
            None,
            [],
            False,
            "training_complete_without_full_evaluation",
            "not_run",
            "skipped_by_operator",
        ),
    ),
)
def test_summary_state_matrix_keeps_evaluation_and_gguf_independent(
    evaluation_scope: str,
    evaluation_max_examples: int | None,
    evaluation_statuses: list[str],
    gguf_requested: bool,
    expected_status: str,
    expected_evaluation_status: str,
    expected_gguf_status: str,
) -> None:
    state = ubuntu_pipeline._derived_summary_state(
        plan=_test_execution_plan(
            evaluation_scope=evaluation_scope,
            evaluation_max_examples=evaluation_max_examples,
            gguf_requested=gguf_requested,
        ),
        evaluation_statuses=evaluation_statuses,
        agent_count=1,
        gguf_count=1 if gguf_requested else 0,
    )

    assert state["status"] == expected_status
    assert state["evaluationStatus"] == expected_evaluation_status
    assert state["evaluationScope"] == evaluation_scope
    assert state["ggufStatus"] == expected_gguf_status


@pytest.mark.parametrize(
    ("evaluation_status", "evaluation_scope", "status"),
    (
        ("smoke_complete", "smoke", "smoke_complete"),
        (
            "not_run",
            "none",
            "training_complete_without_full_evaluation",
        ),
    ),
)
def test_diagnostic_publication_requires_override_and_separate_namespace(
    tmp_path: Path,
    evaluation_status: str,
    evaluation_scope: str,
    status: str,
) -> None:
    smoke_evidence = _write_runtime_binding_smoke_summary_fixture(tmp_path)
    plan = _test_execution_plan(
        evaluation_scope=evaluation_scope,
        evaluation_max_examples=1 if evaluation_scope == "smoke" else None,
        gguf_requested=False,
    )
    summary = {
        "status": status,
        "evaluationStatus": evaluation_status,
        "evaluationScope": evaluation_scope,
        "ggufStatus": "skipped_by_operator",
        "ggufConversionStatus": "skipped_by_operator",
        "ggufTensorEquivalenceStatus": "not_applicable",
        "qualification": "diagnostic_only",
        "promotionEligible": False,
        "preferenceTraining": True,
        "trainingScope": "sft_preference",
        **smoke_evidence,
        "executionPlanSHA256": plan["executionPlanSHA256"],
        **_summary_base_model_lineage_fixture(),
        "agents": {
            "cortex": {
                "sft": _phase_runtime_evidence_fixture("1"),
                "finalPhase": _phase_runtime_evidence_fixture("2"),
            }
        },
    }
    with pytest.raises(RuntimeError, match="--allow-diagnostic-upload"):
        ubuntu_pipeline._upload_publication_contract(
            summary,
            allow_diagnostic_upload=False,
        )

    publication = ubuntu_pipeline._upload_publication_contract(
        summary,
        allow_diagnostic_upload=True,
    )
    assert publication == {
        "remoteNamespace": "diagnostic-runs",
        "qualification": "diagnostic_only",
        "promotionEligible": False,
        "diagnosticUploadOverrideApplied": True,
        "preferenceTraining": True,
        "trainingScope": "sft_preference",
        "phaseRuntimeEvidenceByAgent": {
            "cortex": {
                "sft": _phase_runtime_evidence_fixture("1"),
                "preference": _phase_runtime_evidence_fixture("2"),
            }
        },
        **smoke_evidence,
        "evaluationStatus": evaluation_status,
        "evaluationScope": evaluation_scope,
        "ggufStatus": "skipped_by_operator",
        "ggufConversionStatus": "skipped_by_operator",
        "ggufTensorEquivalenceStatus": "not_applicable",
        "executionPlanSHA256": plan["executionPlanSHA256"],
        **_publication_base_model_lineage_fixture(),
    }


def test_launcher_rejects_diagnostic_upload_without_distinct_override() -> None:
    result = subprocess.run(
        [
            "bash",
            "scripts/ubuntu_train_lumen_full_pipeline.sh",
            "--upload",
            "--eval-smoke",
            "1",
            "--run-id",
            "diagnostic-upload-gate",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "requires --allow-diagnostic-upload" in result.stderr


@pytest.mark.parametrize(
    "evaluation_args",
    (
        ("--no-evaluate", "--eval-smoke", "1"),
        ("--eval-smoke", "1", "--no-evaluate"),
    ),
)
def test_launcher_rejects_conflicting_evaluation_flags_in_any_order(
    evaluation_args: tuple[str, ...],
) -> None:
    result = subprocess.run(
        [
            "bash",
            "scripts/ubuntu_train_lumen_full_pipeline.sh",
            *evaluation_args,
            "--run-id",
            "conflicting-evaluation-flags",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "--no-evaluate and --eval-smoke are mutually exclusive" in result.stderr


@pytest.mark.parametrize(
    "mutation",
    (
        "run_schema",
        "evaluation_sha",
        "evaluation_path",
        "evaluator_code_sha",
        "output_mode_contract_sha",
        "output_mode_contract_record",
        "recovery_counter",
        "config_path",
        "config_sha",
        "chat_template_contract",
        "adapter_directory",
        "finalized_path",
        "finalized_sha",
        "behavior_path",
        "behavior_sha",
        "variant",
        "generation_max_new_tokens",
        "generation_max_sequence_length",
        "generation_seed",
        "generation_repetition_penalty",
        "generation_extra_field",
    ),
)
def test_evaluation_summary_verifier_rejects_controlled_evidence_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    run_path = _write_evaluation_evidence(tmp_path, monkeypatch=monkeypatch)
    evaluation_run = ubuntu_pipeline.read_object(run_path)
    final_phase = _evaluation_final_phase(evaluation_run)
    verified = ubuntu_pipeline._verify_evaluation_outputs(
        tmp_path,
        "cortex",
        final_phase=final_phase,
    )
    assert verified["status"] == "quality_gate_passed"

    manifest = ubuntu_pipeline.read_object(run_path)
    if mutation == "run_schema":
        manifest["schemaVersion"] = "lumen.adapter-evaluation-run/1.1.0"
    elif mutation == "evaluation_sha":
        manifest["evaluationSHA256"] = "0" * 64
    elif mutation == "evaluation_path":
        manifest["evaluationJSONLPath"] = str(
            (tmp_path.parent / "outside-eval.jsonl").resolve()
        )
    elif mutation == "evaluator_code_sha":
        manifest["evaluatorCodeSHA256"] = "0" * 64
    elif mutation == "output_mode_contract_sha":
        manifest["generation"]["outputModeContract"][
            "outputModeContractSHA256"
        ] = "0" * 64
    elif mutation == "output_mode_contract_record":
        contract = manifest["generation"]["outputModeContract"]
        contract["records"][0].update(
            {
                "outputMode": "text",
                "structuredOutputContractSHA256": None,
                "strictJSONRetryEligible": False,
                "strictJSONMaxAttempts": 1,
                "strictJSONRetryContractSHA256": None,
            }
        )
        unsigned_contract = dict(contract)
        unsigned_contract.pop("outputModeContractSHA256")
        contract["outputModeContractSHA256"] = (
            evaluate_adapter._canonical_sha256(unsigned_contract)
        )
    elif mutation == "recovery_counter":
        manifest["formatRecoveryCount"] = 1
    elif mutation == "config_path":
        manifest["configPath"] = str((tmp_path / "configs" / "other.json").resolve())
    elif mutation == "config_sha":
        manifest["configSHA256"] = "0" * 64
    elif mutation == "chat_template_contract":
        manifest["chatTemplateContract"]["generationPrefixOwnership"] = (
            "completion"
        )
    elif mutation == "adapter_directory":
        manifest["adapterDirectory"] = str(
            (tmp_path / "models" / "lora_qwen3_dpo" / "executor").resolve()
        )
    elif mutation == "finalized_path":
        manifest["finalizedVariantManifestPath"] = str(
            (tmp_path / "training" / "other.json").resolve()
        )
    elif mutation == "finalized_sha":
        manifest["finalizedVariantManifestSHA256"] = "0" * 64
    elif mutation == "behavior_path":
        manifest["behaviorManifestPath"] = str(
            (tmp_path / "generated" / "other.json").resolve()
        )
    elif mutation == "behavior_sha":
        manifest["behaviorManifestSHA256"] = "0" * 64
    elif mutation == "variant":
        manifest["variant"] = "internal_only"
    elif mutation == "generation_max_new_tokens":
        manifest["generation"]["maxNewTokens"] = 9
    elif mutation == "generation_max_sequence_length":
        manifest["generation"]["maxSequenceLength"] = 65
    elif mutation == "generation_seed":
        manifest["generation"]["seed"] = 43
    elif mutation == "generation_repetition_penalty":
        manifest["generation"]["repetitionPenalty"] = 1.0
    elif mutation == "generation_extra_field":
        manifest["generation"]["temperature"] = 0
    else:  # pragma: no cover - parametrization is closed above.
        raise AssertionError(mutation)
    manifest.pop("runManifestSHA256", None)
    manifest["runManifestSHA256"] = ubuntu_pipeline.canonical_sha256(manifest)
    ubuntu_pipeline.write_object(run_path, manifest)

    with pytest.raises(RuntimeError):
        ubuntu_pipeline._verify_evaluation_outputs(
            tmp_path,
            "cortex",
            final_phase=final_phase,
        )


def test_evaluation_verifier_rejects_leftover_checkpoint_after_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_path = _write_evaluation_evidence(tmp_path, monkeypatch=monkeypatch)
    final_phase = _evaluation_final_phase(ubuntu_pipeline.read_object(run_path))
    checkpoint = run_path.parent / evaluate_adapter.EVALUATION_CHECKPOINT_FILENAME
    checkpoint.write_text("{}\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="exactly the verified evidence trio"):
        ubuntu_pipeline._verify_evaluation_outputs(
            tmp_path,
            "cortex",
            final_phase=final_phase,
        )


@pytest.mark.parametrize(
    (
        "terminal_status",
        "failed_completion",
        "retry_from_completion",
        "full_count",
        "generated_count",
    ),
    (
        (
            "quality_gate_failed",
            json.dumps(
                {
                    "selectedToolID": None,
                    "intent": "unknown",
                    "reasoningSummary": (
                        "No manifest row applies to intent unknown."
                    ),
                    "status": "no_tool_route",
                    "requiresApproval": False,
                    "nextModel": "mouth",
                },
                separators=(",", ":"),
            ),
            None,
            1,
            None,
        ),
        ("format_failed", "second-invalid-json", "first-invalid-json", 1, None),
        (
            "smoke_failed",
            json.dumps(
                {
                    "selectedToolID": None,
                    "intent": "unknown",
                    "reasoningSummary": (
                        "No manifest row applies to intent unknown."
                    ),
                    "status": "no_tool_route",
                    "requiresApproval": False,
                    "nextModel": "mouth",
                },
                separators=(",", ":"),
            ),
            None,
            2,
            1,
        ),
    ),
)
def test_completed_quality_failure_is_verified_and_classified_without_promotion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    terminal_status: str,
    failed_completion: str,
    retry_from_completion: str | None,
    full_count: int,
    generated_count: int | None,
) -> None:
    run_path = _write_evaluation_evidence(
        tmp_path,
        monkeypatch=monkeypatch,
        completion=failed_completion,
        retry_from_completion=retry_from_completion,
        status=terminal_status,
        quality_gate_passed=False,
        full_case_count=full_count,
        generated_case_count=generated_count,
    )
    final_phase = _evaluation_final_phase(
        ubuntu_pipeline.read_object(run_path)
    )

    with pytest.raises(RuntimeError, match="did not pass"):
        ubuntu_pipeline._verify_evaluation_outputs(
            tmp_path,
            "cortex",
            final_phase=final_phase,
        )

    verified = ubuntu_pipeline._verify_evaluation_outputs(
        tmp_path,
        "cortex",
        final_phase=final_phase,
        require_passing_status=False,
    )
    assert verified["status"] == terminal_status
    assert verified["qualityGatePassed"] is False

    monkeypatch.setattr(
        ubuntu_pipeline,
        "verify_preference",
        lambda _run_root, _agent: final_phase,
    )
    classification = ubuntu_pipeline.classify_completed_evaluation(
        tmp_path,
        "cortex",
    )
    assert classification == {
        "agent": "cortex",
        "state": "completed_quality_failure",
        "status": terminal_status,
        "qualityGatePassed": False,
        "evaluationRunManifest": str(run_path),
        "evaluationRunManifestSHA256": verified["runManifestSHA256"],
    }


def test_evaluation_verifier_binds_image_source_attestation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_path = _write_evaluation_evidence(
        tmp_path,
        monkeypatch=monkeypatch,
        attested_source=True,
    )
    final_phase = _evaluation_final_phase(
        ubuntu_pipeline.read_object(run_path)
    )
    ubuntu_pipeline._verify_evaluation_outputs(
        tmp_path,
        "cortex",
        final_phase=final_phase,
    )

    manifest = ubuntu_pipeline.read_object(run_path)
    manifest["workingTreeDigest"] = "0" * 64
    manifest.pop("runManifestSHA256", None)
    manifest["runManifestSHA256"] = ubuntu_pipeline.canonical_sha256(manifest)
    ubuntu_pipeline.write_object(run_path, manifest)
    with pytest.raises(RuntimeError, match="lineage failed"):
        ubuntu_pipeline._verify_evaluation_outputs(
            tmp_path,
            "cortex",
            final_phase=final_phase,
        )


def test_evaluation_verifier_accepts_evidenced_two_attempt_cortex_recovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    invalid_first_completion = json.dumps(
        {
            "selectedToolID": "mail.send",
            "intent": "files",
            "reasoningSummary": "The requested tool is outside the manifest.",
            "actionStep": {
                "type": "tool_call",
                "toolID": "mail.send",
                "mustPersistBeforeFinal": True,
            },
            "requiresApproval": False,
            "nextModel": "executor",
        },
        separators=(",", ":"),
    )
    run_path = _write_evaluation_evidence(
        tmp_path,
        monkeypatch=monkeypatch,
        retry_from_completion=invalid_first_completion,
    )
    manifest = ubuntu_pipeline.read_object(run_path)
    final_phase = _evaluation_final_phase(manifest)

    verified = ubuntu_pipeline._verify_evaluation_outputs(
        tmp_path,
        "cortex",
        final_phase=final_phase,
    )

    assert verified["status"] == "quality_gate_passed"
    assert verified["initialFormatFailureCount"] == 1
    assert verified["formatRecoveryCount"] == 1
    assert verified["formatFailureCount"] == 0
    candidate_row = ubuntu_pipeline.read_jsonl(
        Path(manifest["candidateOutputsPath"])
    )[0]
    attempts = candidate_row["generationAttempts"]
    assert candidate_row["selectedAttemptIndex"] == 2
    assert [attempt["promptKind"] for attempt in attempts] == [
        "frozen_evaluation",
        "strict_json_retry",
    ]
    assert attempts[0]["formatError"] == "cortex_route_tool_not_in_manifest"
    assert attempts[1]["formatError"] is None
    tool_contracts, _, _ = evaluate_adapter.load_behavior_contract(
        Path(manifest["behaviorManifestPath"])
    )
    record = ubuntu_pipeline.read_jsonl(Path(manifest["evaluationJSONLPath"]))[0]
    primary_messages = evaluate_adapter._structured_output_messages(
        "cortex",
        record["messages"],
        output_mode="json",
        tool_contracts=tool_contracts,
    )
    first_output, _, first_error = evaluate_adapter.normalize_candidate_output(
        "cortex",
        attempts[0]["rawOutput"],
        output_mode="json",
        evaluation_module=evaluate_adapter._load_evaluation_module(),
        tool_contracts=tool_contracts,
    )
    retry_messages = evaluate_adapter._strict_json_retry_messages(
        "cortex",
        primary_messages,
        validation_error=first_error,
        failed_candidate=first_output,
        tool_contracts=tool_contracts,
    )
    assert [attempt["promptSHA256"] for attempt in attempts] == [
        evaluate_adapter._canonical_sha256(primary_messages),
        evaluate_adapter._canonical_sha256(retry_messages),
    ]


@pytest.mark.parametrize(
    "mutation",
    (
        "passed_case_count",
        "critical_failure_count",
        "evidence_complete",
        "promotion_binding",
        "report_agent",
        "report_variant",
    ),
)
def test_evaluation_verifier_independently_rescores_rehashed_forged_reports(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    run_path = _write_evaluation_evidence(tmp_path, monkeypatch=monkeypatch)
    original = ubuntu_pipeline.read_object(run_path)
    final_phase = _evaluation_final_phase(original)

    def forge(report: dict, manifest: dict) -> None:
        if mutation == "passed_case_count":
            report["passedCaseCount"] = 0
        elif mutation == "critical_failure_count":
            report["criticalFailureCount"] = 1
            manifest["criticalFailureCount"] = 1
        elif mutation == "evidence_complete":
            report["evidenceComplete"] = False
        elif mutation == "promotion_binding":
            report["promotionEvidenceBound"] = False
        elif mutation == "report_agent":
            report["agent"] = "executor"
        elif mutation == "report_variant":
            report["variant"] = "internal_only"
        else:  # pragma: no cover - parametrization is closed above.
            raise AssertionError(mutation)

    _rewrite_report_and_enclosing_hashes(run_path, forge)

    with pytest.raises(RuntimeError, match="lineage failed verification"):
        ubuntu_pipeline._verify_evaluation_outputs(
            tmp_path,
            "cortex",
            final_phase=final_phase,
        )


def test_evaluation_verifier_rejects_rehashed_final_config_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_path = _write_evaluation_evidence(tmp_path, monkeypatch=monkeypatch)
    manifest = ubuntu_pipeline.read_object(run_path)
    final_phase = _evaluation_final_phase(manifest)
    config_path = Path(manifest["configPath"])
    config = ubuntu_pipeline.read_object(config_path)
    config["seed"] = 43
    ubuntu_pipeline.write_object(config_path, config)
    manifest["configSHA256"] = ubuntu_pipeline.file_sha256(config_path)
    manifest["generation"]["seed"] = 43
    manifest.pop("runManifestSHA256", None)
    manifest["runManifestSHA256"] = ubuntu_pipeline.canonical_sha256(manifest)
    ubuntu_pipeline.write_object(run_path, manifest)

    with pytest.raises(RuntimeError, match="config or adapter lineage"):
        ubuntu_pipeline._verify_evaluation_outputs(
            tmp_path,
            "cortex",
            final_phase=final_phase,
        )


def test_evaluation_verifier_rejects_rehashed_behavior_manifest_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_path = _write_evaluation_evidence(tmp_path, monkeypatch=monkeypatch)
    manifest = ubuntu_pipeline.read_object(run_path)
    final_phase = _evaluation_final_phase(manifest)
    behavior_path = Path(manifest["behaviorManifestPath"])
    behavior = ubuntu_pipeline.read_object(behavior_path)
    behavior["tools"].append({"id": "files.write", "arguments": []})
    ubuntu_pipeline.write_object(behavior_path, behavior)
    config_path = Path(manifest["configPath"])
    config = ubuntu_pipeline.read_object(config_path)
    config["behaviorManifestFileSHA256"] = ubuntu_pipeline.file_sha256(
        behavior_path
    )
    ubuntu_pipeline.write_object(config_path, config)
    manifest["configSHA256"] = ubuntu_pipeline.file_sha256(config_path)
    _, _, behavior_sha256 = evaluate_adapter.load_behavior_contract(behavior_path)
    manifest["behaviorManifestSHA256"] = behavior_sha256
    manifest.pop("runManifestSHA256", None)
    manifest["runManifestSHA256"] = ubuntu_pipeline.canonical_sha256(manifest)
    ubuntu_pipeline.write_object(run_path, manifest)

    with pytest.raises(RuntimeError, match="exact prepared run"):
        ubuntu_pipeline._verify_evaluation_outputs(
            tmp_path,
            "cortex",
            final_phase=final_phase,
        )


def test_evaluation_verifier_rejects_rehashed_attempt_budget_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_path = _write_evaluation_evidence(tmp_path, monkeypatch=monkeypatch)
    manifest = ubuntu_pipeline.read_object(run_path)
    final_phase = _evaluation_final_phase(manifest)
    candidate_path = Path(manifest["candidateOutputsPath"])
    row = ubuntu_pipeline.read_jsonl(candidate_path)[0]
    attempt = row["generationAttempts"][0]
    attempt["generationTokenBudget"] = 7
    attempt["hitTokenBudget"] = False
    attempt.pop("generationAttemptSHA256", None)
    attempt["generationAttemptSHA256"] = evaluate_adapter._canonical_sha256(
        attempt
    )
    row.pop("candidateRecordSHA256", None)
    row["candidateRecordSHA256"] = evaluate_adapter._canonical_sha256(row)
    candidate_path.write_bytes(evaluate_adapter._jsonl_bytes([row]))
    manifest["candidateOutputsFileSHA256"] = ubuntu_pipeline.file_sha256(
        candidate_path
    )
    manifest.pop("runManifestSHA256", None)
    manifest["runManifestSHA256"] = ubuntu_pipeline.canonical_sha256(manifest)
    ubuntu_pipeline.write_object(run_path, manifest)

    with pytest.raises(RuntimeError, match="lineage failed verification"):
        ubuntu_pipeline._verify_evaluation_outputs(
            tmp_path,
            "cortex",
            final_phase=final_phase,
        )


def test_evaluation_verifier_preserves_valid_smoke_semantics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_path = _write_evaluation_evidence(
        tmp_path,
        monkeypatch=monkeypatch,
        status="smoke_complete",
        quality_gate_passed=False,
        full_case_count=2,
        generated_case_count=1,
    )
    manifest = ubuntu_pipeline.read_object(run_path)

    verified = ubuntu_pipeline._verify_evaluation_outputs(
        tmp_path,
        "cortex",
        final_phase=_evaluation_final_phase(manifest),
    )

    assert verified["status"] == "smoke_complete"
    assert verified["qualityGatePassed"] is False
    report = ubuntu_pipeline.read_object(Path(manifest["evaluationReportPath"]))
    assert report["evaluationSHA256"] == manifest["evaluationSHA256"]
    assert report["variantLineageBound"] is True
    assert report["promotionEvidenceBound"] is False
    assert report["caseCount"] == 1
    assert report["frozenCaseCount"] == 2
    assert report["completeEvaluation"] is False
    assert report["passedCaseCount"] == 1
    assert report["missingOutputCount"] == 0
    assert report["criticalFailureCount"] == 0
    assert report["evidenceComplete"] is True


def test_evaluation_verifier_rejects_smoke_complete_with_a_generated_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selection_only = json.dumps(
        {
            "selectedToolID": "files.read",
            "intent": "files",
            "reasoningSummary": (
                "Manifest row files.read is selected for intent files without "
                "actionStep."
            ),
            "requiresApproval": False,
            "nextModel": "executor",
        },
        separators=(",", ":"),
    )
    run_path = _write_evaluation_evidence(
        tmp_path,
        monkeypatch=monkeypatch,
        status="smoke_complete",
        quality_gate_passed=False,
        completion=selection_only,
        full_case_count=2,
        generated_case_count=1,
    )
    manifest = ubuntu_pipeline.read_object(run_path)
    report = ubuntu_pipeline.read_object(Path(manifest["evaluationReportPath"]))
    assert report["caseCount"] == 1
    assert report["missingOutputCount"] == 0
    assert report["passedCaseCount"] == 0
    assert report["criticalFailureCount"] == 1

    with pytest.raises(RuntimeError, match="lineage failed verification"):
        ubuntu_pipeline._verify_evaluation_outputs(
            tmp_path,
            "cortex",
            final_phase=_evaluation_final_phase(manifest),
        )


def test_evaluation_verifier_reconstructs_non_prefix_semantic_smoke_selection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = []
    for index, prompt in enumerate(
        (
            "Return the alpha structured result.",
            "Return the beta structured result.",
            "Return the gamma structured result.",
        ),
        start=1,
    ):
        record = _evaluation_record(f"eval-{index}")
        record["messages"][-1]["content"] = prompt
        record["metadata"]["name"] = f"semantic-case-{index}"
        records.append(record)
    selected = evaluate_adapter.select_evaluation_records(
        records,
        max_examples=1,
    )[0]
    frozen_records = [
        record for record in records if record["evalID"] != selected["evalID"]
    ] + [selected]
    assert frozen_records[0]["evalID"] != selected["evalID"]

    run_path = _write_evaluation_evidence(
        tmp_path,
        monkeypatch=monkeypatch,
        status="smoke_complete",
        quality_gate_passed=False,
        generated_case_count=1,
        evaluation_records=frozen_records,
    )
    manifest = ubuntu_pipeline.read_object(run_path)
    candidate_rows = ubuntu_pipeline.read_jsonl(
        Path(manifest["candidateOutputsPath"])
    )
    assert [row["evalID"] for row in candidate_rows] == [selected["evalID"]]

    verified = ubuntu_pipeline._verify_evaluation_outputs(
        tmp_path,
        "cortex",
        final_phase=_evaluation_final_phase(manifest),
    )

    assert verified["status"] == "smoke_complete"


def test_evaluation_verifier_rejects_rehashed_smoke_cohort_size_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_path = _write_evaluation_evidence(
        tmp_path,
        monkeypatch=monkeypatch,
        status="smoke_complete",
        quality_gate_passed=False,
        full_case_count=3,
        generated_case_count=1,
    )
    manifest = ubuntu_pipeline.read_object(run_path)
    manifest["generatedCaseCount"] = 2
    manifest.pop("runManifestSHA256", None)
    manifest["runManifestSHA256"] = ubuntu_pipeline.canonical_sha256(manifest)
    ubuntu_pipeline.write_object(run_path, manifest)

    with pytest.raises(RuntimeError, match="case-count lineage failed"):
        ubuntu_pipeline._verify_evaluation_outputs(
            tmp_path,
            "cortex",
            final_phase=_evaluation_final_phase(manifest),
        )


def test_evaluation_verifier_rejects_rehashed_smoke_plan_evidence_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_path = _write_evaluation_evidence(
        tmp_path,
        monkeypatch=monkeypatch,
        status="smoke_complete",
        quality_gate_passed=False,
        full_case_count=3,
        generated_case_count=1,
    )
    manifest = ubuntu_pipeline.read_object(run_path)
    manifest["evaluationMaxExamples"] = 2
    manifest.pop("runManifestSHA256", None)
    manifest["runManifestSHA256"] = ubuntu_pipeline.canonical_sha256(manifest)
    ubuntu_pipeline.write_object(run_path, manifest)

    with pytest.raises(RuntimeError, match="exact prepared run"):
        ubuntu_pipeline._verify_evaluation_outputs(
            tmp_path,
            "cortex",
            final_phase=_evaluation_final_phase(manifest),
        )


def test_evaluation_verifier_leaves_full_evaluation_selection_unchanged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_path = _write_evaluation_evidence(tmp_path, monkeypatch=monkeypatch)
    manifest = ubuntu_pipeline.read_object(run_path)

    def reject_smoke_selection(*_args, **_kwargs):
        raise AssertionError("full evaluation must not apply smoke selection")

    monkeypatch.setattr(
        evaluate_adapter,
        "select_evaluation_records",
        reject_smoke_selection,
    )

    verified = ubuntu_pipeline._verify_evaluation_outputs(
        tmp_path,
        "cortex",
        final_phase=_evaluation_final_phase(manifest),
    )

    assert verified["status"] == "quality_gate_passed"
    report = ubuntu_pipeline.read_object(Path(manifest["evaluationReportPath"]))
    assert report["variantLineageBound"] is True
    assert report["promotionEvidenceBound"] is True
    assert report["caseCount"] == report["frozenCaseCount"] == 1
    assert report["completeEvaluation"] is True


def test_evaluation_verifier_rejects_managed_ancestor_symlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_path = _write_evaluation_evidence(tmp_path, monkeypatch=monkeypatch)
    manifest = ubuntu_pipeline.read_object(run_path)
    original_configs = tmp_path / "configs"
    moved_configs = tmp_path / "real-configs"
    original_configs.rename(moved_configs)
    original_configs.symlink_to(moved_configs, target_is_directory=True)

    with pytest.raises(RuntimeError, match="Managed run path"):
        ubuntu_pipeline._verify_evaluation_outputs(
            tmp_path,
            "cortex",
            final_phase=_evaluation_final_phase(manifest),
        )


@pytest.mark.parametrize(
    "mutation",
    (
        "variant",
        "sft",
        "sft_runtime_model_hash",
        "preference_training_report_hash",
        "training_scope",
        "runtime_smoke_report_hash",
        "missing_gguf",
        "gguf_path",
        "evaluation_report_path",
        "source_integrity",
    ),
)
def test_completed_summary_rejects_rehashed_canonical_evidence_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    summary_path = _write_completed_summary_evidence(
        tmp_path,
        monkeypatch=monkeypatch,
    )
    assert ubuntu_pipeline._verified_completed_summary(
        tmp_path,
        ("cortex",),
    )["status"] == "complete"

    summary = ubuntu_pipeline.read_object(summary_path)
    item = summary["agents"]["cortex"]
    if mutation == "variant":
        summary["variant"] = "internal_only"
    elif mutation == "sft":
        item["sft"] = {"phase": "sft", "adapterSHA256": "0" * 64}
    elif mutation == "sft_runtime_model_hash":
        item["sft"]["runtimeModelBindingSHA256"] = "0" * 64
    elif mutation == "preference_training_report_hash":
        item["finalPhase"]["trainingReportFileSHA256"] = "0" * 64
    elif mutation == "training_scope":
        summary["trainingScope"] = "sft_only"
    elif mutation == "runtime_smoke_report_hash":
        summary["runtimeBindingSmokeReportFileSHA256"] = "0" * 64
    elif mutation == "missing_gguf":
        Path(item["adapterGGUF"]).unlink()
        item["adapterGGUFExists"] = False
        item["adapterGGUFSHA256"] = None
        item["adapterGGUFSizeBytes"] = 0
    elif mutation == "gguf_path":
        item["adapterGGUF"] = str(tmp_path / "other.gguf")
    elif mutation == "evaluation_report_path":
        item["evaluationReport"] = str(tmp_path / "other-report.json")
    elif mutation == "source_integrity":
        summary["ubuntuSourceIntegritySHA256"] = "0" * 64
    else:  # pragma: no cover - parametrization is closed above.
        raise AssertionError(mutation)
    summary.pop("summarySHA256", None)
    summary["summarySHA256"] = ubuntu_pipeline.canonical_sha256(summary)
    ubuntu_pipeline.write_object(summary_path, summary)

    with pytest.raises(RuntimeError):
        ubuntu_pipeline._verified_completed_summary(tmp_path, ("cortex",))


@pytest.mark.parametrize("field", ubuntu_pipeline.ADAPTER_GGUF_SEMANTIC_FIELDS)
def test_completed_summary_rejects_rehashed_gguf_semantic_evidence_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
) -> None:
    summary_path = _write_completed_summary_evidence(
        tmp_path,
        monkeypatch=monkeypatch,
    )
    summary = ubuntu_pipeline.read_object(summary_path)
    item = summary["agents"]["cortex"]
    item[field] = (
        "0" * 64
        if field == "adapterGGUFChatTemplateSHA256"
        else "drifted"
    )
    summary.pop("summarySHA256", None)
    summary["summarySHA256"] = ubuntu_pipeline.canonical_sha256(summary)
    ubuntu_pipeline.write_object(summary_path, summary)

    with pytest.raises(RuntimeError, match="summary GGUF drifted"):
        ubuntu_pipeline._verified_completed_summary(tmp_path, ("cortex",))


def test_completed_summary_rejects_managed_symlink_without_evaluation_recheck(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_completed_summary_evidence(tmp_path, monkeypatch=monkeypatch)
    models = tmp_path / "models"
    real_models = tmp_path / "real-models"
    models.rename(real_models)
    models.symlink_to(real_models, target_is_directory=True)

    with pytest.raises(RuntimeError, match="Managed run path"):
        ubuntu_pipeline._verified_completed_summary(tmp_path, ("cortex",))


def test_resume_rejects_a_minimal_self_hashed_gguf_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gguf = tmp_path / "models" / "lora_qwen3_gguf" / "lumen-cortex-lora.gguf"
    gguf.parent.mkdir(parents=True)
    gguf.write_bytes(_gguf_bytes())
    receipt = (
        tmp_path
        / "models"
        / "lora_qwen3_gguf_receipts"
        / "lumen-cortex-lora.conversion.json"
    )
    receipt.parent.mkdir(parents=True)
    ubuntu_pipeline.write_object(receipt, {"fixture": True})
    summary = {
        "agents": {
            "cortex": {
                "adapterGGUFSHA256": ubuntu_pipeline.file_sha256(gguf),
                "adapterGGUFSizeBytes": gguf.stat().st_size,
            }
        }
    }
    summary["summarySHA256"] = ubuntu_pipeline.canonical_sha256(summary)
    ubuntu_pipeline.write_object(tmp_path / "aio_summary.json", summary)
    monkeypatch.setattr(
        ubuntu_pipeline,
        "_verified_run_manifest",
        lambda *_args: {
            "variant": "internal_plus_public_optimized",
            "executionPlan": _test_execution_plan(),
            "agents": [{"agent": "cortex"}],
        },
    )
    monkeypatch.setattr(
        ubuntu_pipeline,
        "verify_gguf_file",
        lambda *_args: (_ for _ in ()).throw(
            RuntimeError("Missing regular pinned llama.cpp checkout")
        ),
    )

    with pytest.raises(RuntimeError, match="summary failed verification"):
        ubuntu_pipeline.verify_gguf(tmp_path, "cortex")


def test_resume_reuses_gguf_from_a_canonical_summary_with_verified_gguf_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_completed_summary_evidence(tmp_path, monkeypatch=monkeypatch)
    gguf = tmp_path / "models" / "lora_qwen3_gguf" / "lumen-cortex-lora.gguf"

    verified = ubuntu_pipeline.verify_gguf(tmp_path, "cortex")
    assert verified["adapterGGUFSHA256"] == ubuntu_pipeline.file_sha256(gguf)

    gguf.write_bytes(b"GGUF-tampered")
    with pytest.raises(RuntimeError, match="GGUF"):
        ubuntu_pipeline.verify_gguf(tmp_path, "cortex")


def test_resume_rejects_summary_semantics_even_when_digest_and_size_match(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    summary_path = _write_completed_summary_evidence(
        tmp_path,
        monkeypatch=monkeypatch,
    )
    summary = ubuntu_pipeline.read_object(summary_path)
    summary["agents"]["cortex"]["adapterGGUFArchitecture"] = "qwen2"
    monkeypatch.setattr(
        ubuntu_pipeline,
        "_verified_completed_summary",
        lambda *_args: summary,
    )

    with pytest.raises(RuntimeError, match="does not match the completed summary"):
        ubuntu_pipeline.verify_gguf(tmp_path, "cortex")


@pytest.mark.parametrize(
    ("evaluation_scope", "expected_status"),
    [
        ("smoke", "smoke_complete"),
        ("none", "training_complete_without_full_evaluation"),
    ],
)
def test_resume_reuses_verified_gguf_from_diagnostic_summary_states(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    evaluation_scope: str,
    expected_status: str,
) -> None:
    summary_path = _write_completed_summary_evidence(
        tmp_path,
        monkeypatch=monkeypatch,
        evaluation_scope=evaluation_scope,
    )
    assert ubuntu_pipeline.read_object(summary_path)["status"] == expected_status

    verified = ubuntu_pipeline.verify_gguf(tmp_path, "cortex")

    gguf = tmp_path / "models" / "lora_qwen3_gguf" / "lumen-cortex-lora.gguf"
    assert verified["adapterGGUFSHA256"] == ubuntu_pipeline.file_sha256(gguf)


def test_gguf_inventory_requires_exact_prepared_agent_set(
    tmp_path: Path,
) -> None:
    gguf_dir = tmp_path / "models" / "lora_qwen3_gguf"
    receipt_dir = tmp_path / "models" / "lora_qwen3_gguf_receipts"
    gguf_dir.mkdir(parents=True)
    receipt_dir.mkdir(parents=True)
    for agent in ubuntu_pipeline.AGENTS:
        (gguf_dir / f"lumen-{agent}-lora.gguf").write_bytes(_gguf_bytes())
        ubuntu_pipeline.write_object(
            receipt_dir / f"lumen-{agent}-lora.conversion.json",
            {"agent": agent},
        )

    inventory = ubuntu_pipeline._verify_gguf_inventory(
        tmp_path,
        ubuntu_pipeline.AGENTS,
        require_all=True,
    )
    assert set(inventory) == {
        f"lumen-{agent}-lora.gguf" for agent in ubuntu_pipeline.AGENTS
    }

    extra = gguf_dir / "unexpected.gguf"
    extra.write_bytes(_gguf_bytes())
    with pytest.raises(RuntimeError, match="unexpected entries"):
        ubuntu_pipeline._verify_gguf_inventory(
            tmp_path,
            ubuntu_pipeline.AGENTS,
            require_all=True,
        )
    extra.unlink()
    (gguf_dir / "lumen-fleet-lora.gguf").unlink()
    (receipt_dir / "lumen-fleet-lora.conversion.json").unlink()
    with pytest.raises(RuntimeError, match="missing required entries"):
        ubuntu_pipeline._verify_gguf_inventory(
            tmp_path,
            ubuntu_pipeline.AGENTS,
            require_all=True,
        )


def test_completed_summary_rejects_extra_gguf_directory_entry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_completed_summary_evidence(tmp_path, monkeypatch=monkeypatch)
    extra = tmp_path / "models" / "lora_qwen3_gguf" / "extra.gguf"
    extra.write_bytes(_gguf_bytes())

    with pytest.raises(RuntimeError, match="unexpected entries"):
        ubuntu_pipeline._verified_completed_summary(tmp_path, ("cortex",))


def test_required_gguf_summary_fails_without_pinned_reader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    smoke_evidence = _write_runtime_binding_smoke_summary_fixture(tmp_path)
    gguf = tmp_path / "models" / "lora_qwen3_gguf" / "lumen-cortex-lora.gguf"
    gguf.parent.mkdir(parents=True)
    gguf.write_bytes(_gguf_bytes())
    receipt = (
        tmp_path
        / "models"
        / "lora_qwen3_gguf_receipts"
        / "lumen-cortex-lora.conversion.json"
    )
    receipt.parent.mkdir(parents=True)
    ubuntu_pipeline.write_object(receipt, {"fixture": True})
    monkeypatch.setattr(
        ubuntu_pipeline,
        "_verified_run_manifest",
        lambda *_args: {
            "variant": OPTIMIZED_VARIANT,
            "executionPlan": _test_execution_plan(
                evaluation_scope="none",
                gguf_requested=True,
            ),
            "agents": [{"agent": "cortex"}],
            **_summary_base_model_lineage_fixture(),
            **_source_integrity_fixture(),
        },
    )
    monkeypatch.setattr(
        ubuntu_pipeline,
        "_verified_runtime_binding_smoke_summary_evidence",
        lambda *_args: smoke_evidence,
    )
    monkeypatch.setattr(
        ubuntu_pipeline,
        "verify_gguf_file",
        lambda *_args: (_ for _ in ()).throw(
            RuntimeError("Missing regular pinned llama.cpp checkout")
        ),
    )
    monkeypatch.setattr(
        ubuntu_pipeline,
        "verify_sft",
        lambda *_args: {"phase": "sft", **_phase_runtime_evidence_fixture("1")},
    )
    monkeypatch.setattr(
        ubuntu_pipeline,
        "verify_preference",
        lambda *_args: {"phase": "dpo", **_phase_runtime_evidence_fixture("2")},
    )

    with pytest.raises(RuntimeError, match="pinned llama.cpp checkout"):
        ubuntu_pipeline.write_summary(
            run_root=tmp_path,
            agents=("cortex",),
            variant=OPTIMIZED_VARIANT,
            preference=True,
            require_gguf=True,
            require_evaluation=False,
        )


def test_pinned_gguf_reader_requires_exact_clean_revision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout = tmp_path / "llama.cpp"
    reader = checkout / ubuntu_pipeline.GGUF_READER_RELATIVE_PATH
    reader.parent.mkdir(parents=True)
    reader.write_text("print('{}')\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(checkout)], check=True)
    subprocess.run(["git", "-C", str(checkout), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(checkout),
            "-c",
            "user.name=Lumen Test",
            "-c",
            "user.email=lumen-test@example.invalid",
            "commit",
            "-q",
            "-m",
            "fixture",
        ],
        check=True,
    )
    revision = subprocess.check_output(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    monkeypatch.setattr(
        ubuntu_pipeline,
        "DEFAULT_LLAMA_CPP_REVISION",
        revision,
    )

    verified_reader = ubuntu_pipeline._verified_pinned_gguf_reader_script(tmp_path)
    assert verified_reader.path == reader
    assert verified_reader.git_blob_sha1 == subprocess.check_output(
        [
            "git",
            "-C",
            str(checkout),
            "rev-parse",
            f"HEAD:{ubuntu_pipeline.GGUF_READER_RELATIVE_PATH.as_posix()}",
        ],
        text=True,
    ).strip()
    reader.write_text("print('drift')\n", encoding="utf-8")
    gguf = tmp_path / "adapter.gguf"
    gguf.write_bytes(_gguf_bytes())
    with pytest.raises(RuntimeError, match="drifted from the pinned revision"):
        ubuntu_pipeline.verify_gguf_artifact(
            gguf,
            reader_script=verified_reader,
        )
    with pytest.raises(RuntimeError, match="checkout is dirty"):
        ubuntu_pipeline._verified_pinned_gguf_reader_script(tmp_path)


def test_gguf_artifact_verifier_requires_regular_file_size_and_magic(
    tmp_path: Path,
) -> None:
    gguf = tmp_path / "adapter.gguf"
    gguf.write_bytes(_gguf_bytes())
    reader_script = _write_fake_gguf_reader(tmp_path)

    verified = ubuntu_pipeline.verify_gguf_artifact(
        gguf,
        reader_script=reader_script,
    )
    assert verified == {
        "adapterGGUF": str(gguf),
        "adapterGGUFSHA256": ubuntu_pipeline.file_sha256(gguf),
        "adapterGGUFSizeBytes": gguf.stat().st_size,
        "adapterGGUFArchitecture": "qwen3",
        "adapterGGUFType": "adapter",
        "adapterGGUFAdapterType": "lora",
        "adapterGGUFBaseModelID": "Qwen/Qwen3-1.7B",
        "adapterGGUFBaseModelRepoURL": (
            "https://huggingface.co/Qwen/Qwen3-1.7B"
        ),
        "adapterGGUFChatTemplateSource": "shared_base",
        "adapterGGUFChatTemplateSHA256": None,
    }

    gguf.write_bytes(b"NOPE" + _gguf_bytes()[4:])
    with pytest.raises(RuntimeError, match="magic"):
        ubuntu_pipeline.verify_gguf_artifact(gguf, reader_script=reader_script)

    gguf.write_bytes(b"GGUF")
    with pytest.raises(RuntimeError, match="larger"):
        ubuntu_pipeline.verify_gguf_artifact(gguf, reader_script=reader_script)

    target = tmp_path / "target.gguf"
    target.write_bytes(_gguf_bytes())
    gguf.unlink()
    gguf.symlink_to(target)
    with pytest.raises(RuntimeError, match="symlink"):
        ubuntu_pipeline.verify_gguf_artifact(gguf, reader_script=reader_script)


@pytest.mark.parametrize(
    ("key", "mutation", "value", "error"),
    [
        (
            "general.architecture",
            "value",
            "qwen2",
            "semantic metadata drifted",
        ),
        ("general.type", "value", "model", "semantic metadata drifted"),
        ("adapter.type", "value", "ia3", "semantic metadata drifted"),
        (
            "general.base_model.count",
            "value",
            2,
            "exactly one base model",
        ),
        (
            "general.base_model.0.repo_url",
            "value",
            "https://huggingface.co/Qwen/Qwen3-4B",
            "semantic metadata drifted",
        ),
        (
            "general.base_model.count",
            "type",
            "INT32",
            "invalid scalar metadata",
        ),
        (
            "adapter.type",
            "missing",
            None,
            "invalid scalar metadata",
        ),
    ],
)
def test_gguf_artifact_verifier_rejects_semantic_metadata_drift(
    tmp_path: Path,
    key: str,
    mutation: str,
    value: object,
    error: str,
) -> None:
    metadata = _valid_gguf_semantic_metadata()
    if mutation == "missing":
        metadata.pop(key)
    else:
        metadata[key][mutation] = value
    reader_script = _write_fake_gguf_reader(
        tmp_path,
        semantic_metadata=metadata,
    )
    gguf = tmp_path / "adapter.gguf"
    gguf.write_bytes(_gguf_bytes())

    with pytest.raises(RuntimeError, match=error):
        ubuntu_pipeline.verify_gguf_artifact(
            gguf,
            reader_script=reader_script,
        )


def test_gguf_artifact_verifier_records_verified_embedded_chat_template(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chat_template = "{% for message in messages %}{{ message.content }}{% endfor %}"
    expected_sha256 = hashlib.sha256(chat_template.encode("utf-8")).hexdigest()
    monkeypatch.setattr(
        ubuntu_pipeline,
        "PINNED_QWEN3_CHAT_TEMPLATE_SHA256",
        expected_sha256,
    )
    metadata = _valid_gguf_semantic_metadata(chat_template=chat_template)
    reader_script = _write_fake_gguf_reader(
        tmp_path,
        semantic_metadata=metadata,
    )
    gguf = tmp_path / "adapter.gguf"
    gguf.write_bytes(_gguf_bytes(metadata_kv_count=len(metadata)))

    verified = ubuntu_pipeline.verify_gguf_artifact(
        gguf,
        reader_script=reader_script,
    )

    assert verified["adapterGGUFChatTemplateSource"] == "adapter_gguf"
    assert verified["adapterGGUFChatTemplateSHA256"] == expected_sha256


def test_gguf_artifact_verifier_rejects_unpinned_embedded_chat_template(
    tmp_path: Path,
) -> None:
    metadata = _valid_gguf_semantic_metadata(chat_template="drifted-template")
    reader_script = _write_fake_gguf_reader(
        tmp_path,
        semantic_metadata=metadata,
    )
    gguf = tmp_path / "adapter.gguf"
    gguf.write_bytes(_gguf_bytes(metadata_kv_count=len(metadata)))

    with pytest.raises(RuntimeError, match="chat template drifted"):
        ubuntu_pipeline.verify_gguf_artifact(
            gguf,
            reader_script=reader_script,
        )


def test_gguf_artifact_verifier_rejects_header_only_fake(
    tmp_path: Path,
) -> None:
    gguf = tmp_path / "header-only.gguf"
    gguf.write_bytes(_gguf_bytes(payload=b"X"))
    reader_script = _write_fake_gguf_reader(tmp_path)

    with pytest.raises(RuntimeError, match="reader rejected"):
        ubuntu_pipeline.verify_gguf_artifact(
            gguf,
            reader_script=reader_script,
        )


def test_gguf_artifact_verifier_rejects_exponent_overflow_reader_evidence(
    tmp_path: Path,
) -> None:
    gguf = tmp_path / "adapter.gguf"
    gguf.write_bytes(_gguf_bytes())
    reader_script = tmp_path / "overflow_gguf_dump.py"
    reader_script.write_text(
        """from pathlib import Path
import json
import sys

model = Path(sys.argv[1]).resolve()
print(
    '{"filename":' + json.dumps(str(model))
    + ',"endian":"LITTLE","metadata":'
    + '{"GGUF.version":{"strictProbe":1e400},'
    + '"GGUF.tensor_count":{},"GGUF.kv_count":{},"metadata.0":{}},'
    + '"tensors":{"tensor.0":{}}}'
)
""",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="invalid evidence"):
        ubuntu_pipeline.verify_gguf_artifact(
            gguf,
            reader_script=reader_script,
        )


def test_gguf_artifact_verifier_rejects_path_swap_during_reader_execution(
    tmp_path: Path,
) -> None:
    gguf = tmp_path / "adapter.gguf"
    replacement = tmp_path / "replacement.gguf"
    gguf.write_bytes(_gguf_bytes(payload=b"ORIGINAL_UNPARSEABLE_PAYLOAD"))
    replacement.write_bytes(_gguf_bytes())
    reader_script = tmp_path / "swapping_gguf_dump.py"
    reader_script.write_text(
        f"""from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.replace({str(replacement)!r}, {str(gguf)!r})
model = Path(sys.argv[1]).resolve()
data = Path(sys.argv[1]).read_bytes()
if not data.endswith(b"LUMEN_VALID_GGUF_TEST"):
    print("structural GGUF parse failed", file=sys.stderr)
    raise SystemExit(7)
tensor_count = int.from_bytes(data[8:16], "little")
metadata_count = int.from_bytes(data[16:24], "little")
metadata = {{
    "GGUF.version": {{}},
    "GGUF.tensor_count": {{}},
    "GGUF.kv_count": {{}},
}}
metadata.update({{f"metadata.{{index}}": {{}} for index in range(metadata_count)}})
tensors = {{f"tensor.{{index}}": {{}} for index in range(tensor_count)}}
print(json.dumps({{
    "filename": str(model),
    "endian": "LITTLE",
    "metadata": metadata,
    "tensors": tensors,
}}))
""",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="reader rejected|changed while"):
        ubuntu_pipeline.verify_gguf_artifact(
            gguf,
            reader_script=reader_script,
        )


def test_gguf_artifact_verifier_rejects_reader_path_swap_during_execution(
    tmp_path: Path,
) -> None:
    gguf = tmp_path / "adapter.gguf"
    gguf.write_bytes(_gguf_bytes())
    reader_script = tmp_path / "swapped_gguf_dump.py"
    replacement_reader = tmp_path / "replacement_gguf_dump.py"
    replacement_reader.write_text("raise SystemExit(99)\n", encoding="utf-8")
    reader_script.write_text(
        f"""from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.replace({str(replacement_reader)!r}, {str(reader_script)!r})
model = Path(sys.argv[1]).resolve()
data = Path(sys.argv[1]).read_bytes()
tensor_count = int.from_bytes(data[8:16], "little")
metadata_count = int.from_bytes(data[16:24], "little")
metadata = {{
    "GGUF.version": {{}},
    "GGUF.tensor_count": {{}},
    "GGUF.kv_count": {{}},
}}
metadata.update({{f"metadata.{{index}}": {{}} for index in range(metadata_count)}})
tensors = {{f"tensor.{{index}}": {{}} for index in range(tensor_count)}}
print(json.dumps({{
    "filename": str(model),
    "endian": "LITTLE",
    "metadata": metadata,
    "tensors": tensors,
}}))
""",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="reader changed while"):
        ubuntu_pipeline.verify_gguf_artifact(
            gguf,
            reader_script=reader_script,
        )


@pytest.mark.parametrize("version", [2, 3])
def test_gguf_artifact_verifier_accepts_supported_versions(
    tmp_path: Path,
    version: int,
) -> None:
    gguf = tmp_path / f"adapter-v{version}.gguf"
    gguf.write_bytes(_gguf_bytes(version=version))
    reader_script = _write_fake_gguf_reader(tmp_path)

    assert ubuntu_pipeline.verify_gguf_artifact(
        gguf,
        reader_script=reader_script,
    )[
        "adapterGGUFSHA256"
    ] == ubuntu_pipeline.file_sha256(gguf)


@pytest.mark.parametrize(
    "header,error",
    [
        ({"version": 1}, "unsupported version"),
        ({"version": 4}, "unsupported version"),
        ({"tensor_count": 0}, "no tensors"),
        ({"metadata_kv_count": 0}, "no metadata key-values"),
    ],
)
def test_gguf_artifact_verifier_rejects_invalid_fixed_header(
    tmp_path: Path,
    header: dict[str, int],
    error: str,
) -> None:
    gguf = tmp_path / "adapter.gguf"
    gguf.write_bytes(_gguf_bytes(**header))
    reader_script = _write_fake_gguf_reader(tmp_path)

    with pytest.raises(RuntimeError, match=error):
        ubuntu_pipeline.verify_gguf_artifact(gguf, reader_script=reader_script)


def test_completed_summary_rejects_rehashed_non_gguf_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    summary_path = _write_completed_summary_evidence(
        tmp_path,
        monkeypatch=monkeypatch,
    )
    summary = ubuntu_pipeline.read_object(summary_path)
    item = summary["agents"]["cortex"]
    gguf = Path(item["adapterGGUF"])
    gguf.write_bytes(b"NOPE" + _gguf_bytes()[4:])
    item["adapterGGUFSHA256"] = ubuntu_pipeline.file_sha256(gguf)
    item["adapterGGUFSizeBytes"] = gguf.stat().st_size
    summary.pop("summarySHA256", None)
    summary["summarySHA256"] = ubuntu_pipeline.canonical_sha256(summary)
    ubuntu_pipeline.write_object(summary_path, summary)

    with pytest.raises(RuntimeError, match="magic"):
        ubuntu_pipeline._verified_completed_summary(tmp_path, ("cortex",))


def test_resume_gguf_reuse_requires_every_prepared_agent_in_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_completed_summary_evidence(tmp_path, monkeypatch=monkeypatch)
    monkeypatch.setattr(
        ubuntu_pipeline,
        "_verified_run_manifest",
        lambda *_args: {
            "variant": "internal_plus_public_optimized",
            "agents": [{"agent": "cortex"}, {"agent": "executor"}],
        },
    )

    with pytest.raises(RuntimeError):
        ubuntu_pipeline.verify_gguf(tmp_path, "cortex")


def test_upload_snapshot_is_private_and_detached_from_later_host_path_swaps(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    source = run_root / "models" / "adapter.bin"
    source.parent.mkdir(parents=True)
    trusted_payload = b"verified adapter bytes"
    source.write_bytes(trusted_payload)
    snapshot_root = tmp_path / "snapshot"
    snapshot_root.mkdir(mode=0o700)
    contract = ubuntu_pipeline._UploadInputContract(
        relative_path="models/adapter.bin",
        remote_path="runs/run-one/adapters/cortex/adapter.bin",
        expected_sha256=hashlib.sha256(trusted_payload).hexdigest(),
        expected_size=len(trusted_payload),
    )

    snapshotted = ubuntu_pipeline._snapshot_verified_upload_inputs(
        run_root,
        (contract,),
        snapshot_root,
    )
    token = tmp_path / "hf_token"
    token.write_bytes(b"hf_secret_must_not_be_uploaded")
    source.unlink()
    source.symlink_to(token)

    assert len(snapshotted) == 1
    assert snapshotted[0].path.parent == snapshot_root
    assert snapshotted[0].path.read_bytes() == trusted_payload
    assert stat.S_IMODE(snapshotted[0].path.stat().st_mode) == 0o400


def test_upload_snapshot_rejects_digest_drift_before_credentials_are_used(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    source = run_root / "models" / "adapter.bin"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"tampered")
    snapshot_root = tmp_path / "snapshot"
    snapshot_root.mkdir(mode=0o700)
    contract = ubuntu_pipeline._UploadInputContract(
        relative_path="models/adapter.bin",
        remote_path="runs/run-one/adapters/cortex/adapter.bin",
        expected_sha256=hashlib.sha256(b"verified").hexdigest(),
        expected_size=len(b"verified"),
    )

    with pytest.raises(RuntimeError, match="drifted from its verified contract"):
        ubuntu_pipeline._snapshot_verified_upload_inputs(
            run_root,
            (contract,),
            snapshot_root,
        )


def test_upload_snapshot_rejects_symlinked_parent_components(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    payload = b"same bytes do not make a symlink safe"
    (outside / "adapter.bin").write_bytes(payload)
    (run_root / "models").symlink_to(outside, target_is_directory=True)
    snapshot_root = tmp_path / "snapshot"
    snapshot_root.mkdir(mode=0o700)
    contract = ubuntu_pipeline._UploadInputContract(
        relative_path="models/adapter.bin",
        remote_path="runs/run-one/adapters/cortex/adapter.bin",
        expected_sha256=hashlib.sha256(payload).hexdigest(),
        expected_size=len(payload),
    )

    with pytest.raises(RuntimeError, match="without following links"):
        ubuntu_pipeline._snapshot_verified_upload_inputs(
            run_root,
            (contract,),
            snapshot_root,
        )


def test_upload_rejects_a_coherent_adapter_swap_after_summary_verification() -> None:
    old_phase = {
        "phase": "dpo",
        "adapterSHA256": "a" * 64,
        "finalizedVariantManifestSHA256": "b" * 64,
    }
    replacement_phase = {
        "phase": "dpo",
        "adapterSHA256": "c" * 64,
        "finalizedVariantManifestSHA256": "d" * 64,
    }
    summary = {"agents": {"cortex": {"finalPhase": old_phase}}}

    with pytest.raises(RuntimeError, match="drifted from the completed summary"):
        ubuntu_pipeline._verified_upload_final_phase(
            summary,
            "cortex",
            replacement_phase,
        )


def _prepare_minimal_upload_case(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, Any]:
    agent = "cortex"
    run_id = "recoverable-run"
    repo_id = "owner/lumen-adapters"
    source_fields = _source_integrity_fixture()
    source_record = source_fields["ubuntuSourceIntegrity"]
    training_environment = {"trainingEnvironmentSHA256": "e" * 64}
    smoke_evidence = _write_runtime_binding_smoke_summary_fixture(tmp_path)
    runtime_manifest_path = (
        tmp_path / "generated/fine_tuning/adapter_runtime_manifest.json"
    )
    runtime_manifest_path.parent.mkdir(parents=True)
    ubuntu_pipeline.write_object(runtime_manifest_path, {"adapterRepoID": repo_id})
    run_manifest = {
        "runID": run_id,
        "adapterRepoID": repo_id,
        "adapterRuntimeManifestFileSHA256": ubuntu_pipeline.file_sha256(
            runtime_manifest_path
        ),
        "trainingEnvironment": training_environment,
        "agents": [{"agent": agent}],
        **_summary_base_model_lineage_fixture(),
        **source_fields,
    }
    ubuntu_pipeline.write_object(
        tmp_path / "training_environment.json",
        training_environment,
    )
    adapter_dir = tmp_path / "models/lora_qwen3_dpo/cortex"
    adapter_dir.mkdir(parents=True)
    adapter_file = adapter_dir / "adapter_model.safetensors"
    adapter_file.write_bytes(b"recoverable-adapter")
    adapter_payload = {
        "schemaVersion": "lumen.peft-lora-adapter-artifact/1.0.0",
        "artifactType": "peft_lora_directory",
        "trainingPhase": "sft_dpo",
        "parentSFTAdapterSHA256": "9" * 64,
        "files": [
            {
                "path": adapter_file.name,
                "sizeBytes": adapter_file.stat().st_size,
                "sha256": ubuntu_pipeline.file_sha256(adapter_file),
            }
        ],
    }
    adapter_sha = ubuntu_pipeline.canonical_sha256(adapter_payload)
    ubuntu_pipeline.write_object(
        adapter_dir / "adapter_artifact_manifest.json",
        {**adapter_payload, "adapterSHA256": adapter_sha},
    )
    finalized = tmp_path / "training/cortex/dpo/finalized_variant_manifest.json"
    finalized.parent.mkdir(parents=True)
    finalized_payload = {"agent": agent, "trainingPhase": "sft_dpo"}
    finalized_sha = ubuntu_pipeline.canonical_sha256(finalized_payload)
    ubuntu_pipeline.write_object(
        finalized,
        {**finalized_payload, "variantManifestSHA256": finalized_sha},
    )
    sft_record = {
        "phase": "sft",
        "adapterSHA256": "9" * 64,
        **_write_phase_report_fixture(
            tmp_path,
            agent,
            preference=False,
            digest_character="1",
        ),
    }
    preference_record = {
        "phase": "dpo",
        "adapterSHA256": adapter_sha,
        "finalizedVariantManifestSHA256": finalized_sha,
        **_write_phase_report_fixture(
            tmp_path,
            agent,
            preference=True,
            digest_character="2",
        ),
    }
    summary = {
        "status": "complete_without_gguf",
        "evaluationStatus": "quality_gate_passed",
        "evaluationScope": "full",
        "ggufStatus": "skipped_by_operator",
        "ggufConversionStatus": "skipped_by_operator",
        "ggufTensorEquivalenceStatus": "not_applicable",
        "qualification": "quality_gate_passed",
        "promotionEligible": True,
        "preferenceTraining": True,
        "trainingScope": "sft_preference",
        **smoke_evidence,
        "executionPlanSHA256": _test_execution_plan(
            gguf_requested=False
        )["executionPlanSHA256"],
        **_summary_base_model_lineage_fixture(),
        "agents": {
            agent: {
                "sft": sft_record,
                "finalPhase": preference_record,
                "evaluation": None,
                "adapterGGUFExists": False,
            }
        },
    }
    ubuntu_pipeline.write_object(tmp_path / "aio_run_manifest.json", run_manifest)
    ubuntu_pipeline.write_object(tmp_path / "aio_summary.json", summary)
    monkeypatch.setattr(
        ubuntu_pipeline,
        "_verified_run_manifest",
        lambda *_args: run_manifest,
    )
    monkeypatch.setattr(
        ubuntu_pipeline,
        "_verified_completed_summary",
        lambda *_args: summary,
    )
    monkeypatch.setattr(
        ubuntu_pipeline,
        "_verified_runtime_binding_smoke_summary_evidence",
        lambda *_args: smoke_evidence,
    )
    monkeypatch.setattr(
        ubuntu_pipeline,
        "current_source_integrity",
        lambda *_args: source_record,
    )
    monkeypatch.setattr(
        ubuntu_pipeline,
        "verify_sft",
        lambda *_args: sft_record,
    )
    monkeypatch.setattr(
        ubuntu_pipeline,
        "verify_preference",
        lambda *_args: preference_record,
    )
    token = tmp_path / "token"
    token.write_text("hf_test_token\n", encoding="utf-8")
    receipt_dir = tmp_path / "receipts"
    receipt_dir.mkdir(mode=0o700)
    return {
        "agent": agent,
        "run_id": run_id,
        "repo_id": repo_id,
        "token": token,
        "receipt": receipt_dir / "upload.json",
    }


def _install_fake_transactional_upload_hub(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, Any]:
    parent = "1" * 40
    state: dict[str, Any] = {
        "head": parent,
        "private": True,
        "trees": {parent: {}},
        "commits": [SimpleNamespace(commit_id=parent, title="Initial commit")],
        "create_calls": 0,
    }
    download_root = tmp_path / "remote-downloads"
    download_root.mkdir()

    class _Operation:
        def __init__(self, *, path_in_repo: str, path_or_fileobj: str) -> None:
            self.path_in_repo = path_in_repo
            self.path_or_fileobj = path_or_fileobj

    class _API:
        def __init__(self, *, token: str) -> None:
            assert token == "hf_test_token"

        def whoami(self) -> dict[str, str]:
            return {"name": "lumen-test"}

        def create_repo(self, **_kwargs: Any) -> None:
            return None

        def repo_info(self, **_kwargs: Any) -> SimpleNamespace:
            return SimpleNamespace(private=state["private"], sha=state["head"])

        def list_repo_files(self, *, revision: str | None = None, **_kwargs: Any) -> list[str]:
            selected = revision or state["head"]
            return sorted(state["trees"][selected])

        def list_repo_commits(
            self,
            *,
            revision: str | None = None,
            **_kwargs: Any,
        ) -> list[SimpleNamespace]:
            commits = list(state["commits"])
            if revision is None:
                return commits
            for index, commit in enumerate(commits):
                if commit.commit_id == revision:
                    return commits[index:]
            return []

        def create_commit(
            self,
            *,
            operations: list[_Operation],
            commit_message: str,
            parent_commit: str | None,
            **_kwargs: Any,
        ) -> SimpleNamespace:
            assert parent_commit == state["head"]
            state["create_calls"] += 1
            commit_oid = "2" * 40
            tree = dict(state["trees"][state["head"]])
            for operation in operations:
                tree[operation.path_in_repo] = Path(
                    operation.path_or_fileobj
                ).read_bytes()
            state["trees"][commit_oid] = tree
            state["head"] = commit_oid
            state["commits"].insert(
                0,
                SimpleNamespace(
                    commit_id=commit_oid,
                    title=commit_message,
                    parents=[] if parent_commit is None else [parent_commit],
                ),
            )
            return SimpleNamespace(oid=commit_oid)

    def hf_hub_download(
        *,
        filename: str,
        revision: str,
        **_kwargs: Any,
    ) -> str:
        payload = state["trees"][revision][filename]
        destination = download_root / hashlib.sha256(
            f"{revision}:{filename}".encode("utf-8")
        ).hexdigest()
        destination.write_bytes(payload)
        return str(destination)

    hub = ModuleType("huggingface_hub")
    hub.HfApi = _API
    hub.CommitOperationAdd = _Operation
    hub.hf_hub_download = hf_hub_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    return state


def test_upload_receipt_binds_verified_image_source_and_separate_write_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = "cortex"
    run_id = "run-one"
    source_fields = _source_integrity_fixture()
    source_record = source_fields["ubuntuSourceIntegrity"]
    training_environment = {"trainingEnvironmentSHA256": "e" * 64}
    smoke_evidence = _write_runtime_binding_smoke_summary_fixture(tmp_path)
    runtime_manifest_path = (
        tmp_path / "generated/fine_tuning/adapter_runtime_manifest.json"
    )
    runtime_manifest_path.parent.mkdir(parents=True)
    ubuntu_pipeline.write_object(
        runtime_manifest_path,
        {"adapterRepoID": "owner/lumen-adapters"},
    )
    run_manifest = {
        "runID": run_id,
        "adapterRepoID": "owner/lumen-adapters",
        "adapterRuntimeManifestFileSHA256": ubuntu_pipeline.file_sha256(
            runtime_manifest_path
        ),
        "trainingEnvironment": training_environment,
        "agents": [{"agent": agent}],
        **_summary_base_model_lineage_fixture(),
        **source_fields,
    }
    ubuntu_pipeline.write_object(
        tmp_path / "training_environment.json",
        training_environment,
    )
    adapter_dir = tmp_path / "models/lora_qwen3_dpo/cortex"
    adapter_dir.mkdir(parents=True)
    adapter_file = adapter_dir / "adapter_model.safetensors"
    adapter_file.write_bytes(b"adapter")
    adapter_payload = {
        "schemaVersion": "lumen.peft-lora-adapter-artifact/1.0.0",
        "artifactType": "peft_lora_directory",
        "trainingPhase": "sft_dpo",
        "parentSFTAdapterSHA256": "9" * 64,
        "files": [
            {
                "path": adapter_file.name,
                "sizeBytes": adapter_file.stat().st_size,
                "sha256": ubuntu_pipeline.file_sha256(adapter_file),
            }
        ],
    }
    adapter_sha = ubuntu_pipeline.canonical_sha256(adapter_payload)
    ubuntu_pipeline.write_object(
        adapter_dir / "adapter_artifact_manifest.json",
        {**adapter_payload, "adapterSHA256": adapter_sha},
    )
    finalized = tmp_path / "training/cortex/dpo/finalized_variant_manifest.json"
    finalized.parent.mkdir(parents=True)
    finalized_payload = {"agent": agent, "trainingPhase": "sft_dpo"}
    finalized_sha = ubuntu_pipeline.canonical_sha256(finalized_payload)
    ubuntu_pipeline.write_object(
        finalized,
        {**finalized_payload, "variantManifestSHA256": finalized_sha},
    )
    sft_record = {
        "phase": "sft",
        "adapterSHA256": "9" * 64,
        **_write_phase_report_fixture(
            tmp_path,
            agent,
            preference=False,
            digest_character="1",
        ),
    }
    preference_record = {
        "phase": "dpo",
        "adapterSHA256": adapter_sha,
        "finalizedVariantManifestSHA256": finalized_sha,
        **_write_phase_report_fixture(
            tmp_path,
            agent,
            preference=True,
            digest_character="2",
        ),
    }
    summary = {
        "status": "complete_without_gguf",
        "evaluationStatus": "quality_gate_passed",
            "evaluationScope": "full",
            "ggufStatus": "skipped_by_operator",
            "ggufConversionStatus": "skipped_by_operator",
            "ggufTensorEquivalenceStatus": "not_applicable",
        "qualification": "quality_gate_passed",
        "promotionEligible": True,
        "preferenceTraining": True,
        "trainingScope": "sft_preference",
        **smoke_evidence,
        "executionPlanSHA256": _test_execution_plan(
            gguf_requested=False
        )["executionPlanSHA256"],
        **_summary_base_model_lineage_fixture(),
        "agents": {
            agent: {
                "sft": sft_record,
                "finalPhase": preference_record,
                "evaluation": None,
                "adapterGGUFExists": False,
            }
        }
    }
    ubuntu_pipeline.write_object(
        tmp_path / "aio_run_manifest.json",
        run_manifest,
    )
    ubuntu_pipeline.write_object(tmp_path / "aio_summary.json", summary)
    monkeypatch.setattr(
        ubuntu_pipeline,
        "_verified_run_manifest",
        lambda *_args: run_manifest,
    )
    monkeypatch.setattr(
        ubuntu_pipeline,
        "_verified_completed_summary",
        lambda *_args: summary,
    )
    monkeypatch.setattr(
        ubuntu_pipeline,
        "_verified_runtime_binding_smoke_summary_evidence",
        lambda *_args: smoke_evidence,
    )
    monkeypatch.setattr(
        ubuntu_pipeline,
        "current_source_integrity",
        lambda *_args: source_record,
    )
    monkeypatch.setattr(
        ubuntu_pipeline,
        "verify_sft",
        lambda *_args: sft_record,
    )
    monkeypatch.setattr(
        ubuntu_pipeline,
        "verify_preference",
        lambda *_args: preference_record,
    )

    class _Info:
        private = True
        sha = "b" * 40

    class _Commit:
        oid = "b" * 40

    class _API:
        def __init__(self, **_kwargs) -> None:
            pass

        def whoami(self) -> dict:
            return {"name": "lumen-test"}

        def create_repo(self, **_kwargs) -> None:
            return None

        def repo_info(self, **_kwargs) -> _Info:
            return _Info()

        def list_repo_files(self, **_kwargs) -> list[str]:
            return []

        def create_commit(self, **_kwargs) -> _Commit:
            return _Commit()

    class _Operation:
        def __init__(self, **_kwargs) -> None:
            pass

    hub = ModuleType("huggingface_hub")
    hub.HfApi = _API
    hub.CommitOperationAdd = _Operation
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    token = tmp_path / "token"
    token.write_text("hf_test_token\n", encoding="utf-8")
    receipt_dir = tmp_path / "receipts"
    receipt_dir.mkdir()
    receipt_dir.chmod(0o700)
    receipt_path = receipt_dir / "upload.json"

    receipt = ubuntu_pipeline.upload_run(
        run_root=tmp_path,
        agents=(agent,),
        run_id=run_id,
        private=True,
        include_gguf=False,
        token_file=token,
        receipt_path=receipt_path,
    )

    assert receipt_path.is_file()
    assert not (tmp_path / "upload_receipts.json").exists()
    for field, expected in source_fields.items():
        assert receipt[field] == expected


def test_upload_recovers_commit_after_post_commit_crash_and_head_advancement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    upload = _prepare_minimal_upload_case(tmp_path, monkeypatch)
    state = _install_fake_transactional_upload_hub(tmp_path, monkeypatch)
    original_write_once = ubuntu_pipeline._write_once_upload_record
    crash = {"armed": True}

    def crash_before_commit_record(
        path: Path,
        record: Mapping[str, Any],
        **kwargs: Any,
    ) -> dict[str, Any]:
        if path.name == ubuntu_pipeline.UPLOAD_COMMIT_FILENAME and crash["armed"]:
            crash["armed"] = False
            raise RuntimeError("simulated crash after remote commit")
        return original_write_once(path, record, **kwargs)

    monkeypatch.setattr(
        ubuntu_pipeline,
        "_write_once_upload_record",
        crash_before_commit_record,
    )
    with pytest.raises(RuntimeError, match="simulated crash after remote commit"):
        ubuntu_pipeline.upload_run(
            run_root=tmp_path,
            agents=(upload["agent"],),
            run_id=upload["run_id"],
            private=True,
            include_gguf=False,
            token_file=upload["token"],
            receipt_path=upload["receipt"],
        )
    assert state["create_calls"] == 1
    assert not upload["receipt"].exists()
    assert (upload["receipt"].parent / ubuntu_pipeline.UPLOAD_INTENT_FILENAME).is_file()
    assert (upload["receipt"].parent / ubuntu_pipeline.UPLOAD_ATTEMPT_FILENAME).is_file()
    assert not (
        upload["receipt"].parent / ubuntu_pipeline.UPLOAD_COMMIT_FILENAME
    ).exists()

    transaction_oid = state["head"]
    advanced_oid = "3" * 40
    advanced_tree = dict(state["trees"][transaction_oid])
    advanced_tree["unrelated/other-run.txt"] = b"later unrelated commit"
    state["trees"][advanced_oid] = advanced_tree
    state["commits"].insert(
        0,
        SimpleNamespace(
            commit_id=advanced_oid,
            title="Unrelated later upload",
            parents=[transaction_oid],
        ),
    )
    state["head"] = advanced_oid
    monkeypatch.setattr(
        ubuntu_pipeline,
        "_write_once_upload_record",
        original_write_once,
    )

    receipt = ubuntu_pipeline.upload_run(
        run_root=tmp_path,
        agents=(upload["agent"],),
        run_id=upload["run_id"],
        private=True,
        include_gguf=False,
        token_file=upload["token"],
        receipt_path=upload["receipt"],
    )

    assert state["create_calls"] == 1
    assert receipt["commitOID"] == transaction_oid
    assert receipt["headRevision"] == advanced_oid
    assert receipt["remoteVerification"] == "recovered_exact_remote_tree"
    assert receipt["remotePrefix"].endswith(f"{upload['run_id']}/")
    assert any(
        path.endswith(ubuntu_pipeline.UPLOAD_REMOTE_MARKER_FILENAME)
        for path in receipt["uploadedPaths"]
    )
    assert upload["receipt"].is_file()
    for filename in (
        ubuntu_pipeline.UPLOAD_INTENT_FILENAME,
        ubuntu_pipeline.UPLOAD_ATTEMPT_FILENAME,
        ubuntu_pipeline.UPLOAD_COMMIT_FILENAME,
    ):
        assert not (upload["receipt"].parent / filename).exists()

    reverified = ubuntu_pipeline.upload_run(
        run_root=tmp_path,
        agents=(upload["agent"],),
        run_id=upload["run_id"],
        private=True,
        include_gguf=False,
        token_file=upload["token"],
        receipt_path=upload["receipt"],
    )
    assert reverified == receipt
    assert state["create_calls"] == 1

    protected_path = next(
        path
        for path in receipt["uploadedPaths"]
        if not path.endswith(ubuntu_pipeline.UPLOAD_REMOTE_MARKER_FILENAME)
    )
    state["trees"][advanced_oid][protected_path] = b"tampered at current head"
    with pytest.raises(RuntimeError, match="current head content failed verification"):
        ubuntu_pipeline.upload_run(
            run_root=tmp_path,
            agents=(upload["agent"],),
            run_id=upload["run_id"],
            private=True,
            include_gguf=False,
            token_file=upload["token"],
            receipt_path=upload["receipt"],
        )


def test_upload_never_adopts_an_unjournaled_existing_remote_prefix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    upload = _prepare_minimal_upload_case(tmp_path, monkeypatch)
    state = _install_fake_transactional_upload_hub(tmp_path, monkeypatch)
    prefix = f"runs/{upload['run_id']}/"
    state["trees"][state["head"]][f"{prefix}forged.bin"] = b"foreign"

    with pytest.raises(
        RuntimeError,
        match="without a durable local upload attempt",
    ):
        ubuntu_pipeline.upload_run(
            run_root=tmp_path,
            agents=(upload["agent"],),
            run_id=upload["run_id"],
            private=True,
            include_gguf=False,
            token_file=upload["token"],
            receipt_path=upload["receipt"],
        )
    assert state["create_calls"] == 0
    assert not upload["receipt"].exists()


def test_upload_cli_is_private_unless_public_is_explicit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ubuntu_pipeline.py",
            "upload",
            "--run-root",
            str(tmp_path),
            "--agents",
            "cortex",
            "--run-id",
            "run-one",
            "--token-file",
            str(tmp_path / "token"),
        ],
    )
    parsed = ubuntu_pipeline.parse_args()
    assert parsed.public is False
    assert parsed.allow_diagnostic_upload is False

    sys.argv.append("--public")
    assert ubuntu_pipeline.parse_args().public is True

    sys.argv.append("--allow-diagnostic-upload")
    assert ubuntu_pipeline.parse_args().allow_diagnostic_upload is True


def test_sft_only_upload_uses_only_sft_paths_and_requires_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = "cortex"
    run_id = "sft-only-run"
    repo_id = "owner/lumen-adapters"
    source_fields = _source_integrity_fixture()
    smoke_evidence = _write_runtime_binding_smoke_summary_fixture(tmp_path)
    runtime_manifest = (
        tmp_path / "generated" / "fine_tuning" / "adapter_runtime_manifest.json"
    )
    runtime_manifest.parent.mkdir(parents=True)
    ubuntu_pipeline.write_object(runtime_manifest, {"adapterRepoID": repo_id})
    training_environment: dict[str, Any] = {}
    ubuntu_pipeline.write_object(
        tmp_path / "training_environment.json",
        training_environment,
    )
    adapter_dir = tmp_path / "models" / "lora_qwen3_bootstrap" / agent
    adapter_dir.mkdir(parents=True)
    adapter_file = adapter_dir / "adapter_model.safetensors"
    adapter_file.write_bytes(b"sft-only-adapter")
    adapter_payload = {
        "schemaVersion": "lumen.peft-lora-adapter-artifact/1.0.0",
        "artifactType": "peft_lora_directory",
        "trainingPhase": "sft",
        "parentSFTAdapterSHA256": None,
        "files": [
            {
                "path": adapter_file.name,
                "sizeBytes": adapter_file.stat().st_size,
                "sha256": ubuntu_pipeline.file_sha256(adapter_file),
            }
        ],
    }
    adapter_sha = ubuntu_pipeline.canonical_sha256(adapter_payload)
    ubuntu_pipeline.write_object(
        adapter_dir / "adapter_artifact_manifest.json",
        {**adapter_payload, "adapterSHA256": adapter_sha},
    )
    finalized = tmp_path / "training" / agent / "finalized_variant_manifest.json"
    finalized.parent.mkdir(parents=True)
    finalized_payload = {"agent": agent, "trainingPhase": "sft"}
    finalized_sha = ubuntu_pipeline.canonical_sha256(finalized_payload)
    ubuntu_pipeline.write_object(
        finalized,
        {**finalized_payload, "variantManifestSHA256": finalized_sha},
    )
    sft_record = {
        "phase": "sft",
        "adapterSHA256": adapter_sha,
        "finalizedVariantManifestSHA256": finalized_sha,
        **_write_phase_report_fixture(
            tmp_path,
            agent,
            preference=False,
            digest_character="1",
        ),
    }
    plan = _test_execution_plan(evaluation_scope="none", gguf_requested=False)
    summary = {
        "status": "sft_only_diagnostic_complete",
        "trainingScope": "sft_only",
        "preferenceTraining": False,
        "evaluationStatus": "not_run",
        "evaluationScope": "none",
        "ggufStatus": "not_applicable_sft_only",
        "ggufConversionStatus": "not_applicable",
        "ggufTensorEquivalenceStatus": "not_applicable",
        "qualification": "diagnostic_only",
        "promotionEligible": False,
        **smoke_evidence,
        "executionPlanSHA256": plan["executionPlanSHA256"],
        **_summary_base_model_lineage_fixture(),
        "agents": {agent: {"sft": sft_record, "finalPhase": sft_record}},
    }
    run_manifest = {
        "runID": run_id,
        "agents": [{"agent": agent}],
        "adapterRepoID": repo_id,
        "adapterRuntimeManifestFileSHA256": ubuntu_pipeline.file_sha256(
            runtime_manifest
        ),
        "trainingEnvironment": training_environment,
        **_summary_base_model_lineage_fixture(),
        **source_fields,
    }
    ubuntu_pipeline.write_object(tmp_path / "aio_run_manifest.json", run_manifest)
    ubuntu_pipeline.write_object(tmp_path / "aio_summary.json", summary)
    monkeypatch.setattr(
        ubuntu_pipeline,
        "_verified_run_manifest",
        lambda *_args: run_manifest,
    )
    monkeypatch.setattr(
        ubuntu_pipeline,
        "_verified_completed_summary",
        lambda *_args: summary,
    )
    monkeypatch.setattr(
        ubuntu_pipeline,
        "_verified_runtime_binding_smoke_summary_evidence",
        lambda *_args: smoke_evidence,
    )
    monkeypatch.setattr(
        ubuntu_pipeline,
        "current_source_integrity",
        lambda *_args: source_fields["ubuntuSourceIntegrity"],
    )
    monkeypatch.setattr(ubuntu_pipeline, "verify_sft", lambda *_args: sft_record)

    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("SFT-only upload attempted preference verification")

    monkeypatch.setattr(ubuntu_pipeline, "verify_preference", forbidden)
    state = _install_fake_transactional_upload_hub(tmp_path, monkeypatch)
    token = tmp_path / "token"
    token.write_text("hf_test_token\n", encoding="utf-8")
    receipt_dir = tmp_path / "receipts"
    receipt_dir.mkdir(mode=0o700)
    receipt_path = receipt_dir / "upload.json"

    with pytest.raises(RuntimeError, match="--allow-diagnostic-upload"):
        ubuntu_pipeline.upload_run(
            run_root=tmp_path,
            agents=(agent,),
            run_id=run_id,
            private=True,
            include_gguf=False,
            token_file=token,
            receipt_path=receipt_path,
        )
    receipt = ubuntu_pipeline.upload_run(
        run_root=tmp_path,
        agents=(agent,),
        run_id=run_id,
        private=True,
        include_gguf=False,
        token_file=token,
        allow_diagnostic_upload=True,
        receipt_path=receipt_path,
    )

    assert state["create_calls"] == 1
    assert receipt["remoteNamespace"] == "diagnostic-sft-runs"
    assert receipt["trainingScope"] == "sft_only"
    assert receipt["preferenceTraining"] is False
    assert receipt["phaseRuntimeEvidenceByAgent"] == {
        agent: {"sft": ubuntu_pipeline._compact_phase_runtime_evidence(sft_record)}
    }
    assert any(
        path.endswith(f"/adapters/{agent}/adapter_model.safetensors")
        for path in receipt["uploadedPaths"]
    )
    assert any(
        path.endswith(f"/manifests/{agent}/sft_training_report.json")
        for path in receipt["uploadedPaths"]
    )
    assert not any(
        segment in path
        for path in receipt["uploadedPaths"]
        for segment in ("/preference_", "/evaluation/", "/gguf/")
    )


def test_diagnostic_upload_receipt_binds_override_prefix_and_artifact_scope(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = "cortex"
    run_id = "diagnostic-run"
    repo_id = "lumen-owner/lumen-adapters"
    source_fields = _source_integrity_fixture()
    smoke_evidence = _write_runtime_binding_smoke_summary_fixture(tmp_path)
    source_record = source_fields["ubuntuSourceIntegrity"]
    runtime_manifest = (
        tmp_path / "generated" / "fine_tuning" / "adapter_runtime_manifest.json"
    )
    runtime_manifest.parent.mkdir(parents=True)
    ubuntu_pipeline.write_object(runtime_manifest, {"adapterRepoID": repo_id})
    ubuntu_pipeline.write_object(tmp_path / "training_environment.json", {})
    ubuntu_pipeline.write_object(tmp_path / "aio_run_manifest.json", {})
    ubuntu_pipeline.write_object(tmp_path / "aio_summary.json", {})
    adapter_dir = tmp_path / "models" / "lora_qwen3_dpo" / agent
    adapter_dir.mkdir(parents=True)
    adapter_file = adapter_dir / "adapter_model.safetensors"
    adapter_file.write_bytes(b"adapter")
    adapter_payload = {
        "schemaVersion": "lumen.peft-lora-adapter-artifact/1.0.0",
        "artifactType": "peft_lora_directory",
        "trainingPhase": "sft_dpo",
        "parentSFTAdapterSHA256": "9" * 64,
        "files": [
            {
                "path": adapter_file.name,
                "sizeBytes": adapter_file.stat().st_size,
                "sha256": ubuntu_pipeline.file_sha256(adapter_file),
            }
        ],
    }
    adapter_sha = ubuntu_pipeline.canonical_sha256(adapter_payload)
    ubuntu_pipeline.write_object(
        adapter_dir / "adapter_artifact_manifest.json",
        {**adapter_payload, "adapterSHA256": adapter_sha},
    )
    finalized = tmp_path / "training" / agent / "dpo" / "finalized_variant_manifest.json"
    finalized.parent.mkdir(parents=True)
    finalized_payload = {"agent": agent, "trainingPhase": "sft_dpo"}
    finalized_sha = ubuntu_pipeline.canonical_sha256(finalized_payload)
    ubuntu_pipeline.write_object(
        finalized,
        {**finalized_payload, "variantManifestSHA256": finalized_sha},
    )
    sft_record = {
        "phase": "sft",
        "adapterSHA256": "9" * 64,
        **_write_phase_report_fixture(
            tmp_path,
            agent,
            preference=False,
            digest_character="1",
        ),
    }
    preference_record = {
        "phase": "dpo",
        "adapterSHA256": adapter_sha,
        "finalizedVariantManifestSHA256": finalized_sha,
        **_write_phase_report_fixture(
            tmp_path,
            agent,
            preference=True,
            digest_character="2",
        ),
    }
    token_file = tmp_path / "token"
    token_file.write_text("hf_test_token\n", encoding="utf-8")

    summary = {
        "status": "smoke_complete",
        "evaluationStatus": "smoke_complete",
            "evaluationScope": "smoke",
            "ggufStatus": "skipped_by_operator",
            "ggufConversionStatus": "skipped_by_operator",
            "ggufTensorEquivalenceStatus": "not_applicable",
        "qualification": "diagnostic_only",
        "promotionEligible": False,
        "preferenceTraining": True,
        "trainingScope": "sft_preference",
        **smoke_evidence,
        "executionPlanSHA256": _test_execution_plan(
            evaluation_scope="smoke",
            evaluation_max_examples=1,
            gguf_requested=False,
        )["executionPlanSHA256"],
        **_summary_base_model_lineage_fixture(),
        "agents": {
            agent: {
                "sft": sft_record,
                "finalPhase": preference_record,
                "evaluation": None,
                "adapterGGUFExists": False,
            }
        },
    }
    run_manifest = {
        "runID": run_id,
        "agents": [{"agent": agent}],
        "adapterRepoID": repo_id,
        "adapterRuntimeManifestFileSHA256": ubuntu_pipeline.file_sha256(
            runtime_manifest
        ),
        "trainingEnvironment": {},
        **_summary_base_model_lineage_fixture(),
        **source_fields,
    }
    ubuntu_pipeline.write_object(tmp_path / "aio_run_manifest.json", run_manifest)
    ubuntu_pipeline.write_object(tmp_path / "aio_summary.json", summary)
    monkeypatch.setattr(
        ubuntu_pipeline,
        "_verified_run_manifest",
        lambda *_args: run_manifest,
    )
    monkeypatch.setattr(
        ubuntu_pipeline,
        "_verified_completed_summary",
        lambda *_args: summary,
    )
    monkeypatch.setattr(
        ubuntu_pipeline,
        "_verified_runtime_binding_smoke_summary_evidence",
        lambda *_args: smoke_evidence,
    )
    monkeypatch.setattr(
        ubuntu_pipeline,
        "verify_sft",
        lambda *_args: sft_record,
    )
    monkeypatch.setattr(
        ubuntu_pipeline,
        "verify_preference",
        lambda *_args: preference_record,
    )
    monkeypatch.setattr(
        ubuntu_pipeline,
        "current_source_integrity",
        lambda *_args: source_record,
    )

    swap_state = {"done": False}

    class FakeCommitOperationAdd:
        def __init__(self, *, path_in_repo: str, path_or_fileobj: str) -> None:
            self.path_in_repo = path_in_repo
            self.path_or_fileobj = path_or_fileobj
            if not swap_state["done"]:
                adapter_file.unlink()
                adapter_file.symlink_to(token_file)
                swap_state["done"] = True

    class FakeHfApi:
        def __init__(self, *, token: str) -> None:
            assert token == "hf_test_token"
            self.committed = False

        def whoami(self) -> dict:
            return {"name": "lumen-owner"}

        def create_repo(self, **_kwargs) -> None:
            return None

        def repo_info(self, **_kwargs) -> SimpleNamespace:
            revision = "2" * 40 if self.committed else "1" * 40
            return SimpleNamespace(private=True, sha=revision)

        def list_repo_files(self, **_kwargs) -> list[str]:
            return []

        def create_commit(self, *, operations, **_kwargs) -> SimpleNamespace:
            assert operations
            assert all(
                operation.path_in_repo.startswith(
                    "diagnostic-runs/diagnostic-run/"
                )
                for operation in operations
            )
            assert all(
                not Path(operation.path_or_fileobj).is_relative_to(tmp_path)
                for operation in operations
            )
            uploaded = {
                operation.path_in_repo: Path(operation.path_or_fileobj).read_bytes()
                for operation in operations
            }
            assert uploaded[
                "diagnostic-runs/diagnostic-run/adapters/cortex/adapter_model.safetensors"
            ] == b"adapter"
            assert token_file.read_bytes() not in uploaded.values()
            self.committed = True
            return SimpleNamespace(oid="2" * 40)

    fake_hub = ModuleType("huggingface_hub")
    fake_hub.CommitOperationAdd = FakeCommitOperationAdd
    fake_hub.HfApi = FakeHfApi
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)

    receipt = ubuntu_pipeline.upload_run(
        run_root=tmp_path,
        agents=(agent,),
        run_id=run_id,
        private=True,
        include_gguf=False,
        token_file=token_file,
        allow_diagnostic_upload=True,
    )

    assert receipt["schema"] == ubuntu_pipeline.UPLOAD_SCHEMA_VERSION
    assert receipt["remoteNamespace"] == "diagnostic-runs"
    assert receipt["remotePrefix"] == "diagnostic-runs/diagnostic-run/"
    assert receipt["qualification"] == "diagnostic_only"
    assert receipt["promotionEligible"] is False
    assert receipt["diagnosticUploadOverrideApplied"] is True
    assert receipt["evaluationStatus"] == "smoke_complete"
    assert receipt["evaluationScope"] == "smoke"
    assert receipt["ggufStatus"] == "skipped_by_operator"
    assert receipt["ggufIncluded"] is False
    assert swap_state["done"] is True
    assert ubuntu_pipeline.read_object(tmp_path / "upload_receipts.json") == receipt


def test_trainers_publish_only_exact_base_tokenizer_bytes() -> None:
    for filename in ("train_sft.py", "train_dpo.py"):
        source = (
            REPO_ROOT / "tools" / "fine_tuning" / "unsloth" / filename
        ).read_text(encoding="utf-8")
        assert "tokenizer.save_pretrained" not in source
        assert "_publish_exact_base_tokenizer_subset(" in source


def test_controlled_trainers_evaluate_each_epoch_and_checkpoint_preference_steps() -> None:
    sft_source = (REPO_ROOT / "tools/fine_tuning/unsloth/train_sft.py").read_text(
        encoding="utf-8"
    )
    dpo_source = (REPO_ROOT / "tools/fine_tuning/unsloth/train_dpo.py").read_text(
        encoding="utf-8"
    )
    for source in (sft_source, dpo_source):
        assert 'eval_strategy="epoch"' in source or '"eval_strategy": "epoch"' in source
        assert "trainer.evaluate()" in source
    assert 'save_strategy="steps"' in sft_source
    assert "save_steps=checkpoint_save_steps" in sft_source
    assert "save_only_model=False" in sft_source
    assert '"save_strategy": "steps"' in dpo_source
    assert '"save_steps": checkpoint_save_steps' in dpo_source
    assert '"save_only_model": False' in dpo_source


def test_rpo_chosen_nll_evidence_is_visible_only_when_enabled() -> None:
    assert ubuntu_pipeline._verified_rpo_chosen_nll_evidence(
        {"rpoAlpha": None},
        {"evaluation_metrics": {}},
    ) is None
    assert ubuntu_pipeline._verified_rpo_chosen_nll_evidence(
        {"rpoAlpha": 1.0},
        {"evaluation_metrics": {"eval_nll_loss": 0.25}},
    ) == {
        "rpoAlpha": 1.0,
        "metric": "eval_nll_loss",
        "value": 0.25,
    }


@pytest.mark.parametrize(
    "invalid_metric",
    (None, True, "0.25", -0.1, float("nan"), float("inf")),
)
def test_rpo_chosen_nll_evidence_rejects_missing_or_nonfinite_metrics(
    invalid_metric: object,
) -> None:
    with pytest.raises(RuntimeError, match="finite eval_nll_loss"):
        ubuntu_pipeline._verified_rpo_chosen_nll_evidence(
            {"rpoAlpha": 1.0},
            {"evaluation_metrics": {"eval_nll_loss": invalid_metric}},
        )


@pytest.mark.parametrize("rpo_alpha", (None, 1.0))
def test_rpo_runtime_capability_evidence_is_required_and_hash_bound(
    rpo_alpha: float | None,
) -> None:
    if rpo_alpha is None:
        evidence = train_dpo._verify_rpo_runtime_capability(
            dpo_config_class=object,
            dpo_trainer_class=object,
            rpo_alpha=None,
        )
    else:
        unsigned = {
            "schemaVersion": train_dpo.RPO_RUNTIME_CAPABILITY_SCHEMA,
            "status": "verified",
            "rpoAlpha": 1.0,
            "configClass": "trl.DPOConfig",
            "trainerClass": "trl.DPOTrainer",
            "methodSourceSHA256": {
                method: character * 64
                for method, character in zip(
                    (
                        "concatenated_forward",
                        "get_batch_loss_metrics",
                        "prediction_step",
                        "log",
                        "evaluate",
                    ),
                    "abcde",
                    strict=True,
                )
            },
        }
        evidence = {
            **unsigned,
            train_dpo.RPO_RUNTIME_CAPABILITY_HASH_FIELD: (
                train_dpo._canonical_sha256(unsigned)
            ),
        }

    assert ubuntu_pipeline._verified_rpo_runtime_capability_evidence(
        {"rpoAlpha": rpo_alpha},
        {"rpoRuntimeCapability": evidence},
    ) == evidence

    with pytest.raises(RuntimeError, match="exact-runtime RPO capability"):
        ubuntu_pipeline._verified_rpo_runtime_capability_evidence(
            {"rpoAlpha": rpo_alpha},
            {"rpoRuntimeCapability": {**evidence, "status": "tampered"}},
        )

    with pytest.raises(RuntimeError, match="exact-runtime RPO capability"):
        ubuntu_pipeline._verified_rpo_runtime_capability_evidence(
            {"rpoAlpha": rpo_alpha},
            {},
        )


@pytest.mark.parametrize("rpo_alpha", (None, 1.0))
def test_constructed_rpo_binding_evidence_is_required_and_hash_bound(
    rpo_alpha: float | None,
) -> None:
    args = SimpleNamespace(rpo_alpha=rpo_alpha)
    evidence = train_dpo._verify_constructed_rpo_binding(
        SimpleNamespace(args=args),
        args,
        rpo_alpha=rpo_alpha,
    )
    assert ubuntu_pipeline._verified_constructed_rpo_binding_evidence(
        {"rpoAlpha": rpo_alpha},
        {"constructedRPOBinding": evidence},
    ) == evidence

    with pytest.raises(RuntimeError, match="constructed RPO binding"):
        ubuntu_pipeline._verified_constructed_rpo_binding_evidence(
            {"rpoAlpha": rpo_alpha},
            {"constructedRPOBinding": {**evidence, "status": "tampered"}},
        )

    with pytest.raises(RuntimeError, match="constructed RPO binding"):
        ubuntu_pipeline._verified_constructed_rpo_binding_evidence(
            {"rpoAlpha": rpo_alpha},
            {},
        )
