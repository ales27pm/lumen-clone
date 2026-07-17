from __future__ import annotations

import ast
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

from lumen_manifest_crawler.dataset import adapter_evaluation
from tools.fine_tuning.unsloth import evaluate_adapter, ubuntu_pipeline


REPO_ROOT = Path(__file__).resolve().parents[4]
FAKE_IMAGE_DIGEST = "sha256:" + ("a" * 64)
OPTIMIZED_VARIANT = "internal_plus_public_optimized"
VALID_GGUF_TEST_PAYLOAD = (b"\x00" * 64) + b"LUMEN_VALID_GGUF_TEST"


def _gguf_bytes(
    *,
    version: int = 3,
    tensor_count: int = 1,
    metadata_kv_count: int = 1,
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


def _write_fake_gguf_reader(root: Path) -> Path:
    reader = root / "fake_gguf_dump.py"
    reader.write_text(
        """from __future__ import annotations

import json
import sys
from pathlib import Path

model = Path(sys.argv[1]).resolve()
data = model.read_bytes()
if not data.endswith(b"LUMEN_VALID_GGUF_TEST"):
    print("structural GGUF parse failed", file=sys.stderr)
    raise SystemExit(7)
tensor_count = int.from_bytes(data[8:16], "little")
metadata_count = int.from_bytes(data[16:24], "little")
metadata = {
    "GGUF.version": {},
    "GGUF.tensor_count": {},
    "GGUF.kv_count": {},
}
metadata.update({f"metadata.{index}": {} for index in range(metadata_count)})
tensors = {f"tensor.{index}": {} for index in range(tensor_count)}
print(json.dumps({
    "filename": str(model),
    "endian": "LITTLE",
    "metadata": metadata,
    "tensors": tensors,
}))
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
        "preferenceTrainer",
        "preferenceAdapterDir",
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
        "preferenceTrainer": "dpo",
        "preferenceAdapterDir": str(paths["dpo_output_dir"]),
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
) -> Path:
    variant = "internal_plus_public_optimized"
    (run_root / "models" / "lora_qwen3_gguf").mkdir(
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
            evaluation_module=evaluation_module,
            tool_contracts=fixture_tool_contracts,
        )
        if retry_format_error is None:
            raise AssertionError("Retry fixture requires an invalid first completion")
    candidate_rows = []
    for record in selected_records:
        prompt_messages = evaluate_adapter._structured_output_messages(
            agent,
            record["messages"],
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
    finalized = {
        "agent": agent,
        "variant": variant,
        "sourceVariantManifestSHA256": source_variant_sha256,
        "frozenEvaluationSHA256": evaluation_sha256,
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
        **{field: "c" * 40 for field in ubuntu_pipeline.RUNTIME_SOURCE_FIELDS},
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
        "base_model_name": "Qwen/Qwen3-1.7B",
        "baseModelRevision": "1" * 40,
        "baseModelIndexDigest": "2" * 64,
        "baseModelIndexReferencedShardNames": ["model.safetensors"],
        "baseModelIndexShardBindingSHA256": "3" * 64,
        "baseModelArtifactDigest": "4" * 64,
        "baseModelWeightShards": [
            {"filename": "model.safetensors", "sha256": "5" * 64, "size": 1}
        ],
        "baseModelTokenizerDigest": "6" * 64,
        "max_seq_length": 64,
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
        "variantAttestation": {"trainingEnvironmentSHA256": None},
        "adapterExport": {
            "adapterArtifact": "bootstrap",
            "adapterDirectory": "bootstrap",
            "adapterGGUFArtifact": "pending",
        },
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
    structured_eligible = agent in evaluate_adapter.JSON_OUTPUT_AGENTS
    generation = {
        "doSample": False,
        "numBeams": 1,
        "repetitionPenalty": evaluate_adapter.GENERATION_REPETITION_PENALTY,
        "thinkingEnabled": False,
        "maxNewTokens": 8,
        "maxSequenceLength": 64,
        "seed": 42,
        "structuredOutputContractEligible": structured_eligible,
        "structuredOutputContractVersion": (
            evaluate_adapter.STRUCTURED_OUTPUT_CONTRACT_VERSION
        ),
        "structuredOutputContractSHA256": (
            evaluate_adapter._structured_output_contract_sha256(
                agent,
                tool_contracts=fixture_tool_contracts,
            )
        ),
        "strictJSONRetryEligible": structured_eligible,
        "strictJSONMaxAttempts": (
            evaluate_adapter.STRICT_JSON_MAX_ATTEMPTS if structured_eligible else 1
        ),
        "strictJSONRetryContractVersion": (
            evaluate_adapter.STRICT_JSON_RETRY_CONTRACT_VERSION
        ),
        "strictJSONRetryContractSHA256": hashlib.sha256(
            evaluate_adapter.STRICT_JSON_RETRY_INSTRUCTION.encode("utf-8")
        ).hexdigest(),
    }
    run_manifest = {
        "schemaVersion": evaluate_adapter.EVALUATION_RUN_SCHEMA_VERSION,
        "status": status,
        "agent": agent,
        "variant": variant,
        "configPath": str(config_path.resolve()),
        "configSHA256": ubuntu_pipeline.file_sha256(config_path),
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
        "_verified_run_manifest",
        lambda *_args: {
            "variant": variant,
            "behaviorManifest": str(behavior_path.resolve()),
            "behaviorManifestFileSHA256": prepared_behavior_sha256,
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
) -> Path:
    agent = "cortex"
    variant = "internal_plus_public_optimized"
    sft = {"phase": "sft", "adapterSHA256": "a" * 64}
    final_phase = {
        "phase": "dpo",
        "adapterSHA256": "b" * 64,
        "parentSFTAdapterSHA256": "a" * 64,
        "finalizedVariantManifestSHA256": "f" * 64,
    }
    evaluation = {"status": "quality_gate_passed", "qualityGatePassed": True}
    gguf = (
        run_root
        / "models"
        / "lora_qwen3_gguf"
        / "lumen-cortex-lora.gguf"
    )
    gguf.parent.mkdir(parents=True, exist_ok=True)
    gguf.write_bytes(_gguf_bytes())
    reader_script = _write_fake_gguf_reader(run_root)
    evaluation_report = (
        run_root / "evaluation" / agent / "evaluation_report.json"
    )
    evaluation_report.parent.mkdir(parents=True, exist_ok=True)
    evaluation_report.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        ubuntu_pipeline,
        "_verified_run_manifest",
        lambda *_args: {
            "variant": variant,
            "agents": [{"agent": agent}],
        },
    )
    monkeypatch.setattr(ubuntu_pipeline, "verify_sft", lambda *_args: sft)
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
    summary = {
        "schema": "lumen.ubuntu-training-summary/2.0.0",
        "status": "complete",
        "variant": variant,
        "runRoot": str(run_root),
        "preferenceTraining": True,
        "agents": {
            agent: {
                "sft": sft,
                "finalPhase": final_phase,
                "adapterGGUF": str(gguf),
                "adapterGGUFExists": True,
                "adapterGGUFSHA256": ubuntu_pipeline.file_sha256(gguf),
                "adapterGGUFSizeBytes": gguf.stat().st_size,
                "evaluationReport": str(evaluation_report),
                "evaluationReportExists": True,
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

    assert (
        "COPY tools/fine_tuning/unsloth/training_lineage.py "
        "/tmp/lumen-training-lineage.py"
    ) in dockerfile
    assert "lineage.verify_training_dependency_lock(" in dockerfile
    assert "setpriv --reuid=nobody --regid=nogroup --init-groups python" in dockerfile
    assert "lineage.build_resolved_training_environment_snapshot()" in dockerfile
    assert "lineage.verify_resolved_training_environment(environment)" in dockerfile
    assert dockerignore == [
        "**",
        "!tools/",
        "!tools/fine_tuning/",
        "!tools/fine_tuning/unsloth/",
        "!tools/fine_tuning/unsloth/training_lineage.py",
        "!tools/hf_zerogpu/",
        "!tools/hf_zerogpu/space_template/",
        "!tools/hf_zerogpu/space_template/requirements.txt",
    ]
    assert (
        "COPY tools/hf_zerogpu/space_template/requirements.txt "
        "/tmp/lumen-requirements.txt"
    ) in dockerfile
    copy_sources = {
        line.split()[1]
        for line in dockerfile.splitlines()
        if line.startswith("COPY ")
    }
    assert copy_sources == {
        "tools/fine_tuning/unsloth/training_lineage.py",
        "tools/hf_zerogpu/space_template/requirements.txt",
    }


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
        "docker": f"""#!/bin/sh
set -eu
printf '%s\\n' "$*" >> "$FAKE_DOCKER_LOG"
if [ "${{1:-}}" = image ] && [ "${{2:-}}" = inspect ]; then
  printf '{FAKE_IMAGE_DIGEST}\\n'
  exit 0
fi
case "$*" in
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
    assert "PYTORCH_ALLOC_CONF=expandable_segments:True" in launcher
    assert "PYTORCH_CUDA_ALLOC_CONF" not in launcher


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
        and node.func.value.id == "FastLanguageModel"
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
    assert [entry["agent"] for entry in result["agents"]] == list(
        ubuntu_pipeline.AGENTS
    )
    assert not (tmp_path / "run-one").exists()


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
    assert launcher.count('for agent in "${AGENTS[@]}"; do') == 3
    assert "done < <(printf '%s' \"$AGENTS_CSV\" | tr ',' '\\n')" not in launcher


def test_converter_source_preflight_runs_before_the_first_sft_phase() -> None:
    launcher = (
        REPO_ROOT / "scripts/ubuntu_train_lumen_adapters_aio.sh"
    ).read_text(encoding="utf-8")
    first_sft_loop = launcher.index('for agent in "${AGENTS[@]}"; do')

    assert launcher.index('git init "$CONVERTER_REPO"') < first_sft_loop
    assert launcher.index(
        'git -C "$CONVERTER_REPO" status --porcelain=v1 --untracked-files=all'
    ) < first_sft_loop
    assert launcher.index(
        '"$TRAIN_PY" "$CONVERTER" --help >/dev/null'
    ) < first_sft_loop
    assert launcher.index(
        '"$TRAIN_PY" "$GGUF_READER" --help >/dev/null'
    ) < first_sft_loop
    assert launcher.count('git init "$CONVERTER_REPO"') == 1


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
        "resolvedTrainingEnvironmentSHA256": "r" * 64,
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
    monkeypatch.setattr(
        train_sft,
        "_training_environment",
        lambda *_args, **_kwargs: environment,
    )

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
    )
    config = json.loads(
        (run_root / "configs" / "cortex.json").read_text(encoding="utf-8")
    )

    assert config["trainingEnvironmentSHA256"] == environment_sha
    assert config["variantAttestation"]["trainingEnvironmentSHA256"] == environment_sha
    assert config["resolvedTrainingEnvironment"] == lineage[
        "resolvedTrainingEnvironment"
    ]
    assert (
        run_root / "generated" / "agent_manifest" / "AgentBehaviorManifest.json"
    ).is_file()


def test_final_config_switches_to_verified_preference_lineage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    config = {
        "agent": "cortex",
        "preference_trainer": "dpo",
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
        completion=(
            '{"selectedToolID":"files.read","intent":"files",'
            '"reasoningSummary":"Manifest row files.read is selected for intent '
            'files without actionStep.","requiresApproval":false,'
            '"nextModel":"executor"}'
        ),
    )
    evaluation_run = ubuntu_pipeline.read_object(run_path)
    monkeypatch.setattr(
        ubuntu_pipeline,
        "verify_sft",
        lambda *_args: {"adapterSHA256": "a" * 64},
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


@pytest.mark.parametrize(
    "mutation",
    (
        "run_schema",
        "evaluation_sha",
        "evaluation_path",
        "evaluator_code_sha",
        "retry_contract_sha",
        "retry_max_attempts",
        "recovery_counter",
        "config_path",
        "config_sha",
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
    final_phase = {
        "adapterSHA256": "b" * 64,
        "finalizedVariantManifestSHA256": evaluation_run[
            "finalizedVariantManifestSHA256"
        ],
        "parentSFTAdapterSHA256": "9" * 64,
        "phase": "dpo",
    }
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
    elif mutation == "retry_contract_sha":
        manifest["generation"]["strictJSONRetryContractSHA256"] = "0" * 64
    elif mutation == "retry_max_attempts":
        manifest["generation"]["strictJSONMaxAttempts"] = 3
    elif mutation == "recovery_counter":
        manifest["formatRecoveryCount"] = 1
    elif mutation == "config_path":
        manifest["configPath"] = str((tmp_path / "configs" / "other.json").resolve())
    elif mutation == "config_sha":
        manifest["configSHA256"] = "0" * 64
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
    final_phase = {
        "adapterSHA256": "b" * 64,
        "finalizedVariantManifestSHA256": manifest[
            "finalizedVariantManifestSHA256"
        ],
        "parentSFTAdapterSHA256": "9" * 64,
        "phase": "dpo",
    }

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
        tool_contracts=tool_contracts,
    )
    first_output, _, first_error = evaluate_adapter.normalize_candidate_output(
        "cortex",
        attempts[0]["rawOutput"],
        evaluation_module=evaluate_adapter._load_evaluation_module(),
        tool_contracts=tool_contracts,
    )
    retry_messages = evaluate_adapter._strict_json_retry_messages(
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
    final_phase = {
        "adapterSHA256": "b" * 64,
        "finalizedVariantManifestSHA256": original[
            "finalizedVariantManifestSHA256"
        ],
        "parentSFTAdapterSHA256": "9" * 64,
        "phase": "dpo",
    }

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
    final_phase = {
        "adapterSHA256": "b" * 64,
        "finalizedVariantManifestSHA256": manifest[
            "finalizedVariantManifestSHA256"
        ],
        "parentSFTAdapterSHA256": "9" * 64,
        "phase": "dpo",
    }
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
    final_phase = {
        "adapterSHA256": "b" * 64,
        "finalizedVariantManifestSHA256": manifest[
            "finalizedVariantManifestSHA256"
        ],
        "parentSFTAdapterSHA256": "9" * 64,
        "phase": "dpo",
    }
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
    final_phase = {
        "adapterSHA256": "b" * 64,
        "finalizedVariantManifestSHA256": manifest[
            "finalizedVariantManifestSHA256"
        ],
        "parentSFTAdapterSHA256": "9" * 64,
        "phase": "dpo",
    }
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
        final_phase={
            "adapterSHA256": "b" * 64,
            "finalizedVariantManifestSHA256": manifest[
                "finalizedVariantManifestSHA256"
            ],
            "parentSFTAdapterSHA256": "9" * 64,
            "phase": "dpo",
        },
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
            final_phase={
                "adapterSHA256": "b" * 64,
                "finalizedVariantManifestSHA256": manifest[
                    "finalizedVariantManifestSHA256"
                ],
                "parentSFTAdapterSHA256": "9" * 64,
                "phase": "dpo",
            },
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
        final_phase={
            "adapterSHA256": "b" * 64,
            "finalizedVariantManifestSHA256": manifest[
                "finalizedVariantManifestSHA256"
            ],
            "parentSFTAdapterSHA256": "9" * 64,
            "phase": "dpo",
        },
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

    with pytest.raises(RuntimeError, match="Candidate output evidence failed"):
        ubuntu_pipeline._verify_evaluation_outputs(
            tmp_path,
            "cortex",
            final_phase={
                "adapterSHA256": "b" * 64,
                "finalizedVariantManifestSHA256": manifest[
                    "finalizedVariantManifestSHA256"
                ],
                "parentSFTAdapterSHA256": "9" * 64,
                "phase": "dpo",
            },
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
        final_phase={
            "adapterSHA256": "b" * 64,
            "finalizedVariantManifestSHA256": manifest[
                "finalizedVariantManifestSHA256"
            ],
            "parentSFTAdapterSHA256": "9" * 64,
            "phase": "dpo",
        },
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
            final_phase={
                "adapterSHA256": "b" * 64,
                "finalizedVariantManifestSHA256": manifest[
                    "finalizedVariantManifestSHA256"
                ],
                "parentSFTAdapterSHA256": "9" * 64,
                "phase": "dpo",
            },
        )


@pytest.mark.parametrize(
    "mutation",
    (
        "variant",
        "sft",
        "missing_gguf",
        "gguf_path",
        "evaluation_report_path",
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
    elif mutation == "missing_gguf":
        Path(item["adapterGGUF"]).unlink()
        item["adapterGGUFExists"] = False
        item["adapterGGUFSHA256"] = None
        item["adapterGGUFSizeBytes"] = 0
    elif mutation == "gguf_path":
        item["adapterGGUF"] = str(tmp_path / "other.gguf")
    elif mutation == "evaluation_report_path":
        item["evaluationReport"] = str(tmp_path / "other-report.json")
    else:  # pragma: no cover - parametrization is closed above.
        raise AssertionError(mutation)
    summary.pop("summarySHA256", None)
    summary["summarySHA256"] = ubuntu_pipeline.canonical_sha256(summary)
    ubuntu_pipeline.write_object(summary_path, summary)

    with pytest.raises(RuntimeError):
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
            "agents": [{"agent": "cortex"}],
        },
    )

    with pytest.raises(RuntimeError, match="summary failed verification"):
        ubuntu_pipeline.verify_gguf(tmp_path, "cortex")


def test_resume_reuses_gguf_only_from_a_complete_canonical_summary(
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


def test_gguf_inventory_requires_exact_prepared_agent_set(
    tmp_path: Path,
) -> None:
    gguf_dir = tmp_path / "models" / "lora_qwen3_gguf"
    gguf_dir.mkdir(parents=True)
    for agent in ubuntu_pipeline.AGENTS:
        (gguf_dir / f"lumen-{agent}-lora.gguf").write_bytes(_gguf_bytes())

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
    gguf = tmp_path / "models" / "lora_qwen3_gguf" / "lumen-cortex-lora.gguf"
    gguf.parent.mkdir(parents=True)
    gguf.write_bytes(_gguf_bytes())
    monkeypatch.setattr(
        ubuntu_pipeline,
        "_verified_run_manifest",
        lambda *_args: {
            "variant": OPTIMIZED_VARIANT,
            "agents": [{"agent": "cortex"}],
        },
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
    assert ubuntu_pipeline.parse_args().public is False

    sys.argv.append("--public")
    assert ubuntu_pipeline.parse_args().public is True


def test_trainers_save_only_the_unified_fast_tokenizer_format() -> None:
    for filename in ("train_sft.py", "train_dpo.py"):
        source = (
            REPO_ROOT / "tools" / "fine_tuning" / "unsloth" / filename
        ).read_text(encoding="utf-8")
        assert "tokenizer.save_pretrained" in source
        assert "legacy_format=False" in source


def test_controlled_trainers_evaluate_and_save_each_epoch() -> None:
    sft_source = (REPO_ROOT / "tools/fine_tuning/unsloth/train_sft.py").read_text(
        encoding="utf-8"
    )
    dpo_source = (REPO_ROOT / "tools/fine_tuning/unsloth/train_dpo.py").read_text(
        encoding="utf-8"
    )
    for source in (sft_source, dpo_source):
        assert 'eval_strategy="epoch"' in source or '"eval_strategy": "epoch"' in source
        assert 'save_strategy="epoch"' in source or '"save_strategy": "epoch"' in source
        assert "trainer.evaluate()" in source
