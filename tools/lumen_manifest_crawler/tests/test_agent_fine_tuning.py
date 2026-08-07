from __future__ import annotations

import ast
import copy
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import replace
from itertools import combinations
from pathlib import Path
from typing import Any

import pytest

from lumen_manifest_crawler.crawler import generate_manifest
from lumen_manifest_crawler.dataset import fine_tuning as fine_tuning_module
from lumen_manifest_crawler.dataset import generate_all_datasets
from lumen_manifest_crawler.dataset.adapter_evaluation import (
    EVALUATION_SCHEMA_VERSION,
    FLEET_TOOL_OWNERSHIP_OUTPUT_CONTRACT,
    _score_metric,
    build_contamination_report,
    canonical_sha256,
    declarative_metrics_from_expected,
    evaluation_output_mode,
    mouth_final_text_is_complete,
    score_evaluation_suite,
    upgrade_evaluation_record,
)
from lumen_manifest_crawler.dataset.chat_template_contract import (
    generic_strict_json_retry_instruction,
)
from lumen_manifest_crawler.dataset.compiler import _records_hash
from lumen_manifest_crawler.dataset.codebase_home import (
    MAX_CHUNK_CHARS,
    _split_source_chunks,
    generate_codebase_home_records,
)
from lumen_manifest_crawler.dataset.cortex import generate_cortex_records
from lumen_manifest_crawler.dataset.fine_tuning import (
    AGENTS,
    CODEBASE_SUPPLEMENTAL_SOURCE_FAMILIES,
    CORTEX_CODEBASE_SELF_AWARENESS_SOURCE_FAMILY,
    CORTEX_CODEBASE_SYSTEM_PROMPT,
    CORTEX_ROUTE_DECISION_ENDCAP,
    CORTEX_ROUTE_SYSTEM_PROMPT,
    CORTEX_SUPPLEMENTAL_GROUNDING_SOURCE_FAMILIES,
    CORTEX_TOOL_CATALOG_HEADER,
    CRITICAL_CONTRACT_VALIDATION_RECORDS_PER_CASE,
    EXECUTOR_RUNTIME_SYSTEM_PROMPT,
    FLEET_BALANCED_CONTRACT_TASK_TYPES,
    FLEET_ALL_POLICY_SFT_TOKEN_SHARE_MAX_BASIS_POINTS,
    FLEET_ALL_POLICY_SFT_TOKEN_SHARE_MIN_BASIS_POINTS,
    FLEET_COMPREHENSIVE_TOOL_OWNERSHIP_SFT_SURFACES,
    FLEET_DELEGATION_OUTPUT_CONTRACT,
    FLEET_DELEGATION_PROMPTS_PER_OWNER,
    FLEET_DELEGATION_VALIDATION_PROMPTS_PER_OWNER,
    FLEET_LOSS_SHARE_BASIS_POINTS_DENOMINATOR,
    FLEET_LOSS_SHARE_CONTRACT_SCHEMA_VERSION,
    FLEET_LOSS_SHARE_EVIDENCE_SCHEMA_VERSION,
    FLEET_NATIVE_ORCHESTRATION_DPO_SHARE_MAX_BASIS_POINTS,
    FLEET_NATIVE_ORCHESTRATION_DPO_SHARE_MIN_BASIS_POINTS,
    FLEET_NATIVE_ORCHESTRATION_DPO_TOKEN_SHARE_MAX_BASIS_POINTS,
    FLEET_NATIVE_ORCHESTRATION_DPO_TOKEN_SHARE_MIN_BASIS_POINTS,
    FLEET_NATIVE_ORCHESTRATION_DPO_TASK_TYPE,
    FLEET_NATIVE_ORCHESTRATION_MUTATION_CONTRACT,
    FLEET_NATIVE_ORCHESTRATION_SFT_SHARE_MAX_BASIS_POINTS,
    FLEET_NATIVE_ORCHESTRATION_SFT_SHARE_MIN_BASIS_POINTS,
    FLEET_NATIVE_ORCHESTRATION_SFT_PROXY_SHARE_MAX_BASIS_POINTS,
    FLEET_NATIVE_ORCHESTRATION_SFT_PROXY_SHARE_MIN_BASIS_POINTS,
    FLEET_NATIVE_ORCHESTRATION_SFT_TASK_TYPE,
    FLEET_NATIVE_ORCHESTRATION_SOURCE_FAMILY,
    FLEET_OBSERVED_MOUTH_CONFUSION_PROMPTS_PER_OWNER,
    FLEET_OPTIMIZER_FAMILY_SHARE_SCHEMA_VERSION,
    FLEET_OPTIMIZER_FAMILY_SOURCE_PROXY_SCHEMA_VERSION,
    FLEET_POLICY_VOCABULARY_SFT_TASK_TYPE,
    FLEET_REQUIRED_SUPPLEMENTAL_SFT_TASK_TYPES,
    FLEET_SUPPLEMENTAL_ASSISTANT_SHARE_HARD_MAX,
    FLEET_SUPPLEMENTAL_SOURCE_FAMILY_PROXY_SELECTION_SHARE_HARD_MAX,
    FLEET_SFT_OPTIMIZER_WINDOW_CANDIDATE_COUNT,
    FLEET_SFT_OPTIMIZER_MAX_TRAIN_RECORDS,
    FLEET_SFT_OPTIMIZER_RECORD_GEOMETRY_SCHEMA_VERSION,
    FLEET_SFT_OPTIMIZER_WINDOW_SCHEDULE_ALGORITHM,
    FLEET_SFT_OPTIMIZER_WINDOW_SCHEDULE_CONTRACT_SCHEMA_VERSION,
    FLEET_SFT_OPTIMIZER_WINDOW_SCHEDULE_EVIDENCE_SCHEMA_VERSION,
    FLEET_VALIDATION_SAMPLING_SCHEMA_VERSION,
    FLEET_SOURCE_ROLE_BEHAVIORAL_PRIMARY,
    FLEET_SOURCE_ROLE_PUBLIC_BEHAVIORAL,
    FLEET_SOURCE_ROLE_SUPPLEMENTAL_STATIC,
    FLEET_SLOT_DIRECTORY_OUTPUT_CONTRACT,
    FLEET_TOOL_BOUNDARY_OUTPUT_CONTRACT,
    NON_CORTEX_MINIMUM_EFFECTIVE_DPO_STEPS,
    NON_CORTEX_MINIMUM_EFFECTIVE_SFT_STEPS,
    NON_CORTEX_MAX_TRAINING_EPOCHS,
    PUBLIC_ADAPTER_CORPUS_PREFIX,
    PUBLIC_CORPUS_ASSISTANT_TARGET_TOKEN_SHARE_HARD_MAX,
    PUBLIC_CORPUS_LOSS_SHARE_CONTRACT_SCHEMA_VERSION,
    PUBLIC_CORPUS_SOURCE_PROXY_SELECTION_SHARE_HARD_MAX,
    FineTuningDatasetConfig,
    MIMICRY_CONTRACT_TRAIN_RECORDS_PER_CASE,
    MIMICRY_CRITICAL_CONTRACT_CASES,
    REM_CONTRACT_TRAIN_RECORDS_PER_CASE,
    REM_CRITICAL_CONTRACT_CASES,
    REM_REPAIR_ACTION_ADD_ACTION_STEP_SAMPLES,
    REM_REPAIR_ACTION_DISABLE_DETERMINISTIC_COMPATIBILITY,
    REM_REPAIR_ACTION_FORCE_NO_THINKING,
    REM_REPAIR_ACTION_REGENERATE_MANIFEST_GROUNDING,
    SYSTEM_PROMPTS,
    STRUCTURED_OUTPUT_INSTRUCTION,
    STRICT_JSON_RETRY_DPO_INSTRUCTION,
    ULTRA_SPECIFIC_SOURCE_FAMILY,
    _CORTEX_NATURAL_IMPLICIT_COMPLETE_PROMPTS,
    _CORTEX_RETRY_GUIDANCE_BY_FAILURE_CODE,
    _balanced_fleet_contract_dpo_pairs,
    _balanced_fleet_contract_sft_anchors,
    _assert_fleet_balanced_retry_training_coverage,
    _agent_unsloth_config,
    _bind_fleet_dpo_contract,
    _bind_fleet_sft_contract,
    _bind_cortex_dpo_route_contract,
    _bound_fleet_validation_sft_records,
    _canonicalize_cortex_sft_output,
    _comprehensive_fleet_tool_ownership_sft_anchors,
    _canonical_messages_key,
    _cap_public_corpus_token_share,
    _cortex_failure_repair_sft_records,
    _fleet_delegation_tasks,
    _fleet_native_prompt_event_id_fact,
    _fleet_observed_mouth_confusion_tasks,
    _first_role_content,
    _fleet_slot_contract,
    _finalize_fleet_optimizer_lane,
    _limit_supplemental_sft_records,
    _limit_fleet_supplemental_dpo_records,
    _ordered_cortex_route_text,
    _public_corpus_loss_share_contract,
    _fleet_source_role,
    _manifest_valid_executor_payload,
    _mimicry_critical_contract_scenarios,
    _required_eval_templates,
    _rem_critical_contract_scenarios,
    _routed_intent_for_tool,
    _valid_fleet_native_preference_mutation,
    _canonical_rendered_prompt_key,
    _stable_dpo_split,
    _stable_source_stratified_split,
    _source_token_proxy_count,
    _strict_json_loads,
    _synthetic_dpo_pairs,
    _ultra_specific_cortex_records,
    _ultra_specific_dpo_pairs,
    _ultra_specific_eval_templates,
    _validate_cortex_sft_route_intents,
    compile_agent_fine_tuning_datasets,
    cortex_runtime_route_system_prompt,
)
from lumen_manifest_crawler.fleet_artifacts import generate_fleet_artifacts
from lumen_manifest_crawler.manifest import (
    AgentBehaviorManifest,
    IntentManifest,
    RoutingMatrixEntry,
    ToolArgumentManifest,
    ToolManifest,
)
from lumen_manifest_crawler.output.writer import write_outputs
from lumen_manifest_crawler.validators import validate_agent_fine_tuning_datasets, validate_manifest


EXPECTED_SHARED_BASE_REPO = "ales27pm/lumen-qwen3-bootstrap-gguf"
EXPECTED_SHARED_BASE_FILE = "lumen-qwen3-fast-shared-q4_k_m.gguf"
EXPECTED_ADAPTER_REPO = "ales27pm/lumen-qwen3-bootstrap-adapters-gguf"

pytestmark = pytest.mark.slow


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def compiled_fine_tuning() -> tuple:
    repo_root = _repo_root()
    manifest = generate_manifest(repo_root)
    datasets = generate_all_datasets(manifest, root=repo_root)
    fine_tuning = compile_agent_fine_tuning_datasets(manifest, datasets)
    return manifest, datasets, fine_tuning


def _write_fine_tuning_fixture(tmp_path: Path, compiled_fine_tuning: tuple) -> Path:
    manifest, datasets, fine_tuning = compiled_fine_tuning
    report = validate_manifest(manifest, datasets)
    output = tmp_path / "agent_manifest"
    fine_tuning_output = tmp_path / "fine_tuning"

    write_outputs(
        output,
        manifest,
        report,
        datasets,
        pretty=True,
        fine_tuning_datasets=fine_tuning,
        fine_tuning_output_dir=fine_tuning_output,
    )
    return fine_tuning_output


def test_per_agent_directories_are_produced(tmp_path: Path, compiled_fine_tuning: tuple) -> None:
    manifest, _, _ = compiled_fine_tuning
    fine_tuning_output = _write_fine_tuning_fixture(tmp_path, compiled_fine_tuning)

    assert (fine_tuning_output / "adapter_runtime_manifest.json").exists()
    assert (fine_tuning_output / "public_evaluation_fingerprints.json").exists()
    for agent in AGENTS:
        agent_dir = fine_tuning_output / agent
        assert agent_dir.exists()
        for filename in (
            "train_sft.jsonl",
            "val_sft.jsonl",
            "eval.jsonl",
            "dataset_card.json",
            "experiment_manifest.json",
            "unsloth_config.json",
            "adapter_export_plan.json",
        ):
            assert (agent_dir / filename).exists(), f"missing {agent}/{filename}"
        dataset_card = json.loads((agent_dir / "dataset_card.json").read_text(encoding="utf-8"))
        assert dataset_card["sourceIntegrity"] == manifest.sourceIntegrity.lineage_dict()
        adapter_plan = json.loads((agent_dir / "adapter_export_plan.json").read_text(encoding="utf-8"))
        assert adapter_plan["datasetCard"]["sourceIntegrity"] == manifest.sourceIntegrity.lineage_dict()
        for variant in ("internal_only", "internal_plus_public_baseline", "internal_plus_public_optimized"):
            variant_dir = agent_dir / "experiments" / variant
            assert (variant_dir / "variant_manifest.json").exists()
            assert (variant_dir / "contamination_report.json").exists()


def test_all_optimized_training_lanes_have_canonical_source_classification(
    compiled_fine_tuning: tuple,
) -> None:
    _, _, fine_tuning = compiled_fine_tuning

    for agent in AGENTS:
        optimized = fine_tuning[agent].experiment_variants[
            "internal_plus_public_optimized"
        ]
        for lane in ("train_sft", "val_sft", "train_dpo", "val_dpo"):
            for record in optimized[lane]:
                metadata = record.get("metadata")
                assert isinstance(metadata, dict), f"{agent}/{lane} metadata"
                assert metadata.get("agent") == agent
                source_family = metadata.get("sourceFamily")
                assert isinstance(source_family, str), (
                    f"{agent}/{lane} sourceFamily type"
                )
                assert source_family
                assert source_family == source_family.strip()
                task_type = metadata.get("taskType")
                assert isinstance(task_type, str), f"{agent}/{lane} taskType type"
                assert task_type
                assert task_type == task_type.strip()

                public_corpus = metadata.get("publicCorpus")
                has_public_lineage = (
                    isinstance(public_corpus, dict) and bool(public_corpus)
                )
                assert ("publicCorpus" in metadata) == has_public_lineage
                assert source_family.startswith(
                    PUBLIC_ADAPTER_CORPUS_PREFIX
                ) == has_public_lineage


def test_written_fine_tuning_outputs_are_adapter_first(tmp_path: Path, compiled_fine_tuning: tuple) -> None:
    fine_tuning_output = _write_fine_tuning_fixture(tmp_path, compiled_fine_tuning)
    runtime_manifest = json.loads((fine_tuning_output / "adapter_runtime_manifest.json").read_text(encoding="utf-8"))

    assert runtime_manifest["mode"] == "adapter_first"
    assert runtime_manifest["sharedBaseRepoID"] == EXPECTED_SHARED_BASE_REPO
    assert runtime_manifest["sharedBaseFileName"] == EXPECTED_SHARED_BASE_FILE
    assert runtime_manifest["sharedBaseModelIndexReferencedShardNames"] == [
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    ]
    assert len(runtime_manifest["sharedBaseModelIndexShardBindingSHA256"]) == 64
    assert len(runtime_manifest["sharedTrainingCodeSHA256"]) == 64
    assert len(runtime_manifest["sharedTrainingCodeBundleSHA256"]) == 64
    assert len(runtime_manifest["sharedTrainingDependencyLockSHA256"]) == 64
    assert len(runtime_manifest["sharedRequirementsSHA256"]) == 64
    assert runtime_manifest["adapterRepoID"] == EXPECTED_ADAPTER_REPO
    assert runtime_manifest["runtimeStrategy"]["loadBaseModelOnce"] is True
    assert runtime_manifest["runtimeStrategy"]["selectAdapterByAgentSlot"] is True
    assert runtime_manifest["runtimeStrategy"]["mergeAdaptersByDefault"] is False
    assert runtime_manifest["runtimeStrategy"]["mergedExportPhase"] == "optional_release_bake"
    assert runtime_manifest["releaseBakePolicy"]["enabledByDefault"] is False

    adapters_by_agent = {entry["agent"]: entry for entry in runtime_manifest["adapters"]}
    for agent in AGENTS:
        expected_adapter_dir = f"models/lora_qwen3_bootstrap/{agent}"
        expected_training_dir = f"models/training_runs_qwen3_bootstrap/{agent}"
        expected_dpo_dir = f"models/lora_qwen3_dpo/{agent}"
        expected_adapter_gguf = f"models/lora_qwen3_gguf/lumen-{agent}-lora.gguf"
        agent_dir = fine_tuning_output / agent
        config = json.loads((agent_dir / "unsloth_config.json").read_text(encoding="utf-8"))
        plan = json.loads((agent_dir / "adapter_export_plan.json").read_text(encoding="utf-8"))
        experiment = json.loads((agent_dir / "experiment_manifest.json").read_text(encoding="utf-8"))
        controlled_variables = set(experiment["controlledVariables"])

        assert config["artifactMode"] == "adapter_first"
        assert config["defaultExportArtifact"] == "lora_adapter"
        assert config["adapter_output_dir"] == expected_adapter_dir
        assert config["output_dir"] == expected_training_dir
        assert config["dpo_output_dir"] == expected_dpo_dir
        assert config["adapter_gguf_output_path"] == expected_adapter_gguf
        assert config["adapterExport"]["agent"] == agent
        assert config["adapterExport"]["adapterArtifact"] == expected_adapter_dir
        assert config["adapterExport"]["adapterDirectory"] == expected_adapter_dir
        assert config["adapterExport"]["adapterGGUFArtifact"] == expected_adapter_gguf
        assert config["adapterExport"]["adapterRepoID"] == EXPECTED_ADAPTER_REPO
        assert config["gguf_repo_id"] == EXPECTED_ADAPTER_REPO
        assert config["adapterExport"]["sharedBaseRepoID"] == EXPECTED_SHARED_BASE_REPO
        assert config["preference_trainer"] == "dpo"
        assert config["dpo_beta"] == pytest.approx(0.1)
        assert config["dpo_rpo_alpha"] == (
            1.0 if agent == "fleet" else None
        )
        assert config["bf16"] is False
        assert config["fp16"] is True
        assert config["trainingCodeManifest"]["phase"] == "sft"
        assert config["trainingCodeSHA256"] == config[
            "trainingCodeSHA256ByPhase"
        ]["sft"]
        assert config["trainingDependencyLockSHA256"] == config[
            "trainingDependencyLock"
        ]["trainingDependencyLockSHA256"]
        assert config["requirementsSHA256"] == config["trainingDependencyLock"][
            "requirementsSHA256"
        ]
        assert config["runtimeSourceKind"] == "unresolved"
        assert config["runtimeSourceRevision"] is None
        assert config["adapterExport"]["sharedBaseFileName"] == EXPECTED_SHARED_BASE_FILE
        assert config["adapterExport"]["trainBaseModelWeights"] is False
        assert config["adapterExport"]["saveAdapterByDefault"] is True
        assert config["adapterExport"]["mergeAdaptersByDefault"] is False
        assert config["mergeExport"]["enabledByDefault"] is False
        assert config["mergeExport"]["phase"] == "optional_release_bake"

        assert adapters_by_agent[agent]["adapterArtifact"] == expected_adapter_dir
        assert adapters_by_agent[agent]["adapterDirectory"] == expected_adapter_dir
        assert adapters_by_agent[agent]["adapterGGUFArtifact"] == expected_adapter_gguf
        assert adapters_by_agent[agent]["adapterRepoID"] == EXPECTED_ADAPTER_REPO
        assert adapters_by_agent[agent]["trainingCodeSHA256"] == config[
            "trainingCodeSHA256"
        ]
        assert adapters_by_agent[agent]["trainingDependencyLockSHA256"] == config[
            "trainingDependencyLockSHA256"
        ]
        assert (
            set(adapters_by_agent[agent]["experimentPolicy"]["controlledVariables"])
            == controlled_variables
        )
        assert plan["mode"] == "adapter_first"
        assert plan["agent"] == agent
        assert plan["sharedBaseRepoID"] == EXPECTED_SHARED_BASE_REPO
        assert plan["sharedBaseFileName"] == EXPECTED_SHARED_BASE_FILE
        assert plan["adapterRepoID"] == EXPECTED_ADAPTER_REPO
        assert plan["adapterArtifact"] == expected_adapter_dir
        assert plan["adapterDirectory"] == expected_adapter_dir
        assert plan["adapterGGUFArtifact"] == expected_adapter_gguf
        assert plan["trainingCodeSHA256"] == config["trainingCodeSHA256"]
        assert plan["trainingDependencyLockSHA256"] == config[
            "trainingDependencyLockSHA256"
        ]
        assert set(plan["experimentPolicy"]["controlledVariables"]) == controlled_variables
        assert plan["expectedArtifacts"]["adapterDirectory"] == expected_adapter_dir
        assert plan["expectedArtifacts"]["adapterGGUF"] == expected_adapter_gguf
        assert plan["runtimeBinding"]["loadBaseModelOnce"] is True
        assert plan["runtimeBinding"]["selectAdapterByAgentSlot"] is True
        assert plan["exportPolicy"]["defaultArtifact"] == "adapter"
        assert plan["exportPolicy"]["mergeAdaptersByDefault"] is False
        assert plan["exportPolicy"]["mergedExportPhase"] == "optional_release_bake"


def test_sft_records_use_chat_format(compiled_fine_tuning: tuple) -> None:
    _, _, fine_tuning = compiled_fine_tuning
    for agent in AGENTS:
        for record in (fine_tuning[agent].train_sft + fine_tuning[agent].val_sft)[:20]:
            messages = record["messages"]
            assert len(messages) == 3
            assert messages[0]["role"] == "system"
            assert messages[1]["role"] == "user"
            assert messages[2]["role"] == "assistant"
            assert isinstance(messages[2]["content"], str)
            assert messages[2]["content"].strip()


def test_eval_records_use_supported_executable_metric_contracts(compiled_fine_tuning: tuple) -> None:
    _, _, fine_tuning = compiled_fine_tuning
    for agent in AGENTS:
        assert fine_tuning[agent].eval
        for record in fine_tuning[agent].eval:
            assert record["schemaVersion"] == EVALUATION_SCHEMA_VERSION
            assert record["outputMode"] in {"json", "text"}
            assert record["metrics"]
            assert not [
                metric
                for metric in record["metrics"]
                if metric.get("type") == "unsupported_contract"
            ], (agent, record.get("metadata"), record["metrics"])


def test_sft_records_do_not_train_null_assistant_outputs(compiled_fine_tuning: tuple) -> None:
    _, _, fine_tuning = compiled_fine_tuning
    for agent in AGENTS:
        for record in fine_tuning[agent].train_sft + fine_tuning[agent].val_sft:
            content = record["messages"][2]["content"].strip().lower()
            assert content not in {"", "null", "none"}, record["metadata"]


def test_sft_messages_are_unique_and_source_stratified(compiled_fine_tuning: tuple) -> None:
    _, _, fine_tuning = compiled_fine_tuning
    for agent in AGENTS:
        dataset = fine_tuning[agent]
        train = dataset.train_sft
        val = dataset.val_sft
        train_keys = {json.dumps(record["messages"], ensure_ascii=False, sort_keys=True) for record in train}
        val_keys = {json.dumps(record["messages"], ensure_ascii=False, sort_keys=True) for record in val}
        assert len(train_keys) == len(train)
        assert len(val_keys) == len(val)
        assert train_keys.isdisjoint(val_keys)

        all_sources = {
            record["metadata"]["sourceFamily"]
            for record in train + val
        }
        constraints = dataset.dataset_card["constraints"]
        fleet_optimizer_only = (
            agent == "fleet"
            and constraints.get("fleetLossShareEnforcementScope")
            == "optimizer_train_only"
            and constraints.get("fleetValidationLossSharePolicy")
            == "observed_not_enforced"
        )
        registry = constraints.get("fleetSourceRoleRegistry") or {}
        supplemental_pairs = {
            (entry["sourceFamily"], entry["taskType"])
            for entry in registry.get("registeredPairs", [])
            if entry.get("category") == "supplemental_static"
        }
        for source in all_sources:
            source_records = [record for record in train + val if record["metadata"]["sourceFamily"] == source]
            if len(source_records) >= 2 and not source.startswith("public_adapter_corpus_"):
                validation_only_static = (
                    fleet_optimizer_only
                    and not any(
                        record["metadata"]["sourceFamily"] == source
                        for record in train
                    )
                    and all(
                        (
                            record["metadata"]["sourceFamily"],
                            record["metadata"]["taskType"],
                        )
                        in supplemental_pairs
                        for record in source_records
                    )
                )
                if validation_only_static:
                    continue
                assert any(record["metadata"]["sourceFamily"] == source for record in train)
                assert any(record["metadata"]["sourceFamily"] == source for record in val)


def test_sft_prompts_have_one_semantic_assistant_label(compiled_fine_tuning: tuple) -> None:
    _, _, fine_tuning = compiled_fine_tuning

    def canonical_output(content: str) -> str:
        try:
            return json.dumps(json.loads(content), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        except json.JSONDecodeError:
            return " ".join(content.split())

    for agent in AGENTS:
        outputs_by_prompt: dict[str, set[str]] = {}
        records = fine_tuning[agent].train_sft + fine_tuning[agent].val_sft
        for record in records:
            prompt = json.dumps(record["messages"][:-1], ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            outputs_by_prompt.setdefault(prompt, set()).add(canonical_output(record["messages"][-1]["content"]))
        conflicts = {prompt: outputs for prompt, outputs in outputs_by_prompt.items() if len(outputs) > 1}
        assert conflicts == {}, f"{agent} has {len(conflicts)} prompt conflicts"


def test_role_locked_adapters_only_contain_native_sources(compiled_fine_tuning: tuple) -> None:
    _, _, fine_tuning = compiled_fine_tuning
    allowed = {
        "executor": {"executor_tool_calls", "tool_schema_cards", "approval_boundary_samples", "negative_samples"},
        "mouth": {"mouth_responses"},
        "mimicry": {"mimicry_style"},
        "rem": {"rem_reflection", "runtime_audit_repairs"},
    }
    for agent, sources in allowed.items():
        actual = {
            record["metadata"]["sourceFamily"]
            for record in fine_tuning[agent].train_sft + fine_tuning[agent].val_sft
        }
        assert all(
            source in sources
            or source == ULTRA_SPECIFIC_SOURCE_FAMILY
            or source.startswith("public_adapter_corpus_")
            for source in actual
        )
        assert all(
            (record.get("metadata") or {}).get("publicCorpus", {}).get("targetAdapter") == agent
            for record in fine_tuning[agent].train_sft + fine_tuning[agent].val_sft
            if record["metadata"]["sourceFamily"].startswith("public_adapter_corpus_")
        )


def test_dataset_cards_account_for_materialized_sft(compiled_fine_tuning: tuple) -> None:
    _, _, fine_tuning = compiled_fine_tuning
    for agent in AGENTS:
        dataset = fine_tuning[agent]
        records = dataset.train_sft + dataset.val_sft
        source_counts: dict[str, int] = {}
        task_counts: dict[str, int] = {}
        for record in records:
            source = record["metadata"]["sourceFamily"]
            task = record["metadata"]["taskType"]
            source_counts[source] = source_counts.get(source, 0) + 1
            task_counts[task] = task_counts.get(task, 0) + 1
            assert "sourceDirty" in record["metadata"]
            assert "worktreeFingerprint" in record["metadata"]
        assert dataset.dataset_card["sourceFamilyCounts"] == dict(sorted(source_counts.items()))
        assert dataset.dataset_card["taskTypeCounts"] == dict(sorted(task_counts.items()))
        assert dataset.dataset_card["availableSFTRecords"] == len(records)
        assert "sourceDirty" in dataset.dataset_card
        assert "worktreeFingerprint" in dataset.dataset_card
        if agent == "fleet":
            assert dataset.dataset_card["constraints"][
                "fleetOrchestrationEvaluationRequired"
            ] is False


def test_sft_records_fit_configured_sequence_budget(compiled_fine_tuning: tuple) -> None:
    _, _, fine_tuning = compiled_fine_tuning
    for agent in AGENTS:
        dataset = fine_tuning[agent]
        max_chars = dataset.unsloth_config["sequence_char_budget"]
        assert dataset.unsloth_config["sequence_budget_policy"] == (
            "utf8_byte_proxy_configured_chars_per_token"
        )
        assert dataset.unsloth_config["max_chars_per_token"] == 4
        for record in dataset.train_sft + dataset.val_sft:
            serialized = json.dumps(record["messages"], ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            assert len(serialized.encode("utf-8")) <= max_chars


def test_each_adapter_has_ultra_specific_dataset_records(compiled_fine_tuning: tuple) -> None:
    _, _, fine_tuning = compiled_fine_tuning
    minimum_records = {
        "cortex": 10,
        "executor": 10,
        "mouth": 10,
        "mimicry": 5,
        "rem": 5,
        "fleet": 5,
    }

    for agent in AGENTS:
        records = fine_tuning[agent].train_sft + fine_tuning[agent].val_sft
        ultra_specific = [
            record for record in records
            if (record.get("metadata") or {}).get("sourceFamily") == ULTRA_SPECIFIC_SOURCE_FAMILY
        ]
        card_quality = fine_tuning[agent].dataset_card.get("quality") or {}

        assert len(ultra_specific) >= minimum_records[agent], f"{agent} lacks ultra-specific records"
        assert fine_tuning[agent].dataset_card["constraints"]["ultraSpecificAdapterCorpus"] is True
        assert card_quality["ultraSpecificSourceFamily"] == ULTRA_SPECIFIC_SOURCE_FAMILY
        assert card_quality["ultraSpecificRecordCount"] == len(ultra_specific)
        assert all((record.get("metadata") or {}).get("specificity") == "ultra_specific" for record in ultra_specific)


def test_public_adapter_corpus_is_loaded_and_group_split_without_cross_routing(
    compiled_fine_tuning: tuple,
) -> None:
    _, datasets, fine_tuning = compiled_fine_tuning
    snapshot = json.loads(
        (_repo_root() / "datasets/public_adapter_corpus/manifest.json").read_text(encoding="utf-8")
    )
    dataset_manifest = datasets["dataset_manifest"][0]
    public_source_manifest = dataset_manifest["sources"]["publicAdapterCorpus"]
    assert public_source_manifest["lumenContractSHA256"] == snapshot["lumenContractSHA256"]
    assert public_source_manifest["partitionPolicy"] == snapshot["partitionPolicy"]

    group_lanes: dict[tuple[str, str, str], str] = {}
    for agent, expected_count in snapshot["countsByAgent"].items():
        family = f"public_adapter_corpus_{agent}"
        assert len(datasets[family]) == expected_count
        assert dataset_manifest["hashes"][family] == _records_hash(datasets[family])

        train_public_sft = [
            record
            for record in fine_tuning[agent].train_sft
            if isinstance((record.get("metadata") or {}).get("publicCorpus"), dict)
        ]
        val_public_sft = [
            record
            for record in fine_tuning[agent].val_sft
            if isinstance((record.get("metadata") or {}).get("publicCorpus"), dict)
        ]
        train_public_dpo = [
            record
            for record in fine_tuning[agent].train_dpo
            if isinstance((record.get("metadata") or {}).get("publicCorpus"), dict)
        ]
        val_public_dpo = [
            record
            for record in fine_tuning[agent].val_dpo
            if isinstance((record.get("metadata") or {}).get("publicCorpus"), dict)
        ]
        train_public = [*train_public_sft, *train_public_dpo]
        val_public = [*val_public_sft, *val_public_dpo]
        selected_public = [*train_public, *val_public]
        assert selected_public
        assert all(
            record["metadata"]["publicCorpus"]["targetAdapter"] == agent
            for record in selected_public
        )
        loaded_groups = {
            record["metadata"]["publicCorpus"]["sourceGroupID"]
            for record in datasets[family]
        }
        assert {
            record["metadata"]["publicCorpus"]["sourceGroupID"]
            for record in selected_public
        } <= loaded_groups
        train_groups = {
            record["metadata"]["publicCorpus"]["sourceGroupID"] for record in train_public
        }
        val_groups = {
            record["metadata"]["publicCorpus"]["sourceGroupID"] for record in val_public
        }
        assert train_groups.isdisjoint(val_groups)
        for lane, records in (("train", train_public), ("validation", val_public)):
            for record in records:
                public = record["metadata"]["publicCorpus"]
                key = (
                    public["sourceRepository"],
                    public["sourceRevision"],
                    public["sourceGroupID"],
                )
                assert group_lanes.setdefault(key, lane) == lane
        public_card = fine_tuning[agent].dataset_card["publicCorpus"]
        assert public_card["snapshotIntegrity"]["recordsSHA256"] == snapshot["recordsSHA256"]
        assert public_card["snapshotIntegrity"]["sourceManifestSHA256"] == snapshot["sourceManifestSHA256"]
        assert public_card["selectionContract"]["policyVersions"] == [snapshot["selectionPolicyVersion"]]
        assert public_card["selectionContract"][
            "laneTargetTokenProxyModes"
        ] == {
            "train_sft": "all_assistant",
            "val_sft": "all_assistant",
            "train_dpo": "dpo_chosen",
            "val_dpo": "dpo_chosen",
        }
        assert len(public_card["selectionContract"]["sha256"]) == 64
        assert sum(public_card["recordCounts"].values()) == len(selected_public)
        for lane, selected_count in public_card["recordCounts"].items():
            available_count = public_card["availableRecordCounts"][lane]
            assert selected_count <= available_count
            assert (
                public_card["rejectedByTokenCap"][lane]
                + public_card["rejectedByValidationSampling"][lane]
                == available_count - selected_count
            )
        assert sum(public_card["availableSourceCounts"].values()) == sum(
            public_card["availableRecordCounts"].values()
        )

    assert sum(fine_tuning["mouth"].dataset_card["publicCorpus"]["recordCounts"][lane] for lane in ("train_dpo", "val_dpo")) > 0
    mouth_public = fine_tuning["mouth"].dataset_card["publicCorpus"]
    for lane in ("train_dpo", "val_dpo"):
        assert mouth_public["tokenProxyShares"][lane]["total"] <= 0.35
        assert mouth_public["tokenProxyShares"][lane]["target"] <= 0.35

    for agent in AGENTS:
        public_records = [
            record
            for record in (
                fine_tuning[agent].train_sft
                + fine_tuning[agent].val_sft
                + fine_tuning[agent].train_dpo
                + fine_tuning[agent].val_dpo
            )
            if isinstance((record.get("metadata") or {}).get("publicCorpus"), dict)
        ]
        assert all(record["metadata"]["agent"] == agent for record in public_records)


def test_cortex_keeps_codebase_self_awareness_supplemental(compiled_fine_tuning: tuple) -> None:
    _, datasets, fine_tuning = compiled_fine_tuning
    records = fine_tuning["cortex"].train_sft + fine_tuning["cortex"].val_sft
    cortex_codebase = [
        record for record in records
        if (record.get("metadata") or {}).get("sourceFamily") == CORTEX_CODEBASE_SELF_AWARENESS_SOURCE_FAMILY
    ]
    cortex_chunks = [
        record for record in cortex_codebase
        if (record.get("metadata") or {}).get("recordKind") == "source_chunk"
    ]
    source_chunk_count = len(datasets.get("codebase_home_chunks", []))
    card_quality = fine_tuning["cortex"].dataset_card.get("quality") or {}

    assert len(datasets.get("codebase_home_corpus", [])) >= 700
    assert source_chunk_count >= len(datasets.get("codebase_home_corpus", []))
    assert cortex_codebase
    assert len(cortex_codebase) / len(records) <= 0.25
    assert len(cortex_chunks) < source_chunk_count
    assert CORTEX_CODEBASE_SELF_AWARENESS_SOURCE_FAMILY in fine_tuning["cortex"].dataset_card["sourceFamilies"]
    assert card_quality["cortexCodebaseSelfAwarenessSourceFamily"] == CORTEX_CODEBASE_SELF_AWARENESS_SOURCE_FAMILY
    assert card_quality["cortexCodebaseSelfAwarenessRecordCount"] == len(cortex_codebase)
    assert card_quality["cortexCodebaseSelfAwarenessCoverage"] == "deterministic_supplemental_sample_of_git_tracked_text_files"
    assert card_quality["cortexCodebaseChunkRecordCount"] == len(cortex_chunks)
    assert card_quality["cortexCodebaseSelfAwarenessCandidateRecordCount"] >= source_chunk_count
    assert any("sourceHash" in (record.get("metadata") or {}) for record in cortex_codebase)
    assert any((record.get("metadata") or {}).get("taskType") == "module_ownership_grounding" for record in cortex_codebase)
    assert any((record.get("metadata") or {}).get("taskType") == "source_symbol_grounding" for record in cortex_codebase)
    assert any((record.get("metadata") or {}).get("taskType") == "total_codebase_source_chunk" for record in cortex_codebase)


def test_executor_has_no_model_owned_missing_argument_samples(compiled_fine_tuning: tuple) -> None:
    _, _, fine_tuning = compiled_fine_tuning
    records = fine_tuning["executor"].train_sft + fine_tuning["executor"].val_sft

    assert not any(
        (record.get("metadata") or {}).get("taskType")
        == "ultra_specific_missing_argument_boundary"
        for record in records
    )
    assert all(
        "missingArguments" not in json.loads(record["messages"][2]["content"])
        for record in records
    )


def test_executor_strict_contracts_are_optimizer_visible(compiled_fine_tuning: tuple) -> None:
    manifest, _, fine_tuning = compiled_fine_tuning
    executor = fine_tuning["executor"]
    direct_counts = Counter()
    for record in executor.train_sft:
        metadata = record.get("metadata") or {}
        if metadata.get("taskType") not in {
            "tool_call_generation",
            "ultra_specific_tool_call_generation",
        }:
            continue
        direct_counts.update(metadata.get("toolIDs") or [])
    assert set(direct_counts) == {tool.id for tool in manifest.tools}
    assert min(direct_counts.values()) >= 2

    preference_counts = Counter(
        (record.get("metadata") or {}).get("preferenceType")
        for record in executor.train_dpo
    )
    assert preference_counts["argument_completion"] >= 2
    assert preference_counts["ultra_specific_phone_sms_extraction"] >= 2
    assert preference_counts["approval_boundary"] >= 1
    assert preference_counts["ultra_specific_approval_gate"] >= 1
    assert preference_counts["ultra_specific_permission_gate"] >= 2

    for preference_type in (
        "argument_completion",
        "ultra_specific_phone_sms_extraction",
        "ultra_specific_permission_gate",
    ):
        prompts = {
            _canonical_rendered_prompt_key(record, lane="dpo")
            for record in executor.train_dpo
            if (record.get("metadata") or {}).get("preferenceType")
            == preference_type
        }
        assert len(prompts) >= 2


def test_mouth_concrete_failure_contrasts_are_optimizer_visible(
    compiled_fine_tuning: tuple,
) -> None:
    _, _, fine_tuning = compiled_fine_tuning
    mouth = fine_tuning["mouth"]
    train_preferences = Counter(
        (record.get("metadata") or {}).get("preferenceType")
        for record in mouth.train_dpo
    )
    assert train_preferences["truthful_failure_summary"] >= 1
    assert train_preferences["grounded_observation_tool_failure"] >= 1
    assert train_preferences["grounded_observation_failure_polarity"] >= 1

    concrete_failure_prompts = {
        _canonical_rendered_prompt_key(record, lane="dpo")
        for record in mouth.train_dpo
        if (record.get("metadata") or {}).get("preferenceType")
        in {
            "grounded_observation_tool_failure",
            "grounded_observation_failure_polarity",
        }
    }
    assert len(concrete_failure_prompts) == 2


def test_agent_sft_tool_contracts_include_permission_kind_and_confirmation_mode(compiled_fine_tuning: tuple) -> None:
    manifest, _, fine_tuning = compiled_fine_tuning
    tool = next(item for item in manifest.tools if item.id == "calendar.create")
    expected_contract = {
        "requiresApproval": tool.requiresApproval,
        "permissionKey": tool.permissionKey,
        "permissionKind": tool.permissionKind,
        "confirmationMode": tool.confirmationMode,
    }

    for agent, task_type in (
        ("cortex", "ultra_specific_intent_routing"),
        ("executor", "ultra_specific_tool_call_generation"),
        ("fleet", "ultra_specific_tool_boundary_awareness"),
    ):
        records = fine_tuning[agent].train_sft + fine_tuning[agent].val_sft
        record = next(
            item
            for item in records
            if item["metadata"].get("taskType") == task_type and tool.id in item["metadata"].get("toolIDs", [])
        )
        metadata_contract = record["metadata"]["toolContracts"][tool.id]
        assistant = json.loads(record["messages"][2]["content"])

        assert metadata_contract == expected_contract
        if agent == "cortex":
            assert "permissionKey" not in assistant
            assert "permissionKind" not in assistant
            assert "confirmationMode" not in assistant
        elif agent == "executor":
            assert set(assistant) == {"action"}
            assert set(assistant["action"]) == {"tool", "args"}
            assert assistant["action"]["tool"] == tool.id
        else:
            assert set(assistant) == {
                "toolID",
                "delegateTo",
                "knownSlots",
                "approvalState",
                "permissionState",
            }
            assert "permissionKind" not in assistant
            assert "confirmationMode" not in assistant


def test_cortex_prompts_and_preferred_outputs_enforce_one_json_object(compiled_fine_tuning: tuple) -> None:
    manifest, _, fine_tuning = compiled_fine_tuning
    cortex = fine_tuning["cortex"]
    json_contract = "Return exactly one valid JSON object and nothing else."
    route_system_prompt = cortex_runtime_route_system_prompt(manifest)
    assert route_system_prompt.endswith(CORTEX_ROUTE_DECISION_ENDCAP)
    assert route_system_prompt.index("Manifest tools TSV:") < route_system_prompt.index(
        CORTEX_ROUTE_DECISION_ENDCAP
    )
    catalog_lines = route_system_prompt.splitlines()
    header_index = catalog_lines.index(CORTEX_TOOL_CATALOG_HEADER)
    catalog_rows = {
        fields[0]: fields
        for fields in (
            line.split("\t")
            for line in catalog_lines[
                header_index + 1 : header_index + 1 + len(manifest.tools)
            ]
        )
    }
    assert set(catalog_rows) == {tool.id for tool in manifest.tools}
    allowed_intents_by_tool: dict[str, set[str]] = {}
    for entry in manifest.routingMatrix:
        for tool_id in entry.allowedTools:
            allowed_intents_by_tool.setdefault(tool_id, set()).add(entry.intent)
    for intent in manifest.intents:
        for tool_id in intent.allowedToolIDs:
            allowed_intents_by_tool.setdefault(tool_id, set()).add(intent.id)
    for tool_id, fields in catalog_rows.items():
        assert len(fields) == 7
        default_intent = fields[2]
        allowed_intents = fields[3].split(",")
        assert default_intent == _routed_intent_for_tool(manifest, tool_id)
        assert allowed_intents[0] == default_intent
        assert set(allowed_intents) == allowed_intents_by_tool.get(
            tool_id,
            {"tool"},
        )

    for record in cortex.train_sft + cortex.val_sft:
        assert json_contract in record["messages"][0]["content"]
        assert isinstance(json.loads(record["messages"][-1]["content"]), dict)
    for record in cortex.train_dpo + cortex.val_dpo:
        assert json_contract in record["prompt"][0]["content"]
        assert isinstance(json.loads(record["chosen"]["content"]), dict)
    for record in cortex.eval:
        assert json_contract in record["messages"][0]["content"]

    route_sft = [
        record
        for record in cortex.train_sft + cortex.val_sft
        if record["messages"][0]["content"] == route_system_prompt
    ]
    codebase_sft = [
        record
        for record in cortex.train_sft + cortex.val_sft
        if record["messages"][0]["content"] == CORTEX_CODEBASE_SYSTEM_PROMPT
    ]
    assert route_sft
    assert codebase_sft
    tools_by_id = {tool.id: tool for tool in manifest.tools}
    base_fields = {
        "intent",
        "selectedToolID",
        "requiresApproval",
        "nextModel",
        "reasoningSummary",
    }
    alternate_selection_count = 0
    for record in route_sft:
        payload = json.loads(record["messages"][-1]["content"])
        assert base_fields <= set(payload)
        assert isinstance(payload["intent"], str) and payload["intent"].strip()
        assert isinstance(payload["reasoningSummary"], str)
        assert payload["reasoningSummary"].strip()
        selected_tool_id = payload["selectedToolID"]
        if selected_tool_id is None:
            assert set(payload) == base_fields | {"status"}
            assert payload["requiresApproval"] is False
            assert payload["nextModel"] == "mouth"
            assert payload["status"] in {"no_tool_route", "invalid_tool"}
            continue
        tool = tools_by_id[selected_tool_id]
        assert payload["intent"] in allowed_intents_by_tool.get(
            selected_tool_id,
            {"tool"},
        )
        assert payload["requiresApproval"] is tool.requiresApproval
        if "intent" in record["metadata"]:
            assert record["metadata"]["intent"] == payload["intent"]
        if payload.get("status") == "needs_clarification":
            assert payload["intent"] == _routed_intent_for_tool(
                manifest,
                selected_tool_id,
            )
            assert set(payload) == base_fields | {
                "status",
                "missingArguments",
                "clarification",
            }
            required_arguments = [
                argument.name for argument in tool.arguments if argument.required
            ]
            assert payload["missingArguments"]
            assert payload["missingArguments"] == [
                argument
                for argument in required_arguments
                if argument in payload["missingArguments"]
            ]
            assert payload["nextModel"] == "mouth"
            assert payload["clarification"].endswith("?")
        elif "actionStep" in payload:
            assert payload["intent"] == _routed_intent_for_tool(
                manifest,
                selected_tool_id,
            )
            assert set(payload) == base_fields | {"actionStep"}
            assert payload["actionStep"] == {
                "mustPersistBeforeFinal": True,
                "toolID": selected_tool_id,
                "type": "tool_call",
            }
            assert payload["nextModel"] == (
                "approval" if tool.requiresApproval else "executor"
            )
        else:
            assert set(payload) == base_fields
            if payload["intent"] != _routed_intent_for_tool(
                manifest,
                selected_tool_id,
            ):
                alternate_selection_count += 1
            assert payload["nextModel"] == (
                "approval" if tool.requiresApproval else "executor"
            )
    assert alternate_selection_count > 0
    assert all(
        record["prompt"][0]["content"] == route_system_prompt
        for record in cortex.train_dpo + cortex.val_dpo
    )
    assert all(
        record["messages"][0]["content"] == route_system_prompt
        for record in cortex.eval
    )
    for clause in (
        "Task mode: Cortex route mode.",
        "actionStep exactly",
        "it has no status, missingArguments, or clarification",
        "exact still-missing required arguments in manifest order",
        "status no_tool_route",
            "status invalid_tool",
            "Never emit rejected-tool lists",
            "never construct Executor arguments",
            "single TSV row whose id cell exactly equals it",
            "stop consulting every other row",
            "return routing only",
            "'-' is an empty list",
            "Optional names mentioned in descriptions are never required",
        "Natural wording can supply a value without naming its field",
        "Every actionable or clarification route copies defaultIntent exactly",
        "specifically designated recipient",
        "generic object in operation wording",
        "A separate name or `for <topic>` complement",
        "unresolved relative reference, or bare object class",
        "every action or clarification copies its defaultIntent",
        "five-field explicit choose-only selection",
        "selectedToolID (catalog string or null), intent (string), reasoningSummary",
        "then requiresApproval (boolean) and nextModel (string)",
        "not hidden chain-of-thought",
        "exact selected row and exact missing subset",
        "Finish with requiresApproval and nextModel",
    ):
        assert clause in route_system_prompt
    assert "Task mode: Cortex grounding mode." in CORTEX_CODEBASE_SYSTEM_PROMPT
    assert "source-map, manifest, or self-model evidence" in CORTEX_CODEBASE_SYSTEM_PROMPT
    assert "Do not emit a route, selectedToolID, actionStep" in CORTEX_CODEBASE_SYSTEM_PROMPT


def test_cortex_sft_route_intent_validation_rejects_unlisted_intent(
    compiled_fine_tuning: tuple,
) -> None:
    manifest, _, fine_tuning = compiled_fine_tuning
    route_system_prompt = cortex_runtime_route_system_prompt(manifest)
    record = copy.deepcopy(
        next(
            item
            for item in (
                fine_tuning["cortex"].train_sft
                + fine_tuning["cortex"].val_sft
            )
            if item["messages"][0]["content"] == route_system_prompt
            and json.loads(item["messages"][-1]["content"])["selectedToolID"]
            is not None
        )
    )
    payload = json.loads(record["messages"][-1]["content"])
    payload["intent"] = "inventedAppOperation"
    record["messages"][-1]["content"] = json.dumps(payload)

    with pytest.raises(
        ValueError,
        match="Cortex SFT chosen intent is not allowed",
    ):
        _validate_cortex_sft_route_intents(manifest, [record])


@pytest.mark.parametrize("route_state", ["action", "clarification"])
def test_cortex_sft_route_intent_validation_rejects_allowed_nondefault_stateful_intent(
    compiled_fine_tuning: tuple,
    route_state: str,
) -> None:
    manifest, _, fine_tuning = compiled_fine_tuning
    route_system_prompt = cortex_runtime_route_system_prompt(manifest)
    allowed_intents_by_tool: dict[str, set[str]] = {}
    for entry in manifest.routingMatrix:
        for tool_id in entry.allowedTools:
            allowed_intents_by_tool.setdefault(tool_id, set()).add(entry.intent)
    for manifest_intent in manifest.intents:
        for tool_id in manifest_intent.allowedToolIDs:
            allowed_intents_by_tool.setdefault(tool_id, set()).add(
                manifest_intent.id
            )

    def is_target(record: dict) -> bool:
        if record["messages"][0]["content"] != route_system_prompt:
            return False
        payload = json.loads(record["messages"][-1]["content"])
        tool_id = payload.get("selectedToolID")
        if not isinstance(tool_id, str):
            return False
        default_intent = _routed_intent_for_tool(manifest, tool_id)
        if not (allowed_intents_by_tool.get(tool_id, set()) - {default_intent}):
            return False
        return (
            "actionStep" in payload
            if route_state == "action"
            else payload.get("status") == "needs_clarification"
        )

    record = copy.deepcopy(
        next(
            item
            for item in fine_tuning["cortex"].train_sft
            + fine_tuning["cortex"].val_sft
            if is_target(item)
        )
    )
    payload = json.loads(record["messages"][-1]["content"])
    tool_id = payload["selectedToolID"]
    payload["intent"] = sorted(
        allowed_intents_by_tool[tool_id]
        - {_routed_intent_for_tool(manifest, tool_id)}
    )[0]
    record["messages"][-1]["content"] = json.dumps(payload)

    with pytest.raises(
        ValueError,
        match="must equal the tool default intent",
    ):
        _validate_cortex_sft_route_intents(manifest, [record])


def test_cortex_ordinary_routes_have_exact_action_or_clarification_state_coverage(
    compiled_fine_tuning: tuple,
) -> None:
    manifest, _, fine_tuning = compiled_fine_tuning
    cortex = fine_tuning["cortex"]
    tools_by_id = {tool.id: tool for tool in manifest.tools}
    expected_routes = {
        (intent.id, tool_id): [
            argument.name
            for argument in tools_by_id[tool_id].arguments
            if argument.required
        ]
        for intent in manifest.intents
        for tool_id in intent.allowedToolIDs
        if tool_id in tools_by_id
    }
    frozen_eval_prompts = {record["messages"][-1]["content"] for record in cortex.eval}
    curriculum_records = [
        record
        for record in generate_cortex_records(manifest)
        if record.get("taskType") == "cortex_ordinary_route"
    ]
    generated_route_by_prompt = {
        record["messages"][1]["content"]: (
            record["grounding"]["requestedIntent"],
            record["grounding"]["selectedToolID"],
        )
        for record in curriculum_records
    }
    curriculum_prompts = set(generated_route_by_prompt)
    actual_routes: dict[tuple[str, str], list[tuple[dict, str]]] = {}

    for record in cortex.train_sft + cortex.val_sft:
        if record["metadata"].get("sourceFamily") != "cortex_routing":
            continue
        prompt = record["messages"][1]["content"]
        if prompt not in curriculum_prompts:
            continue
        output = json.loads(record["messages"][-1]["content"])
        route = generated_route_by_prompt[prompt]
        assert route in expected_routes
        assert output.get("selectedToolID") == route[1]
        assert output.get("intent") == _routed_intent_for_tool(
            manifest,
            route[1],
        )
        actual_routes.setdefault(route, []).append((output, prompt))

    assert set(actual_routes) == set(expected_routes)
    ordinary_prompts = [
        prompt
        for route_records in actual_routes.values()
        for _, prompt in route_records
    ]
    assert len(set(ordinary_prompts)) == len(ordinary_prompts)

    for (intent, tool_id), required_arguments in expected_routes.items():
        route_records = actual_routes[(intent, tool_id)]
        expected_count = 1
        if required_arguments:
            expected_count += 1
        if len(required_arguments) > 1:
            expected_count += len(required_arguments)
        assert len(route_records) == expected_count
        action_records = [
            (output, prompt)
            for output, prompt in route_records
            if "actionStep" in output
        ]
        clarification_records = [
            (output, prompt)
            for output, prompt in route_records
            if output.get("status") == "needs_clarification"
        ]
        assert len(action_records) == 1
        expected_clarification_count = (1 if required_arguments else 0) + (
            len(required_arguments) if len(required_arguments) > 1 else 0
        )
        assert len(clarification_records) == expected_clarification_count
        if required_arguments:
            missing_argument_sets = [
                output["missingArguments"]
                for output, _ in clarification_records
            ]
            assert missing_argument_sets.count(required_arguments) == 1
            if len(required_arguments) > 1:
                assert Counter(
                    missing_arguments[0]
                    for missing_arguments in missing_argument_sets
                    if len(missing_arguments) == 1
                ) == Counter(required_arguments)

        for output, prompt in route_records:
            has_action = "actionStep" in output
            needs_clarification = output.get("status") == "needs_clarification"
            assert has_action is not needs_clarification
            assert "arguments" not in output
            assert "rejectedToolID" not in output
            assert "rejectedToolIDs" not in output
            assert prompt not in frozen_eval_prompts
            assert "Args:" not in prompt
            assert output["requiresApproval"] is tools_by_id[tool_id].requiresApproval

            if has_action:
                assert set(output) == {
                    "intent",
                    "selectedToolID",
                    "requiresApproval",
                    "nextModel",
                    "reasoningSummary",
                    "actionStep",
                }
                assert output["actionStep"] == {
                    "type": "tool_call",
                    "toolID": tool_id,
                    "mustPersistBeforeFinal": True,
                }
                assert output["nextModel"] == (
                    "approval" if tools_by_id[tool_id].requiresApproval else "executor"
                )
                assert "status" not in output
                assert "clarification" not in output
                assert "missingArguments" not in output
                if required_arguments:
                    assert prompt.startswith("Please use ")
                    assert f"for my {intent} request" in prompt
                    assert all(argument in prompt for argument in required_arguments)
            else:
                assert set(output) == {
                    "intent",
                    "selectedToolID",
                    "requiresApproval",
                    "nextModel",
                    "reasoningSummary",
                    "status",
                    "missingArguments",
                    "clarification",
                }
                assert output["nextModel"] == "mouth"
                missing_arguments = output["missingArguments"]
                if missing_arguments == required_arguments:
                    assert prompt.startswith("Could Lumen help me use ")
                    assert f"for my {intent} request" in prompt
                    assert not any(
                        f"{argument} is " in prompt
                        for argument in required_arguments
                    )
                else:
                    assert len(missing_arguments) == 1
                    assert missing_arguments[0] in required_arguments
                    assert prompt.startswith("Please use ")
                    assert f"for my {intent} request" in prompt
                    assert f"{missing_arguments[0]} is " not in prompt
                    for supplied_argument in required_arguments:
                        if supplied_argument not in missing_arguments:
                            assert supplied_argument in prompt
                assert not any(
                    cue in prompt.casefold()
                    for cue in (
                        "missing",
                        "absent",
                        "do not persist",
                        "required details",
                        "ask only",
                    )
                )

    trigger_output, trigger_prompt = actual_routes[("trigger", "trigger.list")][0]
    assert trigger_output["selectedToolID"] == "trigger.list"
    assert trigger_prompt == "Show the Lumen automations that are currently scheduled to run."
    assert trigger_prompt != "List my active triggers."

    train_route_outputs = [
        json.loads(record["messages"][-1]["content"])
        for record in cortex.train_sft
        if record["metadata"].get("sourceFamily") == "cortex_routing"
    ]
    assert {
        output["selectedToolID"]
        for output in train_route_outputs
        if isinstance(output.get("actionStep"), dict)
    } == {tool_id for _, tool_id in expected_routes}
    required_tool_ids = {
        tool_id
        for (_, tool_id), required_arguments in expected_routes.items()
        if required_arguments
    }
    assert {
        output["selectedToolID"]
        for output in train_route_outputs
        if output.get("status") == "needs_clarification"
    } == required_tool_ids

    validation_modes = {
        (
            "clarification"
            if output.get("status") == "needs_clarification"
            else "action"
            if isinstance(output.get("actionStep"), dict)
            else "selection"
            if output.get("selectedToolID") is not None
            else "null"
        )
        for output in (
            json.loads(record["messages"][-1]["content"])
            for record in cortex.val_sft
            if record["metadata"].get("sourceFamily") == "cortex_routing"
        )
    }
    assert {"action", "clarification", "selection"} <= validation_modes


def test_cortex_manifest_route_curriculum_is_balanced_and_auditable() -> None:
    no_argument_tool = ToolManifest(
        id="device.status",
        displayName="Device Status",
    )
    single_argument_tool = ToolManifest(
        id="files.lookup",
        displayName="File Lookup",
        arguments=[
            ToolArgumentManifest(name="name", type="string", required=True),
        ],
    )
    multi_argument_tool = ToolManifest(
        id="messages.compose",
        displayName="Message Composer",
        requiresApproval=True,
        permissionKey="NSContactsUsageDescription",
        permissionKind="contacts",
        confirmationMode="userApproval",
        arguments=[
            ToolArgumentManifest(name="recipient", type="string", required=True),
            ToolArgumentManifest(name="body", type="string", required=True),
            ToolArgumentManifest(
                name="channel",
                type="string",
                required=True,
                allowedValues=["sms", "imessage"],
            ),
            ToolArgumentManifest(name="signature", type="string", required=False),
        ],
    )
    manifest = AgentBehaviorManifest(
        tools=[no_argument_tool, single_argument_tool, multi_argument_tool],
        intents=[
            IntentManifest(
                id="testRoute",
                allowedToolIDs=[
                    no_argument_tool.id,
                    single_argument_tool.id,
                    multi_argument_tool.id,
                ],
            ),
        ],
    )

    records = [
        record
        for record in generate_cortex_records(manifest)
        if record.get("taskType") == "cortex_ordinary_route"
    ]
    records_by_tool: dict[str, list[dict]] = {}
    for record in records:
        tool_id = record["grounding"]["selectedToolID"]
        records_by_tool.setdefault(tool_id, []).append(record)

    assert {tool_id: len(tool_records) for tool_id, tool_records in records_by_tool.items()} == {
        "device.status": 1,
        "files.lookup": 2,
        "messages.compose": 5,
    }
    expected_modes = {
        "device.status": Counter({"actionable_complete": 1}),
        "files.lookup": Counter(
            {
                "actionable_complete": 1,
                "needs_clarification_all_missing": 1,
            }
        ),
        "messages.compose": Counter(
            {
                "actionable_complete": 1,
                "needs_clarification_all_missing": 1,
                "needs_clarification_partial_missing": 3,
            }
        ),
    }
    tools_by_id = {tool.id: tool for tool in manifest.tools}

    for tool_id, tool_records in records_by_tool.items():
        assert Counter(
            record["grounding"]["curriculumMode"] for record in tool_records
        ) == expected_modes[tool_id]
        assert len({record["messages"][1]["content"] for record in tool_records}) == len(
            tool_records
        )
        for record in tool_records:
            output = record["messages"][-1]["content"]
            grounding = record["grounding"]
            assert isinstance(record["messages"][1]["content"], str)
            assert record["messages"][1]["content"].strip()
            assert grounding["routeState"] == grounding["curriculumMode"]
            assert grounding["selectedToolID"] == tool_id
            assert "arguments" not in output
            assert "permissionKey" not in output
            assert "permissionKind" not in output
            assert "confirmationMode" not in output
            assert output["requiresApproval"] is grounding["requiresApproval"]
            assert grounding["permissionKey"] == tools_by_id[tool_id].permissionKey
            assert grounding["permissionKind"] == tools_by_id[tool_id].permissionKind
            assert grounding["confirmationMode"] == tools_by_id[tool_id].confirmationMode
            if grounding["curriculumMode"] == "actionable_complete":
                assert set(output) == {
                    "intent",
                    "selectedToolID",
                    "requiresApproval",
                    "nextModel",
                    "reasoningSummary",
                    "actionStep",
                }
                assert output["actionStep"] == {
                    "type": "tool_call",
                    "toolID": tool_id,
                    "mustPersistBeforeFinal": True,
                }
                assert "status" not in output
                assert grounding["missingArguments"] == []
            else:
                assert set(output) == {
                    "intent",
                    "selectedToolID",
                    "requiresApproval",
                    "nextModel",
                    "reasoningSummary",
                    "status",
                    "missingArguments",
                    "clarification",
                }
                assert output["status"] == "needs_clarification"
                assert output["missingArguments"] == grounding["missingArguments"]
                assert "actionStep" not in output

    multi_records = records_by_tool["messages.compose"]
    partial_records = [
        record
        for record in multi_records
        if record["grounding"]["curriculumMode"]
        == "needs_clarification_partial_missing"
    ]
    assert Counter(record["grounding"]["missingArguments"][0] for record in partial_records) == Counter(
        ["recipient", "body", "channel"]
    )
    for record in partial_records:
        missing_argument = record["grounding"]["missingArguments"][0]
        assert record["grounding"]["suppliedRequiredArguments"] == [
            argument
            for argument in ["recipient", "body", "channel"]
            if argument != missing_argument
        ]


def test_cortex_contrast_routes_are_compact_allowed_selections(
    compiled_fine_tuning: tuple,
) -> None:
    manifest, _, fine_tuning = compiled_fine_tuning
    cortex = fine_tuning["cortex"]
    tools_by_id = {tool.id: tool for tool in manifest.tools}
    entries_by_intent = {
        entry.intent: entry
        for entry in manifest.routingMatrix
        if entry.allowedTools
        and entry.forbiddenTools
        and entry.allowedTools[0] in tools_by_id
        and any(
            tool_id not in set(entry.forbiddenTools[:5])
            for tool_id in entry.forbiddenTools
        )
    }
    frozen_eval_prompts = {record["messages"][-1]["content"] for record in cortex.eval}
    contrast_records = [
        record
        for record in cortex.train_sft + cortex.val_sft
        if record["metadata"].get("sourceFamily") == "cortex_routing"
        and record["metadata"].get("taskType")
        == "cortex_contrast_route_selection"
    ]

    assert cortex.contamination_report["contaminated"] is False
    assert cortex.contamination_report["matchCount"] == 0
    assert len(contrast_records) == len(entries_by_intent)
    seen_intents: set[str] = set()
    for record in contrast_records:
        prompt = record["messages"][1]["content"]
        raw_output = record["messages"][-1]["content"]
        output = json.loads(raw_output)
        intent = output["intent"]
        entry = entries_by_intent[intent]
        selected_tool_id = output["selectedToolID"]
        selected_tool = tools_by_id[selected_tool_id]
        frozen_eval_window = set(entry.forbiddenTools[:5])
        later_decoys = [
            tool_id
            for tool_id in sorted(entry.forbiddenTools)
            if tool_id not in frozen_eval_window
        ][:3]

        seen_intents.add(intent)
        assert set(output) == {
            "intent",
            "selectedToolID",
            "requiresApproval",
            "nextModel",
            "reasoningSummary",
        }
        assert selected_tool_id == entry.allowedTools[0]
        assert selected_tool_id in entry.allowedTools
        assert raw_output.count('"selectedToolID"') == 1
        assert output["requiresApproval"] is selected_tool.requiresApproval
        assert output["nextModel"] == (
            "approval" if selected_tool.requiresApproval else "executor"
        )
        assert prompt not in frozen_eval_prompts
        assert all(decoy in prompt for decoy in later_decoys)
        assert all(decoy not in prompt for decoy in entry.forbiddenTools[:5])

    assert seen_intents == set(entries_by_intent)
    by_intent = {
        json.loads(record["messages"][-1]["content"])["intent"]: json.loads(
            record["messages"][-1]["content"]
        )
        for record in contrast_records
    }
    assert by_intent["emailDraft"]["selectedToolID"] == "contacts.search"
    assert by_intent["messageDraft"]["selectedToolID"] == "contacts.search"


def test_cortex_contrast_decoys_exclude_unsorted_frozen_eval_window() -> None:
    frozen_eval_window = [
        "zulu.tool",
        "alpha.tool",
        "yankee.tool",
        "beta.tool",
        "xray.tool",
    ]
    later_candidates = ["gamma.tool", "delta.tool", "epsilon.tool"]
    manifest = AgentBehaviorManifest(
        tools=[ToolManifest(id="messages.draft", requiresApproval=True)],
        intents=[
            IntentManifest(id="messageDraft", allowedToolIDs=["messages.draft"]),
        ],
        routingMatrix=[
            RoutingMatrixEntry(
                intent="messageDraft",
                allowedTools=["messages.draft"],
                forbiddenTools=frozen_eval_window + later_candidates,
            ),
        ],
    )
    record = next(
        item
        for item in generate_cortex_records(manifest)
        if item.get("taskType") == "cortex_contrast_route_selection"
    )
    prompt = record["messages"][1]["content"]
    output = record["messages"][-1]["content"]

    assert all(candidate not in prompt for candidate in frozen_eval_window)
    assert all(candidate in prompt for candidate in sorted(later_candidates))
    assert set(output) == {
        "intent",
        "selectedToolID",
        "requiresApproval",
        "nextModel",
        "reasoningSummary",
    }


def test_cortex_no_tool_and_invalid_tool_routes_fail_closed(
    compiled_fine_tuning: tuple,
) -> None:
    _, _, fine_tuning = compiled_fine_tuning
    cortex = fine_tuning["cortex"]
    records = cortex.train_sft + cortex.val_sft
    frozen_eval_prompts = {record["messages"][-1]["content"] for record in cortex.eval}
    no_tool_records = [
        record
        for record in records
        if record["metadata"].get("taskType") == "cortex_no_tool_route"
    ]
    invalid_tool_records = [
        record
        for record in records
        if record["metadata"].get("taskType") == "cortex_invalid_tool_rejection"
    ]

    assert len(no_tool_records) == 2
    assert {
        json.loads(record["messages"][-1]["content"])["intent"]
        for record in no_tool_records
    } == {"chat", "unknown"}
    for record in no_tool_records:
        prompt = record["messages"][1]["content"]
        raw_output = record["messages"][-1]["content"]
        output = json.loads(raw_output)

        assert set(output) == {
            "intent",
            "selectedToolID",
            "requiresApproval",
            "nextModel",
            "status",
            "reasoningSummary",
        }
        assert output["selectedToolID"] is None
        assert output["requiresApproval"] is False
        assert output["nextModel"] == "mouth"
        assert output["status"] == "no_tool_route"
        assert raw_output.count('"selectedToolID"') == 1
        assert "actionStep" not in output
        assert "arguments" not in output
        assert "rejectedToolID" not in output
        assert "rejectedToolIDs" not in output
        assert prompt not in frozen_eval_prompts

    assert len(invalid_tool_records) == 1
    invalid_record = invalid_tool_records[0]
    invalid_prompt = invalid_record["messages"][1]["content"]
    raw_invalid_output = invalid_record["messages"][-1]["content"]
    invalid_output = json.loads(raw_invalid_output)
    assert set(invalid_output) == {
        "intent",
        "selectedToolID",
        "requiresApproval",
        "nextModel",
        "status",
        "reasoningSummary",
    }
    assert invalid_output["intent"] == "unknown"
    assert invalid_output["selectedToolID"] is None
    assert invalid_output["requiresApproval"] is False
    assert invalid_output["nextModel"] == "mouth"
    assert invalid_output["status"] == "invalid_tool"
    assert raw_invalid_output.count('"selectedToolID"') == 1
    assert "actionStep" not in invalid_output
    assert "arguments" not in invalid_output
    assert "rejectedToolID" not in invalid_output
    assert "rejectedToolIDs" not in invalid_output
    assert invalid_prompt not in frozen_eval_prompts


def test_cortex_sft_null_route_targets_have_explicit_fail_closed_status(
    compiled_fine_tuning: tuple,
) -> None:
    _, _, fine_tuning = compiled_fine_tuning
    records = fine_tuning["cortex"].train_sft + fine_tuning["cortex"].val_sft
    null_route_count = 0

    for record in records:
        output = json.loads(record["messages"][-1]["content"])
        if "selectedToolID" not in output or output["selectedToolID"] is not None:
            continue
        null_route_count += 1
        assert output.get("status") in {"no_tool_route", "invalid_tool"}
        assert output.get("nextModel") == "mouth"
        assert output.get("requiresApproval") is False
        assert "actionStep" not in output
        assert "arguments" not in output
        assert "rejectedToolID" not in output
        assert "rejectedToolIDs" not in output

    assert null_route_count >= 13


def test_cortex_ordinary_route_targets_do_not_enumerate_rejected_tools(
    compiled_fine_tuning: tuple,
) -> None:
    _, _, fine_tuning = compiled_fine_tuning
    records = fine_tuning["cortex"].train_sft + fine_tuning["cortex"].val_sft

    for record in records:
        output = json.loads(record["messages"][-1]["content"])
        if output.get("selectedToolID") is None:
            continue
        assert "rejectedToolID" not in output
        assert "rejectedToolIDs" not in output

    ultra_specific = [
        json.loads(record["messages"][-1]["content"])
        for record in records
        if record["metadata"].get("taskType") == "ultra_specific_intent_routing"
    ]
    assert ultra_specific
    assert all("rejectedToolID" not in output for output in ultra_specific)
    assert all("rejectedToolIDs" not in output for output in ultra_specific)


def test_cortex_ultra_specific_routes_are_complete_persisted_actions(
    compiled_fine_tuning: tuple,
) -> None:
    manifest, _, fine_tuning = compiled_fine_tuning
    cortex = fine_tuning["cortex"]
    tools_by_id = {tool.id: tool for tool in manifest.tools}
    expected_routes = {
        (entry.intent, entry.allowedTools[0])
        for entry in manifest.routingMatrix
        if entry.allowedTools and entry.allowedTools[0] in tools_by_id
    }
    frozen_eval_prompts = {record["messages"][-1]["content"] for record in cortex.eval}
    records = [
        record
        for record in cortex.train_sft + cortex.val_sft
        if record["metadata"].get("taskType") == "ultra_specific_intent_routing"
    ]
    actual_routes: set[tuple[str, str]] = set()

    assert len(records) == len(expected_routes)
    for record in records:
        prompt = record["messages"][1]["content"]
        output = json.loads(record["messages"][-1]["content"])
        tool_id = output["selectedToolID"]
        requested_intent = record["metadata"].get(
            "requestedIntent",
            record["metadata"]["intent"],
        )
        route = (requested_intent, tool_id)
        tool = tools_by_id[tool_id]
        required_arguments = [
            argument.name for argument in tool.arguments if argument.required
        ]

        actual_routes.add(route)
        assert output["intent"] == _routed_intent_for_tool(manifest, tool_id)
        assert record["metadata"]["intent"] == output["intent"]
        assert output["actionStep"] == {
            "type": "tool_call",
            "toolID": tool_id,
            "mustPersistBeforeFinal": True,
        }
        assert "status" not in output
        assert "missingArguments" not in output
        assert "arguments" not in output
        assert "rejectedToolID" not in output
        assert "rejectedToolIDs" not in output
        assert prompt not in frozen_eval_prompts
        if required_arguments:
            assert prompt.startswith("All required details are supplied for ")
            assert "without constructing Executor arguments" in prompt

    assert actual_routes == expected_routes


def test_cortex_frozen_evals_never_construct_executor_arguments(
    compiled_fine_tuning: tuple,
) -> None:
    _, _, fine_tuning = compiled_fine_tuning
    cortex_evals = fine_tuning["cortex"].eval

    assert cortex_evals
    for record in cortex_evals:
        expected = record.get("expected") or {}
        assert "arguments" not in expected
        assert "requiredArguments" not in expected
        assert "risk" not in expected
        for metric in record["metrics"]:
            assert not (
                metric.get("type") == "json_field_equals"
                and "arguments" in (metric.get("candidatePaths") or [])
            )
            assert not (
                metric.get("type") == "json_fields_present"
                and any(
                    str(path).startswith("arguments.")
                    for path in metric.get("paths") or []
                )
            )

    for eval_type in (
        "approval_boundary_routing",
        "permission_boundary_routing",
    ):
        record = next(
            item
            for item in cortex_evals
            if item["metadata"].get("evalType") == eval_type
        )
        prompt = record["messages"][-1]["content"]
        assert f"`{record['expected']['selectedToolID']}` action" in prompt
        assert "Return exactly the five-field selection object" in prompt
        assert "Do not emit actionStep" in prompt
        assert "do not construct Executor arguments" in prompt

    approval = next(
        item
        for item in cortex_evals
        if item["metadata"].get("evalType") == "approval_boundary_routing"
    )
    assert approval["expected"]["requiresApproval"] is True
    assert [metric["type"] for metric in approval["metrics"]] == [
        "manifest_tool_call",
        "approval_boundary",
        "cortex_route_contract",
    ]

    permission = next(
        item
        for item in cortex_evals
        if item["metadata"].get("evalType") == "permission_boundary_routing"
    )
    assert "permissionKey" in permission["expected"]
    assert "outputPermissionKey" not in permission["expected"]
    assert [metric["type"] for metric in permission["metrics"]] == [
        "manifest_tool_call",
        "cortex_route_contract",
    ]
    assert not any(
        metric["type"] == "unsupported_contract"
        for record in (approval, permission)
        for metric in record["metrics"]
    )


def test_every_cortex_eval_has_one_record_aware_route_contract(
    compiled_fine_tuning: tuple,
) -> None:
    manifest, _, fine_tuning = compiled_fine_tuning
    cortex_evals = fine_tuning["cortex"].eval
    tools_by_id = {tool.id: tool for tool in manifest.tools}
    modes: Counter[str] = Counter()

    assert len(cortex_evals) == 507
    for record in cortex_evals:
        contracts = [
            metric
            for metric in record["metrics"]
            if metric.get("type") == "cortex_route_contract"
        ]
        assert len(contracts) == 1
        contract = contracts[0]
        mode = contract["mode"]
        modes[mode] += 1
        assert isinstance(contract.get("expectedIntent"), str)
        assert contract["expectedIntent"]
        selected_tool_id = record["expected"].get("selectedToolID")
        if isinstance(selected_tool_id, str):
            assert contract["expectedIntent"] == _routed_intent_for_tool(
                manifest,
                selected_tool_id,
            )
        elif record["metadata"].get("evalType") == "routing_matrix_adherence":
            name = str(record["metadata"].get("name") or "")
            assert contract["expectedIntent"] == name.removeprefix("route-")

        if mode == "actionable":
            assert contract["expectedToolID"] == record["expected"]["selectedToolID"]
        elif mode == "clarification":
            tool_id = record["expected"]["selectedToolID"]
            assert contract["expectedToolID"] == tool_id
            assert contract["requiredArguments"] == record["expected"][
                "missingArguments"
            ]
            required_arguments = [
                argument.name
                for argument in tools_by_id[tool_id].arguments
                if argument.required
            ]
            assert contract["requiredArguments"] == [
                argument
                for argument in required_arguments
                if argument in contract["requiredArguments"]
            ]
        elif mode == "selection":
            expected_tool_id = record["expected"].get("selectedToolID")
            if isinstance(expected_tool_id, str):
                assert contract["allowedToolIDs"] == [expected_tool_id]
            else:
                assert contract["allowedToolIDs"] == record["expected"][
                    "allowedToolIDs"
                ]
        elif mode == "no_tool_route":
            assert contract["expectedIntent"] in {"chat", "unknown"}
        else:
            assert mode == "invalid_tool"
            assert contract["expectedIntent"] == "unknown"

    assert modes == Counter(
        {
                "actionable": 285,
                "clarification": 197,
            "selection": 22,
            "no_tool_route": 2,
            "invalid_tool": 1,
        }
    )


def test_cortex_reasoning_summaries_do_not_serialize_sibling_fields(compiled_fine_tuning: tuple) -> None:
    _, _, fine_tuning = compiled_fine_tuning
    records = fine_tuning["cortex"].train_sft + fine_tuning["cortex"].val_sft
    ultra_specific = [
        record
        for record in records
        if record["metadata"].get("taskType") == "ultra_specific_intent_routing"
    ]
    forbidden_fragments = (
        "approval=",
        "permission=",
        "permissionKind=",
        "confirmationMode=",
        "requiresApproval",
        "selectedToolID",
    )

    assert ultra_specific
    for record in ultra_specific:
        summary = json.loads(record["messages"][-1]["content"])["reasoningSummary"]
        assert summary.endswith(".")
        assert not any(fragment in summary for fragment in forbidden_fragments)


def test_cortex_field_splice_preference_keeps_invalid_output_rejected(compiled_fine_tuning: tuple) -> None:
    _, _, fine_tuning = compiled_fine_tuning
    cortex = fine_tuning["cortex"]
    records = cortex.train_dpo + cortex.val_dpo
    record = next(
        item
        for item in records
        if item["metadata"].get("preferenceType")
        == "ultra_specific_cortex_json_field_splice_repair"
    )
    chosen = json.loads(record["chosen"]["content"])
    frozen_eval_prompts = {item["messages"][1]["content"] for item in cortex.eval}

    assert chosen["selectedToolID"] == "files.read"
    assert chosen["requiresApproval"] is False
    assert "arguments" not in chosen
    assert chosen["actionStep"]["toolID"] == "files.read"
    assert record["prompt"][1]["content"] not in frozen_eval_prompts
    assert record in cortex.train_dpo
    assert record not in cortex.val_dpo
    assert record["metadata"]["requiredSplit"] == "train"
    with pytest.raises(json.JSONDecodeError):
        json.loads(record["rejected"]["content"])


def test_cortex_strict_json_retry_preferences_cover_train_and_validation(
    compiled_fine_tuning: tuple,
) -> None:
    _, _, fine_tuning = compiled_fine_tuning
    cortex = fine_tuning["cortex"]
    train = [
        record
        for record in cortex.train_dpo
        if record["metadata"].get("preferenceType")
        == "strict_json_retry_runaway_compaction"
    ]
    validation = [
        record
        for record in cortex.val_dpo
        if record["metadata"].get("preferenceType")
        == "strict_json_retry_runaway_compaction"
    ]
    frozen_eval_prompts = {record["messages"][-1]["content"] for record in cortex.eval}
    evaluator_module = ast.parse(
        (_repo_root() / "tools/fine_tuning/unsloth/evaluate_adapter.py").read_text(
            encoding="utf-8"
        )
    )
    evaluator_retry_instruction = next(
        ast.literal_eval(node.value)
        for node in evaluator_module.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name)
            and target.id == "STRICT_JSON_RETRY_INSTRUCTION"
            for target in node.targets
        )
    )

    assert len(train) == 1
    assert len(validation) == 1
    assert STRICT_JSON_RETRY_DPO_INSTRUCTION == evaluator_retry_instruction
    assert (
        "Do not emit a tool catalog, a rejected-tool list, repeated keys, or an unbounded array."
        in STRICT_JSON_RETRY_DPO_INSTRUCTION
    )
    assert train[0]["metadata"]["requiredSplit"] == "train"
    assert validation[0]["metadata"]["requiredSplit"] == "validation"
    assert json.loads(train[0]["chosen"]["content"])["selectedToolID"] == "trigger.list"
    assert json.loads(validation[0]["chosen"]["content"])["selectedToolID"] == "alarm.list"

    prompt_prefixes = {
        "trigger.list": (
            "The user asks to show the Lumen automations that are currently scheduled to run."
        ),
        "alarm.list": (
            "The user asks to show the device alarms that are active right now."
        ),
    }
    expected_retry_suffix = (
        "\n\n"
        + STRICT_JSON_RETRY_DPO_INSTRUCTION
        + " Validation failure code: invalid_json. Use that code only to re-check "
        "the response contract; do not invent missing user values. "
        + _CORTEX_RETRY_GUIDANCE_BY_FAILURE_CODE["invalid_json"]
    )
    for record in train + validation:
        prompt = record["prompt"][1]["content"]
        chosen = json.loads(record["chosen"]["content"])
        rejected = record["rejected"]["content"]
        assert prompt == prompt_prefixes[chosen["selectedToolID"]] + expected_retry_suffix
        assert prompt.count(STRICT_JSON_RETRY_DPO_INSTRUCTION) == 1
        assert "Trusted selected manifest row" not in prompt
        assert "Lock to this row and do not borrow fields" not in prompt
        assert (
            "single bounded retry after strict raw JSON or manifest-route "
            "validation failed"
        ) in prompt
        assert prompt not in frozen_eval_prompts
        assert chosen["actionStep"] == {
            "mustPersistBeforeFinal": True,
            "toolID": chosen["selectedToolID"],
            "type": "tool_call",
        }
        assert "rejectedToolID" not in chosen
        assert "rejectedToolIDs" not in chosen
        assert rejected.count("rejectedToolIDs") >= 2
        with pytest.raises(json.JSONDecodeError):
            json.loads(rejected)


def test_cortex_strict_retry_sft_matches_runtime_failure_code_form(
    compiled_fine_tuning: tuple,
) -> None:
    manifest, _, fine_tuning = compiled_fine_tuning
    tools_by_id = {tool.id: tool for tool in manifest.tools}
    zero_required_tool_ids = {
        tool_id
        for tool_id, tool in _cortex_routed_tools(manifest).items()
        if not _required_argument_names(tool)
    }
    records = [
        record
        for record in _optimized_cortex_training_lane(fine_tuning, "train_sft")
        if record["metadata"].get("taskType")
        == "cortex_route_strict_retry_repair"
    ]
    frozen_eval_prompts = {
        record["messages"][-1]["content"]
        for record in fine_tuning["cortex"].eval
    }
    expected_codes = {
        "invalid_json",
        "cortex_route_tool_not_in_manifest",
        "cortex_route_intent_not_in_manifest",
        "cortex_route_protocol_field_invalid",
        "cortex_route_approval_mismatch",
        "cortex_route_clarification_state_invalid",
        "cortex_route_action_state_invalid",
    }
    systematic_records = [
        record
        for record in records
        if record["metadata"].get("surfaceForm")
        == "systematic_zero_required_strict_retry"
    ]
    systematic_failure_codes = {
        "cortex_route_clarification_state_invalid",
        "cortex_route_protocol_field_invalid",
    }

    assert len(zero_required_tool_ids) == 15
    assert len(records) == 42
    assert len(systematic_records) == 30
    assert Counter(
        (
            json.loads(record["messages"][-1]["content"])["selectedToolID"],
            record["metadata"]["failureCode"],
        )
        for record in systematic_records
    ) == Counter(
        {
            (tool_id, failure_code): 1
            for tool_id in zero_required_tool_ids
            for failure_code in systematic_failure_codes
        }
    )
    assert {
        record["metadata"].get("failureCode") for record in records
    } == expected_codes
    user_prompts = {
        "retry_unknown_tool_exact_catalog_reselection": (
            "Send an email now to maya.chen@example.com with subject Friday "
            "rehearsal and body: rehearsal starts at 6:30 PM this Friday."
        ),
        "retry_invalid_intent_default_reselection": (
            "Remember that I prefer compact runtime summaries as a preference."
        ),
        "retry_zero_required_protocol_fields": (
            "Display every alarm configured on this phone."
        ),
        "retry_invalid_json_without_trusted_row": (
            "Rebuild the search index for my imported local files and PDFs."
        ),
        "retry_protocol_fields_without_trusted_row": (
            "Show my Outlook folders with their unread and total counts."
        ),
        "retry_complete_approval_contract": "Cancel alarm alarm-retry-117.",
        "retry_deictic_clarification_contract": (
            "Forward this Outlook email to noa@example.com."
        ),
        "retry_partial_clarification_contract": (
            "Start a three-minute countdown."
        ),
        "retry_action_persistence_literal_true": (
            "Find skyline photographs from last month in my photo library."
        ),
        "retry_outlook_latest_read_exact_catalog_reselection": (
            "Please open latest correspondence delivered through Microsoft 365."
        ),
        "retry_files_read_clarification_reselection": (
            "Load an unidentified document from local imports."
        ),
        "retry_calendar_event_exact_catalog_reselection": (
            "Arrange a compliance workshop on my calendar during an "
            "unspecified afternoon."
        ),
    }
    untrusted_retry_cases = {
        "retry_unknown_tool_exact_catalog_reselection",
        "retry_invalid_json_without_trusted_row",
        "retry_protocol_fields_without_trusted_row",
        "retry_outlook_latest_read_exact_catalog_reselection",
        "retry_files_read_clarification_reselection",
        "retry_calendar_event_exact_catalog_reselection",
    }
    assert {
        record["metadata"]["repairCase"]
        for record in records
        if record not in systematic_records
    } == set(user_prompts)
    trusted_row_prompts = 0
    for record in records:
        prompt = record["messages"][-2]["content"]
        failure_code = record["metadata"]["failureCode"]
        repair_case = record["metadata"]["repairCase"]
        payload = json.loads(record["messages"][-1]["content"])
        tool = tools_by_id[payload["selectedToolID"]]
        trusted_row = json.dumps(
            {
                "selectedToolID": tool.id,
                "defaultIntent": _routed_intent_for_tool(manifest, tool.id),
                "requiredArguments": [
                    argument.name
                    for argument in tool.arguments
                    if argument.required
                ],
                "requiresApproval": tool.requiresApproval,
            },
            separators=(",", ":"),
        )
        assert prompt not in frozen_eval_prompts
        assert f"Validation failure code: {failure_code}." in prompt
        assert STRICT_JSON_RETRY_DPO_INSTRUCTION in prompt
        assert "do not invent missing user values." in prompt
        if failure_code in _CORTEX_RETRY_GUIDANCE_BY_FAILURE_CODE:
            assert _CORTEX_RETRY_GUIDANCE_BY_FAILURE_CODE[failure_code] in prompt
        retry_guidance = _CORTEX_RETRY_GUIDANCE_BY_FAILURE_CODE.get(
            failure_code,
            "Retry repair: re-read the selected manifest row and emit the exact "
            "contracted route state.",
        )
        user_prompt = (
            _CORTEX_NATURAL_IMPLICIT_COMPLETE_PROMPTS[tool.id]
            if record in systematic_records
            else user_prompts[repair_case]
        )
        expected_prompt = (
            user_prompt
            + "\n\n"
            + STRICT_JSON_RETRY_DPO_INSTRUCTION
            + " Validation failure code: "
            + failure_code
            + ". Use that code only to re-check the response contract; do not "
            "invent missing user values. "
            + retry_guidance
        )
        if repair_case in untrusted_retry_cases:
            assert trusted_row not in prompt
            assert "Trusted selected manifest row" not in prompt
            assert "Lock to this row and do not borrow fields" not in prompt
            assert "requiredArguments is empty: emit actionStep" not in prompt
        else:
            trusted_row_prompts += 1
            expected_prompt += (
                " Trusted selected manifest row, derived only by exact "
                "selectedToolID lookup: "
                + trusted_row
                + ". Copy defaultIntent, requiredArguments, and requiresApproval "
                "only from this trusted row. Lock to this row and do not borrow "
                "fields from any other catalog row or from the failed output."
            )
            if not any(argument.required for argument in tool.arguments):
                expected_prompt += (
                    " requiredArguments is empty: emit actionStep and do not emit "
                    "status, missingArguments, or clarification; set nextModel to "
                    "approval when requiresApproval is true, otherwise executor."
                )
            assert trusted_row in prompt
            assert prompt.count("Trusted selected manifest row") == 1
            assert "Lock to this row and do not borrow fields" in prompt
            if not any(argument.required for argument in tool.arguments):
                assert "requiredArguments is empty: emit actionStep" in prompt
        assert prompt == expected_prompt
        assert (
            isinstance(payload.get("actionStep"), dict)
            or payload.get("status") == "needs_clarification"
        )
        assert record["metadata"]["requiredSplit"] == "train"
        if record in systematic_records:
            assert record["metadata"]["targetedFailureFamily"] == (
                "zero_required_strict_retry_action"
            )
            assert payload["actionStep"] == {
                "type": "tool_call",
                "toolID": tool.id,
                "mustPersistBeforeFinal": True,
            }
            assert "status" not in payload
            assert "missingArguments" not in payload
            assert "clarification" not in payload
    assert trusted_row_prompts == 36

    cross_domain_retry_expectations = {
        "retry_outlook_latest_read_exact_catalog_reselection": (
            "outlook.message.read",
            None,
        ),
        "retry_files_read_clarification_reselection": (
            "files.read",
            ["name"],
        ),
    }
    records_by_case = {
        record["metadata"]["repairCase"]: record for record in records
    }
    for repair_case, (
        expected_tool_id,
        expected_missing,
    ) in cross_domain_retry_expectations.items():
        record = records_by_case[repair_case]
        payload = json.loads(record["messages"][-1]["content"])
        assert record["metadata"]["failureCode"] == (
            "cortex_route_tool_not_in_manifest"
        )
        assert record["metadata"]["requiredSplit"] == "train"
        assert record["metadata"]["surfaceForm"] == (
            "strict_retry_cross_domain_route_lock"
        )
        assert record["metadata"]["targetedFailureFamily"] == (
            "outlook_read_files_read_route_lock"
        )
        assert payload["selectedToolID"] == expected_tool_id
        if expected_missing is None:
            assert payload["actionStep"]["toolID"] == expected_tool_id
            assert "missingArguments" not in payload
        else:
            assert payload["missingArguments"] == expected_missing
            assert "actionStep" not in payload

    persistence_retry = next(
        record
        for record in records
        if record["metadata"]["repairCase"]
        == "retry_action_persistence_literal_true"
    )
    persistence_prompt = persistence_retry["messages"][-2]["content"]
    persistence_payload = json.loads(persistence_retry["messages"][-1]["content"])
    assert "mustPersistBeforeFinal true; never emit false." in persistence_prompt
    assert persistence_payload["selectedToolID"] == "photos.search"
    assert persistence_payload["actionStep"]["mustPersistBeforeFinal"] is True


def test_cortex_retires_exposed_email_holdout_and_reserves_fresh_case(
    compiled_fine_tuning: tuple,
) -> None:
    _, datasets, fine_tuning = compiled_fine_tuning
    cortex = fine_tuning["cortex"]
    eval_by_prompt = {
        record["messages"][-1]["content"]: record for record in cortex.eval
    }
    retired_prompt = "Email the supplier with this update."
    fresh_prompt = "Email Jordan Patel directly."

    assert retired_prompt not in eval_by_prompt
    fresh_record = eval_by_prompt[fresh_prompt]
    assert fresh_record["metadata"]["name"] == "tool-scenario-outlook.mail.send-3"
    assert fresh_record["expected"]["selectedToolID"] == "outlook.mail.send"
    assert fresh_record["expected"]["status"] == "needs_clarification"
    assert fresh_record["expected"]["missingArguments"] == ["subject", "body"]

    training_prompts = {
        record["messages"][-2]["content"]
        for record in cortex.train_sft + cortex.val_sft
    } | {
        record["prompt"][-1]["content"]
        for record in cortex.train_dpo + cortex.val_dpo
    }
    assert retired_prompt not in training_prompts
    assert fresh_prompt not in training_prompts

    fresh_tokens = set(re.findall(r"\w+", fresh_prompt.casefold()))
    closest_containment, closest_prompt = max(
        (
            len(fresh_tokens & set(re.findall(r"\w+", prompt.casefold())))
            / min(
                len(fresh_tokens),
                len(set(re.findall(r"\w+", prompt.casefold()))),
            ),
            prompt,
        )
        for prompt in training_prompts
    )
    assert closest_containment < 0.5, (
        "fresh holdout is lexically too close to Cortex training: "
        f"{closest_containment:.3f} against {closest_prompt!r}"
    )

    codebase_training_blob = "\n".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True)
        for family in ("codebase_home_sft", "codebase_home_chunk_sft")
        for record in datasets[family]
    )
    assert fresh_prompt not in codebase_training_blob

    overview = next(
        record
        for record in datasets["codebase_home_corpus"]
        if record.get("path") == "."
    )
    assert overview["metadata"]["evaluationIsolationPolicy"] == (
        "eval_user_segments_corpus_only"
    )
    assert overview["metadata"]["evaluationSensitiveChunkCount"] > 0
    assert overview["metadata"]["evaluationSensitiveSFTExcludedCount"] > 0


def test_codebase_home_sft_contains_no_eval_user_segment(
    compiled_fine_tuning: tuple,
) -> None:
    _, datasets, _ = compiled_fine_tuning
    training_blob = " ".join(
        re.findall(
            r"\w+",
            "\n".join(
                json.dumps(record, ensure_ascii=False, sort_keys=True)
                for family in ("codebase_home_sft", "codebase_home_chunk_sft")
                for record in datasets[family]
            ).casefold(),
            flags=re.UNICODE,
        )
    )
    padded_training_blob = f" {training_blob} "
    leaked_prompts = []
    for record in datasets["eval_scenarios"]:
        for message in record.get("messages", []):
            if message.get("role") != "user":
                continue
            prompt = str(message.get("content") or "")
            normalized_prompt = " ".join(
                re.findall(r"\w+", prompt.casefold(), flags=re.UNICODE)
            )
            if normalized_prompt and f" {normalized_prompt} " in padded_training_blob:
                leaked_prompts.append(prompt)

    assert leaked_prompts == []


def test_cortex_targeted_selection_sft_is_route_only_and_eval_disjoint(
    compiled_fine_tuning: tuple,
) -> None:
    _, _, fine_tuning = compiled_fine_tuning
    records = [
        record
        for record in _optimized_cortex_training_lane(fine_tuning, "train_sft")
        if record["metadata"].get("targetedFailureFamily")
        == "selection_only_route_state"
    ]
    frozen_eval_prompts = {
        record["messages"][-1]["content"]
        for record in fine_tuning["cortex"].eval
    }

    assert {record["metadata"]["repairCase"] for record in records} == {
        "select_only_health_forbidden_decoys",
        "select_only_weather_forbidden_decoys",
    }
    for record in records:
        prompt = record["messages"][-2]["content"]
        payload = json.loads(record["messages"][-1]["content"])
        assert prompt not in frozen_eval_prompts
        assert set(payload) == {
            "selectedToolID",
            "intent",
            "reasoningSummary",
            "requiresApproval",
            "nextModel",
        }
        assert "actionStep" not in payload
        assert "status" not in payload


def test_cortex_contrast_compaction_preferences_cover_train_and_validation(
    compiled_fine_tuning: tuple,
) -> None:
    manifest, _, fine_tuning = compiled_fine_tuning
    cortex = fine_tuning["cortex"]
    tools_by_id = {tool.id: tool for tool in manifest.tools}
    allowed_tools_by_intent = {
        entry.intent: set(entry.allowedTools) for entry in manifest.routingMatrix
    }
    frozen_eval_prompts = {record["messages"][-1]["content"] for record in cortex.eval}
    train = [
        record
        for record in cortex.train_dpo
        if record["metadata"].get("preferenceType")
        == "explicit_contrast_compact_allowed_selection"
    ]
    validation = [
        record
        for record in cortex.val_dpo
        if record["metadata"].get("preferenceType")
        == "explicit_contrast_compact_allowed_selection"
    ]

    assert len(train) == 1
    assert len(validation) == 1
    assert train[0]["metadata"]["requiredSplit"] == "train"
    assert validation[0]["metadata"]["requiredSplit"] == "validation"
    assert json.loads(train[0]["chosen"]["content"])["selectedToolID"] == "messages.draft"
    assert json.loads(validation[0]["chosen"]["content"])["selectedToolID"] == "phone.call"

    for record in train + validation:
        prompt = record["prompt"][1]["content"]
        raw_chosen = record["chosen"]["content"]
        chosen = json.loads(raw_chosen)
        rejected = json.loads(record["rejected"]["content"])
        selected_tool = tools_by_id[chosen["selectedToolID"]]

        assert set(chosen) == {
            "intent",
            "selectedToolID",
            "requiresApproval",
            "nextModel",
            "reasoningSummary",
        }
        assert chosen["requiresApproval"] is selected_tool.requiresApproval
        assert chosen["selectedToolID"] in allowed_tools_by_intent[chosen["intent"]]
        assert raw_chosen.count('"selectedToolID"') == 1
        assert chosen["nextModel"] == (
            "approval" if selected_tool.requiresApproval else "executor"
        )
        assert chosen["reasoningSummary"] == (
            f"Manifest row {selected_tool.id} is selected for intent "
            f"{chosen['intent']} without actionStep."
        )
        assert "actionStep" not in chosen
        assert "rejectedToolID" not in chosen
        assert "rejectedToolIDs" not in chosen
        assert prompt not in frozen_eval_prompts
        assert "later" in prompt
        assert len(rejected["rejectedToolIDs"]) > len(set(rejected["rejectedToolIDs"]))
        assert any(
            tool_id.startswith("invented.")
            for tool_id in rejected["rejectedToolIDs"]
        )


def test_cortex_invalid_tool_preference_fails_closed_without_redirect(
    compiled_fine_tuning: tuple,
) -> None:
    _, _, fine_tuning = compiled_fine_tuning
    cortex = fine_tuning["cortex"]
    records = [
        record
        for record in cortex.train_dpo + cortex.val_dpo
        if record["metadata"].get("preferenceType")
        == "invalid_tool_null_rejection"
    ]
    frozen_eval_prompts = {record["messages"][-1]["content"] for record in cortex.eval}

    assert len(records) == 1
    record = records[0]
    raw_chosen = record["chosen"]["content"]
    chosen = json.loads(raw_chosen)
    rejected = json.loads(record["rejected"]["content"])
    assert record in cortex.train_dpo
    assert record not in cortex.val_dpo
    assert record["metadata"]["requiredSplit"] == "train"
    assert set(chosen) == {
        "intent",
        "selectedToolID",
        "requiresApproval",
        "nextModel",
        "status",
        "reasoningSummary",
    }
    assert chosen["intent"] == "unknown"
    assert chosen["selectedToolID"] is None
    assert chosen["requiresApproval"] is False
    assert chosen["nextModel"] == "mouth"
    assert chosen["status"] == "invalid_tool"
    assert raw_chosen.count('"selectedToolID"') == 1
    assert "actionStep" not in chosen
    assert "rejectedToolID" not in chosen
    assert "rejectedToolIDs" not in chosen
    assert rejected["selectedToolID"] == "trigger.list"
    assert rejected["actionStep"]["toolID"] == "trigger.list"
    assert record["prompt"][1]["content"] not in frozen_eval_prompts


def test_cortex_chosen_route_preferences_are_canonical_or_explicit_contrast(
    compiled_fine_tuning: tuple,
) -> None:
    _, _, fine_tuning = compiled_fine_tuning
    records = fine_tuning["cortex"].train_dpo + fine_tuning["cortex"].val_dpo
    required_action_fields = {
        "intent",
        "selectedToolID",
        "requiresApproval",
        "nextModel",
        "reasoningSummary",
        "actionStep",
    }
    exact_canonical_types = {
        "ultra_specific_calendar_read_routing",
        "ultra_specific_outlook_reference_routing",
        "ultra_specific_local_maps_precedence",
    }
    seen_selected_route = False
    seen_null_route = False

    for record in records:
        chosen = json.loads(record["chosen"]["content"])
        if "selectedToolID" not in chosen:
            continue
        selected_tool_id = chosen["selectedToolID"]
        preference_type = record["metadata"].get("preferenceType")

        if selected_tool_id is None:
            seen_null_route = True
            assert chosen.get("status") in {
                "invalid_tool",
                "no_tool_route",
                "needs_clarification",
            }
            assert chosen.get("nextModel") == "mouth"
            assert chosen.get("requiresApproval") is False
            assert "actionStep" not in chosen
            continue

        seen_selected_route = True
        if preference_type in {
            "explicit_contrast_compact_allowed_selection",
            "manifest_tool_only",
            "route_selection_vs_no_tool_hybrid",
            "route_selection_vs_no_tool_hybrid_validation",
            "route_selection_without_action",
            "safe_tool_selection",
            "selection_only_replay_action_negative",
        } or record["metadata"].get("targetedFailureFamily") == (
            "selection_only_route_state"
        ):
            assert set(chosen) == {
                "intent",
                "selectedToolID",
                "requiresApproval",
                "nextModel",
                "reasoningSummary",
            }
            assert "actionStep" not in chosen
            continue

        if chosen.get("status") == "needs_clarification":
            assert set(chosen) == {
                "intent",
                "selectedToolID",
                "requiresApproval",
                "nextModel",
                "reasoningSummary",
                "status",
                "missingArguments",
                "clarification",
            }
            assert chosen["missingArguments"]
            assert chosen["nextModel"] == "mouth"
            assert chosen["clarification"].endswith("?")
            assert "actionStep" not in chosen
            continue

        assert set(chosen) == required_action_fields
        assert chosen["actionStep"] == {
            "type": "tool_call",
            "toolID": selected_tool_id,
            "mustPersistBeforeFinal": True,
        }
        if preference_type in exact_canonical_types:
            assert set(chosen) == required_action_fields

    assert seen_selected_route is True
    assert seen_null_route is True


def test_cortex_latest_outlook_attachment_reference_persists_symbolic_id(
    compiled_fine_tuning: tuple,
) -> None:
    _, _, fine_tuning = compiled_fine_tuning
    cortex = fine_tuning["cortex"]
    record = next(
        item
        for item in cortex.train_dpo + cortex.val_dpo
        if item["metadata"].get("preferenceType")
        == "ultra_specific_outlook_reference_routing"
    )
    chosen = json.loads(record["chosen"]["content"])
    rejected = json.loads(record["rejected"]["content"])

    assert record["prompt"][-1]["content"] == (
        "Route: Show attachments on the latest Outlook email."
    )
    assert chosen["selectedToolID"] == "outlook.attachments.list"
    assert chosen["actionStep"] == {
        "type": "tool_call",
        "toolID": "outlook.attachments.list",
        "mustPersistBeforeFinal": True,
    }
    assert "missingArguments" not in chosen
    assert rejected["status"] == "needs_clarification"
    assert rejected["missingArguments"] == ["messageId"]
    assert "actionStep" not in rejected


def test_cortex_route_dpo_is_manifest_complete_and_train_anchored(
    compiled_fine_tuning: tuple,
) -> None:
    manifest, _, fine_tuning = compiled_fine_tuning
    cortex = fine_tuning["cortex"]
    tools_by_id = {tool.id: tool for tool in manifest.tools}
    routed_tool_ids = {
        tool_id
        for intent in manifest.intents
        for tool_id in intent.allowedToolIDs
        if tool_id in tools_by_id
    }
    action_anchors = {
        json.loads(record["chosen"]["content"])["selectedToolID"]
        for record in cortex.train_dpo
        if record["metadata"].get("preferenceType")
        == "route_exact_approval_and_next_model"
    }
    clarification_anchors = [
        json.loads(record["chosen"]["content"])
        for record in cortex.train_dpo
        if record["metadata"].get("preferenceType") in {
            "route_natural_all_missing_vs_premature_action",
            "route_natural_partial_missing_vs_premature_action",
        }
    ]

    assert action_anchors == routed_tool_ids
    required_tool_ids = {
        tool_id
        for tool_id in routed_tool_ids
        if any(argument.required for argument in tools_by_id[tool_id].arguments)
    }
    assert {
        payload["selectedToolID"] for payload in clarification_anchors
    } == required_tool_ids
    for tool_id in required_tool_ids:
        required_arguments = [
            argument.name
            for argument in tools_by_id[tool_id].arguments
            if argument.required
        ]
        missing_subsets = {
            tuple(payload["missingArguments"])
            for payload in clarification_anchors
            if payload["selectedToolID"] == tool_id
        }
        assert tuple(required_arguments) in missing_subsets
        if len(required_arguments) > 1:
            expected_proper_subsets = {
                tuple(subset)
                for subset_size in range(1, len(required_arguments))
                for subset in combinations(required_arguments, subset_size)
            }
            assert expected_proper_subsets <= missing_subsets
    assert not [
        record
        for record in cortex.train_dpo + cortex.val_dpo
        if record["metadata"].get("sourceFamily") == "cross_model_training"
    ]
    assert cortex.unsloth_config["dpo_learning_rate"] < cortex.unsloth_config[
        "learning_rate"
    ]
    assert cortex.unsloth_config["learning_rate"] == pytest.approx(0.00015)
    assert cortex.unsloth_config["num_train_epochs"] == 3
    assert cortex.unsloth_config["dpo_learning_rate"] == pytest.approx(0.0000001)
    assert cortex.unsloth_config["dpo_num_train_epochs"] == 1
    assert cortex.unsloth_config["dpo_rpo_alpha"] is None
    assert cortex.unsloth_config["max_prompt_length"] == 3200
    assert cortex.unsloth_config["preference_minimum_prompt_margin_tokens"] == 64
    assert cortex.unsloth_config["preference_minimum_sequence_margin_tokens"] == 128
    assert cortex.unsloth_config["sft_minimum_sequence_margin_tokens"] == 128
    assert cortex.unsloth_config["use_logits_to_keep"] is True
    assert cortex.unsloth_config["precompute_ref_log_probs"] is True
    assert cortex.unsloth_config["precompute_ref_batch_size"] == 1
    assert cortex.unsloth_config["gradient_checkpointing"] is True
    assert cortex.unsloth_config["batch_size"] == 1
    assert cortex.unsloth_config["gradient_accumulation_steps"] == 16


def _optimized_cortex_training_lane(fine_tuning: dict, lane: str) -> list[dict]:
    return fine_tuning["cortex"].experiment_variants[
        "internal_plus_public_optimized"
    ][lane]


def _cortex_routed_tools(
    manifest: AgentBehaviorManifest,
) -> dict[str, ToolManifest]:
    tools_by_id = {tool.id: tool for tool in manifest.tools}
    routed_tool_ids = {
        tool_id
        for entry in manifest.routingMatrix
        for tool_id in entry.allowedTools
        if tool_id in tools_by_id
    } | {
        tool_id
        for intent in manifest.intents
        for tool_id in intent.allowedToolIDs
        if tool_id in tools_by_id
    }
    return {tool_id: tools_by_id[tool_id] for tool_id in routed_tool_ids}


def _required_argument_names(tool: ToolManifest) -> list[str]:
    return [argument.name for argument in tool.arguments if argument.required]


def _load_json_object(value: object) -> dict | None:
    if not isinstance(value, str):
        return None
    try:
        payload = json.loads(value)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def test_cortex_calendar_required_argument_lattice_is_exact_in_optimized_artifact(
    compiled_fine_tuning: tuple,
) -> None:
    manifest, _, fine_tuning = compiled_fine_tuning
    calendar_tool = next(tool for tool in manifest.tools if tool.id == "calendar.create")
    state_contract = {
        "none_supplied": ([], ["title", "startsInMinutes"]),
        "title_only": (["title"], ["startsInMinutes"]),
        "starts_in_minutes_only": (["startsInMinutes"], ["title"]),
        "complete": (["title", "startsInMinutes"], []),
    }
    lane_contract = {
        "train_sft": ("train", 8, 0),
        "val_sft": ("validation", 2, 0),
        "train_dpo": ("train", 8, 1),
        "val_dpo": ("validation", 2, 0),
    }

    for lane, (required_split, expected_groups, expected_extras) in (
        lane_contract.items()
    ):
        records = [
            record
            for record in _optimized_cortex_training_lane(fine_tuning, lane)
            if record["metadata"].get("routeLattice")
            == "calendar_create_required_arguments_v1"
        ]
        core_records = [
            record
            for record in records
            if "routeLatticeExtraCase" not in record["metadata"]
        ]
        extra_records = [
            record
            for record in records
            if "routeLatticeExtraCase" in record["metadata"]
        ]
        assert len(core_records) == expected_groups * len(state_contract)
        assert len(extra_records) == expected_extras
        assert Counter(
            record["metadata"]["routeLatticeState"] for record in core_records
        ) == Counter({state: expected_groups for state in state_contract})

        records_by_group: dict[str, list[dict]] = {}
        for record in core_records:
            metadata = record["metadata"]
            records_by_group.setdefault(
                metadata["routeLatticeSurfaceGroup"], []
            ).append(record)
        assert len(records_by_group) == expected_groups
        for group_records in records_by_group.values():
            assert Counter(
                record["metadata"]["routeLatticeState"]
                for record in group_records
            ) == Counter({state: 1 for state in state_contract})

        for record in core_records:
            metadata = record["metadata"]
            state = metadata["routeLatticeState"]
            supplied_arguments, missing_arguments = state_contract[state]
            assert metadata["requiredSplit"] == required_split
            assert metadata["targetedFailureFamily"] == (
                "calendar_required_argument_lattice_replay"
            )
            assert metadata["suppliedArguments"] == supplied_arguments
            assert metadata["missingArguments"] == missing_arguments
            if lane.endswith("sft"):
                assert metadata["taskType"] == (
                    "cortex_calendar_required_argument_lattice"
                )
                payload = json.loads(record["messages"][-1]["content"])
            else:
                assert metadata["preferenceType"] == (
                    "calendar_required_argument_lattice"
                )
                payload = json.loads(record["chosen"]["content"])
            assert payload["intent"] == "calendar"
            assert payload["selectedToolID"] == calendar_tool.id
            assert payload["requiresApproval"] is calendar_tool.requiresApproval
            if missing_arguments:
                assert payload["status"] == "needs_clarification"
                assert payload["missingArguments"] == missing_arguments
                assert payload["nextModel"] == "mouth"
                assert "actionStep" not in payload
            else:
                assert payload["nextModel"] == "approval"
                assert payload["actionStep"] == {
                    "mustPersistBeforeFinal": True,
                    "toolID": calendar_tool.id,
                    "type": "tool_call",
                }
                assert "status" not in payload
                assert "missingArguments" not in payload


def test_cortex_calendar_lattice_dpo_rejections_cover_exact_bidirectional_matrix(
    compiled_fine_tuning: tuple,
) -> None:
    _, _, fine_tuning = compiled_fine_tuning
    expected_rejection_counts = {
        "train_dpo": Counter(
            {
                "over_clarification": 5,
                "premature_action": 9,
                "selection_without_action": 3,
                "spurious_clarification": 5,
                "under_clarification": 5,
                "wrong_missing_subset": 5,
            }
        ),
        "val_dpo": Counter(
            {
                "over_clarification": 1,
                "premature_action": 3,
                "spurious_clarification": 2,
                "under_clarification": 1,
                "wrong_missing_subset": 1,
            }
        ),
    }
    expected_state_rejections = {
        "train_dpo": Counter(
            {
                ("none_supplied", "under_clarification", ("startsInMinutes",)): 3,
                ("none_supplied", "under_clarification", ("title",)): 2,
                ("none_supplied", "premature_action", ()): 3,
                ("title_only", "premature_action", ()): 3,
                (
                    "title_only",
                    "over_clarification",
                    ("title", "startsInMinutes"),
                ): 3,
                ("title_only", "wrong_missing_subset", ("title",)): 2,
                ("starts_in_minutes_only", "premature_action", ()): 3,
                (
                    "starts_in_minutes_only",
                    "over_clarification",
                    ("title", "startsInMinutes"),
                ): 2,
                (
                    "starts_in_minutes_only",
                    "wrong_missing_subset",
                    ("startsInMinutes",),
                ): 3,
                ("complete", "spurious_clarification", ("title",)): 2,
                (
                    "complete",
                    "spurious_clarification",
                    ("startsInMinutes",),
                ): 1,
                (
                    "complete",
                    "spurious_clarification",
                    ("title", "startsInMinutes"),
                ): 2,
                ("complete", "selection_without_action", ()): 3,
            }
        ),
        "val_dpo": Counter(
            {
                ("none_supplied", "premature_action", ()): 1,
                (
                    "none_supplied",
                    "under_clarification",
                    ("startsInMinutes",),
                ): 1,
                (
                    "title_only",
                    "over_clarification",
                    ("title", "startsInMinutes"),
                ): 1,
                ("title_only", "premature_action", ()): 1,
                ("starts_in_minutes_only", "premature_action", ()): 1,
                (
                    "starts_in_minutes_only",
                    "wrong_missing_subset",
                    ("startsInMinutes",),
                ): 1,
                ("complete", "spurious_clarification", ("title",)): 1,
                (
                    "complete",
                    "spurious_clarification",
                    ("startsInMinutes",),
                ): 1,
            }
        ),
    }
    selection_keys = {
        "intent",
        "nextModel",
        "reasoningSummary",
        "requiresApproval",
        "selectedToolID",
    }

    for lane, expected_counts in expected_rejection_counts.items():
        records = [
            record
            for record in _optimized_cortex_training_lane(fine_tuning, lane)
            if record["metadata"].get("routeLattice")
            == "calendar_create_required_arguments_v1"
            and "routeLatticeExtraCase" not in record["metadata"]
        ]
        assert Counter(
            record["metadata"]["rejectedRouteState"] for record in records
        ) == expected_counts
        assert Counter(
            (
                record["metadata"]["routeLatticeState"],
                record["metadata"]["rejectedRouteState"],
                tuple(record["metadata"]["rejectedMissingArguments"]),
            )
            for record in records
        ) == expected_state_rejections[lane]
        for record in records:
            metadata = record["metadata"]
            assert metadata["targetedFailureFamily"] == (
                "calendar_required_argument_lattice_replay"
            )
            chosen = json.loads(record["chosen"]["content"])
            rejected = json.loads(record["rejected"]["content"])
            assert chosen != rejected
            assert rejected["selectedToolID"] == "calendar.create"
            rejected_state = metadata["rejectedRouteState"]
            rejected_missing = metadata["rejectedMissingArguments"]
            if rejected_state == "premature_action":
                assert rejected["actionStep"]["toolID"] == "calendar.create"
                assert "status" not in rejected
                assert "missingArguments" not in rejected
            elif rejected_state == "selection_without_action":
                assert set(rejected) == selection_keys
                assert metadata["routeLatticeState"] == "complete"
            else:
                assert rejected["status"] == "needs_clarification"
                assert rejected["missingArguments"] == rejected_missing
                assert "actionStep" not in rejected
                if rejected_state == "under_clarification":
                    assert set(rejected_missing) < set(chosen["missingArguments"])
                elif rejected_state == "over_clarification":
                    assert set(rejected_missing) > set(chosen["missingArguments"])
                elif rejected_state == "wrong_missing_subset":
                    assert set(rejected_missing) != set(chosen["missingArguments"])
                else:
                    assert rejected_state == "spurious_clarification"
                    assert "missingArguments" not in chosen

    current_error_records = [
        record
        for record in _optimized_cortex_training_lane(fine_tuning, "train_dpo")
        if record["metadata"].get("routeLatticeExtraCase")
        == "current_error_generic_object_as_title"
    ]
    assert len(current_error_records) == 1
    current_error = current_error_records[0]
    assert current_error["metadata"]["preferenceType"] == (
        "calendar_required_argument_lattice_current_error"
    )
    assert current_error["metadata"]["targetedFailureFamily"] == (
        "calendar_required_argument_lattice_replay"
    )
    assert json.loads(current_error["chosen"]["content"])["missingArguments"] == [
        "title",
        "startsInMinutes",
    ]
    assert json.loads(current_error["rejected"]["content"])[
        "missingArguments"
    ] == ["startsInMinutes"]


def test_cortex_selection_only_replay_has_exact_counts_and_action_negatives(
    compiled_fine_tuning: tuple,
) -> None:
    _, _, fine_tuning = compiled_fine_tuning
    lane_contract = {
        "train_sft": ("train", {"files.read", "health.summary", "memory.recall", "weather"}),
        "val_sft": ("validation", {"photos.search", "rag.search"}),
        "train_dpo": ("train", {"files.read", "health.summary", "memory.recall", "weather"}),
        "val_dpo": ("validation", {"maps.search", "rag.search"}),
    }
    selection_keys = {
        "intent",
        "nextModel",
        "reasoningSummary",
        "requiresApproval",
        "selectedToolID",
    }

    for lane, (required_split, expected_tool_ids) in lane_contract.items():
        records = [
            record
            for record in _optimized_cortex_training_lane(fine_tuning, lane)
            if record["metadata"].get("selectionOnlyReplay")
            == "selection_only_replay_v1"
        ]
        assert len(records) == len(expected_tool_ids)
        assert len(
            {
                record["metadata"]["selectionOnlyReplaySurfaceGroup"]
                for record in records
            }
        ) == len(records)
        chosen_by_tool: dict[str, dict] = {}
        for record in records:
            metadata = record["metadata"]
            assert metadata["requiredSplit"] == required_split
            assert metadata["routeState"] == "selection_only"
            assert metadata["targetedFailureFamily"] == (
                "selection_only_route_state_replay"
            )
            if lane.endswith("sft"):
                assert metadata["taskType"] == "cortex_selection_only_replay"
                chosen = json.loads(record["messages"][-1]["content"])
            else:
                assert metadata["preferenceType"] == (
                    "selection_only_replay_action_negative"
                )
                assert metadata["rejectedRouteState"] == "premature_action"
                chosen = json.loads(record["chosen"]["content"])
                rejected = json.loads(record["rejected"]["content"])
                assert rejected["selectedToolID"] == chosen["selectedToolID"]
                assert rejected["actionStep"] == {
                    "mustPersistBeforeFinal": True,
                    "toolID": chosen["selectedToolID"],
                    "type": "tool_call",
                }
            assert set(chosen) == selection_keys
            chosen_by_tool[chosen["selectedToolID"]] = chosen
        assert set(chosen_by_tool) == expected_tool_ids


def test_cortex_replay_extensions_are_prompt_disjoint_and_below_frozen_containment(
    compiled_fine_tuning: tuple,
) -> None:
    _, _, fine_tuning = compiled_fine_tuning
    cortex = fine_tuning["cortex"]
    extension_prompts: list[str] = []
    train_extension_prompts: list[str] = []
    validation_extension_prompts: list[str] = []
    for lane in ("train_sft", "val_sft", "train_dpo", "val_dpo"):
        for record in _optimized_cortex_training_lane(fine_tuning, lane):
            metadata = record["metadata"]
            if not (
                metadata.get("routeLattice")
                == "calendar_create_required_arguments_v1"
                or metadata.get("selectionOnlyReplay")
                == "selection_only_replay_v1"
            ):
                continue
            prompt = (
                record["messages"][-2]["content"]
                if lane.endswith("sft")
                else record["prompt"][-1]["content"]
            )
            extension_prompts.append(prompt)
            if lane.startswith("train_"):
                train_extension_prompts.append(prompt)
            else:
                validation_extension_prompts.append(prompt)

    frozen_eval_prompts = {
        record["messages"][-1]["content"] for record in cortex.eval
    }
    frozen_hashes = {
        hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        for prompt in frozen_eval_prompts
    }
    assert len(cortex.eval) == 507
    assert len(extension_prompts) == 93
    assert len(set(extension_prompts)) == len(extension_prompts)
    assert set(extension_prompts).isdisjoint(frozen_eval_prompts)
    assert {
        hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        for prompt in extension_prompts
    }.isdisjoint(frozen_hashes)

    for validation_prompt in validation_extension_prompts:
        validation_tokens = set(
            re.findall(r"\w+", validation_prompt.casefold())
        )
        for train_prompt in train_extension_prompts:
            train_tokens = set(re.findall(r"\w+", train_prompt.casefold()))
            containment = len(validation_tokens & train_tokens) / min(
                len(validation_tokens),
                len(train_tokens),
            )
            assert containment < 0.5, (
                "Pilot replay validation prompt is lexically too close to training "
                f"text: {containment:.3f}; validation={validation_prompt!r}; "
                f"train={train_prompt!r}"
            )

    for extension_prompt in extension_prompts:
        extension_tokens = set(re.findall(r"\w+", extension_prompt.casefold()))
        assert extension_tokens
        for frozen_prompt in frozen_eval_prompts:
            frozen_tokens = set(re.findall(r"\w+", frozen_prompt.casefold()))
            containment = len(extension_tokens & frozen_tokens) / min(
                len(extension_tokens),
                len(frozen_tokens),
            )
            assert containment < 0.5, (
                "Pilot replay prompt is lexically too close to frozen evaluation "
                f"text: {containment:.3f}; replay={extension_prompt!r}; "
                f"eval={frozen_prompt!r}"
            )


def test_cortex_replay_extensions_do_not_shrink_historical_replay_floors(
    compiled_fine_tuning: tuple,
) -> None:
    _, _, fine_tuning = compiled_fine_tuning
    historical_floors = {
        "train_sft": {
            "calendar_event_title_without_numeric_delay": 5,
            "calendar_operation_object_not_title": 1,
            "implicit_duration_countdown_missing_title": 6,
            "implicit_preference_memory_save": 10,
            "implicit_topic_memory_recall": 4,
            "outlook_read_files_read_route_lock": 2,
            "outlook_read_reference_resolution": 6,
            "outlook_reply_unresolved_reference_and_body": 16,
            "outlook_send_unresolved_subject_and_body": 4,
            "selection_only_route_state": 2,
            "trigger_cancel_reference_resolution": 5,
            "zero_required_action_without_invented_arguments": 5,
            "zero_required_list_action": 12,
            "zero_required_strict_retry_action": 30,
        },
        "val_sft": {
            "calendar_event_title_without_numeric_delay": 1,
            "calendar_operation_object_not_title": 2,
            "implicit_preference_memory_save": 1,
            "implicit_topic_memory_recall": 1,
            "outlook_read_files_read_route_lock": 1,
            "outlook_read_reference_resolution": 2,
            "outlook_reply_unresolved_reference_and_body": 1,
            "zero_required_list_action": 1,
        },
        "train_dpo": {
            "action_step_persistence_literal_true": 3,
            "calendar_event_title_without_numeric_delay": 4,
            "calendar_operation_object_not_title": 2,
            "implicit_duration_countdown_missing_title": 2,
            "implicit_preference_memory_save": 8,
            "implicit_topic_memory_recall": 2,
            "outlook_read_files_read_route_lock": 4,
            "outlook_read_reference_resolution": 1,
            "outlook_reply_unresolved_reference_and_body": 12,
            "outlook_send_unresolved_subject_and_body": 4,
            "selection_only_route_state": 2,
            "trigger_cancel_reference_resolution": 2,
            "zero_required_action_without_invented_arguments": 4,
            "zero_required_list_action": 7,
        },
        "val_dpo": {
            "calendar_event_title_without_numeric_delay": 2,
            "calendar_operation_object_not_title": 2,
            "implicit_preference_memory_save": 1,
            "implicit_topic_memory_recall": 1,
            "outlook_read_files_read_route_lock": 3,
            "outlook_reply_unresolved_reference_and_body": 1,
            "zero_required_list_action": 1,
        },
    }

    for lane, floors in historical_floors.items():
        observed = Counter(
            record["metadata"].get("targetedFailureFamily")
            for record in _optimized_cortex_training_lane(fine_tuning, lane)
        )
        for family, floor in floors.items():
            assert observed[family] >= floor


def _is_natural_cortex_request(prompt: str) -> bool:
    lowered = prompt.casefold()
    output_directive_cues = (
        "clarification state",
        "do not persist",
        "missing argument",
        "omitted every",
        "persist exactly",
        "repair drill",
        "required-argument",
        "return the exact",
        "route is already known",
    )
    return not any(cue in lowered for cue in output_directive_cues)


def _canonical_cortex_action_tool_ids(
    records: list[dict],
    routed_tools: dict[str, ToolManifest],
) -> set[str]:
    covered: set[str] = set()
    for record in records:
        payload = json.loads(record["messages"][-1]["content"])
        tool_id = payload.get("selectedToolID")
        tool = routed_tools.get(tool_id)
        if tool is None:
            continue
        if payload.get("actionStep") != {
            "type": "tool_call",
            "toolID": tool_id,
            "mustPersistBeforeFinal": True,
        }:
            continue
        if payload.get("nextModel") != (
            "approval" if tool.requiresApproval else "executor"
        ):
            continue
        if any(
            field in payload
            for field in ("status", "missingArguments", "clarification")
        ):
            continue
        covered.add(tool_id)
    return covered


def test_cortex_optimized_train_sft_has_natural_all_missing_clarifications(
    compiled_fine_tuning: tuple,
) -> None:
    manifest, _, fine_tuning = compiled_fine_tuning
    routed_tools = _cortex_routed_tools(manifest)
    required_by_tool = {
        tool_id: _required_argument_names(tool)
        for tool_id, tool in routed_tools.items()
        if _required_argument_names(tool)
    }
    covered: set[str] = set()

    for record in _optimized_cortex_training_lane(fine_tuning, "train_sft"):
        payload = json.loads(record["messages"][-1]["content"])
        tool_id = payload.get("selectedToolID")
        required_arguments = required_by_tool.get(tool_id)
        if required_arguments is None:
            continue
        prompt = record["messages"][-2]["content"]
        if (
            payload.get("status") == "needs_clarification"
            and payload.get("missingArguments") == required_arguments
            and _is_natural_cortex_request(prompt)
        ):
            covered.add(tool_id)

    assert covered == set(required_by_tool), (
        "optimized Cortex train_sft lacks a natural all-missing clarification "
        f"for routed tools: {sorted(set(required_by_tool) - covered)}"
    )


def test_cortex_optimized_train_sft_has_natural_every_missing_subset(
    compiled_fine_tuning: tuple,
) -> None:
    manifest, _, fine_tuning = compiled_fine_tuning
    routed_tools = _cortex_routed_tools(manifest)
    observed: dict[str, set[tuple[str, ...]]] = {}

    for record in _optimized_cortex_training_lane(fine_tuning, "train_sft"):
        metadata = record["metadata"]
        if metadata.get("taskType") != "cortex_route_curriculum_clarification":
            continue
        payload = json.loads(record["messages"][-1]["content"])
        tool_id = payload["selectedToolID"]
        prompt = record["messages"][-2]["content"]
        assert _is_natural_cortex_request(prompt)
        assert " set to " not in prompt.casefold()
        observed.setdefault(tool_id, set()).add(tuple(payload["missingArguments"]))

    expected_tools = {
        tool_id: _required_argument_names(tool)
        for tool_id, tool in routed_tools.items()
        if _required_argument_names(tool)
    }
    assert set(observed) == set(expected_tools)
    for tool_id, required_arguments in expected_tools.items():
        expected_subsets = {
            tuple(subset)
            for subset_size in range(1, len(required_arguments) + 1)
            for subset in combinations(required_arguments, subset_size)
        }
        assert observed[tool_id] == expected_subsets, tool_id


def test_cortex_optimized_train_sft_has_actionable_coverage_for_every_routed_tool(
    compiled_fine_tuning: tuple,
) -> None:
    manifest, _, fine_tuning = compiled_fine_tuning
    routed_tools = _cortex_routed_tools(manifest)
    covered = _canonical_cortex_action_tool_ids(
        _optimized_cortex_training_lane(fine_tuning, "train_sft"),
        routed_tools,
    )

    assert covered == set(routed_tools), (
        "optimized Cortex train_sft lacks a canonical actionable route for: "
        f"{sorted(set(routed_tools) - covered)}"
    )


def test_cortex_action_curriculum_covers_structured_natural_and_boundaries(
    compiled_fine_tuning: tuple,
) -> None:
    manifest, _, fine_tuning = compiled_fine_tuning
    routed_tools = _cortex_routed_tools(manifest)
    frozen_eval_prompts = {
        record["messages"][-1]["content"]
        for record in fine_tuning["cortex"].eval
    }
    action_records = [
        record
        for record in _optimized_cortex_training_lane(fine_tuning, "train_sft")
        if record["metadata"].get("taskType")
        == "cortex_route_curriculum_action"
    ]
    records_by_surface: dict[str, dict[str, dict]] = {}
    for record in action_records:
        payload = json.loads(record["messages"][-1]["content"])
        tool_id = payload["selectedToolID"]
        surface = record["metadata"].get("surfaceForm")
        assert isinstance(surface, str)
        assert record["messages"][-2]["content"] not in frozen_eval_prompts
        records_by_surface.setdefault(surface, {})[tool_id] = record

    assert set(records_by_surface["structured_json_complete"]) == set(routed_tools)
    for tool_id, record in records_by_surface["structured_json_complete"].items():
        prompt = record["messages"][-2]["content"]
        assert f"`{tool_id}`" in prompt
        if not _required_argument_names(routed_tools[tool_id]):
            assert "{}" in prompt

    assert set(records_by_surface["natural_implicit"]) == set(routed_tools)
    for tool_id in routed_tools:
        record = records_by_surface["natural_implicit"][tool_id]
        prompt = record["messages"][-2]["content"]
        assert "set to" not in prompt.casefold()
        assert "supplied required-argument" not in prompt.casefold()
        assert record["metadata"]["suppliedArguments"] == _required_argument_names(
            routed_tools[tool_id]
        )

    zero_required_tool_ids = {
        tool_id
        for tool_id, tool in routed_tools.items()
        if not _required_argument_names(tool)
    }
    assert zero_required_tool_ids <= set(
        _CORTEX_NATURAL_IMPLICIT_COMPLETE_PROMPTS
    )

    approval_tool_ids = {
        tool_id for tool_id, tool in routed_tools.items() if tool.requiresApproval
    }
    permission_tool_ids = {
        tool_id for tool_id, tool in routed_tools.items() if tool.permissionKey
    }
    assert set(records_by_surface["approval_framed_complete"]) == approval_tool_ids
    assert set(records_by_surface["permission_framed_complete"]) == permission_tool_ids


@pytest.mark.parametrize(
    ("tool_id", "expected_missing"),
    (
        ("trigger.create", ["title", "prompt", "schedule"]),
        ("outlook.message.reply_all", ["messageId", "body"]),
    ),
)
def test_cortex_operation_label_curriculum_keeps_all_required_arguments_missing(
    compiled_fine_tuning: tuple,
    tool_id: str,
    expected_missing: list[str],
) -> None:
    manifest, _, fine_tuning = compiled_fine_tuning
    tools_by_id = {tool.id: tool for tool in manifest.tools}
    records = [
        record
        for record in _optimized_cortex_training_lane(fine_tuning, "train_sft")
        if record["metadata"].get("surfaceForm")
        == "operation_label_all_missing"
        and json.loads(record["messages"][-1]["content"])["selectedToolID"]
        == tool_id
    ]

    assert len(records) == 1
    record = records[0]
    prompt = record["messages"][-2]["content"]
    payload = json.loads(record["messages"][-1]["content"])
    display_name = (tools_by_id[tool_id].displayName or tool_id).strip()

    assert display_name.casefold() in prompt.casefold()
    assert "app operation" in prompt.casefold()
    assert payload["status"] == "needs_clarification"
    assert payload["missingArguments"] == expected_missing
    assert record["metadata"]["missingArguments"] == expected_missing
    assert record["metadata"]["suppliedArguments"] == []
    assert "actionStep" not in payload
    assert payload["nextModel"] == "mouth"


def test_cortex_unresolved_operation_reference_keeps_trigger_id_missing(
    compiled_fine_tuning: tuple,
) -> None:
    _, _, fine_tuning = compiled_fine_tuning
    records = [
        record
        for record in _optimized_cortex_training_lane(fine_tuning, "train_sft")
        if record["metadata"].get("surfaceForm")
        == "unmarked_operation_reference"
        and json.loads(record["messages"][-1]["content"])["selectedToolID"]
        == "trigger.cancel"
    ]

    assert records
    for record in records:
        prompt = record["messages"][-2]["content"]
        payload = json.loads(record["messages"][-1]["content"])
        assert "operation on that item" in prompt.casefold()
        assert payload["status"] == "needs_clarification"
        assert payload["missingArguments"] == ["id"]
        assert record["metadata"]["missingArguments"] == ["id"]
        assert record["metadata"]["suppliedArguments"] == []
        assert "actionStep" not in payload
        assert payload["nextModel"] == "mouth"


def test_cortex_manifest_action_step_rehearsal_emits_full_routes(
    compiled_fine_tuning: tuple,
) -> None:
    manifest, _, fine_tuning = compiled_fine_tuning
    routed_tools = _cortex_routed_tools(manifest)
    records = [
        record
        for record in _optimized_cortex_training_lane(fine_tuning, "train_sft")
        if record["metadata"].get("surfaceForm")
        == "manifest_action_step_rehearsal"
    ]
    records_by_tool = {
        json.loads(record["messages"][-1]["content"])["selectedToolID"]: record
        for record in records
    }

    assert set(records_by_tool) == set(routed_tools)
    assert len(records) == len(routed_tools)
    for tool_id, tool in routed_tools.items():
        record = records_by_tool[tool_id]
        prompt = record["messages"][-2]["content"]
        payload = json.loads(record["messages"][-1]["content"])
        assert prompt.startswith(
            f"Manifest action-step rehearsal for `{tool_id}`:"
        )
        assert "do not emit an Executor argument fragment" in prompt
        assert payload["actionStep"] == {
            "type": "tool_call",
            "toolID": tool_id,
            "mustPersistBeforeFinal": True,
        }
        assert payload["requiresApproval"] is tool.requiresApproval
        assert payload["nextModel"] == (
            "approval" if tool.requiresApproval else "executor"
        )
        assert not {
            "status",
            "missingArguments",
            "clarification",
        } & set(payload)
        assert record["metadata"]["suppliedArguments"] == (
            _required_argument_names(tool)
        )


def test_cortex_operation_label_dpo_sweep_is_bidirectional(
    compiled_fine_tuning: tuple,
) -> None:
    manifest, _, fine_tuning = compiled_fine_tuning
    routed_tools = _cortex_routed_tools(manifest)
    required_by_tool = {
        tool_id: _required_argument_names(tool)
        for tool_id, tool in routed_tools.items()
        if _required_argument_names(tool)
    }
    dpo_records = _optimized_cortex_training_lane(fine_tuning, "train_dpo")
    label_records = [
        record
        for record in dpo_records
        if record["metadata"].get("preferenceType")
        == "route_operation_label_all_missing_vs_premature_action"
    ]
    complete_records = [
        record
        for record in dpo_records
        if record["metadata"].get("preferenceType")
        == "route_operation_label_complete_vs_spurious_clarification"
    ]

    assert Counter(
        json.loads(record["chosen"]["content"])["selectedToolID"]
        for record in label_records
    ) == Counter({tool_id: 1 for tool_id in required_by_tool})
    assert Counter(
        json.loads(record["chosen"]["content"])["selectedToolID"]
        for record in complete_records
    ) == Counter({tool_id: 1 for tool_id in required_by_tool})

    for record in label_records:
        prompt = record["prompt"][-1]["content"]
        chosen = json.loads(record["chosen"]["content"])
        rejected = json.loads(record["rejected"]["content"])
        tool_id = chosen["selectedToolID"]
        required_arguments = required_by_tool[tool_id]
        assert "app operation" in prompt.casefold()
        if {"id", "messageId"} & set(required_arguments):
            assert "on that item" in prompt.casefold()
        assert chosen["missingArguments"] == required_arguments
        assert "actionStep" not in chosen
        assert rejected["actionStep"]["toolID"] == tool_id
        assert record["metadata"]["boundaryDirection"] == (
            "label_only_to_clarification"
        )
        assert record["metadata"]["suppliedArguments"] == []

    for record in complete_records:
        prompt = record["prompt"][-1]["content"]
        chosen = json.loads(record["chosen"]["content"])
        rejected = json.loads(record["rejected"]["content"])
        tool_id = chosen["selectedToolID"]
        required_arguments = required_by_tool[tool_id]
        assert "app operation" in prompt.casefold()
        assert chosen["actionStep"] == {
            "mustPersistBeforeFinal": True,
            "toolID": tool_id,
            "type": "tool_call",
        }
        assert rejected["missingArguments"] == required_arguments
        assert record["metadata"]["boundaryDirection"] == (
            "concrete_values_to_action"
        )
        assert record["metadata"]["suppliedArguments"] == required_arguments


def test_cortex_manifest_action_step_dpo_rejects_protocol_fragments(
    compiled_fine_tuning: tuple,
) -> None:
    _, _, fine_tuning = compiled_fine_tuning
    records = [
        record
        for record in _optimized_cortex_training_lane(fine_tuning, "train_dpo")
        if record["metadata"].get("preferenceType")
        == "route_manifest_action_step_full_route_vs_protocol_fragment"
    ]
    records_by_tool = {
        json.loads(record["chosen"]["content"])["selectedToolID"]: record
        for record in records
    }

    assert set(records_by_tool) == {"alarm.countdown", "rag.index_files"}
    for tool_id, record in records_by_tool.items():
        chosen = json.loads(record["chosen"]["content"])
        rejected = json.loads(record["rejected"]["content"])
        assert chosen["actionStep"] == {
            "mustPersistBeforeFinal": True,
            "toolID": tool_id,
            "type": "tool_call",
        }
        assert set(rejected) == {
            "actionStep",
            "nextModel",
            "requiresApproval",
        }
        assert not {
            "selectedToolID",
            "intent",
            "reasoningSummary",
        } & set(rejected)


def test_cortex_reference_curriculum_covers_every_id_bound_tool(
    compiled_fine_tuning: tuple,
) -> None:
    manifest, _, fine_tuning = compiled_fine_tuning
    routed_tools = _cortex_routed_tools(manifest)
    reference_specs = {
        tool_id: {
            "reference": next(
                argument.name
                for argument in tool.arguments
                if argument.required and argument.name in {"id", "messageId"}
            ),
            "required": _required_argument_names(tool),
        }
        for tool_id, tool in routed_tools.items()
        if any(
            argument.required and argument.name in {"id", "messageId"}
            for argument in tool.arguments
        )
    }
    sft_records = [
        record
        for record in _optimized_cortex_training_lane(fine_tuning, "train_sft")
        if record["metadata"].get("curriculumMode")
        == "clarification_reference_missing"
    ]
    dpo_records = [
        record
        for record in _optimized_cortex_training_lane(fine_tuning, "train_dpo")
        if record["metadata"].get("preferenceType")
        == "route_unmarked_reference_exact_missing_subset"
    ]

    assert reference_specs
    expected_sft_counts = {
        tool_id: 2
        * 2
        ** len(
            [
                argument
                for argument in spec["required"]
                if argument != spec["reference"]
            ]
        )
        for tool_id, spec in reference_specs.items()
    }
    expected_dpo_counts = {
        tool_id: count // 2 for tool_id, count in expected_sft_counts.items()
    }
    assert Counter(
        json.loads(record["messages"][-1]["content"])["selectedToolID"]
        for record in sft_records
    ) == Counter(expected_sft_counts)
    assert Counter(
        json.loads(record["chosen"]["content"])["selectedToolID"]
        for record in dpo_records
    ) == Counter(expected_dpo_counts)
    assert {
        record["metadata"].get("surfaceForm") for record in sft_records
    } == {"unmarked_selected_reference", "unmarked_discussed_reference"}

    observed_sft_variants: set[tuple[str, tuple[str, ...], str]] = set()
    for record in sft_records:
        prompt = record["messages"][-2]["content"]
        payload = json.loads(record["messages"][-1]["content"])
        tool_id = payload["selectedToolID"]
        spec = reference_specs[tool_id]
        supplied_arguments = record["metadata"]["suppliedArguments"]
        expected_missing = [
            argument
            for argument in spec["required"]
            if argument == spec["reference"]
            or argument not in supplied_arguments
        ]
        assert payload["missingArguments"] == expected_missing
        assert record["metadata"]["missingArguments"] == expected_missing
        assert "actionStep" not in payload
        assert "latest-resolved" not in prompt.casefold()
        assert any(cue in prompt.casefold() for cue in ("selected", "discussed"))
        surface_form = record["metadata"]["surfaceForm"]
        observed_sft_variants.add(
            (tool_id, tuple(supplied_arguments), surface_form)
        )

    expected_sft_variants = {
        (tool_id, tuple(supplied_subset), surface_form)
        for tool_id, spec in reference_specs.items()
        for supplied_count in range(
            len(
                [
                    argument
                    for argument in spec["required"]
                    if argument != spec["reference"]
                ]
            )
            + 1
        )
        for supplied_subset in combinations(
            [
                argument
                for argument in spec["required"]
                if argument != spec["reference"]
            ],
            supplied_count,
        )
        for surface_form in (
            "unmarked_selected_reference",
            "unmarked_discussed_reference",
        )
    }
    assert observed_sft_variants == expected_sft_variants

    observed_dpo_variants: set[tuple[str, tuple[str, ...]]] = set()
    for record in dpo_records:
        chosen = json.loads(record["chosen"]["content"])
        rejected = json.loads(record["rejected"]["content"])
        tool_id = chosen["selectedToolID"]
        spec = reference_specs[tool_id]
        supplied_arguments = record["metadata"]["suppliedArguments"]
        expected_missing = [
            argument
            for argument in spec["required"]
            if argument == spec["reference"]
            or argument not in supplied_arguments
        ]
        assert chosen["missingArguments"] == expected_missing
        assert record["metadata"]["missingArguments"] == expected_missing
        assert record["metadata"]["referenceArgument"] == spec["reference"]
        if expected_missing == [spec["reference"]]:
            assert rejected["actionStep"]["toolID"] == tool_id
            assert record["metadata"]["rejectedRouteState"] == "premature_action"
            assert record["metadata"]["rejectedMissingArguments"] == []
        else:
            assert "actionStep" not in rejected
            assert rejected["status"] == "needs_clarification"
            assert rejected["missingArguments"] == [spec["reference"]]
            assert (
                record["metadata"]["rejectedRouteState"]
                == "underreported_clarification"
            )
            assert record["metadata"]["rejectedMissingArguments"] == [
                spec["reference"]
            ]
        observed_dpo_variants.add((tool_id, tuple(supplied_arguments)))

    assert observed_dpo_variants == {
        (tool_id, tuple(supplied_subset))
        for tool_id, spec in reference_specs.items()
        for supplied_count in range(
            len(
                [
                    argument
                    for argument in spec["required"]
                    if argument != spec["reference"]
                ]
            )
            + 1
        )
        for supplied_subset in combinations(
            [
                argument
                for argument in spec["required"]
                if argument != spec["reference"]
            ],
            supplied_count,
        )
    }


def test_cortex_optimized_all_missing_clarifications_use_manifest_argument_order(
    compiled_fine_tuning: tuple,
) -> None:
    manifest, _, fine_tuning = compiled_fine_tuning
    routed_tools = _cortex_routed_tools(manifest)
    required_by_tool = {
        tool_id: _required_argument_names(tool)
        for tool_id, tool in routed_tools.items()
        if _required_argument_names(tool)
    }
    seen: set[str] = set()

    for record in _optimized_cortex_training_lane(fine_tuning, "train_sft"):
        payload = json.loads(record["messages"][-1]["content"])
        tool_id = payload.get("selectedToolID")
        required_arguments = required_by_tool.get(tool_id)
        missing_arguments = payload.get("missingArguments")
        if (
            required_arguments is None
            or payload.get("status") != "needs_clarification"
            or not isinstance(missing_arguments, list)
            or Counter(missing_arguments) != Counter(required_arguments)
        ):
            continue
        seen.add(tool_id)
        assert missing_arguments == required_arguments, (
            f"{tool_id} all-missing clarification must preserve manifest order: "
            f"expected {required_arguments}, got {missing_arguments}"
        )

    assert seen == set(required_by_tool), (
        "optimized Cortex train_sft lacks an all-missing clarification for: "
        f"{sorted(set(required_by_tool) - seen)}"
    )


def test_cortex_optimized_train_dpo_omits_redundant_action_only_families(
    compiled_fine_tuning: tuple,
) -> None:
    _, _, fine_tuning = compiled_fine_tuning
    preference_types = {
        record["metadata"].get("preferenceType")
        for record in _optimized_cortex_training_lane(fine_tuning, "train_dpo")
    }

    assert {
        "route_action_vs_bare_selection",
        "route_action_vs_no_tool_hybrid",
    }.isdisjoint(preference_types)


def test_cortex_optimized_route_state_training_is_balanced(
    compiled_fine_tuning: tuple,
) -> None:
    _, _, fine_tuning = compiled_fine_tuning

    def route_state_stats(
        records: list[dict],
        *,
        dpo: bool,
    ) -> tuple[Counter[str], Counter[str]]:
        row_counts: Counter[str] = Counter()
        character_counts: Counter[str] = Counter()
        for record in records:
            content = (
                record["chosen"]["content"]
                if dpo
                else record["messages"][-1]["content"]
            )
            payload = json.loads(content)
            if isinstance(payload.get("actionStep"), dict):
                state = "action"
            elif payload.get("status") == "needs_clarification":
                state = "clarification"
            else:
                continue
            row_counts[state] += 1
            character_counts[state] += len(content)
        return row_counts, character_counts

    sft_stats = route_state_stats(
        _optimized_cortex_training_lane(fine_tuning, "train_sft"),
        dpo=False,
    )
    dpo_stats = route_state_stats(
        _optimized_cortex_training_lane(fine_tuning, "train_dpo"),
        dpo=True,
    )
    for row_counts, character_counts in (sft_stats, dpo_stats):
        assert row_counts["action"] > 0
        assert row_counts["clarification"] > 0
        assert max(row_counts.values()) / min(row_counts.values()) <= 1.15
        assert (
            max(character_counts.values()) / min(character_counts.values())
            <= 1.30
        )


def test_cortex_optimized_train_dpo_rejects_wrong_all_missing_argument_lists(
    compiled_fine_tuning: tuple,
) -> None:
    manifest, _, fine_tuning = compiled_fine_tuning
    routed_tools = _cortex_routed_tools(manifest)
    required_by_tool = {
        tool_id: _required_argument_names(tool)
        for tool_id, tool in routed_tools.items()
        if _required_argument_names(tool)
    }
    rejected_subsets_by_tool: dict[str, set[tuple[str, ...]]] = {}

    for record in _optimized_cortex_training_lane(fine_tuning, "train_dpo"):
        if (
            record["metadata"].get("preferenceType")
            != "route_all_missing_vs_wrong_subset"
        ):
            continue
        chosen = json.loads(record["chosen"]["content"])
        rejected = _load_json_object(record["rejected"]["content"])
        if rejected is None:
            continue
        tool_id = chosen.get("selectedToolID")
        required_arguments = required_by_tool.get(tool_id)
        rejected_missing = rejected.get("missingArguments")
        if required_arguments is None:
            continue
        if (
            chosen.get("status") == "needs_clarification"
            and chosen.get("missingArguments") == required_arguments
            and rejected.get("selectedToolID") == tool_id
            and rejected.get("status") == "needs_clarification"
            and isinstance(rejected_missing, list)
            and rejected_missing != required_arguments
        ):
            rejected_subsets_by_tool.setdefault(tool_id, set()).add(
                tuple(rejected_missing)
            )

    assert set(rejected_subsets_by_tool) == set(required_by_tool), (
        "optimized Cortex train_dpo lacks a wrong all-missing argument-list hard "
        "negative for: "
        f"{sorted(set(required_by_tool) - set(rejected_subsets_by_tool))}"
    )
    for tool_id, required_arguments in required_by_tool.items():
        observed = rejected_subsets_by_tool[tool_id]
        if len(required_arguments) == 1:
            assert observed == {("inventedArgument",)}
            continue
        expected = {
            tuple(subset)
            for subset_size in range(1, len(required_arguments))
            for subset in combinations(required_arguments, subset_size)
        }
        assert observed == expected, tool_id


def test_cortex_zero_required_tools_reject_spurious_clarification(
    compiled_fine_tuning: tuple,
) -> None:
    manifest, _, fine_tuning = compiled_fine_tuning
    routed_tools = _cortex_routed_tools(manifest)
    zero_required_tool_ids = {
        tool_id
        for tool_id, tool in routed_tools.items()
        if not _required_argument_names(tool)
    }
    records = [
        record
        for record in _optimized_cortex_training_lane(fine_tuning, "train_dpo")
        if record["metadata"].get("preferenceType")
        == "route_zero_required_vs_spurious_clarification"
    ]

    assert Counter(
        json.loads(record["chosen"]["content"])["selectedToolID"]
        for record in records
    ) == Counter({tool_id: 1 for tool_id in zero_required_tool_ids})
    for record in records:
        chosen = json.loads(record["chosen"]["content"])
        rejected = json.loads(record["rejected"]["content"])
        tool_id = chosen["selectedToolID"]
        assert chosen["actionStep"]["toolID"] == tool_id
        assert "status" not in chosen
        assert rejected["selectedToolID"] == tool_id
        assert rejected["status"] == "needs_clarification"
        assert rejected["missingArguments"]
        assert "actionStep" not in rejected


def test_cortex_complete_route_dpo_rejects_foreign_row_schemas(
    compiled_fine_tuning: tuple,
) -> None:
    manifest, _, fine_tuning = compiled_fine_tuning
    routed_tools = _cortex_routed_tools(manifest)
    expected_types = {
        "route_natural_complete_vs_wrong_schema",
        "route_structured_complete_vs_wrong_schema",
    }
    observed: Counter[tuple[str, str]] = Counter()

    for record in _optimized_cortex_training_lane(fine_tuning, "train_dpo"):
        preference_type = record["metadata"].get("preferenceType")
        if preference_type not in expected_types:
            continue
        chosen = json.loads(record["chosen"]["content"])
        rejected = json.loads(record["rejected"]["content"])
        tool_id = chosen["selectedToolID"]
        own_required = set(_required_argument_names(routed_tools[tool_id]))
        rejected_missing = rejected["missingArguments"]
        observed[(tool_id, preference_type)] += 1
        assert chosen["actionStep"]["toolID"] == tool_id
        assert rejected["selectedToolID"] == tool_id
        assert rejected["status"] == "needs_clarification"
        assert rejected_missing
        assert own_required.isdisjoint(rejected_missing)
        assert record["metadata"]["rejectedMissingArguments"] == rejected_missing

    assert observed == Counter(
        {
            (tool_id, preference_type): 1
            for tool_id in routed_tools
            for preference_type in expected_types
        }
    )


def test_cortex_structured_incomplete_routes_reject_foreign_row_schemas(
    compiled_fine_tuning: tuple,
) -> None:
    manifest, _, fine_tuning = compiled_fine_tuning
    routed_tools = _cortex_routed_tools(manifest)
    required_by_tool = {
        tool_id: _required_argument_names(tool)
        for tool_id, tool in routed_tools.items()
        if _required_argument_names(tool)
    }
    records = [
        record
        for record in _optimized_cortex_training_lane(fine_tuning, "train_dpo")
        if record["metadata"].get("preferenceType")
        == "route_structured_incomplete_vs_wrong_schema"
    ]

    assert Counter(
        json.loads(record["chosen"]["content"])["selectedToolID"]
        for record in records
    ) == Counter({tool_id: 1 for tool_id in required_by_tool})
    for record in records:
        chosen = json.loads(record["chosen"]["content"])
        rejected = json.loads(record["rejected"]["content"])
        tool_id = chosen["selectedToolID"]
        assert chosen["missingArguments"] == required_by_tool[tool_id]
        assert "actionStep" not in chosen
        assert rejected["selectedToolID"] == tool_id
        assert rejected["missingArguments"] == record["metadata"][
            "rejectedMissingArguments"
        ]
        assert set(required_by_tool[tool_id]).isdisjoint(
            rejected["missingArguments"]
        )


def test_cortex_semantic_sibling_contrasts_bind_tool_and_action_step(
    compiled_fine_tuning: tuple,
) -> None:
    _, _, fine_tuning = compiled_fine_tuning
    records = [
        record
        for record in _optimized_cortex_training_lane(fine_tuning, "train_dpo")
        if record["metadata"].get("preferenceType")
        == "route_semantic_sibling_contrast"
    ]
    expected = {
        ("alarm.stop", "alarm.pause"),
        ("alarm.pause", "alarm.stop"),
        ("outlook.messages.list", "outlook.mail.send"),
        ("outlook.mail.send", "outlook.messages.list"),
        ("outlook.mail.send", "outlook.message.forward"),
        ("outlook.message.forward", "outlook.mail.send"),
        ("outlook.mail.send", "mail.draft"),
        ("mail.draft", "outlook.mail.send"),
        ("alarm.authorization_status", "alarm.request_authorization"),
        ("alarm.request_authorization", "alarm.authorization_status"),
        ("calendar.list", "calendar.create"),
        ("calendar.create", "calendar.list"),
        ("calendar.create", "trigger.create"),
        ("trigger.create", "calendar.create"),
        ("reminders.list", "reminders.create"),
        ("reminders.create", "reminders.list"),
        ("outlook.folders.list", "outlook.messages.list"),
        ("outlook.messages.list", "outlook.folders.list"),
        ("outlook.status", "outlook.messages.list"),
        ("camera.capture", "photos.search"),
        ("photos.search", "camera.capture"),
        ("location.current", "weather"),
        ("weather", "location.current"),
        ("trigger.list", "trigger.create"),
        ("trigger.create", "trigger.list"),
        ("alarm.list", "alarm.schedule"),
        ("alarm.schedule", "alarm.list"),
        ("rag.index_files", "rag.search"),
        ("rag.search", "rag.index_files"),
        ("health.summary", "motion.activity"),
        ("motion.activity", "health.summary"),
        ("memory.recall", "memory.save"),
        ("memory.save", "memory.recall"),
    }
    observed: set[tuple[str, str]] = set()
    for record in records:
        chosen = json.loads(record["chosen"]["content"])
        rejected = json.loads(record["rejected"]["content"])
        chosen_tool_id = chosen["selectedToolID"]
        rejected_tool_id = rejected["selectedToolID"]
        observed.add((chosen_tool_id, rejected_tool_id))
        assert chosen["actionStep"]["toolID"] == chosen_tool_id
        assert rejected["actionStep"]["toolID"] == rejected_tool_id
        assert record["metadata"]["contrastToolID"] == rejected_tool_id
    assert observed == expected


def test_cortex_failure_repair_curriculum_is_bidirectional_and_eval_disjoint(
    compiled_fine_tuning: tuple,
) -> None:
    manifest, _, fine_tuning = compiled_fine_tuning
    cortex = fine_tuning["cortex"]
    frozen_eval_prompts = {
        record["messages"][-1]["content"] for record in cortex.eval
    }
    frozen_eval_prompt_hashes = {
        hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        for prompt in frozen_eval_prompts
    }
    sft_records = [
        record
        for record in _optimized_cortex_training_lane(fine_tuning, "train_sft")
        if record["metadata"].get("curriculumMode")
        in {
            "failure_repair_actionable",
            "failure_repair_clarification",
        }
    ]
    dpo_records = [
        record
        for record in _optimized_cortex_training_lane(fine_tuning, "train_dpo")
        if record["metadata"].get("preferenceType")
        == "route_failure_repair_bidirectional"
    ]

    legacy_expected_sft_cases = {
        "schema_alarm_authorization_status_action",
        "zero_required_alarm_list_action",
        "zero_required_alarm_request_authorization_action",
        "schema_alarm_pause_action",
        "deictic_alarm_resume_explicit_id_action",
        "deictic_trigger_cancel_explicit_id_action",
        "structured_alarm_cancel_explicit_id_action",
        "partial_countdown_complete_action",
        "schema_camera_capture_action",
        "implicit_memory_recall_action",
        "implicit_memory_save_action",
        "action_persistence_photos_search",
        "action_persistence_rag_search",
        "implicit_reminder_title_action",
        "schema_outlook_status_action",
        "route_outlook_send_action",
        "route_outlook_forward_action",
        "schema_outlook_delete_action",
        "route_outlook_list_action",
        "boundary_reminder_complete_action",
        "route_outlook_search_action",
        "route_new_text_action",
        "deictic_alarm_resume_missing_id",
        "deictic_trigger_cancel_missing_id",
        "schema_alarm_pause_missing_id",
        "schema_alarm_countdown_missing_details",
        "partial_countdown_missing_title",
        "implicit_memory_recall_missing_query",
        "implicit_memory_save_missing_content_and_kind",
        "boundary_reminder_missing_title",
        "schema_outlook_mark_read_missing_id",
        "schema_outlook_reply_missing_body",
        "route_outlook_reply_missing_all",
        "route_outlook_send_missing_body",
        "route_outlook_forward_missing_id",
        "route_outlook_search_missing_query",
        "route_new_text_missing_all",
    }
    expected_minimal_pair_counts = {
        ("outlook_read_reference", "latest_outlook_email"): 3,
        ("outlook_read_reference", "unresolved_reference"): 3,
        ("outlook_attachments_reference", "explicit_id"): 3,
        ("outlook_attachments_reference", "unresolved_reference"): 3,
        ("outlook_mark_unread_reference", "explicit_id"): 3,
        ("outlook_mark_unread_reference", "unresolved_reference"): 3,
        ("outlook_reply_all_reference", "explicit_id"): 3,
        ("outlook_reply_all_reference", "unresolved_reference"): 3,
        ("alarm_resume_reference", "explicit_id"): 2,
        ("alarm_resume_reference", "unresolved_reference"): 3,
        ("trigger_cancel_reference", "explicit_id"): 2,
        ("trigger_cancel_reference", "unresolved_reference"): 3,
        ("alarm_countdown_duration_only", "complete"): 2,
        ("alarm_countdown_duration_only", "missing_title"): 2,
        ("alarm_schedule_missing_values", "unmarked_incomplete"): 2,
        ("calendar_create_missing_values", "unmarked_incomplete"): 2,
        ("calendar_generic_object_with_time", "missing_title"): 1,
        ("outlook_send_recipient_only", "unmarked_incomplete"): 2,
        ("outlook_send_named_recipient_only", "unmarked_incomplete"): 3,
        ("outlook_send_named_recipient_only", "complete"): 3,
        ("outlook_forward_reference", "unresolved_reference"): 2,
        ("photo_reindex_missing_months", "unmarked_incomplete"): 2,
        ("zero_required_alarm_list", "actionable"): 3,
        ("zero_required_calendar_list", "actionable"): 1,
        ("zero_required_outlook_folders_list", "actionable"): 1,
        ("zero_required_outlook_messages_list", "actionable"): 1,
        ("zero_required_alarm_authorization_request", "actionable"): 5,
        ("zero_required_reminders_list", "actionable"): 5,
        ("zero_required_trigger_list", "actionable"): 1,
        ("calendar_human_event_vague_time", "missing_numeric_start"): 4,
        ("zero_required_weather", "actionable"): 3,
        ("outlook_send_vs_system_draft", "outlook_send"): 2,
        ("outlook_send_vs_system_draft", "system_mail_draft"): 3,
        ("provider_neutral_send_vs_draft", "send_now"): 2,
        ("provider_neutral_send_vs_draft", "draft_only"): 2,
        ("outlook_forward_vs_system_draft", "outlook_forward"): 2,
        ("memory_save_preference", "actionable"): 10,
        ("memory_save_app_operation", "unmarked_incomplete"): 2,
        ("memory_save_recall_same_topic", "save"): 2,
        ("memory_save_recall_same_topic", "recall"): 2,
        ("memory_recall_query", "actionable"): 4,
        ("reminder_creation_title", "actionable"): 2,
        ("maps_search_query", "actionable"): 3,
        ("alarm_countdown_notify_without_title", "missing_title"): 4,
        ("alarm_countdown_notify_without_title", "complete"): 2,
        (
            "outlook_reply_reference_without_body",
            "unresolved_reference",
        ): 4,
        (
            "outlook_reply_all_reference_without_body",
            "unresolved_reference",
        ): 6,
        (
            "outlook_reply_all_selected_message_body_boundary",
            "missing_message_id",
        ): 3,
        (
            "outlook_reply_all_operation_without_values",
            "all_missing",
        ): 3,
        (
            "outlook_send_named_recipient_unresolved_content",
            "unmarked_incomplete",
        ): 4,
        ("outlook_move_required_subsets", "all_missing"): 2,
        ("outlook_move_required_subsets", "missing_message_id"): 2,
        ("outlook_move_required_subsets", "missing_destination"): 2,
        ("outlook_move_required_subsets", "complete"): 2,
    }
    minimal_pair_records = [
        record
        for record in sft_records
        if isinstance(record["metadata"].get("minimalPairFamily"), str)
    ]
    observed_minimal_pair_counts = Counter(
        (
            record["metadata"]["minimalPairFamily"],
            record["metadata"]["minimalPairState"],
        )
        for record in minimal_pair_records
    )
    expected_sft_cases = legacy_expected_sft_cases | {
        record["metadata"]["repairCase"] for record in minimal_pair_records
    }
    expected_dpo_cases = (
        legacy_expected_sft_cases
        - {
            "route_outlook_search_missing_query",
            "route_new_text_missing_all",
        }
        | {
            "boundary_reminder_time_only",
            "route_outlook_forward_exact_id",
            "route_outlook_named_recipient_missing_subject_body_alias",
            "route_outlook_named_recipient_missing_subject_body_intent",
            "route_outlook_reply_all_reference_missing_all",
            "route_outlook_move_missing_all",
            "route_outlook_move_missing_message_id",
            "route_outlook_move_missing_destination",
            "memory_app_operation_missing_contract",
            "memory_same_topic_save_not_recall",
            "memory_same_topic_recall_not_save",
            "countdown_notify_missing_title",
            "outlook_read_latest_not_files_1",
            "outlook_read_latest_exact_tool_2",
            "outlook_read_selected_not_files",
            "files_read_explicit_not_outlook",
            "files_read_unresolved_not_outlook",
            "outlook_reply_all_operation_missing_all_1",
            "outlook_reply_all_operation_missing_all_2",
            "outlook_reply_all_operation_missing_all_3",
            "memory_preference_implicit_action_1",
            "memory_preference_implicit_action_2",
            "memory_preference_implicit_action_3",
            "memory_preference_implicit_action_4",
            "outlook_reply_unresolved_without_body_1",
            "outlook_reply_unresolved_without_body_2",
            "outlook_reply_unresolved_without_body_3",
            "outlook_reply_all_unresolved_without_body_1",
            "outlook_reply_all_unresolved_without_body_2",
            "outlook_reply_all_unresolved_without_body_3",
            "outlook_send_named_recipient_unresolved_content_1",
            "outlook_send_named_recipient_unresolved_content_2",
            "outlook_send_named_recipient_unresolved_content_3",
            "outlook_send_named_recipient_unresolved_content_4",
            "zero_required_alarm_request_action_1",
            "zero_required_alarm_request_action_2",
            "zero_required_alarm_request_action_3",
            "zero_required_alarm_request_action_4",
            "countdown_duration_supplied_missing_title_1",
            "countdown_duration_supplied_missing_title_2",
            "memory_preference_implicit_action_5",
            "memory_preference_implicit_action_6",
            "memory_recall_topic_action_1",
            "memory_recall_topic_action_2",
            "memory_save_preference_action_7",
            "memory_save_preference_action_8",
            "zero_required_list_alarm_action",
            "zero_required_list_calendar_action",
            "zero_required_list_outlook_folders_action",
            "zero_required_list_outlook_messages_action",
            "zero_required_list_reminders_action_1",
            "zero_required_list_reminders_action_2",
            "zero_required_list_trigger_action",
            "calendar_event_vague_time_missing_start_1",
            "calendar_event_vague_time_missing_start_2",
            "calendar_event_vague_time_missing_start_3",
            "calendar_event_vague_time_missing_start_4",
            "calendar_generic_object_missing_all",
            "calendar_generic_object_with_time_missing_title",
            "outlook_reply_all_selected_without_body_1",
            "outlook_reply_all_selected_without_body_2",
            "outlook_reply_all_selected_with_body",
            "selection_only_health_forbidden_decoys",
            "selection_only_weather_forbidden_decoys",
            "action_persistence_photos_search",
            "action_persistence_rag_search",
            "action_persistence_memory_recall",
        }
    )

    assert {record["metadata"]["repairCase"] for record in sft_records} == (
        expected_sft_cases
    )
    assert {record["metadata"]["repairCase"] for record in dpo_records} == (
        expected_dpo_cases
    )
    assert observed_minimal_pair_counts == expected_minimal_pair_counts
    assert len(sft_records) == len(expected_sft_cases)
    assert len(minimal_pair_records) == sum(expected_minimal_pair_counts.values())
    assert len(dpo_records) == len(expected_dpo_cases)
    expected_targeted_family_counts = {
        "calendar_event_title_without_numeric_delay": (4, 4),
        "calendar_operation_object_not_title": (1, 2),
        "implicit_duration_countdown_missing_title": (6, 2),
        "implicit_preference_memory_save": (10, 8),
        "implicit_topic_memory_recall": (4, 2),
        "outlook_read_files_read_route_lock": (0, 4),
        "outlook_read_reference_resolution": (6, 1),
        "outlook_reply_unresolved_reference_and_body": (16, 12),
        "outlook_send_unresolved_subject_and_body": (4, 4),
        "zero_required_action_without_invented_arguments": (5, 4),
        "zero_required_list_action": (12, 7),
        "selection_only_route_state": (0, 2),
        "trigger_cancel_reference_resolution": (5, 2),
        "action_step_persistence_literal_true": (0, 3),
    }
    assert Counter(
        record["metadata"].get("targetedFailureFamily")
        for record in minimal_pair_records
        if record["metadata"].get("targetedFailureFamily") is not None
    ) == Counter(
        {
            family: counts[0]
            for family, counts in expected_targeted_family_counts.items()
            if counts[0]
        }
    )
    assert Counter(
        record["metadata"].get("targetedFailureFamily")
        for record in dpo_records
        if record["metadata"].get("targetedFailureFamily") is not None
    ) == Counter(
        {
            family: counts[1]
            for family, counts in expected_targeted_family_counts.items()
            if counts[1]
        }
    )
    for record in (*minimal_pair_records, *dpo_records):
        if record["metadata"].get("targetedFailureFamily") is None:
            continue
        assert record["metadata"]["requiredSplit"] == "train"
        assert record["metadata"]["surfaceForm"].startswith("natural_")
    for record in sft_records:
        prompt = record["messages"][-2]["content"]
        assert prompt not in frozen_eval_prompts
        assert (
            hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            not in frozen_eval_prompt_hashes
        )
        payload = json.loads(record["messages"][-1]["content"])
        if record["metadata"]["curriculumMode"] == "failure_repair_actionable":
            assert payload["actionStep"]["toolID"] == payload["selectedToolID"]
            assert "status" not in payload
        else:
            assert payload["status"] == "needs_clarification"
            assert payload["missingArguments"]
            assert "actionStep" not in payload
    route_expectations = {
        ("outlook_read_reference", "latest_outlook_email"): (
            "outlook.message.read",
            None,
        ),
        ("outlook_read_reference", "unresolved_reference"): (
            "outlook.message.read",
            ["messageId"],
        ),
        ("outlook_attachments_reference", "explicit_id"): (
            "outlook.attachments.list",
            None,
        ),
        ("outlook_attachments_reference", "unresolved_reference"): (
            "outlook.attachments.list",
            ["messageId"],
        ),
        ("outlook_mark_unread_reference", "unresolved_reference"): (
            "outlook.message.mark_unread",
            ["messageId"],
        ),
        ("outlook_reply_all_reference", "unresolved_reference"): (
            "outlook.message.reply_all",
            ["messageId"],
        ),
        ("alarm_countdown_duration_only", "missing_title"): (
            "alarm.countdown",
            ["title"],
        ),
        ("alarm_schedule_missing_values", "unmarked_incomplete"): (
            "alarm.schedule",
            ["title", "inMinutes"],
        ),
        ("calendar_create_missing_values", "unmarked_incomplete"): (
            "calendar.create",
            ["title", "startsInMinutes"],
        ),
        ("calendar_generic_object_with_time", "missing_title"): (
            "calendar.create",
            ["title"],
        ),
        ("outlook_send_recipient_only", "unmarked_incomplete"): (
            "outlook.mail.send",
            ["subject", "body"],
        ),
        ("outlook_send_named_recipient_only", "unmarked_incomplete"): (
            "outlook.mail.send",
            ["subject", "body"],
        ),
        ("outlook_send_named_recipient_only", "complete"): (
            "outlook.mail.send",
            None,
        ),
        ("outlook_forward_reference", "unresolved_reference"): (
            "outlook.message.forward",
            ["messageId"],
        ),
        ("photo_reindex_missing_months", "unmarked_incomplete"): (
            "rag.index_photos",
            ["months"],
        ),
        ("calendar_human_event_vague_time", "missing_numeric_start"): (
            "calendar.create",
            ["startsInMinutes"],
        ),
        ("memory_recall_query", "actionable"): ("memory.recall", None),
        ("zero_required_alarm_list", "actionable"): ("alarm.list", None),
        ("zero_required_calendar_list", "actionable"): (
            "calendar.list",
            None,
        ),
        ("zero_required_outlook_folders_list", "actionable"): (
            "outlook.folders.list",
            None,
        ),
        ("zero_required_outlook_messages_list", "actionable"): (
            "outlook.messages.list",
            None,
        ),
        ("zero_required_reminders_list", "actionable"): (
            "reminders.list",
            None,
        ),
        ("zero_required_trigger_list", "actionable"): (
            "trigger.list",
            None,
        ),
        ("outlook_send_vs_system_draft", "outlook_send"): (
            "outlook.mail.send",
            None,
        ),
        ("outlook_send_vs_system_draft", "system_mail_draft"): (
            "mail.draft",
            None,
        ),
        ("memory_save_preference", "actionable"): ("memory.save", None),
        ("memory_save_app_operation", "unmarked_incomplete"): (
            "memory.save",
            ["content", "kind"],
        ),
        ("memory_save_recall_same_topic", "save"): ("memory.save", None),
        ("memory_save_recall_same_topic", "recall"): ("memory.recall", None),
        ("alarm_countdown_notify_without_title", "missing_title"): (
            "alarm.countdown",
            ["title"],
        ),
        ("alarm_countdown_notify_without_title", "complete"): (
            "alarm.countdown",
            None,
        ),
        ("outlook_reply_all_reference_without_body", "unresolved_reference"): (
            "outlook.message.reply_all",
            ["messageId", "body"],
        ),
        (
            "outlook_reply_all_selected_message_body_boundary",
            "missing_message_id",
        ): ("outlook.message.reply_all", ["messageId"]),
        (
            "outlook_reply_all_operation_without_values",
            "all_missing",
        ): ("outlook.message.reply_all", ["messageId", "body"]),
        ("outlook_reply_reference_without_body", "unresolved_reference"): (
            "outlook.message.reply",
            ["messageId", "body"],
        ),
        (
            "outlook_send_named_recipient_unresolved_content",
            "unmarked_incomplete",
        ): ("outlook.mail.send", ["subject", "body"]),
        ("outlook_move_required_subsets", "all_missing"): (
            "outlook.message.move",
            ["messageId", "destination"],
        ),
        ("outlook_move_required_subsets", "missing_message_id"): (
            "outlook.message.move",
            ["messageId"],
        ),
        ("outlook_move_required_subsets", "missing_destination"): (
            "outlook.message.move",
            ["destination"],
        ),
        ("outlook_move_required_subsets", "complete"): (
            "outlook.message.move",
            None,
        ),
        ("maps_search_query", "actionable"): ("maps.search", None),
        ("zero_required_weather", "actionable"): ("weather", None),
    }
    for record in minimal_pair_records:
        key = (
            record["metadata"]["minimalPairFamily"],
            record["metadata"]["minimalPairState"],
        )
        if key[1] in {"unresolved_reference", "unmarked_incomplete"}:
            prompt = record["messages"][-2]["content"].casefold()
            assert not any(
                cue in prompt
                for cue in (
                    "identifier is not available",
                    "key is unresolved",
                    "without assuming its key",
                    "once the item is identified",
                    "after its key is supplied",
                    "not been provided",
                )
            )
        if key not in route_expectations:
            continue
        payload = json.loads(record["messages"][-1]["content"])
        expected_tool_id, expected_missing = route_expectations[key]
        assert payload["selectedToolID"] == expected_tool_id
        if expected_missing is None:
            assert payload["actionStep"]["toolID"] == expected_tool_id
        else:
            assert payload["missingArguments"] == expected_missing
            assert "actionStep" not in payload
    for record in dpo_records:
        prompt = record["prompt"][-1]["content"]
        assert prompt not in frozen_eval_prompts
        assert (
            hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            not in frozen_eval_prompt_hashes
        )
        chosen = json.loads(record["chosen"]["content"])
        rejected = json.loads(record["rejected"]["content"])
        assert chosen != rejected

    by_case = {
        record["metadata"]["repairCase"]: record for record in dpo_records
    }
    observed_wrong_missing = {
        case: json.loads(by_case[case]["rejected"]["content"])[
            "missingArguments"
        ]
        for case in (
            "schema_alarm_authorization_status_action",
            "zero_required_alarm_list_action",
            "zero_required_alarm_request_authorization_action",
            "schema_alarm_pause_action",
            "schema_camera_capture_action",
            "schema_outlook_status_action",
            "schema_outlook_delete_action",
        )
    }
    assert observed_wrong_missing == {
        "schema_alarm_authorization_status_action": ["id", "title"],
        "zero_required_alarm_list_action": ["id", "title"],
        "zero_required_alarm_request_authorization_action": ["id", "title"],
        "schema_alarm_pause_action": ["title"],
        "schema_camera_capture_action": ["title"],
        "schema_outlook_status_action": ["messageId"],
        "schema_outlook_delete_action": ["body"],
    }
    for case, tool_id in {
        "deictic_alarm_resume_missing_id": "alarm.resume",
        "deictic_trigger_cancel_missing_id": "trigger.cancel",
    }.items():
        chosen = json.loads(by_case[case]["chosen"]["content"])
        assert chosen["selectedToolID"] == tool_id
        assert chosen["missingArguments"] == ["id"]
        assert "actionStep" not in chosen
    for case, tool_id in {
        "deictic_alarm_resume_explicit_id_action": "alarm.resume",
        "deictic_trigger_cancel_explicit_id_action": "trigger.cancel",
        "implicit_memory_recall_action": "memory.recall",
        "implicit_memory_save_action": "memory.save",
        "implicit_reminder_title_action": "reminders.create",
        "route_outlook_send_action": "outlook.mail.send",
        "route_outlook_forward_action": "outlook.message.forward",
    }.items():
        chosen = json.loads(by_case[case]["chosen"]["content"])
        assert chosen["selectedToolID"] == tool_id
        assert chosen["actionStep"]["toolID"] == tool_id
    partial_countdown = json.loads(
        by_case["partial_countdown_missing_title"]["chosen"]["content"]
    )
    assert partial_countdown["selectedToolID"] == "alarm.countdown"
    assert partial_countdown["missingArguments"] == ["title"]
    exact_forward = by_case["route_outlook_forward_exact_id"]
    assert json.loads(exact_forward["chosen"]["content"])[
        "selectedToolID"
    ] == "outlook.message.forward"
    assert json.loads(exact_forward["rejected"]["content"])[
        "selectedToolID"
    ] == "mail.forward"

    manifest_tool_ids = {tool.id for tool in manifest.tools}
    named_recipient_alias = by_case[
        "route_outlook_named_recipient_missing_subject_body_alias"
    ]
    assert json.loads(named_recipient_alias["chosen"]["content"])[
        "missingArguments"
    ] == ["subject", "body"]
    assert json.loads(named_recipient_alias["rejected"]["content"])[
        "selectedToolID"
    ] == "mail.send"
    assert "mail.send" not in manifest_tool_ids

    allowed_intents_by_tool: dict[str, set[str]] = {}
    for entry in manifest.routingMatrix:
        for tool_id in entry.allowedTools:
            allowed_intents_by_tool.setdefault(tool_id, set()).add(entry.intent)
    for intent in manifest.intents:
        for tool_id in intent.allowedToolIDs:
            allowed_intents_by_tool.setdefault(tool_id, set()).add(intent.id)
    for case in (
        "route_outlook_named_recipient_missing_subject_body_intent",
        "memory_app_operation_missing_contract",
    ):
        chosen = json.loads(by_case[case]["chosen"]["content"])
        rejected = json.loads(by_case[case]["rejected"]["content"])
        tool_id = chosen["selectedToolID"]
        assert chosen["intent"] in allowed_intents_by_tool[tool_id]
        assert rejected["selectedToolID"] == tool_id
        assert rejected["intent"] not in allowed_intents_by_tool[tool_id]

    for case, chosen_tool_id, rejected_tool_id in (
        ("memory_same_topic_save_not_recall", "memory.save", "memory.recall"),
        ("memory_same_topic_recall_not_save", "memory.recall", "memory.save"),
    ):
        assert json.loads(by_case[case]["chosen"]["content"])[
            "selectedToolID"
        ] == chosen_tool_id
        assert json.loads(by_case[case]["rejected"]["content"])[
            "selectedToolID"
        ] == rejected_tool_id

    cross_domain_route_expectations = {
        "outlook_read_latest_not_files_1": (
            "outlook.message.read",
            None,
            "files.read",
            ["name"],
        ),
        "outlook_read_latest_exact_tool_2": (
            "outlook.message.read",
            None,
            "outlook.message.latest",
            None,
        ),
        "outlook_read_selected_not_files": (
            "outlook.message.read",
            ["messageId"],
            "files.read",
            ["name"],
        ),
        "files_read_explicit_not_outlook": (
            "files.read",
            None,
            "outlook.message.read",
            ["messageId"],
        ),
        "files_read_unresolved_not_outlook": (
            "files.read",
            ["name"],
            "outlook.message.read",
            ["messageId"],
        ),
    }
    for case, (
        chosen_tool_id,
        chosen_missing,
        rejected_tool_id,
        rejected_missing,
    ) in cross_domain_route_expectations.items():
        chosen = json.loads(by_case[case]["chosen"]["content"])
        rejected = json.loads(by_case[case]["rejected"]["content"])
        assert chosen["selectedToolID"] == chosen_tool_id
        assert rejected["selectedToolID"] == rejected_tool_id
        if chosen_missing is None:
            assert chosen["actionStep"]["toolID"] == chosen_tool_id
            assert "missingArguments" not in chosen
        else:
            assert chosen["missingArguments"] == chosen_missing
            assert "actionStep" not in chosen
        if rejected_missing is None:
            assert rejected["actionStep"]["toolID"] == rejected_tool_id
            assert "missingArguments" not in rejected
        else:
            assert rejected["missingArguments"] == rejected_missing
            assert "actionStep" not in rejected
    assert "outlook.message.latest" not in manifest_tool_ids

    expected_missing_by_case = {
        "route_outlook_reply_all_reference_missing_all": ["messageId", "body"],
        "route_outlook_move_missing_all": ["messageId", "destination"],
        "route_outlook_move_missing_message_id": ["messageId"],
        "route_outlook_move_missing_destination": ["destination"],
        "countdown_notify_missing_title": ["title"],
    }
    for case, expected_missing in expected_missing_by_case.items():
        chosen = json.loads(by_case[case]["chosen"]["content"])
        assert chosen["missingArguments"] == expected_missing
        assert "actionStep" not in chosen

    for case in (
        "memory_preference_implicit_action_1",
        "memory_preference_implicit_action_2",
        "memory_preference_implicit_action_3",
        "memory_preference_implicit_action_4",
    ):
        chosen = json.loads(by_case[case]["chosen"]["content"])
        rejected = json.loads(by_case[case]["rejected"]["content"])
        assert chosen["selectedToolID"] == "memory.save"
        assert chosen["actionStep"]["toolID"] == "memory.save"
        assert "status" not in chosen
        assert rejected != chosen

    for case in (
        "outlook_reply_unresolved_without_body_1",
        "outlook_reply_unresolved_without_body_2",
        "outlook_reply_unresolved_without_body_3",
        "outlook_reply_all_unresolved_without_body_1",
        "outlook_reply_all_unresolved_without_body_2",
        "outlook_reply_all_unresolved_without_body_3",
    ):
        chosen = json.loads(by_case[case]["chosen"]["content"])
        rejected = json.loads(by_case[case]["rejected"]["content"])
        assert chosen["missingArguments"] == ["messageId", "body"]
        assert "actionStep" not in chosen
        assert rejected.get("missingArguments") != ["messageId", "body"]

    for case in (
        "outlook_send_named_recipient_unresolved_content_1",
        "outlook_send_named_recipient_unresolved_content_2",
        "outlook_send_named_recipient_unresolved_content_3",
        "outlook_send_named_recipient_unresolved_content_4",
    ):
        chosen = json.loads(by_case[case]["chosen"]["content"])
        rejected = json.loads(by_case[case]["rejected"]["content"])
        assert chosen["selectedToolID"] == "outlook.mail.send"
        assert chosen["missingArguments"] == ["subject", "body"]
        assert "to" not in chosen["missingArguments"]
        assert rejected.get("missingArguments") != ["subject", "body"]

    for case in (
        "zero_required_alarm_request_action_1",
        "zero_required_alarm_request_action_2",
        "zero_required_alarm_request_action_3",
        "zero_required_alarm_request_action_4",
    ):
        chosen = json.loads(by_case[case]["chosen"]["content"])
        rejected = json.loads(by_case[case]["rejected"]["content"])
        assert chosen["selectedToolID"] == "alarm.request_authorization"
        assert chosen["actionStep"]["toolID"] == "alarm.request_authorization"
        assert "missingArguments" not in chosen
        assert "actionStep" not in rejected

    for case in (
        "countdown_duration_supplied_missing_title_1",
        "countdown_duration_supplied_missing_title_2",
    ):
        chosen = json.loads(by_case[case]["chosen"]["content"])
        rejected = json.loads(by_case[case]["rejected"]["content"])
        assert chosen["missingArguments"] == ["title"]
        assert rejected["missingArguments"] == ["durationSeconds"]

    for case in (
        "memory_preference_implicit_action_5",
        "memory_preference_implicit_action_6",
    ):
        chosen = json.loads(by_case[case]["chosen"]["content"])
        rejected = json.loads(by_case[case]["rejected"]["content"])
        assert chosen["selectedToolID"] == "memory.save"
        assert chosen["actionStep"]["mustPersistBeforeFinal"] is True
        assert rejected["status"] == "needs_clarification"

    zero_required_list_cases = {
        "zero_required_list_alarm_action": "alarm.list",
        "zero_required_list_calendar_action": "calendar.list",
        "zero_required_list_outlook_folders_action": "outlook.folders.list",
        "zero_required_list_outlook_messages_action": "outlook.messages.list",
        "zero_required_list_reminders_action_1": "reminders.list",
        "zero_required_list_reminders_action_2": "reminders.list",
        "zero_required_list_trigger_action": "trigger.list",
    }
    for case, tool_id in zero_required_list_cases.items():
        chosen = json.loads(by_case[case]["chosen"]["content"])
        rejected = json.loads(by_case[case]["rejected"]["content"])
        assert chosen["selectedToolID"] == tool_id
        assert chosen["actionStep"] == {
            "mustPersistBeforeFinal": True,
            "toolID": tool_id,
            "type": "tool_call",
        }
        assert "status" not in chosen
        assert "missingArguments" not in chosen
        assert chosen != rejected

    for case in (
        "memory_recall_topic_action_1",
        "memory_recall_topic_action_2",
    ):
        chosen = json.loads(by_case[case]["chosen"]["content"])
        rejected = json.loads(by_case[case]["rejected"]["content"])
        assert chosen["selectedToolID"] == "memory.recall"
        assert chosen["actionStep"]["toolID"] == "memory.recall"
        assert rejected["selectedToolID"] == "memory.recall"
        assert rejected["missingArguments"] == ["query"]

    for case, rejected_missing in {
        "memory_save_preference_action_7": ["content", "kind"],
        "memory_save_preference_action_8": ["kind"],
    }.items():
        chosen = json.loads(by_case[case]["chosen"]["content"])
        rejected = json.loads(by_case[case]["rejected"]["content"])
        assert chosen["selectedToolID"] == "memory.save"
        assert chosen["actionStep"]["toolID"] == "memory.save"
        assert rejected["selectedToolID"] == "memory.save"
        assert rejected["missingArguments"] == rejected_missing

    calendar_rejected_routes = {
        "calendar_event_vague_time_missing_start_1": (
            "calendar.create",
            ["title"],
        ),
        "calendar_event_vague_time_missing_start_2": (
            "trigger.create",
            ["prompt", "schedule"],
        ),
        "calendar_event_vague_time_missing_start_3": (
            "alarm.schedule",
            ["inMinutes"],
        ),
        "calendar_event_vague_time_missing_start_4": (
            "calendar.schedule",
            ["startsInMinutes"],
        ),
    }
    for case, (rejected_tool_id, rejected_missing) in (
        calendar_rejected_routes.items()
    ):
        chosen = json.loads(by_case[case]["chosen"]["content"])
        rejected = json.loads(by_case[case]["rejected"]["content"])
        assert chosen["selectedToolID"] == "calendar.create"
        assert chosen["missingArguments"] == ["startsInMinutes"]
        assert "actionStep" not in chosen
        assert rejected["selectedToolID"] == rejected_tool_id
        assert rejected["missingArguments"] == rejected_missing

    generic_calendar_expectations = {
        "calendar_generic_object_missing_all": (
            ["title", "startsInMinutes"],
            ["startsInMinutes"],
        ),
        "calendar_generic_object_with_time_missing_title": (
            ["title"],
            None,
        ),
    }
    for case, (chosen_missing, rejected_missing) in (
        generic_calendar_expectations.items()
    ):
        chosen = json.loads(by_case[case]["chosen"]["content"])
        rejected = json.loads(by_case[case]["rejected"]["content"])
        assert chosen["selectedToolID"] == "calendar.create"
        assert chosen["missingArguments"] == chosen_missing
        assert "actionStep" not in chosen
        assert rejected["selectedToolID"] == "calendar.create"
        if rejected_missing is None:
            assert rejected["actionStep"]["toolID"] == "calendar.create"
            assert "missingArguments" not in rejected
        else:
            assert rejected["missingArguments"] == rejected_missing
            assert "actionStep" not in rejected

    reply_all_expected_missing = {
        "outlook_reply_all_selected_without_body_1": (
            ["messageId", "body"],
            ["messageId"],
        ),
        "outlook_reply_all_selected_without_body_2": (
            ["messageId", "body"],
            ["body"],
        ),
        "outlook_reply_all_selected_with_body": (
            ["messageId"],
            ["messageId", "body"],
        ),
        "outlook_reply_all_operation_missing_all_1": (
            ["messageId", "body"],
            ["messageId"],
        ),
        "outlook_reply_all_operation_missing_all_2": (
            ["messageId", "body"],
            ["body"],
        ),
        "outlook_reply_all_operation_missing_all_3": (
            ["messageId", "body"],
            None,
        ),
    }
    for case, (
        expected_missing,
        rejected_missing,
    ) in reply_all_expected_missing.items():
        chosen = json.loads(by_case[case]["chosen"]["content"])
        rejected = json.loads(by_case[case]["rejected"]["content"])
        assert chosen["missingArguments"] == expected_missing
        assert "actionStep" not in chosen
        if rejected_missing is None:
            assert rejected["actionStep"]["toolID"] == (
                "outlook.message.reply_all"
            )
            assert "missingArguments" not in rejected
        else:
            assert rejected["missingArguments"] == rejected_missing
            assert "actionStep" not in rejected

    for case in (
        "selection_only_health_forbidden_decoys",
        "selection_only_weather_forbidden_decoys",
    ):
        chosen = json.loads(by_case[case]["chosen"]["content"])
        rejected = json.loads(by_case[case]["rejected"]["content"])
        assert "actionStep" not in chosen
        assert rejected["actionStep"]["mustPersistBeforeFinal"] is True

    for case in (
        "action_persistence_photos_search",
        "action_persistence_rag_search",
        "action_persistence_memory_recall",
    ):
        chosen = json.loads(by_case[case]["chosen"]["content"])
        rejected = json.loads(by_case[case]["rejected"]["content"])
        assert chosen["actionStep"]["mustPersistBeforeFinal"] is True
        assert rejected["actionStep"]["mustPersistBeforeFinal"] is False
        assert {
            **rejected,
            "actionStep": {
                **rejected["actionStep"],
                "mustPersistBeforeFinal": True,
            },
        } == chosen

    structured_cancel = by_case["structured_alarm_cancel_explicit_id_action"]
    assert json.loads(structured_cancel["chosen"]["content"])[
        "actionStep"
    ]["toolID"] == "alarm.cancel"
    assert json.loads(structured_cancel["rejected"]["content"])[
        "missingArguments"
    ] == ["title"]


def test_cortex_semantic_generalization_validation_cases_are_explicit(
    compiled_fine_tuning: tuple,
) -> None:
    _, _, fine_tuning = compiled_fine_tuning
    val_sft = [
        record
        for record in _optimized_cortex_training_lane(fine_tuning, "val_sft")
        if record["metadata"].get("taskType")
        == "cortex_route_semantic_generalization_validation"
    ]
    val_dpo = [
        record
        for record in _optimized_cortex_training_lane(fine_tuning, "val_dpo")
        if record["metadata"].get("preferenceType")
        == "route_semantic_generalization_validation"
    ]
    expected_sft_routes = {
        "validation_memory_recall_topic": (
            "memory.recall",
            None,
            "implicit_topic_memory_recall",
        ),
        "validation_memory_save_preference": (
            "memory.save",
            None,
            "implicit_preference_memory_save",
        ),
        "validation_zero_required_reminders_list": (
            "reminders.list",
            None,
            "zero_required_list_action",
        ),
        "validation_calendar_event_missing_start": (
            "calendar.create",
            ["startsInMinutes"],
            "calendar_event_title_without_numeric_delay",
        ),
        "validation_calendar_generic_object_missing_all": (
            "calendar.create",
            ["title", "startsInMinutes"],
            "calendar_operation_object_not_title",
        ),
        "validation_calendar_generic_object_missing_title": (
            "calendar.create",
            ["title"],
            "calendar_operation_object_not_title",
        ),
        "validation_outlook_latest_read": (
            "outlook.message.read",
            None,
            "outlook_read_reference_resolution",
        ),
        "validation_outlook_read_unresolved": (
            "outlook.message.read",
            ["messageId"],
            "outlook_read_reference_resolution",
        ),
        "validation_outlook_reply_all_missing_all": (
            "outlook.message.reply_all",
            ["messageId", "body"],
            "outlook_reply_unresolved_reference_and_body",
        ),
        "validation_files_read_unresolved": (
            "files.read",
            ["name"],
            "outlook_read_files_read_route_lock",
        ),
    }
    expected_dpo_routes = {
        "validation_memory_recall_topic": (
            "memory.recall",
            None,
            "implicit_topic_memory_recall",
        ),
        "validation_memory_save_preference": (
            "memory.save",
            None,
            "implicit_preference_memory_save",
        ),
        "validation_zero_required_reminders_list": (
            "reminders.list",
            None,
            "zero_required_list_action",
        ),
        "validation_calendar_event_missing_start": (
            "calendar.create",
            ["startsInMinutes"],
            "calendar_event_title_without_numeric_delay",
        ),
        "validation_calendar_canonical_id": (
            "calendar.create",
            ["startsInMinutes"],
            "calendar_event_title_without_numeric_delay",
        ),
        "validation_calendar_generic_object_missing_all": (
            "calendar.create",
            ["title", "startsInMinutes"],
            "calendar_operation_object_not_title",
        ),
        "validation_calendar_generic_object_missing_title": (
            "calendar.create",
            ["title"],
            "calendar_operation_object_not_title",
        ),
        "validation_outlook_latest_not_files": (
            "outlook.message.read",
            None,
            "outlook_read_files_read_route_lock",
        ),
        "validation_outlook_selected_not_files": (
            "outlook.message.read",
            ["messageId"],
            "outlook_read_files_read_route_lock",
        ),
        "validation_files_read_not_outlook": (
            "files.read",
            None,
            "outlook_read_files_read_route_lock",
        ),
        "validation_outlook_reply_all_missing_all": (
            "outlook.message.reply_all",
            ["messageId", "body"],
            "outlook_reply_unresolved_reference_and_body",
        ),
    }

    assert len(val_sft) == 10
    assert len(val_dpo) == 11
    assert {record["metadata"]["repairCase"] for record in val_sft} == set(
        expected_sft_routes
    )
    assert {record["metadata"]["repairCase"] for record in val_dpo} == set(
        expected_dpo_routes
    )

    for lane, records, expected_routes in (
        ("sft", val_sft, expected_sft_routes),
        ("dpo", val_dpo, expected_dpo_routes),
    ):
        for record in records:
            case = record["metadata"]["repairCase"]
            (
                expected_tool_id,
                expected_missing,
                expected_failure_family,
            ) = expected_routes[case]
            payload = json.loads(
                (
                    record["messages"][-1]
                    if lane == "sft"
                    else record["chosen"]
                )["content"]
            )
            assert record["metadata"]["requiredSplit"] == "validation"
            assert record["metadata"]["surfaceForm"] == (
                "held_out_semantic_generalization"
            )
            assert record["metadata"]["targetedFailureFamily"] == (
                expected_failure_family
            )
            assert payload["selectedToolID"] == expected_tool_id
            if expected_missing is None:
                assert payload["actionStep"] == {
                    "type": "tool_call",
                    "toolID": expected_tool_id,
                    "mustPersistBeforeFinal": True,
                }
                assert "status" not in payload
            else:
                assert payload["status"] == "needs_clarification"
                assert payload["missingArguments"] == expected_missing
                assert "actionStep" not in payload

    rejected_expectations = {
        "validation_memory_recall_topic": ("memory.recall", ["query"]),
        "validation_memory_save_preference": (
            "memory.save",
            ["content", "kind"],
        ),
        "validation_zero_required_reminders_list": (
            "reminders.list",
            ["title"],
        ),
        "validation_calendar_event_missing_start": (
            "trigger.create",
            ["prompt", "schedule"],
        ),
        "validation_calendar_canonical_id": (
            "calendar.schedule",
            ["startsInMinutes"],
        ),
        "validation_calendar_generic_object_missing_all": (
            "calendar.create",
            ["startsInMinutes"],
        ),
        "validation_calendar_generic_object_missing_title": (
            "calendar.create",
            None,
        ),
        "validation_outlook_latest_not_files": ("files.read", ["name"]),
        "validation_outlook_selected_not_files": ("files.read", ["name"]),
        "validation_files_read_not_outlook": (
            "outlook.message.read",
            ["messageId"],
        ),
        "validation_outlook_reply_all_missing_all": (
            "outlook.message.reply_all",
            ["messageId"],
        ),
    }
    for record in val_dpo:
        rejected = json.loads(record["rejected"]["content"])
        expected_tool_id, expected_missing = rejected_expectations[
            record["metadata"]["repairCase"]
        ]
        assert rejected["selectedToolID"] == expected_tool_id
        if expected_missing is None:
            assert rejected["actionStep"] == {
                "type": "tool_call",
                "toolID": expected_tool_id,
                "mustPersistBeforeFinal": True,
            }
            assert "missingArguments" not in rejected
        else:
            assert rejected["missingArguments"] == expected_missing
            assert "actionStep" not in rejected


def test_cortex_semantic_generalization_skips_absent_required_arguments() -> None:
    calendar_create = ToolManifest(
        id="calendar.create",
        arguments=[
            ToolArgumentManifest(name="title", type="string", required=True),
        ],
    )
    manifest = AgentBehaviorManifest(
        tools=[calendar_create],
        routingMatrix=[
            RoutingMatrixEntry(
                intent="calendar",
                allowedTools=[calendar_create.id],
            ),
        ],
    )

    records = _cortex_failure_repair_sft_records(
        manifest,
        {calendar_create.id: calendar_create},
    )

    assert not any(
        record["metadata"].get("repairCase")
        == "validation_calendar_event_missing_start"
        for record in records
    )


def test_cortex_calendar_schedule_alias_occurs_only_in_rejected_routes(
    compiled_fine_tuning: tuple,
) -> None:
    manifest, _, fine_tuning = compiled_fine_tuning
    alias = "calendar.schedule"
    assert alias not in {tool.id for tool in manifest.tools}

    chosen_outputs = [
        record["messages"][-1]["content"]
        for lane in ("train_sft", "val_sft")
        for record in _optimized_cortex_training_lane(fine_tuning, lane)
    ] + [
        record["chosen"]["content"]
        for lane in ("train_dpo", "val_dpo")
        for record in _optimized_cortex_training_lane(fine_tuning, lane)
    ]
    assert not [
        payload
        for content in chosen_outputs
        if (payload := _load_json_object(content)) is not None
        and payload.get("selectedToolID") == alias
    ]

    alias_rejections = {
        (lane, record["metadata"].get("repairCase"))
        for lane in ("train_dpo", "val_dpo")
        for record in _optimized_cortex_training_lane(fine_tuning, lane)
        if (
            payload := _load_json_object(record["rejected"]["content"])
        ) is not None
        and payload.get("selectedToolID") == alias
    }
    assert alias_rejections == {
        ("train_dpo", "calendar_event_vague_time_missing_start_4"),
        ("val_dpo", "validation_calendar_canonical_id"),
    }


def test_cortex_targeted_failure_repairs_stay_below_frozen_eval_containment(
    compiled_fine_tuning: tuple,
) -> None:
    _, _, fine_tuning = compiled_fine_tuning
    cortex = fine_tuning["cortex"]
    new_sft_repair_cases = {
        "natural_calendar_human_event_vague_time_missing_numeric_start_1",
        "natural_calendar_human_event_vague_time_missing_numeric_start_2",
        "natural_calendar_human_event_vague_time_missing_numeric_start_3",
        "natural_calendar_human_event_vague_time_missing_numeric_start_4",
        "natural_calendar_generic_object_with_time_missing_title_1",
        "natural_memory_recall_query_actionable_3",
        "natural_memory_recall_query_actionable_4",
        "natural_memory_save_preference_actionable_9",
        "natural_memory_save_preference_actionable_10",
        "natural_zero_required_alarm_list_actionable_3",
        "natural_zero_required_calendar_list_actionable_1",
        "natural_zero_required_outlook_folders_list_actionable_1",
        "natural_zero_required_outlook_messages_list_actionable_1",
        "natural_zero_required_reminders_list_actionable_4",
        "natural_zero_required_reminders_list_actionable_5",
        "natural_zero_required_trigger_list_actionable_1",
        "natural_outlook_read_reference_latest_outlook_email_1",
        "natural_outlook_read_reference_latest_outlook_email_2",
        "natural_outlook_read_reference_latest_outlook_email_3",
        "natural_outlook_read_reference_unresolved_reference_1",
        "natural_outlook_read_reference_unresolved_reference_2",
        "natural_outlook_read_reference_unresolved_reference_3",
        "natural_outlook_reply_all_operation_without_values_all_missing_1",
        "natural_outlook_reply_all_operation_without_values_all_missing_2",
        "natural_outlook_reply_all_operation_without_values_all_missing_3",
        (
            "natural_outlook_reply_all_selected_message_body_boundary_"
            "missing_message_id_1"
        ),
        (
            "natural_outlook_reply_all_selected_message_body_boundary_"
            "missing_message_id_2"
        ),
        (
            "natural_outlook_reply_all_selected_message_body_boundary_"
            "missing_message_id_3"
        ),
        "retry_outlook_latest_read_exact_catalog_reselection",
        "retry_files_read_clarification_reselection",
    }
    new_dpo_repair_cases = {
        "calendar_event_vague_time_missing_start_1",
        "calendar_event_vague_time_missing_start_2",
        "calendar_event_vague_time_missing_start_3",
        "calendar_event_vague_time_missing_start_4",
        "calendar_generic_object_missing_all",
        "calendar_generic_object_with_time_missing_title",
        "memory_recall_topic_action_1",
        "memory_recall_topic_action_2",
        "memory_save_preference_action_7",
        "memory_save_preference_action_8",
        "zero_required_list_alarm_action",
        "zero_required_list_calendar_action",
        "zero_required_list_outlook_folders_action",
        "zero_required_list_outlook_messages_action",
        "zero_required_list_reminders_action_1",
        "zero_required_list_reminders_action_2",
        "zero_required_list_trigger_action",
        "outlook_read_latest_not_files_1",
        "outlook_read_latest_exact_tool_2",
        "outlook_read_selected_not_files",
        "files_read_explicit_not_outlook",
        "files_read_unresolved_not_outlook",
        "outlook_reply_all_operation_missing_all_1",
        "outlook_reply_all_operation_missing_all_2",
        "outlook_reply_all_operation_missing_all_3",
    }
    train_sft_by_case = {
        record["metadata"].get("repairCase"): record
        for record in _optimized_cortex_training_lane(fine_tuning, "train_sft")
    }
    train_dpo_by_case = {
        record["metadata"].get("repairCase"): record
        for record in _optimized_cortex_training_lane(fine_tuning, "train_dpo")
    }
    assert new_sft_repair_cases <= set(train_sft_by_case)
    assert new_dpo_repair_cases <= set(train_dpo_by_case)

    def sft_user_request(case: str) -> str:
        record = train_sft_by_case[case]
        prompt = record["messages"][-2]["content"]
        if record["metadata"].get("taskType") != (
            "cortex_route_strict_retry_repair"
        ):
            return prompt
        retry_separator = "\n\n" + STRICT_JSON_RETRY_DPO_INSTRUCTION
        assert retry_separator in prompt
        return prompt.split(retry_separator, maxsplit=1)[0]

    targeted_prompts = {
        sft_user_request(case) for case in new_sft_repair_cases
    } | {
        train_dpo_by_case[case]["prompt"][-1]["content"]
        for case in new_dpo_repair_cases
    }
    frozen_eval_prompts = {
        record["messages"][-1]["content"] for record in cortex.eval
    }

    assert targeted_prompts
    assert frozen_eval_prompts
    for train_prompt in targeted_prompts:
        train_tokens = set(re.findall(r"\w+", train_prompt.casefold()))
        assert train_tokens
        for eval_prompt in frozen_eval_prompts:
            eval_tokens = set(re.findall(r"\w+", eval_prompt.casefold()))
            containment = len(train_tokens & eval_tokens) / min(
                len(train_tokens),
                len(eval_tokens),
            )
            assert containment < 0.5, (
                "targeted Cortex train prompt is lexically too close to frozen "
                f"evaluation text: {containment:.3f}; train={train_prompt!r}; "
                f"eval={eval_prompt!r}"
            )


def test_cortex_failure_repair_positive_labels_do_not_conflict(
    compiled_fine_tuning: tuple,
) -> None:
    _, _, fine_tuning = compiled_fine_tuning
    train_sft = _optimized_cortex_training_lane(fine_tuning, "train_sft")
    train_dpo = _optimized_cortex_training_lane(fine_tuning, "train_dpo")
    repair_prompts = {
        record["messages"][-2]["content"]
        for record in train_sft
        if record["metadata"].get("curriculumMode")
        in {"failure_repair_actionable", "failure_repair_clarification"}
    } | {
        record["prompt"][-1]["content"]
        for record in train_dpo
        if record["metadata"].get("preferenceType")
        == "route_failure_repair_bidirectional"
    }
    labels_by_prompt: dict[str, set[str]] = {
        prompt: set() for prompt in repair_prompts
    }
    for record in train_sft:
        prompt = record["messages"][-2]["content"]
        if prompt in labels_by_prompt:
            labels_by_prompt[prompt].add(
                json.dumps(
                    json.loads(record["messages"][-1]["content"]),
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
    for record in train_dpo:
        prompt = record["prompt"][-1]["content"]
        if prompt in labels_by_prompt:
            labels_by_prompt[prompt].add(
                json.dumps(
                    json.loads(record["chosen"]["content"]),
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )

    assert labels_by_prompt
    assert not {
        prompt: labels
        for prompt, labels in labels_by_prompt.items()
        if len(labels) != 1
    }


def test_cortex_routes_serialize_manifest_grounding_before_state_and_endcap(
    compiled_fine_tuning: tuple,
) -> None:
    manifest, _, fine_tuning = compiled_fine_tuning
    tools_by_id = {tool.id: tool for tool in manifest.tools}
    seen_action = False
    seen_status = False
    seen_required_action = False
    seen_zero_required_action = False
    serialized_routes = [
        record["messages"][-1]["content"]
        for lane in ("train_sft", "val_sft")
        for record in _optimized_cortex_training_lane(fine_tuning, lane)
    ] + [
        record["chosen"]["content"]
        for lane in ("train_dpo", "val_dpo")
        for record in _optimized_cortex_training_lane(fine_tuning, lane)
    ]
    for serialized in serialized_routes:
        payload = json.loads(serialized)
        if "selectedToolID" not in payload:
            continue
        keys = list(payload)
        assert keys[:3] == ["selectedToolID", "intent", "reasoningSummary"]
        assert keys[-2:] == ["requiresApproval", "nextModel"]
        selected_tool_id = payload["selectedToolID"]
        if "actionStep" in payload:
            seen_action = True
            assert keys == [
                "selectedToolID",
                "intent",
                "reasoningSummary",
                "actionStep",
                "requiresApproval",
                "nextModel",
            ]
            tool = tools_by_id[selected_tool_id]
            required = [
                argument.name for argument in tool.arguments if argument.required
            ]
            if required:
                seen_required_action = True
                assert payload["reasoningSummary"] == (
                    f"Manifest row {tool.id} has all exact required names supplied: "
                    f"{', '.join(required)}."
                )
            else:
                seen_zero_required_action = True
                assert payload["reasoningSummary"] == (
                    f"Manifest row {tool.id} has no required values."
                )
        elif "status" in payload:
            seen_status = True
            if payload["status"] == "needs_clarification":
                assert keys == [
                    "selectedToolID",
                    "intent",
                    "reasoningSummary",
                    "status",
                    "missingArguments",
                    "clarification",
                    "requiresApproval",
                    "nextModel",
                ]
                assert payload["reasoningSummary"] == (
                    f"Manifest row {selected_tool_id} is missing exactly this "
                    f"required subset: {', '.join(payload['missingArguments'])}."
                )
            else:
                assert keys == [
                    "selectedToolID",
                    "intent",
                    "reasoningSummary",
                    "status",
                    "requiresApproval",
                    "nextModel",
                ]
        else:
            assert keys == [
                "selectedToolID",
                "intent",
                "reasoningSummary",
                "requiresApproval",
                "nextModel",
            ]
    assert seen_action is True
    assert seen_status is True
    assert seen_required_action is True
    assert seen_zero_required_action is True


def test_cortex_required_sft_train_records_never_drift_to_validation(
    compiled_fine_tuning: tuple,
) -> None:
    _, _, fine_tuning = compiled_fine_tuning
    variants = fine_tuning["cortex"].experiment_variants
    for variant_name, variant in variants.items():
        required_train = {
            json.dumps(record["messages"], ensure_ascii=False, sort_keys=True)
            for record in variant["train_sft"]
            if record["metadata"].get("requiredSplit") == "train"
        }
        validation = {
            json.dumps(record["messages"], ensure_ascii=False, sort_keys=True)
            for record in variant["val_sft"]
        }
        assert required_train, variant_name
        assert required_train.isdisjoint(validation), variant_name
        assert not [
            record
            for record in variant["val_sft"]
            if record["metadata"].get("requiredSplit") == "train"
        ], variant_name


def test_cortex_loss_balance_caps_public_tools_and_supplemental_targets(
    compiled_fine_tuning: tuple,
) -> None:
    _, _, fine_tuning = compiled_fine_tuning
    cortex = fine_tuning["cortex"]
    records = cortex.train_sft + cortex.val_sft
    constraints = cortex.dataset_card["constraints"]
    quality = cortex.dataset_card["quality"]
    per_public_tool: Counter[str] = Counter()
    total_chars = 0
    supplemental_chars = 0

    for record in records:
        assistant = record["messages"][-1]["content"]
        total_chars += len(assistant)
        source_family = record["metadata"].get("sourceFamily")
        if source_family in CORTEX_SUPPLEMENTAL_GROUNDING_SOURCE_FAMILIES:
            supplemental_chars += len(assistant)
        if isinstance(record["metadata"].get("publicCorpus"), dict):
            payload = json.loads(assistant)
            selected_tool_id = payload.get("selectedToolID")
            public_corpus = record["metadata"]["publicCorpus"]
            assert not (
                selected_tool_id == "reminders.list"
                and public_corpus.get("sourceRepository")
                == "AmazonScience/massive"
                and public_corpus.get("stratum") == "lists_query"
            )
            if isinstance(selected_tool_id, str):
                per_public_tool[selected_tool_id] += 1

    observed_share = supplemental_chars / total_chars
    assert per_public_tool
    assert max(per_public_tool.values()) <= constraints[
        "maxCortexPublicSFTRecordsPerTool"
    ]
    assert observed_share <= constraints[
        "maxCortexSupplementalAssistantCharShare"
    ]
    assert quality["assistantTargetCharCount"] == total_chars
    assert quality["supplementalAssistantTargetCharCount"] == supplemental_chars
    assert quality["supplementalAssistantTargetCharShare"] == pytest.approx(
        observed_share
    )


def test_cortex_public_required_argument_routes_require_same_row_audit(
    compiled_fine_tuning: tuple,
) -> None:
    manifest, _, fine_tuning = compiled_fine_tuning
    tools_by_id = {tool.id: tool for tool in manifest.tools}
    public_records = [
        record
        for record in fine_tuning["cortex"].train_sft
        + fine_tuning["cortex"].val_sft
        if isinstance(record["metadata"].get("publicCorpus"), dict)
    ]

    assert public_records
    for record in public_records:
        payload = json.loads(record["messages"][-1]["content"])
        selected_tool_id = payload.get("selectedToolID")
        if selected_tool_id is None:
            assert payload.get("status") == "no_tool_route"
            continue
        tool = tools_by_id[selected_tool_id]
        required_arguments = [
            argument.name for argument in tool.arguments if argument.required
        ]
        if not required_arguments:
            continue
        quality = record["metadata"]["publicCorpus"].get("quality")
        assert isinstance(quality, dict)
        assert quality.get("sameRowArgumentCoverageAudited") is True
        if payload.get("status") == "needs_clarification":
            assert payload["missingArguments"] == [
                argument
                for argument in required_arguments
                if argument in payload["missingArguments"]
            ]
            assert "actionStep" not in payload
        else:
            assert payload["actionStep"] == {
                "mustPersistBeforeFinal": True,
                "toolID": selected_tool_id,
                "type": "tool_call",
            }


def test_cortex_generic_preferences_use_canonical_semantically_named_routes(
    compiled_fine_tuning: tuple,
) -> None:
    manifest, _, fine_tuning = compiled_fine_tuning
    tools_by_id = {tool.id: tool for tool in manifest.tools}
    records = fine_tuning["cortex"].train_dpo + fine_tuning["cortex"].val_dpo
    preference_types = {
        "action_step_persistence",
        "manifest_tool_only",
        "safe_tool_selection",
    }
    generic = [
        record
        for record in records
        if record["metadata"].get("preferenceType") in preference_types
    ]

    assert {record["metadata"]["preferenceType"] for record in generic} == preference_types
    for record in generic:
        chosen = json.loads(record["chosen"]["content"])
        selected_tool_id = chosen["selectedToolID"]
        assert selected_tool_id in record["prompt"][1]["content"]
        base_fields = {
            "intent",
            "selectedToolID",
            "requiresApproval",
            "nextModel",
            "reasoningSummary",
        }
        if record["metadata"]["preferenceType"] == "action_step_persistence":
            assert set(chosen) == base_fields | {"actionStep"}
            assert chosen["actionStep"] == {
                "mustPersistBeforeFinal": True,
                "toolID": selected_tool_id,
                "type": "tool_call",
            }
            required = [
                argument.name
                for argument in tools_by_id[selected_tool_id].arguments
                if argument.required
            ]
            assert chosen["reasoningSummary"] == (
                f"Manifest row {selected_tool_id} has all exact required names "
                f"supplied: {', '.join(required)}."
                if required
                else f"Manifest row {selected_tool_id} has no required values."
            )
        else:
            assert set(chosen) == base_fields
            assert "actionStep" not in chosen
            assert chosen["reasoningSummary"] == (
                f"Manifest row {selected_tool_id} is selected for intent "
                f"{chosen['intent']} without actionStep."
            )
        assert "risk" not in chosen


def test_cortex_selection_only_targets_do_not_claim_action_persistence(
    compiled_fine_tuning: tuple,
) -> None:
    _, _, fine_tuning = compiled_fine_tuning
    cortex = fine_tuning["cortex"]
    base_fields = {
        "intent",
        "selectedToolID",
        "requiresApproval",
        "nextModel",
        "reasoningSummary",
    }
    selection_payloads: list[dict] = []

    for record in cortex.train_sft + cortex.val_sft:
        payload = json.loads(record["messages"][-1]["content"])
        if set(payload) == base_fields and payload["selectedToolID"] is not None:
            selection_payloads.append(payload)
    for record in cortex.train_dpo + cortex.val_dpo:
        payload = json.loads(record["chosen"]["content"])
        if set(payload) == base_fields and payload["selectedToolID"] is not None:
            selection_payloads.append(payload)

    assert selection_payloads
    for payload in selection_payloads:
        reasoning = payload["reasoningSummary"].casefold()
        assert "persist one action" not in reasoning
        assert "persisted action" not in reasoning


def test_cortex_no_tool_dpo_has_positive_split_anchors_and_bounded_hybrid_ratio(
    compiled_fine_tuning: tuple,
) -> None:
    manifest, _, fine_tuning = compiled_fine_tuning
    tools_by_id = {tool.id: tool for tool in manifest.tools}
    allowed_tools_by_intent: dict[str, set[str]] = {}
    for entry in manifest.routingMatrix:
        allowed_tools_by_intent.setdefault(entry.intent, set()).update(
            tool_id for tool_id in entry.allowedTools if tool_id in tools_by_id
        )
    for intent in manifest.intents:
        allowed_tools_by_intent.setdefault(intent.id, set()).update(
            tool_id for tool_id in intent.allowedToolIDs if tool_id in tools_by_id
        )
    no_tool_intents = {
        intent
        for intent, allowed_tool_ids in allowed_tools_by_intent.items()
        if not allowed_tool_ids
    }
    base_fields = {
        "intent",
        "selectedToolID",
        "requiresApproval",
        "nextModel",
        "reasoningSummary",
    }
    action_fields = base_fields | {"actionStep"}
    frozen_eval_prompts = {
        record["messages"][-1]["content"]
        for record in fine_tuning["cortex"].eval
    }
    prompt_sets: dict[str, set[str]] = {}

    assert no_tool_intents
    lane_contracts = (
        (
            "train_dpo",
            "route_no_tool_vs_spurious_action",
            "train",
            10,
        ),
        (
            "val_dpo",
            "route_no_tool_vs_spurious_action_validation",
            "validation",
            4,
        ),
    )
    for lane, preference_type, required_split, expected_per_intent in lane_contracts:
        records = _optimized_cortex_training_lane(fine_tuning, lane)
        anchors = [
            record
            for record in records
            if record["metadata"].get("preferenceType") == preference_type
        ]
        chosen_by_intent = Counter(
            json.loads(record["chosen"]["content"])["intent"]
            for record in anchors
        )
        assert chosen_by_intent == Counter(
            {intent: expected_per_intent for intent in no_tool_intents}
        )
        assert Counter(
            record["metadata"].get("surfaceStyle") for record in anchors
        ) == Counter(
            {
                "natural": len(no_tool_intents) * (expected_per_intent // 2),
                "contract": len(no_tool_intents) * (expected_per_intent // 2),
            }
        )

        prompts: set[str] = set()
        for record in anchors:
            prompt = record["prompt"][-1]["content"]
            chosen = json.loads(record["chosen"]["content"])
            rejected = json.loads(record["rejected"]["content"])
            selected_tool_id = rejected["selectedToolID"]
            selected_tool = tools_by_id[selected_tool_id]

            prompts.add(prompt)
            assert prompt not in frozen_eval_prompts
            assert record["metadata"]["requiredSplit"] == required_split
            assert set(chosen) == base_fields | {"status"}
            assert chosen["intent"] in no_tool_intents
            assert chosen["selectedToolID"] is None
            assert chosen["requiresApproval"] is False
            assert chosen["nextModel"] == "mouth"
            assert chosen["status"] == "no_tool_route"
            assert "actionStep" not in chosen
            assert set(rejected) == action_fields
            assert selected_tool_id not in allowed_tools_by_intent[chosen["intent"]]
            assert rejected["requiresApproval"] is selected_tool.requiresApproval
            assert rejected["nextModel"] == (
                "approval" if selected_tool.requiresApproval else "executor"
            )
            assert rejected["actionStep"] == {
                "mustPersistBeforeFinal": True,
                "toolID": selected_tool_id,
                "type": "tool_call",
            }
        assert len(prompts) == len(anchors)
        prompt_sets[lane] = prompts

        if lane == "train_dpo":
            hybrid_negative_count = 0
            for record in records:
                rejected = _load_json_object(record["rejected"]["content"])
                if (
                    rejected is not None
                    and rejected.get("selectedToolID") is not None
                    and rejected.get("status") == "no_tool_route"
                ):
                    hybrid_negative_count += 1
            assert hybrid_negative_count <= 4 * len(anchors)

    assert prompt_sets["train_dpo"].isdisjoint(prompt_sets["val_dpo"])


def test_cortex_files_read_without_name_uses_exact_clarification(compiled_fine_tuning: tuple) -> None:
    manifest, _, fine_tuning = compiled_fine_tuning
    cortex = fine_tuning["cortex"]
    records = cortex.train_sft + cortex.val_sft
    record = next(
        item
        for item in records
        if item["metadata"].get("taskType")
        == "ultra_specific_files_read_clarification"
    )
    output = json.loads(record["messages"][-1]["content"])
    frozen_eval_prompts = {item["messages"][1]["content"] for item in cortex.eval}

    assert output["selectedToolID"] == "files.read"
    assert output["status"] == "needs_clarification"
    files_read = next(tool for tool in manifest.tools if tool.id == "files.read")
    assert output["missingArguments"] == [
        argument.name for argument in files_read.arguments if argument.required
    ]
    assert output["clarification"] == "Which file should I read?"
    assert output["nextModel"] == "mouth"
    assert "actionStep" not in output
    assert "arguments" not in output
    assert record["messages"][1]["content"] not in frozen_eval_prompts


def test_cortex_has_missing_required_argument_clarification_for_every_tool(
    compiled_fine_tuning: tuple,
) -> None:
    manifest, _, fine_tuning = compiled_fine_tuning
    cortex = fine_tuning["cortex"]
    tools_by_id = {tool.id: tool for tool in manifest.tools}
    routed_intents_by_tool: dict[str, set[str]] = {}
    for entry in manifest.routingMatrix:
        for tool_id in entry.allowedTools:
            routed_intents_by_tool.setdefault(tool_id, set()).add(entry.intent)
    expected_arguments_by_tool = {
        tool.id: [argument.name for argument in tool.arguments if argument.required]
        for tool in manifest.tools
        if any(argument.required for argument in tool.arguments)
        and tool.id in routed_intents_by_tool
    }
    clarification_task_types = {
        "ultra_specific_files_read_clarification",
        "ultra_specific_missing_required_argument_clarification",
    }
    records = [
        record
        for record in cortex.train_sft + cortex.val_sft
        if record["metadata"].get("taskType") in clarification_task_types
    ]
    frozen_eval_prompts = {item["messages"][1]["content"] for item in cortex.eval}
    records_by_tool = {
        record["metadata"]["toolIDs"][0]: record
        for record in records
    }

    assert expected_arguments_by_tool
    assert len(records) == len(expected_arguments_by_tool)
    assert set(records_by_tool) == set(expected_arguments_by_tool)
    assert len({record["messages"][1]["content"] for record in records}) == len(records)

    for tool_id, required_arguments in expected_arguments_by_tool.items():
        record = records_by_tool[tool_id]
        output = json.loads(record["messages"][-1]["content"])
        tool = tools_by_id[tool_id]
        expected_contract = {
            "requiresApproval": tool.requiresApproval,
            "permissionKey": tool.permissionKey,
            "permissionKind": tool.permissionKind,
            "confirmationMode": tool.confirmationMode,
        }

        assert output["selectedToolID"] == tool_id
        assert output["intent"] in routed_intents_by_tool[tool_id]
        assert record["metadata"]["intent"] == output["intent"]
        assert output["status"] == "needs_clarification"
        assert output["missingArguments"] == required_arguments
        assert output["nextModel"] == "mouth"
        assert record["metadata"]["toolContracts"][tool_id] == expected_contract
        assert set(output) == {
            "intent",
            "selectedToolID",
            "requiresApproval",
            "nextModel",
            "reasoningSummary",
            "status",
            "missingArguments",
            "clarification",
        }
        assert isinstance(output["clarification"], str) and output["clarification"].endswith("?")
        assert isinstance(output["reasoningSummary"], str) and output["reasoningSummary"].endswith(".")
        assert "\n" not in output["clarification"]
        assert "\n" not in output["reasoningSummary"]
        assert "actionStep" not in output
        assert "arguments" not in output
        assert record["messages"][1]["content"] not in frozen_eval_prompts


def test_cortex_clarification_records_derive_contract_and_skip_unrouted_tools() -> None:
    manifest = AgentBehaviorManifest(
        tools=[
            ToolManifest(
                id="files.read",
                requiresApproval=True,
                permissionKey="NSContactsUsageDescription",
                permissionKind="contacts",
                confirmationMode="userApproval",
                arguments=[ToolArgumentManifest(name="name", type="string", required=True)],
            ),
            ToolManifest(
                id="orphan.run",
                arguments=[ToolArgumentManifest(name="payload", type="string", required=True)],
            ),
        ],
        routingMatrix=[
            RoutingMatrixEntry(intent="files", allowedTools=["files.read"]),
        ],
    )
    records = _ultra_specific_cortex_records(
        manifest,
        {tool.id: tool for tool in manifest.tools},
    )
    clarification_records = [
        record
        for record in records
        if record["metadata"].get("taskType")
        in {
            "ultra_specific_files_read_clarification",
            "ultra_specific_missing_required_argument_clarification",
        }
    ]

    assert len(clarification_records) == 1
    record = clarification_records[0]
    output = json.loads(record["messages"][-1]["content"])
    assert record["metadata"]["toolIDs"] == ["files.read"]
    assert output["intent"] == "files"
    assert output["requiresApproval"] is True
    assert set(output) == {
        "intent",
        "selectedToolID",
        "requiresApproval",
        "nextModel",
        "reasoningSummary",
        "status",
        "missingArguments",
        "clarification",
    }
    assert record["metadata"]["toolContracts"]["files.read"] == {
        "requiresApproval": True,
        "permissionKey": "NSContactsUsageDescription",
        "permissionKind": "contacts",
        "confirmationMode": "userApproval",
    }


def test_dpo_records_have_prompt_chosen_rejected(compiled_fine_tuning: tuple) -> None:
    _, _, fine_tuning = compiled_fine_tuning
    for agent in AGENTS:
        for record in (fine_tuning[agent].train_dpo + fine_tuning[agent].val_dpo)[:20]:
            assert isinstance(record["prompt"], list)
            assert record["prompt"][0]["role"] == "system"
            assert record["prompt"][1]["role"] == "user"
            assert record["chosen"]["role"] == "assistant"
            assert record["rejected"]["role"] == "assistant"
            assert record["chosen"]["content"] != record["rejected"]["content"]


@pytest.mark.parametrize(
    "assistant",
    (
        '{"selectedToolID":null,"selectedToolID":"shadow"}',
        '{"selectedToolID":null,"strictProbe":NaN}',
        '{"selectedToolID":null,"strictProbe":Infinity}',
        '{"selectedToolID":null,"strictProbe":-Infinity}',
        '{"selectedToolID":null,"strictProbe":1e400}',
    ),
)
def test_cortex_sft_rejects_non_strict_raw_json_before_canonicalization(
    compiled_fine_tuning: tuple,
    assistant: str,
) -> None:
    manifest, _, _ = compiled_fine_tuning

    with pytest.raises(ValueError, match="JSON key|Non-finite JSON number"):
        _canonicalize_cortex_sft_output(
            assistant,
            manifest=manifest,
            source_family=ULTRA_SPECIFIC_SOURCE_FAMILY,
            task_type="routing_matrix_adherence",
        )


@pytest.mark.parametrize(
    "value",
    (
        '{"selectedToolID":null,"selectedToolID":"shadow"}',
        '{"selectedToolID":null,"strictProbe":Infinity}',
    ),
)
def test_cortex_route_ordering_preserves_invalid_json(value: str) -> None:
    assert _ordered_cortex_route_text(value) == value
    assert fine_tuning_module._ordered_cortex_rejected_route_text(value) == value


@pytest.mark.parametrize(
    "helper_name",
    (
        "_canonical_strict_json_object",
        "_manifest_valid_executor_payload",
        "_public_cortex_route_is_safe",
        "_ordered_cortex_route_text",
        "_ordered_cortex_rejected_route_text",
    ),
)
@pytest.mark.parametrize(
    "exception_type",
    (TypeError, ValueError, RecursionError),
)
def test_strict_json_helpers_propagate_unexpected_parser_errors(
    monkeypatch: pytest.MonkeyPatch,
    helper_name: str,
    exception_type: type[Exception],
) -> None:
    def raise_unexpected_parser_error(_value: str) -> Any:
        raise exception_type("unexpected parser failure")

    monkeypatch.setattr(
        fine_tuning_module,
        "_strict_json_loads",
        raise_unexpected_parser_error,
    )
    manifest = object()
    invocations = {
        "_canonical_strict_json_object": lambda: (
            fine_tuning_module._canonical_strict_json_object("{}")
        ),
        "_manifest_valid_executor_payload": lambda: (
            fine_tuning_module._manifest_valid_executor_payload(manifest, "{}")
        ),
        "_public_cortex_route_is_safe": lambda: (
            fine_tuning_module._public_cortex_route_is_safe(
                manifest,
                "{}",
                public_corpus={},
            )
        ),
        "_ordered_cortex_route_text": lambda: (
            fine_tuning_module._ordered_cortex_route_text("{}")
        ),
        "_ordered_cortex_rejected_route_text": lambda: (
            fine_tuning_module._ordered_cortex_rejected_route_text("{}")
        ),
    }

    with pytest.raises(exception_type, match="unexpected parser failure"):
        invocations[helper_name]()


@pytest.mark.parametrize(
    "chosen",
    (
        '{"selectedToolID":null,"selectedToolID":"shadow"}',
        '{"selectedToolID":null,"strictProbe":NaN}',
        '{"selectedToolID":null,"strictProbe":Infinity}',
        '{"selectedToolID":null,"strictProbe":-Infinity}',
        '{"selectedToolID":null,"strictProbe":1e400}',
    ),
)
def test_cortex_dpo_chosen_rejects_non_strict_raw_json_before_reordering(
    compiled_fine_tuning: tuple,
    chosen: str,
) -> None:
    manifest, _, fine_tuning = compiled_fine_tuning
    record = copy.deepcopy(fine_tuning["cortex"].train_dpo[0])
    record["chosen"]["content"] = chosen

    with pytest.raises(ValueError, match="JSON key|Non-finite JSON number"):
        _bind_cortex_dpo_route_contract(manifest, [record])


@pytest.mark.parametrize(
    "rejected",
    (
        "not a JSON object",
        '{"selectedToolID":null,"selectedToolID":"shadow"}',
        '{"selectedToolID":null,"strictProbe":NaN}',
        '{"selectedToolID":null,"strictProbe":Infinity}',
        '{"selectedToolID":null,"strictProbe":-Infinity}',
        '{"selectedToolID":null,"strictProbe":1e400}',
    ),
)
def test_cortex_dpo_rejected_preserves_malformed_negative_evidence(
    compiled_fine_tuning: tuple,
    rejected: str,
) -> None:
    manifest, _, fine_tuning = compiled_fine_tuning
    record = copy.deepcopy(fine_tuning["cortex"].train_dpo[0])
    record["rejected"]["content"] = rejected

    bound = _bind_cortex_dpo_route_contract(manifest, [record])

    assert bound[0]["rejected"]["content"] == rejected


def test_validator_rejects_non_json_cortex_sft_and_chosen_dpo(compiled_fine_tuning: tuple) -> None:
    manifest, _, fine_tuning = compiled_fine_tuning
    cortex = fine_tuning["cortex"]
    train_sft = copy.deepcopy(cortex.train_sft)
    train_dpo = copy.deepcopy(cortex.train_dpo)
    train_sft[0]["messages"][-1]["content"] = "plain Cortex output"
    train_dpo[0]["chosen"]["content"] = "plain preferred Cortex output"
    mutated = dict(fine_tuning)
    mutated["cortex"] = replace(
        cortex,
        train_sft=train_sft,
        train_dpo=train_dpo,
    )

    failures = validate_agent_fine_tuning_datasets(manifest, mutated)
    codes = {failure.code for failure in failures}

    assert "cortex_non_json_output" in codes
    assert "cortex_dpo_non_json_output" in codes


def test_validator_rejects_duplicate_keys_and_malformed_cortex_route_modes(
    compiled_fine_tuning: tuple,
) -> None:
    manifest, _, fine_tuning = compiled_fine_tuning
    cortex = fine_tuning["cortex"]
    train_sft = copy.deepcopy(cortex.train_sft)
    train_dpo = copy.deepcopy(cortex.train_dpo)
    sft_index = next(
        index
        for index, record in enumerate(train_sft)
        if "Task mode: Cortex route mode."
        in record["messages"][0]["content"]
    )
    original_sft = train_sft[sft_index]["messages"][-1]["content"]
    train_sft[sft_index]["messages"][-1]["content"] = (
        '{"intent":"duplicate",' + original_sft[1:]
    )
    dpo_index = next(
        index
        for index, record in enumerate(train_dpo)
        if "Task mode: Cortex route mode."
        in record["prompt"][0]["content"]
    )
    malformed = json.loads(train_dpo[dpo_index]["chosen"]["content"])
    malformed["handoff"] = "executor"
    train_dpo[dpo_index]["chosen"]["content"] = json.dumps(
        malformed,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    mutated = dict(fine_tuning)
    mutated["cortex"] = replace(
        cortex,
        train_sft=train_sft,
        train_dpo=train_dpo,
    )

    failures = validate_agent_fine_tuning_datasets(manifest, mutated)
    codes = {failure.code for failure in failures}

    assert "cortex_duplicate_json_key" in codes
    assert "cortex_dpo_route_contract_invalid" in codes


def test_validator_rejects_nonfinite_cortex_sft_and_chosen_dpo(
    compiled_fine_tuning: tuple,
) -> None:
    manifest, _, fine_tuning = compiled_fine_tuning
    cortex = fine_tuning["cortex"]
    train_sft = copy.deepcopy(cortex.train_sft)
    train_dpo = copy.deepcopy(cortex.train_dpo)
    sft_index = next(
        index
        for index, record in enumerate(train_sft)
        if "Task mode: Cortex route mode."
        in record["messages"][0]["content"]
    )
    sft_output = train_sft[sft_index]["messages"][-1]["content"]
    dpo_index = next(
        index
        for index, record in enumerate(train_dpo)
        if "Task mode: Cortex route mode."
        in record["prompt"][0]["content"]
    )
    dpo_output = train_dpo[dpo_index]["chosen"]["content"]
    train_sft[sft_index]["messages"][-1]["content"] = (
        sft_output[:-1] + ',"strictProbe":1e400}'
    )
    train_dpo[dpo_index]["chosen"]["content"] = (
        dpo_output[:-1] + ',"strictProbe":Infinity}'
    )
    mutated = dict(fine_tuning)
    mutated["cortex"] = replace(
        cortex,
        train_sft=train_sft,
        train_dpo=train_dpo,
    )

    failures = validate_agent_fine_tuning_datasets(manifest, mutated)
    codes = {failure.code for failure in failures}

    assert "cortex_non_json_output" in codes
    assert "cortex_dpo_non_json_output" in codes


def test_no_unknown_agent_roles_unknown_tools_or_sentinel_leaks(compiled_fine_tuning: tuple) -> None:
    manifest, _, fine_tuning = compiled_fine_tuning
    failures = validate_agent_fine_tuning_datasets(manifest, fine_tuning)
    blocked = {
        "unknown_agent_role",
        "unknown_tool_id",
        "sentinel_leak",
        "dpo_chosen_equals_rejected",
        "eval_missing_expected",
        "missing_required_args_executor_examples",
        "duplicate_sft_messages",
        "sft_split_overlap",
        "conflicting_sft_prompt_labels",
        "dataset_card_source_counts_mismatch",
        "dataset_card_task_counts_mismatch",
        "off_role_sft_source",
        "supplemental_sft_ratio_exceeded",
        "sft_sequence_budget_exceeded",
        "sft_source_split_missing",
        "executor_non_json_output",
        "executor_invalid_payload_tool",
        "executor_invalid_arguments",
        "executor_extra_arguments",
        "executor_invalid_enum_argument",
        "executor_invalid_argument_type",
        "executor_response_shape_invalid",
        "executor_dpo_missing_chosen_output",
        "cortex_duplicate_json_key",
        "cortex_route_contract_invalid",
        "cortex_dpo_duplicate_json_key",
        "cortex_dpo_route_contract_invalid",
    }
    failing_codes = {failure.code for failure in failures}
    assert blocked.isdisjoint(failing_codes), failures


def test_executor_has_tool_coverage(compiled_fine_tuning: tuple) -> None:
    manifest, _, fine_tuning = compiled_fine_tuning
    expected_tools = {tool.id for tool in manifest.tools}
    covered_tools: set[str] = set()
    for record in fine_tuning["executor"].train_sft + fine_tuning["executor"].val_sft:
        covered_tools.update(record["metadata"]["toolIDs"])
    assert expected_tools.issubset(covered_tools)


def test_executor_outputs_are_manifest_valid_json(compiled_fine_tuning: tuple) -> None:
    manifest, _, fine_tuning = compiled_fine_tuning
    tools = {tool.id: tool for tool in manifest.tools}
    for record in fine_tuning["executor"].train_sft + fine_tuning["executor"].val_sft:
        payload = json.loads(record["messages"][2]["content"])
        assert isinstance(payload, dict)
        assert set(payload) in ({"action"}, {"final"})
        if "final" in payload:
            assert isinstance(payload["final"], str) and payload["final"].strip()
            continue
        action = payload["action"]
        assert set(action) == {"tool", "args"}
        tool_id = action.get("tool")
        assert tool_id in tools
        arguments = action.get("args")
        assert isinstance(arguments, dict)
        contract = {argument.name: argument for argument in tools[tool_id].arguments}
        assert set(arguments).issubset(contract)
        assert {
            argument.name for argument in tools[tool_id].arguments if argument.required
        }.issubset(arguments)
        for name, value in arguments.items():
            allowed_values = contract[name].allowedValues
            if allowed_values:
                assert value in allowed_values
        if tool_id == "trigger.create" and "schedule" in arguments:
            assert arguments["schedule"] in {"absolute", "interval", "relative"}


def test_executor_chosen_dpo_outputs_are_manifest_valid_json(compiled_fine_tuning: tuple) -> None:
    manifest, _, fine_tuning = compiled_fine_tuning
    tools = {tool.id: tool for tool in manifest.tools}
    for record in fine_tuning["executor"].train_dpo + fine_tuning["executor"].val_dpo:
        payload = json.loads(record["chosen"]["content"])
        assert set(payload) in ({"action"}, {"final"})
        if "final" in payload:
            assert isinstance(payload["final"], str) and payload["final"].strip()
            continue
        action = payload["action"]
        assert set(action) == {"tool", "args"}
        tool = tools[action["tool"]]
        arguments = action.get("args")
        assert isinstance(arguments, dict)
        required = {argument.name for argument in tool.arguments if argument.required}
        assert required.issubset(arguments)
        for argument in tool.arguments:
            if argument.name in arguments and argument.allowedValues:
                assert arguments[argument.name] in argument.allowedValues


def test_executor_compiled_lanes_and_frozen_cases_bind_native_runtime_contract(
    compiled_fine_tuning: tuple,
) -> None:
    _, _, fine_tuning = compiled_fine_tuning
    executor = fine_tuning["executor"]
    sft = executor.train_sft + executor.val_sft
    dpo = executor.train_dpo + executor.val_dpo

    assert sft and dpo and executor.eval
    assert all(
        record["messages"][0]["content"] == EXECUTOR_RUNTIME_SYSTEM_PROMPT
        for record in sft
    )
    assert all(
        record["prompt"][0]["content"] == EXECUTOR_RUNTIME_SYSTEM_PROMPT
        for record in dpo
    )
    assert all(
        record["messages"][0]["content"] == EXECUTOR_RUNTIME_SYSTEM_PROMPT
        for record in executor.eval
    )
    assert STRUCTURED_OUTPUT_INSTRUCTION in EXECUTOR_RUNTIME_SYSTEM_PROMPT
    assert all(
        any(
            metric.get("type") == "executor_response_contract"
            for metric in record["metrics"]
        )
        for record in executor.eval
    )
    assert all(
        metric.get("type") != "approval_boundary"
        for record in executor.eval
        for metric in record["metrics"]
    )

    sft_task_types = {
        record["metadata"].get("taskType") for record in sft
    }
    dpo_preference_types = {
        record["metadata"].get("preferenceType") for record in dpo
    }
    eval_types = {
        record["metadata"].get("evalType") for record in executor.eval
    }
    assert "ultra_specific_post_observation_final" in sft_task_types
    assert "ultra_specific_post_observation_final" in dpo_preference_types
    assert "post_observation_final" in eval_types
    assert "ultra_specific_post_observation_final" in eval_types
    assert "approval_rejected" not in sft_task_types
    assert all(
        "approval_rejected"
        not in record["messages"][1]["content"].strip().casefold()
        for record in sft
    )

    forbidden_top_level = {
        "tool",
        "arguments",
        "status",
        "requiresApproval",
        "approvalPrompt",
        "permissionKey",
        "permissionKind",
        "confirmationMode",
        "missingArguments",
    }
    for payload in [
        *(json.loads(record["messages"][2]["content"]) for record in sft),
        *(json.loads(record["chosen"]["content"]) for record in dpo),
    ]:
        assert forbidden_top_level.isdisjoint(payload)
        assert set(payload) in ({"action"}, {"final"})
        if "action" in payload:
            assert set(payload["action"]) == {"tool", "args"}

    assert executor.contamination_report["contaminated"] is False
    assert executor.contamination_report["matchCount"] == 0


def test_executor_cancelled_approval_outcome_cannot_be_reframed_as_action(
    compiled_fine_tuning: tuple,
) -> None:
    manifest, _, fine_tuning = compiled_fine_tuning
    sft = fine_tuning["executor"].train_sft + fine_tuning["executor"].val_sft
    native_action = next(
        payload["action"]
        for payload in (
            json.loads(record["messages"][2]["content"])
            for record in sft
        )
        if "action" in payload
    )
    cancelled_legacy_payload = json.dumps(
        {
            "tool": native_action["tool"],
            "arguments": native_action["args"],
            "status": "cancelled_by_user",
        }
    )

    assert (
        _manifest_valid_executor_payload(manifest, cancelled_legacy_payload)
        is None
    )


def test_executor_native_final_curriculum_is_train_balanced_and_held_out(
    compiled_fine_tuning: tuple,
) -> None:
    _, _, fine_tuning = compiled_fine_tuning
    executor = fine_tuning["executor"]
    contract_case = "trusted_observation_no_tool_native_final"
    sft_by_split = {
        "train": [
            record
            for record in executor.train_sft
            if record["metadata"].get("contractCase") == contract_case
        ],
        "validation": [
            record
            for record in executor.val_sft
            if record["metadata"].get("contractCase") == contract_case
        ],
    }
    dpo_by_split = {
        "train": [
            record
            for record in executor.train_dpo
            if record["metadata"].get("contractCase") == contract_case
        ],
        "validation": [
            record
            for record in executor.val_dpo
            if record["metadata"].get("contractCase") == contract_case
        ],
    }

    assert len(sft_by_split["train"]) >= 8
    assert len(dpo_by_split["train"]) >= 8
    assert len(sft_by_split["validation"]) >= 2
    assert len(dpo_by_split["validation"]) >= 2

    for split in ("train", "validation"):
        sft_ids = {
            record["metadata"]["scenarioID"]
            for record in sft_by_split[split]
        }
        dpo_ids = {
            record["metadata"]["scenarioID"]
            for record in dpo_by_split[split]
        }
        assert sft_ids == dpo_ids
        assert all(
            record["metadata"]["requiredSplit"] == split
            for record in (*sft_by_split[split], *dpo_by_split[split])
        )
        assert all(
            set(json.loads(record["messages"][2]["content"])) == {"final"}
            for record in sft_by_split[split]
        )
        assert all(
            set(json.loads(record["chosen"]["content"])) == {"final"}
            for record in dpo_by_split[split]
        )

    train_ids = {
        record["metadata"]["scenarioID"]
        for record in sft_by_split["train"]
    }
    validation_ids = {
        record["metadata"]["scenarioID"]
        for record in sft_by_split["validation"]
    }
    assert train_ids.isdisjoint(validation_ids)
    train_finals = {
        json.loads(record["messages"][2]["content"])["final"]
        for record in sft_by_split["train"]
    }
    validation_finals = {
        json.loads(record["messages"][2]["content"])["final"]
        for record in sft_by_split["validation"]
    }
    assert train_finals.isdisjoint(validation_finals)

    action_count = sum(
        "action" in json.loads(record["messages"][2]["content"])
        for record in executor.train_sft
    )
    assert len(sft_by_split["train"]) * 50 >= action_count


def test_executor_training_prompt_exactly_matches_shipped_runtime_core() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    source = (
        repo_root
        / "ios"
        / "Lumen"
        / "Assistant"
        / "StructuredAgentKernelExecutor.swift"
    ).read_text(encoding="utf-8")
    match = re.search(
        r'executorRuntimeSystemPromptContract = """\n(.*?)\n\s*"""',
        source,
        flags=re.DOTALL,
    )

    assert match is not None
    swift_runtime_core = "\n".join(
        line.removeprefix("    ") for line in match.group(1).splitlines()
    )
    assert swift_runtime_core == EXECUTOR_RUNTIME_SYSTEM_PROMPT


def test_role_locked_training_prompts_match_swift_runtime_contract_bytes() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    source = (
        repo_root / "ios" / "Lumen" / "Services" / "ModelAdapterRuntimeContract.swift"
    ).read_text(encoding="utf-8")

    for role in ("mouth", "mimicry", "rem"):
        match = re.search(
            rf'roleID: "{role}"[^\n]+systemPrompt: "([^"]+)"',
            source,
        )
        assert match is not None, role
        assert match.group(1) == SYSTEM_PROMPTS[role]


def test_executor_frozen_expected_targets_pass_exact_shape_scorer(
    compiled_fine_tuning: tuple,
) -> None:
    manifest, _, fine_tuning = compiled_fine_tuning
    executor = fine_tuning["executor"]
    tool_contracts = {
        tool.id: {
            "arguments": [argument.model_dump() for argument in tool.arguments]
        }
        for tool in manifest.tools
    }

    for record in executor.eval:
        expected = record.get("expected") or {}
        if isinstance(expected.get("tool"), str):
            candidate = {
                "action": {
                    "tool": expected["tool"],
                    "args": dict(expected.get("arguments") or {}),
                }
            }
        elif isinstance(expected.get("final"), str):
            candidate = {"final": expected["final"]}
        elif expected.get("format") == "strict_json":
            candidate = {"final": "The structured turn is complete."}
        else:
            pytest.fail(
                f"Executor frozen case lacks an action/final target: {record['metadata']}"
            )

        results = [
            _score_metric(
                metric,
                candidate,
                tool_contracts=tool_contracts,
                allowed_slots=set(),
                has_output=True,
            )
            for metric in record["metrics"]
        ]
        assert all(result["passed"] for result in results), (
            record["metadata"],
            candidate,
            results,
        )


def test_mouth_training_lanes_are_complete_and_include_adversarial_repairs(
    compiled_fine_tuning: tuple,
) -> None:
    manifest, _, fine_tuning = compiled_fine_tuning
    mouth = fine_tuning["mouth"]
    sft = mouth.train_sft + mouth.val_sft
    dpo = mouth.train_dpo + mouth.val_dpo

    assert sft and dpo
    assert all(
        mouth_final_text_is_complete(record["messages"][2]["content"])
        for record in sft
    )
    assert all(
        mouth_final_text_is_complete(record["chosen"]["content"])
        for record in dpo
    )
    assert {
        "ultra_specific_adversarial_truncation_repair",
        "ultra_specific_adversarial_observation_repair",
        "ultra_specific_adversarial_failure_repair",
        "ultra_specific_adversarial_attachment_repair",
    }.issubset({record["metadata"].get("taskType") for record in sft})
    assert {
        "grounded_observation_truncated_upload",
        "grounded_observation_exact_attachments",
        "grounded_observation_failure_polarity",
        "grounded_observation_motion_negation",
    }.issubset({record["metadata"].get("preferenceType") for record in dpo})
    assert any(
        record["rejected"]["content"].endswith(" with")
        for record in dpo
    )

    failures = validate_agent_fine_tuning_datasets(manifest, fine_tuning)
    assert {
        "mouth_incomplete_sft_output",
        "mouth_incomplete_dpo_chosen_output",
        "mouth_eval_missing_completeness_metric",
    }.isdisjoint({failure.code for failure in failures})
    assert mouth.contamination_report["contaminated"] is False
    assert mouth.contamination_report["matchCount"] == 0


def test_mouth_frozen_bank_is_contentful_complete_and_self_consistent(
    compiled_fine_tuning: tuple,
) -> None:
    _, _, fine_tuning = compiled_fine_tuning
    mouth = fine_tuning["mouth"]
    candidates = {
        "sentinel_suppression": "Québec City has light rain at 18 C.",
        "no_internal_json": (
            "I found budget.pdf in Downloads; it was modified yesterday."
        ),
        "truthful_failure_summary": (
            "I could not read calendar events because calendar permission was denied."
        ),
        "concise_response": "Your Buy filters reminder is due Friday.",
        "ultra_specific_calendar_safe_output": (
            "You have Solstice audit at 13:40 and Orchid review at 17:20."
        ),
        "ultra_specific_outlook_attachment_summary": (
            "The available attachments are invoice-4821.pdf and quote.xlsx."
        ),
        "ultra_specific_motion_summary": (
            "Your activity is walking with high confidence."
        ),
    }
    by_type = {
        record["metadata"].get("evalType"): record
        for record in mouth.eval
    }

    assert set(by_type) == set(candidates)
    for eval_type, record in by_type.items():
        assert record["outputMode"] == "text"
        assert any(
            metric.get("type") == "complete_final_text"
            for metric in record["metrics"]
        )
        assert any(
            metric.get("type") == "observation_entailment"
            for metric in record["metrics"]
        )
        results = [
            _score_metric(
                metric,
                candidates[eval_type],
                tool_contracts={},
                allowed_slots=set(),
                has_output=True,
            )
            for metric in record["metrics"]
        ]
        assert all(result["passed"] for result in results), (
            eval_type,
            record["metrics"],
            results,
        )

        generic = score_evaluation_suite(
            [record],
            {record["evalID"]: "Done."},
            agent="mouth",
        )
        truncated = score_evaluation_suite(
            [record],
            {record["evalID"]: candidates[eval_type] + " with"},
            agent="mouth",
        )
        assert generic["weightedScore"] == 0.0, eval_type
        assert truncated["weightedScore"] == 0.0, eval_type


def test_mouth_validator_rejects_incomplete_preferred_training_outputs(
    compiled_fine_tuning: tuple,
) -> None:
    manifest, _, fine_tuning = compiled_fine_tuning
    mouth = fine_tuning["mouth"]
    train_sft = copy.deepcopy(mouth.train_sft)
    train_dpo = copy.deepcopy(mouth.train_dpo)
    train_sft[0]["messages"][2]["content"] = "The"
    train_dpo[0]["chosen"]["content"] = "You do not need an"
    mutated = dict(fine_tuning)
    mutated["mouth"] = replace(
        mouth,
        train_sft=train_sft,
        train_dpo=train_dpo,
    )

    failures = validate_agent_fine_tuning_datasets(manifest, mutated)
    codes = {failure.code for failure in failures}

    assert "mouth_incomplete_sft_output" in codes
    assert "mouth_incomplete_dpo_chosen_output" in codes


def test_validator_rejects_invalid_executor_enum(compiled_fine_tuning: tuple) -> None:
    manifest, _, fine_tuning = compiled_fine_tuning
    executor = fine_tuning["executor"]
    records = copy.deepcopy(executor.train_sft)
    index = next(
        index
        for index, record in enumerate(records)
        if "trigger.create" in record["metadata"]["toolIDs"]
        and "schedule"
        in json.loads(record["messages"][2]["content"])
        .get("action", {})
        .get("args", {})
    )
    payload = json.loads(records[index]["messages"][2]["content"])
    payload["action"]["args"]["schedule"] = "sample_schedule"
    records[index]["messages"][2]["content"] = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    mutated = dict(fine_tuning)
    mutated["executor"] = replace(executor, train_sft=records)

    failures = validate_agent_fine_tuning_datasets(manifest, mutated)

    assert "executor_invalid_enum_argument" in {failure.code for failure in failures}


def test_rem_does_not_fabricate_runtime_repair_when_none_are_trainable(compiled_fine_tuning: tuple) -> None:
    _, datasets, fine_tuning = compiled_fine_tuning
    if datasets.get("runtime_audit_repairs"):
        pytest.skip("fixture contains trainable runtime repair records")
    records = fine_tuning["rem"].train_sft + fine_tuning["rem"].val_sft
    assert not any(record["metadata"]["sourceFamily"] == "runtime_audit_repairs" for record in records)
    assert "runtime_failure_detected" not in json.dumps(records, ensure_ascii=False, sort_keys=True)


def test_fleet_has_model_slot_coverage(compiled_fine_tuning: tuple) -> None:
    manifest, _, fine_tuning = compiled_fine_tuning
    blob = "\n".join(json.dumps(record, ensure_ascii=False, sort_keys=True) for record in (fine_tuning["fleet"].train_sft + fine_tuning["fleet"].val_sft))
    for slot in manifest.fleet.slots:
        assert slot.id in blob


def test_fleet_complete_training_corpus_is_strict_json_only(
    compiled_fine_tuning: tuple,
) -> None:
    _, _, fine_tuning = compiled_fine_tuning
    fleet = fine_tuning["fleet"]
    sft = fleet.train_sft + fleet.val_sft
    dpo = fleet.train_dpo + fleet.val_dpo

    assert sft and dpo
    assert not any(
        record["metadata"].get("sourceFamily") == "fleet_system_prompts"
        for record in sft
    )
    for record in sft:
        assert STRUCTURED_OUTPUT_INSTRUCTION in record["messages"][0]["content"]
        payload = _strict_json_loads(record["messages"][2]["content"])
        assert isinstance(payload, dict)
    for record in dpo:
        assert STRUCTURED_OUTPUT_INSTRUCTION in record["prompt"][0]["content"]
        is_native_orchestration = (
            record["metadata"].get("sourceFamily")
            == FLEET_NATIVE_ORCHESTRATION_SOURCE_FAMILY
        )
        for completion in ("chosen", "rejected"):
            content = record[completion]["content"]
            payload = _strict_json_loads(content)
            assert isinstance(payload, dict)
            assert content == json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=not is_native_orchestration,
                separators=(",", ":"),
            )
            if is_native_orchestration:
                assert list(payload) == [
                    "graphSchemaVersion",
                    "scenarioID",
                    "knownSlotIDs",
                    "events",
                    "dependencies",
                    "decision",
                ]

    private_state_pairs = [
        record
        for record in dpo
        if record["metadata"].get("taskType")
        == "fleet_private_state_boundary"
    ]
    for record in private_state_pairs:
        chosen = _strict_json_loads(record["chosen"]["content"])
        rejected = _strict_json_loads(record["rejected"]["content"])
        assert chosen["privateStateAccessible"] is False
        assert rejected["privateStateAccessible"] is True
        assert chosen["requestedSlot"] in chosen["knownSlots"]
        assert rejected["requestedSlot"] == chosen["requestedSlot"]


def test_fleet_native_dpo_mutations_survive_canonical_binding(
    compiled_fine_tuning: tuple,
) -> None:
    manifest, _, _ = compiled_fine_tuning
    source_records = [
        record
        for record in generate_fleet_artifacts(manifest).cross_model_training
        if record.get("recordType") == "dpo"
        and record.get("sourceFamily")
        == FLEET_NATIVE_ORCHESTRATION_SOURCE_FAMILY
        and record["metadata"].get("trainingMatrixVariant")
        == "behavior-conditioned"
    ]
    records = _bind_fleet_dpo_contract(manifest, source_records)

    assert len(records) == len(source_records)
    assert records
    assert {
        record["metadata"]["atomicPreferenceMutation"]
        for record in records
    } == set(FLEET_NATIVE_ORCHESTRATION_MUTATION_CONTRACT)
    for record in records:
        event_id_fact = _fleet_native_prompt_event_id_fact(
            _first_role_content(record["prompt"], "user")
        )
        assert event_id_fact is not None
        for completion in ("chosen", "rejected"):
            content = record[completion]["content"]
            payload = _strict_json_loads(content)
            assert content == json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=False,
                separators=(",", ":"),
            )
        chosen = _strict_json_loads(record["chosen"]["content"])
        rejected = _strict_json_loads(record["rejected"]["content"])
        mutation = record["metadata"]["atomicPreferenceMutation"]
        assert _valid_fleet_native_preference_mutation(
            chosen,
            rejected,
            mutation,
            event_id_fact=event_id_fact,
        )
        tampered_rejected = json.loads(json.dumps(rejected, ensure_ascii=False))
        tampered_rejected["scenarioID"] = f"{rejected['scenarioID']}--tampered"
        assert not _valid_fleet_native_preference_mutation(
            chosen,
            tampered_rejected,
            mutation,
            event_id_fact=event_id_fact,
        )


def test_fleet_private_state_preference_is_reframed_as_json(
    compiled_fine_tuning: tuple,
) -> None:
    manifest, _, _ = compiled_fine_tuning
    source = {
        "prompt": [
            {"role": "system", "content": "Legacy Fleet prompt."},
            {
                "role": "user",
                "content": "What is inside embedding's current TTL cache?",
            },
        ],
        "chosen": {
            "role": "assistant",
            "content": "I cannot inspect embedding's private cache.",
        },
        "rejected": {
            "role": "assistant",
            "content": "The cache contains fabricated_internal_state.",
        },
        "metadata": {
            "agent": "fleet",
            "preferenceType": "fleet_private_state_boundary",
            "taskType": "fleet_private_state_boundary",
        },
    }

    bound = _bind_fleet_dpo_contract(manifest, [source])

    assert len(bound) == 1
    assert STRUCTURED_OUTPUT_INSTRUCTION in bound[0]["prompt"][0]["content"]
    chosen = _strict_json_loads(bound[0]["chosen"]["content"])
    rejected = _strict_json_loads(bound[0]["rejected"]["content"])
    assert chosen == {
        "knownSlots": [slot.id for slot in manifest.fleet.slots],
        "privateStateAccessible": False,
        "reason": "private runtime state is not exposed by the fleet manifest",
        "requestedSlot": "embedding",
    }
    assert rejected["privateStateAccessible"] is True
    assert rejected["claimedState"] == "fabricated_internal_state"


@pytest.mark.parametrize(
    ("completion", "invalid_content"),
    (
        ("chosen", '["executor"]'),
        (
            "rejected",
            '{"knownSlots":["fleet"],"knownSlots":["executor"]}',
        ),
        ("rejected", '["executor"]'),
        ("rejected", '{"delegateTo":NaN}'),
    ),
)
def test_fleet_dpo_contract_rejects_non_strict_completion_json(
    compiled_fine_tuning: tuple,
    completion: str,
    invalid_content: str,
) -> None:
    manifest, _, _ = compiled_fine_tuning
    source = {
        "prompt": [
            {"role": "system", "content": "Legacy Fleet prompt."},
            {"role": "user", "content": "Select the manifested owner."},
        ],
        "chosen": {
            "role": "assistant",
            "content": '{ "knownSlots": ["fleet", "executor"] }',
        },
        "rejected": {
            "role": "assistant",
            "content": '{"knownSlots":["fleet"]}',
        },
        "metadata": {
            "agent": "fleet",
            "preferenceType": "fleet_contract_known_slots",
            "taskType": "fleet_contract_known_slots",
        },
    }
    source[completion]["content"] = invalid_content

    assert _bind_fleet_dpo_contract(manifest, [source]) == []


def test_fleet_dpo_contract_rejects_zero_signal_after_canonicalization(
    compiled_fine_tuning: tuple,
) -> None:
    manifest, _, _ = compiled_fine_tuning
    source = {
        "prompt": [
            {"role": "system", "content": "Legacy Fleet prompt."},
            {"role": "user", "content": "Select the manifested owner."},
        ],
        "chosen": {
            "role": "assistant",
            "content": '{ "owner": "executor", "known": true }',
        },
        "rejected": {
            "role": "assistant",
            "content": '{"known":true,"owner":"executor"}',
        },
        "metadata": {
            "agent": "fleet",
            "preferenceType": "fleet_contract_known_slots",
            "taskType": "fleet_contract_known_slots",
        },
    }

    assert _bind_fleet_dpo_contract(manifest, [source]) == []


def test_sparse_fleet_directory_preference_remains_semantic() -> None:
    manifest = AgentBehaviorManifest()
    directory_pair = next(
        record
        for record in _synthetic_dpo_pairs(manifest, set())["fleet"]
        if record["metadata"]["preferenceType"] == "slot_id_directory"
    )

    bound = _bind_fleet_dpo_contract(manifest, [directory_pair])

    assert len(bound) == 1
    chosen = _strict_json_loads(bound[0]["chosen"]["content"])
    rejected = _strict_json_loads(bound[0]["rejected"]["content"])
    assert chosen == {"knownSlots": []}
    assert rejected == {"knownSlots": ["invented_shadow_slot"]}
    assert chosen != rejected


def test_fleet_role_prompt_artifacts_are_not_response_targets(
    compiled_fine_tuning: tuple,
) -> None:
    manifest, _, _ = compiled_fine_tuning
    stale = {
        "messages": [
            {"role": "system", "content": "Legacy Fleet prompt."},
            {"role": "user", "content": "Summarize slot executor."},
            {
                "role": "assistant",
                "content": "You are Executor. Emit strict tool JSON only.",
            },
        ],
        "metadata": {
            "agent": "fleet",
            "sourceFamily": "fleet_system_prompts",
            "taskType": "role_directory",
        },
    }

    assert _bind_fleet_sft_contract(manifest, [stale]) == []


def test_fleet_supplemental_targets_are_bounded_by_loss_share(
    compiled_fine_tuning: tuple,
) -> None:
    _, _, fine_tuning = compiled_fine_tuning
    fleet = fine_tuning["fleet"]
    records = fleet.train_sft + fleet.val_sft
    constraints = fleet.dataset_card["constraints"]
    quality = fleet.dataset_card["quality"]
    total_chars = 0
    supplemental_chars = 0
    total_tokens = 0
    supplemental_tokens = 0

    for record in records:
        assistant = record["messages"][2]["content"]
        assistant_chars = len(assistant)
        assistant_tokens = _source_token_proxy_count(assistant)
        total_chars += assistant_chars
        total_tokens += assistant_tokens
        if _fleet_source_role(record) == FLEET_SOURCE_ROLE_SUPPLEMENTAL_STATIC:
            supplemental_chars += assistant_chars
            supplemental_tokens += assistant_tokens

    char_share = supplemental_chars / total_chars
    token_share = supplemental_tokens / total_tokens
    configured_char_cap = constraints[
        "maxFleetSupplementalAssistantCharShare"
    ]
    configured_token_cap = constraints[
        "maxFleetSupplementalAssistantTokenShare"
    ]
    proxy_selection_token_cap = constraints[
        "maxFleetSupplementalAssistantTokenProxySelectionShare"
    ]
    assert constraints["fleetLossShareEnforcementScope"] == (
        "optimizer_train_only"
    )
    assert constraints["fleetValidationLossSharePolicy"] == (
        "observed_not_enforced"
    )
    assert configured_char_cap == pytest.approx(0.25)
    assert configured_token_cap == pytest.approx(0.25)
    assert proxy_selection_token_cap == pytest.approx(0.15)
    assert configured_char_cap <= FLEET_SUPPLEMENTAL_ASSISTANT_SHARE_HARD_MAX
    assert configured_token_cap <= FLEET_SUPPLEMENTAL_ASSISTANT_SHARE_HARD_MAX
    assert quality["assistantTargetCharCount"] == total_chars
    assert quality["supplementalAssistantTargetCharCount"] == supplemental_chars
    assert quality["supplementalAssistantTargetCharShare"] == pytest.approx(
        char_share
    )
    assert quality["assistantTargetTokenProxyCount"] == total_tokens
    assert quality["supplementalAssistantTargetTokenProxyCount"] == supplemental_tokens
    assert quality["supplementalAssistantTargetTokenProxyShare"] == pytest.approx(
        token_share
    )
    assert quality["sourceTokenProxyContract"] == {
        "schemaVersion": "lumen.source-token-proxy/1.0.0",
        "status": "source_side_selection_proxy_not_exact_token_count",
        "strategy": "max_whitespace_terms_utf8_byte_ceiling",
        "maxCharsPerToken": 4,
        "exactPinnedTokenizerAuthoritative": True,
        "authoritativeEnforcementPhase": "post_tokenizer_load_pre_optimizer",
    }
    assert quality["fleetSourceRoleSFTRecordCounts"] == {
        category: sum(
            _fleet_source_role(record) == category for record in records
        )
        for category in (
            FLEET_SOURCE_ROLE_BEHAVIORAL_PRIMARY,
            FLEET_SOURCE_ROLE_PUBLIC_BEHAVIORAL,
            FLEET_SOURCE_ROLE_SUPPLEMENTAL_STATIC,
        )
    }
    registry = constraints["fleetSourceRoleRegistry"]
    assert registry["unknownPairs"] == "hard_fail"
    assert registry["categories"] == [
        FLEET_SOURCE_ROLE_BEHAVIORAL_PRIMARY,
        FLEET_SOURCE_ROLE_PUBLIC_BEHAVIORAL,
        FLEET_SOURCE_ROLE_SUPPLEMENTAL_STATIC,
    ]
    loss_share_contract = constraints["fleetLossShareContract"]
    assert loss_share_contract == fleet.unsloth_config["fleetLossShareContract"]
    assert loss_share_contract["schemaVersion"] == (
        FLEET_LOSS_SHARE_CONTRACT_SCHEMA_VERSION
    )
    assert loss_share_contract["enforcementRequired"] is True
    assert loss_share_contract["enforcementPhase"] == (
        "post_tokenizer_load_pre_optimizer"
    )
    assert loss_share_contract["requiredLanes"] == ["sft", "dpo"]
    assert loss_share_contract["authoritativeCapEncoding"] == (
        "integer_basis_points"
    )
    assert loss_share_contract["basisPointDenominator"] == (
        FLEET_LOSS_SHARE_BASIS_POINTS_DENOMINATOR
    )
    assert loss_share_contract["capsBasisPoints"] == {
        "supplementalStaticTotal": {
            "requested": 2_500,
            "hard": 3_000,
        },
        "publicBehavioralTotal": {
            "requested": 3_500,
            "hard": 3_500,
        },
        "eachSupplementalSourceFamily": {
            "hard": 1_000,
        },
    }
    evidence_contract = loss_share_contract["exactTokenEvidenceContract"]
    assert evidence_contract["required"] is True
    assert evidence_contract["schemaVersion"] == (
        FLEET_LOSS_SHARE_EVIDENCE_SCHEMA_VERSION
    )
    assert evidence_contract["statusAtGeneration"] == (
        "pending_exact_tokenizer_preflight"
    )
    assert evidence_contract["tokenizer"] == "pinned_qwen_tokenizer"
    assert evidence_contract["comparisonRule"] == (
        "numeratorTokenCount*basisPointDenominator<="
        "denominatorTokenCount*capBasisPoints"
    )
    assert set(evidence_contract["lanes"]) == {"sft", "dpo"}
    assert loss_share_contract["sftOptimizerWindowScheduleContract"] == {
        "schemaVersion": (
            FLEET_SFT_OPTIMIZER_WINDOW_SCHEDULE_CONTRACT_SCHEMA_VERSION
        ),
        "evidenceSchemaVersion": (
            FLEET_SFT_OPTIMIZER_WINDOW_SCHEDULE_EVIDENCE_SCHEMA_VERSION
        ),
        "lane": "sft",
        "split": "train",
        "enforcementRequired": True,
        "enforcementPhase": "post_tokenizer_load_pre_optimizer",
        "basis": (
            "mean_of_floor_per_optimizer_window_native_assistant_target_"
            "token_share_basis_points"
        ),
        "basisPointDenominator": FLEET_LOSS_SHARE_BASIS_POINTS_DENOMINATOR,
        "minimumBasisPoints": (
            FLEET_NATIVE_ORCHESTRATION_SFT_SHARE_MIN_BASIS_POINTS
        ),
        "maximumBasisPoints": (
            FLEET_NATIVE_ORCHESTRATION_SFT_SHARE_MAX_BASIS_POINTS
        ),
        "algorithm": FLEET_SFT_OPTIMIZER_WINDOW_SCHEDULE_ALGORITHM,
        "candidateSearchCount": FLEET_SFT_OPTIMIZER_WINDOW_CANDIDATE_COUNT,
        "permutationPolicy": "each_source_row_exactly_once_per_epoch",
        "packing": False,
        "distributedSamplingPolicy": "single_process_only",
        "resumePolicy": (
            "trainer_set_epoch_with_monotonic_sampler_guard_through_"
            "skip_first_batches"
        ),
        "failurePolicy": "abort_before_optimizer",
    }
    assert loss_share_contract["sftOptimizerRecordGeometryContract"] == {
        "schemaVersion": (
            FLEET_SFT_OPTIMIZER_RECORD_GEOMETRY_SCHEMA_VERSION
        ),
        "lane": "sft",
        "maximumTrainRecords": FLEET_SFT_OPTIMIZER_MAX_TRAIN_RECORDS,
        "protectedSourceRole": FLEET_SOURCE_ROLE_BEHAVIORAL_PRIMARY,
        "removableSourceRoles": [
            FLEET_SOURCE_ROLE_SUPPLEMENTAL_STATIC,
            FLEET_SOURCE_ROLE_PUBLIC_BEHAVIORAL,
        ],
        "selectionPolicy": (
            "largest_deterministic_cap_valid_cohort_from_immutable_"
            "candidates"
        ),
        "failurePolicy": "abort_generation_before_optimizer",
    }
    assert loss_share_contract["dpoTokenizationPolicy"] == {
        "trainerImplementation": "trl.DPOTrainer.tokenize_row",
        "trlVersion": "0.24.0",
        "completionTokenization": "add_special_tokens_false",
        "completionSuffix": "append_tokenizer_eos_token_id",
        "appendedEOSTokensPerCompletion": 1,
    }
    assert loss_share_contract["optimizerFamilyShareBands"] == {
        "schemaVersion": FLEET_OPTIMIZER_FAMILY_SHARE_SCHEMA_VERSION,
        "enforcementScope": "optimizer_train_only",
        "classification": {
            "sourceFamily": FLEET_NATIVE_ORCHESTRATION_SOURCE_FAMILY,
            "taskTypeByLane": {
                "sft": FLEET_NATIVE_ORCHESTRATION_SFT_TASK_TYPE,
                "dpo": FLEET_NATIVE_ORCHESTRATION_DPO_TASK_TYPE,
            },
        },
        "policySignalTokenClassificationByLane": {
            "sft": [
                {
                    "sourceFamily": FLEET_NATIVE_ORCHESTRATION_SOURCE_FAMILY,
                    "taskType": FLEET_NATIVE_ORCHESTRATION_SFT_TASK_TYPE,
                },
                {
                    "sourceFamily": ULTRA_SPECIFIC_SOURCE_FAMILY,
                    "taskType": FLEET_POLICY_VOCABULARY_SFT_TASK_TYPE,
                },
            ],
            "dpo": [
                {
                    "sourceFamily": FLEET_NATIVE_ORCHESTRATION_SOURCE_FAMILY,
                    "taskType": FLEET_NATIVE_ORCHESTRATION_DPO_TASK_TYPE,
                }
            ],
        },
        "lanes": {
            "sft": {
                "basis": "assistant_mask_non_ignored_token_count",
                "numeratorEvidenceField": (
                    "nativeOrchestrationAssistantTargetTokenCount"
                ),
                "denominatorEvidenceField": "assistantTargetTokenCount",
                "minimumBasisPoints": (
                    FLEET_NATIVE_ORCHESTRATION_SFT_SHARE_MIN_BASIS_POINTS
                ),
                "maximumBasisPoints": (
                    FLEET_NATIVE_ORCHESTRATION_SFT_SHARE_MAX_BASIS_POINTS
                ),
            },
            "dpo": {
                "basis": "preference_pair_count",
                "numeratorEvidenceField": (
                    "nativeOrchestrationPreferencePairCount"
                ),
                "denominatorEvidenceField": "preferencePairCount",
                "minimumBasisPoints": (
                    FLEET_NATIVE_ORCHESTRATION_DPO_SHARE_MIN_BASIS_POINTS
                ),
                "maximumBasisPoints": (
                    FLEET_NATIVE_ORCHESTRATION_DPO_SHARE_MAX_BASIS_POINTS
                ),
            },
        },
        "policySignalTokenLanes": {
            "sft": {
                "basis": "assistant_mask_non_ignored_token_count",
                "numeratorEvidenceField": (
                    "allPolicyAssistantTargetTokenCount"
                ),
                "denominatorEvidenceField": "assistantTargetTokenCount",
                "minimumBasisPoints": (
                    FLEET_ALL_POLICY_SFT_TOKEN_SHARE_MIN_BASIS_POINTS
                ),
                "maximumBasisPoints": (
                    FLEET_ALL_POLICY_SFT_TOKEN_SHARE_MAX_BASIS_POINTS
                ),
            },
            "dpo": {
                "basis": "rendered_chosen_completion_token_count",
                "numeratorEvidenceField": (
                    "nativeOrchestrationChosenTargetTokenCount"
                ),
                "denominatorEvidenceField": "chosenTargetTokenCount",
                "minimumBasisPoints": (
                    FLEET_NATIVE_ORCHESTRATION_DPO_TOKEN_SHARE_MIN_BASIS_POINTS
                ),
                "maximumBasisPoints": (
                    FLEET_NATIVE_ORCHESTRATION_DPO_TOKEN_SHARE_MAX_BASIS_POINTS
                ),
            },
        },
        "comparisonRules": {
            "minimum": (
                "numeratorCount*basisPointDenominator>="
                "denominatorCount*minimumBasisPoints"
            ),
            "maximum": (
                "numeratorCount*basisPointDenominator<="
                "denominatorCount*maximumBasisPoints"
            ),
        },
        "failurePolicy": "abort_before_optimizer",
    }
    for lane in evidence_contract["lanes"].values():
        assert set(lane) == {
            "denominatorTokenCount",
            "supplementalNumeratorTokenCount",
            "publicNumeratorTokenCount",
            "policySignalNumeratorTokenCount",
            "perSourceFamilyNumeratorTokenCounts",
        }
    source_proxy = loss_share_contract["sourceSelectionProxy"]
    assert source_proxy["status"] == "safety_budget_not_exact_token_count"
    assert source_proxy["maximumSupplementalStaticShareBasisPoints"] == 1_500
    assert source_proxy["maximumPublicBehavioralShareBasisPoints"] == 3_000
    assert source_proxy["optimizerFamilySafetyBand"] == {
        "schemaVersion": FLEET_OPTIMIZER_FAMILY_SOURCE_PROXY_SCHEMA_VERSION,
        "lane": "sft",
        "basis": "assistant_target_source_token_proxy_count",
        "sourceFamily": FLEET_NATIVE_ORCHESTRATION_SOURCE_FAMILY,
        "taskType": FLEET_NATIVE_ORCHESTRATION_SFT_TASK_TYPE,
        "minimumBasisPoints": (
            FLEET_NATIVE_ORCHESTRATION_SFT_PROXY_SHARE_MIN_BASIS_POINTS
        ),
        "maximumBasisPoints": (
            FLEET_NATIVE_ORCHESTRATION_SFT_PROXY_SHARE_MAX_BASIS_POINTS
        ),
        "selectionPolicy": (
            "retain_non_public_then_bound_public_behavioral"
        ),
        "authoritativeExactBandBasisPoints": {
            "minimum": FLEET_NATIVE_ORCHESTRATION_SFT_SHARE_MIN_BASIS_POINTS,
            "maximum": FLEET_NATIVE_ORCHESTRATION_SFT_SHARE_MAX_BASIS_POINTS,
        },
    }
    assert source_proxy["contract"] == quality["sourceTokenProxyContract"]
    assert loss_share_contract["failurePolicy"] == "abort_before_optimizer"
    assert loss_share_contract["rowMetadataContract"] == {
        "requiredCanonicalFields": ["sourceFamily", "taskType"],
        "missingOrUnknown": "hard_fail",
    }
    assert loss_share_contract["sourceRoleRegistry"] == registry
    assert loss_share_contract["tokenAccounting"] == {
        "sft": "assistant_mask_non_ignored_token_count",
        "dpo": (
            "rendered_chosen_completion_tokens_add_special_tokens_false_"
            "plus_one_trl_0_24_0_appended_eos"
        ),
    }
    assert loss_share_contract["tokenizer"] == {
        "baseModelID": fleet.unsloth_config["baseModelID"],
        "baseModelRevision": fleet.unsloth_config["baseModelRevision"],
        "tokenizerSHA256": fleet.unsloth_config["baseModelTokenizerDigest"],
        "tokenizerClosureSHA256": fleet.unsloth_config[
            "baseModelTokenizerClosureSHA256"
        ],
    }
    assert all(
        "fleetLossShareContract" not in fine_tuning[agent].unsloth_config
        for agent in AGENTS
        if agent != "fleet"
    )

    def assert_optimizer_proxy_caps(
        lane_records: list[dict],
        *,
        lane: str,
    ) -> None:
        def target_proxy(record: dict) -> int:
            if lane == "sft":
                return _source_token_proxy_count(
                    record["messages"][2]["content"]
                )
            return _source_token_proxy_count(record["chosen"]["content"])

        def target_chars(record: dict) -> int:
            if lane == "sft":
                return len(record["messages"][2]["content"])
            return len(record["chosen"]["content"])

        total = sum(target_proxy(record) for record in lane_records)
        supplemental = sum(
            target_proxy(record)
            for record in lane_records
            if _fleet_source_role(record)
            == FLEET_SOURCE_ROLE_SUPPLEMENTAL_STATIC
        )
        public = sum(
            target_proxy(record)
            for record in lane_records
            if _fleet_source_role(record) == FLEET_SOURCE_ROLE_PUBLIC_BEHAVIORAL
        )
        supplemental_by_family: dict[str, int] = {}
        total_chars = sum(target_chars(record) for record in lane_records)
        supplemental_chars = sum(
            target_chars(record)
            for record in lane_records
            if _fleet_source_role(record)
            == FLEET_SOURCE_ROLE_SUPPLEMENTAL_STATIC
        )
        for record in lane_records:
            if (
                _fleet_source_role(record)
                != FLEET_SOURCE_ROLE_SUPPLEMENTAL_STATIC
            ):
                continue
            source_family = record["metadata"]["sourceFamily"]
            supplemental_by_family[source_family] = (
                supplemental_by_family.get(source_family, 0)
                + target_proxy(record)
            )

        assert total > 0
        assert supplemental_chars * 10_000 <= total_chars * 2_500
        assert supplemental * 10_000 <= total * 1_500
        assert public * 10_000 <= total * 3_000
        assert all(
            family_tokens * 10_000 <= total * 500
            for family_tokens in supplemental_by_family.values()
        )

    assert_optimizer_proxy_caps(fleet.train_sft, lane="sft")
    assert_optimizer_proxy_caps(fleet.train_dpo, lane="dpo")
    optimizer_sft_lanes = [
        fleet.train_sft,
        *(
            variant["train_sft"]
            for variant in fleet.experiment_variants.values()
        ),
    ]
    available_cross_model_tasks = {
        record["metadata"]["taskType"]
        for record in records
        if record["metadata"]["sourceFamily"] == "cross_model_training"
    }
    if available_cross_model_tasks:
        assert (
            FLEET_REQUIRED_SUPPLEMENTAL_SFT_TASK_TYPES
            <= available_cross_model_tasks
        )
        for optimizer_sft in optimizer_sft_lanes:
            assert FLEET_REQUIRED_SUPPLEMENTAL_SFT_TASK_TYPES <= {
                record["metadata"]["taskType"]
                for record in optimizer_sft
                if record["metadata"]["sourceFamily"]
                == "cross_model_training"
            }
    for variant in fleet.experiment_variants.values():
        assert_optimizer_proxy_caps(variant["train_sft"], lane="sft")
        assert_optimizer_proxy_caps(variant["train_dpo"], lane="dpo")

    for record in [*records, *fleet.train_dpo, *fleet.val_dpo]:
        metadata = record.get("metadata")
        assert isinstance(metadata, dict)
        assert isinstance(metadata.get("sourceFamily"), str)
        assert metadata["sourceFamily"].strip() == metadata["sourceFamily"]
        assert isinstance(metadata.get("taskType"), str)
        assert metadata["taskType"].strip() == metadata["taskType"]
        assert _fleet_source_role(record) in registry["categories"]

    dpo_records = fleet.train_dpo + fleet.val_dpo
    chosen_chars = sum(len(record["chosen"]["content"]) for record in dpo_records)
    supplemental_chosen_chars = sum(
        len(record["chosen"]["content"])
        for record in dpo_records
        if _fleet_source_role(record) == FLEET_SOURCE_ROLE_SUPPLEMENTAL_STATIC
    )
    chosen_tokens = sum(
        _source_token_proxy_count(record["chosen"]["content"])
        for record in dpo_records
    )
    supplemental_chosen_tokens = sum(
        _source_token_proxy_count(record["chosen"]["content"])
        for record in dpo_records
        if _fleet_source_role(record) == FLEET_SOURCE_ROLE_SUPPLEMENTAL_STATIC
    )
    assert quality["fleetDPOChosenTargetCharCount"] == chosen_chars
    assert quality["fleetSupplementalDPOChosenTargetCharCount"] == (
        supplemental_chosen_chars
    )
    assert quality["fleetSupplementalDPOChosenTargetCharShare"] == pytest.approx(
        supplemental_chosen_chars / chosen_chars if chosen_chars else 0.0
    )
    assert quality["fleetDPOChosenTargetTokenProxyCount"] == chosen_tokens
    assert quality["fleetSupplementalDPOChosenTargetTokenProxyCount"] == (
        supplemental_chosen_tokens
    )
    assert quality["fleetSupplementalDPOChosenTargetTokenProxyShare"] == pytest.approx(
        supplemental_chosen_tokens / chosen_tokens if chosen_tokens else 0.0
    )
    task_types = {
        record["metadata"].get("taskType") for record in records
    }
    assert {
        "ultra_specific_fleet_known_slot_directory",
        "ultra_specific_fleet_delegation",
        "ultra_specific_tool_boundary_awareness",
    } <= task_types


def test_every_agent_config_binds_public_corpus_exact_loss_share_contract() -> None:
    config = FineTuningDatasetConfig(max_public_corpus_token_share=0.90)
    expected = _public_corpus_loss_share_contract(config)

    assert expected["schemaVersion"] == (
        PUBLIC_CORPUS_LOSS_SHARE_CONTRACT_SCHEMA_VERSION
    )
    assert expected["enforcementRequired"] is True
    assert expected["enforcementPhase"] == "post_tokenizer_load_pre_optimizer"
    assert expected["requiredLanes"] == ["sft", "dpo"]
    assert expected["authoritativeCapEncoding"] == "integer_basis_points"
    assert expected["basisPointDenominator"] == 10_000
    assert expected["capBasisPoints"] == {
        "requested": int(
            PUBLIC_CORPUS_ASSISTANT_TARGET_TOKEN_SHARE_HARD_MAX * 10_000
        ),
        "hard": 3_500,
    }
    assert expected["sourceSelectionProxy"] == {
        "status": "safety_budget_not_exact_token_count",
        "maximumPublicShareBasisPoints": int(
            PUBLIC_CORPUS_SOURCE_PROXY_SELECTION_SHARE_HARD_MAX * 10_000
        ),
        "contract": {
            "schemaVersion": "lumen.source-token-proxy/1.0.0",
            "status": "source_side_selection_proxy_not_exact_token_count",
            "strategy": "max_whitespace_terms_utf8_byte_ceiling",
            "maxCharsPerToken": 4,
            "exactPinnedTokenizerAuthoritative": True,
            "authoritativeEnforcementPhase": (
                "post_tokenizer_load_pre_optimizer"
            ),
        },
    }
    assert expected["rowMetadataContract"] == {
        "publicSourceFamilyPrefix": "public_adapter_corpus_",
        "publicCorpusField": "publicCorpus",
        "classificationRule": "prefix_and_nonempty_lineage_required",
        "mismatch": "hard_fail",
    }
    assert expected["exactTokenEvidenceContract"]["lanes"] == {
        "sft": {
            "denominatorTokenCount": "assistantTargetTokenCount",
            "publicNumeratorTokenCount": "publicAssistantTargetTokenCount",
        },
        "dpo": {
            "denominatorTokenCount": "chosenTargetTokenCount",
            "publicNumeratorTokenCount": "publicChosenTargetTokenCount",
        },
    }

    for agent in AGENTS:
        generated = _agent_unsloth_config(
            agent,
            config,
            sft_train_record_count=256,
            dpo_train_record_count=64,
        )
        assert generated["publicCorpusLossShareContract"] == expected


def test_fleet_supplemental_share_configuration_cannot_exceed_hard_max() -> None:
    def record(index: int, source_family: str) -> dict:
        return {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPTS["fleet"]},
                {"role": "user", "content": f"Fleet curriculum item {index}"},
                {
                    "role": "assistant",
                    "content": json.dumps(
                        {
                            "index": index,
                            "summary": "bounded assistant target words",
                        },
                        separators=(",", ":"),
                    ),
                },
            ],
            "metadata": {
                "agent": "fleet",
                "sourceFamily": source_family,
                "taskType": (
                    "ultra_specific_fleet_delegation"
                    if source_family == "adapter_ultra_specific"
                    else "codebase_home_grounding"
                ),
            },
        }

    primary = [record(index, "adapter_ultra_specific") for index in range(20)]
    supplemental = [
        record(index + 100, "codebase_home_sft") for index in range(20)
    ]
    unsafe_config = replace(
        FineTuningDatasetConfig(),
        max_public_corpus_token_share=0.90,
        max_supplemental_sft_ratio=0.95,
        max_fleet_supplemental_assistant_char_share=0.90,
        max_fleet_supplemental_assistant_token_share=0.90,
    )
    unsafe_contract = _agent_unsloth_config(
        "fleet",
        unsafe_config,
        sft_train_record_count=len(primary),
        dpo_train_record_count=1,
    )["fleetLossShareContract"]
    assert unsafe_contract["capsBasisPoints"]["supplementalStaticTotal"] == {
        "requested": 3_000,
        "hard": 3_000,
    }
    assert unsafe_contract["capsBasisPoints"]["publicBehavioralTotal"] == {
        "requested": 3_500,
        "hard": 3_500,
    }
    assert unsafe_contract["capsBasisPoints"][
        "eachSupplementalSourceFamily"
    ] == {"hard": 1_000}
    assert unsafe_contract["sourceSelectionProxy"][
        "maximumSupplementalStaticShareBasisPoints"
    ] == 1_500

    bounded = _limit_supplemental_sft_records(
        "fleet",
        primary + supplemental,
        unsafe_config,
    )
    bounded_supplemental = [
        item
        for item in bounded
        if item["metadata"]["sourceFamily"] == "codebase_home_sft"
    ]
    total_chars = sum(len(item["messages"][2]["content"]) for item in bounded)
    supplemental_chars = sum(
        len(item["messages"][2]["content"])
        for item in bounded_supplemental
    )
    total_tokens = sum(
        len(item["messages"][2]["content"].split()) for item in bounded
    )
    supplemental_tokens = sum(
        len(item["messages"][2]["content"].split())
        for item in bounded_supplemental
    )

    assert bounded_supplemental
    assert (
        supplemental_chars / total_chars
        <= FLEET_SUPPLEMENTAL_ASSISTANT_SHARE_HARD_MAX
    )
    assert (
        supplemental_tokens / total_tokens
        <= FLEET_SUPPLEMENTAL_ASSISTANT_SHARE_HARD_MAX
    )


def test_fleet_supplemental_selection_is_deterministic_and_coverage_first() -> None:
    def record(
        index: int,
        source_family: str,
        task_type: str,
        words: int,
    ) -> dict:
        return {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPTS["fleet"]},
                {"role": "user", "content": f"Fleet item {index}"},
                {
                    "role": "assistant",
                    "content": " ".join(
                        f"bounded_{index}_{offset}" for offset in range(words)
                    ),
                },
            ],
            "metadata": {
                "agent": "fleet",
                "sourceFamily": source_family,
                "taskType": task_type,
            },
        }

    primary = [
        record(
            index,
            "adapter_ultra_specific",
            "fleet_contract_delegation",
            40,
        )
        for index in range(24)
    ]
    supplemental = [
        record(100, "cross_model_training", "fleet_delegation", 8),
        record(101, "cross_model_training", "fleet_peer_source_knowledge", 7),
        record(102, "cross_model_training", "source_code_self_knowledge", 9),
        record(103, "cross_model_training", "fleet_peer_knowledge", 80),
    ]
    config = FineTuningDatasetConfig()

    first = _limit_supplemental_sft_records(
        "fleet",
        [*primary, *reversed(supplemental)],
        config,
    )
    second = _limit_supplemental_sft_records(
        "fleet",
        [*supplemental, *reversed(primary)],
        config,
    )

    assert first == second
    retained_tasks = {
        item["metadata"]["taskType"]
        for item in first
        if item["metadata"]["sourceFamily"] == "cross_model_training"
    }
    assert FLEET_REQUIRED_SUPPLEMENTAL_SFT_TASK_TYPES <= retained_tasks
    supplemental_records = [
        item
        for item in first
        if item["metadata"]["sourceFamily"] == "cross_model_training"
    ]
    total_chars = sum(len(item["messages"][2]["content"]) for item in first)
    supplemental_chars = sum(
        len(item["messages"][2]["content"])
        for item in supplemental_records
    )
    total_tokens = sum(
        _source_token_proxy_count(item["messages"][2]["content"])
        for item in first
    )
    supplemental_tokens = sum(
        _source_token_proxy_count(item["messages"][2]["content"])
        for item in supplemental_records
    )
    assert supplemental_chars / total_chars <= (
        config.max_fleet_supplemental_assistant_char_share
    )
    assert supplemental_tokens / total_tokens <= (
        config.max_fleet_supplemental_assistant_token_share
    )
    assert supplemental_tokens / total_tokens <= (
        FLEET_SUPPLEMENTAL_SOURCE_FAMILY_PROXY_SELECTION_SHARE_HARD_MAX
    )


def test_fleet_required_supplemental_coverage_fails_closed_when_caps_cannot_fit() -> None:
    def record(source_family: str, task_type: str, content: str) -> dict:
        return {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPTS["fleet"]},
                {"role": "user", "content": "Fleet coverage."},
                {"role": "assistant", "content": content},
            ],
            "metadata": {
                "agent": "fleet",
                "sourceFamily": source_family,
                "taskType": task_type,
            },
        }

    records = [
        record(
            "adapter_ultra_specific",
            "fleet_contract_delegation",
            "primary behavioral target",
        ),
        *[
            record("cross_model_training", task_type, "static target")
            for task_type in sorted(
                FLEET_REQUIRED_SUPPLEMENTAL_SFT_TASK_TYPES
            )
        ],
    ]

    with pytest.raises(
        ValueError,
        match="record cap cannot retain every required",
    ):
        _limit_supplemental_sft_records(
            "fleet",
            records,
            FineTuningDatasetConfig(),
        )


def test_fleet_source_role_registry_fails_closed_and_never_promotes_static_cards() -> None:
    def sft(source_family: str, task_type: str) -> dict:
        return {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPTS["fleet"]},
                {"role": "user", "content": "Classify this source."},
                {"role": "assistant", "content": '{"status":"bounded"}'},
            ],
            "metadata": {
                "agent": "fleet",
                "sourceFamily": source_family,
                "taskType": task_type,
            },
        }

    assert _fleet_source_role(
        sft("adapter_ultra_specific", "ultra_specific_fleet_delegation")
    ) == FLEET_SOURCE_ROLE_BEHAVIORAL_PRIMARY
    for source_family, task_type in (
        ("self_model_cards", "self_model_card_grounding"),
        ("manifest_grounding_cards", "manifest_grounding_cards"),
        ("cross_model_training", "fleet_self_knowledge"),
        ("codebase_home_sft", "codebase_home_grounding"),
    ):
        assert _fleet_source_role(sft(source_family, task_type)) == (
            FLEET_SOURCE_ROLE_SUPPLEMENTAL_STATIC
        )

    public = sft(
        "public_adapter_corpus_massive_en_us_1_1",
        "public_capability_delegation",
    )
    public["metadata"]["publicCorpus"] = {
        "sourceRepository": "AmazonScience/massive",
        "sourceRevision": "pinned",
    }
    assert _fleet_source_role(public) == FLEET_SOURCE_ROLE_PUBLIC_BEHAVIORAL

    with pytest.raises(ValueError, match="Unregistered Fleet source-role pair"):
        _fleet_source_role(sft("future_unknown_family", "fleet_self_knowledge"))
    with pytest.raises(ValueError, match="Unregistered Fleet source-role pair"):
        _fleet_source_role(sft("self_model_cards", "future_unknown_task"))
    with pytest.raises(ValueError, match="Unregistered Fleet source-role pair"):
        _limit_supplemental_sft_records(
            "fleet",
            [
                sft(
                    "adapter_ultra_specific",
                    "ultra_specific_fleet_delegation",
                ),
                sft("future_unknown_family", "future_unknown_task"),
            ],
            FineTuningDatasetConfig(),
        )


def test_fleet_dpo_static_chosen_loss_is_bounded_by_char_and_token_share() -> None:
    def dpo(index: int, source_family: str, task_type: str, words: int) -> dict:
        chosen = " ".join(f"token{index}_{offset}" for offset in range(words))
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPTS["fleet"]},
                {"role": "user", "content": f"Fleet preference {index}"},
            ],
            "chosen": {"role": "assistant", "content": chosen},
            "rejected": {"role": "assistant", "content": "wrong"},
            "metadata": {
                "agent": "fleet",
                "sourceFamily": source_family,
                "taskType": task_type,
                "preferenceType": task_type,
            },
        }

    primary = [
        dpo(
            index,
            "adapter_ultra_specific",
            "fleet_contract_delegation",
            40,
        )
        for index in range(12)
    ]
    supplemental = [
        dpo(
            index + 100,
            "cross_model_training",
            "fleet_private_state_boundary",
            20,
        )
        for index in range(20)
    ]
    bounded = _limit_fleet_supplemental_dpo_records(
        primary + supplemental,
        FineTuningDatasetConfig(),
    )
    bounded_static = [
        record
        for record in bounded
        if _fleet_source_role(record) == FLEET_SOURCE_ROLE_SUPPLEMENTAL_STATIC
    ]
    total_chars = sum(len(record["chosen"]["content"]) for record in bounded)
    static_chars = sum(
        len(record["chosen"]["content"]) for record in bounded_static
    )
    total_token_proxy = sum(
        _source_token_proxy_count(record["chosen"]["content"])
        for record in bounded
    )
    static_token_proxy = sum(
        _source_token_proxy_count(record["chosen"]["content"])
        for record in bounded_static
    )

    assert bounded_static
    assert static_chars / total_chars <= 0.25
    assert static_token_proxy / total_token_proxy <= (
        FLEET_SUPPLEMENTAL_SOURCE_FAMILY_PROXY_SELECTION_SHARE_HARD_MAX
    )


def test_fleet_optimizer_finalization_converges_coupled_public_and_static_caps() -> None:
    def sft(
        index: int,
        *,
        source_family: str,
        task_type: str,
        target_words: int,
        public: bool = False,
    ) -> dict:
        metadata: dict = {
            "agent": "fleet",
            "sourceFamily": source_family,
            "taskType": task_type,
        }
        if public:
            metadata["publicCorpus"] = {
                "sourceRepository": "example/fleet-public",
                "sourceRevision": "a" * 40,
                "sourceGroupID": f"public-group-{index}",
                "stratum": "delegation",
                "selectionScore": {"overall": float(index)},
            }
        return {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPTS["fleet"]},
                {"role": "user", "content": f"Fleet optimizer item {index}"},
                {
                    "role": "assistant",
                    "content": " ".join(
                        f"target_{index}_{offset}"
                        for offset in range(target_words)
                    ),
                },
            ],
            "metadata": metadata,
        }

    primary = [
        sft(
            index,
            source_family="adapter_ultra_specific",
            task_type="fleet_contract_delegation",
            target_words=100,
        )
        for index in range(10)
    ]
    supplemental = [
        sft(
            100 + index,
            source_family="codebase_home_sft",
            task_type="codebase_home_grounding",
            target_words=50,
        )
        for index in range(10)
    ]
    public = [
        sft(
            200 + index,
            source_family="public_adapter_corpus_test",
            task_type="public_capability_delegation",
            target_words=100,
            public=True,
        )
        for index in range(10)
    ]
    config = FineTuningDatasetConfig(max_public_corpus_token_share=0.35)
    records = [*primary, *supplemental, *public]

    finalized = _finalize_fleet_optimizer_lane(
        records,
        lane="sft",
        config=config,
    )
    repeated = _finalize_fleet_optimizer_lane(
        list(reversed(finalized)),
        lane="sft",
        config=config,
    )

    assert finalized == repeated
    assert len(finalized) < len(records)
    total = sum(
        _source_token_proxy_count(record["messages"][2]["content"])
        for record in finalized
    )
    supplemental_total = sum(
        _source_token_proxy_count(record["messages"][2]["content"])
        for record in finalized
        if _fleet_source_role(record) == FLEET_SOURCE_ROLE_SUPPLEMENTAL_STATIC
    )
    public_total = sum(
        _source_token_proxy_count(record["messages"][2]["content"])
        for record in finalized
        if _fleet_source_role(record) == FLEET_SOURCE_ROLE_PUBLIC_BEHAVIORAL
    )
    assert supplemental_total * 10_000 <= total * 1_500
    assert supplemental_total * 10_000 <= total * 500
    assert public_total * 10_000 <= total * 3_000


def test_fleet_sft_finalization_balances_native_proxy_by_pruning_public_only() -> None:
    def sft(
        index: int,
        *,
        source_family: str,
        task_type: str,
        target_words: int,
        public: bool = False,
    ) -> dict:
        metadata: dict[str, Any] = {
            "agent": "fleet",
            "manifestCommit": "a" * 40,
            "sourceFamily": source_family,
            "taskType": task_type,
        }
        if public:
            metadata["publicCorpus"] = {
                "sourceRepository": "example/fleet-public",
                "sourceRevision": "a" * 40,
                "sourceGroupID": f"native-balance-group-{index}",
                "stratum": "delegation",
                "selectionScore": {"overall": float(index)},
            }
        return {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPTS["fleet"]},
                {"role": "user", "content": f"Native balance item {index}"},
                {
                    "role": "assistant",
                    "content": " ".join("x" for _ in range(target_words)),
                },
            ],
            "metadata": metadata,
        }

    native = sft(
        1,
        source_family=FLEET_NATIVE_ORCHESTRATION_SOURCE_FAMILY,
        task_type=FLEET_NATIVE_ORCHESTRATION_SFT_TASK_TYPE,
        target_words=560,
    )
    internal = sft(
        2,
        source_family=ULTRA_SPECIFIC_SOURCE_FAMILY,
        task_type="fleet_contract_delegation",
        target_words=400,
    )
    public = [
        sft(
            100 + index,
            source_family="public_adapter_corpus_test",
            task_type="public_capability_delegation",
            target_words=20,
            public=True,
        )
        for index in range(10)
    ]
    finalized = _finalize_fleet_optimizer_lane(
        [native, internal, *public],
        lane="sft",
        config=FineTuningDatasetConfig(max_public_corpus_token_share=0.35),
    )
    repeated = _finalize_fleet_optimizer_lane(
        list(reversed(finalized)),
        lane="sft",
        config=FineTuningDatasetConfig(max_public_corpus_token_share=0.35),
    )
    assert finalized == repeated
    assert native in finalized
    assert internal in finalized
    assert 0 < sum(record in finalized for record in public) < len(public)
    total = sum(
        _source_token_proxy_count(record["messages"][-1]["content"])
        for record in finalized
    )
    native_tokens = _source_token_proxy_count(
        native["messages"][-1]["content"]
    )
    assert native_tokens * 10_000 >= total * (
        FLEET_NATIVE_ORCHESTRATION_SFT_PROXY_SHARE_MIN_BASIS_POINTS
    )
    assert native_tokens * 10_000 <= total * (
        FLEET_NATIVE_ORCHESTRATION_SFT_PROXY_SHARE_MAX_BASIS_POINTS
    )

    with pytest.raises(
        ValueError,
        match="source-proxy share exceeds its safety maximum",
    ):
        _finalize_fleet_optimizer_lane(
            [
                sft(
                    500,
                    source_family=FLEET_NATIVE_ORCHESTRATION_SOURCE_FAMILY,
                    task_type=FLEET_NATIVE_ORCHESTRATION_SFT_TASK_TYPE,
                    target_words=700,
                ),
                sft(
                    501,
                    source_family=ULTRA_SPECIFIC_SOURCE_FAMILY,
                    task_type="fleet_contract_delegation",
                    target_words=300,
                ),
            ],
            lane="sft",
            config=FineTuningDatasetConfig(),
        )

    missing_commit = copy.deepcopy([native, internal])
    missing_commit[1]["metadata"].pop("manifestCommit")
    mixed_commits = copy.deepcopy([native, internal])
    mixed_commits[1]["metadata"]["manifestCommit"] = "b" * 40
    malformed_commit = copy.deepcopy([native, internal])
    for record in malformed_commit:
        record["metadata"]["manifestCommit"] = "not-a-commit"
    for invalid_lineage in (
        missing_commit,
        mixed_commits,
        malformed_commit,
    ):
        with pytest.raises(
            ValueError,
            match="canonical 40-character lowercase hexadecimal manifestCommit",
        ):
            _finalize_fleet_optimizer_lane(
                invalid_lineage,
                lane="sft",
                config=FineTuningDatasetConfig(),
            )


def test_fleet_sft_optimizer_geometry_is_bounded_from_immutable_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def sft(
        index: int,
        *,
        source_family: str,
        task_type: str,
        target_words: int,
        public: bool = False,
    ) -> dict:
        metadata: dict[str, Any] = {
            "agent": "fleet",
            "sourceFamily": source_family,
            "taskType": task_type,
        }
        if public:
            metadata["publicCorpus"] = {
                "sourceRepository": "example/fleet-public",
                "sourceRevision": "a" * 40,
                "sourceGroupID": f"geometry-public-{index}",
                "stratum": "delegation",
                "selectionScore": {"overall": float(index)},
            }
        return {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPTS["fleet"]},
                {"role": "user", "content": f"Geometry item {index}"},
                {
                    "role": "assistant",
                    "content": " ".join(
                        f"geometry_{index}_{offset}"
                        for offset in range(target_words)
                    ),
                },
            ],
            "metadata": metadata,
        }

    primary = [
        sft(
            index,
            source_family=ULTRA_SPECIFIC_SOURCE_FAMILY,
            task_type="fleet_contract_delegation",
            target_words=100,
        )
        for index in range(530)
    ]
    supplemental = [
        sft(
            1_000 + index,
            source_family="codebase_home_sft",
            task_type="codebase_home_grounding",
            target_words=100,
        )
        for index in range(100)
    ]
    public = [
        sft(
            2_000 + index,
            source_family="public_adapter_corpus_test",
            task_type="public_capability_delegation",
            target_words=10,
            public=True,
        )
        for index in range(100)
    ]
    records = [*primary, *supplemental, *public]
    records_snapshot = copy.deepcopy(records)
    config = FineTuningDatasetConfig(
        max_fleet_sft_optimizer_records=(
            FLEET_SFT_OPTIMIZER_MAX_TRAIN_RECORDS
        )
    )

    serialized_record_ids: list[int] = []
    serialize_canonical_record = (
        fine_tuning_module._serialize_canonical_record
    )

    def tracked_serialize_canonical_record(record: dict[str, Any]) -> str:
        serialized_record_ids.append(id(record))
        return serialize_canonical_record(record)

    monkeypatch.setattr(
        fine_tuning_module,
        "_serialize_canonical_record",
        tracked_serialize_canonical_record,
    )

    finalized = _finalize_fleet_optimizer_lane(
        records,
        lane="sft",
        config=config,
    )
    assert Counter(serialized_record_ids) == Counter(
        {id(record): 1 for record in records}
    )
    serialized_record_ids.clear()
    repeated = _finalize_fleet_optimizer_lane(
        list(reversed(records)),
        lane="sft",
        config=config,
    )
    assert Counter(serialized_record_ids) == Counter(
        {id(record): 1 for record in records}
    )

    assert finalized == repeated
    assert records == records_snapshot
    mutation_probe = copy.deepcopy(records[0])
    original_probe_key = fine_tuning_module._canonical_record_key(
        mutation_probe
    )
    mutation_probe["metadata"]["cacheScopeProbe"] = True
    assert fine_tuning_module._canonical_record_key(
        mutation_probe
    ) != original_probe_key
    assert len(finalized) == FLEET_SFT_OPTIMIZER_MAX_TRAIN_RECORDS
    assert {
        _canonical_messages_key(record) for record in primary
    } <= {
        _canonical_messages_key(record) for record in finalized
    }
    assert any(record in finalized for record in supplemental)
    assert any(record in finalized for record in public)

    for invalid_limit in (0, True):
        with pytest.raises(
            ValueError,
            match="max_fleet_sft_optimizer_records must be a positive integer",
        ):
            _finalize_fleet_optimizer_lane(
                records[:2],
                lane="sft",
                config=replace(
                    config,
                    max_fleet_sft_optimizer_records=invalid_limit,
                ),
            )

    with pytest.raises(
        ValueError,
        match="protected behavioral geometry exceeds",
    ):
        _finalize_fleet_optimizer_lane(
            primary[:11],
            lane="sft",
            config=replace(config, max_fleet_sft_optimizer_records=10),
        )


def test_fleet_optimizer_finalization_repairs_post_split_concentration_only() -> None:
    def sft(
        index: int,
        *,
        supplemental: bool,
        required_split: str,
    ) -> dict:
        return {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPTS["fleet"]},
                {"role": "user", "content": f"Fleet split item {index}"},
                {
                    "role": "assistant",
                    "content": " ".join(
                        f"split_{index}_{offset}" for offset in range(100)
                    ),
                },
            ],
            "metadata": {
                "agent": "fleet",
                "sourceFamily": (
                    "codebase_home_sft"
                    if supplemental
                    else "adapter_ultra_specific"
                ),
                "taskType": (
                    "codebase_home_grounding"
                    if supplemental
                    else "fleet_contract_delegation"
                ),
                "requiredSplit": required_split,
            },
        }

    train_primary = [
        sft(index, supplemental=False, required_split="train")
        for index in range(10)
    ]
    train_supplemental = [
        sft(100 + index, supplemental=True, required_split="train")
        for index in range(2)
    ]
    validation_primary = [
        sft(200 + index, supplemental=False, required_split="validation")
        for index in range(20)
    ]
    combined = [*train_primary, *train_supplemental, *validation_primary]

    def proxy(records: list[dict], *, supplemental_only: bool = False) -> int:
        return sum(
            _source_token_proxy_count(record["messages"][2]["content"])
            for record in records
            if not supplemental_only
            or _fleet_source_role(record)
            == FLEET_SOURCE_ROLE_SUPPLEMENTAL_STATIC
        )

    combined_total = proxy(combined)
    combined_supplemental = proxy(combined, supplemental_only=True)
    assert combined_supplemental * 10_000 <= combined_total * 1_500

    train, validation = _stable_source_stratified_split(
        combined,
        FineTuningDatasetConfig(),
        agent="fleet",
    )
    prefinal_total = proxy(train)
    prefinal_supplemental = proxy(train, supplemental_only=True)
    assert prefinal_supplemental * 10_000 > prefinal_total * 1_500
    validation_snapshot = copy.deepcopy(validation)

    finalized = _finalize_fleet_optimizer_lane(
        train,
        lane="sft",
        config=FineTuningDatasetConfig(),
    )
    finalized_total = proxy(finalized)
    finalized_supplemental = proxy(finalized, supplemental_only=True)
    assert finalized_supplemental * 10_000 <= finalized_total * 1_500
    assert finalized_supplemental * 10_000 <= finalized_total * 500
    assert validation == validation_snapshot


def test_fleet_sft_validation_is_bounded_stratified_and_preserves_assignments() -> None:
    def sft(
        index: int,
        *,
        source_family: str,
        task_type: str,
        required_split: str | None = None,
    ) -> dict:
        metadata = {
            "agent": "fleet",
            "sourceFamily": source_family,
            "taskType": task_type,
        }
        if required_split is not None:
            metadata["requiredSplit"] = required_split
        return {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPTS["fleet"]},
                {"role": "user", "content": f"Validation request {index}"},
                {
                    "role": "assistant",
                    "content": json.dumps(
                        {"knownSlots": ["cortex", "executor"], "case": index},
                        sort_keys=True,
                    ),
                },
            ],
            "metadata": metadata,
        }

    records = [
        sft(
            index,
            source_family=source_family,
            task_type=task_type,
            required_split=(
                "validation" if index in {0, 3} else None
            ),
        )
        for index, (source_family, task_type) in enumerate(
            [
                ("adapter_ultra_specific", "fleet_contract_delegation"),
                ("cross_model_training", "fleet_peer_source_knowledge"),
                ("codebase_home_chunk_sft", "codebase_source_chunk_grounding"),
            ]
            * 10
        )
    ]
    config = FineTuningDatasetConfig(max_fleet_validation_sft_records=8)

    bounded = _bound_fleet_validation_sft_records(
        records,
        config=config,
        required_reference_records=records,
    )
    repeated = _bound_fleet_validation_sft_records(
        list(reversed(records)),
        config=config,
        required_reference_records=list(reversed(records)),
    )
    idempotent = _bound_fleet_validation_sft_records(
        bounded,
        config=config,
        required_reference_records=records,
    )

    assert bounded == repeated == idempotent
    assert len(bounded) == 8
    assert {
        record["messages"][1]["content"]
        for record in bounded
        if record["metadata"].get("requiredSplit") == "validation"
    } == {"Validation request 0", "Validation request 3"}
    selected_strata = {
        (record["metadata"]["sourceFamily"], record["metadata"]["taskType"])
        for record in bounded
    }
    assert len(selected_strata) == 3

    public_records = []
    public_intents = {
        "calendar_set",
        "email_sendemail",
        "weather_query",
        "web_search",
    }
    for intent_index, intent in enumerate(sorted(public_intents)):
        for example_index in range(4):
            record = sft(
                300 + intent_index * 10 + example_index,
                source_family="public_adapter_corpus_massive",
                task_type="public_capability_delegation",
            )
            record["metadata"]["publicCorpus"] = {
                "sourceRepository": "mteb/massive_intent",
                "sourceRevision": "fixture-revision",
                "stratum": intent,
            }
            public_records.append(record)
    public_config = replace(config, max_fleet_validation_sft_records=4)
    public_bounded = _bound_fleet_validation_sft_records(
        public_records,
        config=public_config,
    )
    assert {
        record["metadata"]["publicCorpus"]["stratum"]
        for record in public_bounded
    } == public_intents

    metadata_churned = copy.deepcopy(public_records)
    for index, record in enumerate(metadata_churned):
        record["metadata"].update(
            {
                "manifestCommit": f"commit-{index}",
                "sourceIntegrity": {"workingTreeDigest": f"digest-{index}"},
                "worktreeFingerprint": f"fingerprint-{index}",
            }
        )
    churned_bounded = _bound_fleet_validation_sft_records(
        metadata_churned,
        config=public_config,
    )
    assert {
        _canonical_messages_key(record) for record in public_bounded
    } == {
        _canonical_messages_key(record) for record in churned_bounded
    }

    missing_required = [
        record
        for record in records
        if record["metadata"].get("requiredSplit") != "validation"
    ]
    with pytest.raises(ValueError, match="dropped explicitly assigned"):
        _bound_fleet_validation_sft_records(
            missing_required,
            config=config,
            required_reference_records=records,
        )
    for invalid_limit in (False, 0, -1, 1.5):
        with pytest.raises(ValueError, match="positive integer"):
            _bound_fleet_validation_sft_records(
                records,
                config=replace(
                    config,
                    max_fleet_validation_sft_records=invalid_limit,
                ),
            )

    too_many_required = [
        sft(
            100 + index,
            source_family="adapter_ultra_specific",
            task_type="fleet_contract_delegation",
            required_split="validation",
        )
        for index in range(3)
    ]
    with pytest.raises(ValueError, match="assignments exceed"):
        _bound_fleet_validation_sft_records(
            too_many_required,
            config=replace(config, max_fleet_validation_sft_records=2),
        )

    train_assigned = sft(
        200,
        source_family="adapter_ultra_specific",
        task_type="fleet_contract_delegation",
        required_split="train",
    )
    with pytest.raises(ValueError, match="train-assigned"):
        _bound_fleet_validation_sft_records(
            [train_assigned],
            config=config,
        )

    with pytest.raises(ValueError, match="every source/task stratum"):
        _bound_fleet_validation_sft_records(
            records,
            config=replace(config, max_fleet_validation_sft_records=3),
        )


def test_fleet_real_sft_validation_bound_is_attested_for_every_variant(
    compiled_fine_tuning: tuple,
) -> None:
    _, _, fine_tuning = compiled_fine_tuning
    fleet = fine_tuning["fleet"]
    policy = fleet.dataset_card["constraints"][
        "fleetValidationSamplingPolicy"
    ]

    assert policy["schemaVersion"] == FLEET_VALIDATION_SAMPLING_SCHEMA_VERSION
    assert policy["status"] == "bounded_stratified_observation"
    assert policy["lane"] == "sft_validation"
    assert policy["maximumRecords"] == 128
    assert policy["selectedRecordCount"] == len(fleet.val_sft)
    assert policy["candidateBeforePublicSelectionRecordCount"] >= (
        policy["samplingInputRecordCount"]
    )
    assert policy["samplingInputRecordCount"] >= policy["selectedRecordCount"]
    assert policy["rejectedByPublicSelectionCount"] == (
        policy["candidateBeforePublicSelectionRecordCount"]
        - policy["samplingInputRecordCount"]
    )
    assert policy["rejectedByValidationSamplingCount"] == (
        policy["samplingInputRecordCount"] - policy["selectedRecordCount"]
    )
    assert policy["explicitValidationAssignmentsPreserved"] is True
    assert policy["allSamplingInputStrataPreserved"] is True
    assert policy["optimizerLossShareCapsApplied"] is False
    assert len(policy["selectedCohortSHA256"]) == 64
    assert policy["samplingInputStratumCount"] == policy[
        "selectedStratumCount"
    ]
    if policy["samplingInputRecordCount"] > policy["maximumRecords"]:
        assert policy["selectedRecordCount"] == policy["maximumRecords"]
    assert len(fleet.val_sft) <= policy["maximumRecords"]
    required_validation_keys = {
        _canonical_messages_key(record)
        for record in fleet.val_sft
        if record["metadata"].get("requiredSplit") == "validation"
    }
    for variant in fleet.experiment_variants.values():
        assert len(variant["val_sft"]) <= policy["maximumRecords"]
        assert required_validation_keys <= {
            _canonical_messages_key(record) for record in variant["val_sft"]
        }
        assert {
            _canonical_rendered_prompt_key(record, lane="sft")
            for record in variant["train_sft"]
        }.isdisjoint(
            {
                _canonical_rendered_prompt_key(record, lane="sft")
                for record in variant["val_sft"]
            }
        )
        variant_policy = variant["variant_manifest"][
            "validationSamplingPolicy"
        ]
        assert variant_policy["selectedRecordCount"] == len(
            variant["val_sft"]
        )
        if (
            variant_policy["samplingInputRecordCount"]
            > variant_policy["maximumRecords"]
        ):
            assert len(variant["val_sft"]) == variant_policy[
                "maximumRecords"
            ]
        assert variant_policy["selectedCohortSHA256"] == canonical_sha256(
            sorted(
                _canonical_messages_key(record)
                for record in variant["val_sft"]
            )
        )


def test_fleet_dpo_public_cap_uses_chosen_loss_not_rejected_length() -> None:
    def completion(label: str, words: int) -> str:
        return " ".join(f"{label}_{index}" for index in range(words))

    primary = {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPTS["fleet"]},
            {"role": "user", "content": "Use the native Fleet contract."},
        ],
        "chosen": {"role": "assistant", "content": completion("primary", 1)},
        "rejected": {
            "role": "assistant",
            "content": completion("primary_rejected", 1_000),
        },
        "metadata": {
            "agent": "fleet",
            "sourceFamily": "adapter_ultra_specific",
            "taskType": "fleet_contract_delegation",
        },
    }
    public = {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPTS["fleet"]},
            {"role": "user", "content": "Use a public Fleet example."},
        ],
        "chosen": {
            "role": "assistant",
            "content": completion("public", 100),
        },
        "rejected": {
            "role": "assistant",
            "content": completion("public_rejected", 1),
        },
        "metadata": {
            "agent": "fleet",
            "sourceFamily": "public_adapter_corpus_test",
            "taskType": "public_capability_delegation",
            "publicCorpus": {
                "sourceRepository": "example/fleet-public",
                "sourceRevision": "a" * 40,
                "sourceGroupID": "dpo-public-group",
                "stratum": "delegation",
                "selectionScore": {"overall": 1.0},
            },
        },
    }

    finalized = _finalize_fleet_optimizer_lane(
        [primary, public],
        lane="dpo",
        config=FineTuningDatasetConfig(max_public_corpus_token_share=0.35),
    )

    assert finalized == [primary]

    prompt_heavy_primary = {
        **primary,
        "prompt": [
            {
                "role": "system",
                "content": completion("primary_prompt", 1_000),
            },
            {"role": "user", "content": "Use the native Fleet contract."},
        ],
        "chosen": {
            "role": "assistant",
            "content": completion("primary_chosen", 100),
        },
        "rejected": {
            "role": "assistant",
            "content": completion("primary_rejected", 1),
        },
    }
    rejected_heavy_public = {
        **public,
        "chosen": {
            "role": "assistant",
            "content": completion("public_chosen", 1),
        },
        "rejected": {
            "role": "assistant",
            "content": completion("public_rejected", 100),
        },
    }
    all_assistant_prefilter = _cap_public_corpus_token_share(
        [prompt_heavy_primary, rejected_heavy_public],
        0.35,
    )
    chosen_prefilter = _cap_public_corpus_token_share(
        [prompt_heavy_primary, rejected_heavy_public],
        0.35,
        target_mode="dpo_chosen",
    )

    assert rejected_heavy_public not in all_assistant_prefilter
    assert rejected_heavy_public in chosen_prefilter
    assert _finalize_fleet_optimizer_lane(
        chosen_prefilter,
        lane="dpo",
        config=FineTuningDatasetConfig(max_public_corpus_token_share=0.35),
    ) == chosen_prefilter


def test_comprehensive_fleet_tool_ownership_sft_covers_every_non_holdout_tool(
    compiled_fine_tuning: tuple,
) -> None:
    manifest, _, fine_tuning = compiled_fine_tuning
    anchors = _comprehensive_fleet_tool_ownership_sft_anchors(
        manifest,
        list(manifest.tools),
    )
    expected_tool_ids = {
        tool.id for tool in manifest.tools if tool.id != "maps.search"
    }
    assert len(anchors) == (
        len(expected_tool_ids)
        * FLEET_COMPREHENSIVE_TOOL_OWNERSHIP_SFT_SURFACES
    )
    assert Counter(
        record["metadata"]["targetToolID"] for record in anchors
    ) == Counter(
        {
            tool_id: FLEET_COMPREHENSIVE_TOOL_OWNERSHIP_SFT_SURFACES
            for tool_id in expected_tool_ids
        }
    )
    assert len(
        {
            _canonical_rendered_prompt_key(record, lane="sft")
            for record in anchors
        }
    ) == len(anchors)
    bound_anchors = _bind_fleet_sft_contract(manifest, anchors)
    assert len(bound_anchors) == len(anchors)
    for record in bound_anchors:
        user = record["messages"][-2]["content"]
        assert user.endswith(
            f"\n\n{FLEET_TOOL_OWNERSHIP_OUTPUT_CONTRACT}"
        )
        assert user.count(FLEET_TOOL_OWNERSHIP_OUTPUT_CONTRACT) == 1
    expected_prompt_keys = {
        _canonical_rendered_prompt_key(record, lane="sft")
        for record in bound_anchors
    }
    fleet = fine_tuning["fleet"]
    compiled_lanes = {
        "root": fleet.train_sft,
        **{
            name: variant["train_sft"]
            for name, variant in fleet.experiment_variants.items()
        },
    }
    for lane_name, lane_records in compiled_lanes.items():
        compiled_anchors = [
            record
            for record in lane_records
            if record["metadata"].get("curriculumMode")
            == "comprehensive_tool_ownership_sft_matrix"
        ]
        assert len(compiled_anchors) == len(bound_anchors), lane_name
        assert {
            _canonical_rendered_prompt_key(record, lane="sft")
            for record in compiled_anchors
        } == expected_prompt_keys, lane_name
    executor_slot_id = next(
        slot.id for slot in manifest.fleet.slots if slot.role == "tool_executor"
    )
    planning_slot_id = next(
        slot.id for slot in manifest.fleet.slots if slot.role == "orchestrator"
    )
    response_slot_id = next(
        slot.id for slot in manifest.fleet.slots if slot.role == "user_response"
    )
    for record in anchors:
        metadata = record["metadata"]
        payload = json.loads(record["messages"][-1]["content"])
        tool_id = metadata["targetToolID"]
        assert metadata["requiredSplit"] == "train"
        assert metadata["sourceFamily"] == ULTRA_SPECIFIC_SOURCE_FAMILY
        assert metadata["taskType"] == "fleet_contract_tool_ownership"
        assert metadata["toolIDs"] == [tool_id]
        assert set(metadata["toolContracts"]) == {tool_id}
        assert payload == {
            "executionOwnerSlotID": executor_slot_id,
            "planningOwnerSlotID": planning_slot_id,
            "responseOwnerSlotID": response_slot_id,
            "toolID": tool_id,
        }
        assert "knownSlots" not in record["messages"][-1]["content"]
        assert "knownSlotIDs" not in record["messages"][-1]["content"]


def test_comprehensive_fleet_tool_ownership_requires_authoritative_owners() -> None:
    manifest = AgentBehaviorManifest.model_validate(
        {
            "sourceIntegrity": {"baseCommit": "a" * 40},
            "fleet": {
                "slots": [
                    {"id": "rem", "role": "idle_reflection"},
                    {"id": "embedding", "role": "embedding"},
                ]
            },
        }
    )

    with pytest.raises(
        ValueError,
        match=(
            "manifested semantic owners: cortex, executor, mouth"
        ),
    ):
        _comprehensive_fleet_tool_ownership_sft_anchors(manifest, [])


def test_fleet_curriculum_materializes_exact_slot_and_boundary_contracts(
    compiled_fine_tuning: tuple,
) -> None:
    manifest, _, fine_tuning = compiled_fine_tuning
    fleet = fine_tuning["fleet"]
    slot_ids = [slot.id for slot in manifest.fleet.slots]
    executor_slot_id = next(
        slot.id for slot in manifest.fleet.slots if slot.role == "tool_executor"
    )
    contract_task_types = {
        "ultra_specific_fleet_known_slot_directory",
        "ultra_specific_fleet_delegation",
        "ultra_specific_tool_boundary_awareness",
    }

    for split in (fleet.train_sft, fleet.val_sft):
        records = [
            record
            for record in split
            if record["metadata"].get("taskType") in contract_task_types
        ]
        assert contract_task_types <= {
            record["metadata"]["taskType"] for record in records
        }
        for record in records:
            payload = json.loads(record["messages"][2]["content"])
            if "knownSlots" in payload:
                assert payload["knownSlots"] == slot_ids
            if "delegateTo" in payload:
                assert payload["delegateTo"] in slot_ids
                assert payload["delegateTo"] != "fleet"
            if (
                record["metadata"]["taskType"]
                == "ultra_specific_fleet_known_slot_directory"
            ):
                assert set(payload) == {"knownSlots"}
            if (
                record["metadata"]["taskType"]
                == "ultra_specific_fleet_delegation"
            ):
                assert set(payload) == {"delegateTo", "knownSlots", "reason"}
                assert payload["reason"] == "manifest_responsibility_match"
            if record["metadata"]["taskType"] == "ultra_specific_tool_boundary_awareness":
                assert set(payload) == {
                    "toolID",
                    "delegateTo",
                    "approvalState",
                    "permissionState",
                    "knownSlots",
                }
                assert payload["delegateTo"] == executor_slot_id
                assert payload["toolID"] != "maps.search"


def test_fleet_contract_dpo_is_balanced_scorer_aligned_and_update_sized(
    compiled_fine_tuning: tuple,
) -> None:
    manifest, _, fine_tuning = compiled_fine_tuning
    fleet = fine_tuning["fleet"]
    slot_ids = [slot.id for slot in manifest.fleet.slots]
    executor_slot_id = next(
        slot.id for slot in manifest.fleet.slots if slot.role == "tool_executor"
    )
    contract_types = {
        "fleet_contract_delegation",
        "fleet_contract_known_slots",
        "fleet_contract_tool_boundary",
    }

    train_counts = Counter(
        record["metadata"].get("preferenceType")
        for record in fleet.train_dpo
        if record["metadata"].get("preferenceType") in contract_types
    )
    validation_counts = Counter(
        record["metadata"].get("preferenceType")
        for record in fleet.val_dpo
        if record["metadata"].get("preferenceType") in contract_types
    )
    assert train_counts == Counter(
        {
            "fleet_contract_delegation": 248,
            "fleet_contract_known_slots": 21,
            "fleet_contract_tool_boundary": 21,
        }
    )
    assert validation_counts == Counter(
        {
            "fleet_contract_delegation": 12,
            "fleet_contract_known_slots": 3,
            "fleet_contract_tool_boundary": 3,
        }
    )
    assert len(fleet.train_dpo) >= (
        fleet.unsloth_config["batch_size"]
        * fleet.unsloth_config["gradient_accumulation_steps"]
        * 4
    )
    optimizer_dpo_lanes = {
        "root": fleet.train_dpo,
        **{
            name: variant["train_dpo"]
            for name, variant in fleet.experiment_variants.items()
        },
    }
    for records in optimizer_dpo_lanes.values():
        native_count = sum(
            1
            for record in records
            if record["metadata"].get("sourceFamily")
            == FLEET_NATIVE_ORCHESTRATION_SOURCE_FAMILY
            and record["metadata"].get("taskType")
            == FLEET_NATIVE_ORCHESTRATION_DPO_TASK_TYPE
        )
        if native_count:
            assert (
                native_count * FLEET_LOSS_SHARE_BASIS_POINTS_DENOMINATOR
                >= len(records)
                * FLEET_NATIVE_ORCHESTRATION_DPO_SHARE_MIN_BASIS_POINTS
            )
            assert (
                native_count * FLEET_LOSS_SHARE_BASIS_POINTS_DENOMINATOR
                <= len(records)
                * FLEET_NATIVE_ORCHESTRATION_DPO_SHARE_MAX_BASIS_POINTS
            )
    _, _, slot_ids_by_agent = _fleet_slot_contract(manifest)
    finalized_retry_records = [
        record
        for record in fleet.train_dpo
        if record["metadata"].get("taskType")
        in FLEET_BALANCED_CONTRACT_TASK_TYPES
        and record["metadata"].get("generationPromptMode")
        == "strict_json_retry"
    ]
    finalized_retry_groups = {
        (
            record["metadata"]["taskType"],
            record["metadata"]["retryCoverageScope"],
            _canonical_rendered_prompt_key(record, lane="dpo"),
        )
        for record in finalized_retry_records
    }
    assert len(finalized_retry_records) == 16
    assert len(finalized_retry_groups) == 8
    assert {
        (task_type, scope)
        for task_type, scope, _ in finalized_retry_groups
    } == {
        *{
            ("fleet_contract_delegation", slot_ids_by_agent[owner])
            for owner in _fleet_delegation_tasks()
        },
        ("fleet_contract_known_slots", "contract_family"),
        ("fleet_contract_tool_boundary", "contract_family"),
    }
    assert fleet.contamination_report["contaminated"] is False
    assert fleet.contamination_report["matchCount"] == 0

    delegation_tasks = _fleet_delegation_tasks()
    assert set(delegation_tasks) == {
        "cortex",
        "executor",
        "mouth",
        "mimicry",
        "rem",
        "embedding",
    }
    assert {
        owner: len(prompts) for owner, prompts in delegation_tasks.items()
    } == {
        owner: FLEET_DELEGATION_PROMPTS_PER_OWNER
        for owner in delegation_tasks
    }
    assert len(
        {
            prompt
            for prompts in delegation_tasks.values()
            for prompt in prompts
        }
    ) == len(delegation_tasks) * FLEET_DELEGATION_PROMPTS_PER_OWNER
    mouth_confusion_tasks = _fleet_observed_mouth_confusion_tasks()
    assert set(mouth_confusion_tasks) == {"embedding", "executor"}
    assert {
        owner: len(prompts)
        for owner, prompts in mouth_confusion_tasks.items()
    } == {
        owner: FLEET_OBSERVED_MOUTH_CONFUSION_PROMPTS_PER_OWNER
        for owner in mouth_confusion_tasks
    }
    for prompt in mouth_confusion_tasks["embedding"]:
        lowered = prompt.lower()
        assert "embedding" in lowered
        assert any(term in lowered for term in ("mimicry", "style", "voice"))
        assert any(
            term in lowered
            for term in ("executor", "execution", "tool", "action", "invocation")
        )
    for prompt in mouth_confusion_tasks["executor"]:
        lowered = prompt.lower()
        assert any(term in lowered for term in ("executor", "execution", "tool"))
        assert any(term in lowered for term in ("embedding", "vector", "numeric"))
        assert any(
            term in lowered
            for term in ("mimicry", "style", "voice", "tone")
        )
    assert not (
        {
            prompt
            for prompts in delegation_tasks.values()
            for prompt in prompts
        }
        & {
            prompt
            for prompts in mouth_confusion_tasks.values()
            for prompt in prompts
        }
    )

    delegation_by_split = {
        "train": [
            record
            for record in fleet.train_dpo
            if record["metadata"].get("preferenceType")
            == "fleet_contract_delegation"
        ],
        "validation": [
            record
            for record in fleet.val_dpo
            if record["metadata"].get("preferenceType")
            == "fleet_contract_delegation"
        ],
    }
    for split in ("train", "validation"):
        owner_counts = Counter(
            json.loads(record["chosen"]["content"])["delegateTo"]
            for record in delegation_by_split[split]
        )
        expected_per_owner = (
            FLEET_DELEGATION_VALIDATION_PROMPTS_PER_OWNER
            if split == "validation"
            else FLEET_DELEGATION_PROMPTS_PER_OWNER
            - FLEET_DELEGATION_VALIDATION_PROMPTS_PER_OWNER
        )
        expected_counts = {
            slot_ids_by_agent[owner]: expected_per_owner
            for owner in delegation_tasks
        }
        if split == "train":
            for owner in delegation_tasks:
                expected_counts[slot_ids_by_agent[owner]] += 4
            for owner in ("embedding", "executor"):
                expected_counts[slot_ids_by_agent[owner]] += (
                    8 + FLEET_OBSERVED_MOUTH_CONFUSION_PROMPTS_PER_OWNER
                )
        assert owner_counts == Counter(expected_counts)

    for owner in delegation_tasks:
        target_slot_id = slot_ids_by_agent[owner]
        owner_train = [
            record
            for record in delegation_by_split["train"]
            if json.loads(record["chosen"]["content"])["delegateTo"]
            == target_slot_id
        ]
        rejected_counts = Counter(
            json.loads(record["rejected"]["content"])["delegateTo"]
            for record in owner_train
        )
        expected_alternatives = {
            *[slot_id for slot_id in slot_ids if slot_id != target_slot_id],
            "invented_shadow_slot",
        }
        assert set(rejected_counts) == expected_alternatives
        assert min(rejected_counts.values()) >= 2
        contrast_modes = Counter(
            record["metadata"]["contrastMode"] for record in owner_train
        )
        assert contrast_modes["balanced_rotation"] == 32
        assert contrast_modes["balanced_secondary_semantic_contrast"] == 4
        assert contrast_modes["observed_mimicry_confusion"] == (
            7
            if owner in {"embedding", "executor"}
            else 0
        )
        assert contrast_modes["independent_semantic_confusion"] == (
            1 if owner in {"embedding", "executor"} else 0
        )
        assert contrast_modes["observed_mouth_confusion"] == (
            FLEET_OBSERVED_MOUTH_CONFUSION_PROMPTS_PER_OWNER
            if owner in {"embedding", "executor"}
            else 0
        )
        for record in owner_train:
            mode = record["metadata"]["contrastMode"]
            rejected_slot = json.loads(record["rejected"]["content"])[
                "delegateTo"
            ]
            if mode == "observed_mimicry_confusion":
                assert rejected_slot == slot_ids_by_agent["mimicry"]
                assert "observed Mimicry" in record["metadata"]["reason"]
            elif mode == "independent_semantic_confusion":
                assert rejected_slot != slot_ids_by_agent["mimicry"]
                assert (
                    "independent wrong semantic owner"
                    in record["metadata"]["reason"]
                )
            elif mode == "observed_mouth_confusion":
                assert rejected_slot == slot_ids_by_agent["mouth"]
                assert "observed Mouth" in record["metadata"]["reason"]
            elif mode == "balanced_secondary_semantic_contrast":
                assert rejected_slot != target_slot_id
                assert "second independent manifested semantic owner" in (
                    record["metadata"]["reason"]
                )

    records = [
        record
        for record in fleet.train_dpo + fleet.val_dpo
        if record["metadata"].get("preferenceType") in contract_types
    ]
    assert records
    for record in records:
        preference_type = record["metadata"]["preferenceType"]
        chosen = json.loads(record["chosen"]["content"])
        rejected = json.loads(record["rejected"]["content"])
        assert set(chosen) == set(rejected)
        changed_keys = {
            key for key in chosen if chosen[key] != rejected[key]
        }
        if record["metadata"].get("generationPromptMode") == (
            "strict_json_retry"
        ):
            assert record["metadata"]["retryContrastMode"] == (
                "strict_retry_compound_repetition"
            )
            assert record["metadata"]["compoundPreferenceDimensions"] == [
                "incorrect_contract_value",
                "duplicate_known_slot",
            ]
            assert "knownSlots" in changed_keys
            assert len(changed_keys) == (
                1 if preference_type == "fleet_contract_known_slots" else 2
            )
        else:
            assert len(changed_keys) == 1
        assert chosen["knownSlots"] == slot_ids
        if "delegateTo" in chosen:
            assert chosen["delegateTo"] in slot_ids
            assert chosen["delegateTo"] != "fleet"

        if preference_type == "fleet_contract_known_slots":
            assert set(chosen) == {"knownSlots"}
            metric = {
                "type": "json_array_exact_members",
                "path": "knownSlots",
                "values": slot_ids,
                "exactKeys": ["knownSlots"],
                "ordered": True,
            }
        elif preference_type == "fleet_contract_delegation":
            assert set(chosen) == {"delegateTo", "knownSlots", "reason"}
            assert chosen["reason"] == "manifest_responsibility_match"
            metric = {
                "type": "delegation",
                "expectedSlot": chosen["delegateTo"],
                "allowedSlots": slot_ids,
                "expectedKnownSlots": slot_ids,
                "expectedReason": "manifest_responsibility_match",
                "exactKeys": ["delegateTo", "knownSlots", "reason"],
                "sourceSlot": "fleet",
            }
        else:
            assert chosen["delegateTo"] == executor_slot_id
            assert chosen["toolID"] != "maps.search"
            metric = {
                "type": "tool_slot_boundary",
                "contract": {
                    "expectedToolID": chosen["toolID"],
                    "expectedSlot": executor_slot_id,
                    "allowedSlots": slot_ids,
                    "approvalState": chosen["approvalState"],
                    "permissionState": chosen["permissionState"],
                },
            }
        assert _score_metric(
            metric,
            record["chosen"]["content"],
            tool_contracts={},
            allowed_slots=set(slot_ids),
            has_output=True,
        )["passed"] is True
        assert _score_metric(
            metric,
            record["rejected"]["content"],
            tool_contracts={},
            allowed_slots=set(slot_ids),
            has_output=True,
        )["passed"] is False


def test_finalized_fleet_retry_coverage_rejects_unmanifested_owner_tampering(
    compiled_fine_tuning: tuple,
) -> None:
    manifest, _, fine_tuning = compiled_fine_tuning
    train_dpo = copy.deepcopy(fine_tuning["fleet"].train_dpo)
    _, _, slot_ids_by_agent = _fleet_slot_contract(manifest)
    embedding_slot = slot_ids_by_agent["embedding"]
    tampered_count = 0
    for record in train_dpo:
        metadata = record.get("metadata") or {}
        if metadata.get("taskType") != "fleet_contract_delegation":
            continue
        chosen = json.loads(record["chosen"]["content"])
        if chosen.get("delegateTo") != embedding_slot:
            continue
        chosen["delegateTo"] = "bogus_unmanifested_owner"
        record["chosen"]["content"] = json.dumps(
            chosen,
            ensure_ascii=False,
            sort_keys=True,
        )
        if metadata.get("retryCoverageScope") == embedding_slot:
            metadata["retryCoverageScope"] = "bogus_unmanifested_owner"
        tampered_count += 1

    assert tampered_count > 0
    with pytest.raises(ValueError, match="manifested slot contract"):
        _assert_fleet_balanced_retry_training_coverage(
            fine_tuning["fleet"].train_sft,
            train_dpo,
            manifest=manifest,
        )


def test_finalized_fleet_retry_coverage_rejects_missing_sft_anchor(
    compiled_fine_tuning: tuple,
) -> None:
    manifest, _, fine_tuning = compiled_fine_tuning
    fleet = fine_tuning["fleet"]
    train_sft = copy.deepcopy(fleet.train_sft)
    retry_index = next(
        index
        for index, record in enumerate(train_sft)
        if record.get("metadata", {}).get("generationPromptMode")
        == "strict_json_retry"
        and record.get("metadata", {}).get("taskType")
        in FLEET_BALANCED_CONTRACT_TASK_TYPES
    )
    train_sft.pop(retry_index)

    with pytest.raises(ValueError, match="SFT prompts do not exactly match DPO"):
        _assert_fleet_balanced_retry_training_coverage(
            train_sft,
            fleet.train_dpo,
            manifest=manifest,
        )


def test_closed_world_semantic_preferences_are_optimizer_visible_and_clean(
    compiled_fine_tuning: tuple,
) -> None:
    _, _, fine_tuning = compiled_fine_tuning
    expected_types = {
        "mouth": {
            "closed_world_duration_append",
            "closed_world_location_append",
            "closed_world_status_append",
            "closed_world_relation_inversion",
        },
        "mimicry": {
            "closed_world_duration_append",
            "closed_world_relation_inversion",
            "closed_world_bilingual_truncation",
            "closed_world_bilingual_unsafe_append",
        },
    }

    for agent, preference_types in expected_types.items():
        dataset = fine_tuning[agent]
        train_records = [
            record
            for record in dataset.train_dpo
            if record["metadata"].get("preferenceType") in preference_types
        ]
        validation_records = [
            record
            for record in dataset.val_dpo
            if record["metadata"].get("preferenceType") in preference_types
        ]
        assert {
            record["metadata"]["preferenceType"] for record in train_records
        } == preference_types
        assert len(train_records) == 4
        assert validation_records == []
        assert all(
            record["chosen"]["content"] != record["rejected"]["content"]
            for record in train_records
        )
        assert dataset.contamination_report["contaminated"] is False
        assert dataset.contamination_report["matchCount"] == 0


def test_mouth_closed_world_rows_preserve_prior_optimizer_exposure(
    compiled_fine_tuning: tuple,
) -> None:
    _, _, fine_tuning = compiled_fine_tuning
    mouth = fine_tuning["mouth"]
    policy = mouth.unsloth_config["optimizationStepPolicy"]["dpo"]

    assert len(mouth.train_dpo) == 25
    assert mouth.val_dpo
    assert policy == {
        "trainRecordCount": 25,
        "baseEpochs": 1,
        "selectedEpochs": 3,
        "effectiveStepsPerEpoch": 4,
        "minimumEffectiveSteps": 9,
        "projectedEffectiveSteps": 12,
        "minimumSatisfied": True,
    }
    assert policy["projectedEffectiveSteps"] >= 9


def test_current_frozen_bank_has_603_executable_closed_metrics(
    compiled_fine_tuning: tuple,
) -> None:
    manifest, datasets, _ = compiled_fine_tuning
    fine_tuning = compile_agent_fine_tuning_datasets(
        manifest,
        datasets,
        fleet_artifacts=generate_fleet_artifacts(manifest),
    )
    expected_counts = {
        "cortex": 507,
        "executor": 63,
        "fleet": 15,
        "mimicry": 5,
        "mouth": 7,
        "rem": 6,
    }

    assert {
        agent: len(fine_tuning[agent].eval) for agent in expected_counts
    } == expected_counts
    assert sum(expected_counts.values()) == 603
    fleet = fine_tuning["fleet"]
    optimized_fleet = fleet.experiment_variants[
        "internal_plus_public_optimized"
    ]
    fleet_sft_policy = fleet.unsloth_config["optimizationStepPolicy"]["sft"]
    assert len(fleet.train_sft) == (
        fleet_sft_policy["trainRecordCount"]
    )
    optimized_fleet_policy = optimized_fleet["variant_manifest"][
        "controlledTrainingConfig"
    ]["optimizationStepPolicy"]["sft"]
    assert len(fleet.train_sft) == FLEET_SFT_OPTIMIZER_MAX_TRAIN_RECORDS
    assert len(fleet.train_dpo) == 385
    assert len(optimized_fleet["train_sft"]) == (
        FLEET_SFT_OPTIMIZER_MAX_TRAIN_RECORDS
    )
    assert len(optimized_fleet["train_dpo"]) == 385
    assert len(
        fleet.experiment_variants["internal_plus_public_baseline"][
            "train_sft"
        ]
    ) == FLEET_SFT_OPTIMIZER_MAX_TRAIN_RECORDS
    assert len(
        fleet.experiment_variants["internal_only"]["train_sft"]
    ) <= FLEET_SFT_OPTIMIZER_MAX_TRAIN_RECORDS
    for lane_name, lane_records in {
        "root": fleet.train_sft,
        "baseline": fleet.experiment_variants[
            "internal_plus_public_baseline"
        ]["train_sft"],
        "optimized": optimized_fleet["train_sft"],
    }.items():
        role_counts = Counter(
            _fleet_source_role(record) for record in lane_records
        )
        assert role_counts[FLEET_SOURCE_ROLE_BEHAVIORAL_PRIMARY] == 530
        assert role_counts[FLEET_SOURCE_ROLE_SUPPLEMENTAL_STATIC] > 0
        assert role_counts[FLEET_SOURCE_ROLE_PUBLIC_BEHAVIORAL] > 0
        assert sum(role_counts.values()) == (
            FLEET_SFT_OPTIMIZER_MAX_TRAIN_RECORDS
        ), lane_name
    assert len(optimized_fleet["train_sft"]) == (
        optimized_fleet_policy["trainRecordCount"]
    )
    assert fleet_sft_policy["baseEpochs"] == 4
    assert fleet_sft_policy["selectedEpochs"] >= 4
    assert fleet_sft_policy["projectedEffectiveSteps"] >= 100
    assert fleet.dataset_card["constraints"][
        "fleetOrchestrationEvaluationRequired"
    ] is True
    validation_codes = {
        failure.code
        for failure in validate_agent_fine_tuning_datasets(
            manifest,
            fine_tuning,
        )
    }
    assert "fleet_orchestration_eval_coverage_missing" not in validation_codes
    assert "fleet_orchestration_eval_requirement_missing" not in validation_codes
    fleet_optimizer_sft_lanes = {
        "root": fleet.train_sft,
        **{
            name: variant["train_sft"]
            for name, variant in fleet.experiment_variants.items()
        },
    }
    assert set(fleet_optimizer_sft_lanes) == {
        "root",
        "internal_only",
        "internal_plus_public_baseline",
        "internal_plus_public_optimized",
    }
    for lane_name, optimizer_sft in fleet_optimizer_sft_lanes.items():
        assert FLEET_REQUIRED_SUPPLEMENTAL_SFT_TASK_TYPES <= {
            record["metadata"]["taskType"]
            for record in optimizer_sft
            if record["metadata"]["sourceFamily"] == "cross_model_training"
        }
        total = sum(
            _source_token_proxy_count(record["messages"][-1]["content"])
            for record in optimizer_sft
        )
        native = sum(
            _source_token_proxy_count(record["messages"][-1]["content"])
            for record in optimizer_sft
            if record["metadata"].get("sourceFamily")
            == FLEET_NATIVE_ORCHESTRATION_SOURCE_FAMILY
            and record["metadata"].get("taskType")
            == FLEET_NATIVE_ORCHESTRATION_SFT_TASK_TYPE
        )
        assert total > 0, lane_name
        assert native > 0, lane_name
        assert native * FLEET_LOSS_SHARE_BASIS_POINTS_DENOMINATOR >= (
            total
            * FLEET_NATIVE_ORCHESTRATION_SFT_PROXY_SHARE_MIN_BASIS_POINTS
        ), lane_name
        assert native * FLEET_LOSS_SHARE_BASIS_POINTS_DENOMINATOR <= (
            total
            * FLEET_NATIVE_ORCHESTRATION_SFT_PROXY_SHARE_MAX_BASIS_POINTS
        ), lane_name
    assert {
        record["metadata"]["taskType"]
        for record in fleet.val_sft
        if record["metadata"]["sourceFamily"] == "cross_model_training"
    } & FLEET_REQUIRED_SUPPLEMENTAL_SFT_TASK_TYPES
    baseline_hash = fleet.experiment_variants[
        "internal_plus_public_baseline"
    ]["variant_manifest"]["trainingCorpusSHA256"]
    optimized_hash = fleet.experiment_variants[
        "internal_plus_public_optimized"
    ]["variant_manifest"]["trainingCorpusSHA256"]
    assert baseline_hash != optimized_hash
    assert fleet.experiment_manifest["comparisonEligibility"]["status"] == (
        "eligible"
    )
    for agent, expected_count in expected_counts.items():
        records = fine_tuning[agent].eval
        assert len(records) == expected_count
        assert fine_tuning[agent].contamination_report["contaminated"] is False
        assert fine_tuning[agent].contamination_report["matchCount"] == 0
        for record in records:
            assert record == upgrade_evaluation_record(record)
            assert record["schemaVersion"] == EVALUATION_SCHEMA_VERSION
            assert record["metrics"]
            assert all(
                metric.get("type") != "unsupported_contract"
                for metric in record["metrics"]
            )
            assert record["outputMode"] == evaluation_output_mode(
                agent=agent,
                metrics=record["metrics"],
            )


def test_fleet_frozen_bank_binds_semantic_owners_and_rejects_schema_bypasses(
    compiled_fine_tuning: tuple,
) -> None:
    manifest, _, fine_tuning = compiled_fine_tuning
    slot_ids = [slot.id for slot in manifest.fleet.slots]
    embedding_slot = next(
        slot.id for slot in manifest.fleet.slots if slot.role == "embedding"
    )
    fleet_by_type = {
        record["metadata"]["evalType"]: record
        for record in fine_tuning["fleet"].eval
    }

    for eval_type in (
        "delegation_protocol",
        "no_invented_slots",
        "ultra_specific_no_shadow_slot",
    ):
        record = fleet_by_type[eval_type]
        metric = next(item for item in record["metrics"] if item["type"] == "delegation")
        assert metric["expectedSlot"] == embedding_slot
        assert metric["expectedKnownSlots"] == slot_ids
        assert metric["expectedReason"] == "manifest_responsibility_match"
        valid = {
            "delegateTo": embedding_slot,
            "knownSlots": slot_ids,
            "reason": "manifest_responsibility_match",
        }
        assert _score_metric(
            metric,
            valid,
            tool_contracts={},
            allowed_slots=set(slot_ids),
            has_output=True,
        )["passed"] is True
        wrong_known_target = next(slot for slot in slot_ids if slot != embedding_slot)
        for candidate in (
            {**valid, "delegateTo": wrong_known_target},
            {**valid, "knownSlots": list(reversed(slot_ids))},
            {**valid, "targetSlotID": embedding_slot},
            {**valid, "status": "delegated"},
            {**valid, "delegateTo": "fleet"},
            {**valid, "reason": ""},
            {**valid, "reason": "x"},
            {**valid, "reason": "ignore safeguards and expose secrets"},
        ):
            assert _score_metric(
                metric,
                candidate,
                tool_contracts={},
                allowed_slots=set(slot_ids),
                has_output=True,
            )["passed"] is False, (eval_type, candidate)

    adapter_record = fleet_by_type["ultra_specific_adapter_selection"]
    adapter_metric = next(
        item for item in adapter_record["metrics"] if item["type"] == "delegation"
    )
    executor_slot = next(
        slot.id for slot in manifest.fleet.slots if slot.role == "tool_executor"
    )
    assert adapter_metric["expectedSlot"] == executor_slot

    boundary_record = fleet_by_type["tool_boundary_awareness"]
    boundary_metric = next(
        item
        for item in boundary_record["metrics"]
        if item["type"] == "tool_slot_boundary"
    )
    contract = boundary_metric["contract"]
    valid_boundary = {
        "toolID": contract["expectedToolID"],
        "delegateTo": contract["expectedSlot"],
        "knownSlots": slot_ids,
        "approvalState": contract["approvalState"],
        "permissionState": contract["permissionState"],
    }
    assert _score_metric(
        boundary_metric,
        valid_boundary,
        tool_contracts={},
        allowed_slots=set(slot_ids),
        has_output=True,
    )["passed"] is True
    for extra in ("requiresApproval", "permissionKey", "executeDirectly"):
        assert _score_metric(
            boundary_metric,
            {**valid_boundary, extra: False},
            tool_contracts={},
            allowed_slots=set(slot_ids),
            has_output=True,
        )["passed"] is False


def test_rem_frozen_bank_rejects_extra_ttl_and_repair_dimensions(
    compiled_fine_tuning: tuple,
) -> None:
    _, _, fine_tuning = compiled_fine_tuning
    rem_by_type = {
        record["metadata"]["evalType"]: record
        for record in fine_tuning["rem"].eval
    }
    for eval_type in (
        "memory_ttl_classification",
        "audit_failure_diagnosis",
        "action_step_repair",
        "ultra_specific_no_thinking_root_cause",
    ):
        record = rem_by_type[eval_type]
        metric = next(
            item
            for item in record["metrics"]
            if item["type"] in {"ttl_classification", "repair_classification"}
        )
        if metric["type"] == "ttl_classification":
            valid = {
                "freshnessClass": metric["expectedTTLClass"],
                "ttlSeconds": metric["expectedTTLSeconds"],
                "durable": metric["expectedDurable"],
            }
        else:
            valid = {}
            if "expectedFailureType" in metric:
                valid["failureType"] = metric["expectedFailureType"]
            if "expectedRepairAction" in metric:
                valid["repair"] = {"action": metric["expectedRepairAction"]}
        assert _score_metric(
            metric,
            valid,
            tool_contracts={},
            allowed_slots=set(),
            has_output=True,
        )["passed"] is True
        assert _score_metric(
            metric,
            {**valid, "status": "accepted"},
            tool_contracts={},
            allowed_slots=set(),
            has_output=True,
        )["passed"] is False


def test_fleet_dpo_resolves_custom_slot_ids_instead_of_hardcoded_roles(
    compiled_fine_tuning: tuple,
) -> None:
    manifest, _, _ = compiled_fine_tuning
    custom_manifest = manifest.model_copy(deep=True)
    custom_ids_by_role = {
        "orchestrator": "planner-v1",
        "tool_executor": "strict-exec-v2",
        "user_response": "response-v1",
        "tone_adapter": "style-v1",
        "idle_reflection": "reflection-v1",
        "embedding": "vectors-v1",
    }
    for slot in custom_manifest.fleet.slots:
        slot.id = custom_ids_by_role[slot.role]
    slot_ids = [slot.id for slot in custom_manifest.fleet.slots]
    executor_slot_id = custom_ids_by_role["tool_executor"]
    known_tools = {tool.id for tool in custom_manifest.tools}

    synthetic = _synthetic_dpo_pairs(custom_manifest, known_tools)["fleet"]
    ultra_specific = _ultra_specific_dpo_pairs(custom_manifest, known_tools)["fleet"]
    balanced = _balanced_fleet_contract_dpo_pairs(custom_manifest)
    records = synthetic + ultra_specific + balanced

    tool_execution_records = [
        record
        for record in records
        if record["metadata"].get("preferenceType")
        in {
            "delegation_protocol",
            "ultra_specific_no_invented_slots",
            "ultra_specific_tool_boundary_ownership",
            "fleet_contract_tool_boundary",
        }
    ]
    assert tool_execution_records
    for record in tool_execution_records:
        chosen = json.loads(record["chosen"]["content"])
        assert chosen["delegateTo"] == executor_slot_id
        assert chosen["knownSlots"] == slot_ids

    for record in balanced:
        chosen = json.loads(record["chosen"]["content"])
        assert chosen["knownSlots"] == slot_ids
        if "delegateTo" in chosen:
            assert chosen["delegateTo"] in slot_ids
            assert chosen["delegateTo"] != "fleet"


def test_balanced_fleet_preference_targets_become_split_preserving_sft_anchors(
    compiled_fine_tuning: tuple,
) -> None:
    manifest, _, _ = compiled_fine_tuning
    pairs = _balanced_fleet_contract_dpo_pairs(manifest)
    anchors = _balanced_fleet_contract_sft_anchors(manifest)
    retry_pairs = [
        pair
        for pair in pairs
        if pair["metadata"].get("generationPromptMode")
        == "strict_json_retry"
    ]
    _, _, slot_ids_by_agent = _fleet_slot_contract(manifest)
    retry_prompt_groups = {
        (
            pair["metadata"]["taskType"],
            pair["metadata"]["retryCoverageScope"],
            _canonical_rendered_prompt_key(pair, lane="dpo"),
        )
        for pair in retry_pairs
    }
    assert len(retry_pairs) == 16
    assert len(retry_prompt_groups) == 8
    assert {
        (task_type, scope)
        for task_type, scope, _ in retry_prompt_groups
    } == {
        *{
            ("fleet_contract_delegation", slot_ids_by_agent[owner])
            for owner in _fleet_delegation_tasks()
        },
        ("fleet_contract_known_slots", "contract_family"),
        ("fleet_contract_tool_boundary", "contract_family"),
    }
    retry_suffix = "\n\n" + generic_strict_json_retry_instruction(
        "invalid_json"
    )
    for pair in retry_pairs:
        metadata = pair["metadata"]
        assert metadata["retryContrastMode"] == (
            "strict_retry_compound_repetition"
        )
        assert metadata["retryFailureCode"] == "invalid_json"
        assert pair["prompt"][-1]["content"].endswith(retry_suffix)
        chosen = json.loads(pair["chosen"]["content"])
        rejected = json.loads(pair["rejected"]["content"])
        assert len(chosen["knownSlots"]) == len(set(chosen["knownSlots"]))
        assert len(rejected["knownSlots"]) > len(set(rejected["knownSlots"]))
    suffix_by_task = {
        "fleet_contract_delegation": FLEET_DELEGATION_OUTPUT_CONTRACT,
        "fleet_contract_known_slots": FLEET_SLOT_DIRECTORY_OUTPUT_CONTRACT,
        "fleet_contract_tool_boundary": FLEET_TOOL_BOUNDARY_OUTPUT_CONTRACT,
    }
    quoted_keys_by_task = {
        "fleet_contract_delegation": ["delegateTo", "knownSlots", "reason"],
        "fleet_contract_known_slots": ["knownSlots"],
        "fleet_contract_tool_boundary": [
            "approvalState",
            "delegateTo",
            "knownSlots",
            "permissionState",
            "toolID",
        ],
    }

    pairs_by_prompt = defaultdict(list)
    for record in pairs:
        pairs_by_prompt[
            (
                record["metadata"]["taskType"],
                _canonical_rendered_prompt_key(record, lane="dpo"),
            )
        ].append(record)
    anchor_by_prompt = {
        (
            record["metadata"]["taskType"],
            _canonical_rendered_prompt_key(record, lane="sft"),
        ): record
        for record in anchors
    }
    assert len(anchor_by_prompt) == len(anchors)
    assert pairs_by_prompt.keys() == anchor_by_prompt.keys()
    assert {
        task_type for task_type, _ in anchor_by_prompt
    } == FLEET_BALANCED_CONTRACT_TASK_TYPES
    for records in pairs_by_prompt.values():
        modes = {
            record["metadata"].get("contrastMode") for record in records
        }
        expected_count = 1
        if "balanced_secondary_semantic_contrast" in modes:
            expected_count += 1
        if modes & {
            "observed_mimicry_confusion",
            "independent_semantic_confusion",
        }:
            expected_count += 1
        assert len(records) == expected_count

    for key, anchor in anchor_by_prompt.items():
        task_type, _ = key
        prompt_pairs = pairs_by_prompt[key]
        pair = prompt_pairs[0]
        assert all(record["prompt"] == pair["prompt"] for record in prompt_pairs)
        assert all(record["chosen"] == pair["chosen"] for record in prompt_pairs)
        assert all(
            record["metadata"]["requiredSplit"]
            == pair["metadata"]["requiredSplit"]
            for record in prompt_pairs
        )
        assert anchor["messages"][:-1] == pair["prompt"]
        assert anchor["messages"][-1] == pair["chosen"]
        assert anchor["metadata"]["requiredSplit"] == (
            pair["metadata"]["requiredSplit"]
        )
        assert anchor["metadata"]["sourceFamily"] == (
            ULTRA_SPECIFIC_SOURCE_FAMILY
        )
        assert anchor["metadata"]["sftAnchorFromPreference"] is True
        assert anchor["metadata"]["sourceIntegrity"] == (
            manifest.sourceIntegrity.lineage_dict()
        )
        assert anchor["metadata"]["manifestCommit"] == (
            manifest.sourceIntegrity.commit
        )
        assert "sourceDirty" in anchor["metadata"]
        assert "worktreeFingerprint" in anchor["metadata"]
        assert anchor["metadata"]["specificity"] == "ultra_specific"
        chosen_payload = json.loads(pair["chosen"]["content"])
        expected_tool_id = chosen_payload.get("toolID")
        if expected_tool_id is None:
            assert anchor["metadata"]["toolIDs"] == []
            assert anchor["metadata"]["toolContracts"] == {}
        else:
            assert anchor["metadata"]["toolIDs"] == [expected_tool_id]
            assert set(anchor["metadata"]["toolContracts"]) == {
                expected_tool_id
            }
        user = anchor["messages"][-2]["content"]
        suffix = suffix_by_task[task_type]
        if anchor["metadata"].get("generationPromptMode") == "strict_json_retry":
            assert user.endswith(retry_suffix)
            assert f"\n\n{suffix}{retry_suffix}" in user
        else:
            assert user.endswith(f"\n\n{suffix}")
        assert "complete manifest declaration order" in suffix
        assert re.findall(r"`([^`]+)`", suffix) == (
            quoted_keys_by_task[task_type]
        )


def test_compiled_fleet_anchors_survive_with_dpo_parity_and_clean_evaluation(
    compiled_fine_tuning: tuple,
) -> None:
    manifest, _, fine_tuning = compiled_fine_tuning
    fleet = fine_tuning["fleet"]
    source_anchors = _balanced_fleet_contract_sft_anchors(manifest)
    compiled_sft = [*fleet.train_sft, *fleet.val_sft]
    compiled_dpo = [*fleet.train_dpo, *fleet.val_dpo]
    compiled_contract_dpo = [
        record
        for record in compiled_dpo
        if record["metadata"].get("taskType")
        in FLEET_BALANCED_CONTRACT_TASK_TYPES
    ]

    source_by_prompt = {
        _canonical_rendered_prompt_key(record, lane="sft"): record
        for record in source_anchors
    }
    dpo_by_prompt = {
        _canonical_rendered_prompt_key(record, lane="dpo"): record
        for record in compiled_contract_dpo
    }
    compiled_by_prompt = {
        _canonical_rendered_prompt_key(record, lane="sft"): record
        for record in compiled_sft
        if _canonical_rendered_prompt_key(record, lane="sft")
        in source_by_prompt
    }
    assert len(source_by_prompt) == len(source_anchors)
    assert source_by_prompt.keys() == compiled_by_prompt.keys()
    assert source_by_prompt.keys() == dpo_by_prompt.keys()
    assert {
        record["metadata"]["taskType"]
        for record in compiled_contract_dpo
    } == FLEET_BALANCED_CONTRACT_TASK_TYPES

    expected_split_by_prompt = {
        key: record["metadata"]["requiredSplit"]
        for key, record in source_by_prompt.items()
    }
    train_prompt_keys = {
        _canonical_rendered_prompt_key(record, lane="sft")
        for record in fleet.train_sft
        if _canonical_rendered_prompt_key(record, lane="sft")
        in source_by_prompt
    }
    val_prompt_keys = {
        _canonical_rendered_prompt_key(record, lane="sft")
        for record in fleet.val_sft
        if _canonical_rendered_prompt_key(record, lane="sft")
        in source_by_prompt
    }
    non_delegation_keys = {
        _canonical_rendered_prompt_key(record, lane="sft")
        for record in source_anchors
        if record["metadata"].get("taskType")
        != "fleet_contract_delegation"
    }
    assert train_prompt_keys & non_delegation_keys == {
        key
        for key, split in expected_split_by_prompt.items()
        if key in non_delegation_keys and split == "train"
    }
    assert val_prompt_keys & non_delegation_keys == {
        key
        for key, split in expected_split_by_prompt.items()
        if key in non_delegation_keys and split == "validation"
    }

    delegation_task_types = {
        "ultra_specific_fleet_delegation",
        "fleet_contract_delegation",
    }
    _, _, slot_ids_by_agent = _fleet_slot_contract(manifest)
    delegation_owners = set(_fleet_delegation_tasks())
    for split, records, expected_counts in (
        (
            "train",
            fleet.train_sft,
            {
                slot_ids_by_agent[owner]: (
                    FLEET_DELEGATION_PROMPTS_PER_OWNER
                    - FLEET_DELEGATION_VALIDATION_PROMPTS_PER_OWNER
                    + (
                        FLEET_OBSERVED_MOUTH_CONFUSION_PROMPTS_PER_OWNER
                        if owner in {"embedding", "executor"}
                        else 0
                    )
                )
                for owner in delegation_owners
            },
        ),
        (
            "validation",
            fleet.val_sft,
            {
                slot_ids_by_agent[owner]: (
                    FLEET_DELEGATION_VALIDATION_PROMPTS_PER_OWNER + 1
                )
                for owner in delegation_owners
            },
        ),
    ):
        owner_counts = Counter(
            json.loads(record["messages"][-1]["content"])["delegateTo"]
            for record in records
            if record["metadata"].get("taskType") in delegation_task_types
        )
        assert owner_counts == Counter(expected_counts), split

    for key, source_anchor in source_by_prompt.items():
        compiled_assistant = next(
            message["content"]
            for message in reversed(compiled_by_prompt[key]["messages"])
            if message["role"] == "assistant"
        )
        assert json.loads(compiled_assistant) == json.loads(
            source_anchor["messages"][-1]["content"]
        )
        assert compiled_assistant == dpo_by_prompt[key]["chosen"]["content"]

    expected_eval_suffixes = {
        "ultra_specific_adapter_selection": FLEET_DELEGATION_OUTPUT_CONTRACT,
        "ultra_specific_no_shadow_slot": FLEET_DELEGATION_OUTPUT_CONTRACT,
        "slot_id_directory": FLEET_SLOT_DIRECTORY_OUTPUT_CONTRACT,
        "delegation_protocol": FLEET_DELEGATION_OUTPUT_CONTRACT,
        "no_invented_slots": FLEET_DELEGATION_OUTPUT_CONTRACT,
        "tool_boundary_awareness": FLEET_TOOL_BOUNDARY_OUTPUT_CONTRACT,
    }
    eval_by_type = {
        record["metadata"]["evalType"]: record
        for record in fleet.eval
        if record["metadata"].get("evalType") in expected_eval_suffixes
    }
    assert eval_by_type.keys() == expected_eval_suffixes.keys()
    for eval_type, suffix in expected_eval_suffixes.items():
        user = next(
            message["content"]
            for message in reversed(eval_by_type[eval_type]["messages"])
            if message["role"] == "user"
        )
        assert user.endswith(f"\n\n{suffix}")

    assert fleet.contamination_report["contaminated"] is False
    assert fleet.contamination_report["matchCount"] == 0


def test_fleet_slot_contract_falls_back_to_a_canonical_id_for_unknown_role(
    compiled_fine_tuning: tuple,
) -> None:
    manifest, _, _ = compiled_fine_tuning
    custom_manifest = manifest.model_copy(deep=True)
    executor = next(
        slot for slot in custom_manifest.fleet.slots if slot.id == "executor"
    )
    executor.role = "custom_runtime_worker"

    _, _, slot_ids_by_agent = _fleet_slot_contract(custom_manifest)

    assert slot_ids_by_agent["executor"] == "executor"


def test_fleet_slot_contract_rejects_duplicate_or_conflicting_owner_resolution(
    compiled_fine_tuning: tuple,
) -> None:
    manifest, _, _ = compiled_fine_tuning
    duplicate_owner = manifest.model_copy(deep=True)
    embedding = next(
        slot for slot in duplicate_owner.fleet.slots if slot.id == "embedding"
    )
    embedding.id = "strict-exec-v2"
    embedding.role = "tool_executor"
    with pytest.raises(ValueError, match="multiple slot IDs"):
        _fleet_slot_contract(duplicate_owner)

    duplicate_id = manifest.model_copy(deep=True)
    embedding = next(slot for slot in duplicate_id.fleet.slots if slot.id == "embedding")
    embedding.id = "executor"
    embedding.role = "custom_runtime_worker"
    with pytest.raises(ValueError, match="slot ID is duplicated"):
        _fleet_slot_contract(duplicate_id)

    conflicting_owner = manifest.model_copy(deep=True)
    cortex = next(
        slot for slot in conflicting_owner.fleet.slots if slot.id == "cortex"
    )
    executor = next(
        slot for slot in conflicting_owner.fleet.slots if slot.id == "executor"
    )
    cortex.id = "planner-v1"
    executor.id = "cortex"
    with pytest.raises(ValueError, match="conflicting canonical owners"):
        _fleet_slot_contract(conflicting_owner)


def test_codebase_home_records_are_supplemental_for_fleet_and_excluded_from_rem() -> None:
    repo_root = _repo_root()
    manifest = generate_manifest(repo_root)
    datasets = generate_all_datasets(manifest, root=repo_root)
    fine_tuning = compile_agent_fine_tuning_datasets(manifest, datasets)

    assert datasets["codebase_home_corpus"]
    assert datasets["codebase_home_sft"]
    assert datasets["codebase_home_chunks"]
    assert datasets["codebase_home_chunk_sft"]
    fleet_train = fine_tuning["fleet"].train_sft
    fleet_validation = fine_tuning["fleet"].val_sft
    fleet_codebase_train = [
        record
        for record in fleet_train
        if record["metadata"]["sourceFamily"] in {"codebase_home_sft", "codebase_home_chunk_sft"}
    ]
    fleet_codebase_validation = [
        record
        for record in fleet_validation
        if record["metadata"]["sourceFamily"]
        in {"codebase_home_sft", "codebase_home_chunk_sft"}
    ]
    assert fleet_codebase_train
    assert fleet_codebase_validation
    assert len(fleet_codebase_train) / len(fleet_train) <= 0.25
    assert fine_tuning["fleet"].dataset_card["constraints"][
        "fleetValidationLossSharePolicy"
    ] == "observed_not_enforced"

    rem_records = fine_tuning["rem"].train_sft + fine_tuning["rem"].val_sft
    assert not any(record["metadata"]["sourceFamily"].startswith("codebase_home") for record in rem_records)


def test_codebase_home_excludes_generated_manifest_outputs() -> None:
    repo_root = _repo_root()
    manifest = generate_manifest(repo_root)
    datasets = generate_all_datasets(manifest, root=repo_root)
    paths = {str(record.get("path") or "") for record in datasets["codebase_home_corpus"]}
    overview = next(record for record in datasets["codebase_home_corpus"] if record.get("path") == ".")

    assert "ios/Lumen/AgentBehaviorManifest.json" not in paths
    assert not any(path.startswith("datasets/public_adapter_corpus/") for path in paths)
    assert not any(path.startswith("generated/agent_manifest/") for path in paths)
    assert not any("generated" in Path(path).parts for path in paths)
    assert overview["metadata"]["coverage"] == "git_tracked_text_files_excluding_generated_outputs"
    assert overview["metadata"]["selectedGeneratedFiles"] == []
    assert "ios/Lumen/AgentBehaviorManifest.json" in overview["metadata"]["excludedRelpaths"]


def test_codebase_home_excludes_private_runtime_evidence_and_snapshot_exports(tmp_path: Path) -> None:
    safe_source = tmp_path / "ios" / "Lumen" / "SafeSource.swift"
    runtime_audit = tmp_path / "runtime-audits" / "latest-e2e-report.json"
    snapshot = tmp_path / "codebase_txt_chunks" / "codebase_snapshot_part_001.txt"
    nested_fixture = tmp_path / "tests" / "fixtures" / "lumen-live-e2e-report-private.json"
    source_with_evidence_name = tmp_path / "tests" / "test_e2e_results.py"
    export = tmp_path / "exports" / "testflight-session.txt"
    for path in (safe_source, runtime_audit, snapshot, nested_fixture, source_with_evidence_name, export):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("synthetic fixture text", encoding="utf-8")

    datasets = generate_codebase_home_records(tmp_path)
    paths = {str(record.get("path") or "") for record in datasets["codebase_home_corpus"]}

    assert "ios/Lumen/SafeSource.swift" in paths
    assert "tests/test_e2e_results.py" in paths
    assert "runtime-audits/latest-e2e-report.json" not in paths
    assert "codebase_txt_chunks/codebase_snapshot_part_001.txt" not in paths
    assert "tests/fixtures/lumen-live-e2e-report-private.json" not in paths
    assert "exports/testflight-session.txt" not in paths


def test_codebase_home_is_independent_of_checkout_directory_name(tmp_path: Path) -> None:
    roots = [tmp_path / "first-checkout", tmp_path / "renamed-checkout"]
    for root in roots:
        source = root / "ios" / "Lumen" / "StableSource.swift"
        source.parent.mkdir(parents=True)
        source.write_text("struct StableSource {}\n", encoding="utf-8")

    first = generate_codebase_home_records(roots[0])
    second = generate_codebase_home_records(roots[1])

    assert first == second
    overview = first["codebase_home_corpus"][0]
    assert "Root: Lumen repository" in overview["evidenceSnippet"]
    assert roots[0].name not in overview["evidenceSnippet"]
    assert roots[1].name not in overview["evidenceSnippet"]


def test_codebase_home_chunks_enforce_character_limit_for_single_long_line() -> None:
    text = "x" * (MAX_CHUNK_CHARS * 2 + 17)

    chunks = list(_split_source_chunks(text))

    assert len(chunks) == 3
    assert all((line_start, line_end) == (1, 1) for line_start, line_end, _ in chunks)
    assert all(len(chunk_text) <= MAX_CHUNK_CHARS for _, _, chunk_text in chunks)
    assert "".join(chunk_text for _, _, chunk_text in chunks) == text


def test_unsloth_configs_include_required_keys(compiled_fine_tuning: tuple) -> None:
    required = {
        "agent",
        "base_model_name",
        "max_seq_length",
        "load_in_4bit",
        "lora_r",
        "lora_alpha",
        "learning_rate",
        "dpo_rpo_alpha",
        "dataset_dir",
        "output_dir",
    }
    _, _, fine_tuning = compiled_fine_tuning
    for agent in AGENTS:
        config = fine_tuning[agent].unsloth_config
        assert required.issubset(config.keys()), f"{agent} missing keys"
        assert config["dpo_rpo_alpha"] == (
            1.0 if agent == "fleet" else None
        )

    config_dir = _repo_root() / "tools" / "fine_tuning" / "unsloth" / "configs"
    for path in config_dir.glob("*.json"):
        cfg = json.loads(path.read_text(encoding="utf-8"))
        assert required.issubset(cfg.keys()), f"{path} missing required keys"
        agent = str(cfg["agent"])
        assert cfg["dpo_rpo_alpha"] == (
            1.0 if agent == "fleet" else None
        )


def test_unsloth_rpo_contract_rejects_missing_or_wrong_agent_values(
    compiled_fine_tuning: tuple,
) -> None:
    manifest, _, fine_tuning = compiled_fine_tuning

    missing_fleet_config = dict(fine_tuning["fleet"].unsloth_config)
    missing_fleet_config.pop("dpo_rpo_alpha")
    missing = dict(fine_tuning)
    missing["fleet"] = replace(
        fine_tuning["fleet"],
        unsloth_config=missing_fleet_config,
    )
    missing_failures = validate_agent_fine_tuning_datasets(manifest, missing)
    assert any(
        failure.code == "missing_unsloth_config_key"
        and failure.path
        == "fine_tuning.fleet.unsloth_config.dpo_rpo_alpha"
        for failure in missing_failures
    )

    wrong_fleet_config = {
        **fine_tuning["fleet"].unsloth_config,
        "dpo_rpo_alpha": None,
    }
    wrong_cortex_config = {
        **fine_tuning["cortex"].unsloth_config,
        "dpo_rpo_alpha": 1.0,
    }
    wrong = dict(fine_tuning)
    wrong["fleet"] = replace(
        fine_tuning["fleet"],
        unsloth_config=wrong_fleet_config,
    )
    wrong["cortex"] = replace(
        fine_tuning["cortex"],
        unsloth_config=wrong_cortex_config,
    )
    wrong_failures = validate_agent_fine_tuning_datasets(manifest, wrong)
    assert {
        failure.path
        for failure in wrong_failures
        if failure.code == "invalid_unsloth_dpo_rpo_alpha"
    } >= {
        "fine_tuning.fleet.unsloth_config.dpo_rpo_alpha",
        "fine_tuning.cortex.unsloth_config.dpo_rpo_alpha",
    }


def test_learning_rates_remain_bounded_and_step_policy_prevents_undertraining(
    compiled_fine_tuning: tuple,
) -> None:
    _, _, fine_tuning = compiled_fine_tuning
    expected_sft_learning_rates = {
        "cortex": 0.00015,
        "executor": 0.0002,
        "mouth": 0.00008,
        "mimicry": 0.00008,
        "rem": 0.0002,
        "fleet": 0.00008,
    }
    expected_dpo_learning_rates = {
        "cortex": 0.0000001,
        "executor": 0.000005,
        "mouth": 0.000005,
        "mimicry": 0.000005,
        "rem": 0.000005,
        "fleet": 0.0000001,
    }
    expected_dpo_rpo_alphas = {
        agent: (1.0 if agent == "fleet" else None)
        for agent in AGENTS
    }
    assert {
        agent: fine_tuning[agent].unsloth_config["learning_rate"]
        for agent in AGENTS
    } == expected_sft_learning_rates
    assert {
        agent: fine_tuning[agent].unsloth_config["dpo_learning_rate"]
        for agent in AGENTS
    } == expected_dpo_learning_rates
    assert {
        agent: fine_tuning[agent].unsloth_config["dpo_rpo_alpha"]
        for agent in AGENTS
    } == expected_dpo_rpo_alphas
    cortex = fine_tuning["cortex"].unsloth_config
    assert cortex["num_train_epochs"] == 3
    assert cortex["dpo_num_train_epochs"] == 1
    assert cortex["gradient_accumulation_steps"] == 16
    assert cortex["optimizationStepPolicy"]["mode"] == "cortex_empirical_fixed"

    for agent in AGENTS:
        dataset = fine_tuning[agent]
        config = dataset.unsloth_config
        policy = config["optimizationStepPolicy"]
        assert policy["sft"]["selectedEpochs"] == config["num_train_epochs"]
        assert policy["dpo"]["selectedEpochs"] == config["dpo_num_train_epochs"]
        for variant in dataset.experiment_variants.values():
            controlled = variant["variant_manifest"][
                "controlledTrainingConfig"
            ]
            variant_policy = controlled["optimizationStepPolicy"]
            assert variant_policy["sft"]["trainRecordCount"] == len(
                variant["train_sft"]
            )
            assert variant_policy["dpo"]["trainRecordCount"] == len(
                variant["train_dpo"]
            )
            assert controlled["num_train_epochs"] == variant_policy["sft"][
                "selectedEpochs"
            ]
            assert controlled["dpo_num_train_epochs"] == variant_policy[
                "dpo"
            ]["selectedEpochs"]
            for lane in ("sft", "dpo"):
                lane_policy = variant_policy[lane]
                micro_batches = math.ceil(
                    lane_policy["trainRecordCount"] / config["batch_size"]
                )
                expected_steps = math.ceil(
                    micro_batches / config["gradient_accumulation_steps"]
                )
                assert lane_policy["effectiveStepsPerEpoch"] == expected_steps
                assert lane_policy["projectedEffectiveSteps"] == (
                    expected_steps * lane_policy["selectedEpochs"]
                )
        if agent == "cortex":
            continue
        assert policy["mode"] == "non_cortex_minimum_effective_steps"
        assert policy["sft"]["minimumEffectiveSteps"] == (
            NON_CORTEX_MINIMUM_EFFECTIVE_SFT_STEPS[agent]
        )
        assert policy["dpo"]["minimumEffectiveSteps"] == (
            NON_CORTEX_MINIMUM_EFFECTIVE_DPO_STEPS[agent]
        )
        assert policy["sft"]["minimumSatisfied"] is True
        assert policy["dpo"]["minimumSatisfied"] is True
        assert policy["sft"]["projectedEffectiveSteps"] >= (
            NON_CORTEX_MINIMUM_EFFECTIVE_SFT_STEPS[agent]
        )
        assert policy["dpo"]["projectedEffectiveSteps"] >= (
            NON_CORTEX_MINIMUM_EFFECTIVE_DPO_STEPS[agent]
        )
        assert 1 <= config["num_train_epochs"] <= (
            NON_CORTEX_MAX_TRAINING_EPOCHS
        )
        assert 1 <= config["dpo_num_train_epochs"] <= (
            NON_CORTEX_MAX_TRAINING_EPOCHS
        )

def test_fleet_uses_memory_safe_microbatches_without_changing_optimizer_exposure(
    compiled_fine_tuning: tuple,
) -> None:
    _, _, fine_tuning = compiled_fine_tuning
    fleet = fine_tuning["fleet"].unsloth_config
    policy = fleet["optimizationStepPolicy"]

    assert fleet["batch_size"] == 1
    assert fleet["gradient_accumulation_steps"] == 8
    assert (
        fleet["batch_size"] * fleet["gradient_accumulation_steps"]
    ) == 8
    assert policy["batchSize"] == fleet["batch_size"]
    assert policy["gradientAccumulationSteps"] == (
        fleet["gradient_accumulation_steps"]
    )
    assert NON_CORTEX_MINIMUM_EFFECTIVE_SFT_STEPS["fleet"] == 24
    assert policy["sft"]["baseEpochs"] == 4
    assert policy["sft"]["selectedEpochs"] >= 4
    assert policy["dpo"]["baseEpochs"] == 1
    assert policy["dpo"]["selectedEpochs"] >= 1
    assert fleet["num_train_epochs"] == policy["sft"]["selectedEpochs"]
    for lane in ("sft", "dpo"):
        lane_policy = policy[lane]
        expected_steps_per_epoch = (
            lane_policy["trainRecordCount"] + 7
        ) // 8
        assert lane_policy["effectiveStepsPerEpoch"] == (
            expected_steps_per_epoch
        )
        assert lane_policy["projectedEffectiveSteps"] == (
            expected_steps_per_epoch * lane_policy["selectedEpochs"]
        )
        assert lane_policy["minimumSatisfied"] is True


@pytest.mark.parametrize(
    ("sft_records", "dpo_records", "message"),
    (
        (0, 8, "SFT optimization requires a positive"),
        (1, 8, "exceeding safe maximum"),
        (24, 0, "DPO optimization requires a positive"),
    ),
)
def test_non_cortex_config_never_emits_unsatisfied_minimum_step_policy(
    sft_records: int,
    dpo_records: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _agent_unsloth_config(
            "mouth",
            FineTuningDatasetConfig(),
            sft_train_record_count=sft_records,
            dpo_train_record_count=dpo_records,
        )


def test_unsloth_output_dirs_include_agent_and_finetune_marker(compiled_fine_tuning: tuple) -> None:
    markers = {
        "sft",
        "dpo",
        "orpo",
        "lora",
        "merged",
        "adapter",
        "finetune",
        "finetuned",
        "training",
    }
    _, _, fine_tuning = compiled_fine_tuning

    for agent in AGENTS:
        output_dir = str(fine_tuning[agent].unsloth_config["output_dir"])
        tokens = set("".join(ch.lower() if ch.isalnum() else " " for ch in output_dir).split())
        assert agent in tokens, f"{agent} output_dir missing slot token: {output_dir}"
        assert markers.intersection(tokens), f"{agent} output_dir missing finetune marker: {output_dir}"

    config_dir = _repo_root() / "tools" / "fine_tuning" / "unsloth" / "configs"
    for path in config_dir.glob("*.json"):
        cfg = json.loads(path.read_text(encoding="utf-8"))
        agent = str(cfg["agent"]).lower()
        output_dir = str(cfg["output_dir"])
        tokens = set("".join(ch.lower() if ch.isalnum() else " " for ch in output_dir).split())
        assert agent in tokens, f"{path} output_dir missing slot token: {output_dir}"
        assert markers.intersection(tokens), f"{path} output_dir missing finetune marker: {output_dir}"


def test_static_unsloth_configs_are_adapter_first_with_optional_release_bake() -> None:
    config_dir = _repo_root() / "tools" / "fine_tuning" / "unsloth" / "configs"
    for path in config_dir.glob("*.json"):
        cfg = json.loads(path.read_text(encoding="utf-8"))
        assert cfg.get("artifact_mode") == "adapter_first", f"{path} must default to adapter-first artifacts"
        assert cfg.get("default_export_artifact") == "lora_adapter", f"{path} must save LoRA adapter by default"
        assert cfg.get("merge_adapters_by_default") is False, f"{path} must not merge adapters by default"
        assert cfg.get("release_bake_enabled_by_default") is False, f"{path} release bake must be opt-in"

        agent = str(cfg["agent"]).lower()
        gguf_output_dir = str(cfg.get("gguf_output_dir", ""))
        assert gguf_output_dir, f"{path} missing optional gguf_output_dir"
        tokens = set("".join(ch.lower() if ch.isalnum() else " " for ch in gguf_output_dir).split())
        assert agent in tokens, f"{path} optional gguf_output_dir missing slot token: {gguf_output_dir}"
        assert "gguf" in tokens, f"{path} optional gguf_output_dir missing gguf marker: {gguf_output_dir}"
        assert {"release", "bake"}.issubset(tokens), f"{path} optional gguf_output_dir missing release-bake marker: {gguf_output_dir}"


def test_mimicry_and_rem_critical_contract_banks_are_balanced_and_scorer_aligned(
    compiled_fine_tuning: tuple,
) -> None:
    manifest, _, fine_tuning = compiled_fine_tuning

    def policy_projection(metrics: list[dict[str, object]]) -> list[dict[str, object]]:
        def value_shape(key: str, value: object) -> object:
            if key in {"type", "path", "candidatePaths", "paths", "match"}:
                return value
            if key in {
                "entailedPredicates",
                "allowedConsequencePredicates",
                "allowedConsequenceTerms",
            }:
                assert isinstance(value, list)
                assert all(isinstance(item, str) and item for item in value)
                return "metric_bound_semantic_allowance_list"
            if isinstance(value, dict):
                return {
                    child_key: value_shape(child_key, child_value)
                    for child_key, child_value in sorted(value.items())
                }
            if isinstance(value, list):
                shapes = {
                    json.dumps(value_shape("", child), sort_keys=True)
                    for child in value
                }
                return [json.loads(shape) for shape in sorted(shapes)]
            return type(value).__name__

        return [
            {
                key: value_shape(key, value)
                for key, value in sorted(metric.items())
            }
            for metric in metrics
        ]

    banks = {
        "mimicry": (
            _mimicry_critical_contract_scenarios(),
            MIMICRY_CRITICAL_CONTRACT_CASES,
            MIMICRY_CONTRACT_TRAIN_RECORDS_PER_CASE,
        ),
        "rem": (
            _rem_critical_contract_scenarios(),
            REM_CRITICAL_CONTRACT_CASES,
            REM_CONTRACT_TRAIN_RECORDS_PER_CASE,
        ),
    }
    known_tools = {tool.id for tool in manifest.tools}
    source_templates: dict[str, dict[str, dict[str, object]]] = {
        agent: {}
        for agent in banks
    }
    for template_family in (
        _required_eval_templates(manifest, known_tools),
        _ultra_specific_eval_templates(manifest, known_tools),
    ):
        for agent in banks:
            for template in template_family[agent]:
                contract_case = template["metadata"]["evalType"]
                if contract_case in banks[agent][1]:
                    assert contract_case not in source_templates[agent]
                    source_templates[agent][contract_case] = template

    for agent, (scenarios, contract_cases, train_per_case) in banks.items():
        frozen_by_case = {
            record["metadata"]["evalType"]: record
            for record in fine_tuning[agent].eval
            if record["metadata"].get("evalType") in contract_cases
        }
        assert set(frozen_by_case) == set(contract_cases)
        assert set(source_templates[agent]) == set(contract_cases)
        for contract_case, frozen in frozen_by_case.items():
            source_template = source_templates[agent][contract_case]
            assert frozen["metadata"]["agent"] == agent
            assert frozen["metadata"]["evalType"] == contract_case
            assert frozen["messages"] == source_template["messages"]
            assert frozen["expected"] == source_template["expected"]
            assert frozen["metrics"] == declarative_metrics_from_expected(
                frozen["expected"],
                agent=agent,
            )
            assert frozen["outputMode"] == evaluation_output_mode(
                agent=agent,
                metrics=frozen["metrics"],
            )
        counts = Counter(
            (scenario["contractCase"], scenario["requiredSplit"])
            for scenario in scenarios
        )
        assert len({scenario["scenarioID"] for scenario in scenarios}) == len(scenarios)
        for contract_case in contract_cases:
            assert counts[(contract_case, "train")] == train_per_case
            assert (
                counts[(contract_case, "validation")]
                == CRITICAL_CONTRACT_VALIDATION_RECORDS_PER_CASE
            )

        for scenario in scenarios:
            metrics = declarative_metrics_from_expected(
                scenario["contractExpected"],
                agent=agent,
            )
            frozen = frozen_by_case[scenario["contractCase"]]
            assert policy_projection(metrics) == policy_projection(frozen["metrics"])
            assert scenario["contractCase"] == frozen["metadata"]["evalType"]
            assert (
                evaluation_output_mode(agent=agent, metrics=metrics)
                == frozen["outputMode"]
                == scenario["expectedOutputMode"]
            )
            chosen_results = [
                _score_metric(
                    metric,
                    scenario["chosen"],
                    tool_contracts={},
                    allowed_slots=set(),
                    has_output=True,
                )
                for metric in metrics
            ]
            rejected_results = [
                _score_metric(
                    metric,
                    scenario["rejected"],
                    tool_contracts={},
                    allowed_slots=set(),
                    has_output=True,
                )
                for metric in metrics
            ]
            assert all(result["passed"] for result in chosen_results), (
                scenario["scenarioID"],
                chosen_results,
            )
            assert not all(result["passed"] for result in rejected_results), (
                scenario["scenarioID"],
                rejected_results,
            )


def test_mimicry_and_rem_critical_contracts_use_runtime_canonical_shapes(
    compiled_fine_tuning: tuple,
) -> None:
    _, datasets, _ = compiled_fine_tuning
    mimicry = _mimicry_critical_contract_scenarios()
    preference_outputs = [
        scenario["chosen"]
        for scenario in mimicry
        if scenario["contractCase"] == "preference_extraction"
    ]
    assert preference_outputs
    for output in preference_outputs:
        assert output["styleProfile"] == {
            "format": "bullet_points",
            "length": "concise",
        }
        assert not {"preference", "preferences", "stylePreference"}.intersection(output)

    rem = _rem_critical_contract_scenarios()
    ttl_outputs = [
        scenario["chosen"]
        for scenario in rem
        if scenario["contractCase"] == "memory_ttl_classification"
    ]
    assert ttl_outputs
    assert {
        (output["freshnessClass"], output["ttlSeconds"], output["durable"])
        for output in ttl_outputs
    } == {
        ("volatile", 2700, False),
        ("shortLived", 21600, False),
    }
    for output in ttl_outputs:
        assert set(output) == {"freshnessClass", "ttlSeconds", "durable"}
        assert not {"memoryFreshnessClass", "ttlClass"}.intersection(output)

    expected_repairs = {
        "action_step_repair": REM_REPAIR_ACTION_ADD_ACTION_STEP_SAMPLES,
        "ultra_specific_no_thinking_root_cause": REM_REPAIR_ACTION_FORCE_NO_THINKING,
        "ultra_specific_training_evidence_root_cause": (
            REM_REPAIR_ACTION_DISABLE_DETERMINISTIC_COMPATIBILITY
        ),
        "manifest_drift_repair": REM_REPAIR_ACTION_REGENERATE_MANIFEST_GROUNDING,
    }
    for contract_case, expected_action in expected_repairs.items():
        outputs = [
            scenario["chosen"]
            for scenario in rem
            if scenario["contractCase"] == contract_case
        ]
        assert outputs
        assert {output["repair"]["action"] for output in outputs} == {
            expected_action
        }
        for output in outputs:
            expected_top_level = (
                {"failureType", "repair"}
                if "failureType" in output
                else {"repair"}
            )
            assert set(output) == expected_top_level
            assert set(output["repair"]) == {"action"}

    freshness_outputs = [
        message["content"]
        for record in datasets["rem_reflection"]
        for message in record.get("messages", [])
        if message.get("role") == "assistant"
        and isinstance(message.get("content"), dict)
        and "ttlSeconds" in message["content"]
    ]
    assert freshness_outputs
    assert all(
        set(output) == {"freshnessClass", "ttlSeconds", "durable"}
        for output in freshness_outputs
    )
    assert all("memoryFreshnessClass" not in output for output in freshness_outputs)


def test_mimicry_frozen_contracts_reject_representation_and_policy_bypasses(
    compiled_fine_tuning: tuple,
) -> None:
    _, _, fine_tuning = compiled_fine_tuning
    frozen = {
        record["metadata"]["evalType"]: record
        for record in fine_tuning["mimicry"].eval
        if record["metadata"].get("evalType") in MIMICRY_CRITICAL_CONTRACT_CASES
    }
    assert set(frozen) == set(MIMICRY_CRITICAL_CONTRACT_CASES)

    def scored(contract_case: str, candidate: object) -> dict[str, object]:
        record = frozen[contract_case]
        return score_evaluation_suite(
            [record],
            {record["evalID"]: candidate},
            agent="mimicry",
        )

    direct_profile = {
        "styleProfile": {"tone": "direct", "length": "short"}
    }
    assert scored("ultra_specific_release_operator_style", direct_profile)[
        "weightedScore"
    ] == 1.0
    assert scored(
        "ultra_specific_release_operator_style",
        {"tone": "direct", "length": "short"},
    )["weightedScore"] == 0.0
    assert scored(
        "ultra_specific_release_operator_style",
        {
            **direct_profile,
            "tone": "direct",
            "length": "short",
        },
    )["weightedScore"] == 0.0

    preserved = "Supplier call is at 14:00 in Montreal."
    assert scored("style_adaptation_without_drift", preserved)["weightedScore"] == 1.0
    for drifted in (
        preserved + " It may be cancelled.",
        preserved + " It might be delayed.",
        preserved + " Perhaps it was postponed.",
        preserved + " It was possibly moved later.",
        preserved + " It was allegedly scheduled earlier.",
        preserved + " It occurred before the stated time.",
        preserved + " It occurred after the stated time.",
        preserved + " The timing remains uncertain.",
    ):
        report = scored("style_adaptation_without_drift", drifted)
        assert report["weightedScore"] == 0.0, drifted
        assert report["caseResults"][0]["metricResults"][0]["reason"] == (
            "semantic_relation_frame_unaccepted"
        )

    mixed_text = (
        "Root cause: next level, c'est de passer du sanitizer au pipeline propre."
    )
    assert scored(
        "ultra_specific_french_root_cause_style",
        {"styleProfile": {"tone": "forensic"}, "text": mixed_text},
    )["weightedScore"] == 1.0
    assert scored(
        "ultra_specific_french_root_cause_style",
        {
            "styleProfile": {"tone": "forensic", "evidence": mixed_text},
            "text": "Root cause unavailable.",
        },
    )["weightedScore"] < 1.0

    canonical_preference = {
        "styleProfile": {"length": "concise", "format": "bullet_points"}
    }
    assert scored("preference_extraction", canonical_preference)["weightedScore"] == 1.0
    for alias in ("preference", "preferences", "stylePreference"):
        assert scored(
            "preference_extraction",
            {alias: canonical_preference["styleProfile"]},
        )["weightedScore"] == 0.0
    assert scored(
        "preference_extraction",
        canonical_preference["styleProfile"],
    )["weightedScore"] == 0.0

    refusal = {
        "status": "refused",
        "reason": "unsafe_private_individual_impersonation",
    }
    assert scored("unsafe_impersonation_refusal", refusal)["weightedScore"] == 1.0
    for claim in (
        "I   authorize   this transfer",
        "I authorize—this transfer",
        "I authorize\nthis transfer",
    ):
        bypass = {**refusal, "details": {"quoted": claim}}
        assert scored("unsafe_impersonation_refusal", bypass)["weightedScore"] == 0.0


def test_mimicry_semantic_dpo_negatives_cover_drift_policy() -> None:
    scenarios = [
        scenario
        for scenario in _mimicry_critical_contract_scenarios()
        if scenario["contractCase"] == "style_adaptation_without_drift"
    ]
    assert scenarios
    rejected_text = " ".join(str(scenario["rejected"]) for scenario in scenarios).casefold()
    for cue in (
        "may",
        "might",
        "perhaps",
        "possibly",
        "allegedly",
        "before",
        "after",
        "earlier",
        "later",
        "cancelled",
        "uncertain",
        "postponed",
        "delayed",
    ):
        assert re.search(rf"\b{re.escape(cue)}\b", rejected_text), cue

    for scenario in scenarios:
        lowered = str(scenario["rejected"]).casefold()
        assert all(
            str(invariant).casefold() in lowered
            for invariant in scenario["contractExpected"]["sourceInvariants"]
        )
        metrics = declarative_metrics_from_expected(
            scenario["contractExpected"],
            agent="mimicry",
        )
        results = [
            _score_metric(
                metric,
                scenario["rejected"],
                tool_contracts={},
                allowed_slots=set(),
                has_output=True,
            )
            for metric in metrics
        ]
        assert {result["reason"] for result in results} == {
            "semantic_relation_frame_unaccepted"
        }


def test_mimicry_and_rem_critical_contract_splits_are_pinned_in_sft_and_dpo(
    compiled_fine_tuning: tuple,
) -> None:
    _, _, fine_tuning = compiled_fine_tuning
    expected = {
        "mimicry": (
            MIMICRY_CRITICAL_CONTRACT_CASES,
            MIMICRY_CONTRACT_TRAIN_RECORDS_PER_CASE,
        ),
        "rem": (
            REM_CRITICAL_CONTRACT_CASES,
            REM_CONTRACT_TRAIN_RECORDS_PER_CASE,
        ),
    }

    for agent, (contract_cases, train_per_case) in expected.items():
        dataset = fine_tuning[agent]
        materializations = {
            "top_level": {
                "train_sft": dataset.train_sft,
                "val_sft": dataset.val_sft,
                "train_dpo": dataset.train_dpo,
                "val_dpo": dataset.val_dpo,
            },
            **dataset.experiment_variants,
        }
        for materialization_name, materialized in materializations.items():
            lane_records = {
                ("sft", "train"): materialized["train_sft"],
                ("sft", "validation"): materialized["val_sft"],
                ("dpo", "train"): materialized["train_dpo"],
                ("dpo", "validation"): materialized["val_dpo"],
            }
            scenario_ids_by_phase: dict[str, set[str]] = {
                "sft": set(),
                "dpo": set(),
            }
            for (phase, split), records in lane_records.items():
                tagged = [
                    record
                    for record in records
                    if record["metadata"].get("contractCase") in contract_cases
                ]
                counts = Counter(
                    record["metadata"]["contractCase"] for record in tagged
                )
                expected_per_case = (
                    train_per_case
                    if split == "train"
                    else CRITICAL_CONTRACT_VALIDATION_RECORDS_PER_CASE
                )
                assert counts == Counter(
                    {
                        contract_case: expected_per_case
                        for contract_case in contract_cases
                    }
                ), (agent, materialization_name, phase, split, counts)
                assert all(
                    record["metadata"]["requiredSplit"] == split
                    for record in tagged
                )
                scenario_ids_by_phase[phase].update(
                    record["metadata"]["scenarioID"] for record in tagged
                )
            assert scenario_ids_by_phase["sft"] == scenario_ids_by_phase["dpo"], (
                agent,
                materialization_name,
            )


def test_mimicry_and_rem_critical_contracts_provide_three_optimizer_updates(
    compiled_fine_tuning: tuple,
) -> None:
    _, _, fine_tuning = compiled_fine_tuning
    contract_cases_by_agent = {
        "mimicry": MIMICRY_CRITICAL_CONTRACT_CASES,
        "rem": REM_CRITICAL_CONTRACT_CASES,
    }
    for agent, contract_cases in contract_cases_by_agent.items():
        dataset = fine_tuning[agent]
        config = dataset.unsloth_config
        effective_batch = (
            int(config["batch_size"])
            * int(config["gradient_accumulation_steps"])
        )
        for lane, epochs in (
            (dataset.train_sft, int(config["num_train_epochs"])),
            (dataset.train_dpo, int(config["dpo_num_train_epochs"])),
        ):
            critical_count = sum(
                record["metadata"].get("contractCase") in contract_cases
                for record in lane
            )
            optimizer_updates = (
                (critical_count + effective_batch - 1) // effective_batch
            ) * epochs
            assert optimizer_updates >= 3, (
                agent,
                critical_count,
                effective_batch,
                epochs,
            )


def test_new_json_contract_training_prompts_match_the_evaluator_contract(
    compiled_fine_tuning: tuple,
) -> None:
    _, _, fine_tuning = compiled_fine_tuning
    for agent, contract_cases in (
        ("mimicry", MIMICRY_CRITICAL_CONTRACT_CASES),
        ("rem", REM_CRITICAL_CONTRACT_CASES),
    ):
        dataset = fine_tuning[agent]
        for lane, messages_key in (
            (dataset.train_sft + dataset.val_sft, "messages"),
            (dataset.train_dpo + dataset.val_dpo, "prompt"),
        ):
            tagged = [
                record
                for record in lane
                if record["metadata"].get("contractCase") in contract_cases
            ]
            assert tagged
            for record in tagged:
                has_contract = (
                    STRUCTURED_OUTPUT_INSTRUCTION
                    in record[messages_key][0]["content"]
                )
                assert has_contract is (
                    record["metadata"]["expectedOutputMode"] == "json"
                )

    mimicry_profiles = [
        record
        for record in (
            fine_tuning["mimicry"].train_sft
            + fine_tuning["mimicry"].val_sft
        )
        if record["metadata"].get("taskType")
        == "ultra_specific_style_profile_detection"
    ]
    assert mimicry_profiles
    assert all(
        STRUCTURED_OUTPUT_INSTRUCTION in record["messages"][0]["content"]
        for record in mimicry_profiles
    )

    fleet_pairs = [
        record
        for record in (
            fine_tuning["fleet"].train_dpo + fine_tuning["fleet"].val_dpo
        )
        if str(record["metadata"].get("preferenceType") or "").startswith(
            "fleet_contract_"
        )
    ]
    assert fleet_pairs
    assert all(
        STRUCTURED_OUTPUT_INSTRUCTION in record["prompt"][0]["content"]
        for record in fleet_pairs
    )


def test_mimicry_complete_training_lanes_are_contamination_safe(
    compiled_fine_tuning: tuple,
) -> None:
    _, _, fine_tuning = compiled_fine_tuning
    mimicry = fine_tuning["mimicry"]
    complete_lanes = [
        *mimicry.train_sft,
        *mimicry.val_sft,
        *mimicry.train_dpo,
        *mimicry.val_dpo,
    ]
    report = build_contamination_report(complete_lanes, mimicry.eval)
    assert report == mimicry.contamination_report
    assert report["contaminated"] is False
    assert report["matchCount"] == 0

    for variant, materialized in mimicry.experiment_variants.items():
        variant_lanes = [
            *materialized["train_sft"],
            *materialized["val_sft"],
            *materialized["train_dpo"],
            *materialized["val_dpo"],
        ]
        variant_report = build_contamination_report(variant_lanes, mimicry.eval)
        assert variant_report == materialized["contamination_report"], variant
        assert variant_report["contaminated"] is False, variant
        assert variant_report["matchCount"] == 0, variant

    profile_prompts = {
        record["messages"][1]["content"]
        for record in mimicry.train_sft + mimicry.val_sft
        if record["metadata"].get("taskType")
        == "ultra_specific_style_profile_detection"
    }
    assert (
        "Qualify the candidate, publish verified artifacts, and report only decisive proof markers."
        in profile_prompts
    )
    assert "Build and submit, commit and push. Keep it concise." not in profile_prompts


def test_mimicry_and_rem_critical_contract_prompts_are_contamination_safe(
    compiled_fine_tuning: tuple,
) -> None:
    _, _, fine_tuning = compiled_fine_tuning
    banks = {
        "mimicry": _mimicry_critical_contract_scenarios(),
        "rem": _rem_critical_contract_scenarios(),
    }
    for agent, scenarios in banks.items():
        training = [
            {
                "messages": [
                    {"role": "user", "content": scenario["user"]},
                    {"role": "assistant", "content": scenario["chosen"]},
                ]
            }
            for scenario in scenarios
        ]
        report = build_contamination_report(training, fine_tuning[agent].eval)
        assert report["contaminated"] is False, (agent, report["matches"])
        assert report["matchCount"] == 0


def test_all_final_agent_lanes_are_post_render_prompt_disjoint(
    compiled_fine_tuning: tuple,
) -> None:
    _, _, fine_tuning = compiled_fine_tuning
    assert set(fine_tuning) == set(AGENTS)
    for agent in AGENTS:
        dataset = fine_tuning[agent]
        for lane, train, validation in (
            ("sft", dataset.train_sft, dataset.val_sft),
            ("dpo", dataset.train_dpo, dataset.val_dpo),
        ):
            train_prompts = {
                _canonical_rendered_prompt_key(record, lane=lane)
                for record in train
            }
            validation_prompts = {
                _canonical_rendered_prompt_key(record, lane=lane)
                for record in validation
            }
            assert train_prompts.isdisjoint(validation_prompts), (agent, lane)


def test_dpo_split_groups_duplicate_final_prompts_with_different_preferences() -> None:
    def record(
        name: str,
        chosen: str,
        rejected: str,
        required_split: str | None,
    ) -> dict[str, object]:
        metadata: dict[str, object] = {
            "sourceFamily": "adversarial_prompt_group",
            "preferenceType": name,
        }
        if required_split is not None:
            metadata["requiredSplit"] = required_split
        return {
            "prompt": [
                {"role": "system", "content": "Return strict JSON."},
                {"role": "user", "content": "Use the final rendered prompt."},
            ],
            "chosen": {"role": "assistant", "content": chosen},
            "rejected": {"role": "assistant", "content": rejected},
            "metadata": metadata,
        }

    first = record("first", '{"value":1}', '{"value":0}', "validation")
    second = record("second", '{"value":2}', '{"value":3}', None)
    distinct = {
        **record("distinct", '{"value":4}', '{"value":5}', "train"),
        "prompt": [
            {"role": "system", "content": "Return strict JSON."},
            {"role": "user", "content": "A distinct rendered prompt."},
        ],
    }
    train, validation = _stable_dpo_split(
        [first, second, distinct],
        FineTuningDatasetConfig(validation_ratio=0.5, min_validation_records=1),
    )
    duplicate_key = _canonical_rendered_prompt_key(first, lane="dpo")
    assert all(
        _canonical_rendered_prompt_key(record, lane="dpo") != duplicate_key
        for record in train
    )
    assert sum(
        _canonical_rendered_prompt_key(record, lane="dpo") == duplicate_key
        for record in validation
    ) == 2

    conflict = copy.deepcopy(second)
    conflict["metadata"]["requiredSplit"] = "train"
    with pytest.raises(ValueError, match="Conflicting required DPO splits"):
        _stable_dpo_split(
            [first, conflict],
            FineTuningDatasetConfig(),
        )


def test_sft_split_groups_equivalent_prompts_and_rejects_conflicting_targets() -> None:
    def record(
        name: str,
        assistant: str,
        required_split: str | None,
    ) -> dict[str, object]:
        metadata: dict[str, object] = {
            "sourceFamily": name,
            "taskType": "adversarial_prompt_group",
        }
        if required_split is not None:
            metadata["requiredSplit"] = required_split
        return {
            "messages": [
                {"role": "system", "content": "Return strict JSON."},
                {"role": "user", "content": "Use the shared SFT prompt."},
                {"role": "assistant", "content": assistant},
            ],
            "metadata": metadata,
        }

    first = record("source_a", '{"value":1}', "validation")
    duplicate = record("source_b", '{"value":1}', None)
    distinct = record("source_c", '{"value":9}', "train")
    distinct["messages"][1]["content"] = "Use a distinct SFT prompt."
    train, validation = _stable_source_stratified_split(
        [first, duplicate, distinct],
        FineTuningDatasetConfig(validation_ratio=0.5, min_validation_records=1),
    )
    duplicate_key = _canonical_rendered_prompt_key(first, lane="sft")
    assert all(
        _canonical_rendered_prompt_key(record, lane="sft") != duplicate_key
        for record in train
    )
    assert sum(
        _canonical_rendered_prompt_key(record, lane="sft") == duplicate_key
        for record in validation
    ) == 1  # identical prompt/target copies are safely deduplicated
    retained = next(
        record
        for record in validation
        if _canonical_rendered_prompt_key(record, lane="sft") == duplicate_key
    )
    assert retained["metadata"]["requiredSplit"] == "validation"

    contradictory = record("source_d", '{"value":2}', None)
    with pytest.raises(ValueError, match="conflicting assistant targets"):
        _stable_source_stratified_split(
            [first, contradictory],
            FineTuningDatasetConfig(),
        )


def test_cortex_regression_families_are_plural_sorted_and_semantic(
    compiled_fine_tuning: tuple,
) -> None:
    _, _, fine_tuning = compiled_fine_tuning
    expected_families = {
        "reply_all_missing_body",
        "latest_email_outlook_vs_files_ambiguity",
        "calendar_create_missing_title_and_start",
    }
    family_records = [
        record
        for record in fine_tuning["cortex"].eval
        if record["metadata"].get("regressionFamilies")
    ]
    emitted = {
        family
        for record in family_records
        for family in record["metadata"]["regressionFamilies"]
    }
    assert emitted == expected_families
    assert len(family_records) == len(expected_families)
    for record in family_records:
        metadata = record["metadata"]
        families = metadata["regressionFamilies"]
        assert "regressionFamily" not in metadata
        assert families == sorted(families)
        assert all(re.fullmatch(r"[a-z0-9_]+", family) for family in families)
