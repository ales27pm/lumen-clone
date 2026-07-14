from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path

import pytest

from lumen_manifest_crawler.crawler import generate_manifest
from lumen_manifest_crawler.dataset import generate_all_datasets
from lumen_manifest_crawler.dataset.compiler import _records_hash
from lumen_manifest_crawler.dataset.codebase_home import (
    MAX_CHUNK_CHARS,
    _split_source_chunks,
    generate_codebase_home_records,
)
from lumen_manifest_crawler.dataset.fine_tuning import (
    AGENTS,
    CORTEX_CODEBASE_SELF_AWARENESS_SOURCE_FAMILY,
    ULTRA_SPECIFIC_SOURCE_FAMILY,
    compile_agent_fine_tuning_datasets,
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
            assert record["schemaVersion"] == "lumen.adapter-eval/1.0.0"
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
        train = fine_tuning[agent].train_sft
        val = fine_tuning[agent].val_sft
        train_keys = {json.dumps(record["messages"], ensure_ascii=False, sort_keys=True) for record in train}
        val_keys = {json.dumps(record["messages"], ensure_ascii=False, sort_keys=True) for record in val}
        assert len(train_keys) == len(train)
        assert len(val_keys) == len(val)
        assert train_keys.isdisjoint(val_keys)

        all_sources = {
            record["metadata"]["sourceFamily"]
            for record in train + val
        }
        for source in all_sources:
            source_records = [record for record in train + val if record["metadata"]["sourceFamily"] == source]
            if len(source_records) >= 2 and not source.startswith("public_adapter_corpus_"):
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


def test_sft_records_fit_conservative_sequence_budget(compiled_fine_tuning: tuple) -> None:
    _, _, fine_tuning = compiled_fine_tuning
    for agent in AGENTS:
        dataset = fine_tuning[agent]
        max_chars = dataset.unsloth_config["sequence_char_budget"]
        assert dataset.unsloth_config["sequence_budget_policy"] == "conservative_utf8_byte_proxy"
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
        assert len(public_card["selectionContract"]["sha256"]) == 64
        assert sum(public_card["recordCounts"].values()) == len(selected_public)
        for lane, selected_count in public_card["recordCounts"].items():
            available_count = public_card["availableRecordCounts"][lane]
            assert selected_count <= available_count
            assert public_card["rejectedByTokenCap"][lane] == available_count - selected_count
        assert sum(public_card["availableSourceCounts"].values()) == sum(
            public_card["availableRecordCounts"].values()
        )

    assert sum(fine_tuning["mouth"].dataset_card["publicCorpus"]["recordCounts"][lane] for lane in ("train_dpo", "val_dpo")) > 0
    mouth_public = fine_tuning["mouth"].dataset_card["publicCorpus"]
    for lane in ("train_dpo", "val_dpo"):
        assert mouth_public["tokenShares"][lane]["total"] <= 0.35
        assert mouth_public["tokenShares"][lane]["target"] <= 0.35

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


def test_executor_missing_argument_samples_require_required_arguments(compiled_fine_tuning: tuple) -> None:
    manifest, _, fine_tuning = compiled_fine_tuning
    optional_only_tools = {
        tool.id
        for tool in manifest.tools
        if tool.arguments and not any(argument.required for argument in tool.arguments)
    }
    assert optional_only_tools

    for record in fine_tuning["executor"].train_sft + fine_tuning["executor"].val_sft:
        metadata = record.get("metadata") or {}
        if metadata.get("taskType") != "ultra_specific_missing_argument_boundary":
            continue
        assert optional_only_tools.isdisjoint(metadata.get("toolIDs") or [])


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
        assert assistant["permissionKind"] == tool.permissionKind
        assert assistant["confirmationMode"] == tool.confirmationMode


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
        "executor_dpo_missing_chosen_output",
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
        tool_id = payload.get("tool")
        assert tool_id in tools
        arguments = payload.get("arguments")
        assert isinstance(arguments, dict)
        contract = {argument.name: argument for argument in tools[tool_id].arguments}
        assert set(arguments).issubset(contract)
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
        tool = tools[payload["tool"]]
        arguments = payload.get("arguments")
        assert isinstance(arguments, dict)
        required = {argument.name for argument in tool.arguments if argument.required}
        assert required.issubset(arguments)
        for argument in tool.arguments:
            if argument.name in arguments and argument.allowedValues:
                assert arguments[argument.name] in argument.allowedValues


def test_validator_rejects_invalid_executor_enum(compiled_fine_tuning: tuple) -> None:
    manifest, _, fine_tuning = compiled_fine_tuning
    executor = fine_tuning["executor"]
    records = copy.deepcopy(executor.train_sft)
    index = next(
        index
        for index, record in enumerate(records)
        if "trigger.create" in record["metadata"]["toolIDs"]
        and "schedule" in json.loads(record["messages"][2]["content"]).get("arguments", {})
    )
    payload = json.loads(records[index]["messages"][2]["content"])
    payload["arguments"]["schedule"] = "sample_schedule"
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


def test_codebase_home_records_are_supplemental_for_fleet_and_excluded_from_rem() -> None:
    repo_root = _repo_root()
    manifest = generate_manifest(repo_root)
    datasets = generate_all_datasets(manifest, root=repo_root)
    fine_tuning = compile_agent_fine_tuning_datasets(manifest, datasets)

    assert datasets["codebase_home_corpus"]
    assert datasets["codebase_home_sft"]
    assert datasets["codebase_home_chunks"]
    assert datasets["codebase_home_chunk_sft"]
    fleet_records = fine_tuning["fleet"].train_sft + fine_tuning["fleet"].val_sft
    fleet_codebase = [
        record
        for record in fleet_records
        if record["metadata"]["sourceFamily"] in {"codebase_home_sft", "codebase_home_chunk_sft"}
    ]
    assert fleet_codebase
    assert len(fleet_codebase) / len(fleet_records) <= 0.25

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
        "dataset_dir",
        "output_dir",
    }
    _, _, fine_tuning = compiled_fine_tuning
    for agent in AGENTS:
        config = fine_tuning[agent].unsloth_config
        assert required.issubset(config.keys()), f"{agent} missing keys"

    config_dir = _repo_root() / "tools" / "fine_tuning" / "unsloth" / "configs"
    for path in config_dir.glob("*.json"):
        cfg = json.loads(path.read_text(encoding="utf-8"))
        assert required.issubset(cfg.keys()), f"{path} missing required keys"


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
