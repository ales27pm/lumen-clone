from __future__ import annotations

import json

from lumen_manifest_crawler.dataset.compiler import (
    DatasetCompilerConfig,
    _build_tool_schema_records,
    finalize_dataset_manifest,
)
from lumen_manifest_crawler.dataset.cortex import generate_cortex_records
from lumen_manifest_crawler.dataset.executor import (
    generate_approval_boundary_records,
    generate_executor_records,
)
from lumen_manifest_crawler.dataset.fine_tuning import (
    FineTuningDatasetConfig,
    _build_agent_dpo_records,
    _route_record_agents,
)
from lumen_manifest_crawler.manifest import (
    AgentBehaviorManifest,
    IntentManifest,
    ToolArgumentManifest,
    ToolManifest,
)
from lumen_manifest_crawler.improvement_loop import (
    _is_non_blocking_runtime_diagnostic,
    _runtime_gap_category,
    _runtime_root_cause_category,
    _write_testflight_runbook,
)
from lumen_manifest_crawler.validators import validate_manifest


def _tool(tool_id: str, display_name: str) -> ToolManifest:
    return ToolManifest(
        id=tool_id,
        displayName=display_name,
        description=f"Perform {display_name.lower()}",
        requiresApproval=True,
        arguments=[ToolArgumentManifest(name="id", type="string", required=True)],
    )


def test_cortex_tool_labels_have_distinct_prompts() -> None:
    manifest = AgentBehaviorManifest(
        tools=[_tool("alarm.cancel", "Cancel Alarm"), _tool("alarm.pause", "Pause Alarm")],
        intents=[IntentManifest(id="alarm", allowedToolIDs=["alarm.cancel", "alarm.pause"])],
    )

    records = generate_cortex_records(manifest)
    prompts = [record["messages"][1]["content"] for record in records]

    assert len(prompts) == len(set(prompts))


def test_approval_boundary_prompts_identify_tool_and_state() -> None:
    manifest = AgentBehaviorManifest(
        tools=[_tool("alarm.cancel", "Cancel Alarm"), _tool("alarm.pause", "Pause Alarm")]
    )

    records = generate_approval_boundary_records(manifest)
    prompts = [record["input"] for record in records]

    assert len(prompts) == len(set(prompts))
    assert all(f"`{record['tool']}`" in record["input"] for record in records)


def test_approval_ambiguity_requires_a_missing_required_argument() -> None:
    no_arg_tool = ToolManifest(
        id="camera.capture",
        displayName="Capture Photo",
        requiresApproval=True,
        arguments=[],
    )
    manifest = AgentBehaviorManifest(
        tools=[_tool("alarm.cancel", "Cancel Alarm"), no_arg_tool]
    )

    records = generate_approval_boundary_records(manifest)
    ambiguous = [record for record in records if record["scenario"] == "ambiguous_request"]

    assert {record["tool"] for record in ambiguous} == {"alarm.cancel"}
    assert all(record["expectedExecutorOutput"]["missingArguments"] for record in ambiguous)


def test_cross_model_provenance_cannot_bypass_role_locks() -> None:
    record = {
        "taskType": "fleet_private_state_boundary",
        "metadata": {
            "agentRole": "executor",
            "publicCorpus": {"targetAdapter": "executor"},
        },
        "prompt": [{"role": "user", "content": "Inspect a private fleet cache."}],
        "chosen": {"role": "assistant", "content": "I cannot inspect private runtime state."},
        "rejected": {"role": "assistant", "content": "Here is fabricated private state."},
    }
    routed = _route_record_agents(
        source_family="cross_model_training",
        record=record,
        task_type="fleet_private_state_boundary",
        tool_ids=[],
        slot_ids={"executor"},
        slot_roles={"tool_executor"},
    )

    assert routed == ["fleet"]

    manifest = AgentBehaviorManifest(tools=[_tool("alarm.cancel", "Cancel Alarm")])
    dpo = _build_agent_dpo_records(
        manifest,
        {"cross_model_training": [record]},
        FineTuningDatasetConfig(),
        {"alarm.cancel"},
    )
    assert any(
        (item.get("metadata") or {}).get("sourceFamily") == "cross_model_training"
        for item in dpo["fleet"]
    )
    for role_locked_agent in ("cortex", "executor"):
        assert all(
            (item.get("metadata") or {}).get("sourceFamily")
            != "cross_model_training"
            for item in dpo[role_locked_agent]
        )
    assert all("tool" in json.loads(item["chosen"]["content"]) for item in dpo["executor"])


def test_enum_samples_use_manifest_allowed_values() -> None:
    trigger = ToolManifest(
        id="trigger.create",
        displayName="Schedule Agent Run",
        arguments=[
            ToolArgumentManifest(
                name="schedule",
                type="enum",
                required=True,
                allowedValues=["absolute", "interval", "relative"],
            )
        ],
    )
    manifest = AgentBehaviorManifest(tools=[trigger])

    executor_payload = generate_executor_records(manifest)[0]["messages"][-1]["content"]
    schema_record = next(
        record
        for record in _build_tool_schema_records(manifest, DatasetCompilerConfig())
        if record["id"].startswith("schema-required-")
    )
    schema_payload = json.loads(schema_record["messages"][-1]["content"])

    assert executor_payload["arguments"]["schedule"] == "absolute"
    assert schema_payload["arguments"]["schedule"] == "absolute"


def test_final_dataset_manifest_hashes_every_family() -> None:
    base = {"sources": {"rawDatasetFamilies": ["train_sft"]}, "counts": {}, "hashes": {}}
    datasets = {
        "train_sft": [{"id": "train-1"}],
        "embedding_corpus": [{"id": "doc-1"}],
        "reranker_eval_reranking": [],
    }

    result = finalize_dataset_manifest(base, datasets)

    assert result["counts"] == {
        "embedding_corpus": 1,
        "reranker_eval_reranking": 0,
        "train_sft": 1,
    }
    assert set(result["hashes"]) == set(datasets)
    assert result["sources"]["datasetFamilies"] == sorted(datasets)


def test_testflight_runbook_uses_canonical_manifest_base_commit(tmp_path) -> None:
    output = tmp_path / "TESTFLIGHT_RUNBOOK.md"
    state = {
        "manifest": {"fingerprint": "manifest-sha", "baseCommit": "base-sha"},
        "testFlight": {
            "buildLabel": "build-1",
            "expectedExport": "runtime-audit.json",
            "nextIngestCommand": "lumen improve-loop --runtime-audit runtime-audit.json",
            "scenarioQueuePath": "testflight_scenarios.jsonl",
        },
    }

    _write_testflight_runbook(output, state, [])

    contents = output.read_text(encoding="utf-8")
    assert "Manifest base commit: `base-sha`" in contents


def test_validator_rejects_incomplete_or_stale_dataset_manifest() -> None:
    datasets = {"train_sft": [{"id": "train-1"}], "embedding_corpus": [{"id": "doc-1"}]}
    complete = finalize_dataset_manifest({"sources": {}, "counts": {}, "hashes": {}}, datasets)
    corrupted = {
        **complete,
        "counts": {"train_sft": 99},
        "hashes": {**complete["hashes"], "embedding_corpus": "stale"},
    }

    report = validate_manifest(
        AgentBehaviorManifest(),
        {**datasets, "dataset_manifest": [corrupted]},
    )
    codes = {failure.code for failure in report.failures}

    assert "dataset_manifest_count_coverage" in codes
    assert "dataset_manifest_count_mismatch" in codes
    assert "dataset_manifest_hash_mismatch" in codes


def test_runtime_environment_status_remains_non_blocking_when_legacy_root_is_stale() -> None:
    failure = {
        "type": "e2e_runtime_environment_deferred",
        "rootCauseCategory": "stale_or_unclassified_runtime_evidence",
        "trainable": False,
        "e2eScenario": {"modelEvidenceStatus": "runtime_environment_deferred"},
    }

    root_cause = _runtime_root_cause_category(failure)
    category = _runtime_gap_category(failure)

    assert root_cause == "runtime_environment_deferred"
    assert category == "runtime_environment_deferred"
    assert _is_non_blocking_runtime_diagnostic(
        failure,
        category=category,
        skipped_live_generation=False,
    )
