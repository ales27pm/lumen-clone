"""Tests for runtime repair dataset quality gates."""

import json
from pathlib import Path

import pytest

from lumen_manifest_crawler.crawler import generate_manifest
from lumen_manifest_crawler.dataset.compiler import compile_state_of_art_datasets
from lumen_manifest_crawler.dataset.fine_tuning import (
    FineTuningDatasetConfig,
    compile_agent_fine_tuning_datasets,
)
from lumen_manifest_crawler.manifest import AgentBehaviorManifest, ToolManifest


@pytest.fixture(scope="module")
def diagnostic_manifest() -> AgentBehaviorManifest:
    """Use production topology for optimizer-aware repair routing checks."""

    root = Path(__file__).resolve().parents[3]
    manifest = generate_manifest(root)
    return manifest.model_copy(
        update={
            "sourceIntegrity": manifest.sourceIntegrity.model_copy(
                update={"baseCommit": "test-commit"}
            )
        }
    )


def test_export_quality_failures_do_not_become_training_repairs() -> None:
    manifest = AgentBehaviorManifest(tools=[ToolManifest(id="calendar.list")])
    compiled = compile_state_of_art_datasets(
        manifest,
        {},
        runtime_audit_reports=[
            {
                "_source": "agent-grounding-export.json",
                "_sourceFormat": "lumen_in_app_dataset_package",
                "failures": [
                    {
                        "type": "agent_grounding_no_recent_model_traces",
                        "sourceLayer": "agentGroundingRuntimeAudit.exportQuality",
                        "scenario": "Agent Grounding > Export",
                        "actual": "recentTraces is empty",
                    }
                ],
            }
        ],
    )

    assert compiled.records["runtime_audit_repairs"] == []


def test_final_validator_replacement_quality_failure_becomes_rem_repair() -> None:
    manifest = AgentBehaviorManifest(tools=[ToolManifest(id="calendar.list")])
    compiled = compile_state_of_art_datasets(
        manifest,
        {},
        runtime_audit_reports=[
            {
                "_source": "agent-grounding-export-v14.json",
                "_sourceFormat": "lumen_in_app_dataset_package",
                "failures": [
                    {
                        "type": "agent_grounding_model_trace_incomplete",
                        "agent": "executor",
                        "sourceLayer": "agentGroundingRuntimeAudit.exportQuality",
                        "scenario": "What is the weather here?",
                        "actual": "missing=selectedRuntime,modelLoaded",
                    },
                    {
                        "type": "agent_grounding_final_validator_replaced_candidate",
                        "agent": "mouth",
                        "sourceLayer": "agentGroundingRuntimeAudit.exportQuality",
                        "scenario": "Search my calendar for tomorrow",
                        "actual": "replacementSource=safeMessage; rejectionReason=tool-json-leak",
                        "problem": "The final validator replaced the candidate response.",
                    },
                ],
            }
        ],
    )

    repairs = compiled.records["runtime_audit_repairs"]
    assert len(repairs) == 1
    assert repairs[0]["metadata"]["sourceLayer"] == "agentGroundingRuntimeAudit.exportQuality"
    payload = json.loads(repairs[0]["messages"][-1]["content"])
    assert payload["failureType"] == "agent_grounding_final_validator_replaced_candidate"
    assert payload["repair"]["action"] == "add_finalizer_validator_contract_samples"
    assert "final_intent_validator_trace_eval" in payload["repair"]["alsoAdd"]


def test_final_validator_replacement_repair_reaches_rem_sft(
    diagnostic_manifest: AgentBehaviorManifest,
) -> None:
    manifest = diagnostic_manifest
    runtime_reports = [
        {
            "_source": "agent-grounding-export-v14.json",
            "_sourceFormat": "lumen_in_app_dataset_package",
            "failures": [
                {
                    "type": "agent_grounding_final_validator_replaced_candidate",
                    "agent": "mouth",
                    "sourceLayer": "agentGroundingRuntimeAudit.exportQuality",
                    "scenario": "Search my calendar for tomorrow",
                    "actual": "replacementSource=safeMessage; rejectionReason=tool-json-leak",
                    "problem": "The final validator replaced the candidate response.",
                }
            ],
        }
    ]
    compiled = compile_state_of_art_datasets(
        manifest,
        {},
        runtime_audit_reports=runtime_reports,
    )

    fine_tuning = compile_agent_fine_tuning_datasets(
        manifest,
        compiled.records,
        runtime_audit_reports=runtime_reports,
        config=FineTuningDatasetConfig(include_unsloth_config=False),
    )

    rem_records = fine_tuning["rem"].train_sft + fine_tuning["rem"].val_sft
    matching = [
        record
        for record in rem_records
        if (record.get("metadata") or {}).get("sourceFamily") == "runtime_audit_repairs"
        and "add_finalizer_validator_contract_samples" in record["messages"][-1]["content"]
    ]
    assert matching
    assert all((record.get("metadata") or {}).get("taskType") == "runtime_manifest_drift_repair" for record in matching)


def test_duplicate_runtime_failures_are_deduped_for_training_repairs() -> None:
    manifest = AgentBehaviorManifest(tools=[ToolManifest(id="calendar.list")])
    failure = {
        "type": "missing_required_tool_action",
        "agent": "cortex",
        "sourceLayer": "agentModelBehaviorAuditor.repairSamples",
        "scenario": "Search my calendar for next event",
        "actual": "No action step was persisted.",
        "repairSample": {
            "agent": "cortex",
            "violationCode": "missing_required_tool_action",
            "promptPrefix": "Search my calendar for next event",
            "expected": "Intent calendar should select one of: calendar.create, calendar.list",
            "correctedOutput": "Select a manifest-allowed calendar read tool.",
            "lesson": "Tool-backed intents require a manifest-allowed action step.",
            "curriculum": "tool_routing",
        },
    }
    compiled = compile_state_of_art_datasets(
        manifest,
        {},
        runtime_audit_reports=[
            {"_source": "first.json", "_sourceFormat": "lumen_in_app_dataset_package", "failures": [failure]},
            {"_source": "second.json", "_sourceFormat": "lumen_in_app_dataset_package", "failures": [dict(failure)]},
        ],
    )

    repairs = compiled.records["runtime_audit_repairs"]
    assert len(repairs) == 1
    payload = json.loads(repairs[0]["messages"][-1]["content"])
    assert payload["failureType"] == "missing_required_tool_action"
    assert payload["repair"]["action"] == "train_from_in_app_repair_sample"


def test_clean_runtime_reports_are_deduped_by_source_layer() -> None:
    manifest = AgentBehaviorManifest(tools=[ToolManifest(id="calendar.list")])
    compiled = compile_state_of_art_datasets(
        manifest,
        {},
        runtime_audit_reports=[
            {"_source": "first-live-e2e.json", "_sourceFormat": "live-e2e-test-report-json", "_sourceLayer": "e2eTestReport", "failures": []},
            {"_source": "second-live-e2e.json", "_sourceFormat": "live-e2e-test-report-json", "_sourceLayer": "e2eTestReport", "failures": []},
            {"_source": "persistent.json", "_sourceFormat": "persistent_runtime_diagnostics_export", "_sourceLayer": "persistentRuntimeDiagnostics", "failures": []},
        ],
    )

    repairs = compiled.records["runtime_audit_repairs"]
    assert len(repairs) == 2
    assert {record["metadata"]["sourceLayer"] for record in repairs} == {
        "e2eTestReport",
        "persistentRuntimeDiagnostics",
    }
