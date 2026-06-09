"""Tests for runtime repair dataset quality gates."""

import json

from lumen_manifest_crawler.dataset.compiler import compile_state_of_art_datasets
from lumen_manifest_crawler.manifest import AgentBehaviorManifest, ToolManifest


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
