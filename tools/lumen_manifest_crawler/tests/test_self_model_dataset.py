import json
from pathlib import Path

import pytest

from lumen_manifest_crawler.crawler import generate_manifest
from lumen_manifest_crawler.dataset.compiler import (
    MIN_SELF_MODEL_EVAL_SCENARIOS,
    SELF_MODEL_CARD_TYPES,
    compile_state_of_art_datasets,
)
from lumen_manifest_crawler.dataset.fine_tuning import (
    FineTuningDatasetConfig,
    _route_record_agents,
    compile_agent_fine_tuning_datasets,
)
from lumen_manifest_crawler.improvement_loop import _build_testflight_scenario_queue
from lumen_manifest_crawler.manifest import AgentBehaviorManifest, ToolArgumentManifest, ToolManifest, ValidationReport
from lumen_manifest_crawler.output.writer import write_outputs


def _manifest() -> AgentBehaviorManifest:
    return AgentBehaviorManifest(
        tools=[
            ToolManifest(
                id="calendar.create",
                displayName="Create Event",
                description="Create a calendar event",
                requiresApproval=True,
                permissionKey="NSCalendarsFullAccessUsageDescription",
                permissionKind="calendar",
                confirmationMode="userApproval",
                arguments=[ToolArgumentManifest(name="title", type="string", required=True)],
            ),
            ToolManifest(id="device.status", displayName="Device Status", description="Summarize safe device status"),
            ToolManifest(id="rag.search.secure", displayName="Secure RAG Search", description="Search approved RAG sources"),
        ],
    )


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


def test_self_model_cards_cover_required_taxonomy_and_write_top_level_copy(tmp_path) -> None:
    manifest = _manifest()
    compiled = compile_state_of_art_datasets(manifest, {})
    cards = compiled.records["self_model_cards"]

    assert {record["cardType"] for record in cards} == SELF_MODEL_CARD_TYPES
    assert all(record["sourceFamily"] == "self_model_cards" for record in cards)
    assert any(record["cardType"] == "tool_boundary" and "calendar.create" in record["messages"][-1]["content"] for record in cards)

    write_outputs(
        tmp_path,
        manifest,
        ValidationReport(passed=True),
        compiled.records,
        pretty=False,
    )

    top_level_cards = tmp_path / "self_model_cards.jsonl"
    dataset_cards = tmp_path / "dataset" / "self_model_cards.jsonl"
    assert top_level_cards.exists()
    assert dataset_cards.exists()
    assert len(top_level_cards.read_text(encoding="utf-8").splitlines()) == len(cards)


def test_self_model_eval_has_twenty_plus_scenarios_and_enters_testflight_queue() -> None:
    manifest = _manifest()
    compiled = compile_state_of_art_datasets(manifest, {})
    records = compiled.records["self_model_eval"]

    assert len(records) >= MIN_SELF_MODEL_EVAL_SCENARIOS
    assert all(record["sourceFamily"] == "self_model_eval" for record in records)
    assert all(record["taskType"] == "self_model_grounding" for record in records)
    assert any(record["expected"].get("mustRejectUnknownTool") == "system.root.delete" for record in records)

    scenarios = _build_testflight_scenario_queue(
        manifest=manifest,
        datasets=compiled.records,
        fine_tuning_datasets=None,
        limit=40,
    )
    self_model_scenarios = [scenario for scenario in scenarios if scenario["sourceFamily"] == "self_model_eval"]
    assert len(self_model_scenarios) == len(records)
    assert all(scenario["taskType"] == "self_model_grounding" for scenario in self_model_scenarios)


def test_self_model_sft_routes_only_to_fleet_adapter() -> None:
    manifest = _manifest()
    compiled = compile_state_of_art_datasets(manifest, {})

    for source_family in ("self_model_cards", "self_model_sft"):
        records = compiled.records[source_family]
        assert records
        for record in records:
            assert _route_record_agents(
                source_family=source_family,
                record=record,
                task_type=str(record.get("taskType") or source_family),
                tool_ids=list(record.get("toolIDs") or []),
                slot_ids={slot.id for slot in manifest.fleet.slots},
                slot_roles={slot.role for slot in manifest.fleet.slots},
            ) == ["fleet"]


def test_self_model_runtime_failure_ingests_as_rem_repair(
    diagnostic_manifest: AgentBehaviorManifest,
) -> None:
    manifest = diagnostic_manifest
    runtime_reports = [
        {
            "_source": "agent-grounding-export-self-model.json",
            "_sourceFormat": "lumen_in_app_dataset_package",
            "failures": [
                {
                    "type": "self_model_runtime_state_claim_without_evidence",
                    "agent": "fleet",
                    "sourceLayer": "agentGroundingRuntimeAudit.exportQuality",
                    "scenario": "Can you prove the last TestFlight run passed?",
                    "actual": "Claimed pass from generated manifest only.",
                    "problem": "Self-model answer treated static generated evidence as live runtime proof.",
                }
            ],
        }
    ]
    compiled = compile_state_of_art_datasets(manifest, {}, runtime_audit_reports=runtime_reports)

    repairs = compiled.records["runtime_audit_repairs"]
    assert len(repairs) == 1
    payload = json.loads(repairs[0]["messages"][-1]["content"])
    assert payload["failureType"] == "self_model_runtime_state_claim_without_evidence"
    assert payload["repair"]["action"] == "add_runtime_evidence_honesty_samples"
    assert "self_model_eval" in payload["repair"]["alsoAdd"]

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
        and "add_runtime_evidence_honesty_samples" in record["messages"][-1]["content"]
    ]
    assert matching


def test_self_model_score_report_failures_compile_to_repair_records() -> None:
    manifest = _manifest()
    runtime_reports = [
        {
            "_source": "self-model-eval-score.json",
            "_sourceFormat": "self_model_eval_score_report",
            "_sourceLayer": "selfModelEvalScore",
            "scenarioCount": 1,
            "ownsLiveE2EScenarios": False,
            "failures": [
                {
                    "type": "self_model_eval_answer_missing",
                    "agent": "fleet",
                    "expected": ["mustAnswerUnknownWithoutEvidence"],
                    "actual": ["missing_answer"],
                    "scenario": "self-model-current-location",
                    "problem": "A self-model eval scenario was not answered by the model export.",
                    "sourceLayer": "selfModelEvalScore",
                }
            ],
        }
    ]

    compiled = compile_state_of_art_datasets(manifest, {}, runtime_audit_reports=runtime_reports)
    repairs = compiled.records["runtime_audit_repairs"]

    assert len(repairs) == 1
    payload = json.loads(repairs[0]["messages"][-1]["content"])
    assert payload["failureType"] == "self_model_eval_answer_missing"
    assert payload["repair"]["action"] == "add_self_model_eval_execution_samples"
    assert "self_model_eval" in payload["repair"]["alsoAdd"]
