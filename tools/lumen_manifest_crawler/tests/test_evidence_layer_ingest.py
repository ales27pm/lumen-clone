"""Tests for per-layer in-app evidence JSON envelope ingestion."""

from __future__ import annotations

import json
from pathlib import Path

from lumen_manifest_crawler.dataset.runtime_ingest import load_runtime_audit_reports


def test_live_e2e_evidence_layer_envelope_ingests_as_e2e_report(tmp_path: Path) -> None:
    report_path = tmp_path / "lumen-live-e2e-report.json"
    envelope = {
        "schemaVersion": "1.0.0",
        "generatedAt": "2026-05-03T00:00:00Z",
        "exportPolicy": {
            "format": "live-e2e-test-report-json",
            "sourceLayer": "e2eTestReport",
            "ownsLiveE2EScenarios": True,
            "includesDeterministicStaticScenarios": False,
        },
        "payload": {
            "passed": 0,
            "failed": 1,
            "results": [
                {
                    "title": "Training eval: pure chat quality",
                    "prompt": "Explain precision and recall.",
                    "actualIntent": "chat",
                    "expectedIntent": "chat",
                    "passed": False,
                    "failures": ["Required final hint missing: precision"],
                    "finalText": "Recall is about finding things.",
                    "events": [{"phase": "models", "message": "chat fleet ready"}],
                }
            ],
        },
    }
    report_path.write_text(json.dumps(envelope), encoding="utf-8")

    report = load_runtime_audit_reports([report_path])[0]

    assert report["_sourceFormat"] == "live-e2e-test-report-json"
    assert report["scenarioCount"] == 1
    assert len(report["failures"]) == 1
    failure = report["failures"][0]
    assert failure["sourceLayer"] == "e2eTestReport.evidenceLayer"
    assert failure["e2eScenario"]["intent"] == "chat"
    assert failure["actual"] == "Recall is about finding things."


def test_static_scenario_evidence_layer_envelope_is_non_live_and_ignored(tmp_path: Path) -> None:
    report_path = tmp_path / "lumen-static-scenario-checks.json"
    envelope = {
        "schemaVersion": "1.0.0",
        "generatedAt": "2026-05-03T00:00:00Z",
        "exportPolicy": {
            "format": "deterministic-static-scenario-checks-json",
            "sourceLayer": "runtimeScenarioRunner.staticChecks",
            "ownsLiveE2EScenarios": False,
            "includesDeterministicStaticScenarios": True,
        },
        "payload": [
            {
                "id": "calendar::calendar.create",
                "passed": False,
                "failures": [{"type": "scenario_unknown_tool"}],
            }
        ],
    }
    report_path.write_text(json.dumps(envelope), encoding="utf-8")

    report = load_runtime_audit_reports([report_path])[0]

    assert report["_sourceLayer"] == "runtimeScenarioRunner.staticChecks"
    assert report["ownsLiveE2EScenarios"] is False
    assert report["ignoredScenarioResultCount"] == 1
    assert report["failures"] == []


def test_agent_grounding_schema_1_2_package_ingests_as_runtime_audit(tmp_path: Path) -> None:
    report_path = tmp_path / "lumen-agent-grounding-audit.json"
    package = {
        "schemaVersion": "1.2.0",
        "generatedAt": "2026-06-08T23:27:42Z",
        "exportPolicy": {
            "format": "agent-grounding-runtime-json-package",
            "sourceLayer": "agentGroundingRuntimeAudit",
            "ownsLiveE2EScenarios": False,
        },
        "runtimeManifestAudit": {"failures": [], "passed": True},
        "behaviorAudit": {
            "repairSamples": [
                {
                    "agent": "cortex",
                    "violationCode": "missing_required_tool_action",
                    "expected": "calendar.create",
                    "badOutput": "No action step was persisted.",
                    "correctedOutput": "calendar.create(title=\"Appointment\", start=\"2026-06-08T23:30:00Z\")",
                    "promptPrefix": "Set an appointment.",
                }
            ]
        },
        "recentTraces": [],
    }
    report_path.write_text(json.dumps(package), encoding="utf-8")

    report = load_runtime_audit_reports([report_path])[0]

    assert report["_sourceFormat"] == "lumen_in_app_dataset_package"
    assert report["_sourceLayer"] == "agentGroundingRuntimeAudit"
    failure_types = {failure["type"] for failure in report["failures"]}
    assert "missing_required_tool_action" in failure_types
    assert "agent_grounding_no_recent_model_traces" in failure_types


def test_persistent_runtime_diagnostics_export_ingests_bounded_summary(tmp_path: Path) -> None:
    report_path = tmp_path / "persistent-runtime-diagnostics-export.json"
    package = {
        "exportedAt": "2026-06-08T23:28:09Z",
        "appVersion": "1.0.0",
        "deviceModel": "iPhone",
        "systemName": "iOS",
        "systemVersion": "26.0",
        "metricKitPayloads": [{}, {}],
        "ndjson": '{"kind":"event"}\n{"kind":"event"}\n',
        "state": {
            "records": [
                {"id": "passed", "scenario": "plainFastPrompt", "status": "passed"},
                {
                    "id": "expected-cancel",
                    "scenario": "agentCancellation",
                    "status": "cancelled",
                    "metrics": {
                        "didCancel": True,
                        "cancellationReason": "persistent-diagnostics-agent-cancel",
                    },
                },
                {
                    "id": "failed",
                    "scenario": "liveAgentStream",
                    "status": "failed",
                    "remediationProposals": [
                        {
                            "id": "local-chat-model-required",
                            "title": "Select a local chat model",
                            "rationale": "Live inference could not run without a local chat runtime.",
                            "action": "Install or select a local chat model, then rerun the scenario.",
                            "severity": "warning",
                        }
                    ],
                },
            ]
        },
    }
    report_path.write_text(json.dumps(package), encoding="utf-8")

    report = load_runtime_audit_reports([report_path])[0]

    assert report["_sourceFormat"] == "persistent_runtime_diagnostics_export"
    assert report["recordCount"] == 3
    assert report["statusCounts"] == {"cancelled": 1, "failed": 1, "passed": 1}
    assert report["metricKitPayloadCount"] == 2
    assert report["ndjsonLineCount"] == 2
    assert report["remediationProposalCount"] == 1
    assert report["remediationSeverityCounts"] == {"warning": 1}
    assert len(report["failures"]) == 1
    assert report["failures"][0]["scenario"] == "liveAgentStream"
    assert report["failures"][0]["remediationSeverity"] == "warning"
    assert report["failures"][0]["remediationProposals"][0]["id"] == "local-chat-model-required"
    assert report["failures"][0]["remediationProposals"][0]["action"] == "Install or select a local chat model, then rerun the scenario."


def test_empty_trace_evidence_layer_envelope_generates_trace_gap(tmp_path: Path) -> None:
    report_path = tmp_path / "lumen-agent-runtime-traces.json"
    envelope = {
        "schemaVersion": "1.0.0",
        "generatedAt": "2026-05-03T00:00:00Z",
        "exportPolicy": {
            "format": "agent-runtime-traces-json",
            "sourceLayer": "agentBehaviorTraceRecorder",
            "ownsLiveE2EScenarios": False,
            "includesDeterministicStaticScenarios": False,
        },
        "payload": [],
    }
    report_path.write_text(json.dumps(envelope), encoding="utf-8")

    report = load_runtime_audit_reports([report_path])[0]

    assert report["_sourceLayer"] == "agentBehaviorTraceRecorder"
    assert report["traceCount"] == 0
    assert report["failures"][0]["type"] == "agent_grounding_no_recent_model_traces"


def test_self_model_eval_score_report_ingests_as_non_live_runtime_audit(tmp_path: Path) -> None:
    report_path = tmp_path / "self-model-eval-score.json"
    score_report = {
        "schemaVersion": "self_model_eval_score.v1",
        "scenarioCount": 2,
        "answeredCount": 1,
        "passedCount": 0,
        "failedCount": 1,
        "missingCount": 1,
        "allPassed": False,
        "results": [
            {
                "id": "eval-live-proof",
                "name": "self-model-testflight-proof",
                "answered": True,
                "passed": False,
                "score": 0.5,
                "checked": ["mustRequireLiveE2EEvidence"],
                "failures": ["live_e2e_evidence_requirement_missing"],
            },
            {
                "id": "eval-missing",
                "name": "self-model-current-location",
                "answered": False,
                "passed": False,
                "score": 0.0,
                "checked": [],
                "failures": ["missing_answer"],
            },
        ],
    }
    report_path.write_text(json.dumps(score_report), encoding="utf-8")

    report = load_runtime_audit_reports([report_path])[0]

    assert report["_sourceFormat"] == "self_model_eval_score_report"
    assert report["_sourceLayer"] == "selfModelEvalScore"
    assert report["ownsLiveE2EScenarios"] is False
    assert report["scenarioCount"] == 2
    assert report["failedCount"] == 1
    assert report["missingCount"] == 1
    failure_types = {failure["type"] for failure in report["failures"]}
    assert "self_model_runtime_state_claim_without_evidence" in failure_types
    assert "self_model_eval_answer_missing" in failure_types
