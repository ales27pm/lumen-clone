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
