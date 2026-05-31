"""Tests for runtime audit and E2E report ingestion."""

# pylint: disable=missing-function-docstring,line-too-long

from pathlib import Path

from lumen_manifest_crawler.dataset.runtime_ingest import load_runtime_audit_reports


E2E_REPORT = """E2E Test Report
Passed: 5
Failed: 2

Training signals for next run:
• response-quality: 2 issues
• Capture failed prompts + final outputs into next fine-tuning dataset.
• Prioritize scenarios with repeated tool-boundary violations.

✅ Training eval: weather stays grounded
Prompt: What is the weather here and should I carry an umbrella?
Intent: weather / expected weather
Final: I need location access, or a city name, to check the weather.

❌ Training eval: memory save/recall
Prompt: Remember that I prefer concise bullet points, then tell me what you remembered.
Intent: memory / expected memory
Failures: Required final hint missing: remember
Final: Saved: I prefer concise bullet points.

❌ Training eval: communication drafting
Prompt: Draft an email to Alex with a professional update and ask one clarifying question.
Intent: emailDraft / expected emailDraft
Failures: Required final hint missing: question
Final: emailDraft
"""


GENERIC_E2E_REPORT = """E2E Test Report
Passed: 0
Failed: 2

❌ Training eval: memory custom detail
Prompt: Remember that my shop supplier is Delta Parts, then tell me what you remembered.
Intent: memory / expected memory
Failures: Required final hint missing: remember
Final: Saved.

❌ Training eval: email custom detail
Prompt: Draft an email to Sam about the delayed shipment and ask one clarifying question.
Intent: emailDraft / expected emailDraft
Failures: Required final hint missing: question
Final: emailDraft
"""


FINAL_WITH_EMAIL_HEADERS_REPORT = """E2E Test Report
Passed: 0
Failed: 1

❌ Training eval: communication drafting
Prompt: Draft an email to Alex with a professional update and ask one clarifying question.
Intent: emailDraft / expected emailDraft
Failures: Required final hint missing: question
Final: Subject: Project update

Hi Alex,

Here is the update.

Question: should I send this today?
"""


FINAL_WITH_GENERIC_CAPITALIZED_LINES_REPORT = """E2E Test Report
Passed: 0
Failed: 1

❌ Training eval: communication drafting
Prompt: Draft an email to Alex with a professional update and ask one clarifying question.
Intent: emailDraft / expected emailDraft
Failures: Required final hint missing: question
Final: Subject: Project update

Note: this line is part of the email body.
Summary: progress is moving forward.
Question: should I send this today?
"""


JSON_E2E_REPORT = {
    "kind": "lumen_e2e_test_report",
    "passed": 0,
    "failed": 2,
    "scenarios": [
        {
            "name": "web lookup grounding",
            "passed": False,
            "prompt": "Look up the latest release notes and summarize them.",
            "intent": "webLookup",
            "expectedIntent": "webLookup",
            "failures": "Required final hint missing: source",
            "final": "I found the release notes.",
        },
        {
            "name": "missing intent fallback",
            "passed": False,
            "prompt": "Search the web and give me a grounded summary.",
            "expectedIntent": "webLookup",
            "failures": "Required final hint missing: source",
            "final": "Summary complete.",
        },
    ],
}


def test_load_runtime_audit_reports_ingests_text_e2e_report(tmp_path: Path):
    report_path = tmp_path / "e2e-report.txt"
    report_path.write_text(E2E_REPORT, encoding="utf-8")

    reports = load_runtime_audit_reports([report_path])

    assert len(reports) == 1
    report = reports[0]
    assert report["_sourceFormat"] == "lumen_e2e_text_report"
    assert report["passed"] == 5
    assert report["failed"] == 2
    assert report["scenarioCount"] == 3
    assert "response-quality: 2 issues" in report["trainingSignals"]
    assert len(report["failures"]) == 2


def test_e2e_failures_become_repair_samples_with_corrected_outputs(tmp_path: Path):
    report_path = tmp_path / "e2e-report.md"
    report_path.write_text(E2E_REPORT, encoding="utf-8")

    failures = load_runtime_audit_reports([report_path])[0]["failures"]
    by_intent = {failure["e2eScenario"]["intent"]: failure for failure in failures}

    memory = by_intent["memory"]
    assert memory["type"] == "e2e_missing_required_final_hint_remember"
    assert memory["agent"] == "mouth"
    assert memory["sourceLayer"] == "e2eTextReport"
    assert memory["scenario"] == "Remember that I prefer concise bullet points, then tell me what you remembered."
    assert memory["actual"] == "Saved: I prefer concise bullet points."
    assert "remember" in memory["repairSample"]["correctedOutput"].casefold()
    assert memory["repairSample"]["curriculum"] == "grounded_response_quality"

    email = by_intent["emailDraft"]
    assert email["type"] == "e2e_missing_required_final_hint_question"
    assert email["agent"] == "mouth"
    assert "question" in email["repairSample"]["correctedOutput"].casefold()
    assert "Alex" in email["repairSample"]["correctedOutput"]
    assert email["repairSample"]["curriculum"] == "tool_boundary_response_quality"


def test_e2e_corrected_outputs_are_derived_from_prompt_not_fixed_templates(tmp_path: Path):
    report_path = tmp_path / "generic-e2e-report.txt"
    report_path.write_text(GENERIC_E2E_REPORT, encoding="utf-8")

    failures = load_runtime_audit_reports([report_path])[0]["failures"]
    by_intent = {failure["e2eScenario"]["intent"]: failure for failure in failures}

    memory_output = by_intent["memory"]["repairSample"]["correctedOutput"]
    assert "Delta Parts" in memory_output
    assert "concise bullet points" not in memory_output
    assert "remember" in memory_output.casefold()

    email_output = by_intent["emailDraft"]["repairSample"]["correctedOutput"]
    assert "Sam" in email_output
    assert "delayed shipment" in email_output
    assert "Alex" not in email_output
    assert "question" in email_output.casefold()


def test_final_multiline_field_preserves_email_subject_body_headers(tmp_path: Path):
    report_path = tmp_path / "email-final-report.log"
    report_path.write_text(FINAL_WITH_EMAIL_HEADERS_REPORT, encoding="utf-8")

    failure = load_runtime_audit_reports([report_path])[0]["failures"][0]
    actual = failure["actual"]

    assert "Subject: Project update" in actual
    assert "Hi Alex," in actual
    assert "Question: should I send this today?" in actual
    assert failure["repairSample"]["badOutput"] == actual


def test_final_multiline_field_preserves_generic_capitalized_body_lines(tmp_path: Path):
    report_path = tmp_path / "email-final-generic-headers.log"
    report_path.write_text(FINAL_WITH_GENERIC_CAPITALIZED_LINES_REPORT, encoding="utf-8")

    failure = load_runtime_audit_reports([report_path])[0]["failures"][0]
    actual = failure["actual"]

    assert "Subject: Project update" in actual
    assert "Note: this line is part of the email body." in actual
    assert "Summary: progress is moving forward." in actual
    assert "Question: should I send this today?" in actual
    assert failure["repairSample"]["badOutput"] == actual


def test_web_lookup_intent_routes_to_mouth_tool_boundary_curriculum(tmp_path: Path):
    report_path = tmp_path / "web-lookup-report.json"
    import json

    report_path.write_text(json.dumps(JSON_E2E_REPORT), encoding="utf-8")

    failures = load_runtime_audit_reports([report_path])[0]["failures"]
    web_lookup = failures[0]

    assert web_lookup["e2eScenario"]["intent"] == "webLookup"
    assert web_lookup["agent"] == "mouth"
    assert web_lookup["repairSample"]["agent"] == "mouth"
    assert web_lookup["repairSample"]["curriculum"] == "tool_boundary_response_quality"
    assert web_lookup["type"] == web_lookup["repairSample"]["violationCode"]


def test_expected_intent_is_used_when_intent_is_missing(tmp_path: Path):
    report_path = tmp_path / "missing-intent-report.json"
    import json

    report_path.write_text(json.dumps(JSON_E2E_REPORT), encoding="utf-8")

    failures = load_runtime_audit_reports([report_path])[0]["failures"]
    missing_intent = failures[1]

    assert missing_intent["e2eScenario"]["intent"] == "webLookup"
    assert missing_intent["e2eScenario"]["expectedIntent"] == "webLookup"
    assert missing_intent["agent"] == "mouth"
    assert missing_intent["repairSample"]["curriculum"] == "tool_boundary_response_quality"
    assert "unknown" not in missing_intent["type"]


def test_ingestion_flags_e2e_no_model_fallback_as_invalid_evidence(tmp_path: Path):
    report_path = tmp_path / "e2e-no-model-report.json"
    import json

    report = {
        "kind": "lumen_e2e_test_report",
        "passed": 1,
        "failed": 0,
        "scenarios": [
            {
                "name": "chat should run model",
                "passed": True,
                "prompt": "Explain actor isolation in Swift.",
                "intent": "chat",
                "expectedIntent": "chat",
                "failures": [],
                "final": "No model loaded; routing-only checks completed.",
                "events": [{"phase": "models", "message": "no chat model loaded"}],
            }
        ],
    }
    report_path.write_text(json.dumps(report), encoding="utf-8")

    normalized = load_runtime_audit_reports([report_path])[0]

    assert len(normalized["failures"]) == 1
    failure = normalized["failures"][0]
    assert failure["e2eScenario"]["skippedLiveModelRun"] is True
    assert "routing-only fallback is not valid E2E evidence" in failure["expected"][0]
    assert "Load the configured chat model" in failure["repairSample"]["correctedOutput"]


def test_in_app_package_preserves_trace_selected_tool_allowed_count(tmp_path: Path):
    report_path = tmp_path / "lumen-agent-grounding-audit.json"
    import json

    package = {
        "schemaVersion": "1.1.0",
        "generatedAt": "2026-05-03T00:00:00Z",
        "manifestSource": "AgentGrounding/agent_manifest/AgentBehaviorManifest.json",
        "usedRuntimeFallback": False,
        "exportPolicy": {"format": "agent-grounding-runtime-json-package", "ownsLiveE2EScenarios": False},
        "traceSelectedToolAllowedCount": 7,
        "traceParseErrorCount": 3,
        "recentTraces": [
            {"slot": "cortex", "promptPrefix": "route this", "selectedToolID": "calendar.create", "allowedToolIDs": ["calendar.create"]},
            {"slot": "cortex", "promptPrefix": "route this too", "selectedToolID": "web.search", "allowedToolIDs": ["maps.search"]},
        ],
    }
    report_path.write_text(json.dumps(package), encoding="utf-8")

    report = load_runtime_audit_reports([report_path])[0]
    assert report["_sourceFormat"] == "lumen_in_app_dataset_package"
    assert report["ownsLiveE2EScenarios"] is False
    assert report["traceSelectedToolAllowedCount"] == 7
    assert report["traceParseErrorCount"] == 3
    assert "agent_grounding_no_recent_model_traces" not in {failure["type"] for failure in report["failures"]}


def test_in_app_package_backfills_trace_selected_tool_allowed_count_when_missing(tmp_path: Path):
    report_path = tmp_path / "lumen-agent-grounding-audit-backfill.json"
    import json

    package = {
        "schemaVersion": "1.1.0",
        "generatedAt": "2026-05-03T00:00:00Z",
        "manifestSource": "AgentGrounding/agent_manifest/AgentBehaviorManifest.json",
        "usedRuntimeFallback": False,
        "exportPolicy": {"format": "agent-grounding-runtime-json-package", "ownsLiveE2EScenarios": False},
        "recentTraces": [
            {"slot": "cortex", "promptPrefix": "route this", "selectedToolID": "calendar.create", "allowedToolIDs": ["calendar.create"]},
            {"slot": "cortex", "promptPrefix": "route this too", "selectedToolID": "web.search", "allowedToolIDs": ["maps.search"], "parseError": "bad-json"},
            {"slot": "cortex", "promptPrefix": "final", "selectedToolID": None, "allowedToolIDs": [], "parseError": None},
        ],
    }
    report_path.write_text(json.dumps(package), encoding="utf-8")

    report = load_runtime_audit_reports([report_path])[0]
    assert report["traceSelectedToolAllowedCount"] == 1
    assert report["traceParseErrorCount"] == 1


def test_agent_grounding_package_ignores_static_scenario_results_by_default(tmp_path: Path):
    report_path = tmp_path / "lumen-agent-grounding-audit-with-static-scenarios.json"
    import json

    package = {
        "schemaVersion": "1.1.0",
        "generatedAt": "2026-05-03T00:00:00Z",
        "manifestSource": "AgentGrounding/agent_manifest/AgentBehaviorManifest.json",
        "usedRuntimeFallback": False,
        "exportPolicy": {
            "format": "agent-grounding-runtime-json-package",
            "sourceLayer": "agentGroundingRuntimeAudit",
            "ownsLiveE2EScenarios": False,
            "includesDeterministicStaticScenarios": True,
        },
        "scenarioResults": [
            {
                "id": "calendar::calendar.create",
                "passed": False,
                "failures": [
                    {
                        "type": "scenario_unknown_tool",
                        "agent": "cortex",
                        "expected": ["calendar.create"],
                        "actual": "calendar.create",
                        "scenario": "Create a calendar event.",
                        "problem": "Static scenario failure, not model execution.",
                    }
                ],
            }
        ],
        "recentTraces": [
            {"slot": "cortex", "promptPrefix": "route this", "selectedToolID": "calendar.create", "allowedToolIDs": ["calendar.create"]}
        ],
    }
    report_path.write_text(json.dumps(package), encoding="utf-8")

    report = load_runtime_audit_reports([report_path])[0]

    assert report["_sourceLayer"] == "agentGroundingRuntimeAudit"
    assert report["ownsLiveE2EScenarios"] is False
    assert report["ignoredScenarioResultCount"] == 1
    assert all(failure["type"] != "scenario_unknown_tool" for failure in report["failures"])


def test_agent_grounding_package_without_recent_traces_generates_export_quality_failure(tmp_path: Path):
    report_path = tmp_path / "lumen-agent-grounding-audit-empty-traces.json"
    import json

    package = {
        "schemaVersion": "1.1.0",
        "generatedAt": "2026-05-03T00:00:00Z",
        "manifestSource": "AgentGrounding/agent_manifest/AgentBehaviorManifest.json",
        "usedRuntimeFallback": False,
        "exportPolicy": {
            "format": "agent-grounding-runtime-json-package",
            "sourceLayer": "agentGroundingRuntimeAudit",
            "ownsLiveE2EScenarios": False,
        },
        "recentTraces": [],
    }
    report_path.write_text(json.dumps(package), encoding="utf-8")

    report = load_runtime_audit_reports([report_path])[0]

    failures = report["failures"]
    assert len(failures) == 1
    assert failures[0]["type"] == "agent_grounding_no_recent_model_traces"
    assert failures[0]["sourceLayer"] == "agentGroundingRuntimeAudit.exportQuality"
    assert "AgentBehaviorTraceRecorder.record is not wired" in failures[0]["problem"]


def test_e2e_owned_package_can_ingest_live_scenario_results(tmp_path: Path):
    report_path = tmp_path / "lumen-e2e-owned-package.json"
    import json

    package = {
        "schemaVersion": "1.1.0",
        "generatedAt": "2026-05-03T00:00:00Z",
        "manifestSource": "E2ETestRunner",
        "usedRuntimeFallback": False,
        "exportPolicy": {
            "format": "e2e-runtime-json-package",
            "sourceLayer": "e2eTestReport",
            "ownsLiveE2EScenarios": True,
        },
        "scenarioResults": [
            {
                "id": "training-memory-loop",
                "passed": False,
                "failures": [
                    {
                        "type": "missing_required_hint",
                        "agent": "mouth",
                        "expected": ["remember"],
                        "actual": "Saved.",
                        "scenario": "Remember this detail.",
                        "problem": "Required final hint missing: remember",
                    }
                ],
            }
        ],
        "recentTraces": [],
    }
    report_path.write_text(json.dumps(package), encoding="utf-8")

    report = load_runtime_audit_reports([report_path])[0]

    assert report["ownsLiveE2EScenarios"] is True
    assert report["ignoredScenarioResultCount"] == 0
    assert len(report["failures"]) == 1
    assert report["failures"][0]["sourceLayer"] == "e2eTestReport.scenarioResults"


def test_in_app_package_ignores_plaintext_mouth_final_parse_errors(tmp_path: Path):
    report_path = tmp_path / "lumen-agent-grounding-audit-mouth-plaintext.json"
    import json

    package = {
        "schemaVersion": "1.1.0",
        "generatedAt": "2026-05-03T00:00:00Z",
        "manifestSource": "AgentGrounding/agent_manifest/AgentBehaviorManifest.json",
        "usedRuntimeFallback": False,
        "exportPolicy": {
            "format": "agent-grounding-runtime-json-package",
            "sourceLayer": "agentGroundingRuntimeAudit",
            "ownsLiveE2EScenarios": False,
        },
        "recentTraces": [
            {
                "slot": "mouth",
                "stage": "mouth-final",
                "parseError": "noJSONObject",
                "promptPrefix": "Write the final user-facing answer. Do not output JSON.",
                "allowedToolIDs": [],
            }
        ],
    }
    report_path.write_text(json.dumps(package), encoding="utf-8")

    report = load_runtime_audit_reports([report_path])[0]

    assert report["traceParseErrorCount"] == 1
    assert all(failure["type"] != "trace_parse_error" for failure in report["failures"])


def test_in_app_package_ignores_agent_summary_parse_errors(tmp_path: Path):
    report_path = tmp_path / "lumen-agent-grounding-audit-agent-summary.json"
    import json

    package = {
        "schemaVersion": "1.1.0",
        "generatedAt": "2026-05-03T00:00:00Z",
        "manifestSource": "AgentGrounding/agent_manifest/AgentBehaviorManifest.json",
        "usedRuntimeFallback": False,
        "exportPolicy": {
            "format": "agent-grounding-runtime-json-package",
            "sourceLayer": "agentGroundingRuntimeAudit",
            "ownsLiveE2EScenarios": False,
        },
        "recentTraces": [
            {
                "slot": "cortex",
                "stage": "agent-summary",
                "parseError": "missingActionOrFinal",
                "promptPrefix": "User prompt: ... Original final answer: ...",
                "allowedToolIDs": [],
            }
        ],
    }
    report_path.write_text(json.dumps(package), encoding="utf-8")

    report = load_runtime_audit_reports([report_path])[0]

    assert report["traceParseErrorCount"] == 1
    assert all(failure["type"] != "trace_parse_error" for failure in report["failures"])


def test_in_app_package_ignores_cortex_prompt_echo_parse_errors(tmp_path: Path):
    report_path = tmp_path / "lumen-agent-grounding-audit-cortex-prompt-echo.json"
    import json

    package = {
        "schemaVersion": "1.1.0",
        "generatedAt": "2026-05-03T00:00:00Z",
        "manifestSource": "AgentGrounding/agent_manifest/AgentBehaviorManifest.json",
        "usedRuntimeFallback": False,
        "exportPolicy": {
            "format": "agent-grounding-runtime-json-package",
            "sourceLayer": "agentGroundingRuntimeAudit",
            "ownsLiveE2EScenarios": False,
        },
        "recentTraces": [
            {
                "slot": "cortex",
                "stage": "cortex-orchestrator-json",
                "parseError": "noJSONObject",
                "promptPrefix": "You are Lumen Cortex orchestrator step 2. Return exactly one JSON object and no markdown.",
                "rawOutputPrefix": "You are Lumen, a helpful, concise on-device AI assistant.",
                "allowedToolIDs": ["contacts.search", "mail.draft"],
            }
        ],
    }
    report_path.write_text(json.dumps(package), encoding="utf-8")

    report = load_runtime_audit_reports([report_path])[0]

    assert report["traceParseErrorCount"] == 1
    assert all(failure["type"] != "trace_parse_error" for failure in report["failures"])
