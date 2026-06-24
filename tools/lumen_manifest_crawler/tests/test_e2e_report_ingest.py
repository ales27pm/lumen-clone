"""Tests for runtime audit and E2E report ingestion."""

# pylint: disable=missing-function-docstring,line-too-long

import json
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


def test_behavior_repair_samples_skip_meta_instructions(tmp_path: Path):
    report_path = tmp_path / "behavior-audit.json"
    report_path.write_text(
        """{
          "generatedAt": "2026-06-09T01:55:41Z",
          "repairSamples": [
            {
              "agent": "executor",
              "violationCode": "missing_required_tool_argument",
              "promptPrefix": "Read my latest email",
              "badOutput": "message",
              "correctedOutput": "Emit a tool call with every required manifest argument populated, or ask for clarification before tool execution.",
              "lesson": "Executor must satisfy required argument schemas exactly or request clarification."
            },
            {
              "agent": "executor",
              "violationCode": "missing_required_tool_argument",
              "promptPrefix": "Read my latest email",
              "badOutput": "outlook.message.read(message=latest)",
              "correctedOutput": "outlook.messages.list(limit=1)",
              "lesson": "List the mailbox before reading a contextual message reference."
            }
          ]
        }""",
        encoding="utf-8",
    )

    failures = load_runtime_audit_reports([report_path])[0]["failures"]

    assert len(failures) == 1
    assert failures[0]["repairSample"]["correctedOutput"] == "outlook.messages.list(limit=1)"


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
    assert "AgentService" in failure["repairSample"]["correctedOutput"]


def test_ingestion_flags_live_e2e_without_model_evidence_event(tmp_path: Path):
    report_path = tmp_path / "e2e-missing-model-evidence.json"
    import json

    report = {
        "kind": "lumen_e2e_test_report",
        "passed": 1,
        "failed": 0,
        "scenarios": [
            {
                "name": "chat should run model",
                "passed": True,
                "requiresAgentRun": True,
                "prompt": "Explain actor isolation in Swift.",
                "intent": "chat",
                "expectedIntent": "chat",
                "failures": [],
                "final": "Actor isolation protects mutable state.",
                "events": [{"phase": "models", "message": "chat fleet ready"}],
            }
        ],
    }
    report_path.write_text(json.dumps(report), encoding="utf-8")
    (tmp_path / "agent-behavior-traces.jsonl").write_text("", encoding="utf-8")

    normalized = load_runtime_audit_reports([report_path])[0]

    assert len(normalized["failures"]) == 1
    failure = normalized["failures"][0]
    assert failure["rootCauseCategory"] == "no_correlated_model_turn"
    assert failure["e2eScenario"]["skippedLiveModelRun"] is False
    assert "correlated model-backed AgentBehaviorTrace" in failure["expected"][0]


def test_ingestion_accepts_live_e2e_with_model_evidence_event(tmp_path: Path):
    report_path = tmp_path / "e2e-with-model-evidence.json"
    import json

    report = {
        "kind": "lumen_e2e_test_report",
        "passed": 1,
        "failed": 0,
        "scenarios": [
            {
                "name": "chat should run model",
                "passed": True,
                "requiresAgentRun": True,
                "prompt": "Explain actor isolation in Swift.",
                "intent": "chat",
                "expectedIntent": "chat",
                "failures": [],
                "final": "Actor isolation protects mutable state.",
                "events": [
                    {"phase": "models", "message": "chat fleet ready"},
                    {"phase": "model-evidence", "message": "runtime=sharedAdapter, stage=agent-json, elapsedMs=6400, outputTokens=42, adapter=mouth"},
                ],
            }
        ],
    }
    report_path.write_text(json.dumps(report), encoding="utf-8")

    normalized = load_runtime_audit_reports([report_path])[0]

    assert normalized["failures"] == []
    assert normalized["scenarios"][0]["modelEvidenceStatus"] == "valid_model_backed_evidence"


def test_ingestion_matches_sidecar_by_correlation_despite_prompt_mismatch(tmp_path: Path):
    report_path = tmp_path / "e2e-correlated-sidecar.json"
    import json

    e2e_run_id = "11111111-1111-4111-8111-111111111111"
    report = {
        "kind": "lumen_e2e_test_report",
        "passed": 1,
        "failed": 0,
        "results": [
            {
                "scenarioID": "training-general-chat",
                "kind": "training",
                "title": "Training eval: pure chat quality",
                "passed": True,
                "requiresAgentRun": True,
                "prompt": "Explain tradeoffs between precision and recall.",
                "actualIntent": "chat",
                "expectedIntent": "chat",
                "e2eRunID": e2e_run_id,
                "failures": [],
                "finalText": "Precision is exactness; recall is coverage.",
                "events": [{"phase": "model-evidence", "message": "missing fresh AgentBehaviorTrace modelTurn"}],
            }
        ],
    }
    report_path.write_text(json.dumps(report), encoding="utf-8")
    (tmp_path / "agent-behavior-traces.jsonl").write_text(json.dumps({
        "event": "modelTurn",
        "stage": "agent-json-step-0",
        "runtimePath": "agent-model",
        "parseError": None,
        "rawOutputPrefix": "{\"final\":\"Precision is exactness; recall is coverage.\"}",
        "promptPrefix": "Grounded wrapper prompt without the original wording.",
        "scenarioID": "training-general-chat",
        "e2eRunID": e2e_run_id,
    }) + "\n", encoding="utf-8")

    normalized = load_runtime_audit_reports([report_path])[0]

    assert normalized["failures"] == []
    assert normalized["scenarios"][0]["modelEvidenceStatus"] == "valid_model_backed_evidence"
    assert normalized["scenarios"][0]["modelEvidenceTrace"]["matchedBy"] == "correlation"


def test_training_deterministic_model_evidence_is_not_model_backed(tmp_path: Path):
    report_path = tmp_path / "e2e-training-deterministic.json"
    import json

    report = {
        "kind": "lumen_e2e_test_report",
        "passed": 1,
        "failed": 0,
        "scenarios": [
            {
                "id": "training-weather-grounded",
                "name": "Training eval: weather stays grounded",
                "kind": "training",
                "passed": True,
                "requiresAgentRun": True,
                "prompt": "What is the weather here?",
                "intent": "weather",
                "expectedIntent": "weather",
                "failures": [],
                "final": "Weather summary.",
                "events": [{"phase": "model-evidence", "message": "runtime=deterministic-compatibility, kind=policy-first-deterministic, stage=compatibility-tool-action"}],
            }
        ],
    }
    report_path.write_text(json.dumps(report), encoding="utf-8")

    normalized = load_runtime_audit_reports([report_path])[0]

    failure = normalized["failures"][0]
    assert failure["rootCauseCategory"] == "deterministic_compatibility_not_training_evidence"
    assert failure["e2eScenario"]["skippedLiveModelRun"] is False
    assert "deterministic compatibility" in failure["problem"]


def test_missing_sidecar_trace_export_is_distinguishable(tmp_path: Path):
    report_path = tmp_path / "e2e-missing-sidecar.json"
    import json

    report = {
        "kind": "lumen_e2e_test_report",
        "passed": 0,
        "failed": 1,
        "scenarios": [
            {
                "id": "training-general-chat",
                "name": "Training eval: pure chat quality",
                "kind": "training",
                "passed": False,
                "requiresAgentRun": True,
                "prompt": "Explain precision and recall.",
                "intent": "chat",
                "expectedIntent": "chat",
                "failures": ["Live E2E scenario did not record model-backed generation evidence"],
                "final": "Precision is exactness; recall is coverage.",
                "events": [{"phase": "model-evidence", "message": "missing fresh AgentBehaviorTrace modelTurn"}],
            }
        ],
    }
    report_path.write_text(json.dumps(report), encoding="utf-8")

    normalized = load_runtime_audit_reports([report_path])[0]

    failure = normalized["failures"][0]
    assert failure["rootCauseCategory"] == "missing_sidecar_trace_export"
    assert failure["e2eScenario"]["skippedLiveModelRun"] is False
    assert "sidecar export" in failure["problem"]


def test_report_events_classify_empty_output_and_parse_error(tmp_path: Path):
    report_path = tmp_path / "e2e-event-diagnostics.json"
    import json

    report = {
        "kind": "lumen_e2e_test_report",
        "passed": 0,
        "failed": 2,
        "scenarios": [
            {
                "id": "training-empty",
                "name": "Training eval: empty",
                "kind": "training",
                "passed": False,
                "requiresAgentRun": True,
                "prompt": "Empty output prompt.",
                "intent": "chat",
                "expectedIntent": "chat",
                "failures": ["Live E2E scenario did not record model-backed generation evidence"],
                "final": "",
                "events": [{"phase": "model-evidence", "message": "found AgentBehaviorTrace modelTurn but model stream returned no tokens; stage=agent-json-step-0; runtimePath=agent-model; parseError=empty; outputTokens=0"}],
            },
            {
                "id": "training-parse",
                "name": "Training eval: parse",
                "kind": "training",
                "passed": False,
                "requiresAgentRun": True,
                "prompt": "Parse error prompt.",
                "intent": "chat",
                "expectedIntent": "chat",
                "failures": ["Live E2E scenario did not record model-backed generation evidence"],
                "final": "plain text",
                "events": [{"phase": "model-evidence", "message": "found AgentBehaviorTrace modelTurn but parseError=noJSONObject; stage=agent-json-step-0; runtimePath=agent-model; outputTokens=4"}],
            },
        ],
    }
    report_path.write_text(json.dumps(report), encoding="utf-8")

    normalized = load_runtime_audit_reports([report_path])[0]

    failures_by_prompt = {failure["scenario"]: failure for failure in normalized["failures"]}
    assert failures_by_prompt["Empty output prompt."]["rootCauseCategory"] == "agent_model_empty_output"
    assert failures_by_prompt["Parse error prompt."]["rootCauseCategory"] == "agent_model_parse_error"


def test_ingestion_uses_sidecar_empty_agent_json_trace_as_precise_root_cause(tmp_path: Path):
    report_path = tmp_path / "latest-e2e-report.json"
    report_path.write_text(json.dumps({
        "kind": "lumen_e2e_test_report",
        "passed": 0,
        "failed": 1,
        "scenarios": [
            {
                "name": "Training eval: pure chat quality",
                "passed": False,
                "requiresAgentRun": True,
                "prompt": "Explain tradeoffs between precision and recall in retrieval systems in plain English.",
                "intent": "chat",
                "expectedIntent": "chat",
                "failures": ["Live E2E scenario did not record model-backed generation evidence"],
                "final": "Precision is exactness; recall is coverage.",
                "events": [{"phase": "model-evidence", "message": "missing fresh AgentBehaviorTrace modelTurn"}],
            }
        ],
    }), encoding="utf-8")
    (tmp_path / "agent-behavior-traces.jsonl").write_text(json.dumps({
        "event": "modelTurn",
        "stage": "agent-json-step-0",
        "runtimePath": "agent-model",
        "parseError": "empty",
        "rawOutputPrefix": "",
        "promptPrefix": "Explain tradeoffs between precision and recall in retrieval systems in plain English.",
    }) + "\n", encoding="utf-8")
    (tmp_path / "agent-parse-failures.jsonl").write_text(json.dumps({
        "modelName": "agent-json",
        "parseError": "empty",
        "rawOutputPrefix": "",
        "userTurnPrefix": "User request:\nExplain tradeoffs between precision and recall in retrieval systems in plain English.",
    }) + "\n", encoding="utf-8")

    normalized = load_runtime_audit_reports([report_path])[0]

    failure = normalized["failures"][0]
    assert failure["rootCauseCategory"] == "agent_json_empty_generation"
    assert failure["e2eScenario"]["skippedLiveModelRun"] is False
    assert failure["e2eScenario"]["modelEvidenceTrace"]["stage"] == "agent-json-step-0"
    assert "agent-json emitted empty output" in failure["problem"]


def test_ingestion_does_not_match_empty_sidecar_prompt_to_scenario(tmp_path: Path):
    report_path = tmp_path / "latest-e2e-report.json"
    report_path.write_text(json.dumps({
        "kind": "lumen_e2e_test_report",
        "passed": 0,
        "failed": 1,
        "scenarios": [
            {
                "name": "Training eval: pure chat quality",
                "passed": False,
                "requiresAgentRun": True,
                "prompt": "Explain tradeoffs between precision and recall in retrieval systems in plain English.",
                "intent": "chat",
                "expectedIntent": "chat",
                "failures": ["Live E2E scenario did not record model-backed generation evidence"],
                "final": "Precision is exactness; recall is coverage.",
                "events": [],
            }
        ],
    }), encoding="utf-8")
    (tmp_path / "agent-behavior-traces.jsonl").write_text(json.dumps({
        "event": "modelTurn",
        "stage": "agent-json-step-0",
        "runtimePath": "agent-model",
        "parseError": "empty",
        "rawOutputPrefix": "",
        "promptPrefix": "",
    }) + "\n", encoding="utf-8")

    normalized = load_runtime_audit_reports([report_path])[0]

    failure = normalized["failures"][0]
    assert failure["rootCauseCategory"] == "no_correlated_model_turn"
    assert failure["e2eScenario"]["skippedLiveModelRun"] is False


def test_ingestion_rejects_sidecar_outside_scenario_time_window(tmp_path: Path):
    report_path = tmp_path / "latest-e2e-report.json"
    report_path.write_text(json.dumps({
        "kind": "lumen_e2e_test_report",
        "passed": 0,
        "failed": 1,
        "results": [
            {
                "scenarioID": "training-general-chat",
                "title": "Training eval: pure chat quality",
                "passed": False,
                "requiresAgentRun": True,
                "prompt": "Explain tradeoffs between precision and recall in retrieval systems in plain English.",
                "actualIntent": "chat",
                "expectedIntent": "chat",
                "failures": ["Live E2E scenario did not record model-backed generation evidence"],
                "finalText": "Precision is exactness; recall is coverage.",
                "events": [],
                "startedAt": "2026-06-23T10:34:00Z",
                "finishedAt": "2026-06-23T10:34:10Z",
            }
        ],
    }), encoding="utf-8")
    (tmp_path / "agent-behavior-traces.jsonl").write_text(json.dumps({
        "event": "modelTurn",
        "stage": "agent-json-step-0",
        "runtimePath": "agent-model",
        "parseError": "empty",
        "rawOutputPrefix": "",
        "promptPrefix": "Explain tradeoffs between precision and recall in retrieval systems in plain English.",
        "createdAt": "2026-06-23T11:34:00Z",
    }) + "\n", encoding="utf-8")

    normalized = load_runtime_audit_reports([report_path])[0]

    failure = normalized["failures"][0]
    assert failure["rootCauseCategory"] == "no_correlated_model_turn"
    assert failure["e2eScenario"]["skippedLiveModelRun"] is False


def test_text_e2e_report_uses_sidecar_diagnostics(tmp_path: Path):
    report_path = tmp_path / "latest-e2e-report.txt"
    report_path.write_text("""E2E Test Report
Passed: 0
Failed: 1

❌ Training eval: pure chat quality
Prompt: Explain tradeoffs between precision and recall in retrieval systems in plain English.
Intent: chat / expected chat
Failures: Live E2E scenario did not record model-backed generation evidence
Final: Precision is exactness; recall is coverage.
""", encoding="utf-8")
    (tmp_path / "agent-behavior-traces.jsonl").write_text(json.dumps({
        "event": "modelTurn",
        "stage": "agent-json-step-0",
        "runtimePath": "agent-model",
        "parseError": "empty",
        "rawOutputPrefix": "",
        "promptPrefix": "Explain tradeoffs between precision and recall in retrieval systems in plain English.",
    }) + "\n", encoding="utf-8")

    normalized = load_runtime_audit_reports([report_path])[0]

    failure = normalized["failures"][0]
    assert failure["rootCauseCategory"] == "agent_json_empty_generation"
    assert failure["e2eScenario"]["skippedLiveModelRun"] is False


def test_ingestion_distinguishes_missing_empty_parse_valid_and_policy_first_evidence(tmp_path: Path):
    report_path = tmp_path / "e2e-cases.json"
    report_path.write_text(json.dumps({
        "kind": "lumen_e2e_test_report",
        "passed": 2,
        "failed": 3,
        "scenarios": [
            {
                "name": "missing",
                "passed": False,
                "requiresAgentRun": True,
                "prompt": "Missing evidence prompt.",
                "intent": "chat",
                "expectedIntent": "chat",
                "failures": ["Live E2E scenario did not record model-backed generation evidence"],
                "final": "final",
                "events": [],
            },
            {
                "name": "empty",
                "passed": False,
                "requiresAgentRun": True,
                "prompt": "Empty output prompt.",
                "intent": "chat",
                "expectedIntent": "chat",
                "failures": ["Live E2E scenario did not record model-backed generation evidence"],
                "final": "final",
                "events": [],
            },
            {
                "name": "parse",
                "passed": False,
                "requiresAgentRun": True,
                "prompt": "Parse error prompt.",
                "intent": "chat",
                "expectedIntent": "chat",
                "failures": ["Live E2E scenario did not record model-backed generation evidence"],
                "final": "final",
                "events": [],
            },
            {
                "name": "valid",
                "passed": True,
                "requiresAgentRun": True,
                "prompt": "Valid model prompt.",
                "intent": "chat",
                "expectedIntent": "chat",
                "failures": [],
                "final": "final",
                "events": [{"phase": "model-evidence", "message": "runtime=sharedAdapter, kind=model-backed, stage=agent-json, parseError=none"}],
            },
            {
                "name": "policy",
                "passed": True,
                "requiresAgentRun": True,
                "prompt": "Policy first prompt.",
                "intent": "weather",
                "expectedIntent": "weather",
                "failures": [],
                "final": "final",
                "events": [{"phase": "model-evidence", "message": "runtime=deterministic-compatibility, kind=policy-first-deterministic, stage=compatibility-tool-action"}],
            },
        ],
    }), encoding="utf-8")
    (tmp_path / "agent-behavior-traces.jsonl").write_text("\n".join([
        json.dumps({
            "event": "modelTurn",
            "stage": "agent-json-step-0",
            "runtimePath": "agent-model",
            "parseError": "empty",
            "rawOutputPrefix": "",
            "promptPrefix": "Empty output prompt.",
        }),
        json.dumps({
            "event": "modelTurn",
            "stage": "agent-json-step-0",
            "runtimePath": "agent-model",
            "parseError": "noJSONObject",
            "rawOutputPrefix": "plain text",
            "promptPrefix": "Parse error prompt.",
        }),
    ]) + "\n", encoding="utf-8")

    normalized = load_runtime_audit_reports([report_path])[0]

    failures_by_prompt = {failure["scenario"]: failure for failure in normalized["failures"]}
    assert failures_by_prompt["Missing evidence prompt."]["rootCauseCategory"] == "no_correlated_model_turn"
    assert failures_by_prompt["Missing evidence prompt."]["e2eScenario"]["skippedLiveModelRun"] is False
    assert failures_by_prompt["Empty output prompt."]["rootCauseCategory"] == "agent_json_empty_generation"
    assert failures_by_prompt["Parse error prompt."]["rootCauseCategory"] == "agent_json_parse_error"
    assert "Valid model prompt." not in failures_by_prompt
    assert "Policy first prompt." not in failures_by_prompt


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


def test_in_app_package_ignores_non_tool_scoped_parse_errors(tmp_path: Path):
    report_path = tmp_path / "lumen-agent-grounding-audit-direct-chat.json"
    import json

    package = {
        "schemaVersion": "1.1.0",
        "generatedAt": "2026-06-08T00:00:00Z",
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
                "stage": "agent-json",
                "parseError": "noJSONObject",
                "promptPrefix": "Explain precision and recall in plain English.",
                "rawOutputPrefix": "Precision is about exactness; recall is about coverage.",
                "selectedToolID": None,
                "allowedToolIDs": [],
            }
        ],
    }
    report_path.write_text(json.dumps(package), encoding="utf-8")

    report = load_runtime_audit_reports([report_path])[0]

    assert report["traceParseErrorCount"] == 1
    assert all(failure["type"] != "trace_parse_error" for failure in report["failures"])


def test_in_app_package_reports_tool_scoped_parse_errors(tmp_path: Path):
    report_path = tmp_path / "lumen-agent-grounding-audit-tool-scoped-parse.json"
    import json

    package = {
        "schemaVersion": "1.1.0",
        "generatedAt": "2026-06-08T00:00:00Z",
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
                "stage": "agent-json",
                "parseError": "noJSONObject",
                "promptPrefix": "Search my calendar for next event",
                "rawOutputPrefix": "No events in the next 7 days.",
                "selectedToolID": None,
                "allowedToolIDs": ["calendar.list"],
            }
        ],
    }
    report_path.write_text(json.dumps(package), encoding="utf-8")

    report = load_runtime_audit_reports([report_path])[0]

    assert report["traceParseErrorCount"] == 1
    assert any(failure["type"] == "trace_parse_error" for failure in report["failures"])
