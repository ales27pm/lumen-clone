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
                "requiresAgentRun": True,
                "prompt": "Explain actor isolation in Swift.",
                "intent": "chat",
                "expectedIntent": "chat",
                "failures": [],
                "final": "No model loaded; routing-only checks completed.",
                "events": [
                    {"phase": "models", "message": "no chat model loaded"},
                    {"phase": "model-evidence", "message": "AgentService model path was not entered; reason=model not loaded; scenarioID=chat should run model,e2eRunID=11111111-1111-4111-8111-111111111111"},
                ],
            }
        ],
    }
    report_path.write_text(json.dumps(report), encoding="utf-8")

    normalized = load_runtime_audit_reports([report_path])[0]

    assert len(normalized["failures"]) == 1
    failure = normalized["failures"][0]
    assert failure["rootCauseCategory"] is None
    assert failure["e2eScenario"]["skippedLiveModelRun"] is True
    assert failure["e2eScenario"]["modelEvidenceRootCause"] is None
    assert "routing-only fallback is not valid E2E evidence" in failure["expected"][0]
    assert "Load the configured chat model" in failure["repairSample"]["correctedOutput"]
    assert "AssistantKernel model-backed structured generation path" in failure["repairSample"]["correctedOutput"]


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


def test_ingestion_keeps_resource_budget_preflight_out_of_training_repairs(tmp_path: Path):
    report_path = tmp_path / "latest-e2e-report.json"
    import json

    failure = (
        "executor preflight failed: resource-budget-denied-before-prompt-eval; "
        "slot=.executor; budgetReason=strict-live-training.executor-preflight: thermalState=serious"
    )
    report = {
        "kind": "lumen_e2e_test_report",
        "passed": 0,
        "failed": 1,
        "results": [
            {
                "scenarioID": "executor-runtime-preflight",
                "kind": "training",
                "title": "Executor runtime preflight",
                "passed": False,
                "requiresAgentRun": True,
                "prompt": "What is the weather here and should I carry an umbrella?",
                "actualIntent": "preflight",
                "expectedIntent": "weather",
                "failures": [failure],
                "finalText": "",
                "events": [{"phase": "executor-preflight", "message": failure}],
            }
        ],
    }
    report_path.write_text(json.dumps(report), encoding="utf-8")

    normalized = load_runtime_audit_reports([report_path])[0]

    assert len(normalized["failures"]) == 1
    failure_record = normalized["failures"][0]
    assert failure_record["type"] == "e2e_runtime_environment_deferred"
    assert failure_record["rootCauseCategory"] == "runtime_environment_deferred"
    assert failure_record["trainable"] is False
    assert failure_record["repairSample"]["trainable"] is False
    assert "must be exported as diagnostics" in failure_record["expected"][0]


def test_ingestion_keeps_scene_phase_preflight_out_of_training_repairs(tmp_path: Path):
    report_path = tmp_path / "latest-e2e-report.json"
    import json

    failure = "Live E2E preflight blocked model-backed generation before prompt evaluation: live-e2e.pre-scenario: scenePhase=inactive"
    report = {
        "kind": "lumen_e2e_test_report",
        "passed": 0,
        "failed": 1,
        "results": [
            {
                "scenarioID": "live-alarm-countdown-direct",
                "kind": "toolGuard",
                "title": "Live alarm countdown",
                "passed": False,
                "requiresAgentRun": True,
                "prompt": "Start a timer for 10 minutes.",
                "actualIntent": "preflight",
                "expectedIntent": "alarm",
                "failures": [failure],
                "finalText": "Live E2E paused before starting this scenario: live-e2e.pre-scenario: scenePhase=inactive.",
                "events": [{"phase": "live-runtime-preflight", "message": "blocked before model prompt evaluation; reason=live-e2e.pre-scenario: scenePhase=inactive"}],
                "metadata": {
                    "failureKind": "liveRuntimeScenePhaseUnavailable",
                    "budgetPolicy": "foregroundInteractive",
                    "budgetDenialReason": "live-e2e.pre-scenario: scenePhase=inactive",
                    "actionable": "false",
                    "trainingSignal": "false",
                },
            }
        ],
    }
    report_path.write_text(json.dumps(report), encoding="utf-8")

    normalized = load_runtime_audit_reports([report_path])[0]

    failure_record = normalized["failures"][0]
    assert failure_record["type"] == "e2e_runtime_environment_deferred"
    assert failure_record["trainable"] is False
    assert failure_record["repairSample"]["trainable"] is False
    assert failure_record["e2eScenario"]["metadata"]["trainingSignal"] == "false"


def test_ingestion_keeps_alarmkit_unavailable_out_of_training_repairs(tmp_path: Path):
    report_path = tmp_path / "alarmkit-unavailable-e2e-report.json"
    import json

    failure = "AlarmKit runtime unavailable for expected tool alarm.authorization_status; device-runtime evidence required."
    report = {
        "kind": "lumen_e2e_test_report",
        "passed": 0,
        "failed": 1,
        "results": [
            {
                "scenarioID": "live-alarm-status-direct",
                "kind": "toolGuard",
                "title": "Live alarm status",
                "passed": False,
                "requiresAgentRun": True,
                "prompt": "Check alarm authorization status.",
                "actualIntent": "alarm",
                "expectedIntent": "alarm",
                "failures": [failure],
                "finalText": "AlarmKit availability: unavailable (requires iOS 26.0+ and an AlarmKit-capable device runtime).",
                "events": [{"phase": "step", "message": "observation alarm.authorization_status: AlarmKit availability: unavailable"}],
                "metadata": {
                    "expectedToolID": "alarm.authorization_status",
                    "scenarioBankKind": "direct",
                    "failureKind": "liveRuntimeAlarmKitUnavailable",
                    "actionable": "false",
                    "trainingSignal": "false",
                    "runtimeEvidence": "device-runtime-required",
                },
            }
        ],
    }
    report_path.write_text(json.dumps(report), encoding="utf-8")

    normalized = load_runtime_audit_reports([report_path])[0]

    failure_record = normalized["failures"][0]
    assert failure_record["type"] == "e2e_runtime_environment_deferred"
    assert failure_record["trainable"] is False
    assert failure_record["repairSample"]["trainable"] is False
    assert failure_record["e2eScenario"]["metadata"]["failureKind"] == "liveRuntimeAlarmKitUnavailable"


def test_ingestion_keeps_cpu_watchdog_degraded_out_of_training_repairs(tmp_path: Path):
    report_path = tmp_path / "cpu-watchdog-e2e-report.json"
    import json

    report = {
        "kind": "lumen_e2e_test_report",
        "passed": 0,
        "failed": 1,
        "results": [
            {
                "scenarioID": "training-rag-grounding",
                "kind": "training",
                "title": "Training eval: RAG grounding",
                "passed": False,
                "requiresAgentRun": True,
                "prompt": "Search local docs and summarize modules.",
                "actualIntent": "rag",
                "expectedIntent": "rag",
                "failures": ["Live runtime CPU watchdog degraded before completing model-backed scenario."],
                "finalText": "I couldn't complete the structured agent turn because agent-json produced no JSON output. Reason: cpu-watchdog-degraded.",
                "events": [{"phase": "agent-runtime", "message": "cpu-watchdog-degraded"}],
                "metadata": {
                    "failureKind": "liveRuntimeCPUWatchdogDegraded",
                    "actionable": "false",
                    "trainingSignal": "false",
                },
            }
        ],
    }
    report_path.write_text(json.dumps(report), encoding="utf-8")

    normalized = load_runtime_audit_reports([report_path])[0]

    failure_record = normalized["failures"][0]
    assert failure_record["type"] == "e2e_runtime_environment_deferred"
    assert failure_record["trainable"] is False
    assert failure_record["repairSample"]["trainable"] is False
    assert failure_record["e2eScenario"]["metadata"]["failureKind"] == "liveRuntimeCPUWatchdogDegraded"
    assert "Capture failed prompts" not in "\n".join(normalized.get("trainingSignals") or [])


def test_ingestion_keeps_rag_empty_index_out_of_training_repairs(tmp_path: Path):
    report_path = tmp_path / "rag-empty-e2e-report.json"
    import json

    report = {
        "kind": "lumen_e2e_test_report",
        "passed": 0,
        "failed": 1,
        "results": [
            {
                "scenarioID": "training-rag-grounding",
                "kind": "training",
                "title": "Training eval: RAG grounding",
                "passed": False,
                "requiresAgentRun": True,
                "prompt": "Search my files for architecture notes and summarize key modules.",
                "actualIntent": "rag",
                "expectedIntent": "rag",
                "failures": ["RAG empty local index."],
                "finalText": "No matching files found for 'architecture notes'. Your local index appears empty. Import or create local files/notes, then run reindex files.",
                "events": [{"phase": "step", "message": "rag.search: No matching files found for 'architecture notes'. Your local index appears empty."}],
                "metadata": {},
            }
        ],
    }
    report_path.write_text(json.dumps(report), encoding="utf-8")

    normalized = load_runtime_audit_reports([report_path])[0]

    failure_record = normalized["failures"][0]
    assert failure_record["type"] == "e2e_runtime_environment_deferred"
    assert failure_record["trainable"] is False
    assert failure_record["repairSample"]["trainable"] is False
    assert "Capture failed prompts" not in "\n".join(normalized.get("trainingSignals") or [])


def test_ingestion_keeps_internal_routing_json_out_of_training_repairs(tmp_path: Path):
    report_path = tmp_path / "internal-json-e2e-report.json"
    import json

    leaked = '{"intent":"webSearch","nextModel":"rag","reasoningSummary":"bad","requiresApproval":false,"sourceFile":"ios/Lumen/Models/ToolDefinition.swift"}'
    report = {
        "kind": "lumen_e2e_test_report",
        "passed": 0,
        "failed": 1,
        "results": [
            {
                "scenarioID": "training-web-research",
                "kind": "training",
                "title": "Training eval: web research synthesis",
                "passed": False,
                "requiresAgentRun": True,
                "prompt": "Search the web for two recent Swift concurrency best practices and summarize them.",
                "actualIntent": "webSearch",
                "expectedIntent": "webSearch",
                "failures": ["Live agent returned fallback/error text instead of completing the scenario"],
                "finalText": leaked,
                "events": [],
                "metadata": {},
            }
        ],
    }
    report_path.write_text(json.dumps(report), encoding="utf-8")

    normalized = load_runtime_audit_reports([report_path])[0]

    failure_record = normalized["failures"][0]
    assert failure_record["type"] == "e2e_runtime_environment_deferred"
    assert failure_record["trainable"] is False
    assert failure_record["repairSample"]["trainable"] is False
    assert "Capture failed prompts" not in "\n".join(normalized.get("trainingSignals") or [])


def test_ingestion_keeps_partial_internal_routing_json_out_of_training_repairs(tmp_path: Path):
    report_path = tmp_path / "partial-internal-json-e2e-report.json"
    leaked = '{"intent":"webSearch","nextModel":"rag","reasoningSummary":"bad"}'
    report = {
        "kind": "lumen_e2e_test_report",
        "passed": 0,
        "failed": 1,
        "results": [
            {
                "scenarioID": "training-web-research",
                "kind": "training",
                "title": "Training eval: web research synthesis",
                "passed": False,
                "requiresAgentRun": True,
                "prompt": "Search the web for two recent Swift concurrency best practices and summarize them.",
                "actualIntent": "webSearch",
                "expectedIntent": "webSearch",
                "failures": ["Live agent returned fallback/error text instead of completing the scenario"],
                "finalText": leaked,
                "events": [],
                "metadata": {},
            }
        ],
    }
    report_path.write_text(json.dumps(report), encoding="utf-8")

    normalized = load_runtime_audit_reports([report_path])[0]

    failure_record = normalized["failures"][0]
    assert failure_record["trainable"] is False
    assert failure_record["repairSample"]["trainable"] is False
    assert "Capture failed prompts" not in "\n".join(normalized.get("trainingSignals") or [])


def test_ingestion_quarantines_web_no_direct_answer_finalizer_failure(tmp_path: Path):
    report_path = tmp_path / "web-fallback-e2e-report.json"
    report = {
        "kind": "lumen_e2e_test_report",
        "passed": 0,
        "failed": 1,
        "results": [
            {
                "scenarioID": "training-web-research",
                "kind": "training",
                "title": "Training eval: web research synthesis",
                "passed": False,
                "requiresAgentRun": True,
                "prompt": "Search the web for two recent Swift concurrency best practices and summarize them.",
                "actualIntent": "webSearch",
                "expectedIntent": "webSearch",
                "failures": ["Live agent returned fallback/error text instead of completing the scenario"],
                "finalText": "No direct answer from web search. Try a different phrasing, or provide a URL to fetch directly.",
                "events": [{"phase": "step", "message": "web.search returned Swift concurrency results"}],
                "metadata": {},
            }
        ],
    }
    report_path.write_text(json.dumps(report), encoding="utf-8")

    normalized = load_runtime_audit_reports([report_path])[0]

    failure_record = normalized["failures"][0]
    assert failure_record["type"] == "e2e_architecture_finalizer_failure"
    assert failure_record["trainable"] is False
    assert failure_record["repairSample"]["trainable"] is False
    assert "Capture failed prompts" not in "\n".join(normalized.get("trainingSignals") or [])


def test_ingestion_quarantines_stale_outlook_archive_move_alias(tmp_path: Path):
    report_path = tmp_path / "outlook-alias-e2e-report.json"
    report = {
        "kind": "lumen_e2e_test_report",
        "passed": 0,
        "failed": 1,
        "results": [
            {
                "scenarioID": "live-outlook-message-move-direct",
                "kind": "toolGuard",
                "title": "Live outlook.message.move direct",
                "passed": False,
                "requiresAgentRun": True,
                "prompt": "Move my latest Outlook email to Archive.",
                "actualIntent": "outlook",
                "expectedIntent": "outlook",
                "failures": ["deterministic compatibility trace is not live model evidence"],
                "finalText": "Approval required for outlook.message.move. I did not modify Outlook mail yet.",
                "events": [],
                "metadata": {"expectedToolID": "outlook.message.move"},
            }
        ],
    }
    report_path.write_text(json.dumps(report), encoding="utf-8")

    normalized = load_runtime_audit_reports([report_path])[0]

    failure_record = normalized["failures"][0]
    assert failure_record["type"] == "e2e_test_alias_mismatch"
    assert failure_record["rootCauseCategory"] == "stale_test_alias"
    assert failure_record["trainable"] is False
    assert failure_record["repairSample"]["trainable"] is False
    assert "outlook.message.archive" in failure_record["expected"][0]
    assert "Capture failed prompts" not in "\n".join(normalized.get("trainingSignals") or [])


def test_ingestion_keeps_outlook_config_unavailable_out_of_training_repairs(tmp_path: Path):
    report_path = tmp_path / "outlook-config-unavailable-e2e-report.json"
    report = {
        "kind": "lumen_e2e_test_report",
        "passed": 0,
        "failed": 1,
        "results": [
            {
                "scenarioID": "live-outlook-message-read-direct",
                "kind": "toolGuard",
                "title": "Live outlook.message.read direct",
                "passed": False,
                "requiresAgentRun": True,
                "prompt": "Read my latest Outlook email.",
                "actualIntent": "outlook",
                "expectedIntent": "outlook",
                "failures": ["Runtime infrastructure unavailable: Outlook configuration unavailable."],
                "finalText": "Outlook auth is not configured for this device.",
                "events": [{"phase": "step", "message": "observation: Outlook auth is not configured"}],
                "metadata": {
                    "failureKind": "outlookRuntimeUnavailable",
                    "actionable": "false",
                    "trainingSignal": "false",
                    "runtimeEvidence": "tool-configuration-unavailable",
                },
            }
        ],
    }
    report_path.write_text(json.dumps(report), encoding="utf-8")

    normalized = load_runtime_audit_reports([report_path])[0]

    failure_record = normalized["failures"][0]
    assert failure_record["type"] == "e2e_runtime_environment_deferred"
    assert failure_record["trainable"] is False
    assert failure_record["repairSample"]["trainable"] is False
    assert failure_record["e2eScenario"]["metadata"]["failureKind"] == "outlookRuntimeUnavailable"
    assert "Capture failed prompts" not in "\n".join(normalized.get("trainingSignals") or [])


def test_ingestion_quarantines_rag_polluted_fallback_final(tmp_path: Path):
    report_path = tmp_path / "rag-polluted-e2e-report.json"
    report = {
        "kind": "lumen_e2e_test_report",
        "passed": 0,
        "failed": 1,
        "results": [
            {
                "scenarioID": "training-rag-grounding",
                "kind": "training",
                "title": "Training eval: local knowledge grounding",
                "passed": False,
                "requiresAgentRun": True,
                "prompt": "Search my files for architecture notes and summarize key modules.",
                "actualIntent": "rag",
                "expectedIntent": "rag",
                "failures": ["Live agent returned fallback/error text instead of completing the scenario"],
                "finalText": "I'm ready. Please ask again or tell me what you'd like to do next. Key modules: core module details were retrieved from local file snippets [1].",
                "events": [{"phase": "step", "message": "rag.search returned no matching files"}],
                "metadata": {},
            }
        ],
    }
    report_path.write_text(json.dumps(report), encoding="utf-8")

    normalized = load_runtime_audit_reports([report_path])[0]

    failure_record = normalized["failures"][0]
    assert failure_record["type"] == "e2e_architecture_finalizer_failure"
    assert failure_record["trainable"] is False
    assert failure_record["repairSample"]["trainable"] is False
    assert "Capture failed prompts" not in "\n".join(normalized.get("trainingSignals") or [])


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


def test_ingestion_rejects_sidecar_correlation_with_conflicting_ids(tmp_path: Path):
    report_path = tmp_path / "e2e-conflicting-sidecar.json"
    import json

    e2e_run_id = "11111111-1111-4111-8111-111111111111"
    expected_agent_run_id = "22222222-2222-4222-8222-222222222222"
    expected_turn_id = "33333333-3333-4333-8333-333333333333"
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
                "agentRunID": expected_agent_run_id,
                "turnID": expected_turn_id,
                "failures": [],
                "finalText": "Precision is exactness; recall is coverage.",
                "events": [{"phase": "model-evidence", "message": "missing fresh AgentBehaviorTrace modelTurn"}],
            }
        ],
    }
    report_path.write_text(json.dumps(report), encoding="utf-8")
    (tmp_path / "agent-behavior-traces.jsonl").write_text("\n".join([
        json.dumps({
            "event": "modelTurn",
            "stage": "agent-json-step-0",
            "runtimePath": "agent-model",
            "parseError": None,
            "rawOutputPrefix": "{\"final\":\"Precision is exactness; recall is coverage.\"}",
            "promptPrefix": "Grounded wrapper prompt without the original wording.",
            "scenarioID": "training-general-chat",
            "e2eRunID": e2e_run_id,
            "agentRunID": "44444444-4444-4444-8444-444444444444",
            "turnID": expected_turn_id,
        }),
        json.dumps({
            "event": "modelTurn",
            "stage": "agent-json-step-0",
            "runtimePath": "agent-model",
            "parseError": None,
            "rawOutputPrefix": "{\"final\":\"Precision is exactness; recall is coverage.\"}",
            "promptPrefix": "Another grounded wrapper prompt.",
            "scenarioID": "training-general-chat",
            "e2eRunID": e2e_run_id,
            "agentRunID": expected_agent_run_id,
            "turnID": "55555555-5555-4555-8555-555555555555",
        }),
    ]) + "\n", encoding="utf-8")

    normalized = load_runtime_audit_reports([report_path])[0]

    assert normalized["failures"][0]["rootCauseCategory"] == "no_correlated_model_turn"
    assert normalized["failures"][0]["e2eScenario"]["skippedLiveModelRun"] is False
    assert normalized["scenarios"][0]["modelEvidenceStatus"] == "no_correlated_model_turn"


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


def test_live_deterministic_model_evidence_is_not_model_backed(tmp_path: Path):
    report_path = tmp_path / "e2e-live-deterministic.json"
    import json

    report = {
        "kind": "lumen_e2e_test_report",
        "passed": 1,
        "failed": 0,
        "scenarios": [
            {
                "id": "weather-here-no-calendar",
                "name": "Weather here must not create events",
                "kind": "regression",
                "passed": True,
                "requiresAgentRun": True,
                "prompt": "What is the weather here?",
                "intent": "weather",
                "expectedIntent": "weather",
                "failures": [],
                "final": "Weather summary.",
                "events": [{"phase": "model-evidence", "message": "runtime=deterministic-compatibility, kind=policy-first-deterministic, stage=compatibility-final"}],
            }
        ],
    }
    report_path.write_text(json.dumps(report), encoding="utf-8")

    normalized = load_runtime_audit_reports([report_path])[0]

    failure = normalized["failures"][0]
    assert failure["rootCauseCategory"] == "deterministic_compatibility_not_live_evidence"
    assert failure["e2eScenario"]["skippedLiveModelRun"] is False
    assert "deterministic compatibility" in failure["problem"]


def test_policy_first_evidence_mode_accepts_deterministic_trace(tmp_path: Path):
    report_path = tmp_path / "e2e-policy-first.json"
    import json

    report = {
        "kind": "lumen_e2e_test_report",
        "passed": 1,
        "failed": 0,
        "scenarios": [
            {
                "id": "live-alarm-countdown-direct",
                "name": "Live alarm countdown direct",
                "kind": "toolGuard",
                "passed": True,
                "requiresAgentRun": True,
                "evidenceMode": "policyFirstAllowed",
                "prompt": "Start a timer for 10 minutes.",
                "intent": "alarm",
                "expectedIntent": "alarm",
                "failures": [],
                "final": "Approval required.",
                "events": [{"phase": "model-evidence", "message": "runtime=deterministic-compatibility, kind=policy-first-deterministic, stage=compatibility-approval-boundary"}],
            }
        ],
    }
    report_path.write_text(json.dumps(report), encoding="utf-8")

    normalized = load_runtime_audit_reports([report_path])[0]

    assert normalized["failures"] == []
    assert normalized["scenarios"][0]["modelEvidenceStatus"] == "valid_policy_first_evidence"


def test_passed_final_with_validation_json_is_false_success(tmp_path: Path):
    report_path = tmp_path / "e2e-final-artifact.json"
    import json

    report = {
        "kind": "lumen_e2e_test_report",
        "passed": 1,
        "failed": 0,
        "scenarios": [
            {
                "id": "training-memory-loop",
                "name": "Training eval: memory save/recall",
                "kind": "training",
                "passed": True,
                "requiresAgentRun": True,
                "prompt": "Remember that I prefer concise bullet points, then tell me what you remembered.",
                "intent": "memory",
                "expectedIntent": "memory",
                "failures": [],
                "final": "{\"reasoningSummary\":\"Memory tool output could not be validated.\",\"rewrittenFinalAnswer\":\"Memory tool output could not be validated.\",\"requiresApprovalDecision\":\"deny\"}\n\nI remember that you prefer concise bullet points.",
                "events": [{"phase": "model-evidence", "message": "runtime=agent-model, kind=model-backed, stage=agent-json-step-0, parseError=none"}],
            }
        ],
    }
    report_path.write_text(json.dumps(report), encoding="utf-8")

    normalized = load_runtime_audit_reports([report_path])[0]

    failure = normalized["failures"][0]
    assert failure["rootCauseCategory"] == "live_final_validation_artifact"
    assert normalized["scenarios"][0]["outputQualityStatus"] == "live_final_validation_artifact"
    assert "validation fallback" in failure["problem"]


def test_adapter_missing_empty_agent_json_is_runtime_environment_deferred(tmp_path: Path):
    report_path = tmp_path / "latest-e2e-report.json"
    import json

    prompt = "What is the weather here?"
    report = {
        "kind": "lumen_e2e_test_report",
        "passed": 0,
        "failed": 1,
        "scenarios": [
            {
                "id": "weather-here-no-calendar",
                "name": "Weather here must not create events",
                "kind": "regression",
                "passed": False,
                "requiresAgentRun": True,
                "prompt": prompt,
                "intent": "weather",
                "expectedIntent": "weather",
                "failures": ["Live E2E scenario did not record model-backed generation evidence"],
                "final": "Weather tool output could not be validated.",
                "events": [{"phase": "model-evidence", "message": "found primary agent-json modelTurn but runtime readiness failure (executor preflight failed: adapter required but adapter path missing)"}],
            }
        ],
    }
    report_path.write_text(json.dumps(report), encoding="utf-8")
    (tmp_path / "agent-behavior-traces.jsonl").write_text(json.dumps({
        "event": "modelTurn",
        "stage": "agent-json-step-0",
        "runtimePath": "agent-model",
        "parseError": "empty",
        "rawOutputPrefix": "",
        "emptyOutputReason": "executor preflight failed: adapter required but adapter path missing",
        "streamTerminationReason": "executor preflight failed: adapter required but adapter path missing",
        "promptPrefix": prompt,
    }) + "\n", encoding="utf-8")

    normalized = load_runtime_audit_reports([tmp_path])[0]

    failure = normalized["failures"][0]
    assert failure["rootCauseCategory"] == "runtime_environment_deferred"
    assert failure["trainable"] is False


def test_evidence_layer_envelope_preserves_sidecar_correlation(tmp_path: Path):
    report_path = tmp_path / "1-lumen-live-e2e-report.json"
    import json

    envelope = {
        "schemaVersion": "1.0.0",
        "generatedAt": "2026-06-29T03:06:12Z",
        "exportPolicy": {
            "sourceLayer": "e2eTestReport",
            "format": "lumen-evidence-layer-json",
            "ownsLiveE2EScenarios": True,
        },
        "payload": {
            "passed": 1,
            "failed": 0,
            "results": [
                {
                    "scenarioID": "training-general-chat",
                    "title": "Training eval: pure chat quality",
                    "kind": "training",
                    "passed": True,
                    "e2eRunID": "11111111-1111-1111-1111-111111111111",
                    "agentRunID": "22222222-2222-2222-2222-222222222222",
                    "conversationID": "33333333-3333-3333-3333-333333333333",
                    "turnID": "44444444-4444-4444-4444-444444444444",
                    "requiresAgentRun": True,
                    "prompt": "Explain precision and recall.",
                    "expectedIntent": "chat",
                    "actualIntent": "chat",
                    "failures": [],
                    "finalText": "Precision is exactness; recall is coverage.",
                    "events": [{"phase": "model-evidence", "message": "missing fresh AgentBehaviorTrace modelTurn"}],
                    "startedAt": "2026-06-29T03:06:00Z",
                    "finishedAt": "2026-06-29T03:06:20Z",
                }
            ],
        },
    }
    report_path.write_text(json.dumps(envelope), encoding="utf-8")
    (tmp_path / "6-agent-behavior-traces.jsonl").write_text(
        json.dumps({
            "createdAt": "2026-06-29T03:06:10Z",
            "event": "modelTurn",
            "stage": "agent-json-step-0",
            "runtimePath": "agent-model",
            "scenarioID": "training-general-chat",
            "e2eRunID": "11111111-1111-1111-1111-111111111111",
            "agentRunID": "22222222-2222-2222-2222-222222222222",
            "conversationID": "33333333-3333-3333-3333-333333333333",
            "turnID": "44444444-4444-4444-4444-444444444444",
            "promptPrefix": "redacted structured prompt",
            "rawOutputPrefix": "{\"final\":\"Precision is exactness; recall is coverage.\"}",
            "parseError": None,
        }) + "\n",
        encoding="utf-8",
    )

    normalized = load_runtime_audit_reports([report_path])[0]

    assert normalized["failures"] == []
    assert normalized["scenarios"][0]["modelEvidenceStatus"] == "valid_model_backed_evidence"
    assert normalized["scenarios"][0]["scenarioID"] == "training-general-chat"


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
        "emptyOutputReason": "completedWithoutText",
        "streamStarted": True,
        "firstChunkReceived": False,
        "textChunkCount": 0,
        "finalChunkReceived": True,
        "streamTerminationReason": "stop",
        "rawOutputPrefix": "",
        "promptPrefix": "Explain tradeoffs between precision and recall in retrieval systems in plain English.",
    }) + "\n", encoding="utf-8")
    (tmp_path / "agent-parse-failures.jsonl").write_text(json.dumps({
        "modelName": "agent-json",
        "parseError": "empty",
        "emptyOutputReason": "completedWithoutText",
        "streamStarted": True,
        "firstChunkReceived": False,
        "textChunkCount": 0,
        "finalChunkReceived": True,
        "streamTerminationReason": "stop",
        "rawOutputPrefix": "",
        "userTurnPrefix": "User request:\nExplain tradeoffs between precision and recall in retrieval systems in plain English.",
    }) + "\n", encoding="utf-8")

    normalized = load_runtime_audit_reports([report_path])[0]

    failure = normalized["failures"][0]
    assert failure["rootCauseCategory"] == "agent_json_completed_without_text"
    assert failure["e2eScenario"]["skippedLiveModelRun"] is False
    assert failure["e2eScenario"]["modelEvidenceTrace"]["stage"] == "agent-json-step-0"
    assert "agent-json emitted empty output" in failure["problem"]


def test_ingestion_classifies_agent_json_context_overflow_separately(tmp_path: Path):
    report_path = tmp_path / "latest-e2e-report.json"
    prompt = "What is the weather here and should I carry an umbrella?"
    report_path.write_text(json.dumps({
        "kind": "lumen_e2e_test_report",
        "passed": 0,
        "failed": 1,
        "scenarios": [
            {
                "name": "Training eval: weather stays grounded",
                "passed": False,
                "requiresAgentRun": True,
                "prompt": prompt,
                "intent": "weather",
                "expectedIntent": "weather",
                "failures": ["Live E2E scenario did not record model-backed generation evidence"],
                "final": "Weather tool output could not be validated.",
                "events": [{"phase": "model-evidence", "message": "found AgentBehaviorTrace modelTurn but parseError=noJSONObject; stage=agent-repair"}],
            }
        ],
    }), encoding="utf-8")
    (tmp_path / "agent-behavior-traces.jsonl").write_text("\n".join([
        json.dumps({
            "event": "modelTurn",
            "stage": "agent-json-step-0",
            "runtimePath": "agent-model",
            "parseError": "noJSONObject",
            "rawOutputPrefix": "Generation error: Failed to initialize context: Prompt exceeds shared chat context window",
            "promptPrefix": prompt,
        }),
        json.dumps({
            "event": "modelTurn",
            "stage": "agent-repair",
            "runtimePath": "sharedAdapter",
            "parseError": "missingActionOrFinal",
            "rawOutputPrefix": "{}",
            "promptPrefix": prompt,
        }),
    ]) + "\n", encoding="utf-8")

    normalized = load_runtime_audit_reports([report_path])[0]

    failure = normalized["failures"][0]
    assert failure["rootCauseCategory"] == "agent_json_context_overflow"
    assert failure["e2eScenario"]["skippedLiveModelRun"] is False
    assert failure["e2eScenario"]["modelEvidenceTrace"]["stage"] == "agent-json-step-0"
    assert failure["e2eScenario"]["modelEvidenceTrace"]["parseError"] == "contextWindowExceeded"
    assert "prompt exceeded context window" in failure["problem"]


def test_ingestion_prefers_resource_budget_denied_over_cancelled_empty_reason(tmp_path: Path):
    report_path = tmp_path / "latest-e2e-report.json"
    prompt = "Search the web for two recent Swift concurrency best practices and summarize them."
    report_path.write_text(json.dumps({
        "kind": "lumen_e2e_test_report",
        "passed": 0,
        "failed": 1,
        "scenarios": [
            {
                "name": "Training eval: web research synthesis",
                "passed": False,
                "requiresAgentRun": True,
                "prompt": prompt,
                "intent": "webSearch",
                "expectedIntent": "webSearch",
                "failures": ["Live E2E scenario did not record model-backed generation evidence"],
                "final": "Search failed.",
                "events": [{"phase": "model-evidence", "message": "found primary agent-json modelTurn but agent-json emitted empty output"}],
            }
        ],
    }), encoding="utf-8")
    (tmp_path / "agent-behavior-traces.jsonl").write_text(json.dumps({
        "event": "modelTurn",
        "stage": "agent-json-step-0",
        "runtimePath": "agent-model",
        "parseError": "empty",
        "emptyOutputReason": "cancelledBeforeFirstToken",
        "streamStarted": False,
        "firstChunkReceived": False,
        "textChunkCount": 0,
        "finalChunkReceived": False,
        "streamTerminationReason": "resource-budget-denied-before-prompt-eval",
        "rawOutputPrefix": "",
        "promptPrefix": prompt,
    }) + "\n", encoding="utf-8")

    normalized = load_runtime_audit_reports([report_path])[0]

    failure = normalized["failures"][0]
    assert failure["rootCauseCategory"] == "runtime_environment_deferred"
    assert failure["type"] == "e2e_runtime_environment_deferred"
    assert failure["trainable"] is False
    assert failure["repairSample"]["trainable"] is False
    assert "resource-budget-denied-before-prompt-eval" in failure["problem"]


def test_ingestion_keeps_adapter_unavailable_out_of_training_repairs(tmp_path: Path):
    report_path = tmp_path / "latest-e2e-report.json"
    prompt = "What is the weather here and should I carry an umbrella?"
    report_path.write_text(json.dumps({
        "kind": "lumen_e2e_test_report",
        "passed": 0,
        "failed": 1,
        "scenarios": [
            {
                "name": "Training eval: weather stays grounded",
                "passed": False,
                "requiresAgentRun": True,
                "prompt": prompt,
                "intent": "weather",
                "expectedIntent": "weather",
                "failures": ["Live E2E scenario did not record model-backed generation evidence"],
                "final": "Weather tool output could not be validated. Try again or provide a city.",
                "events": [{"phase": "model-evidence", "message": "found primary agent-json modelTurn but parseError=noJSONObject; streamTerminationReason=adapterUnavailable"}],
            }
        ],
    }), encoding="utf-8")
    (tmp_path / "agent-behavior-traces.jsonl").write_text(json.dumps({
        "event": "modelTurn",
        "stage": "agent-json-step-0",
        "runtimePath": "agent-model",
        "parseError": "noJSONObject",
        "rawOutputPrefix": "Generation error: The operation couldn’t be completed. (Lumen.LocalRuntimeError error 0.)",
        "promptPrefix": prompt,
        "streamStarted": False,
        "firstChunkReceived": False,
        "textChunkCount": 0,
        "finalChunkReceived": False,
        "streamTerminationReason": "adapterUnavailable",
    }) + "\n", encoding="utf-8")

    normalized = load_runtime_audit_reports([report_path])[0]

    failure = normalized["failures"][0]
    assert failure["rootCauseCategory"] == "runtime_environment_deferred"
    assert failure["type"] == "e2e_runtime_environment_deferred"
    assert failure["trainable"] is False
    assert failure["repairSample"]["trainable"] is False
    assert "adapterUnavailable" in failure["problem"]


def test_ingestion_prefers_primary_agent_json_trace_over_runtime_init_trace(tmp_path: Path):
    report_path = tmp_path / "latest-e2e-report.json"
    prompt = "What is the weather here and should I carry an umbrella?"
    report_path.write_text(json.dumps({
        "kind": "lumen_e2e_test_report",
        "passed": 0,
        "failed": 1,
        "scenarios": [
            {
                "name": "Training eval: weather stays grounded",
                "passed": False,
                "requiresAgentRun": True,
                "prompt": prompt,
                "intent": "weather",
                "expectedIntent": "weather",
                "failures": ["Live E2E scenario did not record model-backed generation evidence"],
                "final": "Weather tool output could not be validated.",
                "events": [{"phase": "model-evidence", "message": "found AgentBehaviorTrace modelTurn but parseError=noJSONObject; stage=agent-repair"}],
            }
        ],
    }), encoding="utf-8")
    (tmp_path / "agent-behavior-traces.jsonl").write_text("\n".join([
        json.dumps({
            "event": "modelTurn",
            "stage": "agent-json",
            "runtimePath": "model_initialization_failed_prompt_too_large",
            "parseError": "contextWindowExceeded",
            "rawOutputPrefix": "Prompt exceeded context window before generation",
            "promptPrefix": prompt,
        }),
        json.dumps({
            "event": "modelTurn",
            "stage": "agent-json-step-0",
            "runtimePath": "agent-model",
            "parseError": "contextWindowExceeded",
            "rawOutputPrefix": "Prompt exceeded context window before generation",
            "promptPrefix": prompt,
        }),
    ]) + "\n", encoding="utf-8")

    normalized = load_runtime_audit_reports([report_path])[0]

    trace = normalized["failures"][0]["e2eScenario"]["modelEvidenceTrace"]
    assert trace["stage"] == "agent-json-step-0"
    assert trace["runtimePath"] == "agent-model"


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
        "emptyOutputReason": "completedWithoutText",
        "streamStarted": True,
        "firstChunkReceived": False,
        "textChunkCount": 0,
        "finalChunkReceived": True,
        "streamTerminationReason": "stop",
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
        "emptyOutputReason": "completedWithoutText",
        "streamStarted": True,
        "firstChunkReceived": False,
        "textChunkCount": 0,
        "finalChunkReceived": True,
        "streamTerminationReason": "stop",
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
        "emptyOutputReason": "completedWithoutText",
        "streamStarted": True,
        "firstChunkReceived": False,
        "textChunkCount": 0,
        "finalChunkReceived": True,
        "streamTerminationReason": "stop",
        "rawOutputPrefix": "",
        "promptPrefix": "Explain tradeoffs between precision and recall in retrieval systems in plain English.",
    }) + "\n", encoding="utf-8")

    normalized = load_runtime_audit_reports([report_path])[0]

    failure = normalized["failures"][0]
    assert failure["rootCauseCategory"] == "agent_json_completed_without_text"
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
            "emptyOutputReason": "completedWithoutText",
            "streamStarted": True,
            "firstChunkReceived": False,
            "textChunkCount": 0,
            "finalChunkReceived": True,
            "streamTerminationReason": "stop",
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
    assert failures_by_prompt["Empty output prompt."]["rootCauseCategory"] == "agent_json_completed_without_text"
    assert failures_by_prompt["Parse error prompt."]["rootCauseCategory"] == "agent_json_parse_error"
    assert "Valid model prompt." not in failures_by_prompt
    assert failures_by_prompt["Policy first prompt."]["rootCauseCategory"] == "deterministic_compatibility_not_live_evidence"


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


def test_in_app_package_uses_export_quality_failures_without_duplicate_empty_trace_failure(tmp_path: Path):
    report_path = tmp_path / "lumen-agent-grounding-audit-export-quality.json"
    import json

    package = {
        "schemaVersion": "1.3.0",
        "generatedAt": "2026-06-24T00:00:00Z",
        "manifestSource": "AgentGrounding/agent_manifest/AgentBehaviorManifest.json",
        "usedRuntimeFallback": False,
        "exportPolicy": {
            "format": "agent-grounding-runtime-json-package",
            "sourceLayer": "agentGroundingRuntimeAudit",
            "ownsLiveE2EScenarios": False,
        },
        "recentTraces": [],
        "exportQualityFailures": [
            {
                "type": "agent_grounding_no_recent_model_traces",
                "agent": "runtime",
                "expected": ["recent traces"],
                "actual": "recentTraces is empty",
                "scenario": "Agent Grounding > Run Agent Grounding Audit > Export In-App Dataset Package",
                "problem": "The Agent Grounding package exported no recent traces.",
            }
        ],
    }
    report_path.write_text(json.dumps(package), encoding="utf-8")

    report = load_runtime_audit_reports([report_path])[0]

    failures = [
        failure
        for failure in report["failures"]
        if failure["type"] == "agent_grounding_no_recent_model_traces"
    ]
    assert len(failures) == 1
    assert failures[0]["sourceLayer"] == "agentGroundingRuntimeAudit.exportQuality"


def test_in_app_package_preserves_incomplete_model_trace_export_quality_failure(tmp_path: Path):
    report_path = tmp_path / "lumen-agent-grounding-audit-incomplete-trace.json"
    import json

    package = {
        "schemaVersion": "1.3.0",
        "generatedAt": "2026-06-24T00:00:00Z",
        "manifestSource": "AgentGrounding/agent_manifest/AgentBehaviorManifest.json",
        "usedRuntimeFallback": False,
        "exportPolicy": {
            "format": "agent-grounding-runtime-json-package",
            "sourceLayer": "agentGroundingRuntimeAudit",
            "ownsLiveE2EScenarios": False,
        },
        "recentTraces": [
            {
                "event": "modelTurn",
                "slot": "executor",
                "stage": "agent-json-step-0",
                "runtimePath": "agent-model",
                "promptPrefix": "What is the weather here?",
                "allowedToolIDs": ["weather"],
            }
        ],
        "exportQualityFailures": [
            {
                "type": "agent_grounding_model_trace_incomplete",
                "agent": "executor",
                "expected": ["complete runtime evidence"],
                "actual": "missing=selectedRuntime,modelLoaded,outputTokenCount,streamStarted",
                "scenario": "What is the weather here?",
                "problem": "A structured model-turn trace does not carry the minimum runtime evidence.",
            }
        ],
    }
    report_path.write_text(json.dumps(package), encoding="utf-8")

    report = load_runtime_audit_reports([report_path])[0]

    failure = next(
        failure
        for failure in report["failures"]
        if failure["type"] == "agent_grounding_model_trace_incomplete"
    )
    assert failure["sourceLayer"] == "agentGroundingRuntimeAudit.exportQuality"
    assert "selectedRuntime" in failure["actual"]


def test_in_app_package_v14_preserves_final_validator_replacement_quality_failure(tmp_path: Path):
    report_path = tmp_path / "lumen-agent-grounding-audit-final-validator.json"
    import json

    package = {
        "schemaVersion": "1.4.0",
        "generatedAt": "2026-06-24T00:00:00Z",
        "manifestSource": "AgentGrounding/agent_manifest/AgentBehaviorManifest.json",
        "usedRuntimeFallback": False,
        "exportPolicy": {
            "format": "agent-grounding-runtime-json-package",
            "sourceLayer": "agentGroundingRuntimeAudit",
            "ownsLiveE2EScenarios": False,
        },
        "recentTraces": [
            {
                "event": "finalAnswer",
                "slot": "mouth",
                "stage": "compatibility-final",
                "runtimePath": "deterministic-compatibility",
                "promptPrefix": "Search my calendar for tomorrow",
                "rawOutputPrefix": "I could not safely complete the calendar request.",
                "selectedToolID": "calendar.list",
                "allowedToolIDs": ["calendar.list"],
                "finalizerAccepted": False,
                "finalizerRejectionReason": "intent-mismatch",
                "finalValidatorAcceptedCandidate": False,
                "finalValidatorReplacementSource": "safeMessage",
                "finalValidatorRejectionReason": "tool-json-leak",
            }
        ],
        "exportQualityFailures": [
            {
                "type": "agent_grounding_final_validator_replaced_candidate",
                "agent": "mouth",
                "expected": ["final validator accepted candidate"],
                "actual": "replacementSource=safeMessage; rejectionReason=tool-json-leak",
                "scenario": "Search my calendar for tomorrow",
                "problem": "The final validator replaced the candidate response.",
            }
        ],
    }
    report_path.write_text(json.dumps(package), encoding="utf-8")

    report = load_runtime_audit_reports([report_path])[0]

    failure = next(
        failure
        for failure in report["failures"]
        if failure["type"] == "agent_grounding_final_validator_replaced_candidate"
    )
    assert report["_sourceFormat"] == "lumen_in_app_dataset_package"
    assert failure["sourceLayer"] == "agentGroundingRuntimeAudit.exportQuality"
    assert "tool-json-leak" in failure["actual"]


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


def test_agent_grounding_package_embeds_live_e2e_report_with_trace_sidecars(tmp_path: Path):
    report_path = tmp_path / "lumen-agent-grounding-testflight-package.json"
    import json

    e2e_run_id = "11111111-1111-1111-1111-111111111111"
    agent_run_id = "22222222-2222-2222-2222-222222222222"
    conversation_id = "33333333-3333-3333-3333-333333333333"
    turn_id = "44444444-4444-4444-4444-444444444444"
    package = {
        "schemaVersion": "1.7.0",
        "generatedAt": "2026-06-29T00:00:00Z",
        "app": {
            "name": "Lumen",
            "bundleIdentifier": "com.27pm.lumenclone",
            "shortVersion": "1.0.0",
            "buildNumber": "20260629054657",
        },
        "manifestSource": "AgentGrounding/agent_manifest/AgentBehaviorManifest.json",
        "usedRuntimeFallback": False,
        "exportPolicy": {
            "format": "agent-grounding-runtime-json-package",
            "sourceLayer": "agentGroundingRuntimeAudit",
            "ownsLiveE2EScenarios": False,
        },
        "recentTraces": [
            {
                "id": "55555555-5555-5555-5555-555555555555",
                "createdAt": "2026-06-29T00:00:05Z",
                "event": "modelTurn",
                "slot": "executor",
                "stage": "agent-json-step-0",
                "scenarioID": "live-self-model",
                "e2eRunID": e2e_run_id,
                "agentRunID": agent_run_id,
                "conversationID": conversation_id,
                "turnID": turn_id,
                "promptPrefix": "What evidence supports your claim?",
                "rawOutputPrefix": '{"final":"The answer is supported by live E2E evidence."}',
                "runtimePath": "agent-model",
                "selectedRuntime": "llama",
                "modelLoaded": True,
                "outputTokenCount": 12,
                "streamStarted": True,
                "firstChunkReceived": True,
                "textChunkCount": 1,
                "finalChunkReceived": True,
                "allowedToolIDs": [],
                "toolArguments": {},
                "emittedFinalInActionTurn": False,
            }
        ],
        "liveE2EReport": {
            "schemaVersion": "1.0.0",
            "generatedAt": "2026-06-29T00:00:10Z",
            "app": {
                "name": "Lumen",
                "bundleIdentifier": "com.27pm.lumenclone",
                "shortVersion": "1.0.0",
                "buildNumber": "20260629054657",
            },
            "exportPolicy": {
                "format": "live-e2e-test-report-json",
                "sourceLayer": "e2eTestReport",
                "ownsLiveE2EScenarios": True,
                "includesDeterministicStaticScenarios": False,
            },
            "payload": {
                "id": "66666666-6666-6666-6666-666666666666",
                "startedAt": "2026-06-29T00:00:00Z",
                "finishedAt": "2026-06-29T00:00:20Z",
                "passed": 1,
                "failed": 0,
                "results": [
                    {
                        "id": "77777777-7777-7777-7777-777777777777",
                        "scenarioID": "live-self-model",
                        "kind": "standard",
                        "title": "Live self-model evidence",
                        "prompt": "What evidence supports your claim?",
                        "expectedIntent": "rag",
                        "actualIntent": "rag",
                        "e2eRunID": e2e_run_id,
                        "agentRunID": agent_run_id,
                        "conversationID": conversation_id,
                        "turnID": turn_id,
                        "requiresAgentRun": True,
                        "passed": True,
                        "failures": [],
                        "finalText": "The answer is supported by live E2E evidence.",
                        "missingHints": [],
                        "rewriteAttempted": False,
                        "rewriteSuccess": False,
                        "events": [],
                        "startedAt": "2026-06-29T00:00:00Z",
                        "finishedAt": "2026-06-29T00:00:20Z",
                        "rawFinalPrefix": "The answer is supported by live E2E evidence.",
                        "sanitizedFinalPrefix": "The answer is supported by live E2E evidence.",
                        "rawFinalHadUnsafeLeakage": False,
                        "sanitizedFinalRemovedArtifacts": [],
                        "outputHygieneFailures": [],
                        "metadata": {},
                    }
                ],
            },
            "correlatedTraceCount": 2,
            "modelBackedCorrelatedTraceCount": 1,
            "modelBackedCorrelatedScenarioCount": 1,
            "deterministicCompatibilityTraceCount": 1,
            "traceSidecarField": "recentTraces",
        },
    }
    report_path.write_text(json.dumps(package), encoding="utf-8")

    reports = load_runtime_audit_reports([report_path])
    package_report = next(report for report in reports if report["_sourceFormat"] == "lumen_in_app_dataset_package")
    live_report = next(report for report in reports if report["_sourceFormat"] == "live-e2e-test-report-json")

    assert package_report["_sourceLayer"] == "agentGroundingRuntimeAudit"
    assert package_report["appBuildNumber"] == "20260629054657"
    assert package_report["ownsLiveE2EScenarios"] is False
    assert package_report["liveE2ECorrelatedTraceCount"] == 2
    assert package_report["liveE2EModelBackedCorrelatedTraceCount"] == 1
    assert package_report["liveE2EModelBackedCorrelatedScenarioCount"] == 1
    assert package_report["liveE2EDeterministicCompatibilityTraceCount"] == 1
    assert live_report["_sourceLayer"] == "e2eTestReport.evidenceLayer"
    assert live_report["appBuildNumber"] == "20260629054657"
    assert live_report["failures"] == []
    assert live_report["scenarios"][0]["modelEvidenceStatus"] == "valid_model_backed_evidence"
    assert live_report["scenarios"][0]["modelEvidenceTrace"]["matchedBy"] == "correlation"


def test_testflight_agent_grounding_package_preserves_export_metadata(tmp_path: Path):
    import json

    report_path = tmp_path / "lumen-testflight-agent-grounding-current.json"
    package = {
        "schemaVersion": "1.9.0",
        "generatedAt": "2026-06-29T00:00:00Z",
        "exportKind": "testflight-agent-grounding-runtime-export",
        "app": {
            "name": "Lumen",
            "bundleIdentifier": "com.27pm.lumenclone",
            "shortVersion": "1.0.0",
            "buildNumber": "20260629060414",
        },
        "testFlight": {
            "sourceAction": "Agent Grounding > Export TestFlight + Agent Grounding Package",
            "filePrefix": "lumen-testflight-agent-grounding",
            "distributionChannel": "testflight_or_development_sandbox",
            "sandboxReceipt": True,
            "appShortVersion": "1.0.0",
            "appBuildNumber": "20260629060414",
            "liveE2EReportIncluded": False,
            "expectedIngestArgument": "--runtime-audit <exported-testflight-json>",
        },
        "manifestSource": "AgentGrounding/agent_manifest/AgentBehaviorManifest.json",
        "usedRuntimeFallback": False,
        "exportPolicy": {
            "format": "testflight-agent-grounding-runtime-json-package",
            "sourceLayer": "agentGroundingRuntimeAudit",
            "ownsLiveE2EScenarios": False,
            "includesDeterministicStaticScenarios": False,
        },
        "recentTraces": [],
        "runtimeManifestAudit": None,
        "behaviorAudit": None,
        "scenarioResults": [],
        "traceSelectedToolAllowedCount": 0,
        "traceParseErrorCount": 0,
        "improveLoop": {
            "acceptedTraining": [],
            "quarantinedSamples": [],
            "regressionTests": [],
        },
    }
    report_path.write_text(json.dumps(package), encoding="utf-8")

    report = load_runtime_audit_reports([report_path])[0]

    assert report["_sourceFormat"] == "testflight_agent_grounding_package"
    assert report["_sourceLayer"] == "agentGroundingRuntimeAudit"
    assert report["exportKind"] == "testflight-agent-grounding-runtime-export"
    assert report["appBuildNumber"] == "20260629060414"
    assert report["testFlightAppBuildNumber"] == "20260629060414"
    assert report["testFlightSourceAction"] == "Agent Grounding > Export TestFlight + Agent Grounding Package"
    assert report["testFlightDistributionChannel"] == "testflight_or_development_sandbox"
    assert report["testFlightLiveE2EReportIncluded"] is False
    assert report["failures"][0]["scenario"] == "Agent Grounding > Run Agent Grounding Audit > Export TestFlight + Agent Grounding Package"


def test_agent_grounding_package_synthesizes_live_e2e_model_backed_trace_gap(tmp_path: Path):
    import json

    e2e_run_id = "11111111-1111-1111-1111-111111111111"
    agent_run_id = "22222222-2222-2222-2222-222222222222"
    conversation_id = "33333333-3333-3333-3333-333333333333"
    turn_id = "44444444-4444-4444-4444-444444444444"
    package = {
        "schemaVersion": "1.8.0",
        "generatedAt": "2026-06-29T00:00:00Z",
        "manifestSource": "AgentGrounding/agent_manifest/AgentBehaviorManifest.json",
        "usedRuntimeFallback": False,
        "exportPolicy": {
            "format": "agent-grounding-runtime-json-package",
            "sourceLayer": "agentGroundingRuntimeAudit",
            "ownsLiveE2EScenarios": False,
        },
        "recentTraces": [
            {
                "id": "55555555-5555-5555-5555-555555555555",
                "createdAt": "2026-06-29T00:00:05Z",
                "event": "finalAnswer",
                "slot": "cortex",
                "stage": "compatibility-clarification-final",
                "scenarioID": "live-alarm-authorization-status",
                "e2eRunID": e2e_run_id,
                "agentRunID": agent_run_id,
                "conversationID": conversation_id,
                "turnID": turn_id,
                "promptPrefix": "Show alarm permission status.",
                "rawOutputPrefix": "I couldn't safely complete the alarm/timer request.",
                "runtimePath": "deterministic-compatibility",
                "allowedToolIDs": [],
                "toolArguments": {},
                "emittedFinalInActionTurn": True,
            }
        ],
        "liveE2EReport": {
            "schemaVersion": "1.0.0",
            "generatedAt": "2026-06-29T00:00:10Z",
            "exportPolicy": {
                "format": "live-e2e-test-report-json",
                "sourceLayer": "e2eTestReport",
                "ownsLiveE2EScenarios": True,
                "includesDeterministicStaticScenarios": False,
            },
            "payload": {
                "id": "66666666-6666-6666-6666-666666666666",
                "startedAt": "2026-06-29T00:00:00Z",
                "finishedAt": "2026-06-29T00:00:20Z",
                "passed": 0,
                "failed": 1,
                "results": [
                    {
                        "id": "77777777-7777-7777-7777-777777777777",
                        "scenarioID": "live-alarm-authorization-status",
                        "kind": "standard",
                        "title": "Live alarm authorization status",
                        "prompt": "Show alarm permission status.",
                        "expectedIntent": "alarm",
                        "actualIntent": "alarm",
                        "e2eRunID": e2e_run_id,
                        "agentRunID": agent_run_id,
                        "conversationID": conversation_id,
                        "turnID": turn_id,
                        "requiresAgentRun": True,
                        "passed": False,
                        "failures": [
                            "found deterministic-compatibility execution trace but policy-first evidence disabled for this scenario"
                        ],
                        "finalText": "I couldn't safely complete the alarm/timer request.",
                        "events": [],
                    }
                ],
            },
            "correlatedTraceCount": 1,
            "modelBackedCorrelatedTraceCount": 0,
            "modelBackedCorrelatedScenarioCount": 0,
            "deterministicCompatibilityTraceCount": 1,
            "traceSidecarField": "recentTraces",
        },
    }
    report_path = tmp_path / "lumen-agent-grounding-testflight-gap-package.json"
    report_path.write_text(json.dumps(package), encoding="utf-8")

    package_report = next(
        report
        for report in load_runtime_audit_reports([report_path])
        if report["_sourceFormat"] == "lumen_in_app_dataset_package"
    )

    failure = next(
        failure
        for failure in package_report["failures"]
        if failure["type"] == "agent_grounding_live_e2e_model_backed_trace_gap"
    )
    assert package_report["liveE2EModelBackedCorrelatedTraceCount"] == 0
    assert package_report["liveE2EModelBackedCorrelatedScenarioCount"] == 0
    assert package_report["liveE2EDeterministicCompatibilityTraceCount"] == 1
    assert failure["sourceLayer"] == "agentGroundingRuntimeAudit.exportQuality"
    assert "requiredAgentRunScenarioCount=1" in failure["actual"]
    assert "modelBackedCorrelatedTraceCount=0" in failure["actual"]
    assert "modelBackedCorrelatedScenarioCount=0" in failure["actual"]


def test_agent_grounding_package_synthesizes_live_e2e_scenario_coverage_gap_when_trace_count_matches(tmp_path: Path):
    import json

    package = {
        "schemaVersion": "1.8.0",
        "generatedAt": "2026-06-29T00:00:00Z",
        "manifestSource": "AgentGrounding/agent_manifest/AgentBehaviorManifest.json",
        "usedRuntimeFallback": False,
        "exportPolicy": {
            "format": "agent-grounding-runtime-json-package",
            "sourceLayer": "agentGroundingRuntimeAudit",
            "ownsLiveE2EScenarios": False,
        },
        "recentTraces": [],
        "liveE2EReport": {
            "schemaVersion": "1.0.0",
            "generatedAt": "2026-06-29T00:00:10Z",
            "exportPolicy": {
                "format": "live-e2e-test-report-json",
                "sourceLayer": "e2eTestReport",
                "ownsLiveE2EScenarios": True,
                "includesDeterministicStaticScenarios": False,
            },
            "payload": {
                "id": "66666666-6666-6666-6666-666666666666",
                "startedAt": "2026-06-29T00:00:00Z",
                "finishedAt": "2026-06-29T00:00:20Z",
                "passed": 1,
                "failed": 1,
                "results": [
                    {
                        "id": "77777777-7777-7777-7777-777777777777",
                        "scenarioID": "covered-live-scenario",
                        "kind": "standard",
                        "title": "Covered live scenario",
                        "prompt": "What evidence supports your claim?",
                        "expectedIntent": "rag",
                        "actualIntent": "rag",
                        "requiresAgentRun": True,
                        "passed": True,
                        "failures": [],
                        "finalText": "Live E2E evidence.",
                        "events": [],
                    },
                    {
                        "id": "88888888-8888-8888-8888-888888888888",
                        "scenarioID": "missing-live-scenario",
                        "kind": "standard",
                        "title": "Missing live scenario",
                        "prompt": "What source proves the runtime state?",
                        "expectedIntent": "rag",
                        "actualIntent": "rag",
                        "requiresAgentRun": True,
                        "passed": False,
                        "failures": ["Live E2E scenario did not record model-backed generation evidence"],
                        "finalText": "Missing model-backed evidence.",
                        "events": [],
                    },
                ],
            },
            "correlatedTraceCount": 2,
            "modelBackedCorrelatedTraceCount": 2,
            "modelBackedCorrelatedScenarioCount": 1,
            "deterministicCompatibilityTraceCount": 0,
            "traceSidecarField": "recentTraces",
        },
    }
    report_path = tmp_path / "lumen-agent-grounding-testflight-scenario-gap-package.json"
    report_path.write_text(json.dumps(package), encoding="utf-8")

    package_report = next(
        report
        for report in load_runtime_audit_reports([report_path])
        if report["_sourceFormat"] == "lumen_in_app_dataset_package"
    )

    failure = next(
        failure
        for failure in package_report["failures"]
        if failure["type"] == "agent_grounding_live_e2e_model_backed_trace_gap"
    )
    assert package_report["liveE2EModelBackedCorrelatedTraceCount"] == 2
    assert package_report["liveE2EModelBackedCorrelatedScenarioCount"] == 1
    assert "requiredAgentRunScenarioCount=2" in failure["actual"]
    assert "modelBackedCorrelatedTraceCount=2" in failure["actual"]
    assert "modelBackedCorrelatedScenarioCount=1" in failure["actual"]


def test_runtime_audit_ingest_deduplicates_same_e2e_report_id(tmp_path: Path):
    import json

    report_id = "66666666-6666-6666-6666-666666666666"
    result = {
        "id": "77777777-7777-7777-7777-777777777777",
        "scenarioID": "executor-runtime-preflight",
        "kind": "training",
        "title": "Executor runtime preflight",
        "prompt": "Preflight executor runtime",
        "expectedIntent": "weather",
        "actualIntent": "preflight",
        "requiresAgentRun": True,
        "passed": False,
        "failures": [
            "executor preflight failed: resource-budget-denied-before-prompt-eval; thermalState=serious"
        ],
        "events": [],
    }
    plain_report = {
        "id": report_id,
        "startedAt": "2026-06-29T00:00:00Z",
        "finishedAt": "2026-06-29T00:00:20Z",
        "passed": 0,
        "failed": 1,
        "results": [result],
    }
    envelope_report = {
        "schemaVersion": "1.0.0",
        "generatedAt": "2026-06-29T00:00:30Z",
        "exportPolicy": {
            "format": "live-e2e-test-report-json",
            "sourceLayer": "e2eTestReport",
            "ownsLiveE2EScenarios": True,
            "includesDeterministicStaticScenarios": False,
        },
        "payload": plain_report,
    }
    envelope_path = tmp_path / "lumen-live-e2e-report.json"
    latest_path = tmp_path / "latest-e2e-report.json"
    envelope_path.write_text(json.dumps(envelope_report), encoding="utf-8")
    latest_path.write_text(json.dumps(plain_report), encoding="utf-8")

    reports = load_runtime_audit_reports([tmp_path])

    e2e_reports = [report for report in reports if report.get("id") == report_id]
    assert len(e2e_reports) == 1
    assert e2e_reports[0]["failures"][0]["e2eScenario"]["name"] == "Executor runtime preflight"


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
