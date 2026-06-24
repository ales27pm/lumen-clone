from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import Any

from lumen_manifest_crawler.dataset.e2e_policy import e2e_failure_policy


def flatten_e2e_json_report(value: dict[str, Any], *, source: str, source_format: str = "lumen_e2e_test_report", source_layer: str = "e2eTestReport.json", sidecars: dict[str, list[dict[str, Any]]] | None = None) -> dict[str, Any]:
    raw_scenarios = _coerce_e2e_scenarios(value)
    sidecars = sidecars or {}
    failures = []
    scenarios = []
    for raw_scenario in raw_scenarios:
        scenario = dict(raw_scenario)
        diagnosis = _model_evidence_diagnosis_for_scenario(scenario, sidecars)
        if diagnosis:
            scenario["modelEvidenceStatus"] = diagnosis.get("rootCauseCategory")
            scenario["modelEvidenceTrace"] = diagnosis.get("trace")
        scenarios.append(scenario)
        evidence_requires_failure = _scenario_model_evidence_requires_failure(scenario, diagnosis)
        failure_diagnosis = diagnosis if evidence_requires_failure or _has_generic_missing_model_evidence(scenario) else None
        if scenario.get("passed") is not True or evidence_requires_failure or _scenario_skipped_live_model_run(
            scenario,
            sidecar_diagnosis=diagnosis,
        ):
            failures.append(
                e2e_failure_from_scenario(
                    scenario,
                    source_layer=source_layer,
                    sidecar_diagnosis=failure_diagnosis,
                )
            )
    return {
        "_source": source,
        "_sourceFormat": source_format,
        "passed": value.get("passed"),
        "failed": value.get("failed"),
        "trainingSignals": value.get("trainingSignals") or value.get("training_signals") or [],
        "scenarioCount": value.get("scenarioCount") or len(scenarios),
        "failures": failures,
        "scenarios": scenarios,
    }


def _coerce_e2e_scenarios(value: dict[str, Any]) -> list[dict[str, Any]]:
    scenarios = value.get("scenarios")
    if isinstance(scenarios, list):
        return [scenario for scenario in scenarios if isinstance(scenario, dict)]
    results = value.get("results")
    if not isinstance(results, list):
        return []
    coerced: list[dict[str, Any]] = []
    for result in results:
        if not isinstance(result, dict):
            continue
        failures = result.get("failures")
        if isinstance(failures, list):
            failure_text: Any = "; ".join(str(item) for item in failures if item)
        else:
            failure_text = failures
        coerced.append({
            "id": result.get("scenarioID") or result.get("id"),
            "name": result.get("title") or result.get("scenarioID"),
            "passed": result.get("passed") is True,
            "requiresAgentRun": result.get("requiresAgentRun") is True,
            "kind": result.get("kind"),
            "scenarioID": result.get("scenarioID"),
            "e2eRunID": result.get("e2eRunID"),
            "agentRunID": result.get("agentRunID"),
            "conversationID": result.get("conversationID"),
            "turnID": result.get("turnID"),
            "prompt": result.get("prompt"),
            "intent": result.get("actualIntent") or result.get("expectedIntent"),
            "expectedIntent": result.get("expectedIntent"),
            "failures": failure_text,
            "final": result.get("finalText"),
            "events": result.get("events") or [],
            "startedAt": result.get("startedAt"),
            "finishedAt": result.get("finishedAt"),
        })
    return coerced


def e2e_failure_from_scenario(scenario: dict[str, Any], *, source_layer: str, sidecar_diagnosis: dict[str, Any] | None = None) -> dict[str, Any]:
    failure_text = str(scenario.get("failures") or "E2E scenario failed.").strip()
    if sidecar_diagnosis and (_is_generic_missing_model_evidence(failure_text) or failure_text == "E2E scenario failed."):
        failure_text = str(sidecar_diagnosis.get("message") or failure_text)
    if _scenario_skipped_live_model_run(scenario, sidecar_diagnosis=sidecar_diagnosis) and failure_text == "E2E scenario failed.":
        failure_text = "Agent model did not execute: scenario fell back to routing-only checks because no chat model was loaded."
    prompt = str(scenario.get("prompt") or "").strip()
    final = str(scenario.get("final") or "").strip()
    intent = _scenario_intent(scenario)
    required_hint = _extract_required_hint(failure_text)
    policy = e2e_failure_policy(intent, required_hint)
    expected = _expected_for_e2e_failure(scenario, required_hint, sidecar_diagnosis=sidecar_diagnosis)
    corrected = _corrected_output_for_e2e_failure(scenario, required_hint, sidecar_diagnosis=sidecar_diagnosis)
    return {
        "type": policy.failure_type,
        "agent": policy.agent,
        "expected": [expected],
        "actual": final,
        "scenario": prompt,
        "problem": failure_text,
        "rootCauseCategory": sidecar_diagnosis.get("rootCauseCategory") if sidecar_diagnosis else None,
        "sourceLayer": source_layer,
        "e2eScenario": {
            "name": scenario.get("name"),
            "intent": intent,
            "expectedIntent": scenario.get("expectedIntent"),
            "prompt": prompt,
            "final": final,
            "requiredHint": required_hint,
            "skippedLiveModelRun": _scenario_skipped_live_model_run(scenario, sidecar_diagnosis=sidecar_diagnosis),
            "modelEvidenceRootCause": sidecar_diagnosis.get("rootCauseCategory") if sidecar_diagnosis else None,
            "modelEvidenceTrace": sidecar_diagnosis.get("trace") if sidecar_diagnosis else None,
            "modelEvidenceStatus": sidecar_diagnosis.get("rootCauseCategory") if sidecar_diagnosis else scenario.get("modelEvidenceStatus"),
        },
        "repairSample": {
            "agent": policy.agent,
            "violationCode": policy.failure_type,
            "promptPrefix": prompt[:500],
            "expected": expected,
            "badOutput": final[:1000],
            "correctedOutput": corrected,
            "lesson": _lesson_for_e2e_failure(scenario, required_hint, sidecar_diagnosis=sidecar_diagnosis),
            "curriculum": policy.curriculum,
        },
    }


def _scenario_intent(scenario: dict[str, Any]) -> str:
    intent = str(scenario.get("intent") or "").strip()
    if intent:
        return intent
    expected_intent = str(scenario.get("expectedIntent") or "").strip()
    return expected_intent or "unknown"


def _expected_for_e2e_failure(scenario: dict[str, Any], required_hint: str | None, sidecar_diagnosis: dict[str, Any] | None = None) -> str:
    if sidecar_diagnosis:
        category = str(sidecar_diagnosis.get("rootCauseCategory") or "")
        if category == "deterministic_compatibility_not_training_evidence":
            return "Training scenarios must record fresh model-backed AgentBehaviorTrace modelTurn evidence; deterministic compatibility or policy-first traces are not training evidence."
        if category in {"no_correlated_model_turn", "agent_service_not_entered", "missing_sidecar_trace_export"}:
            return "Training scenarios that require an agent run must export a correlated model-backed AgentBehaviorTrace modelTurn or fail closed with the exact missing-path reason."
        return "Agent-json turns must produce non-empty model output that parses as structured JSON before deterministic recovery or final-answer validation can count."
    if _scenario_skipped_live_model_run(scenario):
        return "E2E scenarios that require an agent run must execute an actual loaded chat model; routing-only fallback is not valid E2E evidence."
    if required_hint:
        return f"Final answer must include the required hint `{required_hint}` while preserving the requested intent and user-visible usefulness."
    expected_intent = scenario.get("expectedIntent") or scenario.get("intent") or "expected intent"
    return f"Final answer must satisfy the `{expected_intent}` eval contract without violating tool boundaries."


def _corrected_output_for_e2e_failure(scenario: dict[str, Any], required_hint: str | None, sidecar_diagnosis: dict[str, Any] | None = None) -> str:
    prompt = str(scenario.get("prompt") or "").strip()
    final = str(scenario.get("final") or "").strip()
    intent = _scenario_intent(scenario)
    normalized_intent = intent.casefold()
    if sidecar_diagnosis:
        category = str(sidecar_diagnosis.get("rootCauseCategory") or "")
        if category == "deterministic_compatibility_not_training_evidence":
            return "Route this training scenario through AgentService's model-backed agent-json path and keep deterministic compatibility traces as diagnostics only."
        if category == "missing_sidecar_trace_export":
            return "Export the AgentBehaviorTrace sidecar or include correlated model-evidence events in the live E2E report before using this artifact as training evidence."
        if category in {"no_correlated_model_turn", "agent_service_not_entered"}:
            return "Pass the E2E correlation IDs into AgentService and persist them on AgentBehaviorTrace modelTurn records, or keep the scenario failed with the precise missing-path diagnostic."
        return "Fix the executor-slot agent-json generation path so it emits non-empty structured JSON, or keep this scenario failed with the precise agent-json empty-output parse diagnostic."
    if _scenario_skipped_live_model_run(scenario):
        return "Load the configured chat model/fleet and rerun this scenario through AgentService's model-backed generation path; do not emit routing-only fallback or compatibility output as passing E2E evidence."
    if required_hint and normalized_intent == "memory":
        remembered = _derive_memory_content_from_prompt(prompt)
        base = final if _is_useful_final(final, intent=intent) else f"Remembered: {remembered}."
        return _ensure_required_hint(base, required_hint)
    if required_hint and normalized_intent in {"emaildraft", "email", "maildraft"}:
        draft = _derive_email_draft_from_prompt(prompt, final)
        return _ensure_required_hint(draft, required_hint)
    if required_hint:
        base = final if _is_useful_final(final, intent=intent) else _generic_corrected_output_from_prompt(prompt, intent)
        return _ensure_required_hint(base, required_hint)
    return final or f"Ask a clarification or produce a manifest-compliant final answer for: {prompt}"


def _derive_memory_content_from_prompt(prompt: str) -> str:
    clean = _clean_prompt(prompt)
    patterns = [
        r"\bremember\s+that\s+(.+?)(?:,?\s+then\b|\s+and\s+(?:tell|confirm|say)\b|[.!?]?$)",
        r"\bremember\s+(.+?)(?:,?\s+then\b|\s+and\s+(?:tell|confirm|say)\b|[.!?]?$)",
        r"\bkeep\s+this\s+in\s+mind\s*:?\s*(.+?)(?:,?\s+then\b|[.!?]?$)",
        r"\bsave\s+(?:this|that)?\s*(?:as\s+)?(?:a\s+)?(?:preference|memory|note)?\s*:?\s*(.+?)(?:,?\s+then\b|[.!?]?$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, clean, flags=re.IGNORECASE)
        if match:
            candidate = _clean_derived_fragment(match.group(1))
            if candidate:
                return candidate
    return clean or "the requested memory"


def _derive_email_draft_from_prompt(prompt: str, final: str) -> str:
    if _looks_like_email_draft(final):
        return final.strip()
    clean = _clean_prompt(prompt)
    recipient = _extract_recipient(clean)
    subject_hint = _extract_subject_hint(clean)
    question = _derive_clarifying_question(clean)
    greeting = f"Hi {recipient}," if recipient else "Hi,"
    subject = f"Subject: {subject_hint}" if subject_hint else "Subject: Professional update"
    body = _derive_email_body_sentence(clean)
    return "\n".join([
        subject,
        "",
        greeting,
        "",
        body,
        "",
        question,
    ]).strip()


def _derive_email_body_sentence(prompt: str) -> str:
    lower = prompt.casefold()
    if "professional update" in lower:
        return "Here is a professional update on the current work: progress is moving forward, and I will share the next concrete milestone once the remaining details are confirmed."
    if "update" in lower:
        return "Here is the requested update: progress is moving forward, and I will confirm the next concrete details as soon as they are available."
    if "draft" in lower or "email" in lower:
        return "I wanted to send a clear professional note and confirm the next detail before moving forward."
    return f"I am following up about: {prompt}"


def _derive_clarifying_question(prompt: str) -> str:
    if re.search(r"\b(one|1)\s+clarifying\s+question\b", prompt, flags=re.IGNORECASE):
        return "One clarifying question: what specific deadline, priority, or next step should I align this update with?"
    if re.search(r"\bask\b.*\bquestion\b", prompt, flags=re.IGNORECASE):
        return "Question: what specific detail should I confirm before sending this?"
    return "Question: what detail should I confirm before sending this?"


def _extract_recipient(prompt: str) -> str | None:
    match = re.search(r"\b(?:to|for)\s+([A-Z][A-Za-z0-9_.-]*)\b", prompt)
    if match:
        return match.group(1).strip()
    return None


def _extract_subject_hint(prompt: str) -> str | None:
    match = re.search(r"\babout\s+(.+?)(?:\s+and\s+ask\b|\s+with\b|[.!?]?$)", prompt, flags=re.IGNORECASE)
    if not match:
        return None
    candidate = _clean_derived_fragment(match.group(1))
    if not candidate:
        return None
    return candidate[:1].upper() + candidate[1:]


def _looks_like_email_draft(value: str) -> bool:
    text = value.strip()
    if not text:
        return False
    lowered = text.casefold()
    has_greeting = bool(re.search(r"(?m)^\s*(hi|hello|dear)\b", lowered))
    has_subject = bool(re.search(r"(?m)^\s*subject\s*:", lowered))
    has_question = "?" in text or "question" in lowered
    return (has_greeting or has_subject) and has_question


def _generic_corrected_output_from_prompt(prompt: str, intent: str) -> str:
    clean = _clean_prompt(prompt)
    if clean:
        return f"I handled this `{intent}` request in a manifest-compliant way: {clean}"
    return f"I handled this `{intent}` request in a manifest-compliant way."


def _ensure_required_hint(text: str, required_hint: str | None) -> str:
    cleaned = text.strip()
    if not required_hint:
        return cleaned
    if required_hint.casefold() in cleaned.casefold():
        return cleaned
    if required_hint.casefold() == "question":
        return f"{cleaned}\n\nQuestion: what detail should I confirm before proceeding?"
    return f"{cleaned}\n\n{required_hint}"


def _is_useful_final(final: str, *, intent: str) -> bool:
    stripped = final.strip()
    if not stripped:
        return False
    if stripped.casefold() == intent.casefold():
        return False
    return len(stripped.split()) >= 3


_EXPLICIT_MODEL_EVIDENCE_CATEGORIES = {
    "valid_model_backed_evidence",
    "no_correlated_model_turn",
    "agent_model_empty_output",
    "agent_model_parse_error",
    "deterministic_compatibility_not_training_evidence",
    "agent_service_not_entered",
    "missing_sidecar_trace_export",
    # Backward-compatible names produced by older sidecar diagnostics.
    "agent_json_empty_generation",
    "agent_json_parse_empty",
    "agent_json_parse_error",
}


def _scenario_model_evidence_requires_failure(scenario: dict[str, Any], diagnosis: dict[str, Any] | None) -> bool:
    if scenario.get("requiresAgentRun") is not True or not diagnosis:
        return False
    category = str(diagnosis.get("rootCauseCategory") or "")
    return category in _EXPLICIT_MODEL_EVIDENCE_CATEGORIES and category != "valid_model_backed_evidence"


def _scenario_skipped_live_model_run(scenario: dict[str, Any], sidecar_diagnosis: dict[str, Any] | None = None) -> bool:
    final = str(scenario.get("final") or scenario.get("finalText") or "").casefold()
    failures = str(scenario.get("failures") or "").casefold()
    events = scenario.get("events") if isinstance(scenario.get("events"), list) else []
    event_text = " ".join(str(event) for event in events).casefold()
    haystack = "\n".join([final, failures, event_text])
    if (
        "no model loaded" in haystack
        or "routing-only checks completed" in haystack
    ):
        return True
    if sidecar_diagnosis and sidecar_diagnosis.get("rootCauseCategory") in _EXPLICIT_MODEL_EVIDENCE_CATEGORIES:
        return False
    if (
        "did not record model-backed generation evidence" in haystack
        or "missing fresh agentbehaviortrace modelturn" in haystack
    ):
        return True
    if scenario.get("requiresAgentRun") is True and not _scenario_has_model_evidence_event(events, scenario=scenario):
        return True
    return False


def _model_evidence_diagnosis_for_scenario(scenario: dict[str, Any], sidecars: dict[str, list[dict[str, Any]]]) -> dict[str, Any] | None:
    prompt = str(scenario.get("prompt") or "").strip()
    event_diagnosis = _diagnosis_from_model_evidence_events(scenario)
    if event_diagnosis and event_diagnosis.get("rootCauseCategory") in {
        "valid_model_backed_evidence",
        "deterministic_compatibility_not_training_evidence",
        "agent_service_not_entered",
    }:
        return event_diagnosis
    if not prompt:
        return event_diagnosis
    prompt_key = prompt.casefold()
    traces = sidecars.get("agent_behavior_traces", [])
    sidecar_present = _trace_sidecar_present(sidecars)

    matching_trace, matched_by = _matching_sidecar_trace_for_scenario(scenario, traces, prompt_key)
    if matching_trace is not None:
        raw = str(matching_trace.get("rawOutputPrefix") or "")
        parse_error = matching_trace.get("parseError")
        runtime_path = str(matching_trace.get("runtimePath") or "unknown")
        stage = str(matching_trace.get("stage") or "unknown")
        if _is_model_backed_trace(matching_trace) and raw.strip() and not parse_error:
            return {
                "rootCauseCategory": "valid_model_backed_evidence",
                "message": f"valid model-backed AgentBehaviorTrace modelTurn found; stage={stage}; runtimePath={runtime_path}; matchedBy={matched_by}",
                "trace": _trace_summary(matching_trace, matched_by=matched_by, raw_output_empty=False),
            }
        if runtime_path == "deterministic-compatibility" and _is_training_scenario(scenario):
            return {
                "rootCauseCategory": "deterministic_compatibility_not_training_evidence",
                "message": f"deterministic compatibility trace is not training model evidence; stage={stage}; runtimePath={runtime_path}",
                "trace": _trace_summary(matching_trace, matched_by=matched_by, raw_output_empty=not raw.strip()),
            }
        if not raw.strip():
            category = "agent_model_empty_output"
            if parse_error == "empty":
                category = "agent_json_empty_generation"
            empty_reason = str(matching_trace.get("emptyOutputReason") or "")
            detail = "model stream returned no tokens" if empty_reason == "agent-json-stream-completed-without-text" else "agent-json emitted empty output"
            stage = str(matching_trace.get("stage") or "unknown")
            return {
                "rootCauseCategory": category,
                "message": f"{detail}; parseError={parse_error or 'none'}; stage={stage}; runtimePath={runtime_path}",
                "trace": _trace_summary(matching_trace, matched_by=matched_by, raw_output_empty=True),
            }
        if parse_error:
            category = "agent_json_parse_error" if str(parse_error) in {"noJSONObject", "multipleJSONObjects", "noisyOutput", "malformedEscapeSequence", "incompleteJSON", "invalidJSONObject", "invalidThoughtType", "invalidFinalType", "mixedTurn", "mixedActionShapes", "missingActionOrFinal", "missingActionTool", "invalidActionType", "invalidActionArgsType"} else "agent_model_parse_error"
            return {
                "rootCauseCategory": category,
                "message": f"agent-json output failed to parse; parseError={parse_error}; stage={stage}; runtimePath={runtime_path}",
                "trace": _trace_summary(matching_trace, matched_by=matched_by, raw_output_empty=False),
            }

    matching_parse_failure = next(
        (
            failure
            for failure in sidecars.get("agent_parse_failures", [])
            if str(failure.get("modelName") or "") == "agent-json"
            and _sidecar_text_matches_prompt(str(failure.get("userTurnPrefix") or ""), prompt_key)
            and _sidecar_record_matches_scenario_time(failure, scenario)
        ),
        None,
    )
    if matching_parse_failure is not None:
        raw = str(matching_parse_failure.get("rawOutputPrefix") or "")
        parse_error = str(matching_parse_failure.get("parseError") or "unknown")
        if parse_error == "empty" or not raw.strip():
            return {
                "rootCauseCategory": "agent_json_parse_empty",
                "message": f"agent-json parse failed with empty output; parseError={parse_error}",
                "trace": {
                    "stage": "agent-json",
                    "runtimePath": "unknown",
                    "parseError": parse_error,
                    "rawOutputEmpty": not raw.strip(),
                },
            }
    if event_diagnosis:
        return event_diagnosis
    if scenario.get("requiresAgentRun") is True and not _scenario_reported_no_model_loaded(scenario) and _has_generic_missing_model_evidence(scenario):
        if not sidecar_present:
            return {
                "rootCauseCategory": "missing_sidecar_trace_export",
                "message": "missing AgentBehaviorTrace sidecar export; cannot verify correlated modelTurn evidence",
                "trace": {"sidecarPresent": False},
            }
        return {
            "rootCauseCategory": "no_correlated_model_turn",
            "message": f"no correlated AgentBehaviorTrace modelTurn found; checked {_scenario_correlation_text(scenario)}",
            "trace": {"sidecarPresent": True, "checked": _scenario_correlation_fields(scenario)},
        }
    return None


def _diagnosis_from_model_evidence_events(scenario: dict[str, Any]) -> dict[str, Any] | None:
    events = scenario.get("events") if isinstance(scenario.get("events"), list) else []
    for event in events:
        if not isinstance(event, dict):
            continue
        phase = str(event.get("phase") or "").casefold()
        message = str(event.get("message") or "")
        lowered = message.casefold()
        if phase != "model-evidence":
            continue
        values = _parse_model_evidence_values(message)
        runtime = str(values.get("runtime") or "").casefold()
        kind = str(values.get("kind") or "").casefold()
        parse_error = str(values.get("parseError") or values.get("parseerror") or "none")
        if "no correlated agentbehaviortrace" in lowered:
            return {
                "rootCauseCategory": "no_correlated_model_turn",
                "message": message,
                "trace": {"eventMessage": message, "checked": _scenario_correlation_fields(scenario)},
            }
        if "agentservice model path was not entered" in lowered:
            if "model not loaded" in lowered or _scenario_reported_no_model_loaded(scenario):
                continue
            return {
                "rootCauseCategory": "agent_service_not_entered",
                "message": message,
                "trace": {"eventMessage": message},
            }
        if "missing sidecar trace export" in lowered:
            return {
                "rootCauseCategory": "missing_sidecar_trace_export",
                "message": message,
                "trace": {"eventMessage": message, "sidecarPresent": False},
            }
        if "deterministic-compatibility" in runtime or "policy-first-deterministic" in kind or "deterministic-compatibility execution trace" in lowered:
            if _is_training_scenario(scenario):
                return {
                    "rootCauseCategory": "deterministic_compatibility_not_training_evidence",
                    "message": f"deterministic compatibility trace is not training model evidence; {message}",
                    "trace": {"eventMessage": message, "runtimePath": runtime or "deterministic-compatibility", "stage": values.get("stage")},
                }
            continue
        if "model stream returned no tokens" in lowered or "raw output was empty" in lowered or "agent-json emitted empty output" in lowered:
            return {
                "rootCauseCategory": "agent_model_empty_output",
                "message": message,
                "trace": {"eventMessage": message, "stage": values.get("stage"), "runtimePath": values.get("runtime"), "parseError": parse_error, "rawOutputEmpty": True},
            }
        if "parseerror=" in lowered and parse_error.casefold() not in {"", "none", "nil", "null"}:
            return {
                "rootCauseCategory": "agent_model_parse_error",
                "message": message,
                "trace": {"eventMessage": message, "stage": values.get("stage"), "runtimePath": values.get("runtime"), "parseError": parse_error, "rawOutputEmpty": False},
            }
        if "runtime=" in lowered and "missing" not in lowered and runtime != "deterministic-compatibility":
            return {
                "rootCauseCategory": "valid_model_backed_evidence",
                "message": message,
                "trace": {"eventMessage": message, "stage": values.get("stage"), "runtimePath": values.get("runtime"), "parseError": parse_error, "rawOutputEmpty": False},
            }
    return None


def _parse_model_evidence_values(message: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for match in re.finditer(r"([A-Za-z][A-Za-z0-9_]+)=([^,;]+)", message):
        values[match.group(1)] = match.group(2).strip()
    return values


def _trace_sidecar_present(sidecars: dict[str, list[dict[str, Any]]]) -> bool:
    presence_records = sidecars.get("_sidecar_presence") or []
    if presence_records and isinstance(presence_records[0], dict):
        return presence_records[0].get("agent_behavior_traces") is True
    return "agent_behavior_traces" in sidecars


def _matching_sidecar_trace_for_scenario(
    scenario: dict[str, Any],
    traces: list[dict[str, Any]],
    prompt_key: str,
) -> tuple[dict[str, Any] | None, str]:
    correlated = next(
        (
            trace
            for trace in traces
            if _sidecar_trace_matches_correlation(trace, scenario)
            and str(trace.get("event") or "") == "modelTurn"
            and str(trace.get("stage") or "").startswith("agent-json")
        ),
        None,
    )
    if correlated is not None:
        return correlated, "correlation"
    fallback = next(
        (
            trace
            for trace in traces
            if _sidecar_text_matches_prompt(str(trace.get("promptPrefix") or ""), prompt_key)
            and _sidecar_record_matches_scenario_time(trace, scenario)
            and str(trace.get("event") or "") == "modelTurn"
            and str(trace.get("stage") or "").startswith("agent-json")
        ),
        None,
    )
    if fallback is not None:
        return fallback, "prompt-time"
    return None, "none"


def _sidecar_trace_matches_correlation(trace: dict[str, Any], scenario: dict[str, Any]) -> bool:
    matched_strong_identifier = False
    for key in ("e2eRunID", "agentRunID", "conversationID", "turnID"):
        expected = str(scenario.get(key) or "").strip()
        actual = str(trace.get(key) or "").strip()
        if expected and actual:
            if expected != actual:
                return False
            matched_strong_identifier = True
    expected_scenario = str(scenario.get("scenarioID") or scenario.get("id") or "").strip()
    actual_scenario = str(trace.get("scenarioID") or "").strip()
    if expected_scenario and actual_scenario and expected_scenario != actual_scenario:
        return False
    if matched_strong_identifier:
        return True
    return bool(expected_scenario and actual_scenario and expected_scenario == actual_scenario and _sidecar_record_matches_scenario_time(trace, scenario))


def _is_model_backed_trace(trace: dict[str, Any]) -> bool:
    return str(trace.get("event") or "") == "modelTurn" and str(trace.get("runtimePath") or "") != "deterministic-compatibility"


def _trace_summary(trace: dict[str, Any], *, matched_by: str, raw_output_empty: bool) -> dict[str, Any]:
    return {
        "stage": str(trace.get("stage") or "unknown"),
        "runtimePath": str(trace.get("runtimePath") or "unknown"),
        "parseError": trace.get("parseError"),
        "rawOutputEmpty": raw_output_empty,
        "matchedBy": matched_by,
        "scenarioID": trace.get("scenarioID"),
        "e2eRunID": trace.get("e2eRunID"),
        "agentRunID": trace.get("agentRunID"),
        "conversationID": trace.get("conversationID"),
        "turnID": trace.get("turnID"),
    }


def _is_training_scenario(scenario: dict[str, Any]) -> bool:
    kind = str(scenario.get("kind") or "").casefold()
    if kind == "training":
        return True
    scenario_id = str(scenario.get("scenarioID") or scenario.get("id") or scenario.get("name") or "").casefold()
    return scenario_id.startswith("training")


def _scenario_correlation_fields(scenario: dict[str, Any]) -> dict[str, Any]:
    return {
        "scenarioID": scenario.get("scenarioID") or scenario.get("id"),
        "e2eRunID": scenario.get("e2eRunID"),
        "agentRunID": scenario.get("agentRunID"),
        "conversationID": scenario.get("conversationID"),
        "turnID": scenario.get("turnID"),
    }


def _scenario_correlation_text(scenario: dict[str, Any]) -> str:
    fields = _scenario_correlation_fields(scenario)
    return ",".join(f"{key}={value or 'nil'}" for key, value in fields.items())


def _has_generic_missing_model_evidence(scenario: dict[str, Any]) -> bool:
    failures = str(scenario.get("failures") or "")
    events = scenario.get("events") if isinstance(scenario.get("events"), list) else []
    event_text = " ".join(str(event) for event in events)
    return _is_generic_missing_model_evidence(f"{failures}\n{event_text}") or (
        scenario.get("requiresAgentRun") is True and not _scenario_has_model_evidence_event(events, scenario=scenario)
    )


def _scenario_reported_no_model_loaded(scenario: dict[str, Any]) -> bool:
    final = str(scenario.get("final") or scenario.get("finalText") or "").casefold()
    failures = str(scenario.get("failures") or "").casefold()
    events = scenario.get("events") if isinstance(scenario.get("events"), list) else []
    event_text = " ".join(str(event) for event in events).casefold()
    return "no chat model loaded" in f"{final}\n{failures}\n{event_text}" or "no model loaded" in f"{final}\n{failures}\n{event_text}"


def _sidecar_text_matches_prompt(value: str, prompt_key: str) -> bool:
    normalized = " ".join(value.casefold().split())
    needle = " ".join(prompt_key.split())
    return bool(needle and normalized and (needle in normalized or normalized in needle))


def _sidecar_record_matches_scenario_time(record: dict[str, Any], scenario: dict[str, Any]) -> bool:
    started_at = _parse_iso_datetime(scenario.get("startedAt"))
    finished_at = _parse_iso_datetime(scenario.get("finishedAt"))
    if started_at is None and finished_at is None:
        return True
    created_at = _parse_iso_datetime(record.get("createdAt"))
    if created_at is None:
        return False
    lower_bound = (started_at or finished_at) - timedelta(minutes=5)
    upper_bound = (finished_at or started_at) + timedelta(minutes=5)
    return lower_bound <= created_at <= upper_bound


def _parse_iso_datetime(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _is_generic_missing_model_evidence(text: str) -> bool:
    lowered = text.casefold()
    return (
        "did not record model-backed generation evidence" in lowered
        or "missing fresh agentbehaviortrace modelturn" in lowered
    )


def _scenario_has_model_evidence_event(events: list[Any], *, scenario: dict[str, Any] | None = None) -> bool:
    for event in events:
        if not isinstance(event, dict):
            continue
        phase = str(event.get("phase") or "").casefold()
        message = str(event.get("message") or "").casefold()
        if phase != "model-evidence" or "runtime=" not in message or "missing" in message:
            continue
        is_deterministic = "deterministic-compatibility" in message or "policy-first-deterministic" in message
        if is_deterministic:
            if scenario and _is_training_scenario(scenario):
                continue
            return True
        if "kind=model-backed" in message or "deterministic-compatibility" not in message:
            return True
    return False


def _clean_prompt(prompt: str) -> str:
    return " ".join(str(prompt or "").strip().split())


def _clean_derived_fragment(value: str) -> str:
    cleaned = _clean_prompt(value).strip(" ,.;:!?\"'")
    cleaned = re.sub(r"\bthen\s+(?:tell|confirm|say)\b.*$", "", cleaned, flags=re.IGNORECASE).strip(" ,.;:!?\"'")
    return cleaned


def _lesson_for_e2e_failure(scenario: dict[str, Any], required_hint: str | None, sidecar_diagnosis: dict[str, Any] | None = None) -> str:
    intent = _scenario_intent(scenario)
    if sidecar_diagnosis:
        return f"For `{intent}` E2E evals, empty agent-json output is a model-generation failure, not a skipped live run or a response-quality failure."
    if _scenario_skipped_live_model_run(scenario):
        return f"For `{intent}` E2E evals, a scenario marked as requiring an agent/model run must fail closed unless fresh model-backed generation evidence is recorded."
    if required_hint:
        return f"For `{intent}` E2E evals, the final answer must include required hint `{required_hint}` while remaining natural and useful."
    return f"Use failed `{intent}` E2E prompts and final outputs as next-cycle fine-tuning repair examples."


def _extract_required_hint(text: str) -> str | None:
    match = re.search(r"Required final hint missing:\s*`?([^`\n.]+)`?", text, flags=re.IGNORECASE)
    if not match:
        return None
    return match.group(1).strip().strip("`'\"") or None
