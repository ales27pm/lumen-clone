from __future__ import annotations

import re
from typing import Any

from lumen_manifest_crawler.dataset.e2e_policy import e2e_failure_policy


def flatten_e2e_json_report(value: dict[str, Any], *, source: str, source_format: str = "lumen_e2e_test_report", source_layer: str = "e2eTestReport.json") -> dict[str, Any]:
    scenarios = value.get("scenarios") if isinstance(value.get("scenarios"), list) else []
    failures = [
        e2e_failure_from_scenario(scenario, source_layer=source_layer)
        for scenario in scenarios
        if isinstance(scenario, dict) and (scenario.get("passed") is not True or _scenario_skipped_live_model_run(scenario))
    ]
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


def e2e_failure_from_scenario(scenario: dict[str, Any], *, source_layer: str) -> dict[str, Any]:
    failure_text = str(scenario.get("failures") or "E2E scenario failed.").strip()
    if _scenario_skipped_live_model_run(scenario) and failure_text == "E2E scenario failed.":
        failure_text = "Agent model did not execute: scenario fell back to routing-only checks because no chat model was loaded."
    prompt = str(scenario.get("prompt") or "").strip()
    final = str(scenario.get("final") or "").strip()
    intent = _scenario_intent(scenario)
    required_hint = _extract_required_hint(failure_text)
    policy = e2e_failure_policy(intent, required_hint)
    expected = _expected_for_e2e_failure(scenario, required_hint)
    corrected = _corrected_output_for_e2e_failure(scenario, required_hint)
    return {
        "type": policy.failure_type,
        "agent": policy.agent,
        "expected": [expected],
        "actual": final,
        "scenario": prompt,
        "problem": failure_text,
        "sourceLayer": source_layer,
        "e2eScenario": {
            "name": scenario.get("name"),
            "intent": intent,
            "expectedIntent": scenario.get("expectedIntent"),
            "prompt": prompt,
            "final": final,
            "requiredHint": required_hint,
            "skippedLiveModelRun": _scenario_skipped_live_model_run(scenario),
        },
        "repairSample": {
            "agent": policy.agent,
            "violationCode": policy.failure_type,
            "promptPrefix": prompt[:500],
            "expected": expected,
            "badOutput": final[:1000],
            "correctedOutput": corrected,
            "lesson": _lesson_for_e2e_failure(scenario, required_hint),
            "curriculum": policy.curriculum,
        },
    }


def _scenario_intent(scenario: dict[str, Any]) -> str:
    intent = str(scenario.get("intent") or "").strip()
    if intent:
        return intent
    expected_intent = str(scenario.get("expectedIntent") or "").strip()
    return expected_intent or "unknown"


def _expected_for_e2e_failure(scenario: dict[str, Any], required_hint: str | None) -> str:
    if _scenario_skipped_live_model_run(scenario):
        return "E2E scenarios that require an agent run must execute an actual loaded chat model; routing-only fallback is not valid E2E evidence."
    if required_hint:
        return f"Final answer must include the required hint `{required_hint}` while preserving the requested intent and user-visible usefulness."
    expected_intent = scenario.get("expectedIntent") or scenario.get("intent") or "expected intent"
    return f"Final answer must satisfy the `{expected_intent}` eval contract without violating tool boundaries."


def _corrected_output_for_e2e_failure(scenario: dict[str, Any], required_hint: str | None) -> str:
    prompt = str(scenario.get("prompt") or "").strip()
    final = str(scenario.get("final") or "").strip()
    intent = _scenario_intent(scenario)
    normalized_intent = intent.casefold()
    if _scenario_skipped_live_model_run(scenario):
        return "Load the configured chat model/fleet and rerun this scenario through SlotAgentService; do not emit routing-only fallback as a passing E2E result."
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


def _scenario_skipped_live_model_run(scenario: dict[str, Any]) -> bool:
    final = str(scenario.get("final") or scenario.get("finalText") or "").casefold()
    failures = str(scenario.get("failures") or "").casefold()
    events = scenario.get("events") if isinstance(scenario.get("events"), list) else []
    event_text = " ".join(str(event) for event in events).casefold()
    haystack = "\n".join([final, failures, event_text])
    return "no model loaded" in haystack or "routing-only checks completed" in haystack


def _clean_prompt(prompt: str) -> str:
    return " ".join(str(prompt or "").strip().split())


def _clean_derived_fragment(value: str) -> str:
    cleaned = _clean_prompt(value).strip(" ,.;:!?\"'")
    cleaned = re.sub(r"\bthen\s+(?:tell|confirm|say)\b.*$", "", cleaned, flags=re.IGNORECASE).strip(" ,.;:!?\"'")
    return cleaned


def _lesson_for_e2e_failure(scenario: dict[str, Any], required_hint: str | None) -> str:
    intent = _scenario_intent(scenario)
    if _scenario_skipped_live_model_run(scenario):
        return f"For `{intent}` E2E evals, a scenario marked as requiring an agent/model run must fail closed if no model is loaded."
    if required_hint:
        return f"For `{intent}` E2E evals, the final answer must include required hint `{required_hint}` while remaining natural and useful."
    return f"Use failed `{intent}` E2E prompts and final outputs as next-cycle fine-tuning repair examples."


def _extract_required_hint(text: str) -> str | None:
    match = re.search(r"Required final hint missing:\s*`?([^`\n.]+)`?", text, flags=re.IGNORECASE)
    if not match:
        return None
    return match.group(1).strip().strip("`'\"") or None
