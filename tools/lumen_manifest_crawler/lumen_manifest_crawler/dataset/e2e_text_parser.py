from __future__ import annotations

import re
from typing import Any

REPORT_FIELD_HEADERS = {"Prompt", "Intent", "Failures", "Final"}


def parse_e2e_text_report(text: str, *, source: str) -> dict[str, Any] | None:
    normalized = text.replace("\r\n", "\n")
    if "E2E Test Report" not in normalized and "Training eval:" not in normalized:
        return None

    passed = _extract_int(normalized, r"^Passed:\s*(\d+)", default=None)
    failed = _extract_int(normalized, r"^Failed:\s*(\d+)", default=None)
    training_signals = _extract_training_signals(normalized)
    scenarios = _extract_e2e_scenarios(normalized)
    return {
        "_source": source,
        "_sourceFormat": "lumen_e2e_text_report",
        "passed": passed,
        "failed": failed,
        "trainingSignals": training_signals,
        "scenarioCount": len(scenarios),
        "scenarios": scenarios,
    }


def _extract_e2e_scenarios(text: str) -> list[dict[str, Any]]:
    pattern = re.compile(r"(?m)^([✅❌])\s+Training eval:\s*(.+)$")
    matches = list(pattern.finditer(text))
    scenarios: list[dict[str, Any]] = []
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        scenarios.append(_parse_scenario_block(match.group(1), match.group(2).strip(), body))
    return scenarios


def _parse_scenario_block(status: str, name: str, body: str) -> dict[str, Any]:
    prompt = _extract_field(body, "Prompt")
    intent_line = _extract_field(body, "Intent")
    failures_line = _extract_field(body, "Failures")
    final = _extract_multiline_field(body, "Final")
    intent, expected_intent = _parse_intent_line(intent_line)
    return {
        "name": name,
        "passed": status == "✅",
        "prompt": prompt,
        "intent": intent,
        "expectedIntent": expected_intent,
        "failures": failures_line,
        "final": final,
    }


def _extract_int(text: str, pattern: str, *, default: int | None) -> int | None:
    match = re.search(pattern, text, flags=re.MULTILINE)
    if not match:
        return default
    try:
        return int(match.group(1))
    except ValueError:
        return default


def _extract_field(body: str, field: str) -> str:
    match = re.search(rf"(?m)^{re.escape(field)}:\s*(.*)$", body)
    return match.group(1).strip() if match else ""


def _extract_multiline_field(body: str, field: str) -> str:
    match = re.search(rf"(?m)^{re.escape(field)}:\s*", body)
    if not match:
        return ""

    heading_labels = sorted(REPORT_FIELD_HEADERS - {field})
    heading_pattern = rf"^({'|'.join(map(re.escape, heading_labels))}):\s*" if heading_labels else ""
    stop = re.search(heading_pattern, body[match.end():], flags=re.MULTILINE) if heading_pattern else None
    stop_index = match.end() + stop.start() if stop else len(body)
    value = body[match.end():stop_index]
    return value.strip()


def _parse_intent_line(value: str) -> tuple[str, str | None]:
    if not value:
        return "unknown", None
    parts = [part.strip() for part in value.split("/ expected ", 1)]
    return parts[0], parts[1] if len(parts) > 1 else None


def _extract_training_signals(text: str) -> list[str]:
    signals: list[str] = []
    capture = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("Training signals for next run"):
            capture = True
            continue
        if capture and re.match(r"^[✅❌]\s+Training eval:", stripped):
            break
        if capture and stripped.startswith("•"):
            signals.append(stripped.lstrip("•").strip())
    return signals
