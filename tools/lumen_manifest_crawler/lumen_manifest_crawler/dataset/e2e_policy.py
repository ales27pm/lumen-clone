from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class E2EFailurePolicy:
    agent: str
    curriculum: str
    failure_type: str
    requires_hint: bool


def e2e_failure_policy(intent: str, required_hint: str | None) -> E2EFailurePolicy:
    normalized_intent = intent.strip().casefold() or "unknown"
    if required_hint:
        failure_type = f"e2e_missing_required_final_hint_{_slug(required_hint)}"
    else:
        failure_type = f"e2e_response_quality_{_slug(normalized_intent)}"
    return E2EFailurePolicy(
        agent=_agent_for_intent(normalized_intent),
        curriculum=_curriculum_for_intent(normalized_intent),
        failure_type=failure_type,
        requires_hint=required_hint is not None,
    )


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", value.casefold()).strip("_") or "unknown"


def _agent_for_intent(intent: str) -> str:
    return "mouth" if intent in {"emaildraft", "email", "maildraft", "chat", "memory", "rag", "websearch", "weblookup", "weather", "trigger"} else "cortex"


def _curriculum_for_intent(intent: str) -> str:
    if intent in {"memory", "rag"}:
        return "grounded_response_quality"
    if intent in {"emaildraft", "email", "maildraft", "trigger", "tool", "websearch", "weblookup"}:
        return "tool_boundary_response_quality"
    return "response_quality"
