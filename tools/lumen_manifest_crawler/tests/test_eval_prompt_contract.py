from __future__ import annotations

import pytest

from lumen_manifest_crawler.dataset.adapter_evaluation import (
    upgrade_evaluation_record,
)
from lumen_manifest_crawler.dataset.fine_tuning import (
    STRUCTURED_OUTPUT_INSTRUCTION,
    SYSTEM_PROMPTS,
    _bind_evaluation_output_prompt_contract,
    _eval,
)


@pytest.mark.parametrize(
    ("agent", "eval_type", "expected"),
    (
        (
            "mimicry",
            "preference_extraction",
            {
                "extractPreference": True,
                "expectedPreference": {
                    "format": "bullet_points",
                    "length": "concise",
                },
            },
        ),
        (
            "rem",
            "audit_failure_diagnosis",
            {"failureType": "missing_required_tool_action"},
        ),
    ),
)
def test_json_eval_templates_use_the_training_structured_output_instruction(
    agent: str,
    eval_type: str,
    expected: dict[str, object],
) -> None:
    record = _eval(agent, eval_type, "Held-out request.", expected)

    assert upgrade_evaluation_record(record)["outputMode"] == "json"
    assert record["messages"][0]["content"] == (
        SYSTEM_PROMPTS[agent] + "\n\n" + STRUCTURED_OUTPUT_INSTRUCTION
    )


def test_text_mimicry_eval_template_remains_plain_text() -> None:
    record = _eval(
        "mimicry",
        "style_adaptation_without_drift",
        "Rewrite without changing the supplied facts.",
        {
            "noContentDrift": True,
            "sourceInvariants": ["Harbor review", "09:30", "Halifax"],
            "acceptedGroundedTexts": [
                "Harbor review is at 09:30 in Halifax."
            ],
        },
    )

    assert upgrade_evaluation_record(record)["outputMode"] == "text"
    assert record["messages"][0]["content"] == SYSTEM_PROMPTS["mimicry"]
    assert STRUCTURED_OUTPUT_INSTRUCTION not in record["messages"][0]["content"]


def test_text_eval_rejects_a_json_only_prompt_contract() -> None:
    record = _eval(
        "mimicry",
        "style_adaptation_without_drift",
        "Rewrite without changing the supplied facts.",
        {
            "noContentDrift": True,
            "sourceInvariants": ["Harbor review", "09:30", "Halifax"],
            "acceptedGroundedTexts": [
                "Harbor review is at 09:30 in Halifax."
            ],
        },
    )
    record["messages"][0]["content"] += (
        "\n\n" + STRUCTURED_OUTPUT_INSTRUCTION
    )

    with pytest.raises(ValueError, match="Text-mode evaluation prompt"):
        _bind_evaluation_output_prompt_contract(record)


def test_generic_json_eval_record_is_bound_idempotently() -> None:
    record = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPTS["rem"]},
            {"role": "user", "content": "Classify this repair."},
        ],
        "metrics": [
            {
                "type": "json_field_equals",
                "candidatePaths": ["repairAction"],
                "expected": "add_action_step_samples",
            }
        ],
        "metadata": {
            "agent": "rem",
            "evalType": "generic_repair",
            "mustPass": True,
        },
    }

    once = _bind_evaluation_output_prompt_contract(record)
    twice = _bind_evaluation_output_prompt_contract(once)

    assert twice == once
    assert once["messages"][0]["content"].count(
        STRUCTURED_OUTPUT_INSTRUCTION
    ) == 1
