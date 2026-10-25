"""Split and label checks compare the prompts the training consumer receives."""

import json

import pytest

from test_public_training_output_contract import VALID_ACTION, base, failures_for


@pytest.mark.parametrize("left,right", [(False, 0), (False, None), (0, ""), ([], {}), (None, "")])
def test_distinct_structured_user_contents_are_not_false_duplicates(left, right):
    datasets = base()
    train = datasets["executor"].train_sft[0]
    validation = datasets["executor"].val_sft[0]
    train["messages"][1]["content"] = left
    validation["messages"][1]["content"] = right
    validation["messages"][2]["content"] = VALID_ACTION
    assert failures_for(datasets) == []


@pytest.mark.parametrize("content", [False, 0, None, [], {"b": 2, "a": 1}])
def test_rendered_structured_content_cannot_hide_conflicting_labels(content):
    datasets = base()
    datasets["executor"].train_sft[0]["messages"][1]["content"] = content
    datasets["executor"].val_sft[0]["messages"][1]["content"] = json.dumps(
        content, ensure_ascii=False, sort_keys=True,
    )
    assert "conflicting_sft_prompt_labels" in {failure.code for failure in failures_for(datasets)}


def test_earlier_assistant_context_distinguishes_later_prompts():
    datasets = base()
    train = datasets["executor"].train_sft[0]
    validation = datasets["executor"].val_sft[0]
    train["messages"] += [
        {"role": "user", "content": "Continue after the earlier result."},
        {"role": "assistant", "content": '{"final":"First follow-up."}'},
    ]
    validation["messages"] += [
        {"role": "user", "content": "Continue after the earlier result."},
        {"role": "assistant", "content": '{"final":"Second follow-up."}'},
    ]
    assert failures_for(datasets) == []
