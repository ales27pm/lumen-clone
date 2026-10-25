"""Executor training accepts finite JSON numbers in SFT and preferred DPO."""

import json
from types import SimpleNamespace

import pytest

from lumen_manifest_crawler.validators import (
    _executor_argument_type_is_valid,
    _validate_agent_sft_records,
    _validate_executor_dpo_records,
)


def action(value_json):
    return '{"action":{"tool":"fixture.numeric","args":{"value":' + value_json + '}}}'


def validate_record(value_json, declared_type, record_kind):
    tools = {
        "fixture.numeric": SimpleNamespace(arguments=[
            SimpleNamespace(name="value", type=declared_type, required=True, allowedValues=[]),
        ]),
    }
    failures = []
    assistant = action(value_json)
    if record_kind == "sft":
        _validate_agent_sft_records(
            agent="executor",
            records=[{
                "messages": [
                    {"role": "system", "content": "Return a tool call."},
                    {"role": "user", "content": "Use the numeric fixture."},
                    {"role": "assistant", "content": assistant},
                ],
                "metadata": {"agent": "executor", "toolIDs": ["fixture.numeric"]},
            }],
            known_tools=set(tools),
            tools_by_id=tools,
            tool_arg_map={"fixture.numeric": {"value"}},
            forbidden=set(),
            failures=failures,
        )
    else:
        _validate_executor_dpo_records(
            # Negative examples must remain eligible for the rejected side.
            records=[{"chosen": {"content": assistant}, "rejected": {"content": action("NaN")}}],
            tools_by_id=tools,
            failures=failures,
        )
    return failures


@pytest.mark.parametrize("record_kind", ["sft", "dpo"])
@pytest.mark.parametrize("declared_type", ["number", "float", "double"])
@pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity", "1e400", "-1e400"])
def test_nonfinite_numeric_arguments_cannot_enter_training(record_kind, declared_type, token):
    failures = validate_record(token, declared_type, record_kind)
    assert [failure.code for failure in failures] == ["executor_non_json_output"]
    expected_path = "fine_tuning.executor.sft.0" if record_kind == "sft" else "fine_tuning.executor.dpo.0.chosen"
    assert failures[0].path == expected_path


@pytest.mark.parametrize("record_kind", ["sft", "dpo"])
@pytest.mark.parametrize("declared_type,template", [
    ("object", '{"nested":[{"value":%s}]}'),
    ("array", '[{"nested":[%s]}]'),
])
@pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity", "1e400", "-1e400"])
def test_nested_nonfinite_values_are_rejected(record_kind, declared_type, template, token):
    failures = validate_record(template % token, declared_type, record_kind)
    assert [failure.code for failure in failures] == ["executor_non_json_output"]


@pytest.mark.parametrize("record_kind", ["sft", "dpo"])
@pytest.mark.parametrize("declared_type", ["number", "float", "double"])
@pytest.mark.parametrize("token", [
    "0", "-7", "0.125", "-0.0", "5e-324", "-5e-324",
    "1.7976931348623157e308", "-1.7976931348623157e308",
    str(10**400),
])
def test_finite_numbers_and_large_integers_remain_valid(record_kind, declared_type, token):
    assert validate_record(token, declared_type, record_kind) == []


@pytest.mark.parametrize("record_kind", ["sft", "dpo"])
@pytest.mark.parametrize("declared_type,value", [
    ("string", "NaN Infinity -Infinity 1e400"),
    ("object", {"nested": ["NaN", "Infinity", "-1e400", 0.125]}),
    ("array", [{"text": "Infinity", "number": 1.5}]),
])
def test_quoted_text_and_finite_nested_values_are_not_numeric_tokens(record_kind, declared_type, value):
    assert validate_record(json.dumps(value, allow_nan=False), declared_type, record_kind) == []


@pytest.mark.parametrize("declared_type", ["number", "float", "double", "int", "integer"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), True, "Infinity"])
def test_argument_type_check_rejects_nonfinite_or_non_numeric_values(declared_type, value):
    assert not _executor_argument_type_is_valid(value, declared_type)


@pytest.mark.parametrize("declared_type", ["number", "float", "double", "int", "integer"])
def test_argument_type_check_does_not_convert_large_integers_to_float(declared_type):
    assert _executor_argument_type_is_valid(10**400, declared_type)
