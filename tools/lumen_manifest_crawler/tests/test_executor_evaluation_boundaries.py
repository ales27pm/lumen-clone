"""Public evaluator rejection and aggregation boundaries, using synthetic outputs."""

import json
import math
from copy import deepcopy
from types import SimpleNamespace

import pytest

from lumen_manifest_crawler.dataset.adapter_evaluation import (
    _evaluation_report_aggregates_valid,
    _parse_candidate_json,
    score_evaluation_suite,
)


def record(identity="fixture", *, agent="executor", metrics=None, **extra):
    return {
        "evalID": identity,
        "messages": [{"role": "user", "content": "Synthetic evaluation fixture."}],
        "metrics": metrics or [{"type": "executor_response_contract"}],
        "metadata": {"agent": agent, "critical": True},
        **extra,
    }


def score(candidate, *, contract=None, metrics=None):
    report = score_evaluation_suite(
        [record(metrics=metrics)], {"fixture": candidate},
        tool_contracts={"fixture.tool": contract} if contract is not None else {},
    )
    assert_finite_report(report)
    return report


def assert_finite_report(report):
    json.dumps(report, allow_nan=False)
    assert 0 <= report["weightedScore"] <= 1
    assert all(0 <= value <= 1 for value in report["categoryScores"].values())
    assert _evaluation_report_aggregates_valid(report)


def nested(depth):
    value = 0
    for _ in range(depth):
        value = {"nested": value}
    return value


@pytest.mark.parametrize("number", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("wrap", [lambda value: {"number": value}, lambda value: {"rows": [value]}])
def test_nonfinite_normalized_json_cannot_pass_negative_only_metrics(number, wrap):
    report = score(
        wrap(number), metrics=[{"type": "forbidden_text", "values": ["SYNTHETIC_FORBIDDEN"]}],
    )
    assert report["weightedScore"] == 0.0
    assert report["criticalFailureCount"] == 1
    assert report["caseResults"][0]["metricResults"][0]["reason"] == "non_finite_number"


def test_json_depth_limit_applies_even_to_negative_only_metrics():
    metrics = [{"type": "forbidden_text", "values": ["SYNTHETIC_FORBIDDEN"]}]
    assert score(nested(128), metrics=metrics)["weightedScore"] == 1.0
    report = score(nested(129), metrics=metrics)
    assert report["weightedScore"] == 0.0
    assert report["caseResults"][0]["metricResults"][0]["reason"] == "json_nesting_too_deep"


@pytest.mark.parametrize("candidate", [
    {"final": "\ud800"}, {"final": "Synthetic", 1: "extra"},
    {"final": "Synthetic", "extra": {1}}, {"extra": 10**5000},
])
def test_unserializable_normalized_candidates_raise_fixed_redacted_error(candidate):
    with pytest.raises(ValueError, match="^Candidate outputs cannot be canonically serialized$"):
        score(candidate)


def test_cyclic_normalized_candidate_is_rejected_without_recursive_failure():
    candidate = {}
    candidate["nested"] = candidate
    with pytest.raises(ValueError, match="^Candidate outputs cannot be canonically serialized$"):
        score(candidate)


@pytest.mark.parametrize("weight", [
    True, False, None, "1", 0, -1, float("nan"), float("inf"), float("-inf"), 10**5000,
], ids=lambda value: type(value).__name__)
def test_invalid_evaluation_weight_is_rejected(weight):
    with pytest.raises(ValueError, match="^Evaluation weight must be a finite positive number$"):
        score_evaluation_suite([record(weight=weight)], {"fixture": {"final": "Synthetic result."}})


@pytest.mark.parametrize("weights,expected", [
    ([1e308, 1e308], 0.5), ([1e308, 5e307], 0.666667),
    ([5e-324, 5e-324], 0.5), ([1, 3], 0.25),
])
def test_weighted_score_is_finite_and_reproducible_for_extreme_finite_weights(weights, expected):
    report = score_evaluation_suite(
        [record("pass", weight=weights[0]), record("fail", weight=weights[1])],
        {"pass": {"final": "Synthetic result."}, "fail": {"final": ""}},
    )
    assert_finite_report(report)
    assert math.isfinite(report["weightedScore"])
    assert report["weightedScore"] == expected
    assert _evaluation_report_aggregates_valid(report)


@pytest.mark.parametrize("field,value", [
    ("passedCaseCount", True), ("missingOutputCount", False), ("criticalFailureCount", False),
    ("categoryScores", {"executor_response_contract": True}),
    ("categoryScores", {"executor_response_contract": float("nan")}),
    ("weightedScore", True), ("weightedScore", 10**5000),
], ids=lambda value: type(value).__name__)
def test_aggregate_verifier_rejects_numeric_type_coercion(field, value):
    report = deepcopy(score({"final": "Synthetic result."}))
    report[field] = value
    assert not _evaluation_report_aggregates_valid(report)


def test_aggregate_verifier_cannot_count_a_missing_output_as_a_pass():
    report = deepcopy(score({"final": "Synthetic result."}))
    report["caseResults"][0]["outputPresent"] = False
    report["missingOutputCount"] = 1
    report["evidenceComplete"] = False
    assert not _evaluation_report_aggregates_valid(report)


def test_no_argument_tool_and_default_weight_remain_valid():
    candidate = {"action": {"tool": "fixture.tool", "args": {}}}
    for contract in ({"arguments": []}, SimpleNamespace(arguments=[])):
        report = score(candidate, contract=contract)
        assert report["weightedScore"] == 1.0
        assert report["caseResults"][0]["weight"] == 1.0


@pytest.mark.parametrize("metric_type", ["executor_response_contract", "manifest_tool_call"])
@pytest.mark.parametrize("contract", [
    {}, {"arguments": {}}, {"arguments": [None]},
    {"arguments": [{"name": 1, "type": "string"}]},
    {"arguments": [{"name": "value", "type": ""}]},
    {"arguments": [{"name": "value", "type": "invented_type", "required": False}]},
    {"arguments": [{"name": "value", "type": "string|invented_type", "required": False}]},
    {"arguments": [{"name": "value", "type": "string", "required": "false"}]},
    {"arguments": [{"name": "value", "type": "string", "allowedValues": "one"}]},
    {"arguments": [{"name": "value", "type": "string", "allowedValues": [1]}]},
    {"arguments": [{"name": "value", "type": "string"}] * 2},
])
def test_missing_or_malformed_argument_contract_never_means_no_arguments(contract, metric_type):
    candidate = (
        {"action": {"tool": "fixture.tool", "args": {}}}
        if metric_type == "executor_response_contract"
        else {"tool": "fixture.tool", "arguments": {}}
    )
    report = score(candidate, contract=contract, metrics=[{"type": metric_type}])
    assert report["weightedScore"] == 0.0
    assert report["caseResults"][0]["metricResults"][0]["reason"] in {
        "action_tool_arguments_contract_invalid", "tool_arguments_contract_invalid",
    }


@pytest.mark.parametrize("declared_type,value", [
    ("string", "synthetic"), ("str?", "synthetic"), ("enum", "synthetic"),
    ("Boolean", True), ("integer", 7), ("double", 0.5),
    ("array", []), ("[String]", []), ("dictionary", {}),
    ("nil", None), ("string|number", 0.5), ("null|string?", None),
])
def test_recognized_argument_types_aliases_and_legacy_unions_remain_valid(declared_type, value):
    contract = {"arguments": [{"name": "value", "type": declared_type, "required": False}]}
    for arguments in ({}, {"value": value}):
        report = score({"action": {"tool": "fixture.tool", "args": arguments}}, contract=contract)
        assert report["weightedScore"] == 1.0


def test_manifest_selection_only_metric_does_not_claim_argument_validation():
    report = score(
        {"tool": "fixture.tool"}, contract={},
        metrics=[{"type": "manifest_tool_call", "validateArguments": False}],
    )
    assert report["weightedScore"] == 1.0
    assert report["caseResults"][0]["metricResults"][0]["reason"] == "manifest_tool_selected"


@pytest.mark.parametrize("with_frozen", [False, True])
def test_duplicate_eval_ids_never_reuse_a_single_output_as_two_cases(with_frozen):
    kwargs = {"frozen_evaluation_records": [record()]} if with_frozen else {}
    with pytest.raises(ValueError, match="duplicate evalID"):
        score_evaluation_suite(
            [record(), record()], {"fixture": {"final": "Synthetic result."}}, **kwargs,
        )


def test_allowed_slots_iterable_is_materialized_once_for_all_cases():
    metrics = [{"type": "fixed_slot", "expectedSlot": "executor"}]
    records = [record("one", agent="fleet", metrics=metrics), record("two", agent="fleet", metrics=metrics)]
    outputs = {identity: {"delegateTo": "executor"} for identity in ["one", "two"]}
    report = score_evaluation_suite(records, outputs, allowed_slots=iter(["executor"]))
    assert_finite_report(report)
    assert report["weightedScore"] == 1.0
    assert report["passedCaseCount"] == 2


@pytest.mark.parametrize("mode", [[], {}, False, 0, None, "unsupported"])
def test_invalid_cortex_metric_mode_is_a_failure_not_a_type_error(mode):
    route = {
        "selectedToolID": None, "intent": "probe",
        "reasoningSummary": "No manifest row applies to intent probe.",
        "status": "no_tool_route", "requiresApproval": False, "nextModel": "mouth",
    }
    report = score_evaluation_suite(
        [record(agent="cortex", metrics=[{
            "type": "cortex_route_contract", "mode": mode, "expectedIntent": "probe",
        }])], {"fixture": route},
    )
    assert_finite_report(report)
    assert report["weightedScore"] == 0.0
    assert report["caseResults"][0]["metricResults"][0]["reason"] == "route_contract_mode_invalid"


@pytest.mark.parametrize("metric_type", [[], {}, False, None, 0, "unsupported"])
def test_unsupported_or_wrongly_typed_metric_is_a_sanitized_failure(metric_type):
    report = score({"final": "Synthetic result."}, metrics=[{"type": metric_type}])
    result = report["caseResults"][0]["metricResults"][0]
    assert result["passed"] is False
    assert result["reason"] in {"metric_type_missing", "unsupported_metric_type"}


@pytest.mark.parametrize("mode", [[], {}, False, None, 0, "text"])
def test_wrongly_typed_or_conflicting_output_mode_raises_controlled_value_error(mode):
    with pytest.raises(ValueError, match="^Evaluation outputMode drifted"):
        score_evaluation_suite([record(outputMode=mode)], {"fixture": {"final": "Synthetic result."}})


def test_unknown_agent_cannot_become_complete_evaluation_evidence():
    with pytest.raises(ValueError, match="^Evaluation outputMode cannot be derived for unsupported agent"):
        score_evaluation_suite([record(agent="unsupported")], {"fixture": {"final": "Synthetic result."}})


@pytest.mark.parametrize("text,expected", [
    ('{"value":"\\ud83d\\ude00"}', None),
    ('{"value":"\\ud800"}', "unpaired_unicode_surrogate"),
    ('{"value":NaN}', "invalid_json"),
    ('{"value":1e999}', "non_finite_number"),
    ('{"value":1,"\\u0076alue":2}', "invalid_json"),
])
def test_model_json_parser_preserves_strict_numeric_and_unicode_contract(text, expected):
    _, error = _parse_candidate_json(text)
    assert error == expected


def test_public_json_representation_and_executor_hygiene_remain_strict():
    assert score('{"final":"Synthetic result."}')["weightedScore"] == 0.0
    assert score({"final": "Synthetic result.", "thought": " "})["weightedScore"] == 0.0
    assert score({"final": "Synthetic result.", "extra": True})["weightedScore"] == 0.0
