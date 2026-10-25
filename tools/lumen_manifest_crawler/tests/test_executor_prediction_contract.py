"""Training validation stays compatible without inventing calibration labels."""

import json
from types import SimpleNamespace

import pytest

from lumen_manifest_crawler.validators import _validate_executor_json_contract


def validate(payload):
    failures = []
    _validate_executor_json_contract(
        assistant=json.dumps(payload),
        tools_by_id={"calendar.today": SimpleNamespace(arguments=[])},
        failures=failures,
        path="fixture",
    )
    return {failure.code for failure in failures}


@pytest.mark.parametrize("probability", [0, 1, 0.72])
def test_optional_forecast_accepts_probability(probability):
    assert not validate({
        "action": {"tool": "calendar.today", "args": {}},
        "prediction": {"successProbability": probability},
    })


def test_historical_action_remains_valid_without_invented_probability():
    assert not validate({"action": {"tool": "calendar.today", "args": {}}})


@pytest.mark.parametrize("prediction", [
    None, {}, {"successProbability": True}, {"successProbability": "0.8"},
    {"successProbability": -0.1}, {"successProbability": 1.1},
    {"successProbability": float("nan")}, {"successProbability": float("inf")},
    {"successProbability": 0.9, "approved": True},
])
def test_forecast_metadata_must_be_unambiguous(prediction):
    assert "executor_prediction_invalid" in validate({
        "action": {"tool": "calendar.today", "args": {}}, "prediction": prediction,
    })


def test_prediction_never_licenses_unknown_arguments_or_final_metadata():
    assert "executor_extra_arguments" in validate({
        "action": {"tool": "calendar.today", "args": {"approved": True}},
        "prediction": {"successProbability": 1},
    })
    assert "executor_response_shape_invalid" in validate({
        "final": "Done", "prediction": {"successProbability": 1},
    })


@pytest.mark.parametrize("metadata", [
    r'"prediction":{"successProbability":0.2,"successProbability":0.9}',
    r'"prediction":{},"\u0070rediction":{"successProbability":0.9}',
    r'"prediction":{"successProbability":0.2,"success\u0050robability":0.9}',
])
def test_ambiguous_forecasts_cannot_enter_training(metadata):
    failures = []
    _validate_executor_json_contract(
        assistant='{"action":{"tool":"calendar.today","args":{}},' + metadata + '}',
        tools_by_id={"calendar.today": SimpleNamespace(arguments=[])},
        failures=failures,
        path="fixture",
    )
    assert {failure.code for failure in failures} == {"executor_duplicate_json_key"}
