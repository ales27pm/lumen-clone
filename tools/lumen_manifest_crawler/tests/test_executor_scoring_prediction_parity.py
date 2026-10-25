"""Optional native forecasts keep the same SFT/DPO and offline acceptance rules."""
import json
from types import SimpleNamespace

import pytest

from lumen_manifest_crawler.dataset.adapter_evaluation import _score_metric, score_evaluation_suite
from lumen_manifest_crawler.validators import (
    _validate_agent_sft_records,
    _validate_executor_dpo_records,
)


TOOL_ID = "web.search"
ARGUMENT = {"name": "query", "type": "string", "required": True, "allowedValues": []}
SCORE_TOOLS = {TOOL_ID: {"arguments": [ARGUMENT]}}
TRAIN_TOOLS = {TOOL_ID: SimpleNamespace(arguments=[SimpleNamespace(**ARGUMENT)])}


def action():
    return {"action": {"tool": TOOL_ID, "args": {"query": "synthetic fixture"}}}


def score(payload, *, json_text=False):
    record = {
        "evalID": "executor-prediction-parity-fixture",
        "outputMode": "json",
        "messages": [{"role": "user", "content": "Use the synthetic fixture."}],
        "metrics": [{"type": "executor_response_contract"}],
        "metadata": {"agent": "executor", "critical": True},
    }
    return score_evaluation_suite(
        [record], {record["evalID"]: json.dumps(payload) if json_text else payload},
        tool_contracts=SCORE_TOOLS, agent="executor"
    )


def score_model_json_text(payload):
    return _score_metric(
        {"type": "executor_response_contract"}, json.dumps(payload),
        tool_contracts=SCORE_TOOLS, allowed_slots=set(), has_output=True,
    )


def training_failures(payload, kind):
    assistant = json.dumps(payload)
    failures = []
    if kind == "sft":
        _validate_agent_sft_records(
            agent="executor",
            records=[{
                "messages": [
                    {"role": "system", "content": "Return the native action envelope."},
                    {"role": "user", "content": "Use the synthetic fixture."},
                    {"role": "assistant", "content": assistant},
                ],
                "metadata": {"agent": "executor", "toolIDs": [TOOL_ID]},
            }],
            known_tools={TOOL_ID}, tools_by_id=TRAIN_TOOLS,
            tool_arg_map={TOOL_ID: {"query"}}, forbidden=set(), failures=failures,
        )
    else:
        _validate_executor_dpo_records(
            records=[{"chosen": {"content": assistant}}],
            tools_by_id=TRAIN_TOOLS, failures=failures,
        )
    return failures


@pytest.mark.parametrize("probability", [0, 1, 0.72, 0.0, 1.0, -0.0, 5e-324])
@pytest.mark.parametrize("with_thought", [False, True])
def test_native_prediction_is_accepted_by_training_and_public_scoring(probability, with_thought):
    payload = action()
    payload["prediction"] = {"successProbability": probability}
    if with_thought:
        payload["thought"] = "Check the tool result."
    for kind in ["sft", "dpo"]:
        assert training_failures(payload, kind) == []
    report = score(payload)
    assert report["weightedScore"] == 1.0
    assert report["criticalFailureCount"] == 0
    assert report["caseResults"][0]["metricResults"][0]["reason"] == "native_action_valid"
    assert score_model_json_text(payload)["passed"] is True


@pytest.mark.parametrize("with_thought", [False, True])
def test_absent_prediction_remains_optional_for_public_scoring(with_thought):
    payload = action()
    if with_thought:
        payload["thought"] = "Check the tool result."
    assert "prediction" not in payload
    assert score(payload)["weightedScore"] == 1.0
    assert score_model_json_text(payload)["passed"] is True
    for kind in ["sft", "dpo"]:
        assert training_failures(payload, kind) == []


@pytest.mark.parametrize("prediction", [
    None, {}, [], 0.72, {"successProbability": True}, {"successProbability": "0.72"},
    {"successProbability": None}, {"successProbability": -0.01}, {"successProbability": 1.01},
    {"successProbability": 10**400}, {"successProbability": float("nan")},
    {"successProbability": float("inf")}, {"successProbability": float("-inf")},
    {"successProbability": 0.72, "approved": True}, {"confidence": 0.72},
])
def test_invalid_prediction_is_rejected_by_training_and_public_scoring(prediction):
    payload = action()
    payload["prediction"] = prediction
    report = score(payload)
    assert report["weightedScore"] == 0.0
    assert report["criticalFailureCount"] == 1
    assert score_model_json_text(payload)["passed"] is False
    for kind in ["sft", "dpo"]:
        assert training_failures(payload, kind)


def test_prediction_does_not_license_action_extras_or_unknown_arguments():
    for location in ["top", "action", "arguments"]:
        payload = action()
        payload["prediction"] = {"successProbability": 1}
        target = payload if location == "top" else payload["action"] if location == "action" else payload["action"]["args"]
        target["approved"] = True
        assert score(payload)["weightedScore"] == 0.0


def test_prediction_remains_forbidden_on_final_turns():
    payload = {"final": "Synthetic observation is ready."}
    assert score(payload)["weightedScore"] == 1.0
    payload["prediction"] = {"successProbability": 1}
    report = score(payload)
    assert report["weightedScore"] == 0.0
    assert report["caseResults"][0]["metricResults"][0]["reason"] == "final_top_level_shape_invalid"
    assert score_model_json_text(payload)["passed"] is False


@pytest.mark.parametrize("prediction", ["absent", {"successProbability": 0.72}, {"successProbability": True}])
def test_public_suite_still_requires_normalized_json_mapping(prediction):
    payload = action()
    if prediction != "absent":
        payload["prediction"] = prediction
    report = score(payload, json_text=True)
    assert report["weightedScore"] == 0.0
    assert report["caseResults"][0]["metricResults"][0]["reason"] == "candidate_output_mode_mismatch"
