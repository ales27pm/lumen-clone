import copy
import hashlib
import json
import os
import subprocess
import sys
from uuid import UUID

import pytest
from typer.testing import CliRunner

from lumen_manifest_crawler.cli import app
from lumen_manifest_crawler.dataset.self_prediction_ledger import (
    COVERAGE_SCOPE,
    EXPORT_SCHEMA,
    MAX_EVENTS,
    MAX_INPUT_BYTES,
    LedgerValidationError,
    load_self_prediction_ledger,
    score_self_prediction_ledger,
    write_self_prediction_report,
)


def _export():
    return {"schemaVersion": EXPORT_SCHEMA, "coverageScope": COVERAGE_SCOPE, "events": []}


def _append(export, *, probability=0.8, origin="model", status="success", body=True, cancelled=False, pending=False):
    events = export["events"]
    identifier = str(UUID(int=len(events) + 1)).upper()
    prediction = {
        "kind": "prediction", "seq": len(events) + 1, "id": identifier,
        "runID": str(UUID(int=100)).upper(), "toolID": "device.status", "runtime": "coreAI",
        "probabilityOrigin": origin, "timestamp": "2026-10-25T02:00:00Z",
    }
    if origin == "model":
        prediction["probability"] = probability
    events.append(prediction)
    if not pending:
        events.append({
            "kind": "outcome", "seq": len(events) + 1, "id": identifier,
            "status": status, "toolBodyStarted": body, "cancelled": cancelled,
            "timestamp": "2026-10-25T02:00:01Z",
        })
    return export


def _pair(**kwargs):
    return _append(_export(), **kwargs)


def test_hand_calculated_registry_and_body_brier_ece_are_distinct():
    export = _export()
    _append(export, probability=0.8)
    _append(export, probability=0.3, status="denied", body=False)
    _append(export, probability=0.8, status="failed")
    _append(export, probability=1)
    report = score_self_prediction_ledger(export)
    main = report["registryStatusCalibration"]
    body = report["toolBodyStartedCalibration"]
    # All terminal noncancelled statuses: (.04 + .09 + .64 + 0) / 4.
    assert main["brier"] == pytest.approx(0.1925)
    # Bin .3 contributes .3; bin .8 contributes 2*abs(.8-.5); p=1 contributes0.
    assert main["ece"] == pytest.approx(0.225)
    assert main["baseRate"] == 0.5
    assert main["probabilityRange"] == {"min": 0.3, "max": 1.0}
    assert main["distinctProbabilityCount"] == 3
    assert main["bins"][-1]["count"] == 1
    assert main["bins"][-1]["upperInclusive"] is True
    assert body["count"] == 3
    assert body["brier"] == pytest.approx(0.68 / 3)
    assert body["ece"] == pytest.approx(0.2)
    assert report["coverage"]["scoredToolBodyNotStartedPairCount"] == 1
    assert report["coverage"]["terminalStatusCounts"]["denied"] == 1
    assert report["scoringPolicy"]["target"] == "registry_status_success"


@pytest.mark.parametrize("status", ["failed", "denied", "requiresApproval", "unavailable"])
def test_all_negative_registry_statuses_are_scored_without_claiming_tool_execution(status):
    report = score_self_prediction_ledger(_pair(probability=0.9, status=status, body=False))
    assert report["registryStatusCalibration"]["brier"] == pytest.approx(0.81)
    assert report["toolBodyStartedCalibration"]["count"] == 0
    assert report["toolBodyStartedCalibration"]["brier"] is None


def test_missing_invalid_host_cancelled_and_pending_are_counted_without_fabricating_probabilities():
    export = _export()
    for origin in ["missing", "invalid", "host_repair"]:
        _append(export, origin=origin)
    _append(export, probability=0.9, cancelled=True, status="failed")
    _append(export, probability=0.7, pending=True)
    _append(export, probability=0.6)
    report = score_self_prediction_ledger(export)
    coverage = report["coverage"]
    assert coverage["predictionCount"] == 6
    assert coverage["outcomeCount"] == 5
    assert coverage["missingOutcomeCount"] == 1
    assert coverage["cancelledPairCount"] == 1
    assert coverage["probabilityOriginCounts"] == {"model": 3, "missing": 1, "invalid": 1, "host_repair": 1}
    assert coverage["excludedPredictionCounts"] == {"unresolved": 1, "cancelled": 1, "nonModelProbability": 3}
    assert coverage["scoredPairCount"] == 1
    assert coverage["scoredCoverageOfModelPredictions"] == pytest.approx(1 / 3)
    assert coverage["scoredCoverageOfAllPredictions"] == pytest.approx(1 / 6)
    assert report["registryStatusCalibration"]["brier"] == pytest.approx(0.16)


@pytest.mark.parametrize("export", [_export(), _pair(origin="missing"), _pair(pending=True), _pair(cancelled=True)])
def test_no_eligible_pairs_produces_null_metrics_and_no_pass_claim(export):
    report = score_self_prediction_ledger(export)
    for key in ["registryStatusCalibration", "toolBodyStartedCalibration"]:
        metrics = report[key]
        assert metrics["count"] == 0
        assert metrics["brier"] is None and metrics["ece"] is None and metrics["baseRate"] is None
        assert metrics["probabilityRange"] is None
        assert metrics["unavailableReason"] == "no_eligible_pairs"
        assert len(metrics["bins"]) == 10
    assert "allPassed" not in report
    assert report["ownsLiveE2EScenarios"] is False
    assert not any(value for value in report["evidenceLimits"].values() if type(value) is bool)


def test_constant_perfectly_calibrated_probability_does_not_claim_discrimination():
    export = _pair(probability=0.5)
    _append(export, probability=0.5, status="failed")
    report = score_self_prediction_ledger(export)
    assert report["registryStatusCalibration"]["ece"] == 0
    assert report["registryStatusCalibration"]["brier"] == 0.25
    assert report["registryStatusCalibration"]["probabilityVariationPresent"] is False
    assert report["evidenceLimits"]["discriminationEvaluated"] is False


def test_interleaved_predictions_allow_completion_in_different_order():
    export = _pair()
    _append(export, probability=0.2, status="failed")
    first_prediction, first_outcome, second_prediction, second_outcome = export["events"]
    export["events"] = [first_prediction, second_prediction, second_outcome, first_outcome]
    for sequence, event in enumerate(export["events"], 1):
        event["seq"] = sequence
    assert score_self_prediction_ledger(export)["coverage"]["scoredPairCount"] == 2


def test_wall_clock_rollback_does_not_override_physical_event_order():
    export = _pair()
    export["events"][1]["timestamp"] = "2026-10-25T01:59:59Z"
    assert score_self_prediction_ledger(export)["coverage"]["scoredPairCount"] == 1


@pytest.mark.parametrize("value", [True, False, None, "0.8", -0.1, 1.1, float("nan"), float("inf"), -float("inf"), {}, []])
def test_model_probability_requires_finite_numeric_probability(value):
    with pytest.raises(LedgerValidationError, match="invalid_model_probability"):
        score_self_prediction_ledger(_pair(probability=value))


@pytest.mark.parametrize("origin", ["missing", "invalid", "host_repair"])
@pytest.mark.parametrize("value", [0.7, None])
def test_non_model_origin_cannot_smuggle_a_probability(origin, value):
    export = _pair(origin=origin)
    export["events"][0]["probability"] = value
    with pytest.raises(LedgerValidationError, match="non_model_probability_present"):
        score_self_prediction_ledger(export)


def _tamper(export, case):
    events = export["events"]
    if case == "schema": export["schemaVersion"] = "future"
    elif case == "scope": export["coverageScope"] = "all_runtime_actions"
    elif case == "root_extra": export["authenticated"] = True
    elif case == "root_key_type": export[1] = "payload"
    elif case == "event_extra": events[0]["prompt"] = "PRIVATE_SENTINEL"
    elif case == "event_key_type": events[0][1] = "payload"
    elif case == "kind": events[0]["kind"] = "generated_prediction"
    elif case == "seq_bool": events[0]["seq"] = True
    elif case == "seq_zero": events[0]["seq"] = 0
    elif case == "seq_gap": events[1]["seq"] = 3
    elif case == "seq_float": events[0]["seq"] = 1.0
    elif case == "orphan": events[1]["id"] = str(UUID(int=999))
    elif case == "outcome_first":
        events.reverse()
        for i, event in enumerate(events, 1): event["seq"] = i
    elif case in {"duplicate_prediction", "duplicate_outcome"}:
        event = copy.deepcopy(events[0 if case == "duplicate_prediction" else 1])
        event["seq"] = 3
        event["id"] = event["id"].lower()
        events.append(event)
    elif case == "invalid_origin": events[0]["probabilityOrigin"] = []
    elif case == "model_probability_missing": events[0].pop("probability")
    elif case == "run_id": events[0]["runID"] = "not-an-identifier"
    elif case == "invocation_id": events[0]["id"] = 1
    elif case == "tool_id": events[0]["toolID"] = "device.status\nPRIVATE"
    elif case == "tool_uppercase": events[0]["toolID"] = "Device.Status"
    elif case == "tool_punctuation": events[0]["toolID"] = "device/status"
    elif case == "runtime": events[0]["runtime"] = "x" * 49
    elif case == "runtime_punctuation": events[0]["runtime"] = "coreAI/private"
    elif case == "status": events[1]["status"] = "not_executed"
    elif case == "body_bool": events[1]["toolBodyStarted"] = 1
    elif case == "cancelled_bool": events[1]["cancelled"] = 0
    elif case == "timestamp_naive": events[0]["timestamp"] = "2026-10-25T02:00:00"
    elif case == "timestamp_invalid": events[0]["timestamp"] = "not-an-iso8601-value"
    elif case == "missing_field": events[1].pop("cancelled")
    elif case == "events_not_list": export["events"] = {}
    elif case == "event_limit": export["events"] = [events[0]] * (MAX_EVENTS + 1)
    else: raise AssertionError(case)
    return export


@pytest.mark.parametrize("case", [
    "schema", "scope", "root_extra", "root_key_type", "event_extra", "event_key_type", "kind",
    "seq_bool", "seq_zero", "seq_gap", "seq_float", "orphan", "outcome_first",
    "duplicate_prediction", "duplicate_outcome", "invalid_origin", "model_probability_missing",
    "run_id", "invocation_id", "tool_id", "tool_uppercase", "tool_punctuation", "runtime", "runtime_punctuation", "status", "body_bool", "cancelled_bool",
    "timestamp_naive", "timestamp_invalid", "missing_field", "events_not_list", "event_limit",
])
def test_invalid_exports_fail_closed_instead_of_dropping_bad_rows(case):
    with pytest.raises(LedgerValidationError):
        score_self_prediction_ledger(_tamper(_pair(), case))


@pytest.mark.parametrize("payload", [
    '{"schemaVersion":"self_prediction_ledger.v1","schemaVersion":"future"}',
    '{"x": NaN}', '{"x": Infinity}', '{"x": -Infinity}', '[]', '{',
])
def test_loader_rejects_ambiguous_json(payload, tmp_path):
    path = tmp_path / "ledger.json"
    path.write_text(payload)
    with pytest.raises(LedgerValidationError):
        load_self_prediction_ledger(path)


def test_loader_bounds_input_and_rejects_invalid_utf8(tmp_path):
    path = tmp_path / "ledger.json"
    path.write_bytes(b" " * (MAX_INPUT_BYTES + 1))
    with pytest.raises(LedgerValidationError, match="ledger_input_too_large"):
        load_self_prediction_ledger(path)
    path.write_bytes(b"\xff")
    with pytest.raises(LedgerValidationError, match="invalid_ledger_utf8"):
        load_self_prediction_ledger(path)


def test_loader_reports_numeric_parser_limit_as_a_validation_error(tmp_path):
    path = tmp_path / "ledger.json"
    path.write_text('{"number":' + "9" * 5000 + "}")
    with pytest.raises(LedgerValidationError, match="invalid_ledger_json"):
        load_self_prediction_ledger(path)


def test_cli_emits_small_deterministic_aggregate_json_and_identifies_input_bytes(tmp_path):
    path, output = tmp_path / "ledger.json", tmp_path / "report.json"
    export = _pair()
    path.write_text(json.dumps(export, indent=2))
    result = CliRunner().invoke(app, ["score-self-prediction-ledger", "--ledger", str(path), "--output", str(output)])
    assert result.exit_code == 0, result.output
    report = json.loads(output.read_text())
    assert json.loads(result.output) == report
    assert report["inputSHA256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert report["registryStatusCalibration"]["brier"] == pytest.approx(0.04)
    assert export["events"][0]["runID"] not in output.read_text()
    assert "device.status" not in output.read_text()
    assert len(output.read_bytes()) < 16 * 1024
    second = CliRunner().invoke(app, ["score-self-prediction-ledger", "--ledger", str(path)])
    assert second.exit_code == 0 and second.output == result.output


def test_cli_rejects_invalid_input_without_writing_or_overwriting_report(tmp_path):
    path, output = tmp_path / "ledger.json", tmp_path / "report.json"
    export = _pair()
    export["events"][0]["prompt"] = "PRIVATE_SENTINEL"
    path.write_text(json.dumps(export))
    output.write_text("previous report")
    result = CliRunner().invoke(app, ["score-self-prediction-ledger", "--ledger", str(path), "--output", str(output)])
    assert result.exit_code != 0
    assert "PRIVATE_SENTINEL" not in result.output
    assert output.read_text() == "previous report"


def test_cli_refuses_to_overwrite_the_source_ledger(tmp_path):
    path = tmp_path / "ledger.json"
    original = json.dumps(_pair())
    path.write_text(original)
    result = CliRunner().invoke(app, ["score-self-prediction-ledger", "--ledger", str(path), "--output", str(path)])
    assert result.exit_code != 0
    assert "report_must_not_overwrite_ledger" in result.output
    assert path.read_text() == original


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="POSIX named-pipe regression")
@pytest.mark.parametrize("fifo_target", ["input", "output"])
def test_cli_rejects_unpaired_fifo_without_blocking_or_disclosing_paths(tmp_path, fifo_target):
    source, output = tmp_path / "ledger.json", tmp_path / "report.json"
    original = json.dumps(_pair())
    source.write_text(original)
    fifo = source if fifo_target == "input" else output
    if fifo.exists():
        fifo.unlink()
    os.mkfifo(fifo)
    command = ["score-self-prediction-ledger", "--ledger", str(source), "--output", str(output)]
    # A subprocess timeout bounds a regression in the blocking open itself.
    # No writer/reader is opened: the old implementation hangs indefinitely.
    child = (
        "from typer.testing import CliRunner\n"
        "from lumen_manifest_crawler.cli import app\n"
        f"result = CliRunner().invoke(app, {command!r})\n"
        "assert result.exit_code != 0, result.output\n"
        f"assert {str(tmp_path)!r} not in result.output, result.output\n"
        "print('nonregular_path_rejected')\n"
    )
    completed = subprocess.run([sys.executable, "-c", child], timeout=5, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "nonregular_path_rejected"
    if fifo_target == "input":
        assert not output.exists()
    else:
        assert source.read_text() == original


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="POSIX named-pipe regression")
def test_output_descriptor_type_is_checked_even_when_fifo_has_a_reader(tmp_path):
    source, output = tmp_path / "ledger.json", tmp_path / "report.json"
    source.write_text(json.dumps(_pair()))
    os.mkfifo(output)
    reader = os.open(output, os.O_RDONLY | os.O_NONBLOCK)
    try:
        with pytest.raises(LedgerValidationError, match="report_output_not_regular_file"):
            write_self_prediction_report(output, "report", source_path=source)
        assert os.read(reader, 32) == b"", "No report bytes may be written to a rejected descriptor"
    finally:
        os.close(reader)


@pytest.mark.parametrize("alias_kind", ["symlink", "hardlink"])
def test_report_descriptor_alias_is_rejected_before_source_truncation(tmp_path, alias_kind):
    source, alias = tmp_path / "ledger.json", tmp_path / "alias.json"
    original = json.dumps(_pair())
    source.write_text(original)
    if alias_kind == "symlink":
        alias.symlink_to(source)
    else:
        os.link(source, alias)
    with pytest.raises(LedgerValidationError, match="report_must_not_overwrite_ledger"):
        write_self_prediction_report(alias, "must not truncate", source_path=source)
    assert source.read_text() == original


def test_regular_symlinks_remain_supported_for_input_and_output(tmp_path):
    source, source_link = tmp_path / "ledger.json", tmp_path / "input-link.json"
    report, report_link = tmp_path / "report.json", tmp_path / "output-link.json"
    original = json.dumps(_pair())
    source.write_text(original)
    source_link.symlink_to(source)
    report.write_text("old report" * 5000)
    report_link.symlink_to(report)
    export, digest = load_self_prediction_ledger(source_link)
    assert export == _pair()
    assert digest == hashlib.sha256(source.read_bytes()).hexdigest()
    result = CliRunner().invoke(app, [
        "score-self-prediction-ledger", "--ledger", str(source_link), "--output", str(report_link),
    ])
    assert result.exit_code == 0, result.output
    assert source.read_text() == original
    assert report_link.is_symlink()
    assert json.loads(report.read_text()) == json.loads(result.output)
