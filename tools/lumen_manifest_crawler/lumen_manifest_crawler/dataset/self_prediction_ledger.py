"""Offline calibration of host-recorded, prospective registry-status predictions.

Validation checks the supplied export's structure, not device authenticity or
completeness. Registry success is not a judgment of semantic task correctness.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import UUID

EXPORT_SCHEMA = "self_prediction_ledger.v1"
REPORT_SCHEMA = "self_prediction_ledger_score.v1"
SCORING_POLICY = "registry_status_success.v1"
COVERAGE_SCOPE = "structured_tool_invocations_stored_window"
# The host stores <=2 MiB of JSONL; pretty-printed JSON exports need headroom.
MAX_INPUT_BYTES = 4 * 1024 * 1024
MAX_EVENTS = 4096
BIN_COUNT = 10
PROBABILITY_ORIGINS = {"model", "missing", "invalid", "host_repair"}
TERMINAL_STATUSES = {"success", "failed", "denied", "requiresApproval", "unavailable"}


class LedgerValidationError(ValueError):
    """A content-free failure category safe to display in command output."""


def _strict_json(text: str) -> Any:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise LedgerValidationError("duplicate_json_key")
            result[key] = value
        return result

    def constant(_: str) -> Any:
        raise LedgerValidationError("non_finite_json_number")

    try:
        return json.loads(text, object_pairs_hook=pairs, parse_constant=constant)
    except LedgerValidationError:
        raise
    except (ValueError, RecursionError) as error:
        raise LedgerValidationError("invalid_ledger_json") from error


@contextmanager
def _regular_file_descriptor(path: Path, flags: int, failure: str) -> Iterator[tuple[int, os.stat_result]]:
    # Open first, without blocking on pipes, then validate the actual descriptor.
    # A pathname check alone races with replacement by a FIFO/device/symlink.
    # Normal symlinks to regular files keep their existing supported behavior.
    descriptor = os.open(path, flags | os.O_NONBLOCK, 0o600)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise LedgerValidationError(failure)
        yield descriptor, metadata
    finally:
        os.close(descriptor)


def load_self_prediction_ledger(path: Path) -> tuple[dict[str, Any], str]:
    """Read a regular file within the byte limit; reject pipes/devices before IO."""
    with _regular_file_descriptor(path, os.O_RDONLY, "ledger_input_not_regular_file") as (descriptor, _):
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            data = stream.read(MAX_INPUT_BYTES + 1)
    if len(data) > MAX_INPUT_BYTES:
        raise LedgerValidationError("ledger_input_too_large")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise LedgerValidationError("invalid_ledger_utf8") from error
    value = _strict_json(text)
    if not isinstance(value, dict):
        raise LedgerValidationError("ledger_must_be_object")
    return value, hashlib.sha256(data).hexdigest()


def write_self_prediction_report(path: Path, payload: str, *, source_path: Path) -> None:
    """Write only a regular report file, never a pipe/device or the source ledger."""
    # Do not truncate until the opened descriptor has passed both checks. This
    # also protects a source alias substituted after the CLI's early path check.
    with _regular_file_descriptor(path, os.O_WRONLY | os.O_CREAT, "report_output_not_regular_file") as (descriptor, metadata):
        source = source_path.stat()
        if (metadata.st_dev, metadata.st_ino) == (source.st_dev, source.st_ino):
            raise LedgerValidationError("report_must_not_overwrite_ledger")
        os.ftruncate(descriptor, 0)
        with os.fdopen(descriptor, "w", encoding="utf-8", closefd=False) as stream:
            stream.write(payload)


def _fields(value: Any, required: set[str], optional: set[str], context: str) -> dict[str, Any]:
    if type(value) is not dict or any(type(key) is not str for key in value):
        raise LedgerValidationError(f"invalid_{context}_object")
    if not required <= value.keys() or not value.keys() <= required | optional:
        raise LedgerValidationError(f"invalid_{context}_fields")
    return value


def _uuid(value: Any, context: str) -> str:
    if type(value) is not str or len(value) != 36:
        raise LedgerValidationError(f"invalid_{context}")
    try:
        identifier = UUID(value)
    except ValueError as error:
        raise LedgerValidationError(f"invalid_{context}") from error
    if str(identifier) != value.lower():
        raise LedgerValidationError(f"invalid_{context}")
    return str(identifier)


def _timestamp(value: Any) -> datetime:
    if type(value) is not str or not 20 <= len(value) <= 40:
        raise LedgerValidationError("invalid_timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise LedgerValidationError("invalid_timestamp") from error
    if parsed.tzinfo is None or "T" not in value:
        raise LedgerValidationError("timestamp_timezone_required")
    return parsed


def _probability(value: Any) -> float:
    if type(value) not in (float, int) or not 0 <= value <= 1 or not math.isfinite(value):
        raise LedgerValidationError("invalid_model_probability")
    return float(value)


def _token(value: Any, maximum: int, context: str) -> None:
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-"
    if type(value) is not str or not 0 < len(value) <= maximum or any(char not in allowed for char in value):
        raise LedgerValidationError(f"invalid_{context}")


def _validate_export(export: Any) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    _fields(export, {"schemaVersion", "coverageScope", "events"}, set(), "ledger")
    if export["schemaVersion"] != EXPORT_SCHEMA:
        raise LedgerValidationError("unsupported_ledger_schema")
    if export["coverageScope"] != COVERAGE_SCOPE:
        raise LedgerValidationError("unsupported_coverage_scope")
    events = export["events"]
    if type(events) is not list or len(events) > MAX_EVENTS:
        raise LedgerValidationError("invalid_event_count")
    predictions: dict[str, dict[str, Any]] = {}
    outcomes: dict[str, dict[str, Any]] = {}
    for expected_sequence, event in enumerate(events, 1):
        if type(event) is not dict or type(event.get("kind")) is not str:
            raise LedgerValidationError("invalid_event_kind")
        kind = event["kind"]
        common = {"kind", "seq", "id", "timestamp"}
        if kind == "prediction":
            _fields(event, common | {"runID", "toolID", "runtime", "probabilityOrigin"}, {"probability"}, "prediction")
        elif kind == "outcome":
            _fields(event, common | {"status", "toolBodyStarted", "cancelled"}, set(), "outcome")
        else:
            raise LedgerValidationError("invalid_event_kind")
        if type(event["seq"]) is not int or event["seq"] != expected_sequence:
            raise LedgerValidationError("non_contiguous_event_sequence")
        identifier = _uuid(event["id"], "invocation_id")
        _timestamp(event["timestamp"])
        if kind == "prediction":
            if identifier in predictions:
                raise LedgerValidationError("duplicate_prediction")
            _uuid(event["runID"], "run_id")
            _token(event["toolID"], 128, "tool_id")
            if event["toolID"] != event["toolID"].lower():
                raise LedgerValidationError("invalid_tool_id")
            _token(event["runtime"], 48, "runtime")
            origin = event["probabilityOrigin"]
            if type(origin) is not str or origin not in PROBABILITY_ORIGINS:
                raise LedgerValidationError("invalid_probability_origin")
            if origin == "model":
                _probability(event.get("probability"))
            elif "probability" in event:
                raise LedgerValidationError("non_model_probability_present")
            predictions[identifier] = event
        else:
            if identifier not in predictions:
                raise LedgerValidationError("orphan_or_preceding_outcome")
            if identifier in outcomes:
                raise LedgerValidationError("duplicate_outcome")
            if type(event["status"]) is not str or event["status"] not in TERMINAL_STATUSES:
                raise LedgerValidationError("invalid_outcome_status")
            if type(event["toolBodyStarted"]) is not bool or type(event["cancelled"]) is not bool:
                raise LedgerValidationError("invalid_outcome_flags")
            outcomes[identifier] = event
    return predictions, outcomes


def score_self_prediction_ledger(export: dict[str, Any], *, input_sha256: str | None = None) -> dict[str, Any]:
    """Reject ambiguous exports before scoring their stored, eligible pairs.

    ``input_sha256`` identifies input bytes, not an authenticated runtime. Model
    probabilities predict the registry's terminal status, including policy refusals.
    """
    if input_sha256 is not None and (
        type(input_sha256) is not str or len(input_sha256) != 64
        or any(char not in "0123456789abcdef" for char in input_sha256)
    ):
        raise LedgerValidationError("invalid_input_digest")
    predictions, outcomes = _validate_export(export)
    origin_counts = {origin: 0 for origin in sorted(PROBABILITY_ORIGINS)}
    status_counts = {status: 0 for status in sorted(TERMINAL_STATUSES)}
    excluded = {"unresolved": 0, "cancelled": 0, "nonModelProbability": 0}
    pairs: list[tuple[float, int]] = []
    body_pairs: list[tuple[float, int]] = []
    body_started_count = 0
    cancelled_count = 0
    for identifier, prediction in predictions.items():
        origin = prediction["probabilityOrigin"]
        origin_counts[origin] += 1
        outcome = outcomes.get(identifier)
        if outcome is None:
            excluded["unresolved"] += 1
            continue
        status_counts[outcome["status"]] += 1
        body_started_count += outcome["toolBodyStarted"]
        if outcome["cancelled"]:
            cancelled_count += 1
            excluded["cancelled"] += 1
            continue
        if origin != "model":
            excluded["nonModelProbability"] += 1
            continue
        pair = (float(prediction["probability"]), int(outcome["status"] == "success"))
        pairs.append(pair)
        if outcome["toolBodyStarted"]:
            body_pairs.append(pair)
    prediction_count = len(predictions)
    model_count = origin_counts["model"]
    return {
        "schemaVersion": REPORT_SCHEMA,
        "sourceSchemaVersion": EXPORT_SCHEMA,
        "coverageScope": COVERAGE_SCOPE,
        "inputSHA256": input_sha256,
        "structuralValidation": "passed",
        "ownsLiveE2EScenarios": False,
        "scoringPolicy": {
            "version": SCORING_POLICY,
            "target": "registry_status_success",
            "positiveStatus": "success",
            "negativeStatuses": sorted(TERMINAL_STATUSES - {"success"}),
            "probabilityOriginScored": "model",
            "cancelledAndUnresolved": "excluded_and_counted",
            "toolBodySubset": "same_registry_status_target_conditioned_on_toolBodyStarted",
            "binCount": BIN_COUNT,
            "binning": "equal_width_left_closed_right_open_last_closed",
            "brierInterpretation": "combines_calibration_and_discrimination",
            "eceInterpretation": "bin_and_sample_dependent_not_discrimination_evidence",
        },
        "evidenceLimits": {
            "runtimeAuthenticated": False,
            "preActionOrderingAuthenticated": False,
            "runtimeExhaustivenessVerified": False,
            "semanticTaskCorrectnessEvaluated": False,
            "discriminationEvaluated": False,
            "scope": "supplied_stored_window_only",
        },
        "coverage": {
            "storedEventCount": len(export["events"]),
            "predictionCount": prediction_count,
            "outcomeCount": len(outcomes),
            "pairedCount": len(outcomes),
            "missingOutcomeCount": prediction_count - len(outcomes),
            "cancelledPairCount": cancelled_count,
            "toolBodyStartedPairCount": body_started_count,
            "toolBodyNotStartedPairCount": len(outcomes) - body_started_count,
            "probabilityOriginCounts": origin_counts,
            "terminalStatusCounts": status_counts,
            "excludedPredictionCounts": excluded,
            "scoredPairCount": len(pairs),
            "scoredToolBodyStartedPairCount": len(body_pairs),
            "scoredToolBodyNotStartedPairCount": len(pairs) - len(body_pairs),
            "pairCompletionRate": len(outcomes) / prediction_count if prediction_count else None,
            "modelProbabilityCoverage": model_count / prediction_count if prediction_count else None,
            "scoredCoverageOfModelPredictions": len(pairs) / model_count if model_count else None,
            "scoredCoverageOfAllPredictions": len(pairs) / prediction_count if prediction_count else None,
        },
        "registryStatusCalibration": _calibration(pairs),
        "toolBodyStartedCalibration": _calibration(body_pairs),
    }


def _calibration(pairs: list[tuple[float, int]]) -> dict[str, Any]:
    bins: list[list[tuple[float, int]]] = [[] for _ in range(BIN_COUNT)]
    for probability, outcome in pairs:
        bins[min(int(probability * BIN_COUNT), BIN_COUNT - 1)].append((probability, outcome))
    count = len(pairs)
    details: list[dict[str, Any]] = []
    weighted_gap = 0.0
    for index, group in enumerate(bins):
        mean = math.fsum(probability for probability, _ in group) / len(group) if group else None
        rate = sum(outcome for _, outcome in group) / len(group) if group else None
        gap = abs(mean - rate) if group else None
        if group:
            weighted_gap += len(group) * gap
        details.append({
            "lowerBound": index / BIN_COUNT,
            "upperBound": (index + 1) / BIN_COUNT,
            "upperInclusive": index == BIN_COUNT - 1,
            "count": len(group),
            "meanProbability": mean,
            "successRate": rate,
            "absoluteGap": gap,
        })
    probabilities = [probability for probability, _ in pairs]
    successes = sum(outcome for _, outcome in pairs)
    return {
        "count": count,
        "successCount": successes,
        "failureCount": count - successes,
        "baseRate": successes / count if count else None,
        "meanProbability": math.fsum(probabilities) / count if count else None,
        "brier": math.fsum((probability - outcome) ** 2 for probability, outcome in pairs) / count if count else None,
        "ece": weighted_gap / count if count else None,
        "probabilityRange": {"min": min(probabilities), "max": max(probabilities)} if count else None,
        "distinctProbabilityCount": len(set(probabilities)),
        "bothOutcomeClassesPresent": 0 < successes < count,
        "probabilityVariationPresent": len(set(probabilities)) > 1,
        "unavailableReason": None if count else "no_eligible_pairs",
        "bins": details,
    }
