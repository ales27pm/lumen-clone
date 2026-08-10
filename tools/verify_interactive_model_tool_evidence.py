#!/usr/bin/env python3
"""Verify the privacy-safe physical interactive model/tool evidence package.

This check intentionally accepts only the narrow package emitted by the DEBUG
physical-device validation flow. It does not ingest generic Agent Grounding or
historical E2E exports, and it fails closed when required attribution is absent.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import re
import sys
from typing import Any, Iterable


EXPECTED_BUNDLE_IDENTIFIER = "com.27pm.lumenclone"
EXPECTED_PACKAGE_SCHEMA = "2.0.0"
EXPECTED_EXPORT_KIND = "testflight-agent-grounding-runtime-export"
EXPECTED_SOURCE_ACTION = "E2E Tests > Export Correlated Model/Tool Evidence Package"
EXPECTED_MANIFEST_SOURCE = "interactive-model-tool-validation-live-e2e"
EXPECTED_PACKAGE_FORMAT = "testflight-agent-grounding-runtime-json-package"
EXPECTED_PACKAGE_SOURCE_LAYER = "agentGroundingRuntimeAudit"
EXPECTED_REPORT_SCHEMA = "1.0.0"
EXPECTED_REPORT_FORMAT = "live-e2e-test-report-json"
EXPECTED_REPORT_SOURCE_LAYER = "e2eTestReport"
EXPECTED_SCENARIO_ID = "interactive-model-tool-alarm-authorization"
EXPECTED_SCENARIO_KIND = "toolGuard"
EXPECTED_INTENT = "alarm"
EXPECTED_EVIDENCE_MODE = "modelBackedRequired"
EXPECTED_TOOL_ID = "alarm.authorization_status"

DEFAULT_MAX_AGE_SECONDS = 60 * 60
MAX_FUTURE_CLOCK_SKEW_SECONDS = 5 * 60
MAX_EXPORT_ASSEMBLY_SECONDS = 60
TRACE_TIME_SLOP_SECONDS = 5

OPAQUE_CORRELATION_TOKEN = re.compile(r"^corr_v1_[0-9a-f]{32}$")
REDACTED_TEXT = re.compile(r"^\[redacted sha256=[0-9a-f]{64} chars=[0-9]+\]$")
HASH_SUMMARY = re.compile(r"^[A-Za-z][A-Za-z0-9]*_chars=[0-9]+;sha256=[0-9a-f]{16}$")
OPAQUE_METADATA_KEY = re.compile(r"^metadata_[0-9a-f]{16}$")
LOCAL_PATH = re.compile(
    r"(?:file://|/(?:Users|home|private|var|Volumes|tmp)/|[A-Za-z]:[\\/])",
    re.IGNORECASE,
)

RAW_CORRELATION_KEYS = {"e2eRunID", "agentRunID", "conversationID", "turnID"}
SENSITIVE_FREE_FORM_KEYS = {
    "accelerationDiagnostic",
    "actual",
    "adapterFailureReason",
    "adapterID",
    "adapterPath",
    "badOutput",
    "baseModelPath",
    "correctedOutput",
    "curriculum",
    "expected",
    "failures",
    "finalText",
    "lesson",
    "message",
    "missingHints",
    "modelIdentifier",
    "outputHygieneFailures",
    "problem",
    "prompt",
    "promptPrefix",
    "rawFinalPrefix",
    "rawOutputPrefix",
    "recommendations",
    "recommendedDatasetRepairs",
    "sanitizedFinalPrefix",
    "selectedAdapter",
    "stopSequences",
    "title",
}
SAFE_SCENARIO_METADATA_KEYS = {
    "actionable",
    "attributableModelToolEvidence",
    "expectedToolID",
    "failureKind",
    "missingAdapterSlots",
    "modelFinalTraceCount",
    "modelFinalMatchesNativeObservation",
    "nativeToolObservationStepCount",
    "nativeToolResultEvidenceCount",
    "primaryAgentJSONActionTraceCount",
    "privacyRedacted",
    "readyArtifactCount",
    "remediationApplied",
    "requiredArtifactCount",
    "requiredSlots",
    "runtimeEvidence",
    "scenarioBankKind",
    "toolFailureCode",
    "trainingSignal",
}
ROOT_KEYS = {
    "schemaVersion", "generatedAt", "exportKind", "app", "testFlight",
    "manifestSource", "usedRuntimeFallback", "runtimeManifestAudit", "behaviorAudit",
    "scenarioResults", "recentTraces", "liveE2EReport", "traceSelectedToolAllowedCount",
    "traceParseErrorCount", "exportQualityFailures", "improveLoop", "exportPolicy",
}
APP_KEYS = {"name", "bundleIdentifier", "shortVersion", "buildNumber", "sourceRevision"}
TEST_FLIGHT_KEYS = {
    "sourceAction", "filePrefix", "distributionChannel", "sandboxReceipt",
    "appShortVersion", "appBuildNumber", "liveE2EReportIncluded", "expectedIngestArgument",
}
PACKAGE_POLICY_KEYS = {
    "format", "privacy", "promptPolicy", "traceLimit", "source", "sourceLayer",
    "ownsLiveE2EScenarios", "includesDeterministicStaticScenarios",
    "deterministicScenarioPolicy",
}
LIVE_REPORT_KEYS = {
    "schemaVersion", "generatedAt", "app", "exportPolicy", "payload",
    "correlatedTraceCount", "modelBackedCorrelatedTraceCount",
    "modelBackedCorrelatedScenarioCount", "deterministicCompatibilityTraceCount",
    "traceSidecarField",
}
LIVE_POLICY_KEYS = {
    "format", "sourceLayer", "ownsLiveE2EScenarios",
    "includesDeterministicStaticScenarios", "privacy", "notes",
}
REPORT_PAYLOAD_KEYS = {"id", "startedAt", "finishedAt", "passed", "failed", "results"}
RESULT_KEYS = {
    "id", "scenarioID", "kind", "title", "prompt", "expectedIntent", "actualIntent",
    "e2eRunID", "agentRunID", "conversationID", "turnID", "correlationToken",
    "requiresAgentRun", "evidenceMode", "passed", "failures", "finalText",
    "missingHints", "rewriteAttempted", "rewriteSuccess", "events", "startedAt",
    "finishedAt", "rawFinalPrefix", "sanitizedFinalPrefix", "rawFinalHadUnsafeLeakage",
    "sanitizedFinalRemovedArtifacts", "outputHygieneFailures", "performanceMatrix", "metadata",
}
EVENT_KEYS = {"id", "createdAt", "scenarioID", "phase", "message"}
PERFORMANCE_MATRIX_KEYS = {
    "aneUtilizationPercent", "eventDensityCPUProxyPercent", "gpuUtilizationPercent",
    "peakRAMMB", "averageRAMMB", "sampleCount", "notes", "accelerationDiagnostics",
}
TRACE_KEYS = {
    "id", "createdAt", "event", "slot", "stage", "scenarioID", "correlationToken",
    "intent", "promptPrefix", "rawOutputPrefix", "selectedToolID", "toolArguments",
    "allowedToolIDs", "requiresApproval", "approvalMode", "parseError",
    "emittedFinalInActionTurn", "modelFamily", "baseModelPath", "adapterID",
    "adapterSlot", "adapterPath", "adapterApplied", "adapterScale", "adapterFailureReason",
    "generationElapsedMs", "firstTokenLatencyMs", "outputTokenCount",
    "estimatedPromptTokenCount", "preFirstTokenMs", "messageBuildMs", "decodeMs",
    "tokensPerSecond", "ensureReadyMs", "adapterActivationMs", "runtimePath",
    "activeAdapterSlot", "maxTokensRequested", "maxTokensEffective", "promptCharCount",
    "accelerationDiagnostic", "accelerationDiagnostics", "emptyOutputReason", "streamStarted",
    "selectedRuntime", "selectedAdapter", "modelIdentifier", "modelLoaded", "stopSequences",
    "temperature", "topP", "cancellationStateBeforeStream", "firstChunkReceived",
    "textChunkCount", "finalChunkReceived", "streamTerminationReason",
    "successfulObservationCount", "finalizerAccepted", "finalizerRejectionReason",
    "finalValidatorAcceptedCandidate", "finalValidatorReplacementSource",
    "finalValidatorRejectionReason", "selfModel",
}
SELF_MODEL_KEYS = {
    "included", "schemaVersion", "mode", "activeSlot", "sourceIDs",
    "runtimeEvidenceSourceLayer", "selectedToolID", "requiresApproval", "approvalMode",
}
IMPROVE_LOOP_KEYS = {
    "schemaVersion", "generatedAt", "acceptedTraining", "quarantinedSamples",
    "regressionTests", "counters",
}
IMPROVE_LOOP_COUNTER_KEYS = {
    "accepted", "quarantined", "regression", "staleTraceRejected",
    "legacyToolNamespaceRejected", "architectureFailureRejected", "resourceFallbackRejected",
}
QUALITY_FAILURE_KEYS = {
    "type", "agent", "expected", "actual", "scenario", "problem", "sourceLayer",
}


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_redacted_text(value: str) -> bool:
    return not value or REDACTED_TEXT.fullmatch(value) is not None or HASH_SUMMARY.fullmatch(value) is not None


def _strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _strings(item)


def _privacy_failures(
    value: Any,
    *,
    location: str = "$",
    parent_key: str | None = None,
) -> list[str]:
    failures: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_location = f"{location}.{key}"
            if key in RAW_CORRELATION_KEYS and child is not None:
                failures.append(f"{child_location} contains a raw correlation identifier")
            if key == "correlationToken" and child not in (None, ""):
                if not isinstance(child, str) or OPAQUE_CORRELATION_TOKEN.fullmatch(child) is None:
                    failures.append(f"{child_location} is not an opaque corr_v1 token")
            if key in SENSITIVE_FREE_FORM_KEYS:
                for text in _strings(child):
                    if not _is_redacted_text(text):
                        failures.append(f"{child_location} contains non-redacted free-form text")
            if key == "metadata" and isinstance(child, dict):
                for metadata_key, metadata_value in child.items():
                    metadata_location = f"{child_location}.{metadata_key}"
                    if metadata_key in SAFE_SCENARIO_METADATA_KEYS:
                        continue
                    if OPAQUE_METADATA_KEY.fullmatch(metadata_key) is None:
                        failures.append(f"{metadata_location} is not a privacy-safe metadata key")
                    for text in _strings(metadata_value):
                        if not _is_redacted_text(text):
                            failures.append(f"{metadata_location} contains non-redacted metadata")
            failures.extend(
                _privacy_failures(child, location=child_location, parent_key=key)
            )
    elif isinstance(value, list):
        for index, child in enumerate(value):
            failures.extend(
                _privacy_failures(child, location=f"{location}[{index}]", parent_key=parent_key)
            )
    elif isinstance(value, str) and LOCAL_PATH.search(value):
        failures.append(f"{location} contains a local filesystem path")
    return failures


def _parse_timestamp(value: Any, location: str, failures: list[str]) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        failures.append(f"{location} must be a non-empty ISO-8601 timestamp")
        return None
    normalized = value.strip()
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        failures.append(f"{location} is not a valid ISO-8601 timestamp")
        return None
    if parsed.tzinfo is None:
        failures.append(f"{location} must include a UTC offset")
        return None
    return parsed.astimezone(timezone.utc)


def _require_mapping(value: Any, location: str, failures: list[str]) -> dict[str, Any]:
    if not isinstance(value, dict):
        failures.append(f"{location} must be a JSON object")
        return {}
    return value


def _require_list(value: Any, location: str, failures: list[str]) -> list[Any]:
    if not isinstance(value, list):
        failures.append(f"{location} must be a JSON array")
        return []
    return value


def _require_exact(value: Any, expected: Any, location: str, failures: list[str]) -> None:
    if type(value) is not type(expected) or value != expected:
        failures.append(f"{location}={value!r}; expected {expected!r}")


def _reject_unknown_keys(
    value: dict[str, Any],
    allowed: set[str],
    location: str,
    failures: list[str],
) -> None:
    for key in value:
        if key not in allowed:
            failures.append(f"{location}.{key} is not part of the accepted evidence schema")


def _require_positive_int(value: Any, location: str, failures: list[str]) -> None:
    if not _is_int(value) or value <= 0:
        failures.append(f"{location}={value!r}; expected a positive integer")


def _validate_app_identity(
    app: dict[str, Any],
    *,
    location: str,
    expected_source_revision: str,
    expected_build_number: str,
    failures: list[str],
) -> None:
    _reject_unknown_keys(app, APP_KEYS, location, failures)
    _require_exact(
        app.get("bundleIdentifier"),
        EXPECTED_BUNDLE_IDENTIFIER,
        f"{location}.bundleIdentifier",
        failures,
    )
    _require_exact(
        app.get("sourceRevision"),
        expected_source_revision,
        f"{location}.sourceRevision",
        failures,
    )
    _require_exact(
        app.get("buildNumber"),
        expected_build_number,
        f"{location}.buildNumber",
        failures,
    )


def _validate_model_trace_common(
    trace: dict[str, Any],
    *,
    location: str,
    failures: list[str],
) -> None:
    _require_exact(trace.get("event"), "modelTurn", f"{location}.event", failures)
    stage = trace.get("stage")
    if not isinstance(stage, str) or not stage.startswith("agent-json"):
        failures.append(f"{location}.stage={stage!r}; expected a primary agent-json stage")
    _require_exact(trace.get("runtimePath"), "agent-model", f"{location}.runtimePath", failures)
    _require_exact(trace.get("parseError"), None, f"{location}.parseError", failures)
    _require_exact(trace.get("streamStarted"), True, f"{location}.streamStarted", failures)
    _require_exact(trace.get("modelLoaded"), True, f"{location}.modelLoaded", failures)
    _require_exact(trace.get("firstChunkReceived"), True, f"{location}.firstChunkReceived", failures)
    _require_positive_int(trace.get("textChunkCount"), f"{location}.textChunkCount", failures)
    _require_exact(trace.get("finalChunkReceived"), True, f"{location}.finalChunkReceived", failures)
    _require_positive_int(trace.get("outputTokenCount"), f"{location}.outputTokenCount", failures)

    selected_runtime = trace.get("selectedRuntime")
    if not isinstance(selected_runtime, str) or not selected_runtime.strip():
        failures.append(f"{location}.selectedRuntime must identify the model runtime")
    model_identifier = trace.get("modelIdentifier")
    if not isinstance(model_identifier, str) or HASH_SUMMARY.fullmatch(model_identifier) is None:
        failures.append(f"{location}.modelIdentifier must be a privacy-safe model summary")
    raw_output = trace.get("rawOutputPrefix")
    if not isinstance(raw_output, str) or HASH_SUMMARY.fullmatch(raw_output) is None:
        failures.append(f"{location}.rawOutputPrefix must be a non-empty privacy-safe output summary")


def _validate_action_trace(
    trace: dict[str, Any],
    *,
    location: str,
    failures: list[str],
) -> None:
    _validate_model_trace_common(trace, location=location, failures=failures)
    _require_exact(trace.get("stage"), "agent-json-step-0", f"{location}.stage", failures)
    _require_exact(trace.get("selectedToolID"), EXPECTED_TOOL_ID, f"{location}.selectedToolID", failures)
    _require_exact(trace.get("allowedToolIDs"), [EXPECTED_TOOL_ID], f"{location}.allowedToolIDs", failures)
    _require_exact(trace.get("toolArguments"), {}, f"{location}.toolArguments", failures)
    _require_exact(trace.get("requiresApproval"), False, f"{location}.requiresApproval", failures)
    if trace.get("approvalMode") not in (None, ""):
        failures.append(f"{location}.approvalMode={trace.get('approvalMode')!r}; expected no approval mode")
    _require_exact(
        trace.get("emittedFinalInActionTurn"),
        False,
        f"{location}.emittedFinalInActionTurn",
        failures,
    )


def _validate_final_trace(
    trace: dict[str, Any],
    *,
    location: str,
    failures: list[str],
) -> None:
    _validate_model_trace_common(trace, location=location, failures=failures)
    _require_exact(trace.get("stage"), "agent-json-step-1", f"{location}.stage", failures)
    _require_exact(trace.get("selectedToolID"), None, f"{location}.selectedToolID", failures)
    _require_exact(trace.get("allowedToolIDs"), [EXPECTED_TOOL_ID], f"{location}.allowedToolIDs", failures)
    _require_exact(trace.get("toolArguments"), {}, f"{location}.toolArguments", failures)
    _require_exact(
        trace.get("emittedFinalInActionTurn"),
        True,
        f"{location}.emittedFinalInActionTurn",
        failures,
    )
    _require_positive_int(
        trace.get("successfulObservationCount"),
        f"{location}.successfulObservationCount",
        failures,
    )
    _require_exact(trace.get("finalizerAccepted"), True, f"{location}.finalizerAccepted", failures)


def verify_evidence_package(
    package: Any,
    *,
    expected_source_revision: str,
    expected_build_number: str,
    now: datetime | None = None,
    max_age_seconds: int = DEFAULT_MAX_AGE_SECONDS,
) -> list[str]:
    """Return every contract failure; an empty list means the package passed."""

    failures: list[str] = []
    if not isinstance(expected_source_revision, str) or not expected_source_revision.strip():
        failures.append("expected source revision must be non-empty")
    if not isinstance(expected_build_number, str) or not expected_build_number.strip():
        failures.append("expected build number must be non-empty")
    if not _is_int(max_age_seconds) or max_age_seconds <= 0:
        failures.append("max_age_seconds must be a positive integer")
        return failures
    if now is None:
        now = datetime.now(timezone.utc)
    elif now.tzinfo is None:
        failures.append("verification clock must include a UTC offset")
        return failures
    else:
        now = now.astimezone(timezone.utc)

    root = _require_mapping(package, "$", failures)
    if not root:
        return failures
    _reject_unknown_keys(root, ROOT_KEYS, "$", failures)

    _require_exact(root.get("schemaVersion"), EXPECTED_PACKAGE_SCHEMA, "$.schemaVersion", failures)
    _require_exact(root.get("exportKind"), EXPECTED_EXPORT_KIND, "$.exportKind", failures)
    _require_exact(root.get("manifestSource"), EXPECTED_MANIFEST_SOURCE, "$.manifestSource", failures)
    _require_exact(root.get("usedRuntimeFallback"), False, "$.usedRuntimeFallback", failures)

    package_app = _require_mapping(root.get("app"), "$.app", failures)
    _validate_app_identity(
        package_app,
        location="$.app",
        expected_source_revision=expected_source_revision,
        expected_build_number=expected_build_number,
        failures=failures,
    )

    test_flight = _require_mapping(root.get("testFlight"), "$.testFlight", failures)
    _reject_unknown_keys(test_flight, TEST_FLIGHT_KEYS, "$.testFlight", failures)
    _require_exact(test_flight.get("sourceAction"), EXPECTED_SOURCE_ACTION, "$.testFlight.sourceAction", failures)
    _require_exact(test_flight.get("liveE2EReportIncluded"), True, "$.testFlight.liveE2EReportIncluded", failures)
    _require_exact(test_flight.get("appBuildNumber"), expected_build_number, "$.testFlight.appBuildNumber", failures)

    package_policy = _require_mapping(root.get("exportPolicy"), "$.exportPolicy", failures)
    _reject_unknown_keys(package_policy, PACKAGE_POLICY_KEYS, "$.exportPolicy", failures)
    _require_exact(package_policy.get("format"), EXPECTED_PACKAGE_FORMAT, "$.exportPolicy.format", failures)
    _require_exact(
        package_policy.get("sourceLayer"),
        EXPECTED_PACKAGE_SOURCE_LAYER,
        "$.exportPolicy.sourceLayer",
        failures,
    )
    _require_exact(
        package_policy.get("ownsLiveE2EScenarios"),
        False,
        "$.exportPolicy.ownsLiveE2EScenarios",
        failures,
    )
    _require_exact(
        package_policy.get("includesDeterministicStaticScenarios"),
        False,
        "$.exportPolicy.includesDeterministicStaticScenarios",
        failures,
    )
    _require_exact(root.get("scenarioResults"), [], "$.scenarioResults", failures)
    _require_exact(root.get("runtimeManifestAudit"), None, "$.runtimeManifestAudit", failures)
    _require_exact(root.get("behaviorAudit"), None, "$.behaviorAudit", failures)

    improve_loop = _require_mapping(root.get("improveLoop"), "$.improveLoop", failures)
    _reject_unknown_keys(improve_loop, IMPROVE_LOOP_KEYS, "$.improveLoop", failures)
    for field in ("acceptedTraining", "quarantinedSamples", "regressionTests"):
        _require_exact(improve_loop.get(field), [], f"$.improveLoop.{field}", failures)
    improve_loop_counters = _require_mapping(
        improve_loop.get("counters"),
        "$.improveLoop.counters",
        failures,
    )
    _reject_unknown_keys(
        improve_loop_counters,
        IMPROVE_LOOP_COUNTER_KEYS,
        "$.improveLoop.counters",
        failures,
    )

    package_generated_at = _parse_timestamp(root.get("generatedAt"), "$.generatedAt", failures)
    if package_generated_at is not None:
        age = (now - package_generated_at).total_seconds()
        if age > max_age_seconds:
            failures.append(
                f"$.generatedAt is stale by {int(age)} seconds; maximum age is {max_age_seconds} seconds"
            )
        if age < -MAX_FUTURE_CLOCK_SKEW_SECONDS:
            failures.append("$.generatedAt is implausibly in the future")

    live = _require_mapping(root.get("liveE2EReport"), "$.liveE2EReport", failures)
    _reject_unknown_keys(live, LIVE_REPORT_KEYS, "$.liveE2EReport", failures)
    _require_exact(live.get("schemaVersion"), EXPECTED_REPORT_SCHEMA, "$.liveE2EReport.schemaVersion", failures)
    _require_exact(live.get("traceSidecarField"), "recentTraces", "$.liveE2EReport.traceSidecarField", failures)
    live_app = _require_mapping(live.get("app"), "$.liveE2EReport.app", failures)
    _validate_app_identity(
        live_app,
        location="$.liveE2EReport.app",
        expected_source_revision=expected_source_revision,
        expected_build_number=expected_build_number,
        failures=failures,
    )
    live_policy = _require_mapping(live.get("exportPolicy"), "$.liveE2EReport.exportPolicy", failures)
    _reject_unknown_keys(
        live_policy,
        LIVE_POLICY_KEYS,
        "$.liveE2EReport.exportPolicy",
        failures,
    )
    _require_exact(live_policy.get("format"), EXPECTED_REPORT_FORMAT, "$.liveE2EReport.exportPolicy.format", failures)
    _require_exact(
        live_policy.get("sourceLayer"),
        EXPECTED_REPORT_SOURCE_LAYER,
        "$.liveE2EReport.exportPolicy.sourceLayer",
        failures,
    )
    _require_exact(
        live_policy.get("ownsLiveE2EScenarios"),
        True,
        "$.liveE2EReport.exportPolicy.ownsLiveE2EScenarios",
        failures,
    )
    _require_exact(
        live_policy.get("includesDeterministicStaticScenarios"),
        False,
        "$.liveE2EReport.exportPolicy.includesDeterministicStaticScenarios",
        failures,
    )

    live_generated_at = _parse_timestamp(
        live.get("generatedAt"),
        "$.liveE2EReport.generatedAt",
        failures,
    )
    if live_generated_at is not None and package_generated_at is not None:
        if live_generated_at > package_generated_at + timedelta(seconds=MAX_EXPORT_ASSEMBLY_SECONDS):
            failures.append("$.liveE2EReport.generatedAt is later than its parent package")
        live_age = (now - live_generated_at).total_seconds()
        if live_age > max_age_seconds:
            failures.append("$.liveE2EReport.generatedAt is stale")
        if live_age < -MAX_FUTURE_CLOCK_SKEW_SECONDS:
            failures.append("$.liveE2EReport.generatedAt is implausibly in the future")

    payload = _require_mapping(live.get("payload"), "$.liveE2EReport.payload", failures)
    _reject_unknown_keys(payload, REPORT_PAYLOAD_KEYS, "$.liveE2EReport.payload", failures)
    _require_exact(payload.get("passed"), 1, "$.liveE2EReport.payload.passed", failures)
    _require_exact(payload.get("failed"), 0, "$.liveE2EReport.payload.failed", failures)
    results = _require_list(payload.get("results"), "$.liveE2EReport.payload.results", failures)
    if len(results) != 1:
        failures.append(f"$.liveE2EReport.payload.results has {len(results)} entries; expected exactly 1")
    result = _require_mapping(results[0], "$.liveE2EReport.payload.results[0]", failures) if results else {}

    report_started_at = _parse_timestamp(payload.get("startedAt"), "$.liveE2EReport.payload.startedAt", failures)
    report_finished_at = _parse_timestamp(payload.get("finishedAt"), "$.liveE2EReport.payload.finishedAt", failures)
    if report_started_at is not None and report_finished_at is not None and report_started_at > report_finished_at:
        failures.append("$.liveE2EReport.payload.startedAt is after finishedAt")
    if live_generated_at is not None and report_finished_at is not None and report_finished_at > live_generated_at:
        failures.append("$.liveE2EReport payload finished after the embedded report export")

    result_started_at: datetime | None = None
    result_finished_at: datetime | None = None
    correlation_token: str | None = None
    if result:
        result_location = "$.liveE2EReport.payload.results[0]"
        _reject_unknown_keys(result, RESULT_KEYS, result_location, failures)
        _require_exact(result.get("scenarioID"), EXPECTED_SCENARIO_ID, f"{result_location}.scenarioID", failures)
        _require_exact(result.get("kind"), EXPECTED_SCENARIO_KIND, f"{result_location}.kind", failures)
        _require_exact(result.get("expectedIntent"), EXPECTED_INTENT, f"{result_location}.expectedIntent", failures)
        _require_exact(result.get("actualIntent"), EXPECTED_INTENT, f"{result_location}.actualIntent", failures)
        _require_exact(result.get("requiresAgentRun"), True, f"{result_location}.requiresAgentRun", failures)
        _require_exact(result.get("evidenceMode"), EXPECTED_EVIDENCE_MODE, f"{result_location}.evidenceMode", failures)
        _require_exact(result.get("passed"), True, f"{result_location}.passed", failures)
        _require_exact(result.get("failures"), [], f"{result_location}.failures", failures)
        _require_exact(result.get("missingHints"), [], f"{result_location}.missingHints", failures)
        _require_exact(
            result.get("outputHygieneFailures"),
            [],
            f"{result_location}.outputHygieneFailures",
            failures,
        )
        _require_exact(result.get("rawFinalHadUnsafeLeakage"), False, f"{result_location}.rawFinalHadUnsafeLeakage", failures)

        correlation_value = result.get("correlationToken")
        if isinstance(correlation_value, str) and OPAQUE_CORRELATION_TOKEN.fullmatch(correlation_value):
            correlation_token = correlation_value
        else:
            failures.append(f"{result_location}.correlationToken must be a non-empty opaque corr_v1 token")

        metadata = _require_mapping(result.get("metadata"), f"{result_location}.metadata", failures)
        expected_metadata = {
            "privacyRedacted": "true",
            "expectedToolID": EXPECTED_TOOL_ID,
            "attributableModelToolEvidence": "true",
            "primaryAgentJSONActionTraceCount": "1",
            "modelFinalTraceCount": "1",
            "modelFinalMatchesNativeObservation": "true",
            "nativeToolObservationStepCount": "1",
            "nativeToolResultEvidenceCount": "1",
        }
        for key, expected in expected_metadata.items():
            _require_exact(metadata.get(key), expected, f"{result_location}.metadata.{key}", failures)

        events = _require_list(result.get("events"), f"{result_location}.events", failures)
        for index, event in enumerate(events):
            if isinstance(event, dict):
                _reject_unknown_keys(event, EVENT_KEYS, f"{result_location}.events[{index}]", failures)
        tool_result_events = [
            event
            for event in events
            if isinstance(event, dict) and event.get("phase") == "tool-result"
        ]
        if len(tool_result_events) != 1:
            failures.append(
                f"{result_location}.events contains {len(tool_result_events)} tool-result phases; expected exactly 1"
            )

        performance_matrix = result.get("performanceMatrix")
        if performance_matrix is not None:
            matrix = _require_mapping(
                performance_matrix,
                f"{result_location}.performanceMatrix",
                failures,
            )
            _reject_unknown_keys(
                matrix,
                PERFORMANCE_MATRIX_KEYS,
                f"{result_location}.performanceMatrix",
                failures,
            )
            _require_exact(
                matrix.get("accelerationDiagnostics"),
                None,
                f"{result_location}.performanceMatrix.accelerationDiagnostics",
                failures,
            )
            notes = _require_list(
                matrix.get("notes"),
                f"{result_location}.performanceMatrix.notes",
                failures,
            )
            for index, note in enumerate(notes):
                if not isinstance(note, str) or not _is_redacted_text(note):
                    failures.append(
                        f"{result_location}.performanceMatrix.notes[{index}] is not privacy-redacted"
                    )

        result_started_at = _parse_timestamp(result.get("startedAt"), f"{result_location}.startedAt", failures)
        result_finished_at = _parse_timestamp(result.get("finishedAt"), f"{result_location}.finishedAt", failures)
        if result_started_at is not None and result_finished_at is not None and result_started_at > result_finished_at:
            failures.append(f"{result_location}.startedAt is after finishedAt")
        if report_started_at is not None and result_started_at is not None and result_started_at < report_started_at:
            failures.append(f"{result_location}.startedAt predates the owning report")
        if report_finished_at is not None and result_finished_at is not None and result_finished_at > report_finished_at:
            failures.append(f"{result_location}.finishedAt exceeds the owning report")

    _require_exact(
        live.get("modelBackedCorrelatedScenarioCount"),
        1,
        "$.liveE2EReport.modelBackedCorrelatedScenarioCount",
        failures,
    )
    model_trace_count = live.get("modelBackedCorrelatedTraceCount")
    if not _is_int(model_trace_count) or model_trace_count < 2:
        failures.append(
            "$.liveE2EReport.modelBackedCorrelatedTraceCount must be an integer >= 2"
        )
    _require_exact(
        live.get("deterministicCompatibilityTraceCount"),
        0,
        "$.liveE2EReport.deterministicCompatibilityTraceCount",
        failures,
    )

    traces = _require_list(root.get("recentTraces"), "$.recentTraces", failures)
    trace_objects: list[tuple[int, dict[str, Any]]] = [
        (index, trace) for index, trace in enumerate(traces) if isinstance(trace, dict)
    ]
    if len(trace_objects) != len(traces):
        failures.append("$.recentTraces must contain only JSON objects")
    for index, trace in trace_objects:
        _reject_unknown_keys(trace, TRACE_KEYS, f"$.recentTraces[{index}]", failures)
        _require_exact(
            trace.get("accelerationDiagnostics"),
            None,
            f"$.recentTraces[{index}].accelerationDiagnostics",
            failures,
        )
        tool_arguments = trace.get("toolArguments")
        if isinstance(tool_arguments, dict):
            for argument_key, argument_value in tool_arguments.items():
                if not isinstance(argument_value, str) or HASH_SUMMARY.fullmatch(argument_value) is None:
                    failures.append(
                        f"$.recentTraces[{index}].toolArguments.{argument_key} is not a privacy-safe value summary"
                    )
        self_model = trace.get("selfModel")
        if isinstance(self_model, dict):
            _reject_unknown_keys(
                self_model,
                SELF_MODEL_KEYS,
                f"$.recentTraces[{index}].selfModel",
                failures,
            )
            for key in (
                "schemaVersion", "mode", "activeSlot", "sourceIDs",
                "runtimeEvidenceSourceLayer", "approvalMode",
            ):
                for text in _strings(self_model.get(key)):
                    if not _is_redacted_text(text):
                        failures.append(
                            f"$.recentTraces[{index}].selfModel.{key} contains non-redacted text"
                        )
    correlated = [
        (index, trace)
        for index, trace in trace_objects
        if correlation_token is not None and trace.get("correlationToken") == correlation_token
    ]
    reported_correlated_count = live.get("correlatedTraceCount")
    if not _is_int(reported_correlated_count) or reported_correlated_count < 2:
        failures.append("$.liveE2EReport.correlatedTraceCount must be an integer >= 2")
    elif reported_correlated_count != len(correlated):
        failures.append(
            "$.liveE2EReport.correlatedTraceCount does not match recentTraces joined by correlationToken"
        )
    if _is_int(model_trace_count) and _is_int(reported_correlated_count) and model_trace_count > reported_correlated_count:
        failures.append("$.liveE2EReport.modelBackedCorrelatedTraceCount exceeds correlatedTraceCount")

    for index, trace in trace_objects:
        token = trace.get("correlationToken")
        if token not in (None, "") and token != correlation_token:
            failures.append(f"$.recentTraces[{index}].correlationToken does not belong to the only report scenario")

    action_candidates = [
        (index, trace)
        for index, trace in correlated
        if trace.get("stage") == "agent-json-step-0" or trace.get("selectedToolID") is not None
    ]
    final_candidates = [
        (index, trace)
        for index, trace in correlated
        if trace.get("stage") == "agent-json-step-1" or trace.get("emittedFinalInActionTurn") is True
    ]
    if len(action_candidates) != 1:
        failures.append(f"$.recentTraces has {len(action_candidates)} correlated action candidates; expected exactly 1")
    else:
        index, action_trace = action_candidates[0]
        _validate_action_trace(action_trace, location=f"$.recentTraces[{index}]", failures=failures)
    if len(final_candidates) != 1:
        failures.append(f"$.recentTraces has {len(final_candidates)} correlated final candidates; expected exactly 1")
    else:
        index, final_trace = final_candidates[0]
        _validate_final_trace(final_trace, location=f"$.recentTraces[{index}]", failures=failures)

    for index, trace in correlated:
        if trace.get("runtimePath") == "deterministic-compatibility":
            failures.append(f"$.recentTraces[{index}] is correlated deterministic-compatibility evidence")
        trace_created_at = _parse_timestamp(trace.get("createdAt"), f"$.recentTraces[{index}].createdAt", failures)
        if trace_created_at is None:
            continue
        if result_started_at is not None and trace_created_at < result_started_at - timedelta(seconds=TRACE_TIME_SLOP_SECONDS):
            failures.append(f"$.recentTraces[{index}].createdAt predates the scenario run")
        if result_finished_at is not None and trace_created_at > result_finished_at + timedelta(seconds=TRACE_TIME_SLOP_SECONDS):
            failures.append(f"$.recentTraces[{index}].createdAt is after the scenario run")

    quality_failures = root.get("exportQualityFailures")
    if quality_failures is not None:
        quality_items = _require_list(quality_failures, "$.exportQualityFailures", failures)
        if quality_items:
            failures.append(
                f"$.exportQualityFailures contains {len(quality_items)} failures; expected none"
            )
        for index, item in enumerate(quality_items):
            if not isinstance(item, dict):
                failures.append(f"$.exportQualityFailures[{index}] must be a JSON object")
                continue
            _reject_unknown_keys(
                item,
                QUALITY_FAILURE_KEYS,
                f"$.exportQualityFailures[{index}]",
                failures,
            )
            failure_type = item.get("type")
            problem = item.get("problem")
            if (isinstance(failure_type, str) and "trace_gap" in failure_type.lower()) or (
                isinstance(problem, str) and "trace gap" in problem.lower()
            ):
                failures.append(f"$.exportQualityFailures[{index}] reports a live E2E trace gap")

    failures.extend(_privacy_failures(root))
    return list(dict.fromkeys(failures))


def verify_evidence_file(
    path: Path,
    *,
    expected_source_revision: str,
    expected_build_number: str,
    now: datetime | None = None,
    max_age_seconds: int = DEFAULT_MAX_AGE_SECONDS,
) -> list[str]:
    try:
        package = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return [f"package does not exist: {path}"]
    except (OSError, UnicodeError) as error:
        return [f"could not read package {path}: {error}"]
    except json.JSONDecodeError as error:
        return [f"invalid package JSON at line {error.lineno}, column {error.colno}: {error.msg}"]
    return verify_evidence_package(
        package,
        expected_source_revision=expected_source_revision,
        expected_build_number=expected_build_number,
        now=now,
        max_age_seconds=max_age_seconds,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fail-closed verification for Lumen's physical interactive model/tool evidence package."
    )
    parser.add_argument("package", type=Path, help="Exported package JSON path.")
    parser.add_argument("--expected-source-revision", required=True, help="Exact LumenGitSHA expected in the app and embedded report.")
    parser.add_argument("--expected-build-number", required=True, help="Exact CFBundleVersion expected in the app and embedded report.")
    parser.add_argument(
        "--max-age-seconds",
        type=int,
        default=DEFAULT_MAX_AGE_SECONDS,
        help=f"Maximum package/report age (default: {DEFAULT_MAX_AGE_SECONDS}).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    package_path = args.package.expanduser().resolve()
    failures = verify_evidence_file(
        package_path,
        expected_source_revision=args.expected_source_revision,
        expected_build_number=args.expected_build_number,
        max_age_seconds=args.max_age_seconds,
    )
    if failures:
        print("Interactive model/tool evidence verification failed:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1
    print("Interactive model/tool evidence verification passed")
    print(f"package={package_path}")
    print(f"bundleIdentifier={EXPECTED_BUNDLE_IDENTIFIER}")
    print(f"sourceRevision={args.expected_source_revision}")
    print(f"buildNumber={args.expected_build_number}")
    print(f"scenario={EXPECTED_SCENARIO_ID}")
    print(f"tool={EXPECTED_TOOL_ID}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
