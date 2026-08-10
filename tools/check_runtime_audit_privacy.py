#!/usr/bin/env python3
"""Fail closed when tracked runtime evidence uses a legacy raw-content format."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AUDIT_ROOT = ROOT / "runtime-audits"
DEFAULT_HISTORY_REVISION = "HEAD"
FORBIDDEN_HISTORY_PREFIXES = ("exports/",)

REDACTED_TEXT = re.compile(r"^\[redacted sha256=[0-9a-f]{64} chars=[0-9]+\]$")
HASH_SUMMARY = re.compile(r"^[A-Za-z][A-Za-z0-9]*_chars=[0-9]+;sha256=[0-9a-f]{16}$")
OPAQUE_METADATA_KEY = re.compile(r"^metadata_[0-9a-f]{16}$")
OPAQUE_TOOL_ARGUMENT_KEY = re.compile(r"^toolArg_[0-9a-f]{16}$")
OPAQUE_CORRELATION_TOKEN = re.compile(r"^corr_(?:v1|hash_v2)_[0-9a-f]{32}$")
EMAIL = re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.IGNORECASE)
SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
ALLOWED_NON_EVIDENCE_FILES = {
    ".DS_Store",
    "AGENTS.md",
    "PRIVACY_QUARANTINE.md",
}
PRIVACY_SAFE_FILE_PREFIXES = (
    "accepted_training-redacted-v1",
    "agent-behavior-traces-redacted-v1",
    "agent-parse-failures-redacted-v1",
    "agent-parse-noise-redacted-v1",
    "e2e-results-redacted-v1",
    "latest-e2e-report-redacted-v1",
    "lumen-agent-runtime-traces-redacted-v1",
    "lumen-live-e2e-report-redacted-v1",
    "lumen-model-behaviour-audit-redacted-v1",
    "lumen-runtime-registry-audit-redacted-v1",
    "lumen-static-scenario-checks-redacted-v1",
    "lumen-testflight-agent-grounding-redacted-v1",
    "persistent-runtime-diagnostics-redacted-v2",
    "quarantined_samples-redacted-v1",
    "regression_tests-redacted-v1",
)
RAW_CORRELATION_KEYS = {
    "e2eRunID",
    "agentRunID",
    "conversationID",
    "turnID",
}
SENSITIVE_FREE_FORM_KEYS = {
    "accelerationDiagnostic",
    "actual",
    "adapterID",
    "adapterFailureReason",
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
    "selectedJSONPrefix",
    "stopSequences",
    "streamedFinalPrefix",
    "streamedThoughtPrefix",
    "systemPromptPrefix",
    "title",
    "userTurnPrefix",
    "modelIdentifier",
}
SAFE_METADATA_KEYS = {
    "actionable",
    "attributableModelToolEvidence",
    "expectedToolID",
    "failureKind",
    "missingAdapterSlots",
    "modelFinalMatchesNativeObservation",
    "modelFinalTraceCount",
    "nativeToolObservationStepCount",
    "nativeToolResultEvidenceCount",
    "privacyRedacted",
    "primaryAgentJSONActionTraceCount",
    "readyArtifactCount",
    "remediationApplied",
    "requiredArtifactCount",
    "requiredSlots",
    "runtimeEvidence",
    "scenarioBankKind",
    "toolFailureCode",
    "trainingSignal",
}
SENSITIVE_SELF_MODEL_KEYS = {
    "activeSlot",
    "approvalMode",
    "mode",
    "runtimeEvidenceSourceLayer",
    "schemaVersion",
    "sourceIDs",
}

# These are the encoded field names owned by the versioned privacy-safe evidence
# formats above. A new field must be classified here before tracked evidence can
# use it; arbitrary dictionary labels can themselves contain private data.
ALLOWED_DOCUMENT_KEYS = frozenset(
    """
    accelerationDiagnostic accelerationDiagnostics accepted acceptedTraining action
    activeAdapterSlot activeCampaignID activeCountsByCategory activeLaunchID activeRunID
    activeScenario activeSlot activeStartedAt actual actualIntent adapterActivationMs
    adapterApplied adapterFailureReason adapterID adapterPath adapterScale adapterSlot agent
    agentGroundingElapsedMs agentRunID allowedToolIDs aneUtilizationPercent app
    appBecameInactiveOrBackgroundDuringRun appBuildNumber appShortVersion appVersion
    approvalMode at authority averageRAMMB badOutput baseModelPath behaviorAudit
    bridgedToolCount buildNumber bundleIdentifier bytes15Minutes bytes1Minute bytes24Hours
    bytesByCategory24Hours campaign campaignID cancellationReasonToken
    cancellationStateBeforeStream canonicalToolID cleanCancellationBeforeTermination code
    completedRunIDs conversationID correctedOutput correlatedTraceCount correlationToken
    counters cpuWatchdog createdAt curriculum decodeMs degradedCategories
    delayBetweenRunsSeconds deterministicCompatibilityTraceCount deterministicScenarioPolicy
    deviceModel didCancel didFallback didUseFastPath diskBytesAfter diskBytesBefore diskWrite
    disposition distributionChannel e2eRunID emittedFinalInActionTurn emptyOutputReason enabled
    ensureReadyMs errorCodeTokens estimatedPromptTokenCount estimatedPromptTokens event
    eventDensityCPUProxyPercent events evidenceMode expected expectedIngestArgument expectedIntent
    expectedToolID exportKind exportPolicy exportQualityFailures exportScope exportedAt failed
    failedCount failureSummaryCharacters failureSummaryToken failures fallbackReasonToken
    filePrefix fileToken finalChunkReceived finalText finalValidatorAcceptedCandidate
    finalValidatorRejectionReason finalValidatorReplacementSource finalizerAccepted
    finalizerRejectionReason finishedAt firstChunkReceived firstTokenLatencyMs format generatedAt
    generationActive generationElapsedMs gpuUtilizationPercent groundingChars
    groundingSectionCount id improveLoop included includesDeterministicStaticScenarios
    inputToolCount intent isPaused isRunning kind lastCancellationReasonToken
    lastCrashResumeStatusToken lastFirstTokenLatencyMs lastPromptFinalChars
    lastRemediationSummaryToken lastUpdatedAt latestScenario lesson liveE2EReport
    liveE2EReportIncluded lowPowerMode manifestSource maxRunsPerScenario maxTokensEffective
    maxTokensRequested memoryCount memoryWarningCount message messageBuildMs messageCharacters
    messageToken metadata metricKitPayloads metrics missingHints mode
    modelBackedCorrelatedScenarioCount modelBackedCorrelatedTraceCount modelFamily
    modelIdentifier modelLoaded name ndjson notes outputHygieneFailures outputTokenCount phase
    ownsLiveE2EScenarios parseError passed passedCount payload payloadBytes peakRAMMB
    performanceMatrix privacy problem prompt promptBodyBytes promptCharCount promptFinalChars
    promptInitialChars promptLatencyClass promptPolicy promptPrefix promptRedactionModeToken
    promptToken quarantined quarantinedSamples rawFinalHadUnsafeLeakage rawFinalPrefix
    rawOutputPrefix realDenied realScenePhase realThermalState recentTraces recommendations
    recommendedDatasetRepairs record recordID records regression regressionTests
    remediationProposals repairSamples requiresAgentRun requiresApproval residentMemoryMB results
    rewriteAttempted rewriteSuccess runContinuously runtimeEvidenceSourceLayer
    runtimeManifestAudit runtimePath sampleCount sampleType sandboxReceipt sanitizedFinalPrefix
    sanitizedFinalRemovedArtifacts scenario scenarioBankKind scenarioID scenarioResults scenarios
    scenePhase schemaVersion score selectedAdapter selectedRuntime selectedToolID selfModel
    severity shortVersion simulatedDenied simulatedScenePhase simulatedThermalState skippedCount
    slot source sourceAction sourceCommit sourceIDs sourceLayer sourceRevision
    sourceRevisionToken sourceTraceID stage staleTraceRejected startedAt state status
    stopSequences streamStarted streamTerminationReason streamingUpdateCount
    successfulObservationCount summaryToken systemName systemVersion temperature testFlight
    textChunkCount thermalState timestamp title tokensPerSecond toolArguments toolCount topP
    totalMemoryMB totalsByCategory traceCount traceLimit traceParseErrorCount
    traceSelectedToolAllowedCount traceSidecarField turnID type uiUpdateCount updatedAt
    usedRuntimeFallback values violationCode violationCount violations
    architectureFailureRejected legacyToolNamespaceRejected resourceFallbackRejected
    """.split()
)
PERSISTENT_STRUCTURAL_METADATA_KEYS = {
    "backgrounddenied", "criticaldenied", "elapsedms", "estimatedprompttokens",
    "estimatedtokens", "finalchars", "finalpromptchars", "firsttokenlatencyms",
    "initialchars", "latencyclass", "lowpowerallowed", "maxsteps", "memorycount",
    "phase", "promptchars", "realdenied", "realexpectedallowed", "sectioncount",
    "seriousdenied", "source", "surface", "targethz", "toolcount",
}
SAFE_CATEGORY_MAP_KEYS = {
    "chatGeneration", "conversation", "diagnostics", "logs", "memory", "modelLoad",
    "modelMetadata", "persistence", "rag", "triggers", "voice",
}


def registered_tool_argument_keys() -> frozenset[str]:
    manifest_path = ROOT / "ios" / "Lumen" / "AgentBehaviorManifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return frozenset()
    keys: set[str] = set()
    for tool in manifest.get("tools", []):
        if not isinstance(tool, dict):
            continue
        for argument in tool.get("arguments", []):
            if isinstance(argument, dict) and isinstance(argument.get("name"), str):
                keys.add(argument["name"])
    return frozenset(keys)


REGISTERED_TOOL_ARGUMENT_KEYS = registered_tool_argument_keys()


def is_legacy_sensitive_name(name: str) -> bool:
    if "lumen-live-e2e-report" in name:
        return "lumen-live-e2e-report-redacted-v1" not in name
    if "latest-e2e-report" in name:
        return "latest-e2e-report-redacted-v1" not in name
    if "e2e-results" in name:
        return "e2e-results-redacted-v1" not in name
    if "agent-behavior-traces" in name:
        return "agent-behavior-traces-redacted-v1" not in name
    if "lumen-testflight-agent-grounding" in name:
        return "lumen-testflight-agent-grounding-redacted-v1" not in name
    return False


def is_privacy_safe_candidate(name: str) -> bool:
    for suffix in (".json", ".jsonl"):
        if not name.endswith(suffix):
            continue
        stem = name[: -len(suffix)]
        return any(
            stem == prefix or stem.startswith(f"{prefix}-")
            for prefix in PRIVACY_SAFE_FILE_PREFIXES
        )
    return False


def is_redacted_text(value: str) -> bool:
    return REDACTED_TEXT.fullmatch(value) is not None or HASH_SUMMARY.fullmatch(value) is not None


def load_documents(path: Path) -> Iterable[Any]:
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if line.strip():
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError as error:
                        raise ValueError(f"invalid JSONL at line {line_number}") from error
        return
    try:
        yield json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError("invalid JSON") from error


def strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from strings(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from strings(item)


def is_allowlisted_document_key(key: str, *, parent_key: str | None) -> bool:
    if key in ALLOWED_DOCUMENT_KEYS:
        return True
    if parent_key == "metadata":
        return key in SAFE_METADATA_KEYS or OPAQUE_METADATA_KEY.fullmatch(key) is not None
    if parent_key == "toolArguments":
        return key in REGISTERED_TOOL_ARGUMENT_KEYS or OPAQUE_TOOL_ARGUMENT_KEY.fullmatch(key) is not None
    if parent_key == "values":
        return key in PERSISTENT_STRUCTURAL_METADATA_KEYS or OPAQUE_METADATA_KEY.fullmatch(key) is not None
    if parent_key in {"activeCountsByCategory", "bytesByCategory24Hours", "totalsByCategory"}:
        return key in SAFE_CATEGORY_MAP_KEYS
    return False


def validate_redacted_document(
    value: Any,
    *,
    location: str = "$",
    parent_key: str | None = None,
) -> list[str]:
    failures: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_location = f"{location}.{key}"
            if not is_allowlisted_document_key(key, parent_key=parent_key):
                failures.append(f"{child_location} exposes a non-allowlisted JSON key")
            if EMAIL.search(key) or SSN.search(key):
                failures.append(f"{child_location} exposes a sensitive identifier in a JSON key")
            if key in RAW_CORRELATION_KEYS and child is not None:
                failures.append(f"{child_location} contains a raw correlation identifier")
            if key == "correlationToken" and isinstance(child, str) and child:
                if OPAQUE_CORRELATION_TOKEN.fullmatch(child) is None:
                    failures.append(f"{child_location} contains a non-opaque correlation token")
            if key in SENSITIVE_FREE_FORM_KEYS:
                for text in strings(child):
                    if text and not is_redacted_text(text):
                        failures.append(f"{child_location} contains non-redacted free-form text")
            if key == "metadata" and isinstance(child, dict):
                for metadata_key, metadata_value in child.items():
                    if metadata_key not in SAFE_METADATA_KEYS and OPAQUE_METADATA_KEY.fullmatch(metadata_key) is None:
                        failures.append(
                            f"{child_location}.{metadata_key} exposes a non-allowlisted metadata key"
                        )
                    if metadata_key not in SAFE_METADATA_KEYS:
                        for text in strings(metadata_value):
                            if text and not is_redacted_text(text):
                                failures.append(
                                    f"{child_location}.{metadata_key} contains non-redacted metadata"
                                )
            if key == "selfModel" and isinstance(child, dict):
                for self_model_key in SENSITIVE_SELF_MODEL_KEYS:
                    for text in strings(child.get(self_model_key)):
                        if text and not is_redacted_text(text):
                            failures.append(
                                f"{child_location}.{self_model_key} contains non-redacted self-model text"
                            )
            failures.extend(
                validate_redacted_document(child, location=child_location, parent_key=key)
            )
    elif isinstance(value, str):
        if EMAIL.search(value) or SSN.search(value):
            failures.append(f"{location} contains a raw sensitive identifier")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            failures.extend(
                validate_redacted_document(child, location=f"{location}[{index}]", parent_key=parent_key)
            )
    return failures


def check_runtime_audits(audit_root: Path) -> list[str]:
    failures: list[str] = []
    if not audit_root.exists():
        return [f"runtime audit root does not exist: {audit_root}"]

    for path in sorted(audit_root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(audit_root)
        if relative.as_posix() in ALLOWED_NON_EVIDENCE_FILES:
            continue
        if is_legacy_sensitive_name(path.name):
            failures.append(f"{relative}: legacy raw-content runtime artifact name is forbidden")
            continue
        if not is_privacy_safe_candidate(path.name):
            failures.append(
                f"{relative}: unrecognized or unversioned runtime artifact is forbidden"
            )
            continue
        try:
            documents = list(load_documents(path))
        except (OSError, UnicodeError, ValueError) as error:
            failures.append(f"{relative}: {error}")
            continue
        for index, document in enumerate(documents):
            for failure in validate_redacted_document(document):
                failures.append(f"{relative} document {index + 1}: {failure}")
    return failures


def check_git_history_privacy(
    repository_root: Path,
    *,
    revision: str = DEFAULT_HISTORY_REVISION,
) -> list[str]:
    """Reject private export paths reachable from the release history.

    ``exports/`` is intentionally ignored and machine-local. Checking only the
    current tree is insufficient because a deleted export remains downloadable
    from every reachable commit until history is purged.
    """

    try:
        result = subprocess.run(
            [
                "git",
                "-C",
                str(repository_root),
                "rev-list",
                "--objects",
                revision,
                "--",
                *FORBIDDEN_HISTORY_PREFIXES,
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as error:
        return [f"could not inspect Git history for private exports: {error}"]

    if result.returncode != 0:
        detail = result.stderr.strip() or f"git exited with status {result.returncode}"
        return [f"could not inspect Git history revision {revision!r}: {detail}"]

    forbidden_paths: set[str] = set()
    for line in result.stdout.splitlines():
        _object_id, separator, path = line.partition(" ")
        path = path.strip()
        if not separator or not path:
            continue
        # ``rev-list --objects`` also emits the directory tree object as the
        # bare path ``exports``. Only descendants are private export files.
        if any(path.startswith(prefix) for prefix in FORBIDDEN_HISTORY_PREFIXES):
            forbidden_paths.add(path)

    return [
        f"Git history revision {revision!r} exposes forbidden private export path: {path}"
        for path in sorted(forbidden_paths)
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-root", type=Path, default=DEFAULT_AUDIT_ROOT)
    parser.add_argument("--repository-root", type=Path, default=ROOT)
    parser.add_argument("--history-revision", default=DEFAULT_HISTORY_REVISION)
    args = parser.parse_args()

    failures = check_runtime_audits(args.audit_root)
    failures.extend(
        check_git_history_privacy(
            args.repository_root,
            revision=args.history_revision,
        )
    )
    if failures:
        print("Runtime audit privacy validation failed:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1
    print("Runtime audit privacy validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
