from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


TOOLS_DIR = Path(__file__).resolve().parents[2]
SCRIPT = TOOLS_DIR / "verify_interactive_model_tool_evidence.py"
SOURCE_REVISION = "a" * 40
BUILD_NUMBER = "204"
WORKING_TREE_DIGEST = "c" * 64
NOW = datetime(2026, 8, 9, 20, 0, tzinfo=timezone.utc)
TOKEN = "corr_v1_" + ("b" * 32)


def _load_verifier():
    spec = importlib.util.spec_from_file_location("verify_interactive_model_tool_evidence", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


VERIFIER = _load_verifier()


def _iso(seconds_from_now: int, *, now: datetime = NOW) -> str:
    return (now + timedelta(seconds=seconds_from_now)).isoformat().replace("+00:00", "Z")


def _redacted(chars: int = 12, digest: str = "c") -> str:
    return f"[redacted sha256={digest * 64} chars={chars}]"


def _summary(label: str, chars: int = 12, digest: str = "d") -> str:
    return f"{label}_chars={chars};sha256={digest * 16}"


def _app() -> dict:
    return {
        "name": "Lumen",
        "bundleIdentifier": "com.27pm.lumenclone",
        "shortVersion": "1.0.0",
        "buildNumber": BUILD_NUMBER,
        "sourceRevision": SOURCE_REVISION,
        "workingTreeDigest": WORKING_TREE_DIGEST,
        "sourceDirtyState": False,
        "executionEnvironment": "physical-iPhone",
    }


def _trace(*, action: bool, now: datetime = NOW) -> dict:
    return {
        "id": "11111111-1111-4111-8111-111111111111" if action else "22222222-2222-4222-8222-222222222222",
        "createdAt": _iso(-90 if action else -60, now=now),
        "event": "modelTurn",
        "slot": "executor",
        "stage": "agent-json-step-0" if action else "agent-json-step-1",
        "scenarioID": _summary("scenarioID"),
        "correlationToken": TOKEN,
        "intent": "alarm",
        "promptPrefix": _summary("prompt"),
        "rawOutputPrefix": _summary("rawOutput"),
        "selectedToolID": "alarm.authorization_status" if action else None,
        "toolArguments": {},
        "allowedToolIDs": ["alarm.authorization_status"],
        "requiresApproval": False,
        "approvalMode": None,
        "parseError": None,
        "emittedFinalInActionTurn": not action,
        "modelIdentifier": _summary("modelIdentifier"),
        "outputTokenCount": 12,
        "runtimePath": "agent-model",
        "streamStarted": True,
        "selectedRuntime": "llama",
        "modelLoaded": True,
        "firstChunkReceived": True,
        "textChunkCount": 2,
        "finalChunkReceived": True,
        "successfulObservationCount": 0 if action else 1,
        "finalizerAccepted": None if action else True,
    }


def valid_package(*, now: datetime = NOW) -> dict:
    return {
            "schemaVersion": "2.0.0",
            "generatedAt": _iso(-30, now=now),
            "exportKind": "testflight-agent-grounding-runtime-export",
            "app": _app(),
            "testFlight": {
                "sourceAction": "E2E Tests > Export Correlated Model/Tool Evidence Package",
                "filePrefix": "lumen-testflight-agent-grounding-redacted-v1",
                "distributionChannel": "testflight_or_unknown",
                "sandboxReceipt": False,
                "appShortVersion": "1.0.0",
                "appBuildNumber": BUILD_NUMBER,
                "liveE2EReportIncluded": True,
                "expectedIngestArgument": "--runtime-audit <exported-testflight-json>",
            },
            "manifestSource": "interactive-model-tool-validation-live-e2e",
            "usedRuntimeFallback": False,
            "runtimeManifestAudit": None,
            "behaviorAudit": None,
            "scenarioResults": [],
            "recentTraces": [_trace(action=True, now=now), _trace(action=False, now=now)],
            "liveE2EReport": {
                "schemaVersion": "1.0.0",
                "generatedAt": _iso(-31, now=now),
                "app": _app(),
                "exportPolicy": {
                    "format": "live-e2e-test-report-json",
                    "sourceLayer": "e2eTestReport",
                    "ownsLiveE2EScenarios": True,
                    "includesDeterministicStaticScenarios": False,
                    "privacy": "privacy-safe",
                    "notes": [],
                },
                "payload": {
                    "id": "33333333-3333-4333-8333-333333333333",
                    "startedAt": _iso(-120, now=now),
                    "finishedAt": _iso(-40, now=now),
                    "passed": 1,
                    "failed": 0,
                    "results": [
                        {
                            "id": "44444444-4444-4444-8444-444444444444",
                            "scenarioID": "interactive-model-tool-alarm-authorization",
                            "kind": "toolGuard",
                            "title": _redacted(),
                            "prompt": _redacted(digest="d"),
                            "expectedIntent": "alarm",
                            "actualIntent": "alarm",
                            "e2eRunID": None,
                            "agentRunID": None,
                            "conversationID": None,
                            "turnID": None,
                            "correlationToken": TOKEN,
                            "requiresAgentRun": True,
                            "evidenceMode": "modelBackedRequired",
                            "passed": True,
                            "failures": [],
                            "finalText": _redacted(digest="e"),
                            "missingHints": [],
                            "rewriteAttempted": False,
                            "rewriteSuccess": False,
                            "events": [
                                {
                                    "id": "55555555-5555-4555-8555-555555555555",
                                    "createdAt": _iso(-70, now=now),
                                    "scenarioID": "interactive-model-tool-alarm-authorization",
                                    "phase": "tool-result",
                                    "message": _redacted(digest="f"),
                                }
                            ],
                            "startedAt": _iso(-110, now=now),
                            "finishedAt": _iso(-45, now=now),
                            "rawFinalPrefix": "",
                            "sanitizedFinalPrefix": "",
                            "rawFinalHadUnsafeLeakage": False,
                            "sanitizedFinalRemovedArtifacts": [],
                            "outputHygieneFailures": [],
                            "metadata": {
                                "privacyRedacted": "true",
                                "expectedToolID": "alarm.authorization_status",
                                "attributableModelToolEvidence": "true",
                                "primaryAgentJSONActionTraceCount": "1",
                                "modelFinalTraceCount": "1",
                                "modelFinalMatchesNativeObservation": "true",
                                "nativeToolObservationStepCount": "1",
                                "nativeToolResultEvidenceCount": "1",
                                "scenarioBankKind": "direct",
                            },
                        }
                    ],
                },
                "correlatedTraceCount": 2,
                "modelBackedCorrelatedTraceCount": 2,
                "modelBackedCorrelatedScenarioCount": 1,
                "deterministicCompatibilityTraceCount": 0,
                "traceSidecarField": "recentTraces",
            },
            "traceSelectedToolAllowedCount": 1,
            "traceParseErrorCount": 0,
            "exportQualityFailures": [],
            "improveLoop": {
                "schemaVersion": "1.0.0",
                "generatedAt": _iso(-30, now=now),
                "acceptedTraining": [],
                "quarantinedSamples": [],
                "regressionTests": [],
                "counters": {},
            },
            "exportPolicy": {
                "format": "testflight-agent-grounding-runtime-json-package",
                "privacy": "privacy-safe",
                "promptPolicy": "redacted",
                "traceLimit": 64,
                "source": "runtime",
                "sourceLayer": "agentGroundingRuntimeAudit",
                "ownsLiveE2EScenarios": False,
                "includesDeterministicStaticScenarios": False,
                "deterministicScenarioPolicy": "omitted",
            },
        }


def _verify(
    package: dict,
    *,
    now: datetime = NOW,
    max_age_seconds: int = VERIFIER.DEFAULT_MAX_AGE_SECONDS,
) -> list[str]:
    return VERIFIER.verify_evidence_package(
        package,
        expected_source_revision=SOURCE_REVISION,
        expected_build_number=BUILD_NUMBER,
        expected_working_tree_digest=WORKING_TREE_DIGEST,
        now=now,
        max_age_seconds=max_age_seconds,
    )


def test_valid_physical_interactive_evidence_package_passes() -> None:
    assert _verify(valid_package()) == []


def _set(path: tuple[object, ...], value: object):
    def mutate(package: dict) -> None:
        target: object = package
        for key in path[:-1]:
            target = target[key]  # type: ignore[index]
        target[path[-1]] = value  # type: ignore[index]

    return mutate


def _append(path: tuple[object, ...], value: object):
    def mutate(package: dict) -> None:
        target: object = package
        for key in path:
            target = target[key]  # type: ignore[index]
        target.append(value)  # type: ignore[union-attr]

    return mutate


RESULT = ("liveE2EReport", "payload", "results", 0)
ACTION = ("recentTraces", 0)
FINAL = ("recentTraces", 1)


@pytest.mark.parametrize(
    ("mutate", "expected_failure"),
    [
        (_set(("app", "bundleIdentifier"), "com.example.fake"), "bundleIdentifier"),
        (_set(("app", "sourceRevision"), "b" * 40), "sourceRevision"),
        (_set(("app", "workingTreeDigest"), "d" * 64), "workingTreeDigest"),
        (_set(("app", "sourceDirtyState"), True), "sourceDirtyState"),
        (_set(("app", "executionEnvironment"), "iOS-simulator"), "executionEnvironment"),
        (_set(("liveE2EReport", "app", "buildNumber"), "205"), "buildNumber"),
        (_set(("liveE2EReport", "app", "sourceDirtyState"), True), "sourceDirtyState"),
        (_set(("liveE2EReport", "app", "executionEnvironment"), "iOS-simulator"), "executionEnvironment"),
        (_set(("testFlight", "sourceAction"), "Agent Grounding > Export"), "sourceAction"),
        (_set(("exportPolicy", "ownsLiveE2EScenarios"), True), "ownsLiveE2EScenarios"),
        (_set(("liveE2EReport", "exportPolicy", "ownsLiveE2EScenarios"), False), "ownsLiveE2EScenarios"),
        (_set(("liveE2EReport", "payload", "passed"), 0), "payload.passed"),
        (
            _set(("liveE2EReport", "payload", "finishedAt"), _iso(-20)),
            "finished after the embedded report export",
        ),
        (_set(RESULT + ("scenarioID",), "some-other-scenario"), "scenarioID"),
        (_set(RESULT + ("evidenceMode",), "policyFirstAllowed"), "evidenceMode"),
        (_set(RESULT + ("passed",), False), ".passed"),
        (_set(RESULT + ("metadata", "expectedToolID"), "alarm.list"), "expectedToolID"),
        (_set(RESULT + ("metadata", "attributableModelToolEvidence"), "false"), "attributableModelToolEvidence"),
        (_set(RESULT + ("metadata", "primaryAgentJSONActionTraceCount"), "2"), "primaryAgentJSONActionTraceCount"),
        (_set(RESULT + ("metadata", "modelFinalTraceCount"), "0"), "modelFinalTraceCount"),
        (_set(RESULT + ("metadata", "modelFinalMatchesNativeObservation"), "false"), "modelFinalMatchesNativeObservation"),
        (_set(RESULT + ("metadata", "nativeToolObservationStepCount"), "0"), "nativeToolObservationStepCount"),
        (_set(RESULT + ("metadata", "nativeToolResultEvidenceCount"), "0"), "nativeToolResultEvidenceCount"),
        (_set(RESULT + ("correlationToken",), "not-opaque"), "correlationToken"),
        (_set(("liveE2EReport", "modelBackedCorrelatedScenarioCount"), 0), "modelBackedCorrelatedScenarioCount"),
        (_set(("liveE2EReport", "modelBackedCorrelatedTraceCount"), 1), "modelBackedCorrelatedTraceCount"),
        (_set(("liveE2EReport", "deterministicCompatibilityTraceCount"), 1), "deterministicCompatibilityTraceCount"),
        (_set(ACTION + ("stage",), "agent-json-step-2"), ".stage"),
        (_set(ACTION + ("runtimePath",), "deterministic-compatibility"), "runtimePath"),
        (_set(ACTION + ("streamStarted",), False), "streamStarted"),
        (_set(ACTION + ("modelLoaded",), False), "modelLoaded"),
        (_set(ACTION + ("textChunkCount",), 0), "textChunkCount"),
        (_set(ACTION + ("selectedToolID",), "alarm.list"), "selectedToolID"),
        (_set(ACTION + ("allowedToolIDs",), ["alarm.authorization_status", "alarm.list"]), "allowedToolIDs"),
        (_set(ACTION + ("requiresApproval",), True), "requiresApproval"),
        (_set(ACTION + ("approvalMode",), "userApproval"), "approvalMode"),
        (_set(ACTION + ("parseError",), "noJSONObject"), "parseError"),
        (_set(ACTION + ("emittedFinalInActionTurn",), True), "emittedFinalInActionTurn"),
        (_set(FINAL + ("selectedToolID",), "alarm.authorization_status"), "selectedToolID"),
        (_set(FINAL + ("emittedFinalInActionTurn",), False), "emittedFinalInActionTurn"),
        (_set(FINAL + ("successfulObservationCount",), 0), "successfulObservationCount"),
        (_set(FINAL + ("finalizerAccepted",), False), "finalizerAccepted"),
        (_set(RESULT + ("events",), []), "tool-result phases"),
        (
            _set(RESULT + ("events", 0, "scenarioID"), "some-other-scenario"),
            "events[0].scenarioID",
        ),
        (
            _set(RESULT + ("events", 0, "createdAt"), "2099-01-01T00:00:00Z"),
            "events[0].createdAt",
        ),
        (
            _set(RESULT + ("events", 0, "id"), "not-a-uuid"),
            "canonical UUIDv4",
        ),
        (
            _set(RESULT + ("events", 0, "message"), _summary("observation")),
            "redacted native observation",
        ),
        (
            _set(RESULT + ("events", 0, "createdAt"), _iso(-100)),
            "native tool-result event predates",
        ),
        (_set(RESULT + ("prompt",), "Check my private alarm details"), "non-redacted free-form text"),
        (_set(ACTION + ("rawOutputPrefix",), '{"action":"raw"}'), "rawOutputPrefix"),
        (_set(ACTION + ("baseModelPath",), "/Users/test/private/model.gguf"), "local filesystem path"),
        (_set(ACTION + ("rawPrivateOutput",), "unredacted output"), "accepted evidence schema"),
        (_set(RESULT + ("e2eRunID",), "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"), "raw correlation identifier"),
        (_set(FINAL + ("correlationToken",), "corr_v1_" + ("e" * 32)), "does not belong"),
        (
            _append(
                ("exportQualityFailures",),
                {
                    "type": "agent_grounding_live_e2e_model_backed_trace_gap",
                    "problem": _summary("problem"),
                },
            ),
            "trace gap",
        ),
        (
            _append(
                ("exportQualityFailures",),
                {
                    "type": "agent_grounding_trace_schema_incomplete",
                    "problem": _summary("problem"),
                },
            ),
            "expected none",
        ),
    ],
)
def test_mutations_fail_closed(mutate, expected_failure: str) -> None:
    package = deepcopy(valid_package())
    mutate(package)

    failures = _verify(package)

    assert failures
    assert any(expected_failure in failure for failure in failures), failures


def test_stale_package_fails_closed() -> None:
    package = valid_package()
    package["generatedAt"] = _iso(-3601)

    failures = _verify(package)

    assert any("stale" in failure for failure in failures)


def test_pre_attestation_package_fails_closed() -> None:
    package = valid_package()
    for app in (package["app"], package["liveE2EReport"]["app"]):
        app.pop("workingTreeDigest")
        app.pop("sourceDirtyState")
        app.pop("executionEnvironment")

    failures = _verify(package)

    assert any("workingTreeDigest" in failure for failure in failures)
    assert any("sourceDirtyState" in failure for failure in failures)
    assert any("executionEnvironment" in failure for failure in failures)


def test_fresh_package_with_stale_embedded_report_fails_closed() -> None:
    package = valid_package()
    package["liveE2EReport"]["payload"]["startedAt"] = _iso(-180)
    package["liveE2EReport"]["payload"]["finishedAt"] = _iso(-121)
    package["liveE2EReport"]["payload"]["results"][0]["startedAt"] = _iso(-170)
    package["liveE2EReport"]["payload"]["results"][0]["finishedAt"] = _iso(-130)
    package["liveE2EReport"]["payload"]["results"][0]["events"][0]["createdAt"] = _iso(-140)
    package["recentTraces"][0]["createdAt"] = _iso(-160)
    package["recentTraces"][1]["createdAt"] = _iso(-135)

    failures = _verify(package, max_age_seconds=120)

    assert any("payload.finishedAt is stale" in failure for failure in failures), failures


def test_excessive_embedded_report_export_assembly_gap_fails_closed() -> None:
    package = valid_package()
    gap = VERIFIER.MAX_EXPORT_ASSEMBLY_SECONDS + 1
    package["liveE2EReport"]["generatedAt"] = _iso(-30 - gap)
    package["liveE2EReport"]["payload"]["finishedAt"] = _iso(-30 - gap - 1)
    package["liveE2EReport"]["payload"]["results"][0]["finishedAt"] = _iso(-30 - gap - 2)
    package["recentTraces"][0]["createdAt"] = _iso(-100)
    package["recentTraces"][1]["createdAt"] = _iso(-95)

    failures = _verify(package)

    assert any("export assembly gap" in failure for failure in failures), failures


def test_cli_accepts_valid_fixture_and_rejects_mutation(tmp_path: Path) -> None:
    current_now = datetime.now(timezone.utc)
    package = valid_package(now=current_now)
    path = tmp_path / "evidence.json"
    path.write_text(json.dumps(package), encoding="utf-8")
    command = [
        sys.executable,
        str(SCRIPT),
        str(path),
        "--expected-source-revision",
        SOURCE_REVISION,
        "--expected-build-number",
        BUILD_NUMBER,
        "--expected-working-tree-digest",
        WORKING_TREE_DIGEST,
    ]

    passed = subprocess.run(command, text=True, capture_output=True, check=False)

    assert passed.returncode == 0, passed.stderr
    assert "verification passed" in passed.stdout

    package["recentTraces"][1]["finalizerAccepted"] = False
    path.write_text(json.dumps(package), encoding="utf-8")
    failed = subprocess.run(command, text=True, capture_output=True, check=False)

    assert failed.returncode == 1
    assert "finalizerAccepted" in failed.stderr


def _receipt_command(package_path: Path, receipt_path: Path) -> list[str]:
    return [
        sys.executable,
        str(SCRIPT),
        str(package_path),
        "--expected-source-revision",
        SOURCE_REVISION,
        "--expected-build-number",
        BUILD_NUMBER,
        "--expected-working-tree-digest",
        WORKING_TREE_DIGEST,
        "--receipt-output",
        str(receipt_path),
    ]


def _parse_iso8601(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def test_cli_writes_hash_bound_receipt_only_after_strict_pass(tmp_path: Path) -> None:
    package = valid_package(now=datetime.now(timezone.utc))
    raw_bytes = json.dumps(
        package,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    package_path = tmp_path / "evidence.json"
    receipt_path = tmp_path / "receipt.json"
    package_path.write_bytes(raw_bytes)
    started_at = datetime.now(timezone.utc)

    completed = subprocess.run(
        _receipt_command(package_path, receipt_path),
        text=True,
        capture_output=True,
        check=False,
    )
    finished_at = datetime.now(timezone.utc)

    assert completed.returncode == 0, completed.stderr
    assert "verification passed" in completed.stdout
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["schemaVersion"] == VERIFIER.VERIFIER_RECEIPT_SCHEMA
    assert receipt["status"] == "verified-at-assessment"
    assert receipt["scope"] == "physical-device-debug-interactive-model-tool"
    assert receipt["clockSource"] == "host-system-utc"
    assert receipt["maxAgeSeconds"] == VERIFIER.DEFAULT_MAX_AGE_SECONDS
    assert receipt["package"] == {
        "sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "byteCount": len(raw_bytes),
        "schemaVersion": "2.0.0",
        "exportKind": "testflight-agent-grounding-runtime-export",
    }
    assert receipt["binding"] == {
        "bundleIdentifier": "com.27pm.lumenclone",
        "sourceRevision": SOURCE_REVISION,
        "buildNumber": BUILD_NUMBER,
        "workingTreeDigest": WORKING_TREE_DIGEST,
        "sourceDirtyState": False,
        "executionEnvironment": "physical-iPhone",
        "scenarioID": "interactive-model-tool-alarm-authorization",
        "toolID": "alarm.authorization_status",
    }
    assert receipt["verifier"] == {
        "name": "verify_interactive_model_tool_evidence",
        "contractVersion": VERIFIER.VERIFIER_CONTRACT_VERSION,
        "sourceSha256": hashlib.sha256(SCRIPT.read_bytes()).hexdigest(),
    }

    verified_at = _parse_iso8601(receipt["verifiedAt"])
    assert started_at <= verified_at <= finished_at
    freshness_anchors = [
        _parse_iso8601(package["generatedAt"]),
        _parse_iso8601(package["liveE2EReport"]["generatedAt"]),
        _parse_iso8601(package["liveE2EReport"]["payload"]["startedAt"]),
        _parse_iso8601(package["liveE2EReport"]["payload"]["finishedAt"]),
        _parse_iso8601(
            package["liveE2EReport"]["payload"]["results"][0]["startedAt"]
        ),
        _parse_iso8601(
            package["liveE2EReport"]["payload"]["results"][0]["finishedAt"]
        ),
        _parse_iso8601(
            package["liveE2EReport"]["payload"]["results"][0]["events"][0]["createdAt"]
        ),
        *[
            _parse_iso8601(trace["createdAt"])
            for trace in package["recentTraces"]
        ],
    ]
    assert _parse_iso8601(receipt["validUntil"]) == min(freshness_anchors) + timedelta(
        seconds=VERIFIER.DEFAULT_MAX_AGE_SECONDS
    )
    receipt_text = receipt_path.read_text(encoding="utf-8")
    assert str(package_path) not in receipt_text
    assert str(tmp_path) not in receipt_text


def test_receipt_hash_tracks_raw_byte_mutation_without_exposing_content(tmp_path: Path) -> None:
    package = valid_package(now=datetime.now(timezone.utc))
    compact_path = tmp_path / "compact.json"
    pretty_path = tmp_path / "pretty.json"
    compact_bytes = json.dumps(package, separators=(",", ":")).encode("utf-8")
    pretty_bytes = (json.dumps(package, indent=2, sort_keys=True) + "\n").encode("utf-8")
    compact_path.write_bytes(compact_bytes)
    pretty_path.write_bytes(pretty_bytes)
    compact_receipt = tmp_path / "compact-receipt.json"
    pretty_receipt = tmp_path / "pretty-receipt.json"

    compact_result = subprocess.run(
        _receipt_command(compact_path, compact_receipt),
        text=True,
        capture_output=True,
        check=False,
    )
    pretty_result = subprocess.run(
        _receipt_command(pretty_path, pretty_receipt),
        text=True,
        capture_output=True,
        check=False,
    )

    assert compact_result.returncode == 0, compact_result.stderr
    assert pretty_result.returncode == 0, pretty_result.stderr
    compact_payload = json.loads(compact_receipt.read_text(encoding="utf-8"))
    pretty_payload = json.loads(pretty_receipt.read_text(encoding="utf-8"))
    assert compact_payload["package"]["sha256"] == hashlib.sha256(compact_bytes).hexdigest()
    assert pretty_payload["package"]["sha256"] == hashlib.sha256(pretty_bytes).hexdigest()
    assert compact_payload["package"]["sha256"] != pretty_payload["package"]["sha256"]
    assert "recentTraces" not in compact_payload
    assert "liveE2EReport" not in compact_payload


def test_cli_does_not_write_receipt_when_strict_verification_fails(tmp_path: Path) -> None:
    package = valid_package(now=datetime.now(timezone.utc))
    package["recentTraces"][1]["finalizerAccepted"] = False
    package_path = tmp_path / "invalid-evidence.json"
    receipt_path = tmp_path / "receipt.json"
    package_path.write_text(json.dumps(package), encoding="utf-8")

    completed = subprocess.run(
        _receipt_command(package_path, receipt_path),
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 1
    assert "finalizerAccepted" in completed.stderr
    assert not receipt_path.exists()


def test_cli_does_not_write_receipt_for_stale_package(tmp_path: Path) -> None:
    package = valid_package(
        now=datetime.now(timezone.utc) - timedelta(
            seconds=VERIFIER.DEFAULT_MAX_AGE_SECONDS + 60
        )
    )
    package_path = tmp_path / "stale-evidence.json"
    receipt_path = tmp_path / "receipt.json"
    package_path.write_text(json.dumps(package), encoding="utf-8")

    completed = subprocess.run(
        _receipt_command(package_path, receipt_path),
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 1
    assert "stale" in completed.stderr
    assert not receipt_path.exists()


def test_fresh_wrapper_cannot_receipt_historical_correlated_execution(
    tmp_path: Path,
) -> None:
    current_now = datetime.now(timezone.utc)
    package = valid_package(now=current_now)
    package["liveE2EReport"]["payload"]["startedAt"] = _iso(-7_200, now=current_now)
    result = package["liveE2EReport"]["payload"]["results"][0]
    result["startedAt"] = _iso(-7_190, now=current_now)
    result["finishedAt"] = _iso(-7_180, now=current_now)
    result["events"][0]["createdAt"] = _iso(-7_185, now=current_now)
    package["recentTraces"][0]["createdAt"] = _iso(-7_188, now=current_now)
    package["recentTraces"][1]["createdAt"] = _iso(-7_184, now=current_now)
    package_path = tmp_path / "fresh-wrapper-old-execution.json"
    receipt_path = tmp_path / "receipt.json"
    package_path.write_text(json.dumps(package), encoding="utf-8")

    completed = subprocess.run(
        _receipt_command(package_path, receipt_path),
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 1
    assert "freshness window expired" in completed.stderr
    assert not receipt_path.exists()


def test_receipt_output_cannot_replace_evidence_package(tmp_path: Path) -> None:
    package = valid_package(now=datetime.now(timezone.utc))
    package_path = tmp_path / "evidence.json"
    original_bytes = json.dumps(package).encode("utf-8")
    package_path.write_bytes(original_bytes)

    completed = subprocess.run(
        _receipt_command(package_path, package_path),
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 1
    assert "must not replace the evidence package" in completed.stderr
    assert package_path.read_bytes() == original_bytes


def test_atomic_receipt_write_cleans_temporary_file_on_replace_failure(
    tmp_path: Path,
) -> None:
    output_directory = tmp_path / "receipt.json"
    output_directory.mkdir()

    with pytest.raises(OSError):
        VERIFIER._write_receipt_atomic(output_directory, {"status": "test"})

    assert list(tmp_path.glob(".receipt.json.*.tmp")) == []


def test_atomic_receipt_write_rejects_expiry_before_replacement(tmp_path: Path) -> None:
    receipt_path = tmp_path / "receipt.json"

    with pytest.raises(ValueError, match="expired before receipt replacement"):
        VERIFIER._write_receipt_atomic(
            receipt_path,
            {"status": "verified-at-assessment"},
            not_after=datetime.now(timezone.utc) - timedelta(seconds=1),
        )

    assert not receipt_path.exists()
    assert list(tmp_path.glob(".receipt.json.*.tmp")) == []
