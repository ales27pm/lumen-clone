from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from lumen_manifest_crawler import improvement_loop as loop
from lumen_manifest_crawler.dataset.runtime_ingest import load_runtime_audit_reports
from lumen_manifest_crawler.improvement_loop import (
    AgentImprovementLoopConfig,
    _assess_runtime_audit_proof,
    _annotate_runtime_reports_for_training,
    _build_testflight_plan,
    _build_gap_report,
    _ingestion_runtime_reports,
    _verify_runtime_audit_at_host_now,
)


CURRENT_REVISION = "e84fd7d41021fe93136637687f6e9094d39bc0f3"
HISTORICAL_REVISION = "95174d975da515cf8625212592721cd0baa7bfa5"
BUILD_NUMBER = "20260810031810"


class _Manifest:
    def output_dict(self) -> dict[str, object]:
        return {"sourceIntegrity": {"baseCommit": CURRENT_REVISION}}


def _config(**overrides: object) -> AgentImprovementLoopConfig:
    values: dict[str, object] = {
        "root": Path("."),
        "output": Path("generated/agent_manifest"),
        "app_run_mode": "device-debug",
    }
    values.update(overrides)
    return AgentImprovementLoopConfig(**values)  # type: ignore[arg-type]


def _report(
    *,
    revision: str,
    source_layer: str,
    include_report_finish: bool,
) -> dict[str, object]:
    report: dict[str, object] = {
        "_source": f"d156.json#{source_layer}",
        "_sourceLayer": source_layer,
        "generatedAt": "2026-08-10T03:20:38Z",
        "appBuildNumber": BUILD_NUMBER,
        "app": {
            "buildNumber": BUILD_NUMBER,
            "sourceRevision": revision,
        },
        "failures": [],
    }
    if include_report_finish:
        report["reportFinishedAt"] = "2026-08-10T03:20:26Z"
    return report


def test_d156_like_source_mismatch_and_age_remain_historical() -> None:
    reports = [
        _report(
            revision=HISTORICAL_REVISION,
            source_layer="agentGroundingRuntimeAudit",
            include_report_finish=False,
        ),
        _report(
            revision=HISTORICAL_REVISION,
            source_layer="e2eTestReport.evidenceLayer",
            include_report_finish=True,
        ),
    ]
    config = _config(runtime_audit_reference_time="2026-08-10T09:35:43Z")

    ingested = _ingestion_runtime_reports(reports, config)
    current, proof = _assess_runtime_audit_proof(
        ingested,
        config,
        expected_source_revision=CURRENT_REVISION,
    )
    plan = _build_testflight_plan(
        config,
        _Manifest(),
        reports,
        [],
        ingestion_runtime_reports=ingested,
        runtime_proof=proof,
    )

    assert ingested == reports
    assert current == []
    assert proof["status"] == "historical-source-revision-mismatch"
    assert proof["currentProofComplete"] is False
    assert proof["verifiedReportCount"] == 0
    assert proof["historicalReportCount"] == 2
    assert proof["staleReportCount"] == 2
    assert proof["sourceRevisionMismatchReportCount"] == 2
    assert proof["freshnessUnverifiedReportCount"] == 0
    assert proof["basis"]["validUntil"] == "2026-08-10T04:20:26+00:00"
    assert plan["status"] == "historical-runtime-audit-ingested"
    assert plan["proofStatus"] == "historical-source-revision-mismatch"
    assert plan["currentRuntimeAuditProvided"] is False
    assert plan["currentRuntimeAuditReportCount"] == 0
    assert plan["historicalRuntimeAuditReportCount"] == 2
    assert plan["staleRuntimeAuditReportCount"] == 2
    assert plan["sourceRevisionMismatchRuntimeAuditReportCount"] == 2


def test_public_artifact_sanitizer_uses_relative_paths_and_content_refs(
    tmp_path: Path,
) -> None:
    root = tmp_path / "Users" / "private-user" / "repo"
    root.mkdir(parents=True)
    audit = tmp_path / "private" / "runtime" / "diagnostic-name.json"
    audit.parent.mkdir(parents=True)
    raw_bytes = b'{"privacy":"redacted"}\n'
    audit.write_bytes(raw_bytes)
    digest = hashlib.sha256(raw_bytes).hexdigest()
    reports = [{
        "_source": f"{audit}#liveE2EReport",
        "_artifactSha256": digest,
        "_artifactByteCount": len(raw_bytes),
    }]
    replacements = loop._runtime_audit_public_replacements((audit,), reports)

    public = loop._sanitize_public_artifact(
        {
            "root": str(root),
            "command": [
                str(Path(sys.executable).resolve()),
                "--root",
                str(root),
                "--runtime-audit",
                str(audit),
            ],
            "cwd": str(root),
            "stdoutTail": f"wrote {root / 'generated'} from {audit}",
            "otherHostPaths": [
                "/Volumes/AlexisSecret/model.gguf",
                "/Users/alex/Client Secret/model.gguf",
                "/Library/Private/config.json",
                "/usr/local/private-tool",
                "/Network/Servers/private-share",
                "file:///Users/alex/private.json",
                "file://localhost/Users/alex/private.json",
                "file:/Users/alex/private.json",
                "</Users/alex/Client Secret/report.json>",
                "[report](</Users/alex/Client Secret/report.json>)",
            ],
            "nested": {
                "/Users/alex/Client Secret/report.json": "private key canary",
            },
            "_artifactSha256": digest,
            "_artifactByteCount": len(raw_bytes),
        },
        root=root,
        replacements=replacements,
    )
    serialized = json.dumps(public, sort_keys=True)

    assert str(root) not in serialized
    assert str(audit) not in serialized
    assert "private-user" not in serialized
    assert "diagnostic-name.json" not in serialized
    assert str(Path(sys.executable).resolve()) not in serialized
    assert "AlexisSecret" not in serialized
    assert "/Library/" not in serialized
    assert "/usr/local/" not in serialized
    assert "/Network/" not in serialized
    assert "Client Secret" not in serialized
    assert "file:" not in serialized.casefold()
    assert f"runtime-audit-sha256-{digest}" in serialized
    assert public["root"] == "."
    assert public["cwd"] == "."
    assert public["stdoutTail"].startswith("wrote ./generated from ")
    assert "_artifactSha256" not in public
    assert "_artifactByteCount" not in public

    with pytest.raises(ValueError, match="key collision"):
        loop._sanitize_public_artifact(
            {"/Users/alex/one": 1, "/Users/alex/two": 2},
            root=root,
            replacements={},
        )


def test_explicit_reference_can_classify_fresh_but_never_current_proof() -> None:
    reports = [
        _report(
            revision=CURRENT_REVISION,
            source_layer="agentGroundingRuntimeAudit",
            include_report_finish=False,
        ),
        _report(
            revision=CURRENT_REVISION,
            source_layer="e2eTestReport.evidenceLayer",
            include_report_finish=True,
        ),
    ]
    config = _config(runtime_audit_reference_time="2026-08-10T03:20:50Z")

    current, proof = _assess_runtime_audit_proof(
        reports,
        config,
        expected_source_revision=CURRENT_REVISION,
    )
    plan = _build_testflight_plan(
        config,
        _Manifest(),
        reports,
        [],
        ingestion_runtime_reports=reports,
        runtime_proof=proof,
    )

    assert current == []
    assert proof["status"] == "fresh-at-explicit-reference-unverified"
    assert proof["freshAtExplicitReferenceReportCount"] == 2
    assert proof["verifiedReportCount"] == 0
    assert proof["currentProofComplete"] is False
    assert proof["basis"]["referenceTimeStatus"] == "valid"
    assert proof["basis"]["referenceTimeTrust"] == "caller-supplied-unverified"
    assert proof["basis"]["currentProofTrusted"] is False
    assert proof["basis"]["verificationReceiptProvided"] is False
    assert plan["status"] == "historical-runtime-audit-ingested"
    assert plan["currentRuntimeAuditProvided"] is False
    assert plan["currentRuntimeAuditReportCount"] == 0
    assert plan["freshAtExplicitReferenceRuntimeAuditReportCount"] == 2


def test_d156_like_without_reference_is_source_mismatched_and_freshness_unverified() -> None:
    reports = [
        _report(
            revision=HISTORICAL_REVISION,
            source_layer="agentGroundingRuntimeAudit",
            include_report_finish=False,
        ),
        _report(
            revision=HISTORICAL_REVISION,
            source_layer="e2eTestReport.evidenceLayer",
            include_report_finish=True,
        ),
    ]

    current, proof = _assess_runtime_audit_proof(
        reports,
        _config(),
        expected_source_revision=CURRENT_REVISION,
    )

    assert current == []
    assert proof["status"] == "historical-source-revision-mismatch"
    assert proof["sourceRevisionMismatchReportCount"] == 2
    assert proof["freshnessUnverifiedReportCount"] == 2
    assert proof["staleReportCount"] == 0
    assert proof["currentProofComplete"] is False
    assert proof["basis"]["validUntil"] == "2026-08-10T04:20:26+00:00"


def test_missing_reference_preserves_ingestion_but_marks_freshness_unverified() -> None:
    reports = [
        _report(
            revision=CURRENT_REVISION,
            source_layer="agentGroundingRuntimeAudit",
            include_report_finish=False,
        )
    ]
    reports[0]["_artifactSha256"] = "a" * 64
    reports[0]["_artifactByteCount"] = 123
    config = _config()

    ingested = _ingestion_runtime_reports(reports, config)
    current, proof = _assess_runtime_audit_proof(
        ingested,
        config,
        expected_source_revision=CURRENT_REVISION,
    )

    assert ingested == reports
    assert current == []
    assert proof["status"] == "historical-unverified"
    assert proof["freshnessUnverifiedReportCount"] == 1
    assert proof["freshAtExplicitReferenceReportCount"] == 0
    assert proof["historicalReportCount"] == 1
    assert proof["basis"]["referenceTimeStatus"] == "not-provided"

    training_reports = _annotate_runtime_reports_for_training(ingested, proof)
    assert training_reports[0]["_runtimeProofStatus"] == "historical-unverified"
    assert training_reports[0]["_runtimeCurrentProof"] is False
    assert training_reports[0]["_runtimeHistoricalObservation"] is True
    assert "_artifactSha256" not in training_reports[0]
    assert "_artifactByteCount" not in training_reports[0]
    assert "_runtimeProofStatus" not in reports[0]


def test_required_testflight_audit_fails_closed_without_verifier_receipt() -> None:
    report = _report(
        revision=CURRENT_REVISION,
        source_layer="agentGroundingRuntimeAudit",
        include_report_finish=False,
    )
    config = _config(
        app_run_mode="testflight",
        testflight_build_label=BUILD_NUMBER,
        require_testflight_runtime_audit=True,
        runtime_audit_reference_time="2026-08-10T03:20:50Z",
    )
    _, proof = _assess_runtime_audit_proof(
        [report],
        config,
        expected_source_revision=CURRENT_REVISION,
    )
    required_families = {
        "train_sft",
        "validation_sft",
        "eval_scenarios",
        "dpo_preference_pairs",
        "tool_schema_cards",
        "manifest_grounding_cards",
        "self_model_cards",
        "self_model_sft",
        "self_model_eval",
        "runtime_audit_repairs",
    }

    gaps = _build_gap_report(
        manifest=SimpleNamespace(tools=[]),
        validation_report=SimpleNamespace(failures=[], warnings=[]),
        datasets={family: [{}] for family in required_families},
        fine_tuning_datasets=None,
        runtime_reports=[report],
        all_runtime_reports=[report],
        runtime_proof=proof,
        command_results=[],
        config=config,
    )

    proof_gaps = [
        gap
        for gap in gaps
        if gap["category"] == "testflight_runtime_proof_unverified"
    ]
    assert len(proof_gaps) == 1
    assert proof_gaps[0]["severity"] == "error"
    assert proof_gaps[0]["evidence"]["proofStatus"] == (
        "fresh-at-explicit-reference-unverified"
    )


def test_verify_now_invokes_strict_verifier_and_persists_assessment_only(
    tmp_path: Path,
    monkeypatch,
) -> None:
    package = tmp_path / "evidence.json"
    package.write_bytes(b"{}")
    verifier_path = tmp_path / "tools" / "verify_interactive_model_tool_evidence.py"
    verifier_path.parent.mkdir(parents=True)
    verifier_bytes = b"# trusted verifier fixture\n"
    verifier_path.write_bytes(verifier_bytes)
    verifier_sha256 = hashlib.sha256(verifier_bytes).hexdigest()
    package_sha256 = hashlib.sha256(b"{}").hexdigest()
    report = {
        "_source": str(package),
        "_sourceFormat": "testflight_agent_grounding_package",
        "_sourceLayer": "agentGroundingRuntimeAudit",
        "manifestSource": "interactive-model-tool-validation-live-e2e",
        "_artifactSha256": package_sha256,
        "_artifactByteCount": 2,
        "failures": [],
    }
    verified_at = datetime.now(timezone.utc)
    valid_until = verified_at + timedelta(minutes=30)
    captured: dict[str, object] = {}

    def fake_run(command, **_kwargs):
        captured["command"] = command
        assert Path(command[1]) != verifier_path
        assert Path(command[1]).read_bytes() == verifier_bytes
        receipt_path = Path(command[command.index("--receipt-output") + 1])
        receipt_path.write_text(json.dumps({
            "schemaVersion": "lumen.interactive_model_tool_verifier_receipt/1.1.0",
            "status": "verified-at-assessment",
            "scope": "physical-device-debug-interactive-model-tool",
            "verifiedAt": verified_at.isoformat(),
            "validUntil": valid_until.isoformat(),
            "package": {
                "sha256": hashlib.sha256(b"{}").hexdigest(),
                "byteCount": 2,
            },
            "binding": {
                "bundleIdentifier": "com.27pm.lumenclone",
                "sourceRevision": CURRENT_REVISION,
                "buildNumber": BUILD_NUMBER,
                "workingTreeDigest": "c" * 64,
                "sourceDirtyState": False,
                "executionEnvironment": "physical-iPhone",
                "scenarioID": "interactive-model-tool-alarm-authorization",
                "toolID": "alarm.authorization_status",
            },
            "verifier": {
                "name": "verify_interactive_model_tool_evidence",
                "contractVersion": "1.1.0",
                "sourceSha256": verifier_sha256,
            },
        }), encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(
        loop,
        "_tracked_verifier_bytes",
        lambda _root, _revision: verifier_bytes,
    )
    monkeypatch.setattr(
        loop,
        "_working_file_matches_revision",
        lambda _root, _revision, _path: True,
    )
    monkeypatch.setattr(loop.subprocess, "run", fake_run)
    config = _config(
        root=tmp_path,
        deterministic=False,
        verify_runtime_audit_now=True,
        runtime_audit_expected_build_number=BUILD_NUMBER,
    )
    assessment = _verify_runtime_audit_at_host_now(
        [report],
        config,
        expected_source_revision=CURRENT_REVISION,
        expected_working_tree_digest="c" * 64,
    )
    _, proof = _assess_runtime_audit_proof(
        [report],
        config,
        expected_source_revision=CURRENT_REVISION,
        verification_assessment=assessment,
    )
    plan = _build_testflight_plan(
        config, _Manifest(), [report], [], runtime_proof=proof
    )

    command = captured["command"]
    assert "--receipt-output" in command
    assert command[command.index("--expected-working-tree-digest") + 1] == "c" * 64
    assert "--reference-time" not in command
    assert assessment["proofSatisfiedAtAssessment"] is True
    assert assessment["workingTreeDigest"] == "c" * 64
    assert assessment["sourceDirtyState"] is False
    assert assessment["executionEnvironment"] == "physical-iPhone"
    assert plan["status"] == "verified-at-assessment"
    assert plan["proofSatisfiedAtAssessment"] is True
    assert plan["packageSha256"] == package_sha256
    assert plan["scope"] == "physical-device-debug-interactive-model-tool"
    assert plan["currentRuntimeAuditProvided"] is False


def test_verify_now_is_explicitly_unsupported_for_testflight(monkeypatch) -> None:
    monkeypatch.setattr(
        loop.subprocess,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("not called")),
    )
    assessment = _verify_runtime_audit_at_host_now(
        [],
        _config(
            app_run_mode="testflight",
            deterministic=False,
            verify_runtime_audit_now=True,
        ),
        expected_source_revision=CURRENT_REVISION,
        expected_working_tree_digest="c" * 64,
    )

    assert assessment["status"] == "unsupported-debug-receipt-for-testflight"
    assert assessment["proofSatisfiedAtAssessment"] is False


def test_runtime_ingest_binds_every_normalized_report_to_one_byte_snapshot(
    tmp_path: Path,
) -> None:
    package = {
        "schemaVersion": "2.0.0",
        "generatedAt": "2026-08-10T03:20:38Z",
        "manifestSource": "interactive-model-tool-validation-live-e2e",
        "exportPolicy": {
            "format": "testflight-agent-grounding-runtime-json-package",
            "sourceLayer": "agentGroundingRuntimeAudit",
            "ownsLiveE2EScenarios": False,
        },
        "recentTraces": [],
        "liveE2EReport": {
            "schemaVersion": "1.0.0",
            "generatedAt": "2026-08-10T03:20:37Z",
            "exportPolicy": {
                "format": "live-e2e-test-report-json",
                "sourceLayer": "e2eTestReport",
                "ownsLiveE2EScenarios": True,
            },
            "payload": {
                "id": "33333333-3333-4333-8333-333333333333",
                "startedAt": "2026-08-10T03:20:00Z",
                "finishedAt": "2026-08-10T03:20:30Z",
                "passed": 0,
                "failed": 0,
                "results": [],
            },
        },
    }
    raw_bytes = json.dumps(package, separators=(",", ":")).encode("utf-8")
    path = tmp_path / "package.json"
    path.write_bytes(raw_bytes)

    reports = load_runtime_audit_reports([path])

    assert len(reports) == 2
    assert {
        (report["_artifactSha256"], report["_artifactByteCount"])
        for report in reports
    } == {(hashlib.sha256(raw_bytes).hexdigest(), len(raw_bytes))}


def test_verify_now_rejects_bytes_changed_after_ingest(
    tmp_path: Path,
    monkeypatch,
) -> None:
    package = tmp_path / "evidence.json"
    loaded_bytes = b"{}"
    package.write_bytes(loaded_bytes)
    report = {
        "_source": str(package),
        "_sourceFormat": "testflight_agent_grounding_package",
        "_sourceLayer": "agentGroundingRuntimeAudit",
        "manifestSource": "interactive-model-tool-validation-live-e2e",
        "_artifactSha256": hashlib.sha256(loaded_bytes).hexdigest(),
        "_artifactByteCount": len(loaded_bytes),
        "failures": [],
    }
    package.write_bytes(b'{"mutated":true}')
    monkeypatch.setattr(
        loop.subprocess,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("not called")),
    )

    assessment = _verify_runtime_audit_at_host_now(
        [report],
        _config(
            root=tmp_path,
            deterministic=False,
            verify_runtime_audit_now=True,
            runtime_audit_expected_build_number=BUILD_NUMBER,
        ),
        expected_source_revision=CURRENT_REVISION,
        expected_working_tree_digest="c" * 64,
    )

    assert assessment["status"] == "strict-verifier-loaded-artifact-mismatch"
    assert assessment["failureCode"] == "loaded_report_bytes_do_not_match_verifier_input"


def test_verify_now_fails_closed_in_deterministic_mode() -> None:
    assessment = _verify_runtime_audit_at_host_now(
        [],
        _config(verify_runtime_audit_now=True),
        expected_source_revision=CURRENT_REVISION,
    )

    assert assessment["status"] == "strict-verifier-requires-non-deterministic-mode"
    assert assessment["proofSatisfiedAtAssessment"] is False


def test_verify_now_rejects_untracked_verifier_implementation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    package = tmp_path / "evidence.json"
    raw_bytes = b"{}"
    package.write_bytes(raw_bytes)
    verifier_path = tmp_path / "tools" / "verify_interactive_model_tool_evidence.py"
    verifier_path.parent.mkdir(parents=True)
    verifier_path.write_bytes(b"# modified verifier\n")
    report = {
        "_source": str(package),
        "_sourceFormat": "testflight_agent_grounding_package",
        "_sourceLayer": "agentGroundingRuntimeAudit",
        "manifestSource": "interactive-model-tool-validation-live-e2e",
        "_artifactSha256": hashlib.sha256(raw_bytes).hexdigest(),
        "_artifactByteCount": len(raw_bytes),
        "failures": [],
    }
    monkeypatch.setattr(
        loop,
        "_tracked_verifier_bytes",
        lambda _root, _revision: b"# committed verifier\n",
    )
    monkeypatch.setattr(
        loop.subprocess,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("not called")),
    )

    assessment = _verify_runtime_audit_at_host_now(
        [report],
        _config(
            root=tmp_path,
            deterministic=False,
            verify_runtime_audit_now=True,
            runtime_audit_expected_build_number=BUILD_NUMBER,
        ),
        expected_source_revision=CURRENT_REVISION,
        expected_working_tree_digest="c" * 64,
    )

    assert assessment["status"] == "strict-verifier-implementation-untrusted"
    assert assessment["proofSatisfiedAtAssessment"] is False


def test_requested_verification_failure_is_a_hard_gap() -> None:
    config = _config(
        deterministic=True,
        verify_runtime_audit_now=True,
        runtime_audit_expected_build_number=BUILD_NUMBER,
    )
    assessment = _verify_runtime_audit_at_host_now(
        [], config, expected_source_revision=CURRENT_REVISION
    )
    _, proof = _assess_runtime_audit_proof(
        [],
        config,
        expected_source_revision=CURRENT_REVISION,
        verification_assessment=assessment,
    )
    required_families = {
        "train_sft", "validation_sft", "eval_scenarios",
        "dpo_preference_pairs", "tool_schema_cards", "manifest_grounding_cards",
        "self_model_cards", "self_model_sft", "self_model_eval",
        "runtime_audit_repairs",
    }

    gaps = _build_gap_report(
        manifest=SimpleNamespace(tools=[]),
        validation_report=SimpleNamespace(failures=[], warnings=[]),
        datasets={family: [{}] for family in required_families},
        fine_tuning_datasets=None,
        runtime_reports=[],
        runtime_proof=proof,
        command_results=[],
        config=config,
    )

    strict_gap = next(
        gap
        for gap in gaps
        if gap["category"] == "runtime_audit_strict_verification_failed"
    )
    assert strict_gap["severity"] == "error"
    assert strict_gap["evidence"]["proofStatus"] == (
        "strict-verifier-requires-non-deterministic-mode"
    )
