from __future__ import annotations

import copy
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from lumen_manifest_crawler import developer_cycle as dc
from lumen_manifest_crawler.developer_cycle import DeveloperCycleConfig, run_developer_cycle


def test_non_git_zip_behavior_skips_git_diff(tmp_path: Path) -> None:
    report, exit_code = run_developer_cycle(
        DeveloperCycleConfig(
            root=tmp_path,
            portable=True,
            dry_run=True,
            skip_generation=True,
            skip_improvement_loop=True,
        )
    )

    assert exit_code == 0
    assert report["environment"]["checkoutKind"] == "non_git_zip_or_export"
    static_phase = [phase for phase in report["phases"] if phase["id"] == "phase1_static_source_validation"][0]
    git_diff = [command for command in static_phase["commands"] if command["name"] == "git_diff_check"][0]
    assert git_diff["status"] == "skipped"
    assert git_diff["skipReason"] == "not inside a git worktree"


def test_linux_portable_mode_marks_xcode_skipped(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(dc.platform, "system", lambda: "Linux")
    monkeypatch.setattr(dc.shutil, "which", lambda _name: None)

    report, exit_code = run_developer_cycle(
        DeveloperCycleConfig(
            root=tmp_path,
            portable=True,
            dry_run=True,
            skip_generation=True,
            skip_improvement_loop=True,
        )
    )

    assert exit_code == 0
    assert report["environment"]["executionKind"] == "linux_codex_static"
    assert report["xcodeValidationStatus"] == "skipped"
    xcode_phase = [phase for phase in report["phases"] if phase["id"] == "phase5_xcode_validation"][0]
    assert xcode_phase["reason"] == "portable mode"


def test_macos_without_xcode_skips_or_fails_depending_flags(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(dc.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(dc.shutil, "which", lambda _name: None)

    skipped_report, skipped_exit = run_developer_cycle(
        DeveloperCycleConfig(
            root=tmp_path,
            dry_run=True,
            skip_generation=True,
            skip_improvement_loop=True,
        )
    )
    failed_report, failed_exit = run_developer_cycle(
        DeveloperCycleConfig(
            root=tmp_path,
            with_xcode=True,
            dry_run=True,
            skip_generation=True,
            skip_improvement_loop=True,
        )
    )

    assert skipped_exit == 0
    assert skipped_report["xcodeValidationStatus"] == "skipped"
    assert failed_exit == 1
    assert failed_report["xcodeValidationStatus"] == "failed"


def test_runtime_audit_paths_are_passed_into_improvement_loop(tmp_path: Path, monkeypatch) -> None:
    captured: dict[str, object] = {}
    explicit = tmp_path / "audit.json"

    def fake_loop(config):
        captured["runtime_audit_paths"] = config.runtime_audit_paths
        return SimpleNamespace(
            state={
                "validation": {"failureCount": 0, "warningCount": 0},
                "runtime": {
                    "reportCount": 1,
                    "failureCount": 0,
                    "rawFailureCount": 0,
                    "skippedLiveModelGenerationCount": 0,
                },
                "testFlight": {
                    "currentRuntimeAuditProvided": False,
                    "proofStatus": "historical-source-revision-mismatch",
                    "proofBasis": {
                        "currentProofTrusted": False,
                        "verificationReceiptProvided": False,
                    },
                    "ingestedRuntimeAuditReportCount": 1,
                    "nextIngestCommand": "python -m lumen_manifest_crawler improve-loop --runtime-audit export.json",
                    "scenarioQueuePath": "testflight_scenarios.jsonl",
                },
                "passed": True,
                "gapCount": 0,
                "criticalGapCount": 0,
                "errorGapCount": 0,
            }
        )

    monkeypatch.setattr(dc, "run_agent_improvement_loop", fake_loop)
    monkeypatch.setattr(dc, "_run_static_validation", lambda root, environment, dry_run: {"commands": [], "staticValidationPassed": True})
    monkeypatch.setattr(dc, "_run_xcode_validation", lambda root, environment, config: dc._xcode_phase("skipped", [], "test"))
    monkeypatch.setattr(
        dc,
        "analyze_reports",
        lambda root, paths: {
            "schemaVersion": "test",
            "reportCount": 1,
            "runtimeFailureCount": 0,
            "runtimeFailures": [],
            "plainFindings": [],
        },
    )

    report, exit_code = run_developer_cycle(
        DeveloperCycleConfig(root=tmp_path, runtime_audit_paths=(explicit,))
    )

    assert exit_code == 0
    assert report["historicalRuntimeInputPresent"] is True
    assert report["currentRuntimeProofPresent"] is False
    assert report["runtimeEvidencePresent"] is False
    assert report["runtimeProofStatus"] == "historical-source-revision-mismatch"
    paths = tuple(Path(path) for path in captured["runtime_audit_paths"])  # type: ignore[arg-type]
    assert explicit in paths
    assert tmp_path / "exports" in paths
    assert tmp_path / "runtime-audits" in paths
    assert tmp_path / "generated/agent_improvement_loop" not in paths


def test_final_json_report_contains_required_fields(tmp_path: Path) -> None:
    report, _ = run_developer_cycle(
        DeveloperCycleConfig(
            root=tmp_path,
            portable=True,
            dry_run=True,
            skip_generation=True,
            skip_improvement_loop=True,
        )
    )

    required = {
        "staticValidationPassed",
        "manifestValidationPassed",
        "historicalRuntimeInputPresent",
        "currentRuntimeProofPresent",
        "runtimeProofStatus",
        "runtimeProofBasis",
        "runtimeEvidencePresent",
        "improvementLoopPassed",
        "improvementLoopOutputContractPassed",
        "xcodeValidationStatus",
        "trainingStatus",
        "overallPortablePassed",
        "overallDeviceDebugDiagnosticPassed",
        "overallReleaseCandidatePassed",
    }
    assert required.issubset(report)
    assert (tmp_path / "generated/developer_framework/developer_cycle_report.json").exists()
    assert (tmp_path / "generated/developer_framework/DEVELOPER_CYCLE_REPORT.md").exists()


def test_historical_runtime_input_cannot_make_release_candidate_pass(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        dc,
        "_run_static_validation",
        lambda root, environment, dry_run: {
            "commands": [],
            "staticValidationPassed": True,
        },
    )
    monkeypatch.setattr(
        dc,
        "_run_generation_and_loop",
        lambda root, config: (
            {
                "id": "phase2_manifest_dataset_generation",
                "commands": [],
                "manifestValidationPassed": True,
            },
            None,
        ),
    )
    monkeypatch.setattr(
        dc,
        "_run_runtime_ingestion",
        lambda root, paths, dry_run: {
            "id": "phase3_runtime_audit_ingestion",
            "commands": [],
            "historicalRuntimeInputPresent": True,
            "runtimeFailureCount": 0,
        },
    )
    monkeypatch.setattr(
        dc,
        "_summarize_improvement_loop",
        lambda root, config, result, **_kwargs: {
            "id": "phase4_improvement_loop_preparation",
            "commands": [],
            "improvementLoopPassed": True,
            "improvementLoopOutputContractPassed": True,
            "historicalRuntimeInputPresent": True,
            "currentRuntimeProofPresent": False,
            "declaredCurrentRuntimeProofPresent": False,
            "runtimeProofStatus": "historical-source-revision-mismatch",
            "runtimeProofBasis": {
                "currentProofTrusted": False,
                "verificationReceiptProvided": False,
            },
            "rawRuntimeFailureCount": 0,
            "skippedLiveModelGenerationCount": 0,
        },
    )
    monkeypatch.setattr(
        dc,
        "_run_xcode_validation",
        lambda root, environment, config: dc._xcode_phase("passed", [], ""),
    )
    monkeypatch.setattr(
        dc,
        "_run_training_profile",
        lambda root, config: {
            "id": "phase6_training_hf_profile",
            "commands": [],
            "trainingStatus": "not_requested",
        },
    )

    report, exit_code = run_developer_cycle(DeveloperCycleConfig(root=tmp_path))

    assert exit_code == 0
    assert report["overallPortablePassed"] is True
    assert report["historicalRuntimeInputPresent"] is True
    assert report["currentRuntimeProofPresent"] is False
    assert report["runtimeEvidencePresent"] is False
    assert report["runtimeProofStatus"] == "historical-source-revision-mismatch"
    assert report["overallReleaseCandidatePassed"] is False


def _receipt_backed_state(
    root: Path,
    package: Path,
    *,
    revision: str,
    build_number: str,
) -> dict:
    raw_bytes = package.read_bytes()
    return {
        "runtimeAuditInputs": [
            "<runtime-audit-input-redacted>",
            f"runtime-audit-sha256-{hashlib.sha256(raw_bytes).hexdigest()}",
        ],
        "manifest": {
            "baseCommit": revision,
            "dirtyState": False,
            "workingTreeDigest": "c" * 64,
        },
        "testFlight": {
            "mode": "device-debug",
            "currentRuntimeAuditProvided": False,
            "proofSatisfiedAtAssessment": True,
            "proofStatus": "verified-at-assessment",
            "proofBasis": {
                "expectedSourceRevision": revision,
                "expectedBuildNumber": build_number,
                "verificationReceiptProvided": True,
                "proofSatisfiedAtAssessment": True,
                "scope": "physical-device-debug-interactive-model-tool",
                "verifiedAt": "2026-08-10T10:00:00Z",
                "validUntil": "2026-08-10T12:00:00Z",
                "packageSha256": hashlib.sha256(raw_bytes).hexdigest(),
                "packageByteCount": len(raw_bytes),
                "sourceRevision": revision,
                "buildNumber": build_number,
                "workingTreeDigest": "c" * 64,
                "sourceDirtyState": False,
                "executionEnvironment": "physical-iPhone",
                "scenarioID": "interactive-model-tool-alarm-authorization",
                "toolID": "alarm.authorization_status",
                "verifierName": "verify_interactive_model_tool_evidence",
                "verifierContractVersion": "1.1.0",
                "verifierSourceSha256": "b" * 64,
            },
            "ingestedRuntimeAuditReportCount": 1,
        },
    }


def test_developer_cycle_accepts_only_current_invocation_bound_receipt(
    tmp_path: Path,
    monkeypatch,
) -> None:
    revision = "a" * 40
    build_number = "205"
    package = tmp_path / "audit.json"
    package.write_bytes(b'{"evidence":"model-tool"}')
    config = DeveloperCycleConfig(
        root=tmp_path,
        app_run_mode="device-debug",
        verify_runtime_audit_now=True,
        runtime_audit_expected_build_number=build_number,
        runtime_audit_paths=(package,),
    )
    receipt_backed = _receipt_backed_state(
        tmp_path,
        package,
        revision=revision,
        build_number=build_number,
    )
    invocation = {
        "verificationRequested": True,
        "expectedBuildNumber": build_number,
        "insideExpectedWorktree": True,
        "checkoutCleanAtInvocation": True,
        "sourceDirtyState": False,
        "workingTreeDigest": "c" * 64,
        "headRevision": revision,
    }
    monkeypatch.setattr(dc, "_git_head_revision", lambda _root: revision)
    monkeypatch.setattr(
        dc,
        "_repository_working_tree_provenance",
        lambda _root: ("c" * 64, False),
    )
    monkeypatch.setattr(
        dc,
        "_working_file_matches_revision",
        lambda _root, _revision, _path: True,
    )

    legacy_or_untrusted = {
        "testFlight": {
            "currentRuntimeAuditProvided": True,
            "proofStatus": "fresh-at-explicit-reference-unverified",
            "proofBasis": {
                "currentProofTrusted": False,
                "verificationReceiptProvided": False,
            },
            "ingestedRuntimeAuditReportCount": 1,
        }
    }
    assert dc._runtime_proof_boundary(legacy_or_untrusted)["accepted"] is False
    assessed = dc._runtime_proof_boundary(
        receipt_backed,
        root=tmp_path,
        config=config,
        current_loop_executed=True,
        invocation=invocation,
        now=datetime(2026, 8, 10, 11, 0, tzinfo=timezone.utc),
    )
    expired = dc._runtime_proof_boundary(
        receipt_backed,
        root=tmp_path,
        config=config,
        current_loop_executed=True,
        invocation=invocation,
        now=datetime(2026, 8, 10, 13, 0, tzinfo=timezone.utc),
    )
    assert assessed["accepted"] is True
    assert assessed["bindingFailures"] == []
    assert expired["accepted"] is False
    assert "receipt_expired_or_time_invalid" in expired["bindingFailures"]


def test_persisted_receipt_is_rejected_when_loop_was_skipped(
    tmp_path: Path,
    monkeypatch,
) -> None:
    revision = "a" * 40
    package = tmp_path / "audit.json"
    package.write_bytes(b'{"evidence":"model-tool"}')
    config = DeveloperCycleConfig(
        root=tmp_path,
        skip_improvement_loop=True,
        app_run_mode="device-debug",
        verify_runtime_audit_now=True,
        runtime_audit_expected_build_number="205",
        runtime_audit_paths=(package,),
    )
    state = _receipt_backed_state(
        tmp_path,
        package,
        revision=revision,
        build_number="205",
    )
    invocation = {
        "verificationRequested": True,
        "expectedBuildNumber": "205",
        "insideExpectedWorktree": True,
        "checkoutCleanAtInvocation": True,
        "sourceDirtyState": False,
        "workingTreeDigest": "c" * 64,
        "headRevision": revision,
    }
    monkeypatch.setattr(dc, "_git_head_revision", lambda _root: revision)
    monkeypatch.setattr(
        dc,
        "_repository_working_tree_provenance",
        lambda _root: ("c" * 64, False),
    )
    monkeypatch.setattr(
        dc,
        "_working_file_matches_revision",
        lambda _root, _revision, _path: True,
    )

    boundary = dc._runtime_proof_boundary(
        state,
        root=tmp_path,
        config=config,
        current_loop_executed=False,
        invocation=invocation,
        now=datetime(2026, 8, 10, 11, 0, tzinfo=timezone.utc),
    )

    assert boundary["accepted"] is False
    assert "improvement_loop_not_executed_this_invocation" in boundary["bindingFailures"]


def test_receipt_binding_rejects_build_checkout_identity_and_package_drift(
    tmp_path: Path,
    monkeypatch,
) -> None:
    revision = "a" * 40
    package = tmp_path / "audit.json"
    original_bytes = b'{"evidence":"model-tool"}'
    package.write_bytes(original_bytes)
    config = DeveloperCycleConfig(
        root=tmp_path,
        app_run_mode="device-debug",
        verify_runtime_audit_now=True,
        runtime_audit_expected_build_number="205",
        runtime_audit_paths=(package,),
    )
    state = _receipt_backed_state(
        tmp_path,
        package,
        revision=revision,
        build_number="205",
    )
    invocation = {
        "verificationRequested": True,
        "expectedBuildNumber": "205",
        "insideExpectedWorktree": True,
        "checkoutCleanAtInvocation": True,
        "sourceDirtyState": False,
        "workingTreeDigest": "c" * 64,
        "headRevision": revision,
    }
    monkeypatch.setattr(dc, "_git_head_revision", lambda _root: revision)
    monkeypatch.setattr(
        dc,
        "_repository_working_tree_provenance",
        lambda _root: ("c" * 64, False),
    )
    monkeypatch.setattr(
        dc,
        "_working_file_matches_revision",
        lambda _root, _revision, _path: True,
    )
    assessment_time = datetime(2026, 8, 10, 11, 0, tzinfo=timezone.utc)

    wrong_build = copy.deepcopy(state)
    wrong_build["testFlight"]["proofBasis"]["buildNumber"] = "204"
    dirty_invocation = {**invocation, "checkoutCleanAtInvocation": False}
    wrong_scope = copy.deepcopy(state)
    wrong_scope["testFlight"]["proofBasis"]["scope"] = "testflight"

    assert dc._runtime_proof_boundary(
        wrong_build,
        root=tmp_path,
        config=config,
        current_loop_executed=True,
        invocation=invocation,
        now=assessment_time,
    )["accepted"] is False
    assert dc._runtime_proof_boundary(
        state,
        root=tmp_path,
        config=config,
        current_loop_executed=True,
        invocation=dirty_invocation,
        now=assessment_time,
    )["accepted"] is False
    assert dc._runtime_proof_boundary(
        wrong_scope,
        root=tmp_path,
        config=config,
        current_loop_executed=True,
        invocation=invocation,
        now=assessment_time,
    )["accepted"] is False

    package.write_bytes(b'{"evidence":"model-TOOL"}')
    mutated = dc._runtime_proof_boundary(
        state,
        root=tmp_path,
        config=config,
        current_loop_executed=True,
        invocation=invocation,
        now=assessment_time,
    )
    assert mutated["accepted"] is False
    assert "receipt_package_not_uniquely_present" in mutated["bindingFailures"]


def test_developer_cycle_verify_command_is_explicitly_non_deterministic(
    tmp_path: Path,
) -> None:
    command = dc._improve_loop_command(
        tmp_path,
        DeveloperCycleConfig(
            root=tmp_path,
            app_run_mode="device-debug",
            verify_runtime_audit_now=True,
            runtime_audit_expected_build_number="205",
        ),
    )

    assert "--non-deterministic" in command
    assert "--verify-runtime-audit-now" in command


def test_runtime_ingestion_sanitizes_all_tracked_side_reports(
    tmp_path: Path,
    monkeypatch,
) -> None:
    audit = tmp_path / "Client Secret" / "diagnostic-name.json"
    audit.parent.mkdir(parents=True)
    raw_bytes = b'{"privacy":"redacted"}\n'
    audit.write_bytes(raw_bytes)
    digest = hashlib.sha256(raw_bytes).hexdigest()
    monkeypatch.setattr(
        dc,
        "analyze_reports",
        lambda _root, _paths: {
            "schemaVersion": "test",
            "reportCount": 1,
            "runtimeFailureCount": 0,
            "runtimeFailures": [],
            "plainFindings": [{
                "source": str(audit),
                "message": "file://localhost/Users/alex/private.json",
            }],
        },
    )
    monkeypatch.setattr(
        dc,
        "load_runtime_audit_reports",
        lambda _paths: [{
            "_source": str(audit),
            "_artifactSha256": digest,
            "_artifactByteCount": len(raw_bytes),
        }],
    )

    dc._run_runtime_ingestion(tmp_path, (audit,), dry_run=False)

    for name in ("framework_report.json", "runtime_report_index.json"):
        contents = (
            tmp_path / "generated" / "developer_framework" / name
        ).read_text(encoding="utf-8")
        assert str(tmp_path) not in contents
        assert "diagnostic-name.json" not in contents
        assert "Client Secret" not in contents
        assert "file:" not in contents.casefold()
        assert f"runtime-audit-sha256-{digest}" in contents


def test_developer_cycle_rechecks_receipt_after_long_phases(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = DeveloperCycleConfig(
        root=tmp_path,
        app_run_mode="device-debug",
        verify_runtime_audit_now=True,
        runtime_audit_expected_build_number="205",
    )
    loop_result = SimpleNamespace(state={"testFlight": {}})
    monkeypatch.setattr(dc, "_capture_runtime_proof_invocation", lambda *_args: {})
    monkeypatch.setattr(
        dc,
        "_run_static_validation",
        lambda *_args, **_kwargs: {"commands": [], "staticValidationPassed": True},
    )
    monkeypatch.setattr(
        dc,
        "_run_generation_and_loop",
        lambda *_args: (
            {"commands": [], "manifestValidationPassed": True},
            loop_result,
        ),
    )
    monkeypatch.setattr(
        dc,
        "_run_runtime_ingestion",
        lambda *_args, **_kwargs: {
            "commands": [],
            "historicalRuntimeInputPresent": True,
            "runtimeFailureCount": 0,
        },
    )
    monkeypatch.setattr(
        dc,
        "_summarize_improvement_loop",
        lambda *_args, **_kwargs: {
            "commands": [],
            "improvementLoopPassed": True,
            "improvementLoopOutputContractPassed": True,
            "historicalRuntimeInputPresent": True,
            "runtimeProofSatisfiedAtAssessment": True,
            "runtimeProofStatus": "verified-at-assessment",
            "runtimeProofBasis": {},
            "runtimeProofBindingFailures": [],
            "rawRuntimeFailureCount": 0,
            "skippedLiveModelGenerationCount": 0,
        },
    )
    monkeypatch.setattr(
        dc,
        "_run_xcode_validation",
        lambda *_args: dc._xcode_phase("passed", [], ""),
    )
    monkeypatch.setattr(
        dc,
        "_run_training_profile",
        lambda *_args: {"commands": [], "trainingStatus": "not_requested"},
    )
    monkeypatch.setattr(
        dc,
        "_runtime_proof_boundary",
        lambda *_args, **_kwargs: {
            "accepted": False,
            "declared": True,
            "status": "verified-at-assessment",
            "basis": {},
            "bindingFailures": ["receipt_expired_or_time_invalid"],
        },
    )

    report, _ = run_developer_cycle(config)

    assert report["runtimeProofSatisfiedAtAssessment"] is False
    assert report["overallDeviceDebugDiagnosticPassed"] is False
    assert report["overallReleaseCandidatePassed"] is False
    assert report["runtimeProofBindingFailures"] == [
        "receipt_expired_or_time_invalid"
    ]
