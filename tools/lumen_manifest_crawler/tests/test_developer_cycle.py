from __future__ import annotations

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
    assert report["runtimeEvidencePresent"] is True
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
        "runtimeEvidencePresent",
        "improvementLoopPassed",
        "improvementLoopOutputContractPassed",
        "xcodeValidationStatus",
        "trainingStatus",
        "overallPortablePassed",
        "overallReleaseCandidatePassed",
    }
    assert required.issubset(report)
    assert (tmp_path / "generated/developer_framework/developer_cycle_report.json").exists()
    assert (tmp_path / "generated/developer_framework/DEVELOPER_CYCLE_REPORT.md").exists()
