from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from lumen_manifest_crawler.developer_framework import (
    EvidenceLayer,
    FrameworkEnvironment,
    FrameworkJob,
    FrameworkJobRunner,
    FrameworkJobState,
    analyze_reports,
    build_framework_jobs,
    load_framework_snapshot,
    resolve_environment,
    run_framework_job,
    _index_html,
)


# ---------------------------------------------------------------------------
# Existing tests (unchanged)
# ---------------------------------------------------------------------------


def test_framework_jobs_include_macos_and_ubuntu_profiles(tmp_path: Path) -> None:
    mac_jobs = {job.id: job for job in build_framework_jobs(tmp_path, FrameworkEnvironment.MACOS)}
    ubuntu_jobs = {job.id: job for job in build_framework_jobs(tmp_path, FrameworkEnvironment.UBUNTU)}

    assert "adapter-invariants" in mac_jobs
    assert "improve-loop" in mac_jobs
    assert mac_jobs["improve-loop"].evidence_layer == EvidenceLayer.STATIC_SOURCE

    assert "ubuntu-preflight" in ubuntu_jobs
    assert "train-adapters" in ubuntu_jobs
    assert "hf-resolve" in ubuntu_jobs
    assert ubuntu_jobs["train-adapters"].requires_confirmation is True


def test_framework_snapshot_is_valid_without_generated_artifacts(tmp_path: Path) -> None:
    snapshot = load_framework_snapshot(tmp_path, FrameworkEnvironment.MACOS)

    assert snapshot["schemaVersion"] == "lumen.developer_framework/1.0.0"
    assert snapshot["authoritativeLiveLayer"] == EvidenceLayer.LIVE_E2E.value
    assert snapshot["gapCount"] == 0
    live = [layer for layer in snapshot["evidenceLayers"] if layer["id"] == EvidenceLayer.LIVE_E2E.value][0]
    assert live["ownsScenarioPassFail"] is True


def test_framework_snapshot_reads_loop_gaps_and_prompts(tmp_path: Path) -> None:
    loop = tmp_path / "generated" / "agent_improvement_loop"
    loop.mkdir(parents=True)
    (loop / "loop_gaps.json").write_text(
        json.dumps({"gaps": [{"severity": "error", "title": "Missing traces"}]}),
        encoding="utf-8",
    )
    (loop / "next_action_prompts.jsonl").write_text(
        json.dumps({"taskType": "codebase_improvement"}) + "\n",
        encoding="utf-8",
    )

    snapshot = load_framework_snapshot(tmp_path, FrameworkEnvironment.MACOS)

    assert snapshot["gapCount"] == 1
    assert snapshot["gaps"][0]["title"] == "Missing traces"
    assert snapshot["nextActionPrompts"][0]["taskType"] == "codebase_improvement"


def test_analyze_reports_flags_plain_no_model_evidence(tmp_path: Path) -> None:
    report = tmp_path / "e2e.log"
    report.write_text("No model loaded; routing-only checks completed.", encoding="utf-8")

    analysis = analyze_reports(tmp_path, [report])

    assert analysis["plainFindings"][0]["type"] == "invalid_live_e2e_no_model"


def test_resolve_environment_accepts_explicit_values() -> None:
    assert resolve_environment("macos") == FrameworkEnvironment.MACOS
    assert resolve_environment("ubuntu") == FrameworkEnvironment.UBUNTU


# ---------------------------------------------------------------------------
# EvidenceLayer enum
# ---------------------------------------------------------------------------


def test_evidence_layer_values_are_stable() -> None:
    assert EvidenceLayer.STATIC_SOURCE.value == "static_source"
    assert EvidenceLayer.LOCAL_VALIDATION.value == "local_validation"
    assert EvidenceLayer.SIMULATOR_VALIDATION.value == "simulator_validation"
    assert EvidenceLayer.DEVICE_RUNTIME.value == "device_runtime"
    assert EvidenceLayer.LIVE_E2E.value == "live_e2e"
    assert EvidenceLayer.TRAINING_FEEDBACK.value == "training_feedback"


def test_only_live_e2e_owns_scenario_pass_fail() -> None:
    owners = [layer for layer in EvidenceLayer if layer == EvidenceLayer.LIVE_E2E]
    non_owners = [layer for layer in EvidenceLayer if layer != EvidenceLayer.LIVE_E2E]
    assert len(owners) == 1
    for layer in non_owners:
        assert layer != EvidenceLayer.LIVE_E2E


# ---------------------------------------------------------------------------
# FrameworkEnvironment enum
# ---------------------------------------------------------------------------


def test_framework_environment_values_are_stable() -> None:
    assert FrameworkEnvironment.AUTO.value == "auto"
    assert FrameworkEnvironment.MACOS.value == "macos"
    assert FrameworkEnvironment.UBUNTU.value == "ubuntu"


def test_framework_environment_is_string_subclass() -> None:
    # FrameworkEnvironment(str, Enum) means values compare equal to strings
    assert FrameworkEnvironment.MACOS == "macos"
    assert FrameworkEnvironment.UBUNTU == "ubuntu"


# ---------------------------------------------------------------------------
# resolve_environment
# ---------------------------------------------------------------------------


def test_resolve_environment_accepts_enum_input() -> None:
    assert resolve_environment(FrameworkEnvironment.MACOS) == FrameworkEnvironment.MACOS
    assert resolve_environment(FrameworkEnvironment.UBUNTU) == FrameworkEnvironment.UBUNTU


def test_resolve_environment_auto_returns_macos_on_darwin() -> None:
    with patch("platform.system", return_value="Darwin"):
        result = resolve_environment(FrameworkEnvironment.AUTO)
    assert result == FrameworkEnvironment.MACOS


def test_resolve_environment_auto_returns_ubuntu_on_linux() -> None:
    with patch("platform.system", return_value="Linux"):
        result = resolve_environment(FrameworkEnvironment.AUTO)
    assert result == FrameworkEnvironment.UBUNTU


def test_resolve_environment_auto_defaults_to_macos_on_unknown_os() -> None:
    with patch("platform.system", return_value="Windows"):
        result = resolve_environment(FrameworkEnvironment.AUTO)
    assert result == FrameworkEnvironment.MACOS


# ---------------------------------------------------------------------------
# FrameworkJob.output_dict
# ---------------------------------------------------------------------------


def test_framework_job_output_dict_has_all_required_keys(tmp_path: Path) -> None:
    job = FrameworkJob(
        id="test-job",
        title="Test job",
        environment=FrameworkEnvironment.MACOS,
        evidence_layer=EvidenceLayer.LOCAL_VALIDATION,
        command=(sys.executable, "--version"),
        description="A test job.",
        outputs=("generated/out.json",),
        requires_confirmation=False,
    )
    d = job.output_dict()

    assert d["id"] == "test-job"
    assert d["title"] == "Test job"
    assert d["environment"] == "macos"
    assert d["evidenceLayer"] == "local_validation"
    assert d["command"] == [sys.executable, "--version"]
    assert d["description"] == "A test job."
    assert d["outputs"] == ["generated/out.json"]
    assert d["requiresConfirmation"] is False


def test_framework_job_output_dict_confirmation_flag(tmp_path: Path) -> None:
    job = FrameworkJob(
        id="confirm-job",
        title="Confirm",
        environment=FrameworkEnvironment.UBUNTU,
        evidence_layer=EvidenceLayer.TRAINING_FEEDBACK,
        command=(sys.executable,),
        description="Needs confirm.",
        requires_confirmation=True,
    )
    assert job.output_dict()["requiresConfirmation"] is True


# ---------------------------------------------------------------------------
# FrameworkJobState.output_dict
# ---------------------------------------------------------------------------


def test_framework_job_state_output_dict_idle_state() -> None:
    state = FrameworkJobState()
    d = state.output_dict()

    assert d["jobID"] is None
    assert d["status"] == "idle"
    assert d["startedAt"] is None
    assert d["endedAt"] is None
    assert d["durationSeconds"] is None
    assert d["returncode"] is None
    assert d["command"] == []
    assert d["log"] == []


def test_framework_job_state_duration_seconds_computed_when_started() -> None:
    before = time.time()
    state = FrameworkJobState(job_id="x", status="running", started_at=before)
    d = state.output_dict()

    assert d["durationSeconds"] is not None
    assert d["durationSeconds"] >= 0.0


def test_framework_job_state_duration_seconds_uses_ended_at_when_present() -> None:
    start = 1000.0
    end = 1005.0
    state = FrameworkJobState(job_id="x", status="passed", started_at=start, ended_at=end)
    d = state.output_dict()
    assert d["durationSeconds"] == 5.0


# ---------------------------------------------------------------------------
# build_framework_jobs – common jobs present in both environments
# ---------------------------------------------------------------------------


def test_build_framework_jobs_common_jobs_exist_in_both_environments(tmp_path: Path) -> None:
    for env in (FrameworkEnvironment.MACOS, FrameworkEnvironment.UBUNTU):
        job_ids = {job.id for job in build_framework_jobs(tmp_path, env)}
        assert "status" in job_ids, f"status job missing in {env}"
        assert "ingest-runtime" in job_ids, f"ingest-runtime job missing in {env}"


def test_build_framework_jobs_common_jobs_have_correct_evidence_layers(tmp_path: Path) -> None:
    mac_jobs = {job.id: job for job in build_framework_jobs(tmp_path, FrameworkEnvironment.MACOS)}
    assert mac_jobs["status"].evidence_layer == EvidenceLayer.LOCAL_VALIDATION
    assert mac_jobs["ingest-runtime"].evidence_layer == EvidenceLayer.DEVICE_RUNTIME


def test_build_framework_jobs_macos_does_not_include_ubuntu_jobs(tmp_path: Path) -> None:
    mac_job_ids = {job.id for job in build_framework_jobs(tmp_path, FrameworkEnvironment.MACOS)}
    assert "ubuntu-preflight" not in mac_job_ids
    assert "train-adapters" not in mac_job_ids
    assert "convert-adapters" not in mac_job_ids


def test_build_framework_jobs_ubuntu_does_not_include_macos_jobs(tmp_path: Path) -> None:
    ubuntu_job_ids = {job.id for job in build_framework_jobs(tmp_path, FrameworkEnvironment.UBUNTU)}
    assert "adapter-invariants" not in ubuntu_job_ids
    assert "build-readiness" not in ubuntu_job_ids
    assert "improve-loop" not in ubuntu_job_ids
    assert "visual-dashboard" not in ubuntu_job_ids


def test_build_framework_jobs_macos_includes_expected_jobs(tmp_path: Path) -> None:
    mac_job_ids = {job.id for job in build_framework_jobs(tmp_path, FrameworkEnvironment.MACOS)}
    expected = {"status", "ingest-runtime", "adapter-invariants", "build-readiness", "improve-loop", "visual-dashboard"}
    assert expected <= mac_job_ids


def test_build_framework_jobs_ubuntu_includes_expected_jobs(tmp_path: Path) -> None:
    ubuntu_job_ids = {job.id for job in build_framework_jobs(tmp_path, FrameworkEnvironment.UBUNTU)}
    expected = {"status", "ingest-runtime", "ubuntu-preflight", "train-adapters", "convert-adapters", "hf-resolve", "hf-upload-dry-run"}
    assert expected <= ubuntu_job_ids


def test_build_framework_jobs_ubuntu_training_jobs_require_confirmation(tmp_path: Path) -> None:
    ubuntu_jobs = {job.id: job for job in build_framework_jobs(tmp_path, FrameworkEnvironment.UBUNTU)}
    assert ubuntu_jobs["train-adapters"].requires_confirmation is True
    assert ubuntu_jobs["convert-adapters"].requires_confirmation is True


def test_build_framework_jobs_ubuntu_hf_jobs_do_not_require_confirmation(tmp_path: Path) -> None:
    ubuntu_jobs = {job.id: job for job in build_framework_jobs(tmp_path, FrameworkEnvironment.UBUNTU)}
    assert ubuntu_jobs["hf-resolve"].requires_confirmation is False


def test_build_framework_jobs_improve_loop_outputs_include_generated(tmp_path: Path) -> None:
    mac_jobs = {job.id: job for job in build_framework_jobs(tmp_path, FrameworkEnvironment.MACOS)}
    outputs = mac_jobs["improve-loop"].outputs
    assert any("generated/agent_manifest" in o for o in outputs)
    assert any("generated/agent_improvement_loop" in o for o in outputs)


def test_build_framework_jobs_commands_include_root_path(tmp_path: Path) -> None:
    mac_jobs = {job.id: job for job in build_framework_jobs(tmp_path, FrameworkEnvironment.MACOS)}
    improve_cmd = mac_jobs["improve-loop"].command
    assert str(tmp_path) in improve_cmd


# ---------------------------------------------------------------------------
# load_framework_snapshot – additional coverage
# ---------------------------------------------------------------------------


def test_framework_snapshot_environment_value_is_returned(tmp_path: Path) -> None:
    snapshot_mac = load_framework_snapshot(tmp_path, FrameworkEnvironment.MACOS)
    snapshot_ubuntu = load_framework_snapshot(tmp_path, FrameworkEnvironment.UBUNTU)
    assert snapshot_mac["environment"] == "macos"
    assert snapshot_ubuntu["environment"] == "ubuntu"


def test_framework_snapshot_root_key_is_string(tmp_path: Path) -> None:
    snapshot = load_framework_snapshot(tmp_path, FrameworkEnvironment.MACOS)
    assert isinstance(snapshot["root"], str)
    assert snapshot["root"] == str(tmp_path.resolve())


def test_framework_snapshot_all_evidence_layers_present(tmp_path: Path) -> None:
    snapshot = load_framework_snapshot(tmp_path, FrameworkEnvironment.MACOS)
    layer_ids = {layer["id"] for layer in snapshot["evidenceLayers"]}
    for layer in EvidenceLayer:
        assert layer.value in layer_ids


def test_framework_snapshot_non_live_layers_do_not_own_pass_fail(tmp_path: Path) -> None:
    snapshot = load_framework_snapshot(tmp_path, FrameworkEnvironment.MACOS)
    for entry in snapshot["evidenceLayers"]:
        if entry["id"] == EvidenceLayer.LIVE_E2E.value:
            assert entry["ownsScenarioPassFail"] is True
        else:
            assert entry["ownsScenarioPassFail"] is False


def test_framework_snapshot_available_jobs_are_serializable(tmp_path: Path) -> None:
    snapshot = load_framework_snapshot(tmp_path, FrameworkEnvironment.MACOS)
    # Should not raise
    serialized = json.dumps(snapshot)
    assert len(serialized) > 0


def test_framework_snapshot_reads_testflight_scenarios(tmp_path: Path) -> None:
    loop = tmp_path / "generated" / "agent_improvement_loop"
    loop.mkdir(parents=True)
    (loop / "testflight_scenarios.jsonl").write_text(
        json.dumps({"scenario": "launch_app", "priority": 1}) + "\n",
        encoding="utf-8",
    )

    snapshot = load_framework_snapshot(tmp_path, FrameworkEnvironment.MACOS)

    assert len(snapshot["testflightScenarios"]) == 1
    assert snapshot["testflightScenarios"][0]["scenario"] == "launch_app"


def test_framework_snapshot_reads_visual_summary(tmp_path: Path) -> None:
    visual = tmp_path / "generated" / "visual_improve_loop"
    visual.mkdir(parents=True)
    (visual / "visual_improve_loop_summary.json").write_text(
        json.dumps({"loopPassed": True, "gapCount": 0}),
        encoding="utf-8",
    )

    snapshot = load_framework_snapshot(tmp_path, FrameworkEnvironment.MACOS)

    assert snapshot["visualSummary"]["loopPassed"] is True


def test_framework_snapshot_reads_hf_manifest(tmp_path: Path) -> None:
    hf_dir = tmp_path / "generated" / "hf_artifacts"
    hf_dir.mkdir(parents=True)
    (hf_dir / "lumen_hf_artifact_manifest.resolved.json").write_text(
        json.dumps({"artifacts": []}),
        encoding="utf-8",
    )

    snapshot = load_framework_snapshot(tmp_path, FrameworkEnvironment.MACOS)

    assert "artifacts" in snapshot["hfManifest"]


def test_framework_snapshot_gaps_capped_at_80(tmp_path: Path) -> None:
    loop = tmp_path / "generated" / "agent_improvement_loop"
    loop.mkdir(parents=True)
    gaps = [{"severity": "warning", "title": f"gap_{i}"} for i in range(100)]
    (loop / "loop_gaps.json").write_text(json.dumps({"gaps": gaps}), encoding="utf-8")

    snapshot = load_framework_snapshot(tmp_path, FrameworkEnvironment.MACOS)

    assert len(snapshot["gaps"]) == 80
    assert snapshot["gapCount"] == 100


def test_framework_snapshot_malformed_gaps_json_returns_zero_gaps(tmp_path: Path) -> None:
    loop = tmp_path / "generated" / "agent_improvement_loop"
    loop.mkdir(parents=True)
    (loop / "loop_gaps.json").write_text("{bad json}", encoding="utf-8")

    snapshot = load_framework_snapshot(tmp_path, FrameworkEnvironment.MACOS)

    assert snapshot["gapCount"] == 0
    assert snapshot["gaps"] == []


def test_framework_snapshot_gaps_json_with_non_list_gaps_returns_zero(tmp_path: Path) -> None:
    loop = tmp_path / "generated" / "agent_improvement_loop"
    loop.mkdir(parents=True)
    (loop / "loop_gaps.json").write_text(json.dumps({"gaps": "not a list"}), encoding="utf-8")

    snapshot = load_framework_snapshot(tmp_path, FrameworkEnvironment.MACOS)

    assert snapshot["gapCount"] == 0


# ---------------------------------------------------------------------------
# analyze_reports
# ---------------------------------------------------------------------------


def test_analyze_reports_schema_version(tmp_path: Path) -> None:
    analysis = analyze_reports(tmp_path, [])
    assert analysis["schemaVersion"] == "lumen.developer_framework.report_analysis/1.0.0"


def test_analyze_reports_empty_paths_returns_zero_reports(tmp_path: Path) -> None:
    analysis = analyze_reports(tmp_path, [])
    assert analysis["reportCount"] == 0
    assert analysis["runtimeFailureCount"] == 0
    assert analysis["plainFindings"] == []


def test_analyze_reports_flags_python_traceback(tmp_path: Path) -> None:
    report = tmp_path / "error.log"
    report.write_text("Traceback (most recent call last):\n  File foo.py\nValueError: bad", encoding="utf-8")

    analysis = analyze_reports(tmp_path, [report])

    types = [f["type"] for f in analysis["plainFindings"]]
    assert "python_traceback" in types


def test_analyze_reports_flags_xcodebuild_failure(tmp_path: Path) -> None:
    report = tmp_path / "build.log"
    report.write_text("xcodebuild: Build FAILED\nerror: compilation error", encoding="utf-8")

    analysis = analyze_reports(tmp_path, [report])

    types = [f["type"] for f in analysis["plainFindings"]]
    assert "xcodebuild_failure" in types


def test_analyze_reports_flags_hf_upload_failure(tmp_path: Path) -> None:
    report = tmp_path / "hf.log"
    report.write_text("Hugging Face upload failed with error 403", encoding="utf-8")

    analysis = analyze_reports(tmp_path, [report])

    types = [f["type"] for f in analysis["plainFindings"]]
    assert "hf_upload_failure" in types


def test_analyze_reports_flags_routing_only_checks(tmp_path: Path) -> None:
    report = tmp_path / "e2e.log"
    report.write_text("routing-only checks completed", encoding="utf-8")

    analysis = analyze_reports(tmp_path, [report])

    types = [f["type"] for f in analysis["plainFindings"]]
    assert "invalid_live_e2e_no_model" in types


def test_analyze_reports_clean_file_has_no_findings(tmp_path: Path) -> None:
    report = tmp_path / "clean.log"
    report.write_text("All checks passed successfully.", encoding="utf-8")

    analysis = analyze_reports(tmp_path, [report])

    assert analysis["plainFindings"] == []


def test_analyze_reports_finding_source_is_file_path(tmp_path: Path) -> None:
    report = tmp_path / "err.log"
    report.write_text("Traceback (most recent call last):", encoding="utf-8")

    analysis = analyze_reports(tmp_path, [report])

    assert analysis["plainFindings"][0]["source"] == str(report)


def test_analyze_reports_scan_files_in_directory(tmp_path: Path) -> None:
    subdir = tmp_path / "logs"
    subdir.mkdir()
    (subdir / "nested.log").write_text("No model loaded", encoding="utf-8")

    analysis = analyze_reports(tmp_path, [subdir])

    types = [f["type"] for f in analysis["plainFindings"]]
    assert "invalid_live_e2e_no_model" in types


def test_analyze_reports_nonexistent_path_is_ignored(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist.log"
    analysis = analyze_reports(tmp_path, [missing])
    assert analysis["plainFindings"] == []


def test_analyze_reports_relative_paths_resolved_against_root(tmp_path: Path) -> None:
    report = tmp_path / "trace.log"
    report.write_text("Traceback (most recent call last):", encoding="utf-8")

    analysis = analyze_reports(tmp_path, [Path("trace.log")])

    types = [f["type"] for f in analysis["plainFindings"]]
    assert "python_traceback" in types


def test_analyze_reports_multiple_findings_in_one_file(tmp_path: Path) -> None:
    report = tmp_path / "multi.log"
    report.write_text(
        "Traceback (most recent call last):\nxcodebuild error: build failed",
        encoding="utf-8",
    )

    analysis = analyze_reports(tmp_path, [report])

    types = {f["type"] for f in analysis["plainFindings"]}
    assert "python_traceback" in types
    assert "xcodebuild_failure" in types


def test_analyze_reports_severity_error_for_traceback(tmp_path: Path) -> None:
    report = tmp_path / "tb.log"
    report.write_text("Traceback (most recent call last):", encoding="utf-8")

    analysis = analyze_reports(tmp_path, [report])

    finding = next(f for f in analysis["plainFindings"] if f["type"] == "python_traceback")
    assert finding["severity"] == "error"


def test_analyze_reports_severity_warning_for_hf_failure(tmp_path: Path) -> None:
    report = tmp_path / "hf.log"
    report.write_text("Hugging Face upload failed", encoding="utf-8")

    analysis = analyze_reports(tmp_path, [report])

    finding = next(f for f in analysis["plainFindings"] if f["type"] == "hf_upload_failure")
    assert finding["severity"] == "warning"


# ---------------------------------------------------------------------------
# FrameworkJobRunner
# ---------------------------------------------------------------------------


def test_framework_job_runner_initial_snapshot_is_idle(tmp_path: Path) -> None:
    runner = FrameworkJobRunner(tmp_path, FrameworkEnvironment.MACOS)
    snap = runner.snapshot()
    assert snap["status"] == "idle"
    assert snap["jobID"] is None


def test_framework_job_runner_returns_error_for_unknown_job(tmp_path: Path) -> None:
    runner = FrameworkJobRunner(tmp_path, FrameworkEnvironment.MACOS)
    ok, message = runner.start("nonexistent-job-id")
    assert ok is False
    assert "Unknown framework job" in message


def test_framework_job_runner_rejects_concurrent_start(tmp_path: Path) -> None:
    runner = FrameworkJobRunner(tmp_path, FrameworkEnvironment.MACOS)
    # Manually set state to running to simulate in-progress job
    with runner.lock:
        runner.state.status = "running"

    ok, message = runner.start("status")
    assert ok is False
    assert "already running" in message.lower()


def test_framework_job_runner_start_known_job_returns_started(tmp_path: Path) -> None:
    runner = FrameworkJobRunner(tmp_path, FrameworkEnvironment.MACOS)
    ok, message = runner.start("status")
    assert ok is True
    assert message == "started"
    # Wait briefly for the job to progress
    time.sleep(0.3)
    snap = runner.snapshot()
    assert snap["jobID"] == "status"
    assert snap["status"] in {"running", "passed", "failed"}


# ---------------------------------------------------------------------------
# run_framework_job
# ---------------------------------------------------------------------------


def test_run_framework_job_raises_value_error_for_unknown_job(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unknown framework job"):
        run_framework_job(tmp_path, "nonexistent-job-id", FrameworkEnvironment.MACOS)


# ---------------------------------------------------------------------------
# _index_html
# ---------------------------------------------------------------------------


def test_index_html_contains_expected_page_structure(tmp_path: Path) -> None:
    html = _index_html(tmp_path, FrameworkEnvironment.MACOS)
    assert "<!doctype html>" in html
    assert "<title>Lumen Developer Framework</title>" in html
    assert "Whitelisted Jobs" in html
    assert "Evidence Layers" in html
    assert "Gaps" in html


def test_index_html_escapes_root_path(tmp_path: Path) -> None:
    # Use a path with characters that must be HTML-escaped
    evil_path = tmp_path / "root<script>"
    html = _index_html(evil_path, FrameworkEnvironment.MACOS)
    assert "<script>" not in html
    assert "&lt;script&gt;" in html


def test_index_html_includes_environment_value(tmp_path: Path) -> None:
    html_mac = _index_html(tmp_path, FrameworkEnvironment.MACOS)
    html_ubuntu = _index_html(tmp_path, FrameworkEnvironment.UBUNTU)
    assert "macos" in html_mac
    assert "ubuntu" in html_ubuntu


def test_index_html_contains_api_endpoints(tmp_path: Path) -> None:
    html = _index_html(tmp_path, FrameworkEnvironment.MACOS)
    assert "/status.json" in html
    assert "/jobs.json" in html
    assert "/run/" in html


# ---------------------------------------------------------------------------
# Regression / boundary tests
# ---------------------------------------------------------------------------


def test_jsonl_reader_skips_empty_lines_and_non_dict_values(tmp_path: Path) -> None:
    """Ensure load_framework_snapshot ignores blank lines and non-dict JSONL entries."""
    loop = tmp_path / "generated" / "agent_improvement_loop"
    loop.mkdir(parents=True)
    content = (
        "\n"
        + json.dumps({"taskType": "valid"}) + "\n"
        + "\n"
        + json.dumps(["not", "a", "dict"]) + "\n"
        + json.dumps({"taskType": "also_valid"}) + "\n"
    )
    (loop / "next_action_prompts.jsonl").write_text(content, encoding="utf-8")

    snapshot = load_framework_snapshot(tmp_path, FrameworkEnvironment.MACOS)

    assert len(snapshot["nextActionPrompts"]) == 2
    assert snapshot["nextActionPrompts"][0]["taskType"] == "valid"
    assert snapshot["nextActionPrompts"][1]["taskType"] == "also_valid"


def test_jsonl_reader_respects_limit(tmp_path: Path) -> None:
    """Scenarios list should be capped at 80 records."""
    loop = tmp_path / "generated" / "agent_improvement_loop"
    loop.mkdir(parents=True)
    lines = [json.dumps({"id": i}) for i in range(100)]
    (loop / "testflight_scenarios.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")

    snapshot = load_framework_snapshot(tmp_path, FrameworkEnvironment.MACOS)

    assert len(snapshot["testflightScenarios"]) == 80


def test_build_framework_jobs_all_have_non_empty_command(tmp_path: Path) -> None:
    for env in (FrameworkEnvironment.MACOS, FrameworkEnvironment.UBUNTU):
        for job in build_framework_jobs(tmp_path, env):
            assert len(job.command) > 0, f"Job {job.id} has empty command"


def test_build_framework_jobs_all_ids_are_unique(tmp_path: Path) -> None:
    for env in (FrameworkEnvironment.MACOS, FrameworkEnvironment.UBUNTU):
        jobs = build_framework_jobs(tmp_path, env)
        ids = [job.id for job in jobs]
        assert len(ids) == len(set(ids)), f"Duplicate job IDs in {env} profile"
