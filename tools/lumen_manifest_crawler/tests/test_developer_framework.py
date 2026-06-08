from __future__ import annotations

import json
from pathlib import Path

from lumen_manifest_crawler.developer_framework import (
    EvidenceLayer,
    FrameworkEnvironment,
    UBUNTU_TRAINING_JOB_IDS,
    _is_loopback_host,
    analyze_reports,
    build_framework_jobs,
    load_framework_snapshot,
    resolve_environment,
)


def test_framework_jobs_include_macos_and_ubuntu_profiles(tmp_path: Path) -> None:
    mac_jobs = {job.id: job for job in build_framework_jobs(tmp_path, FrameworkEnvironment.MACOS)}
    ubuntu_jobs = {job.id: job for job in build_framework_jobs(tmp_path, FrameworkEnvironment.UBUNTU)}

    assert "adapter-invariants" in mac_jobs
    assert "improve-loop" in mac_jobs
    assert mac_jobs["improve-loop"].evidence_layer == EvidenceLayer.STATIC_SOURCE

    assert "ubuntu-preflight" in ubuntu_jobs
    assert "train-adapters" in ubuntu_jobs
    assert "hf-resolve" in ubuntu_jobs
    assert "hf-upload-dry-run" in ubuntu_jobs
    assert ubuntu_jobs["train-adapters"].requires_confirmation is True
    assert UBUNTU_TRAINING_JOB_IDS[-1] == "hf-upload-dry-run"


def test_framework_snapshot_is_valid_without_generated_artifacts(tmp_path: Path) -> None:
    snapshot = load_framework_snapshot(tmp_path, FrameworkEnvironment.MACOS)

    assert snapshot["schemaVersion"] == "lumen.developer_framework/1.0.0"
    assert snapshot["authoritativeLiveLayer"] == EvidenceLayer.LIVE_E2E.value
    assert snapshot["gapCount"] == 0
    live = [layer for layer in snapshot["evidenceLayers"] if layer["id"] == EvidenceLayer.LIVE_E2E.value][0]
    assert live["ownsScenarioPassFail"] is True
    assert snapshot["adapterRuntime"]["runtimeShape"] == "one_shared_chat_base_plus_role_lora_adapters"
    assert {role["id"] for role in snapshot["adapterRuntime"]["roles"]} == {"cortex", "executor", "mouth", "mimicry", "rem", "fleet"}
    assert "runtime_trace_presence" in snapshot["adapterRuntime"]["promotionGates"]


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


def test_analyze_reports_uses_narrow_hf_upload_failure_heuristic(tmp_path: Path) -> None:
    noisy = tmp_path / "hf-noisy.log"
    noisy.write_text("Hugging Face model card says prior training failed but no upload was attempted.", encoding="utf-8")
    upload = tmp_path / "hf-upload.log"
    upload.write_text("huggingface.co upload failed: token expired", encoding="utf-8")

    analysis = analyze_reports(tmp_path, [noisy, upload])

    hf_findings = [finding for finding in analysis["plainFindings"] if finding["type"] == "hf_upload_failure"]
    assert len(hf_findings) == 1
    assert hf_findings[0]["source"] == str(upload)


def test_loopback_host_detection_rejects_public_bindings() -> None:
    assert _is_loopback_host("127.0.0.1") is True
    assert _is_loopback_host("localhost") is True
    assert _is_loopback_host("0.0.0.0") is False


def test_resolve_environment_accepts_explicit_values() -> None:
    assert resolve_environment("macos") == FrameworkEnvironment.MACOS
    assert resolve_environment("ubuntu") == FrameworkEnvironment.UBUNTU
