"""Unified developer-cycle orchestration for Lumen.

The developer cycle is intentionally a coordinator. Manifest generation,
dataset compilation, runtime-audit ingestion, and improvement-loop outputs stay
owned by the existing crawler/framework modules.
"""

from __future__ import annotations

import json
import platform
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from lumen_manifest_crawler.developer_framework import (
    FrameworkEnvironment,
    UBUNTU_TRAINING_JOB_IDS,
    analyze_reports,
    build_framework_jobs,
    run_framework_job,
)
from lumen_manifest_crawler.improvement_loop import AgentImprovementLoopConfig, run_agent_improvement_loop


REPORT_SCHEMA_VERSION = "lumen.developer_cycle/1.0.0"
DEFAULT_FRAMEWORK_DIR = Path("generated/developer_framework")
DEFAULT_MANIFEST_DIR = Path("generated/agent_manifest")
DEFAULT_LOOP_DIR = Path("generated/agent_improvement_loop")
TRAINING_JOB_IDS = UBUNTU_TRAINING_JOB_IDS


@dataclass(frozen=True)
class DeveloperCycleConfig:
    root: Path
    runtime_audit_paths: tuple[Path, ...] = ()
    portable: bool = False
    with_xcode: bool = False
    with_training_plan: bool = False
    run_training: bool = False
    fail_on_gaps: bool = False
    fail_on_static: bool = False
    fail_on_validation: bool = False
    require_runtime_audit: bool = False
    json_output: bool = False
    dry_run: bool = False
    skip_generation: bool = False
    skip_improvement_loop: bool = False


def run_developer_cycle(config: DeveloperCycleConfig) -> tuple[dict[str, Any], int]:
    root = config.root.resolve()
    framework_dir = root / DEFAULT_FRAMEWORK_DIR
    framework_dir.mkdir(parents=True, exist_ok=True)

    environment = detect_environment(root)
    phases: list[dict[str, Any]] = [_environment_phase(environment)]

    static_phase = _run_static_validation(root, environment, dry_run=config.dry_run)
    phases.append(static_phase)
    static_passed = _phase_passed(static_phase)

    manifest_phase, loop_result = _run_generation_and_loop(root, config)
    phases.append(manifest_phase)

    runtime_phase = _run_runtime_ingestion(root, config.runtime_audit_paths, dry_run=config.dry_run)
    phases.append(runtime_phase)

    improvement_phase = _summarize_improvement_loop(root, config, loop_result)
    phases.append(improvement_phase)

    xcode_phase = _run_xcode_validation(root, environment, config)
    phases.append(xcode_phase)

    training_phase = _run_training_profile(root, config)
    phases.append(training_phase)

    manifest_validation_passed = bool(manifest_phase.get("manifestValidationPassed", False))
    runtime_evidence_present = bool(runtime_phase.get("runtimeEvidencePresent", False))
    improvement_loop_passed = bool(improvement_phase.get("improvementLoopPassed", False))
    improvement_loop_contract_passed = bool(improvement_phase.get("improvementLoopOutputContractPassed", False))
    xcode_status = str(xcode_phase.get("xcodeValidationStatus", "skipped"))
    training_status = str(training_phase.get("trainingStatus", "not_requested"))

    overall_portable_passed = (
        static_passed
        and manifest_validation_passed
        and improvement_loop_passed
        and improvement_loop_contract_passed
        and training_status not in {"failed"}
    )
    overall_release_candidate_passed = (
        overall_portable_passed
        and xcode_status == "passed"
        and runtime_evidence_present
    )

    report: dict[str, Any] = {
        "schemaVersion": REPORT_SCHEMA_VERSION,
        "root": str(root),
        "environment": environment,
        "phases": phases,
        "staticValidationPassed": static_passed,
        "manifestValidationPassed": manifest_validation_passed,
        "runtimeEvidencePresent": runtime_evidence_present,
        "improvementLoopPassed": improvement_loop_passed,
        "improvementLoopOutputContractPassed": improvement_loop_contract_passed,
        "xcodeValidationStatus": xcode_status,
        "trainingStatus": training_status,
        "overallPortablePassed": overall_portable_passed,
        "overallReleaseCandidatePassed": overall_release_candidate_passed,
        "runtimeFailures": {
            "runtimeFailureCount": runtime_phase.get("runtimeFailureCount", 0),
            "rawRuntimeFailureCount": improvement_phase.get("rawRuntimeFailureCount", 0),
            "skippedLiveModelGenerationCount": improvement_phase.get("skippedLiveModelGenerationCount", 0),
        },
        "outputArtifacts": _output_artifacts(root),
        "nextRecommendedCommand": _next_recommended_command(report_flags={
            "static": static_passed,
            "manifest": manifest_validation_passed,
            "runtime": runtime_evidence_present,
            "loop": improvement_loop_passed,
            "xcode": xcode_status,
        }),
        "options": {
            "portable": config.portable,
            "withXcode": config.with_xcode,
            "withTrainingPlan": config.with_training_plan,
            "runTraining": config.run_training,
            "failOnGaps": config.fail_on_gaps,
            "failOnStatic": config.fail_on_static,
            "failOnValidation": config.fail_on_validation,
            "requireRuntimeAudit": config.require_runtime_audit,
            "dryRun": config.dry_run,
            "skipGeneration": config.skip_generation,
            "skipImprovementLoop": config.skip_improvement_loop,
            "runtimeAuditPaths": [str(path) for path in config.runtime_audit_paths],
        },
    }

    _write_json(framework_dir / "developer_cycle_report.json", report)
    _write_markdown_report(framework_dir / "DEVELOPER_CYCLE_REPORT.md", report)

    exit_code = _exit_code_for_report(report, config)
    return report, exit_code


def detect_environment(root: Path) -> dict[str, Any]:
    system = platform.system()
    git = _git_worktree_status(root)
    xcodebuild_path = shutil.which("xcodebuild")
    is_macos = system == "Darwin"
    is_linux = system == "Linux"
    has_xcode = bool(is_macos and xcodebuild_path)
    if not git["insideWorktree"]:
        checkout_kind = "non_git_zip_or_export"
    else:
        checkout_kind = "git_checkout"
    if is_linux:
        execution_kind = "linux_codex_static"
    elif is_macos and has_xcode:
        execution_kind = "macos_with_xcode"
    elif is_macos:
        execution_kind = "macos_without_xcode"
    else:
        execution_kind = "static_unknown_platform"
    return {
        "checkoutKind": checkout_kind,
        "executionKind": execution_kind,
        "insideGitWorktree": git["insideWorktree"],
        "gitTopLevel": git["topLevel"],
        "platformSystem": system,
        "platformMachine": platform.machine(),
        "pythonExecutable": sys.executable,
        "xcodebuildPath": xcodebuild_path,
        "hasXcodebuild": bool(xcodebuild_path),
    }


def _run_static_validation(root: Path, environment: dict[str, Any], *, dry_run: bool) -> dict[str, Any]:
    commands: list[tuple[str, list[str]]] = [
        ("agent_kernel_boundary", ["python3", "tools/check_agent_kernel_boundary.py"]),
        ("agent_kernel_boundary_strict", ["python3", "tools/check_agent_kernel_boundary.py", "--strict"]),
        ("adapter_runtime_invariants", ["python3", "tools/check_adapter_runtime_invariants.py"]),
        ("ios_lora_hardening_invariants", ["python3", "tools/check_ios_lora_hardening_invariants.py"]),
        ("msal_release_config", ["python3", "scripts/validate-msal-ios-release-config.py"]),
        ("ios_signing_capabilities", ["python3", "scripts/validate_ios_signing_capabilities.py"]),
        ("no_shell_subprocess", ["python3", "tools/security/check_no_shell_subprocess.py"]),
        ("ios_build_readiness", ["bash", "scripts/check-ios-build-readiness.sh"]),
    ]
    results = [_run_command(name, command, root, dry_run=dry_run) for name, command in commands]
    if environment.get("insideGitWorktree"):
        results.append(_run_command("git_diff_check", ["git", "diff", "--check"], root, dry_run=dry_run))
    else:
        results.append(_skipped_command("git_diff_check", ["git", "diff", "--check"], root, "not inside a git worktree"))
    return {
        "id": "phase1_static_source_validation",
        "title": "Phase 1 - Static Source Validation",
        "status": "passed" if all(result["status"] in {"passed", "skipped"} for result in results) else "failed",
        "commands": results,
        "staticValidationPassed": all(result["status"] in {"passed", "skipped"} for result in results),
    }


def _environment_phase(environment: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": "phase0_environment_detection",
        "title": "Phase 0 - Environment Detection",
        "status": "passed",
        "environment": environment,
        "commands": [],
    }


def _run_generation_and_loop(root: Path, config: DeveloperCycleConfig) -> tuple[dict[str, Any], Any | None]:
    if config.skip_generation or config.skip_improvement_loop:
        reason = "generation skipped by --skip-generation" if config.skip_generation else "improvement loop skipped by --skip-improvement-loop"
        return (
            {
                "id": "phase2_manifest_dataset_generation",
                "title": "Phase 2 - Manifest and Dataset Generation",
                "status": "skipped",
                "reason": reason,
                "commands": [_skipped_command("improve_loop_generation", _improve_loop_command(root, config), root, reason)],
                "manifestValidationPassed": _existing_manifest_validation_passed(root),
            },
            None,
        )
    command = _improve_loop_command(root, config)
    if config.dry_run:
        return (
            {
                "id": "phase2_manifest_dataset_generation",
                "title": "Phase 2 - Manifest and Dataset Generation",
                "status": "skipped",
                "reason": "dry-run",
                "commands": [_skipped_command("improve_loop_generation", command, root, "dry-run")],
                "manifestValidationPassed": _existing_manifest_validation_passed(root),
            },
            None,
        )

    result = run_agent_improvement_loop(
        AgentImprovementLoopConfig(
            root=root,
            output=root / DEFAULT_MANIFEST_DIR,
            loop_output=root / DEFAULT_LOOP_DIR,
            runtime_audit_paths=tuple(_runtime_audit_inputs(root, config.runtime_audit_paths)),
            deterministic=True,
            pretty=True,
            strict=True,
            generate_system_prompts=True,
            generate_agent_fine_tuning=True,
            require_testflight_runtime_audit=config.require_runtime_audit,
        )
    )
    validation = result.state.get("validation", {}) if isinstance(result.state, dict) else {}
    manifest_passed = int(validation.get("failureCount") or 0) == 0
    return (
        {
            "id": "phase2_manifest_dataset_generation",
            "title": "Phase 2 - Manifest and Dataset Generation",
            "status": "passed" if manifest_passed else "failed",
            "commands": [_recorded_command("improve_loop_generation", command, root, 0, "called run_agent_improvement_loop()", "")],
            "manifestValidationPassed": manifest_passed,
            "validation": validation,
            "outputArtifacts": _manifest_artifacts(root),
        },
        result,
    )


def _run_runtime_ingestion(root: Path, runtime_audit_paths: Sequence[Path], *, dry_run: bool) -> dict[str, Any]:
    paths = _runtime_audit_inputs(root, tuple(runtime_audit_paths))
    command = [
        "python3",
        "-m",
        "lumen_manifest_crawler",
        "framework",
        "diagnose",
        "--root",
        str(root),
        "--output",
        str(root / DEFAULT_FRAMEWORK_DIR / "framework_report.json"),
    ]
    for path in paths:
        command.extend(["--path", str(path)])
    if dry_run:
        report = {
            "schemaVersion": "lumen.developer_framework.report_analysis/1.0.0",
            "reportCount": 0,
            "runtimeFailureCount": 0,
            "runtimeFailures": [],
            "plainFindings": [],
        }
        status = "skipped"
        command_result = _skipped_command("framework_diagnose", command, root, "dry-run")
    else:
        report = analyze_reports(root, paths)
        output = root / DEFAULT_FRAMEWORK_DIR / "framework_report.json"
        _write_json(output, report)
        _write_json(root / DEFAULT_FRAMEWORK_DIR / "runtime_report_index.json", _runtime_report_index(paths, report))
        status = "passed"
        command_result = _recorded_command("framework_diagnose", command, root, 0, "called analyze_reports()", "")
    return {
        "id": "phase3_runtime_audit_ingestion",
        "title": "Phase 3 - Runtime-Audit/Report Ingestion",
        "status": status,
        "commands": [command_result],
        "inputPaths": [str(path) for path in paths],
        "runtimeEvidencePresent": int(report.get("reportCount") or 0) > 0,
        "runtimeFailureCount": int(report.get("runtimeFailureCount") or 0),
        "plainFindingCount": len(report.get("plainFindings") or []),
        "outputArtifacts": [
            str(root / DEFAULT_FRAMEWORK_DIR / "framework_report.json"),
            str(root / DEFAULT_FRAMEWORK_DIR / "runtime_report_index.json"),
        ],
    }


def _summarize_improvement_loop(root: Path, config: DeveloperCycleConfig, loop_result: Any | None) -> dict[str, Any]:
    command = _improve_loop_command(root, config)
    state = loop_result.state if loop_result is not None else _read_json(root / DEFAULT_LOOP_DIR / "loop_state.json") or {}
    contract = _improvement_loop_output_contract(root, state)
    if config.skip_improvement_loop:
        return {
            "id": "phase4_improvement_loop_preparation",
            "title": "Phase 4 - Improvement-Loop Preparation",
            "status": "skipped",
            "reason": "skipped by --skip-improvement-loop",
            "commands": [_skipped_command("improve_loop", command, root, "skipped by --skip-improvement-loop")],
            "improvementLoopPassed": _existing_loop_passed(root),
            "improvementLoopOutputContractPassed": contract["passed"],
            "improvementLoopOutputContract": contract,
            "gapCount": _existing_gap_count(root),
            "rawRuntimeFailureCount": 0,
            "skippedLiveModelGenerationCount": 0,
            "outputArtifacts": _loop_artifacts(root),
        }
    runtime = state.get("runtime", {}) if isinstance(state, dict) else {}
    skipped_live_count = int(runtime.get("skippedLiveModelGenerationCount") or 0)
    passed = bool(state.get("passed", False)) and skipped_live_count == 0
    if loop_result is None and not state:
        status = "skipped"
        command_result = _skipped_command("improve_loop", command, root, "no loop execution in this run")
    else:
        status = "passed" if passed else "failed"
        command_result = _recorded_command("improve_loop", command, root, 0, "called run_agent_improvement_loop()", "")
    return {
        "id": "phase4_improvement_loop_preparation",
        "title": "Phase 4 - Improvement-Loop Preparation",
        "status": status,
        "commands": [command_result],
        "improvementLoopPassed": passed,
        "improvementLoopOutputContractPassed": contract["passed"],
        "improvementLoopOutputContract": contract,
        "gapCount": int(state.get("gapCount") or 0) if isinstance(state, dict) else 0,
        "criticalGapCount": int(state.get("criticalGapCount") or 0) if isinstance(state, dict) else 0,
        "errorGapCount": int(state.get("errorGapCount") or 0) if isinstance(state, dict) else 0,
        "runtimeEvidencePresent": int(runtime.get("reportCount") or 0) > 0,
        "runtimeFailureCount": int(runtime.get("failureCount") or 0),
        "rawRuntimeFailureCount": int(runtime.get("rawFailureCount") or 0),
        "skippedLiveModelGenerationCount": skipped_live_count,
        "outputArtifacts": _loop_artifacts(root),
    }


def _run_xcode_validation(root: Path, environment: dict[str, Any], config: DeveloperCycleConfig) -> dict[str, Any]:
    command = ["bash", "scripts/validate_lumen_ios.sh"]
    if config.portable:
        return _xcode_phase("skipped", [_skipped_command("xcode_validation", command, root, "portable mode")], "portable mode")
    has_xcode = bool(environment.get("hasXcodebuild"))
    is_macos = environment.get("platformSystem") == "Darwin"
    if not has_xcode:
        reason = "xcodebuild unavailable"
        if config.with_xcode:
            return _xcode_phase("failed", [_recorded_command("xcode_validation", command, root, 127, "", reason)], reason)
        return _xcode_phase("skipped", [_skipped_command("xcode_validation", command, root, reason)], reason)
    if not is_macos:
        reason = "xcodebuild exists but platform is not macOS"
        if config.with_xcode:
            return _xcode_phase("failed", [_recorded_command("xcode_validation", command, root, 2, "", reason)], reason)
        return _xcode_phase("skipped", [_skipped_command("xcode_validation", command, root, reason)], reason)
    if config.dry_run:
        return _xcode_phase("skipped", [_skipped_command("xcode_validation", command, root, "dry-run")], "dry-run")
    result = _run_command("xcode_validation", command, root, dry_run=False)
    return _xcode_phase("passed" if result["status"] == "passed" else "failed", [result], "")


def _run_training_profile(root: Path, config: DeveloperCycleConfig) -> dict[str, Any]:
    requested = config.with_training_plan or config.run_training
    if not requested:
        return {
            "id": "phase6_training_hf_profile",
            "title": "Phase 6 - Optional Training/HF Artifact Profile",
            "status": "skipped",
            "trainingStatus": "not_requested",
            "reason": "training profile is opt-in",
            "jobs": [],
            "commands": [],
        }
    jobs_by_id = {job.id: job for job in build_framework_jobs(root, FrameworkEnvironment.UBUNTU)}
    jobs = [jobs_by_id[job_id] for job_id in TRAINING_JOB_IDS if job_id in jobs_by_id]
    if config.with_training_plan and not config.run_training:
        return {
            "id": "phase6_training_hf_profile",
            "title": "Phase 6 - Optional Training/HF Artifact Profile",
            "status": "planned",
            "trainingStatus": "planned",
            "jobs": [job.output_dict() for job in jobs],
            "commands": [_skipped_command(job.id, list(job.command), root, "planned only") for job in jobs],
        }
    results: list[dict[str, Any]] = []
    if config.dry_run:
        results = [_skipped_command(job.id, list(job.command), root, "dry-run") for job in jobs]
        return {
            "id": "phase6_training_hf_profile",
            "title": "Phase 6 - Optional Training/HF Artifact Profile",
            "status": "planned",
            "trainingStatus": "planned",
            "jobs": [job.output_dict() for job in jobs],
            "commands": results,
        }
    for job in jobs:
        started = time.time()
        returncode = run_framework_job(root, job.id, FrameworkEnvironment.UBUNTU)
        results.append(
            _recorded_command(
                job.id,
                list(job.command),
                root,
                returncode,
                "",
                "",
                duration_seconds=round(time.time() - started, 2),
            )
        )
        if returncode != 0:
            break
    passed = all(result.get("status") == "passed" for result in results)
    return {
        "id": "phase6_training_hf_profile",
        "title": "Phase 6 - Optional Training/HF Artifact Profile",
        "status": "passed" if passed else "failed",
        "trainingStatus": "passed" if passed else "failed",
        "jobs": [job.output_dict() for job in jobs],
        "commands": results,
    }


def _run_command(name: str, command: Sequence[str], cwd: Path, *, dry_run: bool) -> dict[str, Any]:
    if dry_run:
        return _skipped_command(name, command, cwd, "dry-run")
    started = time.time()
    try:
        completed = subprocess.run(
            list(command),
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=None,
        )
        return _recorded_command(
            name,
            command,
            cwd,
            completed.returncode,
            completed.stdout,
            completed.stderr,
            duration_seconds=round(time.time() - started, 2),
        )
    except OSError as exc:
        return _recorded_command(
            name,
            command,
            cwd,
            127,
            "",
            str(exc),
            duration_seconds=round(time.time() - started, 2),
        )


def _recorded_command(
    name: str,
    command: Sequence[str],
    cwd: Path,
    returncode: int,
    stdout: str,
    stderr: str,
    *,
    duration_seconds: float = 0.0,
) -> dict[str, Any]:
    return {
        "name": name,
        "command": list(command),
        "commandText": shlex.join(list(command)),
        "cwd": str(cwd),
        "returncode": returncode,
        "status": "passed" if returncode == 0 else "failed",
        "durationSeconds": duration_seconds,
        "stdoutTail": _tail(stdout),
        "stderrTail": _tail(stderr),
    }


def _skipped_command(name: str, command: Sequence[str], cwd: Path, reason: str) -> dict[str, Any]:
    return {
        "name": name,
        "command": list(command),
        "commandText": shlex.join(list(command)),
        "cwd": str(cwd),
        "returncode": None,
        "status": "skipped",
        "skipReason": reason,
        "durationSeconds": 0.0,
        "stdoutTail": "",
        "stderrTail": "",
    }


def _xcode_phase(status: str, commands: list[dict[str, Any]], reason: str) -> dict[str, Any]:
    phase: dict[str, Any] = {
        "id": "phase5_xcode_validation",
        "title": "Phase 5 - Optional macOS/Xcode Validation",
        "status": status,
        "xcodeValidationStatus": status,
        "commands": commands,
    }
    if reason:
        phase["reason"] = reason
    return phase


def _runtime_audit_inputs(root: Path, runtime_audit_paths: Sequence[Path]) -> list[Path]:
    # Keep runtime evidence inputs separate from generated loop outputs. The
    # improve-loop ingester expects exported app/runtime audit payloads, not the
    # LOOP_REPORT, gap files, scenario queues, or command logs it generates.
    defaults = [root / "exports", root / "runtime-audits"]
    explicit = [path if path.is_absolute() else root / path for path in runtime_audit_paths]
    return [*defaults, *explicit]


def _runtime_audit_defaults(root: Path, runtime_audit_paths: Sequence[Path]) -> list[Path]:
    return _runtime_audit_inputs(root, runtime_audit_paths)


def _improve_loop_command(root: Path, config: DeveloperCycleConfig) -> list[str]:
    command = [
        "python3",
        "-m",
        "lumen_manifest_crawler",
        "improve-loop",
        "--root",
        str(root),
        "--output",
        str(root / DEFAULT_MANIFEST_DIR),
        "--loop-output",
        str(root / DEFAULT_LOOP_DIR),
        "--generate-system-prompts",
        "--generate-agent-fine-tuning",
    ]
    for path in _runtime_audit_inputs(root, config.runtime_audit_paths):
        command.extend(["--runtime-audit", str(path)])
    if config.require_runtime_audit:
        command.append("--require-testflight-runtime-audit")
    return command


def _git_worktree_status(root: Path) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return {"insideWorktree": False, "topLevel": None}
    if completed.returncode != 0:
        return {"insideWorktree": False, "topLevel": None}
    return {"insideWorktree": True, "topLevel": completed.stdout.strip() or None}


def _phase_passed(phase: dict[str, Any]) -> bool:
    commands = phase.get("commands") if isinstance(phase.get("commands"), list) else []
    return all(command.get("status") in {"passed", "skipped", "planned"} for command in commands if isinstance(command, dict))


def _existing_manifest_validation_passed(root: Path) -> bool:
    payload = _read_json(root / DEFAULT_MANIFEST_DIR / "manifest_validation_report.json")
    if not isinstance(payload, dict):
        return False
    failures = payload.get("failures")
    return isinstance(failures, list) and len(failures) == 0


def _existing_loop_passed(root: Path) -> bool:
    payload = _read_json(root / DEFAULT_LOOP_DIR / "loop_state.json")
    return bool(isinstance(payload, dict) and payload.get("passed") is True)


def _existing_gap_count(root: Path) -> int:
    payload = _read_json(root / DEFAULT_LOOP_DIR / "loop_gaps.json")
    gaps = payload.get("gaps") if isinstance(payload, dict) else []
    return len(gaps) if isinstance(gaps, list) else 0


def _runtime_report_index(paths: Iterable[Path], report: dict[str, Any]) -> dict[str, Any]:
    return {
        "schemaVersion": "lumen.developer_cycle.runtime_report_index/1.0.0",
        "inputPaths": [str(path) for path in paths],
        "reportCount": report.get("reportCount", 0),
        "runtimeFailureCount": report.get("runtimeFailureCount", 0),
        "plainFindingCount": len(report.get("plainFindings") or []),
    }


def _improvement_loop_output_contract(root: Path, state: Any) -> dict[str, Any]:
    required_files = _loop_artifacts(root)
    missing_files = [path for path in required_files if not Path(path).exists()]
    required_state_keys = {
        "schemaVersion",
        "runtimeAuditInputs",
        "runtime",
        "testFlight",
        "validation",
        "gapCount",
        "passed",
    }
    state_keys = set(state.keys()) if isinstance(state, dict) else set()
    missing_state_keys = sorted(required_state_keys - state_keys)
    testflight = state.get("testFlight") if isinstance(state, dict) and isinstance(state.get("testFlight"), dict) else {}
    next_ingest_command = str(testflight.get("nextIngestCommand") or "")
    scenario_queue_path = str(testflight.get("scenarioQueuePath") or "")
    contract_failures = []
    if missing_files:
        contract_failures.append("missing required loop output files")
    if missing_state_keys:
        contract_failures.append("loop_state.json is missing required keys")
    if "improve-loop" not in next_ingest_command or "--runtime-audit" not in next_ingest_command:
        contract_failures.append("testFlight.nextIngestCommand does not point to improve-loop runtime-audit ingestion")
    if scenario_queue_path != "testflight_scenarios.jsonl":
        contract_failures.append("testFlight.scenarioQueuePath must be testflight_scenarios.jsonl")
    return {
        "schemaVersion": "lumen.developer_cycle.improvement_loop_output_contract/1.0.0",
        "passed": not contract_failures,
        "requiredFiles": required_files,
        "missingFiles": missing_files,
        "requiredLoopStateKeys": sorted(required_state_keys),
        "missingLoopStateKeys": missing_state_keys,
        "nextIngestCommand": next_ingest_command,
        "scenarioQueuePath": scenario_queue_path,
        "failures": contract_failures,
    }


def _manifest_artifacts(root: Path) -> list[str]:
    base = root / DEFAULT_MANIFEST_DIR
    return [
        str(base / "AgentBehaviorManifest.json"),
        str(base / "AgentBehaviorManifest.md"),
        str(base / "manifest_validation_report.json"),
        str(base / "dataset_manifest.json"),
        str(base / "dataset_index.csv"),
        str(base / "fleet_system_prompts.json"),
        str(base / "fine_tuning"),
    ]


def _loop_artifacts(root: Path) -> list[str]:
    base = root / DEFAULT_LOOP_DIR
    return [
        str(base / "LOOP_REPORT.md"),
        str(base / "loop_state.json"),
        str(base / "loop_gaps.json"),
        str(base / "GAP_TRIAGE.md"),
        str(base / "gap_triage.json"),
        str(base / "TESTFLIGHT_RUNBOOK.md"),
        str(base / "testflight_scenarios.jsonl"),
    ]


def _output_artifacts(root: Path) -> list[str]:
    return [
        str(root / DEFAULT_FRAMEWORK_DIR / "developer_cycle_report.json"),
        str(root / DEFAULT_FRAMEWORK_DIR / "DEVELOPER_CYCLE_REPORT.md"),
        str(root / DEFAULT_FRAMEWORK_DIR / "framework_report.json"),
        str(root / DEFAULT_FRAMEWORK_DIR / "runtime_report_index.json"),
        *_manifest_artifacts(root),
        *_loop_artifacts(root),
    ]


def _next_recommended_command(report_flags: dict[str, Any]) -> str:
    if not report_flags.get("static"):
        return "python3 -m lumen_manifest_crawler developer-cycle --root . --portable --fail-on-static"
    if not report_flags.get("manifest") or not report_flags.get("loop"):
        return "python3 -m lumen_manifest_crawler improve-loop --root . --output generated/agent_manifest --loop-output generated/agent_improvement_loop"
    if not report_flags.get("runtime"):
        return "python3 -m lumen_manifest_crawler developer-cycle --root . --runtime-audit <exported-testflight-json>"
    if report_flags.get("xcode") != "passed":
        return "python3 -m lumen_manifest_crawler developer-cycle --root . --with-xcode"
    return "python3 -m lumen_manifest_crawler framework status --root ."


def _exit_code_for_report(report: dict[str, Any], config: DeveloperCycleConfig) -> int:
    if config.with_xcode and report.get("xcodeValidationStatus") != "passed":
        return 1
    if config.fail_on_static and not report.get("staticValidationPassed"):
        return 1
    if config.fail_on_validation and (
        not report.get("manifestValidationPassed")
        or not report.get("improvementLoopPassed")
        or report.get("xcodeValidationStatus") == "failed"
        or report.get("trainingStatus") == "failed"
    ):
        return 1
    if config.require_runtime_audit and not report.get("runtimeEvidencePresent"):
        return 1
    if config.fail_on_gaps:
        gap_count = 0
        for phase in report.get("phases", []):
            if isinstance(phase, dict):
                gap_count += int(phase.get("gapCount") or 0)
        if gap_count > 0 or not report.get("runtimeEvidencePresent"):
            return 1
    return 0


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_markdown_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Lumen Developer Cycle Report",
        "",
        f"- Root: `{report['root']}`",
        f"- Environment: `{report['environment']['checkoutKind']}` / `{report['environment']['executionKind']}`",
        f"- Static validation passed: `{report['staticValidationPassed']}`",
        f"- Manifest validation passed: `{report['manifestValidationPassed']}`",
        f"- Runtime evidence present: `{report['runtimeEvidencePresent']}`",
        f"- Improvement loop passed: `{report['improvementLoopPassed']}`",
        f"- Improvement-loop output contract passed: `{report['improvementLoopOutputContractPassed']}`",
        f"- Xcode validation: `{report['xcodeValidationStatus']}`",
        f"- Training status: `{report['trainingStatus']}`",
        f"- Portable pass: `{report['overallPortablePassed']}`",
        f"- Release-candidate pass: `{report['overallReleaseCandidatePassed']}`",
        "",
        "## Phase Summary",
        "",
    ]
    for phase in report.get("phases", []):
        if not isinstance(phase, dict):
            continue
        lines.extend([
            f"### {phase.get('title', phase.get('id', 'Phase'))}",
            "",
            f"- Status: `{phase.get('status', 'passed' if _phase_passed(phase) else 'failed')}`",
        ])
        if phase.get("reason"):
            lines.append(f"- Reason: {phase['reason']}")
        commands = phase.get("commands") if isinstance(phase.get("commands"), list) else []
        for command in commands:
            if not isinstance(command, dict):
                continue
            lines.append(f"- `{command.get('commandText', '')}` -> `{command.get('status')}`")
            if command.get("skipReason"):
                lines.append(f"  - skipped: {command['skipReason']}")
            elif command.get("returncode") not in {0, None}:
                lines.append(f"  - return code: `{command.get('returncode')}`")
        artifacts = phase.get("outputArtifacts") if isinstance(phase.get("outputArtifacts"), list) else []
        if artifacts:
            lines.append("- Outputs:")
            lines.extend(f"  - `{artifact}`" for artifact in artifacts)
        contract = phase.get("improvementLoopOutputContract") if isinstance(phase.get("improvementLoopOutputContract"), dict) else None
        if contract:
            lines.append(f"- Improvement-loop output contract: `{contract.get('passed')}`")
            for failure in contract.get("failures") or []:
                lines.append(f"  - {failure}")
        lines.append("")
    lines.extend([
        "## Runtime Evidence",
        "",
        f"- Runtime failures: `{report['runtimeFailures']['runtimeFailureCount']}`",
        f"- Raw runtime failures: `{report['runtimeFailures']['rawRuntimeFailureCount']}`",
        f"- Skipped live model generations: `{report['runtimeFailures']['skippedLiveModelGenerationCount']}`",
        "",
        "## Next Command",
        "",
        f"```bash\n{report['nextRecommendedCommand']}\n```",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _tail(text: str, max_chars: int = 12000) -> str:
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]
