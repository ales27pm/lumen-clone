# pylint: disable=line-too-long
"""Closed-loop runner for manifest generation and runtime-audit/TestFlight handoff."""

from __future__ import annotations

# pylint: disable=line-too-long,too-many-lines,too-many-branches,too-many-statements,too-many-locals,too-many-arguments,too-many-nested-blocks,missing-function-docstring,missing-class-docstring

import hashlib
import json
import shlex
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from lumen_manifest_crawler.crawler import generate_manifest
from lumen_manifest_crawler.dataset import generate_all_datasets
from lumen_manifest_crawler.dataset.fine_tuning import compile_agent_fine_tuning_datasets
from lumen_manifest_crawler.dataset.runtime_ingest import load_runtime_audit_reports
from lumen_manifest_crawler.fleet_artifacts import generate_fleet_artifacts
from lumen_manifest_crawler.output.writer import write_outputs
from lumen_manifest_crawler.validators import validate_agent_fine_tuning_datasets, validate_manifest

DETERMINISTIC_LOOP_TIMESTAMP = "1970-01-01T00:00:00+00:00"
LOOP_SCHEMA_VERSION = "1.1.0"
DEFAULT_LOOP_DIR = Path("generated/agent_improvement_loop")
TESTFLIGHT_SCENARIOS_FILE = "testflight_scenarios.jsonl"
TESTFLIGHT_RUNBOOK_FILE = "TESTFLIGHT_RUNBOOK.md"
EXPORT_DATASET_INSTRUCTION = "Export the in-app dataset package JSON from Agent Grounding."


@dataclass(frozen=True)
class LoopCommandResult:
    name: str
    command: list[str]
    cwd: str
    returncode: int
    stdout_tail: str
    stderr_tail: str

    @property
    def passed(self) -> bool:
        return self.returncode == 0

    def output_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "command": self.command,
            "cwd": self.cwd,
            "returncode": self.returncode,
            "passed": self.passed,
            "stdoutTail": self.stdout_tail,
            "stderrTail": self.stderr_tail,
        }


@dataclass(frozen=True)
class AgentImprovementLoopConfig:
    root: Path
    output: Path
    loop_output: Path = DEFAULT_LOOP_DIR
    runtime_audit_paths: tuple[Path, ...] = ()
    deterministic: bool = True
    pretty: bool = True
    strict: bool = True
    generate_system_prompts: bool = True
    generate_agent_fine_tuning: bool = True
    fine_tuning_output: Path | None = None
    cross_model_train_dir: Path | None = None
    build_command: tuple[str, ...] = ()
    test_command: tuple[str, ...] = ()
    train_command: tuple[str, ...] = ()
    fail_on_validation: bool = False
    max_tail_chars: int = 12000
    dry_run_commands: bool = False
    app_run_mode: str = "testflight"
    testflight_build_label: str | None = None
    require_testflight_runtime_audit: bool = False
    testflight_scenario_limit: int = 120


@dataclass(frozen=True)
class AgentImprovementLoopResult:
    state: dict[str, Any]
    gaps: list[dict[str, Any]]
    next_prompts: list[dict[str, Any]]
    command_results: list[LoopCommandResult]
    testflight_scenarios: list[dict[str, Any]]

    @property
    def passed(self) -> bool:
        hard_gaps = [gap for gap in self.gaps if gap.get("severity") in {"critical", "error"}]
        failed_commands = [result for result in self.command_results if not result.passed]
        return not hard_gaps and not failed_commands


def run_agent_improvement_loop(config: AgentImprovementLoopConfig) -> AgentImprovementLoopResult:
    """Run one closed-loop improvement pass.

    The loop intentionally performs one deterministic iteration, not an actual
    infinite process. External automation can repeat this command forever. This
    keeps every cycle auditable, diffable, and safe to stop or roll back.

    The live runtime stage is represented explicitly as a TestFlight handoff:
    this command writes a TestFlight runbook and scenario queue. The human or CI
    build system compiles/distributes the app, the tester runs Agent Grounding in
    the TestFlight build, exports the in-app dataset package JSON, and the next
    loop iteration ingests that JSON with --runtime-audit.
    """
    root = config.root.resolve()
    output = config.output.resolve()
    loop_output = config.loop_output.resolve()
    loop_output.mkdir(parents=True, exist_ok=True)

    started_at = DETERMINISTIC_LOOP_TIMESTAMP if config.deterministic else datetime.now(timezone.utc).isoformat()
    command_results: list[LoopCommandResult] = []

    command_results.append(_run_optional_command("pre_generation_tests", config.test_command, root, config))

    manifest = generate_manifest(root)
    runtime_reports = load_runtime_audit_reports(list(config.runtime_audit_paths))
    datasets = generate_all_datasets(
        manifest,
        runtime_audit_paths=list(config.runtime_audit_paths) if config.runtime_audit_paths else None,
        deterministic=config.deterministic,
    )
    validation_report = validate_manifest(manifest, datasets, strict=config.strict)
    fleet_artifacts = generate_fleet_artifacts(manifest) if config.generate_system_prompts else None

    fine_tuning_datasets = None
    if config.generate_agent_fine_tuning:
        fine_tuning_datasets = compile_agent_fine_tuning_datasets(
            manifest,
            datasets,
            fleet_artifacts=fleet_artifacts,
            runtime_audit_reports=runtime_reports,
        )
        ft_failures = validate_agent_fine_tuning_datasets(
            manifest,
            fine_tuning_datasets,
            runtime_audit_reports=runtime_reports,
        )
        existing_failures = list(getattr(validation_report, "failures", []))
        existing_failures.extend(ft_failures)
        validation_report = validation_report.model_copy(
            update={"failures": existing_failures, "passed": not existing_failures}
        )

    write_outputs(
        output,
        manifest,
        validation_report,
        datasets,
        pretty=config.pretty,
        fleet_artifacts=fleet_artifacts,
        cross_model_train_dir=config.cross_model_train_dir,
        incremental_fingerprint=_manifest_fingerprint(manifest),
        fine_tuning_datasets=fine_tuning_datasets,
        fine_tuning_output_dir=config.fine_tuning_output,
    )

    command_results.append(_run_optional_command("build_for_testflight", config.build_command, root, config))
    command_results.append(_run_optional_command("train", config.train_command, root, config))

    testflight_scenarios = _build_testflight_scenario_queue(
        manifest=manifest,
        datasets=datasets,
        fine_tuning_datasets=fine_tuning_datasets,
        limit=config.testflight_scenario_limit,
    )
    testflight_plan = _build_testflight_plan(config, manifest, runtime_reports, testflight_scenarios)

    dataset_summary = _dataset_summary(datasets, fine_tuning_datasets)
    runtime_summary = _runtime_summary(runtime_reports)
    command_summary = [result.output_dict() for result in command_results if result.command]
    gaps = _build_gap_report(
        manifest=manifest,
        validation_report=validation_report,
        datasets=datasets,
        fine_tuning_datasets=fine_tuning_datasets,
        runtime_reports=runtime_reports,
        command_results=command_results,
        config=config,
    )
    next_prompts = _build_next_action_prompts(gaps, runtime_reports, command_results, testflight_plan)

    source_integrity = getattr(manifest, "sourceIntegrity", None)
    fleet_manifest = getattr(manifest, "fleet", None)
    state = {
        "schemaVersion": LOOP_SCHEMA_VERSION,
        "startedAt": started_at,
        "completedAt": DETERMINISTIC_LOOP_TIMESTAMP if config.deterministic else datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "output": str(output),
        "runtimeAuditInputs": [str(path) for path in config.runtime_audit_paths],
        "manifest": {
            "commit": getattr(source_integrity, "commit", None),
            "fingerprint": _manifest_fingerprint(manifest),
            "toolCount": len(manifest.tools),
            "intentCount": len(manifest.intents),
            "modelSlotCount": len(getattr(fleet_manifest, "slots", [])),
            "routingEntryCount": len(manifest.routingMatrix),
        },
        "dataset": dataset_summary,
        "runtime": runtime_summary,
        "testFlight": testflight_plan,
        "validation": {
            "failureCount": len(validation_report.failures),
            "warningCount": len(validation_report.warnings),
            "failures": [_model_dump(failure) for failure in validation_report.failures],
            "warnings": [_model_dump(warning) for warning in validation_report.warnings],
        },
        "commands": command_summary,
        "gapCount": len(gaps),
        "criticalGapCount": sum(1 for gap in gaps if gap.get("severity") == "critical"),
        "errorGapCount": sum(1 for gap in gaps if gap.get("severity") == "error"),
        "passed": not any(gap.get("severity") in {"critical", "error"} for gap in gaps) and all(result.passed for result in command_results),
        "nextActionPromptCount": len(next_prompts),
    }

    _write_json(loop_output / "loop_state.json", state)
    _write_json(loop_output / "loop_gaps.json", {"gaps": gaps})
    _write_jsonl(loop_output / "next_action_prompts.jsonl", next_prompts)
    _write_jsonl(loop_output / TESTFLIGHT_SCENARIOS_FILE, testflight_scenarios)
    _write_markdown_report(loop_output / "LOOP_REPORT.md", state, gaps, next_prompts)
    _write_testflight_runbook(loop_output / TESTFLIGHT_RUNBOOK_FILE, state, testflight_scenarios)

    result = AgentImprovementLoopResult(
        state=state,
        gaps=gaps,
        next_prompts=next_prompts,
        command_results=command_results,
        testflight_scenarios=testflight_scenarios,
    )
    return result


def _run_optional_command(name: str, command: tuple[str, ...], cwd: Path, config: AgentImprovementLoopConfig) -> LoopCommandResult:
    if not command:
        return LoopCommandResult(name=name, command=[], cwd=str(cwd), returncode=0, stdout_tail="", stderr_tail="")
    if config.dry_run_commands:
        return LoopCommandResult(
            name=name,
            command=list(command),
            cwd=str(cwd),
            returncode=0,
            stdout_tail="dry-run: command not executed",
            stderr_tail="",
        )
    try:
        completed = subprocess.run(
            list(command),
            cwd=cwd,
            text=True,
            capture_output=True,
            check=False,
            timeout=None,
        )
    except OSError as exc:
        return LoopCommandResult(
            name=name,
            command=list(command),
            cwd=str(cwd),
            returncode=127,
            stdout_tail="",
            stderr_tail=_tail(str(exc), config.max_tail_chars),
        )
    return LoopCommandResult(
        name=name,
        command=list(command),
        cwd=str(cwd),
        returncode=completed.returncode,
        stdout_tail=_tail(completed.stdout, config.max_tail_chars),
        stderr_tail=_tail(completed.stderr, config.max_tail_chars),
    )


def _tail(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def _manifest_fingerprint(manifest: Any) -> str:
    payload = json.dumps(_canonicalize(manifest.output_dict()), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _canonicalize(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _canonicalize(child) for key, child in sorted(value.items())}
    if isinstance(value, list):
        return [_canonicalize(item) for item in value]
    return value


def _dataset_summary(datasets: dict[str, list[dict[str, Any]]], fine_tuning_datasets: Any) -> dict[str, Any]:
    families = {
        name: len(records)
        for name, records in sorted(datasets.items())
        if name != "dataset_manifest"
    }
    out: dict[str, Any] = {
        "familyCount": len(families),
        "recordCount": sum(families.values()),
        "families": families,
    }
    if fine_tuning_datasets:
        out["agentFineTuning"] = {
            agent: {
                "trainSFT": len(dataset.train_sft),
                "valSFT": len(dataset.val_sft),
                "trainDPO": len(dataset.train_dpo),
                "valDPO": len(dataset.val_dpo),
                "eval": len(dataset.eval),
            }
            for agent, dataset in sorted(fine_tuning_datasets.items())
        }
    return out


def _runtime_summary(runtime_reports: list[dict[str, Any]]) -> dict[str, Any]:
    failures = [failure for report in runtime_reports for failure in report.get("failures", []) if isinstance(failure, dict)]
    by_type: dict[str, int] = {}
    by_layer: dict[str, int] = {}
    for failure in failures:
        by_type[str(failure.get("type") or "unknown")] = by_type.get(str(failure.get("type") or "unknown"), 0) + 1
        by_layer[str(failure.get("sourceLayer") or "unknown")] = by_layer.get(str(failure.get("sourceLayer") or "unknown"), 0) + 1
    return {
        "reportCount": len(runtime_reports),
        "failureCount": len(failures),
        "failureTypes": dict(sorted(by_type.items())),
        "sourceLayers": dict(sorted(by_layer.items())),
    }


def _build_testflight_scenario_queue(
    *,
    manifest: Any,
    datasets: dict[str, list[dict[str, Any]]],
    fine_tuning_datasets: Any,
    limit: int,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    candidates.extend(_build_trace_export_scenarios(manifest))
    candidates.extend(_build_trace_integrity_scenarios())

    for record in datasets.get("eval_scenarios", []):
        candidates.append(_scenario_from_eval_record(record, source_family="eval_scenarios"))

    if fine_tuning_datasets:
        for agent, dataset in sorted(fine_tuning_datasets.items()):
            for record in dataset.eval:
                scenario = _scenario_from_eval_record(record, source_family=f"agent_eval:{agent}")
                scenario["agent"] = agent
                candidates.append(scenario)

    for entry in manifest.routingMatrix:
        candidates.append({
            "id": _stable_id("routing", entry.intent, entry.allowedTools),
            "sourceFamily": "routing_matrix",
            "agent": "cortex",
            "taskType": "routing_matrix_adherence",
            "prompt": f"Test intent `{entry.intent}` in the TestFlight app and verify the selected tool is one of: {', '.join(entry.allowedTools) or 'none'}.",
            "expected": {
                "intent": entry.intent,
                "allowedToolIDs": list(entry.allowedTools),
                "mustUseManifestToolIDsOnly": True,
            },
            "testFlightInstructions": [
                "Open the TestFlight build of Lumen.",
                "Use the normal chat/app surface, not a mocked harness.",
                "Enter or adapt the prompt naturally.",
                "Run Agent Grounding Audit after the interaction batch.",
                "Export the in-app dataset package JSON.",
            ],
        })

    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate.get("id") or _stable_id(candidate))
        if key in seen:
            continue
        seen.add(key)
        candidate["id"] = key
        deduped.append(candidate)
    return deduped[: max(0, limit)]


def _build_trace_export_scenarios(manifest: Any) -> list[dict[str, Any]]:
    scenarios: list[dict[str, Any]] = []
    routing_entries = sorted(
        (entry for entry in manifest.routingMatrix if getattr(entry, "allowedTools", None)),
        key=lambda entry: str(entry.intent),
    )
    for entry in routing_entries[:3]:
        allowed = [str(tool_id) for tool_id in list(entry.allowedTools)[:5]]
        tool_hint = ", ".join(allowed) if allowed else "manifest-allowed tool IDs"
        scenarios.append({
            "id": _stable_id("trace_export_coverage", entry.intent, allowed),
            "sourceFamily": "trace_export_coverage",
            "agent": "runtime",
            "taskType": "runtime_trace_export_coverage",
            "prompt": f"Trigger intent `{entry.intent}` with a realistic request that should select one of: {tool_hint}.",
            "expected": {
                "intent": entry.intent,
                "traceField": "traceSelectedToolAllowedCount",
                "requiresRecentTrace": True,
                "allowedToolIDs": allowed,
            },
            "testFlightInstructions": [
                "Run the prompt in the real TestFlight app.",
                f"After the batch, {EXPORT_DATASET_INSTRUCTION}",
                "Verify the export includes `traceSelectedToolAllowedCount` and that recent traces keep allowedToolIDs for tool-selection turns.",
            ],
        })

    scenarios.append({
        "id": _stable_id("trace_export_coverage", "chat_intent_no_tool"),
        "sourceFamily": "trace_export_coverage",
        "agent": "runtime",
        "taskType": "runtime_trace_export_coverage",
        "prompt": "Ask a normal chat-only question that should not call tools, then verify the exported runtime traces still include prompt prefixes and parse diagnostics.",
        "expected": {
            "intent": "chat",
            "traceField": "traceSelectedToolAllowedCount",
            "requiresRecentTrace": True,
            "selectedToolExpected": False,
        },
        "testFlightInstructions": [
            "Run the prompt in the real TestFlight app without developer harnesses.",
            EXPORT_DATASET_INSTRUCTION,
            "Verify the export includes `traceSelectedToolAllowedCount` and at least one trace for this interaction.",
        ],
    })
    return scenarios


def _build_trace_integrity_scenarios() -> list[dict[str, Any]]:
    return [
        {
            "id": _stable_id("trace_integrity", "parse_error_free_tool_turn"),
            "sourceFamily": "trace_integrity",
            "agent": "runtime",
            "taskType": "runtime_trace_integrity",
            "prompt": "Run one tool-backed task and verify the exported dataset shows `traceParseErrorCount` does not increase unexpectedly.",
            "expected": {
                "traceField": "traceParseErrorCount",
                "requiresRecentTrace": True,
                "expectedDirection": "non_increasing_for_stable_build",
            },
            "testFlightInstructions": [
                "Run the prompt in the real TestFlight app.",
                EXPORT_DATASET_INSTRUCTION,
                "Verify `traceParseErrorCount` exists and inspect whether parse errors are regressing.",
            ],
        },
        {
            "id": _stable_id("trace_integrity", "mixed_prompts_trace_consistency"),
            "sourceFamily": "trace_integrity",
            "agent": "runtime",
            "taskType": "runtime_trace_integrity",
            "prompt": "Run a mixed batch of chat and tool prompts, then verify the export includes both `traceSelectedToolAllowedCount` and `traceParseErrorCount`.",
            "expected": {
                "traceFields": ["traceSelectedToolAllowedCount", "traceParseErrorCount"],
                "requiresRecentTrace": True,
            },
            "testFlightInstructions": [
                "Run mixed prompts through the normal app UI.",
                EXPORT_DATASET_INSTRUCTION,
                "Confirm both trace metrics are present for loop ingestion.",
            ],
        },
    ]


def _scenario_from_eval_record(record: dict[str, Any], *, source_family: str) -> dict[str, Any]:
    messages = record.get("messages") if isinstance(record.get("messages"), list) else []
    prompt = ""
    for message in messages:
        if isinstance(message, dict) and message.get("role") == "user":
            prompt = str(message.get("content") or "")
            break
    expected = record.get("expected") if isinstance(record.get("expected"), dict) else {}
    metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    return {
        "id": str(record.get("id") or _stable_id(source_family, prompt, expected)),
        "sourceFamily": source_family,
        "agent": str(metadata.get("agent") or record.get("agentRole") or "runtime"),
        "taskType": str(record.get("taskType") or metadata.get("evalType") or "runtime_eval"),
        "prompt": prompt,
        "expected": expected,
        "metadata": metadata,
        "testFlightInstructions": [
            "Install or update the current TestFlight build.",
            "Run this prompt through the real app UI and current bundled model/runtime.",
            "Do not use mocked developer harnesses for this pass.",
            f"After the batch, {EXPORT_DATASET_INSTRUCTION}",
            "Feed the exported JSON into the next loop with --runtime-audit.",
        ],
    }


def _build_testflight_plan(
    config: AgentImprovementLoopConfig,
    manifest: Any,
    runtime_reports: list[dict[str, Any]],
    scenarios: list[dict[str, Any]],
) -> dict[str, Any]:
    has_runtime = bool(runtime_reports)
    return {
        "mode": config.app_run_mode,
        "buildLabel": config.testflight_build_label,
        "status": "runtime-audit-ingested" if has_runtime else "awaiting-testflight-runtime-audit",
        "requiresTestFlightAppRun": config.app_run_mode.casefold() == "testflight",
        "requireRuntimeAuditForPass": config.require_testflight_runtime_audit,
        "runtimeAuditProvided": has_runtime,
        "scenarioQueuePath": TESTFLIGHT_SCENARIOS_FILE,
        "runbookPath": TESTFLIGHT_RUNBOOK_FILE,
        "scenarioCount": len(scenarios),
        "expectedExport": "lumen-in-app-dataset-*.json from Agent Grounding > Export In-App Dataset Package",
        "nextIngestCommand": shlex.join([
            "python",
            "-m",
            "lumen_manifest_crawler",
            "improve-loop",
            "--root",
            str(config.root.resolve()),
            "--output",
            str(config.output.resolve()),
            "--loop-output",
            str(config.loop_output.resolve()),
            "--runtime-audit",
            "<exported-testflight-json>",
        ]),
        "manifestFingerprint": _manifest_fingerprint(manifest),
    }


def _build_gap_report(  # NOSONAR
    *,
    manifest: Any,
    validation_report: Any,
    datasets: dict[str, list[dict[str, Any]]],
    fine_tuning_datasets: Any,
    runtime_reports: list[dict[str, Any]],
    command_results: list[LoopCommandResult],
    config: AgentImprovementLoopConfig,
) -> list[dict[str, Any]]:
    gaps: list[dict[str, Any]] = []

    if config.app_run_mode.casefold() == "testflight" and not runtime_reports:
        resolved_loop_output = config.loop_output.resolve()
        gaps.append({
            "id": _stable_id("testflight_runtime_pending", config.testflight_build_label or "unlabeled"),
            "severity": "error" if config.require_testflight_runtime_audit else "warning",
            "category": "testflight_runtime_pending",
            "title": "TestFlight in-app audit export has not been ingested yet",
            "evidence": {
                "expectedExport": "lumen-in-app-dataset-*.json",
                "source": "Agent Grounding > Export In-App Dataset Package",
                "scenarioQueue": str(resolved_loop_output / "testflight_scenarios.jsonl"),
                "runbook": str(resolved_loop_output / "TESTFLIGHT_RUNBOOK.md"),
            },
            "recommendedAction": "Compile/distribute the TestFlight build, run Agent Grounding in the app, export the in-app dataset package JSON, then rerun improve-loop with --runtime-audit <json>.",
        })

    for failure in validation_report.failures:
        dumped = _model_dump(failure)
        gaps.append({
            "id": _stable_id("validation", dumped),
            "severity": "error",
            "category": "validation",
            "title": dumped.get("code") or "validation_failure",
            "evidence": dumped,
            "recommendedAction": "Fix source extraction or dataset generation until manifest validation is clean.",
        })

    for warning in validation_report.warnings:
        dumped = _model_dump(warning)
        gaps.append({
            "id": _stable_id("warning", dumped),
            "severity": "warning",
            "category": "validation_warning",
            "title": dumped.get("code") or "validation_warning",
            "evidence": dumped,
            "recommendedAction": "Review warning and either improve extraction coverage or intentionally document the exception.",
        })

    for result in command_results:
        if result.command and not result.passed:
            gaps.append({
                "id": _stable_id("command", result.output_dict()),
                "severity": "critical",
                "category": "command_failure",
                "title": f"{result.name} command failed",
                "evidence": result.output_dict(),
                "recommendedAction": "Fix the failing command before trusting this loop iteration.",
            })

    runtime_failures = [failure for report in runtime_reports for failure in report.get("failures", []) if isinstance(failure, dict)]
    for failure in runtime_failures[:200]:
        failure_type = str(failure.get("type") or "runtime_failure")
        severity = "critical" if any(token in failure_type for token in ["unknown_tool", "sentinel", "not_allowed"]) else "error"
        gaps.append({
            "id": _stable_id("runtime", failure),
            "severity": severity,
            "category": "runtime_drift",
            "title": failure_type,
            "evidence": failure,
            "recommendedAction": _runtime_recommendation(failure_type),
        })

    required_families = {
        "train_sft",
        "validation_sft",
        "eval_scenarios",
        "dpo_preference_pairs",
        "tool_schema_cards",
        "manifest_grounding_cards",
        "runtime_audit_repairs",
    }
    for family in sorted(required_families):
        if len(datasets.get(family, [])) == 0:
            gaps.append({
                "id": _stable_id("empty_family", family),
                "severity": "error" if family != "runtime_audit_repairs" else "warning",
                "category": "dataset_coverage",
                "title": f"Empty dataset family: {family}",
                "evidence": {"family": family},
                "recommendedAction": f"Add generators or runtime inputs that produce {family} records.",
            })

    if fine_tuning_datasets:
        for agent, dataset in sorted(fine_tuning_datasets.items()):
            if not dataset.train_sft:
                gaps.append({
                    "id": _stable_id("agent_empty_sft", agent),
                    "severity": "error",
                    "category": "agent_fine_tuning_coverage",
                    "title": f"No SFT records for {agent}",
                    "evidence": {"agent": agent},
                    "recommendedAction": f"Add or route role-specific examples for the {agent} agent.",
                })
            if not dataset.eval:
                gaps.append({
                    "id": _stable_id("agent_empty_eval", agent),
                    "severity": "warning",
                    "category": "agent_eval_coverage",
                    "title": f"No eval records for {agent}",
                    "evidence": {"agent": agent},
                    "recommendedAction": f"Add must-pass eval scenarios for the {agent} agent.",
                })

    tool_count = len(manifest.tools)
    eval_count = len(datasets.get("eval_scenarios", []))
    if tool_count and eval_count < tool_count * 5:
        gaps.append({
            "id": _stable_id("eval_coverage", {"toolCount": tool_count, "evalCount": eval_count}),
            "severity": "warning",
            "category": "eval_coverage",
            "title": "Eval scenario coverage is below five records per tool",
            "evidence": {"toolCount": tool_count, "evalScenarioCount": eval_count, "minimumExpected": tool_count * 5},
            "recommendedAction": "Expand natural, argument, approval, permission, and adversarial scenario generation per tool.",
        })

    return sorted(gaps, key=lambda gap: (str(gap.get("severity")), str(gap.get("category")), str(gap.get("title"))))


def _runtime_recommendation(failure_type: str) -> str:
    if "unknown_tool" in failure_type or "unmanifested" in failure_type or "missing_live_tool" in failure_type:
        return "Regenerate the manifest from Swift source, then add unknown-tool DPO contrast samples."
    if "argument" in failure_type:
        return "Regenerate executor schema cards and add missing-argument clarification examples."
    if "approval" in failure_type:
        return "Add approval-boundary SFT/DPO records and verify the UI confirmation path."
    if "sentinel" in failure_type:
        return "Add Mouth sanitizer and persisted-step sentinel suppression regression samples."
    if "not_allowed" in failure_type or "routing" in failure_type:
        return "Add Cortex routing contrast samples for the violated intent/tool pair."
    return "Convert this failure into a REM repair sample and add a regression eval."


def _build_next_action_prompts(
    gaps: list[dict[str, Any]],
    runtime_reports: list[dict[str, Any]],
    command_results: list[LoopCommandResult],
    testflight_plan: dict[str, Any],
) -> list[dict[str, Any]]:
    prompts: list[dict[str, Any]] = []
    for gap in gaps[:80]:
        prompts.append({
            "id": _stable_id("prompt", gap),
            "taskType": "codebase_improvement" if gap.get("category") != "testflight_runtime_pending" else "testflight_runtime_audit",
            "priority": _priority_for_gap(gap),
            "messages": [
                {
                    "role": "system",
                    "content": "You are improving the Lumen agent dataset loop. Make real source changes only. Do not invent tool IDs, do not weaken privacy policy, and keep generated artifacts deterministic.",
                },
                {
                    "role": "user",
                    "content": _gap_prompt(gap, testflight_plan),
                },
            ],
            "metadata": {
                "gapID": gap.get("id"),
                "category": gap.get("category"),
                "severity": gap.get("severity"),
                "runtimeReportCount": len(runtime_reports),
                "failedCommandCount": sum(1 for result in command_results if result.command and not result.passed),
                "testFlightStatus": testflight_plan.get("status"),
                "testFlightScenarioQueue": testflight_plan.get("scenarioQueuePath"),
            },
        })
    if not prompts:
        prompts.append({
            "id": _stable_id("prompt", "expand_next_loop"),
            "taskType": "loop_expansion",
            "priority": "medium",
            "messages": [
                {"role": "system", "content": "You are improving the Lumen agent dataset loop."},
                {"role": "user", "content": "No blocking gaps were detected. Expand the next loop by adding one new TestFlight scenario family, one runtime trace field exported by the in-app dataset package, and one dataset quality gate while preserving deterministic output."},
            ],
            "metadata": {"category": "continuous_expansion", "testFlightStatus": testflight_plan.get("status")},
        })
    return prompts


def _priority_for_gap(gap: dict[str, Any]) -> str:
    return {
        "critical": "highest",
        "error": "high",
        "warning": "medium",
    }.get(str(gap.get("severity")), "low")


def _gap_prompt(gap: dict[str, Any], testflight_plan: dict[str, Any]) -> str:
    return (
        "Fix or expand the Lumen agent improvement loop for this gap.\n\n"
        f"Severity: {gap.get('severity')}\n"
        f"Category: {gap.get('category')}\n"
        f"Title: {gap.get('title')}\n"
        f"Recommended action: {gap.get('recommendedAction')}\n\n"
        "TestFlight phase:\n"
        f"{json.dumps(testflight_plan, ensure_ascii=False, indent=2, sort_keys=True)}\n\n"
        "Evidence JSON:\n"
        f"{json.dumps(gap.get('evidence'), ensure_ascii=False, indent=2, sort_keys=True)}\n\n"
        "Required outcome: modify the crawler, in-app Agent Grounding audit, runtime trace schema, dataset compiler, TestFlight runbook, tests, or workflow scripts so the next TestFlight loop iteration has stronger live-runtime coverage or removes the drift."
    )


def _model_dump(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, dict):
        return value
    return {"value": str(value)}


def _stable_id(*parts: Any) -> str:
    payload = json.dumps(parts, ensure_ascii=False, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def _write_markdown_report(path: Path, state: dict[str, Any], gaps: list[dict[str, Any]], prompts: list[dict[str, Any]]) -> None:
    lines = [
        "# Lumen Agent Improvement Loop Report",
        "",
        f"- Passed: `{state['passed']}`",
        f"- Tools: `{state['manifest']['toolCount']}`",
        f"- Intents: `{state['manifest']['intentCount']}`",
        f"- Model slots: `{state['manifest']['modelSlotCount']}`",
        f"- Dataset records: `{state['dataset']['recordCount']}`",
        f"- Runtime audit reports: `{state['runtime']['reportCount']}`",
        f"- Runtime failures: `{state['runtime']['failureCount']}`",
        f"- TestFlight status: `{state['testFlight']['status']}`",
        f"- TestFlight scenarios: `{state['testFlight']['scenarioCount']}`",
        f"- Gaps: `{len(gaps)}`",
        f"- Next action prompts: `{len(prompts)}`",
        "",
        "## TestFlight handoff",
        "",
        "Run `TESTFLIGHT_RUNBOOK.md` in the real TestFlight app, export the in-app dataset package JSON, then rerun this command with `--runtime-audit <exported-json>`.",
        "",
        "## Top gaps",
        "",
    ]
    for gap in gaps[:30]:
        lines.extend([
            f"### {gap.get('severity', 'unknown').upper()} — {gap.get('title')}",
            "",
            f"- Category: `{gap.get('category')}`",
            f"- Recommendation: {gap.get('recommendedAction')}",
            "",
        ])
    if not gaps:
        lines.append("No blocking gaps detected. The next loop should expand TestFlight runtime coverage.")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _write_testflight_runbook(path: Path, state: dict[str, Any], scenarios: list[dict[str, Any]]) -> None:
    lines = [
        "# TestFlight In-App Runtime Runbook",
        "",
        "This is the live-runtime phase of the Lumen improvement loop. Do not replace this with mocked unit tests. The point is to run the current app build through TestFlight, then export what the shipped app observed.",
        "",
        "## Build identity",
        "",
        f"- Manifest fingerprint: `{state['manifest']['fingerprint']}`",
        f"- Manifest commit: `{state['manifest']['commit']}`",
        f"- Build label: `{state['testFlight'].get('buildLabel')}`",
        f"- Expected export: `{state['testFlight']['expectedExport']}`",
        "",
        "## Required app flow",
        "",
        "1. Compile/archive the app and distribute it through TestFlight.",
        "2. Install or update that TestFlight build on the device.",
        "3. Use the normal app surface for scenario prompts. Do not use a mocked harness for this pass.",
        "4. Open the in-app Agent Grounding screen.",
        "5. Tap `Run Agent Grounding Audit`.",
        "6. Tap `Export In-App Dataset Package`.",
        "7. Share/save the produced `lumen-in-app-dataset-*.json` file.",
        "8. Feed it into the next loop:",
        "",
        "```bash",
        state["testFlight"]["nextIngestCommand"],
        "```",
        "",
        "## Scenario queue",
        "",
        f"Full machine-readable queue: `{state['testFlight']['scenarioQueuePath']}`",
        "",
    ]
    for index, scenario in enumerate(scenarios[:30], start=1):
        prompt = str(scenario.get("prompt") or "").replace("\n", " ").strip()
        lines.extend([
            f"### {index}. {scenario.get('taskType')}",
            "",
            f"- Agent: `{scenario.get('agent')}`",
            f"- Source: `{scenario.get('sourceFamily')}`",
            f"- Prompt: {prompt}",
            "",
        ])
    if len(scenarios) > 30:
        lines.append(f"Additional scenarios omitted from this Markdown view: `{len(scenarios) - 30}`. Use `testflight_scenarios.jsonl` for the full queue.")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
