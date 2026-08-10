# pylint: disable=line-too-long
"""Closed-loop runner for manifest generation and runtime-audit/TestFlight handoff."""

from __future__ import annotations

# pylint: disable=line-too-long,too-many-lines,too-many-branches,too-many-statements,too-many-locals,too-many-arguments,too-many-nested-blocks,missing-function-docstring,missing-class-docstring

import hashlib
import json
import re
import shlex
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

from lumen_manifest_crawler.crawler import generate_manifest
from lumen_manifest_crawler.dataset import generate_all_datasets
from lumen_manifest_crawler.dataset.fine_tuning import compile_agent_fine_tuning_datasets
from lumen_manifest_crawler.dataset.runtime_ingest import (
    load_runtime_audit_reports,
    without_internal_artifact_bindings,
)
from lumen_manifest_crawler.fleet_artifacts import generate_fleet_artifacts
from lumen_manifest_crawler.output.writer import write_outputs
from lumen_manifest_crawler.validators import validate_agent_fine_tuning_datasets, validate_manifest

DETERMINISTIC_LOOP_TIMESTAMP = "1970-01-01T00:00:00+00:00"
LOOP_SCHEMA_VERSION = "1.2.0"
DEFAULT_RUNTIME_AUDIT_MAX_AGE_SECONDS = 60 * 60
RUNTIME_AUDIT_MAX_FUTURE_SKEW_SECONDS = 5 * 60
STRICT_RUNTIME_RECEIPT_SCHEMA = "lumen.interactive_model_tool_verifier_receipt/1.1.0"
STRICT_RUNTIME_RECEIPT_STATUS = "verified-at-assessment"
STRICT_RUNTIME_RECEIPT_SCOPE = "physical-device-debug-interactive-model-tool"
STRICT_RUNTIME_VERIFIER_NAME = "verify_interactive_model_tool_evidence"
STRICT_RUNTIME_VERIFIER_CONTRACT_VERSION = "1.1.0"
SUPPORTED_APP_RUN_MODES = frozenset({"testflight", "device-debug"})
APP_BEHAVIOR_MANIFEST_PATH = Path("ios/Lumen/AgentBehaviorManifest.json")
LOCAL_FILE_URL_PATTERN = re.compile(r"file:[^\r\n]*", re.IGNORECASE)
LOCAL_ABSOLUTE_PATH_PATTERN = re.compile(
    r"(?m)(?<![:/.\w])/(?!/)(?![A-Za-z][A-Za-z0-9:_-]*\s*>)[^\r\n]*"
)
DEFAULT_LOOP_DIR = Path("generated/agent_improvement_loop")
TESTFLIGHT_SCENARIOS_FILE = "testflight_scenarios.jsonl"
TESTFLIGHT_RUNBOOK_FILE = "TESTFLIGHT_RUNBOOK.md"
EXPORT_DATASET_INSTRUCTION = "Export the TestFlight + Agent Grounding package JSON from Agent Grounding."
EXPLICIT_MODEL_EVIDENCE_CATEGORIES = {
    "agent_json_empty_generation",
    "agent_json_parse_empty",
    "agent_json_parse_error",
    "agent_model_empty_output",
    "agent_model_parse_error",
    "no_correlated_model_turn",
    "deterministic_compatibility_not_training_evidence",
    "deterministic_compatibility_not_live_evidence",
    "agent_service_not_entered",
    "missing_sidecar_trace_export",
}


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
    runtime_audit_reference_time: str | None = None
    runtime_audit_max_age_seconds: int = DEFAULT_RUNTIME_AUDIT_MAX_AGE_SECONDS
    verify_runtime_audit_now: bool = False
    runtime_audit_expected_build_number: str | None = None


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
    the TestFlight build, exports the TestFlight + Agent Grounding package JSON, and the next
    loop iteration ingests that JSON with --runtime-audit.
    """
    if config.app_run_mode.casefold() not in SUPPORTED_APP_RUN_MODES:
        raise ValueError(
            "app_run_mode must be exactly 'testflight' or 'device-debug'"
        )
    root = config.root.resolve()
    output = config.output.resolve()
    loop_output = config.loop_output.resolve()
    loop_output.mkdir(parents=True, exist_ok=True)

    started_at = DETERMINISTIC_LOOP_TIMESTAMP if config.deterministic else datetime.now(timezone.utc).isoformat()
    command_results: list[LoopCommandResult] = []

    command_results.append(_run_optional_command("pre_generation_tests", config.test_command, root, config))

    manifest = generate_manifest(root)
    runtime_reports = load_runtime_audit_reports(list(config.runtime_audit_paths))
    ingestion_runtime_reports = _ingestion_runtime_reports(runtime_reports, config)
    source_integrity = getattr(manifest, "sourceIntegrity", None)
    verification_assessment = _verify_runtime_audit_at_host_now(
        ingestion_runtime_reports,
        config,
        expected_source_revision=getattr(source_integrity, "baseCommit", None),
        expected_working_tree_digest=getattr(
            source_integrity,
            "workingTreeDigest",
            None,
        ),
        source_dirty_state=getattr(source_integrity, "dirtyState", None),
    )
    _current_proof_reports, runtime_proof = _assess_runtime_audit_proof(
        ingestion_runtime_reports,
        config,
        expected_source_revision=getattr(source_integrity, "baseCommit", None),
        verification_assessment=verification_assessment,
    )
    training_runtime_reports = _annotate_runtime_reports_for_training(
        ingestion_runtime_reports,
        runtime_proof,
    )
    datasets = generate_all_datasets(
        manifest,
        root=root,
        runtime_audit_reports=training_runtime_reports,
        deterministic=config.deterministic,
    )
    validation_report = validate_manifest(manifest, datasets, strict=config.strict)
    should_write_full_fleet_artifacts = (
        config.generate_system_prompts or config.cross_model_train_dir is not None
    )
    fleet_artifacts = (
        generate_fleet_artifacts(manifest)
        if should_write_full_fleet_artifacts or config.generate_agent_fine_tuning
        else None
    )
    output_fleet_artifacts = (
        fleet_artifacts if should_write_full_fleet_artifacts else None
    )

    fine_tuning_datasets = None
    if config.generate_agent_fine_tuning:
        fine_tuning_datasets = compile_agent_fine_tuning_datasets(
            manifest,
            datasets,
            fleet_artifacts=fleet_artifacts,
            runtime_audit_reports=training_runtime_reports,
        )
        ft_failures = validate_agent_fine_tuning_datasets(
            manifest,
            fine_tuning_datasets,
            runtime_audit_reports=training_runtime_reports,
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
        fleet_artifacts=output_fleet_artifacts,
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
    testflight_plan = _build_testflight_plan(
        config,
        manifest,
        runtime_reports,
        testflight_scenarios,
        ingestion_runtime_reports=ingestion_runtime_reports,
        runtime_proof=runtime_proof,
    )

    dataset_summary = _dataset_summary(datasets, fine_tuning_datasets)
    runtime_summary = _runtime_summary(
        ingestion_runtime_reports,
        all_runtime_reports=runtime_reports,
        runtime_proof=runtime_proof,
    )
    public_replacements = _runtime_audit_public_replacements(
        config.runtime_audit_paths,
        runtime_reports,
    )
    command_summary = [
        _sanitize_public_artifact(
            result.output_dict(),
            root=root,
            replacements=public_replacements,
        )
        for result in command_results
        if result.command
    ]
    gaps = _build_gap_report(
        manifest=manifest,
        validation_report=validation_report,
        datasets=datasets,
        fine_tuning_datasets=fine_tuning_datasets,
        runtime_reports=ingestion_runtime_reports,
        all_runtime_reports=runtime_reports,
        runtime_proof=runtime_proof,
        command_results=command_results,
        config=config,
    )
    next_prompts = _build_next_action_prompts(gaps, ingestion_runtime_reports, command_results, testflight_plan)

    fleet_manifest = getattr(manifest, "fleet", None)
    state = {
        "schemaVersion": LOOP_SCHEMA_VERSION,
        "startedAt": started_at,
        "completedAt": DETERMINISTIC_LOOP_TIMESTAMP if config.deterministic else datetime.now(timezone.utc).isoformat(),
        "root": _public_repo_path(root, root),
        "output": _public_repo_path(output, root),
        "runtimeAuditInputs": sorted(set(public_replacements.values())),
        "manifest": {
            "baseCommit": getattr(source_integrity, "baseCommit", None),
            "workingTreeDigest": getattr(source_integrity, "workingTreeDigest", None),
            "dirtyState": getattr(source_integrity, "dirtyState", None),
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

    triage = _build_gap_triage(gaps, ingestion_runtime_reports, config)
    state["triage"] = triage["summary"]

    state = _sanitize_public_artifact(
        state,
        root=root,
        replacements=public_replacements,
    )
    gaps = _sanitize_public_artifact(
        gaps,
        root=root,
        replacements=public_replacements,
    )
    triage = _sanitize_public_artifact(
        triage,
        root=root,
        replacements=public_replacements,
    )
    next_prompts = _sanitize_public_artifact(
        next_prompts,
        root=root,
        replacements=public_replacements,
    )
    testflight_scenarios = _sanitize_public_artifact(
        testflight_scenarios,
        root=root,
        replacements=public_replacements,
    )

    _write_json(loop_output / "loop_state.json", state)
    _write_json(loop_output / "loop_gaps.json", {"gaps": gaps})
    _write_json(loop_output / "gap_triage.json", triage)
    _write_jsonl(loop_output / "next_action_prompts.jsonl", next_prompts)
    _write_jsonl(loop_output / TESTFLIGHT_SCENARIOS_FILE, testflight_scenarios)
    _write_markdown_report(loop_output / "LOOP_REPORT.md", state, gaps, next_prompts)
    _write_gap_triage_markdown(loop_output / "GAP_TRIAGE.md", triage)
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


def _public_repo_path(path: Path, root: Path) -> str:
    """Return a stable repository-relative path for tracked artifacts."""

    try:
        relative = path.resolve().relative_to(root.resolve())
    except ValueError:
        return "<external-path-redacted>"
    return "." if relative == Path(".") else relative.as_posix()


def _runtime_audit_public_replacements(
    configured_paths: Iterable[Path],
    runtime_reports: Iterable[dict[str, Any]],
) -> dict[str, str]:
    """Map private runtime-audit locations to content-derived public refs."""

    replacements: dict[str, str] = {}
    refs_by_source: dict[str, set[str]] = {}
    for report in runtime_reports:
        sha256 = report.get("_artifactSha256")
        if not isinstance(sha256, str) or re.fullmatch(r"[0-9a-f]{64}", sha256) is None:
            continue
        reference = f"runtime-audit-sha256-{sha256}"
        source = str(report.get("_source") or "").split("#", 1)[0]
        if source:
            refs_by_source.setdefault(source, set()).add(reference)
            replacements[source] = reference

    for configured in configured_paths:
        resolved = str(configured.resolve())
        matched = {
            reference
            for source, references in refs_by_source.items()
            if source == resolved or source.startswith(resolved.rstrip("/") + "/")
            for reference in references
        }
        if len(matched) == 1:
            replacements[resolved] = next(iter(matched))
        elif matched:
            digest = hashlib.sha256("\n".join(sorted(matched)).encode("utf-8")).hexdigest()
            replacements[resolved] = f"runtime-audit-set-sha256-{digest}"
        elif configured.is_file():
            try:
                digest = hashlib.sha256(configured.read_bytes()).hexdigest()
            except OSError:
                replacements[resolved] = "<runtime-audit-input-redacted>"
            else:
                replacements[resolved] = f"runtime-audit-sha256-{digest}"
        else:
            replacements[resolved] = "<runtime-audit-input-redacted>"

    return replacements


def _sanitize_public_artifact(
    value: Any,
    *,
    root: Path,
    replacements: dict[str, str],
) -> Any:
    """Remove machine-local paths from values written to tracked artifacts."""

    if isinstance(value, dict):
        sanitized_dict: dict[Any, Any] = {}
        for key, child in value.items():
            if key in {"_artifactSha256", "_artifactByteCount"}:
                continue
            sanitized_key = (
                _sanitize_public_artifact(
                    key,
                    root=root,
                    replacements=replacements,
                )
                if isinstance(key, str)
                else key
            )
            if sanitized_key in sanitized_dict:
                raise ValueError("public artifact key collision after sanitization")
            sanitized_dict[sanitized_key] = _sanitize_public_artifact(
                child,
                root=root,
                replacements=replacements,
            )
        return sanitized_dict
    if isinstance(value, list):
        return [
            _sanitize_public_artifact(child, root=root, replacements=replacements)
            for child in value
        ]
    if isinstance(value, tuple):
        return [
            _sanitize_public_artifact(child, root=root, replacements=replacements)
            for child in value
        ]
    if not isinstance(value, str):
        return value

    sanitized = value
    executable = str(Path(sys.executable).resolve())
    sanitized = sanitized.replace(executable, "python")
    for private_path, public_ref in sorted(
        replacements.items(),
        key=lambda item: len(item[0]),
        reverse=True,
    ):
        sanitized = sanitized.replace(private_path, public_ref)
    sanitized = sanitized.replace(str(root.resolve()), ".")
    sanitized = LOCAL_FILE_URL_PATTERN.sub("<local-file-url-redacted>", sanitized)
    return LOCAL_ABSOLUTE_PATH_PATTERN.sub("<local-path-redacted>", sanitized)


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
    embedding_summary = _embedding_dataset_summary(datasets)
    if embedding_summary is not None:
        out["embedding"] = embedding_summary
    reranker_summary = _reranker_dataset_summary(datasets)
    if reranker_summary is not None:
        out["reranker"] = reranker_summary
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


def _embedding_dataset_summary(datasets: dict[str, list[dict[str, Any]]]) -> dict[str, Any] | None:
    card = _first_dataset_card(datasets.get("embedding_dataset_card", []))
    counts = {
        "corpusCount": len(datasets.get("embedding_corpus", [])),
        "trainPairCount": len(datasets.get("embedding_train_pairs", [])),
        "valPairCount": len(datasets.get("embedding_val_pairs", [])),
        "trainTripletCount": len(datasets.get("embedding_train_triplets", [])),
        "valTripletCount": len(datasets.get("embedding_val_triplets", [])),
        "hardNegativeCount": len(datasets.get("embedding_hard_negatives", [])),
        "evalCount": len(datasets.get("embedding_eval_retrieval", [])),
    }
    if not any(counts.values()) and not card:
        return None
    return {
        "model": card.get("model") or "Qwen/Qwen3-Embedding-0.6B",
        "fallbackModel": "current-baseline-embedding-model",
        "teacherModel": card.get("teacherModel") or "Qwen/Qwen3-Embedding-4B",
        "usedFallback": False,
        **counts,
        "pairCount": counts["trainPairCount"] + counts["valPairCount"],
        "tripletCount": counts["trainTripletCount"] + counts["valTripletCount"],
        "generated": any(counts.values()),
        "metrics": {
            "recallAt1": 0.0,
            "recallAt5": 0.0,
            "mrr": 0.0,
            "ndcgAt5": 0.0,
            "hardNegativeAccuracy": 0.0,
            "toolRetrievalAccuracy": 0.0,
            "sourceMapRetrievalAccuracy": 0.0,
            "runtimeRepairRetrievalAccuracy": 0.0,
        },
        "datasetCard": {
            "schemaVersion": card.get("schemaVersion"),
            "task": card.get("task"),
            "promotionMetrics": card.get("promotionMetrics", {}),
            "families": card.get("families", []),
        },
    }


def _reranker_dataset_summary(datasets: dict[str, list[dict[str, Any]]]) -> dict[str, Any] | None:
    card = _first_dataset_card(datasets.get("reranker_dataset_card", []))
    counts = {
        "trainPairCount": len(datasets.get("reranker_train_pairs", [])),
        "valPairCount": len(datasets.get("reranker_val_pairs", [])),
        "hardNegativePairCount": len(datasets.get("reranker_hard_negative_pairs", [])),
        "evalCount": len(datasets.get("reranker_eval_reranking", [])),
    }
    if not any(counts.values()) and not card:
        return None
    return {
        "model": card.get("model") or "Qwen/Qwen3-Reranker-0.6B",
        "fallbackMode": "embedding-only",
        "teacherModel": card.get("teacherModel") or "Qwen/Qwen3-Reranker-4B",
        "enabledByDefault": False,
        **counts,
        "pairCount": counts["trainPairCount"] + counts["valPairCount"],
        "generated": any(counts.values()),
        "metrics": {
            "rerankedRecallAt1": 0.0,
            "rerankedNdcgAt5": 0.0,
            "hardNegativePairAccuracy": 0.0,
            "top5ReorderWinRate": 0.0,
            "p95RerankLatencyRegression": 0.0,
        },
        "datasetCard": {
            "schemaVersion": card.get("schemaVersion"),
            "task": card.get("task"),
            "promotionMetrics": card.get("promotionMetrics", {}),
            "families": card.get("families", []),
        },
    }


def _first_dataset_card(records: list[dict[str, Any]]) -> dict[str, Any]:
    if records and isinstance(records[0], dict):
        return records[0]
    return {}


def _runtime_summary(
    runtime_reports: list[dict[str, Any]],
    *,
    all_runtime_reports: list[dict[str, Any]] | None = None,
    runtime_proof: dict[str, Any] | None = None,
) -> dict[str, Any]:
    failures = [failure for report in runtime_reports for failure in report.get("failures", []) if isinstance(failure, dict)]
    executable_failures = [failure for failure in failures if not _is_skipped_live_model_generation(failure)]
    skipped_failures = [failure for failure in failures if _is_skipped_live_model_generation(failure)]
    total_report_count = len(all_runtime_reports) if all_runtime_reports is not None else len(runtime_reports)
    proof = runtime_proof if isinstance(runtime_proof, dict) else {}
    by_type: dict[str, int] = {}
    by_all_type: dict[str, int] = {}
    by_layer: dict[str, int] = {}
    for failure in executable_failures:
        by_type[str(failure.get("type") or "unknown")] = by_type.get(str(failure.get("type") or "unknown"), 0) + 1
    for failure in failures:
        by_all_type[str(failure.get("type") or "unknown")] = by_all_type.get(str(failure.get("type") or "unknown"), 0) + 1
        by_layer[str(failure.get("sourceLayer") or "unknown")] = by_layer.get(str(failure.get("sourceLayer") or "unknown"), 0) + 1
    return {
        "reportCount": len(runtime_reports),
        "totalReportCount": total_report_count,
        "buildRejectedReportCount": max(0, total_report_count - len(runtime_reports)),
        "currentProofReportCount": 0,
        "verifiedAtAssessmentReportCount": int(
            proof.get("verifiedReportCount") or 0
        ),
        "freshAtExplicitReferenceReportCount": int(
            proof.get("freshAtExplicitReferenceReportCount") or 0
        ),
        "historicalReportCount": int(proof.get("historicalReportCount") or 0),
        "staleReportCount": int(proof.get("staleReportCount") or 0),
        "sourceRevisionMismatchReportCount": int(
            proof.get("sourceRevisionMismatchReportCount") or 0
        ),
        "freshnessUnverifiedReportCount": int(
            proof.get("freshnessUnverifiedReportCount") or 0
        ),
        "failureCount": len(executable_failures),
        "rawFailureCount": len(failures),
        "skippedLiveModelGenerationCount": len(skipped_failures),
        "failureTypes": dict(sorted(by_type.items())),
        "allFailureTypes": dict(sorted(by_all_type.items())),
        "sourceLayers": dict(sorted(by_layer.items())),
    }


AGENT_JSON_MODEL_ROOT_CAUSES = {
    "agent_json_empty_stream",
    "agent_json_completed_without_text",
    "agent_json_stop_before_first_token",
    "agent_json_resource_budget_denied_before_first_token",
    "agent_json_cancelled_before_first_token",
    "agent_json_decode_budget_zero",
    "agent_json_model_not_loaded",
    "agent_json_slot_unavailable",
    "agent_json_runtime_unavailable",
    "agent_json_empty_generation",
    "agent_json_parse_empty",
    "agent_json_parse_error",
    "agent_json_context_overflow",
}

RUNTIME_ENVIRONMENT_ROOT_CAUSES = {
    "runtime_environment_deferred",
}

NON_BLOCKING_RUNTIME_DIAGNOSTIC_ROOT_CAUSES = {
    "deterministic_compatibility_not_training_evidence",
    "deterministic_compatibility_not_live_evidence",
    "runtime_environment_deferred",
}


def _is_skipped_live_model_generation(failure: dict[str, Any]) -> bool:
    root_cause = str(failure.get("rootCauseCategory") or "")
    explicit_categories = EXPLICIT_MODEL_EVIDENCE_CATEGORIES | AGENT_JSON_MODEL_ROOT_CAUSES
    if root_cause in explicit_categories:
        return False
    scenario = failure.get("e2eScenario")
    if isinstance(scenario, dict) and scenario.get("modelEvidenceRootCause") in explicit_categories:
        return False
    if isinstance(scenario, dict) and scenario.get("skippedLiveModelRun") is True:
        return True
    return failure.get("skippedLiveModelRun") is True


def _runtime_gap_category(failure: dict[str, Any]) -> str:
    root_cause = _runtime_root_cause_category(failure)
    if str(failure.get("type") or "") == "e2e_architecture_finalizer_failure":
        return "architecture_finalizer_failure"
    if str(failure.get("type") or "") == "persistent_diagnostics_scenario_not_passed" and failure.get("remediationProposals"):
        return "persistent_diagnostics_remediation"
    if root_cause == "agent_json_context_overflow":
        return "prompt_budget_overflow"
    if root_cause in RUNTIME_ENVIRONMENT_ROOT_CAUSES:
        return root_cause
    if root_cause in EXPLICIT_MODEL_EVIDENCE_CATEGORIES or root_cause in AGENT_JSON_MODEL_ROOT_CAUSES:
        return root_cause
    if _is_skipped_live_model_generation(failure):
        return "skipped_live_model_generation"
    if root_cause == "manifest_mismatch":
        return "manifest_mismatch"
    if root_cause == "permission_config_issue":
        return "runtime_permission_config"
    return "runtime_drift"


def _runtime_root_cause_category(failure: dict[str, Any]) -> str:
    explicit = str(failure.get("rootCauseCategory") or "")
    if explicit in RUNTIME_ENVIRONMENT_ROOT_CAUSES:
        return explicit
    if explicit in EXPLICIT_MODEL_EVIDENCE_CATEGORIES or explicit in AGENT_JSON_MODEL_ROOT_CAUSES:
        return explicit
    scenario = failure.get("e2eScenario")
    if isinstance(scenario, dict):
        for key in ("modelEvidenceRootCause", "modelEvidenceStatus"):
            scenario_root = str(scenario.get(key) or "")
            if scenario_root in RUNTIME_ENVIRONMENT_ROOT_CAUSES:
                return scenario_root
            if scenario_root in EXPLICIT_MODEL_EVIDENCE_CATEGORIES or scenario_root in AGENT_JSON_MODEL_ROOT_CAUSES:
                return scenario_root
    failure_type = str(failure.get("type") or "").lower()
    if failure_type == "e2e_runtime_environment_deferred":
        return "runtime_environment_deferred"
    actual = str(failure.get("actual") or failure.get("final") or "").lower()
    problem = str(failure.get("problem") or "").lower()
    combined = f"{failure_type}\n{actual}\n{problem}"

    if _is_skipped_live_model_generation(failure):
        return "skipped_live_model_generation"
    if any(token in combined for token in ["unknown_tool", "not_allowed", "static_manifest", "manifest"]):
        return "manifest_mismatch"
    if any(token in combined for token in [
        "permission",
        "access was denied",
        "access denied",
        "not signed in",
        "sign in",
        "authorization",
        "oauth",
        "unavailable in this build",
        "tools are unavailable",
    ]):
        return "permission_config_issue"
    if any(token in combined for token in ["unimplemented", "unsupported", "missing implementation", "not implemented"]):
        return "true_missing_implementation"
    return "stale_or_unclassified_runtime_evidence"


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

    for record in datasets.get("self_model_eval", []):
        candidates.append(_scenario_from_eval_record(record, source_family="self_model_eval"))
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
                "Export the TestFlight + Agent Grounding package JSON.",
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
    *,
    ingestion_runtime_reports: list[dict[str, Any]] | None = None,
    runtime_proof: dict[str, Any] | None = None,
) -> dict[str, Any]:
    has_runtime = bool(runtime_reports)
    ingested_reports = (
        ingestion_runtime_reports
        if ingestion_runtime_reports is not None
        else runtime_reports
    )
    proof = runtime_proof if isinstance(runtime_proof, dict) else {}
    verified_report_count = int(proof.get("verifiedReportCount") or 0)
    proof_satisfied_at_assessment = (
        proof.get("proofSatisfiedAtAssessment") is True
    )
    proof_basis = (
        proof.get("basis") if isinstance(proof.get("basis"), dict) else {}
    )
    build_rejected_count = max(0, len(runtime_reports) - len(ingested_reports))
    if proof_satisfied_at_assessment:
        status = STRICT_RUNTIME_RECEIPT_STATUS
    elif ingested_reports:
        status = "historical-runtime-audit-ingested"
    elif has_runtime:
        status = "runtime-audit-build-rejected"
    else:
        status = "awaiting-runtime-audit"
    return {
        "mode": config.app_run_mode,
        "buildLabel": config.testflight_build_label,
        "status": status,
        "proofStatus": proof.get("status") or "not-assessed",
        "proofBasis": proof_basis,
        "proofSatisfiedAtAssessment": proof_satisfied_at_assessment,
        "verifiedAt": proof_basis.get("verifiedAt"),
        "validUntil": proof_basis.get("validUntil"),
        "packageSha256": proof_basis.get("packageSha256"),
        "scope": proof_basis.get("scope"),
        "requiresTestFlightAppRun": config.app_run_mode.casefold() == "testflight",
        "requireRuntimeAuditForPass": config.require_testflight_runtime_audit,
        "runtimeAuditProvided": has_runtime,
        "currentRuntimeAuditProvided": False,
        "currentRuntimeAuditProvidedSemantics": "non-enduring-always-false",
        "runtimeAuditReportCount": len(runtime_reports),
        "ingestedRuntimeAuditReportCount": len(ingested_reports),
        "buildSelectedRuntimeAuditReportCount": len(ingested_reports),
        "buildRejectedRuntimeAuditReportCount": build_rejected_count,
        "currentRuntimeAuditReportCount": 0,
        "currentRuntimeAuditReportCountSemantics": "non-enduring-always-zero",
        "verifiedAtAssessmentRuntimeAuditReportCount": verified_report_count,
        "freshAtExplicitReferenceRuntimeAuditReportCount": int(
            proof.get("freshAtExplicitReferenceReportCount") or 0
        ),
        "historicalRuntimeAuditReportCount": int(
            proof.get("historicalReportCount") or 0
        ),
        "staleRuntimeAuditReportCount": int(proof.get("staleReportCount") or 0),
        "sourceRevisionMismatchRuntimeAuditReportCount": int(
            proof.get("sourceRevisionMismatchReportCount") or 0
        ),
        "freshnessUnverifiedRuntimeAuditReportCount": int(
            proof.get("freshnessUnverifiedReportCount") or 0
        ),
        "scenarioQueuePath": TESTFLIGHT_SCENARIOS_FILE,
        "runbookPath": TESTFLIGHT_RUNBOOK_FILE,
        "scenarioCount": len(scenarios),
        "expectedExport": "lumen-testflight-agent-grounding-*.json from Agent Grounding > Export TestFlight + Agent Grounding Package",
        "nextIngestCommand": shlex.join([
            "python",
            "-m",
            "lumen_manifest_crawler",
            "improve-loop",
            "--root",
            _public_repo_path(config.root, config.root),
            "--output",
            _public_repo_path(config.output, config.root),
            "--loop-output",
            _public_repo_path(config.loop_output, config.root),
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
    all_runtime_reports: list[dict[str, Any]] | None = None,
    runtime_proof: dict[str, Any] | None = None,
    command_results: list[LoopCommandResult],
    config: AgentImprovementLoopConfig,
) -> list[dict[str, Any]]:
    gaps: list[dict[str, Any]] = []
    mismatch_runtime_reports = all_runtime_reports if all_runtime_reports is not None else runtime_reports

    if config.app_run_mode.casefold() == "testflight" and not mismatch_runtime_reports:
        resolved_loop_output = config.loop_output.resolve()
        gaps.append({
            "id": _stable_id("testflight_runtime_pending", config.testflight_build_label or "unlabeled"),
            "severity": "error" if config.require_testflight_runtime_audit else "warning",
            "category": "testflight_runtime_pending",
            "title": "TestFlight in-app audit export has not been ingested yet",
            "evidence": {
                "expectedExport": "lumen-testflight-agent-grounding-*.json",
                "source": "Agent Grounding > Export TestFlight + Agent Grounding Package",
                "scenarioQueue": str(resolved_loop_output / "testflight_scenarios.jsonl"),
                "runbook": str(resolved_loop_output / "TESTFLIGHT_RUNBOOK.md"),
            },
            "recommendedAction": "Compile/distribute the TestFlight build, run Agent Grounding in the app, export the TestFlight + Agent Grounding package JSON, then rerun improve-loop with --runtime-audit <json>.",
        })

    build_mismatch_gap = _testflight_runtime_build_mismatch_gap(mismatch_runtime_reports, config)
    if build_mismatch_gap is not None:
        gaps.append(build_mismatch_gap)

    proof = runtime_proof if isinstance(runtime_proof, dict) else {}
    if (
        config.verify_runtime_audit_now
        and proof.get("proofSatisfiedAtAssessment") is not True
    ):
        gaps.append({
            "id": _stable_id("runtime_audit_strict_verification_failed", proof),
            "severity": "error",
            "category": "runtime_audit_strict_verification_failed",
            "title": "Requested strict runtime-audit verification did not pass",
            "evidence": {
                "proofStatus": proof.get("status"),
                "verificationFailureCode": (
                    proof.get("basis", {}).get("verificationFailureCode")
                    if isinstance(proof.get("basis"), dict)
                    else None
                ),
                "proofSatisfiedAtAssessment": False,
            },
            "recommendedAction": (
                "Run non-deterministically with exactly one unchanged physical-device "
                "DEBUG package, the exact expected build number, and a passing strict verifier."
            ),
        })
    if (
        config.app_run_mode.casefold() == "testflight"
        and config.require_testflight_runtime_audit
        and mismatch_runtime_reports
        and build_mismatch_gap is None
        and proof.get("currentProofComplete") is not True
    ):
        gaps.append({
            "id": _stable_id("testflight_runtime_proof_unverified", proof),
            "severity": "error",
            "category": "testflight_runtime_proof_unverified",
            "title": "TestFlight runtime audit lacks TestFlight-capable verification",
            "evidence": {
                "proofStatus": proof.get("status"),
                "proofBasis": proof.get("basis"),
                "historicalReportCount": proof.get("historicalReportCount"),
                "staleReportCount": proof.get("staleReportCount"),
                "sourceRevisionMismatchReportCount": proof.get(
                    "sourceRevisionMismatchReportCount"
                ),
                "freshnessUnverifiedReportCount": proof.get(
                    "freshnessUnverifiedReportCount"
                ),
                "debugReceiptUnsupportedForTestFlight": (
                    proof.get("status")
                    == "unsupported-debug-receipt-for-testflight"
                ),
            },
            "recommendedAction": (
                "Use a distinct TestFlight-capable verification path bound to the exact "
                "manifest source revision, TestFlight build, package hash, and host time. "
                "The physical-device DEBUG receipt is explicitly insufficient for this gate."
            ),
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
        skipped_live_generation = _is_skipped_live_model_generation(failure)
        severity = _runtime_gap_severity(failure_type, failure, skipped_live_generation=skipped_live_generation)
        category = _runtime_gap_category(failure)
        if _is_non_blocking_runtime_diagnostic(
            failure,
            category=category,
            skipped_live_generation=skipped_live_generation,
        ):
            continue
        evidence = dict(failure)
        evidence["rootCauseCategory"] = _runtime_root_cause_category(failure)
        gaps.append({
            "id": _stable_id("runtime", failure),
            "severity": severity,
            "category": category,
            "title": failure_type,
            "evidence": evidence,
            "recommendedAction": _runtime_recommendation(failure_type, failure=failure, skipped_live_generation=skipped_live_generation),
        })

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
    }
    for family in sorted(required_families):
        if len(datasets.get(family, [])) == 0:
            gaps.append({
                "id": _stable_id("empty_family", family),
                "severity": "error",
                "category": "dataset_coverage",
                "title": f"Empty dataset family: {family}",
                "evidence": {"family": family},
                "recommendedAction": f"Add generators or runtime inputs that produce {family} records.",
            })
    if runtime_reports and len(datasets.get("runtime_audit_repairs", [])) == 0:
        gaps.append({
            "id": _stable_id("empty_family", "runtime_audit_repairs"),
            "severity": "warning",
            "category": "dataset_coverage",
            "title": "Empty dataset family: runtime_audit_repairs",
            "evidence": {"family": "runtime_audit_repairs"},
            "recommendedAction": "Add current-build runtime inputs that produce runtime_audit_repairs records.",
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


def _testflight_runtime_build_mismatch_gap(
    runtime_reports: list[dict[str, Any]],
    config: AgentImprovementLoopConfig,
) -> dict[str, Any] | None:
    expected_build = str(
        config.runtime_audit_expected_build_number
        or config.testflight_build_label
        or ""
    ).strip()
    if config.app_run_mode.casefold() != "testflight" or not expected_build or not runtime_reports:
        return None

    mismatched: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    matching = 0
    for report in runtime_reports:
        build_number = _runtime_report_app_build_number(report)
        evidence = {
            "source": report.get("_source"),
            "sourceFormat": report.get("_sourceFormat"),
            "sourceLayer": report.get("_sourceLayer"),
            "appBuildNumber": build_number,
            "generatedAt": report.get("generatedAt"),
        }
        if build_number == expected_build:
            matching += 1
        elif build_number:
            mismatched.append(evidence)
        else:
            missing.append(evidence)

    if matching > 0:
        return None
    if not mismatched and not missing:
        return None

    return {
        "id": _stable_id("testflight_runtime_build_mismatch", {
            "expectedBuild": expected_build,
            "mismatched": mismatched,
            "missing": missing,
        }),
        "severity": "error",
        "category": "testflight_runtime_build_mismatch",
        "title": "Runtime audit export does not prove the current TestFlight build",
        "evidence": {
            "expectedBuildLabel": expected_build,
            "matchingReportCount": matching,
            "mismatchedReportCount": len(mismatched),
            "missingBuildReportCount": len(missing),
            "mismatchedReports": mismatched[:10],
            "missingBuildReports": missing[:10],
            "rootCauseCategory": "stale_audit_evidence",
        },
        "recommendedAction": "Install build "
        f"{expected_build}, run Agent Grounding in that TestFlight app, export the TestFlight + Agent Grounding package JSON, and ingest only that current-build package.",
    }


def _ingestion_runtime_reports(
    runtime_reports: list[dict[str, Any]],
    config: AgentImprovementLoopConfig,
) -> list[dict[str, Any]]:
    """Select reports for historical repair/training ingestion by build only.

    This is intentionally not a proof-freshness decision. Device-debug and
    unlabeled inputs remain useful historical repair inputs, while TestFlight
    runs with an explicit build label continue to reject other builds.
    """
    return [
        report
        for report in runtime_reports
        if _runtime_report_matches_testflight_build(report, config)
    ]


def _annotate_runtime_reports_for_training(
    runtime_reports: list[dict[str, Any]],
    runtime_proof: dict[str, Any],
) -> list[dict[str, Any]]:
    """Bind repair inputs to their non-current evidence classification.

    Runtime reports remain useful for historical repairs even when they cannot
    prove the current build. Keep that ingestion lane, but make the boundary an
    explicit compiler input so neither clean nor failing reports can be turned
    into an unqualified current-runtime claim.
    """

    proof_status = str(runtime_proof.get("status") or "historical-unverified")
    return [
        {
            **without_internal_artifact_bindings(report),
            "_runtimeProofStatus": proof_status,
            "_runtimeCurrentProof": False,
            "_runtimeHistoricalObservation": True,
        }
        for report in runtime_reports
    ]


def _verify_runtime_audit_at_host_now(
    runtime_reports: list[dict[str, Any]],
    config: AgentImprovementLoopConfig,
    *,
    expected_source_revision: str | None,
    expected_working_tree_digest: str | None = None,
    source_dirty_state: bool | None = False,
) -> dict[str, Any] | None:
    """Run the strict DEBUG verifier over one raw package and capture its receipt."""

    if not config.verify_runtime_audit_now:
        return None
    if config.deterministic:
        return {
            "status": "strict-verifier-requires-non-deterministic-mode",
            "proofSatisfiedAtAssessment": False,
            "failureCode": "host_clock_verification_forbidden_in_deterministic_mode",
        }
    if config.app_run_mode.casefold() != "device-debug":
        return {
            "status": "unsupported-debug-receipt-for-testflight",
            "proofSatisfiedAtAssessment": False,
            "failureCode": "debug_receipt_scope_mismatch",
        }
    if source_dirty_state is not False:
        return {
            "status": "strict-verifier-source-not-clean",
            "proofSatisfiedAtAssessment": False,
            "failureCode": "source_worktree_not_clean_at_manifest_assessment",
        }
    expected_revision = str(expected_source_revision or "").strip()
    expected_digest = str(expected_working_tree_digest or "").strip().lower()
    expected_build = str(config.runtime_audit_expected_build_number or "").strip()
    if not expected_revision:
        return {
            "status": "strict-verifier-source-revision-missing",
            "proofSatisfiedAtAssessment": False,
            "failureCode": "expected_source_revision_missing",
        }
    if not expected_build:
        return {
            "status": "strict-verifier-build-number-missing",
            "proofSatisfiedAtAssessment": False,
            "failureCode": "expected_build_number_missing",
        }
    if not re.fullmatch(r"[0-9a-f]{64}", expected_digest):
        return {
            "status": "strict-verifier-working-tree-digest-missing",
            "proofSatisfiedAtAssessment": False,
            "failureCode": "expected_working_tree_digest_missing_or_invalid",
        }

    package_paths = sorted({
        path
        for report in runtime_reports
        if _is_strict_interactive_package_report(report)
        if (path := _runtime_report_raw_package_path(report, config.root)) is not None
    })
    if len(package_paths) != 1:
        return {
            "status": "strict-verifier-package-not-unique",
            "proofSatisfiedAtAssessment": False,
            "failureCode": "expected_exactly_one_raw_package",
            "candidatePackageCount": len(package_paths),
        }

    package_path = package_paths[0]
    try:
        raw_bytes = package_path.read_bytes()
    except OSError:
        return {
            "status": "strict-verifier-package-unreadable",
            "proofSatisfiedAtAssessment": False,
            "failureCode": "raw_package_unreadable",
        }
    package_reports = [
        report
        for report in runtime_reports
        if _runtime_report_belongs_to_package(report, package_path, config.root)
    ]
    loaded_binding_values = [
        (report.get("_artifactSha256"), report.get("_artifactByteCount"))
        for report in package_reports
    ]
    loaded_bindings_valid = all(
        isinstance(sha256, str)
        and isinstance(byte_count, int)
        and not isinstance(byte_count, bool)
        for sha256, byte_count in loaded_binding_values
    )
    loaded_bindings = (
        set(loaded_binding_values) if loaded_bindings_valid else set()
    )
    expected_loaded_binding = (hashlib.sha256(raw_bytes).hexdigest(), len(raw_bytes))
    if (
        not package_reports
        or not loaded_bindings_valid
        or len(loaded_bindings) != 1
        or next(iter(loaded_bindings), None) != expected_loaded_binding
    ):
        return {
            "status": "strict-verifier-loaded-artifact-mismatch",
            "proofSatisfiedAtAssessment": False,
            "failureCode": "loaded_report_bytes_do_not_match_verifier_input",
            "packageOwnedReportCount": len(package_reports),
        }
    verifier_path = config.root.resolve() / "tools" / "verify_interactive_model_tool_evidence.py"
    try:
        verifier_bytes = verifier_path.read_bytes()
    except OSError:
        return {
            "status": "strict-verifier-implementation-unreadable",
            "proofSatisfiedAtAssessment": False,
            "failureCode": "verifier_implementation_unreadable",
        }
    tracked_verifier_bytes = _tracked_verifier_bytes(
        config.root,
        expected_revision,
    )
    if tracked_verifier_bytes is None or tracked_verifier_bytes != verifier_bytes:
        return {
            "status": "strict-verifier-implementation-untrusted",
            "proofSatisfiedAtAssessment": False,
            "failureCode": "verifier_bytes_do_not_match_source_revision",
        }
    if not _working_file_matches_revision(
        config.root,
        expected_revision,
        APP_BEHAVIOR_MANIFEST_PATH,
    ):
        return {
            "status": "strict-verifier-app-manifest-untrusted",
            "proofSatisfiedAtAssessment": False,
            "failureCode": "app_behavior_manifest_does_not_match_source_revision",
        }
    verifier_sha256 = hashlib.sha256(verifier_bytes).hexdigest()
    with tempfile.TemporaryDirectory(prefix="lumen-runtime-proof-") as temporary:
        temporary_root = Path(temporary)
        receipt_path = temporary_root / "receipt.json"
        verifier_snapshot_path = temporary_root / "verified-runtime-evidence.py"
        verifier_snapshot_path.write_bytes(verifier_bytes)
        command = [
            sys.executable,
            str(verifier_snapshot_path),
            str(package_path),
            "--expected-source-revision",
            expected_revision,
            "--expected-build-number",
            expected_build,
            "--expected-working-tree-digest",
            expected_digest,
            "--max-age-seconds",
            str(config.runtime_audit_max_age_seconds),
            "--receipt-output",
            str(receipt_path),
        ]
        try:
            completed = subprocess.run(
                command,
                cwd=config.root.resolve(),
                check=False,
                capture_output=True,
                text=True,
                errors="replace",
                timeout=120,
            )
        except (OSError, subprocess.SubprocessError):
            return {
                "status": "strict-verifier-execution-failed",
                "proofSatisfiedAtAssessment": False,
                "failureCode": "verifier_process_failed",
            }
        if completed.returncode != 0 or not receipt_path.is_file():
            return {
                "status": "strict-verifier-rejected",
                "proofSatisfiedAtAssessment": False,
                "failureCode": "strict_verifier_rejected_package",
                "verifierReturnCode": completed.returncode,
            }
        try:
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {
                "status": "strict-verifier-receipt-invalid",
                "proofSatisfiedAtAssessment": False,
                "failureCode": "receipt_parse_failed",
            }

        try:
            if verifier_snapshot_path.read_bytes() != verifier_bytes:
                return {
                    "status": "strict-verifier-snapshot-mutated",
                    "proofSatisfiedAtAssessment": False,
                    "failureCode": "executed_verifier_snapshot_changed",
                }
        except OSError:
            return {
                "status": "strict-verifier-snapshot-unreadable",
                "proofSatisfiedAtAssessment": False,
                "failureCode": "executed_verifier_snapshot_unreadable",
            }

    try:
        if verifier_path.read_bytes() != verifier_bytes:
            return {
                "status": "strict-verifier-implementation-mutated",
                "proofSatisfiedAtAssessment": False,
                "failureCode": "verifier_changed_during_verification",
            }
    except OSError:
        return {
            "status": "strict-verifier-implementation-unreadable",
            "proofSatisfiedAtAssessment": False,
            "failureCode": "verifier_implementation_unreadable_after_verification",
        }

    if not _working_file_matches_revision(
        config.root,
        expected_revision,
        APP_BEHAVIOR_MANIFEST_PATH,
    ):
        return {
            "status": "strict-verifier-app-manifest-mutated",
            "proofSatisfiedAtAssessment": False,
            "failureCode": "app_behavior_manifest_changed_during_verification",
        }

    try:
        if package_path.read_bytes() != raw_bytes:
            return {
                "status": "strict-verifier-package-mutated",
                "proofSatisfiedAtAssessment": False,
                "failureCode": "raw_package_changed_during_verification",
            }
    except OSError:
        return {
            "status": "strict-verifier-package-unreadable",
            "proofSatisfiedAtAssessment": False,
            "failureCode": "raw_package_unreadable_after_verification",
        }

    receipt_failure = _strict_receipt_failure(
        receipt,
        raw_bytes=raw_bytes,
        expected_source_revision=expected_revision,
        expected_build_number=expected_build,
        expected_working_tree_digest=expected_digest,
        expected_verifier_sha256=verifier_sha256,
    )
    if receipt_failure is not None:
        return {
            "status": "strict-verifier-receipt-invalid",
            "proofSatisfiedAtAssessment": False,
            "failureCode": receipt_failure,
        }
    binding = receipt["binding"]
    package = receipt["package"]
    verifier = receipt["verifier"]
    return {
        "status": STRICT_RUNTIME_RECEIPT_STATUS,
        "proofSatisfiedAtAssessment": True,
        "verifiedAt": receipt["verifiedAt"],
        "validUntil": receipt["validUntil"],
        "scope": receipt["scope"],
        "packageSha256": package["sha256"],
        "packageByteCount": package["byteCount"],
        "sourceRevision": binding["sourceRevision"],
        "buildNumber": binding["buildNumber"],
        "workingTreeDigest": binding["workingTreeDigest"],
        "sourceDirtyState": binding["sourceDirtyState"],
        "executionEnvironment": binding["executionEnvironment"],
        "scenarioID": binding["scenarioID"],
        "toolID": binding["toolID"],
        "verifierName": verifier["name"],
        "verifierContractVersion": verifier["contractVersion"],
        "verifierSourceSha256": verifier["sourceSha256"],
        "verifiedReportCount": sum(
            1
            for report in runtime_reports
            if _runtime_report_belongs_to_package(report, package_path, config.root)
        ),
    }


def _is_strict_interactive_package_report(report: dict[str, Any]) -> bool:
    return bool(
        report.get("_sourceFormat") == "testflight_agent_grounding_package"
        and report.get("_sourceLayer") == "agentGroundingRuntimeAudit"
        and report.get("manifestSource")
        == "interactive-model-tool-validation-live-e2e"
    )


def _runtime_report_raw_package_path(
    report: dict[str, Any],
    root: Path,
) -> Path | None:
    source = str(report.get("_source") or "").strip()
    if not source:
        return None
    path = Path(source)
    if not path.is_absolute():
        path = root.resolve() / path
    return path.resolve()


def _runtime_report_belongs_to_package(
    report: dict[str, Any],
    package_path: Path,
    root: Path,
) -> bool:
    source = str(report.get("_source") or "")
    if source.endswith("#liveE2EReport"):
        source = source.removesuffix("#liveE2EReport")
    if not source:
        return False
    path = Path(source)
    if not path.is_absolute():
        path = root.resolve() / path
    return path.resolve() == package_path


def _tracked_verifier_bytes(root: Path, revision: str) -> bytes | None:
    """Read the verifier implementation committed at the asserted revision."""

    return _tracked_file_bytes(
        root,
        revision,
        Path("tools/verify_interactive_model_tool_evidence.py"),
    )


def _tracked_file_bytes(
    root: Path,
    revision: str,
    relative_path: Path,
) -> bytes | None:
    """Read one repository file exactly as committed at ``revision``."""

    try:
        completed = subprocess.run(
            [
                "git",
                "show",
                f"{revision}:{relative_path.as_posix()}",
            ],
            cwd=root.resolve(),
            check=False,
            capture_output=True,
            timeout=20,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return bytes(completed.stdout)


def _working_file_matches_revision(
    root: Path,
    revision: str,
    relative_path: Path,
) -> bool:
    """Return whether a working-tree file matches its asserted commit bytes."""

    try:
        working_bytes = (root.resolve() / relative_path).read_bytes()
    except OSError:
        return False
    tracked_bytes = _tracked_file_bytes(root, revision, relative_path)
    return tracked_bytes is not None and working_bytes == tracked_bytes


def _strict_receipt_failure(
    receipt: Any,
    *,
    raw_bytes: bytes,
    expected_source_revision: str,
    expected_build_number: str,
    expected_working_tree_digest: str,
    expected_verifier_sha256: str,
) -> str | None:
    if not isinstance(receipt, dict):
        return "receipt_not_object"
    package = receipt.get("package")
    binding = receipt.get("binding")
    verifier = receipt.get("verifier")
    if not isinstance(package, dict) or not isinstance(binding, dict) or not isinstance(verifier, dict):
        return "receipt_sections_missing"
    expected = {
        "schemaVersion": STRICT_RUNTIME_RECEIPT_SCHEMA,
        "status": STRICT_RUNTIME_RECEIPT_STATUS,
        "scope": STRICT_RUNTIME_RECEIPT_SCOPE,
    }
    if any(receipt.get(key) != value for key, value in expected.items()):
        return "receipt_identity_mismatch"
    expected_binding = {
        "bundleIdentifier": "com.27pm.lumenclone",
        "sourceRevision": expected_source_revision,
        "buildNumber": expected_build_number,
        "workingTreeDigest": expected_working_tree_digest,
        "sourceDirtyState": False,
        "executionEnvironment": "physical-iPhone",
        "scenarioID": "interactive-model-tool-alarm-authorization",
        "toolID": "alarm.authorization_status",
    }
    if any(binding.get(key) != value for key, value in expected_binding.items()):
        return "receipt_binding_mismatch"
    if package.get("sha256") != hashlib.sha256(raw_bytes).hexdigest():
        return "receipt_package_hash_mismatch"
    if package.get("byteCount") != len(raw_bytes):
        return "receipt_package_size_mismatch"
    verified_at, verified_status = _parse_runtime_audit_reference_time(receipt.get("verifiedAt"))
    valid_until, valid_status = _parse_runtime_audit_reference_time(receipt.get("validUntil"))
    if verified_status != "valid" or valid_status != "valid" or verified_at is None or valid_until is None:
        return "receipt_time_invalid"
    if verified_at > valid_until or datetime.now(timezone.utc) >= valid_until:
        return "receipt_expired"
    expected_verifier = {
        "name": STRICT_RUNTIME_VERIFIER_NAME,
        "contractVersion": STRICT_RUNTIME_VERIFIER_CONTRACT_VERSION,
        "sourceSha256": expected_verifier_sha256,
    }
    if any(verifier.get(key) != value for key, value in expected_verifier.items()):
        return "receipt_verifier_identity_mismatch"
    return None


def _assess_runtime_audit_proof(
    runtime_reports: list[dict[str, Any]],
    config: AgentImprovementLoopConfig,
    *,
    expected_source_revision: str | None,
    verification_assessment: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Classify current proof without changing the historical ingestion set."""

    expected_revision = str(expected_source_revision or "").strip()
    reference_time, reference_error = _parse_runtime_audit_reference_time(
        config.runtime_audit_reference_time
    )
    max_age_valid = (
        isinstance(config.runtime_audit_max_age_seconds, int)
        and not isinstance(config.runtime_audit_max_age_seconds, bool)
        and config.runtime_audit_max_age_seconds > 0
    )
    expected_build = str(
        config.runtime_audit_expected_build_number
        or config.testflight_build_label
        or ""
    ).strip()
    build_basis_valid = (
        config.app_run_mode.casefold() != "testflight" or bool(expected_build)
    )
    reference_classification_configured = bool(
        expected_revision
        and reference_time is not None
        and max_age_valid
        and build_basis_valid
    )

    reference_fresh_reports: list[dict[str, Any]] = []
    stale_report_count = 0
    source_revision_mismatch_count = 0
    freshness_unverified_count = 0
    future_timestamp_count = 0
    valid_until_candidates: list[datetime] = []

    for report in runtime_reports:
        source_revision_matches = (
            bool(expected_revision)
            and _runtime_report_source_revision(report) == expected_revision
        )
        if not source_revision_matches:
            source_revision_mismatch_count += 1

        timestamps, timestamps_valid = _runtime_report_proof_timestamps(report)
        timestamp_unverified = (
            reference_time is None
            or not max_age_valid
            or not timestamps
            or not timestamps_valid
            or not build_basis_valid
        )
        stale = False
        future = False
        if timestamps_valid and timestamps and max_age_valid:
            valid_until_candidates.extend(
                timestamp + timedelta(seconds=config.runtime_audit_max_age_seconds)
                for timestamp in timestamps
            )
        if not timestamp_unverified and reference_time is not None:
            ages = [
                (reference_time - timestamp).total_seconds()
                for timestamp in timestamps
            ]
            stale = any(
                age > config.runtime_audit_max_age_seconds
                for age in ages
            )
            future = any(
                age < -RUNTIME_AUDIT_MAX_FUTURE_SKEW_SECONDS
                for age in ages
            )
        if stale:
            stale_report_count += 1
        if future:
            future_timestamp_count += 1
        if timestamp_unverified or future:
            freshness_unverified_count += 1

        build_matches = (
            not expected_build
            or _runtime_report_app_build_number(report) == expected_build
        )
        if (
            reference_classification_configured
            and source_revision_matches
            and build_matches
            and not stale
            and not future
            and not timestamp_unverified
        ):
            reference_fresh_reports.append(report)

    # A caller-supplied reference time remains classification-only. The only
    # positive lane is the receipt created by the strict verifier in this run.
    current_proof_complete = False
    proof_satisfied_at_assessment = bool(
        isinstance(verification_assessment, dict)
        and verification_assessment.get("proofSatisfiedAtAssessment") is True
        and verification_assessment.get("status") == STRICT_RUNTIME_RECEIPT_STATUS
        and verification_assessment.get("scope") == STRICT_RUNTIME_RECEIPT_SCOPE
        and config.app_run_mode.casefold() == "device-debug"
    )
    historical_report_count = len(runtime_reports)
    if proof_satisfied_at_assessment:
        status = STRICT_RUNTIME_RECEIPT_STATUS
    elif isinstance(verification_assessment, dict):
        status = str(verification_assessment.get("status") or "strict-verifier-rejected")
    elif not runtime_reports:
        status = "not-provided"
    elif source_revision_mismatch_count:
        status = "historical-source-revision-mismatch"
    elif stale_report_count:
        status = "historical-stale"
    elif len(reference_fresh_reports) == len(runtime_reports):
        status = "fresh-at-explicit-reference-unverified"
    elif reference_fresh_reports:
        status = "mixed-reference-fresh-and-historical"
    else:
        status = "historical-unverified"

    basis = {
        "expectedSourceRevision": expected_revision or None,
        "referenceTime": (
            reference_time.isoformat() if reference_time is not None else None
        ),
        "referenceTimeStatus": (
            "valid"
            if reference_time is not None
            else reference_error
        ),
        "referenceTimeTrust": (
            "caller-supplied-unverified"
            if reference_time is not None
            else "absent"
        ),
        "maxAgeSeconds": config.runtime_audit_max_age_seconds,
        "maxAgeStatus": "valid" if max_age_valid else "invalid",
        "validUntil": (
            min(valid_until_candidates).isoformat()
            if valid_until_candidates
            else None
        ),
        "expectedBuildNumber": expected_build or None,
        "buildSelectionStatus": (
            "valid"
            if build_basis_valid and (expected_build or config.app_run_mode.casefold() != "testflight")
            else "unverified"
        ),
        "referenceClassificationConfigured": reference_classification_configured,
        "currentProofTrusted": False,
        "verificationReceiptProvided": proof_satisfied_at_assessment,
        "proofSatisfiedAtAssessment": proof_satisfied_at_assessment,
    }
    if proof_satisfied_at_assessment and verification_assessment is not None:
        basis.update({
            "verifiedAt": verification_assessment.get("verifiedAt"),
            "validUntil": verification_assessment.get("validUntil"),
            "scope": verification_assessment.get("scope"),
            "packageSha256": verification_assessment.get("packageSha256"),
            "packageByteCount": verification_assessment.get("packageByteCount"),
            "sourceRevision": verification_assessment.get("sourceRevision"),
            "buildNumber": verification_assessment.get("buildNumber"),
            "workingTreeDigest": verification_assessment.get(
                "workingTreeDigest"
            ),
            "sourceDirtyState": verification_assessment.get(
                "sourceDirtyState"
            ),
            "executionEnvironment": verification_assessment.get(
                "executionEnvironment"
            ),
            "scenarioID": verification_assessment.get("scenarioID"),
            "toolID": verification_assessment.get("toolID"),
            "verifierName": verification_assessment.get("verifierName"),
            "verifierContractVersion": verification_assessment.get(
                "verifierContractVersion"
            ),
            "verifierSourceSha256": verification_assessment.get(
                "verifierSourceSha256"
            ),
            "currentProofTrusted": False,
        })
    elif isinstance(verification_assessment, dict):
        basis["verificationFailureCode"] = verification_assessment.get(
            "failureCode"
        )
    return [], {
        "status": status,
        "basis": basis,
        "currentProofComplete": current_proof_complete,
        "proofSatisfiedAtAssessment": proof_satisfied_at_assessment,
        "verifiedReportCount": int(
            verification_assessment.get("verifiedReportCount") or 0
        ) if proof_satisfied_at_assessment and verification_assessment else 0,
        "freshAtExplicitReferenceReportCount": len(reference_fresh_reports),
        "historicalReportCount": historical_report_count,
        "staleReportCount": stale_report_count,
        "sourceRevisionMismatchReportCount": source_revision_mismatch_count,
        "freshnessUnverifiedReportCount": freshness_unverified_count,
        "futureTimestampReportCount": future_timestamp_count,
    }


def _parse_runtime_audit_reference_time(
    value: str | None,
) -> tuple[datetime | None, str]:
    if not isinstance(value, str) or not value.strip():
        return None, "not-provided"
    normalized = value.strip()
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None, "invalid"
    if parsed.tzinfo is None:
        return None, "invalid"
    return parsed.astimezone(timezone.utc), "valid"


def _runtime_report_proof_timestamps(
    report: dict[str, Any],
) -> tuple[list[datetime], bool]:
    raw_values = [report.get("generatedAt")]
    if report.get("reportFinishedAt") not in (None, ""):
        raw_values.append(report.get("reportFinishedAt"))
    timestamps: list[datetime] = []
    for value in raw_values:
        parsed, status = _parse_runtime_audit_reference_time(
            value if isinstance(value, str) else None
        )
        if parsed is None or status != "valid":
            return [], False
        timestamps.append(parsed)
    return timestamps, bool(timestamps)


def _runtime_report_source_revision(report: dict[str, Any]) -> str | None:
    source_revision = report.get("appSourceRevision")
    if source_revision:
        return str(source_revision)
    app = report.get("app")
    if isinstance(app, dict) and app.get("sourceRevision"):
        return str(app.get("sourceRevision"))
    return None


def _runtime_report_matches_testflight_build(
    report: dict[str, Any],
    config: AgentImprovementLoopConfig,
) -> bool:
    expected_build = str(config.testflight_build_label or "").strip()
    if config.app_run_mode.casefold() != "testflight" or not expected_build:
        return True
    return _runtime_report_app_build_number(report) == expected_build


def _runtime_report_app_build_number(report: dict[str, Any]) -> str | None:
    build_number = report.get("appBuildNumber")
    if build_number:
        return str(build_number)
    app = report.get("app")
    if isinstance(app, dict) and app.get("buildNumber"):
        return str(app.get("buildNumber"))
    return None


def _runtime_recommendation(
    failure_type: str,
    *,
    failure: dict[str, Any] | None = None,
    skipped_live_generation: bool = False,
) -> str:
    if skipped_live_generation:
        return "Rerun this scenario through the live app/model path and export fresh E2E evidence before treating it as a tool failure."
    if isinstance(failure, dict) and failure.get("trainable") is False:
        return "Quarantine this architecture/runtime/finalizer failure from SFT; add a deterministic regression test or runtime diagnostic instead."
    if failure_type == "agent_grounding_no_recent_model_traces":
        return "Fix runtime trace instrumentation or rerun the app before exporting; do not train from empty-trace evidence."
    if failure_type == "persistent_diagnostics_scenario_not_passed":
        action = _first_remediation_action(failure)
        if action:
            return action
        return "Fix the diagnostics scenario or app runtime path, then rerun persistent diagnostics before using the artifact."
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
    if failure_type == "trace_parse_error":
        return "Fix the tool-scoped trace producer or parser contract, then add a regression eval for the affected tool scope."
    return "Convert this failure into a REM repair sample and add a regression eval."


def _is_non_blocking_runtime_diagnostic(
    failure: dict[str, Any],
    *,
    category: str,
    skipped_live_generation: bool,
) -> bool:
    root_cause = _runtime_root_cause_category(failure)
    if root_cause in NON_BLOCKING_RUNTIME_DIAGNOSTIC_ROOT_CAUSES:
        return True
    if skipped_live_generation and failure.get("trainable") is False:
        return True
    return category == "architecture_finalizer_failure" and failure.get("trainable") is False


def _runtime_gap_severity(
    failure_type: str,
    failure: dict[str, Any],
    *,
    skipped_live_generation: bool,
) -> str:
    if skipped_live_generation:
        return "warning"
    if failure_type == "persistent_diagnostics_scenario_not_passed":
        return {
            "info": "warning",
            "warning": "error",
            "critical": "critical",
        }.get(str(failure.get("remediationSeverity") or ""), "error")
    if any(token in failure_type for token in ["unknown_tool", "sentinel", "not_allowed"]):
        return "critical"
    return "error"


def _first_remediation_action(failure: dict[str, Any] | None) -> str | None:
    if not isinstance(failure, dict):
        return None
    proposals = failure.get("remediationProposals")
    if not isinstance(proposals, list):
        return None
    for proposal in proposals:
        if not isinstance(proposal, dict):
            continue
        action = str(proposal.get("action") or "").strip()
        if action:
            return action
    return None


def _build_gap_triage(
    gaps: list[dict[str, Any]],
    runtime_reports: list[dict[str, Any]],
    config: AgentImprovementLoopConfig,
) -> dict[str, Any]:
    groups: dict[str, dict[str, Any]] = {}
    root_cause_counts: dict[str, int] = {}
    category_counts: dict[str, int] = {}
    severity_counts: dict[str, int] = {}

    for gap in gaps:
        category = str(gap.get("category") or "unknown")
        severity = str(gap.get("severity") or "unknown")
        evidence = gap.get("evidence") if isinstance(gap.get("evidence"), dict) else {}
        failure_type = str(evidence.get("type") or gap.get("title") or "unknown")
        group = _failure_group(failure_type, evidence)
        root_cause = str(evidence.get("rootCauseCategory") or _gap_root_cause_category(gap))

        category_counts[category] = category_counts.get(category, 0) + 1
        severity_counts[severity] = severity_counts.get(severity, 0) + 1
        root_cause_counts[root_cause] = root_cause_counts.get(root_cause, 0) + 1

        bucket = groups.setdefault(group, {
            "group": group,
            "count": 0,
            "categories": {},
            "severities": {},
            "rootCauseCategories": {},
            "skippedLiveModelGenerationCount": 0,
            "freshRuntimeFailureCount": 0,
            "examples": [],
            "recommendedAction": gap.get("recommendedAction"),
            "status": "deferred",
        })
        bucket["count"] += 1
        bucket["categories"][category] = bucket["categories"].get(category, 0) + 1
        bucket["severities"][severity] = bucket["severities"].get(severity, 0) + 1
        bucket["rootCauseCategories"][root_cause] = bucket["rootCauseCategories"].get(root_cause, 0) + 1
        if root_cause == "skipped_live_model_generation":
            bucket["skippedLiveModelGenerationCount"] += 1
            bucket["status"] = "needs_fresh_runtime_evidence"
        elif root_cause == "stale_audit_evidence":
            bucket["status"] = "needs_fresh_runtime_evidence"
        else:
            bucket["freshRuntimeFailureCount"] += 1
            if root_cause == "permission_config_issue":
                bucket["status"] = "code_or_configuration_fix_required"
            elif root_cause == "manifest_mismatch":
                bucket["status"] = "manifest_reconciliation_required"
        if len(bucket["examples"]) < 5:
            scenario = evidence.get("e2eScenario") if isinstance(evidence.get("e2eScenario"), dict) else {}
            bucket["examples"].append({
                "title": gap.get("title"),
                "category": category,
                "severity": severity,
                "rootCauseCategory": root_cause,
                "prompt": scenario.get("prompt") or evidence.get("scenario"),
                "actual": evidence.get("actual"),
                "skippedLiveModelRun": scenario.get("skippedLiveModelRun"),
            })

    sorted_groups = sorted(groups.values(), key=lambda item: (-int(item["count"]), str(item["group"])))
    runtime_failures = [failure for report in runtime_reports for failure in report.get("failures", []) if isinstance(failure, dict)]
    skipped_runtime = [failure for failure in runtime_failures if _is_skipped_live_model_generation(failure)]
    fresh_runtime = [failure for failure in runtime_failures if not _is_skipped_live_model_generation(failure)]

    return {
        "schemaVersion": "1.0.0",
        "generatedAt": DETERMINISTIC_LOOP_TIMESTAMP if config.deterministic else datetime.now(timezone.utc).isoformat(),
        "summary": {
            "totalGaps": len(gaps),
            "runtimeAuditReports": len(runtime_reports),
            "rawRuntimeFailureCount": len(runtime_failures),
            "freshRuntimeFailureCount": len(fresh_runtime),
            "skippedLiveModelGenerationCount": len(skipped_runtime),
            "categoryCounts": dict(sorted(category_counts.items())),
            "severityCounts": dict(sorted(severity_counts.items())),
            "rootCauseCounts": dict(sorted(root_cause_counts.items())),
            "groupCounts": {str(group["group"]): int(group["count"]) for group in sorted_groups},
            "classificationRule": "skippedLiveModelRun=true remains a gap but is not counted as a fresh runtime failure.",
        },
        "groups": sorted_groups,
    }


def _gap_root_cause_category(gap: dict[str, Any]) -> str:
    category = str(gap.get("category") or "")
    if category == "validation":
        return "manifest_mismatch"
    if category == "command_failure":
        return "permission_config_issue"
    if category in {"testflight_runtime_pending", "testflight_runtime_build_mismatch"}:
        return "stale_audit_evidence"
    return "stale_or_unclassified_runtime_evidence"


def _failure_group(failure_type: str, evidence: dict[str, Any]) -> str:
    if failure_type.startswith("e2e_response_quality_"):
        return failure_type.removeprefix("e2e_response_quality_")
    scenario = evidence.get("e2eScenario") if isinstance(evidence.get("e2eScenario"), dict) else {}
    intent = scenario.get("intent") or scenario.get("expectedIntent")
    if intent:
        return str(intent)
    if "." in failure_type:
        return failure_type.split(".", maxsplit=1)[0]
    return failure_type or "unknown"


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
                {"role": "user", "content": "No blocking gaps were detected. Expand the next loop by adding one new TestFlight scenario family, one runtime trace field exported by the TestFlight + Agent Grounding package, and one dataset quality gate while preserving deterministic output."},
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
        f"- Raw runtime failures: `{state['runtime'].get('rawFailureCount', state['runtime']['failureCount'])}`",
        f"- Skipped live model generation: `{state['runtime'].get('skippedLiveModelGenerationCount', 0)}`",
        f"- Runtime evidence status: `{state['testFlight']['status']}`",
        f"- Assessment proof status: `{state['testFlight'].get('proofStatus', 'not-assessed')}`",
        f"- Proof satisfied at assessment: `{state['testFlight'].get('proofSatisfiedAtAssessment', False)}`",
        f"- Verified-at-assessment reports: `{state['testFlight'].get('verifiedAtAssessmentRuntimeAuditReportCount', 0)}`",
        f"- Assessment proof valid until: `{state['testFlight'].get('validUntil')}`",
        f"- Historical runtime audit reports: `{state['testFlight'].get('historicalRuntimeAuditReportCount', 0)}`",
        f"- Build-rejected runtime audit reports: `{state['testFlight'].get('buildRejectedRuntimeAuditReportCount', 0)}`",
        f"- TestFlight scenarios: `{state['testFlight']['scenarioCount']}`",
        f"- Gaps: `{len(gaps)}`",
        f"- Next action prompts: `{len(prompts)}`",
        "",
        "## TestFlight handoff",
        "",
        "Run `TESTFLIGHT_RUNBOOK.md` in the real TestFlight app, export the TestFlight + Agent Grounding package JSON, then rerun this command with `--runtime-audit <exported-json>`.",
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
    _write_markdown_lines(path, lines)


def _write_gap_triage_markdown(path: Path, triage: dict[str, Any]) -> None:
    summary = triage.get("summary") if isinstance(triage.get("summary"), dict) else {}
    groups = triage.get("groups") if isinstance(triage.get("groups"), list) else []
    lines = [
        "# Agent Improvement Gap Triage",
        "",
        f"- Total gaps: `{summary.get('totalGaps', 0)}`",
        f"- Raw runtime failures: `{summary.get('rawRuntimeFailureCount', 0)}`",
        f"- Fresh runtime failures: `{summary.get('freshRuntimeFailureCount', 0)}`",
        f"- Skipped live model generation: `{summary.get('skippedLiveModelGenerationCount', 0)}`",
        f"- Classification rule: {summary.get('classificationRule', '')}",
        "",
        "## Root Cause Counts",
        "",
    ]
    root_counts = summary.get("rootCauseCounts") if isinstance(summary.get("rootCauseCounts"), dict) else {}
    for name, count in sorted(root_counts.items()):
        lines.append(f"- `{name}`: `{count}`")
    if not root_counts:
        lines.append("- None")

    lines.extend(["", "## Failure Groups", ""])
    for group in groups:
        if not isinstance(group, dict):
            continue
        lines.extend([
            f"### {group.get('group')}",
            "",
            f"- Count: `{group.get('count')}`",
            f"- Status: `{group.get('status')}`",
            f"- Root causes: `{json.dumps(group.get('rootCauseCategories', {}), sort_keys=True)}`",
            f"- Categories: `{json.dumps(group.get('categories', {}), sort_keys=True)}`",
            f"- Fresh runtime failures: `{group.get('freshRuntimeFailureCount', 0)}`",
            f"- Skipped live model generation: `{group.get('skippedLiveModelGenerationCount', 0)}`",
            f"- Recommended action: {group.get('recommendedAction')}",
            "",
        ])
        examples = group.get("examples") if isinstance(group.get("examples"), list) else []
        for example in examples[:3]:
            prompt = str(example.get("prompt") or "").replace("\n", " ").strip()
            actual = str(example.get("actual") or "").replace("\n", " ").strip()
            lines.append(f"  - `{example.get('rootCauseCategory')}` | skipped=`{example.get('skippedLiveModelRun')}` | prompt: {prompt} | actual: {actual[:240]}")
        lines.append("")

    _write_markdown_lines(path, lines)


def _write_testflight_runbook(path: Path, state: dict[str, Any], scenarios: list[dict[str, Any]]) -> None:
    manifest_state = state["manifest"]
    manifest_commit = manifest_state.get("baseCommit", manifest_state.get("commit"))
    lines = [
        "# TestFlight In-App Runtime Runbook",
        "",
        "This is the live-runtime phase of the Lumen improvement loop. Do not replace this with mocked unit tests. The point is to run the current app build through TestFlight, then export what the shipped app observed.",
        "",
        "## Build identity",
        "",
        f"- Manifest fingerprint: `{manifest_state['fingerprint']}`",
        f"- Manifest base commit: `{manifest_commit}`",
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
        "6. Tap `Export TestFlight + Agent Grounding Package`.",
        "7. Share/save the produced `lumen-testflight-agent-grounding-*.json` file.",
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
    _write_markdown_lines(path, lines)


def _write_markdown_lines(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(line.rstrip() for line in lines).rstrip() + "\n", encoding="utf-8")
