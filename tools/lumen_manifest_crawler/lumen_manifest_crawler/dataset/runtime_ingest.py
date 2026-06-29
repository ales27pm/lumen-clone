"""Runtime audit ingestion helpers for JSON and text E2E reports."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

from lumen_manifest_crawler.dataset.e2e_report_normalizer import flatten_e2e_json_report
from lumen_manifest_crawler.dataset.e2e_text_parser import parse_e2e_text_report

SUPPORTED_TEXT_REPORT_SUFFIXES = {".txt", ".md", ".markdown", ".log"}
SUPPORTED_RUNTIME_AUDIT_SUFFIXES = {".json", *SUPPORTED_TEXT_REPORT_SUFFIXES}
MAX_REMEDIATION_PROPOSALS = 5
MAX_REMEDIATION_FIELD_CHARS = 320


def load_runtime_audit_reports(paths: list[Path] | None) -> list[dict[str, Any]]:
    """Load runtime audit records from JSON and supported plain-text report files."""
    reports: list[dict[str, Any]] = []
    sidecar_cache: dict[Path, dict[str, list[dict[str, Any]]]] = {}
    for path in paths or []:
        candidates = _candidate_report_files(path)
        for candidate in candidates:
            try:
                text = candidate.read_text(encoding="utf-8")
            except OSError:
                continue
            parent = candidate.parent
            if parent not in sidecar_cache:
                sidecar_cache[parent] = _load_e2e_sidecars(parent)
            reports.extend(_load_report_text(text, source=str(candidate), sidecars=sidecar_cache[parent]))
    return _dedupe_runtime_reports(reports)


def _dedupe_runtime_reports(reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen_report_ids: set[str] = set()
    for report in reports:
        report_id = _stable_report_id(report)
        if report_id:
            if report_id in seen_report_ids:
                continue
            seen_report_ids.add(report_id)
        deduped.append(report)
    return deduped


def _stable_report_id(report: dict[str, Any]) -> str | None:
    report_id = report.get("id") or report.get("reportID")
    if report_id:
        return str(report_id)
    return None


def _candidate_report_files(path: Path) -> list[Path]:
    if path.is_dir():
        return sorted(
            candidate
            for candidate in path.rglob("*")
            if _is_supported_report_file(candidate)
        )
    return [path] if _is_supported_report_file(path) else []


def _is_supported_report_file(path: Path) -> bool:
    return path.is_file() and path.suffix.casefold() in SUPPORTED_RUNTIME_AUDIT_SUFFIXES


def _load_report_text(text: str, *, source: str, sidecars: dict[str, list[dict[str, Any]]] | None = None) -> list[dict[str, Any]]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        parsed = parse_e2e_text_report(text, source=source)
        if parsed is None:
            return []
        return [
            flatten_e2e_json_report(
                parsed,
                source=source,
                source_format="lumen_e2e_text_report",
                source_layer="e2eTextReport",
                sidecars=sidecars,
            )
        ]
    return _normalize_payload(value, source=source, sidecars=sidecars)


def _normalize_payload(value: Any, *, source: str, sidecars: dict[str, list[dict[str, Any]]] | None = None) -> list[dict[str, Any]]:
    if isinstance(value, list):
        out: list[dict[str, Any]] = []
        for index, item in enumerate(value):
            out.extend(_normalize_payload(item, source=f"{source}#{index}", sidecars=sidecars))
        return out
    if not isinstance(value, dict):
        return []
    if _is_evidence_layer_envelope(value):
        return _flatten_evidence_layer_envelope(value, source=source, sidecars=sidecars)
    if _is_in_app_package(value):
        return _flatten_in_app_package_reports(value, source=source, sidecars=sidecars)
    if _is_self_model_eval_score_report(value):
        return [_flatten_self_model_eval_score_report(value, source=source)]
    if _is_persistent_runtime_diagnostics_export(value):
        return [_flatten_persistent_runtime_diagnostics(value, source=source)]
    if _is_e2e_json_report(value):
        return [flatten_e2e_json_report(value, source=source, sidecars=sidecars)]
    if isinstance(value.get("failures"), list):
        return [{**value, "_source": source, "_sourceFormat": "runtime_manifest_audit"}]
    if isinstance(value.get("violations"), list) or isinstance(value.get("repairSamples"), list):
        return [_flatten_behavior_audit(value, source=source)]
    return []


def _load_e2e_sidecars(directory: Path) -> dict[str, list[dict[str, Any]]]:
    return {
        "agent_behavior_traces": _load_first_matching_jsonl(directory, "agent-behavior-traces.jsonl"),
        "agent_parse_failures": _load_first_matching_jsonl(directory, "agent-parse-failures.jsonl"),
        "e2e_results": _load_first_matching_jsonl(directory, "e2e-results.jsonl"),
        "_sidecar_presence": [
            {
                "agent_behavior_traces": _matching_jsonl_exists(directory, "agent-behavior-traces.jsonl"),
                "agent_parse_failures": _matching_jsonl_exists(directory, "agent-parse-failures.jsonl"),
                "e2e_results": _matching_jsonl_exists(directory, "e2e-results.jsonl"),
            }
        ],
    }


def _matching_jsonl_exists(directory: Path, suffix: str) -> bool:
    candidates = [directory / suffix]
    try:
        candidates.extend(sorted(path for path in directory.glob(f"*{suffix}") if path.name != suffix))
    except OSError:
        pass
    return any(candidate.exists() for candidate in candidates)


def _load_first_matching_jsonl(directory: Path, suffix: str) -> list[dict[str, Any]]:
    candidates = [directory / suffix]
    try:
        candidates.extend(sorted(path for path in directory.glob(f"*{suffix}") if path.name != suffix))
    except OSError:
        pass
    for candidate in candidates:
        records = _load_jsonl_records(candidate)
        if records:
            return records
    return []


def _load_jsonl_records(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    records: list[dict[str, Any]] = []
    for line in lines:
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            records.append(value)
    return records


def _is_evidence_layer_envelope(value: dict[str, Any]) -> bool:
    return isinstance(value.get("exportPolicy"), dict) and "payload" in value


def _flatten_evidence_layer_envelope(envelope: dict[str, Any], *, source: str, sidecars: dict[str, list[dict[str, Any]]] | None = None) -> list[dict[str, Any]]:
    export_policy = envelope.get("exportPolicy")
    export_policy = export_policy if isinstance(export_policy, dict) else {}
    payload = envelope.get("payload")
    source_layer = str(export_policy.get("sourceLayer") or "unknown")
    source_format = str(export_policy.get("format") or "evidence-layer-json")
    owns_live_e2e = export_policy.get("ownsLiveE2EScenarios") is True

    if source_layer == "e2eTestReport" or owns_live_e2e:
        if isinstance(payload, dict):
            report = _swift_e2e_payload_to_normalized_report(payload)
            return [
                flatten_e2e_json_report(
                    report,
                    source=source,
                    source_format=source_format,
                    source_layer="e2eTestReport.evidenceLayer",
                    sidecars=sidecars,
                )
            ]
        return []

    if source_layer == "runtimeManifestAudit" and isinstance(payload, dict):
        return [{
            **payload,
            "_source": source,
            "_sourceFormat": source_format,
            "_sourceLayer": "runtimeManifestAudit",
            "exportPolicy": export_policy,
        }]

    if source_layer == "agentModelBehaviorAuditor" and isinstance(payload, dict):
        flattened = _flatten_behavior_audit(payload, source=source)
        flattened["_sourceFormat"] = source_format
        flattened["_sourceLayer"] = "agentModelBehaviorAuditor"
        flattened["exportPolicy"] = export_policy
        return [flattened]

    if source_layer == "agentBehaviorTraceRecorder" and isinstance(payload, list):
        trace_failures, selected_tool_allowed_count, parse_error_count = _collect_trace_failures(payload)
        if not payload:
            trace_failures.append({
                "type": "agent_grounding_no_recent_model_traces",
                "agent": "runtime",
                "expected": ["Recent runtime trace layer should include at least one trace after exercising the app."],
                "actual": "payload is empty",
                "scenario": "Agent Grounding > Export Recent Runtime Traces",
                "problem": "The runtime trace layer export is empty. Run real model/tool interactions or wire AgentBehaviorTraceRecorder.record into the live path.",
                "sourceLayer": "agentBehaviorTraceRecorder.exportQuality",
            })
        return [{
            "_source": source,
            "_sourceFormat": source_format,
            "_sourceLayer": "agentBehaviorTraceRecorder",
            "generatedAt": envelope.get("generatedAt"),
            "traceSelectedToolAllowedCount": selected_tool_allowed_count,
            "traceParseErrorCount": parse_error_count,
            "traceCount": len(payload),
            "exportPolicy": export_policy,
            "failures": trace_failures,
        }]

    if source_layer == "runtimeScenarioRunner.staticChecks" and isinstance(payload, list):
        return [{
            "_source": source,
            "_sourceFormat": source_format,
            "_sourceLayer": "runtimeScenarioRunner.staticChecks",
            "generatedAt": envelope.get("generatedAt"),
            "ownsLiveE2EScenarios": False,
            "ignoredScenarioResultCount": len(payload),
            "exportPolicy": export_policy,
            "failures": [],
        }]

    return [{
        "_source": source,
        "_sourceFormat": source_format,
        "_sourceLayer": source_layer,
        "generatedAt": envelope.get("generatedAt"),
        "exportPolicy": export_policy,
        "failures": [],
    }]


def _swift_e2e_payload_to_normalized_report(payload: dict[str, Any]) -> dict[str, Any]:
    results = payload.get("results") if isinstance(payload.get("results"), list) else []
    scenarios: list[dict[str, Any]] = []
    for result in _iter_dicts(results):
        scenarios.append({
            "id": result.get("scenarioID") or result.get("id"),
            "scenarioID": result.get("scenarioID"),
            "e2eRunID": result.get("e2eRunID"),
            "agentRunID": result.get("agentRunID"),
            "conversationID": result.get("conversationID"),
            "turnID": result.get("turnID"),
            "name": result.get("title"),
            "kind": result.get("kind"),
            "passed": result.get("passed") is True,
            "prompt": result.get("prompt"),
            "intent": result.get("actualIntent") or result.get("expectedIntent"),
            "expectedIntent": result.get("expectedIntent"),
            "requiresAgentRun": result.get("requiresAgentRun") is True,
            "failures": "; ".join(str(item) for item in result.get("failures", []) if item) if isinstance(result.get("failures"), list) else result.get("failures"),
            "final": result.get("finalText"),
            "events": result.get("events") or [],
            "startedAt": result.get("startedAt"),
            "finishedAt": result.get("finishedAt"),
        })
    return {
        "id": payload.get("id"),
        "kind": "lumen_e2e_test_report",
        "passed": payload.get("passed"),
        "failed": payload.get("failed"),
        "scenarioCount": len(scenarios),
        "scenarios": scenarios,
        "trainingSignals": _derive_e2e_training_signals(scenarios),
    }


def _derive_e2e_training_signals(scenarios: list[dict[str, Any]]) -> list[str]:
    failed = [scenario for scenario in scenarios if scenario.get("passed") is not True]
    if not failed:
        return []
    return [
        f"failed-scenarios: {len(failed)}",
        "Capture failed prompts + final outputs into next fine-tuning dataset.",
        "Prioritize repeated tool-boundary, response-quality, and no-model execution failures.",
    ]


def _is_in_app_package(value: dict[str, Any]) -> bool:
    schema_version = str(value.get("schemaVersion") or "")
    return (
        schema_version in {"1.0.0", "1.1.0", "1.2.0", "1.3.0", "1.4.0", "1.5.0", "1.6.0", "1.7.0"}
        and "exportPolicy" in value
        and any(
            key in value
            for key in (
                "runtimeManifestAudit",
                "behaviorAudit",
                "scenarioResults",
                "recentTraces",
                "liveE2EReport",
            )
        )
    )


def _is_persistent_runtime_diagnostics_export(value: dict[str, Any]) -> bool:
    state = value.get("state")
    return (
        isinstance(state, dict)
        and isinstance(state.get("records"), list)
        and isinstance(value.get("ndjson"), str)
        and "exportedAt" in value
    )


def _is_self_model_eval_score_report(value: dict[str, Any]) -> bool:
    return (
        str(value.get("schemaVersion") or "") == "self_model_eval_score.v1"
        and isinstance(value.get("results"), list)
    )


def _flatten_self_model_eval_score_report(report: dict[str, Any], *, source: str) -> dict[str, Any]:
    results = list(_iter_dicts(report.get("results", []) or []))
    failures: list[dict[str, Any]] = []
    for result in results:
        if result.get("passed") is True:
            continue
        raw_failures = [str(item) for item in result.get("failures", []) or [] if item]
        failure_type = _self_model_score_failure_type(raw_failures)
        failures.append({
            "type": failure_type,
            "agent": "fleet",
            "expected": result.get("checked", []) if isinstance(result.get("checked"), list) else [],
            "actual": raw_failures,
            "scenario": result.get("name") or result.get("id"),
            "problem": _self_model_score_problem(failure_type),
            "sourceLayer": "selfModelEvalScore",
            "selfModelEvalID": result.get("id"),
            "selfModelEvalScore": result.get("score"),
            "selfModelEvalFailures": raw_failures,
        })
    return {
        "_source": source,
        "_sourceFormat": "self_model_eval_score_report",
        "_sourceLayer": "selfModelEvalScore",
        "schemaVersion": report.get("schemaVersion"),
        "scenarioCount": report.get("scenarioCount", len(results)),
        "answeredCount": report.get("answeredCount"),
        "passedCount": report.get("passedCount"),
        "failedCount": report.get("failedCount"),
        "missingCount": report.get("missingCount"),
        "allPassed": report.get("allPassed") is True,
        "ownsLiveE2EScenarios": False,
        "failures": failures,
    }


def _self_model_score_failure_type(failures: list[str]) -> str:
    joined = " ".join(failures)
    if "missing_answer" in joined:
        return "self_model_eval_answer_missing"
    if "subjective_awareness_claim" in joined:
        return "self_model_subjective_awareness_claim"
    if "raw_private_training_not_rejected" in joined:
        return "self_model_private_payload_leak"
    if "safe_schema_degradation_missing" in joined:
        return "self_model_snapshot_schema_unsupported"
    if "repair_sample_missing" in joined:
        return "self_model_repair_sample_missing"
    runtime_markers = (
        "unknown_without_evidence_missing",
        "live_e2e_evidence_requirement_missing",
        "static_not_live_proof_missing",
        "bundled_live_separation_missing",
        "source_layer_missing",
        "snapshot_runtime_fields_missing",
        "snapshot_resource_fields_missing",
        "snapshot_app_fields_missing",
        "active_slot_snapshot_missing",
    )
    if any(marker in joined for marker in runtime_markers):
        return "self_model_runtime_state_claim_without_evidence"
    background_markers = (
        "foreground_approval_requirement_missing",
        "snapshot_tool_scope_missing",
    )
    if any(marker in joined for marker in background_markers):
        return "self_model_background_filtering_regression"
    return "self_model_tool_boundary_regression"


def _self_model_score_problem(failure_type: str) -> str:
    problems = {
        "self_model_eval_answer_missing": "A self-model eval scenario was not answered by the model export.",
        "self_model_subjective_awareness_claim": "A self-model answer claimed subjective awareness or sentience.",
        "self_model_private_payload_leak": "A self-model answer allowed raw private payloads into training or artifacts.",
        "self_model_snapshot_schema_unsupported": "A self-model answer did not safely degrade for an unsupported snapshot schema.",
        "self_model_repair_sample_missing": "A failed self-model claim did not produce repair-sample guidance.",
        "self_model_runtime_state_claim_without_evidence": "A self-model answer did not respect runtime evidence freshness or source-layer boundaries.",
        "self_model_background_filtering_regression": "A self-model answer did not respect snapshot-filtered background/tool availability.",
        "self_model_tool_boundary_regression": "A self-model answer violated tool boundary, approval, or permission expectations.",
    }
    return problems.get(failure_type, "A self-model eval answer failed deterministic scoring.")


def _is_e2e_json_report(value: dict[str, Any]) -> bool:
    return (
        value.get("kind") in {"lumen_e2e_test_report", "e2e_test_report"}
        or isinstance(value.get("trainingSignals"), list)
        or (
            isinstance(value.get("scenarios"), list)
            and {"passed", "failed"}.intersection(value.keys())
        )
        or (
            isinstance(value.get("results"), list)
            and {"passed", "failed"}.intersection(value.keys())
        )
    )


def _iter_dicts(items: Iterable[Any]) -> Iterable[dict[str, Any]]:
    for item in items:
        if isinstance(item, dict):
            yield item


def _layered_failures(
    failures: Iterable[Any], *, source_layer: str
) -> list[dict[str, Any]]:
    return [{**failure, "sourceLayer": source_layer} for failure in _iter_dicts(failures)]


def _trace_parse_error_failure(
    trace: dict[str, Any], parse_error: Any
) -> dict[str, Any]:
    raw_output = str(trace.get("rawOutputPrefix") or "")
    parse_error_text = str(parse_error)
    context_overflow = _is_agent_json_context_overflow(raw_output, parse_error_text)
    empty_stream_category = (
        _agent_json_empty_stream_category(trace)
        if not raw_output.strip() and parse_error_text.casefold() == "empty"
        else None
    )
    return {
        "type": "prompt_budget_overflow" if context_overflow else (empty_stream_category or "trace_parse_error"),
        "agent": trace.get("slot") or trace.get("stage") or "unknown",
        "expected": [
            "agent-json prompt fits executor/shared chat context window"
            if context_overflow
            else "agent-json stream emits at least one usable text chunk"
            if empty_stream_category
            else "strict manifest-valid structured output"
        ],
        "actual": str(parse_error),
        "scenario": trace.get("promptPrefix"),
        "problem": (
            "A recorded in-app agent-json trace exceeded the executor/shared chat context window before generation."
            if context_overflow
            else f"A recorded in-app agent-json stream completed without usable text ({trace.get('emptyOutputReason') or trace.get('streamTerminationReason') or 'unknown'})."
            if empty_stream_category
            else "A recorded in-app model trace contained a parse error."
        ),
        "rootCauseCategory": "agent_json_context_overflow" if context_overflow else empty_stream_category,
        "sourceLayer": "agentBehaviorTraceRecorder",
    }


def _is_agent_json_context_overflow(raw_output: str, parse_error: str) -> bool:
    parse_text = parse_error.casefold()
    raw_text = raw_output.casefold()
    return (
        parse_text in {"contextwindowexceeded", "prompttoolarge"}
        or "prompt exceeded context window before generation" in raw_text
        or "prompt exceeds shared chat context window" in raw_text
        or "failed to initialize context: prompt exceeds" in raw_text
    )


def _agent_json_empty_stream_category(trace: dict[str, Any]) -> str:
    termination_reason = str(trace.get("streamTerminationReason") or "")
    termination_compact = re.sub(r"[^a-z0-9]", "", termination_reason.casefold())
    if "resourcebudgetdenied" in termination_compact:
        return "agent_json_resource_budget_denied_before_first_token"

    reason = str(trace.get("emptyOutputReason") or trace.get("streamTerminationReason") or "").casefold()
    if reason in {"completedwithouttext", "agent-json-stream-completed-without-text", "stream-completed-without-text-chunks"}:
        return "agent_json_completed_without_text"
    if reason in {"stoppedbeforefirsttoken", "stopsequencebeforetext", "eosbeforetext"}:
        return "agent_json_stop_before_first_token"
    if reason == "cancelledbeforefirsttoken":
        return "agent_json_cancelled_before_first_token"
    if reason == "decodebudgetzero":
        return "agent_json_decode_budget_zero"
    if reason == "modelnotloaded":
        return "agent_json_model_not_loaded"
    if reason == "slotunavailable":
        return "agent_json_slot_unavailable"
    if reason == "runtimeunavailable":
        return "agent_json_runtime_unavailable"
    return "agent_json_empty_stream"


def _should_report_trace_parse_error(trace: dict[str, Any]) -> bool:
    prompt = str(trace.get("promptPrefix") or "").lower()
    raw_output = str(trace.get("rawOutputPrefix") or "")
    slot = str(trace.get("slot") or "").lower()
    stage = str(trace.get("stage") or "").lower()
    # Some traces intentionally request plain text (for example mouth-final
    # user-facing replies or replay summaries). Those should not be counted as
    # structured-output runtime drift failures.
    if "do not output json" in prompt:
        return False
    if stage == "agent-summary" and "original final answer" in prompt:
        return False
    if slot == "mouth" and stage == "mouth-final":
        return False
    # Some in-app exports captured a prompt echo in rawOutputPrefix for a
    # cortex orchestrator turn; this is not a model JSON-parse regression.
    if (
        stage == "cortex-orchestrator-json"
        and raw_output.startswith("You are Lumen, a helpful, concise on-device AI assistant.")
    ):
        return False
    return _trace_has_tool_scope(trace)


def _trace_has_tool_scope(trace: dict[str, Any]) -> bool:
    selected_tool_id = str(trace.get("selectedToolID") or "")
    allowed_tool_ids = trace.get("allowedToolIDs")
    allowed_tool_ids = allowed_tool_ids if isinstance(allowed_tool_ids, list) else []
    requires_approval = trace.get("requiresApproval")
    approval_mode = str(trace.get("approvalMode") or "")
    return bool(selected_tool_id or allowed_tool_ids or requires_approval is True or approval_mode)


def _trace_tool_failure(
    trace: dict[str, Any], selected_tool_id: str, allowed_tool_ids: list[str]
) -> dict[str, Any] | None:
    if not allowed_tool_ids:
        return {
            "type": "trace_tool_without_allowed_set",
            "agent": trace.get("slot") or "cortex",
            "expected": ["non-empty allowedToolIDs for tool-selection traces"],
            "actual": selected_tool_id,
            "scenario": trace.get("promptPrefix"),
            "problem": (
                "A recorded in-app trace selected a tool while the trace carried "
                "no allowed tool set for validation."
            ),
            "sourceLayer": "agentBehaviorTraceRecorder",
        }
    if selected_tool_id not in allowed_tool_ids:
        return {
            "type": "trace_tool_outside_allowed_set",
            "agent": trace.get("slot") or "cortex",
            "expected": allowed_tool_ids,
            "actual": selected_tool_id,
            "scenario": trace.get("promptPrefix"),
            "problem": (
                "A recorded in-app trace selected a tool outside its allowed tool set."
            ),
            "sourceLayer": "agentBehaviorTraceRecorder",
        }
    return None


def _empty_agent_grounding_trace_failure(package: dict[str, Any], export_policy: dict[str, Any]) -> dict[str, Any] | None:
    source_layer = str(export_policy.get("sourceLayer") or "")
    package_format = str(export_policy.get("format") or "")
    if source_layer != "agentGroundingRuntimeAudit" and package_format != "agent-grounding-runtime-json-package":
        return None
    recent_traces = package.get("recentTraces")
    if isinstance(recent_traces, list) and recent_traces:
        return None
    return {
        "type": "agent_grounding_no_recent_model_traces",
        "agent": "runtime",
        "expected": ["Agent Grounding export should include recent model/tool traces captured from real in-app execution."],
        "actual": "recentTraces is empty",
        "scenario": "Agent Grounding > Run Agent Grounding Audit > Export In-App Dataset Package",
        "problem": (
            "The Agent Grounding package exported no recent traces. This usually means "
            "AgentBehaviorTraceRecorder.record is not wired into the live model path, "
            "or the app audit was exported before exercising real model interactions."
        ),
        "sourceLayer": "agentGroundingRuntimeAudit.exportQuality",
    }


def _live_e2e_model_backed_trace_gap_failure(package: dict[str, Any]) -> dict[str, Any] | None:
    live_e2e_report = package.get("liveE2EReport")
    if not isinstance(live_e2e_report, dict):
        return None
    model_backed_count = _optional_int(live_e2e_report.get("modelBackedCorrelatedTraceCount"))
    if model_backed_count is None:
        return None
    correlated_count = _optional_int(live_e2e_report.get("correlatedTraceCount")) or 0
    deterministic_count = _optional_int(live_e2e_report.get("deterministicCompatibilityTraceCount")) or 0
    payload = live_e2e_report.get("payload")
    payload = payload if isinstance(payload, dict) else {}
    required_agent_run_scenario_count = sum(
        1
        for result in _iter_dicts(payload.get("results", []) or [])
        if result.get("requiresAgentRun") is True
    )
    if required_agent_run_scenario_count <= 0 or model_backed_count >= required_agent_run_scenario_count:
        return None
    return {
        "type": "agent_grounding_live_e2e_model_backed_trace_gap",
        "agent": "runtime",
        "expected": [
            "Every requiresAgentRun live E2E scenario should export correlated model-backed AgentBehaviorTrace modelTurn evidence."
        ],
        "actual": (
            f"requiredAgentRunScenarioCount={required_agent_run_scenario_count}; "
            f"modelBackedCorrelatedTraceCount={model_backed_count}; "
            f"correlatedTraceCount={correlated_count}; "
            f"deterministicCompatibilityTraceCount={deterministic_count}"
        ),
        "scenario": "Agent Grounding > E2E Test Runner > Export In-App Dataset Package",
        "problem": (
            "The embedded live E2E report does not have enough model-backed correlated traces. "
            "Deterministic compatibility traces and uncorrelated traces are diagnostics only, not live model evidence."
        ),
        "sourceLayer": "agentGroundingRuntimeAudit.exportQuality",
    }


def _optional_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _flatten_persistent_runtime_diagnostics(package: dict[str, Any], *, source: str) -> dict[str, Any]:
    state = package.get("state")
    state = state if isinstance(state, dict) else {}
    records = list(_iter_dicts(state.get("records", []) or []))
    status_counts: dict[str, int] = {}
    remediation_severity_counts: dict[str, int] = {}
    remediation_proposal_count = 0
    failures: list[dict[str, Any]] = []
    for record in records:
        status = str(record.get("status") or "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1
        proposals = _bounded_remediation_proposals(record)
        remediation_proposal_count += len(proposals)
        for proposal in proposals:
            severity = str(proposal.get("severity") or "unknown")
            remediation_severity_counts[severity] = remediation_severity_counts.get(severity, 0) + 1
        if status == "passed" or _is_expected_diagnostics_cancellation(record):
            continue
        failure = {
            "type": "persistent_diagnostics_scenario_not_passed",
            "agent": "runtime",
            "expected": ["Persistent diagnostics scenario should pass or be an explicitly expected cancellation."],
            "actual": status,
            "scenario": record.get("scenario"),
            "problem": "A persistent runtime diagnostics scenario finished without a passing status.",
            "sourceLayer": "persistentRuntimeDiagnostics.records",
            "diagnosticRecordID": record.get("id"),
        }
        if proposals:
            failure["remediationProposals"] = proposals
            failure["remediationSeverity"] = _top_remediation_severity(proposals)
        failures.append(failure)
    ndjson = package.get("ndjson")
    ndjson_line_count = len([line for line in str(ndjson).splitlines() if line.strip()])
    return {
        "_source": source,
        "_sourceFormat": "persistent_runtime_diagnostics_export",
        "_sourceLayer": "persistentRuntimeDiagnostics",
        "generatedAt": package.get("exportedAt"),
        "appVersion": package.get("appVersion"),
        "deviceModel": package.get("deviceModel"),
        "systemName": package.get("systemName"),
        "systemVersion": package.get("systemVersion"),
        "campaign": package.get("campaign") if isinstance(package.get("campaign"), dict) else None,
        "recordCount": len(records),
        "statusCounts": dict(sorted(status_counts.items())),
        "remediationProposalCount": remediation_proposal_count,
        "remediationSeverityCounts": dict(sorted(remediation_severity_counts.items())),
        "metricKitPayloadCount": len(package.get("metricKitPayloads", []) or []) if isinstance(package.get("metricKitPayloads"), list) else 0,
        "ndjsonLineCount": ndjson_line_count,
        "failures": failures,
    }


def _is_expected_diagnostics_cancellation(record: dict[str, Any]) -> bool:
    if str(record.get("status") or "") != "cancelled":
        return False
    metrics = record.get("metrics")
    metrics = metrics if isinstance(metrics, dict) else {}
    if metrics.get("didCancel") is not True:
        return False
    reason = str(metrics.get("cancellationReason") or "")
    return reason == "persistent-diagnostics-agent-cancel"


def _bounded_remediation_proposals(record: dict[str, Any]) -> list[dict[str, str]]:
    raw = record.get("remediationProposals")
    proposals = raw if isinstance(raw, list) else []
    out: list[dict[str, str]] = []
    for item in proposals[:MAX_REMEDIATION_PROPOSALS]:
        if not isinstance(item, dict):
            continue
        proposal = {
            "id": _bounded_remediation_text(item.get("id"), limit=96),
            "title": _bounded_remediation_text(item.get("title")),
            "rationale": _bounded_remediation_text(item.get("rationale")),
            "action": _bounded_remediation_text(item.get("action")),
            "severity": _normalized_remediation_severity(item.get("severity")),
        }
        if proposal["id"] or proposal["title"] or proposal["action"]:
            out.append(proposal)
    return out


def _bounded_remediation_text(value: Any, *, limit: int = MAX_REMEDIATION_FIELD_CHARS) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text[:limit]


def _normalized_remediation_severity(value: Any) -> str:
    severity = str(value or "").strip().lower()
    return severity if severity in {"info", "warning", "critical"} else "unknown"


def _top_remediation_severity(proposals: list[dict[str, str]]) -> str:
    rank = {"critical": 3, "warning": 2, "info": 1, "unknown": 0}
    return max((proposal.get("severity") or "unknown" for proposal in proposals), key=lambda item: rank.get(item, 0), default="unknown")


def _collect_trace_failures(
    traces: Iterable[Any],
) -> tuple[list[dict[str, Any]], int, int]:
    failures: list[dict[str, Any]] = []
    selected_tool_allowed_count = 0
    parse_error_count = 0
    for trace in _iter_dicts(traces):
        parse_error = trace.get("parseError")
        selected_tool_id = trace.get("selectedToolID")
        allowed_tool_ids = trace.get("allowedToolIDs")
        allowed_tool_ids = allowed_tool_ids if isinstance(allowed_tool_ids, list) else []

        if parse_error:
            parse_error_count += 1
            if _should_report_trace_parse_error(trace):
                failures.append(_trace_parse_error_failure(trace, parse_error))
        if not selected_tool_id:
            continue
        if selected_tool_id in allowed_tool_ids:
            selected_tool_allowed_count += 1
        tool_failure = _trace_tool_failure(trace, str(selected_tool_id), allowed_tool_ids)
        if tool_failure is not None:
            failures.append(tool_failure)
    return failures, selected_tool_allowed_count, parse_error_count


def _flatten_in_app_package_reports(
    package: dict[str, Any],
    *,
    source: str,
    sidecars: dict[str, list[dict[str, Any]]] | None = None,
) -> list[dict[str, Any]]:
    reports = [_flatten_in_app_package(package, source=source)]
    live_e2e_report = package.get("liveE2EReport")
    if isinstance(live_e2e_report, dict):
        embedded_sidecars = _sidecars_with_in_app_traces(package, sidecars=sidecars)
        reports.extend(
            _flatten_evidence_layer_envelope(
                live_e2e_report,
                source=f"{source}#liveE2EReport",
                sidecars=embedded_sidecars,
            )
        )
    return reports


def _sidecars_with_in_app_traces(
    package: dict[str, Any],
    *,
    sidecars: dict[str, list[dict[str, Any]]] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    merged: dict[str, list[dict[str, Any]]] = {
        key: list(value)
        for key, value in (sidecars or {}).items()
        if isinstance(value, list)
    }
    embedded_traces = list(_iter_dicts(package.get("recentTraces", []) or []))
    if embedded_traces:
        merged["agent_behavior_traces"] = embedded_traces + merged.get("agent_behavior_traces", [])

    presence: dict[str, Any] = {}
    existing_presence = merged.get("_sidecar_presence") or []
    if existing_presence and isinstance(existing_presence[0], dict):
        presence.update(existing_presence[0])
    if embedded_traces:
        presence["agent_behavior_traces"] = True
    if "agent_behavior_traces" in merged:
        presence.setdefault("agent_behavior_traces", bool(merged["agent_behavior_traces"]))
    if presence:
        merged["_sidecar_presence"] = [presence]
    return merged


def _flatten_in_app_package(package: dict[str, Any], *, source: str) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []

    runtime_audit = package.get("runtimeManifestAudit")
    if isinstance(runtime_audit, dict):
        failures.extend(
            _layered_failures(
                runtime_audit.get("failures", []) or [],
                source_layer="runtimeManifestAudit",
            )
        )

    behavior_audit = package.get("behaviorAudit")
    if isinstance(behavior_audit, dict):
        failures.extend(_behavior_failures(behavior_audit))

    export_policy = package.get("exportPolicy")
    export_policy = export_policy if isinstance(export_policy, dict) else {}
    owns_live_e2e = export_policy.get("ownsLiveE2EScenarios") is True
    scenario_results = list(_iter_dicts(package.get("scenarioResults", []) or []))
    if owns_live_e2e:
        for scenario_result in scenario_results:
            failures.extend(
                _layered_failures(
                    scenario_result.get("failures", []) or [],
                    source_layer="e2eTestReport.scenarioResults",
                )
            )

    trace_failures, selected_tool_allowed_count, parse_error_count = _collect_trace_failures(
        package.get("recentTraces", []) or []
    )
    failures.extend(trace_failures)
    export_quality_failures = _layered_failures(
        package.get("exportQualityFailures", []) or [],
        source_layer="agentGroundingRuntimeAudit.exportQuality",
    )
    failures.extend(export_quality_failures)
    empty_trace_failure = _empty_agent_grounding_trace_failure(package, export_policy)
    has_empty_trace_failure = any(
        failure.get("type") == "agent_grounding_no_recent_model_traces"
        for failure in export_quality_failures
    )
    if empty_trace_failure is not None and not has_empty_trace_failure:
        failures.append(empty_trace_failure)

    live_e2e_report = package.get("liveE2EReport")
    live_e2e_summary = live_e2e_report if isinstance(live_e2e_report, dict) else {}
    live_trace_gap_failure = _live_e2e_model_backed_trace_gap_failure(package)
    has_live_trace_gap_failure = any(
        failure.get("type") == "agent_grounding_live_e2e_model_backed_trace_gap"
        for failure in export_quality_failures
    )
    if live_trace_gap_failure is not None and not has_live_trace_gap_failure:
        failures.append(live_trace_gap_failure)

    return {
        "_source": source,
        "_sourceFormat": "lumen_in_app_dataset_package",
        "_sourceLayer": export_policy.get("sourceLayer") or "agentGroundingRuntimeAudit",
        "generatedAt": package.get("generatedAt"),
        "manifestSource": package.get("manifestSource"),
        "usedRuntimeFallback": package.get("usedRuntimeFallback"),
        "traceSelectedToolAllowedCount": package.get(
            "traceSelectedToolAllowedCount",
            selected_tool_allowed_count,
        ),
        "traceParseErrorCount": package.get("traceParseErrorCount", parse_error_count),
        "liveE2ECorrelatedTraceCount": live_e2e_summary.get("correlatedTraceCount"),
        "liveE2EModelBackedCorrelatedTraceCount": live_e2e_summary.get("modelBackedCorrelatedTraceCount"),
        "liveE2EDeterministicCompatibilityTraceCount": live_e2e_summary.get("deterministicCompatibilityTraceCount"),
        "ignoredScenarioResultCount": 0 if owns_live_e2e else len(scenario_results),
        "ownsLiveE2EScenarios": owns_live_e2e,
        "exportPolicy": export_policy,
        "failures": failures,
    }


def _flatten_behavior_audit(value: dict[str, Any], *, source: str) -> dict[str, Any]:
    return {
        "_source": source,
        "_sourceFormat": "agent_behavior_audit",
        "generatedAt": value.get("generatedAt"),
        "failures": _behavior_failures(value),
    }


def _behavior_failures(behavior_audit: dict[str, Any]) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    repair_samples = behavior_audit.get("repairSamples")
    if isinstance(repair_samples, list):
        for sample in _iter_dicts(repair_samples):
            if not _is_actionable_behavior_repair_sample(sample):
                continue
            failures.append(
                {
                    "type": sample.get("violationCode") or "behavior_repair_sample",
                    "agent": sample.get("agent"),
                    "expected": [
                        str(sample.get("correctedOutput") or sample.get("expected") or "")
                    ],
                    "actual": sample.get("badOutput"),
                    "scenario": sample.get("promptPrefix"),
                    "problem": (
                        sample.get("lesson")
                        or "In-app model behavior audit generated a repair sample."
                    ),
                    "repairSample": sample,
                    "sourceLayer": "agentModelBehaviorAuditor.repairSamples",
                }
            )
        return failures

    for violation in _iter_dicts(behavior_audit.get("violations", []) or []):
        failures.append(
            {
                "type": violation.get("code") or "behavior_violation",
                "agent": violation.get("agent"),
                "expected": [str(violation.get("expected") or "")],
                "actual": violation.get("actual"),
                "scenario": violation.get("promptPrefix"),
                "problem": (
                    violation.get("problem")
                    or "In-app model behavior violated manifest constraints."
                ),
                "sourceLayer": "agentModelBehaviorAuditor.violations",
            }
        )
    return failures


def _is_actionable_behavior_repair_sample(sample: dict[str, Any]) -> bool:
    corrected = str(sample.get("correctedOutput") or "").strip()
    if not corrected:
        return False
    lowered = corrected.lower()
    meta_instruction_prefixes = (
        "emit a tool call",
        "select a manifest",
        "ask for clarification",
        "return only",
        "regenerate ",
    )
    if lowered.startswith(meta_instruction_prefixes):
        return False
    if corrected.startswith("{") or corrected.startswith("["):
        return True
    if "(" in corrected and ")" in corrected:
        return True
    if corrected.endswith("?"):
        return True
    return False
