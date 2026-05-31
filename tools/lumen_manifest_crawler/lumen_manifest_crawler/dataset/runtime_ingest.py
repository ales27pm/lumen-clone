"""Runtime audit ingestion helpers for JSON and text E2E reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from lumen_manifest_crawler.dataset.e2e_report_normalizer import flatten_e2e_json_report
from lumen_manifest_crawler.dataset.e2e_text_parser import parse_e2e_text_report

SUPPORTED_TEXT_REPORT_SUFFIXES = {".txt", ".md", ".markdown", ".log"}
SUPPORTED_RUNTIME_AUDIT_SUFFIXES = {".json", *SUPPORTED_TEXT_REPORT_SUFFIXES}


def load_runtime_audit_reports(paths: list[Path] | None) -> list[dict[str, Any]]:
    """Load runtime audit records from JSON and supported plain-text report files."""
    reports: list[dict[str, Any]] = []
    for path in paths or []:
        candidates = _candidate_report_files(path)
        for candidate in candidates:
            try:
                text = candidate.read_text(encoding="utf-8")
            except OSError:
                continue
            reports.extend(_load_report_text(text, source=str(candidate)))
    return reports


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


def _load_report_text(text: str, *, source: str) -> list[dict[str, Any]]:
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
            )
        ]
    return _normalize_payload(value, source=source)


def _normalize_payload(value: Any, *, source: str) -> list[dict[str, Any]]:
    if isinstance(value, list):
        out: list[dict[str, Any]] = []
        for index, item in enumerate(value):
            out.extend(_normalize_payload(item, source=f"{source}#{index}"))
        return out
    if not isinstance(value, dict):
        return []
    if _is_evidence_layer_envelope(value):
        return _flatten_evidence_layer_envelope(value, source=source)
    if _is_in_app_package(value):
        return [_flatten_in_app_package(value, source=source)]
    if _is_e2e_json_report(value):
        return [flatten_e2e_json_report(value, source=source)]
    if isinstance(value.get("failures"), list):
        return [{**value, "_source": source, "_sourceFormat": "runtime_manifest_audit"}]
    if isinstance(value.get("violations"), list) or isinstance(value.get("repairSamples"), list):
        return [_flatten_behavior_audit(value, source=source)]
    return []


def _is_evidence_layer_envelope(value: dict[str, Any]) -> bool:
    return isinstance(value.get("exportPolicy"), dict) and "payload" in value


def _flatten_evidence_layer_envelope(envelope: dict[str, Any], *, source: str) -> list[dict[str, Any]]:
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
            "name": result.get("title"),
            "passed": result.get("passed") is True,
            "prompt": result.get("prompt"),
            "intent": result.get("actualIntent") or result.get("expectedIntent"),
            "expectedIntent": result.get("expectedIntent"),
            "failures": "; ".join(str(item) for item in result.get("failures", []) if item) if isinstance(result.get("failures"), list) else result.get("failures"),
            "final": result.get("finalText"),
            "events": result.get("events") or [],
        })
    return {
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
        schema_version in {"1.0.0", "1.1.0"}
        and "exportPolicy" in value
        and any(
            key in value
            for key in (
                "runtimeManifestAudit",
                "behaviorAudit",
                "scenarioResults",
                "recentTraces",
            )
        )
    )


def _is_e2e_json_report(value: dict[str, Any]) -> bool:
    return (
        value.get("kind") in {"lumen_e2e_test_report", "e2e_test_report"}
        or isinstance(value.get("trainingSignals"), list)
        or (
            isinstance(value.get("scenarios"), list)
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
    return {
        "type": "trace_parse_error",
        "agent": trace.get("slot") or trace.get("stage") or "unknown",
        "expected": ["strict manifest-valid structured output"],
        "actual": str(parse_error),
        "scenario": trace.get("promptPrefix"),
        "problem": "A recorded in-app model trace contained a parse error.",
        "sourceLayer": "agentBehaviorTraceRecorder",
    }


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
    return True


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
    empty_trace_failure = _empty_agent_grounding_trace_failure(package, export_policy)
    if empty_trace_failure is not None:
        failures.append(empty_trace_failure)

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
