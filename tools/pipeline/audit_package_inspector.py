#!/usr/bin/env python3
"""Inspect Lumen in-app audit packages before using them for training.

The crawler already flattens runtime audits into failure records. This module is
more forensic: it preserves counts for the actual in-app package shape so the
pipeline can tell the difference between "we ingested a JSON" and "we ingested a
real Qwen3 adapter-runtime audit with usable traces/training signals".
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable
import json

from audit_to_adapter_contract import (
    IN_APP_DATASET_EXPORT_FORMAT,
    IN_APP_DATASET_PACKAGE_SCHEMA_VERSIONS,
    IN_APP_DATASET_SOURCE_LAYER,
    LIVE_RUNTIME_SLOTS,
    REDACTED_IN_APP_DATASET_EXPORT_FORMAT,
    REDACTED_IN_APP_DATASET_EXPORT_KIND,
    REDACTED_IN_APP_DATASET_FILE_PREFIX,
    REDACTED_IN_APP_DATASET_PACKAGE_SCHEMA_VERSION,
    REDACTED_IN_APP_DATASET_PRIVACY_POLICY,
    REDACTED_IN_APP_DATASET_PROMPT_POLICY,
    REDACTED_IN_APP_DATASET_SOURCE_ACTIONS,
)


@dataclass
class AuditInspection:
    source: str
    source_format: str = "unknown"
    source_layer: str = "unknown"
    schema_version: str | None = None
    generated_at: str | None = None
    is_in_app_package: bool = False
    trace_count: int = 0
    model_turn_count: int = 0
    qwen3_model_turn_count: int = 0
    shared_adapter_runtime_turn_count: int = 0
    adapter_applied_true_count: int = 0
    adapter_applied_false_count: int = 0
    adapter_applied_missing_count: int = 0
    trace_parse_error_count: int = 0
    trace_selected_tool_allowed_count: int = 0
    package_trace_parse_error_count: int = 0
    package_trace_selected_tool_allowed_count: int = 0
    used_runtime_fallback: bool | None = None
    behavior_violation_count: int = 0
    repair_sample_count: int = 0
    accepted_training_count: int = 0
    quarantined_sample_count: int = 0
    regression_test_count: int = 0
    e2e_scenario_count: int = 0
    e2e_failed_count: int = 0
    e2e_training_signal_count: int = 0
    diagnostics_record_count: int = 0
    diagnostics_failed_count: int = 0
    diagnostics_remediation_proposal_count: int = 0
    adapter_slots_seen: dict[str, int] = field(default_factory=dict)
    slots_seen: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def has_adapter_traces(self) -> bool:
        return self.adapter_applied_true_count + self.adapter_applied_false_count > 0

    @property
    def has_training_signals(self) -> bool:
        return (self.accepted_training_count + self.regression_test_count + self.repair_sample_count) > 0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AuditInspectionSummary:
    inspections: list[AuditInspection]

    def _sum(self, attr: str) -> int:
        return sum(getattr(item, attr) for item in self.inspections)

    @property
    def file_count(self) -> int:
        return len(self.inspections)

    @property
    def in_app_package_count(self) -> int:
        return sum(1 for item in self.inspections if item.is_in_app_package)

    @property
    def trace_count(self) -> int:
        return self._sum("trace_count")

    @property
    def adapter_applied_true_count(self) -> int:
        return self._sum("adapter_applied_true_count")

    @property
    def adapter_applied_false_count(self) -> int:
        return self._sum("adapter_applied_false_count")

    @property
    def adapter_applied_missing_count(self) -> int:
        return self._sum("adapter_applied_missing_count")

    @property
    def accepted_training_count(self) -> int:
        return self._sum("accepted_training_count")

    @property
    def regression_test_count(self) -> int:
        return self._sum("regression_test_count")

    @property
    def e2e_scenario_count(self) -> int:
        return self._sum("e2e_scenario_count")

    @property
    def e2e_failed_count(self) -> int:
        return self._sum("e2e_failed_count")

    @property
    def e2e_training_signal_count(self) -> int:
        return self._sum("e2e_training_signal_count")

    @property
    def diagnostics_record_count(self) -> int:
        return self._sum("diagnostics_record_count")

    @property
    def diagnostics_failed_count(self) -> int:
        return self._sum("diagnostics_failed_count")

    @property
    def warnings(self) -> list[str]:
        out: list[str] = []
        for item in self.inspections:
            out.extend(f"{item.source}: {warning}" for warning in item.warnings)
        return out

    @property
    def errors(self) -> list[str]:
        out: list[str] = []
        for item in self.inspections:
            out.extend(f"{item.source}: {error}" for error in item.errors)
        return out

    @property
    def has_adapter_traces(self) -> bool:
        return any(item.has_adapter_traces for item in self.inspections)

    @property
    def has_training_signals(self) -> bool:
        return any(item.has_training_signals for item in self.inspections)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": "lumen.audit_to_adapter_pipeline.audit_inspection/1.0.0",
            "fileCount": self.file_count,
            "inAppPackageCount": self.in_app_package_count,
            "traceCount": self.trace_count,
            "adapterAppliedTrueCount": self.adapter_applied_true_count,
            "adapterAppliedFalseCount": self.adapter_applied_false_count,
            "adapterAppliedMissingCount": self.adapter_applied_missing_count,
            "acceptedTrainingCount": self.accepted_training_count,
            "regressionTestCount": self.regression_test_count,
            "e2eScenarioCount": self.e2e_scenario_count,
            "e2eFailedCount": self.e2e_failed_count,
            "e2eTrainingSignalCount": self.e2e_training_signal_count,
            "diagnosticsRecordCount": self.diagnostics_record_count,
            "diagnosticsFailedCount": self.diagnostics_failed_count,
            "hasAdapterTraces": self.has_adapter_traces,
            "hasTrainingSignals": self.has_training_signals,
            "warnings": self.warnings,
            "errors": self.errors,
            "files": [item.as_dict() for item in self.inspections],
        }


def inspect_audit_files(paths: Iterable[Path]) -> AuditInspectionSummary:
    return AuditInspectionSummary([inspect_audit_file(path) for path in sorted({path.resolve() for path in paths})])


def inspect_audit_file(path: Path) -> AuditInspection:
    inspection = AuditInspection(source=str(path))
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        inspection.errors.append(f"could not read file: {exc}")
        return inspection

    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        # Sidecar JSONL files are valid training evidence, but not runtime audit
        # packages. Count only syntactically valid JSONL records.
        return inspect_jsonl_sidecar(path, text)

    inspect_payload(value, inspection)
    finalize_inspection(inspection)
    return inspection


def inspect_jsonl_sidecar(path: Path, text: str) -> AuditInspection:
    inspection = AuditInspection(source=str(path), source_format="jsonl_sidecar", source_layer="improveLoopSidecar")
    name = path.name.lower()
    records: list[Any] = []
    invalid_count = 0
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            invalid_count += 1
    if invalid_count:
        inspection.warnings.append(f"ignored {invalid_count} invalid JSONL line(s)")
    if name.startswith("accepted_training"):
        inspection.accepted_training_count = len(records)
    elif name.startswith("quarantined_samples"):
        inspection.quarantined_sample_count = len(records)
    elif name.startswith("regression_tests"):
        inspection.regression_test_count = len(records)
    else:
        inspection.warnings.append("JSON parse failed and filename is not a known improve-loop JSONL sidecar")
    return inspection


def _claims_redacted_v1_package(value: dict[str, Any]) -> bool:
    policy = value.get("exportPolicy") if isinstance(value.get("exportPolicy"), dict) else {}
    test_flight = value.get("testFlight") if isinstance(value.get("testFlight"), dict) else {}
    return (
        value.get("exportKind") == REDACTED_IN_APP_DATASET_EXPORT_KIND
        or policy.get("format") == REDACTED_IN_APP_DATASET_EXPORT_FORMAT
        or test_flight.get("filePrefix") == REDACTED_IN_APP_DATASET_FILE_PREFIX
        or (
            value.get("schemaVersion") == REDACTED_IN_APP_DATASET_PACKAGE_SCHEMA_VERSION
            and any(key in value for key in ("testFlight", "recentTraces", "improveLoop"))
        )
    )


def _redacted_v1_contract_errors(value: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    expected = {
        "schemaVersion": REDACTED_IN_APP_DATASET_PACKAGE_SCHEMA_VERSION,
        "exportKind": REDACTED_IN_APP_DATASET_EXPORT_KIND,
    }
    for key, expected_value in expected.items():
        if value.get(key) != expected_value:
            errors.append(f"{key}={value.get(key)!r}; expected {expected_value!r}")
    required_types = {
        "generatedAt": str,
        "app": dict,
        "manifestSource": str,
        "usedRuntimeFallback": bool,
        "traceSelectedToolAllowedCount": int,
        "traceParseErrorCount": int,
    }
    for key, expected_type in required_types.items():
        field = value.get(key)
        if type(field) is not expected_type or (expected_type is str and not field):
            errors.append(f"{key} must be a non-empty {expected_type.__name__}")

    policy = value.get("exportPolicy") if isinstance(value.get("exportPolicy"), dict) else {}
    expected_policy = {
        "format": REDACTED_IN_APP_DATASET_EXPORT_FORMAT,
        "privacy": REDACTED_IN_APP_DATASET_PRIVACY_POLICY,
        "promptPolicy": REDACTED_IN_APP_DATASET_PROMPT_POLICY,
        "sourceLayer": IN_APP_DATASET_SOURCE_LAYER,
        "ownsLiveE2EScenarios": False,
    }
    for key, expected_value in expected_policy.items():
        if policy.get(key) != expected_value:
            errors.append(f"exportPolicy.{key} does not match redacted-v1")
    includes_static = policy.get("includesDeterministicStaticScenarios")
    if not isinstance(includes_static, bool):
        errors.append("exportPolicy.includesDeterministicStaticScenarios must be boolean")

    traces = value.get("recentTraces")
    scenarios = value.get("scenarioResults")
    if not isinstance(traces, list):
        errors.append("recentTraces must be an array")
    if not isinstance(scenarios, list):
        errors.append("scenarioResults must be an array")
    elif includes_static is False and scenarios:
        errors.append("scenarioResults must be empty when static scenarios are omitted")

    test_flight = value.get("testFlight") if isinstance(value.get("testFlight"), dict) else {}
    if test_flight.get("filePrefix") != REDACTED_IN_APP_DATASET_FILE_PREFIX:
        errors.append("testFlight.filePrefix does not identify redacted-v1")
    source_action = test_flight.get("sourceAction")
    if source_action not in REDACTED_IN_APP_DATASET_SOURCE_ACTIONS:
        errors.append("testFlight.sourceAction is not a current export action")
    if (
        source_action == REDACTED_IN_APP_DATASET_SOURCE_ACTIONS[1]
        and value.get("manifestSource") != "interactive-model-tool-validation-live-e2e"
    ):
        errors.append("interactive model/tool exports require the exact manifestSource")
    live_included = test_flight.get("liveE2EReportIncluded")
    live_report = value.get("liveE2EReport")
    if not isinstance(live_included, bool) or live_included != isinstance(live_report, dict):
        errors.append("testFlight.liveE2EReportIncluded does not match liveE2EReport")
    if source_action == REDACTED_IN_APP_DATASET_SOURCE_ACTIONS[1] and not live_included:
        errors.append("interactive model/tool exports must include liveE2EReport")

    if value.get("exportQualityFailures") != []:
        errors.append("exportQualityFailures must be empty")

    improve_loop = value.get("improveLoop") if isinstance(value.get("improveLoop"), dict) else {}
    for key in ("acceptedTraining", "quarantinedSamples", "regressionTests"):
        if improve_loop.get(key) != []:
            errors.append(f"improveLoop.{key} must be empty in shareable evidence")

    if isinstance(live_report, dict):
        live_policy = live_report.get("exportPolicy") if isinstance(live_report.get("exportPolicy"), dict) else {}
        if live_report.get("schemaVersion") != "1.0.0":
            errors.append("liveE2EReport.schemaVersion must be '1.0.0'")
        if live_policy.get("format") != "live-e2e-test-report-json":
            errors.append("liveE2EReport export format is invalid")
        if live_policy.get("sourceLayer") != "e2eTestReport" or live_policy.get("ownsLiveE2EScenarios") is not True:
            errors.append("only embedded e2eTestReport may own live scenario results")
        if live_report.get("traceSidecarField") != "recentTraces" or not isinstance(live_report.get("payload"), dict):
            errors.append("liveE2EReport must use the recentTraces sidecar and object payload")
    return errors


def inspect_payload(value: Any, inspection: AuditInspection) -> None:
    if isinstance(value, list):
        for item in value:
            inspect_payload(item, inspection)
        return
    if not isinstance(value, dict):
        inspection.errors.append("top-level payload is not a JSON object/list")
        return

    if _claims_redacted_v1_package(value):
        errors = _redacted_v1_contract_errors(value)
        if errors:
            inspection.errors.extend(errors)
            return
        inspect_in_app_package(value, inspection)
        return

    if is_evidence_layer_envelope(value):
        payload = value.get("payload")
        export_policy = value.get("exportPolicy") if isinstance(value.get("exportPolicy"), dict) else {}
        inspection.source_format = str(export_policy.get("format") or "evidence-layer-json")
        inspection.source_layer = str(export_policy.get("sourceLayer") or "unknown")
        inspection.generated_at = str(value.get("generatedAt") or "") or None
        if inspection.source_layer == "e2eTestReport" or export_policy.get("ownsLiveE2EScenarios") is True:
            if isinstance(payload, dict):
                inspect_e2e_report_payload(payload, inspection)
            else:
                inspection.warnings.append("e2eTestReport payload is not a JSON object")
            return
        inspect_payload(payload, inspection)
        return

    if is_in_app_dataset_package(value):
        inspect_in_app_package(value, inspection)
        return

    if is_persistent_runtime_diagnostics_export(value):
        inspect_persistent_runtime_diagnostics(value, inspection)
        return

    if is_e2e_report_payload(value):
        inspect_e2e_report_payload(value, inspection)
        return

    if isinstance(value.get("violations"), list) or isinstance(value.get("repairSamples"), list):
        inspection.source_format = "agent_behavior_audit"
        behavior = value
        inspection.behavior_violation_count += len([item for item in behavior.get("violations", []) if isinstance(item, dict)])
        inspection.repair_sample_count += len([item for item in behavior.get("repairSamples", []) if isinstance(item, dict)])
        return

    if isinstance(value.get("failures"), list):
        inspection.source_format = "runtime_manifest_audit"
        inspection.source_layer = str(value.get("_sourceLayer") or "runtimeManifestAudit")
        return

    inspection.warnings.append("unrecognized JSON audit shape")


def is_evidence_layer_envelope(value: dict[str, Any]) -> bool:
    return isinstance(value.get("exportPolicy"), dict) and "payload" in value



def is_persistent_runtime_diagnostics_export(value: dict[str, Any]) -> bool:
    state = value.get("state")
    return (
        isinstance(state, dict)
        and isinstance(state.get("records"), list)
        and isinstance(value.get("ndjson"), str)
        and "exportedAt" in value
    )


def is_e2e_report_payload(value: dict[str, Any]) -> bool:
    return (
        value.get("kind") in {"lumen_e2e_test_report", "e2e_test_report"}
        or isinstance(value.get("trainingSignals"), list)
        or isinstance(value.get("training_signals"), list)
        or isinstance(value.get("scenarios"), list)
        or isinstance(value.get("results"), list)
    )


def inspect_e2e_report_payload(report: dict[str, Any], inspection: AuditInspection) -> None:
    if inspection.source_format == "unknown":
        inspection.source_format = "lumen_e2e_test_report"
    if inspection.source_layer == "unknown":
        inspection.source_layer = "e2eTestReport.json"
    scenarios = report.get("scenarios")
    if not isinstance(scenarios, list):
        scenarios = report.get("results") if isinstance(report.get("results"), list) else []
    inspection.e2e_scenario_count += len([item for item in scenarios if isinstance(item, dict)])
    failed = 0
    for scenario in scenarios:
        if isinstance(scenario, dict) and scenario.get("passed") is not True:
            failed += 1
    explicit_failed = report.get("failed")
    if isinstance(explicit_failed, int):
        failed = max(failed, explicit_failed)
    inspection.e2e_failed_count += failed
    signals = report.get("trainingSignals") or report.get("training_signals") or []
    if isinstance(signals, list):
        inspection.e2e_training_signal_count += len(signals)
    if inspection.e2e_scenario_count == 0:
        inspection.warnings.append("E2E report contains no scenario results")


def inspect_persistent_runtime_diagnostics(package: dict[str, Any], inspection: AuditInspection) -> None:
    inspection.source_format = "persistent_runtime_diagnostics_export"
    inspection.source_layer = "persistentRuntimeDiagnostics"
    inspection.generated_at = str(package.get("exportedAt") or "") or None
    state = package.get("state") if isinstance(package.get("state"), dict) else {}
    records = [item for item in state.get("records", []) if isinstance(item, dict)]
    inspection.diagnostics_record_count += len(records)
    inspection.diagnostics_failed_count += len(
        [
            item
            for item in records
            if str(item.get("status") or "") != "passed" and not is_expected_diagnostics_cancellation(item)
        ]
    )
    inspection.diagnostics_remediation_proposal_count += sum(
        len(item.get("remediationProposals", []))
        for item in records
        if isinstance(item.get("remediationProposals"), list)
    )
    if not records:
        inspection.warnings.append("persistent diagnostics export contains no records")


def is_expected_diagnostics_cancellation(record: dict[str, Any]) -> bool:
    if str(record.get("status") or "") != "cancelled":
        return False
    metrics = record.get("metrics") if isinstance(record.get("metrics"), dict) else {}
    return metrics.get("didCancel") is True and str(metrics.get("cancellationReason") or "") == "persistent-diagnostics-agent-cancel"


def is_in_app_dataset_package(value: dict[str, Any]) -> bool:
    if _claims_redacted_v1_package(value):
        return not _redacted_v1_contract_errors(value)
    return (
        str(value.get("schemaVersion") or "") in IN_APP_DATASET_PACKAGE_SCHEMA_VERSIONS
        and isinstance(value.get("exportPolicy"), dict)
        and any(key in value for key in ("runtimeManifestAudit", "behaviorAudit", "scenarioResults", "recentTraces", "improveLoop"))
    )


def inspect_in_app_package(package: dict[str, Any], inspection: AuditInspection) -> None:
    export_policy = package.get("exportPolicy") if isinstance(package.get("exportPolicy"), dict) else {}
    inspection.is_in_app_package = True
    inspection.schema_version = str(package.get("schemaVersion") or "") or None
    inspection.source_format = str(export_policy.get("format") or "lumen_in_app_dataset_package")
    inspection.source_layer = str(export_policy.get("sourceLayer") or IN_APP_DATASET_SOURCE_LAYER)
    inspection.generated_at = str(package.get("generatedAt") or "") or None
    inspection.used_runtime_fallback = package.get("usedRuntimeFallback") if isinstance(package.get("usedRuntimeFallback"), bool) else None
    inspection.package_trace_selected_tool_allowed_count = int(package.get("traceSelectedToolAllowedCount") or 0)
    inspection.package_trace_parse_error_count = int(package.get("traceParseErrorCount") or 0)

    if inspection.source_format not in {IN_APP_DATASET_EXPORT_FORMAT, REDACTED_IN_APP_DATASET_EXPORT_FORMAT}:
        inspection.warnings.append(f"unexpected export format: {inspection.source_format}")
    if inspection.source_layer != IN_APP_DATASET_SOURCE_LAYER:
        inspection.warnings.append(f"unexpected source layer: {inspection.source_layer}")
    if inspection.used_runtime_fallback is True:
        inspection.warnings.append("usedRuntimeFallback=true; generated data may represent fallback behavior")

    behavior_audit = package.get("behaviorAudit") if isinstance(package.get("behaviorAudit"), dict) else {}
    inspection.behavior_violation_count += len([item for item in behavior_audit.get("violations", []) if isinstance(item, dict)])
    inspection.repair_sample_count += len([item for item in behavior_audit.get("repairSamples", []) if isinstance(item, dict)])

    traces = package.get("recentTraces") if isinstance(package.get("recentTraces"), list) else []
    for trace in traces:
        if isinstance(trace, dict):
            inspect_trace(trace, inspection)
    if not traces:
        inspection.warnings.append("recentTraces is empty; this audit cannot prove live adapter behavior")

    if traces:
        if inspection.package_trace_selected_tool_allowed_count != inspection.trace_selected_tool_allowed_count:
            inspection.warnings.append(
                "traceSelectedToolAllowedCount package summary differs from recomputed trace count: "
                f"{inspection.package_trace_selected_tool_allowed_count} != {inspection.trace_selected_tool_allowed_count}"
            )
        if inspection.package_trace_parse_error_count != inspection.trace_parse_error_count:
            inspection.warnings.append(
                "traceParseErrorCount package summary differs from recomputed trace count: "
                f"{inspection.package_trace_parse_error_count} != {inspection.trace_parse_error_count}"
            )

    improve_loop = package.get("improveLoop") if isinstance(package.get("improveLoop"), dict) else {}
    inspection.accepted_training_count += len([item for item in improve_loop.get("acceptedTraining", []) if isinstance(item, dict)])
    inspection.quarantined_sample_count += len([item for item in improve_loop.get("quarantinedSamples", []) if isinstance(item, dict)])
    inspection.regression_test_count += len([item for item in improve_loop.get("regressionTests", []) if isinstance(item, dict)])

    live_report = package.get("liveE2EReport")
    if isinstance(live_report, dict) and isinstance(live_report.get("payload"), dict):
        inspect_e2e_report_payload(live_report["payload"], inspection)


def inspect_trace(trace: dict[str, Any], inspection: AuditInspection) -> None:
    inspection.trace_count += 1
    slot = str(trace.get("slot") or "unknown").lower()
    inspection.slots_seen[slot] = inspection.slots_seen.get(slot, 0) + 1

    if trace.get("parseError"):
        inspection.trace_parse_error_count += 1
    if trace.get("selectedToolID") and trace.get("selectedToolID") in (trace.get("allowedToolIDs") or []):
        inspection.trace_selected_tool_allowed_count += 1

    if str(trace.get("event") or "") != "modelTurn":
        return
    inspection.model_turn_count += 1

    model_family = str(trace.get("modelFamily") or "").lower()
    runtime_path = str(trace.get("runtimePath") or "")
    adapter_slot = str(trace.get("adapterSlot") or trace.get("activeAdapterSlot") or "").lower()
    adapter_applied = trace.get("adapterApplied")

    if adapter_slot:
        inspection.adapter_slots_seen[adapter_slot] = inspection.adapter_slots_seen.get(adapter_slot, 0) + 1
    if model_family == "qwen3":
        inspection.qwen3_model_turn_count += 1
    if runtime_path == "sharedAdapter":
        inspection.shared_adapter_runtime_turn_count += 1

    if adapter_applied is True:
        inspection.adapter_applied_true_count += 1
    elif adapter_applied is False:
        inspection.adapter_applied_false_count += 1
    elif model_family == "qwen3" or runtime_path == "sharedAdapter":
        inspection.adapter_applied_missing_count += 1


def finalize_inspection(inspection: AuditInspection) -> None:
    if inspection.is_in_app_package:
        missing_live_slots = [slot for slot in LIVE_RUNTIME_SLOTS if inspection.adapter_slots_seen.get(slot, 0) == 0 and inspection.slots_seen.get(slot, 0) > 0]
        if missing_live_slots:
            inspection.warnings.append("slot traces exist without matching adapterSlot evidence: " + ",".join(missing_live_slots))
        if inspection.qwen3_model_turn_count > 0 and inspection.shared_adapter_runtime_turn_count == 0:
            inspection.warnings.append("qwen3 model turns exist but runtimePath=sharedAdapter was not observed")
        if inspection.shared_adapter_runtime_turn_count > 0 and inspection.adapter_applied_true_count == 0:
            inspection.warnings.append("sharedAdapter runtime observed but no adapterApplied=true trace was found")
        if inspection.adapter_applied_false_count > 0:
            inspection.warnings.append(f"adapterApplied=false observed {inspection.adapter_applied_false_count} time(s)")
        if not inspection.has_training_signals:
            inspection.warnings.append("package contains no accepted training, regression tests, or repair samples")


def assert_audit_requirements(summary: AuditInspectionSummary, *, require_runtime_audit: bool, require_adapter_traces: bool, require_training_signals: bool) -> list[str]:
    errors: list[str] = []
    if require_runtime_audit and summary.file_count == 0:
        errors.append("no runtime audit files found")
    if require_runtime_audit and summary.in_app_package_count == 0:
        errors.append("no Lumen in-app dataset package audit found")
    if require_adapter_traces and not summary.has_adapter_traces:
        errors.append("no adapter trace evidence found; expected adapterApplied/adapterSlot in recentTraces")
    if require_training_signals and not summary.has_training_signals:
        errors.append("no improve-loop training signals found in audit inputs")
    errors.extend(summary.errors)
    return errors
