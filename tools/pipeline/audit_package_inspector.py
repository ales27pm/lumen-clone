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


def inspect_payload(value: Any, inspection: AuditInspection) -> None:
    if isinstance(value, list):
        for item in value:
            inspect_payload(item, inspection)
        return
    if not isinstance(value, dict):
        inspection.errors.append("top-level payload is not a JSON object/list")
        return

    if is_evidence_layer_envelope(value):
        payload = value.get("payload")
        export_policy = value.get("exportPolicy") if isinstance(value.get("exportPolicy"), dict) else {}
        inspection.source_format = str(export_policy.get("format") or "evidence-layer-json")
        inspection.source_layer = str(export_policy.get("sourceLayer") or "unknown")
        inspection.generated_at = str(value.get("generatedAt") or "") or None
        inspect_payload(payload, inspection)
        return

    if is_in_app_dataset_package(value):
        inspect_in_app_package(value, inspection)
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


def is_in_app_dataset_package(value: dict[str, Any]) -> bool:
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

    if inspection.source_format != IN_APP_DATASET_EXPORT_FORMAT:
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
