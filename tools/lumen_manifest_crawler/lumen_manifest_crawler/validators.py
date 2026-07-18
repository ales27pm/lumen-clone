"""Validation rules for manifest extraction and compiled training/eval datasets."""

from __future__ import annotations

# pylint: disable=line-too-long,too-many-lines,too-many-branches,too-many-statements,too-many-locals,too-many-arguments,too-many-nested-blocks,missing-function-docstring

import hashlib
import json
import math
import re
from collections import Counter
from collections.abc import Iterable
from typing import Any

from lumen_manifest_crawler.dataset.adapter_evaluation import mouth_final_text_is_complete
from lumen_manifest_crawler.fleet_artifacts import generate_orchestration_evals
from lumen_manifest_crawler.manifest import AgentBehaviorManifest, ValidationFailure, ValidationReport, ValidationWarning

DEFAULT_SUPPORTED_JSON_TYPES = {"string", "double", "int", "bool", "array", "object", "null", "number", "enum"}
VAGUE_TYPES = {"any", "unknown", "dictionary", "dict"}
SUPPORTED_PERMISSION_KINDS = {"calendar", "reminders", "contacts", "location", "microphone", "speech", "camera", "photos", "motion", "health", "notifications", "alarms"}
SUPPORTED_CONFIRMATION_MODES = {"none", "userApproval"}
STRICT_TOOL_ID_DATASET_FAMILIES = {"tool_schema_cards", "runtime_audit_repairs", "dpo_preference_pairs", "self_model_cards", "self_model_sft", "self_model_eval"}
STRICT_WARNING_CODES = {"tool_missing_description", "vague_argument_type", "inferred_tool_definition", "ambiguous_intent_tools", "freshness_missing_ttl"}
MIN_EVAL_SCENARIOS_PER_TOOL = 5
MIN_SELF_MODEL_EVAL_SCENARIOS = 20
REQUIRED_SELF_MODEL_CARD_TYPES = {
    "slot_contract",
    "tool_boundary",
    "permission_boundary",
    "context_budget_profile",
    "runtime_evidence_policy",
    "artifact_policy",
    "known_gap",
    "repair_sample",
}
INFERRED_TOOL_ARGUMENT_DESCRIPTION_PREFIX = "Inferred from ToolDefinition description"
FORBIDDEN_ARGUMENT_NAMES = {"true", "false"}
FORBIDDEN_CODEBASE_HOME_PATHS = {"ios/Lumen/AgentBehaviorManifest.json"}
FORBIDDEN_CODEBASE_HOME_PREFIXES = ("datasets/public_adapter_corpus/", "generated/agent_manifest/")
PUBLIC_CORPUS_ALLOWED_LICENSES = {"Apache-2.0", "CC-BY-4.0", "MIT"}
PUBLIC_CORPUS_REQUIRED_FIELDS = {
    "targetAdapter",
    "sourceRepository",
    "sourceRevision",
    "sourceLicense",
    "sourceLicenseURL",
    "sourceURL",
    "sourcePath",
    "sourceContentSHA256",
    "sourceArtifactSHA256",
    "sourceGroupID",
    "partitionKind",
    "sourcePartition",
    "transformationVersion",
    "transformedContentSHA256",
    "attribution",
}
PUBLIC_CORPUS_PARTITION_KINDS = {"ml_split", "reference_corpus"}
PUBLIC_CORPUS_ML_TRAINING_PARTITIONS = {"train"}
PUBLIC_CORPUS_RAW_ID_KEYS = {
    "createddate",
    "messageid",
    "messagetreeid",
    "parentid",
    "userid",
    "workerid",
}
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
GIT_REVISION_PATTERN = re.compile(r"^[0-9a-f]{40}$")
ADAPTER_ULTRA_SPECIFIC_SOURCE_FAMILY = "adapter_ultra_specific"
PUBLIC_ADAPTER_CORPUS_PREFIX = "public_adapter_corpus_"
ADAPTER_ROLE_SOURCE_FAMILIES = {
    "executor": {"executor_tool_calls", "tool_schema_cards", "approval_boundary_samples", "negative_samples"},
    "mouth": {"mouth_responses"},
    "mimicry": {"mimicry_style"},
    "rem": {"rem_reflection", "runtime_audit_repairs"},
}
ADAPTER_CODEBASE_SUPPLEMENTAL_SOURCE_FAMILIES = {
    "codebase_home_corpus",
    "codebase_home_sft",
    "codebase_home_chunks",
    "codebase_home_chunk_sft",
    "cortex_codebase_self_awareness",
}
FANOUT_INTENTS = {
    "alarm",
    "calendar",
    "emailDraft",
    "files",
    "maps",
    "memory",
    "messageDraft",
    "note",
    "outlook",
    "phoneCall",
    "photos",
    "rag",
    "reminder",
    "trigger",
    "weather",
    "webSearch",
}
CORTEX_ROUTE_SYSTEM_MARKER = "Task mode: Cortex route mode."
CORTEX_ROUTE_BASE_FIELDS = {
    "intent",
    "selectedToolID",
    "requiresApproval",
    "nextModel",
    "reasoningSummary",
}


class _DuplicateJSONKeyError(ValueError):
    pass


class _NonFiniteJSONNumberError(ValueError):
    pass


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in pairs:
        if key in payload:
            raise _DuplicateJSONKeyError(key)
        payload[key] = value
    return payload


def _reject_nonfinite_json_number(value: str) -> None:
    raise _NonFiniteJSONNumberError(value)


def _parse_finite_json_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        _reject_nonfinite_json_number(value)
    return parsed


def validate_manifest(manifest: AgentBehaviorManifest, dataset_records: dict[str, list[dict]] | None = None, *, strict: bool = False) -> ValidationReport:  # NOSONAR
    failures: list[ValidationFailure] = []
    warnings: list[ValidationWarning] = []

    tool_ids = [tool.id for tool in manifest.tools]
    tool_counts = Counter(tool_ids)
    for tool_id, count in tool_counts.items():
        if count > 1:
            failures.append(ValidationFailure(code="duplicate_tool_id", message=f"Duplicate tool id: {tool_id}", path="tools"))

    known_tools = set(tool_ids)
    for intent in manifest.intents:
        for tool_id in intent.allowedToolIDs:
            if tool_id not in known_tools:
                failures.append(ValidationFailure(code="unknown_intent_tool", message=f"Intent {intent.id} references missing tool {tool_id}", path=f"intents.{intent.id}"))

    raw_supported = manifest.agentProtocols.executorOutput.get("supportedJSONTypes")
    supported_types = set(raw_supported) if raw_supported else DEFAULT_SUPPORTED_JSON_TYPES
    normalized_supported = {str(t).lower() for t in supported_types}
    # An enum is a schema-level restriction on a JSON string, not a distinct
    # AgentJSONValue runtime case. Keep it valid when the extracted executor
    # contract proves string support.
    if "string" in normalized_supported:
        normalized_supported.add("enum")
    for tool in manifest.tools:
        if getattr(tool, "inferred", False):
            warnings.append(
                ValidationWarning(
                    code="inferred_tool_definition",
                    message=f"Tool {tool.id} was inferred from a {tool.inferredSource or 'literal'} and may be missing approval, permission, argument, and description metadata.",
                    path=f"tools.{tool.id}",
                )
            )
        if not tool.description:
            warnings.append(ValidationWarning(code="tool_missing_description", message=f"Tool {tool.id} has no description", path=f"tools.{tool.id}"))
        if tool.permissionKind and tool.permissionKind not in SUPPORTED_PERMISSION_KINDS:
            failures.append(ValidationFailure(code="unsupported_permission_kind", message=f"Tool {tool.id} uses unsupported permission kind {tool.permissionKind}", path=f"tools.{tool.id}.permissionKind"))
        if tool.confirmationMode:
            if tool.confirmationMode not in SUPPORTED_CONFIRMATION_MODES:
                failures.append(ValidationFailure(code="unsupported_confirmation_mode", message=f"Tool {tool.id} uses unsupported confirmation mode {tool.confirmationMode}", path=f"tools.{tool.id}.confirmationMode"))
            expected_confirmation = "userApproval" if tool.requiresApproval else "none"
            if tool.confirmationMode != expected_confirmation:
                failures.append(ValidationFailure(code="confirmation_mode_approval_mismatch", message=f"Tool {tool.id} confirmationMode={tool.confirmationMode} does not match requiresApproval={tool.requiresApproval}", path=f"tools.{tool.id}.confirmationMode"))
        for arg in tool.arguments:
            if arg.name.strip().lower() in FORBIDDEN_ARGUMENT_NAMES:
                failures.append(ValidationFailure(code="literal_value_argument_name", message=f"Tool {tool.id} declares literal value as argument name: {arg.name}", path=f"tools.{tool.id}.arguments.{arg.name}"))
            if (arg.description or "").startswith(INFERRED_TOOL_ARGUMENT_DESCRIPTION_PREFIX):
                failures.append(ValidationFailure(code="inferred_tool_argument_contract", message=f"Tool {tool.id}.{arg.name} argument contract is inferred from description text instead of a declared capability contract", path=f"tools.{tool.id}.arguments.{arg.name}"))
            arg_type = arg.type.lower()
            if arg_type in VAGUE_TYPES:
                warnings.append(ValidationWarning(code="vague_argument_type", message=f"Tool {tool.id}.{arg.name} uses vague type {arg.type}", path=f"tools.{tool.id}.arguments.{arg.name}"))
            if arg_type != "enum" and arg_type not in normalized_supported:
                failures.append(ValidationFailure(code="unsupported_argument_type", message=f"Tool {tool.id}.{arg.name} uses unsupported type {arg.type}", path=f"tools.{tool.id}.arguments.{arg.name}"))

    for slot in manifest.fleet.slots:
        if not slot.role:
            failures.append(ValidationFailure(code="model_slot_missing_role", message=f"Model slot {slot.id} has no role", path=f"fleet.slots.{slot.id}"))

    for entry in manifest.routingMatrix:
        if len(entry.allowedTools) > 1 and entry.intent not in FANOUT_INTENTS:
            warnings.append(ValidationWarning(code="ambiguous_intent_tools", message=f"Intent {entry.intent} has multiple allowed tools", path=f"routingMatrix.{entry.intent}"))

    for freshness in manifest.memory.freshnessClasses:
        if freshness.ttlSeconds is None and not freshness.durable:
            warnings.append(ValidationWarning(code="freshness_missing_ttl", message=f"Freshness class {freshness.id} has no TTL or durable marker", path=f"memory.freshnessClasses.{freshness.id}"))

    if dataset_records:
        _validate_dataset_records(manifest, dataset_records, failures)

    if strict:
        strict_failures = [
            ValidationFailure(code=f"strict_{warning.code}", message=warning.message, path=warning.path)
            for warning in warnings
            if warning.code in STRICT_WARNING_CODES
        ]
        failures.extend(strict_failures)

    return ValidationReport(passed=not failures, failures=failures, warnings=warnings)


def _validate_dataset_records(manifest: AgentBehaviorManifest, records: dict[str, list[dict]], failures: list[ValidationFailure]) -> None:  # NOSONAR
    if "dataset_manifest" in records:
        _validate_dataset_manifest_integrity(records, failures)
    forbidden = set(manifest.sentinels.forbiddenInUserOutput)
    known_tools = {tool.id for tool in manifest.tools}
    approval_tools = {tool.id for tool in manifest.tools if tool.requiresApproval}

    covered_required_tools: set[str] = set()
    covered_approval_tools: set[str] = set()
    compiled_ids: set[str] = set()
    eval_scenarios_by_tool: Counter[str] = Counter()
    eval_tool_records: dict[str, list[dict[str, Any]]] = {tool.id: [] for tool in manifest.tools}

    for name, dataset in records.items():
        for index, record in enumerate(dataset):
            _validate_compiled_record_shape(name, index, record, failures, compiled_ids)
            if name in {"mouth_responses", "mimicry_style", "train_sft", "validation_sft", "tool_schema_cards", "runtime_audit_repairs", "dpo_preference_pairs", "self_model_cards", "self_model_sft", "self_model_eval"}:
                for sentinel in forbidden:
                    if sentinel and _record_model_visible_text_contains(record, sentinel):
                        failures.append(ValidationFailure(code="sentinel_leak", message=f"Sentinel {sentinel} leaked in {name}[{index}]", path=f"dataset.{name}.{index}"))
            if name in {"executor_tool_calls", "approval_boundary_samples"}:
                tool_id = _find_tool_id(record)
                if tool_id:
                    if tool_id not in known_tools:
                        failures.append(ValidationFailure(code="unknown_executor_tool", message=f"Executor dataset references unknown tool {tool_id}", path=f"dataset.{name}.{index}"))
                    covered_required_tools.add(tool_id)
                    if tool_id in approval_tools:
                        covered_approval_tools.add(tool_id)
            if name in STRICT_TOOL_ID_DATASET_FAMILIES:
                for tool_id in _extract_declared_tool_ids(record):
                    if tool_id not in known_tools and not _looks_like_intentionally_invalid_tool(tool_id):
                        failures.append(ValidationFailure(code="unknown_compiled_tool", message=f"Compiled dataset references unknown tool {tool_id}", path=f"dataset.{name}.{index}"))
            if name == "cortex_routing":
                tool_id = _find_selected_tool_id(record)
                if tool_id and tool_id not in known_tools:
                    failures.append(ValidationFailure(code="unknown_cortex_tool", message=f"Cortex dataset references unknown tool {tool_id}", path=f"dataset.{name}.{index}"))
            if name == "eval_scenarios":
                tool_id = _find_eval_expected_tool_id(record)
                if tool_id:
                    if tool_id not in known_tools:
                        failures.append(ValidationFailure(code="unknown_eval_tool", message=f"Eval scenario references unknown tool {tool_id}", path=f"dataset.{name}.{index}"))
                    elif record.get("taskType") == "tool_runtime_scenario_selection":
                        eval_scenarios_by_tool[tool_id] += 1
                        eval_tool_records.setdefault(tool_id, []).append(record)
            if name in {"codebase_home_corpus", "codebase_home_sft", "codebase_home_chunks", "codebase_home_chunk_sft"}:
                _validate_codebase_home_record(name, index, record, failures)

    self_model_cards = records.get("self_model_cards", [])
    self_model_eval = records.get("self_model_eval", [])
    if self_model_cards:
        card_types = {str(record.get("cardType") or ((record.get("metadata") or {}).get("cardType") if isinstance(record.get("metadata"), dict) else "")) for record in self_model_cards}
        missing = sorted(REQUIRED_SELF_MODEL_CARD_TYPES - card_types)
        if missing:
            failures.append(ValidationFailure(code="missing_self_model_card_types", message=f"Self-model cards missing required card types: {', '.join(missing)}", path="dataset.self_model_cards"))
    if self_model_eval and len(self_model_eval) < MIN_SELF_MODEL_EVAL_SCENARIOS:
        failures.append(ValidationFailure(code="missing_self_model_eval_scenarios", message=f"Self-model eval family has {len(self_model_eval)} scenarios; expected at least {MIN_SELF_MODEL_EVAL_SCENARIOS}", path="dataset.self_model_eval"))

    for tool in manifest.tools:
        if any(arg.required for arg in tool.arguments) and tool.id not in covered_required_tools:
            failures.append(ValidationFailure(code="missing_executor_sample", message=f"Tool {tool.id} has required args but no executor sample", path=f"tools.{tool.id}"))
        if tool.requiresApproval and tool.id not in covered_approval_tools:
            failures.append(ValidationFailure(code="missing_approval_sample", message=f"Tool {tool.id} requires approval but has no approval dataset sample", path=f"tools.{tool.id}"))
        if eval_scenarios_by_tool[tool.id] < MIN_EVAL_SCENARIOS_PER_TOOL:
            failures.append(ValidationFailure(code="missing_tool_eval_scenarios", message=f"Tool {tool.id} has {eval_scenarios_by_tool[tool.id]} runtime eval scenarios; expected at least {MIN_EVAL_SCENARIOS_PER_TOOL}", path=f"dataset.eval_scenarios.{tool.id}"))
        scenarios = eval_tool_records.get(tool.id, [])
        natural = [r for r in scenarios if (r.get("metadata") or {}).get("scenarioKind") == "natural_intent"]
        explicit = [r for r in scenarios if (r.get("metadata") or {}).get("scenarioKind") == "explicit_tool_schema"]
        if len(natural) < 2:
            failures.append(ValidationFailure(code="missing_natural_tool_eval_scenarios", message=f"Tool {tool.id} has {len(natural)} natural intent eval scenarios; expected at least 2", path=f"dataset.eval_scenarios.{tool.id}"))
        if not explicit:
            failures.append(ValidationFailure(code="missing_explicit_schema_eval", message=f"Tool {tool.id} is missing explicit schema eval scenarios", path=f"dataset.eval_scenarios.{tool.id}"))
        covered_args: set[str] = set()
        has_approval = False
        has_permission = False
        for record in scenarios:
            metadata = record.get("metadata") or {}
            scenario_kind = metadata.get("scenarioKind")
            arg_cov = metadata.get("argumentCoverage")
            if isinstance(arg_cov, list):
                covered_args.update(arg for arg in arg_cov if isinstance(arg, str))
            if metadata.get("approvalCoverage") is True:
                has_approval = True
            if metadata.get("permissionCoverage") is True:
                has_permission = True
            if scenario_kind == "natural_intent":
                if metadata.get("toolIDVisibleInPrompt") is not False:
                    failures.append(ValidationFailure(code="tool_id_leak_in_natural_eval", message=f"Tool {tool.id} natural eval metadata marks tool id visible", path=f"dataset.eval_scenarios.{tool.id}"))
                prompt_text = "\n".join(
                    message.get("content", "") for message in record.get("messages", []) if isinstance(message, dict)
                )
                if _has_explicit_tool_id_reference(prompt_text, tool.id):
                    failures.append(ValidationFailure(code="tool_id_leak_in_natural_eval", message=f"Tool {tool.id} leaked in natural intent prompt", path=f"dataset.eval_scenarios.{tool.id}"))
        required_args = {arg.name for arg in tool.arguments if arg.required}
        missing_args = sorted(required_args - covered_args)
        if missing_args:
            failures.append(ValidationFailure(code="missing_argument_eval_coverage", message=f"Tool {tool.id} missing argument coverage for: {', '.join(missing_args)}", path=f"dataset.eval_scenarios.{tool.id}"))
        if tool.requiresApproval and not has_approval:
            failures.append(ValidationFailure(code="missing_approval_eval_coverage", message=f"Tool {tool.id} requires approval coverage in eval scenarios", path=f"dataset.eval_scenarios.{tool.id}"))
        if tool.permissionKey and not has_permission:
            failures.append(ValidationFailure(code="missing_permission_eval_coverage", message=f"Tool {tool.id} requires permission coverage in eval scenarios", path=f"dataset.eval_scenarios.{tool.id}"))


def _validate_dataset_manifest_integrity(
    records: dict[str, list[dict]],
    failures: list[ValidationFailure],
) -> None:
    manifest_records = records.get("dataset_manifest")
    if not isinstance(manifest_records, list) or len(manifest_records) != 1 or not isinstance(manifest_records[0], dict):
        failures.append(
            ValidationFailure(
                code="invalid_dataset_manifest",
                message="dataset_manifest must contain exactly one manifest object",
                path="dataset.dataset_manifest",
            )
        )
        return

    manifest_record = manifest_records[0]
    families = {name: family for name, family in records.items() if name != "dataset_manifest"}
    expected_names = set(families)
    counts = manifest_record.get("counts")
    hashes = manifest_record.get("hashes")
    sources = manifest_record.get("sources")
    declared_families = sources.get("datasetFamilies") if isinstance(sources, dict) else None

    if not isinstance(counts, dict) or set(counts) != expected_names:
        failures.append(ValidationFailure(code="dataset_manifest_count_coverage", message="dataset manifest counts must cover every materialized family exactly once", path="dataset.dataset_manifest.counts"))
    if not isinstance(hashes, dict) or set(hashes) != expected_names:
        failures.append(ValidationFailure(code="dataset_manifest_hash_coverage", message="dataset manifest hashes must cover every materialized family exactly once", path="dataset.dataset_manifest.hashes"))
    if declared_families != sorted(expected_names):
        failures.append(ValidationFailure(code="dataset_manifest_family_coverage", message="dataset manifest family inventory does not match materialized families", path="dataset.dataset_manifest.sources.datasetFamilies"))

    for name, family in sorted(families.items()):
        if isinstance(counts, dict) and counts.get(name) != len(family):
            failures.append(ValidationFailure(code="dataset_manifest_count_mismatch", message=f"dataset manifest count for {name} does not match materialized records", path=f"dataset.dataset_manifest.counts.{name}"))
        expected_hash = _canonical_records_hash(family)
        if isinstance(hashes, dict) and hashes.get(name) != expected_hash:
            failures.append(ValidationFailure(code="dataset_manifest_hash_mismatch", message=f"dataset manifest hash for {name} does not match materialized records", path=f"dataset.dataset_manifest.hashes.{name}"))


def _canonical_records_hash(records: list[dict[str, Any]]) -> str:
    payload = "\n".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        for record in records
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()



def _has_explicit_tool_id_reference(prompt_text: str, tool_id: str) -> bool:
    if not prompt_text or not tool_id:
        return False
    if "." in tool_id:
        return tool_id.casefold() in prompt_text.casefold()

    escaped = re.escape(tool_id)
    explicit_patterns = (
        rf"`{escaped}`",
        rf'[\'\"]{escaped}[\'\"]',
        rf"\btool\s+{escaped}\b",
        rf"\buse\s+{escaped}\b",
    )
    lowered = prompt_text.lower()
    return any(re.search(pattern, lowered, flags=re.IGNORECASE) for pattern in explicit_patterns)


def _validate_codebase_home_record(name: str, index: int, record: dict[str, Any], failures: list[ValidationFailure]) -> None:
    for path in _record_path_values(record):
        normalized_path = path.replace("\\", "/").strip("/")
        contains_generated_directory = "generated" in normalized_path.split("/")
        if (
            path in FORBIDDEN_CODEBASE_HOME_PATHS
            or contains_generated_directory
            or any(path.startswith(prefix) for prefix in FORBIDDEN_CODEBASE_HOME_PREFIXES)
        ):
            failures.append(
                ValidationFailure(
                    code="generated_output_in_codebase_home",
                    message=f"{name}[{index}] ingests generated output path {path}; codebase-home records must not depend on generated manifest artifacts",
                    path=f"dataset.{name}.{index}.path",
                )
            )


def _record_path_values(record: dict[str, Any]) -> set[str]:
    paths: set[str] = set()
    for value in (record.get("path"), (record.get("metadata") or {}).get("path") if isinstance(record.get("metadata"), dict) else None):
        if isinstance(value, str) and value:
            paths.add(value)

    for message in record.get("messages", []):
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            continue
        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            continue
        path = payload.get("path") if isinstance(payload, dict) else None
        if isinstance(path, str) and path:
            paths.add(path)
    return paths


def _validate_compiled_record_shape(name: str, index: int, record: dict, failures: list[ValidationFailure], seen_ids: set[str]) -> None:  # NOSONAR
    if name == "dataset_manifest":
        return
    if name in {"train_sft", "validation_sft", "eval_scenarios", "tool_schema_cards", "manifest_grounding_cards", "runtime_audit_repairs", "dpo_preference_pairs", "self_model_cards", "self_model_sft", "self_model_eval"}:
        record_id = record.get("id")
        if not isinstance(record_id, str) or not record_id:
            failures.append(ValidationFailure(code="compiled_record_missing_id", message=f"{name}[{index}] has no stable id", path=f"dataset.{name}.{index}"))
        elif record_id in seen_ids:
            failures.append(ValidationFailure(code="duplicate_compiled_record_id", message=f"Duplicate compiled dataset id {record_id}", path=f"dataset.{name}.{index}"))
        else:
            seen_ids.add(record_id)
    if name in {"train_sft", "validation_sft", "eval_scenarios", "tool_schema_cards", "manifest_grounding_cards", "runtime_audit_repairs", "self_model_cards", "self_model_sft", "self_model_eval"}:
        messages = record.get("messages")
        if not isinstance(messages, list) or not messages:
            failures.append(ValidationFailure(code="compiled_record_missing_messages", message=f"{name}[{index}] has no messages array", path=f"dataset.{name}.{index}"))
        else:
            for message_index, message in enumerate(messages):
                if not isinstance(message, dict) or message.get("role") not in {"system", "user", "assistant", "tool"} or not isinstance(message.get("content"), str):
                    failures.append(ValidationFailure(code="invalid_chat_message", message=f"{name}[{index}].messages[{message_index}] is not canonical chat format", path=f"dataset.{name}.{index}.messages.{message_index}"))
    if name == "self_model_cards":
        card_type = record.get("cardType") or ((record.get("metadata") or {}).get("cardType") if isinstance(record.get("metadata"), dict) else None)
        if card_type not in REQUIRED_SELF_MODEL_CARD_TYPES:
            failures.append(ValidationFailure(code="invalid_self_model_card_type", message=f"self_model_cards record has invalid cardType {card_type}", path=f"dataset.{name}.{index}.cardType"))
        if record.get("sourceFamily") != "self_model_cards":
            failures.append(ValidationFailure(code="self_model_card_missing_source_family", message="self_model_cards record missing sourceFamily marker", path=f"dataset.{name}.{index}.sourceFamily"))
    if name in {"self_model_sft", "self_model_eval"} and record.get("sourceFamily") != name:
        failures.append(ValidationFailure(code="self_model_record_missing_source_family", message=f"{name} record missing sourceFamily marker", path=f"dataset.{name}.{index}.sourceFamily"))
    if name == "runtime_audit_repairs":
        if record.get("sourceFamily") != "runtime_audit_repairs":
            failures.append(ValidationFailure(code="runtime_repair_missing_source_family", message="runtime_audit_repairs record missing sourceFamily marker", path=f"dataset.{name}.{index}.sourceFamily"))
        metadata = record.get("metadata")
        if not isinstance(metadata, dict) or not str(metadata.get("source") or "").strip() or not str(metadata.get("sourceFile") or "").strip():
            failures.append(ValidationFailure(code="runtime_repair_missing_provenance", message="runtime_audit_repairs record missing metadata.source or metadata.sourceFile", path=f"dataset.{name}.{index}.metadata"))
        if not _runtime_repair_has_action(record):
            failures.append(ValidationFailure(code="runtime_repair_missing_action", message="runtime_audit_repairs assistant payload must contain repair.action", path=f"dataset.{name}.{index}.messages"))
        if not _runtime_repair_has_failure_type(record):
            failures.append(ValidationFailure(code="runtime_repair_missing_failure_type", message="runtime_audit_repairs assistant payload must contain failureType", path=f"dataset.{name}.{index}.messages"))
    if name == "dpo_preference_pairs":
        if not isinstance(record.get("prompt"), list) or not isinstance(record.get("chosen"), dict) or not isinstance(record.get("rejected"), dict):
            failures.append(ValidationFailure(code="invalid_dpo_pair", message=f"{name}[{index}] is missing prompt/chosen/rejected", path=f"dataset.{name}.{index}"))


def _record_model_visible_text_contains(record: dict, needle: str) -> bool:
    """Check only prompt/completion text visible to the trained model.

    Grounding metadata may intentionally contain forbidden sentinel strings as a
    blacklist. Treating the whole JSON record as trainable text makes the
    validator report its own guardrail as a leak.
    """
    for value in _model_visible_values(record):
        if needle in value:
            return True
    return False


def _model_visible_values(record: dict) -> Iterable[str]:
    for message in record.get("messages", []):
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if isinstance(content, str):
            yield content
        elif isinstance(content, dict):
            yield from _string_values(content)
    for message in record.get("prompt", []):
        if isinstance(message, dict) and isinstance(message.get("content"), str):
            yield message["content"]
    for key in ("input", "output", "prompt", "completion", "response"):
        value = record.get(key)
        if isinstance(value, str):
            yield value
        elif isinstance(value, dict):
            yield from _string_values(value)


def _string_values(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from _string_values(child)
    elif isinstance(value, list):
        for child in value:
            yield from _string_values(child)


def _runtime_repair_has_action(record: dict[str, Any]) -> bool:
    messages = record.get("messages")
    if not isinstance(messages, list):
        return False
    assistant_messages = [message for message in messages if isinstance(message, dict) and message.get("role") == "assistant"]
    if not assistant_messages:
        return False
    content = assistant_messages[-1].get("content")
    if not isinstance(content, str) or not content.strip():
        return False
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        return False
    if not isinstance(payload, dict):
        return False
    repair = payload.get("repair")
    return isinstance(repair, dict) and isinstance(repair.get("action"), str) and bool(repair.get("action").strip())


def _runtime_repair_has_failure_type(record: dict[str, Any]) -> bool:
    messages = record.get("messages")
    if not isinstance(messages, list):
        return False
    assistant_messages = [message for message in messages if isinstance(message, dict) and message.get("role") == "assistant"]
    if not assistant_messages:
        return False
    content = assistant_messages[-1].get("content")
    if not isinstance(content, str) or not content.strip():
        return False
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        return False
    if not isinstance(payload, dict):
        return False
    value = payload.get("failureType")
    return isinstance(value, str) and bool(value.strip())


def _find_tool_id(record: dict) -> str | None:
    if isinstance(record.get("tool"), str):
        return record["tool"]
    for message in record.get("messages", []):
        content = message.get("content") if isinstance(message, dict) else None
        if isinstance(content, dict) and isinstance(content.get("tool"), str):
            return content["tool"]
    expected = record.get("expectedExecutorOutput")
    if isinstance(expected, dict) and isinstance(expected.get("tool"), str):
        return expected["tool"]
    return None


def _find_selected_tool_id(record: dict) -> str | None:
    for message in record.get("messages", []):
        content = message.get("content") if isinstance(message, dict) else None
        if isinstance(content, dict) and isinstance(content.get("selectedToolID"), str):
            return content["selectedToolID"]
    return None


def _find_eval_expected_tool_id(record: dict) -> str | None:
    expected = record.get("expected")
    if isinstance(expected, dict):
        for key in ("selectedToolID", "tool"):
            value = expected.get(key)
            if isinstance(value, str):
                return value
    return None


def _extract_declared_tool_ids(record: dict) -> set[str]:
    raw = record.get("toolIDs")
    if isinstance(raw, list):
        return {value for value in raw if isinstance(value, str)}
    tool_id = record.get("toolID")
    if isinstance(tool_id, str):
        return {tool_id}
    return set()


def _looks_like_intentionally_invalid_tool(tool_id: str) -> bool:
    lowered = tool_id.lower()
    if lowered.endswith(("fake", "invalid")):
        return True
    tokens = [token for token in re.split(r"[^a-z0-9]+", lowered) if token]
    return "invalid" in tokens


def validate_agent_fine_tuning_datasets(  # NOSONAR
    manifest: AgentBehaviorManifest,
    datasets: dict[str, Any],
    runtime_audit_reports: list[dict[str, Any]] | None = None,
) -> list[ValidationFailure]:
    failures: list[ValidationFailure] = []
    runtime_audit_reports = runtime_audit_reports or []

    known_agents = {"cortex", "executor", "mouth", "mimicry", "rem", "fleet"}
    known_tools = {tool.id for tool in manifest.tools}
    tools_by_id = {tool.id: tool for tool in manifest.tools}
    tool_arg_map = {tool.id: {arg.name for arg in tool.arguments if arg.required} for tool in manifest.tools}
    approval_tools = {tool.id for tool in manifest.tools if tool.requiresApproval}
    permission_tools = {tool.id for tool in manifest.tools if tool.permissionKey}
    slot_ids = {slot.id for slot in manifest.fleet.slots}
    forbidden = {sentinel for sentinel in manifest.sentinels.forbiddenInUserOutput if sentinel}

    for agent in sorted(known_agents):
        if agent not in datasets:
            failures.append(ValidationFailure(code="missing_agent_dataset", message=f"Missing dataset for agent {agent}", path=f"fine_tuning.{agent}"))
    for agent, ds in datasets.items():
        if agent not in known_agents:
            failures.append(ValidationFailure(code="unknown_agent_role", message=f"Unknown agent role {agent}", path=f"fine_tuning.{agent}"))
            continue

        if not isinstance(ds.train_sft, list):
            failures.append(ValidationFailure(code="missing_train_sft", message=f"{agent} train_sft missing", path=f"fine_tuning.{agent}.train_sft"))
            continue
        if not isinstance(ds.val_sft, list):
            failures.append(ValidationFailure(code="missing_val_sft", message=f"{agent} val_sft missing", path=f"fine_tuning.{agent}.val_sft"))
            continue
        if not isinstance(ds.eval, list):
            failures.append(ValidationFailure(code="missing_eval", message=f"{agent} eval missing", path=f"fine_tuning.{agent}.eval"))
            continue
        if not isinstance(ds.dataset_card, dict):
            failures.append(ValidationFailure(code="missing_dataset_card", message=f"{agent} dataset_card missing", path=f"fine_tuning.{agent}.dataset_card"))
            continue
        if not isinstance(ds.unsloth_config, dict):
            failures.append(ValidationFailure(code="missing_unsloth_config", message=f"{agent} unsloth_config missing", path=f"fine_tuning.{agent}.unsloth_config"))
            continue

        _validate_sft_collection_integrity(agent=agent, ds=ds, failures=failures)
        _validate_agent_sft_records(
            agent=agent,
            records=ds.train_sft + ds.val_sft,
            known_tools=known_tools,
            tools_by_id=tools_by_id,
            tool_arg_map=tool_arg_map,
            forbidden=forbidden,
            failures=failures,
        )
        if agent == "executor":
            _validate_executor_dpo_records(
                records=ds.train_dpo + ds.val_dpo,
                tools_by_id=tools_by_id,
                failures=failures,
            )
        _validate_agent_dpo_records(agent=agent, records=ds.train_dpo + ds.val_dpo, failures=failures)
        _validate_agent_eval_records(agent=agent, records=ds.eval, failures=failures, known_tools=known_tools)
        _validate_unsloth_config(agent=agent, config=ds.unsloth_config, failures=failures)

        if agent == "cortex":
            available = _dataset_card_int(ds.dataset_card, "availableSFTRecords")
            trained = len(ds.train_sft) + len(ds.val_sft)
            if available >= 100 and trained < 100:
                failures.append(ValidationFailure(code="cortex_min_records_not_met", message=f"Cortex has {trained} records but at least 100 are available", path="fine_tuning.cortex"))
        if agent == "executor":
            _validate_executor_tool_coverage(ds, known_tools, failures)
            _validate_executor_required_args(ds, tool_arg_map, failures)
        if agent == "mouth":
            if not any((record.get("metadata") or {}).get("evalType") == "sentinel_suppression" for record in ds.eval):
                failures.append(ValidationFailure(code="mouth_missing_sentinel_eval", message="Mouth eval is missing sentinel suppression coverage", path="fine_tuning.mouth.eval"))
        if agent == "fleet":
            _validate_fleet_slot_coverage(ds, slot_ids, failures)
            _validate_fleet_orchestration_eval_coverage(
                manifest=manifest,
                ds=ds,
                failures=failures,
            )

        _validate_natural_intent_tool_leaks(agent=agent, ds=ds, failures=failures, known_tools=known_tools)
        _validate_boundary_coverage(agent=agent, ds=ds, approval_tools=approval_tools, permission_tools=permission_tools, failures=failures)

    return failures


def _dataset_card_int(card: dict[str, Any], key: str) -> int:
    value = card.get(key)
    return value if isinstance(value, int) else 0


def _validate_sft_collection_integrity(
    *,
    agent: str,
    ds: Any,
    failures: list[ValidationFailure],
) -> None:
    train_keys = [_canonical_sft_messages_key(record) for record in ds.train_sft]
    val_keys = [_canonical_sft_messages_key(record) for record in ds.val_sft]
    all_keys = train_keys + val_keys
    duplicate_count = len(all_keys) - len(set(all_keys))
    if duplicate_count:
        failures.append(ValidationFailure(code="duplicate_sft_messages", message=f"{agent} contains {duplicate_count} message-identical SFT records", path=f"fine_tuning.{agent}"))
    if set(train_keys).intersection(val_keys):
        failures.append(ValidationFailure(code="sft_split_overlap", message=f"{agent} train and validation SFT splits overlap", path=f"fine_tuning.{agent}"))

    outputs_by_prompt: dict[str, set[str]] = {}
    for record in ds.train_sft + ds.val_sft:
        prompt_key = _canonical_sft_prompt_key(record)
        outputs_by_prompt.setdefault(prompt_key, set()).add(_canonical_sft_output_key(record))
    conflict_count = sum(1 for outputs in outputs_by_prompt.values() if len(outputs) > 1)
    if conflict_count:
        failures.append(ValidationFailure(code="conflicting_sft_prompt_labels", message=f"{agent} contains {conflict_count} prompts with conflicting assistant labels", path=f"fine_tuning.{agent}"))

    records = ds.train_sft + ds.val_sft
    source_counts = _sft_metadata_counts(records, "sourceFamily")
    task_counts = _sft_metadata_counts(records, "taskType")
    card_source_counts = ds.dataset_card.get("sourceFamilyCounts")
    card_task_counts = ds.dataset_card.get("taskTypeCounts")
    if card_source_counts != source_counts:
        failures.append(ValidationFailure(code="dataset_card_source_counts_mismatch", message=f"{agent} dataset card source counts do not match materialized SFT records", path=f"fine_tuning.{agent}.dataset_card.sourceFamilyCounts"))
    if card_task_counts != task_counts:
        failures.append(ValidationFailure(code="dataset_card_task_counts_mismatch", message=f"{agent} dataset card task counts do not match materialized SFT records", path=f"fine_tuning.{agent}.dataset_card.taskTypeCounts"))

    allowed_sources = ADAPTER_ROLE_SOURCE_FAMILIES.get(agent)
    if allowed_sources is not None:
        invalid_sources = sorted(
            source
            for source in source_counts
            if source not in allowed_sources
            and source != ADAPTER_ULTRA_SPECIFIC_SOURCE_FAMILY
            and not source.startswith(PUBLIC_ADAPTER_CORPUS_PREFIX)
        )
        if invalid_sources:
            failures.append(ValidationFailure(code="off_role_sft_source", message=f"{agent} contains off-role sources: {', '.join(invalid_sources)}", path=f"fine_tuning.{agent}"))

    if agent in {"cortex", "fleet"} and records:
        supplemental_count = sum(source_counts.get(source, 0) for source in ADAPTER_CODEBASE_SUPPLEMENTAL_SOURCE_FAMILIES)
        if supplemental_count / len(records) > 0.251:
            failures.append(ValidationFailure(code="supplemental_sft_ratio_exceeded", message=f"{agent} codebase grounding exceeds 25% of materialized SFT", path=f"fine_tuning.{agent}"))

    max_sequence_length = ds.unsloth_config.get("max_seq_length")
    if isinstance(max_sequence_length, int) and max_sequence_length > 0:
        max_chars = ds.unsloth_config.get("sequence_char_budget")
        if not isinstance(max_chars, int) or max_chars <= 0:
            max_chars = max_sequence_length * 2
        for index, record in enumerate(records):
            serialized = json.dumps(record.get("messages"), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            if len(serialized.encode("utf-8")) > max_chars:
                failures.append(ValidationFailure(code="sft_sequence_budget_exceeded", message=f"{agent} SFT record exceeds the conservative {max_sequence_length}-token byte-proxy budget", path=f"fine_tuning.{agent}.sft.{index}"))

    train_sources = _sft_metadata_counts(ds.train_sft, "sourceFamily")
    val_sources = _sft_metadata_counts(ds.val_sft, "sourceFamily")
    for source, count in source_counts.items():
        if (
            count >= 2
            and not source.startswith(PUBLIC_ADAPTER_CORPUS_PREFIX)
            and (source not in train_sources or source not in val_sources)
        ):
            failures.append(ValidationFailure(code="sft_source_split_missing", message=f"{agent} source {source} is not represented in both train and validation", path=f"fine_tuning.{agent}"))


def _canonical_sft_messages_key(record: dict[str, Any]) -> str:
    messages = record.get("messages")
    if not isinstance(messages, list):
        return json.dumps(messages, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    canonical = [
        {
            "role": str(message.get("role") or ""),
            "content": _canonical_model_output(str(message.get("content") or ""))
            if message.get("role") == "assistant"
            else str(message.get("content") or ""),
        }
        if isinstance(message, dict)
        else {"role": "unknown", "content": str(message)}
        for message in messages
    ]
    return json.dumps(canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _canonical_sft_prompt_key(record: dict[str, Any]) -> str:
    messages = record.get("messages")
    prompt = messages[:-1] if isinstance(messages, list) else messages
    return json.dumps(prompt, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _canonical_sft_output_key(record: dict[str, Any]) -> str:
    messages = record.get("messages")
    if not isinstance(messages, list) or not messages or not isinstance(messages[-1], dict):
        return ""
    return _canonical_model_output(str(messages[-1].get("content") or ""))


def _canonical_model_output(content: str) -> str:
    try:
        return json.dumps(json.loads(content), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except json.JSONDecodeError:
        return " ".join(content.split())


def _sft_metadata_counts(records: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for record in records:
        metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
        counts[str(metadata.get(key) or "unknown")] += 1
    return {value: counts[value] for value in sorted(counts)}


def _validate_agent_sft_records(  # NOSONAR
    *,
    agent: str,
    records: list[dict[str, Any]],
    known_tools: set[str],
    tools_by_id: dict[str, Any],
    tool_arg_map: dict[str, set[str]],
    forbidden: set[str],
    failures: list[ValidationFailure],
) -> None:
    for index, rec in enumerate(records):
        messages = rec.get("messages")
        if not isinstance(messages, list) or len(messages) < 3:
            failures.append(ValidationFailure(code="invalid_chat_format", message=f"{agent} SFT record must use system/user/assistant chat format", path=f"fine_tuning.{agent}.sft.{index}"))
            continue

        assistant = next((m.get("content", "") for m in messages if isinstance(m, dict) and m.get("role") == "assistant"), "")
        system = next((m.get("content", "") for m in messages if isinstance(m, dict) and m.get("role") == "system"), "")
        if not isinstance(assistant, str) or not assistant.strip():
            failures.append(ValidationFailure(code="empty_assistant_output", message=f"{agent} has empty assistant output", path=f"fine_tuning.{agent}.sft.{index}"))
        if agent == "cortex":
            _validate_cortex_json_object_contract(
                assistant=assistant,
                failures=failures,
                path=f"fine_tuning.{agent}.sft.{index}",
                preference_chosen=False,
                route_contract=(
                    isinstance(system, str)
                    and CORTEX_ROUTE_SYSTEM_MARKER in system
                ),
            )
        if agent == "mouth" and not mouth_final_text_is_complete(assistant):
            failures.append(
                ValidationFailure(
                    code="mouth_incomplete_sft_output",
                    message="Mouth SFT output must be complete user-facing text",
                    path=f"fine_tuning.{agent}.sft.{index}",
                )
            )
        for sentinel in forbidden:
            if sentinel in assistant:
                failures.append(ValidationFailure(code="sentinel_leak", message=f"{agent} leaked sentinel `{sentinel}`", path=f"fine_tuning.{agent}.sft.{index}"))

        metadata = rec.get("metadata")
        if not isinstance(metadata, dict):
            failures.append(ValidationFailure(code="missing_sft_metadata", message=f"{agent} SFT metadata missing", path=f"fine_tuning.{agent}.sft.{index}.metadata"))
            continue
        if metadata.get("agent") != agent:
            failures.append(ValidationFailure(code="unknown_agent_role", message=f"SFT record metadata.agent mismatch for {agent}", path=f"fine_tuning.{agent}.sft.{index}.metadata.agent"))

        _validate_public_corpus_metadata(
            metadata.get("publicCorpus"),
            agent=agent,
            path=f"fine_tuning.{agent}.sft.{index}.metadata.publicCorpus",
            failures=failures,
        )

        tool_ids = metadata.get("toolIDs")
        if isinstance(tool_ids, list):
            for tool_id in tool_ids:
                if not isinstance(tool_id, str):
                    continue
                if tool_id not in known_tools:
                    failures.append(ValidationFailure(code="unknown_tool_id", message=f"{agent} references unknown tool {tool_id}", path=f"fine_tuning.{agent}.sft.{index}.metadata.toolIDs"))
        if agent == "executor" and isinstance(tool_ids, list):
            _validate_executor_json_contract(
                assistant=assistant,
                tools_by_id=tools_by_id,
                failures=failures,
                path=f"fine_tuning.{agent}.sft.{index}",
            )
            task_type = str(metadata.get("taskType") or "")
            source_family = str(metadata.get("sourceFamily") or "")
            if task_type not in {"tool_call_generation", "argument_completion", "required_args"} and source_family not in {"executor_tool_calls", "approval_boundary_samples"}:
                continue
            if not _should_enforce_required_args(assistant):
                continue
            for tool_id in tool_ids:
                required_args = tool_arg_map.get(tool_id, set())
                if not required_args:
                    continue
                if not _assistant_mentions_required_args(assistant, required_args):
                    failures.append(ValidationFailure(code="executor_missing_required_args", message=f"Executor sample for {tool_id} missing required args in assistant output", path=f"fine_tuning.{agent}.sft.{index}"))


def _validate_cortex_json_object_contract(
    *,
    assistant: Any,
    failures: list[ValidationFailure],
    path: str,
    preference_chosen: bool,
    route_contract: bool,
) -> None:
    prefix = "cortex_dpo" if preference_chosen else "cortex"
    label = "Cortex DPO chosen output" if preference_chosen else "Cortex SFT output"
    try:
        payload = json.loads(
            assistant,
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_nonfinite_json_number,
            parse_float=_parse_finite_json_float,
        )
    except _DuplicateJSONKeyError as exc:
        failures.append(
            ValidationFailure(
                code=f"{prefix}_duplicate_json_key",
                message=f"{label} repeats JSON key `{exc.args[0]}`",
                path=path,
            )
        )
        return
    except (json.JSONDecodeError, TypeError, _NonFiniteJSONNumberError):
        failures.append(
            ValidationFailure(
                code=f"{prefix}_non_json_output",
                message=f"{label} must be strict JSON",
                path=path,
            )
        )
        return
    if not isinstance(payload, dict):
        failures.append(
            ValidationFailure(
                code=f"{prefix}_non_object_output",
                message=f"{label} must be a JSON object",
                path=path,
            )
        )
        return
    if route_contract and not _valid_cortex_route_payload(payload):
        failures.append(
            ValidationFailure(
                code=f"{prefix}_route_contract_invalid",
                message=f"{label} does not match one exact Cortex route mode",
                path=path,
            )
        )


def _valid_cortex_route_payload(payload: dict[str, Any]) -> bool:
    if (
        not isinstance(payload.get("intent"), str)
        or not payload["intent"].strip()
        or type(payload.get("requiresApproval")) is not bool
        or not isinstance(payload.get("nextModel"), str)
        or not isinstance(payload.get("reasoningSummary"), str)
        or not payload["reasoningSummary"].strip()
        or "tool" in payload
        or "arguments" in payload
    ):
        return False

    selected_tool_id = payload.get("selectedToolID")
    status = payload.get("status")
    if status in {"no_tool_route", "invalid_tool"}:
        return (
            set(payload) == CORTEX_ROUTE_BASE_FIELDS | {"status"}
            and selected_tool_id is None
            and payload["requiresApproval"] is False
            and payload["nextModel"] == "mouth"
        )
    if status == "needs_clarification":
        missing_arguments = payload.get("missingArguments")
        clarification = payload.get("clarification")
        return (
            set(payload)
            == CORTEX_ROUTE_BASE_FIELDS
            | {"status", "missingArguments", "clarification"}
            and isinstance(selected_tool_id, str)
            and bool(selected_tool_id)
            and payload["nextModel"] == "mouth"
            and isinstance(missing_arguments, list)
            and bool(missing_arguments)
            and all(
                isinstance(argument, str) and bool(argument)
                for argument in missing_arguments
            )
            and len(missing_arguments) == len(set(missing_arguments))
            and isinstance(clarification, str)
            and clarification.strip().endswith("?")
        )
    if not isinstance(selected_tool_id, str) or not selected_tool_id:
        return False

    expected_next_model = (
        "approval" if payload["requiresApproval"] else "executor"
    )
    if payload["nextModel"] != expected_next_model:
        return False
    if set(payload) == CORTEX_ROUTE_BASE_FIELDS:
        return True
    action_step = payload.get("actionStep")
    return (
        set(payload) == CORTEX_ROUTE_BASE_FIELDS | {"actionStep"}
        and isinstance(action_step, dict)
        and set(action_step) == {"type", "toolID", "mustPersistBeforeFinal"}
        and action_step.get("type") == "tool_call"
        and action_step.get("toolID") == selected_tool_id
        and action_step.get("mustPersistBeforeFinal") is True
    )


def _validate_executor_json_contract(
    *,
    assistant: str,
    tools_by_id: dict[str, Any],
    failures: list[ValidationFailure],
    path: str,
) -> None:
    try:
        payload = json.loads(assistant)
    except (json.JSONDecodeError, TypeError):
        failures.append(ValidationFailure(code="executor_non_json_output", message="Executor SFT output must be a strict JSON object", path=path))
        return
    if not isinstance(payload, dict):
        failures.append(ValidationFailure(code="executor_non_object_output", message="Executor SFT output must be a JSON object", path=path))
        return

    top_level_keys = set(payload)
    if "thought" in payload and not isinstance(payload.get("thought"), str):
        failures.append(ValidationFailure(code="executor_response_shape_invalid", message="Executor thought must be a string when present", path=path))
        return
    if "final" in payload:
        if top_level_keys not in ({"final"}, {"final", "thought"}):
            failures.append(ValidationFailure(code="executor_response_shape_invalid", message="Executor final output may contain only final and optional thought", path=path))
            return
        final = payload.get("final")
        if not isinstance(final, str) or not final.strip():
            failures.append(ValidationFailure(code="executor_response_shape_invalid", message="Executor final output must contain non-empty final text", path=path))
        return
    if top_level_keys not in ({"action"}, {"action", "thought"}):
        failures.append(ValidationFailure(code="executor_response_shape_invalid", message="Executor output must use the native action or final envelope without legacy metadata", path=path))
        return
    action = payload.get("action")
    if not isinstance(action, dict) or set(action) != {"tool", "args"}:
        failures.append(ValidationFailure(code="executor_response_shape_invalid", message="Executor action must contain exactly tool and args", path=path))
        return
    tool_id = action.get("tool")
    if not isinstance(tool_id, str) or tool_id not in tools_by_id:
        failures.append(ValidationFailure(code="executor_invalid_payload_tool", message="Executor action must contain one manifest tool id", path=path))
        return
    arguments = action.get("args")
    if not isinstance(arguments, dict):
        failures.append(ValidationFailure(code="executor_invalid_arguments", message=f"Executor action for {tool_id} must contain an args object", path=path))
        return

    tool = tools_by_id[tool_id]
    arguments_by_name = {argument.name: argument for argument in tool.arguments}
    unknown_arguments = sorted(set(arguments).difference(arguments_by_name))
    if unknown_arguments:
        failures.append(ValidationFailure(code="executor_extra_arguments", message=f"Executor payload for {tool_id} has extra arguments: {', '.join(unknown_arguments)}", path=path))
    for name, value in arguments.items():
        argument = arguments_by_name.get(name)
        if argument is None:
            continue
        if argument.allowedValues and value not in argument.allowedValues:
            failures.append(ValidationFailure(code="executor_invalid_enum_argument", message=f"Executor payload for {tool_id}.{name} must use a manifest allowed value", path=path))
        if not _executor_argument_type_is_valid(value, argument.type):
            failures.append(ValidationFailure(code="executor_invalid_argument_type", message=f"Executor payload for {tool_id}.{name} has the wrong JSON type", path=path))

    missing_required = {
        argument.name
        for argument in tool.arguments
        if argument.required and argument.name not in arguments
    }
    if missing_required:
        failures.append(ValidationFailure(code="executor_missing_required_args", message=f"Executor action for {tool_id} omits required arguments", path=path))


def _validate_executor_dpo_records(
    *,
    records: list[dict[str, Any]],
    tools_by_id: dict[str, Any],
    failures: list[ValidationFailure],
) -> None:
    """Require the preferred Executor response to obey the production tool contract.

    Rejected responses are intentionally allowed to demonstrate malformed calls;
    only the chosen side is eligible to teach the adapter its output contract.
    """
    for index, record in enumerate(records):
        chosen = record.get("chosen")
        assistant = chosen.get("content") if isinstance(chosen, dict) else None
        if not isinstance(assistant, str):
            failures.append(
                ValidationFailure(
                    code="executor_dpo_missing_chosen_output",
                    message="Executor DPO record must contain a chosen assistant response",
                    path=f"fine_tuning.executor.dpo.{index}",
                )
            )
            continue
        _validate_executor_json_contract(
            assistant=assistant,
            tools_by_id=tools_by_id,
            failures=failures,
            path=f"fine_tuning.executor.dpo.{index}.chosen",
        )


def _executor_argument_type_is_valid(value: Any, declared_type: str) -> bool:
    type_name = declared_type.strip().lower()
    if type_name in {"string", "enum"}:
        return isinstance(value, str)
    if type_name in {"bool", "boolean"}:
        return isinstance(value, bool)
    if type_name in {"int", "integer"}:
        return isinstance(value, int) and not isinstance(value, bool)
    if type_name in {"number", "float", "double"}:
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if type_name in {"object", "dictionary"}:
        return isinstance(value, dict)
    if type_name in {"array", "list"}:
        return isinstance(value, list)
    if type_name in {"null", "none", "nil"}:
        return value is None
    return False


def _assistant_mentions_required_args(assistant: str, required_args: set[str]) -> bool:
    try:
        parsed = json.loads(assistant)
    except json.JSONDecodeError:
        lowered = assistant.lower()
        return all(arg.lower() in lowered for arg in required_args)

    if isinstance(parsed, dict):
        action = parsed.get("action")
        args = action.get("args") if isinstance(action, dict) else None
        if isinstance(args, dict):
            return required_args.issubset(set(args.keys()))
    return False


def _should_enforce_required_args(assistant: str) -> bool:
    try:
        payload = json.loads(assistant)
    except json.JSONDecodeError:
        return True
    if not isinstance(payload, dict):
        return True
    return isinstance(payload.get("action"), dict)


def _validate_agent_dpo_records(*, agent: str, records: list[dict[str, Any]], failures: list[ValidationFailure]) -> None:
    for index, rec in enumerate(records):
        metadata = rec.get("metadata") if isinstance(rec.get("metadata"), dict) else {}
        _validate_public_corpus_metadata(
            metadata.get("publicCorpus"),
            agent=agent,
            path=f"fine_tuning.{agent}.dpo.{index}.metadata.publicCorpus",
            failures=failures,
        )
        prompt = rec.get("prompt")
        chosen = rec.get("chosen")
        rejected = rec.get("rejected")
        if not isinstance(prompt, list) or not isinstance(chosen, dict) or not isinstance(rejected, dict):
            failures.append(ValidationFailure(code="invalid_dpo_pair", message=f"{agent} DPO record missing prompt/chosen/rejected", path=f"fine_tuning.{agent}.dpo.{index}"))
            continue
        chosen_text = chosen.get("content")
        rejected_text = rejected.get("content")
        if not isinstance(chosen_text, str) or not isinstance(rejected_text, str):
            failures.append(ValidationFailure(code="invalid_dpo_pair", message=f"{agent} DPO chosen/rejected content missing", path=f"fine_tuning.{agent}.dpo.{index}"))
            continue
        if agent == "cortex":
            system = next(
                (
                    message.get("content", "")
                    for message in prompt
                    if isinstance(message, dict)
                    and message.get("role") == "system"
                ),
                "",
            )
            _validate_cortex_json_object_contract(
                assistant=chosen_text,
                failures=failures,
                path=f"fine_tuning.{agent}.dpo.{index}.chosen",
                preference_chosen=True,
                route_contract=(
                    isinstance(system, str)
                    and CORTEX_ROUTE_SYSTEM_MARKER in system
                ),
            )
        if agent == "mouth" and not mouth_final_text_is_complete(chosen_text):
            failures.append(
                ValidationFailure(
                    code="mouth_incomplete_dpo_chosen_output",
                    message="Mouth DPO chosen output must be complete user-facing text",
                    path=f"fine_tuning.{agent}.dpo.{index}.chosen",
                )
            )
        if chosen_text == rejected_text:
            failures.append(ValidationFailure(code="dpo_chosen_equals_rejected", message=f"{agent} DPO chosen == rejected", path=f"fine_tuning.{agent}.dpo.{index}"))


def _validate_public_corpus_metadata(
    value: Any,
    *,
    agent: str,
    path: str,
    failures: list[ValidationFailure],
) -> None:
    if value is None:
        return
    if not isinstance(value, dict):
        failures.append(ValidationFailure(code="invalid_public_corpus_metadata", message="publicCorpus metadata must be an object", path=path))
        return

    missing = sorted(
        key
        for key in PUBLIC_CORPUS_REQUIRED_FIELDS
        if not isinstance(value.get(key), str) or not str(value.get(key)).strip()
    )
    if missing:
        failures.append(ValidationFailure(code="public_corpus_missing_provenance", message=f"Public corpus metadata is missing: {', '.join(missing)}", path=path))

    target = str(value.get("targetAdapter") or "").strip().lower()
    if target != agent:
        failures.append(ValidationFailure(code="public_corpus_adapter_mismatch", message=f"Public corpus target {target or '<missing>'} does not match {agent}", path=f"{path}.targetAdapter"))

    license_id = str(value.get("sourceLicense") or "").strip()
    if license_id not in PUBLIC_CORPUS_ALLOWED_LICENSES:
        failures.append(ValidationFailure(code="public_corpus_license_not_allowed", message=f"Public corpus license {license_id or '<missing>'} is not allowlisted", path=f"{path}.sourceLicense"))

    revision = str(value.get("sourceRevision") or "").strip().lower()
    if revision and GIT_REVISION_PATTERN.fullmatch(revision) is None:
        failures.append(ValidationFailure(code="public_corpus_revision_not_pinned", message="Public corpus revision must be a full 40-character lowercase Git revision", path=f"{path}.sourceRevision"))

    for field in ("sourceContentSHA256", "sourceArtifactSHA256", "sourceGroupID", "transformedContentSHA256"):
        digest = str(value.get(field) or "").strip().lower()
        if digest and SHA256_PATTERN.fullmatch(digest) is None:
            failures.append(ValidationFailure(code="public_corpus_invalid_digest", message=f"{field} must be a 64-character lowercase SHA-256 digest", path=f"{path}.{field}"))

    partition_kind = str(value.get("partitionKind") or "").strip().lower()
    source_partition = str(value.get("sourcePartition") or "").strip().lower()
    if partition_kind not in PUBLIC_CORPUS_PARTITION_KINDS:
        failures.append(ValidationFailure(code="public_corpus_invalid_partition_kind", message=f"Public corpus partitionKind {partition_kind or '<missing>'} is not supported", path=f"{path}.partitionKind"))
    elif partition_kind == "ml_split" and source_partition not in PUBLIC_CORPUS_ML_TRAINING_PARTITIONS:
        failures.append(ValidationFailure(code="public_corpus_heldout_split_ingested", message=f"ML partition {source_partition or '<missing>'} is not approved for training", path=f"{path}.sourcePartition"))

    for field in ("sourceURL", "sourceLicenseURL"):
        url = str(value.get(field) or "").strip()
        if url and not url.startswith("https://"):
            failures.append(ValidationFailure(code="public_corpus_insecure_source_url", message=f"{field} must use HTTPS", path=f"{path}.{field}"))

    leaked_keys = sorted(_public_corpus_raw_id_keys(value))
    if leaked_keys:
        failures.append(ValidationFailure(code="public_corpus_raw_identifier_leak", message=f"Public corpus metadata retained raw source identifiers: {', '.join(leaked_keys)}", path=path))


def _public_corpus_raw_id_keys(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = re.sub(r"[^a-z0-9]", "", str(key).lower())
            if normalized in PUBLIC_CORPUS_RAW_ID_KEYS:
                found.add(str(key))
            found.update(_public_corpus_raw_id_keys(child))
    elif isinstance(value, list):
        for child in value:
            found.update(_public_corpus_raw_id_keys(child))
    return found


def _validate_agent_eval_records(
    *,
    agent: str,
    records: list[dict[str, Any]],
    failures: list[ValidationFailure],
    known_tools: set[str],
) -> None:
    for index, rec in enumerate(records):
        expected = rec.get("expected")
        if not isinstance(expected, dict):
            failures.append(ValidationFailure(code="eval_missing_expected", message=f"{agent} eval has no expected field", path=f"fine_tuning.{agent}.eval.{index}"))
            continue
        messages = rec.get("messages")
        if not isinstance(messages, list) or len(messages) < 2:
            failures.append(ValidationFailure(code="invalid_chat_format", message=f"{agent} eval must contain messages", path=f"fine_tuning.{agent}.eval.{index}"))
        for key in ("selectedToolID", "tool"):
            value = expected.get(key)
            if isinstance(value, str) and value not in known_tools:
                failures.append(ValidationFailure(code="unknown_tool_id", message=f"{agent} eval expected references unknown tool {value}", path=f"fine_tuning.{agent}.eval.{index}.expected.{key}"))
        if agent == "mouth":
            metrics = rec.get("metrics")
            if not isinstance(metrics, list) or not any(
                isinstance(metric, dict)
                and metric.get("type") == "complete_final_text"
                for metric in metrics
            ):
                failures.append(
                    ValidationFailure(
                        code="mouth_eval_missing_completeness_metric",
                        message="Every Mouth evaluation must enforce complete final text",
                        path=f"fine_tuning.{agent}.eval.{index}.metrics",
                    )
                )


def _validate_unsloth_config(*, agent: str, config: dict[str, Any], failures: list[ValidationFailure]) -> None:
    required = {
        "agent",
        "base_model_name",
        "max_seq_length",
        "load_in_4bit",
        "lora_r",
        "lora_alpha",
        "learning_rate",
        "dataset_dir",
        "output_dir",
    }
    for key in required:
        if key not in config:
            failures.append(ValidationFailure(code="missing_unsloth_config_key", message=f"{agent} missing unsloth key {key}", path=f"fine_tuning.{agent}.unsloth_config.{key}"))


def _validate_executor_tool_coverage(ds: Any, known_tools: set[str], failures: list[ValidationFailure]) -> None:
    covered: set[str] = set()
    for record in ds.train_sft + ds.val_sft:
        metadata = record.get("metadata") if isinstance(record, dict) else None
        tool_ids = metadata.get("toolIDs") if isinstance(metadata, dict) else None
        if isinstance(tool_ids, list):
            covered.update(tool_id for tool_id in tool_ids if isinstance(tool_id, str))
    missing = sorted(tool for tool in known_tools if tool not in covered)
    if missing:
        failures.append(ValidationFailure(code="executor_tool_coverage_missing", message=f"Executor missing tool coverage for: {', '.join(missing[:10])}", path="fine_tuning.executor"))


def _validate_executor_required_args(ds: Any, tool_arg_map: dict[str, set[str]], failures: list[ValidationFailure]) -> None:  # NOSONAR
    for record in ds.train_sft + ds.val_sft:
        metadata = record.get("metadata")
        if not isinstance(metadata, dict):
            continue
        tool_ids = metadata.get("toolIDs")
        if not isinstance(tool_ids, list):
            continue
        task_type = str(metadata.get("taskType") or "")
        source_family = str(metadata.get("sourceFamily") or "")
        if task_type not in {"tool_call_generation", "argument_completion", "required_args"} and source_family not in {"executor_tool_calls", "approval_boundary_samples"}:
            continue
        assistant = next((m.get("content", "") for m in record.get("messages", []) if isinstance(m, dict) and m.get("role") == "assistant"), "")
        if not _should_enforce_required_args(assistant):
            continue
        for tool_id in tool_ids:
            required = tool_arg_map.get(tool_id, set())
            if required and not _assistant_mentions_required_args(assistant, required):
                failures.append(ValidationFailure(code="missing_required_args_executor_examples", message=f"Executor example missing required args for {tool_id}", path="fine_tuning.executor"))


def _has_runtime_repair_sample(ds: Any) -> bool:
    for record in ds.train_sft + ds.val_sft:
        metadata = record.get("metadata")
        if isinstance(metadata, dict) and metadata.get("sourceFamily") == "runtime_audit_repairs":
            return True
        if isinstance(metadata, dict) and metadata.get("taskType") == "runtime_manifest_drift_repair":
            return True
    return False


def _validate_fleet_slot_coverage(ds: Any, slot_ids: set[str], failures: list[ValidationFailure]) -> None:
    blob = "\n".join(json.dumps(record, ensure_ascii=False, sort_keys=True) for record in (ds.train_sft + ds.val_sft))
    for slot_id in sorted(slot_ids):
        if slot_id not in blob:
            failures.append(ValidationFailure(code="fleet_slot_coverage_missing", message=f"fleet missing role-card coverage for slot {slot_id}", path="fine_tuning.fleet"))


def _validate_fleet_orchestration_eval_coverage(
    *,
    manifest: AgentBehaviorManifest,
    ds: Any,
    failures: list[ValidationFailure],
) -> None:
    expected_scenarios = {
        str(record.get("metadata", {}).get("scenarioID") or "")
        for record in generate_orchestration_evals(manifest)
    }
    expected_scenarios.discard("")
    actual_scenarios = [
        str(record.get("metadata", {}).get("scenarioID") or "")
        for record in ds.eval
        if record.get("metadata", {}).get("evalType")
        == "fleet_orchestration_event_graph_eval"
    ]
    actual_scenario_set = {scenario for scenario in actual_scenarios if scenario}
    if (
        actual_scenario_set != expected_scenarios
        or len(actual_scenarios) != len(expected_scenarios)
    ):
        missing = sorted(expected_scenarios - actual_scenario_set)
        unexpected = sorted(actual_scenario_set - expected_scenarios)
        failures.append(
            ValidationFailure(
                code="fleet_orchestration_eval_coverage_missing",
                message=(
                    "Fleet orchestration eval coverage must exactly match the "
                    "manifest-derived scenarios; "
                    f"missing={missing}, unexpected={unexpected}, "
                    f"actualCount={len(actual_scenarios)}, "
                    f"expectedCount={len(expected_scenarios)}"
                ),
                path="fine_tuning.fleet.eval",
            )
        )


def _validate_natural_intent_tool_leaks(*, agent: str, ds: Any, failures: list[ValidationFailure], known_tools: set[str]) -> None:  # NOSONAR
    for index, rec in enumerate(ds.eval):
        metadata = rec.get("metadata")
        if not isinstance(metadata, dict):
            continue
        if metadata.get("evalType") not in {"tool_runtime_scenario_selection", "routing_matrix_adherence"}:
            continue
        messages = rec.get("messages")
        if not isinstance(messages, list):
            continue
        prompt = "\n".join(
            msg.get("content", "")
            for msg in messages
            if isinstance(msg, dict) and msg.get("role") == "user" and isinstance(msg.get("content"), str)
        )
        expected = rec.get("expected")
        if not isinstance(expected, dict):
            continue
        tool_id = expected.get("selectedToolID") or expected.get("tool")
        if isinstance(tool_id, str) and tool_id in known_tools:
            if tool_id in prompt and "natural" in prompt.lower():
                failures.append(ValidationFailure(code="natural_intent_tool_id_leak", message=f"{agent} eval prompt leaks tool id {tool_id}", path=f"fine_tuning.{agent}.eval.{index}"))


def _validate_boundary_coverage(
    *,
    agent: str,
    ds: Any,
    approval_tools: set[str],
    permission_tools: set[str],
    failures: list[ValidationFailure],
) -> None:
    if agent not in {"cortex", "executor"}:
        return
    has_approval = any(
        (record.get("metadata") or {}).get("risk") == "approval_required"
        for record in (ds.train_sft + ds.val_sft)
    )
    if approval_tools and not has_approval:
        failures.append(ValidationFailure(code="missing_approval_boundary_examples", message=f"{agent} missing approval boundary examples", path=f"fine_tuning.{agent}"))
    has_permission = any(
        (record.get("metadata") or {}).get("risk") == "permissioned"
        for record in (ds.train_sft + ds.val_sft)
    )
    if permission_tools and not has_permission:
        failures.append(ValidationFailure(code="missing_permission_boundary_examples", message=f"{agent} missing permission boundary examples", path=f"fine_tuning.{agent}"))
