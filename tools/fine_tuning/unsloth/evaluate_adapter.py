from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

try:
    from lumen_manifest_crawler.dataset.chat_template_contract import (
        apply_non_thinking_chat_template,
        canonical_non_thinking_messages,
        strip_terminal_non_thinking_directive,
        verify_chat_template_contract,
    )
except ImportError:
    from tools.lumen_manifest_crawler.lumen_manifest_crawler.dataset.chat_template_contract import (
        apply_non_thinking_chat_template,
        canonical_non_thinking_messages,
        strip_terminal_non_thinking_directive,
        verify_chat_template_contract,
    )

try:
    from .export_gguf import (
        _validate_config as _validate_export_config,
        _verified_release_bake_lineage,
    )
    from .train_sft import (
        _controlled_torch_dtype,
        _load_verified_runtime_tokenizer_source,
        _require_unsloth_before_transformers,
        _runtime_tokenizer_evidence,
        _seed_everything,
        _verify_runtime_model_binding,
        _verify_runtime_tokenizer_binding,
    )
except ImportError:
    module_dir = str(Path(__file__).resolve().parent)
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)
    from export_gguf import (  # type: ignore
        _validate_config as _validate_export_config,
        _verified_release_bake_lineage,
    )
    from train_sft import (  # type: ignore
        _controlled_torch_dtype,
        _load_verified_runtime_tokenizer_source,
        _require_unsloth_before_transformers,
        _runtime_tokenizer_evidence,
        _seed_everything,
        _verify_runtime_model_binding,
        _verify_runtime_tokenizer_binding,
    )


EVALUATION_RUN_SCHEMA_VERSION = "lumen.adapter-evaluation-run/1.5.0"
EVALUATION_CHECKPOINT_SCHEMA_VERSION = (
    "lumen.adapter-evaluation-checkpoint/1.0.0"
)
EVALUATION_CHECKPOINT_CONTRACT_SCHEMA_VERSION = (
    "lumen.adapter-evaluation-checkpoint-contract/1.0.0"
)
EVALUATION_CHECKPOINT_ENTRY_SCHEMA_VERSION = (
    "lumen.adapter-evaluation-checkpoint-entry/1.0.0"
)
EVALUATION_CHECKPOINT_FILENAME = "evaluation_checkpoint.json"
EVALUATION_CHECKPOINT_HASH_FIELD = "evaluationCheckpointSHA256"
EVALUATION_CHECKPOINT_MAX_BYTES = 256 << 20
EVALUATION_FINAL_FILENAMES = (
    "candidate_outputs.jsonl",
    "evaluation_report.json",
    "evaluation_run_manifest.json",
)
EVALUATION_ATOMIC_WRITE_TARGET_FILENAMES = (
    EVALUATION_CHECKPOINT_FILENAME,
    *EVALUATION_FINAL_FILENAMES,
)
EVALUATION_ATOMIC_WRITE_TEMP_NAME_PATTERN = re.compile(
    r"^\.(?:"
    + "|".join(
        re.escape(name) for name in EVALUATION_ATOMIC_WRITE_TARGET_FILENAMES
    )
    + r")\.[a-z0-9_]{8}\.tmp$"
)
UBUNTU_SOURCE_INTEGRITY_FIELDS = (
    "workingTreeDigest",
    "ubuntuOrchestrationCodeSHA256",
    "ubuntuSourceIntegritySHA256",
    "ubuntuSourceIntegrity",
)
CANDIDATE_OUTPUT_SCHEMA_VERSION = "lumen.adapter-eval-candidate/1.2.0"
GENERATION_ATTEMPT_SCHEMA_VERSION = "lumen.adapter-eval-generation-attempt/1.0.0"
STRUCTURED_OUTPUT_CONTRACT_VERSION = "lumen.adapter-eval-json-object-contract/1.1.0"
OUTPUT_MODE_CONTRACT_VERSION = "lumen.adapter-eval-output-mode-contract/1.0.0"
CORTEX_ROUTING_CONTEXT_VERSION = "lumen.adapter-eval-cortex-routing-context/1.0.0"
STRICT_JSON_RETRY_CONTRACT_VERSION = "lumen.adapter-eval-strict-json-retry/1.4.0"
STRICT_JSON_MAX_ATTEMPTS = 2
GENERATION_REPETITION_PENALTY = 1.1
STRUCTURED_OUTPUT_INSTRUCTION = (
    "Response format contract: output exactly one valid JSON object. Do not include "
    "prose, markdown, code fences, or hidden reasoning."
)
CORTEX_ROUTE_INSTRUCTION = (
    "Cortex route mode: use the manifest catalog below as exact runtime truth. "
    "The catalog is TSV: defaultIntent is the canonical intent for an ordinary "
    "action request, allowedIntents is the comma-separated set of manifest-routed "
    "intents, required '-' means no required arguments, and approval 1 means true "
    "while 0 means false. "
    "Never invent, rename, pluralize, or abbreviate a tool ID. If the user names a "
    "catalog ID explicitly, copy it exactly. Match ordinary requests to catalog names "
    "and descriptions. Choose selectedToolID first, then find the single TSV row whose "
    "id cell exactly equals it and stop consulting every other row. "
    "Every actionable or clarification route copies defaultIntent exactly; never "
    "echo meta-language such as app "
    "action, operation, request, or capability as intent. Only an explicit choose-only "
    "intent-category request may use a different value, and it must occur verbatim in "
    "that row's allowedIntents cell. Copy required and approval only from that row. If "
    "the request explicitly says choose or select only, return routing only, or do not "
    "begin the action, emit the five common fields and stop. Otherwise start with that "
    "row's required names; '-' is an empty list. When required is '-', the missing "
    "set is empty: emit actionStep and never borrow a field, status, or clarification "
    "from another row. "
    "Optional names mentioned in descriptions are never required. Natural wording can "
    "supply a value without naming its field. A concrete topic after about, regarding, "
    "or concerning supplies `query`. A complete proposition introduced by that supplies "
    "`content`; explicit personal-preference wording supplies both that content and the "
    "preference `kind`, even though Cortex never emits Executor arguments. A specifically "
    "designated recipient may "
    "be an address, person, organization, or role such as the supplier and supplies "
    "`to`. An event title must be a distinct name, topic, or description separate from "
    "the calendar-create operation. The generic object in operation wording such as "
    "create, add, or schedule an event, calendar event, appointment, meeting, or calendar "
    "entry does not supply `title`. A separate name or `for <topic>` complement does supply "
    "`title`, even when that topic is a simple noun. A precise relative delay supplies "
    "`inMinutes` or `startsInMinutes`, while a vague daypart or scheduling adverb does not. "
    "Operation wording supplies no required values. A standalone pronoun, deictic phrase, "
    "unresolved relative reference, or bare object class does not supply an identifier, "
    "path, query, title, body, content, or kind. Recognize explicit content but never "
    "guess an absent value. Audit each required name literally before choosing a route "
    "state. A tool display name or operation phrase never supplies a same-named "
    "argument: asking to schedule an agent run supplies no title, prompt, or schedule; "
    "asking to reply or reply-all supplies no body; a countdown duration plus alert "
    "wording supplies no title. References such as that item or the selected message "
    "supply no id or messageId. One narrow runtime-supported exception applies: an "
    "explicit latest, last, or newest email reference for an Outlook message "
    "operation supplies the "
    "symbolic `messageId` value `latest`; generic latest-item wording and selected or "
    "current message references remain unresolved. Only concrete user values or that "
    "narrow symbolic value remove those names from the missing set. Remove a "
    "required name only when this user request supplies its concrete value; do not copy "
    "Executor arguments into the route. If names remain, emit "
    "all of them in manifest order in missingArguments and omit actionStep; if none "
    "remain, emit actionStep. Never infer one required value from another. Always emit "
    "top-level keys in this visible order: selectedToolID (catalog string or null), "
    "intent (string), reasoningSummary (one concise manifest-row grounding sentence), "
    "then status, missingArguments, and clarification or actionStep when applicable, "
    "then requiresApproval (boolean) and nextModel (string). The reasoningSummary is "
    "not hidden chain-of-thought. For an actionable route, it states that the exact "
    "selected row has no required values or names all exact required values as "
    "supplied. For a clarification, it states the exact selected row and exact missing "
    "subset. For a "
    "complete actionable request, also emit actionStep exactly as "
    '{"type":"tool_call","toolID":"<same selectedToolID>",'
    '"mustPersistBeforeFinal":true}; set requiresApproval from the catalog and set '
    "nextModel to approval when true, otherwise executor. When required arguments are "
    "missing, keep the canonical selectedToolID and catalog requiresApproval, omit "
    "actionStep, set nextModel to mouth, set status to needs_clarification, and emit "
    "missingArguments plus one clarification. Use no_tool_route only when no catalog "
    "tool applies, and invalid_tool only for a requested nonexistent ID. Do not emit "
    "status on complete actionable routes."
)
CORTEX_ROUTE_DECISION_ENDCAP = (
    "Final route decision: selectedToolID is an exact column-1 ID. Lock to that "
    "one row: every action or clarification copies its defaultIntent; only a "
    "five-field explicit choose-only selection may use another allowedIntent. Required "
    "'-' means empty and can never produce missingArguments or clarification; required "
    "names and approval come from that row only. "
    "Treat concrete natural implicit values as supplied, including a specifically "
    "designated recipient role. Operation wording, standalone pronouns, and unresolved "
    "relative references such as that item, this one, the latest item, the selected "
    "message, or the entry discussed earlier do not supply an identifier or other "
    "required value by themselves. The narrow runtime-supported exception is an "
    "explicit latest, last, or newest email reference for an Outlook message "
    "operation, which supplies the "
    "symbolic messageId value `latest`; it does not apply to generic latest items or "
    "selected/current message wording. In contrast, a that-clause containing a complete "
    "proposition supplies content, a concrete topic after about, regarding, or concerning "
    "supplies query, personal-preference wording supplies preference kind, and a distinct "
    "event name, topic, or `for <topic>` complement supplies title. The generic event-like "
    "object of a create, add, or schedule operation is only the object class and does not "
    "supply title. Precise relative delays can supply numeric time "
    "fields; vague dayparts cannot. If any required value is "
    "absent, summarize that row and exact missing subset before status, "
    "missingArguments, and clarification; omit actionStep. Otherwise summarize that "
    "the row has no required values or that every exact required name is supplied, "
    "then emit actionStep. Finish with requiresApproval and nextModel."
)
CORTEX_TOOL_CATALOG_HEADER = (
    "Manifest tools TSV: id\tname\tdefaultIntent\tallowedIntents\trequired\tapproval\tdescription"
)
STRICT_JSON_RETRY_INSTRUCTION = (
    "This is the single bounded retry after strict raw JSON or manifest-route "
    "validation failed. Re-read the manifest catalog and the user's request. "
    "Emit a fresh, complete JSON object now. Output JSON only: no prose, markdown, "
    "code fences, comments, or hidden reasoning. Start with { and stop after its "
    "matching }. Keep the object concise. Do not emit a tool catalog, a rejected-tool "
    "list, repeated keys, or an unbounded array. Do not repeat or repair the previous "
    "output. For Cortex, a trusted exact-row digest may follow; treat it as "
    "authoritative manifest data."
)
GENERIC_STRICT_JSON_RETRY_INSTRUCTION = (
    "This is the single bounded retry after strict raw JSON validation failed. "
    "Re-read the response-format contract and the user's request. Emit a fresh, "
    "complete JSON object now. Output JSON only: no prose, markdown, code fences, "
    "comments, or hidden reasoning. Start with { and stop after its matching }. "
    "Keep the object concise. Do not emit a tool catalog, repeat or repair the "
    "previous output, emit duplicate keys, or emit an unbounded array."
)
_CORTEX_RETRY_GUIDANCE_BY_FAILURE_CODE = {
    "invalid_json": (
        "Retry repair: discard the malformed text and emit the complete Cortex "
        "route object from scratch. Include selectedToolID, intent, and "
        "reasoningSummary before the route state, then requiresApproval and "
        "nextModel. Never emit a flat tool call or Executor arguments."
    ),
    "cortex_route_protocol_field_invalid": (
        "Retry repair: emit the complete Cortex route object from scratch. "
        "Include selectedToolID, intent, and reasoningSummary before the route "
        "state, then requiresApproval and nextModel. Never emit a flat tool call, "
        "an actionStep-only fragment, or Executor arguments."
    ),
    "cortex_route_tool_not_in_manifest": (
        "Retry repair: reselect an exact column-1 tool ID from the catalog; "
        "never reuse or mutate the invalid ID."
    ),
    "cortex_route_intent_not_in_manifest": (
        "Retry repair: after selecting one row, copy its defaultIntent for an "
        "ordinary request or a verbatim allowedIntents value for an explicit "
        "choose-only request."
    ),
    "cortex_route_clarification_state_invalid": (
        "Retry repair: reread the selected row's required names. If every "
        "required value is supplied, emit an actionable route with actionStep; "
        "otherwise emit only the exact absent names in manifest order."
    ),
    "cortex_route_action_state_invalid": (
        "Retry repair: reread the selected row's required names. If any required "
        "value is absent, omit actionStep and ask for only the exact absent names "
        "in manifest order; never infer them. Otherwise emit actionStep with exactly "
        "type tool_call, the matching selectedToolID as toolID, and "
        "mustPersistBeforeFinal true; never emit false."
    ),
}
SUPPORTED_AGENTS = ("cortex", "executor", "mouth", "mimicry", "rem", "fleet")
SUPPORTED_OUTPUT_MODES = frozenset({"json", "text"})
CORTEX_FORBIDDEN_ROUTE_FIELDS = frozenset({"rejectedToolID", "rejectedToolIDs"})
_CORTEX_ROUTE_PREFIX_FIELDS = (
    "selectedToolID",
    "intent",
    "reasoningSummary",
)
_CORTEX_ROUTE_SUFFIX_FIELDS = ("requiresApproval", "nextModel")
_CORTEX_ACTION_STEP_FIELDS = ("type", "toolID", "mustPersistBeforeFinal")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_TOOL_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_.]*$")
_REGRESSION_FAMILY_PATTERN = re.compile(
    r"^[a-z][a-z0-9]*(?:[._-][a-z0-9]+)*$"
)
_MESSAGE_ROLES = frozenset({"system", "user", "assistant"})


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a finalized per-agent LoRA adapter over its frozen evaluation "
            "suite and produce promotion-compatible candidate outputs and scores."
        )
    )
    parser.add_argument("--config", required=True, help="Finalized adapter config JSON.")
    parser.add_argument(
        "--adapter-dir",
        help="Override config adapter_output_dir (the finalized manifest must still bind it).",
    )
    parser.add_argument(
        "--finalized-variant-manifest",
        help=(
            "Override the finalized manifest path. Defaults to config "
            "finalized_variant_manifest or <output_dir>/finalized_variant_manifest.json."
        ),
    )
    parser.add_argument(
        "--eval-jsonl",
        help="Override the frozen eval JSONL. Defaults to <dataset_dir>/eval.jsonl.",
    )
    parser.add_argument(
        "--behavior-manifest",
        default="generated/agent_manifest/AgentBehaviorManifest.json",
        help="Behavior manifest supplying the exact tool and fleet contracts used by scoring.",
    )
    parser.add_argument(
        "--output-dir",
        help="Evaluation output directory. Defaults to <finalized-manifest-dir>/evaluation.",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        help=(
            "Deterministic semantic sample size for a smoke run. System-prompt and "
            "evalID revisions do not reshuffle the sample. Omit to run the full "
            "frozen suite."
        ),
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1024,
        help="Bounded generation budget per example (default: 1024; maximum: 4096).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Atomically replace existing evaluator output files.",
    )
    parser.add_argument(
        "--verify-checkpoint-only",
        action="store_true",
        help=(
            "Verify an exact recoverable interrupted evaluation state without "
            "loading the model or publishing final outputs."
        ),
    )
    return parser.parse_args(argv)


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


_SMOKE_METADATA_STRING_DIMENSIONS = (
    "agent",
    "evalType",
    "name",
    "scenarioKind",
    "scenarioID",
    "sourceClass",
    "coverageFamily",
)
_SMOKE_EXPECTED_CATEGORICAL_FIELDS = (
    "status",
    "scenarioKind",
    "permissionKey",
    "repairAction",
    "failureType",
    "expectedTTLClass",
    "strategy",
    "expectedStopReason",
    "tone",
    "length",
    "format",
    "delegateTo",
    "aggregationOwnerSlotID",
    "expectedAggregationOwnerSlotID",
)
_SMOKE_EXPECTED_LIST_DIMENSIONS = (
    "missingArguments",
    "requiredEventTypes",
    "expectedDelegatedSlotIDs",
    "requiredContextKeys",
    "forbiddenContextKeys",
)
_SMOKE_METRIC_CATEGORICAL_FIELDS = (
    "mode",
    "expectedIntent",
    "expectedRepairAction",
    "expectedTTLClass",
    "expectedStopReason",
)
_SMOKE_TOOL_FIELDS = ("selectedToolID", "tool", "toolID")
_SMOKE_METRIC_TOOL_FIELDS = ("expectedToolID", "forbiddenToolID")


def _smoke_semantic_token(value: Any) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Evaluation record contains a non-JSON semantic smoke value"
        ) from exc


def _semantic_smoke_projection(record: Mapping[str, Any]) -> dict[str, Any]:
    """Return stable case semantics, excluding generated IDs and system text."""

    messages = record.get("messages")
    metadata = record.get("metadata")
    metrics = record.get("metrics")
    if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes)):
        raise ValueError("Evaluation record lacks messages for semantic smoke selection")
    if not isinstance(metadata, Mapping):
        raise ValueError("Evaluation record lacks metadata for semantic smoke selection")
    if not isinstance(metrics, Sequence) or isinstance(metrics, (str, bytes)) or not metrics:
        raise ValueError("Evaluation record lacks metrics for semantic smoke selection")

    eval_type = metadata.get("evalType")
    agent = metadata.get("agent")
    if not isinstance(agent, str) or not agent.strip():
        raise ValueError("Evaluation record lacks an agent for semantic smoke selection")
    if not isinstance(eval_type, str) or not eval_type.strip():
        raise ValueError("Evaluation record lacks an evalType for semantic smoke selection")

    semantic_metadata: dict[str, Any] = {}
    for key in _SMOKE_METADATA_STRING_DIMENSIONS:
        if key not in metadata or metadata[key] is None:
            continue
        value = metadata[key]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"Evaluation record metadata.{key} is invalid for semantic smoke selection"
            )
        semantic_metadata[key] = value
    for key in (
        "critical",
        "mustPass",
        "approvalCoverage",
        "permissionCoverage",
    ):
        if key not in metadata or metadata[key] is None:
            continue
        value = metadata[key]
        if type(value) is not bool:
            raise ValueError(
                f"Evaluation record metadata.{key} is invalid for semantic smoke selection"
            )
        semantic_metadata[key] = value
    if "argumentCoverage" in metadata and metadata["argumentCoverage"] is not None:
        value = metadata["argumentCoverage"]
        if (
            not isinstance(value, Sequence)
            or isinstance(value, (str, bytes))
            or any(not isinstance(item, str) or not item for item in value)
        ):
            raise ValueError(
                "Evaluation record metadata.argumentCoverage is invalid for "
                "semantic smoke selection"
            )
        semantic_metadata["argumentCoverage"] = list(value)
    if "regressionFamilies" in metadata:
        value = metadata["regressionFamilies"]
        if (
            type(value) is not list
            or not value
            or any(
                not isinstance(item, str)
                or len(item) > 128
                or _REGRESSION_FAMILY_PATTERN.fullmatch(item) is None
                for item in value
            )
            or len(set(value)) != len(value)
        ):
            raise ValueError(
                "Evaluation record metadata.regressionFamilies is invalid for "
                "semantic smoke selection"
            )
        semantic_metadata["regressionFamilies"] = sorted(value)
    if "scenario" in metadata and metadata["scenario"] is not None:
        semantic_metadata["scenario"] = metadata["scenario"]

    scenario_messages: list[dict[str, str]] = []
    for index, message in enumerate(messages):
        if not isinstance(message, Mapping):
            raise ValueError(
                f"Evaluation record messages[{index}] is invalid for semantic smoke selection"
            )
        role = message.get("role")
        content = message.get("content")
        if (
            not isinstance(role, str)
            or role not in {"system", "user", "assistant"}
            or not isinstance(content, str)
            or not content.strip()
        ):
            raise ValueError(
                f"Evaluation record messages[{index}] is invalid for semantic smoke selection"
            )
        if role != "system":
            scenario_messages.append({"role": role, "content": content})
    if not scenario_messages:
        raise ValueError(
            "Evaluation record lacks non-system messages for semantic smoke selection"
        )

    semantic_metrics: list[dict[str, Any]] = []
    for index, metric in enumerate(metrics):
        if not isinstance(metric, Mapping):
            raise ValueError(
                f"Evaluation record metrics[{index}] is invalid for semantic smoke selection"
            )
        metric_type = metric.get("type")
        if not isinstance(metric_type, str) or not metric_type.strip():
            raise ValueError(
                f"Evaluation record metrics[{index}] lacks a type for semantic smoke selection"
            )
        semantic_metrics.append(dict(metric))

    expected = record.get("expected")
    if expected is not None and not isinstance(expected, Mapping):
        raise ValueError(
            "Evaluation record expected contract is invalid for semantic smoke selection"
        )

    projection: dict[str, Any] = {
        "metadata": semantic_metadata,
        "messages": scenario_messages,
        "metrics": semantic_metrics,
    }
    if expected is not None:
        projection["expected"] = dict(expected)
    output_mode = record.get("outputMode")
    if output_mode is not None:
        if not isinstance(output_mode, str) or not output_mode:
            raise ValueError(
                "Evaluation record outputMode is invalid for semantic smoke selection"
            )
        projection["outputMode"] = output_mode
    return projection


def _semantic_smoke_sort_key(record: Mapping[str, Any]) -> str:
    """Return a stable scenario key that deliberately ignores ID/system churn."""

    return _canonical_sha256(_semantic_smoke_projection(record))


def _add_smoke_feature(
    features: dict[tuple[str, str], int],
    *,
    priority: int,
    dimension: str,
    value: Any,
) -> None:
    features[(dimension, _smoke_semantic_token(value))] = priority


def _add_smoke_string_list_features(
    features: dict[tuple[str, str], int],
    *,
    priority: int,
    dimension: str,
    value: Any,
) -> None:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or any(not isinstance(item, str) or not item for item in value)
    ):
        raise ValueError(
            f"Evaluation record {dimension} is invalid for semantic smoke selection"
        )
    values = list(value)
    _add_smoke_feature(
        features,
        priority=priority,
        dimension=f"{dimension}.signature",
        value=values,
    )
    _add_smoke_feature(
        features,
        priority=priority,
        dimension=f"{dimension}.count",
        value=len(values),
    )
    for item in values:
        _add_smoke_feature(
            features,
            priority=max(1, priority - 1),
            dimension=f"{dimension}.item",
            value=item,
        )


def _semantic_smoke_coverage_features(
    record: Mapping[str, Any],
) -> dict[tuple[str, str], int]:
    """Extract only behavior-relevant, frozen dimensions used for smoke coverage."""

    projection = _semantic_smoke_projection(record)
    metadata = projection["metadata"]
    metrics = projection["metrics"]
    expected = projection.get("expected", {})
    features: dict[tuple[str, str], int] = {}

    _add_smoke_feature(
        features,
        priority=4,
        dimension="metadata.evalType",
        value=metadata["evalType"],
    )
    if metadata.get("critical") is True or metadata.get("mustPass") is True:
        _add_smoke_feature(
            features,
            priority=5,
            dimension="metadata.criticalEvalType",
            value=metadata["evalType"],
        )
    for key in ("scenarioKind", "scenarioID", "sourceClass"):
        if key in metadata:
            _add_smoke_feature(
                features,
                priority=4 if key == "scenarioKind" else 3,
                dimension=f"metadata.{key}",
                value=metadata[key],
            )
    for key in ("coverageFamily", "approvalCoverage", "permissionCoverage"):
        if key in metadata:
            _add_smoke_feature(
                features,
                priority=4 if key == "coverageFamily" else 3,
                dimension=f"metadata.{key}",
                value=metadata[key],
            )
    if "argumentCoverage" in metadata:
        _add_smoke_string_list_features(
            features,
            priority=3,
            dimension="metadata.argumentCoverage",
            value=metadata["argumentCoverage"],
        )
    for regression_family in metadata.get("regressionFamilies", []):
        _add_smoke_feature(
            features,
            priority=5,
            dimension="metadata.regressionFamily",
            value=regression_family,
        )
    if "scenario" in metadata:
        _add_smoke_feature(
            features,
            priority=3,
            dimension="metadata.scenario",
            value=metadata["scenario"],
        )
    if "name" in metadata:
        _add_smoke_feature(
            features,
            priority=1,
            dimension="metadata.name",
            value=metadata["name"],
        )

    for metric in metrics:
        metric_type = metric["type"]
        _add_smoke_feature(
            features,
            priority=4,
            dimension="metric.type",
            value=metric_type,
        )
        for key in _SMOKE_METRIC_CATEGORICAL_FIELDS:
            if key in metric:
                value = metric[key]
                if not isinstance(value, str) or not value:
                    raise ValueError(
                        f"Evaluation metric {key} is invalid for semantic smoke selection"
                    )
                _add_smoke_feature(
                    features,
                    priority=4 if key == "mode" else 3,
                    dimension=f"metric.{metric_type}.{key}",
                    value=value,
                )
        for key in _SMOKE_METRIC_TOOL_FIELDS:
            if key in metric:
                value = metric[key]
                if not isinstance(value, str) or not value:
                    raise ValueError(
                        f"Evaluation metric {key} is invalid for semantic smoke selection"
                    )
                _add_smoke_feature(
                    features,
                    priority=3,
                    dimension=f"metric.{key}",
                    value=value,
                )
        if "requiredArguments" in metric:
            _add_smoke_string_list_features(
                features,
                priority=3,
                dimension=f"metric.{metric_type}.requiredArguments",
                value=metric["requiredArguments"],
            )
        for key, value in metric.items():
            if key != "type" and type(value) is bool:
                _add_smoke_feature(
                    features,
                    priority=3,
                    dimension=f"metric.{metric_type}.{key}",
                    value=value,
                )

    for key in _SMOKE_TOOL_FIELDS:
        if key not in expected:
            continue
        value = expected[key]
        if value is not None and (not isinstance(value, str) or not value):
            raise ValueError(
                f"Evaluation expected.{key} is invalid for semantic smoke selection"
            )
        _add_smoke_feature(
            features,
            priority=3,
            dimension=f"expected.{key}",
            value=value,
        )
    for key in _SMOKE_EXPECTED_CATEGORICAL_FIELDS:
        if key not in expected:
            continue
        value = expected[key]
        if value is not None and (not isinstance(value, str) or not value):
            raise ValueError(
                f"Evaluation expected.{key} is invalid for semantic smoke selection"
            )
        _add_smoke_feature(
            features,
            priority=4 if key in {"status", "scenarioKind"} else 3,
            dimension=f"expected.{key}",
            value=value,
        )
    if "boundaryContract" in expected:
        value = expected["boundaryContract"]
        if not isinstance(value, Mapping) or not value:
            raise ValueError(
                "Evaluation expected.boundaryContract is invalid for semantic smoke selection"
            )
        _add_smoke_feature(
            features,
            priority=3,
            dimension="expected.boundaryContract",
            value=value,
        )
    for key in _SMOKE_EXPECTED_LIST_DIMENSIONS:
        if key in expected:
            _add_smoke_string_list_features(
                features,
                priority=3,
                dimension=f"expected.{key}",
                value=expected[key],
            )
    for key, value in expected.items():
        if type(value) is bool:
            _add_smoke_feature(
                features,
                priority=3,
                dimension=f"expected.boolean.{key}",
                value=value,
            )
    return features


def select_evaluation_records(
    records: Sequence[dict[str, Any]],
    *,
    max_examples: int | None,
) -> list[dict[str, Any]]:
    """Greedily select a stable cohort that maximizes behavioral coverage."""

    if max_examples is None:
        return list(records)
    if type(max_examples) is not int or max_examples <= 0:
        raise ValueError("max_examples must be a positive integer")
    if max_examples > len(records):
        raise ValueError("max_examples exceeds the frozen evaluation case count")

    candidates: list[
        tuple[str, dict[tuple[str, str], int], dict[str, Any]]
    ] = []
    for record in records:
        semantic_key = _semantic_smoke_sort_key(record)
        candidates.append(
            (semantic_key, _semantic_smoke_coverage_features(record), record)
        )

    feature_frequencies: dict[tuple[str, str], int] = {}
    for _, features, _ in candidates:
        for feature in features:
            feature_frequencies[feature] = feature_frequencies.get(feature, 0) + 1

    def coverage_score(
        features: Mapping[tuple[str, str], int],
        *,
        covered: set[tuple[str, str]],
    ) -> tuple[int, ...]:
        score: list[int] = []
        for priority in (5, 4, 3, 2, 1):
            uncovered = [
                feature
                for feature, feature_priority in features.items()
                if feature_priority == priority and feature not in covered
            ]
            score.extend(
                (
                    -len(uncovered),
                    -sum(
                        1_000_000 // feature_frequencies[feature]
                        for feature in uncovered
                    ),
                )
            )
        return tuple(score)

    covered: set[tuple[str, str]] = set()
    selected: list[dict[str, Any]] = []
    while len(selected) < max_examples:
        best_index = min(
            range(len(candidates)),
            key=lambda index: (
                coverage_score(candidates[index][1], covered=covered),
                candidates[index][0],
            ),
        )
        _, features, record = candidates.pop(best_index)
        selected.append(record)
        covered.update(features)
    return selected


def _verified_evaluation_execution_plan(
    cfg: Mapping[str, Any],
    *,
    max_examples: int | None,
    frozen_case_count: int,
) -> dict[str, Any] | None:
    raw_plan = cfg.get("runExecutionPlan")
    if raw_plan is None:
        return None
    from tools.fine_tuning.unsloth.ubuntu_pipeline import (
        _verified_execution_plan,
    )

    plan = _verified_execution_plan(raw_plan)
    scope = plan["evaluationScope"]
    planned_max_examples = plan["evaluationMaxExamples"]
    if scope == "none":
        raise ValueError("The prepared execution plan disables evaluation")
    if scope == "full" and max_examples is not None:
        raise ValueError("Full evaluation cannot use --max-examples")
    if scope == "smoke" and max_examples != planned_max_examples:
        raise ValueError("--max-examples drifted from the prepared smoke plan")
    if scope == "smoke" and planned_max_examples >= frozen_case_count:
        raise ValueError(
            "The prepared smoke cohort must be smaller than the frozen evaluation suite"
        )
    return plan


def _evaluation_outcome(
    *,
    complete_evaluation: bool,
    format_failure_count: int,
    report: Mapping[str, Any],
) -> tuple[str, bool]:
    all_scored_cases_passed = (
        report.get("evidenceComplete") is True
        and report.get("criticalFailureCount") == 0
        and type(report.get("caseCount")) is int
        and report["caseCount"] > 0
        and report.get("passedCaseCount") == report["caseCount"]
    )
    quality_gate_passed = (
        complete_evaluation
        and format_failure_count == 0
        and all_scored_cases_passed
    )
    if complete_evaluation:
        status = (
            "format_failed"
            if format_failure_count
            else "quality_gate_passed"
            if quality_gate_passed
            else "quality_gate_failed"
        )
    else:
        status = (
            "smoke_complete"
            if format_failure_count == 0 and all_scored_cases_passed
            else "smoke_failed"
        )
    return status, quality_gate_passed


def _evaluation_exit_code(*, status: str, format_failure_count: int) -> int:
    if format_failure_count:
        return 2
    if status in {"quality_gate_failed", "smoke_failed"}:
        return 3
    return 0


def _evaluation_report_scope_valid(
    report: Mapping[str, Any],
    *,
    selected_case_count: int,
    frozen_case_count: int,
) -> bool:
    complete_evaluation = selected_case_count == frozen_case_count
    return (
        selected_case_count > 0
        and frozen_case_count >= selected_case_count
        and report.get("variantLineageBound") is True
        and report.get("frozenCaseCount") == frozen_case_count
        and report.get("caseCount") == selected_case_count
        and report.get("completeEvaluation") is complete_evaluation
        and report.get("promotionEvidenceBound") is complete_evaluation
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_duplicate_json_keys(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"Duplicate JSON key is not allowed: {key}")
        value[key] = item
    return value


def _reject_nonfinite_json_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON number is not allowed: {value}")


def _parse_finite_json_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        _reject_nonfinite_json_constant(value)
    return parsed


def _strict_json_loads(value: str) -> Any:
    return json.loads(
        value,
        object_pairs_hook=_reject_duplicate_json_keys,
        parse_constant=_reject_nonfinite_json_constant,
        parse_float=_parse_finite_json_float,
    )


def _read_file_snapshot(path: Path, *, label: str) -> tuple[bytes, str]:
    try:
        with path.open("rb") as handle:
            file_stat = os.fstat(handle.fileno())
            if not stat.S_ISREG(file_stat.st_mode):
                raise ValueError(f"{label} must be a regular file: {path}")
            payload = handle.read()
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"{label} not found: {path}") from exc
    except OSError as exc:
        raise ValueError(f"{label} could not be read: {path}") from exc
    return payload, hashlib.sha256(payload).hexdigest()


def _load_json_object_bytes(
    payload: bytes,
    *,
    path: Path,
    label: str,
) -> dict[str, Any]:
    try:
        value = _strict_json_loads(payload.decode("utf-8"))
    except (UnicodeError, ValueError, RecursionError) as exc:
        raise ValueError(f"{label} is not valid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return value


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    payload, _ = _read_file_snapshot(path, label=label)
    return _load_json_object_bytes(payload, path=path, label=label)


def _load_evaluation_config_snapshot(path: Path) -> tuple[dict[str, Any], str]:
    payload, file_sha256 = _read_file_snapshot(path, label="Evaluation config")
    strict_config = _load_json_object_bytes(
        payload,
        path=path,
        label="Evaluation config",
    )
    return _validate_export_config(strict_config, path=path), file_sha256


def load_evaluation_config(path: Path) -> dict[str, Any]:
    config, _ = _load_evaluation_config_snapshot(path)
    return config


def _load_evaluation_module() -> ModuleType:
    repo_root = Path(__file__).resolve().parents[3]
    crawler_root = repo_root / "tools" / "lumen_manifest_crawler"
    if crawler_root.is_dir() and str(crawler_root) not in sys.path:
        sys.path.insert(0, str(crawler_root))
    from lumen_manifest_crawler.dataset import adapter_evaluation

    return adapter_evaluation


def _validate_prompt_messages(value: Any, *, path: Path, line_number: int) -> None:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{path}:{line_number} messages must be a non-empty list")
    expected_role = "user"
    for index, message in enumerate(value):
        if not isinstance(message, dict) or set(message) != {"role", "content"}:
            raise ValueError(
                f"{path}:{line_number} messages[{index}] must contain only role and content"
            )
        role = message.get("role")
        content = message.get("content")
        if role not in _MESSAGE_ROLES or not isinstance(content, str) or not content.strip():
            raise ValueError(f"{path}:{line_number} messages[{index}] is invalid")
        if index == 0 and role == "system":
            continue
        if role != expected_role:
            raise ValueError(
                f"{path}:{line_number} prompt roles must alternate after an optional leading system message"
            )
        expected_role = "assistant" if expected_role == "user" else "user"
    if value[-1].get("role") != "user":
        raise ValueError(f"{path}:{line_number} prompt must end with a user message")


def load_evaluation_records(
    path: Path,
    *,
    agent: str,
    evaluation_module: ModuleType,
) -> tuple[list[dict[str, Any]], str]:
    if not path.is_file():
        raise FileNotFoundError(f"Frozen evaluation JSONL not found: {path}")
    records: list[dict[str, Any]] = []
    seen_eval_ids: set[str] = set()
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not raw_line.strip():
            continue
        try:
            raw_record = _strict_json_loads(raw_line)
        except ValueError as exc:
            raise ValueError(f"{path}:{line_number} is not valid JSON") from exc
        if not isinstance(raw_record, dict):
            raise ValueError(f"{path}:{line_number} must be a JSON object")
        _validate_prompt_messages(
            raw_record.get("messages"),
            path=path,
            line_number=line_number,
        )
        record = evaluation_module.upgrade_evaluation_record(raw_record)
        eval_id = record.get("evalID")
        record_agent = str((record.get("metadata") or {}).get("agent") or "").strip().lower()
        if not isinstance(eval_id, str) or not eval_id:
            raise ValueError(f"{path}:{line_number} has no stable evalID")
        if eval_id in seen_eval_ids:
            raise ValueError(f"{path}:{line_number} duplicates evalID {eval_id}")
        if record_agent != agent:
            raise ValueError(
                f"{path}:{line_number} belongs to agent {record_agent or '<missing>'}, expected {agent}"
            )
        metrics = record.get("metrics")
        if not isinstance(metrics, list) or not metrics or any(
            not isinstance(metric, dict) for metric in metrics
        ):
            raise ValueError(f"{path}:{line_number} has an invalid metric contract")
        seen_eval_ids.add(eval_id)
        records.append(record)
    if not records:
        raise ValueError(f"Frozen evaluation JSONL is empty: {path}")
    return records, evaluation_module.canonical_sha256(records)


def _behavior_contract_from_manifest(
    manifest: dict[str, Any],
) -> tuple[dict[str, Any], set[str], str]:
    raw_tools = manifest.get("tools")
    if not isinstance(raw_tools, list) or not raw_tools:
        raise ValueError("Behavior manifest tools must be a non-empty list")
    tool_contracts: dict[str, Any] = {}
    for index, tool in enumerate(raw_tools):
        if not isinstance(tool, dict):
            raise ValueError(f"Behavior manifest tools[{index}] must be an object")
        tool_id = tool.get("id")
        arguments = tool.get("arguments")
        if (
            not isinstance(tool_id, str)
            or _TOOL_ID_PATTERN.fullmatch(tool_id) is None
            or tool_id in tool_contracts
            or not isinstance(arguments, list)
        ):
            raise ValueError(f"Behavior manifest tools[{index}] has an invalid or duplicate ID")
        argument_names: set[str] = set()
        for argument_index, argument in enumerate(arguments):
            if not isinstance(argument, dict):
                raise ValueError(
                    f"Behavior manifest {tool_id} arguments[{argument_index}] must be an object"
                )
            name = argument.get("name")
            declared_type = argument.get("type")
            required = argument.get("required")
            allowed_values = argument.get("allowedValues")
            if (
                not isinstance(name, str)
                or not name
                or name in argument_names
                or not isinstance(declared_type, str)
                or not declared_type
                or type(required) is not bool
                or (allowed_values is not None and not isinstance(allowed_values, list))
            ):
                raise ValueError(
                    f"Behavior manifest {tool_id} arguments[{argument_index}] is invalid"
                )
            argument_names.add(name)
        tool_contracts[tool_id] = tool

    routed_intents_by_tool: dict[str, list[str]] = {
        tool_id: [] for tool_id in tool_contracts
    }
    raw_routing_matrix = manifest.get("routingMatrix", [])
    if not isinstance(raw_routing_matrix, list):
        raise ValueError("Behavior manifest routingMatrix must be a list")
    for index, entry in enumerate(
        sorted(
            raw_routing_matrix,
            key=lambda item: str(item.get("intent") or "")
            if isinstance(item, Mapping)
            else "",
        )
    ):
        if not isinstance(entry, Mapping):
            raise ValueError(
                f"Behavior manifest routingMatrix[{index}] must be an object"
            )
        intent_id = entry.get("intent")
        allowed_tools = entry.get("allowedTools")
        if (
            not isinstance(intent_id, str)
            or not intent_id
            or not isinstance(allowed_tools, list)
            or any(not isinstance(tool_id, str) for tool_id in allowed_tools)
        ):
            raise ValueError(
                f"Behavior manifest routingMatrix[{index}] is invalid"
            )
        for tool_id in allowed_tools:
            if tool_id in routed_intents_by_tool:
                routed_intents_by_tool[tool_id].append(intent_id)

    raw_intents = manifest.get("intents", [])
    if not isinstance(raw_intents, list):
        raise ValueError("Behavior manifest intents must be a list")
    for index, intent in enumerate(
        sorted(
            raw_intents,
            key=lambda item: str(item.get("id") or "")
            if isinstance(item, Mapping)
            else "",
        )
    ):
        if not isinstance(intent, Mapping):
            raise ValueError(f"Behavior manifest intents[{index}] must be an object")
        intent_id = intent.get("id")
        allowed_tool_ids = intent.get("allowedToolIDs")
        if (
            not isinstance(intent_id, str)
            or not intent_id
            or not isinstance(allowed_tool_ids, list)
            or any(not isinstance(tool_id, str) for tool_id in allowed_tool_ids)
        ):
            raise ValueError(f"Behavior manifest intents[{index}] is invalid")
        for tool_id in allowed_tool_ids:
            if tool_id in routed_intents_by_tool:
                routed_intents_by_tool[tool_id].append(intent_id)

    for tool_id, tool in list(tool_contracts.items()):
        ordered_intents = list(dict.fromkeys(routed_intents_by_tool[tool_id]))
        default_intent = ordered_intents[0] if ordered_intents else "tool"
        allowed_intents = [
            default_intent,
            *sorted(set(ordered_intents) - {default_intent}),
        ]
        tool_contracts[tool_id] = {
            **tool,
            "defaultIntent": default_intent,
            "allowedIntents": allowed_intents,
        }

    fleet = manifest.get("fleet")
    raw_slots = fleet.get("slots") if isinstance(fleet, dict) else None
    if not isinstance(raw_slots, list) or not raw_slots:
        raise ValueError("Behavior manifest fleet.slots must be a non-empty list")
    allowed_slots: set[str] = set()
    for index, slot in enumerate(raw_slots):
        slot_id = slot.get("id") if isinstance(slot, dict) else None
        if (
            not isinstance(slot_id, str)
            or not slot_id
            or slot_id in allowed_slots
        ):
            raise ValueError(f"Behavior manifest fleet.slots[{index}] is invalid")
        allowed_slots.add(slot_id)
    return tool_contracts, allowed_slots, _canonical_sha256(manifest)


def _load_behavior_contract_snapshot(
    path: Path,
) -> tuple[dict[str, Any], set[str], str, str]:
    payload, file_sha256 = _read_file_snapshot(path, label="Behavior manifest")
    manifest = _load_json_object_bytes(
        payload,
        path=path,
        label="Behavior manifest",
    )
    tool_contracts, allowed_slots, canonical_sha256 = (
        _behavior_contract_from_manifest(manifest)
    )
    return tool_contracts, allowed_slots, canonical_sha256, file_sha256


def load_behavior_contract(path: Path) -> tuple[dict[str, Any], set[str], str]:
    tool_contracts, allowed_slots, canonical_sha256, _ = (
        _load_behavior_contract_snapshot(path)
    )
    return tool_contracts, allowed_slots, canonical_sha256


def validate_scoring_contracts(
    records: Sequence[Mapping[str, Any]],
    *,
    tool_contracts: Mapping[str, Any],
    allowed_slots: set[str],
) -> None:
    referenced_tools: set[str] = set()
    referenced_slots: set[str] = set()
    for record in records:
        for metric in record["metrics"]:
            expected_tool = metric.get("expectedToolID")
            if isinstance(expected_tool, str):
                referenced_tools.add(expected_tool)
            raw_allowed_tools = metric.get("allowedToolIDs")
            if isinstance(raw_allowed_tools, list):
                referenced_tools.update(
                    value for value in raw_allowed_tools if isinstance(value, str)
                )
            candidates: list[Mapping[str, Any]] = [metric]
            if isinstance(metric.get("contract"), Mapping):
                candidates.append(metric["contract"])
            for candidate in candidates:
                for key in (
                    "expectedSlot",
                    "expectedAggregationOwnerSlotID",
                ):
                    value = candidate.get(key)
                    if isinstance(value, str):
                        referenced_slots.add(value)
                for key in (
                    "allowedSlots",
                    "knownSlotIDs",
                    "expectedDelegatedSlotIDs",
                ):
                    value = candidate.get(key)
                    if isinstance(value, list):
                        referenced_slots.update(
                            item for item in value if isinstance(item, str)
                        )
    missing_tools = sorted(referenced_tools - set(tool_contracts))
    missing_slots = sorted(referenced_slots - allowed_slots)
    if missing_tools:
        raise ValueError(
            "Behavior manifest is missing evaluation tool contracts: "
            + ", ".join(missing_tools)
        )
    if missing_slots:
        raise ValueError(
            "Behavior manifest is missing evaluation fleet slots: "
            + ", ".join(missing_slots)
        )


def load_finalized_manifest(
    path: Path,
    *,
    cfg: Mapping[str, Any],
    evaluation_sha256: str,
    evaluation_module: ModuleType,
) -> dict[str, Any]:
    finalized = _load_json_object(path, label="Finalized variant manifest")
    expected_sha = finalized.get("variantManifestSHA256")
    unsigned = dict(finalized)
    unsigned.pop("variantManifestSHA256", None)
    if (
        not isinstance(expected_sha, str)
        or _SHA256_PATTERN.fullmatch(expected_sha) is None
        or evaluation_module.canonical_sha256(unsigned) != expected_sha
    ):
        raise ValueError("Finalized variant manifest integrity check failed")
    agent = str(cfg.get("agent") or "").strip().lower()
    variant = cfg.get("variant")
    source_sha = cfg.get("variantManifestSHA256")
    artifact = finalized.get("artifact")
    if (
        finalized.get("agent") != agent
        or finalized.get("variant") != variant
        or finalized.get("sourceVariantManifestSHA256") != source_sha
        or finalized.get("frozenEvaluationSHA256") != evaluation_sha256
        or not isinstance(artifact, dict)
        or artifact.get("status") != "trained"
        or _SHA256_PATTERN.fullmatch(str(artifact.get("adapterSHA256") or "")) is None
    ):
        raise ValueError(
            "Finalized variant manifest is not bound to the selected config, adapter, and frozen evaluation"
        )
    validator = getattr(evaluation_module, "_valid_variant_manifest", None)
    if validator is None or not validator(
        finalized,
        agent=agent,
        expected_variant=variant,
        require_trained_artifact=True,
    ):
        raise ValueError("Finalized variant manifest failed the controlled lineage contract")
    return finalized


def _model_device(model: Any) -> Any:
    try:
        return next(model.parameters()).device
    except (AttributeError, StopIteration, TypeError) as exc:
        raise RuntimeError("Unable to resolve the adapter model device") from exc


def _move_model_inputs(value: Any, device: Any) -> Any:
    if hasattr(value, "to"):
        return value.to(device)
    return value


def generate_completion(
    model: Any,
    tokenizer: Any,
    messages: Sequence[Mapping[str, str]],
    *,
    max_seq_length: int,
    max_new_tokens: int,
    torch_module: ModuleType | Any | None = None,
) -> tuple[str, int, int, int]:
    try:
        encoded = apply_non_thinking_chat_template(
            tokenizer,
            list(messages),
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
    except (AttributeError, TypeError, ValueError) as exc:
        raise RuntimeError("Tokenizer could not render the frozen evaluation prompt") from exc
    if isinstance(encoded, Mapping):
        model_inputs = dict(encoded)
    else:
        model_inputs = {"input_ids": encoded}
    input_ids = model_inputs.get("input_ids")
    shape = getattr(input_ids, "shape", None)
    if shape is None or len(shape) != 2 or int(shape[0]) != 1:
        raise RuntimeError("Tokenizer must return one rank-two input_ids tensor")
    input_token_count = int(shape[-1])
    remaining_tokens = max_seq_length - input_token_count
    generation_budget = min(max_new_tokens, remaining_tokens)
    if input_token_count <= 0 or generation_budget <= 0:
        raise RuntimeError(
            "Frozen evaluation prompt consumes the configured maximum sequence length"
        )
    device = _model_device(model)
    moved_inputs = {
        key: _move_model_inputs(value, device)
        for key, value in model_inputs.items()
    }
    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": generation_budget,
        "do_sample": False,
        "num_beams": 1,
        "repetition_penalty": GENERATION_REPETITION_PENALTY,
        "use_cache": True,
    }
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if eos_token_id is not None:
        generation_kwargs["eos_token_id"] = eos_token_id
    if pad_token_id is not None or eos_token_id is not None:
        generation_kwargs["pad_token_id"] = (
            pad_token_id if pad_token_id is not None else eos_token_id
        )
    if torch_module is None:
        import torch as torch_module  # type: ignore[no-redef]

    with torch_module.inference_mode():
        generated = model.generate(**moved_inputs, **generation_kwargs)
    sequences = getattr(generated, "sequences", generated)
    output_shape = getattr(sequences, "shape", None)
    if output_shape is None or len(output_shape) != 2 or int(output_shape[0]) != 1:
        raise RuntimeError("Model generation must return one rank-two token sequence")
    output_token_count = int(output_shape[-1])
    if output_token_count < input_token_count:
        raise RuntimeError("Model generation returned fewer tokens than the input prompt")
    generated_token_count = output_token_count - input_token_count
    if generated_token_count > generation_budget:
        raise RuntimeError("Model generation exceeded the configured token budget")
    generated_ids = sequences[0][input_token_count:]
    completion = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    if not isinstance(completion, str):
        raise RuntimeError("Tokenizer decode did not return text")
    return (
        completion,
        input_token_count,
        generated_token_count,
        generation_budget,
    )


def _validate_output_mode_for_agent(agent: str, output_mode: Any) -> str:
    if not isinstance(output_mode, str) or output_mode not in SUPPORTED_OUTPUT_MODES:
        raise ValueError("Evaluation record has an invalid outputMode")
    if agent in {"cortex", "executor", "fleet", "rem"} and output_mode != "json":
        raise ValueError(f"{agent} evaluation outputMode must be json")
    if agent == "mouth" and output_mode != "text":
        raise ValueError("mouth evaluation outputMode must be text")
    if agent not in SUPPORTED_AGENTS:
        raise ValueError(f"Unsupported evaluation agent: {agent}")
    return output_mode


def _record_output_mode(record: Mapping[str, Any], *, agent: str) -> str:
    record_agent = str((record.get("metadata") or {}).get("agent") or "").strip().lower()
    if record_agent != agent:
        raise ValueError(
            f"Evaluation record belongs to agent {record_agent or '<missing>'}, expected {agent}"
        )
    return _validate_output_mode_for_agent(agent, record.get("outputMode"))


def normalize_candidate_output(
    agent: str,
    completion: str,
    *,
    output_mode: str,
    evaluation_module: ModuleType,
    tool_contracts: Mapping[str, Any] | None = None,
) -> tuple[Any, str, str | None]:
    resolved_output_mode = _validate_output_mode_for_agent(agent, output_mode)
    if resolved_output_mode == "text":
        has_text = bool(completion.strip())
        return completion, "text" if has_text else "empty_text", (
            None if has_text else "empty_candidate_output"
        )
    parsed, json_error = evaluation_module._parse_candidate_json(completion)
    if json_error is not None:
        return completion, "invalid_json", json_error
    if not isinstance(parsed, dict):
        return completion, "invalid_json", "json_output_must_be_an_object"
    if agent == "cortex" and _contains_forbidden_cortex_route_field(parsed):
        return completion, "invalid_json", "forbidden_cortex_route_field"
    if agent == "cortex" and not tool_contracts:
        return parsed, "invalid_cortex_route", "cortex_route_manifest_contract_missing"
    if agent == "cortex":
        assert tool_contracts is not None
        route_error = _cortex_manifest_route_error(parsed, tool_contracts)
        if route_error is not None:
            return parsed, "invalid_cortex_route", route_error
    return parsed, "json_object", None


def _contains_forbidden_cortex_route_field(value: Any) -> bool:
    if isinstance(value, Mapping):
        return bool(CORTEX_FORBIDDEN_ROUTE_FIELDS.intersection(value)) or any(
            _contains_forbidden_cortex_route_field(child)
            for child in value.values()
        )
    if isinstance(value, list):
        return any(_contains_forbidden_cortex_route_field(child) for child in value)
    return False


def _cortex_manifest_route_error(
    route: Mapping[str, Any],
    tool_contracts: Mapping[str, Any],
) -> str | None:
    """Validate only candidate-caused manifest and route-protocol failures."""

    base_fields = set(_CORTEX_ROUTE_PREFIX_FIELDS + _CORTEX_ROUTE_SUFFIX_FIELDS)
    if (
        not isinstance(route.get("intent"), str)
        or not route["intent"].strip()
        or type(route.get("requiresApproval")) is not bool
        or not isinstance(route.get("nextModel"), str)
        or not route["nextModel"].strip()
        or not isinstance(route.get("reasoningSummary"), str)
        or not route["reasoningSummary"].strip()
    ):
        return "cortex_route_protocol_field_invalid"

    selected_tool_id = route.get("selectedToolID")
    if selected_tool_id is None:
        expected_key_order = (
            *_CORTEX_ROUTE_PREFIX_FIELDS,
            "status",
            *_CORTEX_ROUTE_SUFFIX_FIELDS,
        )
        if (
            set(route) != base_fields | {"status"}
            or route.get("requiresApproval") is not False
            or route.get("nextModel") != "mouth"
            or route.get("status") not in {"no_tool_route", "invalid_tool"}
        ):
            return "cortex_route_null_state_invalid"
        if tuple(route) != expected_key_order:
            return "cortex_route_key_order_invalid"
        if route.get("reasoningSummary") != (
            f"No manifest row applies to intent {route['intent']}."
        ):
            return "cortex_route_reasoning_summary_mismatch"
        return None
    if not isinstance(selected_tool_id, str) or not selected_tool_id:
        return "cortex_route_selected_tool_invalid"

    tool = tool_contracts.get(selected_tool_id)
    if not isinstance(tool, Mapping):
        return "cortex_route_tool_not_in_manifest"
    expected_approval = tool.get("requiresApproval")
    raw_arguments = tool.get("arguments")
    default_intent = tool.get("defaultIntent")
    allowed_intents = tool.get("allowedIntents")
    if (
        type(expected_approval) is not bool
        or not isinstance(raw_arguments, list)
        or not isinstance(default_intent, str)
        or not default_intent
        or not isinstance(allowed_intents, list)
        or not allowed_intents
        or any(not isinstance(intent, str) or not intent for intent in allowed_intents)
        or len(set(allowed_intents)) != len(allowed_intents)
        or allowed_intents[0] != default_intent
    ):
        return "cortex_route_manifest_contract_invalid"
    if route["intent"] not in allowed_intents:
        return "cortex_route_intent_not_in_manifest"
    required_arguments: list[str] = []
    for argument in raw_arguments:
        if not isinstance(argument, Mapping):
            return "cortex_route_manifest_contract_invalid"
        name = argument.get("name")
        required = argument.get("required")
        if not isinstance(name, str) or not name or type(required) is not bool:
            return "cortex_route_manifest_contract_invalid"
        if required:
            required_arguments.append(name)
    if route.get("requiresApproval") is not expected_approval:
        return "cortex_route_approval_mismatch"

    if route.get("status") == "needs_clarification":
        if route["intent"] != default_intent:
            return "cortex_route_intent_not_in_manifest"
        missing_arguments = route.get("missingArguments")
        clarification = route.get("clarification")
        if (
            set(route)
            != base_fields | {"status", "missingArguments", "clarification"}
            or not isinstance(missing_arguments, list)
            or not missing_arguments
            or any(not isinstance(value, str) for value in missing_arguments)
            or len(set(missing_arguments)) != len(missing_arguments)
            or missing_arguments
            != [
                argument
                for argument in required_arguments
                if argument in missing_arguments
            ]
            or route.get("nextModel") != "mouth"
            or not isinstance(clarification, str)
            or not clarification.strip().endswith("?")
        ):
            return "cortex_route_clarification_state_invalid"
        expected_key_order = (
            *_CORTEX_ROUTE_PREFIX_FIELDS,
            "status",
            "missingArguments",
            "clarification",
            *_CORTEX_ROUTE_SUFFIX_FIELDS,
        )
        if tuple(route) != expected_key_order:
            return "cortex_route_key_order_invalid"
        expected_summary = (
            f"Manifest row {selected_tool_id} is missing exactly this required subset: "
            f"{', '.join(missing_arguments)}."
        )
        if route.get("reasoningSummary") != expected_summary:
            return "cortex_route_reasoning_summary_mismatch"
        return None

    expected_next_model = "approval" if expected_approval else "executor"
    if set(route) == base_fields:
        if route.get("nextModel") != expected_next_model:
            return "cortex_route_selection_state_invalid"
        if tuple(route) != _CORTEX_ROUTE_PREFIX_FIELDS + _CORTEX_ROUTE_SUFFIX_FIELDS:
            return "cortex_route_key_order_invalid"
        expected_summary = (
            f"Manifest row {selected_tool_id} is selected for intent "
            f"{route['intent']} without actionStep."
        )
        if route.get("reasoningSummary") != expected_summary:
            return "cortex_route_reasoning_summary_mismatch"
        return None

    if "actionStep" in route and route["intent"] != default_intent:
        return "cortex_route_intent_not_in_manifest"
    action_step = route.get("actionStep")
    if (
        set(route) != base_fields | {"actionStep"}
        or route.get("nextModel") != expected_next_model
        or not isinstance(action_step, Mapping)
        or set(action_step) != {"type", "toolID", "mustPersistBeforeFinal"}
        or action_step.get("type") != "tool_call"
        or action_step.get("toolID") != selected_tool_id
        or action_step.get("mustPersistBeforeFinal") is not True
    ):
        return "cortex_route_action_state_invalid"
    expected_key_order = (
        *_CORTEX_ROUTE_PREFIX_FIELDS,
        "actionStep",
        *_CORTEX_ROUTE_SUFFIX_FIELDS,
    )
    if (
        tuple(route) != expected_key_order
        or tuple(action_step) != _CORTEX_ACTION_STEP_FIELDS
    ):
        return "cortex_route_key_order_invalid"
    expected_summary = (
        f"Manifest row {selected_tool_id} has no required values."
        if not required_arguments
        else (
            f"Manifest row {selected_tool_id} has all exact required names supplied: "
            f"{', '.join(required_arguments)}."
        )
    )
    if route.get("reasoningSummary") != expected_summary:
        return "cortex_route_reasoning_summary_mismatch"
    return None


def _structured_output_messages(
    agent: str,
    messages: Sequence[Mapping[str, str]],
    *,
    output_mode: str,
    tool_contracts: Mapping[str, Any] | None = None,
) -> list[dict[str, str]]:
    copied = [
        {"role": str(message["role"]), "content": str(message["content"])}
        for message in messages
    ]
    resolved_output_mode = _validate_output_mode_for_agent(agent, output_mode)
    if resolved_output_mode == "text":
        return canonical_non_thinking_messages(copied)
    if agent == "cortex" and not tool_contracts:
        raise ValueError("Cortex structured output requires manifest tool contracts")
    instructions = [STRUCTURED_OUTPUT_INSTRUCTION]
    if agent == "cortex":
        assert tool_contracts is not None
        instructions.extend(
            (
                CORTEX_ROUTE_INSTRUCTION,
                _cortex_tool_catalog_instruction(tool_contracts),
                CORTEX_ROUTE_DECISION_ENDCAP,
            )
        )
    contract = "\n\n".join(instructions)
    if copied and copied[0]["role"] == "system":
        existing = copied[0]["content"].rstrip()
        if existing == contract or existing.endswith("\n\n" + contract):
            return canonical_non_thinking_messages(copied)
        if (
            STRUCTURED_OUTPUT_INSTRUCTION in existing
            or CORTEX_TOOL_CATALOG_HEADER in existing
        ):
            raise ValueError(
                f"{agent} evaluation prompt contains a drifted structured-output contract"
            )
        copied[0]["content"] = existing + "\n\n" + contract
    else:
        copied.insert(
            0,
            {"role": "system", "content": contract},
        )
    return canonical_non_thinking_messages(copied)


def _cortex_tool_catalog_instruction(
    tool_contracts: Mapping[str, Any],
) -> str:
    lines = [CORTEX_TOOL_CATALOG_HEADER]
    for tool_id, raw_tool in sorted(tool_contracts.items()):
        if not isinstance(tool_id, str) or not isinstance(raw_tool, Mapping):
            raise ValueError("Cortex routing context contains an invalid tool contract")
        display_name = raw_tool.get("displayName")
        description = raw_tool.get("description")
        requires_approval = raw_tool.get("requiresApproval")
        raw_arguments = raw_tool.get("arguments")
        default_intent = raw_tool.get("defaultIntent")
        allowed_intents = raw_tool.get("allowedIntents")
        if (
            not isinstance(display_name, str)
            or not display_name.strip()
            or not isinstance(description, str)
            or not description.strip()
            or type(requires_approval) is not bool
            or not isinstance(raw_arguments, list)
            or not isinstance(default_intent, str)
            or not default_intent
            or not isinstance(allowed_intents, list)
            or not allowed_intents
            or any(
                not isinstance(intent, str)
                or not intent
                or any(separator in intent for separator in (",", "\t", "\r", "\n"))
                for intent in allowed_intents
            )
            or len(set(allowed_intents)) != len(allowed_intents)
            or allowed_intents[0] != default_intent
        ):
            raise ValueError(f"Cortex routing context has an invalid contract for {tool_id}")
        required_arguments: list[str] = []
        for argument in raw_arguments:
            if not isinstance(argument, Mapping):
                raise ValueError(f"Cortex routing context has invalid arguments for {tool_id}")
            name = argument.get("name")
            required = argument.get("required")
            if not isinstance(name, str) or not name or type(required) is not bool:
                raise ValueError(f"Cortex routing context has invalid arguments for {tool_id}")
            if required:
                required_arguments.append(name)
        compact_description = " ".join(description.split())
        compact_description = re.sub(
            r"\s+Args:\s.*$",
            "",
            compact_description,
        ).strip()
        if not compact_description:
            raise ValueError(
                f"Cortex routing context has an invalid description for {tool_id}"
            )
        lines.append(
            f"{tool_id}\t{display_name.strip()}\t"
            f"{default_intent}\t{','.join(allowed_intents)}\t"
            f"{','.join(required_arguments) or '-'}\t"
            f"{'1' if requires_approval else '0'}\t{compact_description}"
        )
    return "\n".join(lines)


def _structured_output_contract_sha256(
    agent: str,
    *,
    output_mode: str,
    tool_contracts: Mapping[str, Any] | None = None,
) -> str | None:
    resolved_output_mode = _validate_output_mode_for_agent(agent, output_mode)
    if resolved_output_mode == "text":
        return None
    messages = _structured_output_messages(
        agent,
        [{"role": "user", "content": "contract hash sentinel"}],
        output_mode=resolved_output_mode,
        tool_contracts=tool_contracts,
    )
    return _canonical_sha256(messages[0]["content"])


def _strict_json_retry_instruction(agent: str) -> str:
    if agent not in SUPPORTED_AGENTS:
        raise ValueError(f"Unsupported evaluation agent: {agent}")
    return (
        STRICT_JSON_RETRY_INSTRUCTION
        if agent == "cortex"
        else GENERIC_STRICT_JSON_RETRY_INSTRUCTION
    )


def _strict_json_retry_contract_sha256(
    agent: str,
    *,
    output_mode: str,
) -> str | None:
    resolved_output_mode = _validate_output_mode_for_agent(agent, output_mode)
    if resolved_output_mode == "text":
        return None
    return hashlib.sha256(
        _strict_json_retry_instruction(agent).encode("utf-8")
    ).hexdigest()


def _evaluation_output_mode_contract(
    records: Sequence[Mapping[str, Any]],
    *,
    agent: str,
    tool_contracts: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    seen_eval_ids: set[str] = set()
    for record in records:
        eval_id = record.get("evalID")
        if not isinstance(eval_id, str) or not eval_id or eval_id in seen_eval_ids:
            raise ValueError("Output-mode contract requires unique stable evalID values")
        seen_eval_ids.add(eval_id)
        output_mode = _record_output_mode(record, agent=agent)
        json_mode = output_mode == "json"
        entries.append(
            {
                "evalID": eval_id,
                "outputMode": output_mode,
                "structuredOutputContractSHA256": _structured_output_contract_sha256(
                    agent,
                    output_mode=output_mode,
                    tool_contracts=tool_contracts,
                ),
                "strictJSONRetryEligible": json_mode,
                "strictJSONMaxAttempts": STRICT_JSON_MAX_ATTEMPTS if json_mode else 1,
                "strictJSONRetryContractSHA256": _strict_json_retry_contract_sha256(
                    agent,
                    output_mode=output_mode,
                ),
            }
        )
    if not entries:
        raise ValueError("Output-mode contract requires evaluation records")
    contract = {
        "schemaVersion": OUTPUT_MODE_CONTRACT_VERSION,
        "structuredOutputContractVersion": STRUCTURED_OUTPUT_CONTRACT_VERSION,
        "strictJSONRetryContractVersion": STRICT_JSON_RETRY_CONTRACT_VERSION,
        "records": entries,
    }
    return {
        **contract,
        "outputModeContractSHA256": _canonical_sha256(contract),
    }


def _trusted_cortex_retry_manifest_row(
    failed_candidate: Any,
    tool_contracts: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """Resolve retry facts only through an exact manifest-key lookup."""

    if not isinstance(failed_candidate, Mapping) or not tool_contracts:
        return None
    selected_tool_id = failed_candidate.get("selectedToolID")
    if not isinstance(selected_tool_id, str) or not selected_tool_id:
        return None
    raw_tool = tool_contracts.get(selected_tool_id)
    if not isinstance(raw_tool, Mapping):
        return None

    default_intent = raw_tool.get("defaultIntent")
    allowed_intents = raw_tool.get("allowedIntents")
    requires_approval = raw_tool.get("requiresApproval")
    raw_arguments = raw_tool.get("arguments")
    if (
        not isinstance(default_intent, str)
        or not default_intent
        or not isinstance(allowed_intents, list)
        or not allowed_intents
        or allowed_intents[0] != default_intent
        or any(not isinstance(intent, str) or not intent for intent in allowed_intents)
        or type(requires_approval) is not bool
        or not isinstance(raw_arguments, list)
    ):
        return None

    required_arguments: list[str] = []
    argument_names: set[str] = set()
    for argument in raw_arguments:
        if not isinstance(argument, Mapping):
            return None
        name = argument.get("name")
        required = argument.get("required")
        if (
            not isinstance(name, str)
            or not name
            or name in argument_names
            or type(required) is not bool
        ):
            return None
        argument_names.add(name)
        if required:
            required_arguments.append(name)

    return {
        "selectedToolID": selected_tool_id,
        "defaultIntent": default_intent,
        "requiredArguments": required_arguments,
        "requiresApproval": requires_approval,
    }


def _cortex_retry_transition_error(
    failed_candidate: Any,
    validation_error: str | None,
    retry_candidate: Any,
    tool_contracts: Mapping[str, Any] | None,
    retry_locked_row: Mapping[str, Any] | None = None,
) -> str | None:
    """Reject a valid retry that escapes the first attempt's trusted route scope."""

    if not isinstance(retry_candidate, Mapping) or not tool_contracts:
        return None

    trusted_row = _trusted_cortex_retry_manifest_row(
        retry_locked_row,
        tool_contracts,
    ) or _trusted_cortex_retry_manifest_row(failed_candidate, tool_contracts)
    if trusted_row is not None:
        if retry_candidate.get("selectedToolID") != trusted_row["selectedToolID"]:
            return "cortex_route_retry_tool_drift"
        return None

    if (
        validation_error != "cortex_route_tool_not_in_manifest"
        or not isinstance(failed_candidate, Mapping)
    ):
        return None
    failed_intent = failed_candidate.get("intent")
    if not isinstance(failed_intent, str) or not failed_intent:
        return None

    manifest_intents: set[str] = set()
    for raw_tool in tool_contracts.values():
        if not isinstance(raw_tool, Mapping):
            continue
        allowed_intents = raw_tool.get("allowedIntents")
        if not isinstance(allowed_intents, list):
            continue
        manifest_intents.update(
            intent
            for intent in allowed_intents
            if isinstance(intent, str) and intent
        )
    if (
        failed_intent in manifest_intents
        and retry_candidate.get("intent") != failed_intent
    ):
        return "cortex_route_retry_intent_drift"
    return None


def _trusted_cortex_retry_row_instruction(
    failed_candidate: Any,
    tool_contracts: Mapping[str, Any] | None,
) -> str:
    row = _trusted_cortex_retry_manifest_row(failed_candidate, tool_contracts)
    if row is None:
        return ""
    instruction = (
        " Trusted selected manifest row, derived only by exact selectedToolID lookup: "
        + json.dumps(row, ensure_ascii=False, separators=(",", ":"))
        + ". Copy defaultIntent, requiredArguments, and requiresApproval only from "
        "this trusted row. Lock to this row and do not borrow fields from any other "
        "catalog row or from the failed output."
    )
    if not row["requiredArguments"]:
        instruction += (
            " requiredArguments is empty: emit actionStep and do not emit status, "
            "missingArguments, or clarification; set nextModel to approval when "
            "requiresApproval is true, otherwise executor."
        )
    return instruction


def _quoted_manifest_tool_candidate(
    messages: Sequence[Mapping[str, str]],
    tool_contracts: Mapping[str, Any] | None,
) -> dict[str, str] | None:
    """Recover one explicitly quoted manifest ID without trusting malformed JSON."""

    if not messages or not tool_contracts:
        return None
    user_content = str(messages[-1].get("content", ""))
    candidates = [
        tool_id
        for tool_id in sorted(tool_contracts)
        if f"`{tool_id}`" in user_content
        or json.dumps(tool_id, ensure_ascii=False) in user_content
    ]
    if len(candidates) != 1:
        return None
    return {"selectedToolID": candidates[0]}


def _cortex_retry_locked_manifest_row(
    messages: Sequence[Mapping[str, str]],
    failed_candidate: Any,
    tool_contracts: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """Return the exact row the retry prompt will bind, if one is trustworthy."""

    trusted_row = _trusted_cortex_retry_manifest_row(
        failed_candidate,
        tool_contracts,
    )
    if trusted_row is not None:
        return trusted_row
    quoted_candidate = _quoted_manifest_tool_candidate(messages, tool_contracts)
    return _trusted_cortex_retry_manifest_row(quoted_candidate, tool_contracts)


def _strict_json_retry_messages(
    agent: str,
    messages: Sequence[Mapping[str, str]],
    *,
    validation_error: str | None = None,
    failed_candidate: Any = None,
    tool_contracts: Mapping[str, Any] | None = None,
) -> list[dict[str, str]]:
    _validate_output_mode_for_agent(agent, "json")
    copied = [
        {"role": str(message["role"]), "content": str(message["content"])}
        for message in messages
    ]
    if not copied or copied[-1]["role"] != "user":
        raise RuntimeError("Strict JSON retry requires a prompt ending in a user message")
    retry_instruction = _strict_json_retry_instruction(agent)
    if validation_error is not None:
        if not re.fullmatch(r"[a-z][a-z0-9_]{0,127}", validation_error):
            raise ValueError("Strict JSON retry received an invalid failure code")
        retry_instruction += (
            " Validation failure code: "
            + validation_error
            + ". Use that code only to re-check the response contract; do not "
            "invent missing user values."
        )
        if agent == "cortex":
            retry_instruction += " " + _CORTEX_RETRY_GUIDANCE_BY_FAILURE_CODE.get(
                validation_error,
                "Retry repair: re-read the selected manifest row and emit the exact "
                "contracted route state.",
            )
    if agent == "cortex":
        trusted_candidate = _cortex_retry_locked_manifest_row(
            copied,
            failed_candidate,
            tool_contracts,
        )
        retry_instruction += _trusted_cortex_retry_row_instruction(
            trusted_candidate,
            tool_contracts,
        )
    copied[-1]["content"] = (
        strip_terminal_non_thinking_directive(copied[-1]["content"])
        + "\n\n"
        + retry_instruction
    )
    return canonical_non_thinking_messages(copied)


def _generation_attempt_record(
    *,
    attempt_index: int,
    prompt_kind: str,
    messages: Sequence[Mapping[str, str]],
    completion: str,
    output_kind: str,
    format_error: str | None,
    input_token_count: int,
    generated_token_count: int,
    generation_token_budget: int,
) -> dict[str, Any]:
    attempt = {
        "schemaVersion": GENERATION_ATTEMPT_SCHEMA_VERSION,
        "attemptIndex": attempt_index,
        "promptKind": prompt_kind,
        "promptSHA256": _canonical_sha256(list(messages)),
        "rawOutput": completion,
        "outputKind": output_kind,
        "formatError": format_error,
        "inputTokenCount": input_token_count,
        "generatedTokenCount": generated_token_count,
        "generationTokenBudget": generation_token_budget,
        "hitTokenBudget": generated_token_count >= generation_token_budget,
    }
    attempt["generationAttemptSHA256"] = _canonical_sha256(attempt)
    return attempt


def evaluate_records(
    records: Sequence[Mapping[str, Any]],
    *,
    agent: str,
    model: Any,
    tokenizer: Any,
    max_seq_length: int,
    max_new_tokens: int,
    evaluation_module: ModuleType,
    tool_contracts: Mapping[str, Any] | None = None,
    torch_module: ModuleType | Any | None = None,
    on_case_completed: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], int, int, int]:
    outputs: dict[str, Any] = {}
    output_rows: list[dict[str, Any]] = []
    format_failure_count = 0
    initial_format_failure_count = 0
    format_recovery_count = 0
    for index, raw_record in enumerate(records, start=1):
        record = evaluation_module.upgrade_evaluation_record(raw_record)
        output_mode = _record_output_mode(record, agent=agent)
        json_mode = output_mode == "json"
        prompt_messages = _structured_output_messages(
            agent,
            record["messages"],
            output_mode=output_mode,
            tool_contracts=tool_contracts,
        )
        attempt_limit = STRICT_JSON_MAX_ATTEMPTS if json_mode else 1
        generation_attempts: list[dict[str, Any]] = []
        output: Any = ""
        output_kind = "empty_text"
        format_error: str | None = "empty_candidate_output"
        input_tokens = 0
        generated_tokens = 0
        generation_token_budget = 0
        first_attempt_output: Any = None
        first_attempt_error: str | None = None
        cortex_retry_locked_row: dict[str, Any] | None = None
        for attempt_index in range(1, attempt_limit + 1):
            (
                completion,
                input_tokens,
                generated_tokens,
                generation_token_budget,
            ) = generate_completion(
                model,
                tokenizer,
                prompt_messages,
                max_seq_length=max_seq_length,
                max_new_tokens=max_new_tokens,
                torch_module=torch_module,
            )
            output, output_kind, format_error = normalize_candidate_output(
                agent,
                completion,
                output_mode=output_mode,
                evaluation_module=evaluation_module,
                tool_contracts=tool_contracts,
            )
            if attempt_index == 2 and agent == "cortex" and format_error is None:
                transition_error = _cortex_retry_transition_error(
                    first_attempt_output,
                    first_attempt_error,
                    output,
                    tool_contracts,
                    cortex_retry_locked_row,
                )
                if transition_error is not None:
                    output_kind = "invalid_cortex_route"
                    format_error = transition_error
            generation_attempts.append(
                _generation_attempt_record(
                    attempt_index=attempt_index,
                    prompt_kind=(
                        "frozen_evaluation"
                        if attempt_index == 1
                        else "strict_json_retry"
                    ),
                    messages=prompt_messages,
                    completion=completion,
                    output_kind=output_kind,
                    format_error=format_error,
                    input_token_count=input_tokens,
                    generated_token_count=generated_tokens,
                    generation_token_budget=generation_token_budget,
                )
            )
            if attempt_index == 1:
                first_attempt_output = output
                first_attempt_error = format_error
            if format_error is None:
                break
            if attempt_index == 1 and json_mode:
                initial_format_failure_count += 1
                if agent == "cortex":
                    cortex_retry_locked_row = _cortex_retry_locked_manifest_row(
                        prompt_messages,
                        output,
                        tool_contracts,
                    )
                prompt_messages = _strict_json_retry_messages(
                    agent,
                    prompt_messages,
                    validation_error=format_error,
                    failed_candidate=output,
                    tool_contracts=tool_contracts,
                )

        if (
            len(generation_attempts) == STRICT_JSON_MAX_ATTEMPTS
            and generation_attempts[0]["formatError"] is not None
            and format_error is None
        ):
            format_recovery_count += 1
        if format_error is not None:
            if len(generation_attempts) == 1:
                initial_format_failure_count += 1
            format_failure_count += 1
        eval_id = str(record["evalID"])
        outputs[eval_id] = output
        row = {
            "schemaVersion": CANDIDATE_OUTPUT_SCHEMA_VERSION,
            "evalID": eval_id,
            "agent": agent,
            "outputMode": output_mode,
            "output": output,
            "outputKind": output_kind,
            "formatError": format_error,
            "inputTokenCount": input_tokens,
            "generatedTokenCount": generated_tokens,
            "selectedAttemptIndex": len(generation_attempts),
            "generationAttempts": generation_attempts,
        }
        row["candidateRecordSHA256"] = _canonical_sha256(row)
        output_rows.append(row)
        if on_case_completed is not None:
            # The callback is deliberately synchronous. A caller that journals
            # the row must durably commit it before this loop advances to the
            # next selected evaluation case.
            on_case_completed(row)
        print(
            f"[{agent}] evaluated {index}/{len(records)} {eval_id} ({output_kind})",
            flush=True,
        )
    return (
        outputs,
        output_rows,
        format_failure_count,
        initial_format_failure_count,
        format_recovery_count,
    )


def load_inference_model(
    cfg: Mapping[str, Any],
    *,
    adapter_dir: Path,
) -> tuple[Any, Any, dict[str, Any]]:
    if cfg.get("load_in_4bit") is not True:
        raise ValueError("Evaluation requires the controlled load_in_4bit=true config")
    _require_unsloth_before_transformers()
    try:
        from unsloth import FastLanguageModel  # type: ignore
        from peft import PeftModel  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Evaluation requires the pinned Unsloth, PEFT, and PyTorch environment"
        ) from exc
    # Seed only after Unsloth has patched Transformers, but before model loading.
    _seed_everything(int(cfg["seed"]))
    (
        expected_runtime_tokenizer,
        runtime_tokenizer_snapshot_path,
        runtime_tokenizer_snapshot_verification,
    ) = _load_verified_runtime_tokenizer_source(cfg)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(runtime_tokenizer_snapshot_path),
        revision=cfg["baseModelRevision"],
        tokenizer_name=str(runtime_tokenizer_snapshot_path),
        max_seq_length=int(cfg["max_seq_length"]),
        dtype=_controlled_torch_dtype(cfg),
        load_in_4bit=True,
        local_files_only=True,
        trust_remote_code=False,
        use_exact_model_name=True,
    )
    runtime_model_binding = _verify_runtime_model_binding(
        cfg,
        runtime_model=model,
        snapshot_path=runtime_tokenizer_snapshot_path,
        snapshot_verification=runtime_tokenizer_snapshot_verification,
    )
    runtime_tokenizer_binding = _verify_runtime_tokenizer_binding(
        cfg,
        expected_tokenizer=expected_runtime_tokenizer,
        runtime_tokenizer=tokenizer,
        snapshot_path=runtime_tokenizer_snapshot_path,
        snapshot_verification=runtime_tokenizer_snapshot_verification,
    )
    runtime_tokenizer_evidence = _runtime_tokenizer_evidence(
        cfg,
        snapshot_path=runtime_tokenizer_snapshot_path,
        snapshot_verification=runtime_tokenizer_snapshot_verification,
        runtime_model_binding=runtime_model_binding,
        runtime_binding=runtime_tokenizer_binding,
    )
    model = PeftModel.from_pretrained(
        model,
        str(adapter_dir),
        is_trainable=False,
    )
    FastLanguageModel.for_inference(model)
    model.eval()
    return model, tokenizer, runtime_tokenizer_evidence


def _atomic_write_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_descriptor = os.open(
            path.parent,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return (
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n"
            for row in rows
        )
    ).encode("utf-8")


def _validated_generation_attempts(
    row: Mapping[str, Any],
    *,
    agent: str,
    output_mode: str,
    evaluation_module: ModuleType,
    tool_contracts: Mapping[str, Any] | None,
    cortex_retry_locked_row: Mapping[str, Any] | None,
    expected_prompt_sha256: Sequence[str],
    path: Path,
    line_number: int,
) -> tuple[list[Mapping[str, Any]], Any]:
    attempts = row.get("generationAttempts")
    selected_attempt_index = row.get("selectedAttemptIndex")
    expected_attempt_keys = {
        "schemaVersion",
        "attemptIndex",
        "promptKind",
        "promptSHA256",
        "rawOutput",
        "outputKind",
        "formatError",
        "inputTokenCount",
        "generatedTokenCount",
        "generationTokenBudget",
        "hitTokenBudget",
        "generationAttemptSHA256",
    }
    resolved_output_mode = _validate_output_mode_for_agent(agent, output_mode)
    json_mode = resolved_output_mode == "json"
    maximum_attempts = STRICT_JSON_MAX_ATTEMPTS if json_mode else 1
    if (
        not isinstance(attempts, list)
        or not attempts
        or len(attempts) > maximum_attempts
        or type(selected_attempt_index) is not int
        or selected_attempt_index != len(attempts)
    ):
        raise ValueError(f"{path}:{line_number} has invalid generation attempt evidence")

    normalized_attempt_outputs: list[Any] = []
    for expected_index, attempt in enumerate(attempts, start=1):
        if not isinstance(attempt, dict) or set(attempt) != expected_attempt_keys:
            raise ValueError(f"{path}:{line_number} has invalid generation attempt evidence")
        expected_sha256 = attempt.get("generationAttemptSHA256")
        unsigned = dict(attempt)
        unsigned.pop("generationAttemptSHA256", None)
        raw_output = attempt.get("rawOutput")
        format_error = attempt.get("formatError")
        if (
            attempt.get("schemaVersion") != GENERATION_ATTEMPT_SCHEMA_VERSION
            or type(attempt.get("attemptIndex")) is not int
            or attempt.get("attemptIndex") != expected_index
            or attempt.get("promptKind")
            != ("frozen_evaluation" if expected_index == 1 else "strict_json_retry")
            or not isinstance(attempt.get("promptSHA256"), str)
            or _SHA256_PATTERN.fullmatch(attempt["promptSHA256"]) is None
            or expected_index > len(expected_prompt_sha256)
            or attempt["promptSHA256"] != expected_prompt_sha256[expected_index - 1]
            or not isinstance(raw_output, str)
            or (format_error is not None and not isinstance(format_error, str))
            or type(attempt.get("inputTokenCount")) is not int
            or attempt["inputTokenCount"] <= 0
            or type(attempt.get("generatedTokenCount")) is not int
            or attempt["generatedTokenCount"] < 0
            or type(attempt.get("generationTokenBudget")) is not int
            or attempt["generationTokenBudget"] <= 0
            or attempt["generatedTokenCount"] > attempt["generationTokenBudget"]
            or type(attempt.get("hitTokenBudget")) is not bool
            or attempt["hitTokenBudget"]
            != (
                attempt["generatedTokenCount"] >= attempt["generationTokenBudget"]
            )
            or not isinstance(expected_sha256, str)
            or _SHA256_PATTERN.fullmatch(expected_sha256) is None
            or _canonical_sha256(unsigned) != expected_sha256
        ):
            raise ValueError(f"{path}:{line_number} has invalid generation attempt evidence")
        normalized_output, expected_kind, expected_error = normalize_candidate_output(
            agent,
            raw_output,
            output_mode=resolved_output_mode,
            evaluation_module=evaluation_module,
            tool_contracts=tool_contracts,
        )
        if expected_index == 2 and agent == "cortex" and expected_error is None:
            transition_error = _cortex_retry_transition_error(
                normalized_attempt_outputs[0],
                attempts[0]["formatError"],
                normalized_output,
                tool_contracts,
                cortex_retry_locked_row,
            )
            if transition_error is not None:
                expected_kind = "invalid_cortex_route"
                expected_error = transition_error
        if (
            attempt.get("outputKind") != expected_kind
            or format_error != expected_error
        ):
            raise ValueError(f"{path}:{line_number} has inconsistent generation attempt evidence")
        normalized_attempt_outputs.append(normalized_output)

    if json_mode:
        if len(attempts) == 1 and attempts[0]["formatError"] is not None:
            raise ValueError(f"{path}:{line_number} omits the bounded strict JSON retry")
        if len(attempts) == STRICT_JSON_MAX_ATTEMPTS and attempts[0]["formatError"] is None:
            raise ValueError(f"{path}:{line_number} contains an ineligible strict JSON retry")
    elif len(attempts) != 1:
        raise ValueError(f"{path}:{line_number} retries a non-JSON candidate")

    return attempts, normalized_attempt_outputs[-1]


def _validated_candidate_row(
    row: Any,
    *,
    agent: str,
    expected_record: Mapping[str, Any],
    evaluation_module: ModuleType,
    tool_contracts: Mapping[str, Any] | None,
    path: Path,
    line_number: int,
) -> tuple[str, Any]:
    """Reconstruct one candidate solely from its raw generation attempts."""

    expected_keys = {
        "schemaVersion",
        "evalID",
        "agent",
        "outputMode",
        "output",
        "outputKind",
        "formatError",
        "inputTokenCount",
        "generatedTokenCount",
        "selectedAttemptIndex",
        "generationAttempts",
        "candidateRecordSHA256",
    }
    if not isinstance(row, dict) or set(row) != expected_keys:
        raise ValueError(f"{path}:{line_number} has an invalid candidate record")
    expected_eval_id = expected_record.get("evalID")
    expected_sha256 = row.get("candidateRecordSHA256")
    unsigned = dict(row)
    unsigned.pop("candidateRecordSHA256", None)
    eval_id = row.get("evalID")
    expected_output_mode = _record_output_mode(expected_record, agent=agent)
    if (
        row.get("schemaVersion") != CANDIDATE_OUTPUT_SCHEMA_VERSION
        or row.get("agent") != agent
        or not isinstance(eval_id, str)
        or not eval_id
        or eval_id != expected_eval_id
        or row.get("outputMode") != expected_output_mode
        or not isinstance(expected_sha256, str)
        or _SHA256_PATTERN.fullmatch(expected_sha256) is None
        or _canonical_sha256(unsigned) != expected_sha256
        or type(row.get("inputTokenCount")) is not int
        or row["inputTokenCount"] <= 0
        or type(row.get("generatedTokenCount")) is not int
        or row["generatedTokenCount"] < 0
    ):
        raise ValueError(f"{path}:{line_number} failed candidate lineage validation")
    record_messages = expected_record["messages"]
    primary_messages = _structured_output_messages(
        agent,
        record_messages,
        output_mode=expected_output_mode,
        tool_contracts=tool_contracts,
    )
    expected_prompt_sha256 = [_canonical_sha256(primary_messages)]
    cortex_retry_locked_row: Mapping[str, Any] | None = None
    if expected_output_mode == "json":
        raw_attempts = row.get("generationAttempts")
        first_attempt = (
            raw_attempts[0]
            if isinstance(raw_attempts, list)
            and raw_attempts
            and isinstance(raw_attempts[0], Mapping)
            else None
        )
        first_raw_output = (
            first_attempt.get("rawOutput")
            if isinstance(first_attempt, Mapping)
            and isinstance(first_attempt.get("rawOutput"), str)
            else None
        )
        first_output: Any = None
        first_error: str | None = None
        if first_raw_output is not None:
            first_output, _, first_error = normalize_candidate_output(
                agent,
                first_raw_output,
                output_mode=expected_output_mode,
                evaluation_module=evaluation_module,
                tool_contracts=tool_contracts,
            )
        if agent == "cortex":
            cortex_retry_locked_row = _cortex_retry_locked_manifest_row(
                primary_messages,
                first_output,
                tool_contracts,
            )
        expected_prompt_sha256.append(
            _canonical_sha256(
                _strict_json_retry_messages(
                    agent,
                    primary_messages,
                    validation_error=first_error,
                    failed_candidate=first_output,
                    tool_contracts=tool_contracts,
                )
            )
        )
    attempts, selected_output = _validated_generation_attempts(
        row,
        agent=agent,
        output_mode=expected_output_mode,
        evaluation_module=evaluation_module,
        tool_contracts=tool_contracts,
        cortex_retry_locked_row=cortex_retry_locked_row,
        expected_prompt_sha256=expected_prompt_sha256,
        path=path,
        line_number=line_number,
    )
    selected_attempt = attempts[-1]
    if (
        row.get("outputKind") != selected_attempt.get("outputKind")
        or row.get("formatError") != selected_attempt.get("formatError")
        or row.get("inputTokenCount") != selected_attempt.get("inputTokenCount")
        or row.get("generatedTokenCount")
        != selected_attempt.get("generatedTokenCount")
        or _canonical_sha256(row.get("output"))
        != _canonical_sha256(selected_output)
    ):
        raise ValueError(f"{path}:{line_number} has inconsistent selected attempt evidence")
    # The outer evidence row is canonically serialized with sorted keys, so its
    # nested output object cannot preserve the model's protocol-significant key
    # order. Score the independently replayed selected raw attempt instead.
    return eval_id, selected_output


def _checkpoint_record_bindings(
    records: Sequence[Mapping[str, Any]],
    *,
    agent: str,
    evaluation_module: ModuleType,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    upgraded: list[dict[str, Any]] = []
    bindings: list[dict[str, Any]] = []
    seen: set[str] = set()
    for case_index, raw_record in enumerate(records, start=1):
        record = evaluation_module.upgrade_evaluation_record(raw_record)
        eval_id = record.get("evalID")
        record_agent = str(
            (record.get("metadata") or {}).get("agent") or ""
        ).strip().lower()
        if (
            not isinstance(eval_id, str)
            or not eval_id
            or eval_id in seen
            or record_agent != agent
        ):
            raise ValueError(
                "Selected evaluation records are not a unique ordered agent cohort"
            )
        seen.add(eval_id)
        upgraded.append(record)
        bindings.append(
            {
                "caseIndex": case_index,
                "evalID": eval_id,
                "evaluationRecordSHA256": _canonical_sha256(record),
            }
        )
    if not bindings:
        raise ValueError("Selected evaluation checkpoint cohort must not be empty")
    return upgraded, bindings


def _checkpoint_execution_plan_binding(
    evaluation_plan: Mapping[str, Any] | None,
    *,
    max_examples: int | None,
    frozen_case_count: int,
) -> dict[str, Any]:
    if evaluation_plan is not None:
        return {
            "bindingKind": "prepared_execution_plan",
            "executionPlan": dict(evaluation_plan),
            "executionPlanSHA256": evaluation_plan.get("executionPlanSHA256"),
        }
    unsigned = {
        "bindingKind": "derived_legacy_evaluation_request",
        "evaluationScope": "smoke" if max_examples is not None else "full",
        "evaluationMaxExamples": max_examples,
        "frozenCaseCount": frozen_case_count,
    }
    return {**unsigned, "executionPlanSHA256": _canonical_sha256(unsigned)}


def _evaluation_checkpoint_contract(
    *,
    agent: str,
    variant: str,
    config_path: Path,
    config_file_sha256: str,
    evaluator_path: Path,
    adapter_dir: Path,
    adapter_sha256: str,
    finalized_path: Path,
    finalized_sha256: str,
    evaluation_path: Path,
    evaluation_file_sha256: str,
    evaluation_sha256: str,
    behavior_manifest_path: Path,
    behavior_manifest_file_sha256: str,
    behavior_manifest_sha256: str,
    output_dir: Path,
    evaluation_plan: Mapping[str, Any] | None,
    max_examples: int | None,
    frozen_case_count: int,
    selected_records: Sequence[Mapping[str, Any]],
    evaluation_module: ModuleType,
    generation: Mapping[str, Any],
) -> dict[str, Any]:
    upgraded, record_bindings = _checkpoint_record_bindings(
        selected_records,
        agent=agent,
        evaluation_module=evaluation_module,
    )
    unsigned: dict[str, Any] = {
        "schemaVersion": EVALUATION_CHECKPOINT_CONTRACT_SCHEMA_VERSION,
        "agent": agent,
        "variant": variant,
        "configPath": str(config_path),
        "configFileSHA256": config_file_sha256,
        "evaluatorCodePath": str(evaluator_path),
        "evaluatorCodeSHA256": _file_sha256(evaluator_path),
        "adapterDirectory": str(adapter_dir),
        "adapterSHA256": adapter_sha256,
        "finalizedVariantManifestPath": str(finalized_path),
        "finalizedVariantManifestSHA256": finalized_sha256,
        "evaluationJSONLPath": str(evaluation_path),
        "evaluationJSONLFileSHA256": evaluation_file_sha256,
        "evaluationSHA256": evaluation_sha256,
        "behaviorManifestPath": str(behavior_manifest_path),
        "behaviorManifestFileSHA256": behavior_manifest_file_sha256,
        "behaviorManifestSHA256": behavior_manifest_sha256,
        "outputDirectory": str(output_dir),
        "candidateOutputsPath": str(output_dir / "candidate_outputs.jsonl"),
        "evaluationReportPath": str(output_dir / "evaluation_report.json"),
        "evaluationRunManifestPath": str(
            output_dir / "evaluation_run_manifest.json"
        ),
        "executionPlanBinding": _checkpoint_execution_plan_binding(
            evaluation_plan,
            max_examples=max_examples,
            frozen_case_count=frozen_case_count,
        ),
        "selectedRecordCount": len(record_bindings),
        "selectedRecordOrderSHA256": _canonical_sha256(record_bindings),
        "selectedRecordsSHA256": _canonical_sha256(upgraded),
        "generation": dict(generation),
    }
    return {
        **unsigned,
        "evaluationCheckpointContractSHA256": _canonical_sha256(unsigned),
    }


_RUNTIME_EVIDENCE_KEYS = {
    "baseModelTokenizerDigest",
    "baseModelTokenizerFiles",
    "baseModelTokenizerClosureSHA256",
    "baseModelGenerationConfigFile",
    "baseModelTokenizerSnapshotPath",
    "baseModelTokenizerSnapshotVerification",
    "baseModelRuntimeSnapshotPath",
    "baseModelRuntimeSnapshotVerification",
    "runtimeModelBinding",
    "runtimeTokenizerBinding",
}


def _verified_checkpoint_runtime_evidence(
    value: Any,
    *,
    cfg: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _RUNTIME_EVIDENCE_KEYS:
        raise ValueError("Evaluation checkpoint runtime evidence has an invalid schema")
    evidence = dict(value)
    expected_static = {
        "baseModelTokenizerDigest": cfg.get("baseModelTokenizerDigest"),
        "baseModelTokenizerFiles": cfg.get("baseModelTokenizerFiles"),
        "baseModelTokenizerClosureSHA256": cfg.get(
            "baseModelTokenizerClosureSHA256"
        ),
        "baseModelGenerationConfigFile": cfg.get(
            "baseModelGenerationConfigFile"
        ),
        "baseModelTokenizerSnapshotPath": cfg.get(
            "baseModelTokenizerSnapshotPath"
        ),
        "baseModelTokenizerSnapshotVerification": cfg.get(
            "baseModelTokenizerSnapshotVerification"
        ),
        "baseModelRuntimeSnapshotPath": cfg.get("baseModelRuntimeSnapshotPath"),
        "baseModelRuntimeSnapshotVerification": cfg.get(
            "baseModelRuntimeSnapshotVerification"
        ),
    }
    if any(evidence.get(field) != expected for field, expected in expected_static.items()):
        raise ValueError("Evaluation checkpoint runtime snapshot evidence drifted")
    from tools.fine_tuning.unsloth.ubuntu_pipeline import (
        _verified_runtime_model_binding as verify_recorded_model_binding,
        _verified_runtime_tokenizer_binding as verify_recorded_tokenizer_binding,
    )

    snapshot_verification = expected_static[
        "baseModelRuntimeSnapshotVerification"
    ]
    if not isinstance(snapshot_verification, Mapping):
        raise ValueError("Evaluation checkpoint runtime snapshot is invalid")
    verify_recorded_model_binding(
        evidence.get("runtimeModelBinding"),
        config=cfg,
        snapshot_verification=snapshot_verification,
    )
    verify_recorded_tokenizer_binding(
        evidence.get("runtimeTokenizerBinding"),
        config=cfg,
        snapshot_verification=snapshot_verification,
    )
    return evidence


def _checkpoint_entry(
    *,
    case_index: int,
    record: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    unsigned = {
        "schemaVersion": EVALUATION_CHECKPOINT_ENTRY_SCHEMA_VERSION,
        "caseIndex": case_index,
        "evalID": record.get("evalID"),
        "evaluationRecordSHA256": _canonical_sha256(record),
        "candidateRecord": dict(candidate),
        "candidateRecordSHA256": candidate.get("candidateRecordSHA256"),
    }
    return {
        **unsigned,
        "evaluationCheckpointEntrySHA256": _canonical_sha256(unsigned),
    }


def _write_evaluation_checkpoint(
    path: Path,
    *,
    contract: Mapping[str, Any],
    runtime_evidence: Mapping[str, Any] | None,
    entries: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    expected_count = contract.get("selectedRecordCount")
    if (
        type(expected_count) is not int
        or expected_count <= 0
        or len(entries) > expected_count
    ):
        raise ValueError("Evaluation checkpoint completion count is invalid")
    unsigned = {
        "schemaVersion": EVALUATION_CHECKPOINT_SCHEMA_VERSION,
        "status": (
            "ready_for_finalization"
            if len(entries) == expected_count
            else "in_progress"
        ),
        "contract": dict(contract),
        "runtimeEvidence": (
            dict(runtime_evidence) if runtime_evidence is not None else None
        ),
        "completedCaseCount": len(entries),
        "completedCases": [dict(entry) for entry in entries],
    }
    checkpoint = {
        **unsigned,
        EVALUATION_CHECKPOINT_HASH_FIELD: _canonical_sha256(unsigned),
    }
    _atomic_write_bytes(path, _json_bytes(checkpoint))
    return checkpoint


def _require_private_checkpoint_path(path: Path) -> None:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Evaluation checkpoint is not a regular file: {path}")
    observed = path.stat(follow_symlinks=False)
    if (
        not stat.S_ISREG(observed.st_mode)
        or observed.st_uid != os.geteuid()
        or stat.S_IMODE(observed.st_mode) != 0o600
        or observed.st_size <= 0
        or observed.st_size > EVALUATION_CHECKPOINT_MAX_BYTES
    ):
        raise ValueError(
            "Evaluation checkpoint must be a bounded process-owned mode-0600 "
            f"file: {path}"
        )


def _fsync_directory_path(path: Path) -> None:
    directory_descriptor = os.open(
        path,
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)


def _durably_create_private_directory(path: Path) -> None:
    """Create every missing component privately and persist each parent entry."""

    missing: list[Path] = []
    cursor = path
    while not cursor.exists():
        if cursor.is_symlink():
            raise ValueError(
                f"Evaluation output path contains a dangling symlink: {cursor}"
            )
        missing.append(cursor)
        parent = cursor.parent
        if parent == cursor:
            raise ValueError(
                f"Evaluation output has no existing directory ancestor: {path}"
            )
        cursor = parent
    if cursor.is_symlink() or not cursor.is_dir():
        raise ValueError(
            f"Evaluation output ancestor is not a regular directory: {cursor}"
        )

    for directory in reversed(missing):
        try:
            directory.mkdir(mode=0o700)
        except FileExistsError as error:
            raise ValueError(
                "Evaluation output path changed while its private directory "
                f"was being created: {directory}"
            ) from error
        observed = directory.stat(follow_symlinks=False)
        if (
            not stat.S_ISDIR(observed.st_mode)
            or observed.st_uid != os.geteuid()
            or stat.S_IMODE(observed.st_mode) != 0o700
        ):
            raise ValueError(
                "New evaluation output directory must be process-owned mode 0700: "
                f"{directory}"
            )
        # Persist the new inode before its name, then persist the name in the
        # parent. A checkpoint fsync inside the child is not a substitute for
        # making the parent's newly-created directory entry durable.
        _fsync_directory_path(directory)
        _fsync_directory_path(directory.parent)


def _require_private_evaluation_directory(path: Path, *, create: bool) -> None:
    if create and not path.exists():
        _durably_create_private_directory(path)
    if path.is_symlink() or not path.is_dir():
        raise ValueError(f"Evaluation output is not a regular directory: {path}")
    observed = path.stat(follow_symlinks=False)
    if (
        not stat.S_ISDIR(observed.st_mode)
        or observed.st_uid != os.geteuid()
        or stat.S_IMODE(observed.st_mode) != 0o700
    ):
        raise ValueError(
            f"Evaluation output must be process-owned mode 0700: {path}"
        )


def _verified_evaluation_directory_entries(
    path: Path,
    *,
    allowed_names: set[str],
    required_names: set[str],
) -> set[str]:
    entries = list(path.iterdir())
    observed_names = {entry.name for entry in entries}
    if (
        len(observed_names) != len(entries)
        or not required_names.issubset(observed_names)
        or not observed_names.issubset(allowed_names)
    ):
        raise ValueError("Evaluation output directory has an unrecognized state")
    for entry in entries:
        if entry.is_symlink() or not entry.is_file():
            raise ValueError(f"Evaluation output entry is unsafe: {entry}")
        observed = entry.stat(follow_symlinks=False)
        if (
            not stat.S_ISREG(observed.st_mode)
            or observed.st_uid != os.geteuid()
            or stat.S_IMODE(observed.st_mode) & 0o077
        ):
            raise ValueError(
                f"Evaluation output entry must be private and process-owned: {entry}"
            )
    return observed_names


def _remove_verified_atomic_write_orphans(path: Path) -> tuple[str, ...]:
    """Remove exact private writer temps after the checkpoint was verified.

    Callers must cryptographically verify the existing checkpoint before calling
    this helper. The private directory prevents cross-user replacement; the
    descriptor-relative re-stat prevents deleting a changed directory entry.
    Temp contents are never opened or trusted.
    """

    _require_private_evaluation_directory(path, create=False)
    directory_descriptor = os.open(
        path,
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        orphan_names = tuple(
            sorted(
                name
                for name in os.listdir(directory_descriptor)
                if EVALUATION_ATOMIC_WRITE_TEMP_NAME_PATTERN.fullmatch(name)
                is not None
            )
        )
        verified_entries: dict[str, tuple[int, ...]] = {}
        for name in orphan_names:
            try:
                observed = os.stat(
                    name,
                    dir_fd=directory_descriptor,
                    follow_symlinks=False,
                )
            except OSError as error:
                raise ValueError(
                    f"Evaluation atomic-write orphan could not be verified: {name}"
                ) from error
            if (
                not stat.S_ISREG(observed.st_mode)
                or observed.st_uid != os.geteuid()
                or stat.S_IMODE(observed.st_mode) != 0o600
            ):
                raise ValueError(
                    "Evaluation atomic-write orphan must be a process-owned "
                    f"mode-0600 regular file: {name}"
                )
            verified_entries[name] = (
                observed.st_dev,
                observed.st_ino,
                observed.st_mode,
                observed.st_uid,
                observed.st_size,
                observed.st_mtime_ns,
                observed.st_ctime_ns,
            )

        removed_any = False
        try:
            for name in orphan_names:
                try:
                    observed = os.stat(
                        name,
                        dir_fd=directory_descriptor,
                        follow_symlinks=False,
                    )
                except OSError as error:
                    raise ValueError(
                        "Evaluation atomic-write orphan changed before removal: "
                        f"{name}"
                    ) from error
                current_entry = (
                    observed.st_dev,
                    observed.st_ino,
                    observed.st_mode,
                    observed.st_uid,
                    observed.st_size,
                    observed.st_mtime_ns,
                    observed.st_ctime_ns,
                )
                if current_entry != verified_entries[name]:
                    raise ValueError(
                        "Evaluation atomic-write orphan changed before removal: "
                        f"{name}"
                    )
                os.unlink(name, dir_fd=directory_descriptor)
                removed_any = True
        finally:
            if removed_any:
                os.fsync(directory_descriptor)
        return orphan_names
    finally:
        os.close(directory_descriptor)


def _require_recoverable_checkpoint_directory(
    path: Path,
    *,
    completed_case_count: int,
    selected_case_count: int,
) -> None:
    final_names = set(EVALUATION_FINAL_FILENAMES)
    observed = _verified_evaluation_directory_entries(
        path,
        allowed_names={EVALUATION_CHECKPOINT_FILENAME, *final_names},
        required_names={EVALUATION_CHECKPOINT_FILENAME},
    )
    published_final_subset = observed & final_names
    if published_final_subset and completed_case_count != selected_case_count:
        raise ValueError(
            "Partially published final evidence requires a complete checkpoint"
        )


def _verify_evaluation_checkpoint(
    path: Path,
    *,
    expected_contract: Mapping[str, Any],
    selected_records: Sequence[Mapping[str, Any]],
    agent: str,
    cfg: Mapping[str, Any],
    evaluation_module: ModuleType,
    tool_contracts: Mapping[str, Any] | None,
) -> dict[str, Any]:
    _require_private_checkpoint_path(path)
    checkpoint = _load_json_object(path, label="Evaluation checkpoint")
    expected_keys = {
        "schemaVersion",
        "status",
        "contract",
        "runtimeEvidence",
        "completedCaseCount",
        "completedCases",
        EVALUATION_CHECKPOINT_HASH_FIELD,
    }
    declared = checkpoint.get(EVALUATION_CHECKPOINT_HASH_FIELD)
    unsigned = dict(checkpoint)
    unsigned.pop(EVALUATION_CHECKPOINT_HASH_FIELD, None)
    entries = checkpoint.get("completedCases")
    completed_count = checkpoint.get("completedCaseCount")
    if (
        set(checkpoint) != expected_keys
        or checkpoint.get("schemaVersion")
        != EVALUATION_CHECKPOINT_SCHEMA_VERSION
        or not isinstance(declared, str)
        or _SHA256_PATTERN.fullmatch(declared) is None
        or _canonical_sha256(unsigned) != declared
        or checkpoint.get("contract") != dict(expected_contract)
        or type(completed_count) is not int
        or completed_count < 0
        or not isinstance(entries, list)
        or completed_count != len(entries)
        or completed_count > len(selected_records)
        or checkpoint.get("status")
        != (
            "ready_for_finalization"
            if completed_count == len(selected_records)
            else "in_progress"
        )
    ):
        raise ValueError("Evaluation checkpoint failed its exact self-hashed schema")

    runtime_value = checkpoint.get("runtimeEvidence")
    runtime_evidence = (
        _verified_checkpoint_runtime_evidence(runtime_value, cfg=cfg)
        if runtime_value is not None
        else None
    )
    if completed_count > 0 and runtime_evidence is None:
        raise ValueError("Evaluation checkpoint candidates lack runtime evidence")

    outputs: dict[str, Any] = {}
    output_rows: list[dict[str, Any]] = []
    verified_entries: list[dict[str, Any]] = []
    seen_eval_ids: set[str] = set()
    expected_entry_keys = {
        "schemaVersion",
        "caseIndex",
        "evalID",
        "evaluationRecordSHA256",
        "candidateRecord",
        "candidateRecordSHA256",
        "evaluationCheckpointEntrySHA256",
    }
    for expected_index, raw_entry in enumerate(entries, start=1):
        record = evaluation_module.upgrade_evaluation_record(
            selected_records[expected_index - 1]
        )
        if not isinstance(raw_entry, Mapping):
            raise ValueError("Evaluation checkpoint contains a non-object case")
        entry = dict(raw_entry)
        entry_declared = entry.pop("evaluationCheckpointEntrySHA256", None)
        candidate = entry.get("candidateRecord")
        eval_id = entry.get("evalID")
        if (
            set(raw_entry) != expected_entry_keys
            or entry.get("schemaVersion")
            != EVALUATION_CHECKPOINT_ENTRY_SCHEMA_VERSION
            or entry.get("caseIndex") != expected_index
            or not isinstance(eval_id, str)
            or eval_id in seen_eval_ids
            or eval_id != record.get("evalID")
            or entry.get("evaluationRecordSHA256")
            != _canonical_sha256(record)
            or not isinstance(candidate, Mapping)
            or entry.get("candidateRecordSHA256")
            != candidate.get("candidateRecordSHA256")
            or not isinstance(entry_declared, str)
            or _SHA256_PATTERN.fullmatch(entry_declared) is None
            or _canonical_sha256(entry) != entry_declared
        ):
            raise ValueError(
                "Evaluation checkpoint is not an exact unique selected-record prefix"
            )
        verified_eval_id, selected_output = _validated_candidate_row(
            dict(candidate),
            agent=agent,
            expected_record=record,
            evaluation_module=evaluation_module,
            tool_contracts=tool_contracts,
            path=path,
            line_number=expected_index,
        )
        if verified_eval_id != eval_id:
            raise ValueError("Evaluation checkpoint candidate identity drifted")
        seen_eval_ids.add(eval_id)
        outputs[eval_id] = selected_output
        output_rows.append(dict(candidate))
        verified_entries.append(dict(raw_entry))
    return {
        "checkpoint": checkpoint,
        "runtimeEvidence": runtime_evidence,
        "entries": verified_entries,
        "outputs": outputs,
        "outputRows": output_rows,
    }


def _recover_evaluation_checkpoint_directory(
    path: Path,
    *,
    expected_contract: Mapping[str, Any],
    selected_records: Sequence[Mapping[str, Any]],
    agent: str,
    cfg: Mapping[str, Any],
    evaluation_module: ModuleType,
    tool_contracts: Mapping[str, Any] | None,
) -> dict[str, Any]:
    _require_private_evaluation_directory(path, create=False)
    checkpoint_path = path / EVALUATION_CHECKPOINT_FILENAME
    recovered = _verify_evaluation_checkpoint(
        checkpoint_path,
        expected_contract=expected_contract,
        selected_records=selected_records,
        agent=agent,
        cfg=cfg,
        evaluation_module=evaluation_module,
        tool_contracts=tool_contracts,
    )
    _remove_verified_atomic_write_orphans(path)
    _require_recoverable_checkpoint_directory(
        path,
        completed_case_count=len(recovered["entries"]),
        selected_case_count=len(selected_records),
    )
    return recovered


def _candidate_failure_counts(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[int, int, int]:
    format_failures = 0
    initial_failures = 0
    recoveries = 0
    for row in rows:
        attempts = row.get("generationAttempts")
        if not isinstance(attempts, list) or not attempts:
            raise ValueError("Verified candidate row lost its generation attempts")
        first_error = attempts[0].get("formatError")
        final_error = row.get("formatError")
        if first_error is not None:
            initial_failures += 1
        if final_error is not None:
            format_failures += 1
        if len(attempts) == STRICT_JSON_MAX_ATTEMPTS and first_error is not None and final_error is None:
            recoveries += 1
    return format_failures, initial_failures, recoveries


def _remove_evaluation_checkpoint(path: Path) -> None:
    _require_private_checkpoint_path(path)
    path.unlink()
    directory_descriptor = os.open(
        path.parent,
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0),
    )
    try:
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)


def load_candidate_outputs(
    path: Path,
    *,
    agent: str,
    evaluation_records: Sequence[Mapping[str, Any]],
    tool_contracts: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Load evaluator output without trusting duplicate or mutated JSONL rows."""

    if not path.is_file():
        raise FileNotFoundError(f"Candidate output JSONL not found: {path}")
    evaluation_module = _load_evaluation_module()
    expected_records: list[Mapping[str, Any]] = []
    seen_expected_eval_ids: set[str] = set()
    for index, raw_record in enumerate(evaluation_records):
        record = evaluation_module.upgrade_evaluation_record(raw_record)
        eval_id = record.get("evalID")
        record_agent = str((record.get("metadata") or {}).get("agent") or "").strip().lower()
        messages = record.get("messages")
        if (
            not isinstance(eval_id, str)
            or not eval_id
            or eval_id in seen_expected_eval_ids
            or record_agent != agent
            or not isinstance(messages, list)
            or not messages
        ):
            raise ValueError(
                f"Expected evaluation record {index + 1} is not uniquely bound to agent {agent}"
            )
        seen_expected_eval_ids.add(eval_id)
        expected_records.append(record)
    if not expected_records:
        raise ValueError("Expected evaluation records must not be empty")

    outputs: dict[str, Any] = {}
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not raw_line.strip():
            continue
        row, row_error = evaluation_module._parse_candidate_json(raw_line)
        if row_error is not None:
            raise ValueError(f"{path}:{line_number} is not valid unique-key JSON")
        expected_index = len(outputs)
        if expected_index >= len(expected_records):
            raise ValueError(f"{path}:{line_number} failed candidate lineage validation")
        eval_id, selected_output = _validated_candidate_row(
            row,
            agent=agent,
            expected_record=expected_records[expected_index],
            evaluation_module=evaluation_module,
            tool_contracts=tool_contracts,
            path=path,
            line_number=line_number,
        )
        if eval_id in outputs:
            raise ValueError(f"{path}:{line_number} failed candidate lineage validation")
        outputs[eval_id] = selected_output
    if not outputs:
        raise ValueError(f"Candidate output JSONL is empty: {path}")
    expected_record_order = [str(record["evalID"]) for record in expected_records]
    if list(outputs) != expected_record_order:
        missing = sorted(set(expected_record_order) - set(outputs))
        extra = sorted(set(outputs) - set(expected_record_order))
        raise ValueError(
            "Candidate output evalID set does not match the frozen evaluation "
            "records or is out of order: "
            f"missing={missing} extra={extra}"
        )
    return outputs


def _check_output_paths(paths: Sequence[Path], *, overwrite: bool) -> None:
    resolved = [path.resolve() for path in paths]
    if len(set(resolved)) != len(resolved):
        raise ValueError("Evaluator output paths must be distinct")
    for path in paths:
        if path.is_symlink() or (path.exists() and not path.is_file()):
            raise ValueError(f"Evaluator output path is not a regular file: {path}")
        if path.exists() and not overwrite:
            raise FileExistsError(
                f"Evaluator output already exists (pass --overwrite to replace it): {path}"
            )


def run(args: argparse.Namespace) -> int:
    if args.max_examples is not None and args.max_examples <= 0:
        raise ValueError("--max-examples must be positive")
    if args.max_new_tokens <= 0 or args.max_new_tokens > 4096:
        raise ValueError("--max-new-tokens must be between 1 and 4096")

    config_path = Path(args.config).resolve()
    cfg, config_file_sha256 = _load_evaluation_config_snapshot(config_path)
    chat_template_contract_sha256 = verify_chat_template_contract(
        cfg.get("chatTemplateContract")
    )
    from tools.fine_tuning.unsloth.ubuntu_source_integrity import (
        validate_attestation_record,
    )

    expected_source_fields: dict[str, Any] = {}
    if cfg.get("runtimeSourceBindingMethod") == (
        "git_clean_worktree_plus_ubuntu_orchestration_manifest"
    ):
        source_integrity = cfg.get("ubuntuSourceIntegrity")
        if not isinstance(source_integrity, Mapping):
            raise ValueError("Evaluation config is missing Ubuntu source integrity")
        verified_source_integrity = validate_attestation_record(source_integrity)
        expected_source_fields = {
            "workingTreeDigest": verified_source_integrity["workingTreeDigest"],
            "ubuntuOrchestrationCodeSHA256": verified_source_integrity[
                "ubuntuOrchestrationCodeSHA256"
            ],
            "ubuntuSourceIntegritySHA256": verified_source_integrity[
                "sourceIntegritySHA256"
            ],
            "ubuntuSourceIntegrity": verified_source_integrity,
        }
        if any(cfg.get(key) != value for key, value in expected_source_fields.items()):
            raise ValueError("Evaluation config Ubuntu source-integrity fields drifted")
    agent = str(cfg["agent"]).strip().lower()
    if agent not in SUPPORTED_AGENTS:
        raise ValueError(f"Unsupported evaluation agent: {agent}")
    adapter_dir = Path(args.adapter_dir or cfg["adapter_output_dir"]).resolve()
    finalized_path = Path(
        args.finalized_variant_manifest
        or cfg.get("finalized_variant_manifest")
        or (Path(str(cfg["output_dir"])) / "finalized_variant_manifest.json")
    ).resolve()
    eval_path = Path(
        args.eval_jsonl or (Path(str(cfg["dataset_dir"])) / "eval.jsonl")
    ).resolve()
    behavior_manifest_path = Path(args.behavior_manifest).resolve()
    requested_output_dir = Path(
        args.output_dir or (finalized_path.parent / "evaluation")
    )
    if requested_output_dir.is_symlink():
        raise ValueError(
            f"Evaluation output directory must not be a symlink: {requested_output_dir}"
        )
    output_dir = requested_output_dir.resolve()
    candidate_path = output_dir / "candidate_outputs.jsonl"
    report_path = output_dir / "evaluation_report.json"
    run_manifest_path = output_dir / "evaluation_run_manifest.json"
    checkpoint_path = output_dir / EVALUATION_CHECKPOINT_FILENAME

    evaluation_module = _load_evaluation_module()
    all_records, evaluation_sha256 = load_evaluation_records(
        eval_path,
        agent=agent,
        evaluation_module=evaluation_module,
    )
    evaluation_file_sha256 = _file_sha256(eval_path)
    evaluation_plan = _verified_evaluation_execution_plan(
        cfg,
        max_examples=args.max_examples,
        frozen_case_count=len(all_records),
    )
    (
        tool_contracts,
        allowed_slots,
        behavior_manifest_sha256,
        behavior_manifest_file_sha256,
    ) = _load_behavior_contract_snapshot(behavior_manifest_path)
    expected_behavior_file_sha256 = cfg.get("behaviorManifestFileSHA256")
    if (
        not isinstance(expected_behavior_file_sha256, str)
        or _SHA256_PATTERN.fullmatch(expected_behavior_file_sha256) is None
        or behavior_manifest_file_sha256 != expected_behavior_file_sha256
    ):
        raise ValueError("Behavior manifest file drifted from the finalized evaluation config")
    validate_scoring_contracts(
        all_records,
        tool_contracts=tool_contracts,
        allowed_slots=allowed_slots,
    )
    finalized = load_finalized_manifest(
        finalized_path,
        cfg=cfg,
        evaluation_sha256=evaluation_sha256,
        evaluation_module=evaluation_module,
    )

    cfg["adapter_output_dir"] = str(adapter_dir)
    cfg["finalized_variant_manifest"] = str(finalized_path)
    lineage = _verified_release_bake_lineage(cfg)
    artifact_sha256 = lineage["adapterSHA256"]
    if artifact_sha256 != finalized["artifact"]["adapterSHA256"]:
        raise ValueError("Verified adapter artifact digest does not match finalized lineage")

    selected_records = select_evaluation_records(
        all_records,
        max_examples=args.max_examples,
    )
    complete_evaluation = (
        evaluation_plan["evaluationScope"] == "full"
        if evaluation_plan is not None
        else len(selected_records) == len(all_records)
    )
    generation_evidence = {
        "doSample": False,
        "numBeams": 1,
        "repetitionPenalty": GENERATION_REPETITION_PENALTY,
        "thinkingEnabled": False,
        "maxNewTokens": int(args.max_new_tokens),
        "maxSequenceLength": int(cfg["max_seq_length"]),
        "seed": int(cfg["seed"]),
        "outputModeContract": _evaluation_output_mode_contract(
            selected_records,
            agent=agent,
            tool_contracts=tool_contracts,
        ),
    }
    evaluator_path = Path(__file__).resolve()
    checkpoint_contract = _evaluation_checkpoint_contract(
        agent=agent,
        variant=str(cfg["variant"]),
        config_path=config_path,
        config_file_sha256=config_file_sha256,
        evaluator_path=evaluator_path,
        adapter_dir=adapter_dir,
        adapter_sha256=artifact_sha256,
        finalized_path=finalized_path,
        finalized_sha256=str(finalized["variantManifestSHA256"]),
        evaluation_path=eval_path,
        evaluation_file_sha256=evaluation_file_sha256,
        evaluation_sha256=evaluation_sha256,
        behavior_manifest_path=behavior_manifest_path,
        behavior_manifest_file_sha256=behavior_manifest_file_sha256,
        behavior_manifest_sha256=behavior_manifest_sha256,
        output_dir=output_dir,
        evaluation_plan=evaluation_plan,
        max_examples=args.max_examples,
        frozen_case_count=len(all_records),
        selected_records=selected_records,
        evaluation_module=evaluation_module,
        generation=generation_evidence,
    )

    verify_checkpoint_only = bool(
        getattr(args, "verify_checkpoint_only", False)
    )
    if verify_checkpoint_only and not output_dir.exists():
        raise FileNotFoundError(
            f"Evaluation checkpoint directory not found: {output_dir}"
        )
    _require_private_evaluation_directory(
        output_dir,
        create=not verify_checkpoint_only,
    )
    final_names = {
        candidate_path.name,
        report_path.name,
        run_manifest_path.name,
    }
    checkpoint_exists = checkpoint_path.exists() or checkpoint_path.is_symlink()
    if checkpoint_exists:
        recovered = _recover_evaluation_checkpoint_directory(
            output_dir,
            expected_contract=checkpoint_contract,
            selected_records=selected_records,
            agent=agent,
            cfg=cfg,
            evaluation_module=evaluation_module,
            tool_contracts=tool_contracts,
        )
    else:
        _verified_evaluation_directory_entries(
            output_dir,
            allowed_names={EVALUATION_CHECKPOINT_FILENAME, *final_names},
            required_names=set(),
        )
        if verify_checkpoint_only:
            raise FileNotFoundError(
                f"Evaluation checkpoint not found: {checkpoint_path}"
            )
        _check_output_paths(
            (candidate_path, report_path, run_manifest_path),
            overwrite=bool(args.overwrite),
        )
        _write_evaluation_checkpoint(
            checkpoint_path,
            contract=checkpoint_contract,
            runtime_evidence=None,
            entries=(),
        )
        recovered = _verify_evaluation_checkpoint(
            checkpoint_path,
            expected_contract=checkpoint_contract,
            selected_records=selected_records,
            agent=agent,
            cfg=cfg,
            evaluation_module=evaluation_module,
            tool_contracts=tool_contracts,
        )
    if verify_checkpoint_only:
        print(
            "Verified evaluation checkpoint: "
            f"{checkpoint_path} completed="
            f"{len(recovered['entries'])}/{len(selected_records)}"
        )
        return 0

    entries = list(recovered["entries"])
    outputs = dict(recovered["outputs"])
    output_rows = list(recovered["outputRows"])
    runtime_tokenizer_evidence = recovered["runtimeEvidence"]
    if len(entries) < len(selected_records):
        model, tokenizer, observed_runtime_evidence = load_inference_model(
            cfg,
            adapter_dir=adapter_dir,
        )
        if (
            verify_chat_template_contract(
                cfg.get("chatTemplateContract"),
                tokenizer=tokenizer,
            )
            != chat_template_contract_sha256
        ):
            raise RuntimeError("Loaded tokenizer chat-template contract digest drifted")
        if (
            runtime_tokenizer_evidence is not None
            and observed_runtime_evidence != runtime_tokenizer_evidence
        ):
            raise RuntimeError(
                "Resumed evaluation runtime evidence drifted from the checkpoint"
            )
        runtime_tokenizer_evidence = observed_runtime_evidence
        _write_evaluation_checkpoint(
            checkpoint_path,
            contract=checkpoint_contract,
            runtime_evidence=runtime_tokenizer_evidence,
            entries=entries,
        )

        def persist_completed_case(row: dict[str, Any]) -> None:
            case_index = len(entries) + 1
            record = evaluation_module.upgrade_evaluation_record(
                selected_records[case_index - 1]
            )
            entry = _checkpoint_entry(
                case_index=case_index,
                record=record,
                candidate=row,
            )
            _write_evaluation_checkpoint(
                checkpoint_path,
                contract=checkpoint_contract,
                runtime_evidence=runtime_tokenizer_evidence,
                entries=(*entries, entry),
            )
            entries.append(entry)

        (
            generated_outputs,
            generated_rows,
            _generated_format_failures,
            _generated_initial_failures,
            _generated_recoveries,
        ) = evaluate_records(
            selected_records[len(entries) :],
            agent=agent,
            model=model,
            tokenizer=tokenizer,
            max_seq_length=int(cfg["max_seq_length"]),
            max_new_tokens=int(args.max_new_tokens),
            evaluation_module=evaluation_module,
            tool_contracts=tool_contracts,
            on_case_completed=persist_completed_case,
        )
        outputs.update(generated_outputs)
        output_rows.extend(generated_rows)
    else:
        print(
            "Evaluation checkpoint already covers the complete selected cohort; "
            "finalizing without loading the model.",
            flush=True,
        )

    recovered = _verify_evaluation_checkpoint(
        checkpoint_path,
        expected_contract=checkpoint_contract,
        selected_records=selected_records,
        agent=agent,
        cfg=cfg,
        evaluation_module=evaluation_module,
        tool_contracts=tool_contracts,
    )
    if len(recovered["entries"]) != len(selected_records):
        raise RuntimeError("Evaluation checkpoint is not ready for finalization")
    outputs = dict(recovered["outputs"])
    output_rows = list(recovered["outputRows"])
    runtime_tokenizer_evidence = recovered["runtimeEvidence"]
    if runtime_tokenizer_evidence is None:
        raise RuntimeError("Complete evaluation checkpoint lacks runtime evidence")
    (
        format_failure_count,
        initial_format_failure_count,
        format_recovery_count,
    ) = _candidate_failure_counts(output_rows)
    controlled_lineage_builder = getattr(
        evaluation_module,
        "_variant_controlled_lineage",
        None,
    )
    if controlled_lineage_builder is None:
        raise RuntimeError("Evaluation module lacks controlled-lineage scoring support")
    report = evaluation_module.score_evaluation_suite(
        selected_records,
        outputs,
        frozen_evaluation_records=all_records,
        tool_contracts=tool_contracts,
        allowed_slots=allowed_slots,
        agent=agent,
        variant=cfg["variant"],
        controlled_lineage=controlled_lineage_builder(finalized),
        variant_manifest=finalized,
        artifact_sha256=artifact_sha256,
    )
    if not _evaluation_report_scope_valid(
        report,
        selected_case_count=len(selected_records),
        frozen_case_count=len(all_records),
    ):
        raise RuntimeError(
            "Evaluation report scope could not bind to the finalized adapter lineage"
        )

    candidate_bytes = _jsonl_bytes(output_rows)
    report_bytes = _json_bytes(report)
    candidate_file_sha256 = hashlib.sha256(candidate_bytes).hexdigest()
    report_file_sha256 = hashlib.sha256(report_bytes).hexdigest()
    status, quality_gate_passed = _evaluation_outcome(
        complete_evaluation=complete_evaluation,
        format_failure_count=format_failure_count,
        report=report,
    )
    execution_plan_evidence = (
        {
            "executionPlanSHA256": evaluation_plan["executionPlanSHA256"],
            "evaluationScope": evaluation_plan["evaluationScope"],
            "evaluationMaxExamples": evaluation_plan["evaluationMaxExamples"],
        }
        if evaluation_plan is not None
        else {}
    )
    run_manifest: dict[str, Any] = {
        "schemaVersion": EVALUATION_RUN_SCHEMA_VERSION,
        "status": status,
        "evaluatorCodePath": str(evaluator_path),
        "evaluatorCodeSHA256": _file_sha256(evaluator_path),
        "agent": agent,
        "variant": cfg["variant"],
        "configPath": str(config_path),
        "configSHA256": config_file_sha256,
        "chatTemplateContract": cfg["chatTemplateContract"],
        **runtime_tokenizer_evidence,
        "adapterDirectory": str(adapter_dir),
        "adapterSHA256": artifact_sha256,
        "finalizedVariantManifestPath": str(finalized_path),
        "finalizedVariantManifestSHA256": finalized["variantManifestSHA256"],
        "evaluationJSONLPath": str(eval_path),
        "evaluationSHA256": evaluation_sha256,
        "behaviorManifestPath": str(behavior_manifest_path),
        "behaviorManifestSHA256": behavior_manifest_sha256,
        "candidateOutputsPath": str(candidate_path),
        "candidateOutputsFileSHA256": candidate_file_sha256,
        "candidateOutputsSHA256": report["candidateOutputsSHA256"],
        "evaluationReportPath": str(report_path),
        "evaluationReportFileSHA256": report_file_sha256,
        "evaluationReportSHA256": report["reportSHA256"],
        "fullCaseCount": len(all_records),
        "generatedCaseCount": len(selected_records),
        "completeEvaluation": complete_evaluation,
        **execution_plan_evidence,
        "initialFormatFailureCount": initial_format_failure_count,
        "formatRecoveryCount": format_recovery_count,
        "formatFailureCount": format_failure_count,
        "criticalFailureCount": report["criticalFailureCount"],
        "qualityGatePassed": quality_gate_passed,
        **expected_source_fields,
        "generation": generation_evidence,
    }
    run_manifest["runManifestSHA256"] = _canonical_sha256(run_manifest)

    _atomic_write_bytes(candidate_path, candidate_bytes)
    _atomic_write_bytes(report_path, report_bytes)
    _atomic_write_bytes(run_manifest_path, _json_bytes(run_manifest))
    _remove_evaluation_checkpoint(checkpoint_path)
    print(f"Wrote candidate outputs: {candidate_path}")
    print(f"Wrote scored report: {report_path}")
    print(f"Wrote evaluation run manifest: {run_manifest_path}")
    print(
        f"Evaluation status={status} weightedScore={report['weightedScore']} "
        f"criticalFailures={report['criticalFailureCount']} "
        f"formatFailures={format_failure_count} "
        f"formatRecoveries={format_recovery_count}"
    )
    return _evaluation_exit_code(
        status=status,
        format_failure_count=format_failure_count,
    )


def main() -> None:
    raise SystemExit(run(parse_args()))


if __name__ == "__main__":
    main()
