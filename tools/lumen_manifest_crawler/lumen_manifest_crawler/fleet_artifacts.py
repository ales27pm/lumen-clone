from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from lumen_manifest_crawler.dataset.adapter_evaluation import canonical_sha256
from lumen_manifest_crawler.dataset.chat_template_contract import (
    generic_strict_json_retry_instruction,
)
from lumen_manifest_crawler.manifest import AgentBehaviorManifest, ModelSlotManifest, ToolManifest
from lumen_manifest_crawler.runtime_prompt_contract import (
    FLEET_SYSTEM_PROMPT_CONTRACT_SCHEMA_VERSION,
    RUNTIME_PROMPT_COMPOSER_POLICY_ID,
    RUNTIME_PROMPT_COMPOSER_POLICY_SHA256,
    prompt_sha256,
)


ORCHESTRATION_DERIVATION_SCHEMA_VERSION = (
    "lumen.fleet-graph-derivation/1.0.0"
)
ORCHESTRATION_EVENT_ID_GRAMMAR = "<scenarioID>::event::<one-based two-digit order>"
ORCHESTRATION_BEHAVIOR_CONDITIONED_VARIANT = "behavior-conditioned"
ORCHESTRATION_BEHAVIOR_CONDITIONED_REPLICAS = 8
ORCHESTRATION_BEHAVIOR_CONDITIONED_SFT_REPLICAS = 8
ORCHESTRATION_CORE_FAILURE_CONTRAST_MODE = "core_failure_family_atomic"
ORCHESTRATION_CORE_FAILURE_FAMILY_MUTATIONS = {
    "parallel-dependencies": "top_level_dependencies_omission",
    "no-delegation": "event_type_vocabulary",
    "approval-boundary": "event_completeness_contract",
    "unavailable-boundary": "decision_aggregation_owner_omission",
    "context-handoff": "decision_strategy_role",
    "aggregation-owner": "dependency_endpoint_role",
    "nonexistent-slot-negative": "terminal_stop_reason",
    "duplicate-suppression": "event_payload_schema",
    "sequential-dependencies": "scenario_identity_role",
}
_ORCHESTRATION_TOP_LEVEL_OMISSION_KEY = "dependencies"
ORCHESTRATION_TRAINING_IDENTITY_SCHEMA_VERSION = (
    "lumen.fleet-training-identity/1.4.0"
)
ORCHESTRATION_TRAINING_SCENARIO_ID_WIDTH = 6
ORCHESTRATION_TRAINING_FACT_ID_WIDTH = 6
ORCHESTRATION_TRAINING_IDENTITY_PREFIX = "id-"
_ORCHESTRATION_TRAINING_IDENTITY_ALPHABET = "abcdefghijklmnopqrstuvwxyz"
_ORCHESTRATION_SHARED_TRAINING_IDENTITY_SURFACES = (
    ((6,), ""),
    ((7,), ""),
    ((5, 3), "-"),
    ((5, 3), "_"),
    ((6, 4), "-"),
    ((6, 4), "_"),
    ((7, 6), "-"),
    ((13, 12, 10), "_"),
)
_ORCHESTRATION_TRAINING_IDENTITY_SURFACES = {
    "scenario": _ORCHESTRATION_SHARED_TRAINING_IDENTITY_SURFACES,
    "fact": _ORCHESTRATION_SHARED_TRAINING_IDENTITY_SURFACES,
}
_ORCHESTRATION_VARIED_FACT_KINDS_BY_BEHAVIOR = {
    "no-delegation": frozenset({"trusted-evidence"}),
    "sequential-dependencies": frozenset({"executor-observation"}),
    "parallel-dependencies": frozenset({"parallel-join"}),
    "context-handoff": frozenset(
        {
            "approved-action",
            "allowed-executor-context-1",
            "allowed-executor-context-2",
            "forbidden-executor-context-1",
            "forbidden-executor-context-2",
        }
    ),
    "duplicate-suppression": frozenset({"shared-work"}),
    "aggregation-owner": frozenset({"validated-response"}),
    "approval-boundary": frozenset({"approval-policy-snapshot"}),
    "unavailable-boundary": frozenset({"permission-check", "request"}),
    "nonexistent-slot-negative": frozenset({"slot-directory-snapshot"}),
}
ORCHESTRATION_ATOMIC_MUTATION_KINDS = (
    "terminal_decision_contract",
    "event_type_vocabulary",
    "event_completeness_contract",
    "event_order",
    "event_id_grammar",
    "dependency_endpoint_reference",
    "event_payload_schema",
    "delegated_slot_contract",
)
ORCHESTRATION_OUTPUT_INTERFACE = (
    "Canonical output interface: emit top-level keys in this exact order: "
    "`graphSchemaVersion`, `scenarioID`, `knownSlotIDs`, `events`, "
    "`dependencies`, then `decision`. The top-level keys are exactly that set. "
    "Each event emits `id`, then `type`, then only its required payload fields. "
    "Each dependency emits exactly `fromEventID`, `kind`, then `toEventID`, "
    "with `kind` set to "
    "`requires`. `events` and `dependencies` are arrays; `decision` is an "
    "object; `graphSchemaVersion` and `scenarioID` are strings; and "
    "`knownSlotIDs` and `delegatedSlotIDs` are ordered string arrays. "
    "The `decision` object is last and emits exactly `strategy`, "
    "`delegatedSlotIDs`, `aggregationOwnerSlotID`, then `stopReason`. "
    "`aggregationOwnerSlotID` is a known-slot string or null, while "
    "`stopReason` and `strategy` are strings. Each event has `id` and `type` "
    "plus only the fields required by its behavior, with no wrapper objects or "
    "invented aliases. Copy the supplied graph schema version, scenario ID, "
    "and known slot IDs exactly. Substitute the actual supplied scenario ID into "
    "every event ID using a two-digit one-based order; never emit the literal "
    "`<scenarioID>` placeholder. Every dependency endpoint references an emitted "
    "event ID. `delegatedSlotIDs` is the first-event-order list of unique known "
    "`targetSlotID` values from `delegate` events. Emit one terminal `stop` event "
    "whose `reason` exactly equals `decision.stopReason`, no more than 12 events "
    "or 16 dependencies, and stop immediately after the minimal graph's closing "
    "brace."
)


@dataclass(frozen=True)
class FleetArtifacts:
    system_prompts: dict[str, dict[str, Any]]
    cross_model_training: list[dict[str, Any]]
    orchestration_evals: list[dict[str, Any]]
    markdown: str


def _compact_orchestration_training_digest(digest: str, *, width: int) -> str:
    """Encode a SHA-256 identity densely while retaining deterministic entropy."""

    if re.fullmatch(r"[0-9a-f]{64}", digest) is None or width <= 0:
        raise ValueError("Fleet compact identity input is invalid")
    radix = len(_ORCHESTRATION_TRAINING_IDENTITY_ALPHABET)
    value = int(digest, 16) % (radix**width)
    encoded = ""
    while value:
        value, remainder = divmod(value, radix)
        encoded = _ORCHESTRATION_TRAINING_IDENTITY_ALPHABET[remainder] + encoded
    return encoded.rjust(width, _ORCHESTRATION_TRAINING_IDENTITY_ALPHABET[0])


def _register_orchestration_training_identity(
    *,
    identity_registry: dict[str, str] | None,
    identity: str,
    digest: str,
    identity_class: str,
) -> str:
    """Require compact training identities to remain injective per generation."""

    if identity_registry is None:
        return identity
    registered_digest = identity_registry.get(identity)
    if registered_digest is not None and registered_digest != digest:
        raise ValueError(f"Fleet {identity_class} identity collision")
    identity_registry[identity] = digest
    return identity


def _orchestration_training_identity_surface_index(
    *,
    identity_class: str,
    behavior: str,
    replica_index: int | None,
    lane: str,
    fact_kind: str | None,
    digest: str,
) -> int:
    """Choose every bounded surface once per conditioned behavior, opaquely."""

    surfaces = _ORCHESTRATION_TRAINING_IDENTITY_SURFACES[identity_class]
    # Vary one behavior-critical external fact per graph instead of only the
    # request ID. These values occupy payload/dependency positions in the
    # frozen contract. Keeping the remaining facts compact preserves the
    # independently enforced 50-60% optimizer-token band.
    if identity_class == "fact":
        varied_fact_kinds = _ORCHESTRATION_VARIED_FACT_KINDS_BY_BEHAVIOR.get(
            behavior
        )
        if varied_fact_kinds is None:
            raise ValueError("Unknown Fleet identity behavior class")
        if fact_kind not in varied_fact_kinds:
            return 0
    if replica_index is None:
        return (
            0
            if identity_class == "fact"
            else int(digest[:8], 16) % len(surfaces)
        )
    # Keep the scenario identity and every behavior-critical fact on the same
    # opaque surface for a conditioned instance. This makes one replica expose
    # a complete long-form graph envelope without using a scenario/fact prefix
    # or shape cue. The remaining facts stay compact so the independently
    # enforced optimizer-token band remains bounded.
    permutation_digest = canonical_sha256(
        {
            "schemaVersion": ORCHESTRATION_TRAINING_IDENTITY_SCHEMA_VERSION,
            "surfacePermutation": True,
            "behaviorClass": behavior,
            "lane": lane,
        }
    )
    return (int(permutation_digest[:8], 16) + replica_index) % len(surfaces)


def _format_orchestration_training_identity(
    *,
    identity_class: str,
    digest: str,
    surface_index: int,
) -> str:
    """Render one opaque digest with bounded short and long identifier shapes."""

    surfaces = _ORCHESTRATION_TRAINING_IDENTITY_SURFACES[identity_class]
    if surface_index < 0 or surface_index >= len(surfaces):
        raise ValueError("Fleet training identity surface index is invalid")
    segment_widths, separator = surfaces[surface_index]
    encoded = _compact_orchestration_training_digest(
        digest,
        width=sum(segment_widths),
    )
    segments: list[str] = []
    offset = 0
    for width in segment_widths:
        segments.append(encoded[offset : offset + width])
        offset += width
    return ORCHESTRATION_TRAINING_IDENTITY_PREFIX + separator.join(segments)


def _opaque_orchestration_training_identity(
    *,
    identity_class: str,
    behavior: str,
    variant: str,
    replica_index: int | None,
    lane: str,
    fact_kind: str | None = None,
    identity_registry: dict[str, str] | None = None,
) -> str:
    """Return a deterministic identity whose visible form carries no matrix cue."""

    if identity_class not in {"scenario", "fact"}:
        raise ValueError("Unknown Fleet training identity class")
    if identity_class == "fact" and not fact_kind:
        raise ValueError("Fleet fact identities require an independent fact kind")
    digest = canonical_sha256(
        {
            "schemaVersion": ORCHESTRATION_TRAINING_IDENTITY_SCHEMA_VERSION,
            "identityClass": identity_class,
            "behaviorClass": behavior,
            "trainingMatrixVariant": variant,
            "behaviorConditionedInstanceIndex": (
                replica_index + 1 if replica_index is not None else None
            ),
            "lane": lane,
            "factKind": fact_kind,
        }
    )
    surface_index = _orchestration_training_identity_surface_index(
        identity_class=identity_class,
        behavior=behavior,
        replica_index=replica_index,
        lane=lane,
        fact_kind=fact_kind,
        digest=digest,
    )
    identity = _format_orchestration_training_identity(
        identity_class=identity_class,
        digest=digest,
        surface_index=surface_index,
    )
    return _register_orchestration_training_identity(
        identity_registry=identity_registry,
        identity=identity,
        digest=digest,
        identity_class=identity_class,
    )


def _orchestration_training_scenario_id(
    *,
    behavior: str,
    variant: str,
    replica_index: int | None,
    lane: str = "sft",
    identity_registry: dict[str, str] | None = None,
) -> str:
    return _opaque_orchestration_training_identity(
        identity_class="scenario",
        behavior=behavior,
        variant=variant,
        replica_index=replica_index,
        lane=lane,
        identity_registry=identity_registry,
    )


def _orchestration_training_fact_id(
    *,
    behavior: str,
    variant: str,
    replica_index: int | None,
    fact_kind: str,
    lane: str = "sft",
    identity_registry: dict[str, str] | None = None,
) -> str:
    return _opaque_orchestration_training_identity(
        identity_class="fact",
        behavior=behavior,
        variant=variant,
        replica_index=replica_index,
        lane=lane,
        fact_kind=fact_kind,
        identity_registry=identity_registry,
    )


def _is_opaque_orchestration_training_identity(
    value: str,
    *,
    identity_class: str,
) -> bool:
    if identity_class not in {"scenario", "fact"}:
        raise ValueError("Unknown Fleet training identity class")
    prefix = ORCHESTRATION_TRAINING_IDENTITY_PREFIX
    if not value.startswith(prefix):
        return False
    payload = value[len(prefix) :]
    for segment_widths, separator in _ORCHESTRATION_TRAINING_IDENTITY_SURFACES[
        identity_class
    ]:
        pattern = separator.join(rf"[a-z]{{{width}}}" for width in segment_widths)
        if re.fullmatch(pattern, payload) is not None:
            return True
    return False


def _is_opaque_orchestration_training_scenario_id(value: str) -> bool:
    return _is_opaque_orchestration_training_identity(
        value,
        identity_class="scenario",
    )


def _is_opaque_orchestration_training_fact_id(value: str) -> bool:
    return _is_opaque_orchestration_training_identity(
        value,
        identity_class="fact",
    )


def generate_fleet_artifacts(manifest: AgentBehaviorManifest) -> FleetArtifacts:
    return FleetArtifacts(
        system_prompts=generate_fleet_system_prompts(manifest),
        cross_model_training=generate_cross_model_training(manifest),
        orchestration_evals=generate_orchestration_evals(manifest),
        markdown=generate_manifest_markdown(manifest),
    )


def generate_fleet_system_prompts(manifest: AgentBehaviorManifest) -> dict[str, dict[str, Any]]:
    tools_by_slot = _tools_by_slot(manifest)
    source_map = _source_code_map(manifest)
    prompts: dict[str, dict[str, Any]] = {}
    for slot in sorted(manifest.fleet.slots, key=lambda item: item.id):
        topology = manifest.fleetTopology.slots.get(slot.id)
        public_directory = _public_model_directory(manifest, current_slot_id=slot.id)
        routing_table = _routing_table(manifest)
        routing_rules = {entry["intent"]: {"allowedTools": entry["allowedTools"], "forbiddenTools": entry["forbiddenTools"]} for entry in routing_table}
        available_tools = tools_by_slot.get(slot.id, [])
        compact_payload = {
            "slotID": slot.id,
            "role": slot.role,
            "purpose": topology.purpose if topology else _slot_purpose_fallback(slot),
            "responsibilities": sorted(slot.responsibilities),
            "availableTools": [_tool_payload(tool) for tool in available_tools],
            "modelDirectory": public_directory,
            "routingRules": routing_table,
            "topology": topology.model_dump() if topology else {},
            "memory": manifest.memory.model_dump(),
            "sentinelPolicy": manifest.sentinels.model_dump(),
            "sourceCodeMap": source_map,
            "slotSource": _slot_source_payload(slot),
            "fleetIdentity": {
                "agentName": manifest.app.name,
                "singleEntityInstruction": "All model slots are coordinated components of one logical Lumen agent. If work is outside your scope, delegate or route it using manifest-defined rules instead of improvising.",
            },
        }
        prompt = _system_prompt_text(manifest, slot, compact_payload)
        prompts[slot.id] = {
            "promptContractSchemaVersion": FLEET_SYSTEM_PROMPT_CONTRACT_SCHEMA_VERSION,
            "fleetContractVersion": manifest.fleet.contractVersion,
            "composerPolicyID": RUNTIME_PROMPT_COMPOSER_POLICY_ID,
            "composerPolicySHA256": RUNTIME_PROMPT_COMPOSER_POLICY_SHA256,
            "systemPromptSHA256": prompt_sha256(prompt),
            "slotID": slot.id,
            "role": slot.role,
            "systemPrompt": prompt,
            "contextPayload": compact_payload,
            "system_prompt": prompt,
            "model_directory": public_directory,
            "routing_rules": routing_rules,
            "source_code_map": source_map,
        }
    return prompts


def generate_cross_model_training(manifest: AgentBehaviorManifest) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    slots = sorted(manifest.fleet.slots, key=lambda item: item.id)
    records.extend(_fleet_whole_system_records(manifest))
    records.extend(_orchestration_training_records(manifest))
    for slot in slots:
        records.extend(_self_knowledge_records(manifest, slot))
        records.extend(_source_code_self_knowledge_records(manifest, slot))
        records.extend(_tool_origin_records(manifest, slot))
        records.extend(_routing_origin_records(manifest, slot))
    for source in slots:
        for target in slots:
            if source.id == target.id:
                continue
            records.extend(_peer_knowledge_records(manifest, source, target))
            records.extend(_peer_source_knowledge_records(manifest, source, target))
            records.extend(_delegation_records(manifest, source, target))
            records.extend(_private_state_boundary_records(manifest, source, target))
    return records


def generate_orchestration_evals(manifest: AgentBehaviorManifest) -> list[dict[str, Any]]:
    """Return Fleet-owned, manifest-grounded orchestration evaluation records."""
    records: list[dict[str, Any]] = []
    for scenario in _orchestration_eval_scenarios(manifest):
        graph = scenario["graph"]
        decision = graph["decision"]
        derivation = _orchestration_derivation_contract(scenario)
        if _derive_orchestration_graph_from_contract(derivation) != graph:
            raise ValueError(
                f"Fleet holdout derivation does not uniquely rebuild graph: {scenario['id']}"
            )
        prompt = _orchestration_eval_prompt(scenario, derivation)
        metadata = {
            **_native_orchestration_metadata(manifest, scenario["id"]),
            "behaviorClass": scenario["behaviorClass"],
            "holdoutInstance": True,
            "expectedCandidateHashSchemaVersion": "lumen.eval-candidate-hash/1.0.0",
            "expectedCandidateSHA256": canonical_sha256(graph),
            "expectedCandidateTopologyHashSchemaVersion": (
                "lumen.eval-candidate-topology-hash/1.0.0"
            ),
            "expectedCandidateTopologySHA256": canonical_sha256(
                _orchestration_topology_contract(graph)
            ),
        }
        records.append({
            "id": _record_id("orchestration-eval", scenario["id"]),
            "schemaVersion": "2.1.0",
            "recordType": "eval",
            "sourceFamily": "fleet_orchestration_native",
            "agentRole": "fleet",
            "taskType": "fleet_orchestration_event_graph_eval",
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            "expected": {
                "metricVersion": "1.0.0",
                "graphSchemaVersion": graph["graphSchemaVersion"],
                "scenarioID": scenario["id"],
                "strategy": decision["strategy"],
                "knownSlotIDs": graph["knownSlotIDs"],
                "expectedDelegatedSlotIDs": decision["delegatedSlotIDs"],
                "expectedAggregationOwnerSlotID": decision["aggregationOwnerSlotID"],
                "expectedStopReason": decision["stopReason"],
                "requiredEventTypes": [event["type"] for event in graph["events"]],
                "requiredDependencies": graph["dependencies"],
                "requiresCanonicalDerivation": True,
                "canonicalDerivation": derivation,
                "mustUseKnownSlotsOnly": True,
                "mustNotExposePrivateState": True,
                **scenario["evalConstraints"],
            },
            "metadata": metadata,
        })
    return records


def _orchestration_eval_prompt(
    scenario: dict[str, Any],
    derivation: dict[str, Any],
) -> str:
    """Provide request/state facts; require policy derivation, not transcription."""

    return _canonical_orchestration_prompt(scenario, derivation)


def _canonical_orchestration_prompt(
    scenario: dict[str, Any],
    derivation: dict[str, Any],
) -> str:
    """Use one non-leaking prompt grammar for training and holdout graphs."""

    facts = derivation["facts"]
    return "\n".join(
        [
            "Return exactly one JSON Fleet event-graph object and nothing else.",
            (
                "Derive the graph from the canonical Fleet policy; no target graph, "
                "event list, dependency list, decision object, or answer hash is supplied."
            ),
            (
                f"Behavior class `{scenario['behaviorClass']}`: "
                f"{scenario['prompt']}"
            ),
            (
                "Canonical policy conditions (each condition has the same meaning in "
                "training and evaluation): "
                f"{json.dumps(derivation['policyConditions'], sort_keys=True)}"
            ),
            (
                f"Graph schema `{derivation['graphSchemaVersion']}`; scenario "
                f"`{derivation['scenarioID']}`; known slots "
                f"{json.dumps(derivation['knownSlotIDs'], ensure_ascii=False)}."
            ),
            ORCHESTRATION_OUTPUT_INTERFACE,
            (
                "Canonical event IDs are derived solely from scenario identity and "
                f"order using `{ORCHESTRATION_EVENT_ID_GRAMMAR}`."
            ),
            (
                "Trusted request/state facts (these are inputs, not output fields): "
                f"{json.dumps(facts, ensure_ascii=False, sort_keys=True)}"
            ),
            (
                "Apply the behavior-class policy to derive the minimal canonical stages, "
                "direct-prerequisite edges, delegated-slot directory, aggregation owner, "
                "strategy, and terminal reason."
            ),
            "Expose no private state and do not invent a slot, tool, fact, or stage.",
        ]
    )


def _events_of_type(
    graph: dict[str, Any],
    event_type: str,
) -> list[dict[str, Any]]:
    events = graph.get("events")
    if not isinstance(events, list):
        raise ValueError("Fleet derivation graph has no events")
    matched = [
        event
        for event in events
        if isinstance(event, dict) and event.get("type") == event_type
    ]
    if not matched:
        raise ValueError(f"Fleet derivation graph lacks {event_type}")
    return matched


def _one_event(
    graph: dict[str, Any],
    event_type: str,
) -> dict[str, Any]:
    matched = _events_of_type(graph, event_type)
    if len(matched) != 1:
        raise ValueError(
            f"Fleet derivation graph requires one {event_type}, got {len(matched)}"
        )
    return matched[0]


def _delegation_to(
    graph: dict[str, Any],
    slot_id: str,
) -> dict[str, Any]:
    matched = [
        event
        for event in _events_of_type(graph, "delegate")
        if event.get("targetSlotID") == slot_id
    ]
    if len(matched) != 1:
        raise ValueError(
            f"Fleet derivation graph requires one delegation to {slot_id}"
        )
    return matched[0]


_ORCHESTRATION_POLICY_CONDITION_KEYS = (
    "requestNormalizationRequired",
    "policyAuditRequired",
    "trustedContextSnapshotProvided",
    "executorObservationProvided",
    "parallelJoinRequired",
    "contextBoundaryReviewRequired",
    "candidateBranchesProvided",
    "aggregationInputVerificationRequired",
    "responseValidationRequired",
    "approvalPolicyEvaluationRequired",
    "permissionPreflightRequired",
    "slotDirectorySnapshotProvided",
    "rejectionRecordRequired",
)
_HOLDOUT_POLICY_CONDITION_BY_BEHAVIOR = {
    "no-delegation": ("trustedContextSnapshotProvided",),
    "sequential-dependencies": ("executorObservationProvided",),
    "parallel-dependencies": ("parallelJoinRequired",),
    "context-handoff": ("contextBoundaryReviewRequired",),
    "duplicate-suppression": ("candidateBranchesProvided",),
    "aggregation-owner": (
        "aggregationInputVerificationRequired",
        "responseValidationRequired",
    ),
    "approval-boundary": ("approvalPolicyEvaluationRequired",),
    "unavailable-boundary": ("permissionPreflightRequired",),
    "nonexistent-slot-negative": (
        "slotDirectorySnapshotProvided",
        "rejectionRecordRequired",
    ),
}


def _orchestration_policy_conditions(
    *,
    behavior: str,
    training_variant: str | None,
) -> dict[str, bool]:
    conditions = {key: False for key in _ORCHESTRATION_POLICY_CONDITION_KEYS}
    if training_variant == ORCHESTRATION_BEHAVIOR_CONDITIONED_VARIANT:
        for key in _HOLDOUT_POLICY_CONDITION_BY_BEHAVIOR.get(behavior, ()):
            conditions[key] = True
    elif training_variant == "normalized-intake":
        conditions["requestNormalizationRequired"] = True
        for key in _HOLDOUT_POLICY_CONDITION_BY_BEHAVIOR.get(behavior, ()):
            conditions[key] = True
    elif training_variant == "policy-audited":
        conditions["policyAuditRequired"] = True
        for key in _HOLDOUT_POLICY_CONDITION_BY_BEHAVIOR.get(behavior, ()):
            conditions[key] = True
    elif training_variant == "normalization-policy-audited":
        conditions["requestNormalizationRequired"] = True
        conditions["policyAuditRequired"] = True
        for key in _HOLDOUT_POLICY_CONDITION_BY_BEHAVIOR.get(behavior, ()):
            conditions[key] = True
    elif training_variant == "core":
        # The optimizer-visible core example teaches the same canonical
        # behavior topology as the frozen evaluation contract. Its request
        # facts and identifiers remain independently generated below.
        for key in _HOLDOUT_POLICY_CONDITION_BY_BEHAVIOR.get(behavior, ()):
            conditions[key] = True
    elif training_variant is None:
        for key in _HOLDOUT_POLICY_CONDITION_BY_BEHAVIOR.get(behavior, ()):
            conditions[key] = True
    else:
        raise ValueError(f"Unknown Fleet training matrix variant: {training_variant}")
    return conditions


def _training_orchestration_facts(
    scenario: dict[str, Any],
) -> dict[str, Any]:
    """Extract semantic inputs without copying a stage, edge, or decision."""

    graph = scenario["graph"]
    behavior = str(scenario["behaviorClass"])
    if behavior == "no-delegation":
        evidence = _one_event(graph, "trusted_context_verified")
        return {"trustedEvidenceStatus": evidence["evidenceStatus"]}
    if behavior == "sequential-dependencies":
        return {
            "peerContext": {
                slot: _delegation_to(graph, slot)["contextKeys"]
                for slot in ("cortex", "executor", "mouth")
            }
        }
    if behavior == "parallel-dependencies":
        return {
            "peerContext": {
                slot: _delegation_to(graph, slot)["contextKeys"]
                for slot in ("cortex", "executor", "mimicry", "mouth")
            }
        }
    if behavior == "context-handoff":
        handoff = _delegation_to(graph, "executor")
        return {
            "allowedExecutorContext": handoff["contextKeys"],
            "forbiddenExecutorContext": handoff["excludes"],
        }
    if behavior == "duplicate-suppression":
        dispatch = _one_event(graph, "delegate")
        return {
            "workOwnerSlot": dispatch["targetSlotID"],
            "sharedWorkKey": dispatch["workKey"],
        }
    if behavior == "aggregation-owner":
        return {
            "renderContext": _delegation_to(graph, "mouth")["contextKeys"],
        }
    if behavior == "approval-boundary":
        request = _one_event(graph, "request_received")
        boundary = _one_event(graph, "approval_boundary")
        return {
            "toolIdentifier": request["toolID"],
            "approvalState": boundary["approvalState"],
        }
    if behavior == "unavailable-boundary":
        request = _one_event(graph, "request_received")
        unavailable = _one_event(graph, "capability_unavailable")
        return {
            "toolIdentifier": request["toolID"],
            "permissionKey": unavailable["permissionKey"],
            "permissionState": unavailable["permissionState"],
        }
    if behavior == "nonexistent-slot-negative":
        request = _one_event(graph, "request_received")
        return {"requestedSlotIdentifier": request["requestedSlotID"]}
    raise ValueError(f"Unknown Fleet training derivation behavior: {behavior}")


def _training_feature_facts(
    *,
    base_facts: dict[str, Any],
    behavior: str,
    variant: str,
    replica_index: int | None,
    conditions: dict[str, bool],
    identity_registry: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Add external identifiers needed by supported compositional conditions."""

    facts = json.loads(json.dumps(base_facts, ensure_ascii=False))

    def fact(kind: str) -> str:
        return _orchestration_training_fact_id(
            behavior=behavior,
            variant=variant,
            replica_index=replica_index,
            fact_kind=kind,
            identity_registry=identity_registry,
        )

    # Context handoff follows the frozen schema: its request event is keyed by
    # the approved action, not by an unrelated request identity.
    if (
        behavior != "context-handoff"
        or conditions["requestNormalizationRequired"]
    ):
        facts["requestIdentifier"] = fact("request")
    if behavior == "duplicate-suppression":
        facts["sharedWorkKey"] = fact("shared-work")
    if behavior == "nonexistent-slot-negative":
        facts["requestedSlotIdentifier"] = fact("requested-unlisted-slot")
    if behavior in {"sequential-dependencies", "parallel-dependencies"}:
        facts["peerContext"] = {
            slot_id: [
                fact(f"peer-context-{slot_id}-{index}")
                for index, _ in enumerate(context_keys, start=1)
            ]
            for slot_id, context_keys in facts["peerContext"].items()
        }
    if behavior == "context-handoff":
        facts["allowedExecutorContext"] = [
            fact(f"allowed-executor-context-{index}")
            for index, _ in enumerate(
                facts["allowedExecutorContext"],
                start=1,
            )
        ]
        facts["forbiddenExecutorContext"] = [
            fact(f"forbidden-executor-context-{index}")
            for index, _ in enumerate(
                facts["forbiddenExecutorContext"],
                start=1,
            )
        ]
    if behavior == "aggregation-owner":
        facts["renderContext"] = [
            fact(f"render-context-{index}")
            for index, _ in enumerate(facts["renderContext"], start=1)
        ]
    if conditions["trustedContextSnapshotProvided"]:
        facts.update(
            {
                "trustedContextSnapshotIdentifier": fact(
                    "trusted-context-snapshot"
                ),
                "trustedEvidenceIdentifier": fact("trusted-evidence"),
            }
        )
    if conditions["executorObservationProvided"]:
        facts["executorObservationIdentifier"] = fact("executor-observation")
    if conditions["parallelJoinRequired"]:
        facts.update(
            {
                "parallelBranchIdentifiers": [
                    fact("parallel-executor-branch"),
                    fact("parallel-mimicry-branch"),
                ],
                "joinIdentifier": fact("parallel-join"),
            }
        )
    if conditions["contextBoundaryReviewRequired"]:
        facts.update(
            {
                "approvedActionIdentifier": fact("approved-action"),
                "executorResultIdentifier": fact("executor-result"),
            }
        )
    if conditions["candidateBranchesProvided"]:
        facts["candidateBranchIdentifiers"] = [
            fact("candidate-branch-a"),
            fact("candidate-branch-b"),
        ]
    if (
        conditions["aggregationInputVerificationRequired"]
        or conditions["responseValidationRequired"]
    ):
        result_ids = {
            "executor": fact("aggregation-executor-result"),
            "mimicry": fact("aggregation-mimicry-result"),
        }
        facts["availableResultIdentifiersBySlot"] = result_ids
        facts["verifiedInputResultIdentifiers"] = [
            result_ids["executor"],
            result_ids["mimicry"],
        ]
        facts["responseIdentifier"] = fact("validated-response")
    if conditions["approvalPolicyEvaluationRequired"]:
        facts.update(
            {
                "approvalPolicySnapshotIdentifier": fact(
                    "approval-policy-snapshot"
                ),
                "userApprovalRequestIdentifier": fact("user-approval-request"),
            }
        )
    if conditions["permissionPreflightRequired"]:
        facts["permissionCheckIdentifier"] = fact("permission-check")
    if conditions["slotDirectorySnapshotProvided"]:
        facts["slotDirectorySnapshotIdentifier"] = fact(
            "slot-directory-snapshot"
        )
    if conditions["rejectionRecordRequired"]:
        facts["rejectionIdentifier"] = fact("rejection-record")
    if conditions["policyAuditRequired"]:
        facts.update(
            {
                "policyAuditSnapshotIdentifier": fact(
                    "policy-audit-snapshot"
                ),
                "completionAuditRecordIdentifier": fact(
                    "completion-audit-record"
                ),
            }
        )
    return facts


def _training_orchestration_derivation_contract(
    scenario: dict[str, Any],
    *,
    facts: dict[str, Any] | None = None,
) -> dict[str, Any]:
    graph = scenario["graph"]
    variant = str(scenario.get("trainingMatrixVariant") or "")
    if variant not in {
        "core",
        ORCHESTRATION_BEHAVIOR_CONDITIONED_VARIANT,
        "normalized-intake",
        "policy-audited",
        "normalization-policy-audited",
    }:
        raise ValueError(f"Unknown Fleet training matrix variant: {variant}")
    return {
        "schemaVersion": ORCHESTRATION_DERIVATION_SCHEMA_VERSION,
        "eventIDGrammar": ORCHESTRATION_EVENT_ID_GRAMMAR,
        "graphSchemaVersion": graph["graphSchemaVersion"],
        "scenarioID": graph["scenarioID"],
        "behaviorClass": str(scenario["behaviorClass"]),
        "trainingMatrixVariant": variant,
        "trainingIdentityLane": "sft",
        "behaviorConditionedInstanceIndex": scenario.get(
            "behaviorConditionedInstanceIndex"
        ),
        "policyConditions": _orchestration_policy_conditions(
            behavior=str(scenario["behaviorClass"]),
            training_variant=variant,
        ),
        "knownSlotIDs": graph["knownSlotIDs"],
        "facts": facts if facts is not None else _training_orchestration_facts(scenario),
    }


def _orchestration_derivation_contract(
    scenario: dict[str, Any],
) -> dict[str, Any]:
    """Extract only external request/state facts from a frozen graph."""

    persisted = scenario.get("canonicalDerivation")
    if isinstance(persisted, dict):
        return json.loads(json.dumps(persisted, ensure_ascii=False))
    if scenario.get("trainingMatrixVariant") is not None:
        return _training_orchestration_derivation_contract(scenario)

    graph = scenario["graph"]
    behavior_class = str(scenario["behaviorClass"])
    request = _one_event(graph, "request_received")
    facts: dict[str, Any]
    if behavior_class == "no-delegation":
        snapshot = _one_event(graph, "trusted_context_snapshot_loaded")
        evidence = _one_event(graph, "trusted_context_verified")
        facts = {
            "requestIdentifier": request["requestID"],
            "trustedContextSnapshotIdentifier": snapshot["contextSnapshotID"],
            "trustedEvidenceIdentifier": evidence["evidenceID"],
            "trustedEvidenceStatus": evidence["evidenceStatus"],
        }
    elif behavior_class == "sequential-dependencies":
        observation = _one_event(graph, "result_received")
        facts = {
            "requestIdentifier": request["requestID"],
            "executorObservationIdentifier": observation["observationID"],
            "peerContext": {
                slot_id: _delegation_to(graph, slot_id)["contextKeys"]
                for slot_id in ("cortex", "executor", "mouth")
            },
        }
    elif behavior_class == "parallel-dependencies":
        join = _one_event(graph, "branch_join_verified")
        facts = {
            "requestIdentifier": request["requestID"],
            "parallelBranchIdentifiers": list(join["branchIDs"]),
            "joinIdentifier": join["joinID"],
            "peerContext": {
                slot_id: _delegation_to(graph, slot_id)["contextKeys"]
                for slot_id in ("cortex", "executor", "mimicry", "mouth")
            },
        }
    elif behavior_class == "context-handoff":
        boundary = _one_event(graph, "context_boundary_checked")
        result = _one_event(graph, "result_received")
        facts = {
            "approvedActionIdentifier": request["actionID"],
            "allowedExecutorContext": boundary["allowedContextKeys"],
            "forbiddenExecutorContext": boundary["excludes"],
            "executorResultIdentifier": result["resultID"],
        }
    elif behavior_class == "duplicate-suppression":
        candidates = _events_of_type(graph, "work_candidate_identified")
        dispatch = _one_event(graph, "delegate")
        facts = {
            "requestIdentifier": request["requestID"],
            "candidateBranchIdentifiers": [
                candidate["branchID"] for candidate in candidates
            ],
            "workOwnerSlot": dispatch["targetSlotID"],
            "sharedWorkKey": dispatch["workKey"],
        }
    elif behavior_class == "aggregation-owner":
        available = _events_of_type(graph, "result_available")
        verified = _one_event(graph, "aggregation_inputs_verified")
        response = _one_event(graph, "response_validated")
        facts = {
            "requestIdentifier": request["requestID"],
            "availableResultIdentifiersBySlot": {
                event["sourceSlotID"]: event["resultID"]
                for event in available
            },
            "verifiedInputResultIdentifiers": verified["inputResultIDs"],
            "renderContext": _delegation_to(graph, "mouth")["contextKeys"],
            "responseIdentifier": response["responseID"],
        }
    elif behavior_class == "approval-boundary":
        policy = _one_event(graph, "approval_policy_evaluated")
        approval = _one_event(graph, "request_user_approval")
        facts = {
            "requestIdentifier": request["requestID"],
            "toolIdentifier": request["toolID"],
            "approvalState": policy["approvalState"],
            "approvalPolicySnapshotIdentifier": policy["policySnapshotID"],
            "userApprovalRequestIdentifier": approval["approvalRequestID"],
        }
    elif behavior_class == "unavailable-boundary":
        check = _one_event(graph, "permission_state_checked")
        facts = {
            "requestIdentifier": request["requestID"],
            "toolIdentifier": request["toolID"],
            "permissionCheckIdentifier": check["permissionCheckID"],
            "permissionKey": check["permissionKey"],
            "permissionState": check["permissionState"],
        }
    elif behavior_class == "nonexistent-slot-negative":
        snapshot = _one_event(graph, "slot_directory_snapshot_loaded")
        rejection = _one_event(graph, "rejection_recorded")
        facts = {
            "requestIdentifier": request["requestID"],
            "requestedSlotIdentifier": request["requestedSlotID"],
            "slotDirectorySnapshotIdentifier": snapshot["directorySnapshotID"],
            "rejectionIdentifier": rejection["rejectionID"],
        }
    else:
        raise ValueError(f"Unknown Fleet derivation behavior: {behavior_class}")

    return {
        "schemaVersion": ORCHESTRATION_DERIVATION_SCHEMA_VERSION,
        "eventIDGrammar": ORCHESTRATION_EVENT_ID_GRAMMAR,
        "graphSchemaVersion": graph["graphSchemaVersion"],
        "scenarioID": graph["scenarioID"],
        "behaviorClass": behavior_class,
        "policyConditions": _orchestration_policy_conditions(
            behavior=behavior_class,
            training_variant=None,
        ),
        "knownSlotIDs": graph["knownSlotIDs"],
        "facts": facts,
    }


def _derive_training_core_orchestration_graph(
    *,
    scenario_id: str,
    known_slots: list[str],
    behavior: str,
    facts: dict[str, Any],
) -> dict[str, Any]:  # NOSONAR
    """Apply the canonical training policy to semantic state facts."""

    if behavior == "no-delegation":
        events = [
            _orchestration_event("request", "request_received"),
            _orchestration_event(
                "evidence",
                "trusted_context_verified",
                evidenceStatus=facts["trustedEvidenceStatus"],
            ),
            _orchestration_event("stop", "stop", reason="trusted_context_complete"),
        ]
        edges = [
            _orchestration_dependency("request", "evidence"),
            _orchestration_dependency("evidence", "stop"),
        ]
        strategy, delegated, owner, stop = (
            "no_delegation", [], None, "trusted_context_complete"
        )
    elif behavior == "sequential-dependencies":
        context = facts["peerContext"]
        events = [
            _orchestration_event("request", "request_received"),
            _orchestration_event("plan", "delegate", targetSlotID="cortex", contextKeys=context["cortex"]),
            _orchestration_event("execute", "delegate", targetSlotID="executor", contextKeys=context["executor"]),
            _orchestration_event("respond", "delegate", targetSlotID="mouth", contextKeys=context["mouth"]),
            _orchestration_event("stop", "stop", reason="grounded_response_complete"),
        ]
        edges = [
            _orchestration_dependency("request", "plan"),
            _orchestration_dependency("plan", "execute"),
            _orchestration_dependency("execute", "respond"),
            _orchestration_dependency("respond", "stop"),
        ]
        strategy, delegated, owner, stop = (
            "sequential", ["cortex", "executor", "mouth"], "mouth",
            "grounded_response_complete",
        )
    elif behavior == "parallel-dependencies":
        context = facts["peerContext"]
        events = [
            _orchestration_event("request", "request_received"),
            _orchestration_event("plan", "delegate", targetSlotID="cortex", contextKeys=context["cortex"]),
            _orchestration_event("execute", "delegate", targetSlotID="executor", contextKeys=context["executor"]),
            _orchestration_event("style", "delegate", targetSlotID="mimicry", contextKeys=context["mimicry"]),
            _orchestration_event("aggregate", "delegate", targetSlotID="mouth", contextKeys=context["mouth"]),
            _orchestration_event("stop", "stop", reason="parallel_results_aggregated"),
        ]
        edges = [
            _orchestration_dependency("request", "plan"),
            _orchestration_dependency("plan", "execute"),
            _orchestration_dependency("plan", "style"),
            _orchestration_dependency("execute", "aggregate"),
            _orchestration_dependency("style", "aggregate"),
            _orchestration_dependency("aggregate", "stop"),
        ]
        strategy, delegated, owner, stop = (
            "parallel_then_aggregate",
            ["cortex", "executor", "mimicry", "mouth"],
            "mouth",
            "parallel_results_aggregated",
        )
    elif behavior == "context-handoff":
        allowed = facts["allowedExecutorContext"]
        forbidden = facts["forbiddenExecutorContext"]
        events = [
            _orchestration_event("request", "request_received"),
            _orchestration_event("handoff", "delegate", targetSlotID="executor", contextKeys=allowed, excludes=forbidden),
            _orchestration_event("result", "result_received", sourceSlotID="executor"),
            _orchestration_event("stop", "stop", reason="bounded_handoff_complete"),
        ]
        edges = [
            _orchestration_dependency("request", "handoff"),
            _orchestration_dependency("handoff", "result"),
            _orchestration_dependency("result", "stop"),
        ]
        strategy, delegated, owner, stop = (
            "bounded_handoff", ["executor"], None, "bounded_handoff_complete"
        )
    elif behavior == "duplicate-suppression":
        target = facts["workOwnerSlot"]
        work_key = facts["sharedWorkKey"]
        events = [
            _orchestration_event("request", "request_received"),
            _orchestration_event("first", "delegate", targetSlotID=target, workKey=work_key),
            _orchestration_event("duplicate", "duplicate_suppressed", targetSlotID=target, workKey=work_key),
            _orchestration_event("result", "result_received", sourceSlotID=target, workKey=work_key),
            _orchestration_event("stop", "stop", reason="unique_work_complete"),
        ]
        edges = [
            _orchestration_dependency("request", "first"),
            _orchestration_dependency("first", "duplicate"),
            _orchestration_dependency("first", "result"),
            _orchestration_dependency("duplicate", "stop"),
            _orchestration_dependency("result", "stop"),
        ]
        strategy, delegated, owner, stop = (
            "deduplicated", [target], None, "unique_work_complete"
        )
    elif behavior == "aggregation-owner":
        result_slots = list(facts["availableResultIdentifiersBySlot"])
        if result_slots != ["executor", "mimicry"]:
            raise ValueError("Fleet aggregation training facts have invalid result slots")
        events = [
            _orchestration_event("request", "request_received"),
            _orchestration_event("tool-result", "result_available", sourceSlotID=result_slots[0]),
            _orchestration_event("style-result", "result_available", sourceSlotID=result_slots[1]),
            _orchestration_event("aggregate", "delegate", targetSlotID="mouth", contextKeys=facts["renderContext"]),
            _orchestration_event("stop", "stop", reason="single_owner_finalized"),
        ]
        edges = [
            _orchestration_dependency("request", "tool-result"),
            _orchestration_dependency("request", "style-result"),
            _orchestration_dependency("tool-result", "aggregate"),
            _orchestration_dependency("style-result", "aggregate"),
            _orchestration_dependency("aggregate", "stop"),
        ]
        strategy, delegated, owner, stop = (
            "aggregate", ["mouth"], "mouth", "single_owner_finalized"
        )
    elif behavior == "approval-boundary":
        tool = facts["toolIdentifier"]
        events = [
            _orchestration_event("request", "request_received", toolID=tool),
            _orchestration_event("boundary", "approval_boundary", toolID=tool, approvalState=facts["approvalState"]),
            _orchestration_event("approval", "request_user_approval", toolID=tool),
            _orchestration_event("stop", "stop", reason="awaiting_user_approval"),
        ]
        edges = [
            _orchestration_dependency("request", "boundary"),
            _orchestration_dependency("boundary", "approval"),
            _orchestration_dependency("approval", "stop"),
        ]
        strategy, delegated, owner, stop = (
            "approval_boundary", [], None, "awaiting_user_approval"
        )
    elif behavior == "unavailable-boundary":
        tool = facts["toolIdentifier"]
        events = [
            _orchestration_event("request", "request_received", toolID=tool),
            _orchestration_event(
                "availability", "capability_unavailable", toolID=tool,
                permissionKey=facts["permissionKey"],
                permissionState=facts["permissionState"],
            ),
            _orchestration_event("stop", "stop", reason="required_capability_unavailable"),
        ]
        edges = [
            _orchestration_dependency("request", "availability"),
            _orchestration_dependency("availability", "stop"),
        ]
        strategy, delegated, owner, stop = (
            "unavailable_boundary", [], None, "required_capability_unavailable"
        )
    elif behavior == "nonexistent-slot-negative":
        requested = facts["requestedSlotIdentifier"]
        events = [
            _orchestration_event("request", "request_received", requestedSlotID=requested),
            _orchestration_event("directory", "slot_directory_checked", requestedSlotID=requested, slotExists=False),
            _orchestration_event("reject", "invalid_slot_rejected", requestedSlotID=requested),
            _orchestration_event("stop", "stop", reason="requested_slot_not_manifested"),
        ]
        edges = [
            _orchestration_dependency("request", "directory"),
            _orchestration_dependency("directory", "reject"),
            _orchestration_dependency("reject", "stop"),
        ]
        strategy, delegated, owner, stop = (
            "reject_invalid_slot", [], None, "requested_slot_not_manifested"
        )
    else:
        raise ValueError(f"Unknown Fleet training derivation behavior: {behavior}")

    return _orchestration_graph(
        scenario_id=scenario_id,
        known_slot_ids=known_slots,
        strategy=strategy,
        events=events,
        dependencies=edges,
        delegated_slot_ids=delegated,
        aggregation_owner_slot_id=owner,
        stop_reason=stop,
    )


def _derive_orchestration_graph_from_contract(
    derivation: dict[str, Any],
) -> dict[str, Any]:  # NOSONAR
    """Rebuild the exact graph from canonical policy plus external facts."""

    if derivation.get("schemaVersion") != ORCHESTRATION_DERIVATION_SCHEMA_VERSION:
        raise ValueError("Unsupported Fleet graph derivation schema")
    if derivation.get("eventIDGrammar") != ORCHESTRATION_EVENT_ID_GRAMMAR:
        raise ValueError("Unsupported Fleet event-ID grammar")
    scenario_id = str(derivation["scenarioID"])
    known_slots = list(derivation["knownSlotIDs"])
    behavior = str(derivation["behaviorClass"])
    training_variant = derivation.get("trainingMatrixVariant")
    training_identity_lane = derivation.get("trainingIdentityLane")
    training_instance_index = derivation.get(
        "behaviorConditionedInstanceIndex"
    )
    policy_conditions = derivation.get("policyConditions")
    facts = derivation["facts"]
    if not isinstance(facts, dict):
        raise ValueError("Fleet graph derivation facts must be an object")
    if (
        not isinstance(policy_conditions, dict)
        or set(policy_conditions) != set(_ORCHESTRATION_POLICY_CONDITION_KEYS)
        or not all(isinstance(value, bool) for value in policy_conditions.values())
    ):
        raise ValueError("Fleet graph policy conditions are invalid")
    normalization_required = policy_conditions["requestNormalizationRequired"]
    audit_required = policy_conditions["policyAuditRequired"]
    enabled_features = {
        key
        for key, enabled in policy_conditions.items()
        if enabled
        and key
        not in {"requestNormalizationRequired", "policyAuditRequired"}
    }
    expected_features = _HOLDOUT_POLICY_CONDITION_BY_BEHAVIOR.get(behavior)
    if expected_features is None:
        raise ValueError("Fleet graph behavior has no policy-condition contract")

    if training_variant is not None:
        if training_variant not in {
            "core",
            ORCHESTRATION_BEHAVIOR_CONDITIONED_VARIANT,
            "normalized-intake",
            "policy-audited",
            "normalization-policy-audited",
        }:
            raise ValueError("Fleet training derivation variant is invalid")
        if training_identity_lane not in {"sft", "dpo"}:
            raise ValueError("Fleet training derivation identity lane is invalid")
        replica_index: int | None = None
        if training_variant == ORCHESTRATION_BEHAVIOR_CONDITIONED_VARIANT:
            if (
                not isinstance(training_instance_index, int)
                or isinstance(training_instance_index, bool)
                or training_instance_index < 1
                or training_instance_index
                > ORCHESTRATION_BEHAVIOR_CONDITIONED_REPLICAS
            ):
                raise ValueError(
                    "Fleet conditioned derivation instance index is invalid"
                )
            replica_index = training_instance_index - 1
        elif training_instance_index is not None:
            raise ValueError(
                "Fleet non-conditioned derivation has an instance index"
            )
        expected_scenario_id = _orchestration_training_scenario_id(
            behavior=behavior,
            variant=training_variant,
            replica_index=replica_index,
            lane=str(training_identity_lane),
        )
        if (
            not _is_opaque_orchestration_training_scenario_id(scenario_id)
            or scenario_id != expected_scenario_id
        ):
            raise ValueError("Fleet training scenario identity is invalid")
        expected_training_conditions = _orchestration_policy_conditions(
            behavior=behavior,
            training_variant=str(training_variant),
        )
        if policy_conditions != expected_training_conditions:
            raise ValueError("Fleet training policy-condition combination is invalid")
    elif (
        training_identity_lane is not None
        or training_instance_index is not None
    ):
        raise ValueError("Fleet non-training derivation has training identity fields")

    if normalization_required or audit_required:
        variant = str(training_variant or "")
        if not variant:
            variant = (
                "normalization-policy-audited"
                if normalization_required and audit_required
                else "normalized-intake"
                if normalization_required
                else "policy-audited"
            )
        expected_training_conditions = _orchestration_policy_conditions(
            behavior=behavior,
            training_variant=variant,
        )
        if policy_conditions != expected_training_conditions:
            raise ValueError("Fleet training policy-condition combination is invalid")
        core_graph = _derive_training_core_orchestration_graph(
            scenario_id=scenario_id,
            known_slots=known_slots,
            behavior=behavior,
            facts=facts,
        )
        variant_graph = _orchestration_training_variant_graph(
            core_graph,
            scenario_id=scenario_id,
            event_namespace=scenario_id,
            request_id=facts.get("requestIdentifier"),
            variant=variant,
            policy_snapshot_id=facts.get(
                "policyAuditSnapshotIdentifier"
            ),
            completion_record_id=facts.get(
                "completionAuditRecordIdentifier"
            ),
        )
        return _apply_training_policy_condition_support(
            variant_graph,
            behavior=behavior,
            conditions=policy_conditions,
            facts=facts,
        )
    if enabled_features != set(expected_features):
        raise ValueError("Fleet holdout policy-condition combination is invalid")

    def event(index: int, event_type: str, **payload: Any) -> dict[str, Any]:
        return _orchestration_event(f"e{index:02d}", event_type, **payload)

    def dependency(source: int, target: int) -> dict[str, str]:
        return _orchestration_dependency(f"e{source:02d}", f"e{target:02d}")

    if behavior == "no-delegation":
        events = [
            event(1, "request_received", requestID=facts["requestIdentifier"]),
            event(
                2,
                "trusted_context_snapshot_loaded",
                contextSnapshotID=facts["trustedContextSnapshotIdentifier"],
            ),
            event(
                3,
                "trusted_context_verified",
                evidenceID=facts["trustedEvidenceIdentifier"],
                evidenceStatus=facts["trustedEvidenceStatus"],
            ),
            event(4, "stop", reason="trusted_context_complete"),
        ]
        edges = [dependency(1, 2), dependency(2, 3), dependency(3, 4)]
        strategy, delegated, owner, stop = (
            "no_delegation",
            [],
            None,
            "trusted_context_complete",
        )
    elif behavior == "sequential-dependencies":
        context = facts["peerContext"]
        events = [
            event(1, "request_received", requestID=facts["requestIdentifier"]),
            event(2, "delegate", targetSlotID="cortex", contextKeys=context["cortex"]),
            event(3, "delegate", targetSlotID="executor", contextKeys=context["executor"]),
            event(
                4,
                "result_received",
                sourceSlotID="executor",
                observationID=facts["executorObservationIdentifier"],
            ),
            event(5, "delegate", targetSlotID="mouth", contextKeys=context["mouth"]),
            event(6, "stop", reason="grounded_response_complete"),
        ]
        edges = [dependency(i, i + 1) for i in range(1, 6)]
        strategy, delegated, owner, stop = (
            "sequential",
            ["cortex", "executor", "mouth"],
            "mouth",
            "grounded_response_complete",
        )
    elif behavior == "parallel-dependencies":
        context = facts["peerContext"]
        branches = facts["parallelBranchIdentifiers"]
        events = [
            event(1, "request_received", requestID=facts["requestIdentifier"]),
            event(2, "delegate", targetSlotID="cortex", contextKeys=context["cortex"]),
            event(3, "delegate", targetSlotID="executor", branchID=branches[0], contextKeys=context["executor"]),
            event(4, "delegate", targetSlotID="mimicry", branchID=branches[1], contextKeys=context["mimicry"]),
            event(5, "branch_join_verified", branchIDs=branches, joinID=facts["joinIdentifier"]),
            event(6, "delegate", targetSlotID="mouth", contextKeys=context["mouth"]),
            event(7, "stop", reason="parallel_results_aggregated"),
        ]
        edges = [
            dependency(1, 2), dependency(2, 3), dependency(2, 4),
            dependency(3, 5), dependency(4, 5), dependency(5, 6),
            dependency(6, 7),
        ]
        strategy, delegated, owner, stop = (
            "parallel_then_aggregate",
            ["cortex", "executor", "mimicry", "mouth"],
            "mouth",
            "parallel_results_aggregated",
        )
    elif behavior == "context-handoff":
        allowed = facts["allowedExecutorContext"]
        forbidden = facts["forbiddenExecutorContext"]
        events = [
            event(1, "request_received", actionID=facts["approvedActionIdentifier"]),
            event(2, "context_boundary_checked", allowedContextKeys=allowed, excludes=forbidden),
            event(3, "delegate", targetSlotID="executor", contextKeys=allowed, excludes=forbidden),
            event(4, "result_received", sourceSlotID="executor", resultID=facts["executorResultIdentifier"]),
            event(5, "stop", reason="bounded_handoff_complete"),
        ]
        edges = [dependency(i, i + 1) for i in range(1, 5)]
        strategy, delegated, owner, stop = (
            "bounded_handoff", ["executor"], None, "bounded_handoff_complete"
        )
    elif behavior == "duplicate-suppression":
        branches = facts["candidateBranchIdentifiers"]
        target = facts["workOwnerSlot"]
        work_key = facts["sharedWorkKey"]
        events = [
            event(1, "request_received", requestID=facts["requestIdentifier"]),
            event(2, "work_candidate_identified", branchID=branches[0], targetSlotID=target, workKey=work_key),
            event(3, "delegate", targetSlotID=target, workKey=work_key),
            event(4, "work_candidate_identified", branchID=branches[1], targetSlotID=target, workKey=work_key),
            event(5, "duplicate_suppressed", targetSlotID=target, workKey=work_key),
            event(6, "result_received", sourceSlotID=target, workKey=work_key),
            event(7, "stop", reason="unique_work_complete"),
        ]
        edges = [
            dependency(1, 2), dependency(1, 4), dependency(2, 3),
            dependency(3, 6), dependency(4, 5), dependency(5, 7),
            dependency(6, 7),
        ]
        strategy, delegated, owner, stop = (
            "deduplicated", [target], None, "unique_work_complete"
        )
    elif behavior == "aggregation-owner":
        results = facts["availableResultIdentifiersBySlot"]
        events = [
            event(1, "request_received", requestID=facts["requestIdentifier"]),
            event(2, "result_available", resultID=results["executor"], sourceSlotID="executor"),
            event(3, "result_available", resultID=results["mimicry"], sourceSlotID="mimicry"),
            event(4, "aggregation_inputs_verified", inputResultIDs=facts["verifiedInputResultIdentifiers"]),
            event(5, "delegate", targetSlotID="mouth", contextKeys=facts["renderContext"]),
            event(6, "response_validated", responseID=facts["responseIdentifier"], sourceSlotID="mouth"),
            event(7, "stop", reason="single_owner_finalized"),
        ]
        edges = [
            dependency(1, 2), dependency(1, 3), dependency(2, 4),
            dependency(3, 4), dependency(4, 5), dependency(5, 6),
            dependency(6, 7),
        ]
        strategy, delegated, owner, stop = (
            "aggregate", ["mouth"], "mouth", "single_owner_finalized"
        )
    elif behavior == "approval-boundary":
        tool = facts["toolIdentifier"]
        events = [
            event(1, "request_received", requestID=facts["requestIdentifier"], toolID=tool),
            event(2, "approval_policy_evaluated", approvalState=facts["approvalState"], policySnapshotID=facts["approvalPolicySnapshotIdentifier"], toolID=tool),
            event(3, "approval_boundary", approvalState="required", toolID=tool),
            event(4, "request_user_approval", approvalRequestID=facts["userApprovalRequestIdentifier"], toolID=tool),
            event(5, "stop", reason="awaiting_user_approval"),
        ]
        edges = [dependency(i, i + 1) for i in range(1, 5)]
        strategy, delegated, owner, stop = (
            "approval_boundary", [], None, "awaiting_user_approval"
        )
    elif behavior == "unavailable-boundary":
        tool = facts["toolIdentifier"]
        permission_key = facts["permissionKey"]
        permission_state = facts["permissionState"]
        events = [
            event(1, "request_received", requestID=facts["requestIdentifier"], toolID=tool),
            event(2, "permission_state_checked", permissionCheckID=facts["permissionCheckIdentifier"], permissionKey=permission_key, permissionState=permission_state, toolID=tool),
            event(3, "capability_unavailable", permissionKey=permission_key, permissionState=permission_state, toolID=tool),
            event(4, "stop", reason="required_capability_unavailable"),
        ]
        edges = [dependency(1, 2), dependency(2, 3), dependency(3, 4)]
        strategy, delegated, owner, stop = (
            "unavailable_boundary", [], None, "required_capability_unavailable"
        )
    elif behavior == "nonexistent-slot-negative":
        requested = facts["requestedSlotIdentifier"]
        events = [
            event(1, "request_received", requestID=facts["requestIdentifier"], requestedSlotID=requested),
            event(2, "slot_directory_snapshot_loaded", directorySnapshotID=facts["slotDirectorySnapshotIdentifier"]),
            event(3, "slot_directory_checked", requestedSlotID=requested, slotExists=False),
            event(4, "invalid_slot_rejected", requestedSlotID=requested),
            event(5, "rejection_recorded", rejectionID=facts["rejectionIdentifier"], requestedSlotID=requested),
            event(6, "stop", reason="requested_slot_not_manifested"),
        ]
        edges = [dependency(i, i + 1) for i in range(1, 6)]
        strategy, delegated, owner, stop = (
            "reject_invalid_slot", [], None, "requested_slot_not_manifested"
        )
    else:
        raise ValueError(f"Unknown Fleet derivation behavior: {behavior}")

    return _orchestration_graph(
        scenario_id=scenario_id,
        known_slot_ids=known_slots,
        strategy=strategy,
        events=events,
        dependencies=edges,
        delegated_slot_ids=delegated,
        aggregation_owner_slot_id=owner,
        stop_reason=stop,
    )


def generate_manifest_markdown(manifest: AgentBehaviorManifest) -> str:
    lines: list[str] = []
    source_map = _source_code_map(manifest)
    lines.append(f"# {manifest.app.name} Agent Behavior Manifest")
    lines.append("")
    lines.append("## Source Integrity")
    lines.append(f"- Base commit: `{manifest.sourceIntegrity.baseCommit or 'unknown'}`")
    lines.append(
        f"- Working-tree digest: `{manifest.sourceIntegrity.workingTreeDigest or 'unknown'}`"
    )
    lines.append(f"- Dirty source state: `{manifest.sourceIntegrity.dirtyState}`")
    lines.append(f"- Source files: {len(manifest.sourceIntegrity.files)}")
    if source_map["files"]:
        lines.append("- Source map:")
        for entry in source_map["files"][:80]:
            lines.append(f"  - `{entry['path']}`: {', '.join(entry['domains']) or 'general'}")
        if len(source_map["files"]) > 80:
            lines.append(f"  - ... {len(source_map['files']) - 80} more files omitted from Markdown summary")
    lines.append("")
    lines.append("## System Identity")
    lines.append("- Lumen is one logical agent composed of specialized model slots.")
    lines.append("- Each slot must know its own contract, peer slot contracts, routing boundaries, source-code origin, and the public map of the codebase extracted into this manifest.")
    lines.append("- Slots must coordinate as one coherent entity instead of acting like unrelated assistants.")
    lines.append("")
    lines.append("## Model Fleet Slots")
    lines.append(f"- Contract version: `{manifest.fleet.contractVersion}`")
    for slot in sorted(manifest.fleet.slots, key=lambda item: item.id):
        topology = manifest.fleetTopology.slots.get(slot.id)
        lines.append(f"### `{slot.id}`")
        lines.append(f"- Role: {slot.role}")
        lines.append(f"- Source: `{slot.source or 'unknown'}`")
        lines.append(f"- Purpose: {(topology.purpose if topology else _slot_purpose_fallback(slot))}")
        if slot.responsibilities:
            lines.append("- Responsibilities:")
            for responsibility in sorted(slot.responsibilities):
                lines.append(f"  - {responsibility}")
        if topology:
            lines.append(f"- Accepts: {topology.inputSignature}")
            lines.append(f"- Returns: {topology.outputSignature}")
            lines.append(f"- Calls: {', '.join(topology.calls) or 'none'}")
            lines.append(f"- Called by: {', '.join(topology.calledBy) or 'none'}")
        lines.append("")

    lines.append("## Tools")
    for tool in sorted(manifest.tools, key=lambda item: item.id):
        lines.append(f"### `{tool.id}`")
        lines.append(f"- Display name: {tool.displayName or tool.id}")
        lines.append(f"- Description: {tool.description or 'No description extracted.'}")
        lines.append(f"- Source: `{tool.source or tool.inferredSource or 'unknown'}`")
        lines.append(f"- Inferred: {tool.inferred}")
        lines.append(f"- Requires approval: {tool.requiresApproval}")
        lines.append(f"- Permission key: {tool.permissionKey or 'none'}")
        if tool.arguments:
            lines.append("- Arguments:")
            for argument in tool.arguments:
                required = "required" if argument.required else "optional"
                source = f" Source: `{argument.source}`." if argument.source else ""
                lines.append(f"  - `{argument.name}`: {argument.type}, {required}. {argument.description or ''}{source}".rstrip())
        else:
            lines.append("- Arguments: none")
        if tool.description:
            lines.append(f"- Example: Use `{tool.id}` only when the user intent maps to this manifest tool and all required arguments are known.")
        lines.append("")

    lines.append("## UserIntents")
    for intent in sorted(manifest.intents, key=lambda item: item.id):
        lines.append(f"- `{intent.id}` → allowed tools: {', '.join(intent.allowedToolIDs) or 'none'}; source: `{intent.source or 'unknown'}`")
    lines.append("")

    lines.append("## Routing Rules")
    for entry in sorted(manifest.routingMatrix, key=lambda item: item.intent):
        lines.append(f"- `{entry.intent}` → allowed: {', '.join(entry.allowedTools) or 'none'}; forbidden examples: {', '.join(entry.forbiddenTools[:8]) or 'none'}")
    lines.append("")

    lines.append("## Memory Scopes")
    lines.append(f"- Scopes: {', '.join(sorted(manifest.memory.scopes)) or 'none'}")
    for freshness in sorted(manifest.memory.freshnessClasses, key=lambda item: item.id):
        ttl = "durable" if freshness.durable else f"ttlSeconds={freshness.ttlSeconds}"
        lines.append(f"- `{freshness.id}`: {ttl}; source: `{freshness.source or 'unknown'}`")
    lines.append("")

    lines.append("## Permissions")
    permission_tools = [tool for tool in sorted(manifest.tools, key=lambda item: item.id) if tool.permissionKey or tool.requiresApproval]
    if permission_tools:
        for tool in permission_tools:
            lines.append(f"- `{tool.id}`: permission={tool.permissionKey or 'none'}, requiresApproval={tool.requiresApproval}")
    else:
        lines.append("- No permission-bound tools extracted.")
    lines.append("")

    lines.append("## Sentinel Policy")
    for sentinel in sorted(manifest.sentinels.forbiddenInUserOutput):
        lines.append(f"- `{sentinel}` must never appear in user-visible output.")
    lines.append("")

    lines.append("## Fleet Topology")
    for slot_id, topology in sorted(manifest.fleetTopology.slots.items()):
        lines.append(f"- `{slot_id}` calls [{', '.join(topology.calls)}] and is called by [{', '.join(topology.calledBy)}].")
    lines.append(f"- External handoff tools: {', '.join(manifest.fleetTopology.externalHandoffTools) or 'none'}")
    lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _system_prompt_text(manifest: AgentBehaviorManifest, slot: ModelSlotManifest, payload: dict[str, Any]) -> str:
    directory_lines = [f"- {entry['slotID']} ({entry['role']}): {entry['purpose']}" for entry in payload["modelDirectory"]]
    tool_lines = [f"- {tool['id']}: {tool['description']} [source: {tool.get('source') or 'unknown'}]" for tool in payload["availableTools"]]
    route_lines = [f"- {route['intent']} -> {', '.join(route['allowedTools']) or 'no tool'}" for route in payload["routingRules"]]
    responsibility_lines = [f"- {item}" for item in payload["responsibilities"]] or ["- Follow the role contract extracted from the Swift source."]
    sentinel_lines = [f"- {sentinel}" for sentinel in payload["sentinelPolicy"].get("forbiddenInUserOutput", [])] or ["- none extracted"]
    source_lines = [f"- {entry['path']}: {', '.join(entry['domains']) or 'general'}" for entry in payload["sourceCodeMap"].get("files", [])[:30]]
    handoff_tools = payload.get("topology", {}).get("externalHandoffTools", []) if isinstance(payload.get("topology"), dict) else []
    handoff_line = "Use manifest-listed handoff tools when available; otherwise return a structured routing instruction for the host orchestrator, not a fake tool call."
    if handoff_tools:
        handoff_line = f"Use only these manifest-listed handoff tools for explicit slot delegation: {', '.join(handoff_tools)}."
    lines = [
        f"You are `{slot.id}`, the `{slot.role}` slot inside the unified {manifest.app.name} agent fleet.",
        "You are part of a single unified agent named Lumen.",
        "You are one component of a single logical agent named Lumen; do not act like a separate assistant.",
        f"Your purpose: {payload['purpose']}",
        f"Your Swift/source origin: {payload['slotSource'].get('source') or 'unknown'}.",
        "You have manifest-derived awareness of the codebase map, fleet topology, source lineage, tools, intents, routing rules, memory policy, and peer roles.",
        "This is not full raw source-code text. It is the extracted, hashed, public operational map of the code that defines your runtime contract.",
        "If a task is outside your scope, delegate or route using the fleet topology and approved manifest tools. Never invent a slot, tool, permission, memory scope, or source file.",
        handoff_line,
        "Never claim ignorance of other manifest-defined parts of the system; describe public peer capabilities from the model directory and route private work instead.",
        "Never claim access to private runtime state, hidden chain-of-thought, full user data, or raw source not present in the manifest/source map.",
        "",
        "Your responsibilities:",
        *responsibility_lines,
        "",
        "Your available tools:",
        *(tool_lines or ["- none directly assigned; route or delegate when needed."]),
        "",
        "Model directory:",
        *(directory_lines or ["- no peer slots extracted."]),
        "",
        "Routing rules:",
        *(route_lines or ["- no explicit routing matrix extracted; ask for clarification before acting outside scope."]),
        "",
        "Source-code map summary:",
        *(source_lines or ["- no source files extracted."]),
        "",
        "Memory scopes:",
        f"- {', '.join(payload['memory'].get('scopes', [])) or 'none'}",
        "",
        "Forbidden user-visible sentinels:",
        *sentinel_lines,
        "",
        "Return outputs that match your slot contract. Preserve the illusion of one coherent Lumen agent by coordinating with peers instead of improvising.",
    ]
    return "\n".join(lines)


def _fleet_whole_system_records(manifest: AgentBehaviorManifest) -> list[dict[str, Any]]:
    source_map = _source_code_map(manifest)
    payload = {
        "identity": "Lumen is one logical agent composed of specialized model slots.",
        "slotCount": len(manifest.fleet.slots),
        "toolCount": len(manifest.tools),
        "intentCount": len(manifest.intents),
        "sourceFileCount": len(manifest.sourceIntegrity.files),
        "fleetSlots": [_slot_source_payload(slot) for slot in sorted(manifest.fleet.slots, key=lambda item: item.id)],
        "sourceCodeMap": source_map,
        "rules": [
            "Use manifest-defined tools only.",
            "Use manifest-defined slots only.",
            "Delegate outside-scope work through the topology.",
            "Never expose private runtime state or hidden reasoning.",
            "Act as one coherent Lumen agent, not as unrelated sub-assistants.",
        ],
    }
    records: list[dict[str, Any]] = []
    for slot in sorted(manifest.fleet.slots, key=lambda item: item.id):
        records.append({
            "id": _record_id("whole-system", slot.id),
            "schemaVersion": "2.1.0",
            "recordType": "sft",
            "agentRole": slot.role,
            "taskType": "fleet_whole_system_identity",
            "messages": [
                {"role": "system", "content": f"You are {slot.id}. Answer from the manifest-derived source map and fleet topology."},
                {"role": "user", "content": "Explain how Lumen is one entity made of multiple agents, and summarize the source-code map you know."},
                {"role": "assistant", "content": json.dumps(payload, ensure_ascii=False, sort_keys=True)},
            ],
            "metadata": {"sourceFileCount": len(manifest.sourceIntegrity.files), "toolCount": len(manifest.tools)},
        })
    return records


def _serialize_orchestration_graph(graph: dict[str, Any]) -> str:
    """Serialize a graph in the runtime contract's canonical teaching order."""

    top_level_keys = {
        "graphSchemaVersion",
        "scenarioID",
        "knownSlotIDs",
        "events",
        "dependencies",
        "decision",
    }
    if set(graph) != top_level_keys:
        raise ValueError("Fleet graph has a noncanonical top-level schema")
    events = graph.get("events")
    dependencies = graph.get("dependencies")
    decision = graph.get("decision")
    if (
        not isinstance(events, list)
        or not isinstance(dependencies, list)
        or not isinstance(decision, dict)
    ):
        raise ValueError("Fleet graph has invalid ordered containers")

    ordered_events: list[dict[str, Any]] = []
    for event in events:
        if (
            not isinstance(event, dict)
            or not isinstance(event.get("id"), str)
            or not isinstance(event.get("type"), str)
        ):
            raise ValueError("Fleet graph has an invalid event object")
        ordered_events.append(
            {
                "id": event["id"],
                "type": event["type"],
                **{
                    key: event[key]
                    for key in sorted(set(event) - {"id", "type"})
                },
            }
        )

    ordered_dependencies: list[dict[str, Any]] = []
    for dependency in dependencies:
        if not isinstance(dependency, dict) or set(dependency) != {
            "fromEventID",
            "kind",
            "toEventID",
        }:
            raise ValueError("Fleet graph has an invalid dependency object")
        ordered_dependencies.append(
            {
                "fromEventID": dependency["fromEventID"],
                "kind": dependency["kind"],
                "toEventID": dependency["toEventID"],
            }
        )

    if set(decision) != {
        "strategy",
        "delegatedSlotIDs",
        "aggregationOwnerSlotID",
        "stopReason",
    }:
        raise ValueError("Fleet graph has an invalid decision object")
    ordered_graph = {
        "graphSchemaVersion": graph["graphSchemaVersion"],
        "scenarioID": graph["scenarioID"],
        "knownSlotIDs": graph["knownSlotIDs"],
        "events": ordered_events,
        "dependencies": ordered_dependencies,
        "decision": {
            "strategy": decision["strategy"],
            "delegatedSlotIDs": decision["delegatedSlotIDs"],
            "aggregationOwnerSlotID": decision["aggregationOwnerSlotID"],
            "stopReason": decision["stopReason"],
        },
    }
    return json.dumps(
        ordered_graph,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _serialize_orchestration_rejection(
    graph: dict[str, Any],
    *,
    contrast_mode: str | None,
    failure_family_mutation: str | None = None,
) -> str:
    if contrast_mode != ORCHESTRATION_CORE_FAILURE_CONTRAST_MODE:
        return _serialize_orchestration_graph(graph)
    if failure_family_mutation == "top_level_dependencies_omission":
        expected_keys = {
            "graphSchemaVersion",
            "scenarioID",
            "knownSlotIDs",
            "events",
            "decision",
        }
        if set(graph) != expected_keys:
            raise ValueError(
                "Fleet top-level omission rejection is not missing exactly "
                f"{_ORCHESTRATION_TOP_LEVEL_OMISSION_KEY!r}"
            )
        canonicalizable = json.loads(json.dumps(graph, ensure_ascii=False))
        canonicalizable[_ORCHESTRATION_TOP_LEVEL_OMISSION_KEY] = []
        ordered = json.loads(_serialize_orchestration_graph(canonicalizable))
        ordered.pop(_ORCHESTRATION_TOP_LEVEL_OMISSION_KEY)
        return json.dumps(ordered, ensure_ascii=False, separators=(",", ":"))
    if failure_family_mutation == "decision_aggregation_owner_omission":
        decision = graph.get("decision")
        if set(graph) != {
            "graphSchemaVersion",
            "scenarioID",
            "knownSlotIDs",
            "events",
            "dependencies",
            "decision",
        } or not isinstance(decision, dict) or set(decision) != {
            "strategy",
            "delegatedSlotIDs",
            "stopReason",
        }:
            raise ValueError(
                "Fleet decision omission rejection is not missing exactly "
                "'aggregationOwnerSlotID'"
            )
        canonicalizable = json.loads(json.dumps(graph, ensure_ascii=False))
        canonicalizable["decision"]["aggregationOwnerSlotID"] = None
        ordered = json.loads(_serialize_orchestration_graph(canonicalizable))
        ordered["decision"].pop("aggregationOwnerSlotID")
        return json.dumps(ordered, ensure_ascii=False, separators=(",", ":"))
    return _serialize_orchestration_graph(graph)


def _orchestration_training_records(manifest: AgentBehaviorManifest) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    identity_registry: dict[str, str] = {}
    for scenario in _orchestration_training_scenarios(
        manifest,
        identity_registry=identity_registry,
    ):
        derivation = _orchestration_derivation_contract(scenario)
        if _derive_orchestration_graph_from_contract(derivation) != scenario["graph"]:
            raise ValueError(
                "Fleet training derivation does not uniquely rebuild graph: "
                f"{scenario['id']}"
            )
        base_prompt = [
            {
                "role": "system",
                "content": (
                    "You are the Lumen Fleet orchestration policy. Return a manifest-grounded event graph. "
                    "Use only known slot IDs, preserve explicit dependencies and boundaries, and stop once the request is complete."
                ),
            },
            {
                "role": "user",
                "content": _orchestration_training_prompt(
                    scenario,
                    derivation,
                    lane="sft",
                ),
            },
        ]
        metadata = {
            **_native_orchestration_metadata(manifest, scenario["id"]),
            "behaviorClass": scenario["behaviorClass"],
            "trainingMatrixVariant": scenario["trainingMatrixVariant"],
            # All eight independent behavior-conditioned instances and the
            # canonical core are SFT optimizer-visible and DPO-visible. The
            # core SFT lane teaches initial generation; only its paired DPO
            # lane teaches strict retry recovery. The combined wrapper remains
            # validation-only. Frozen facts, prompts, IDs, and exact graphs
            # remain unseen.
            "requiredSplit": (
                "validation"
                if scenario["trainingMatrixVariant"] in {
                    "normalized-intake",
                    "policy-audited",
                    "normalization-policy-audited",
                }
                else "train"
            ),
            "trainingTopologySHA256": canonical_sha256(
                _orchestration_topology_contract(scenario["graph"])
            ),
            "derivationSchemaVersion": derivation["schemaVersion"],
            "canonicalDerivationSHA256": canonical_sha256(derivation),
            **_orchestration_generation_prompt_metadata(
                scenario,
                lane="sft",
            ),
        }
        atomic_mutation = scenario.get("atomicPreferenceMutation")
        sft_optimizer_visible = scenario.get("sftOptimizerVisible", True) is True
        metadata["sftOptimizerVisible"] = sft_optimizer_visible
        if isinstance(atomic_mutation, str):
            metadata["atomicPreferenceMutation"] = atomic_mutation
            metadata["behaviorConditionedInstanceIndex"] = scenario[
                "behaviorConditionedInstanceIndex"
            ]
            metadata["topologyCoverageMode"] = (
                "trained_policy_topology_unseen_frozen_instance"
            )
        if sft_optimizer_visible:
            records.append({
                "id": _record_id("orchestration-sft", scenario["id"]),
                "schemaVersion": "2.1.0",
                "recordType": "sft",
                "sourceFamily": "fleet_orchestration_native",
                "agentRole": "fleet",
                "taskType": "fleet_orchestration_event_graph",
                "messages": [
                    *base_prompt,
                    {
                        "role": "assistant",
                        "content": _serialize_orchestration_graph(
                            scenario["graph"]
                        ),
                    },
                ],
                "metadata": metadata,
            })
        rejected_graph = scenario.get("rejectedGraph")
        if isinstance(rejected_graph, dict):
            preference_scenario = _orchestration_preference_scenario(
                scenario,
                identity_registry=identity_registry,
            )
            preference_derivation = preference_scenario[
                "canonicalDerivation"
            ]
            if (
                _derive_orchestration_graph_from_contract(
                    preference_derivation
                )
                != preference_scenario["graph"]
            ):
                raise ValueError(
                    "Fleet preference derivation does not uniquely rebuild graph: "
                    f"{preference_scenario['id']}"
                )
            preference_prompt = [
                base_prompt[0],
                {
                    "role": "user",
                    "content": _orchestration_training_prompt(
                        preference_scenario,
                        preference_derivation,
                        lane="dpo",
                    ),
                },
            ]
            preference_metadata = {
                **metadata,
                **_native_orchestration_metadata(
                    manifest,
                    preference_scenario["id"],
                ),
                "trainingTopologySHA256": canonical_sha256(
                    _orchestration_topology_contract(
                        preference_scenario["graph"]
                    )
                ),
                "canonicalDerivationSHA256": canonical_sha256(
                    preference_derivation
                ),
                **_orchestration_generation_prompt_metadata(
                    preference_scenario,
                    lane="dpo",
                ),
                "preferenceSourceScenarioID": scenario["id"],
                "sftAnchorScenarioID": (
                    scenario["id"]
                    if sft_optimizer_visible
                    else _orchestration_training_scenario_id(
                        behavior=str(scenario["behaviorClass"]),
                        variant=ORCHESTRATION_BEHAVIOR_CONDITIONED_VARIANT,
                        replica_index=0,
                        identity_registry=identity_registry,
                    )
                ),
                "sftAnchorBindingMode": (
                    "exact_scenario"
                    if sft_optimizer_visible
                    else "topology_equivalent"
                ),
            }
            records.append({
                "id": _record_id(
                    "orchestration-dpo",
                    preference_scenario["id"],
                ),
                "schemaVersion": "2.1.0",
                "recordType": "dpo",
                "sourceFamily": "fleet_orchestration_native",
                "agentRole": "fleet",
                "taskType": "fleet_orchestration_event_graph_preference",
                "prompt": preference_prompt,
                "chosen": {
                    "role": "assistant",
                    "content": _serialize_orchestration_graph(
                        preference_scenario["graph"]
                    ),
                },
                "rejected": {
                    "role": "assistant",
                    "content": _serialize_orchestration_rejection(
                        preference_scenario["rejectedGraph"],
                        contrast_mode=preference_scenario.get(
                            "preferenceContrastMode"
                        ),
                        failure_family_mutation=preference_scenario.get(
                            "coreFailureFamilyMutation"
                        ),
                    ),
                },
                "metadata": {
                    **preference_metadata,
                    "preferenceType": "manifest_grounded_orchestration",
                    "lesson": scenario["preferenceLesson"],
                    **(
                        {
                            "preferenceContrastMode": preference_scenario[
                                "preferenceContrastMode"
                            ],
                        }
                        if isinstance(
                            preference_scenario.get("preferenceContrastMode"),
                            str,
                        )
                        else {}
                    ),
                    **(
                        {
                            "coreFailureFamilyMutation": preference_scenario[
                                "coreFailureFamilyMutation"
                            ],
                        }
                        if isinstance(
                            preference_scenario.get(
                                "coreFailureFamilyMutation"
                            ),
                            str,
                        )
                        else {}
                    ),
                },
            })
    return records


def _orchestration_training_prompt(
    scenario: dict[str, Any],
    derivation: dict[str, Any] | None = None,
    *,
    lane: str = "sft",
) -> str:
    """Teach canonical policy derivation from state facts, never graph copying."""

    prompt = _canonical_orchestration_prompt(
        scenario,
        derivation or _orchestration_derivation_contract(scenario),
    )
    if _orchestration_training_uses_retry_prompt(scenario, lane=lane):
        prompt += "\n\n" + generic_strict_json_retry_instruction("invalid_json")
    return prompt


def _orchestration_training_uses_retry_prompt(
    scenario: dict[str, Any],
    *,
    lane: str = "sft",
) -> bool:
    if lane not in {"sft", "dpo"}:
        raise ValueError(f"Unknown Fleet orchestration training lane: {lane}")
    variant = scenario.get("trainingMatrixVariant")
    return (lane == "dpo" and variant == "core") or (
        variant == ORCHESTRATION_BEHAVIOR_CONDITIONED_VARIANT
        and scenario.get("behaviorConditionedInstanceIndex")
        == ORCHESTRATION_BEHAVIOR_CONDITIONED_REPLICAS
    )


def _orchestration_generation_prompt_metadata(
    scenario: dict[str, Any],
    *,
    lane: str,
) -> dict[str, str]:
    if _orchestration_training_uses_retry_prompt(scenario, lane=lane):
        return {
            "generationPromptMode": "strict_json_retry",
            "retryFailureCode": "invalid_json",
        }
    return {"generationPromptMode": "initial_generation"}


_PREFERENCE_PRESERVED_FACT_KEYS = {
    "approvalState",
    "permissionKey",
    "permissionState",
    "toolIdentifier",
    "trustedEvidenceStatus",
    "workOwnerSlot",
}

def _preference_natural_fact_alias(
    source: str,
    *,
    identity_registry: dict[str, str] | None = None,
) -> str:
    """Map one source literal into an independent opaque DPO fact identity."""

    digest = canonical_sha256(
        {
            "schemaVersion": "lumen.fleet-preference-fact-identity/1.2.0",
            "source": source,
        }
    )
    identity = _format_orchestration_training_identity(
        identity_class="fact",
        digest=digest,
        surface_index=(
            int(digest[:8], 16)
            % len(_ORCHESTRATION_TRAINING_IDENTITY_SURFACES["fact"])
        ),
    )
    return _register_orchestration_training_identity(
        identity_registry=identity_registry,
        identity=identity,
        digest=digest,
        identity_class="fact",
    )


def _preference_training_facts(
    source: Any,
    *,
    known_slots: set[str],
    key: str | None = None,
    identity_registry: dict[str, str] | None = None,
) -> Any:
    if isinstance(source, dict):
        return {
            child_key: _preference_training_facts(
                child,
                known_slots=known_slots,
                key=child_key,
                identity_registry=identity_registry,
            )
            for child_key, child in source.items()
        }
    if isinstance(source, list):
        return [
            _preference_training_facts(
                child,
                known_slots=known_slots,
                key=key,
                identity_registry=identity_registry,
            )
            for child in source
        ]
    if (
        isinstance(source, str)
        and key not in _PREFERENCE_PRESERVED_FACT_KEYS
        and source not in known_slots
    ):
        return _preference_natural_fact_alias(
            source,
            identity_registry=identity_registry,
        )
    return source


def _preference_graph_replacements(
    source: Any,
    rebound: Any,
    *,
    key: str | None = None,
) -> dict[str, str]:
    """Bind exact chosen fact changes into a structurally different negative."""

    if isinstance(source, dict) and isinstance(rebound, dict):
        replacements: dict[str, str] = {}
        for child_key in source.keys() & rebound.keys():
            if child_key in {
                "fromEventID",
                "id",
                "scenarioID",
                "toEventID",
            }:
                continue
            replacements.update(
                _preference_graph_replacements(
                    source[child_key],
                    rebound[child_key],
                    key=child_key,
                )
            )
        return replacements
    if isinstance(source, list) and isinstance(rebound, list):
        replacements = {}
        if len(source) != len(rebound):
            return replacements
        for left, right in zip(source, rebound, strict=True):
            replacements.update(
                _preference_graph_replacements(left, right, key=key)
            )
        return replacements
    if (
        isinstance(source, str)
        and isinstance(rebound, str)
        and source != rebound
    ):
        return {source: rebound}
    return {}


def _replace_prompt_string_values(
    prompt: str,
    replacements: dict[str, str],
) -> str:
    """Replace original prompt literals once without rewriting inserted values."""

    selected = {
        source: target
        for source, target in replacements.items()
        if source and source != target
    }
    if not selected:
        return prompt
    pattern = re.compile(
        "|".join(
            re.escape(source)
            for source in sorted(selected, key=lambda value: (-len(value), value))
        )
    )
    return pattern.sub(lambda match: selected[match.group(0)], prompt)


def _orchestration_preference_scenario(
    source: dict[str, Any],
    *,
    identity_registry: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Build a DPO-only prompt/answer instance disjoint from every SFT anchor."""

    scenario = json.loads(json.dumps(source, ensure_ascii=False))
    source_id = str(source["id"])
    behavior = str(scenario["behaviorClass"])
    variant = str(scenario["trainingMatrixVariant"])
    instance_index = scenario.get("behaviorConditionedInstanceIndex")
    replica_index = (
        int(instance_index) - 1
        if variant == ORCHESTRATION_BEHAVIOR_CONDITIONED_VARIANT
        else None
    )
    registry = identity_registry if identity_registry is not None else {}
    preference_id = _orchestration_training_scenario_id(
        behavior=behavior,
        variant=variant,
        replica_index=replica_index,
        lane="dpo",
        identity_registry=registry,
    )
    chosen = scenario["graph"]
    preference_derivation = json.loads(
        json.dumps(scenario["canonicalDerivation"], ensure_ascii=False)
    )
    preference_derivation["scenarioID"] = preference_id
    preference_derivation["trainingIdentityLane"] = "dpo"
    facts = preference_derivation.get("facts")
    if not isinstance(facts, dict):
        raise ValueError("Fleet preference derivation facts are missing")
    preference_derivation["facts"] = _preference_training_facts(
        facts,
        known_slots=set(chosen["knownSlotIDs"]),
        identity_registry=registry,
    )
    preference_graph = _derive_orchestration_graph_from_contract(
        preference_derivation
    )
    replacements = _preference_graph_replacements(
        chosen,
        preference_graph,
    )

    atomic_mutation = scenario.get("atomicPreferenceMutation")
    if isinstance(atomic_mutation, str):
        try:
            mutation_index = ORCHESTRATION_ATOMIC_MUTATION_KINDS.index(
                atomic_mutation
            )
        except ValueError as exc:
            raise ValueError(
                f"Unknown Fleet atomic preference mutation: {atomic_mutation}"
            ) from exc
        preference_rejected = _atomic_orchestration_rejection(
            preference_graph,
            mutation_index=mutation_index,
            event_id_fact=_orchestration_event_id_negative_fact(
                preference_derivation["facts"]
            ),
        )
    else:
        preference_rejected = _replace_exact_string_values(
            scenario["rejectedGraph"],
            replacements,
        )
        preference_rejected["scenarioID"] = preference_id
        preference_rejected = _canonicalize_orchestration_event_ids(
            preference_rejected
        )
        if variant == "core":
            preference_rejected = _core_failure_family_rejection(
                preference_graph,
                behavior=behavior,
                event_id_fact=_orchestration_event_id_negative_fact(
                    preference_derivation["facts"]
                ),
            )
            scenario["preferenceContrastMode"] = (
                ORCHESTRATION_CORE_FAILURE_CONTRAST_MODE
            )
            scenario["coreFailureFamilyMutation"] = (
                ORCHESTRATION_CORE_FAILURE_FAMILY_MUTATIONS[behavior]
            )

    preference_prompt = _replace_prompt_string_values(
        str(scenario["prompt"]),
        {
            **replacements,
            source_id: preference_id,
        },
    )

    scenario.update(
        {
            "id": preference_id,
            "prompt": preference_prompt,
            "graph": preference_graph,
            "rejectedGraph": preference_rejected,
            "canonicalDerivation": preference_derivation,
            "evalConstraints": _replace_exact_string_values(
                scenario["evalConstraints"],
                replacements,
            ),
        }
    )
    return scenario


def _orchestration_scenarios(manifest: AgentBehaviorManifest) -> list[dict[str, Any]]:  # NOSONAR
    known_slots = sorted(slot.id for slot in manifest.fleet.slots)
    known_slot_set = set(known_slots)
    scenarios: list[dict[str, Any]] = []

    def add(
        scenario_id: str,
        prompt: str,
        *,
        strategy: str,
        events: list[dict[str, Any]],
        dependencies: list[dict[str, str]],
        delegated_slot_ids: list[str],
        aggregation_owner_slot_id: str | None,
        stop_reason: str,
        eval_constraints: dict[str, Any],
        rejected_events: list[dict[str, Any]] | None = None,
        rejected_dependencies: list[dict[str, str]] | None = None,
        preference_lesson: str = "Prefer the manifest-grounded orchestration graph.",
    ) -> None:
        graph = _orchestration_graph(
            scenario_id=scenario_id,
            known_slot_ids=known_slots,
            strategy=strategy,
            events=events,
            dependencies=dependencies,
            delegated_slot_ids=delegated_slot_ids,
            aggregation_owner_slot_id=aggregation_owner_slot_id,
            stop_reason=stop_reason,
        )
        scenario: dict[str, Any] = {
            "id": scenario_id,
            "behaviorClass": scenario_id,
            "trainingMatrixVariant": "core",
            "prompt": prompt,
            "graph": graph,
            "evalConstraints": eval_constraints,
        }
        if rejected_events is not None:
            scenario["rejectedGraph"] = _orchestration_graph(
                scenario_id=scenario_id,
                known_slot_ids=known_slots,
                strategy="invalid_boundary_bypass",
                events=rejected_events,
                dependencies=rejected_dependencies or [],
                delegated_slot_ids=_delegated_slots_from_events(rejected_events),
                aggregation_owner_slot_id=None,
                stop_reason="invalid_boundary_bypass",
            )
            scenario["preferenceLesson"] = preference_lesson
        scenarios.append(scenario)

    add(
        "no-delegation",
        "The request is already fully answered by trusted current context. Produce an orchestration graph without assigning peer work.",
        strategy="no_delegation",
        events=[
            _orchestration_event("request", "request_received"),
            _orchestration_event("evidence", "trusted_context_verified", evidenceStatus="complete"),
            _orchestration_event("stop", "stop", reason="trusted_context_complete"),
        ],
        dependencies=[
            _orchestration_dependency("request", "evidence"),
            _orchestration_dependency("evidence", "stop"),
        ],
        delegated_slot_ids=[],
        aggregation_owner_slot_id=None,
        stop_reason="trusted_context_complete",
        eval_constraints={"maximumDelegationCount": 0, "mustNotDelegate": True},
    )

    if {"cortex", "executor", "mouth"}.issubset(known_slot_set):
        add(
            "sequential-dependencies",
            "Route an actionable request through planning, strict tool execution, and a grounded final response in dependency order.",
            strategy="sequential",
            events=[
                _orchestration_event("request", "request_received"),
                _orchestration_event("plan", "delegate", targetSlotID="cortex", contextKeys=["userRequest", "availableTools"]),
                _orchestration_event("execute", "delegate", targetSlotID="executor", contextKeys=["approvedPlan", "toolContract"]),
                _orchestration_event("respond", "delegate", targetSlotID="mouth", contextKeys=["trustedObservation", "userVisibleState"]),
                _orchestration_event("stop", "stop", reason="grounded_response_complete"),
            ],
            dependencies=[
                _orchestration_dependency("request", "plan"),
                _orchestration_dependency("plan", "execute"),
                _orchestration_dependency("execute", "respond"),
                _orchestration_dependency("respond", "stop"),
            ],
            delegated_slot_ids=["cortex", "executor", "mouth"],
            aggregation_owner_slot_id="mouth",
            stop_reason="grounded_response_complete",
            eval_constraints={
                "mustRespectDependencyOrder": True,
                "expectedSequence": ["cortex", "executor", "mouth"],
            },
        )

    if {"cortex", "executor", "mimicry", "mouth"}.issubset(known_slot_set):
        add(
            "parallel-dependencies",
            "Plan once, run independent execution and style-analysis branches in parallel, then combine them in one grounded response.",
            strategy="parallel_then_aggregate",
            events=[
                _orchestration_event("request", "request_received"),
                _orchestration_event("plan", "delegate", targetSlotID="cortex", contextKeys=["userRequest", "availableTools"]),
                _orchestration_event("execute", "delegate", targetSlotID="executor", contextKeys=["approvedPlan", "toolContract"]),
                _orchestration_event("style", "delegate", targetSlotID="mimicry", contextKeys=["styleHints", "locale"]),
                _orchestration_event("aggregate", "delegate", targetSlotID="mouth", contextKeys=["trustedObservation", "styleProfile"]),
                _orchestration_event("stop", "stop", reason="parallel_results_aggregated"),
            ],
            dependencies=[
                _orchestration_dependency("request", "plan"),
                _orchestration_dependency("plan", "execute"),
                _orchestration_dependency("plan", "style"),
                _orchestration_dependency("execute", "aggregate"),
                _orchestration_dependency("style", "aggregate"),
                _orchestration_dependency("aggregate", "stop"),
            ],
            delegated_slot_ids=["cortex", "executor", "mimicry", "mouth"],
            aggregation_owner_slot_id="mouth",
            stop_reason="parallel_results_aggregated",
            eval_constraints={
                "parallelBranches": [["executor"], ["mimicry"]],
                "mustWaitForAllDependenciesBeforeAggregation": True,
            },
        )

    if "executor" in known_slot_set:
        add(
            "context-handoff",
            "Hand an approved action to the strict executor with only the bounded context needed to produce a manifest-valid call.",
            strategy="bounded_handoff",
            events=[
                _orchestration_event("request", "request_received"),
                _orchestration_event(
                    "handoff",
                    "delegate",
                    targetSlotID="executor",
                    contextKeys=["approvedPlan", "toolID", "argumentCandidates", "permissionState", "approvalState"],
                    excludes=["rawConversation", "privatePeerState", "hiddenReasoning"],
                ),
                _orchestration_event("result", "result_received", sourceSlotID="executor"),
                _orchestration_event("stop", "stop", reason="bounded_handoff_complete"),
            ],
            dependencies=[
                _orchestration_dependency("request", "handoff"),
                _orchestration_dependency("handoff", "result"),
                _orchestration_dependency("result", "stop"),
            ],
            delegated_slot_ids=["executor"],
            aggregation_owner_slot_id=None,
            stop_reason="bounded_handoff_complete",
            eval_constraints={
                "requiredContextKeys": ["approvedPlan", "toolID", "argumentCandidates", "permissionState", "approvalState"],
                "forbiddenContextKeys": ["rawConversation", "privatePeerState", "hiddenReasoning"],
            },
        )

        duplicate_chosen = [
            _orchestration_event("request", "request_received"),
            _orchestration_event("first", "delegate", targetSlotID="executor", workKey="strict-tool-call-1"),
            _orchestration_event("duplicate", "duplicate_suppressed", targetSlotID="executor", workKey="strict-tool-call-1"),
            _orchestration_event("result", "result_received", sourceSlotID="executor", workKey="strict-tool-call-1"),
            _orchestration_event("stop", "stop", reason="unique_work_complete"),
        ]
        duplicate_rejected = [
            _orchestration_event("request", "request_received"),
            _orchestration_event("first", "delegate", targetSlotID="executor", workKey="strict-tool-call-1"),
            _orchestration_event("second", "delegate", targetSlotID="executor", workKey="strict-tool-call-1"),
            _orchestration_event("stop", "stop", reason="duplicate_work_dispatched"),
        ]
        add(
            "duplicate-suppression",
            "Two candidate branches request the same executor work key. Dispatch it once and suppress the duplicate.",
            strategy="deduplicated",
            events=duplicate_chosen,
            dependencies=[
                _orchestration_dependency("request", "first"),
                _orchestration_dependency("first", "duplicate"),
                _orchestration_dependency("first", "result"),
                _orchestration_dependency("duplicate", "stop"),
                _orchestration_dependency("result", "stop"),
            ],
            delegated_slot_ids=["executor"],
            aggregation_owner_slot_id=None,
            stop_reason="unique_work_complete",
            eval_constraints={"mustSuppressDuplicateDelegation": True, "maximumDelegationsPerWorkKey": 1},
            rejected_events=duplicate_rejected,
            rejected_dependencies=[
                _orchestration_dependency("request", "first"),
                _orchestration_dependency("first", "second"),
                _orchestration_dependency("second", "stop"),
            ],
            preference_lesson="Dispatch each semantic work key at most once.",
        )

    if {"executor", "mimicry", "mouth"}.issubset(known_slot_set):
        add(
            "aggregation-owner",
            "Combine an executor observation and a style profile. Assign exactly one manifest slot to own the user-facing aggregation.",
            strategy="aggregate",
            events=[
                _orchestration_event("request", "request_received"),
                _orchestration_event("tool-result", "result_available", sourceSlotID="executor"),
                _orchestration_event("style-result", "result_available", sourceSlotID="mimicry"),
                _orchestration_event("aggregate", "delegate", targetSlotID="mouth", contextKeys=["trustedObservation", "styleProfile"]),
                _orchestration_event("stop", "stop", reason="single_owner_finalized"),
            ],
            dependencies=[
                _orchestration_dependency("request", "tool-result"),
                _orchestration_dependency("request", "style-result"),
                _orchestration_dependency("tool-result", "aggregate"),
                _orchestration_dependency("style-result", "aggregate"),
                _orchestration_dependency("aggregate", "stop"),
            ],
            delegated_slot_ids=["mouth"],
            aggregation_owner_slot_id="mouth",
            stop_reason="single_owner_finalized",
            eval_constraints={"mustHaveExactlyOneAggregationOwner": True, "aggregationOwnerSlotID": "mouth"},
        )

    approval_tool = next((tool for tool in sorted(manifest.tools, key=lambda item: item.id) if tool.requiresApproval), None)
    if approval_tool is not None:
        approval_chosen = [
            _orchestration_event("request", "request_received", toolID=approval_tool.id),
            _orchestration_event("boundary", "approval_boundary", toolID=approval_tool.id, approvalState="required"),
            _orchestration_event("approval", "request_user_approval", toolID=approval_tool.id),
            _orchestration_event("stop", "stop", reason="awaiting_user_approval"),
        ]
        approval_rejected = [
            _orchestration_event("request", "request_received", toolID=approval_tool.id),
            _orchestration_event("execute", "delegate", targetSlotID="executor", toolID=approval_tool.id, approvalState="missing"),
            _orchestration_event("stop", "stop", reason="executed_without_approval"),
        ]
        add(
            "approval-boundary",
            f"The request requires `{approval_tool.id}`, but user approval has not been granted. Respect the approval boundary.",
            strategy="approval_boundary",
            events=approval_chosen,
            dependencies=[
                _orchestration_dependency("request", "boundary"),
                _orchestration_dependency("boundary", "approval"),
                _orchestration_dependency("approval", "stop"),
            ],
            delegated_slot_ids=[],
            aggregation_owner_slot_id=None,
            stop_reason="awaiting_user_approval",
            eval_constraints={"mustRequestApproval": True, "mustNotExecuteBeforeApproval": True, "toolID": approval_tool.id},
            rejected_events=approval_rejected,
            rejected_dependencies=[
                _orchestration_dependency("request", "execute"),
                _orchestration_dependency("execute", "stop"),
            ],
            preference_lesson="Stop at the approval boundary instead of delegating execution.",
        )

    permission_tool = next((tool for tool in sorted(manifest.tools, key=lambda item: item.id) if tool.permissionKey), None)
    if permission_tool is not None:
        unavailable_chosen = [
            _orchestration_event("request", "request_received", toolID=permission_tool.id),
            _orchestration_event(
                "availability",
                "capability_unavailable",
                toolID=permission_tool.id,
                permissionKey=permission_tool.permissionKey,
                permissionState="denied",
            ),
            _orchestration_event("stop", "stop", reason="required_capability_unavailable"),
        ]
        unavailable_rejected = [
            _orchestration_event("request", "request_received", toolID=permission_tool.id),
            _orchestration_event("execute", "delegate", targetSlotID="executor", toolID=permission_tool.id, permissionState="denied"),
            _orchestration_event("stop", "stop", reason="delegated_unavailable_capability"),
        ]
        add(
            "unavailable-boundary",
            f"The permission required for `{permission_tool.id}` is denied. Do not delegate unavailable work or fabricate a result.",
            strategy="unavailable_boundary",
            events=unavailable_chosen,
            dependencies=[
                _orchestration_dependency("request", "availability"),
                _orchestration_dependency("availability", "stop"),
            ],
            delegated_slot_ids=[],
            aggregation_owner_slot_id=None,
            stop_reason="required_capability_unavailable",
            eval_constraints={
                "mustNotDelegateUnavailableCapability": True,
                "mustNotFabricateResult": True,
                "toolID": permission_tool.id,
            },
            rejected_events=unavailable_rejected,
            rejected_dependencies=[
                _orchestration_dependency("request", "execute"),
                _orchestration_dependency("execute", "stop"),
            ],
            preference_lesson="Stop with explicit unavailable evidence instead of delegating denied work.",
        )

    invented_slot = _invented_slot_id(known_slot_set)
    nonexistent_chosen = [
        _orchestration_event("request", "request_received", requestedSlotID=invented_slot),
        _orchestration_event("directory", "slot_directory_checked", requestedSlotID=invented_slot, slotExists=False),
        _orchestration_event("reject", "invalid_slot_rejected", requestedSlotID=invented_slot),
        _orchestration_event("stop", "stop", reason="requested_slot_not_manifested"),
    ]
    nonexistent_rejected = [
        _orchestration_event("request", "request_received", requestedSlotID=invented_slot),
        _orchestration_event("delegate", "delegate", targetSlotID=invented_slot),
        _orchestration_event("stop", "stop", reason="invented_slot_dispatched"),
    ]
    add(
        "nonexistent-slot-negative",
        f"A request explicitly asks Fleet to delegate to nonexistent slot `{invented_slot}`. Reject the invented slot using the manifest directory.",
        strategy="reject_invalid_slot",
        events=nonexistent_chosen,
        dependencies=[
            _orchestration_dependency("request", "directory"),
            _orchestration_dependency("directory", "reject"),
            _orchestration_dependency("reject", "stop"),
        ],
        delegated_slot_ids=[],
        aggregation_owner_slot_id=None,
        stop_reason="requested_slot_not_manifested",
        eval_constraints={"mustRejectSlotID": invented_slot, "maximumDelegationCount": 0},
        rejected_events=nonexistent_rejected,
        rejected_dependencies=[
            _orchestration_dependency("request", "delegate"),
            _orchestration_dependency("delegate", "stop"),
        ],
        preference_lesson="Reject every target slot absent from the manifest directory.",
    )
    return scenarios


def _orchestration_training_scenarios(
    manifest: AgentBehaviorManifest,
    *,
    identity_registry: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Expand every policy cell with fresh train identities and one held-out matrix."""

    matrix: list[dict[str, Any]] = []
    registry = identity_registry if identity_registry is not None else {}
    for behavior_index, scenario in enumerate(_orchestration_scenarios(manifest)):
        matrix.append(
            _orchestration_training_variant(
                manifest,
                scenario,
                variant="core",
                boundary_value_index=0,
                identity_registry=registry,
            )
        )
        for replica_index in range(ORCHESTRATION_BEHAVIOR_CONDITIONED_REPLICAS):
            matrix.append(
                _orchestration_training_variant(
                    manifest,
                    scenario,
                    variant=ORCHESTRATION_BEHAVIOR_CONDITIONED_VARIANT,
                    boundary_value_index=replica_index + 1,
                    replica_index=replica_index,
                    atomic_mutation_index=(
                        behavior_index * ORCHESTRATION_BEHAVIOR_CONDITIONED_REPLICAS
                        + replica_index
                    ),
                    identity_registry=registry,
                )
            )
        matrix.append(
            _orchestration_training_variant(
                manifest,
                scenario,
                variant="normalization-policy-audited",
                boundary_value_index=3,
                identity_registry=registry,
            )
        )
    return matrix


def _orchestration_training_variant(
    manifest: AgentBehaviorManifest,
    source: dict[str, Any],
    *,
    variant: str,
    boundary_value_index: int,
    replica_index: int | None = None,
    atomic_mutation_index: int | None = None,
    identity_registry: dict[str, str] | None = None,
) -> dict[str, Any]:
    scenario = json.loads(json.dumps(source, ensure_ascii=False))
    behavior_class = str(source["behaviorClass"])
    is_core = variant == "core"
    scenario_id = _orchestration_training_scenario_id(
        behavior=behavior_class,
        variant=variant,
        replica_index=replica_index,
        identity_registry=identity_registry,
    )
    scenario["id"] = scenario_id
    scenario["behaviorClass"] = behavior_class
    scenario["trainingMatrixVariant"] = variant
    scenario.pop("canonicalDerivation", None)

    replacements = _training_boundary_replacements(
        manifest,
        scenario,
        behavior_class=behavior_class,
        boundary_value_index=boundary_value_index,
    )
    if replacements:
        scenario = _replace_exact_string_values(scenario, replacements)
    if variant == ORCHESTRATION_BEHAVIOR_CONDITIONED_VARIANT:
        if replica_index is None:
            raise ValueError(
                "Behavior-conditioned Fleet instances require a replica index"
            )
        scenario = _vary_behavior_conditioned_training_facts(
            scenario,
            behavior=behavior_class,
            replica_index=replica_index,
        )
    scenario["id"] = scenario_id
    scenario["behaviorClass"] = behavior_class
    scenario["trainingMatrixVariant"] = variant
    scenario_prompt = str(scenario["prompt"])
    policy_conditions = _orchestration_policy_conditions(
        behavior=behavior_class,
        training_variant=variant,
    )
    base_facts = _training_orchestration_facts(scenario)
    derivation_facts = _training_feature_facts(
        base_facts=base_facts,
        behavior=behavior_class,
        variant=variant,
        replica_index=replica_index,
        conditions=policy_conditions,
        identity_registry=identity_registry,
    )
    if variant == ORCHESTRATION_BEHAVIOR_CONDITIONED_VARIANT:
        if replica_index is None:
            raise ValueError(
                "Behavior-conditioned Fleet prompt requires a replica index"
            )
        scenario_prompt = _behavior_conditioned_training_prompt(
            scenario_prompt,
            behavior=behavior_class,
            facts=derivation_facts,
            replica_index=replica_index,
        )
    scenario["prompt"] = scenario_prompt
    scenario = _replace_exact_string_values(
        scenario,
        _preference_graph_replacements(base_facts, derivation_facts),
    )
    scenario["id"] = scenario_id
    scenario["behaviorClass"] = behavior_class
    scenario["trainingMatrixVariant"] = variant
    request_id = derivation_facts.get("requestIdentifier")
    graph_request_id = (
        None
        if behavior_class == "context-handoff"
        and variant in {"core", ORCHESTRATION_BEHAVIOR_CONDITIONED_VARIANT}
        else request_id
    )
    scenario["graph"] = _orchestration_training_variant_graph(
        scenario["graph"],
        scenario_id=scenario_id,
        event_namespace=scenario_id,
        request_id=graph_request_id,
        variant=(
            ORCHESTRATION_BEHAVIOR_CONDITIONED_VARIANT
            if is_core
            else variant
        ),
        policy_snapshot_id=derivation_facts.get(
            "policyAuditSnapshotIdentifier"
        ),
        completion_record_id=derivation_facts.get(
            "completionAuditRecordIdentifier"
        ),
    )
    scenario["graph"] = _apply_training_policy_condition_support(
        scenario["graph"],
        behavior=behavior_class,
        conditions=policy_conditions,
        facts=derivation_facts,
    )
    if is_core:
        scenario["sftOptimizerVisible"] = True
    if variant == ORCHESTRATION_BEHAVIOR_CONDITIONED_VARIANT:
        if replica_index is None or atomic_mutation_index is None:
            raise ValueError(
                "Behavior-conditioned Fleet instances require replica and "
                "atomic-mutation indices"
            )
        scenario["rejectedGraph"] = _atomic_orchestration_rejection(
            scenario["graph"],
            mutation_index=atomic_mutation_index,
            event_id_fact=_orchestration_event_id_negative_fact(
                derivation_facts
            ),
        )
        scenario["behaviorConditionedInstanceIndex"] = replica_index + 1
        scenario["sftOptimizerVisible"] = (
            replica_index < ORCHESTRATION_BEHAVIOR_CONDITIONED_SFT_REPLICAS
        )
        scenario["atomicPreferenceMutation"] = (
            _atomic_orchestration_mutation_kind(atomic_mutation_index)
        )
        scenario["preferenceLesson"] = (
            "Prefer the exact behavior-conditioned schema over one atomic contract mutation."
        )
    elif isinstance(scenario.get("rejectedGraph"), dict):
        scenario["rejectedGraph"] = _orchestration_training_variant_graph(
            scenario["rejectedGraph"],
            scenario_id=scenario_id,
            event_namespace=f"{scenario_id}-negative",
            request_id=graph_request_id,
            variant=(
                ORCHESTRATION_BEHAVIOR_CONDITIONED_VARIANT
                if is_core
                else variant
            ),
            policy_snapshot_id=derivation_facts.get(
                "policyAuditSnapshotIdentifier"
            ),
            completion_record_id=derivation_facts.get(
                "completionAuditRecordIdentifier"
            ),
        )
    if is_core:
        # Every behavior receives one exact SFT core anchor and one DPO-only
        # single-family failure contrast. The preference instance is rebound
        # to a disjoint DPO identity later.
        scenario["rejectedGraph"] = json.loads(
            json.dumps(scenario["graph"], ensure_ascii=False)
        )
        scenario["preferenceLesson"] = (
            "Prefer the exact canonical graph over one deterministic failure "
            "in the observed behavior contract."
        )
    scenario["canonicalDerivation"] = (
        _training_orchestration_derivation_contract(
            scenario,
            facts=derivation_facts,
        )
    )
    return scenario


_BEHAVIOR_CONDITIONED_PROMPT_FRAMES = (
    "Resolve this concrete Fleet state under the canonical policy: {request}",
    "Given this operational state, derive the minimal Fleet graph: {request}",
    "Apply the Fleet contract to the following situation: {request}",
    "Construct the canonical orchestration outcome for this case: {request}",
    "Use the manifest policy to resolve this request state: {request}",
    "Derive the bounded Fleet event graph for this scenario: {request}",
    "From these trusted conditions, produce the canonical Fleet route: {request}",
    "Evaluate this Fleet situation and emit only its minimal policy graph: {request}",
)

_SEQUENTIAL_CONTEXT_VARIANTS = (
    (("taskIntent", "capabilityDirectory"), ("authorizedPlan", "argumentSchema"), ("verifiedObservation", "responsePolicy")),
    (("goalStatement", "toolAvailability"), ("validatedSteps", "toolContract"), ("groundedResult", "userVisibleState")),
    (("requestedOutcome", "routingConstraints"), ("approvedAction", "typedArguments"), ("trustedToolResult", "renderPolicy")),
    (("actionObjective", "manifestCapabilities"), ("executionPlan", "permissionDecision"), ("validatedObservation", "answerContract")),
    (("userGoal", "enabledToolSet"), ("boundedPlan", "approvalDecision"), ("groundingEvidence", "responseStyle")),
    (("taskBrief", "executionBoundaries"), ("approvedProcedure", "inputSchema"), ("toolEvidence", "presentationPolicy")),
    (("requestSummary", "availableOperations"), ("validatedOperation", "parameterContract"), ("observedOutcome", "responseBoundary")),
    (("desiredResult", "routingPolicy"), ("authorizedAction", "toolSchema"), ("trustedOutcome", "userResponsePolicy")),
)

_PARALLEL_CONTEXT_VARIANTS = (
    (("taskIntent", "capabilityDirectory"), ("authorizedToolTask", "toolBoundary"), ("toneEvidence", "localePreference"), ("joinedObservation", "styleContract")),
    (("goalStatement", "slotAvailability"), ("validatedExecution", "argumentSchema"), ("voiceSignals", "languagePreference"), ("verifiedResult", "renderPolicy")),
    (("requestedOutcome", "routingConstraints"), ("approvedAction", "toolContract"), ("registerEvidence", "localeProfile"), ("groundedObservation", "responseStyle")),
    (("actionObjective", "manifestCapabilities"), ("executionPlan", "permissionDecision"), ("toneProfile", "languageChoice"), ("validatedToolResult", "styleGuide")),
    (("userGoal", "enabledSlotSet"), ("boundedToolPlan", "approvalDecision"), ("voicePattern", "localeRule"), ("trustedEvidence", "presentationContract")),
    (("taskBrief", "parallelBoundaries"), ("approvedProcedure", "inputSchema"), ("styleEvidence", "languageRule"), ("toolOutcome", "renderBoundary")),
    (("requestSummary", "availablePeers"), ("validatedOperation", "parameterContract"), ("toneSignals", "localeContract"), ("observedOutcome", "responsePolicy")),
    (("desiredResult", "fanoutPolicy"), ("authorizedAction", "toolSchema"), ("registerSignals", "languageProfile"), ("trustedOutcome", "adaptiveStyle")),
)

_HANDOFF_CONTEXT_VARIANTS = (
    (("validatedPlan", "manifestToolReference", "typedArguments", "permissionDecision", "approvalDecision"), ("rawConversation", "peerPrivateState", "hiddenReasoning")),
    (("approvedAction", "canonicalToolID", "validatedParameters", "permissionState", "approvalState"), ("conversationTranscript", "peerRuntimeState", "reasoningTrace")),
    (("authorizedPlan", "manifestOperation", "boundedArguments", "permissionResult", "consentResult"), ("chatHistory", "peerMemorySnapshot", "chainOfThought")),
    (("validatedAction", "toolDirectoryEntry", "typedInputs", "permissionCheck", "approvalCheck"), ("rawUserThread", "privatePeerContext", "internalDeliberation")),
    (("approvedProcedure", "manifestToolID", "validatedFields", "permissionVerdict", "approvalVerdict"), ("fullTranscript", "peerHiddenState", "reasoningTokens")),
    (("authorizedOperation", "toolContractID", "boundedInputs", "permissionOutcome", "consentOutcome"), ("conversationArchive", "peerRuntimeSnapshot", "internalReasoningTrace")),
    (("validatedProcedure", "manifestCapabilityID", "typedFields", "permissionAuthorization", "approvalAuthorization"), ("rawDialogue", "peerPrivateMemory", "hiddenScratchpad")),
    (("approvedToolPlan", "canonicalCapability", "validatedArguments", "permissionDecision", "approvalDecision"), ("conversationTranscript", "peerRuntimeSnapshot", "internalReasoningTrace")),
)

_DUPLICATE_WORK_KEYS = (
    "train-calendar-read",
    "train-file-fetch",
    "train-contact-lookup",
    "train-reminder-read",
    "train-mail-lookup",
    "train-location-query",
    "train-document-fetch",
    "train-tool-observation",
)

_AGGREGATION_CONTEXT_VARIANTS = (
    ("verifiedToolEvidence", "toneProfile"),
    ("groundedObservation", "localeStyle"),
    ("validatedResult", "voiceContract"),
    ("trustedToolOutcome", "registerGuide"),
    ("verifiedExecution", "languageStyle"),
    ("groundingEvidence", "responseVoice"),
    ("observedResult", "presentationStyle"),
    ("trustedObservation", "adaptiveStyleGuide"),
)


def _behavior_conditioned_training_prompt(
    request: str,
    *,
    behavior: str,
    facts: dict[str, Any],
    replica_index: int,
) -> str:
    framed = _BEHAVIOR_CONDITIONED_PROMPT_FRAMES[
        replica_index % len(_BEHAVIOR_CONDITIONED_PROMPT_FRAMES)
    ].format(request=request)
    return framed + " " + _behavior_conditioned_fact_narrative(
        behavior=behavior,
        facts=facts,
    )


def _behavior_conditioned_fact_narrative(
    *,
    behavior: str,
    facts: dict[str, Any],
) -> str:
    """Describe supplied opaque facts in the same natural surface used at runtime."""

    request_id = (
        str(facts["requestIdentifier"])
        if behavior != "context-handoff"
        else None
    )
    if behavior == "no-delegation":
        return (
            f"Request `{request_id}` is resolved by trusted snapshot "
            f"`{facts['trustedContextSnapshotIdentifier']}` and evidence "
            f"`{facts['trustedEvidenceIdentifier']}`."
        )
    if behavior == "sequential-dependencies":
        return (
            f"Request `{request_id}` must preserve the supplied peer contexts and "
            f"ground the response in observation "
            f"`{facts['executorObservationIdentifier']}`."
        )
    if behavior == "parallel-dependencies":
        branches = facts["parallelBranchIdentifiers"]
        return (
            f"Request `{request_id}` has independent branches `{branches[0]}` and "
            f"`{branches[1]}` whose supplied results meet at "
            f"`{facts['joinIdentifier']}`."
        )
    if behavior == "context-handoff":
        return (
            f"Approved action `{facts['approvedActionIdentifier']}` permits only the "
            f"listed bounded context and yields executor result "
            f"`{facts['executorResultIdentifier']}`."
        )
    if behavior == "duplicate-suppression":
        branches = facts["candidateBranchIdentifiers"]
        return (
            f"Request `{request_id}` presents candidate branches `{branches[0]}` and "
            f"`{branches[1]}` for the same work key "
            f"`{facts['sharedWorkKey']}`."
        )
    if behavior == "aggregation-owner":
        results = facts["availableResultIdentifiersBySlot"]
        return (
            f"Request `{request_id}` supplies verified results "
            f"`{results['executor']}` and `{results['mimicry']}` and expects response "
            f"`{facts['responseIdentifier']}`."
        )
    if behavior == "approval-boundary":
        return (
            f"Request `{request_id}` for `{facts['toolIdentifier']}` has approval "
            f"state `{facts['approvalState']}` under policy snapshot "
            f"`{facts['approvalPolicySnapshotIdentifier']}`."
        )
    if behavior == "unavailable-boundary":
        return (
            f"Request `{request_id}` for `{facts['toolIdentifier']}` has permission "
            f"state `{facts['permissionState']}` from check "
            f"`{facts['permissionCheckIdentifier']}`."
        )
    if behavior == "nonexistent-slot-negative":
        return (
            f"Request `{request_id}` names unlisted destination "
            f"`{facts['requestedSlotIdentifier']}` against directory snapshot "
            f"`{facts['slotDirectorySnapshotIdentifier']}`."
        )
    raise ValueError(f"Unknown Fleet behavior-conditioned narrative: {behavior}")


def _vary_behavior_conditioned_training_facts(
    source: dict[str, Any],
    *,
    behavior: str,
    replica_index: int,
) -> dict[str, Any]:
    """Vary real request facts without changing the canonical graph topology."""

    scenario = json.loads(json.dumps(source, ensure_ascii=False))
    index = replica_index % ORCHESTRATION_BEHAVIOR_CONDITIONED_REPLICAS
    graph = scenario["graph"]

    if behavior == "no-delegation":
        _one_event(graph, "trusted_context_verified")["evidenceStatus"] = (
            "complete" if index % 2 == 0 else "sufficient"
        )
    elif behavior == "sequential-dependencies":
        contexts = _SEQUENTIAL_CONTEXT_VARIANTS[index]
        for slot_id, context_keys in zip(
            ("cortex", "executor", "mouth"),
            contexts,
            strict=True,
        ):
            _delegation_to(graph, slot_id)["contextKeys"] = list(context_keys)
    elif behavior == "parallel-dependencies":
        contexts = _PARALLEL_CONTEXT_VARIANTS[index]
        for slot_id, context_keys in zip(
            ("cortex", "executor", "mimicry", "mouth"),
            contexts,
            strict=True,
        ):
            _delegation_to(graph, slot_id)["contextKeys"] = list(context_keys)
    elif behavior == "context-handoff":
        allowed, excluded = _HANDOFF_CONTEXT_VARIANTS[index]
        handoff = _delegation_to(graph, "executor")
        handoff["contextKeys"] = list(allowed)
        handoff["excludes"] = list(excluded)
    elif behavior == "duplicate-suppression":
        old_key = _one_event(graph, "delegate")["workKey"]
        scenario = _replace_exact_string_values(
            scenario,
            {str(old_key): _DUPLICATE_WORK_KEYS[index]},
        )
    elif behavior == "aggregation-owner":
        _delegation_to(graph, "mouth")["contextKeys"] = list(
            _AGGREGATION_CONTEXT_VARIANTS[index]
        )
    elif behavior == "approval-boundary":
        # The runtime reports both absent/missing approval and a policy-required
        # boundary. Teach the external state independently from the boundary's
        # required action so the model does not reinterpret "missing" as
        # capability unavailability.
        _one_event(graph, "approval_boundary")["approvalState"] = (
            "missing" if index % 2 == 0 else "required"
        )
    elif behavior not in {
        "unavailable-boundary",
        "nonexistent-slot-negative",
    }:
        raise ValueError(f"Unknown Fleet semantic fact behavior: {behavior}")
    if behavior == "context-handoff":
        handoff = _delegation_to(scenario["graph"], "executor")
        scenario["evalConstraints"]["requiredContextKeys"] = list(
            handoff["contextKeys"]
        )
        scenario["evalConstraints"]["forbiddenContextKeys"] = list(
            handoff["excludes"]
        )
    return scenario


def _training_boundary_replacements(
    manifest: AgentBehaviorManifest,
    scenario: dict[str, Any],
    *,
    behavior_class: str,
    boundary_value_index: int,
) -> dict[str, str]:
    constraints = scenario.get("evalConstraints")
    if not isinstance(constraints, dict):
        return {}
    old_tool_id = constraints.get("toolID")
    if not isinstance(old_tool_id, str):
        return {}
    if behavior_class == "approval-boundary":
        eligible = [
            tool
            for tool in sorted(manifest.tools, key=lambda item: item.id)
            if tool.requiresApproval
        ]
    elif behavior_class == "unavailable-boundary":
        eligible = [
            tool
            for tool in sorted(manifest.tools, key=lambda item: item.id)
            if tool.permissionKey
        ]
    else:
        return {}
    if not eligible:
        return {}
    # The frozen evaluation intentionally selects the final eligible tool.
    # Cycle through the remaining real tools so additional replicas change
    # boundary facts without leaking the holdout tool identity.
    training_eligible = eligible[:-1] if len(eligible) > 1 else eligible
    selected = training_eligible[boundary_value_index % len(training_eligible)]
    replacements = {old_tool_id: selected.id}
    old_permission_key = next(
        (
            tool.permissionKey
            for tool in manifest.tools
            if tool.id == old_tool_id and tool.permissionKey
        ),
        None,
    )
    if old_permission_key and selected.permissionKey:
        replacements[old_permission_key] = selected.permissionKey
    return replacements


def _replace_exact_string_values(value: Any, replacements: dict[str, str]) -> Any:
    if isinstance(value, dict):
        return {
            key: _replace_exact_string_values(child, replacements)
            for key, child in value.items()
        }
    if isinstance(value, list):
        return [_replace_exact_string_values(child, replacements) for child in value]
    if isinstance(value, str):
        replaced = replacements.get(value)
        if replaced is not None:
            return replaced
        for old, new in replacements.items():
            value = value.replace(f"`{old}`", f"`{new}`")
        return value
    return value


def _orchestration_training_variant_graph(
    source: dict[str, Any],
    *,
    scenario_id: str,
    event_namespace: str,
    request_id: str | None,
    variant: str,
    policy_snapshot_id: str | None,
    completion_record_id: str | None,
) -> dict[str, Any]:
    graph = json.loads(json.dumps(source, ensure_ascii=False))
    graph["scenarioID"] = scenario_id
    id_map = {
        str(event["id"]): f"{event_namespace}-{event['id']}"
        for event in graph["events"]
    }
    for event in graph["events"]:
        event["id"] = id_map[str(event["id"])]
        if event.get("type") == "request_received" and request_id is not None:
            event["requestID"] = request_id
    for dependency in graph["dependencies"]:
        dependency["fromEventID"] = id_map[str(dependency["fromEventID"])]
        dependency["toEventID"] = id_map[str(dependency["toEventID"])]

    request_event_id = str(graph["events"][0]["id"])
    stop_event_id = str(graph["events"][-1]["id"])
    behavior_conditioned = variant == ORCHESTRATION_BEHAVIOR_CONDITIONED_VARIANT
    normalization_required = variant in {
        "normalized-intake",
        "normalization-policy-audited",
    }
    audit_required = variant in {
        "policy-audited",
        "normalization-policy-audited",
    }
    if not behavior_conditioned and not normalization_required and not audit_required:
        raise ValueError(f"Unknown Fleet training matrix variant: {variant}")

    request_successor_id = request_event_id
    if normalization_required:
        if not isinstance(request_id, str):
            raise ValueError("Fleet normalized graph lacks an opaque request fact")
        normalization_event_id = f"{event_namespace}-request-normalized"
        for dependency in graph["dependencies"]:
            if dependency["fromEventID"] == request_event_id:
                dependency["fromEventID"] = normalization_event_id
        graph["events"].insert(
            1,
            _orchestration_event(
                normalization_event_id,
                "request_normalized",
                requestID=request_id,
                normalizationProfile="manifest_fields_only",
            ),
        )
        graph["dependencies"].insert(
            0,
            _orchestration_dependency(request_event_id, normalization_event_id),
        )
        request_successor_id = normalization_event_id

    if audit_required:
        if not isinstance(policy_snapshot_id, str) or not isinstance(
            completion_record_id,
            str,
        ):
            raise ValueError("Fleet audited graph lacks opaque audit facts")
        policy_event_id = f"{event_namespace}-policy-snapshot"
        completion_event_id = f"{event_namespace}-completion-audit"
        for dependency in graph["dependencies"]:
            if dependency["fromEventID"] == request_successor_id:
                dependency["fromEventID"] = policy_event_id
            if dependency["toEventID"] == stop_event_id:
                dependency["toEventID"] = completion_event_id
        graph["events"].insert(
            2 if normalization_required else 1,
            _orchestration_event(
                policy_event_id,
                "policy_snapshot_loaded",
                policySnapshotID=policy_snapshot_id,
            ),
        )
        graph["events"].insert(
            len(graph["events"]) - 1,
            _orchestration_event(
                completion_event_id,
                "completion_audit_recorded",
                completionRecordID=completion_record_id,
            ),
        )
        graph["dependencies"].insert(
            1 if normalization_required else 0,
            _orchestration_dependency(request_successor_id, policy_event_id),
        )
        graph["dependencies"].append(
            _orchestration_dependency(completion_event_id, stop_event_id)
        )
    if behavior_conditioned:
        graph["decision"]["delegatedSlotIDs"] = _delegated_slots_from_events(
            graph["events"]
        )
        return _canonicalize_orchestration_event_ids(graph)

    suffix = (
        "normalization_policy_audited"
        if normalization_required and audit_required
        else "normalized_input"
        if normalization_required
        else "policy_audited"
    )

    decision = graph["decision"]
    decision["strategy"] = f"{decision['strategy']}--{suffix}"
    stop_reason = f"{decision['stopReason']}--{suffix}"
    decision["stopReason"] = stop_reason
    graph["events"][-1]["reason"] = stop_reason
    decision["delegatedSlotIDs"] = _delegated_slots_from_events(graph["events"])
    return _canonicalize_orchestration_event_ids(graph)


def _atomic_orchestration_rejection(
    source: dict[str, Any],
    *,
    mutation_index: int,
    event_id_fact: str | None = None,
) -> dict[str, Any]:
    """Return a scorer-invalid graph changing one isolated contract dimension."""

    graph = json.loads(json.dumps(source, ensure_ascii=False))
    mutation_kind = _atomic_orchestration_mutation_kind(mutation_index)
    if mutation_kind == "terminal_decision_contract":
        graph = _terminal_decision_rejection(graph)
    elif mutation_kind == "event_type_vocabulary":
        mutable_events = [
            event
            for event in graph["events"]
            if event.get("type") not in {"request_received", "stop"}
        ]
        if not mutable_events:
            raise ValueError("Atomic Fleet rejection lacks a behavior event")
        mutable_events[0]["type"] = _natural_noncanonical_event_type_alias(
            graph,
            mutable_events[0],
        )
    elif mutation_kind == "event_completeness_contract":
        graph = _atomic_event_completeness_rejection(graph)
    elif mutation_kind == "event_order":
        first, second = _atomic_event_order_indices(graph["events"])
        first_id = graph["events"][first]["id"]
        second_id = graph["events"][second]["id"]
        first_body = dict(graph["events"][second])
        second_body = dict(graph["events"][first])
        first_body["id"] = first_id
        second_body["id"] = second_id
        graph["events"][first] = first_body
        graph["events"][second] = second_body
    elif mutation_kind == "event_id_grammar":
        source_id = graph["events"][0]["id"]
        replacement_id = _fact_derived_noncanonical_event_id(event_id_fact)
        if any(
            event.get("id") == replacement_id
            for event in graph["events"][1:]
        ):
            raise ValueError("Fleet chosen graph already uses the invalid event ID")
        graph["events"][0]["id"] = replacement_id
        for dependency in graph["dependencies"]:
            for endpoint in ("fromEventID", "toEventID"):
                if dependency.get(endpoint) == source_id:
                    dependency[endpoint] = replacement_id
    elif mutation_kind == "dependency_endpoint_reference":
        dependency = graph["dependencies"][-1]
        replacement_endpoint = graph["events"][0]["id"]
        dependency["toEventID"] = replacement_endpoint
    elif mutation_kind == "event_payload_schema":
        payload_event = next(
            (
                event
                for event in graph["events"]
                if event.get("type") in _ATOMIC_REQUIRED_EVENT_PAYLOAD_KEYS
            ),
            None,
        )
        if payload_event is None:
            raise ValueError("Atomic Fleet rejection lacks a payload event")
        required_key = _ATOMIC_REQUIRED_EVENT_PAYLOAD_KEYS[
            str(payload_event["type"])
        ]
        if required_key not in payload_event:
            raise ValueError(
                "Atomic Fleet rejection payload is missing its chosen field"
            )
        payload_event.pop(required_key)
    elif mutation_kind == "delegated_slot_contract":
        delegated = graph["decision"]["delegatedSlotIDs"]
        if not isinstance(delegated, list):
            raise ValueError("Atomic Fleet rejection has invalid chosen delegation")
        if delegated:
            delegated[0] = "invented_shadow_slot"
        else:
            graph["decision"]["delegatedSlotIDs"] = ["invented_shadow_slot"]
    else:  # pragma: no cover - guarded by _atomic_orchestration_mutation_kind
        raise ValueError(f"Unknown Fleet atomic mutation: {mutation_kind}")
    _validate_atomic_orchestration_rejection(
        source,
        graph,
        mutation_kind=mutation_kind,
        event_id_fact=event_id_fact,
    )
    return graph


def _orchestration_top_level_schema_omission_rejection(
    source: dict[str, Any],
) -> dict[str, Any]:
    """Return valid JSON missing one required graph key seen in failed output."""

    graph = json.loads(json.dumps(source, ensure_ascii=False))
    expected_keys = {
        "graphSchemaVersion",
        "scenarioID",
        "knownSlotIDs",
        "events",
        "dependencies",
        "decision",
    }
    if set(graph) != expected_keys or not isinstance(
        graph.get(_ORCHESTRATION_TOP_LEVEL_OMISSION_KEY),
        list,
    ):
        raise ValueError("Fleet omission rejection lacks a complete graph surface")
    graph.pop(_ORCHESTRATION_TOP_LEVEL_OMISSION_KEY)
    return graph


def _core_failure_family_rejection(
    source: dict[str, Any],
    *,
    behavior: str,
    event_id_fact: str | None = None,
) -> dict[str, Any]:
    """Return one deterministic core negative in one observed failure family."""

    try:
        mutation = ORCHESTRATION_CORE_FAILURE_FAMILY_MUTATIONS[behavior]
    except KeyError as exc:
        raise ValueError(
            f"Unknown Fleet core failure-family behavior: {behavior}"
        ) from exc

    if mutation == "top_level_dependencies_omission":
        return _orchestration_top_level_schema_omission_rejection(source)
    if mutation == "event_completeness_contract":
        graph = json.loads(json.dumps(source, ensure_ascii=False))
        return _atomic_event_omission_rejection(graph)
    if mutation in {
        "event_type_vocabulary",
        "dependency_endpoint_role",
        "event_payload_schema",
    }:
        atomic_kind = {
            "dependency_endpoint_role": "dependency_endpoint_reference",
        }.get(mutation, mutation)
        return _atomic_orchestration_rejection(
            source,
            mutation_index=ORCHESTRATION_ATOMIC_MUTATION_KINDS.index(
                atomic_kind
            ),
            event_id_fact=event_id_fact,
        )

    graph = json.loads(json.dumps(source, ensure_ascii=False))
    decision = graph.get("decision")
    events = graph.get("events")
    if not isinstance(decision, dict) or not isinstance(events, list):
        raise ValueError("Fleet core rejection lacks a canonical graph surface")
    if mutation == "decision_aggregation_owner_omission":
        if "aggregationOwnerSlotID" not in decision:
            raise ValueError("Fleet core decision lacks its aggregation owner field")
        decision.pop("aggregationOwnerSlotID")
    elif mutation == "decision_strategy_role":
        if decision.get("strategy") != "bounded_handoff":
            raise ValueError(
                "Fleet context-handoff rejection lacks its canonical strategy"
            )
        decision["strategy"] = "context_handoff"
    elif mutation == "terminal_stop_reason":
        if (
            not events
            or not isinstance(events[-1], dict)
            or events[-1].get("type") != "stop"
            or not isinstance(decision.get("stopReason"), str)
            or events[-1].get("reason") != decision.get("stopReason")
        ):
            raise ValueError("Fleet core rejection lacks a canonical stop event")
        stop_reason = _natural_noncanonical_stop_reason_alias(
            graph,
            str(decision["stopReason"]),
        )
        decision["stopReason"] = stop_reason
        events[-1]["reason"] = stop_reason
    elif mutation == "scenario_identity_role":
        if not isinstance(event_id_fact, str) or not (
            _is_opaque_orchestration_training_fact_id(event_id_fact)
        ):
            raise ValueError(
                "Fleet scenario-identity rejection lacks an opaque fact identity"
            )
        if graph.get("scenarioID") == event_id_fact:
            raise ValueError(
                "Fleet scenario-identity rejection fact is already the scenario"
            )
        graph["scenarioID"] = event_id_fact
        graph = _canonicalize_orchestration_event_ids(graph)
    else:  # pragma: no cover - mapping and branches are kept exhaustive above
        raise ValueError(f"Unknown Fleet core failure-family mutation: {mutation}")
    return graph


_ATOMIC_REQUIRED_EVENT_PAYLOAD_KEYS = {
    "trusted_context_snapshot_loaded": "contextSnapshotID",
    "delegate": "targetSlotID",
    "context_boundary_checked": "allowedContextKeys",
    "work_candidate_identified": "branchID",
    "result_available": "sourceSlotID",
    "approval_policy_evaluated": "policySnapshotID",
    "permission_state_checked": "permissionCheckID",
    "slot_directory_snapshot_loaded": "directorySnapshotID",
}

_CANONICAL_ORCHESTRATION_STRATEGIES = (
    "no_delegation",
    "sequential",
    "parallel_then_aggregate",
    "bounded_handoff",
    "deduplicated",
    "aggregate",
    "approval_boundary",
    "unavailable_boundary",
    "reject_invalid_slot",
)

_NONCANONICAL_EVENT_TYPE_SUFFIX_ALIASES = {
    "available": "observed",
    "checked": "verified",
    "completed": "finalized",
    "evaluated": "checked",
    "identified": "selected",
    "joined": "merged",
    "loaded": "verified",
    "received": "accepted",
    "recorded": "verified",
    "rejected": "denied",
}


def _rotated_canonical_strategy(strategy: Any) -> str:
    if not isinstance(strategy, str):
        raise ValueError("Fleet chosen strategy is malformed")
    try:
        index = _CANONICAL_ORCHESTRATION_STRATEGIES.index(strategy)
    except ValueError as exc:
        raise ValueError("Fleet chosen strategy is not canonical") from exc
    return _CANONICAL_ORCHESTRATION_STRATEGIES[
        (index + 1) % len(_CANONICAL_ORCHESTRATION_STRATEGIES)
    ]


def _natural_noncanonical_event_type_alias(
    graph: dict[str, Any],
    event: dict[str, Any],
) -> str:
    canonical_event_type = event.get("type")
    if not isinstance(canonical_event_type, str) or not canonical_event_type:
        raise ValueError("Fleet chosen behavior event type is malformed")
    tokens = canonical_event_type.split("_")
    candidates: list[str] = []
    replacement = _NONCANONICAL_EVENT_TYPE_SUFFIX_ALIASES.get(tokens[-1])
    if replacement is not None:
        candidates.append("_".join([*tokens[:-1], replacement]))
    if len(tokens) >= 3:
        # Dropping one semantic qualifier produces the kind of plausible but
        # scorer-invalid alias seen when a model remembers the topology but not
        # the exact manifest vocabulary.
        candidates.append("_".join([*tokens[:-2], tokens[-1]]))
    candidates.extend(
        (
            tokens[0]
            + "".join(token.title() for token in tokens[1:]),
            f"fleet.{canonical_event_type}",
        )
    )
    aliases = tuple(
        dict.fromkeys(
            candidate
            for candidate in candidates
            if candidate and candidate != canonical_event_type
        )
    )
    if not aliases:
        raise ValueError("Fleet event type has no noncanonical near alias")
    digest = canonical_sha256(
        {
            "schemaVersion": "lumen.fleet-event-type-negative/1.1.0",
            "scenarioID": graph.get("scenarioID"),
            "canonicalEventType": canonical_event_type,
        }
    )
    return aliases[int(digest[:8], 16) % len(aliases)]


def _natural_noncanonical_stop_reason_alias(
    graph: dict[str, Any],
    stop_reason: str,
) -> str:
    """Return a coherent but noncanonical terminal reason near-miss."""

    tokens = stop_reason.split("_")
    candidates: list[str] = []
    if len(tokens) >= 3:
        candidates.append("_".join([*tokens[:-2], tokens[-1]]))
    candidates.extend(
        (
            f"{stop_reason}_verified",
            f"{stop_reason}_complete",
        )
    )
    aliases = tuple(
        dict.fromkeys(
            candidate
            for candidate in candidates
            if candidate and candidate != stop_reason
        )
    )
    digest = canonical_sha256(
        {
            "schemaVersion": "lumen.fleet-stop-reason-negative/1.0.0",
            "scenarioID": graph.get("scenarioID"),
            "canonicalStopReason": stop_reason,
        }
    )
    return aliases[int(digest[:8], 16) % len(aliases)]


def _terminal_decision_rejection(
    source: dict[str, Any],
) -> dict[str, Any]:
    """Mutate the coupled terminal decision while preserving graph shape."""

    graph = json.loads(json.dumps(source, ensure_ascii=False))
    decision = graph.get("decision")
    events = graph.get("events")
    if (
        not isinstance(decision, dict)
        or not isinstance(events, list)
        or not events
        or not isinstance(events[-1], dict)
        or events[-1].get("type") != "stop"
        or not isinstance(decision.get("stopReason"), str)
        or events[-1].get("reason") != decision.get("stopReason")
    ):
        raise ValueError("Fleet chosen graph lacks a canonical terminal decision")
    decision["strategy"] = _rotated_canonical_strategy(
        decision.get("strategy")
    )
    stop_reason = _natural_noncanonical_stop_reason_alias(
        graph,
        str(decision["stopReason"]),
    )
    decision["stopReason"] = stop_reason
    events[-1]["reason"] = stop_reason
    return graph


def _orchestration_event_id_negative_fact(
    facts: dict[str, Any],
) -> str:
    """Select the frozen-schema identity used for an event-ID role negative."""

    candidate = facts.get("requestIdentifier")
    if candidate is None:
        candidate = facts.get("approvedActionIdentifier")
    if not isinstance(candidate, str) or not (
        _is_opaque_orchestration_training_fact_id(candidate)
    ):
        raise ValueError("Fleet rejection lacks an opaque event-ID fact")
    return candidate


def _fact_derived_noncanonical_event_id(fact_id: str | None) -> str:
    """Use a supplied request fact as the wrong event namespace."""

    if not isinstance(fact_id, str) or not (
        _is_opaque_orchestration_training_fact_id(fact_id)
    ):
        raise ValueError("Fleet rejection lacks an opaque request fact")
    return f"{fact_id}::event::01"


def _atomic_event_omission_index(events: Any) -> int:
    """Choose a behavior event whose removal preserves delegation ordering."""

    if not isinstance(events, list):
        raise ValueError("Fleet chosen events are malformed")
    for index, event in enumerate(events):
        if (
            isinstance(event, dict)
            and event.get("type")
            not in {"request_received", "delegate", "stop"}
        ):
            return index
    raise ValueError("Fleet chosen graph lacks an omittable behavior event")


def _atomic_event_omission_rejection(
    graph: dict[str, Any],
) -> dict[str, Any]:
    """Remove one event while preserving graph closure and canonical identities."""

    events = graph.get("events")
    dependencies = graph.get("dependencies")
    if not isinstance(events, list) or not isinstance(dependencies, list):
        raise ValueError("Fleet chosen graph lacks events or dependencies")
    omission_index = _atomic_event_omission_index(events)
    omitted = events[omission_index]
    omitted_id = omitted.get("id") if isinstance(omitted, dict) else None
    if not isinstance(omitted_id, str):
        raise ValueError("Fleet chosen omission event lacks an identity")
    predecessors = [
        dependency["fromEventID"]
        for dependency in dependencies
        if isinstance(dependency, dict)
        and dependency.get("toEventID") == omitted_id
        and isinstance(dependency.get("fromEventID"), str)
    ]
    successors = [
        dependency["toEventID"]
        for dependency in dependencies
        if isinstance(dependency, dict)
        and dependency.get("fromEventID") == omitted_id
        and isinstance(dependency.get("toEventID"), str)
    ]
    if not predecessors or not successors:
        raise ValueError("Fleet chosen omission event is not an interior graph node")
    retained = [
        dependency
        for dependency in dependencies
        if isinstance(dependency, dict)
        and dependency.get("fromEventID") != omitted_id
        and dependency.get("toEventID") != omitted_id
    ]
    retained_edges = {
        (dependency["fromEventID"], dependency["toEventID"])
        for dependency in retained
    }
    for predecessor in predecessors:
        for successor in successors:
            edge = (predecessor, successor)
            if predecessor != successor and edge not in retained_edges:
                retained.append(
                    _orchestration_dependency(predecessor, successor)
                )
                retained_edges.add(edge)
    graph["events"].pop(omission_index)
    graph["dependencies"] = retained
    return _canonicalize_orchestration_event_ids(graph)


def _atomic_terminal_stop_omission_rejection(
    graph: dict[str, Any],
) -> dict[str, Any]:
    """Remove the terminal stop and only edges incident to that event."""

    events = graph.get("events")
    dependencies = graph.get("dependencies")
    if (
        not isinstance(events, list)
        or not events
        or not isinstance(dependencies, list)
        or not isinstance(events[-1], dict)
        or events[-1].get("type") != "stop"
        or not isinstance(events[-1].get("id"), str)
    ):
        raise ValueError("Fleet chosen graph lacks a terminal stop event")
    stop_id = events[-1]["id"]
    graph["events"] = events[:-1]
    graph["dependencies"] = [
        dependency
        for dependency in dependencies
        if isinstance(dependency, dict)
        and dependency.get("fromEventID") != stop_id
        and dependency.get("toEventID") != stop_id
    ]
    emitted_ids = {
        event["id"]
        for event in graph["events"]
        if isinstance(event, dict) and isinstance(event.get("id"), str)
    }
    if any(
        dependency.get("fromEventID") not in emitted_ids
        or dependency.get("toEventID") not in emitted_ids
        for dependency in graph["dependencies"]
    ):
        raise ValueError("Fleet terminal omission leaves an open dependency")
    return graph


def _atomic_event_completeness_rejection(
    graph: dict[str, Any],
) -> dict[str, Any]:
    """Rotate deterministically between interior and terminal event omission."""

    strategy = graph.get("decision", {}).get("strategy")
    if not isinstance(strategy, str):
        raise ValueError("Fleet chosen graph lacks a canonical strategy")
    try:
        strategy_index = _CANONICAL_ORCHESTRATION_STRATEGIES.index(strategy)
    except ValueError as exc:
        raise ValueError("Fleet chosen graph lacks a canonical strategy") from exc
    if strategy_index % 2 == 0:
        return _atomic_terminal_stop_omission_rejection(graph)
    return _atomic_event_omission_rejection(graph)


def _atomic_event_order_indices(events: Any) -> tuple[int, int]:
    """Choose an adjacent event swap that preserves delegation ordering."""

    if not isinstance(events, list) or not all(
        isinstance(event, dict) for event in events
    ):
        raise ValueError("Fleet chosen events are malformed")
    delegated = _delegated_slots_from_events(events)
    for first in range(1, len(events) - 2):
        second = first + 1
        if events[first].get("type") == events[second].get("type"):
            continue
        reordered = list(events)
        reordered[first], reordered[second] = (
            reordered[second],
            reordered[first],
        )
        if _delegated_slots_from_events(reordered) == delegated:
            return first, second
    raise ValueError("Fleet chosen graph lacks a safe adjacent event-order swap")


def _validate_atomic_orchestration_rejection(
    chosen: dict[str, Any],
    rejected: dict[str, Any],
    *,
    mutation_kind: str,
    event_id_fact: str | None = None,
) -> None:
    """Type-check one mutation and prove every other graph value is unchanged."""

    if mutation_kind not in ORCHESTRATION_ATOMIC_MUTATION_KINDS:
        raise ValueError(f"Unknown Fleet atomic mutation: {mutation_kind}")
    expected = json.loads(json.dumps(chosen, ensure_ascii=False))
    if mutation_kind == "terminal_decision_contract":
        expected = _terminal_decision_rejection(expected)
    elif mutation_kind == "event_type_vocabulary":
        mutable_events = [
            event
            for event in expected.get("events", [])
            if isinstance(event, dict)
            and event.get("type") not in {"request_received", "stop"}
        ]
        if not mutable_events or not isinstance(mutable_events[0].get("type"), str):
            raise ValueError("Fleet chosen behavior event is malformed")
        mutable_events[0]["type"] = _natural_noncanonical_event_type_alias(
            expected,
            mutable_events[0],
        )
    elif mutation_kind == "event_completeness_contract":
        expected = _atomic_event_completeness_rejection(expected)
    elif mutation_kind == "event_order":
        events = expected.get("events")
        first, second = _atomic_event_order_indices(events)
        first_id = events[first]["id"]
        second_id = events[second]["id"]
        first_body = dict(events[second])
        second_body = dict(events[first])
        first_body["id"] = first_id
        second_body["id"] = second_id
        events[first] = first_body
        events[second] = second_body
    elif mutation_kind == "event_id_grammar":
        events = expected.get("events")
        if not isinstance(events, list) or not events or not isinstance(
            events[0].get("id") if isinstance(events[0], dict) else None,
            str,
        ):
            raise ValueError("Fleet chosen event identity is malformed")
        source_id = events[0]["id"]
        replacement_id = _fact_derived_noncanonical_event_id(event_id_fact)
        if any(event.get("id") == replacement_id for event in events[1:]):
            raise ValueError("Fleet chosen graph already uses the invalid event ID")
        events[0]["id"] = replacement_id
        for dependency in expected.get("dependencies", []):
            if not isinstance(dependency, dict):
                raise ValueError("Fleet chosen dependency is malformed")
            for endpoint in ("fromEventID", "toEventID"):
                if dependency.get(endpoint) == source_id:
                    dependency[endpoint] = replacement_id
    elif mutation_kind == "dependency_endpoint_reference":
        dependencies = expected.get("dependencies")
        events = expected.get("events")
        if (
            not isinstance(dependencies, list)
            or not dependencies
            or not isinstance(dependencies[-1], dict)
            or not isinstance(events, list)
            or len(events) < 2
            or not all(isinstance(event, dict) for event in events)
        ):
            raise ValueError("Fleet chosen dependency is malformed")
        terminal_event = events[-1]
        replacement_endpoint = events[0].get("id")
        if (
            terminal_event.get("type") != "stop"
            or dependencies[-1].get("toEventID") != terminal_event.get("id")
            or not isinstance(replacement_endpoint, str)
            or replacement_endpoint == terminal_event.get("id")
        ):
            raise ValueError("Fleet chosen dependency endpoint is malformed")
        dependencies[-1]["toEventID"] = replacement_endpoint
    elif mutation_kind == "event_payload_schema":
        payload_event = next(
            (
                event
                for event in expected.get("events", [])
                if isinstance(event, dict)
                and event.get("type") in _ATOMIC_REQUIRED_EVENT_PAYLOAD_KEYS
            ),
            None,
        )
        if payload_event is None:
            raise ValueError("Fleet chosen payload event is malformed")
        required_key = _ATOMIC_REQUIRED_EVENT_PAYLOAD_KEYS[
            str(payload_event["type"])
        ]
        if required_key not in payload_event:
            raise ValueError("Fleet chosen payload field is missing")
        payload_event.pop(required_key)
    else:
        delegated = expected.get("decision", {}).get("delegatedSlotIDs")
        if not isinstance(delegated, list) or not all(
            isinstance(slot, str) for slot in delegated
        ):
            raise ValueError("Fleet chosen delegation directory is malformed")
        if delegated:
            delegated[0] = "invented_shadow_slot"
        else:
            expected["decision"]["delegatedSlotIDs"] = [
                "invented_shadow_slot"
            ]
    if rejected != expected:
        raise ValueError(
            "Fleet atomic rejection changes more than its typed contract dimension"
        )


def _atomic_orchestration_mutation_kind(mutation_index: int) -> str:
    return ORCHESTRATION_ATOMIC_MUTATION_KINDS[
        mutation_index % len(ORCHESTRATION_ATOMIC_MUTATION_KINDS)
    ]


def _orchestration_scalar_leaf_differences(
    chosen: Any,
    rejected: Any,
    *,
    path: str = "$",
) -> list[str]:
    """Return scalar paths that differ, rejecting structural mutations."""

    if type(chosen) is not type(rejected):
        return [path]
    if isinstance(chosen, dict):
        if set(chosen) != set(rejected):
            return [path]
        differences: list[str] = []
        for key in sorted(chosen):
            differences.extend(
                _orchestration_scalar_leaf_differences(
                    chosen[key],
                    rejected[key],
                    path=f"{path}.{key}",
                )
            )
        return differences
    if isinstance(chosen, list):
        if len(chosen) != len(rejected):
            return [path]
        differences = []
        for index, (chosen_item, rejected_item) in enumerate(
            zip(chosen, rejected, strict=True)
        ):
            differences.extend(
                _orchestration_scalar_leaf_differences(
                    chosen_item,
                    rejected_item,
                    path=f"{path}[{index}]",
                )
            )
        return differences
    return [] if chosen == rejected else [path]


def _apply_training_policy_condition_support(
    source: dict[str, Any],
    *,
    behavior: str,
    conditions: dict[str, bool],
    facts: dict[str, Any],
) -> dict[str, Any]:  # NOSONAR
    """Materialize holdout-relevant conditions inside distinct train topologies."""

    graph = json.loads(json.dumps(source, ensure_ascii=False))
    events = graph["events"]
    dependencies = graph["dependencies"]

    def one(event_type: str) -> dict[str, Any]:
        matched = [event for event in events if event.get("type") == event_type]
        if len(matched) != 1:
            raise ValueError(
                f"Fleet condition support needs one {event_type}, got {len(matched)}"
            )
        return matched[0]

    def delegation(slot_id: str) -> dict[str, Any]:
        matched = [
            event
            for event in events
            if event.get("type") == "delegate"
            and event.get("targetSlotID") == slot_id
        ]
        if len(matched) != 1:
            raise ValueError(
                f"Fleet condition support needs one delegation to {slot_id}"
            )
        return matched[0]

    def remove_edge(source_id: str, target_id: str) -> None:
        matched = [
            index
            for index, edge in enumerate(dependencies)
            if edge.get("fromEventID") == source_id
            and edge.get("toEventID") == target_id
        ]
        if len(matched) != 1:
            raise ValueError("Fleet condition support cannot locate direct edge")
        dependencies.pop(matched[0])

    def add_edge(source_id: str, target_id: str) -> None:
        dependencies.append(_orchestration_dependency(source_id, target_id))

    def insert_between(
        source_event: dict[str, Any],
        target_event: dict[str, Any],
        inserted: dict[str, Any],
    ) -> None:
        remove_edge(source_event["id"], target_event["id"])
        events.insert(events.index(target_event), inserted)
        add_edge(source_event["id"], inserted["id"])
        add_edge(inserted["id"], target_event["id"])

    normalized = one("request_normalized") if conditions["requestNormalizationRequired"] else None
    policy_snapshot = one("policy_snapshot_loaded") if conditions["policyAuditRequired"] else None
    request = one("request_received")
    # When both wrappers are present, policy_snapshot is the final entry gate:
    # request -> normalization -> policy -> behavior-specific graph.
    state_entry = policy_snapshot or normalized or request

    if conditions["trustedContextSnapshotProvided"]:
        evidence = one("trusted_context_verified")
        evidence["evidenceID"] = facts["trustedEvidenceIdentifier"]
        insert_between(
            state_entry,
            evidence,
            _orchestration_event(
                "support-context-snapshot",
                "trusted_context_snapshot_loaded",
                contextSnapshotID=facts["trustedContextSnapshotIdentifier"],
            ),
        )
    if conditions["executorObservationProvided"]:
        executor = delegation("executor")
        mouth = delegation("mouth")
        insert_between(
            executor,
            mouth,
            _orchestration_event(
                "support-executor-observation",
                "result_received",
                sourceSlotID="executor",
                observationID=facts["executorObservationIdentifier"],
            ),
        )
    if conditions["parallelJoinRequired"]:
        executor = delegation("executor")
        mimicry = delegation("mimicry")
        mouth = delegation("mouth")
        branches = facts["parallelBranchIdentifiers"]
        executor["branchID"] = branches[0]
        mimicry["branchID"] = branches[1]
        remove_edge(executor["id"], mouth["id"])
        remove_edge(mimicry["id"], mouth["id"])
        join = _orchestration_event(
            "support-parallel-join",
            "branch_join_verified",
            branchIDs=branches,
            joinID=facts["joinIdentifier"],
        )
        events.insert(events.index(mouth), join)
        add_edge(executor["id"], join["id"])
        add_edge(mimicry["id"], join["id"])
        add_edge(join["id"], mouth["id"])
    if conditions["contextBoundaryReviewRequired"]:
        executor = delegation("executor")
        result = one("result_received")
        request["actionID"] = facts["approvedActionIdentifier"]
        result["resultID"] = facts["executorResultIdentifier"]
        insert_between(
            state_entry,
            executor,
            _orchestration_event(
                "support-context-boundary",
                "context_boundary_checked",
                allowedContextKeys=executor["contextKeys"],
                excludes=executor["excludes"],
            ),
        )
    if conditions["candidateBranchesProvided"]:
        dispatch = delegation(str(facts["workOwnerSlot"]))
        duplicate = one("duplicate_suppressed")
        result = one("result_received")
        request["requestID"] = facts["requestIdentifier"]
        branches = facts["candidateBranchIdentifiers"]
        candidate_a = _orchestration_event(
            "support-candidate-a",
            "work_candidate_identified",
            branchID=branches[0],
            targetSlotID=dispatch["targetSlotID"],
            workKey=dispatch["workKey"],
        )
        candidate_b = _orchestration_event(
            "support-candidate-b",
            "work_candidate_identified",
            branchID=branches[1],
            targetSlotID=dispatch["targetSlotID"],
            workKey=dispatch["workKey"],
        )
        remove_edge(state_entry["id"], dispatch["id"])
        remove_edge(dispatch["id"], duplicate["id"])
        events.insert(events.index(dispatch), candidate_a)
        events.insert(events.index(duplicate), candidate_b)
        add_edge(state_entry["id"], candidate_a["id"])
        add_edge(state_entry["id"], candidate_b["id"])
        add_edge(candidate_a["id"], dispatch["id"])
        add_edge(candidate_b["id"], duplicate["id"])
        if result.get("workKey") != dispatch.get("workKey"):
            raise ValueError("Fleet dedup support lost the shared work key")
    if conditions["aggregationInputVerificationRequired"]:
        available = [event for event in events if event.get("type") == "result_available"]
        mouth = delegation("mouth")
        result_ids = facts["availableResultIdentifiersBySlot"]
        for result_event in available:
            result_event["resultID"] = result_ids[result_event["sourceSlotID"]]
            remove_edge(result_event["id"], mouth["id"])
        verified = _orchestration_event(
            "support-aggregation-verification",
            "aggregation_inputs_verified",
            inputResultIDs=facts["verifiedInputResultIdentifiers"],
        )
        events.insert(events.index(mouth), verified)
        for result_event in available:
            add_edge(result_event["id"], verified["id"])
        add_edge(verified["id"], mouth["id"])
    if conditions["responseValidationRequired"]:
        available = [event for event in events if event.get("type") == "result_available"]
        result_ids = facts["availableResultIdentifiersBySlot"]
        for result_event in available:
            result_event["resultID"] = result_ids[result_event["sourceSlotID"]]
        mouth = delegation("mouth")
        completion = (
            one("completion_audit_recorded")
            if conditions["policyAuditRequired"]
            else one("stop")
        )
        insert_between(
            mouth,
            completion,
            _orchestration_event(
                "support-response-validation",
                "response_validated",
                responseID=facts["responseIdentifier"],
                sourceSlotID="mouth",
            ),
        )
    if conditions["approvalPolicyEvaluationRequired"]:
        boundary = one("approval_boundary")
        approval = one("request_user_approval")
        request["requestID"] = facts["requestIdentifier"]
        approval["approvalRequestID"] = facts["userApprovalRequestIdentifier"]
        boundary["approvalState"] = "required"
        insert_between(
            state_entry,
            boundary,
            _orchestration_event(
                "support-approval-policy",
                "approval_policy_evaluated",
                approvalState=facts["approvalState"],
                policySnapshotID=facts["approvalPolicySnapshotIdentifier"],
                toolID=facts["toolIdentifier"],
            ),
        )
    if conditions["permissionPreflightRequired"]:
        unavailable = one("capability_unavailable")
        request["requestID"] = facts["requestIdentifier"]
        insert_between(
            state_entry,
            unavailable,
            _orchestration_event(
                "support-permission-preflight",
                "permission_state_checked",
                permissionCheckID=facts["permissionCheckIdentifier"],
                permissionKey=unavailable["permissionKey"],
                permissionState=unavailable["permissionState"],
                toolID=facts["toolIdentifier"],
            ),
        )
    if conditions["slotDirectorySnapshotProvided"]:
        directory = one("slot_directory_checked")
        request["requestID"] = facts["requestIdentifier"]
        insert_between(
            state_entry,
            directory,
            _orchestration_event(
                "support-directory-snapshot",
                "slot_directory_snapshot_loaded",
                directorySnapshotID=facts["slotDirectorySnapshotIdentifier"],
            ),
        )
    if conditions["rejectionRecordRequired"]:
        rejection = one("invalid_slot_rejected")
        completion = (
            one("completion_audit_recorded")
            if conditions["policyAuditRequired"]
            else one("stop")
        )
        request["requestID"] = facts["requestIdentifier"]
        insert_between(
            rejection,
            completion,
            _orchestration_event(
                "support-rejection-record",
                "rejection_recorded",
                rejectionID=facts["rejectionIdentifier"],
                requestedSlotID=rejection["requestedSlotID"],
            ),
        )

    return _canonicalize_orchestration_event_ids(graph)


def _orchestration_eval_scenarios(
    manifest: AgentBehaviorManifest,
) -> list[dict[str, Any]]:  # NOSONAR
    """Build one topology-distinct frozen instance for every behavior class."""

    known_slots = sorted(slot.id for slot in manifest.fleet.slots)
    known_slot_set = set(known_slots)
    scenarios: list[dict[str, Any]] = []

    def add(
        scenario_id: str,
        behavior_class: str,
        prompt: str,
        *,
        strategy: str,
        events: list[dict[str, Any]],
        dependencies: list[dict[str, str]],
        delegated_slot_ids: list[str],
        aggregation_owner_slot_id: str | None,
        stop_reason: str,
        eval_constraints: dict[str, Any],
    ) -> None:
        scenarios.append(
            {
                "id": scenario_id,
                "behaviorClass": behavior_class,
                "prompt": prompt,
                "graph": _orchestration_graph(
                    scenario_id=scenario_id,
                    known_slot_ids=known_slots,
                    strategy=strategy,
                    events=events,
                    dependencies=dependencies,
                    delegated_slot_ids=delegated_slot_ids,
                    aggregation_owner_slot_id=aggregation_owner_slot_id,
                    stop_reason=stop_reason,
                ),
                "evalConstraints": eval_constraints,
            }
        )

    add(
        "holdout-context-terminal-17",
        "no-delegation",
        (
            "Request `holdout-request-17` is fully resolved by trusted context snapshot "
            "`holdout-context-snapshot-17`; verify evidence `holdout-proof-17` and stop "
            "without assigning peer work."
        ),
        strategy="no_delegation",
        events=[
            _orchestration_event(
                "ctx-intake-17",
                "request_received",
                requestID="holdout-request-17",
            ),
            _orchestration_event(
                "ctx-load-17",
                "trusted_context_snapshot_loaded",
                contextSnapshotID="holdout-context-snapshot-17",
            ),
            _orchestration_event(
                "ctx-proof-17",
                "trusted_context_verified",
                evidenceID="holdout-proof-17",
                evidenceStatus="sufficient",
            ),
            _orchestration_event(
                "ctx-terminal-17",
                "stop",
                reason="trusted_context_complete",
            ),
        ],
        dependencies=[
            _orchestration_dependency("ctx-intake-17", "ctx-load-17"),
            _orchestration_dependency("ctx-load-17", "ctx-proof-17"),
            _orchestration_dependency("ctx-proof-17", "ctx-terminal-17"),
        ],
        delegated_slot_ids=[],
        aggregation_owner_slot_id=None,
        stop_reason="trusted_context_complete",
        eval_constraints={"maximumDelegationCount": 0, "mustNotDelegate": True},
    )

    if {"cortex", "executor", "mouth"}.issubset(known_slot_set):
        add(
            "holdout-linear-tool-route-26",
            "sequential-dependencies",
            (
                "For request `holdout-linear-request-26`, plan in Cortex, execute in "
                "Executor, verify observation `holdout-observation-26`, and only then "
                "delegate the grounded response to Mouth."
            ),
            strategy="sequential",
            events=[
                _orchestration_event(
                    "seq-intake-26",
                    "request_received",
                    requestID="holdout-linear-request-26",
                ),
                _orchestration_event(
                    "seq-route-26",
                    "delegate",
                    targetSlotID="cortex",
                    contextKeys=["requestEnvelope", "manifestToolDirectory"],
                ),
                _orchestration_event(
                    "seq-call-26",
                    "delegate",
                    targetSlotID="executor",
                    contextKeys=["validatedActionPlan", "toolSchemaSnapshot"],
                ),
                _orchestration_event(
                    "seq-observe-26",
                    "result_received",
                    sourceSlotID="executor",
                    observationID="holdout-observation-26",
                ),
                _orchestration_event(
                    "seq-answer-26",
                    "delegate",
                    targetSlotID="mouth",
                    contextKeys=["verifiedToolObservation", "responsePolicy"],
                ),
                _orchestration_event(
                    "seq-terminal-26",
                    "stop",
                    reason="grounded_response_complete",
                ),
            ],
            dependencies=[
                _orchestration_dependency("seq-intake-26", "seq-route-26"),
                _orchestration_dependency("seq-route-26", "seq-call-26"),
                _orchestration_dependency("seq-call-26", "seq-observe-26"),
                _orchestration_dependency("seq-observe-26", "seq-answer-26"),
                _orchestration_dependency("seq-answer-26", "seq-terminal-26"),
            ],
            delegated_slot_ids=["cortex", "executor", "mouth"],
            aggregation_owner_slot_id="mouth",
            stop_reason="grounded_response_complete",
            eval_constraints={
                "mustRespectDependencyOrder": True,
                "expectedSequence": ["cortex", "executor", "mouth"],
            },
        )

    if {"cortex", "executor", "mimicry", "mouth"}.issubset(known_slot_set):
        add(
            "holdout-fork-join-response-31",
            "parallel-dependencies",
            (
                "For request `holdout-parallel-request-31`, let Cortex route two independent "
                "branches: Executor work `holdout-tool-branch-31` and Mimicry work "
                "`holdout-style-branch-31`. Verify join `holdout-join-31` before Mouth responds."
            ),
            strategy="parallel_then_aggregate",
            events=[
                _orchestration_event(
                    "par-intake-31",
                    "request_received",
                    requestID="holdout-parallel-request-31",
                ),
                _orchestration_event(
                    "par-route-31",
                    "delegate",
                    targetSlotID="cortex",
                    contextKeys=["planInput", "slotCapabilityMap"],
                ),
                _orchestration_event(
                    "par-tool-31",
                    "delegate",
                    targetSlotID="executor",
                    branchID="holdout-tool-branch-31",
                    contextKeys=["approvedToolTask", "toolBoundarySnapshot"],
                ),
                _orchestration_event(
                    "par-style-31",
                    "delegate",
                    targetSlotID="mimicry",
                    branchID="holdout-style-branch-31",
                    contextKeys=["toneEvidence", "languagePreference"],
                ),
                _orchestration_event(
                    "par-join-31",
                    "branch_join_verified",
                    branchIDs=["holdout-tool-branch-31", "holdout-style-branch-31"],
                    joinID="holdout-join-31",
                ),
                _orchestration_event(
                    "par-render-31",
                    "delegate",
                    targetSlotID="mouth",
                    contextKeys=["verifiedObservation", "renderStyleContract"],
                ),
                _orchestration_event(
                    "par-terminal-31",
                    "stop",
                    reason="parallel_results_aggregated",
                ),
            ],
            dependencies=[
                _orchestration_dependency("par-intake-31", "par-route-31"),
                _orchestration_dependency("par-route-31", "par-tool-31"),
                _orchestration_dependency("par-route-31", "par-style-31"),
                _orchestration_dependency("par-tool-31", "par-join-31"),
                _orchestration_dependency("par-style-31", "par-join-31"),
                _orchestration_dependency("par-join-31", "par-render-31"),
                _orchestration_dependency("par-render-31", "par-terminal-31"),
            ],
            delegated_slot_ids=["cortex", "executor", "mimicry", "mouth"],
            aggregation_owner_slot_id="mouth",
            stop_reason="parallel_results_aggregated",
            eval_constraints={
                "parallelBranches": [["executor"], ["mimicry"]],
                "mustWaitForAllDependenciesBeforeAggregation": True,
            },
        )

    if "executor" in known_slot_set:
        handoff_context = [
            "validatedActionPlan",
            "manifestToolID",
            "validatedArguments",
            "permissionDecision",
            "approvalDecision",
        ]
        handoff_excludes = [
            "conversationTranscript",
            "peerRuntimeSnapshot",
            "internalReasoningTrace",
        ]
        add(
            "holdout-minimal-executor-handoff-44",
            "context-handoff",
            (
                "For approved action `holdout-action-44`, enforce the least-context boundary "
                "before handing work to Executor and return only the bounded result."
            ),
            strategy="bounded_handoff",
            events=[
                _orchestration_event(
                    "handoff-intake-44",
                    "request_received",
                    actionID="holdout-action-44",
                ),
                _orchestration_event(
                    "handoff-filter-44",
                    "context_boundary_checked",
                    allowedContextKeys=handoff_context,
                    excludes=handoff_excludes,
                ),
                _orchestration_event(
                    "handoff-dispatch-44",
                    "delegate",
                    targetSlotID="executor",
                    contextKeys=handoff_context,
                    excludes=handoff_excludes,
                ),
                _orchestration_event(
                    "handoff-result-44",
                    "result_received",
                    sourceSlotID="executor",
                    resultID="holdout-executor-result-44",
                ),
                _orchestration_event(
                    "handoff-terminal-44",
                    "stop",
                    reason="bounded_handoff_complete",
                ),
            ],
            dependencies=[
                _orchestration_dependency("handoff-intake-44", "handoff-filter-44"),
                _orchestration_dependency("handoff-filter-44", "handoff-dispatch-44"),
                _orchestration_dependency("handoff-dispatch-44", "handoff-result-44"),
                _orchestration_dependency("handoff-result-44", "handoff-terminal-44"),
            ],
            delegated_slot_ids=["executor"],
            aggregation_owner_slot_id=None,
            stop_reason="bounded_handoff_complete",
            eval_constraints={
                "mustRespectDependencyOrder": True,
                "requiredContextKeys": handoff_context,
                "forbiddenContextKeys": handoff_excludes,
            },
        )

        duplicate_work_key = "holdout-calendar-read-73"
        add(
            "holdout-shared-work-dedup-73",
            "duplicate-suppression",
            (
                f"Branches `holdout-branch-a-73` and `holdout-branch-b-73` request the "
                f"same Executor work key `{duplicate_work_key}`. Register both candidates, "
                "dispatch once, suppress once, and stop after the one result."
            ),
            strategy="deduplicated",
            events=[
                _orchestration_event(
                    "dedup-intake-73",
                    "request_received",
                    requestID="holdout-dedup-request-73",
                ),
                _orchestration_event(
                    "dedup-candidate-a-73",
                    "work_candidate_identified",
                    branchID="holdout-branch-a-73",
                    targetSlotID="executor",
                    workKey=duplicate_work_key,
                ),
                _orchestration_event(
                    "dedup-dispatch-73",
                    "delegate",
                    targetSlotID="executor",
                    workKey=duplicate_work_key,
                ),
                _orchestration_event(
                    "dedup-candidate-b-73",
                    "work_candidate_identified",
                    branchID="holdout-branch-b-73",
                    targetSlotID="executor",
                    workKey=duplicate_work_key,
                ),
                _orchestration_event(
                    "dedup-suppress-73",
                    "duplicate_suppressed",
                    targetSlotID="executor",
                    workKey=duplicate_work_key,
                ),
                _orchestration_event(
                    "dedup-result-73",
                    "result_received",
                    sourceSlotID="executor",
                    workKey=duplicate_work_key,
                ),
                _orchestration_event(
                    "dedup-terminal-73",
                    "stop",
                    reason="unique_work_complete",
                ),
            ],
            dependencies=[
                _orchestration_dependency("dedup-intake-73", "dedup-candidate-a-73"),
                _orchestration_dependency("dedup-intake-73", "dedup-candidate-b-73"),
                _orchestration_dependency("dedup-candidate-a-73", "dedup-dispatch-73"),
                _orchestration_dependency("dedup-dispatch-73", "dedup-result-73"),
                _orchestration_dependency("dedup-candidate-b-73", "dedup-suppress-73"),
                _orchestration_dependency("dedup-suppress-73", "dedup-terminal-73"),
                _orchestration_dependency("dedup-result-73", "dedup-terminal-73"),
            ],
            delegated_slot_ids=["executor"],
            aggregation_owner_slot_id=None,
            stop_reason="unique_work_complete",
            eval_constraints={
                "mustRespectDependencyOrder": True,
                "mustSuppressDuplicateDelegation": True,
                "maximumDelegationsPerWorkKey": 1,
            },
        )

    if {"executor", "mimicry", "mouth"}.issubset(known_slot_set):
        add(
            "holdout-single-render-owner-58",
            "aggregation-owner",
            (
                "Tool result `holdout-tool-result-58` and style result "
                "`holdout-style-result-58` are independently ready. Verify both inputs, "
                "assign Mouth as the sole render owner, validate response "
                "`holdout-response-58`, and stop."
            ),
            strategy="aggregate",
            events=[
                _orchestration_event(
                    "owner-intake-58",
                    "request_received",
                    requestID="holdout-owner-request-58",
                ),
                _orchestration_event(
                    "owner-tool-58",
                    "result_available",
                    resultID="holdout-tool-result-58",
                    sourceSlotID="executor",
                ),
                _orchestration_event(
                    "owner-style-58",
                    "result_available",
                    resultID="holdout-style-result-58",
                    sourceSlotID="mimicry",
                ),
                _orchestration_event(
                    "owner-ready-58",
                    "aggregation_inputs_verified",
                    inputResultIDs=[
                        "holdout-tool-result-58",
                        "holdout-style-result-58",
                    ],
                ),
                _orchestration_event(
                    "owner-render-58",
                    "delegate",
                    targetSlotID="mouth",
                    contextKeys=["verifiedObservation", "adaptiveStyleGuide"],
                ),
                _orchestration_event(
                    "owner-check-58",
                    "response_validated",
                    responseID="holdout-response-58",
                    sourceSlotID="mouth",
                ),
                _orchestration_event(
                    "owner-terminal-58",
                    "stop",
                    reason="single_owner_finalized",
                ),
            ],
            dependencies=[
                _orchestration_dependency("owner-intake-58", "owner-tool-58"),
                _orchestration_dependency("owner-intake-58", "owner-style-58"),
                _orchestration_dependency("owner-tool-58", "owner-ready-58"),
                _orchestration_dependency("owner-style-58", "owner-ready-58"),
                _orchestration_dependency("owner-ready-58", "owner-render-58"),
                _orchestration_dependency("owner-render-58", "owner-check-58"),
                _orchestration_dependency("owner-check-58", "owner-terminal-58"),
            ],
            delegated_slot_ids=["mouth"],
            aggregation_owner_slot_id="mouth",
            stop_reason="single_owner_finalized",
            eval_constraints={
                "mustHaveExactlyOneAggregationOwner": True,
                "aggregationOwnerSlotID": "mouth",
                "mustRespectDependencyOrder": True,
            },
        )

    approval_tools = [
        tool
        for tool in sorted(manifest.tools, key=lambda item: item.id)
        if tool.requiresApproval
    ]
    if approval_tools:
        approval_tool = approval_tools[-1]
        add(
            "holdout-approval-stop-64",
            "approval-boundary",
            (
                f"Request `holdout-approval-request-64` requires `{approval_tool.id}`. "
                "Approval is absent; evaluate policy snapshot `holdout-approval-policy-64`, "
                "emit one approval boundary and one user-approval request, then stop."
            ),
            strategy="approval_boundary",
            events=[
                _orchestration_event(
                    "approval-intake-64",
                    "request_received",
                    requestID="holdout-approval-request-64",
                    toolID=approval_tool.id,
                ),
                _orchestration_event(
                    "approval-policy-64",
                    "approval_policy_evaluated",
                    approvalState="missing",
                    policySnapshotID="holdout-approval-policy-64",
                    toolID=approval_tool.id,
                ),
                _orchestration_event(
                    "approval-boundary-64",
                    "approval_boundary",
                    approvalState="required",
                    toolID=approval_tool.id,
                ),
                _orchestration_event(
                    "approval-ask-64",
                    "request_user_approval",
                    approvalRequestID="holdout-user-approval-64",
                    toolID=approval_tool.id,
                ),
                _orchestration_event(
                    "approval-terminal-64",
                    "stop",
                    reason="awaiting_user_approval",
                ),
            ],
            dependencies=[
                _orchestration_dependency("approval-intake-64", "approval-policy-64"),
                _orchestration_dependency("approval-policy-64", "approval-boundary-64"),
                _orchestration_dependency("approval-boundary-64", "approval-ask-64"),
                _orchestration_dependency("approval-ask-64", "approval-terminal-64"),
            ],
            delegated_slot_ids=[],
            aggregation_owner_slot_id=None,
            stop_reason="awaiting_user_approval",
            eval_constraints={
                "mustRespectDependencyOrder": True,
                "mustRequestApproval": True,
                "mustNotExecuteBeforeApproval": True,
                "toolID": approval_tool.id,
            },
        )

    permission_tools = [
        tool
        for tool in sorted(manifest.tools, key=lambda item: item.id)
        if tool.permissionKey
    ]
    if permission_tools:
        permission_tool = permission_tools[-1]
        add(
            "holdout-permission-denied-stop-82",
            "unavailable-boundary",
            (
                f"Request `holdout-permission-request-82` needs `{permission_tool.id}`, "
                f"whose `{permission_tool.permissionKey}` permission is denied. Verify the "
                "permission check, record capability unavailability, and stop without delegation."
            ),
            strategy="unavailable_boundary",
            events=[
                _orchestration_event(
                    "permission-intake-82",
                    "request_received",
                    requestID="holdout-permission-request-82",
                    toolID=permission_tool.id,
                ),
                _orchestration_event(
                    "permission-check-82",
                    "permission_state_checked",
                    permissionCheckID="holdout-permission-check-82",
                    permissionKey=permission_tool.permissionKey,
                    permissionState="denied",
                    toolID=permission_tool.id,
                ),
                _orchestration_event(
                    "permission-unavailable-82",
                    "capability_unavailable",
                    permissionKey=permission_tool.permissionKey,
                    permissionState="denied",
                    toolID=permission_tool.id,
                ),
                _orchestration_event(
                    "permission-terminal-82",
                    "stop",
                    reason="required_capability_unavailable",
                ),
            ],
            dependencies=[
                _orchestration_dependency("permission-intake-82", "permission-check-82"),
                _orchestration_dependency("permission-check-82", "permission-unavailable-82"),
                _orchestration_dependency("permission-unavailable-82", "permission-terminal-82"),
            ],
            delegated_slot_ids=[],
            aggregation_owner_slot_id=None,
            stop_reason="required_capability_unavailable",
            eval_constraints={
                "mustRespectDependencyOrder": True,
                "mustNotDelegateUnavailableCapability": True,
                "mustNotFabricateResult": True,
                "toolID": permission_tool.id,
            },
        )

    invented_slot = _invented_eval_slot_id(known_slot_set)
    add(
        "holdout-directory-rejection-91",
        "nonexistent-slot-negative",
        (
            f"Request `holdout-directory-request-91` names target `{invented_slot}`. "
            "Load directory snapshot `holdout-directory-snapshot-91`, verify absence, "
            "reject the target, record rejection `holdout-rejection-91`, and stop."
        ),
        strategy="reject_invalid_slot",
        events=[
            _orchestration_event(
                "directory-intake-91",
                "request_received",
                requestID="holdout-directory-request-91",
                requestedSlotID=invented_slot,
            ),
            _orchestration_event(
                "directory-snapshot-91",
                "slot_directory_snapshot_loaded",
                directorySnapshotID="holdout-directory-snapshot-91",
            ),
            _orchestration_event(
                "directory-check-91",
                "slot_directory_checked",
                requestedSlotID=invented_slot,
                slotExists=False,
            ),
            _orchestration_event(
                "directory-reject-91",
                "invalid_slot_rejected",
                requestedSlotID=invented_slot,
            ),
            _orchestration_event(
                "directory-record-91",
                "rejection_recorded",
                rejectionID="holdout-rejection-91",
                requestedSlotID=invented_slot,
            ),
            _orchestration_event(
                "directory-terminal-91",
                "stop",
                reason="requested_slot_not_manifested",
            ),
        ],
        dependencies=[
            _orchestration_dependency("directory-intake-91", "directory-snapshot-91"),
            _orchestration_dependency("directory-snapshot-91", "directory-check-91"),
            _orchestration_dependency("directory-check-91", "directory-reject-91"),
            _orchestration_dependency("directory-reject-91", "directory-record-91"),
            _orchestration_dependency("directory-record-91", "directory-terminal-91"),
        ],
        delegated_slot_ids=[],
        aggregation_owner_slot_id=None,
        stop_reason="requested_slot_not_manifested",
        eval_constraints={
            "mustRespectDependencyOrder": True,
            "mustRejectSlotID": invented_slot,
            "maximumDelegationCount": 0,
        },
    )

    _assert_orchestration_holdouts_disjoint(
        training_scenarios=_orchestration_training_scenarios(manifest),
        eval_scenarios=scenarios,
    )
    return scenarios


def _orchestration_graph(
    *,
    scenario_id: str,
    known_slot_ids: list[str],
    strategy: str,
    events: list[dict[str, Any]],
    dependencies: list[dict[str, str]],
    delegated_slot_ids: list[str],
    aggregation_owner_slot_id: str | None,
    stop_reason: str,
) -> dict[str, Any]:
    return _canonicalize_orchestration_event_ids({
        "graphSchemaVersion": "1.0.0",
        "scenarioID": scenario_id,
        "knownSlotIDs": known_slot_ids,
        "events": events,
        "dependencies": dependencies,
        "decision": {
            "strategy": strategy,
            "delegatedSlotIDs": delegated_slot_ids,
            "aggregationOwnerSlotID": aggregation_owner_slot_id,
            "stopReason": stop_reason,
        },
    })


def _canonicalize_orchestration_event_ids(
    graph: dict[str, Any],
) -> dict[str, Any]:
    """Derive every event ID from scenario identity and canonical order."""

    canonical = json.loads(json.dumps(graph, ensure_ascii=False))
    scenario_id = str(canonical.get("scenarioID") or "")
    events = canonical.get("events")
    dependencies = canonical.get("dependencies")
    if not scenario_id or not isinstance(events, list) or not isinstance(
        dependencies,
        list,
    ):
        raise ValueError("Fleet graph cannot derive canonical event IDs")
    id_map: dict[str, str] = {}
    for index, event in enumerate(events, start=1):
        if not isinstance(event, dict) or not isinstance(event.get("id"), str):
            raise ValueError("Fleet graph event is missing its source ID")
        old_id = event["id"]
        if old_id in id_map:
            raise ValueError("Fleet graph source event IDs must be unique")
        canonical_id = f"{scenario_id}::event::{index:02d}"
        id_map[old_id] = canonical_id
        event["id"] = canonical_id
    for dependency in dependencies:
        if not isinstance(dependency, dict):
            raise ValueError("Fleet graph dependency must be an object")
        from_id = dependency.get("fromEventID")
        to_id = dependency.get("toEventID")
        if from_id not in id_map or to_id not in id_map:
            raise ValueError("Fleet graph dependency references an unknown event")
        dependency["fromEventID"] = id_map[str(from_id)]
        dependency["toEventID"] = id_map[str(to_id)]
    canonical_positions = {
        event["id"]: index for index, event in enumerate(events)
    }
    dependencies.sort(
        key=lambda dependency: (
            canonical_positions[dependency["fromEventID"]],
            canonical_positions[dependency["toEventID"]],
            dependency["kind"],
        )
    )
    return canonical


def _orchestration_event(event_id: str, event_type: str, **payload: Any) -> dict[str, Any]:
    return {"id": event_id, "type": event_type, **payload}


def _orchestration_dependency(source: str, target: str) -> dict[str, str]:
    return {"fromEventID": source, "toEventID": target, "kind": "requires"}


def _delegated_slots_from_events(events: list[dict[str, Any]]) -> list[str]:
    return [
        str(event["targetSlotID"])
        for event in events
        if event.get("type") == "delegate" and isinstance(event.get("targetSlotID"), str)
    ]


def _invented_slot_id(known_slot_ids: set[str]) -> str:
    candidate = "invented_shadow_slot"
    suffix = 2
    while candidate in known_slot_ids:
        candidate = f"invented_shadow_slot_{suffix}"
        suffix += 1
    return candidate


def _invented_eval_slot_id(known_slot_ids: set[str]) -> str:
    candidate = "holdout_unlisted_router_slot"
    suffix = 2
    while candidate in known_slot_ids:
        candidate = f"holdout_unlisted_router_slot_{suffix}"
        suffix += 1
    return candidate


def _orchestration_topology_contract(graph: dict[str, Any]) -> dict[str, Any]:
    """Normalize away instance wording while retaining graph/schema/path semantics."""

    events = graph.get("events") if isinstance(graph.get("events"), list) else []
    event_positions = {
        str(event.get("id")): index
        for index, event in enumerate(events)
        if isinstance(event, dict) and isinstance(event.get("id"), str)
    }
    normalized_events: list[dict[str, Any]] = []
    for event in events:
        if not isinstance(event, dict):
            normalized_events.append({"invalidEventType": type(event).__name__})
            continue
        payload_contract: dict[str, Any] = {}
        for key, value in sorted(event.items()):
            if key in {"id", "type"}:
                continue
            if key in {"targetSlotID", "sourceSlotID"} and isinstance(value, str):
                payload_contract[key] = {"type": "slot_path", "value": value}
            else:
                payload_contract[key] = _orchestration_value_shape(value)
        normalized_events.append(
            {
                "type": event.get("type"),
                "payloadContract": payload_contract,
            }
        )
    normalized_dependencies: list[dict[str, Any]] = []
    dependencies = graph.get("dependencies")
    if isinstance(dependencies, list):
        for dependency in dependencies:
            if not isinstance(dependency, dict):
                normalized_dependencies.append(
                    {"invalidDependencyType": type(dependency).__name__}
                )
                continue
            normalized_dependencies.append(
                {
                    "fromEventIndex": event_positions.get(
                        str(dependency.get("fromEventID"))
                    ),
                    "toEventIndex": event_positions.get(
                        str(dependency.get("toEventID"))
                    ),
                    "kind": dependency.get("kind"),
                }
            )
    decision = graph.get("decision") if isinstance(graph.get("decision"), dict) else {}
    return {
        "fingerprintSchemaVersion": "lumen.eval-candidate-topology-hash/1.0.0",
        "graphSchemaVersion": graph.get("graphSchemaVersion"),
        "knownSlotIDs": graph.get("knownSlotIDs"),
        "events": normalized_events,
        "dependencies": normalized_dependencies,
        "pathContract": {
            "delegatedSlotIDs": decision.get("delegatedSlotIDs"),
            "aggregationOwnerSlotID": decision.get("aggregationOwnerSlotID"),
            "hasStrategy": isinstance(decision.get("strategy"), str),
            "hasStopReason": isinstance(decision.get("stopReason"), str),
        },
    }


def _orchestration_value_shape(value: Any) -> Any:
    if value is None:
        return {"type": "null"}
    if type(value) is bool:
        return {"type": "boolean"}
    if type(value) is int:
        return {"type": "integer"}
    if type(value) is float:
        return {"type": "number"}
    if isinstance(value, str):
        return {"type": "string"}
    if isinstance(value, list):
        return {
            "type": "array",
            "length": len(value),
            "items": [_orchestration_value_shape(child) for child in value],
        }
    if isinstance(value, dict):
        return {
            "type": "object",
            "fields": {
                key: _orchestration_value_shape(child)
                for key, child in sorted(value.items())
            },
        }
    return {"type": type(value).__name__}


def _assert_orchestration_holdouts_disjoint(
    *,
    training_scenarios: list[dict[str, Any]],
    eval_scenarios: list[dict[str, Any]],
) -> None:
    """Keep frozen instances disjoint while requiring learned policy coverage.

    Exact prompts, facts, identifiers, and graphs remain held out. The canonical
    behavior topology is the contract being trained, so each evaluation topology
    must be represented by a controlled fresh behavior-conditioned instance
    instead of occupying an unlearnable Boolean cell.
    """

    training_graphs = [
        scenario["graph"]
        for scenario in training_scenarios
        if isinstance(scenario.get("graph"), dict)
    ]
    training_exact_hashes = {canonical_sha256(graph) for graph in training_graphs}
    training_topology_hashes = {
        canonical_sha256(_orchestration_topology_contract(graph))
        for graph in training_graphs
    }
    training_scenario_ids = {
        str(scenario.get("id"))
        for scenario in training_scenarios
        if isinstance(scenario.get("id"), str)
    }
    training_holdout_values = {
        value
        for graph in training_graphs
        for value in _orchestration_scalar_values(graph)
        if isinstance(value, str) and value.startswith("holdout")
    }
    if training_holdout_values:
        raise ValueError(
            "Fleet training graph contains frozen holdout identifiers: "
            f"{sorted(training_holdout_values)}"
        )
    seen_eval_exact: set[str] = set()
    seen_eval_topologies: set[str] = set()
    for scenario in eval_scenarios:
        graph = scenario.get("graph")
        if not isinstance(graph, dict):
            raise ValueError(f"Fleet holdout graph missing: {scenario.get('id')}")
        exact_hash = canonical_sha256(graph)
        topology_hash = canonical_sha256(_orchestration_topology_contract(graph))
        scenario_id = str(scenario.get("id") or "")
        if scenario_id in training_scenario_ids:
            raise ValueError(
                f"Fleet holdout scenario identity collides with training: {scenario_id}"
            )
        if exact_hash in training_exact_hashes:
            raise ValueError(
                f"Fleet holdout exact graph collides with training: {scenario['id']}"
            )
        topology_matches = [
            training
            for training in training_scenarios
            if (
                training.get("behaviorClass") == scenario.get("behaviorClass")
                and training.get("trainingMatrixVariant")
                == ORCHESTRATION_BEHAVIOR_CONDITIONED_VARIANT
                and isinstance(training.get("graph"), dict)
                and canonical_sha256(
                    _orchestration_topology_contract(training["graph"])
                )
                == topology_hash
            )
        ]
        if len(topology_matches) != ORCHESTRATION_BEHAVIOR_CONDITIONED_REPLICAS:
            raise ValueError(
                "Fleet holdout lacks controlled trained-topology coverage: "
                f"scenario={scenario['id']} observed={len(topology_matches)} "
                f"required={ORCHESTRATION_BEHAVIOR_CONDITIONED_REPLICAS}"
            )
        replica_identity_values = [
            training.get("id") for training in topology_matches
        ]
        replica_identities = set(replica_identity_values)
        replica_graph_hashes = {
            canonical_sha256(training["graph"])
            for training in topology_matches
        }
        replica_index_values = [
            training.get("behaviorConditionedInstanceIndex")
            for training in topology_matches
        ]
        replica_indices = set(replica_index_values)
        replica_mutations = {
            training.get("atomicPreferenceMutation")
            for training in topology_matches
        }
        if (
            len(replica_identities)
            != ORCHESTRATION_BEHAVIOR_CONDITIONED_REPLICAS
            or not all(
                isinstance(identity, str) and identity
                for identity in replica_identity_values
            )
            or len(replica_graph_hashes)
            != ORCHESTRATION_BEHAVIOR_CONDITIONED_REPLICAS
            or not all(type(index) is int for index in replica_index_values)
            or replica_indices
            != set(range(1, ORCHESTRATION_BEHAVIOR_CONDITIONED_REPLICAS + 1))
            or replica_mutations
            != {
                "terminal_decision_contract",
                "event_type_vocabulary",
                "event_completeness_contract",
                "event_order",
                "event_id_grammar",
                "dependency_endpoint_reference",
                "event_payload_schema",
                "delegated_slot_contract",
            }
        ):
            raise ValueError(
                "Fleet holdout trained-topology replicas are not distinct and "
                f"complete: scenario={scenario['id']}"
            )
        if topology_hash not in training_topology_hashes:
            raise ValueError(
                f"Fleet holdout topology is absent from training: {scenario['id']}"
            )
        if exact_hash in seen_eval_exact:
            raise ValueError(f"Duplicate Fleet holdout graph: {scenario['id']}")
        if topology_hash in seen_eval_topologies:
            raise ValueError(f"Duplicate Fleet holdout topology: {scenario['id']}")
        seen_eval_exact.add(exact_hash)
        seen_eval_topologies.add(topology_hash)


def _orchestration_prompt_reconstructs_graph(
    prompt: str,
    graph: dict[str, Any],
) -> bool:
    """Compatibility predicate: prompt binds inputs without copying the target."""

    serialized = json.dumps(
        graph,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    event_ids = [
        event.get("id")
        for event in graph.get("events", [])
        if isinstance(event, dict) and isinstance(event.get("id"), str)
    ]
    return (
        str(graph.get("scenarioID") or "") in prompt
        and all(str(slot) in prompt for slot in graph.get("knownSlotIDs", []))
        and "Trusted request/state facts" in prompt
        and serialized not in prompt
        and canonical_sha256(graph) not in prompt
        and not any(event_id in prompt for event_id in event_ids)
        and '"events"' not in prompt
        and '"dependencies"' not in prompt
        and '"decision"' not in prompt
    )


def _orchestration_scalar_values(value: Any) -> list[Any]:
    values: list[Any] = []
    if isinstance(value, dict):
        for child in value.values():
            values.extend(_orchestration_scalar_values(child))
    elif isinstance(value, list):
        for child in value:
            values.extend(_orchestration_scalar_values(child))
    elif value is None or isinstance(value, (str, bool, int, float)):
        values.append(value)
    return values


def _native_orchestration_metadata(manifest: AgentBehaviorManifest, scenario_id: str) -> dict[str, Any]:
    return {
        "sourceClass": "lumen_native_manifest_derived",
        "derivedFrom": "AgentBehaviorManifest",
        "sourceIntegrity": manifest.sourceIntegrity.lineage_dict(),
        # Compatibility for consumers of the pre-source-snapshot schema.
        "manifestCommit": manifest.sourceIntegrity.commit,
        "scenarioID": scenario_id,
        "eventGraphSchemaVersion": "1.0.0",
    }


def _self_knowledge_records(manifest: AgentBehaviorManifest, slot: ModelSlotManifest) -> list[dict[str, Any]]:
    topology = manifest.fleetTopology.slots.get(slot.id)
    payload = {
        "slotID": slot.id,
        "role": slot.role,
        "source": slot.source,
        "purpose": topology.purpose if topology else _slot_purpose_fallback(slot),
        "availablePeers": sorted(peer.id for peer in manifest.fleet.slots if peer.id != slot.id),
        "memoryScopes": sorted(manifest.memory.scopes),
        "sourceCodeBoundary": "I know the manifest-derived code map and source origins, not arbitrary unextracted source text or private runtime state.",
    }
    return [{
        "id": _record_id("self", slot.id),
        "schemaVersion": "2.1.0",
        "recordType": "sft",
        "agentRole": slot.role,
        "taskType": "fleet_self_knowledge",
        "messages": [
            {"role": "system", "content": f"You are {slot.id}. Answer only from AgentBehaviorManifest."},
            {"role": "user", "content": "Who are you inside Lumen, what source defines you, and what can you do?"},
            {"role": "assistant", "content": json.dumps(payload, ensure_ascii=False, sort_keys=True)},
        ],
    }]


def _source_code_self_knowledge_records(manifest: AgentBehaviorManifest, slot: ModelSlotManifest) -> list[dict[str, Any]]:
    source_map = _source_code_map(manifest)
    payload = {
        "slotID": slot.id,
        "slotSource": _slot_source_payload(slot),
        "sourceIntegrity": manifest.sourceIntegrity.lineage_dict(),
        # Compatibility for existing training-record consumers.
        "sourceIntegrityCommit": manifest.sourceIntegrity.commit,
        "sourceFiles": source_map["files"],
        "domains": source_map["domains"],
        "knownSourceBoundary": (
            "Manifest-derived map only; never invent source text or private state."
        ),
    }
    return [{
        "id": _record_id("source-self", slot.id),
        "schemaVersion": "2.1.0",
        "recordType": "sft",
        "agentRole": slot.role,
        "taskType": "source_code_self_knowledge",
        "messages": [
            {"role": "system", "content": f"You are {slot.id}. Use source-code lineage from the manifest only."},
            {"role": "user", "content": "What parts of the Lumen source code do you know from your manifest, and what are your limits?"},
            {"role": "assistant", "content": json.dumps(payload, ensure_ascii=False, sort_keys=True)},
        ],
    }]


def _tool_origin_records(manifest: AgentBehaviorManifest, slot: ModelSlotManifest) -> list[dict[str, Any]]:
    tools = sorted(manifest.tools, key=lambda item: item.id)
    payload = {
        "toolRegistry": [_tool_payload(tool) for tool in tools],
        "toolCount": len(tools),
        "rule": "Only tool IDs listed in this manifest-derived registry are valid. Source fields explain where the tool contract came from.",
    }
    return [{
        "id": _record_id("tool-origin", slot.id),
        "schemaVersion": "2.1.0",
        "recordType": "sft",
        "agentRole": slot.role,
        "taskType": "source_tool_registry_knowledge",
        "messages": [
            {"role": "system", "content": f"You are {slot.id}. Explain tools from the manifest registry."},
            {"role": "user", "content": "Which tools exist in Lumen, where do their contracts come from, and what must you never do?"},
            {"role": "assistant", "content": json.dumps(payload, ensure_ascii=False, sort_keys=True)},
        ],
    }]


def _routing_origin_records(manifest: AgentBehaviorManifest, slot: ModelSlotManifest) -> list[dict[str, Any]]:
    payload = {
        "intents": [intent.model_dump() for intent in sorted(manifest.intents, key=lambda item: item.id)],
        "routingMatrix": _routing_table(manifest),
        "rule": "Cortex and the unified fleet must obey these routing constraints and reject/clarify instead of inventing a path.",
    }
    return [{
        "id": _record_id("routing-origin", slot.id),
        "schemaVersion": "2.1.0",
        "recordType": "sft",
        "agentRole": slot.role,
        "taskType": "source_routing_knowledge",
        "messages": [
            {"role": "system", "content": f"You are {slot.id}. Explain routing from the manifest."},
            {"role": "user", "content": "How does Lumen know which tool or peer should handle a request?"},
            {"role": "assistant", "content": json.dumps(payload, ensure_ascii=False, sort_keys=True)},
        ],
    }]


def _peer_knowledge_records(manifest: AgentBehaviorManifest, source: ModelSlotManifest, target: ModelSlotManifest) -> list[dict[str, Any]]:
    target_topology = manifest.fleetTopology.slots.get(target.id)
    payload = {
        "slotID": target.id,
        "role": target.role,
        "source": target.source,
        "purpose": target_topology.purpose if target_topology else _slot_purpose_fallback(target),
        "inputSignature": target_topology.inputSignature if target_topology else "Role-specific input defined by manifest.",
        "outputSignature": target_topology.outputSignature if target_topology else "Role-specific output defined by manifest.",
    }
    return [{
        "id": _record_id("peer", source.id, target.id),
        "schemaVersion": "2.1.0",
        "recordType": "sft",
        "agentRole": source.role,
        "taskType": "fleet_peer_knowledge",
        "messages": [
            {"role": "system", "content": f"You are {source.id}. Describe peers only from the manifest public directory."},
            {"role": "user", "content": f"What do you know about {target.id}, including its source and boundaries?"},
            {"role": "assistant", "content": json.dumps(payload, ensure_ascii=False, sort_keys=True)},
        ],
    }]


def _peer_source_knowledge_records(manifest: AgentBehaviorManifest, source: ModelSlotManifest, target: ModelSlotManifest) -> list[dict[str, Any]]:
    payload = {
        "sourceSlot": source.id,
        "targetSlot": target.id,
        "targetSource": _slot_source_payload(target),
        "relationship": "peer",
        "coordinationRule": "Route public-role work; never claim private state.",
    }
    return [{
        "id": _record_id("peer-source", source.id, target.id),
        "schemaVersion": "2.1.0",
        "recordType": "sft",
        "agentRole": source.role,
        "taskType": "fleet_peer_source_knowledge",
        "messages": [
            {"role": "system", "content": f"You are {source.id}. Know peer source origins without crossing private-state boundaries."},
            {"role": "user", "content": f"Where is {target.id}'s public role defined, and how should you coordinate with it?"},
            {"role": "assistant", "content": json.dumps(payload, ensure_ascii=False, sort_keys=True)},
        ],
    }]


def _delegation_records(manifest: AgentBehaviorManifest, source: ModelSlotManifest, target: ModelSlotManifest) -> list[dict[str, Any]]:
    task = _delegation_task_for(target)
    handoff_tools = manifest.fleetTopology.externalHandoffTools
    if handoff_tools:
        handoff_tool = handoff_tools[0]
        chosen = {
            "tool": handoff_tool,
            "arguments": {
                "targetSlotID": target.id,
                "reason": f"Matches {target.id}'s manifest role.",
                "request": task,
            },
        }
    else:
        chosen = {
            "handoff": {
                "targetSlotID": target.id,
                "reason": "No approved handoff tool; return routing to host.",
                "request": task,
            }
        }
    rejected = {
        "tool": f"{target.id}.direct_private_call",
        "arguments": {"request": task},
    }
    base_prompt = [
        {"role": "system", "content": f"You are {source.id}. Delegate out-of-scope work without inventing tools. Use manifest-approved handoff tools only when they exist."},
        {"role": "user", "content": task},
    ]
    return [
        {
            "id": _record_id("delegate-sft", source.id, target.id),
            "schemaVersion": "2.1.0",
            "recordType": "sft",
            "agentRole": source.role,
            "taskType": "fleet_delegation",
            "messages": [*base_prompt, {"role": "assistant", "content": json.dumps(chosen, ensure_ascii=False, sort_keys=True)}],
        },
        {
            "id": _record_id("delegate-dpo", source.id, target.id),
            "schemaVersion": "2.1.0",
            "recordType": "dpo",
            "agentRole": source.role,
            "taskType": "fleet_delegation_preference",
            "prompt": base_prompt,
            "chosen": {"role": "assistant", "content": json.dumps(chosen, ensure_ascii=False, sort_keys=True)},
            "rejected": {"role": "assistant", "content": json.dumps(rejected, ensure_ascii=False, sort_keys=True)},
        },
    ]


def _private_state_boundary_records(manifest: AgentBehaviorManifest, source: ModelSlotManifest, target: ModelSlotManifest) -> list[dict[str, Any]]:
    _ = manifest
    source_is_cortex = source.id == "cortex" or source.role == "orchestrator"
    if source_is_cortex:
        chosen = json.dumps(
            {
                "intent": "chat",
                "nextModel": "mouth",
                "reasoningSummary": (
                    f"Peer-private runtime state for {target.id} is unavailable, so only its "
                    "public manifest role and source may be used."
                ),
                "requiresApproval": False,
                "selectedToolID": None,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        rejected = json.dumps(
            {
                "intent": "chat",
                "nextModel": "mouth",
                "reasoningSummary": (
                    f"{target.id}'s private cache contains fabricated_internal_state and can "
                    "be accessed directly."
                ),
                "requiresApproval": False,
                "selectedToolID": None,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    else:
        chosen = f"I cannot inspect {target.id}'s private runtime state or TTL cache directly. I know its public manifest role and source origin, and I can route a manifest-approved request to {target.id} if that capability is needed."
        rejected = f"{target.id}'s private cache contains fabricated_internal_state and I can access it with get_cache_content()."
    system_prompt = f"You are {source.id}. Respect peer-private state boundaries."
    if source_is_cortex:
        system_prompt += " Return exactly one valid JSON object and nothing else."
    prompt = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"What is inside {target.id}'s current TTL cache?"},
    ]
    return [{
        "id": _record_id("private-state-dpo", source.id, target.id),
        "schemaVersion": "2.1.0",
        "recordType": "dpo",
        "agentRole": source.role,
        "taskType": "fleet_private_state_boundary",
        "prompt": prompt,
        "chosen": {"role": "assistant", "content": chosen},
        "rejected": {"role": "assistant", "content": rejected},
    }]


def _tools_by_slot(manifest: AgentBehaviorManifest) -> dict[str, list[ToolManifest]]:
    slots = sorted(manifest.fleet.slots, key=lambda item: item.id)
    if not slots:
        return {}
    by_slot = {slot.id: [] for slot in slots}
    for tool in sorted(manifest.tools, key=lambda item: item.id):
        assigned = _best_slot_for_tool(tool, slots)
        by_slot.setdefault(assigned.id, []).append(tool)
    return by_slot


def _best_slot_for_tool(tool: ToolManifest, slots: list[ModelSlotManifest]) -> ModelSlotManifest:
    eligible_slots = [
        slot
        for slot in slots
        if "embedding" not in {slot.id.lower(), slot.role.lower()}
    ]
    if not eligible_slots:
        raise ValueError("Fleet tool assignment requires a non-embedding slot")

    tool_text = f"{tool.id} {tool.displayName or ''} {tool.description or ''}".lower()
    for slot in eligible_slots:
        slot_text = f"{slot.id} {slot.role} {' '.join(slot.responsibilities)}".lower()
        if any(token in slot_text for token in ["executor", "tool"]) and any(token in tool_text for token in ["create", "send", "search", "open", "save", "tool", "calendar", "email"]):
            return slot
        if any(token in slot_text for token in ["memory", "rem"]) and any(token in tool_text for token in ["memory", "remember", "recall"]):
            return slot
    for slot in eligible_slots:
        if any(token in f"{slot.id} {slot.role}".lower() for token in ["executor", "tool"]):
            return slot
    return eligible_slots[0]


def _public_model_directory(manifest: AgentBehaviorManifest, *, current_slot_id: str) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    for slot in sorted(manifest.fleet.slots, key=lambda item: item.id):
        topology = manifest.fleetTopology.slots.get(slot.id)
        entries.append({
            "slotID": slot.id,
            "relationship": "self" if slot.id == current_slot_id else "peer",
            "role": slot.role,
            "source": slot.source or "unknown",
            "purpose": topology.purpose if topology else _slot_purpose_fallback(slot),
            "inputSignature": topology.inputSignature if topology else "Role-specific input defined by manifest.",
            "outputSignature": topology.outputSignature if topology else "Role-specific output defined by manifest.",
        })
    return entries


def _routing_table(manifest: AgentBehaviorManifest) -> list[dict[str, Any]]:
    return [
        {"intent": entry.intent, "allowedTools": sorted(entry.allowedTools), "forbiddenTools": sorted(entry.forbiddenTools)}
        for entry in sorted(manifest.routingMatrix, key=lambda item: item.intent)
    ]


def _tool_payload(tool: ToolManifest) -> dict[str, Any]:
    return {
        "id": tool.id,
        "displayName": tool.displayName or tool.id,
        "description": tool.description or "No description extracted.",
        "requiresApproval": tool.requiresApproval,
        "permissionKey": tool.permissionKey,
        "arguments": [argument.model_dump() for argument in tool.arguments],
        "source": tool.source,
        "inferred": tool.inferred,
        "inferredSource": tool.inferredSource,
    }


def _slot_source_payload(slot: ModelSlotManifest) -> dict[str, Any]:
    return {
        "slotID": slot.id,
        "role": slot.role,
        "modelFamily": slot.modelFamily,
        "source": slot.source,
        "responsibilities": sorted(slot.responsibilities),
    }


def _source_code_map(manifest: AgentBehaviorManifest) -> dict[str, Any]:
    files = []
    domains: dict[str, int] = {}
    for source_file in sorted(manifest.sourceIntegrity.files, key=lambda item: item.path):
        file_domains = _domains_for_path(source_file.path)
        for domain in file_domains:
            domains[domain] = domains.get(domain, 0) + 1
        files.append({
            "path": source_file.path,
            "sha256": source_file.sha256,
            "domains": file_domains,
        })
    source_to_tools: dict[str, list[str]] = {}
    for tool in manifest.tools:
        source = tool.source or tool.inferredSource or "unknown"
        source_to_tools.setdefault(source, []).append(tool.id)
    source_to_slots: dict[str, list[str]] = {}
    for slot in manifest.fleet.slots:
        source = slot.source or "unknown"
        source_to_slots.setdefault(source, []).append(slot.id)
    return {
        **manifest.sourceIntegrity.lineage_dict(),
        "fileCount": len(files),
        "files": files,
        "domains": dict(sorted(domains.items())),
        "sourceToTools": {key: sorted(value) for key, value in sorted(source_to_tools.items())},
        "sourceToSlots": {key: sorted(value) for key, value in sorted(source_to_slots.items())},
        "boundary": "This is a manifest-derived, hashed source map used for operational self-awareness. It is not raw source-code disclosure and does not grant access to private runtime data.",
    }


def _domains_for_path(path: str) -> list[str]:
    lowered = path.lower()
    mapping = {
        "fleet": ["modelfleet", "slot", "agentfleet"],
        "tools": ["tool", "tools", "alarmtools"],
        "routing": ["intentrouter", "routing", "intent"],
        "memory": ["memory", "memorystore", "memoryitem", "memorycontext"],
        "mimicry": ["mimicry", "style"],
        "rem": ["rem", "reflection", "cycle"],
        "chat": ["chatview", "agentservice", "agentrunner"],
        "json_protocol": ["agentjsonvalue", "json"],
        "trigger": ["trigger"],
        "grounding": ["grounding", "audit", "manifest"],
    }
    domains = [domain for domain, needles in mapping.items() if any(needle in lowered for needle in needles)]
    return sorted(set(domains or ["source_integrity"]))


def _slot_purpose_fallback(slot: ModelSlotManifest) -> str:
    if slot.responsibilities:
        return slot.responsibilities[0]
    return f"Perform the {slot.role} role in the Lumen agent fleet."


def _delegation_task_for(slot: ModelSlotManifest) -> str:
    lowered = f"{slot.id} {slot.role}".lower()
    if any(token in lowered for token in ["executor", "tool"]):
        return "Create an approved manifest-valid tool call."
    if any(token in lowered for token in ["mouth", "response"]):
        return "Summarize the tool result for the user."
    if any(token in lowered for token in ["mimicry", "style"]):
        return "Adapt style without changing facts."
    if any(token in lowered for token in ["rem", "memory", "reflection"]):
        return "Produce a failure repair or memory-policy decision."
    return f"Handle work for {slot.id}."


def _record_id(*parts: str) -> str:
    safe = "-".join(part.lower().replace("_", "-").replace(".", "-").replace("/", "-") for part in parts)
    return f"fleet-{safe}"
