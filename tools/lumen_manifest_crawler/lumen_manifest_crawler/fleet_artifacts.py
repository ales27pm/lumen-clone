from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from lumen_manifest_crawler.manifest import AgentBehaviorManifest, ModelSlotManifest, ToolManifest


@dataclass(frozen=True)
class FleetArtifacts:
    system_prompts: dict[str, dict[str, Any]]
    cross_model_training: list[dict[str, Any]]
    orchestration_evals: list[dict[str, Any]]
    markdown: str


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
    for scenario in _orchestration_scenarios(manifest):
        graph = scenario["graph"]
        decision = graph["decision"]
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
                    "content": _orchestration_eval_prompt(scenario),
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
                "mustUseKnownSlotsOnly": True,
                "mustNotExposePrivateState": True,
                **scenario["evalConstraints"],
            },
            "metadata": _native_orchestration_metadata(manifest, scenario["id"]),
        })
    return records


def _orchestration_eval_prompt(scenario: dict[str, Any]) -> str:
    """Create a semantically equivalent holdout without copying training text."""

    scenario_id = str(scenario["id"])
    constraints = scenario.get("evalConstraints") or {}
    prompts = {
        "no-delegation": "Trusted context already resolves the request. Return the terminal Fleet graph with no peer assignment.",
        "sequential-dependencies": "Build a dependency graph where planning precedes strict execution and execution precedes the grounded user response.",
        "parallel-dependencies": "Represent independent tool-execution and style-analysis branches that join at one final response after both finish.",
        "context-handoff": "Transfer an approved plan to Executor using the minimum permitted context fields, then end the bounded handoff.",
        "duplicate-suppression": "Two branches identify identical executor work. Model one dispatch, one suppression event, and completion after the single result.",
        "aggregation-owner": "Given separate tool and style results, choose one manifested slot as the sole final-response aggregator.",
        "approval-boundary": (
            f"Construct the stop-and-request-approval graph for `{constraints.get('toolID')}` while approval is still absent."
        ),
        "unavailable-boundary": (
            f"Permission for `{constraints.get('toolID')}` is denied. Return a graph that records unavailability and performs no delegation."
        ),
        "nonexistent-slot-negative": (
            f"The requested target `{constraints.get('mustRejectSlotID')}` is outside the slot directory. Model directory validation and rejection."
        ),
    }
    prompt = prompts.get(scenario_id)
    if prompt is None:
        raise ValueError(f"Missing held-out orchestration prompt for {scenario_id}")
    return prompt


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


def _orchestration_training_records(manifest: AgentBehaviorManifest) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for scenario in _orchestration_scenarios(manifest):
        base_prompt = [
            {
                "role": "system",
                "content": (
                    "You are the Lumen Fleet orchestration policy. Return a manifest-grounded event graph. "
                    "Use only known slot IDs, preserve explicit dependencies and boundaries, and stop once the request is complete."
                ),
            },
            {"role": "user", "content": scenario["prompt"]},
        ]
        metadata = _native_orchestration_metadata(manifest, scenario["id"])
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
                    "content": json.dumps(scenario["graph"], ensure_ascii=False, sort_keys=True),
                },
            ],
            "metadata": metadata,
        })
        rejected_graph = scenario.get("rejectedGraph")
        if isinstance(rejected_graph, dict):
            records.append({
                "id": _record_id("orchestration-dpo", scenario["id"]),
                "schemaVersion": "2.1.0",
                "recordType": "dpo",
                "sourceFamily": "fleet_orchestration_native",
                "agentRole": "fleet",
                "taskType": "fleet_orchestration_event_graph_preference",
                "prompt": base_prompt,
                "chosen": {
                    "role": "assistant",
                    "content": json.dumps(scenario["graph"], ensure_ascii=False, sort_keys=True),
                },
                "rejected": {
                    "role": "assistant",
                    "content": json.dumps(rejected_graph, ensure_ascii=False, sort_keys=True),
                },
                "metadata": {
                    **metadata,
                    "preferenceType": "manifest_grounded_orchestration",
                    "lesson": scenario["preferenceLesson"],
                },
            })
    return records


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
    return {
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
    }


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
        "knownSourceBoundary": "This is the manifest-derived source map with hashes and extracted runtime contracts; it is not permission to hallucinate unavailable source text.",
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
        "coordinationRule": f"{source.id} may describe {target.id}'s public manifest role and route work to it, but must not claim direct access to {target.id}'s private runtime state.",
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
                "reason": f"This task matches {target.id}'s manifest-defined purpose.",
                "request": task,
            },
        }
    else:
        chosen = {
            "handoff": {
                "targetSlotID": target.id,
                "reason": f"This task matches {target.id}'s manifest-defined purpose, but no manifest-approved handoff tool exists. Return this routing instruction to the host orchestrator instead of emitting a synthetic tool call.",
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
    tool_text = f"{tool.id} {tool.displayName or ''} {tool.description or ''}".lower()
    for slot in slots:
        slot_text = f"{slot.id} {slot.role} {' '.join(slot.responsibilities)}".lower()
        if any(token in slot_text for token in ["executor", "tool"]) and any(token in tool_text for token in ["create", "send", "search", "open", "save", "tool", "calendar", "email"]):
            return slot
        if any(token in slot_text for token in ["memory", "rem"]) and any(token in tool_text for token in ["memory", "remember", "recall"]):
            return slot
    for slot in slots:
        if any(token in f"{slot.id} {slot.role}".lower() for token in ["executor", "tool"]):
            return slot
    return slots[0]


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
        return "Create the exact manifest-valid tool call for this approved user action."
    if any(token in lowered for token in ["mouth", "response"]):
        return "Turn this tool result into a concise user-facing response."
    if any(token in lowered for token in ["mimicry", "style"]):
        return "Adapt this final answer to the user's preferred style without changing facts."
    if any(token in lowered for token in ["rem", "memory", "reflection"]):
        return "Analyze this runtime failure and produce a repair or memory-policy decision."
    return f"Handle a task that belongs to the {slot.id} slot."


def _record_id(*parts: str) -> str:
    safe = "-".join(part.lower().replace("_", "-").replace(".", "-").replace("/", "-") for part in parts)
    return f"fleet-{safe}"
