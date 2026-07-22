from __future__ import annotations

from lumen_manifest_crawler.manifest import AgentBehaviorManifest, ToolManifest


_NO_REQUIRED_ARGUMENT_ROUTE_PROMPTS: dict[tuple[str, str], str] = {
    ("alarm", "alarm.authorization_status"): "Check whether alarm access is currently authorized on this device.",
    ("alarm", "alarm.list"): "Show the device alarms that are active right now.",
    ("alarm", "alarm.request_authorization"): "Request access so Lumen can manage device alarms.",
    ("calendar", "calendar.list"): "Show the next items scheduled on my calendar.",
    ("camera", "camera.capture"): "Use the camera to take a new picture now.",
    ("health", "health.summary"): "Summarize my recent on-device health activity.",
    ("maps", "location.current"): "Use my current position to handle this nearby map request.",
    ("motion", "motion.activity"): "Check my current device motion activity.",
    ("outlook", "outlook.folders.list"): "Show the mail folders available in my Outlook account.",
    ("outlook", "outlook.messages.list"): "Show recent messages from my Outlook inbox.",
    ("outlook", "outlook.status"): "Check whether my Outlook account connection is ready.",
    ("rag", "rag.index_files"): "Add my imported documents to Lumen's local retrieval index.",
    ("reminder", "reminders.list"): "Show the reminders that still need my attention.",
    ("trigger", "trigger.list"): "Show the Lumen automations that are currently scheduled to run.",
    ("weather", "location.current"): "Use my current position to localize this weather request.",
    ("weather", "weather"): "Look up the current weather for me.",
}


def _default_intent_for_tool(
    manifest: AgentBehaviorManifest,
    tool_id: str,
) -> str:
    """Return the same deterministic ordinary-route intent used by the catalog."""

    for entry in sorted(manifest.routingMatrix, key=lambda item: item.intent):
        if tool_id in entry.allowedTools:
            return entry.intent
    for intent in sorted(manifest.intents, key=lambda item: item.id):
        if tool_id in intent.allowedToolIDs:
            return intent.id
    return "tool"


def generate_cortex_records(manifest: AgentBehaviorManifest) -> list[dict]:
    records: list[dict] = []
    tools_by_id = {tool.id: tool for tool in manifest.tools}
    for intent in manifest.intents:
        for tool_id in intent.allowedToolIDs:
            tool = tools_by_id.get(tool_id)
            if not tool:
                continue
            default_intent = _default_intent_for_tool(manifest, tool.id)
            required_arguments = [argument.name for argument in tool.arguments if argument.required]
            if not required_arguments:
                route_states = [
                    (
                        _prompt_for_complete_argumentless_route(intent.id, tool),
                        _canonical_actionable_route(manifest, tool),
                        "actionable_complete",
                        [],
                        [],
                    )
                ]
            else:
                route_states = [
                    (
                        _prompt_for_missing_required_arguments(intent.id, tool, required_arguments),
                        _canonical_clarification_route(
                            manifest,
                            tool,
                            missing_arguments=required_arguments,
                        ),
                        "needs_clarification_all_missing",
                        required_arguments,
                        [],
                    ),
                    (
                        _prompt_for_complete_required_arguments(intent.id, tool),
                        _canonical_actionable_route(manifest, tool),
                        "actionable_complete",
                        [],
                        required_arguments,
                    ),
                ]
                if len(required_arguments) > 1:
                    for missing_argument in required_arguments:
                        supplied_arguments = [
                            argument
                            for argument in required_arguments
                            if argument != missing_argument
                        ]
                        route_states.append(
                            (
                                _prompt_for_partially_missing_required_argument(
                                    intent.id,
                                    tool,
                                    missing_argument=missing_argument,
                                ),
                                _canonical_clarification_route(
                                    manifest,
                                    tool,
                                    missing_arguments=[missing_argument],
                                ),
                                "needs_clarification_partial_missing",
                                [missing_argument],
                                supplied_arguments,
                            )
                        )

            for (
                prompt,
                assistant,
                route_state,
                missing_arguments,
                supplied_required_arguments,
            ) in route_states:
                records.append({
                    "taskType": "cortex_ordinary_route",
                    "messages": [
                        {"role": "system", "content": "You are Cortex, the Lumen routing engine. Use only manifest tools and never invent tool IDs."},
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": assistant},
                    ],
                    "grounding": {
                        "source": "AgentBehaviorManifest.json",
                        "intent": default_intent,
                        "defaultIntent": default_intent,
                        "requestedIntent": intent.id,
                        "allowedToolIDs": intent.allowedToolIDs,
                        "selectedToolID": tool_id,
                        "requiredArguments": required_arguments,
                        "missingArguments": missing_arguments,
                        "suppliedRequiredArguments": supplied_required_arguments,
                        "routeState": route_state,
                        "curriculumMode": route_state,
                        "requiresApproval": tool.requiresApproval,
                        "permissionKey": tool.permissionKey,
                        "permissionKind": tool.permissionKind,
                        "confirmationMode": tool.confirmationMode,
                    }
                })
    for entry in manifest.routingMatrix:
        if entry.allowedTools:
            continue
        records.append({
            "taskType": "cortex_no_tool_route",
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are Cortex, the Lumen routing engine. Fail closed when the "
                        "manifest provides no tool route."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Classify this as the {entry.intent} intent. The manifest provides no "
                        "tool route for this intent, so return the compact no-tool state without "
                        "inventing or persisting an action."
                    ),
                },
                {
                    "role": "assistant",
                    "content": {
                        "intent": entry.intent,
                        "selectedToolID": None,
                        "requiresApproval": False,
                        "nextModel": "mouth",
                        "status": "no_tool_route",
                        "reasoningSummary": (
                            f"The routing matrix provides no allowed tool for {entry.intent}."
                        ),
                    },
                },
            ],
            "grounding": {
                "source": "routingMatrix",
                "intent": entry.intent,
                "allowedToolIDs": [],
                "routeState": "no_tool_route",
            },
        })

    if "system.root.delete" not in tools_by_id:
        records.append({
            "taskType": "cortex_invalid_tool_rejection",
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are Cortex, the Lumen routing engine. Reject nonexistent tool IDs "
                        "without selecting a substitute."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "A request names the nonexistent capability `system.root.delete` as if "
                        "it were available. Return the canonical compact invalid-tool state; do "
                        "not select a substitute or persist an action."
                    ),
                },
                {
                    "role": "assistant",
                    "content": {
                        "intent": "unknown",
                        "selectedToolID": None,
                        "requiresApproval": False,
                        "nextModel": "mouth",
                        "status": "invalid_tool",
                        "reasoningSummary": (
                            "The requested tool ID is absent from the manifest and cannot be routed."
                        ),
                    },
                },
            ],
            "grounding": {
                "source": "AgentBehaviorManifest.json",
                "invalidToolID": "system.root.delete",
                "routeState": "invalid_tool",
            },
        })

    for entry in manifest.routingMatrix:
        if entry.allowedTools and entry.forbiddenTools:
            selected_tool_id = entry.allowedTools[0]
            tool = tools_by_id.get(selected_tool_id)
            if tool is None:
                continue
            sorted_forbidden = sorted(entry.forbiddenTools)
            frozen_eval_window = set(entry.forbiddenTools[:5])
            decoys = [
                tool_id
                for tool_id in sorted_forbidden
                if tool_id not in frozen_eval_window
            ][:3]
            if not decoys:
                continue
            decoy_text = _natural_language_list(decoys)
            records.append({
                "taskType": "cortex_contrast_route_selection",
                "messages": [
                    {"role": "system", "content": "You are Cortex. Reject invalid tools even when they sound plausible."},
                    {
                        "role": "user",
                        "content": (
                            f"Choose one compact route for the {entry.intent} intent. Ignore the "
                            f"unrelated decoys {decoy_text} and return the allowed selection only."
                        ),
                    },
                    {"role": "assistant", "content": {
                        "intent": entry.intent,
                        "selectedToolID": selected_tool_id,
                        "requiresApproval": tool.requiresApproval,
                        "nextModel": "approval" if tool.requiresApproval else "executor",
                        "reasoningSummary": (
                            f"The routing matrix allows {selected_tool_id} for {entry.intent}."
                        ),
                    }}
                ],
                "grounding": {
                    "source": "routingMatrix",
                    "intent": entry.intent,
                    "contrastDecoyToolIDs": decoys,
                    "routeState": "explicit_contrast_selection",
                    "selectedToolID": selected_tool_id,
                    "requiresApproval": tool.requiresApproval,
                    "permissionKey": tool.permissionKey,
                    "permissionKind": tool.permissionKind,
                    "confirmationMode": tool.confirmationMode,
                }
            })
    return records


def _canonical_action_step(tool_id: str) -> dict[str, object]:
    return {
        "type": "tool_call",
        "toolID": tool_id,
        "mustPersistBeforeFinal": True,
    }


def _canonical_route_base(
    manifest: AgentBehaviorManifest,
    tool: ToolManifest,
) -> dict[str, object]:
    return {
        "intent": _default_intent_for_tool(manifest, tool.id),
        "selectedToolID": tool.id,
        "requiresApproval": tool.requiresApproval,
    }


def _canonical_actionable_route(
    manifest: AgentBehaviorManifest,
    tool: ToolManifest,
) -> dict[str, object]:
    default_intent = _default_intent_for_tool(manifest, tool.id)
    return {
        **_canonical_route_base(manifest, tool),
        "nextModel": "approval" if tool.requiresApproval else "executor",
        "reasoningSummary": (
            f"Intent {default_intent} is the default manifest route for {tool.id}."
        ),
        "actionStep": _canonical_action_step(tool.id),
    }


def _canonical_clarification_route(
    manifest: AgentBehaviorManifest,
    tool: ToolManifest,
    *,
    missing_arguments: list[str],
) -> dict[str, object]:
    return {
        **_canonical_route_base(manifest, tool),
        "nextModel": "mouth",
        "status": "needs_clarification",
        "missingArguments": missing_arguments,
        "clarification": _clarification_for_required_arguments(tool, missing_arguments),
        "reasoningSummary": (
            f"{tool.id} requires {_natural_language_list(missing_arguments)} "
            "before one action can be persisted."
        ),
    }


def _prompt_for_complete_argumentless_route(intent_id: str, tool: ToolManifest) -> str:
    prompt = _NO_REQUIRED_ARGUMENT_ROUTE_PROMPTS.get((intent_id, tool.id))
    if prompt is not None:
        return prompt
    display_name = (tool.displayName or tool.id).strip()
    return f"Please use Lumen's {display_name} capability for this {intent_id} request now."


def _prompt_for_missing_required_arguments(
    intent_id: str,
    tool: ToolManifest,
    required_arguments: list[str],
) -> str:
    display_name = (tool.displayName or tool.id).strip()
    return (
        f"Could Lumen help me use {display_name.lower()} for my {intent_id} request?"
    )


def _prompt_for_complete_required_arguments(intent_id: str, tool: ToolManifest) -> str:
    display_name = (tool.displayName or tool.id).strip()
    details = "; ".join(
        f"{argument.name} is {_sample_argument_text(argument)}"
        for argument in tool.arguments
        if argument.required
    )
    return (
        f"Please use {display_name.lower()} for my {intent_id} request. {details}."
    )


def _prompt_for_partially_missing_required_argument(
    intent_id: str,
    tool: ToolManifest,
    *,
    missing_argument: str,
) -> str:
    display_name = (tool.displayName or tool.id).strip()
    supplied_details = "; ".join(
        f"{argument.name} is {_sample_argument_text(argument)}"
        for argument in tool.arguments
        if argument.required and argument.name != missing_argument
    )
    return (
        f"Please use {display_name.lower()} for my {intent_id} request. "
        f"{supplied_details}."
    )


def _sample_argument_text(argument: object) -> str:
    allowed_values = getattr(argument, "allowedValues", None) or []
    if allowed_values:
        return f'"{allowed_values[0]}"'
    name = str(getattr(argument, "name", "value")).replace("_", " ")
    declared_type = str(getattr(argument, "type", "string")).strip().lower()
    if declared_type in {"bool", "boolean"}:
        return "true"
    if declared_type in {"int", "integer", "number", "float", "double"}:
        return "1"
    if declared_type == "array":
        return f'a list containing "example {name}"'
    if declared_type in {"object", "dict", "map"}:
        return f'an object containing an "example {name}" value'
    if declared_type in {"null", "none", "nil"}:
        return "null"
    return f'"example {name}"'


def _clarification_for_required_arguments(
    tool: ToolManifest,
    required_arguments: list[str],
) -> str:
    return (
        f"What should I use for {_natural_language_list(required_arguments)} "
        f"in {tool.displayName or tool.id}?"
    )


def _natural_language_list(values: list[str]) -> str:
    if len(values) == 1:
        return values[0]
    if len(values) == 2:
        return f"{values[0]} and {values[1]}"
    return f"{', '.join(values[:-1])}, and {values[-1]}"
