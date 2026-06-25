from __future__ import annotations

from lumen_manifest_crawler.manifest import AgentBehaviorManifest


def generate_cortex_records(manifest: AgentBehaviorManifest) -> list[dict]:
    records: list[dict] = []
    tools_by_id = {tool.id: tool for tool in manifest.tools}
    for intent in manifest.intents:
        for tool_id in intent.allowedToolIDs:
            tool = tools_by_id.get(tool_id)
            if not tool:
                continue
            records.append({
                "messages": [
                    {"role": "system", "content": "You are Cortex, the Lumen routing engine. Use only manifest tools and never invent tool IDs."},
                    {"role": "user", "content": _prompt_for_intent(intent.id, tool_id)},
                    {"role": "assistant", "content": {
                        "intent": intent.id,
                        "selectedToolID": tool_id,
                        "requiresApproval": tool.requiresApproval,
                        "permissionKey": tool.permissionKey,
                        "permissionKind": tool.permissionKind,
                        "confirmationMode": tool.confirmationMode,
                        "nextModel": "approval" if tool.requiresApproval else "executor",
                        "reasoningSummary": f"Intent {intent.id} is allowed to use {tool_id} by the manifest routing matrix."
                    }}
                ],
                "grounding": {
                    "source": "AgentBehaviorManifest.json",
                    "intent": intent.id,
                    "allowedToolIDs": intent.allowedToolIDs,
                    "selectedToolID": tool_id,
                    "requiresApproval": tool.requiresApproval,
                    "permissionKey": tool.permissionKey,
                    "permissionKind": tool.permissionKind,
                    "confirmationMode": tool.confirmationMode,
                }
            })
    for entry in manifest.routingMatrix:
        if entry.allowedTools and entry.forbiddenTools:
            selected_tool_id = entry.allowedTools[0]
            tool = tools_by_id.get(selected_tool_id)
            if tool is None:
                continue
            records.append({
                "messages": [
                    {"role": "system", "content": "You are Cortex. Reject invalid tools even when they sound plausible."},
                    {"role": "user", "content": f"For intent {entry.intent}, should I use {entry.forbiddenTools[0]}?"},
                    {"role": "assistant", "content": {
                        "intent": entry.intent,
                        "selectedToolID": selected_tool_id,
                        "rejectedToolID": entry.forbiddenTools[0],
                        "requiresApproval": tool.requiresApproval,
                        "permissionKey": tool.permissionKey,
                        "permissionKind": tool.permissionKind,
                        "confirmationMode": tool.confirmationMode,
                        "nextModel": "approval" if tool.requiresApproval else "executor",
                        "reasoningSummary": f"{entry.forbiddenTools[0]} is not allowed for {entry.intent}; use {selected_tool_id}."
                    }}
                ],
                "grounding": {
                    "source": "routingMatrix",
                    "intent": entry.intent,
                    "selectedToolID": selected_tool_id,
                    "requiresApproval": tool.requiresApproval,
                    "permissionKey": tool.permissionKey,
                    "permissionKind": tool.permissionKind,
                    "confirmationMode": tool.confirmationMode,
                }
            })
    return records


def _prompt_for_intent(intent_id: str, tool_id: str) -> str:
    if "map" in tool_id or "local" in intent_id.lower():
        return "Find a hardware store nearby."
    if "calendar" in tool_id:
        return "Create a calendar event for a meeting in 10 minutes."
    if "mail" in tool_id or "email" in tool_id:
        return "Draft an email update."
    if "web" in tool_id:
        return "Search for current SwiftData migration details."
    return f"Handle user intent {intent_id}."
