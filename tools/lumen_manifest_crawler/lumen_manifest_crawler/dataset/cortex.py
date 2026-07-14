from __future__ import annotations

from lumen_manifest_crawler.manifest import AgentBehaviorManifest, ToolManifest


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
                    {"role": "user", "content": _prompt_for_intent(intent.id, tool)},
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


def _prompt_for_intent(intent_id: str, tool: ToolManifest) -> str:
    natural_prompts = {
        "calendar.create": "Create a calendar event for a meeting in 10 minutes.",
        "calendar.list": "What calendar events are coming up next?",
        "location.current": "What is my current location?",
        "mail.draft": "Draft an email update without sending it.",
        "maps.directions": "Open directions to the nearest hardware store.",
        "maps.search": "Find a hardware store nearby.",
        "outlook.draft.create": "Create and save an Outlook draft update.",
        "outlook.mail.send": "Send the approved Outlook email update.",
        "web.fetch": "Read the documentation page at the URL I supplied.",
        "web.search": "Search for current SwiftData migration details.",
    }
    if tool.id in natural_prompts:
        return f"{natural_prompts[tool.id]} Route it specifically as the `{intent_id}` intent."

    display_name = (tool.displayName or tool.id).strip()
    description = (tool.description or f"Use {display_name}").strip().rstrip(".")
    return f"{description}. Handle this `{intent_id}` request with the {display_name} capability."
