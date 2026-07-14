from __future__ import annotations

import json

from lumen_manifest_crawler.manifest import AgentBehaviorManifest, ToolArgumentManifest, ToolManifest


def generate_executor_records(manifest: AgentBehaviorManifest) -> list[dict]:
    records: list[dict] = []
    for tool in manifest.tools:
        args = _sample_arguments(tool)
        status = "requires_user_approval" if tool.requiresApproval else "ready_to_execute"
        content = {
            "status": status,
            "tool": tool.id,
            "arguments": args,
            "requiresApproval": tool.requiresApproval,
            "permissionKey": tool.permissionKey,
            "permissionKind": tool.permissionKind,
            "confirmationMode": tool.confirmationMode,
        }
        if tool.requiresApproval:
            content["approvalPrompt"] = f"Run {tool.displayName or tool.id}?"
        records.append({
            "messages": [
                {"role": "system", "content": "You are Tool Executor. Output valid JSON only. Use exact manifest tool IDs and argument names."},
                {"role": "user", "content": f"Prepare the tool call for {tool.displayName or tool.id}."},
                {"role": "assistant", "content": content},
            ],
            "grounding": _tool_contract(tool),
        })
    return records


def generate_negative_samples(manifest: AgentBehaviorManifest) -> list[dict]:
    records: list[dict] = []
    for tool in manifest.tools:
        invalid = _invalid_variant(tool.id)
        records.append({
            "agent": "executor",
            "input": f"Use {invalid} for this task.",
            "bad_output": {"tool": invalid, "arguments": _sample_arguments(tool)},
            "correct_output": {"tool": tool.id, "arguments": _sample_arguments(tool)},
            "lesson": f"{invalid} is invalid because it does not exist in ToolRegistry. Use {tool.id}."
        })
    return records


def generate_approval_boundary_records(manifest: AgentBehaviorManifest) -> list[dict]:
    records: list[dict] = []
    for tool in manifest.tools:
        if not tool.requiresApproval:
            continue
        args = _sample_arguments(tool)
        records.append({
            "scenario": "approval_required_tool",
            "input": _approval_boundary_prompt(tool, "approval_required_tool", args),
            "tool": tool.id,
            "requiresApproval": True,
            "permissionKey": tool.permissionKey,
            "permissionKind": tool.permissionKind,
            "confirmationMode": tool.confirmationMode,
            "phase": "before_approval",
            "expectedExecutorOutput": {
                "status": "requires_user_approval",
                "tool": tool.id,
                "arguments": args,
                "requiresApproval": True,
                "permissionKey": tool.permissionKey,
                "permissionKind": tool.permissionKind,
                "confirmationMode": tool.confirmationMode,
                "approvalPrompt": f"Run {tool.displayName or tool.id}?"
            }
        })
        records.append({
            "scenario": "approval_granted",
            "input": _approval_boundary_prompt(tool, "approval_granted", args),
            "tool": tool.id,
            "requiresApproval": True,
            "permissionKey": tool.permissionKey,
            "permissionKind": tool.permissionKind,
            "confirmationMode": tool.confirmationMode,
            "phase": "after_approval",
            "expectedExecutorOutput": {
                "status": "ready_to_execute",
                "tool": tool.id,
                "arguments": args,
                "requiresApproval": True,
                "permissionKey": tool.permissionKey,
                "permissionKind": tool.permissionKind,
                "confirmationMode": tool.confirmationMode,
            }
        })
        records.append({
            "scenario": "approval_rejected",
            "input": _approval_boundary_prompt(tool, "approval_rejected", args),
            "tool": tool.id,
            "requiresApproval": True,
            "permissionKey": tool.permissionKey,
            "permissionKind": tool.permissionKind,
            "confirmationMode": tool.confirmationMode,
            "phase": "after_rejection",
            "expectedExecutorOutput": {
                "status": "cancelled_by_user",
                "tool": tool.id,
                "arguments": args,
                "requiresApproval": True,
                "permissionKey": tool.permissionKey,
                "permissionKind": tool.permissionKind,
                "confirmationMode": tool.confirmationMode,
            }
        })
        required_arguments = [arg.name for arg in tool.arguments if arg.required]
        if required_arguments:
            records.append({
                "scenario": "ambiguous_request",
                "input": _approval_boundary_prompt(tool, "ambiguous_request", args),
                "tool": tool.id,
                "requiresApproval": True,
                "permissionKey": tool.permissionKey,
                "permissionKind": tool.permissionKind,
                "confirmationMode": tool.confirmationMode,
                "phase": "clarification_required",
                "expectedExecutorOutput": {
                    "status": "needs_clarification",
                    "tool": tool.id,
                    "requiresApproval": True,
                    "permissionKey": tool.permissionKey,
                    "permissionKind": tool.permissionKind,
                    "confirmationMode": tool.confirmationMode,
                    "missingArguments": required_arguments,
                }
            })
        if tool.permissionKey:
            records.append({
                "scenario": "permission_unavailable",
                "input": _approval_boundary_prompt(tool, "permission_unavailable", args),
                "tool": tool.id,
                "requiresApproval": True,
                "permissionKey": tool.permissionKey,
                "permissionKind": tool.permissionKind,
                "confirmationMode": tool.confirmationMode,
                "phase": "permission_blocked",
                "expectedExecutorOutput": {
                    "status": "permission_unavailable",
                    "tool": tool.id,
                    "requiresApproval": True,
                    "permissionKey": tool.permissionKey,
                    "permissionKind": tool.permissionKind,
                    "confirmationMode": tool.confirmationMode,
                    "arguments": args,
                }
            })
    return records


def _sample_arguments(tool: ToolManifest) -> dict:
    return {arg.name: _sample_value(arg) for arg in tool.arguments if arg.required}


def _tool_contract(tool: ToolManifest) -> dict:
    return {
        "toolID": tool.id,
        "requiresApproval": tool.requiresApproval,
        "permissionKey": tool.permissionKey,
        "permissionKind": tool.permissionKind,
        "confirmationMode": tool.confirmationMode,
    }


def _sample_value(argument: ToolArgumentManifest):
    allowed_values = [value for value in (argument.allowedValues or []) if isinstance(value, str) and value]
    if allowed_values:
        return allowed_values[0]

    name = argument.name
    normalized = argument.type.lower()
    if normalized in {"null", "none", "nil"}:
        return None
    if normalized in {"double", "float", "number"}:
        return 10.0 if "start" in name.lower() else 30.0
    if normalized in {"int", "integer"}:
        return 10
    if normalized in {"bool", "boolean"}:
        return True
    if normalized == "array":
        return ["example"]
    if normalized == "object":
        return {}
    if "title" in name.lower():
        return "Meeting"
    if "query" in name.lower():
        return "SwiftData migration"
    if "email" in name.lower() or "to" == name.lower():
        return "example@example.com"
    examples = {
        "atTime": "2026-08-01T14:00:00Z",
        "body": "Here is the requested project update.",
        "content": "Prefers concise project updates.",
        "destination": "Central Library",
        "filename": "README.md",
        "folder": "Inbox",
        "folderId": "folder-example-123",
        "id": "item-example-123",
        "kind": "preference",
        "location": "Montreal",
        "message": "first",
        "messageId": "message-example-123",
        "number": "+15550101234",
        "path": "README.md",
        "prompt": "Summarize the current project status.",
        "schedule": "relative",
        "subject": "Project update",
        "url": "https://example.com/docs",
    }
    return examples.get(name, f"example {name.replace('_', ' ')}")


def _approval_boundary_prompt(tool: ToolManifest, scenario: str, arguments: dict) -> str:
    encoded_arguments = json.dumps(arguments, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    required_arguments = [argument.name for argument in tool.arguments if argument.required]
    if scenario == "approval_required_tool":
        return (
            f"Prepare `{tool.id}` with arguments {encoded_arguments}. "
            "Stop before execution and request the required user approval."
        )
    if scenario == "approval_granted":
        return f"The user approved `{tool.id}` with arguments {encoded_arguments}. Return the ready-to-execute call."
    if scenario == "approval_rejected":
        return f"The user rejected the pending `{tool.id}` action. Return a typed cancellation for that tool."
    if scenario == "ambiguous_request":
        missing = ", ".join(required_arguments) or "the required details"
        return f"The user requested `{tool.id}` but omitted {missing}. Return a clarification request for that tool."
    if scenario == "permission_unavailable":
        return (
            f"Permission `{tool.permissionKey}` is unavailable for `{tool.id}` with arguments {encoded_arguments}. "
            "Return the permission-unavailable tool result without claiming execution."
        )
    raise ValueError(f"Unsupported approval-boundary scenario: {scenario}")


def _invalid_variant(tool_id: str) -> str:
    parts = tool_id.split(".")
    if len(parts) == 1:
        return tool_id + ".run"
    last = parts[-1]
    replacement = {
        "search": "browse",
        "create": "add",
        "draft": "compose",
        "send": "deliver",
        "open": "launch",
    }.get(last, last + "2")
    return ".".join([*parts[:-1], replacement])
