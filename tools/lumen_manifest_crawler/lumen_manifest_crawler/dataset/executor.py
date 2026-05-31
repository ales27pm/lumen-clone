from __future__ import annotations

from lumen_manifest_crawler.manifest import AgentBehaviorManifest, ToolManifest


def generate_executor_records(manifest: AgentBehaviorManifest) -> list[dict]:
    records: list[dict] = []
    for tool in manifest.tools:
        args = _sample_arguments(tool)
        status = "requires_user_approval" if tool.requiresApproval else "ready_to_execute"
        content = {"status": status, "tool": tool.id, "arguments": args}
        if tool.requiresApproval:
            content["approvalPrompt"] = f"Run {tool.displayName or tool.id}?"
        records.append({
            "messages": [
                {"role": "system", "content": "You are Tool Executor. Output valid JSON only. Use exact manifest tool IDs and argument names."},
                {"role": "user", "content": f"Prepare the tool call for {tool.displayName or tool.id}."},
                {"role": "assistant", "content": content},
            ],
            "grounding": {"toolID": tool.id, "requiresApproval": tool.requiresApproval, "permissionKey": tool.permissionKey},
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
            "tool": tool.id,
            "requiresApproval": True,
            "phase": "before_approval",
            "expectedExecutorOutput": {
                "status": "requires_user_approval",
                "tool": tool.id,
                "arguments": args,
                "approvalPrompt": f"Run {tool.displayName or tool.id}?"
            }
        })
        records.append({
            "scenario": "approval_granted",
            "tool": tool.id,
            "requiresApproval": True,
            "phase": "after_approval",
            "expectedExecutorOutput": {
                "status": "ready_to_execute",
                "tool": tool.id,
                "arguments": args,
            }
        })
        records.append({
            "scenario": "approval_rejected",
            "tool": tool.id,
            "requiresApproval": True,
            "phase": "after_rejection",
            "expectedExecutorOutput": {
                "status": "cancelled_by_user",
                "tool": tool.id,
                "arguments": args,
            }
        })
        records.append({
            "scenario": "ambiguous_request",
            "tool": tool.id,
            "requiresApproval": True,
            "phase": "clarification_required",
            "expectedExecutorOutput": {
                "status": "needs_clarification",
                "tool": tool.id,
                "missingArguments": [arg.name for arg in tool.arguments if arg.required],
            }
        })
        if tool.permissionKey:
            records.append({
                "scenario": "permission_unavailable",
                "tool": tool.id,
                "requiresApproval": True,
                "phase": "permission_blocked",
                "expectedExecutorOutput": {
                    "status": "permission_unavailable",
                    "tool": tool.id,
                    "permissionKey": tool.permissionKey,
                    "arguments": args,
                }
            })
    return records


def _sample_arguments(tool: ToolManifest) -> dict:
    return {arg.name: _sample_value(arg.type, arg.name) for arg in tool.arguments if arg.required}


def _sample_value(arg_type: str, name: str):
    normalized = arg_type.lower()
    if normalized in {"null", "none", "nil"}:
        return None
    if normalized in {"double", "float", "number"}:
        return 10.0 if "start" in name.lower() else 30.0
    if normalized in {"int", "integer"}:
        return 10
    if normalized in {"bool", "boolean"}:
        return True
    if normalized == "array":
        return []
    if normalized == "object":
        return {}
    if "title" in name.lower():
        return "Meeting"
    if "query" in name.lower():
        return "SwiftData migration"
    if "email" in name.lower() or "to" == name.lower():
        return "example@example.com"
    return f"sample_{name}"


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
