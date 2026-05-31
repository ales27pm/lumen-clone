from __future__ import annotations

from lumen_manifest_crawler.manifest import AgentBehaviorManifest


def generate_rem_records(manifest: AgentBehaviorManifest) -> list[dict]:
    records: list[dict] = []
    existing_tool_ids = {tool.id for tool in manifest.tools}
    if manifest.tools:
        valid_tool = manifest.tools[0].id
        invalid_tool = _unique_invalid_variant(valid_tool, existing_tool_ids)
        records.append({
            "messages": [
                {"role": "system", "content": "You are REM. Analyze traces, compress memory, and recommend training repairs from the manifest."},
                {"role": "user", "content": f"The executor attempted {invalid_tool}, but the manifest only contains {valid_tool}."},
                {"role": "assistant", "content": {
                    "issue": "invalid_tool_id",
                    "invalidOutput": invalid_tool,
                    "validReplacement": valid_tool,
                    "recommendedTrainingRecord": {
                        "agent": "executor",
                        "lesson": f"Use {valid_tool}. Never emit {invalid_tool} because it is not present in ToolRegistry."
                    }
                }}
            ],
            "grounding": {"source": "AgentBehaviorManifest.json", "role": "rem"}
        })
    for freshness in manifest.memory.freshnessClasses:
        records.append({
            "messages": [
                {"role": "system", "content": "You are REM. Apply memory freshness and TTL policy from the manifest."},
                {"role": "user", "content": f"Categorize a memory for freshness class {freshness.id}."},
                {"role": "assistant", "content": {
                    "memoryFreshnessClass": freshness.id,
                    "ttlSeconds": freshness.ttlSeconds,
                    "durable": freshness.durable,
                    "action": "preserve" if freshness.durable else "prune_after_ttl"
                }}
            ],
            "grounding": {"source": freshness.source or "AgentBehaviorManifest.json"}
        })
    return records


def _unique_invalid_variant(tool_id: str, existing_tool_ids: set[str]) -> str:
    candidate = _invalid_variant(tool_id)
    if candidate not in existing_tool_ids:
        return candidate
    suffix = 1
    while True:
        regenerated = f"{candidate}Invalid{suffix}"
        if regenerated not in existing_tool_ids:
            return regenerated
        suffix += 1


def _invalid_variant(tool_id: str) -> str:
    parts = tool_id.split(".")
    if len(parts) < 2:
        return tool_id + ".fake"
    replacement = "browse" if parts[-1] == "search" else parts[-1] + "Fake"
    return ".".join([*parts[:-1], replacement])
