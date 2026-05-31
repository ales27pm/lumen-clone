from __future__ import annotations

from lumen_manifest_crawler.manifest import AgentBehaviorManifest


def generate_mouth_records(manifest: AgentBehaviorManifest) -> list[dict]:
    records: list[dict] = []
    for tool in manifest.tools:
        if tool.requiresApproval:
            records.append({
                "messages": [
                    {"role": "system", "content": "You are Mouth. Produce user-facing text only. Do not choose tools or reveal internal markers."},
                    {"role": "user", "content": f"Cortex says {tool.id} requires approval. Ask the user for confirmation."},
                    {"role": "assistant", "content": f"Do you want me to run {tool.displayName or tool.id}?"},
                ],
                "grounding": {"responseMode": "approval_request", "forbidden": list(manifest.sentinels.forbiddenInUserOutput)},
            })
        else:
            records.append({
                "messages": [
                    {"role": "system", "content": "You are Mouth. Produce final concise user-facing text only."},
                    {"role": "user", "content": f"Executor completed {tool.id}."},
                    {"role": "assistant", "content": "Done."},
                ],
                "grounding": {"responseMode": "completion", "toolID": tool.id},
            })
    return records
