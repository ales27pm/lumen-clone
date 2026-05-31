from __future__ import annotations

from lumen_manifest_crawler.manifest import AgentBehaviorManifest


def generate_mimicry_records(manifest: AgentBehaviorManifest) -> list[dict]:
    records: list[dict] = []
    base_states = [
        ("This build keeps failing again. I'm losing my mind.", "frustrated_focused", "direct"),
        ("Can you just give me the fix without the essay?", "impatient_technical", "concise"),
        ("I need this to work on device, not as a fake demo.", "precision_demanding", "clinical"),
    ]
    for user_text, state, tone in base_states:
        records.append({
            "messages": [
                {"role": "system", "content": "You are Mimicry. Detect user state and produce style constraints only. Do not answer the user directly."},
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": {
                    "detectedState": state,
                    "styleProfile": {
                        "length": "short",
                        "tone": tone,
                        "warmth": "low",
                        "confidence": "high",
                        "avoid": ["long introduction", "generic reassurance", "overexplaining"],
                    }
                }}
            ],
            "grounding": {"source": "AgentBehaviorManifest.json", "role": "mimicry"}
        })
    return records
