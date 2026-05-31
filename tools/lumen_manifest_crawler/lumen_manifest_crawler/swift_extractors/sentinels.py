from __future__ import annotations

import re

from lumen_manifest_crawler.swift_extractors.base import SwiftExtractor, SwiftFile, string_literals


DEFAULT_FORBIDDEN = {
    "<user_final_text>",
    "<private_reasoning>",
    "<tool_json>",
    "<internal_state>",
    "<scratchpad>",
    "<hidden_reasoning>",
}
SENTINEL_WORDS = {
    "final",
    "hidden",
    "internal",
    "private",
    "reasoning",
    "scratchpad",
    "state",
    "tool",
    "user",
}


class SentinelExtractor(SwiftExtractor):
    target_names = ("ChatView.swift", "AgentService.swift", "ComposeController.swift")

    def extract(self, file: SwiftFile, manifest) -> None:
        found = set(manifest.sentinels.forbiddenInUserOutput)
        found.update(DEFAULT_FORBIDDEN)
        for literal in string_literals(file.text):
            lowered = literal.lower()
            if self._looks_like_internal_sentinel(lowered):
                found.add(literal)
            if "private_reasoning" in lowered or "scratchpad" in lowered or "internal_state" in lowered:
                found.add(literal)
        manifest.sentinels.forbiddenInUserOutput = sorted(found)

    @staticmethod
    def _looks_like_internal_sentinel(value: str) -> bool:
        if not re.fullmatch(r"<[a-z0-9_\-]+>", value):
            return False
        inner = value[1:-1]
        tokens = set(re.split(r"[_\-]+", inner))
        return bool(tokens.intersection(SENTINEL_WORDS))
