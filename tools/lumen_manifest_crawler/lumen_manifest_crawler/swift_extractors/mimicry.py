from __future__ import annotations

import re

from lumen_manifest_crawler.swift_extractors.base import SwiftExtractor, SwiftFile, enum_cases, string_literals

MIMICRY_HINT_PATTERN = re.compile(r"\b(?:tone|style|direct|concise)\b", flags=re.I)


class MimicryExtractor(SwiftExtractor):
    target_names = ("MimicryProfile.swift",)

    def extract(self, file: SwiftFile, manifest) -> None:
        states = set(enum_cases(file.text, "MimicryState")) | set(enum_cases(file.text, "DetectedUserTone"))
        hints = []
        for literal in string_literals(file.text):
            if MIMICRY_HINT_PATTERN.search(literal):
                hints.append(literal)
        if states or hints:
            manifest.agentProtocols.cortexOutput["mimicryProfile"] = {
                "states": sorted(states),
                "constraints": sorted(dict.fromkeys(hints))[:100],
                "source": file.relpath,
            }
