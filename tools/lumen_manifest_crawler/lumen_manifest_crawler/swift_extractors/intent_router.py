from __future__ import annotations

import re

from lumen_manifest_crawler.manifest import IntentManifest, RoutingMatrixEntry
from lumen_manifest_crawler.swift_extractors.base import SwiftExtractor, SwiftFile, enum_cases, string_literals


class IntentRouterExtractor(SwiftExtractor):
    target_names = ("IntentRouter.swift",)

    def extract(self, file: SwiftFile, manifest) -> None:
        intent_ids = set(enum_cases(file.text, "UserIntent"))
        existing = {i.id for i in manifest.intents}
        for intent_id in sorted(intent_ids):
            if intent_id not in existing:
                manifest.intents.append(IntentManifest(id=intent_id, source=file.relpath))

        known_tool_ids = {t.id for t in manifest.tools}
        for intent in manifest.intents:
            if intent.id not in intent_ids:
                continue
            allowed = self._tools_near_name(file.text, intent.id, known_tool_ids)
            if allowed:
                intent.allowedToolIDs = sorted(set(intent.allowedToolIDs).union(allowed))

        known_tools = sorted({t.id for t in manifest.tools})
        manifest.routingMatrix = [
            RoutingMatrixEntry(
                intent=intent.id,
                allowedTools=sorted(intent.allowedToolIDs),
                forbiddenTools=[tool_id for tool_id in known_tools if tool_id not in intent.allowedToolIDs][:25],
            )
            for intent in manifest.intents
        ]

    @staticmethod
    def _tools_near_name(text: str, name: str, known_tool_ids: set[str]) -> list[str]:
        allowed: set[str] = set()
        for block in IntentRouterExtractor._switch_case_blocks(text, name):
            for literal in string_literals(block):
                if literal in known_tool_ids:
                    allowed.add(literal)
        if allowed:
            return sorted(allowed)

        # Fallback for non-switch routing tables: inspect only the line or small assignment
        # that contains the exact intent name, not the whole switch/router body.
        pattern = re.compile(rf"(?m)^.*\b{re.escape(name)}\b.*$")
        for match in pattern.finditer(text):
            line = match.group(0)
            for literal in string_literals(line):
                if literal in known_tool_ids:
                    allowed.add(literal)
        return sorted(allowed)

    @staticmethod
    def _switch_case_blocks(text: str, name: str) -> list[str]:
        blocks: list[str] = []
        case_pattern = re.compile(rf"\bcase\s+\.{re.escape(name)}\b\s*:")
        next_case_pattern = re.compile(r"\n\s*case\s+\.")
        for match in case_pattern.finditer(text):
            start = match.end()
            next_case = next_case_pattern.search(text, start)
            end = next_case.start() if next_case else IntentRouterExtractor._case_block_end(text, start)
            blocks.append(text[start:end])
        return blocks

    @staticmethod
    def _case_block_end(text: str, start: int) -> int:
        closing = text.find("\n        }", start)
        if closing != -1:
            return closing
        closing = text.find("\n    }", start)
        if closing != -1:
            return closing
        return min(len(text), start + 500)
