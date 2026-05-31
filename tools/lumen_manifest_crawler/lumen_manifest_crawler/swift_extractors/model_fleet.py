from __future__ import annotations

import re

from lumen_manifest_crawler.manifest import ModelSlotManifest
from lumen_manifest_crawler.swift_extractors.base import SwiftExtractor, SwiftFile, argument_value, balanced_call_blocks, clean_swift_string, string_literals


DEFAULT_SLOT_ROLES: dict[str, str] = {
    "cortex": "orchestrator",
    "executor": "tool_executor",
    "toolExecutor": "tool_executor",
    "mouth": "user_response",
    "mimicry": "tone_adapter",
    "rem": "idle_reflection",
}

DEFAULT_RESPONSIBILITIES: dict[str, list[str]] = {
    "orchestrator": ["intent routing", "tool selection", "model coordination", "task planning"],
    "tool_executor": ["strict JSON generation", "tool argument validation", "approval boundary enforcement"],
    "user_response": ["final user-facing response", "spoken output", "clarification"],
    "tone_adapter": ["tone detection", "style adaptation", "response rewriting"],
    "idle_reflection": ["memory pruning", "dataset generation", "failure analysis", "manifest audit"],
}


class ModelFleetExtractor(SwiftExtractor):
    target_names = ("ModelFleet.swift", "FleetStatusCard.swift")

    def extract(self, file: SwiftFile, manifest) -> None:
        contract = self._extract_contract(file.text)
        if contract:
            manifest.fleet.contractVersion = contract

        seen = {slot.id for slot in manifest.fleet.slots}
        for block in balanced_call_blocks(file.text, "LumenModelSlotContract") + balanced_call_blocks(file.text, "ModelSlot"):
            slot_id = self._extract_slot_id(block)
            if not slot_id or slot_id in seen:
                continue
            seen.add(slot_id)
            role = self._extract_role(block, slot_id)
            manifest.fleet.slots.append(
                ModelSlotManifest(
                    id=slot_id,
                    role=role,
                    modelFamily=clean_swift_string(argument_value(block, "modelFamily")),
                    responsibilities=self._extract_responsibilities(block, role),
                    source=file.relpath,
                )
            )

        # Safe fallback: if known slot names are literals in the fleet file, capture them.
        for literal in string_literals(file.text):
            if literal in DEFAULT_SLOT_ROLES and literal not in seen:
                role = DEFAULT_SLOT_ROLES[literal]
                seen.add(literal)
                manifest.fleet.slots.append(
                    ModelSlotManifest(
                        id=literal,
                        role=role,
                        responsibilities=DEFAULT_RESPONSIBILITIES.get(role, []),
                        source=file.relpath,
                    )
                )

    @staticmethod
    def _extract_contract(text: str) -> str | None:
        patterns = [
            r"contractVersion\s*=\s*\"([^\"]+)\"",
            r"fleetContractVersion\s*=\s*\"([^\"]+)\"",
            r"version\s*:\s*\"([^\"]+)\"",
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1)
        return None

    @staticmethod
    def _extract_slot_id(block: str) -> str | None:
        for label in ("id", "slot", "slotID", "agentID", "name"):
            value = clean_swift_string(argument_value(block, label))
            if value:
                return value
        literals = string_literals(block)
        return literals[0] if literals else None

    @staticmethod
    def _extract_role(block: str, slot_id: str) -> str:
        raw = clean_swift_string(argument_value(block, "role"))
        if raw:
            return raw
        return DEFAULT_SLOT_ROLES.get(slot_id, slot_id)

    @staticmethod
    def _extract_responsibilities(block: str, role: str) -> list[str]:
        if "responsibilities" not in block:
            return DEFAULT_RESPONSIBILITIES.get(role, [])
        extracted = ModelFleetExtractor._extract_array_literals(block, "responsibilities")
        values = [value for value in extracted if len(value.split()) > 1]
        return values or DEFAULT_RESPONSIBILITIES.get(role, [])

    @staticmethod
    def _extract_array_literals(block: str, label: str) -> list[str]:
        match = re.search(rf"\b{re.escape(label)}\s*:\s*\[", block)
        if not match:
            match = re.search(rf"\b{re.escape(label)}\s*=\s*\[", block)
        if not match:
            return []
        start = match.end() - 1
        depth = 0
        in_string = False
        escaped = False
        for index in range(start, len(block)):
            char = block[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
            else:
                if char == '"':
                    in_string = True
                elif char == "[":
                    depth += 1
                elif char == "]":
                    depth -= 1
                    if depth == 0:
                        return string_literals(block[start:index + 1])
        return []
