from __future__ import annotations

import re

from lumen_manifest_crawler.swift_extractors.base import SwiftExtractor, SwiftFile, enum_cases

VALID_AGENT_JSON_TYPES = {"array", "bool", "null", "number", "object", "string"}
NUMERIC_MARKERS = {"double", "float", "int", "integer", "number"}


class AgentJSONValueExtractor(SwiftExtractor):
    target_names = ("AgentJSONValue.swift",)

    def extract(self, file: SwiftFile, manifest) -> None:
        cases = {case for case in enum_cases(file.text, "AgentJSONValue") if case in VALID_AGENT_JSON_TYPES}
        inferred = self._infer_types(file.text)
        supported = sorted(cases.union(inferred))
        if supported:
            manifest.agentProtocols.executorOutput["supportedJSONTypes"] = supported
            manifest.agentProtocols.executorOutput["jsonTypeSource"] = file.relpath

    @staticmethod
    def _infer_types(text: str) -> set[str]:
        lowered = text.lower()
        inferred: set[str] = set()
        for value_type in VALID_AGENT_JSON_TYPES:
            if re.search(rf"\bcase\s+{re.escape(value_type)}\b", lowered):
                inferred.add(value_type)
        if inferred:
            return inferred

        for value_type in VALID_AGENT_JSON_TYPES:
            if re.search(rf"\b{re.escape(value_type)}\b", lowered):
                inferred.add(value_type)
        if any(re.search(rf"\b{re.escape(marker)}\b", lowered) for marker in NUMERIC_MARKERS):
            inferred.add("number")
        return inferred.intersection(VALID_AGENT_JSON_TYPES)
