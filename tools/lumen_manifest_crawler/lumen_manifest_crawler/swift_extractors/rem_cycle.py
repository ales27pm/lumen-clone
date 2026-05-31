from __future__ import annotations

from lumen_manifest_crawler.swift_extractors.base import SwiftExtractor, SwiftFile, balanced_call_blocks, string_literals


class RemCycleExtractor(SwiftExtractor):
    target_names = ("RemCycleService.swift", "REMService.swift")

    def extract(self, file: SwiftFile, manifest) -> None:
        report_fields: set[str] = set()
        for struct_name in ("RemCycleReport", "REMReport", "TrainingRecord"):
            for block in balanced_call_blocks(file.text, struct_name):
                report_fields.update(string_literals(block))
        literals = [s for s in string_literals(file.text) if "training" in s.lower() or "reflection" in s.lower() or "memory" in s.lower()]
        if report_fields or literals:
            existing = manifest.agentProtocols.executorOutput.get("remCycle") or {}
            existing_fields = set(existing.get("reportFields", []))
            existing_hints = list(existing.get("hints", []))
            existing_sources = existing.get("source", [])
            if isinstance(existing_sources, str):
                existing_sources = [existing_sources]
            merged_hints = list(dict.fromkeys([*existing_hints, *literals]))[:100]
            merged_sources = list(dict.fromkeys([*existing_sources, file.relpath]))
            manifest.agentProtocols.executorOutput["remCycle"] = {
                "reportFields": sorted(existing_fields.union(report_fields)),
                "hints": merged_hints,
                "source": merged_sources,
            }
