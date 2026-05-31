from __future__ import annotations

import re

from lumen_manifest_crawler.manifest import FreshnessClassManifest
from lumen_manifest_crawler.swift_extractors.base import SwiftExtractor, SwiftFile, enum_cases, string_literals

DURABLE_NAMES = {"durable", "permanent", "pinned"}
TTL_NAME_PATTERN = r"(?:ephemeral|session|durable|permanent|project)\w*(?:ttl|ttlseconds|lifetime|duration|expiration|expiry|seconds)\w*|(?:ttl|ttlseconds|lifetime|duration|expiration|expiry)\w*(?:ephemeral|session|durable|permanent|project)\w*"


class MemoryExtractor(SwiftExtractor):
    target_names = ("MemoryItem.swift", "MemoryStore.swift", "MemoryContextItem.swift")

    def extract(self, file: SwiftFile, manifest) -> None:
        scopes = set(manifest.memory.scopes)
        for enum_name in ("Scope", "MemoryScope", "MemoryContextScope"):
            scopes.update(enum_cases(file.text, enum_name))
        for literal in string_literals(file.text):
            if literal in {"currentTurn", "session", "userPreference", "project", "durableFact", "person", "conversation"}:
                scopes.add(literal)
        manifest.memory.scopes = sorted(scopes)

        existing = {f.id for f in manifest.memory.freshnessClasses}
        freshness_cases = set(enum_cases(file.text, "MemoryFreshnessClass")) | set(enum_cases(file.text, "FreshnessClass"))
        for name in sorted(freshness_cases):
            if name not in existing:
                manifest.memory.freshnessClasses.append(
                    FreshnessClassManifest(
                        id=name,
                        ttlSeconds=self._ttl_near(file.text, name),
                        durable=name.lower() in DURABLE_NAMES,
                        source=file.relpath,
                    )
                )
                existing.add(name)

        for ttl_name, ttl in self._extract_ttl_constants(file.text):
            if ttl_name not in existing:
                manifest.memory.freshnessClasses.append(
                    FreshnessClassManifest(id=ttl_name, ttlSeconds=ttl, durable=ttl_name.lower() in DURABLE_NAMES, source=file.relpath)
                )
                existing.add(ttl_name)

    @staticmethod
    def _ttl_near(text: str, name: str) -> int | None:
        pattern = "\\b" + re.escape(name) + "\\b"
        match = re.search(pattern, text)
        if not match:
            return None
        window = text[match.start(): min(len(text), match.end() + 500)]
        num = re.search(r"(\d+)\s*(?:seconds|second|minutes|minute|hours|hour|days|day)", window, flags=re.I)
        if not num:
            return None
        value = int(num.group(1))
        unit = re.search(r"\d+\s*([A-Za-z]+)", num.group(0))
        factor = 1
        if unit:
            u = unit.group(1).lower()
            if u.startswith("minute"):
                factor = 60
            elif u.startswith("hour"):
                factor = 3600
            elif u.startswith("day"):
                factor = 86400
        return value * factor

    @staticmethod
    def _extract_ttl_constants(text: str) -> list[tuple[str, int | None]]:
        out: list[tuple[str, int | None]] = []
        declaration = re.compile(
            rf"\b(?:static\s+)?(?:let|var)\s+(?P<name>{TTL_NAME_PATTERN})\b(?:\s*:[^=\n]+)?\s*=\s*(?P<value>\d+)?",
            flags=re.I,
        )
        for match in declaration.finditer(text):
            raw_name = match.group("name").casefold()
            raw_value = match.group("value")
            freshness = MemoryExtractor._freshness_name_from_ttl_identifier(raw_name)
            if freshness:
                out.append((freshness, int(raw_value) if raw_value else None))
        return out

    @staticmethod
    def _freshness_name_from_ttl_identifier(identifier: str) -> str | None:
        for name in ("ephemeral", "session", "durable", "permanent", "project"):
            if re.search(rf"(?:^|_){name}(?:_|$)|{name}", identifier, flags=re.I):
                return name.casefold()
        return None
