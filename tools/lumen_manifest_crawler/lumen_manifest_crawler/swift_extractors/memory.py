from __future__ import annotations

import re

from lumen_manifest_crawler.manifest import FreshnessClassManifest
from lumen_manifest_crawler.swift_extractors.base import (
    SwiftExtractor,
    SwiftFile,
    enum_cases,
    string_literals,
    strip_comments,
)

DURABLE_NAMES = {"durable", "permanent", "pinned"}
TTL_NAME_PATTERN = r"(?:ephemeral|session|durable|permanent|project)\w*(?:ttl|ttlseconds|lifetime|duration|expiration|expiry|seconds)\w*|(?:ttl|ttlseconds|lifetime|duration|expiration|expiry)\w*(?:ephemeral|session|durable|permanent|project)\w*"
RUNTIME_TTL_POLICY_PATTERN = re.compile(
    r"\breturn\s+TTLPolicy\s*\(\s*freshness\s*:\s*\.(?P<freshness>[A-Za-z_][A-Za-z0-9_]*)"
    r"\s*,\s*ttl\s*:\s*(?P<ttl>nil|\d+(?:\s*\*\s*\d+)*)\s*\)"
)


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

        existing_by_id = {f.id: f for f in manifest.memory.freshnessClasses}
        freshness_cases = set(enum_cases(file.text, "MemoryFreshnessClass")) | set(enum_cases(file.text, "FreshnessClass"))
        for name in sorted(freshness_cases):
            if name not in existing_by_id:
                freshness = FreshnessClassManifest(
                    id=name,
                    ttlSeconds=self._ttl_near(file.text, name),
                    durable=name.lower() in DURABLE_NAMES,
                    source=file.relpath,
                )
                manifest.memory.freshnessClasses.append(freshness)
                existing_by_id[name] = freshness

        for name, ttl in self._runtime_ttl_policies(file.text).items():
            if ttl is None or name.casefold() in DURABLE_NAMES | {"timeless"}:
                continue
            freshness = existing_by_id.get(name)
            if freshness is None:
                freshness = FreshnessClassManifest(id=name)
                manifest.memory.freshnessClasses.append(freshness)
                existing_by_id[name] = freshness
            freshness.ttlSeconds = ttl
            freshness.durable = False
            freshness.source = file.relpath

        for ttl_name, ttl in self._extract_ttl_constants(file.text):
            if ttl_name not in existing_by_id:
                freshness = FreshnessClassManifest(
                    id=ttl_name,
                    ttlSeconds=ttl,
                    durable=ttl_name.lower() in DURABLE_NAMES,
                    source=file.relpath,
                )
                manifest.memory.freshnessClasses.append(freshness)
                existing_by_id[ttl_name] = freshness

    @staticmethod
    def _runtime_ttl_policies(text: str) -> dict[str, int | None]:
        policies: dict[str, set[int | None]] = {}
        for match in RUNTIME_TTL_POLICY_PATTERN.finditer(strip_comments(text)):
            freshness = match.group("freshness")
            raw_ttl = match.group("ttl")
            ttl = None
            if raw_ttl != "nil":
                factors = [int(value.strip()) for value in raw_ttl.split("*")]
                ttl = 1
                for factor in factors:
                    ttl *= factor
            policies.setdefault(freshness, set()).add(ttl)

        resolved: dict[str, int | None] = {}
        for freshness, values in policies.items():
            numeric_values = {value for value in values if value is not None}
            if len(numeric_values) > 1:
                raise ValueError(
                    f"Memory freshness class {freshness!r} has conflicting runtime TTLs: "
                    f"{sorted(numeric_values)}"
                )
            resolved[freshness] = next(iter(numeric_values), None)
        return resolved

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
