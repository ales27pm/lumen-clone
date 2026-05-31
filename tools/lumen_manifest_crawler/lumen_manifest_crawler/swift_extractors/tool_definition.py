from __future__ import annotations

import re

from lumen_manifest_crawler.manifest import ToolArgumentManifest, ToolManifest
from lumen_manifest_crawler.swift_extractors.base import (
    SwiftExtractor,
    SwiftFile,
    argument_value,
    balanced_call_blocks,
    bool_value,
    clean_swift_string,
    string_literals,
)


TOOL_ID_PATTERN = re.compile(r"[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)*")
TOOL_ID_LABEL_PATTERN = re.compile(r"\b(?:id|toolID|toolId|identifier)\s*:\s*\"([^\"]+)\"")
TOOL_ID_COLLECTION_PATTERN = re.compile(r"\b(?:toolIDs|toolIds|allowedToolIDs|allowedToolIds)\b[^\n=:\[]*[:=]\s*\[(?P<body>.*?)\]", flags=re.S)
ARG_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
OPTIONAL_PREFIX_PATTERN = re.compile(r"^optional\s+", flags=re.I)
ARG_TYPE_HINT_WORDS = {"uuid", "fallback"}


class ToolDefinitionExtractor(SwiftExtractor):
    target_names = ("ToolDefinition.swift",)

    def extract(self, file: SwiftFile, manifest) -> None:
        seen: set[str] = {tool.id for tool in manifest.tools}
        for block in balanced_call_blocks(file.text, "ToolDefinition"):
            tool_id = self._extract_tool_id(block)
            if not tool_id or tool_id in seen:
                continue
            seen.add(tool_id)
            description = clean_swift_string(argument_value(block, "description"))
            manifest.tools.append(
                ToolManifest(
                    id=tool_id,
                    displayName=clean_swift_string(argument_value(block, "displayName"))
                    or clean_swift_string(argument_value(block, "name"))
                    or tool_id,
                    description=description,
                    requiresApproval=bool_value(argument_value(block, "requiresApproval"), False),
                    permissionKey=clean_swift_string(argument_value(block, "permissionKey")),
                    arguments=self._extract_arguments(block, file.relpath, description),
                    source=file.relpath,
                )
            )

        # Literal fallback only accepts explicit tool-id labels/collections, then marks entries as inferred.
        for literal in self._fallback_tool_literals(file.text):
            if literal not in seen:
                seen.add(literal)
                manifest.tools.append(
                    ToolManifest(
                        id=literal,
                        displayName=literal,
                        requiresApproval=False,
                        arguments=[],
                        source=file.relpath,
                        inferred=True,
                        inferredSource="literal",
                    )
                )

    def _extract_tool_id(self, block: str) -> str | None:
        for label in ("id", "toolID", "toolId", "identifier"):
            value = clean_swift_string(argument_value(block, label))
            if value and self._looks_like_tool_id(value):
                return value
        labelled = [match.group(1) for match in TOOL_ID_LABEL_PATTERN.finditer(block)]
        for value in labelled:
            if self._looks_like_tool_id(value):
                return value
        return None

    def _extract_arguments(self, block: str, source: str, description: str | None) -> list[ToolArgumentManifest]:
        args: list[ToolArgumentManifest] = []
        seen: set[str] = set()
        for callee in ("ToolArgument", "ToolParameter", "ArgumentDefinition", "ParameterDefinition"):
            for arg_block in balanced_call_blocks(block, callee):
                name = clean_swift_string(argument_value(arg_block, "name")) or self._first_string(arg_block)
                if not name or name in seen:
                    continue
                seen.add(name)
                arg_type = (
                    clean_swift_string(argument_value(arg_block, "type"))
                    or clean_swift_string(argument_value(arg_block, "valueType"))
                    or self._infer_type_from_block(arg_block)
                )
                args.append(
                    ToolArgumentManifest(
                        name=name,
                        type=self._normalize_type(arg_type or "string"),
                        required=not bool_value(argument_value(arg_block, "optional"), False)
                        and bool_value(argument_value(arg_block, "required"), True),
                        description=clean_swift_string(argument_value(arg_block, "description")),
                        source=source,
                    )
                )
        for inferred in self._extract_args_from_description(description, source):
            if inferred.name not in seen:
                seen.add(inferred.name)
                args.append(inferred)
        return args

    @staticmethod
    def _extract_args_from_description(description: str | None, source: str) -> list[ToolArgumentManifest]:
        if not description:
            return []
        match = re.search(r"\bArgs?\s*:\s*(?P<body>.*?)(?:\.|$)", description, flags=re.I)
        if not match:
            return []
        body = match.group("body").strip()
        if not body or body.lower() in {"none", "no args", "n/a"}:
            return []

        normalized = body.replace(" or ", ", ").replace(" plus ", ", ").replace(" depending on ", ", ")
        specs: list[tuple[str, bool]] = []
        optional_group = False
        for raw in re.split(r"[,;/]", normalized):
            token = raw.strip()
            if not token:
                continue
            token = re.sub(r"\(.*?\)", "", token).strip()
            token_lower = token.lower()
            token_optional = optional_group or token_lower.startswith("optional ")
            token = OPTIONAL_PREFIX_PATTERN.sub("", token).strip()
            if not token:
                optional_group = True
                continue
            parts = token.split()
            name = parts[0].strip("`'\".:") if parts else token.strip("`'\".:")
            if name.lower() in {"none", "args", "arg"}:
                continue
            if name.lower() in ARG_TYPE_HINT_WORDS and len(parts) > 1:
                continue
            if not ARG_NAME_PATTERN.fullmatch(name):
                continue
            if not any(existing == name for existing, _ in specs):
                specs.append((name, token_optional))
            optional_group = token_optional

        return [
            ToolArgumentManifest(
                name=name,
                type=ToolDefinitionExtractor._infer_arg_type_from_name(name),
                required=not optional,
                description=f"Inferred from ToolDefinition description Args contract: {body}",
                source=source,
            )
            for name, optional in specs
        ]

    @staticmethod
    def _infer_arg_type_from_name(name: str) -> str:
        lowered = name.lower()
        if any(marker in lowered for marker in ("minutes", "seconds", "count", "limit", "months", "duration", "interval")):
            return "number"
        if lowered in {"repeats"} or lowered.startswith("is") or lowered.startswith("has") or lowered.startswith("should"):
            return "bool"
        if lowered in {"query", "title", "subject", "body", "message", "text", "recipient", "number", "destination", "location", "city", "url", "name", "kind", "content", "schedule", "timestamp", "attime", "id", "email", "to", "prompt"}:
            return "string"
        return "string"

    @staticmethod
    def _fallback_tool_literals(text: str) -> list[str]:
        values: list[str] = []
        for match in TOOL_ID_LABEL_PATTERN.finditer(text):
            value = match.group(1)
            if ToolDefinitionExtractor._looks_like_tool_id(value):
                values.append(value)
        for match in TOOL_ID_COLLECTION_PATTERN.finditer(text):
            for literal in string_literals(match.group("body")):
                if ToolDefinitionExtractor._looks_like_tool_id(literal):
                    values.append(literal)
        return list(dict.fromkeys(values))

    @staticmethod
    def _first_string(block: str) -> str | None:
        vals = string_literals(block)
        return vals[0] if vals else None

    @staticmethod
    def _infer_type_from_block(block: str) -> str:
        lowered = block.lower()
        if "double" in lowered or "float" in lowered or "number" in lowered:
            return "number"
        if "int" in lowered:
            return "number"
        if "bool" in lowered:
            return "bool"
        if "array" in lowered or "list" in lowered:
            return "array"
        if "object" in lowered or "dictionary" in lowered:
            return "object"
        return "string"

    @staticmethod
    def _normalize_type(raw: str) -> str:
        value = raw.strip().lower().replace(".", "")
        mapping = {
            "str": "string",
            "string": "string",
            "text": "string",
            "double": "number",
            "float": "number",
            "number": "number",
            "int": "number",
            "integer": "number",
            "bool": "bool",
            "boolean": "bool",
            "array": "array",
            "list": "array",
            "object": "object",
            "dictionary": "object",
            "dict": "object",
            "nil": "null",
            "none": "null",
            "null": "null",
        }
        return mapping.get(value, value or "string")

    @staticmethod
    def _looks_like_tool_id(value: str) -> bool:
        return bool(TOOL_ID_PATTERN.fullmatch(value))
