#!/usr/bin/env python3
"""Static Release hardening guard for Lumen runtime/product surfaces."""

from __future__ import annotations

import pathlib
import re
import sys
from dataclasses import dataclass
from enum import Enum, auto

ROOT = pathlib.Path(__file__).resolve().parents[1]
SOURCE_ROOTS = [ROOT / "ios" / "Lumen"]
DOC_ROOTS = [
    ROOT / "README.md",
    ROOT / "CLAUDE.md",
    ROOT / "docs" / "RUNTIME_STATUS_MATRIX.md",
    ROOT / "docs" / "AGENT_KERNEL_MIGRATION_STATUS.md",
    ROOT / "docs" / "VALIDATION.md",
    ROOT / "docs" / "APP_INTENTS.md",
    ROOT / "docs" / "BACKGROUND_PROCESSING.md",
    ROOT / "docs" / "TOOL_SECURITY_MODEL.md",
]

ALGORITHMIC_PHILOSOPHY_CANONICAL_ROOT = pathlib.Path(
    "generated/algorithmic_philosophies"
)
ALGORITHMIC_PHILOSOPHY_APP_ROOT = pathlib.Path(
    "ios/Lumen/Resources/AlgorithmicPhilosophies"
)
ALGORITHMIC_PHILOSOPHY_MIRROR_FILES = (
    pathlib.Path("latent_liturgy/latent_liturgy.html"),
    pathlib.Path("latent_liturgy/latent_liturgy.js"),
)
REMOVED_P5_RUNTIME_PATHS = (
    ALGORITHMIC_PHILOSOPHY_CANONICAL_ROOT / "p5.min.js",
    ALGORITHMIC_PHILOSOPHY_APP_ROOT / "p5.min.js",
)
P5_REFERENCE_PATTERN = re.compile(r"(?<![A-Za-z0-9_])p5(?=$|[^A-Za-z0-9_])", re.IGNORECASE)

FORBIDDEN_SOURCE_PATTERNS = {
    "removed generation sentinel": re.compile(r"\bgenerationNotImplemented\b"),
    "removed embedding sentinel": re.compile(r"\bembeddingExtractionNotImplemented\b"),
    "old generic fallback copy": re.compile(r"limited local mode", re.IGNORECASE),
    "staged runtime wording": re.compile(r"staged:\s*implementation missing|generation is staged|runtime staged", re.IGNORECASE),
    "mock backend registration": re.compile(r"register\s*\([^)]*mock|for:\s*\.mock", re.IGNORECASE),
    "removed legacy tool command API": re.compile(r"\bexecuteLegacyTool\s*\("),
    "background compatibility bridge surface": re.compile(r"\bBackgroundToolBridge(?:Policy|Assessment)\b|bridgeMappingUnavailable"),
    "release legacy bridge exclusion wording": re.compile(r"Legacy agent bridge is excluded from Release builds|compatibility shim"),
    "gguf release precondition crash": re.compile(r"\bprecondition(?:Failure)?\s*\([^\n]*(?:GGUF|Unavailable GGUF)"),
}

RELEASE_FORBIDDEN_SOURCE_PATTERNS = {
    "production TODO marker": re.compile(r"\b(?:TODO|FIXME|XXX)\b"),
    "production stub marker": re.compile(r"\b(?:stub|stubbed|stubbing)\b", re.IGNORECASE),
    "production not-implemented marker": re.compile(r"\b(?:not implemented|not-implemented|unimplemented)\b", re.IGNORECASE),
    "production precondition crash": re.compile(r"\bprecondition(?:Failure)?\s*\("),
}

FORBIDDEN_DOC_PATTERNS = {
    "shipped partial wording": re.compile(r"\bpartial\b", re.IGNORECASE),
    "shipped planned wording": re.compile(r"\bplanned\b", re.IGNORECASE),
    "shipped compatibility bridge wording": re.compile(r"compatibility bridge", re.IGNORECASE),
    "stale background bridge policy wording": re.compile(r"\bBackgroundToolBridge(?:Policy|Assessment)\b|bridgeMappingUnavailable"),
}

DEBUG_ONLY_PATTERNS = {
    "legacy compatibility bridge implementation": re.compile(r"\benum\s+LegacyAgentCompatibilityBridge\b"),
    "legacy bridge API surface": re.compile(r"\bfunc\s+runLegacyAgentBridge\s*\("),
    "legacy bridge call": re.compile(r"\.runLegacyAgentBridge\s*\("),
    "legacy compatibility bridge call": re.compile(r"\bLegacyAgentCompatibilityBridge\.(runLegacyAgentService|runSlotAgentKernelCompatibility|runSlotAgentCompatibility)\b"),
    "unavailable gguf bridge construction": re.compile(r"\bUnavailableGGUFNativeBridge\s*\("),
    "developer trace raw encoding": re.compile(r"\bencoder\.encode\s*\(\s*trace\s*\)"),
    "receipt-based debug authorization": re.compile(
        r"\bappStoreReceiptURL\b|\.lastPathComponent\s*==\s*\"sandboxReceipt\""
    ),
    "Microsoft Graph runtime client-ID override": re.compile(
        r"\bMicrosoftGraphRuntimeConfig\b|\bMSALClientIDOverride\b"
    ),
    "Microsoft Graph debug editor surface": re.compile(
        r"\b(?:microsoftClientID|debugClientIDEditor|debugConfigurationSection|defaultRedirectURI)\b"
        r"|\"(?:Debug configuration|Microsoft Entra client ID|Enter app client ID|Use this client ID|"
        r"Effective client ID:|Effective redirect URI:|Effective authority URL:)"
    ),
}

UNSAFE_DIAGNOSTIC_PATTERNS = {
    "public raw error diagnostic": re.compile(
        r"(?:String\s*\(\s*describing:\s*error\s*\)|error\.localizedDescription)\s*,\s*privacy:\s*\.public"
    ),
    "public raw sensitive diagnostic interpolation": re.compile(
        r"\\\(\s*(?:query|userPrompt|systemPrompt|developerPrompt|rawOutput|rawModelOutput|reasoningText|visibleAnswer|line|title|name|url\.lastPathComponent|path|loadedPath|adapterPath|assignment\.localPath|message\s*\?\?\s*\"\")\s*,\s*privacy:\s*\.public"
    ),
    "raw sensitive logger interpolation": re.compile(
        r"(?:\blogger|Logger\s*\([^)]*\))\.(?:debug|info|notice|warning|error|fault|log)\s*\([^\n]*\\\(\s*(?:query|userPrompt|systemPrompt|developerPrompt|rawOutput|rawModelOutput|line)\s*\)"
    ),
    "raw persistent diagnostic field": re.compile(
        r"\bPersistentDiagnosticEvent\s*\([^\n]*(?:message:\s*(?:userPrompt|systemPrompt|developerPrompt|rawOutput|rawModelOutput|visibleAnswer|query|content)|values:\s*[^\n]*(?:toolArgs|arguments|userPrompt|systemPrompt|developerPrompt|rawOutput|rawModelOutput))"
    ),
}

LOSSY_RAG_MEMORY_PATTERNS = {
    "lossy SwiftData fetch empty fallback": re.compile(
        r"try\?\s*[^;\n]*fetch\s*\(\s*FetchDescriptor\s*<\s*(?:RAGChunk|MemoryItem)\s*>\s*\([^)]*\)\s*\)\s*\)\s*\?\?\s*\[\]"
    ),
    "lossy empty JSON export fallback": re.compile(r'\breturn\s+"\[\]"'),
}

LOSSY_RUNTIME_MODEL_PATTERNS = {
    "lossy headless stored-model fetch empty fallback": re.compile(
        r"try\?\s*[^;\n]*fetch\s*\(\s*FetchDescriptor\s*<\s*StoredModel\s*>\s*\([^)]*\)\s*\)\s*\)\s*\?\?\s*\[\]"
    ),
    "lossy model bootstrap stored-model fetch empty fallback": re.compile(
        r"try\?\s*[^;\n]*fetch\s*\(\s*FetchDescriptor\s*<\s*StoredModel\s*>\s*\([^)]*\)\s*\)\s*\)\s*\?\?\s*\[\]"
    ),
    "lossy rem cycle stored-model fetch empty fallback": re.compile(
        r"try\?\s*[^;\n]*fetch\s*\(\s*FetchDescriptor\s*<\s*StoredModel\s*>\s*\([^)]*\)\s*\)\s*\)\s*\?\?\s*\[\]"
    ),
    "lossy settings e2e stored-model fetch empty fallback": re.compile(
        r"try\?\s*[^;\n]*fetch\s*\(\s*FetchDescriptor\s*<\s*StoredModel\s*>\s*\([^)]*\)\s*\)\s*\)\s*\?\?\s*\[\]"
    ),
}

LOSSY_MODEL_INTEGRITY_PATTERNS = {
    "lossy installed-model integrity filter": re.compile(
        r"\.filter\s*\{\s*ModelFileIntegrity\.validateInstalledFile\s*\(\s*\$0\s*\)\s*\}"
    ),
}

LOSSY_SETTINGS_MODEL_DIRECTORY_PATTERNS = {
    "lossy settings model directory fallback": re.compile(
        r"try\?\s*ModelStorage\.modelsDirectoryURLOrThrow\s*\("
    ),
    "lossy settings model files directory empty fallback": re.compile(
        r"try\?\s*[^;\n]*contentsOfDirectory\s*\("
    ),
}

LOSSY_SETTINGS_IMPORTED_FILE_PATTERNS = {
    "lossy settings imported files wrapper": re.compile(r"\bFileStore\.importedFiles\s*\("),
    "lossy settings imports directory fallback": re.compile(
        r"try\?\s*FileStore\.importsDirectoryOrThrow\s*\("
    ),
    "raw settings models path": re.compile(r"Models path:\s*\\\([^)]*\.path[^)]*\)"),
}

LOSSY_TRIGGER_PATTERNS = {
    "lossy trigger persist nil fallback": re.compile(r"catch\s*\{\s*return\s+nil\s*\}"),
    "lossy trigger scheduler fetch silent return": re.compile(
        r"guard\s+let\s+\w+\s*=\s*try\?\s*[^;\n]*fetch\s*\(\s*FetchDescriptor\s*<\s*Trigger\s*>\s*\([^)]*\)\s*\)\s*else\s*\{\s*return\s*\}"
    ),
    "lossy trigger tool ignored save": re.compile(r"try\?\s*ctx\.save\s*\(\s*\)"),
    "lossy trigger tool fetch empty fallback": re.compile(
        r"try\?\s*ctx\.fetch\s*\(\s*FetchDescriptor\s*<\s*Trigger\s*>\s*\([^)]*\)\s*\)\s*\)\s*\?\?\s*\[\]"
    ),
    "generic trigger intent no-result fallback": re.compile(r'"No result\."'),
}

LOSSY_RAG_MEMORY_CALLS = [
    ("lossy memory recall wrapper", re.compile(r"\bMemoryStore\.recall\s*\("), "ios/Lumen/Services/MemoryStore.swift"),
    ("lossy memory export wrapper", re.compile(r"\bMemoryStore\.exportJSON\s*\("), "ios/Lumen/Services/MemoryStore.swift"),
    ("lossy RAG counts wrapper", re.compile(r"\bRAGStore\.counts\s*\("), "ios/Lumen/Services/RAGStore.swift"),
    ("lossy RAG chunks wrapper", re.compile(r"\bRAGStore\.chunks\s*\("), "ios/Lumen/Services/RAGStore.swift"),
    ("lossy RAG imported-file index wrapper", re.compile(r"\bRAGStore\.indexImportedFiles\s*\("), "ios/Lumen/Services/RAGStore.swift"),
    ("lossy RAG photo index wrapper", re.compile(r"\bRAGStore\.indexPhotos\s*\("), "ios/Lumen/Services/RAGStore.swift"),
    ("lossy RAG note index wrapper", re.compile(r"\bRAGStore\.indexNote\s*\("), "ios/Lumen/Services/RAGStore.swift"),
    ("lossy RAG retrieve wrapper", re.compile(r"\bRAGEngine\s*\(\s*\)\.retrieve\s*\("), "ios/Lumen/RAG/RAGEngine.swift"),
    ("lossy memory engine search wrapper", re.compile(r"\bMemoryEngine\s*\(\s*\)\.search\s*\("), "ios/Lumen/Memory/MemoryEngine.swift"),
]

LOSSY_IMPORTED_FILE_PATTERNS = {
    "lossy imported files wrapper": re.compile(r"\bFileStore\.importedFiles\s*\("),
    "lossy imported files directory empty fallback": re.compile(
        r"try\?\s*[^;\n]*contentsOfDirectory\s*\([^;\n]*\)\s*\)\s*\?\?\s*\[\]"
    ),
}

LOSSY_IMPORTED_FILE_WRITE_PATTERNS = {
    "lossy imported-file write wrapper": re.compile(r"\bFileStore\.importFile\s*\("),
}

LOSSY_ATTACHMENT_METADATA_PATTERNS = {
    "lossy attachment size zero fallback": re.compile(r"\.intValue\s*\?\?\s*0"),
    "lossy attachment data read fallback": re.compile(r"try\?\s*Data\s*\(\s*contentsOf\s*:"),
    "lossy attachment attributed decode fallback": re.compile(r"try\?\s*NSAttributedString\s*\("),
    "lossy attachment PDF empty fallback": re.compile(r"PDFDocument\s*\(\s*url\s*:\s*[^)]*\)\s*else\s*\{\s*return\s+\"\""),
    "lossy raw attachment extraction wrapper": re.compile(r"\brawExtractText\s*\("),
}

UNSAFE_DEVELOPER_TRACE_PATTERNS = {
    "raw developer trace encoding": re.compile(r"encoder\.encode\s*\(\s*trace\s*\)"),
    "raw attachment name in trace context": re.compile(r"title\s*:\s*attachment\.name"),
    "raw attachment path in trace context": re.compile(r"source\s*:\s*attachment\.path"),
    "raw history content in trace context": re.compile(r"content\s*:\s*item\.content\s*,"),
}

LOSSY_FILE_TOOL_READ_PATTERNS = {
    "lossy file tool data read fallback": re.compile(r"try\?\s*Data\s*\(\s*contentsOf\s*:"),
    "generic file tool open failure": re.compile(r'return\s+"Couldn[\'’]?t open PDF\."'),
    "generic file tool read failure": re.compile(r'return\s+"Couldn[\'’]?t read'),
}

LOSSY_RAG_FILE_EXTRACTION_PATTERNS = {
    "lossy RAG file read fallback": re.compile(r"try\?\s*Data\s*\(\s*contentsOf\s*:"),
    "lossy RAG attributed decode fallback": re.compile(r"try\?\s*NSAttributedString\s*\("),
}

LOSSY_MEMORY_CAPTURE_QUEUE_PATTERNS = {
    "lossy memory capture pending count fallback": re.compile(
        r"try\?\s*(?:MemoryCaptureQueue\.)?pendingCount\s*\([^;\n]*\)\s*\)\s*\?\?\s*(?:0|1)"
    ),
}

LOSSY_MEMORY_REMEMBER_PATTERNS = {
    "lossy memory remember try fallback": re.compile(
        r"try\?\s+await\s+MemoryStore\.remember\s*\("
    ),
}

UNSAFE_MEMORY_TOOL_SAVE_PATTERNS = {
    "raw memory save content echo": re.compile(r'return\s+"Saved:\s*\\\(\s*trimmed\s*\)"'),
    "raw memory save localized error": re.compile(
        r'return\s+"Failed to save memory:\s*\\\(\s*error\.localizedDescription\s*\)"'
    ),
    "throwing memory tool save path": re.compile(r"try\s+await\s+MemoryStore\.remember\s*\("),
}

UNSAFE_CALENDAR_TOOL_PATTERNS = {
    "raw calendar tool localized error": re.compile(r"error\.localizedDescription"),
}

UNSAFE_ALARM_TOOL_PATTERNS = {
    "raw alarm tool localized error": re.compile(r"error\.localizedDescription"),
}

UNSAFE_HEALTH_TOOL_PATTERNS = {
    "raw health tool localized error": re.compile(r"error\.localizedDescription"),
}

UNSAFE_CONTACTS_TOOL_PATTERNS = {
    "raw contacts tool localized error": re.compile(r"error\.localizedDescription"),
}

LOSSY_DEVELOPER_DIAGNOSTIC_PATTERNS = {
    "lossy developer imported-files wrapper": re.compile(r"\bFileStore\.importedFiles\s*\("),
    "lossy developer model-files fallback": re.compile(
        r"try\?\s*[^;\n]*contentsOfDirectory\s*\([^;\n]*\)\s*\)\s*\?\?\s*\[\]"
    ),
    "raw developer models path": re.compile(r"Models path:\s*\\\([^)]*\.path\)"),
}

MODEL_CATALOG_RELEASE_FORBIDDEN_PATTERNS = {
    "release model catalog fallback wording": re.compile(r"\bfallback\b", re.IGNORECASE),
    "release model catalog mock wording": re.compile(r"\bmock\b", re.IGNORECASE),
    "release model catalog staged wording": re.compile(r"\bstaged\b", re.IGNORECASE),
    "release model catalog unavailable wording": re.compile(r"\bunavailable\b", re.IGNORECASE),
    "release model catalog not-implemented wording": re.compile(
        r"\b(?:not implemented|not-implemented|unimplemented)\b",
        re.IGNORECASE,
    ),
}

MODEL_CATALOG_RELEASE_FILES = {
    "ios/Lumen/Services/LLM/Models/BuiltInModelCatalog.swift",
    "ios/Lumen/Services/ModelFamilySelection.swift",
    "ios/Lumen/Services/ModelFleetCatalog.swift",
}

MODEL_CATALOG_CONTRACT_FILE = "ios/Lumen/Services/ModelAdapterRuntimeContract.swift"
MODEL_FAMILY_SELECTION_FILE = "ios/Lumen/Services/ModelFamilySelection.swift"
MUTABLE_MODEL_RESOLUTION_PATTERN = re.compile(r"\bresolve/main\b", re.IGNORECASE)
IMMUTABLE_REVISION_PATTERN = re.compile(r"^[0-9a-fA-F]{40}$")
SHA256_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")
SAFE_MODEL_PATH_COMPONENT_PATTERN = re.compile(r"^[A-Za-z0-9_.~-]+$")

DETERMINISTIC_COMPATIBILITY_EXECUTION_FILES = {
    "ios/Lumen/Services/AgentService.swift",
    "ios/Lumen/Services/SlotAgentService.swift",
}


@dataclass(frozen=True)
class _SwiftCallBlock:
    line_number: int
    text: str


def _swift_call_blocks(text: str, callee: str) -> list[_SwiftCallBlock]:
    """Return balanced Swift call expressions while ignoring strings/comments."""
    blocks: list[_SwiftCallBlock] = []
    pattern = re.compile(rf"\b{re.escape(callee)}\s*\(")
    for match in pattern.finditer(text):
        opening = text.find("(", match.start(), match.end())
        if opening < 0:
            continue
        depth = 0
        index = opening
        in_string = False
        escaped = False
        line_comment = False
        block_comment_depth = 0
        while index < len(text):
            character = text[index]
            following = text[index + 1] if index + 1 < len(text) else ""
            if line_comment:
                if character == "\n":
                    line_comment = False
                index += 1
                continue
            if block_comment_depth:
                if character == "/" and following == "*":
                    block_comment_depth += 1
                    index += 2
                    continue
                if character == "*" and following == "/":
                    block_comment_depth -= 1
                    index += 2
                    continue
                index += 1
                continue
            if in_string:
                if escaped:
                    escaped = False
                elif character == "\\":
                    escaped = True
                elif character == '"':
                    in_string = False
                index += 1
                continue
            if character == "/" and following == "/":
                line_comment = True
                index += 2
                continue
            if character == "/" and following == "*":
                block_comment_depth = 1
                index += 2
                continue
            if character == '"':
                in_string = True
                index += 1
                continue
            if character == "(":
                depth += 1
            elif character == ")":
                depth -= 1
                if depth == 0:
                    blocks.append(
                        _SwiftCallBlock(
                            line_number=text.count("\n", 0, match.start()) + 1,
                            text=text[match.start() : index + 1],
                        )
                    )
                    break
            index += 1
    return blocks


def _swift_literal_argument(block: str, name: str) -> str | None:
    match = re.search(
        rf'\b{re.escape(name)}\s*:\s*"((?:\\.|[^"\\])*)"',
        block,
    )
    return match.group(1) if match else None


def _swift_has_argument(block: str, name: str) -> bool:
    return re.search(rf"\b{re.escape(name)}\s*:", block) is not None


def _swift_nil_argument(block: str, name: str) -> bool:
    return re.search(rf"\b{re.escape(name)}\s*:\s*nil\b", block) is not None


def _is_safe_model_basename(value: str) -> bool:
    return (
        bool(value)
        and value == value.strip()
        and value not in {".", ".."}
        and "/" not in value
        and "\\" not in value
        and SAFE_MODEL_PATH_COMPONENT_PATTERN.fullmatch(value) is not None
    )


def _is_safe_model_source_path(value: str) -> bool:
    if not value or value != value.strip() or "\\" in value:
        return False
    components = value.split("/")
    return all(
        component not in {"", ".", ".."}
        and SAFE_MODEL_PATH_COMPONENT_PATTERN.fullmatch(component) is not None
        for component in components
    )


def _model_pin_violation(
    relative: str,
    line_number: int,
    field: str,
    value: str | None,
    pattern: re.Pattern[str],
) -> str | None:
    if value is not None and pattern.fullmatch(value):
        return None
    expected = "40-character commit hash" if pattern is IMMUTABLE_REVISION_PATTERN else "64-character SHA-256"
    rendered = value if value is not None else "missing or non-literal"
    return (
        f"{relative}:{line_number}: selectable model {field} must be an immutable {expected}; "
        f"found {rendered!r}"
    )


def _scan_model_catalog_contract(relative: str, text: str) -> list[str]:
    violations: list[str] = []
    destination_file_names: list[tuple[str, int]] = []

    if relative == MODEL_FAMILY_SELECTION_FILE:
        for block in _swift_call_blocks(text, "CatalogModel"):
            for field in ("sourceRevision", "expectedSHA256"):
                if not _swift_has_argument(block.text, field):
                    violations.append(
                        f"{relative}:{block.line_number}: selectable CatalogModel must provide {field}"
                    )
        return violations

    if relative != MODEL_CATALOG_CONTRACT_FILE:
        return violations

    for block in _swift_call_blocks(text, "LumenTrainedModelRuntimeContract"):
        shared_file_name = _swift_literal_argument(block.text, "sharedBaseFileName")
        if shared_file_name is None or not _is_safe_model_basename(shared_file_name):
            violations.append(
                f"{relative}:{block.line_number}: selectable shared-base fileName must be one safe basename"
            )
        else:
            destination_file_names.append((shared_file_name, block.line_number))

        for field, pattern in (
            ("sharedBaseSourceRevision", IMMUTABLE_REVISION_PATTERN),
            ("sharedBaseExpectedSHA256", SHA256_PATTERN),
        ):
            violation = _model_pin_violation(
                relative,
                block.line_number,
                field,
                _swift_literal_argument(block.text, field),
                pattern,
            )
            if violation:
                violations.append(violation)

        if not _swift_has_argument(block.text, "embeddingRepoID"):
            violations.append(
                f"{relative}:{block.line_number}: selectable runtime contract must declare embeddingRepoID explicitly"
            )
        elif not _swift_nil_argument(block.text, "embeddingRepoID"):
            embedding_file_name = _swift_literal_argument(block.text, "embeddingFileName")
            if embedding_file_name is None or not _is_safe_model_basename(embedding_file_name):
                violations.append(
                    f"{relative}:{block.line_number}: selectable embedding fileName must be one safe basename"
                )
            else:
                destination_file_names.append((embedding_file_name, block.line_number))
            for field, pattern in (
                ("embeddingSourceRevision", IMMUTABLE_REVISION_PATTERN),
                ("embeddingExpectedSHA256", SHA256_PATTERN),
            ):
                violation = _model_pin_violation(
                    relative,
                    block.line_number,
                    field,
                    _swift_literal_argument(block.text, field),
                    pattern,
                )
                if violation:
                    violations.append(violation)

    for block in _swift_call_blocks(text, "LumenAdapterRoleContract"):
        adapter_file_name = _swift_literal_argument(block.text, "adapterFileName")
        if adapter_file_name is None or not _is_safe_model_basename(adapter_file_name):
            violations.append(
                f"{relative}:{block.line_number}: selectable adapter fileName must be one safe basename"
            )
        else:
            destination_file_names.append((adapter_file_name, block.line_number))

        for field, pattern in (
            ("adapterSourceRevision", IMMUTABLE_REVISION_PATTERN),
            ("adapterExpectedSHA256", SHA256_PATTERN),
        ):
            violation = _model_pin_violation(
                relative,
                block.line_number,
                field,
                _swift_literal_argument(block.text, field),
                pattern,
            )
            if violation:
                violations.append(violation)

        literal_source_path = _swift_literal_argument(block.text, "adapterSourcePath")
        if literal_source_path is not None:
            if not _is_safe_model_source_path(literal_source_path):
                violations.append(
                    f"{relative}:{block.line_number}: selectable adapter sourcePath contains an unsafe component"
                )
        else:
            helper_match = re.search(
                r'\badapterSourcePath\s*:\s*qwen3AdapterSourcePath\s*\(\s*"([^"\\]+)"\s*\)',
                block.text,
            )
            if (
                helper_match is None
                or adapter_file_name is None
                or helper_match.group(1) != adapter_file_name
            ):
                violations.append(
                    f"{relative}:{block.line_number}: selectable adapter sourcePath must be a safe literal or use the validated destination fileName"
                )

    seen_file_names: dict[str, tuple[str, int]] = {}
    for file_name, line_number in destination_file_names:
        key = file_name.casefold()
        if key in seen_file_names:
            prior_name, prior_line = seen_file_names[key]
            violations.append(
                f"{relative}:{line_number}: selectable destination fileName {file_name!r} duplicates "
                f"{prior_name!r} from line {prior_line}"
            )
        else:
            seen_file_names[key] = (file_name, line_number)

    return violations


def rel(path: pathlib.Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def iter_files(roots: list[pathlib.Path], suffixes: tuple[str, ...]) -> list[pathlib.Path]:
    files: list[pathlib.Path] = []
    for root in roots:
        if root.is_file() and root.suffix in suffixes:
            files.append(root)
        elif root.exists():
            files.extend(path for path in root.rglob("*") if path.is_file() and path.suffix in suffixes)
    return sorted(files)


class _ReleaseTruth(Enum):
    FALSE = auto()
    TRUE = auto()
    UNKNOWN = auto()


def _release_not(value: _ReleaseTruth) -> _ReleaseTruth:
    if value is _ReleaseTruth.FALSE:
        return _ReleaseTruth.TRUE
    if value is _ReleaseTruth.TRUE:
        return _ReleaseTruth.FALSE
    return _ReleaseTruth.UNKNOWN


def _release_and(left: _ReleaseTruth, right: _ReleaseTruth) -> _ReleaseTruth:
    if _ReleaseTruth.FALSE in {left, right}:
        return _ReleaseTruth.FALSE
    if left is _ReleaseTruth.TRUE and right is _ReleaseTruth.TRUE:
        return _ReleaseTruth.TRUE
    return _ReleaseTruth.UNKNOWN


def _release_or(left: _ReleaseTruth, right: _ReleaseTruth) -> _ReleaseTruth:
    if _ReleaseTruth.TRUE in {left, right}:
        return _ReleaseTruth.TRUE
    if left is _ReleaseTruth.FALSE and right is _ReleaseTruth.FALSE:
        return _ReleaseTruth.FALSE
    return _ReleaseTruth.UNKNOWN


class _ReleaseConditionParser:
    """Evaluate a Swift compilation condition with DEBUG fixed to false.

    Platform checks and custom compilation flags remain unknown. The caller only
    treats a branch as debug-only when this evaluator can prove that its
    condition is false in Release; parse failures therefore fail open to
    UNKNOWN/release-reachable rather than hiding source from the guard.
    """

    TOKEN_PATTERN = re.compile(r"&&|\|\||==|!=|!|\(|\)|[A-Za-z_][A-Za-z0-9_]*|\S")

    def __init__(self, expression: str) -> None:
        self.tokens = self.TOKEN_PATTERN.findall(expression)
        self.index = 0
        self.valid = True

    def parse(self) -> _ReleaseTruth:
        if not self.tokens:
            return _ReleaseTruth.UNKNOWN
        value = self._parse_or()
        if not self.valid or self.index != len(self.tokens):
            return _ReleaseTruth.UNKNOWN
        return value

    def _peek(self) -> str | None:
        if self.index >= len(self.tokens):
            return None
        return self.tokens[self.index]

    def _accept(self, token: str) -> bool:
        if self._peek() != token:
            return False
        self.index += 1
        return True

    def _parse_or(self) -> _ReleaseTruth:
        value = self._parse_and()
        while self._accept("||"):
            value = _release_or(value, self._parse_and())
        return value

    def _parse_and(self) -> _ReleaseTruth:
        value = self._parse_equality()
        while self._accept("&&"):
            value = _release_and(value, self._parse_equality())
        return value

    def _parse_equality(self) -> _ReleaseTruth:
        value = self._parse_unary()
        while self._peek() in {"==", "!="}:
            operator = self._peek()
            self.index += 1
            right = self._parse_unary()
            if _ReleaseTruth.UNKNOWN in {value, right}:
                value = _ReleaseTruth.UNKNOWN
            else:
                equal = value is right
                value = _ReleaseTruth.TRUE if equal else _ReleaseTruth.FALSE
                if operator == "!=":
                    value = _release_not(value)
        return value

    def _parse_unary(self) -> _ReleaseTruth:
        if self._accept("!"):
            return _release_not(self._parse_unary())
        return self._parse_primary()

    def _parse_primary(self) -> _ReleaseTruth:
        if self._accept("("):
            value = self._parse_or()
            if not self._accept(")"):
                self.valid = False
            return value

        token = self._peek()
        if token is None:
            self.valid = False
            return _ReleaseTruth.UNKNOWN
        self.index += 1

        if token == "DEBUG":
            return _ReleaseTruth.FALSE
        if token == "true":
            return _ReleaseTruth.TRUE
        if token == "false":
            return _ReleaseTruth.FALSE
        if token == "defined":
            return self._parse_defined()

        if self._accept("("):
            self._consume_call_arguments()
        return _ReleaseTruth.UNKNOWN

    def _parse_defined(self) -> _ReleaseTruth:
        parenthesized = self._accept("(")
        flag = self._peek()
        if flag is None:
            self.valid = False
            return _ReleaseTruth.UNKNOWN
        self.index += 1
        if parenthesized and not self._accept(")"):
            self.valid = False
            return _ReleaseTruth.UNKNOWN
        return _ReleaseTruth.FALSE if flag == "DEBUG" else _ReleaseTruth.UNKNOWN

    def _consume_call_arguments(self) -> None:
        depth = 1
        while self.index < len(self.tokens) and depth > 0:
            token = self.tokens[self.index]
            self.index += 1
            if token == "(":
                depth += 1
            elif token == ")":
                depth -= 1
        if depth != 0:
            self.valid = False


def _release_condition_truth(directive: str) -> _ReleaseTruth:
    condition = re.sub(r"//.*$", "", directive, count=1)
    condition = re.sub(r"^#(?:if|elseif)\b", "", condition, count=1).strip()
    return _ReleaseConditionParser(condition).parse()


def _is_else_directive(stripped: str) -> bool:
    return re.match(r"^#else(?:\s|//|$)", stripped) is not None


@dataclass
class _ConditionalFrame:
    parent_release_reachable: bool
    prior_branch_definitely_true: bool
    current_release_reachable: bool
    saw_else: bool = False


def debug_stack_for_lines(lines: list[str]) -> list[bool]:
    stack: list[_ConditionalFrame] = []
    states: list[bool] = []
    for line in lines:
        stripped = line.strip()
        if re.match(r"^#if\b", stripped):
            truth = _release_condition_truth(stripped)
            parent_release_reachable = (
                stack[-1].current_release_reachable if stack else True
            )
            stack.append(
                _ConditionalFrame(
                    parent_release_reachable=parent_release_reachable,
                    prior_branch_definitely_true=truth is _ReleaseTruth.TRUE,
                    current_release_reachable=(
                        parent_release_reachable and truth is not _ReleaseTruth.FALSE
                    ),
                )
            )
        elif re.match(r"^#elseif\b", stripped) and stack:
            frame = stack[-1]
            truth = _release_condition_truth(stripped)
            frame.current_release_reachable = (
                frame.parent_release_reachable
                and not frame.prior_branch_definitely_true
                and truth is not _ReleaseTruth.FALSE
            )
            frame.prior_branch_definitely_true = (
                frame.prior_branch_definitely_true or truth is _ReleaseTruth.TRUE
            )
        elif _is_else_directive(stripped) and stack:
            frame = stack[-1]
            frame.current_release_reachable = (
                frame.parent_release_reachable
                and not frame.prior_branch_definitely_true
                and not frame.saw_else
            )
            frame.prior_branch_definitely_true = True
            frame.saw_else = True
        elif stripped.startswith("#endif") and stack:
            stack.pop()
        states.append(bool(stack) and not stack[-1].current_release_reachable)
    return states


def _is_non_source_line(stripped: str) -> bool:
    return (
        not stripped
        or stripped.startswith("//")
        or stripped.startswith("/*")
        or stripped.startswith("*")
        or stripped.startswith("#")
    )


def scan_source() -> list[str]:
    violations: list[str] = []
    for path in iter_files(SOURCE_ROOTS, (".swift", ".h", ".m", ".mm")):
        text = path.read_text(encoding="utf-8")
        lines = text.splitlines()
        debug_states = debug_stack_for_lines(lines)
        relative = rel(path)
        for line_number, line in enumerate(lines, start=1):
            debug_only = debug_states[line_number - 1]
            stripped = line.strip()
            if (
                relative == "ios/Lumen/Assistant/LegacyAgentCompatibilityBridge.swift"
                and not debug_only
                and not _is_non_source_line(stripped)
            ):
                violations.append(
                    f"{relative}:{line_number}: legacy compatibility bridge file must be fully inside #if DEBUG: {stripped}"
                )
            for label, pattern in FORBIDDEN_SOURCE_PATTERNS.items():
                if pattern.search(line):
                    violations.append(f"{relative}:{line_number}: {label}: {line.strip()}")
            for label, pattern in DEBUG_ONLY_PATTERNS.items():
                if pattern.search(line) and not debug_only:
                    violations.append(f"{relative}:{line_number}: {label} must be inside #if DEBUG: {line.strip()}")
            if not debug_only:
                if MUTABLE_MODEL_RESOLUTION_PATTERN.search(line):
                    violations.append(
                        f"{relative}:{line_number}: mutable model resolve/main reference is forbidden in Release: {line.strip()}"
                    )
                for label, pattern in RELEASE_FORBIDDEN_SOURCE_PATTERNS.items():
                    if pattern.search(line):
                        violations.append(f"{relative}:{line_number}: {label}: {line.strip()}")
                for label, pattern in UNSAFE_DIAGNOSTIC_PATTERNS.items():
                    if pattern.search(line):
                        violations.append(f"{relative}:{line_number}: {label}: {line.strip()}")
            for label, pattern in LOSSY_RAG_MEMORY_PATTERNS.items():
                if pattern.search(line):
                    violations.append(f"{relative}:{line_number}: {label}: {line.strip()}")
            if relative == "ios/Lumen/Assistant/HeadlessAgentKernelRunner.swift":
                label = "lossy headless stored-model fetch empty fallback"
                if LOSSY_RUNTIME_MODEL_PATTERNS[label].search(line):
                    violations.append(
                        f"{relative}:{line_number}: {label}; surface model catalog fetch failure instead of empty fleet: {line.strip()}"
                    )
            if relative == "ios/Lumen/Services/ModelLaunchBootstrap.swift":
                label = "lossy model bootstrap stored-model fetch empty fallback"
                if LOSSY_RUNTIME_MODEL_PATTERNS[label].search(line):
                    violations.append(
                        f"{relative}:{line_number}: {label}; surface model catalog fetch failure instead of missing/zero artifacts: {line.strip()}"
                    )
            if relative == "ios/Lumen/Services/RemCycleService.swift":
                label = "lossy rem cycle stored-model fetch empty fallback"
                if LOSSY_RUNTIME_MODEL_PATTERNS[label].search(line):
                    violations.append(
                        f"{relative}:{line_number}: {label}; surface model catalog fetch failure instead of empty REM fleet: {line.strip()}"
                    )
            if relative == "ios/Lumen/Views/SettingsView.swift":
                label = "lossy settings e2e stored-model fetch empty fallback"
                if LOSSY_RUNTIME_MODEL_PATTERNS[label].search(line):
                    violations.append(
                        f"{relative}:{line_number}: {label}; surface live E2E model catalog fetch failure instead of no-model run: {line.strip()}"
                    )
                for label, pattern in LOSSY_SETTINGS_MODEL_DIRECTORY_PATTERNS.items():
                    if pattern.search(line):
                        violations.append(
                            f"{relative}:{line_number}: {label}; use ModelStorage.modelFilesWithDiagnostics so model storage failures are not empty or unavailable success: {line.strip()}"
                        )
                for label, pattern in LOSSY_SETTINGS_IMPORTED_FILE_PATTERNS.items():
                    if pattern.search(line):
                        guidance = (
                            "hash local path diagnostics instead of exposing raw storage paths"
                            if label == "raw settings models path"
                            else "use FileStore.importedFilesWithDiagnostics so import storage failures are not empty or unavailable success"
                        )
                        violations.append(f"{relative}:{line_number}: {label}; {guidance}: {line.strip()}")
            if relative == "ios/Lumen/Services/SlotModelRuntimeCoordinator.swift":
                for label, pattern in LOSSY_MODEL_INTEGRITY_PATTERNS.items():
                    if pattern.search(line):
                        violations.append(
                            f"{relative}:{line_number}: {label}; log sanitized model artifact integrity diagnostics instead of dropping invalid candidates: {line.strip()}"
                        )
            if relative == "ios/Lumen/Services/TriggerScheduler.swift":
                for label, pattern in LOSSY_TRIGGER_PATTERNS.items():
                    if label == "lossy trigger persist nil fallback" and pattern.search(line):
                        violations.append(
                            f"{relative}:{line_number}: {label}; surface trigger persistence failure instead of nil/no result: {line.strip()}"
                        )
                    if label == "lossy trigger scheduler fetch silent return" and pattern.search(line):
                        violations.append(
                            f"{relative}:{line_number}: {label}; surface trigger fetch failure instead of silent scheduling no-op: {line.strip()}"
                        )
            if relative == "ios/Lumen/Services/Tools/TriggerTools.swift":
                for label in ("lossy trigger tool ignored save", "lossy trigger tool fetch empty fallback"):
                    pattern = LOSSY_TRIGGER_PATTERNS[label]
                    if pattern.search(line):
                        violations.append(
                            f"{relative}:{line_number}: {label}; surface trigger tool persistence/fetch failure instead of success or empty list: {line.strip()}"
                        )
            if relative == "ios/Lumen/AppIntents/LumenRunTriggerIntent.swift":
                label = "generic trigger intent no-result fallback"
                if LOSSY_TRIGGER_PATTERNS[label].search(line):
                    violations.append(
                        f"{relative}:{line_number}: {label}; render a degraded diagnostic instead of generic trigger success text: {line.strip()}"
                    )
            for label, pattern, owner in LOSSY_RAG_MEMORY_CALLS:
                if relative != owner and pattern.search(line):
                    violations.append(
                        f"{relative}:{line_number}: {label}; use the diagnostic API in product code: {line.strip()}"
                    )
            if relative in {
                "ios/Lumen/Services/RAGStore.swift",
                "ios/Lumen/Services/Tools/FilesTools.swift",
            }:
                for label, pattern in LOSSY_IMPORTED_FILE_PATTERNS.items():
                    if pattern.search(line):
                        violations.append(
                            f"{relative}:{line_number}: {label}; use importedFilesWithDiagnostics so import storage failures are not empty results: {line.strip()}"
                        )
            if relative in {
                "ios/Lumen/Views/ChatView.swift",
                "ios/Lumen/Views/SourcesView.swift",
            }:
                for label, pattern in LOSSY_IMPORTED_FILE_WRITE_PATTERNS.items():
                    if pattern.search(line):
                        violations.append(
                            f"{relative}:{line_number}: {label}; use importFileWithDiagnostics so import copy failures are not silent no-ops: {line.strip()}"
                        )
            if relative == "ios/Lumen/Models/ChatAttachment.swift":
                for label, pattern in LOSSY_ATTACHMENT_METADATA_PATTERNS.items():
                    if pattern.search(line):
                        violations.append(
                            f"{relative}:{line_number}: {label}; surface unreadable attachment metadata/extraction as diagnostics instead of reporting fake zero-byte or empty content: {line.strip()}"
                        )
            if relative == "ios/Lumen/Services/PromptBudget.swift":
                label = "lossy raw attachment extraction wrapper"
                if LOSSY_ATTACHMENT_METADATA_PATTERNS[label].search(line):
                    violations.append(
                        f"{relative}:{line_number}: {label}; use AttachmentResolver.extractTextWithDiagnostics so prompt assembly can distinguish empty attachments from extraction failures: {line.strip()}"
                    )
            if relative == "ios/Lumen/Services/LLM/DeveloperTrace.swift":
                label = "raw developer trace encoding"
                if UNSAFE_DEVELOPER_TRACE_PATTERNS[label].search(line):
                    violations.append(
                        f"{relative}:{line_number}: {label}; encode redactedForPersistence() so stored developer traces do not persist prompts, outputs, memory, tool args, or paths: {line.strip()}"
                    )
            if relative == "ios/Lumen/Views/ChatView.swift":
                for label in (
                    "raw attachment name in trace context",
                    "raw attachment path in trace context",
                    "raw history content in trace context",
                ):
                    pattern = UNSAFE_DEVELOPER_TRACE_PATTERNS[label]
                    if pattern.search(line):
                        violations.append(
                            f"{relative}:{line_number}: {label}; store hash/count trace context metadata instead of raw attachment/history text: {line.strip()}"
                        )
            if relative == "ios/Lumen/Services/Tools/FilesTools.swift":
                for label, pattern in LOSSY_FILE_TOOL_READ_PATTERNS.items():
                    if pattern.search(line):
                        violations.append(
                            f"{relative}:{line_number}: {label}; surface imported-file read/open/decode failures with sanitized diagnostics: {line.strip()}"
                        )
            if relative == "ios/Lumen/Services/RAGStore.swift":
                for label, pattern in LOSSY_RAG_FILE_EXTRACTION_PATTERNS.items():
                    if pattern.search(line):
                        violations.append(
                            f"{relative}:{line_number}: {label}; surface RAG file extraction failures as typed diagnostics instead of collapsing read/decode errors: {line.strip()}"
                        )
            if relative in {
                "ios/Lumen/Memory/MemoryCaptureQueue.swift",
                "ios/Lumen/AppIntents/LumenAddMemoryIntent.swift",
            }:
                for label, pattern in LOSSY_MEMORY_CAPTURE_QUEUE_PATTERNS.items():
                    if pattern.search(line):
                        violations.append(
                            f"{relative}:{line_number}: {label}; surface queue read failures instead of inventing a pending-count value: {line.strip()}"
                        )
            for label, pattern in LOSSY_MEMORY_REMEMBER_PATTERNS.items():
                if pattern.search(line):
                    violations.append(
                        f"{relative}:{line_number}: {label}; use rememberWithDiagnostics or explicit do/catch so memory persistence failures are not hidden: {line.strip()}"
                    )
            if relative == "ios/Lumen/Services/Tools/MemoryTools.swift":
                for label, pattern in UNSAFE_MEMORY_TOOL_SAVE_PATTERNS.items():
                    if pattern.search(line):
                        violations.append(
                            f"{relative}:{line_number}: {label}; memory.save must use rememberWithDiagnostics and must not echo raw content or localized errors: {line.strip()}"
                        )
            if relative == "ios/Lumen/Services/Tools/CalendarTools.swift":
                for label, pattern in UNSAFE_CALENDAR_TOOL_PATTERNS.items():
                    if pattern.search(line):
                        violations.append(
                            f"{relative}:{line_number}: {label}; calendar/reminder tools must surface sanitized provider diagnostics instead of raw localized errors: {line.strip()}"
                        )
            if relative == "ios/Lumen/Services/Tools/AlarmTools.swift":
                for label, pattern in UNSAFE_ALARM_TOOL_PATTERNS.items():
                    if pattern.search(line):
                        violations.append(
                            f"{relative}:{line_number}: {label}; alarm tools must surface sanitized provider diagnostics instead of raw localized errors: {line.strip()}"
                        )
            if relative == "ios/Lumen/Services/Tools/HealthTools.swift":
                for label, pattern in UNSAFE_HEALTH_TOOL_PATTERNS.items():
                    if pattern.search(line):
                        violations.append(
                            f"{relative}:{line_number}: {label}; health tools must surface sanitized provider diagnostics instead of raw localized errors: {line.strip()}"
                        )
            if relative == "ios/Lumen/Services/Tools/ContactsTools.swift":
                for label, pattern in UNSAFE_CONTACTS_TOOL_PATTERNS.items():
                    if pattern.search(line):
                        violations.append(
                            f"{relative}:{line_number}: {label}; contacts tools must surface sanitized provider diagnostics instead of raw localized errors: {line.strip()}"
                        )
            if relative == "ios/Lumen/Developer/DeveloperFramework.swift":
                for label, pattern in LOSSY_DEVELOPER_DIAGNOSTIC_PATTERNS.items():
                    if pattern.search(line):
                        violations.append(
                            f"{relative}:{line_number}: {label}; developer diagnostics must use diagnostic APIs and hashed path metadata: {line.strip()}"
                        )
            if relative in MODEL_CATALOG_RELEASE_FILES and not debug_only:
                for label, pattern in MODEL_CATALOG_RELEASE_FORBIDDEN_PATTERNS.items():
                    if pattern.search(line):
                        violations.append(
                            f"{relative}:{line_number}: {label}; Release model catalog entries must not advertise fallback/mock/staged/unavailable model surfaces: {line.strip()}"
                        )
            if (
                relative in DETERMINISTIC_COMPATIBILITY_EXECUTION_FILES
                and not debug_only
                and re.search(r"\boptions\.allowDeterministicCompatibility\b", line)
            ):
                violations.append(
                    f"{relative}:{line_number}: raw deterministic compatibility flag in Release-compiled execution path; use allowsDeterministicCompatibilityExecution: {line.strip()}"
                )
        violations.extend(_scan_model_catalog_contract(relative, text))
    return violations


def scan_docs() -> list[str]:
    violations: list[str] = []
    for path in iter_files(DOC_ROOTS, (".md",)):
        relative = rel(path)
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            for label, pattern in FORBIDDEN_DOC_PATTERNS.items():
                if pattern.search(line):
                    violations.append(f"{relative}:{line_number}: {label}: {line.strip()}")
    return violations


def scan_redistribution_resources() -> list[str]:
    """Fail closed when the removed LGPL runtime or mirror drift returns."""

    violations: list[str] = []

    for relative_path in REMOVED_P5_RUNTIME_PATHS:
        path = ROOT / relative_path
        if path.exists():
            violations.append(
                f"{relative_path.as_posix()}: removed p5 runtime must not be distributed"
            )

    for relative_file in ALGORITHMIC_PHILOSOPHY_MIRROR_FILES:
        canonical_relative = ALGORITHMIC_PHILOSOPHY_CANONICAL_ROOT / relative_file
        app_relative = ALGORITHMIC_PHILOSOPHY_APP_ROOT / relative_file
        canonical_path = ROOT / canonical_relative
        app_path = ROOT / app_relative

        missing = False
        for relative_path, path in (
            (canonical_relative, canonical_path),
            (app_relative, app_path),
        ):
            if not path.is_file():
                violations.append(
                    f"{relative_path.as_posix()}: required algorithmic philosophy resource is missing"
                )
                missing = True
        if missing:
            continue

        canonical_bytes = canonical_path.read_bytes()
        app_bytes = app_path.read_bytes()
        if canonical_bytes != app_bytes:
            violations.append(
                f"{app_relative.as_posix()}: algorithmic philosophy app resource differs from "
                f"canonical {canonical_relative.as_posix()}"
            )

        for relative_path, payload in (
            (canonical_relative, canonical_bytes),
            (app_relative, app_bytes),
        ):
            try:
                text = payload.decode("utf-8")
            except UnicodeDecodeError:
                violations.append(
                    f"{relative_path.as_posix()}: algorithmic philosophy resource must be UTF-8 text"
                )
                continue
            if P5_REFERENCE_PATTERN.search(text):
                violations.append(
                    f"{relative_path.as_posix()}: removed p5 runtime reference must not be distributed"
                )

    return violations


def main() -> int:
    violations = scan_source() + scan_docs() + scan_redistribution_resources()
    if violations:
        print("Release hardening violations detected:", file=sys.stderr)
        for violation in violations:
            print(f"  {violation}", file=sys.stderr)
        return 1
    print("Release hardening guard passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
