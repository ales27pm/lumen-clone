from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

from lumen_manifest_crawler.output.hashing import normalized_repo_path, sha256_file

CODEBASE_HOME_SCHEMA_VERSION = "1.1.0"

INCLUDED_SUFFIXES = {
    ".csv",
    ".entitlements",
    ".json",
    ".md",
    ".pbxproj",
    ".plist",
    ".py",
    ".sh",
    ".swift",
    ".toml",
    ".txt",
    ".xcconfig",
    ".xcprivacy",
    ".xcscheme",
    ".yaml",
    ".yml",
}

INCLUDED_FILENAMES = {
    ".gitignore",
    ".python-version",
    "Package.resolved",
}

IGNORED_DIRS = {
    ".git",
    ".build",
    ".cache",
    ".codex",
    ".local",
    ".pytest_cache",
    ".swiftpm",
    ".venv",
    "DerivedData",
    "__pycache__",
    "build",
    "dist",
    "models",
    "node_modules",
}

EXCLUDED_PREFIXES = (
    "generated/agent_improvement_loop/",
    "generated/agent_manifest/cross_model_training/",
    "generated/agent_manifest/dataset/",
    "generated/agent_manifest/embedding/",
    "generated/agent_manifest/fine_tuning/",
    "generated/fine_tuning/",
    "generated/visual_improve_loop/",
)

SELECTED_GENERATED_FILES = {
    "generated/agent_manifest/AgentBehaviorManifest.json",
    "generated/agent_manifest/AgentBehaviorManifest.md",
    "generated/agent_manifest/AgentBehaviorManifest.pretty.json",
    "generated/agent_manifest/dataset_index.csv",
    "generated/agent_manifest/dataset_manifest.json",
    "generated/agent_manifest/fleet_system_prompts.json",
    "generated/agent_manifest/routing_matrix.csv",
    "generated/agent_manifest/runtime_grounding_bundle.json",
    "generated/agent_manifest/runtime_grounding_prompt.md",
    "generated/agent_manifest/tool_registry.csv",
}

MAX_FILE_BYTES = 1_250_000
MAX_RECORDS = 10_000
MAX_CHUNK_CHARS = 2_400
MAX_CHUNK_LINES = 80
MAX_SYMBOLS = 80
MAX_IMPORTS = 60
MAX_SNIPPET_CHARS = 1_600


def generate_codebase_home_records(root: Path) -> dict[str, list[dict[str, Any]]]:
    """Build deterministic codebase-home grounding records.

    These records teach the fleet where app logic lives: modules, workflows,
    public symbols, imports, and bounded source evidence. They intentionally
    skip generated/build/model/private-local directories and do not ingest
    arbitrary runtime data.
    """

    root = root.resolve()
    corpus: list[dict[str, Any]] = []
    sft: list[dict[str, Any]] = []
    chunks: list[dict[str, Any]] = []
    chunk_sft: list[dict[str, Any]] = []

    for path in _iter_codebase_files(root):
        text = _read_text_file(path)
        if text is None:
            continue
        record = _code_file_record(root, path, text)
        if record is None:
            continue
        source_chunks = _chunk_records(root, path, text, record)
        record["metadata"]["lineCount"] = len(text.splitlines())
        record["metadata"]["chunkCount"] = len(source_chunks)
        corpus.append(record)
        sft.append(_sft_record(record))
        chunks.extend(source_chunks)
        chunk_sft.extend(_chunk_sft_record(chunk, record) for chunk in source_chunks)
        if len(corpus) >= MAX_RECORDS:
            break

    overview = _overview_record(root, corpus, chunks)
    corpus.insert(0, overview)
    sft.insert(0, _sft_record(overview))

    return {
        "codebase_home_corpus": corpus,
        "codebase_home_sft": sft,
        "codebase_home_chunks": chunks,
        "codebase_home_chunk_sft": chunk_sft,
    }


def _iter_codebase_files(root: Path):
    tracked = _git_tracked_files(root)
    if tracked:
        for relpath in tracked:
            if not _include_relpath(relpath):
                continue
            path = (root / relpath).resolve()
            if path.is_file():
                yield path
        return

    for path in _walk_files(root):
        relpath = normalized_repo_path(root, path)
        if _include_relpath(relpath):
            yield path.resolve()


def _walk_files(base: Path):
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames[:] = sorted(name for name in dirnames if name not in IGNORED_DIRS and not name.endswith(".xcuserdata"))
        for filename in sorted(filenames):
            path = Path(dirpath) / filename
            if _include_path(path):
                yield path


def _git_tracked_files(root: Path) -> list[str]:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "ls-files", "-z"],
            check=True,
            capture_output=True,
            text=False,
        )
    except (OSError, subprocess.CalledProcessError):
        return []
    return sorted(
        item.decode("utf-8", errors="replace")
        for item in result.stdout.split(b"\0")
        if item
    )


def _include_path(path: Path) -> bool:
    return path.suffix in INCLUDED_SUFFIXES or path.name in INCLUDED_FILENAMES


def _include_relpath(relpath: str) -> bool:
    if relpath in SELECTED_GENERATED_FILES:
        return True
    parts = Path(relpath).parts
    if any(part in IGNORED_DIRS or part.endswith(".xcuserdata") for part in parts):
        return False
    if any(relpath.startswith(prefix) for prefix in EXCLUDED_PREFIXES):
        return False
    if parts and parts[0] == "generated":
        return False
    return _include_path(Path(relpath))


def _read_text_file(path: Path) -> str | None:
    try:
        if path.stat().st_size > MAX_FILE_BYTES:
            return None
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def _code_file_record(root: Path, path: Path, text: str) -> dict[str, Any] | None:
    relpath = normalized_repo_path(root, path)
    symbols = _extract_symbols(path, text)
    imports = _extract_imports(path, text)
    module = _module_name(relpath)
    responsibility = _responsibility(relpath, symbols, text)
    snippet = _evidence_snippet(text)
    file_hash = sha256_file(path)

    return {
        "id": _stable_id("codebase_home", relpath, file_hash),
        "schemaVersion": CODEBASE_HOME_SCHEMA_VERSION,
        "sourceFamily": "codebase_home_corpus",
        "taskType": "codebase_home_grounding",
        "path": relpath,
        "module": module,
        "language": _language(path),
        "sha256": file_hash,
        "symbols": symbols[:MAX_SYMBOLS],
        "imports": imports[:MAX_IMPORTS],
        "responsibility": responsibility,
        "evidenceSnippet": snippet,
        "metadata": {
            "agentRole": "fleet",
            "privacy": "static_repo_source_only",
            "byteCount": len(text.encode("utf-8", errors="replace")),
            "recordKind": "file_summary",
            "coverage": "git_tracked_text_file",
        },
    }


def _overview_record(root: Path, corpus: list[dict[str, Any]], chunks: list[dict[str, Any]]) -> dict[str, Any]:
    modules: dict[str, int] = {}
    languages: dict[str, int] = {}
    total_lines = 0
    for record in corpus:
        modules[str(record.get("module") or "unknown")] = modules.get(str(record.get("module") or "unknown"), 0) + 1
        languages[str(record.get("language") or "unknown")] = languages.get(str(record.get("language") or "unknown"), 0) + 1
        metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
        total_lines += int(metadata.get("lineCount") or 0)
    top_modules = sorted(modules.items(), key=lambda item: (-item[1], item[0]))[:30]
    text = "\n".join(
        [
            "Lumen codebase home overview.",
            f"Root: {root.name}",
            "Scanned deterministic git-tracked text source, tests, docs, scripts, configs, and selected generated manifest artifacts.",
            f"Files: {len(corpus)}",
            f"Source chunks: {len(chunks)}",
            f"Lines: {total_lines}",
            "Top modules:",
            *[f"- {name}: {count} files" for name, count in top_modules],
        ]
    )
    return {
        "id": _stable_id("codebase_home_overview", top_modules, languages),
        "schemaVersion": CODEBASE_HOME_SCHEMA_VERSION,
        "sourceFamily": "codebase_home_corpus",
        "taskType": "codebase_home_overview",
        "path": ".",
        "module": "repo",
        "language": "mixed",
        "sha256": _stable_id("overview_sha", top_modules, languages),
        "symbols": [],
        "imports": [],
        "responsibility": text,
        "evidenceSnippet": text,
        "metadata": {
            "agentRole": "fleet",
            "privacy": "static_repo_source_only",
            "fileCount": len(corpus),
            "chunkCount": len(chunks),
            "lineCount": total_lines,
            "languages": dict(sorted(languages.items())),
            "modules": dict(sorted(modules.items())),
            "coverage": "git_tracked_text_files_plus_selected_manifest_artifacts",
            "selectedGeneratedFiles": sorted(SELECTED_GENERATED_FILES),
            "excludedPrefixes": sorted(EXCLUDED_PREFIXES),
            "recordKind": "repo_overview",
        },
    }


def _chunk_records(root: Path, path: Path, text: str, file_record: dict[str, Any]) -> list[dict[str, Any]]:
    relpath = normalized_repo_path(root, path)
    source_hash = str(file_record.get("sha256") or sha256_file(path))
    chunks = list(_split_source_chunks(text))
    records: list[dict[str, Any]] = []
    for index, (line_start, line_end, chunk_text) in enumerate(chunks):
        chunk_hash = hashlib.sha256(chunk_text.encode("utf-8", errors="replace")).hexdigest()
        records.append(
            {
                "id": _stable_id("codebase_home_chunk", relpath, source_hash, index, line_start, line_end, chunk_hash),
                "schemaVersion": CODEBASE_HOME_SCHEMA_VERSION,
                "sourceFamily": "codebase_home_chunks",
                "taskType": "codebase_source_chunk",
                "path": relpath,
                "module": file_record.get("module"),
                "language": file_record.get("language"),
                "sha256": source_hash,
                "chunkSHA256": chunk_hash,
                "chunkIndex": index,
                "chunkCount": len(chunks),
                "lineStart": line_start,
                "lineEnd": line_end,
                "text": chunk_text,
                "metadata": {
                    "agentRole": "cortex",
                    "sourceRecordID": file_record.get("id"),
                    "privacy": "static_repo_source_only",
                    "recordKind": "source_chunk",
                    "coverage": "complete_file_text_chunk",
                },
            }
        )
    return records


def _split_source_chunks(text: str):
    lines = text.splitlines()
    if not lines:
        yield 1, 1, ""
        return

    current: list[str] = []
    current_chars = 0
    start_line = 1
    for line_number, line in enumerate(lines, start=1):
        projected_chars = current_chars + len(line) + 1
        if current and (len(current) >= MAX_CHUNK_LINES or projected_chars > MAX_CHUNK_CHARS):
            yield start_line, line_number - 1, "\n".join(current)
            current = []
            current_chars = 0
            start_line = line_number
        current.append(line)
        current_chars += len(line) + 1

    if current:
        yield start_line, start_line + len(current) - 1, "\n".join(current)


def _sft_record(record: dict[str, Any]) -> dict[str, Any]:
    path = str(record.get("path") or ".")
    module = str(record.get("module") or "unknown")
    symbols = record.get("symbols") if isinstance(record.get("symbols"), list) else []
    imports = record.get("imports") if isinstance(record.get("imports"), list) else []
    answer = {
        "path": path,
        "module": module,
        "language": record.get("language"),
        "responsibility": record.get("responsibility"),
        "symbols": symbols[:24],
        "imports": imports[:24],
        "sourceHash": record.get("sha256"),
        "evidenceSnippet": record.get("evidenceSnippet"),
    }
    return {
        "id": _stable_id("codebase_home_sft", record.get("id")),
        "schemaVersion": CODEBASE_HOME_SCHEMA_VERSION,
        "sourceFamily": "codebase_home_sft",
        "taskType": str(record.get("taskType") or "codebase_home_grounding"),
        "messages": [
            {
                "role": "system",
                "content": "You are Lumen Fleet. Ground answers in the app's actual codebase home map.",
            },
            {
                "role": "user",
                "content": f"Where does Lumen implement `{module}` behavior for `{path}`?",
            },
            {
                "role": "assistant",
                "content": json.dumps(answer, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
            },
        ],
        "metadata": {
            "agentRole": "fleet",
            "path": path,
            "module": module,
            "sourceRecordID": record.get("id"),
            "privacy": "static_repo_source_only",
        },
    }


def _chunk_sft_record(chunk: dict[str, Any], file_record: dict[str, Any]) -> dict[str, Any]:
    path = str(chunk.get("path") or "")
    module = str(chunk.get("module") or file_record.get("module") or "unknown")
    line_start = int(chunk.get("lineStart") or 1)
    line_end = int(chunk.get("lineEnd") or line_start)
    answer = {
        "path": path,
        "module": module,
        "language": chunk.get("language"),
        "sourceHash": chunk.get("sha256"),
        "chunkHash": chunk.get("chunkSHA256"),
        "chunkIndex": chunk.get("chunkIndex"),
        "chunkCount": chunk.get("chunkCount"),
        "lineStart": line_start,
        "lineEnd": line_end,
        "chunkText": chunk.get("text") or "",
        "boundary": "This is static tracked source text. Use it for source grounding only; do not infer private runtime state from it.",
    }
    return {
        "id": _stable_id("codebase_home_chunk_sft", chunk.get("id")),
        "schemaVersion": CODEBASE_HOME_SCHEMA_VERSION,
        "sourceFamily": "codebase_home_chunk_sft",
        "taskType": "codebase_source_chunk_grounding",
        "messages": [
            {
                "role": "system",
                "content": "You are Cortex. Memorize Lumen source chunks by path, line range, hash, and adapter boundary.",
            },
            {
                "role": "user",
                "content": f"Ground Cortex on `{path}` lines {line_start}-{line_end}.",
            },
            {
                "role": "assistant",
                "content": json.dumps(answer, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
            },
        ],
        "metadata": {
            "agentRole": "cortex",
            "path": path,
            "module": module,
            "sourceRecordID": file_record.get("id"),
            "sourceChunkID": chunk.get("id"),
            "lineStart": line_start,
            "lineEnd": line_end,
            "sourceHash": chunk.get("sha256"),
            "chunkHash": chunk.get("chunkSHA256"),
            "privacy": "static_repo_source_only",
            "recordKind": "source_chunk_sft",
            "coverage": "complete_file_text_chunk",
        },
    }


def _extract_symbols(path: Path, text: str) -> list[str]:
    if path.suffix == ".swift":
        patterns = [
            r"\b(?:class|struct|enum|actor|protocol|extension)\s+([A-Za-z_][A-Za-z0-9_]*)",
            r"\bfunc\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(",
            r"\b(?:let|var)\s+([A-Za-z_][A-Za-z0-9_]*)\s*[:=]",
        ]
    elif path.suffix == ".py":
        patterns = [
            r"^\s*class\s+([A-Za-z_][A-Za-z0-9_]*)",
            r"^\s*def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(",
        ]
    elif path.suffix == ".md":
        patterns = [r"^#{1,3}\s+(.+)$"]
    else:
        patterns = []

    symbols: list[str] = []
    seen: set[str] = set()
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.MULTILINE):
            value = " ".join(match.group(1).strip().split())
            if value and value not in seen:
                seen.add(value)
                symbols.append(value)
    return symbols


def _extract_imports(path: Path, text: str) -> list[str]:
    if path.suffix == ".swift":
        pattern = r"^\s*import\s+([A-Za-z_][A-Za-z0-9_]*)"
    elif path.suffix == ".py":
        pattern = r"^\s*(?:from\s+([A-Za-z_][A-Za-z0-9_.]*)\s+import|import\s+([A-Za-z_][A-Za-z0-9_.]*))"
    else:
        return []

    imports: list[str] = []
    seen: set[str] = set()
    for match in re.finditer(pattern, text, flags=re.MULTILINE):
        value = next((group for group in match.groups() if group), "")
        if value and value not in seen:
            seen.add(value)
            imports.append(value)
    return imports


def _module_name(relpath: str) -> str:
    parts = Path(relpath).parts
    if len(parts) >= 3 and parts[0] == "ios" and parts[1] == "Lumen":
        return "/".join(parts[2:-1]) or "app"
    if parts and parts[0] in {"tools", "docs"}:
        return "/".join(parts[: min(len(parts) - 1, 3)]) or parts[0]
    return "/".join(parts[:-1]) or "repo"


def _responsibility(relpath: str, symbols: list[str], text: str) -> str:
    header = _leading_comment_or_heading(text)
    if header:
        return header
    name = Path(relpath).stem
    symbol_text = ", ".join(symbols[:10])
    if symbol_text:
        return f"`{relpath}` owns {name} behavior and defines: {symbol_text}."
    return f"`{relpath}` is static Lumen source for the `{_module_name(relpath)}` module."


def _leading_comment_or_heading(text: str) -> str:
    lines: list[str] = []
    for raw in text.splitlines()[:40]:
        stripped = raw.strip()
        if not stripped:
            if lines:
                break
            continue
        if stripped.startswith("# "):
            return stripped.lstrip("# ").strip()
        if stripped.startswith("///") or stripped.startswith("//"):
            cleaned = stripped.lstrip("/").strip()
            if cleaned:
                lines.append(cleaned)
                continue
        if stripped.startswith('"""'):
            cleaned = stripped.strip('"').strip()
            if cleaned:
                lines.append(cleaned)
                continue
        if lines:
            break
    return " ".join(lines)[:300]


def _evidence_snippet(text: str) -> str:
    lines = []
    for raw in text.splitlines():
        stripped = raw.rstrip()
        if not stripped:
            if lines:
                lines.append("")
            continue
        lines.append(stripped[:240])
        if sum(len(line) + 1 for line in lines) >= MAX_SNIPPET_CHARS:
            break
    return "\n".join(lines).strip()[:MAX_SNIPPET_CHARS]


def _language(path: Path) -> str:
    return {
        ".csv": "csv",
        ".entitlements": "plist",
        ".json": "json",
        ".md": "markdown",
        ".pbxproj": "xcode_project",
        ".plist": "plist",
        ".py": "python",
        ".sh": "shell",
        ".swift": "swift",
        ".toml": "toml",
        ".txt": "text",
        ".xcconfig": "xcconfig",
        ".xcprivacy": "json",
        ".xcscheme": "xml",
        ".yaml": "yaml",
        ".yml": "yaml",
    }.get(path.suffix, path.suffix.lstrip(".") or "text")


def _stable_id(*parts: Any) -> str:
    payload = json.dumps(parts, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]
