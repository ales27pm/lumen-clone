from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

from lumen_manifest_crawler.output.hashing import normalized_repo_path, sha256_file

CODEBASE_HOME_SCHEMA_VERSION = "1.0.0"

INCLUDED_SUFFIXES = {
    ".swift",
    ".py",
    ".md",
    ".json",
    ".plist",
    ".entitlements",
    ".xcscheme",
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
    "generated",
    "models",
    "node_modules",
}

MAX_FILE_BYTES = 256_000
MAX_RECORDS = 700
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

    for path in _iter_codebase_files(root):
        record = _code_file_record(root, path)
        if record is None:
            continue
        corpus.append(record)
        sft.append(_sft_record(record))
        if len(corpus) >= MAX_RECORDS:
            break

    overview = _overview_record(root, corpus)
    corpus.insert(0, overview)
    sft.insert(0, _sft_record(overview))

    return {
        "codebase_home_corpus": corpus,
        "codebase_home_sft": sft,
    }


def _iter_codebase_files(root: Path):
    preferred_roots = [root / "ios" / "Lumen", root / "tools", root / "docs"]
    seen: set[Path] = set()
    for base in preferred_roots:
        if not base.exists():
            continue
        for path in _walk_files(base):
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            yield resolved


def _walk_files(base: Path):
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames[:] = sorted(name for name in dirnames if name not in IGNORED_DIRS and not name.endswith(".xcuserdata"))
        for filename in sorted(filenames):
            path = Path(dirpath) / filename
            if path.suffix in INCLUDED_SUFFIXES:
                yield path


def _code_file_record(root: Path, path: Path) -> dict[str, Any] | None:
    try:
        if path.stat().st_size > MAX_FILE_BYTES:
            return None
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None

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
        },
    }


def _overview_record(root: Path, corpus: list[dict[str, Any]]) -> dict[str, Any]:
    modules: dict[str, int] = {}
    languages: dict[str, int] = {}
    for record in corpus:
        modules[str(record.get("module") or "unknown")] = modules.get(str(record.get("module") or "unknown"), 0) + 1
        languages[str(record.get("language") or "unknown")] = languages.get(str(record.get("language") or "unknown"), 0) + 1
    top_modules = sorted(modules.items(), key=lambda item: (-item[1], item[0]))[:30]
    text = "\n".join(
        [
            "Lumen codebase home overview.",
            f"Root: {root.name}",
            "Scanned static repo source, docs, and tool scripts only.",
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
            "languages": dict(sorted(languages.items())),
            "modules": dict(sorted(modules.items())),
        },
    }


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
        ".swift": "swift",
        ".py": "python",
        ".md": "markdown",
        ".json": "json",
        ".plist": "plist",
        ".entitlements": "plist",
        ".xcscheme": "xml",
    }.get(path.suffix, path.suffix.lstrip(".") or "text")


def _stable_id(*parts: Any) -> str:
    payload = json.dumps(parts, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]
