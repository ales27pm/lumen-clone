#!/usr/bin/env python3
"""Static Release hardening guard for Lumen runtime/product surfaces."""

from __future__ import annotations

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SOURCE_ROOTS = [ROOT / "ios" / "Lumen"]
DOC_ROOTS = [
    ROOT / "README.md",
    ROOT / "CLAUDE.md",
    ROOT / "docs" / "RUNTIME_STATUS_MATRIX.md",
    ROOT / "docs" / "AGENT_KERNEL_MIGRATION_STATUS.md",
    ROOT / "docs" / "VALIDATION.md",
]

FORBIDDEN_SOURCE_PATTERNS = {
    "removed generation sentinel": re.compile(r"\bgenerationNotImplemented\b"),
    "removed embedding sentinel": re.compile(r"\bembeddingExtractionNotImplemented\b"),
    "old generic fallback copy": re.compile(r"limited local mode", re.IGNORECASE),
    "staged runtime wording": re.compile(r"staged:\s*implementation missing|generation is staged|runtime staged", re.IGNORECASE),
    "mock backend registration": re.compile(r"register\s*\([^)]*mock|for:\s*\.mock", re.IGNORECASE),
}

FORBIDDEN_DOC_PATTERNS = {
    "shipped partial wording": re.compile(r"\bpartial\b", re.IGNORECASE),
    "shipped planned wording": re.compile(r"\bplanned\b", re.IGNORECASE),
    "shipped compatibility bridge wording": re.compile(r"compatibility bridge", re.IGNORECASE),
}

DEBUG_ONLY_PATTERNS = {
    "legacy bridge call": re.compile(r"\.runLegacyAgentBridge\s*\("),
    "legacy compatibility bridge call": re.compile(r"\bLegacyAgentCompatibilityBridge\.(runLegacyAgentService|runSlotAgentKernelCompatibility|runSlotAgentCompatibility)\b"),
    "unavailable gguf bridge construction": re.compile(r"\bUnavailableGGUFNativeBridge\s*\("),
}


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


def _condition_contains_debug(directive: str) -> bool:
    condition = re.sub(r"//.*$", "", directive)
    for match in re.finditer(r"\bDEBUG\b", condition):
        prefix = condition[: match.start()].rstrip()
        while True:
            normalized = prefix
            while normalized.endswith("("):
                normalized = normalized[:-1].rstrip()
            normalized = re.sub(r"\bdefined\s*$", "", normalized).rstrip()
            if normalized == prefix:
                break
            prefix = normalized
        if not prefix.endswith("!"):
            return True
    return False


def _is_else_directive(stripped: str) -> bool:
    return re.match(r"^#else(?:\s|//|$)", stripped) is not None


def debug_stack_for_lines(lines: list[str]) -> list[bool]:
    stack: list[bool] = []
    states: list[bool] = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("#if"):
            stack.append(_condition_contains_debug(stripped))
        elif stripped.startswith("#elseif") and stack:
            stack[-1] = _condition_contains_debug(stripped)
        elif _is_else_directive(stripped) and stack:
            stack[-1] = not stack[-1]
        elif stripped.startswith("#endif") and stack:
            stack.pop()
        states.append(any(stack))
    return states


def scan_source() -> list[str]:
    violations: list[str] = []
    for path in iter_files(SOURCE_ROOTS, (".swift", ".h", ".m", ".mm")):
        text = path.read_text(encoding="utf-8")
        lines = text.splitlines()
        debug_states = debug_stack_for_lines(lines)
        relative = rel(path)
        for line_number, line in enumerate(lines, start=1):
            debug_only = debug_states[line_number - 1]
            for label, pattern in FORBIDDEN_SOURCE_PATTERNS.items():
                if pattern.search(line):
                    violations.append(f"{relative}:{line_number}: {label}: {line.strip()}")
            for label, pattern in DEBUG_ONLY_PATTERNS.items():
                if pattern.search(line) and not debug_only:
                    violations.append(f"{relative}:{line_number}: {label} must be inside #if DEBUG: {line.strip()}")
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


def main() -> int:
    violations = scan_source() + scan_docs()
    if violations:
        print("Release hardening violations detected:", file=sys.stderr)
        for violation in violations:
            print(f"  {violation}", file=sys.stderr)
        return 1
    print("Release hardening guard passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
