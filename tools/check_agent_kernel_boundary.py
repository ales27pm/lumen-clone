#!/usr/bin/env python3
"""Guard the Agent Kernel migration boundary.

This first-pass guard intentionally allows the legacy call sites that exist at
the start of the migration. Each migration PR should delete entries from
ALLOWED_LEGACY_CALLERS until the allowlist is empty. The script fails when a new
production path starts calling legacy runtimes directly.
"""
from __future__ import annotations

import argparse
import pathlib
import re
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_ROOTS = [REPO_ROOT / "ios" / "Lumen"]

LEGACY_PATTERNS = {
    "AgentService.shared.run": re.compile(r"\bAgentService\.shared\.run\b"),
    "SlotAgentService.shared.run": re.compile(r"\bSlotAgentService\.shared\.run\b"),
    "RolePipelineAgentService.shared.run": re.compile(r"\bRolePipelineAgentService\.shared\.run\b"),
    "AgentRunner.runHeadless": re.compile(r"\bAgentRunner\.runHeadless\b"),
    "ToolExecutor.shared.execute": re.compile(r"\bToolExecutor\.shared\.execute\b"),
    "LegacySecureToolExecutor": re.compile(r"\bLegacySecureToolExecutor\b"),
}

# Line-specific snapshot of known callers before the Agent Kernel migration.
# Shrink this list as each entrypoint is moved behind AssistantKernel.run(...).
ALLOWED_LEGACY_CALLERS = {
    ("ios/Lumen/Services/AgentService.swift", 1324, "SlotAgentService.shared.run"),
}

ALLOWED_MIGRATION_FILES = {
    "ios/Lumen/Assistant/AgentKernelContracts.swift",
    "ios/Lumen/Assistant/AssistantKernel+Streaming.swift",
}


def repo_relative(path: pathlib.Path) -> str:
    return path.resolve().relative_to(REPO_ROOT).as_posix()


def iter_swift_files(roots: list[pathlib.Path]):
    for root in roots:
        if root.is_file() and root.suffix == ".swift":
            yield root
        elif root.exists():
            yield from root.rglob("*.swift")


def scan_file(path: pathlib.Path) -> list[tuple[int, str, str]]:
    text = path.read_text(encoding="utf-8")
    findings: list[tuple[int, str, str]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        for label, pattern in LEGACY_PATTERNS.items():
            if pattern.search(line):
                findings.append((line_number, label, line.strip()))
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="*", help="Files or directories to scan; defaults to ios/Lumen")
    parser.add_argument("--strict", action="store_true", help="Fail on every legacy call, including migration shims and the initial allowlist")
    args = parser.parse_args()

    roots = [REPO_ROOT / root for root in args.roots] if args.roots else DEFAULT_ROOTS
    violations: list[str] = []
    legacy_inventory: list[str] = []

    for path in sorted(iter_swift_files(roots)):
        rel = repo_relative(path)
        findings = scan_file(path)
        if not findings:
            continue
        for line_number, label, line in findings:
            record = f"{rel}:{line_number}: {label}: {line}"
            allowed_record = (rel, line_number, label)
            if not args.strict and rel in ALLOWED_MIGRATION_FILES:
                continue
            if args.strict or allowed_record not in ALLOWED_LEGACY_CALLERS:
                violations.append(record)
            else:
                legacy_inventory.append(record)

    if violations:
        print("Agent Kernel boundary violations detected:", file=sys.stderr)
        for item in violations:
            print(f"  {item}", file=sys.stderr)
        print("\nRoute new work through AssistantKernel.run(...) or add a temporary allowlist entry with a removal PR.", file=sys.stderr)
        return 1

    print("Agent Kernel boundary guard passed.")
    if legacy_inventory:
        print(f"Known legacy callers still allowlisted: {len(legacy_inventory)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
