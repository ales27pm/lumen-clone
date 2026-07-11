#!/usr/bin/env python3
"""Guard the Agent Kernel migration boundary.

This guard allows only exact documented compatibility bridge files while the
Agent Kernel migration finishes. The script fails when a new production path
starts calling legacy runtimes directly.
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

DOCUMENTED_COMPATIBILITY_BRIDGES = {
    "ios/Lumen/Assistant/LegacyAgentCompatibilityBridge.swift": {
        "AgentService.shared.run": (
            "temporary kernel-owned bridge for diagnostics and tool-capable "
            "legacy agent event streams; remove when AgentKernel emits native "
            "tool stages for those paths"
        ),
        "SlotAgentService.shared.run": (
            "temporary slot-agent bridge for diagnostics and deterministic "
            "compatibility responses; remove when those paths are kernel-native"
        ),
    },
}


def repo_relative(path: pathlib.Path) -> str:
    return path.resolve().relative_to(REPO_ROOT).as_posix()


def iter_swift_files(roots: list[pathlib.Path]):
    for root in roots:
        if root.is_file() and root.suffix == ".swift":
            yield root
        elif root.exists():
            yield from root.rglob("*.swift")


def scan_file(path: pathlib.Path) -> list[tuple[int, str, str, bool]]:
    text = path.read_text(encoding="utf-8")
    findings: list[tuple[int, str, str, bool]] = []
    conditional_stack: list[bool] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if re.match(r"#if\s+DEBUG\b", stripped):
            conditional_stack.append(True)
            continue
        if stripped.startswith("#elseif") and conditional_stack:
            conditional_stack[-1] = bool(re.match(r"#elseif\s+DEBUG\b", stripped))
            continue
        if stripped.startswith("#else") and conditional_stack:
            conditional_stack[-1] = False
            continue
        if stripped.startswith("#endif") and conditional_stack:
            conditional_stack.pop()
            continue

        is_debug_only = any(conditional_stack)
        for label, pattern in LEGACY_PATTERNS.items():
            if pattern.search(line):
                findings.append((line_number, label, stripped, is_debug_only))
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
        for line_number, label, line, is_debug_only in findings:
            record = f"{rel}:{line_number}: {label}: {line}"
            if label in DOCUMENTED_COMPATIBILITY_BRIDGES.get(rel, {}):
                if args.strict and not is_debug_only:
                    violations.append(record)
                else:
                    legacy_inventory.append(record)
            else:
                violations.append(record)

    if not args.roots:
        streaming_path = REPO_ROOT / "ios" / "Lumen" / "Assistant" / "AssistantKernel+Streaming.swift"
        executor_path = REPO_ROOT / "ios" / "Lumen" / "Assistant" / "StructuredAgentKernelExecutor.swift"
        streaming = streaming_path.read_text(encoding="utf-8") if streaming_path.exists() else ""
        if not executor_path.exists():
            violations.append("ios/Lumen/Assistant/StructuredAgentKernelExecutor.swift: missing Release-native structured executor")
        if "structuredMode == .requiredAgentJSON" not in streaming or "StructuredAgentKernelExecutor" not in streaming:
            violations.append(
                "ios/Lumen/Assistant/AssistantKernel+Streaming.swift: AssistantKernel.run must route requiredAgentJSON to StructuredAgentKernelExecutor"
            )

    if violations:
        print("Agent Kernel boundary violations detected:", file=sys.stderr)
        for item in violations:
            print(f"  {item}", file=sys.stderr)
        print("\nRoute new work through AssistantKernel.run(...) or keep temporary bridges DEBUG-only.", file=sys.stderr)
        return 1

    print("Agent Kernel boundary guard passed.")
    if legacy_inventory:
        print(f"Documented compatibility bridges: {len(legacy_inventory)}")
        if args.strict:
            for item in legacy_inventory:
                print(f"  {item}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
