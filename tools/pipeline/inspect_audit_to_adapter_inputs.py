#!/usr/bin/env python3
"""Strict inspection CLI for Lumen audit inputs before dataset generation."""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

from audit_package_inspector import assert_audit_requirements, inspect_audit_files
from audit_to_adapter_contract import CONTRACT, runtime_audit_candidates


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect Lumen audit inputs before ingest/training.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--runtime-audit", action="append", default=[], help="Audit file or glob. May be repeated.")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--require-runtime-audit", action="store_true")
    parser.add_argument("--require-adapter-traces", action="store_true")
    parser.add_argument("--require-training-signals", action="store_true")
    parser.add_argument("--write", type=Path, default=None, help="Write inspection JSON to this path.")
    return parser.parse_args(argv)


def resolve(root: Path, path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    return candidate if candidate.is_absolute() else root / candidate


def expand_runtime_audits(root: Path, explicit: list[str]) -> list[Path]:
    if not explicit:
        return runtime_audit_candidates(root)
    found: list[Path] = []
    for raw in explicit:
        pattern = str(resolve(root, raw))
        matches = [Path(path).resolve() for path in glob.glob(pattern, recursive=True)]
        if not matches and Path(pattern).exists():
            matches = [Path(pattern).resolve()]
        found.extend(path for path in matches if path.is_file())
    return sorted({path for path in found})


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.root.resolve()
    paths = expand_runtime_audits(root, args.runtime_audit)
    summary = inspect_audit_files(paths)
    payload = summary.as_dict()
    payload["contractSchema"] = CONTRACT.schema_version
    payload["contractFamily"] = CONTRACT.family
    payload["contractMode"] = CONTRACT.mode

    errors = assert_audit_requirements(
        summary,
        require_runtime_audit=args.require_runtime_audit,
        require_adapter_traces=args.require_adapter_traces,
        require_training_signals=args.require_training_signals,
    )
    payload["ok"] = not errors
    payload["requirementErrors"] = errors

    if args.write:
        target = resolve(root, args.write)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(f"contract={CONTRACT.schema_version} family={CONTRACT.family} mode={CONTRACT.mode}")
        print(f"files={summary.file_count} in_app_packages={summary.in_app_package_count} traces={summary.trace_count}")
        print(f"adapterApplied true={summary.adapter_applied_true_count} false={summary.adapter_applied_false_count} missing={summary.adapter_applied_missing_count}")
        print(f"trainingSignals accepted={summary.accepted_training_count} regression={summary.regression_test_count}")
        for warning in summary.warnings[:40]:
            print(f"WARN {warning}")
        for error in errors:
            print(f"FAIL {error}")
        if not errors:
            print("PASS audit input inspection")

    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
