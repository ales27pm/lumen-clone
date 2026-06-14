#!/usr/bin/env python3
"""Validate the Lumen audit -> adapter -> iOS runtime pipeline contract."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from audit_to_adapter_contract import CONTRACT, runtime_audit_candidates, validate_repository_alignment, write_contract_json


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Lumen's audit-to-adapter pipeline alignment.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--json", action="store_true", help="Emit machine-readable validation output.")
    parser.add_argument("--write-contract-json", type=Path, default=None, help="Write the canonical pipeline contract to this JSON path.")
    parser.add_argument("--require-generated-artifacts", action="store_true", help="Also require trained adapters, GGUF adapters, and shared base artifacts to exist locally.")
    parser.add_argument("--require-runtime-audit", action="store_true", help="Fail if no runtime audit JSON candidates are present.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.root.resolve()

    errors = validate_repository_alignment(root, require_generated_artifacts=args.require_generated_artifacts)
    audits = runtime_audit_candidates(root)
    if args.require_runtime_audit and not audits:
        errors.append("no runtime audit JSONs found in configured audit globs")

    if args.write_contract_json is not None:
        output = args.write_contract_json
        if not output.is_absolute():
            output = root / output
        write_contract_json(output)

    payload = {
        "schema": "lumen.audit_to_adapter_pipeline.validation/1.0.0",
        "contract_schema": CONTRACT.schema_version,
        "family": CONTRACT.family,
        "mode": CONTRACT.mode,
        "live_runtime_slots": list(CONTRACT.live_runtime_slots),
        "trained_adapter_roles": list(CONTRACT.trained_adapter_roles),
        "runtime_audit_candidates": [str(path.relative_to(root)) if path.is_relative_to(root) else str(path) for path in audits],
        "stage_ids": [stage.id for stage in CONTRACT.stages],
        "ok": not errors,
        "errors": errors,
    }

    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(f"contract={payload['contract_schema']} family={payload['family']} mode={payload['mode']}")
        print("stages=" + ",".join(payload["stage_ids"]))
        print(f"runtime_audits={len(audits)}")
        if errors:
            print("\nFAIL")
            for error in errors:
                print(f"- {error}")
        else:
            print("PASS audit-to-adapter pipeline alignment")

    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
