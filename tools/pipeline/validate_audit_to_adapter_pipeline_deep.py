#!/usr/bin/env python3
"""Deep validator for Lumen's audit-to-adapter pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from audit_package_inspector import assert_audit_requirements, inspect_audit_files
from audit_to_adapter_contract import CONTRACT, runtime_audit_candidates, validate_repository_alignment, write_contract_json


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deep-validate Lumen's audit-to-adapter pipeline alignment.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--write-contract-json", type=Path, default=None)
    parser.add_argument("--write-audit-inspection", type=Path, default=None)
    parser.add_argument("--require-generated-artifacts", action="store_true")
    parser.add_argument("--require-runtime-audit", action="store_true")
    parser.add_argument("--require-adapter-traces", action="store_true")
    parser.add_argument("--require-training-signals", action="store_true")
    return parser.parse_args(argv)


def _resolve(root: Path, target: Path) -> Path:
    return target if target.is_absolute() else root / target


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.root.resolve()
    audits = runtime_audit_candidates(root)
    audit_summary = inspect_audit_files(audits)
    errors = validate_repository_alignment(root, require_generated_artifacts=args.require_generated_artifacts)
    errors.extend(
        assert_audit_requirements(
            audit_summary,
            require_runtime_audit=args.require_runtime_audit,
            require_adapter_traces=args.require_adapter_traces,
            require_training_signals=args.require_training_signals,
        )
    )

    if args.write_contract_json is not None:
        write_contract_json(_resolve(root, args.write_contract_json))
    if args.write_audit_inspection is not None:
        path = _resolve(root, args.write_audit_inspection)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(audit_summary.as_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    payload = {
        "schema": "lumen.audit_to_adapter_pipeline.deep_validation/1.0.0",
        "contract_schema": CONTRACT.schema_version,
        "family": CONTRACT.family,
        "mode": CONTRACT.mode,
        "stage_ids": [stage.id for stage in CONTRACT.stages],
        "auditInspection": audit_summary.as_dict(),
        "ok": not errors,
        "errors": errors,
    }

    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(f"contract={CONTRACT.schema_version} family={CONTRACT.family} mode={CONTRACT.mode}")
        print("stages=" + ",".join(payload["stage_ids"]))
        print(f"audit_files={audit_summary.file_count} in_app_packages={audit_summary.in_app_package_count} traces={audit_summary.trace_count}")
        print(f"adapterApplied true={audit_summary.adapter_applied_true_count} false={audit_summary.adapter_applied_false_count} missing={audit_summary.adapter_applied_missing_count}")
        print(f"trainingSignals accepted={audit_summary.accepted_training_count} regression={audit_summary.regression_test_count}")
        for warning in audit_summary.warnings[:40]:
            print(f"WARN {warning}")
        for error in errors:
            print(f"FAIL {error}")
        if not errors:
            print("PASS deep audit-to-adapter pipeline validation")

    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
