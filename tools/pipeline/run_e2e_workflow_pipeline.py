#!/usr/bin/env python3
"""Async all-in-one E2E workflow runner for Lumen pipeline stages.

The script intentionally delegates domain-specific work to the existing pipeline
CLIs. Its job is orchestration: preserve the active virtualenv, run stages with
async subprocesses, stream logs, write machine-readable state, and stop safely on
failures.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shlex
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class StagePlan:
    id: str
    description: str
    command: list[str]


@dataclass
class StageResult:
    id: str
    description: str
    command: list[str]
    returncode: int
    elapsed_s: float
    started_at: str
    finished_at: str
    log_path: str | None


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def executable_python(path: Path) -> str | None:
    return str(path) if path.is_file() and os.access(path, os.X_OK) else None


def default_python(root: Path) -> str:
    # Prefer the repository virtualenv path itself. Do not resolve this symlink:
    # resolving .venv/bin/python to /usr/bin/python3.12 drops venv site-packages.
    candidate = executable_python(root / ".venv" / "bin" / "python")
    if candidate:
        return candidate

    venv = os.environ.get("VIRTUAL_ENV")
    if venv:
        candidate = executable_python(Path(venv) / "bin" / "python")
        if candidate:
            return candidate

    return sys.executable


def resolve(root: Path, path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    return candidate if candidate.is_absolute() else root / candidate


def resolve_under_root(root: Path, path: str | Path, *, label: str) -> Path:
    resolved = resolve(root, path).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise SystemExit(f"{label} must stay under repository root: {resolved}") from exc
    return resolved


def validate_executable(root: Path, value: str | Path, *, label: str) -> str:
    path = resolve(root, value)
    absolute = path if path.is_absolute() else root / path
    if not absolute.is_file() or not os.access(absolute, os.X_OK):
        raise SystemExit(f"{label} must exist and be executable: {absolute}")
    return str(absolute)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Lumen E2E workflow pipeline stages asynchronously.")
    parser.add_argument("--root", type=Path, default=repo_root())
    parser.add_argument("--python", default=None, help="Python interpreter for pipeline helper scripts. Defaults to repo .venv.")
    parser.add_argument("--train-python", default=None, help="Python interpreter for train/convert stages. Defaults to LUMEN_TRAIN_PYTHON or repo .venv.")
    parser.add_argument("--agents", default=None, help="Comma-separated adapter roles to train/convert; forwarded to the base runner.")
    parser.add_argument("--runtime-audit", action="append", default=[], help="Runtime audit file or glob. May be repeated. Latest-only filtering is handled by the shared contract.")
    parser.add_argument("--state-file", type=Path, default=Path("generated/agent_improvement_loop/e2e_workflow_state.json"))
    parser.add_argument("--logs-dir", type=Path, default=Path("generated/agent_improvement_loop/e2e_logs"))
    parser.add_argument("--inspection-output", type=Path, default=Path("generated/agent_improvement_loop/e2e_audit_input_inspection.json"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--require-runtime-audit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-adapter-traces", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-training-signals", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--inspect", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--validate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ingest", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--train-adapters", action="store_true", help="Run adapter SFT training after ingest.")
    parser.add_argument("--convert-adapters", action="store_true", help="Convert trained adapters to GGUF after training or ingest.")
    parser.add_argument("--upload-adapters", action="store_true", help="Upload adapter GGUF artifacts with the HF CLI.")
    parser.add_argument("--upload-base", action="store_true", help="Upload the shared base GGUF artifact with the HF CLI.")
    parser.add_argument("--hf-private", action="store_true")
    parser.add_argument("--large-folder-upload", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--assistant-only-loss", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stop-on-error", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--json", action="store_true", help="Print final state as JSON.")
    return parser.parse_args(argv)


def runtime_audit_flags(args: argparse.Namespace) -> list[str]:
    flags: list[str] = []
    for audit in args.runtime_audit:
        flags.extend(["--runtime-audit", str(audit)])
    if args.require_runtime_audit:
        flags.append("--require-runtime-audit")
    return flags


def runner_common_flags(args: argparse.Namespace) -> list[str]:
    flags = ["--root", str(args.root), "--python", str(args.python), "--train-python", str(args.train_python), "--seed", str(args.seed)]
    if args.agents:
        flags.extend(["--agents", args.agents])
    if args.hf_private:
        flags.append("--hf-private")
    flags.append("--large-folder-upload" if args.large_folder_upload else "--no-large-folder-upload")
    flags.append("--assistant-only-loss" if args.assistant_only_loss else "--no-assistant-only-loss")
    flags.append("--dry-run") if args.dry_run else None
    flags.append("--stop-on-error" if args.stop_on_error else "--no-stop-on-error")
    return flags


def build_stage_plan(args: argparse.Namespace) -> list[StagePlan]:
    root = args.root
    python = str(args.python)
    plans: list[StagePlan] = []

    if args.inspect:
        cmd = [
            python,
            "tools/pipeline/inspect_audit_to_adapter_inputs.py",
            "--root",
            str(root),
            "--write",
            str(args.inspection_output),
            *runtime_audit_flags(args),
        ]
        if args.require_adapter_traces:
            cmd.append("--require-adapter-traces")
        if args.require_training_signals:
            cmd.append("--require-training-signals")
        plans.append(StagePlan("inspect-audits", "Inspect latest runtime audit inputs", cmd))

    if args.validate:
        cmd = [
            python,
            "tools/pipeline/run_audit_to_adapter_pipeline.py",
            "--mode",
            "validate",
            *runner_common_flags(args),
            *runtime_audit_flags(args),
        ]
        plans.append(StagePlan("contract-validate", "Validate audit-to-adapter contract alignment", cmd))

    if args.ingest:
        cmd = [
            python,
            "tools/pipeline/run_audit_to_adapter_pipeline.py",
            "--mode",
            "ingest",
            *runner_common_flags(args),
            *runtime_audit_flags(args),
        ]
        plans.append(StagePlan("ingest", "Ingest audits, crawl code, and generate datasets", cmd))

    optional_runner_modes = [
        (args.train_adapters, "train-adapters", "Train role adapters"),
        (args.convert_adapters, "convert-adapters", "Convert role adapters to GGUF"),
        (args.upload_adapters, "upload-adapters", "Upload adapter GGUF artifacts"),
        (args.upload_base, "upload-base", "Upload shared base GGUF artifact"),
    ]
    for enabled, mode, description in optional_runner_modes:
        if not enabled:
            continue
        cmd = [
            python,
            "tools/pipeline/run_audit_to_adapter_pipeline.py",
            "--mode",
            mode,
            *runner_common_flags(args),
            *runtime_audit_flags(args),
        ]
        plans.append(StagePlan(mode, description, cmd))

    return plans


def build_env(root: Path, args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    pipeline_path = str((root / "tools/pipeline").resolve())
    crawler_path = str((root / "tools/lumen_manifest_crawler").resolve())
    venv_bin = root / ".venv" / "bin"
    existing_pythonpath = env.get("PYTHONPATH")
    additions = os.pathsep.join([pipeline_path, crawler_path])
    env["PYTHONPATH"] = additions if not existing_pythonpath else f"{additions}{os.pathsep}{existing_pythonpath}"
    if venv_bin.is_dir():
        existing_path = env.get("PATH", "")
        env["PATH"] = f"{venv_bin}{os.pathsep}{existing_path}" if existing_path else str(venv_bin)
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("PYTHONHASHSEED", str(args.seed))
    env.setdefault("LUMEN_TRAIN_SEED", str(args.seed))
    env.setdefault("LUMEN_E2E_WORKFLOW_PIPELINE", "1")
    return env


async def run_stage(root: Path, plan: StagePlan, args: argparse.Namespace, env: dict[str, str]) -> StageResult:
    logs_dir = resolve_under_root(root, args.logs_dir, label="--logs-dir")
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"{len(list(logs_dir.glob('*.log'))) + 1:02d}-{plan.id}.log"

    printable = [str(part) for part in plan.command]
    print(f"\n== {plan.description}")
    print(shlex.join(printable))

    started_at = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()

    if args.dry_run:
        log_path.write_text(shlex.join(printable) + "\n", encoding="utf-8")
        finished_at = datetime.now(timezone.utc).isoformat()
        return StageResult(plan.id, plan.description, printable, 0, 0.0, started_at, finished_at, str(log_path.relative_to(root)))

    process = await asyncio.create_subprocess_exec(
        *printable,
        cwd=root,
        env=env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )

    with log_path.open("w", encoding="utf-8") as handle:
        assert process.stdout is not None
        while True:
            line = await process.stdout.readline()
            if not line:
                break
            text = line.decode("utf-8", errors="replace")
            print(text, end="")
            handle.write(text)

    returncode = await process.wait()
    elapsed = time.perf_counter() - started
    finished_at = datetime.now(timezone.utc).isoformat()
    return StageResult(plan.id, plan.description, printable, returncode, elapsed, started_at, finished_at, str(log_path.relative_to(root)))


def write_state(root: Path, args: argparse.Namespace, results: list[StageResult], *, ok: bool) -> None:
    if args.dry_run:
        return
    state_path = resolve_under_root(root, args.state_file, label="--state-file")
    state_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "lumen.e2e_workflow_pipeline.state/1.0.0",
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "ok": ok,
        "root": str(root),
        "python": str(args.python),
        "train_python": str(args.train_python),
        "runtime_audit_count_requested": len(args.runtime_audit),
        "stages": [asdict(result) for result in results],
    }
    state_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


async def async_main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.root = args.root.resolve()
    if not (args.root / "ios/Lumen").exists():
        raise SystemExit(f"root does not look like lumen-clone: {args.root}")

    args.python = validate_executable(args.root, args.python or default_python(args.root), label="--python")
    args.train_python = validate_executable(args.root, args.train_python or os.environ.get("LUMEN_TRAIN_PYTHON") or default_python(args.root), label="--train-python")
    args.logs_dir = resolve_under_root(args.root, args.logs_dir, label="--logs-dir")
    args.state_file = resolve_under_root(args.root, args.state_file, label="--state-file")
    args.inspection_output = resolve_under_root(args.root, args.inspection_output, label="--inspection-output")

    plans = build_stage_plan(args)
    if not plans:
        raise SystemExit("no stages selected")

    env = build_env(args.root, args)
    results: list[StageResult] = []

    for plan in plans:
        result = await run_stage(args.root, plan, args, env)
        results.append(result)
        ok_so_far = all(stage.returncode == 0 for stage in results)
        write_state(args.root, args, results, ok=ok_so_far)
        if result.returncode != 0 and args.stop_on_error:
            if args.json:
                print(json.dumps({"ok": False, "failed_stage": result.id, "stages": [asdict(item) for item in results]}, ensure_ascii=False, indent=2, sort_keys=True))
            return result.returncode

    ok = all(result.returncode == 0 for result in results)
    write_state(args.root, args, results, ok=ok)

    if args.json:
        print(json.dumps({"ok": ok, "stages": [asdict(item) for item in results]}, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print("\nCompleted E2E workflow stages:")
        for result in results:
            print(f"- {result.id}: rc={result.returncode} elapsed={result.elapsed_s:.1f}s log={result.log_path}")
        print(f"state={args.state_file.relative_to(args.root)}")

    return 0 if ok else 1


def main(argv: Sequence[str] | None = None) -> int:
    return asyncio.run(async_main(argv))


if __name__ == "__main__":
    raise SystemExit(main())
