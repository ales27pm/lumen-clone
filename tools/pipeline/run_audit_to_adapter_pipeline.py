#!/usr/bin/env python3
"""Contract-driven runner for the Lumen audit -> adapter -> app pipeline.

This is a thin, explicit orchestrator. It does not replace the lower-level tools;
it wires them together using `audit_to_adapter_contract.py` so the app catalog,
training configs, upload repos, and runtime expectations stay aligned.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Sequence

from audit_to_adapter_contract import (
    ADAPTER_GGUF_DIR,
    ADAPTER_REPO_ID,
    CONTRACT,
    FINE_TUNING_OUTPUT_DIR,
    LOOP_OUTPUT_DIR,
    MANIFEST_OUTPUT_DIR,
    PIPELINE_STATE_FILE,
    SHARED_BASE_FILE_NAME,
    SHARED_BASE_GGUF_DIR,
    SHARED_BASE_MODEL_ID,
    SHARED_BASE_REPO_ID,
    TRAINED_ADAPTER_ROLES,
    adapter_file_name,
    expand_runtime_audit_paths,
    trained_adapter_path,
    validate_repository_alignment,
)


@dataclass
class StageResult:
    stage: str
    command: list[str]
    returncode: int
    elapsed_s: float
    started_at: str
    finished_at: str


StageFn = Callable[[Path, argparse.Namespace], list[StageResult]]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def executable_python(path: Path) -> str | None:
    return str(path) if path.is_file() and os.access(path, os.X_OK) else None


def default_python(root: Path | None = None) -> str:
    root = root or repo_root()
    candidate = executable_python(root / ".venv" / "bin" / "python")
    if candidate:
        return candidate
    venv = os.environ.get("VIRTUAL_ENV")
    if venv:
        candidate = executable_python(Path(venv) / "bin" / "python")
        if candidate:
            return candidate
    return sys.executable


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Lumen's audit-to-adapter pipeline.")
    parser.add_argument("--root", type=Path, default=repo_root())
    parser.add_argument(
        "--mode",
        choices=["validate", "ingest", "train-adapters", "convert-adapters", "upload-adapters", "upload-base", "full"],
        default="validate",
    )
    parser.add_argument("--agents", default=",".join(TRAINED_ADAPTER_ROLES))
    parser.add_argument("--runtime-audit", action="append", default=[], help="Runtime audit file or glob. May be repeated.")
    parser.add_argument("--require-runtime-audit", action="store_true")
    parser.add_argument("--require-generated-artifacts", action="store_true")
    parser.add_argument("--state-file", type=Path, default=PIPELINE_STATE_FILE)
    parser.add_argument("--python", default=None, help="Python interpreter for crawler/helper scripts. Defaults to repo .venv when present.")
    parser.add_argument("--train-python", default=None, help="Python interpreter for Unsloth training/conversion. Defaults to LUMEN_TRAIN_PYTHON or repo .venv.")
    parser.add_argument("--converter", type=Path, default=Path.home() / ".unsloth/llama.cpp/convert_lora_to_gguf.py")
    parser.add_argument("--base-model-id", default=SHARED_BASE_MODEL_ID)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--assistant-only-loss", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--hf-private", action="store_true")
    parser.add_argument("--large-folder-upload", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stop-on-error", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args(argv)


def parse_agents(raw: str) -> list[str]:
    agents = [part.strip().lower() for part in raw.split(",") if part.strip()]
    invalid = [agent for agent in agents if agent not in TRAINED_ADAPTER_ROLES]
    if invalid:
        raise SystemExit(f"Invalid agent(s): {', '.join(invalid)}. Valid: {', '.join(TRAINED_ADAPTER_ROLES)}")
    return agents or list(TRAINED_ADAPTER_ROLES)


def resolve(root: Path, path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    return candidate if candidate.is_absolute() else root / candidate


def resolve_under_root(root: Path, path: str | Path, *, label: str) -> Path:
    resolved = resolve(root, path).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise SystemExit(f"{label} must be under repository root: {resolved}") from exc
    return resolved


def validate_executable_path(root: Path, value: str | Path, *, label: str) -> str:
    path = resolve(root, value).resolve()
    if not path.is_file() or not os.access(path, os.X_OK):
        raise SystemExit(f"{label} must exist and be executable: {path}")
    return str(path)


def validate_script_path(root: Path, value: str | Path, *, label: str) -> Path:
    path = resolve(root, value).resolve()
    if not path.is_file():
        raise SystemExit(f"{label} must be a file: {path}")
    return path


def validate_tool_paths(root: Path, args: argparse.Namespace) -> None:
    # This runner is intended for controlled developer/CI environments. Dynamic
    # interpreter/script paths are validated before command construction and are
    # passed to subprocess as separate argv entries with shell=False.
    if args.python is None:
        args.python = default_python(root)
    if args.train_python is None:
        args.train_python = os.environ.get("LUMEN_TRAIN_PYTHON") or default_python(root)
    args.python = validate_executable_path(root, args.python, label="--python")
    args.train_python = validate_executable_path(root, args.train_python, label="--train-python")
    if args.mode in {"convert-adapters", "full"}:
        args.converter = validate_script_path(root, args.converter, label="--converter")


def validate_stage_command(command: Sequence[str]) -> None:
    if not command:
        raise SystemExit("refusing to execute empty command")
    if command[0] == "hf":
        return
    first = Path(command[0])
    if not first.is_file() or not os.access(first, os.X_OK):
        raise SystemExit(f"refusing to execute non-executable command: {command[0]}")


def run_stage(root: Path, args: argparse.Namespace, stage: str, command: Sequence[str | Path]) -> StageResult:
    printable = [str(part) for part in command]
    validate_stage_command(printable)
    print(f"\n== {stage}")
    print(shlex.join(printable))
    started_at = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    if args.dry_run:
        finished_at = datetime.now(timezone.utc).isoformat()
        result = StageResult(stage, printable, 0, 0.0, started_at, finished_at)
        getattr(args, "_stage_results").append(result)
        return result
    env = os.environ.copy()
    crawler = str((root / "tools/lumen_manifest_crawler").resolve())
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = crawler if not existing_pythonpath else f"{crawler}{os.pathsep}{existing_pythonpath}"
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("PYTHONHASHSEED", str(args.seed))
    env.setdefault("LUMEN_TRAIN_SEED", str(args.seed))
    completed = subprocess.run(printable, cwd=root, env=env, check=False, shell=False)
    elapsed = time.perf_counter() - started
    finished_at = datetime.now(timezone.utc).isoformat()
    result = StageResult(stage, printable, completed.returncode, elapsed, started_at, finished_at)
    getattr(args, "_stage_results").append(result)
    if completed.returncode != 0 and args.stop_on_error:
        write_state(root, args, getattr(args, "_stage_results"))
        raise SystemExit(completed.returncode)
    return result


def write_state(root: Path, args: argparse.Namespace, results: Sequence[StageResult]) -> None:
    if args.dry_run:
        return
    path = resolve_under_root(root, args.state_file, label="--state-file")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "lumen.audit_to_adapter_pipeline.run_state/1.0.0",
        "contract_schema": CONTRACT.schema_version,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "agents": parse_agents(args.agents),
        "stages": [asdict(result) for result in results],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def validate(root: Path, args: argparse.Namespace) -> list[StageResult]:
    started_at = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    errors = validate_repository_alignment(root, require_generated_artifacts=args.require_generated_artifacts)
    audits = expand_runtime_audit_paths(root, args.runtime_audit)
    if args.require_runtime_audit and not audits:
        errors.append("no runtime audit JSONs found")
    if errors:
        print("\nPipeline contract validation failed:")
        for error in errors:
            print(f"- {error}")
        result = StageResult("validate", [], 1, time.perf_counter() - started, started_at, datetime.now(timezone.utc).isoformat())
        getattr(args, "_stage_results").append(result)
        if args.stop_on_error:
            write_state(root, args, getattr(args, "_stage_results"))
            raise SystemExit(1)
        return [result]
    print(f"PASS {CONTRACT.schema_version} ({CONTRACT.family}/{CONTRACT.mode})")
    print(f"Runtime audits discovered: {len(audits)}")
    return []


def ingest(root: Path, args: argparse.Namespace) -> list[StageResult]:
    audits = expand_runtime_audit_paths(root, args.runtime_audit)
    if args.require_runtime_audit and not audits:
        raise SystemExit("no runtime audit JSONs found; pass --runtime-audit or export app runtime audits")
    command: list[str | Path] = [
        args.python,
        "-m",
        "lumen_manifest_crawler",
        "improve-loop",
        "--root",
        root,
        "--output",
        MANIFEST_OUTPUT_DIR,
        "--loop-output",
        LOOP_OUTPUT_DIR,
        "--fine-tuning-output",
        FINE_TUNING_OUTPUT_DIR,
        "--testflight-scenario-limit",
        "120",
        "--app-run-mode",
        "testflight",
        "--strict",
        "--deterministic",
        "--pretty",
        "--generate-system-prompts",
        "--generate-agent-fine-tuning",
    ]
    for audit in audits:
        command.extend(["--runtime-audit", audit])
    return [run_stage(root, args, "ingest audits + crawl code + generate datasets", command)]


def train_adapters(root: Path, args: argparse.Namespace) -> list[StageResult]:
    results: list[StageResult] = []
    for agent in parse_agents(args.agents):
        command: list[str | Path] = [
            args.train_python,
            "tools/fine_tuning/unsloth/train_sft.py",
            "--config",
            f"tools/fine_tuning/unsloth/configs_qwen3_bootstrap/{agent}.json",
            "--seed",
            str(args.seed),
        ]
        if args.assistant_only_loss:
            command.append("--assistant-only-loss")
        results.append(run_stage(root, args, f"train {agent} adapter", command))
    return results


def convert_adapters(root: Path, args: argparse.Namespace) -> list[StageResult]:
    out_dir = root / ADAPTER_GGUF_DIR
    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
    results: list[StageResult] = []
    for agent in parse_agents(args.agents):
        source = trained_adapter_path(agent)
        outfile = ADAPTER_GGUF_DIR / adapter_file_name(agent)
        command: list[str | Path] = [
            args.train_python,
            args.converter,
            source,
            "--outfile",
            outfile,
            "--base-model-id",
            args.base_model_id,
        ]
        results.append(run_stage(root, args, f"convert {agent} adapter to GGUF", command))
    return results


def upload_adapters(root: Path, args: argparse.Namespace) -> list[StageResult]:
    adapter_dir = root / ADAPTER_GGUF_DIR
    if not adapter_dir.is_dir():
        raise SystemExit(f"Adapter GGUF directory '{ADAPTER_GGUF_DIR}' does not exist. Run convert-adapters first.")
    if not any(adapter_dir.glob("*.gguf")):
        raise SystemExit(f"Adapter GGUF directory '{ADAPTER_GGUF_DIR}' contains no .gguf files. Run convert-adapters first.")
    create_cmd: list[str | Path] = ["hf", "repos", "create", ADAPTER_REPO_ID, "--type", "model", "--exist-ok"]
    if args.hf_private:
        create_cmd.append("--private")
    return [
        run_stage(root, args, "ensure adapter HF repo", create_cmd),
        run_stage(root, args, "upload adapter GGUFs", ["hf", "upload", ADAPTER_REPO_ID, ADAPTER_GGUF_DIR, ".", "--repo-type", "model"]),
    ]


def upload_base(root: Path, args: argparse.Namespace) -> list[StageResult]:
    create_cmd: list[str | Path] = ["hf", "repos", "create", SHARED_BASE_REPO_ID, "--type", "model", "--exist-ok"]
    if args.hf_private:
        create_cmd.append("--private")
    shared_base_dir = root / SHARED_BASE_GGUF_DIR
    if args.large_folder_upload:
        if not shared_base_dir.is_dir():
            raise SystemExit(f"shared base GGUF not found; run the base export pipeline first (expected directory: {SHARED_BASE_GGUF_DIR})")
        upload_cmd: list[str | Path] = ["hf", "upload-large-folder", SHARED_BASE_REPO_ID, SHARED_BASE_GGUF_DIR, "--repo-type", "model"]
    else:
        shared_base_file = shared_base_dir / SHARED_BASE_FILE_NAME
        if not shared_base_file.is_file():
            raise SystemExit(f"shared base GGUF not found; run the base export pipeline first (expected file: {SHARED_BASE_GGUF_DIR / SHARED_BASE_FILE_NAME})")
        upload_cmd = ["hf", "upload", SHARED_BASE_REPO_ID, SHARED_BASE_GGUF_DIR / SHARED_BASE_FILE_NAME, SHARED_BASE_FILE_NAME, "--repo-type", "model"]
    return [
        run_stage(root, args, "ensure shared base HF repo", create_cmd),
        run_stage(root, args, "upload shared base GGUF", upload_cmd),
    ]


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args._stage_results = []
    root = args.root.resolve()
    if not (root / "ios/Lumen").exists():
        raise SystemExit(f"root does not look like lumen-clone: {root}")
    validate_tool_paths(root, args)
    os.chdir(root)

    pipelines: dict[str, list[StageFn]] = {
        "validate": [validate],
        "ingest": [validate, ingest],
        "train-adapters": [validate, train_adapters],
        "convert-adapters": [validate, convert_adapters],
        "upload-adapters": [validate, upload_adapters],
        "upload-base": [validate, upload_base],
        "full": [validate, ingest, train_adapters, convert_adapters, upload_adapters],
    }
    history: list[StageResult] = getattr(args, "_stage_results")
    for stage_fn in pipelines[args.mode]:
        stage_fn(root, args)

    write_state(root, args, history)
    if history:
        print("\nCompleted stages:")
        for result in history:
            print(f"- {result.stage}: rc={result.returncode} elapsed={result.elapsed_s:.1f}s")
    return 1 if any(result.returncode != 0 for result in history) else 0


if __name__ == "__main__":
    raise SystemExit(main())
