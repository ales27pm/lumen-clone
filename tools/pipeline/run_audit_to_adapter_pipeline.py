#!/usr/bin/env python3
"""Contract-driven runner for the Lumen audit -> adapter -> app pipeline.

This is a thin, explicit orchestrator. It does not replace the lower-level tools;
it wires them together using `audit_to_adapter_contract.py` so the app catalog,
training configs, upload repos, and runtime expectations stay aligned.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

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
    runtime_audit_candidates,
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


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Lumen's audit-to-adapter pipeline.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
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
    parser.add_argument("--python", default=sys.executable, help="Python interpreter for crawler/helper scripts.")
    parser.add_argument("--train-python", default=os.environ.get("LUMEN_TRAIN_PYTHON", sys.executable), help="Python interpreter for Unsloth training/conversion.")
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


def rel(root: Path, path: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def expand_runtime_audits(root: Path, explicit: Iterable[str]) -> list[Path]:
    if not explicit:
        return runtime_audit_candidates(root)
    found: list[Path] = []
    for raw in explicit:
        pattern = str(resolve(root, raw))
        matches = [Path(path).resolve() for path in glob.glob(pattern)]
        if not matches and Path(pattern).exists():
            matches = [Path(pattern).resolve()]
        found.extend(path for path in matches if path.is_file())
    return sorted({path for path in found})


def run_stage(root: Path, args: argparse.Namespace, stage: str, command: Sequence[str | Path]) -> StageResult:
    printable = [str(part) for part in command]
    print(f"\n== {stage}")
    print(shlex.join(printable))
    started_at = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    if args.dry_run:
        finished_at = datetime.now(timezone.utc).isoformat()
        return StageResult(stage, printable, 0, 0.0, started_at, finished_at)
    env = os.environ.copy()
    crawler = str((root / "tools/lumen_manifest_crawler").resolve())
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = crawler if not existing_pythonpath else f"{crawler}{os.pathsep}{existing_pythonpath}"
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("PYTHONHASHSEED", str(args.seed))
    env.setdefault("LUMEN_TRAIN_SEED", str(args.seed))
    completed = subprocess.run(printable, cwd=root, env=env, check=False)
    elapsed = time.perf_counter() - started
    finished_at = datetime.now(timezone.utc).isoformat()
    result = StageResult(stage, printable, completed.returncode, elapsed, started_at, finished_at)
    if completed.returncode != 0 and args.stop_on_error:
        write_state(root, args, [result])
        raise SystemExit(completed.returncode)
    return result


def write_state(root: Path, args: argparse.Namespace, results: Sequence[StageResult]) -> None:
    if args.dry_run:
        return
    path = resolve(root, args.state_file)
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
    errors = validate_repository_alignment(root, require_generated_artifacts=args.require_generated_artifacts)
    audits = expand_runtime_audits(root, args.runtime_audit)
    if args.require_runtime_audit and not audits:
        errors.append("no runtime audit JSONs found")
    if errors:
        print("\nPipeline contract validation failed:")
        for error in errors:
            print(f"- {error}")
        if args.stop_on_error:
            raise SystemExit(1)
    else:
        print(f"PASS {CONTRACT.schema_version} ({CONTRACT.family}/{CONTRACT.mode})")
        print(f"Runtime audits discovered: {len(audits)}")
    return []


def ingest(root: Path, args: argparse.Namespace) -> list[StageResult]:
    audits = expand_runtime_audits(root, args.runtime_audit)
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
    if not args.converter.exists() and not args.dry_run:
        raise SystemExit(f"missing converter: {args.converter}")
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
    if args.large_folder_upload:
        upload_cmd: list[str | Path] = ["hf", "upload-large-folder", SHARED_BASE_REPO_ID, SHARED_BASE_GGUF_DIR, "--repo-type", "model"]
    else:
        upload_cmd = ["hf", "upload", SHARED_BASE_REPO_ID, SHARED_BASE_GGUF_DIR / SHARED_BASE_FILE_NAME, SHARED_BASE_FILE_NAME, "--repo-type", "model"]
    return [
        run_stage(root, args, "ensure shared base HF repo", create_cmd),
        run_stage(root, args, "upload shared base GGUF", upload_cmd),
    ]


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.root.resolve()
    if not (root / "ios/Lumen").exists():
        raise SystemExit(f"root does not look like lumen-clone: {root}")
    os.chdir(root)

    results: list[StageResult] = []
    if args.mode == "validate":
        results.extend(validate(root, args))
    elif args.mode == "ingest":
        validate(root, args)
        results.extend(ingest(root, args))
    elif args.mode == "train-adapters":
        validate(root, args)
        results.extend(train_adapters(root, args))
    elif args.mode == "convert-adapters":
        validate(root, args)
        results.extend(convert_adapters(root, args))
    elif args.mode == "upload-adapters":
        validate(root, args)
        results.extend(upload_adapters(root, args))
    elif args.mode == "upload-base":
        validate(root, args)
        results.extend(upload_base(root, args))
    elif args.mode == "full":
        validate(root, args)
        results.extend(ingest(root, args))
        results.extend(train_adapters(root, args))
        results.extend(convert_adapters(root, args))
        results.extend(upload_adapters(root, args))
    else:
        raise SystemExit(f"unsupported mode: {args.mode}")

    write_state(root, args, results)
    if results:
        print("\nCompleted stages:")
        for result in results:
            print(f"- {result.stage}: rc={result.returncode} elapsed={result.elapsed_s:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
