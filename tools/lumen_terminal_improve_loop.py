#!/usr/bin/env python3
"""Terminal-only AIO improve-loop launcher for Lumen.

This is the local, menu-driven replacement for the browser/HTML control panel.
It performs the complete adapter-first workflow from the terminal:

1. preflight checks (drift guard + Qwen3 config strictness);
2. code crawl + manifest/dataset generation;
3. in-app JSON ingestion;
4. embedding-state augmentation;
5. adapter-first release manifest generation;
6. Qwen3 role-adapter training;
7. LoRA-to-GGUF adapter conversion (with explicit base validation);
8. optional Hugging Face uploads (via `hf repos create` / `hf upload`).

The script runs fixed repo-rooted commands only. It never accepts arbitrary shell
commands from a web page or text field.

It is also resumable: each stage is recorded in a pipeline_state.json with input
hashes, command argv, status, output paths, timestamps, and a wall clock. With
``--resume`` the script will skip stages whose input hashes have not changed
and whose previous run succeeded.
"""
# pylint: disable=too-many-lines,missing-class-docstring,missing-function-docstring,line-too-long
# cspell:words Qwen qwen QWEN GGUF GGUFs gguf coreml mlmodel toolcalls scenarioresults adapterapplied jsons PYTHONUNBUFFERED PYTHONHASHSEED

import argparse
import glob
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

AGENTS = ("cortex", "executor", "mouth", "mimicry", "rem", "fleet")

DEFAULT_OUTPUT = Path("generated/agent_manifest")
DEFAULT_LOOP_OUTPUT = Path("generated/agent_improvement_loop")
DEFAULT_FINE_TUNING_OUTPUT = Path("generated/fine_tuning")
DEFAULT_QWEN3_CONFIG_DIR = Path("tools/fine_tuning/unsloth/configs_qwen3_bootstrap")
DEFAULT_LORA_DIR = Path("models/lora_qwen3_bootstrap")
DEFAULT_LORA_GGUF_DIR = Path("models/lora_qwen3_gguf")
DEFAULT_RELEASE_OUTPUT_ROOT = Path("models/gguf_release_bake_qwen3_bootstrap")
DEFAULT_RELEASE_MANIFEST = Path("generated/fine_tuning/qwen3_bootstrap_release_bake_gguf_manifest.json")
DEFAULT_ADAPTER_REPO = "ales27pm/lumen-qwen3-bootstrap-adapters-gguf"
DEFAULT_BASE_REPO = "ales27pm/lumen-qwen3-bootstrap-gguf"
DEFAULT_BASE_FILE_NAME = "lumen-qwen3-fast-shared-q4_k_m.gguf"
DEFAULT_BASE_FILE = Path("models/base_qwen3_fast") / DEFAULT_BASE_FILE_NAME
DEFAULT_STATE_FILE = Path("generated/agent_improvement_loop/pipeline_state.json")
DEFAULT_BASE_MODEL_ID = "Qwen/Qwen3-1.7B"
ADAPTER_RUNTIME_INVARIANTS_SCRIPT = Path("tools/check_adapter_runtime_invariants.py")
LOOP_STATE_FILE_NAME = "loop_state.json"

# Substrings that indicate a config still points at the pre-Qwen3 family. The
# Qwen3 bootstrap config dir must NEVER reference Qwen2.x bases.
FORBIDDEN_BASE_TOKENS = ("qwen2", "qwen-2")
REQUIRED_BASE_TOKENS = ("qwen3",)

RUNTIME_INCLUDE_HINTS = (
    "runtime",
    "audit",
    "grounding",
    "agent-grounding",
    "agent_grounding",
    "live-e2e",
    "e2e",
    "testflight",
    "in-app",
    "in_app",
    "trace",
)
RUNTIME_EXCLUDE_HINTS = (
    "loop_state",
    "loop_gaps",
    "next_action_prompts",
    "testflight_scenarios",
    "release_bake_gguf_manifest",
    "adapter_runtime_manifest",
    "dataset_manifest",
    "visual_improve_loop_summary",
)
RUNTIME_JSON_KEY_HINTS = {
    "traces",
    "events",
    "failures",
    "runtime",
    "toolcalls",
    "tool_calls",
    "scenarioresults",
}


@dataclass
class StageRecord:
    name: str
    status: str  # "pending" | "ok" | "fail" | "skipped"
    returncode: int = 0
    elapsed_s: float = 0.0
    started_at: str = ""
    finished_at: str = ""
    argv: list[str] = field(default_factory=list)
    inputs: list[str] = field(default_factory=list)
    input_hash: str = ""
    outputs: list[str] = field(default_factory=list)
    note: str = ""


@dataclass
class RunResult:
    name: str
    returncode: int
    elapsed: float


class Terminal:
    def __init__(self, quiet: bool = False) -> None:
        self.quiet = quiet

    def print(self, text: str = "") -> None:
        if not self.quiet:
            print(text)

    def title(self, text: str) -> None:
        self.print("\n" + "=" * 88)
        self.print(text)
        self.print("=" * 88)

    def info(self, text: str) -> None:
        self.print(f"• {text}")

    def ok(self, text: str) -> None:
        self.print(f"✓ {text}")

    def warn(self, text: str) -> None:
        self.print(f"⚠ {text}")

    def fail(self, text: str) -> None:
        self.print(f"✗ {text}")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else root / path


def rel(root: Path, path: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def parse_agents(raw: str) -> list[str]:
    agents = [x.strip().lower() for x in raw.split(",") if x.strip()]
    invalid = [a for a in agents if a not in AGENTS]
    if invalid:
        raise SystemExit(f"Invalid agent(s): {', '.join(invalid)}. Valid: {', '.join(AGENTS)}")
    return agents or list(AGENTS)


def env(root: Path, args: argparse.Namespace) -> dict[str, str]:
    out = os.environ.copy()
    crawler = str((root / "tools/lumen_manifest_crawler").resolve())
    existing = out.get("PYTHONPATH")
    out["PYTHONPATH"] = crawler if not existing else f"{crawler}{os.pathsep}{existing}"
    out.setdefault("PYTHONUNBUFFERED", "1")
    if getattr(args, "seed", None) is not None:
        out.setdefault("PYTHONHASHSEED", str(int(args.seed)))
        out.setdefault("LUMEN_TRAIN_SEED", str(int(args.seed)))
    return out


def training_python() -> str:
    configured = os.environ.get("LUMEN_TRAIN_PYTHON", "").strip()
    if configured:
        return configured
    local_unsloth = Path.cwd() / ".venv-unsloth" / "bin" / "python"
    if local_unsloth.exists():
        return str(local_unsloth)
    if sys.version_info >= (3, 10):
        return sys.executable
    py310 = shutil.which("python3.10")
    return py310 or sys.executable


# ---------------------------------------------------------------------------
# Pipeline state (resumable)
# ---------------------------------------------------------------------------


def _hash_paths(root: Path, paths: Iterable[Path]) -> str:
    hasher = hashlib.sha256()
    seen: list[Path] = []
    for path in paths:
        try:
            resolved = path.resolve()
        except (OSError, RuntimeError):
            continue
        if not resolved.exists():
            continue
        seen.append(resolved)
    for path in sorted(seen):
        try:
            stat = path.stat()
        except OSError:
            continue
        hasher.update(rel(root, path).encode("utf-8"))
        hasher.update(b"|")
        hasher.update(str(stat.st_size).encode("utf-8"))
        hasher.update(b"|")
        hasher.update(str(int(stat.st_mtime_ns)).encode("utf-8"))
        hasher.update(b"\n")
    return hasher.hexdigest()


class PipelineState:
    def __init__(self, root: Path, path: Path, *, resume: bool, dry_run: bool) -> None:
        self.root = root
        self.path = path
        self.resume = resume
        self.dry_run = dry_run
        self.records: dict[str, StageRecord] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        for name, raw in (payload.get("stages") or {}).items():
            if not isinstance(raw, dict):
                continue
            self.records[name] = StageRecord(
                name=name,
                status=str(raw.get("status", "pending")),
                returncode=int(raw.get("returncode", 0)),
                elapsed_s=float(raw.get("elapsed_s", 0.0)),
                started_at=str(raw.get("started_at", "")),
                finished_at=str(raw.get("finished_at", "")),
                argv=list(raw.get("argv", []) or []),
                inputs=list(raw.get("inputs", []) or []),
                input_hash=str(raw.get("input_hash", "")),
                outputs=list(raw.get("outputs", []) or []),
                note=str(raw.get("note", "")),
            )

    def can_skip(self, name: str, input_hash: str) -> bool:
        if not self.resume:
            return False
        record = self.records.get(name)
        if not record or record.status != "ok":
            return False
        return bool(input_hash) and record.input_hash == input_hash

    def write(self) -> None:
        if self.dry_run:
            return
        payload = {
            "schema": "lumen.improve_loop.pipeline_state/1.0.0",
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "stages": {
                name: {
                    "status": rec.status,
                    "returncode": rec.returncode,
                    "elapsed_s": round(rec.elapsed_s, 3),
                    "started_at": rec.started_at,
                    "finished_at": rec.finished_at,
                    "argv": rec.argv,
                    "inputs": rec.inputs,
                    "input_hash": rec.input_hash,
                    "outputs": rec.outputs,
                    "note": rec.note,
                }
                for name, rec in self.records.items()
            },
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def _record_stage(
    state: PipelineState | None,
    *,
    root: Path,
    name: str,
    stage_status: str,
    argv: Sequence[str],
    input_paths: Sequence[Path],
    input_hash: str,
    output_paths: Sequence[Path],
    returncode: int = 0,
    elapsed_s: float = 0.0,
    started_at: str = "",
    finished_at: str = "",
    note: str = "",
) -> None:
    if state is None:
        return
    state.records[name] = StageRecord(
        name=name,
        status=stage_status,
        returncode=returncode,
        elapsed_s=elapsed_s,
        started_at=started_at,
        finished_at=finished_at,
        argv=list(argv),
        inputs=[rel(root, p) for p in input_paths],
        input_hash=input_hash,
        outputs=[rel(root, p) for p in output_paths],
        note=note,
    )
    state.write()


def _stream_output(process: subprocess.Popen[str], quiet_commands: bool) -> None:
    assert process.stdout is not None
    for line in process.stdout:
        if not quiet_commands:
            print(line.rstrip())


def run(
    term: Terminal,
    root: Path,
    name: str,
    argv: Sequence[str | Path],
    *,
    args: argparse.Namespace,
    state: PipelineState | None = None,
    inputs: Sequence[Path] | None = None,
    outputs: Sequence[Path] | None = None,
) -> RunResult:
    printable = [str(x) for x in argv]
    term.title(name)
    term.info(shlex.join(printable))

    input_paths = list(inputs or [])
    input_hash = _hash_paths(root, input_paths) if input_paths else ""

    if state is not None and state.can_skip(name, input_hash):
        prior = state.records[name]
        term.ok(f"resume: skipping {name} (inputs unchanged, prior {prior.elapsed_s:.1f}s)")
        return RunResult(name, 0, 0.0)

    started = time.perf_counter()
    started_iso = datetime.now(timezone.utc).isoformat()

    if args.dry_run:
        term.warn("dry-run: command not executed")
        _record_stage(
            state,
            root=root,
            name=name,
            stage_status="skipped",
            argv=printable,
            input_paths=input_paths,
            input_hash=input_hash,
            output_paths=list(outputs or []),
            started_at=started_iso,
            finished_at=started_iso,
            note="dry-run",
        )
        return RunResult(name, 0, 0.0)

    process = subprocess.Popen(
        printable,
        cwd=root,
        env=env(root, args),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        errors="replace",
    )
    _stream_output(process, args.quiet_commands)
    code = process.wait()
    elapsed = time.perf_counter() - started
    finished_iso = datetime.now(timezone.utc).isoformat()
    if code == 0:
        term.ok(f"{name} passed in {elapsed:.1f}s")
    else:
        term.fail(f"{name} failed with {code} after {elapsed:.1f}s")

    _record_stage(
        state,
        root=root,
        name=name,
        stage_status="ok" if code == 0 else "fail",
        returncode=code,
        elapsed_s=elapsed,
        started_at=started_iso,
        finished_at=finished_iso,
        argv=printable,
        input_paths=input_paths,
        input_hash=input_hash,
        output_paths=list(outputs or []),
    )

    if code != 0 and args.stop_on_error:
        raise SystemExit(code)
    return RunResult(name, code, elapsed)


# ---------------------------------------------------------------------------
# Runtime JSON discovery
# ---------------------------------------------------------------------------


def runtime_json_candidate(path: Path) -> bool:
    name = path.name.lower()
    if not path.is_file() or path.suffix.lower() != ".json":
        return False
    if any(token in name for token in RUNTIME_EXCLUDE_HINTS):
        return False
    if not any(token in name for token in RUNTIME_INCLUDE_HINTS):
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(payload, dict):
        return False
    lowered_keys = {str(k).lower() for k in payload.keys()}
    if lowered_keys.intersection(RUNTIME_JSON_KEY_HINTS):
        return True
    sample = json.dumps(payload, ensure_ascii=False)[:30000].lower()
    return "adapterapplied" in sample or "agent grounding" in sample or "runtime" in sample or "trace" in sample


def discover_runtime_jsons(root: Path, explicit: Iterable[str]) -> list[Path]:
    patterns = list(explicit) or [
        "exports/*.json",
        "generated/runtime_audits/*.json",
        "generated/runtime_audit/*.json",
        "generated/testflight_exports/*.json",
        "generated/agent_improvement_loop/runtime_audits/*.json",
    ]
    found: list[Path] = []
    for pattern in patterns:
        absolute_pattern = str(resolve(root, pattern))
        for raw in glob.glob(absolute_pattern):
            path = Path(raw).resolve()
            if runtime_json_candidate(path):
                found.append(path)
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in sorted(found):
        if path not in seen:
            seen.add(path)
            unique.append(path)
    return unique


def print_runtime_jsons(term: Terminal, root: Path, paths: Sequence[Path]) -> None:
    if not paths:
        term.warn("no in-app JSON exports found; the loop will create TestFlight handoff artifacts")
        return
    term.ok(f"in-app JSON exports discovered: {len(paths)}")
    for path in paths:
        term.info(rel(root, path))


def _audit_has_adapter_applied(audit_path: Path) -> bool:
    try:
        blob = audit_path.read_text(encoding="utf-8", errors="replace").lower()
    except OSError:
        return False
    return "adapterapplied" in blob


# ---------------------------------------------------------------------------
# Strict Qwen3 config validation
# ---------------------------------------------------------------------------


def _validate_qwen3_agent_config(root: Path, cfg_path: Path, agent: str) -> list[str]:
    errors: list[str] = []
    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"invalid JSON in {rel(root, cfg_path)}: {exc}"]

    base = str(cfg.get("base_model_name", "")).lower()
    if not base:
        return [f"{rel(root, cfg_path)}: base_model_name is empty"]

    if any(token in base for token in FORBIDDEN_BASE_TOKENS):
        errors.append(
            f"{rel(root, cfg_path)}: base_model_name '{cfg.get('base_model_name')}' "
            "still points at a pre-Qwen3 family in the Qwen3 bootstrap config dir."
        )
    if not any(token in base for token in REQUIRED_BASE_TOKENS):
        errors.append(
            f"{rel(root, cfg_path)}: base_model_name '{cfg.get('base_model_name')}' "
            "must reference Qwen3 in the Qwen3 bootstrap config dir."
        )
    if cfg.get("agent", "").strip().lower() != agent:
        errors.append(f"{rel(root, cfg_path)}: agent field must equal '{agent}'")
    if cfg.get("merge_adapters_by_default", False):
        errors.append(f"{rel(root, cfg_path)}: merge_adapters_by_default must be false (adapter-first).")
    if cfg.get("release_bake_enabled_by_default", False):
        errors.append(
            f"{rel(root, cfg_path)}: release_bake_enabled_by_default must be false (adapter-first)."
        )
    return errors


def validate_qwen3_configs(root: Path, args: argparse.Namespace) -> list[str]:
    """Return a list of human-readable validation errors. Empty list = OK."""
    errors: list[str] = []
    cfg_dir = resolve(root, args.config_dir)
    if not cfg_dir.exists():
        errors.append(f"missing Qwen3 bootstrap config dir: {rel(root, cfg_dir)}")
        return errors

    for agent in parse_agents(args.agents):
        cfg_path = cfg_dir / f"{agent}.json"
        if not cfg_path.exists():
            errors.append(f"missing Qwen3 bootstrap config: {rel(root, cfg_path)}")
            continue
        errors.extend(_validate_qwen3_agent_config(root, cfg_path, agent))

    return errors


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------


def preflight(term: Terminal, root: Path, args: argparse.Namespace, state: PipelineState | None = None) -> list[RunResult]:
    term.title("Preflight")
    required = [
        root / "tools/lumen_manifest_crawler/lumen_manifest_crawler/cli.py",
        root / "tools/lumen_manifest_crawler/lumen_manifest_crawler/improvement_loop.py",
        root / "tools/fine_tuning/unsloth/train_sft.py",
        root / "tools/fine_tuning/unsloth/export_gguf.py",
        root / ADAPTER_RUNTIME_INVARIANTS_SCRIPT,
        root / "docs/ADAPTER_RUNTIME_IMPROVE_LOOP.md",
        root / "ios/Lumen",
    ]
    missing = [p for p in required if not p.exists()]
    if missing:
        for path in missing:
            term.fail(f"missing {rel(root, path)}")
        if args.stop_on_error:
            raise SystemExit(2)
    else:
        term.ok("required files are present")

    config_errors = validate_qwen3_configs(root, args)
    if config_errors:
        for err in config_errors:
            term.fail(err)
        if args.fail_if_missing_qwen3_config or args.stop_on_error:
            raise SystemExit(2)
    else:
        term.ok("Qwen3 bootstrap configs validated (all reference a Qwen3 base, adapter-first)")

    results = [
        run(
            term,
            root,
            "adapter runtime drift guard",
            [sys.executable, str(ADAPTER_RUNTIME_INVARIANTS_SCRIPT)],
            args=args,
            state=state,
            inputs=[
                root / ADAPTER_RUNTIME_INVARIANTS_SCRIPT,
                root / "tools/lumen_terminal_improve_loop.py",
            ],
        )
    ]
    if not args.skip_pytest:
        results.append(
            run(
                term,
                root,
                "manifest crawler tests",
                [sys.executable, "-m", "pytest", "tools/lumen_manifest_crawler/tests"],
                args=args,
                state=state,
                inputs=[root / "tools/lumen_manifest_crawler"],
            )
        )
    return results


def crawl_ingest_generate(term: Terminal, root: Path, args: argparse.Namespace, state: PipelineState | None = None) -> list[RunResult]:
    audits = discover_runtime_jsons(root, args.runtime_audit)
    print_runtime_jsons(term, root, audits)

    if args.require_adapter_traces:
        has_trace = any(_audit_has_adapter_applied(audit) for audit in audits)
        if not has_trace:
            term.fail(
                "--require-adapter-traces: no in-app audit contains 'adapterApplied' evidence."
            )
            if args.stop_on_error or args.fail_if_missing_qwen3_config:
                raise SystemExit(2)

    loop_state_path = resolve(root, args.loop_output) / LOOP_STATE_FILE_NAME

    improve = [
        sys.executable,
        "-m",
        "lumen_manifest_crawler",
        "improve-loop",
        "--root",
        str(root),
        "--output",
        str(resolve(root, args.output)),
        "--loop-output",
        str(resolve(root, args.loop_output)),
        "--fine-tuning-output",
        str(resolve(root, args.fine_tuning_output)),
        "--testflight-scenario-limit",
        str(args.testflight_scenario_limit),
        "--app-run-mode",
        args.app_run_mode,
        "--strict",
        "--deterministic",
        "--pretty",
        "--generate-system-prompts",
        "--generate-agent-fine-tuning",
    ]
    if args.require_runtime_audit:
        improve.append("--require-testflight-runtime-audit")
    for audit in audits:
        improve.extend(["--runtime-audit", str(audit)])

    results = [
        run(
            term,
            root,
            "crawl code + ingest JSON + generate datasets",
            improve,
            args=args,
            state=state,
            inputs=[root / "ios/Lumen", *audits],
            outputs=[
                resolve(root, args.output) / "dataset_manifest.json",
                loop_state_path,
            ],
        )
    ]

    augment = [
        sys.executable,
        "tools/augment_loop_state_embedding.py",
        "--loop-state",
        str(loop_state_path),
        "--embedding-dir",
        str(resolve(root, args.output) / "embedding"),
        "--print-summary",
    ]
    results.append(
        run(
            term,
            root,
            "augment loop summary with embedding state",
            augment,
            args=args,
            state=state,
            inputs=[
                loop_state_path,
                resolve(root, args.output) / "embedding",
            ],
        )
    )

    export = [
        sys.executable,
        "tools/fine_tuning/unsloth/export_gguf.py",
        "--config-dir",
        str(resolve(root, args.config_dir)),
        "--agents",
        args.agents,
        "--quantization",
        args.quantization,
        "--output-root",
        str(resolve(root, args.release_output_root)),
        "--manifest-output",
        str(resolve(root, args.release_manifest)),
        "--skip-upload",
    ]
    if args.release_bake:
        export.append("--release-bake")
    results.append(
        run(
            term,
            root,
            "adapter-first manifest / optional release bake",
            export,
            args=args,
            state=state,
            inputs=[resolve(root, args.config_dir)],
            outputs=[resolve(root, args.release_manifest)],
        )
    )
    return results


def _train_intent_classifier(
    term: Terminal,
    root: Path,
    args: argparse.Namespace,
    train_py: Path,
    state: PipelineState | None,
) -> list[RunResult]:
    dataset_path = resolve(root, args.intent_dataset_output)
    model_path = resolve(root, args.intent_model_output)
    coreml_path = resolve(root, args.intent_coreml_output)
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    coreml_path.parent.mkdir(parents=True, exist_ok=True)

    results = [
        run(
            term,
            root,
            "generate intent-classifier dataset",
            [
                train_py,
                "tools/intent_classifier/generate_intent_dataset.py",
                "--out",
                str(dataset_path),
                "--repeat",
                str(int(args.intent_dataset_repeat)),
            ],
            args=args,
            state=state,
            inputs=[
                root / "tools/intent_classifier/generate_intent_dataset.py",
                root / "tools/intent_classifier/intents.schema.json",
            ],
            outputs=[dataset_path],
        ),
        run(
            term,
            root,
            "train intent-classifier model",
            [
                train_py,
                "tools/intent_classifier/train_intent_classifier.py",
                "--dataset",
                str(dataset_path),
                "--model-out",
                str(model_path),
            ],
            args=args,
            state=state,
            inputs=[
                dataset_path,
                root / "tools/intent_classifier/train_intent_classifier.py",
            ],
            outputs=[model_path],
        ),
    ]
    if args.skip_intent_coreml_export:
        return results

    results.append(
        run(
            term,
            root,
            "export intent-classifier CoreML model",
            [
                train_py,
                "tools/intent_classifier/export_coreml_intent_classifier.py",
                "--model",
                str(model_path),
                "--out",
                str(coreml_path),
            ],
            args=args,
            state=state,
            inputs=[
                model_path,
                root / "tools/intent_classifier/export_coreml_intent_classifier.py",
            ],
            outputs=[coreml_path],
        )
    )
    return results


def train(
    term: Terminal,
    root: Path,
    args: argparse.Namespace,
    state: PipelineState | None = None,
) -> list[RunResult]:
    cfg_dir = resolve(root, args.config_dir)
    if not cfg_dir.exists():
        raise SystemExit(f"Missing config dir: {cfg_dir}")
    config_errors = validate_qwen3_configs(root, args)
    if config_errors:
        for err in config_errors:
            term.fail(err)
        raise SystemExit(2)
    train_py = training_python()
    term.info(f"training interpreter: {train_py}")

    results: list[RunResult] = []
    for agent in parse_agents(args.agents):
        cfg = cfg_dir / f"{agent}.json"
        if not cfg.exists():
            raise SystemExit(f"Missing config: {cfg}")
        argv: list[str | Path] = [
            train_py,
            "tools/fine_tuning/unsloth/train_sft.py",
            "--config",
            str(cfg),
        ]
        if args.seed is not None:
            argv.extend(["--seed", str(int(args.seed))])
        if args.resume:
            argv.append("--resume-from-checkpoint")
        if args.assistant_only_loss:
            argv.append("--assistant-only-loss")
        results.append(
            run(
                term,
                root,
                f"train {agent} adapter",
                argv,
                args=args,
                state=state,
                inputs=[cfg, resolve(root, args.fine_tuning_output) / agent],
                outputs=[resolve(root, args.lora_dir) / agent],
            )
        )
    if args.train_intent_classifier:
        results.extend(_train_intent_classifier(term, root, args, train_py, state))
    return results


def converter(root: Path, args: argparse.Namespace) -> Path:
    if args.converter:
        return resolve(root, args.converter)
    return Path.home() / ".unsloth/llama.cpp/convert_lora_to_gguf.py"


def _resolve_base_for_convert(args: argparse.Namespace) -> tuple[list[str], str]:
    """Return (extra argv tokens, human description) for base model selection.

    The official llama.cpp `convert_lora_to_gguf.py` requires either a `--base`
    pointing at a directory with a `config.json` of the base model, or a
    `--base-model-id` Hugging Face repo id. Without one of these, conversion
    silently falls back to whatever `adapter_config.json` references, which
    breaks reproducibility and leaks network access.
    """
    if args.base_model_dir:
        return ["--base", str(Path(args.base_model_dir).expanduser())], f"--base {args.base_model_dir}"
    if args.base_model_id:
        return ["--base-model-id", str(args.base_model_id)], f"--base-model-id {args.base_model_id}"
    raise SystemExit(
        "convert: either --base-model-dir or --base-model-id is required to make "
        "LoRA→GGUF conversion reproducible. Use --base-model-id Qwen/Qwen3-1.7B by default."
    )


def convert(term: Terminal, root: Path, args: argparse.Namespace, state: PipelineState | None = None) -> list[RunResult]:
    script = converter(root, args)
    if not script.exists() and not args.dry_run:
        raise SystemExit(f"Missing converter: {script}")
    base_args, base_desc = _resolve_base_for_convert(args)
    term.info(f"convert base: {base_desc}")
    out_dir = resolve(root, args.lora_gguf_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results: list[RunResult] = []
    manifest_entries: list[dict[str, Any]] = []
    for agent in parse_agents(args.agents):
        source = resolve(root, args.lora_dir) / agent
        if not source.exists() and not args.dry_run:
            term.warn(f"missing trained adapter dir: {rel(root, source)}")
            if args.stop_on_error:
                raise SystemExit(2)
            continue
        outfile = out_dir / f"lumen-{agent}-lora.gguf"
        argv: list[str | Path] = [training_python(), str(script), str(source), "--outfile", str(outfile)]
        argv.extend(base_args)
        result = run(
            term,
            root,
            f"convert {agent} adapter to GGUF",
            argv,
            args=args,
            state=state,
            inputs=[source],
            outputs=[outfile],
        )
        results.append(result)
        manifest_entries.append(
            {
                "agent": agent,
                "source": rel(root, source),
                "outfile": rel(root, outfile),
                "base": base_desc,
                "returncode": result.returncode,
                "elapsed_s": round(result.elapsed, 3),
            }
        )

    if not args.dry_run and manifest_entries:
        manifest_path = out_dir / "convert_manifest.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "schema": "lumen.improve_loop.convert_manifest/1.0.0",
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "base": base_desc,
                    "entries": manifest_entries,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        term.ok(f"wrote convert manifest: {rel(root, manifest_path)}")
    return results


def upload_adapters(term: Terminal, root: Path, args: argparse.Namespace, state: PipelineState | None = None) -> list[RunResult]:
    if args.skip_upload:
        term.warn("skip-upload enabled")
        return []
    out_dir = resolve(root, args.lora_gguf_dir)
    if not out_dir.exists() and not args.dry_run:
        raise SystemExit(f"Missing adapter GGUF dir: {out_dir}")
    create = ["hf", "repos", "create", args.adapter_repo, "--type", "model", "--exist-ok"]
    if args.hf_private:
        create.append("--private")
    return [
        run(
            term,
            root,
            "ensure adapter HF repo",
            create,
            args=args,
            state=state,
            inputs=[],
        ),
        run(
            term,
            root,
            "upload adapter GGUFs",
            ["hf", "upload", args.adapter_repo, str(out_dir), ".", "--repo-type", "model"],
            args=args,
            state=state,
            inputs=[out_dir],
        ),
    ]


def upload_base(term: Terminal, root: Path, args: argparse.Namespace, state: PipelineState | None = None) -> list[RunResult]:
    if args.skip_upload:
        term.warn("skip-upload enabled")
        return []
    base = resolve(root, args.base_file)
    if not base.exists() and not args.dry_run:
        raise SystemExit(f"Missing shared base GGUF: {base}")
    create = ["hf", "repos", "create", args.base_repo, "--type", "model", "--exist-ok"]
    if args.hf_private:
        create.append("--private")
    upload_cmd = ["hf"]
    if args.large_folder_upload:
        upload_cmd += ["upload-large-folder", args.base_repo, str(base.parent), "--repo-type", "model"]
    else:
        upload_cmd += ["upload", args.base_repo, str(base), DEFAULT_BASE_FILE_NAME, "--repo-type", "model"]
    return [
        run(
            term,
            root,
            "ensure base HF repo",
            create,
            args=args,
            state=state,
            inputs=[],
        ),
        run(
            term,
            root,
            "upload shared Qwen3 base",
            upload_cmd,
            args=args,
            state=state,
            inputs=[base],
        ),
    ]


def status(term: Terminal, root: Path, args: argparse.Namespace) -> None:
    term.title("Latest generated state")
    paths = [
        resolve(root, args.loop_output) / "LOOP_REPORT.md",
        resolve(root, args.loop_output) / "loop_gaps.json",
        resolve(root, args.loop_output) / "next_action_prompts.jsonl",
        resolve(root, args.output) / "dataset_manifest.json",
        resolve(root, args.output) / "embedding" / "dataset_card.json",
        resolve(root, args.release_manifest),
        resolve(root, args.state_file),
    ]
    for path in paths:
        if not path.exists():
            term.warn(f"missing {rel(root, path)}")
            continue
        term.title(rel(root, path))
        text = path.read_text(encoding="utf-8", errors="replace")
        if path.suffix == ".json":
            try:
                text = json.dumps(json.loads(text), indent=2, ensure_ascii=False)
            except json.JSONDecodeError:
                pass
        print(text[:12000])


def full(term: Terminal, root: Path, args: argparse.Namespace, state: PipelineState | None = None) -> None:
    preflight(term, root, args, state)
    crawl_ingest_generate(term, root, args, state)
    train(term, root, args, state)
    convert(term, root, args, state)
    if args.upload_after_full:
        upload_adapters(term, root, args, state)
    status(term, root, args)


def menu(term: Terminal) -> str:
    term.title("Lumen terminal AIO improve-loop")
    print("1) Preflight: drift guard + tests + Qwen3 config check")
    print("2) Crawl code + ingest in-app JSONs + generate datasets")
    print("3) Train Qwen3 role adapters")
    print("4) Convert trained LoRA adapters to GGUF (with explicit base)")
    print("5) Upload adapter GGUFs to Hugging Face (hf repos create)")
    print("6) Upload shared Qwen3 base GGUF to Hugging Face")
    print("7) Full local cycle: preflight → crawl/ingest/datasets → train → convert → status")
    print("8) Full local cycle + upload adapters")
    print("9) Show latest report/gaps/dataset summaries + pipeline state")
    print("0) Exit")
    return input("\nSelect: ").strip().lower()


def interactive(term: Terminal, root: Path, args: argparse.Namespace, state: PipelineState | None) -> int:
    recoverable_menu_errors = (OSError, RuntimeError, ValueError, subprocess.SubprocessError)
    actions: dict[str, Callable[[], object]] = {
        "1": lambda: preflight(term, root, args, state),
        "2": lambda: crawl_ingest_generate(term, root, args, state),
        "3": lambda: train(term, root, args, state),
        "4": lambda: convert(term, root, args, state),
        "5": lambda: upload_adapters(term, root, args, state),
        "6": lambda: upload_base(term, root, args, state),
        "7": lambda: full(term, root, args, state),
        "8": lambda: full(term, root, argparse.Namespace(**{**vars(args), "upload_after_full": True}), state),
        "9": lambda: status(term, root, args),
    }
    while True:
        choice = menu(term)
        if choice in {"0", "q", "quit", "exit"}:
            return 0
        action = actions.get(choice)
        if not action:
            term.warn("unknown selection")
            continue
        try:
            action()
        except KeyboardInterrupt:
            term.warn("interrupted")
        except recoverable_menu_errors as exc:
            term.fail(str(exc))
            if args.stop_on_error:
                return 1
        input("\nPress Enter to return to menu...")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Terminal-only AIO improve-loop launcher for Lumen.")
    parser.add_argument("--root", type=Path, default=repo_root())
    parser.add_argument("--mode", choices=["menu", "preflight", "generate", "train", "convert", "upload-adapters", "upload-base", "full", "status"], default="menu")
    parser.add_argument("--agents", default=",".join(AGENTS))
    parser.add_argument("--runtime-audit", action="append", default=[], help="Runtime JSON glob/path. Can be repeated. Defaults to exports/*.json and generated export dirs.")
    parser.add_argument("--require-runtime-audit", action="store_true")
    parser.add_argument("--require-adapter-traces", action="store_true", help="Fail generate stage unless at least one in-app audit shows adapterApplied evidence.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--loop-output", type=Path, default=DEFAULT_LOOP_OUTPUT)
    parser.add_argument("--fine-tuning-output", type=Path, default=DEFAULT_FINE_TUNING_OUTPUT)
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_QWEN3_CONFIG_DIR)
    parser.add_argument("--lora-dir", type=Path, default=DEFAULT_LORA_DIR)
    parser.add_argument("--lora-gguf-dir", type=Path, default=DEFAULT_LORA_GGUF_DIR)
    parser.add_argument("--release-output-root", type=Path, default=DEFAULT_RELEASE_OUTPUT_ROOT)
    parser.add_argument("--release-manifest", type=Path, default=DEFAULT_RELEASE_MANIFEST)
    parser.add_argument("--quantization", default="q4_k_m")
    parser.add_argument("--release-bake", action="store_true", help="Manual only: produce role-baked GGUFs. Leave off for adapter-first runtime.")
    parser.add_argument("--converter", type=Path, default=None)
    parser.add_argument("--base-model-id", default=DEFAULT_BASE_MODEL_ID, help="Hugging Face repo id of the Qwen3 base used for LoRA→GGUF conversion (passed to convert_lora_to_gguf.py --base-model-id).")
    parser.add_argument("--base-model-dir", type=Path, default=None, help="Local directory containing the Qwen3 base model config.json for LoRA→GGUF conversion (passed to convert_lora_to_gguf.py --base). Overrides --base-model-id.")
    parser.add_argument("--adapter-repo", default=DEFAULT_ADAPTER_REPO)
    parser.add_argument("--base-repo", default=DEFAULT_BASE_REPO)
    parser.add_argument("--base-file", type=Path, default=DEFAULT_BASE_FILE)
    parser.add_argument("--hf-private", action="store_true")
    parser.add_argument("--large-folder-upload", action="store_true", help="Use 'hf upload-large-folder' for the shared base upload (resumable, recommended for >1GB).")
    parser.add_argument("--skip-upload", action="store_true")
    parser.add_argument("--upload-after-full", action="store_true")
    parser.add_argument("--app-run-mode", default="testflight")
    parser.add_argument("--testflight-scenario-limit", type=int, default=120)
    parser.add_argument("--skip-pytest", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--quiet-commands", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--fail-if-missing-qwen3-config", action="store_true", help="Hard-fail preflight if any Qwen3 bootstrap config is missing or still references a non-Qwen3 base.")
    parser.add_argument("--state-file", type=Path, default=DEFAULT_STATE_FILE, help="Resumable pipeline state file. Each stage records inputs/argv/outputs/status.")
    parser.add_argument("--resume", action="store_true", help="Skip stages whose recorded inputs are unchanged and previous run succeeded. Also passes --resume-from-checkpoint to train_sft.py.")
    parser.add_argument("--seed", type=int, default=None, help="Deterministic seed forwarded to train_sft.py (PYTHONHASHSEED, torch, numpy, transformers).")
    parser.add_argument("--assistant-only-loss", action="store_true", help="Train only on assistant turns (TRL assistant_only_loss).")
    parser.add_argument("--train-intent-classifier", action=argparse.BooleanOptionalAction, default=False, help="Also run lightweight intent-classifier dataset generation and sklearn training/export.")
    parser.add_argument("--intent-dataset-output", type=Path, default=Path("generated/intent_classifier/intent_dataset.jsonl"))
    parser.add_argument("--intent-model-output", type=Path, default=Path("generated/intent_classifier/intent_model.pkl"))
    parser.add_argument("--intent-coreml-output", type=Path, default=Path("generated/intent_classifier/IntentClassifier.mlmodel"))
    parser.add_argument("--intent-dataset-repeat", type=int, default=20, help="How many times to replicate curated seed prompts per intent.")
    parser.add_argument("--skip-intent-coreml-export", action="store_true", help="Skip CoreML export for intent classifier when training is enabled.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    root = resolve(Path.cwd(), args.root).resolve()
    os.chdir(root)
    term = Terminal(quiet=args.quiet)

    state_path = resolve(root, args.state_file)
    state = PipelineState(root, state_path, resume=args.resume, dry_run=args.dry_run)

    dispatch: dict[str, Callable[[], object]] = {
        "preflight": lambda: preflight(term, root, args, state),
        "generate": lambda: crawl_ingest_generate(term, root, args, state),
        "train": lambda: train(term, root, args, state),
        "convert": lambda: convert(term, root, args, state),
        "upload-adapters": lambda: upload_adapters(term, root, args, state),
        "upload-base": lambda: upload_base(term, root, args, state),
        "full": lambda: full(term, root, args, state),
        "status": lambda: status(term, root, args),
    }
    if args.mode == "menu":
        return interactive(term, root, args, state)
    dispatch[args.mode]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
