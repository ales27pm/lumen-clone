#!/usr/bin/env python3
"""Fully automated visual runner for the Lumen improvement loop.

This script orchestrates the repository-side loop end to end:

1. optional tests;
2. manifest/dataset/fleet prompt generation;
3. adapter-first per-agent fine-tuning dataset generation;
4. TestFlight scenario/runbook output;
5. adapter-first GGUF release-bake manifest generation;
6. optional explicit release-bake GGUF export;
7. visual terminal summary;
8. standalone HTML/SVG/JSON dashboard output.

The actual TestFlight app execution still has to happen on a device or CI/App Store
pipeline that can run the iOS build. This runner automates everything around that
real-app handoff and automatically ingests exported runtime audit JSON files when
provided or discovered.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
import webbrowser
from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

ANSI_RESET = "\033[0m"
ANSI_BOLD = "\033[1m"
ANSI_DIM = "\033[2m"
ANSI_GREEN = "\033[32m"
ANSI_YELLOW = "\033[33m"
ANSI_RED = "\033[31m"
ANSI_BLUE = "\033[34m"
ANSI_CYAN = "\033[36m"

DEFAULT_AGENTS = "cortex,executor,mouth,mimicry,rem,fleet"
RUNTIME_AUDIT_NAME_HINTS = (
    "runtime",
    "audit",
    "in-app-dataset",
    "in_app_dataset",
    "agent-grounding",
    "agent_grounding",
    "testflight",
)
RUNTIME_AUDIT_EXCLUDE_HINTS = (
    "loop_state",
    "loop_gaps",
    "next_action_prompts",
    "testflight_scenarios",
    "release_bake_gguf_manifest",
    "adapter_runtime_manifest",
    "dataset_manifest",
)


@dataclass(slots=True)
class StepResult:
    name: str
    command: list[str]
    cwd: str
    started_at: str
    ended_at: str
    duration_seconds: float
    returncode: int
    stdout_tail: str = ""
    stderr_tail: str = ""
    skipped: bool = False
    notes: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.returncode == 0

    @property
    def status(self) -> str:
        if self.skipped:
            return "skipped"
        return "passed" if self.passed else "failed"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "command": self.command,
            "cwd": self.cwd,
            "startedAt": self.started_at,
            "endedAt": self.ended_at,
            "durationSeconds": round(self.duration_seconds, 3),
            "returncode": self.returncode,
            "status": self.status,
            "passed": self.passed,
            "skipped": self.skipped,
            "stdoutTail": self.stdout_tail,
            "stderrTail": self.stderr_tail,
            "notes": self.notes,
        }


@dataclass(slots=True)
class LoopArtifacts:
    state: dict[str, Any]
    gaps: list[dict[str, Any]]
    next_prompts: list[dict[str, Any]]
    testflight_scenarios: list[dict[str, Any]]
    release_bake_manifest: dict[str, Any]
    adapter_runtime_manifest: dict[str, Any]


class Console:
    def __init__(self, *, color: bool, verbose: bool) -> None:
        self.color = color
        self.verbose = verbose

    def paint(self, value: str, color: str) -> str:
        if not self.color:
            return value
        return f"{color}{value}{ANSI_RESET}"

    def title(self, value: str) -> None:
        print(self.paint(f"\n{value}", ANSI_BOLD + ANSI_CYAN))

    def step(self, index: int, total: int, name: str) -> None:
        print(self.paint(f"\n[{index:02d}/{total:02d}] {name}", ANSI_BOLD + ANSI_BLUE))

    def ok(self, value: str) -> None:
        print(self.paint(f"✓ {value}", ANSI_GREEN))

    def warn(self, value: str) -> None:
        print(self.paint(f"⚠ {value}", ANSI_YELLOW))

    def fail(self, value: str) -> None:
        print(self.paint(f"✗ {value}", ANSI_RED))

    def info(self, value: str) -> None:
        print(self.paint(f"• {value}", ANSI_DIM))

    def stream(self, value: str) -> None:
        if self.verbose:
            print(self.paint(f"    {value.rstrip()}", ANSI_DIM))


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.root.resolve()
    dashboard_dir = args.dashboard_output.resolve()
    output = args.output.resolve()
    loop_output = args.loop_output.resolve()
    fine_tuning_output = args.fine_tuning_output.resolve()
    console = Console(color=not args.no_color and sys.stdout.isatty(), verbose=not args.quiet_commands)

    started_at = now_iso()
    console.title("Lumen Visual Improve-Loop")
    console.info(f"repo: {root}")
    console.info(f"manifest output: {output}")
    console.info(f"loop output: {loop_output}")
    console.info(f"dashboard output: {dashboard_dir}")

    ensure_directory(dashboard_dir)
    ensure_directory(output)
    ensure_directory(loop_output)
    ensure_directory(fine_tuning_output)

    env = build_env(root)
    runtime_audits = collect_runtime_audits(args, root, console)
    steps: list[StepResult] = []
    command_queue = build_command_queue(args, root, output, loop_output, fine_tuning_output, runtime_audits)

    total = len(command_queue) + 2
    index = 1
    console.step(index, total, "preflight")
    preflight = run_preflight(root, args, console)
    steps.append(preflight)
    index += 1
    if not preflight.passed and not args.keep_going:
        artifacts = read_artifacts(loop_output, fine_tuning_output, args.release_bake_manifest_output.resolve())
        write_visual_outputs(
            root=root,
            dashboard_dir=dashboard_dir,
            output=output,
            loop_output=loop_output,
            fine_tuning_output=fine_tuning_output,
            args=args,
            started_at=started_at,
            ended_at=now_iso(),
            steps=steps,
            artifacts=artifacts,
        )
        return 2

    for command_spec in command_queue:
        console.step(index, total, command_spec["name"])
        index += 1
        result = run_command(
            name=command_spec["name"],
            command=command_spec["command"],
            cwd=root,
            env=env,
            console=console,
            tail_chars=args.tail_chars,
            dry_run=command_spec.get("dry_run", False),
            allow_failure=command_spec.get("allow_failure", False),
        )
        steps.append(result)
        if result.passed or result.skipped:
            console.ok(f"{result.name}: {result.status} in {result.duration_seconds:.1f}s")
        else:
            console.fail(f"{result.name}: failed with exit code {result.returncode}")
            if not args.keep_going and not command_spec.get("allow_failure", False):
                break

    console.step(index, total, "visual dashboard")
    artifacts = read_artifacts(loop_output, fine_tuning_output, args.release_bake_manifest_output.resolve())
    dashboard_files = write_visual_outputs(
        root=root,
        dashboard_dir=dashboard_dir,
        output=output,
        loop_output=loop_output,
        fine_tuning_output=fine_tuning_output,
        args=args,
        started_at=started_at,
        ended_at=now_iso(),
        steps=steps,
        artifacts=artifacts,
    )
    steps.append(
        StepResult(
            name="visual dashboard",
            command=[],
            cwd=str(root),
            started_at=started_at,
            ended_at=now_iso(),
            duration_seconds=0.0,
            returncode=0,
            skipped=False,
            notes=[str(path) for path in dashboard_files],
        )
    )
    console.ok(f"dashboard: {dashboard_files['html']}")
    console.ok(f"summary: {dashboard_files['summary']}")
    console.ok(f"pipeline svg: {dashboard_files['svg']}")

    print_terminal_summary(console, artifacts, steps, dashboard_files)

    if args.open_dashboard:
        webbrowser.open(dashboard_files["html"].as_uri())

    failed_steps = [step for step in steps if not step.passed and not step.skipped]
    hard_gaps = [gap for gap in artifacts.gaps if str(gap.get("severity")) in {"critical", "error"}]
    if failed_steps:
        return 1
    if args.fail_on_gaps and hard_gaps:
        return 1
    return 0


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full Lumen improve-loop with live terminal visuals and an HTML/SVG dashboard.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--root", type=Path, default=Path("."), help="Repository root.")
    parser.add_argument("--output", type=Path, default=Path("generated/agent_manifest"), help="Agent manifest output directory.")
    parser.add_argument("--loop-output", type=Path, default=Path("generated/agent_improvement_loop"), help="Improve-loop output directory.")
    parser.add_argument("--fine-tuning-output", type=Path, default=Path("generated/fine_tuning"), help="Per-agent fine-tuning output directory.")
    parser.add_argument("--dashboard-output", type=Path, default=Path("generated/visual_improve_loop"), help="Visual report output directory.")
    parser.add_argument("--runtime-audit", action="append", type=Path, default=[], help="Runtime audit JSON file or directory. Can be passed multiple times.")
    parser.add_argument("--no-auto-discover-runtime-audit", action="store_true", help="Disable discovery of exported runtime audit JSON files.")
    parser.add_argument("--test-command", default=None, help="Override test command. Use an empty string with --skip-tests to disable.")
    parser.add_argument("--skip-tests", action="store_true", help="Skip the default test command.")
    parser.add_argument("--build-command", default=None, help="Optional build/TestFlight command passed through to improve-loop.")
    parser.add_argument("--train-command", default=None, help="Optional train command passed through to improve-loop.")
    parser.add_argument("--dry-run-commands", action="store_true", help="Record build/train commands without executing them inside improve-loop.")
    parser.add_argument("--release-bake", action="store_true", help="Explicitly run optional adapter merge/GGUF export.")
    parser.add_argument("--skip-release-bake-existing", action="store_true", help="Reuse existing release-baked GGUF files when --release-bake is enabled.")
    parser.add_argument("--release-bake-python", default=sys.executable, help="Python interpreter for tools/fine_tuning/unsloth/export_gguf.py.")
    parser.add_argument("--config-dir", type=Path, default=Path("tools/fine_tuning/unsloth/configs"), help="Unsloth config directory.")
    parser.add_argument("--agents", default=DEFAULT_AGENTS, help="Comma-separated agents for release-bake manifest/export.")
    parser.add_argument("--quantization", default="q4_k_m", help="GGUF quantization for release-bake export.")
    parser.add_argument("--release-bake-output-root", type=Path, default=Path("models/gguf_release_bake"), help="Release-bake GGUF output root.")
    parser.add_argument("--release-bake-manifest-output", type=Path, default=Path("generated/fine_tuning/release_bake_gguf_manifest.json"), help="Release-bake manifest output path.")
    parser.add_argument("--hf-repo-id", default=None, help="Optional Hugging Face repo id for release-baked GGUF upload.")
    parser.add_argument("--hf-private", action="store_true", help="Create HF repo as private if upload is enabled.")
    parser.add_argument("--skip-upload", action="store_true", help="Skip upload even when --hf-repo-id is set.")
    parser.add_argument("--strict", action=argparse.BooleanOptionalAction, default=True, help="Run improve-loop in strict mode.")
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True, help="Use deterministic generation.")
    parser.add_argument("--pretty", action=argparse.BooleanOptionalAction, default=True, help="Write pretty manifest output.")
    parser.add_argument("--generate-system-prompts", action=argparse.BooleanOptionalAction, default=True, help="Generate fleet prompts/cross-model artifacts.")
    parser.add_argument("--generate-agent-fine-tuning", action=argparse.BooleanOptionalAction, default=True, help="Generate per-agent datasets.")
    parser.add_argument("--app-run-mode", default="testflight", help="Live app runtime mode passed to improve-loop.")
    parser.add_argument("--testflight-build-label", default=None, help="Build/version label in runbook.")
    parser.add_argument("--require-testflight-runtime-audit", action="store_true", help="Treat missing TestFlight audit as a hard gap.")
    parser.add_argument("--testflight-scenario-limit", type=int, default=120, help="Maximum TestFlight scenarios to generate.")
    parser.add_argument("--fail-on-gaps", action="store_true", help="Exit non-zero if critical/error gaps are present.")
    parser.add_argument("--keep-going", action="store_true", help="Continue later stages even if a command fails.")
    parser.add_argument("--open-dashboard", action="store_true", help="Open the generated HTML dashboard in a browser.")
    parser.add_argument("--quiet-commands", action="store_true", help="Suppress live command output; tails still land in dashboard.")
    parser.add_argument("--tail-chars", type=int, default=16000, help="Maximum command tail chars kept in dashboard.")
    parser.add_argument("--no-color", action="store_true", help="Disable ANSI colours.")
    return parser.parse_args(argv)


def build_env(root: Path) -> dict[str, str]:
    env = os.environ.copy()
    crawler_root = str((root / "tools" / "lumen_manifest_crawler").resolve())
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = crawler_root if not existing else f"{crawler_root}{os.pathsep}{existing}"
    env.setdefault("PYTHONUNBUFFERED", "1")
    return env


def run_preflight(root: Path, args: argparse.Namespace, console: Console) -> StepResult:
    started = time.perf_counter()
    started_at = now_iso()
    notes: list[str] = []
    errors: list[str] = []

    required_paths = [
        root / "tools" / "lumen_manifest_crawler" / "lumen_manifest_crawler" / "cli.py",
        root / "tools" / "lumen_manifest_crawler" / "lumen_manifest_crawler" / "improvement_loop.py",
        root / "tools" / "fine_tuning" / "unsloth" / "export_gguf.py",
        root / "tools" / "fine_tuning" / "unsloth" / "configs",
        root / "ios" / "Lumen",
    ]
    for path in required_paths:
        if path.exists():
            notes.append(f"found {path.relative_to(root)}")
            console.ok(f"found {path.relative_to(root)}")
        else:
            errors.append(f"missing {path}")
            console.fail(f"missing {path}")

    if args.release_bake:
        interpreter = shutil.which(args.release_bake_python) if os.path.sep not in args.release_bake_python else args.release_bake_python
        if interpreter:
            notes.append(f"release bake python: {interpreter}")
        else:
            errors.append(f"release bake interpreter not found: {args.release_bake_python}")

    ended_at = now_iso()
    return StepResult(
        name="preflight",
        command=[],
        cwd=str(root),
        started_at=started_at,
        ended_at=ended_at,
        duration_seconds=time.perf_counter() - started,
        returncode=0 if not errors else 2,
        stdout_tail="\n".join(notes),
        stderr_tail="\n".join(errors),
        notes=notes,
    )


def collect_runtime_audits(args: argparse.Namespace, root: Path, console: Console) -> list[Path]:
    explicit: list[Path] = []
    for raw in args.runtime_audit:
        path = raw if raw.is_absolute() else root / raw
        if path.exists():
            explicit.append(path.resolve())
        else:
            console.warn(f"runtime audit path not found: {path}")

    discovered: list[Path] = []
    if not args.no_auto_discover_runtime_audit:
        search_roots = [
            root / "generated" / "runtime_audits",
            root / "generated" / "runtime_audit",
            root / "generated" / "agent_improvement_loop" / "runtime_audits",
            root / "generated" / "testflight_exports",
            root / "exports",
        ]
        for search_root in search_roots:
            if search_root.exists():
                discovered.extend(find_runtime_audit_json(search_root))

    merged: list[Path] = []
    seen: set[Path] = set()
    for path in explicit + discovered:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        merged.append(resolved)

    if merged:
        console.ok(f"runtime audit inputs: {len(merged)}")
        for path in merged[:10]:
            console.info(str(path))
    else:
        console.warn("no runtime audit JSON detected; loop will produce TestFlight handoff artifacts")
    return merged


def find_runtime_audit_json(path: Path) -> list[Path]:
    if path.is_file():
        return [path] if is_runtime_audit_candidate(path) else []
    return sorted(candidate for candidate in path.rglob("*.json") if is_runtime_audit_candidate(candidate))


def is_runtime_audit_candidate(path: Path) -> bool:
    lowered = path.name.lower()
    if any(excluded in lowered for excluded in RUNTIME_AUDIT_EXCLUDE_HINTS):
        return False
    if not any(hint in lowered for hint in RUNTIME_AUDIT_NAME_HINTS):
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if isinstance(payload, dict):
        keys = {str(key).lower() for key in payload.keys()}
        if {"failures", "traces", "runtime", "events", "toolcalls", "tool_calls"}.intersection(keys):
            return True
        serialized = json.dumps(payload, ensure_ascii=False).lower()[:20000]
        return "trace" in serialized or "runtime" in serialized or "agent grounding" in serialized
    return False


def build_command_queue(
    args: argparse.Namespace,
    root: Path,
    output: Path,
    loop_output: Path,
    fine_tuning_output: Path,
    runtime_audits: list[Path],
) -> list[dict[str, Any]]:
    commands: list[dict[str, Any]] = []
    python = sys.executable

    if not args.skip_tests:
        test_command = split_shell(args.test_command) if args.test_command else [python, "-m", "pytest", "tools/lumen_manifest_crawler/tests"]
        commands.append({"name": "test manifest crawler", "command": test_command})

    improve_loop = [
        python,
        "-m",
        "lumen_manifest_crawler",
        "improve-loop",
        "--root",
        str(root),
        "--output",
        str(output),
        "--loop-output",
        str(loop_output),
        "--fine-tuning-output",
        str(fine_tuning_output),
        "--testflight-scenario-limit",
        str(args.testflight_scenario_limit),
        "--app-run-mode",
        args.app_run_mode,
    ]
    improve_loop.append("--strict" if args.strict else "--no-strict")
    improve_loop.append("--deterministic" if args.deterministic else "--non-deterministic")
    improve_loop.append("--pretty" if args.pretty else "--no-pretty")
    improve_loop.append("--generate-system-prompts" if args.generate_system_prompts else "--no-generate-system-prompts")
    improve_loop.append("--generate-agent-fine-tuning" if args.generate_agent_fine_tuning else "--no-generate-agent-fine-tuning")
    if args.dry_run_commands:
        improve_loop.append("--dry-run-commands")
    if args.require_testflight_runtime_audit:
        improve_loop.append("--require-testflight-runtime-audit")
    if args.testflight_build_label:
        improve_loop.extend(["--testflight-build-label", args.testflight_build_label])
    if args.build_command:
        improve_loop.extend(["--build-command", args.build_command])
    if args.train_command:
        improve_loop.extend(["--train-command", args.train_command])
    for audit in runtime_audits:
        improve_loop.extend(["--runtime-audit", str(audit)])
    commands.append({"name": "improve-loop generation", "command": improve_loop})

    export_script = root / "tools" / "fine_tuning" / "unsloth" / "export_gguf.py"
    release_manifest = args.release_bake_manifest_output.resolve()
    export_command = [
        args.release_bake_python,
        str(export_script),
        "--config-dir",
        str((root / args.config_dir).resolve() if not args.config_dir.is_absolute() else args.config_dir.resolve()),
        "--agents",
        args.agents,
        "--quantization",
        args.quantization,
        "--output-root",
        str((root / args.release_bake_output_root).resolve() if not args.release_bake_output_root.is_absolute() else args.release_bake_output_root.resolve()),
        "--manifest-output",
        str(release_manifest),
    ]
    if args.release_bake:
        export_command.append("--release-bake")
        if args.skip_release_bake_existing:
            export_command.append("--skip-existing")
        if args.hf_repo_id:
            export_command.extend(["--hf-repo-id", args.hf_repo_id])
        if args.hf_private:
            export_command.append("--hf-private")
        if args.skip_upload:
            export_command.append("--skip-upload")
        commands.append({"name": "optional GGUF release bake", "command": export_command})
    else:
        commands.append({"name": "adapter-first release-bake manifest", "command": export_command})

    return commands


def split_shell(command: str | None) -> list[str]:
    if not command:
        return []
    return shlex.split(command)


def run_command(
    *,
    name: str,
    command: list[str],
    cwd: Path,
    env: dict[str, str],
    console: Console,
    tail_chars: int,
    dry_run: bool = False,
    allow_failure: bool = False,
) -> StepResult:
    started = time.perf_counter()
    started_at = now_iso()
    if dry_run:
        console.info("dry-run: " + shlex.join(command))
        return StepResult(
            name=name,
            command=command,
            cwd=str(cwd),
            started_at=started_at,
            ended_at=now_iso(),
            duration_seconds=time.perf_counter() - started,
            returncode=0,
            stdout_tail="dry-run",
            skipped=True,
        )

    console.info(shlex.join(command))
    tail: deque[str] = deque()
    try:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            errors="replace",
        )
    except OSError as exc:
        return StepResult(
            name=name,
            command=command,
            cwd=str(cwd),
            started_at=started_at,
            ended_at=now_iso(),
            duration_seconds=time.perf_counter() - started,
            returncode=127 if not allow_failure else 0,
            stderr_tail=str(exc),
        )

    assert process.stdout is not None
    for line in process.stdout:
        tail.append(line)
        while sum(len(item) for item in tail) > tail_chars and tail:
            tail.popleft()
        console.stream(line)
    returncode = process.wait()
    ended_at = now_iso()
    return StepResult(
        name=name,
        command=command,
        cwd=str(cwd),
        started_at=started_at,
        ended_at=ended_at,
        duration_seconds=time.perf_counter() - started,
        returncode=returncode if not allow_failure else 0,
        stdout_tail="".join(tail),
    )


def read_artifacts(loop_output: Path, fine_tuning_output: Path, release_bake_manifest_path: Path) -> LoopArtifacts:
    state = read_json(loop_output / "loop_state.json")
    gaps_payload = read_json(loop_output / "loop_gaps.json")
    gaps = gaps_payload.get("gaps", []) if isinstance(gaps_payload, dict) else []
    next_prompts = read_jsonl(loop_output / "next_action_prompts.jsonl")
    testflight_scenarios = read_jsonl(loop_output / "testflight_scenarios.jsonl")
    release_bake_manifest = read_json(release_bake_manifest_path)
    adapter_runtime_manifest = read_json(fine_tuning_output / "adapter_runtime_manifest.json")
    return LoopArtifacts(
        state=state if isinstance(state, dict) else {},
        gaps=[item for item in gaps if isinstance(item, dict)],
        next_prompts=[item for item in next_prompts if isinstance(item, dict)],
        testflight_scenarios=[item for item in testflight_scenarios if isinstance(item, dict)],
        release_bake_manifest=release_bake_manifest if isinstance(release_bake_manifest, dict) else {},
        adapter_runtime_manifest=adapter_runtime_manifest if isinstance(adapter_runtime_manifest, dict) else {},
    )


def read_json(path: Path) -> Any:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"_readError": str(exc), "_path": str(path)}


def read_jsonl(path: Path) -> list[Any]:
    if not path.exists():
        return []
    records: list[Any] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            records.append(json.loads(stripped))
        except Exception as exc:
            records.append({"_readError": str(exc), "_path": str(path), "_line": line_number})
    return records


def write_visual_outputs(
    *,
    root: Path,
    dashboard_dir: Path,
    output: Path,
    loop_output: Path,
    fine_tuning_output: Path,
    args: argparse.Namespace,
    started_at: str,
    ended_at: str,
    steps: list[StepResult],
    artifacts: LoopArtifacts,
) -> dict[str, Path]:
    ensure_directory(dashboard_dir)
    severity_counts = Counter(str(gap.get("severity") or "unknown") for gap in artifacts.gaps)
    summary = build_summary(
        root=root,
        output=output,
        loop_output=loop_output,
        fine_tuning_output=fine_tuning_output,
        args=args,
        started_at=started_at,
        ended_at=ended_at,
        steps=steps,
        artifacts=artifacts,
        severity_counts=severity_counts,
    )
    svg = build_pipeline_svg(steps, artifacts)
    html_report = build_html_dashboard(summary, steps, artifacts, svg)

    summary_path = dashboard_dir / "visual_improve_loop_summary.json"
    html_path = dashboard_dir / "index.html"
    svg_path = dashboard_dir / "pipeline.svg"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    html_path.write_text(html_report, encoding="utf-8")
    svg_path.write_text(svg, encoding="utf-8")
    return {"summary": summary_path, "html": html_path, "svg": svg_path}


def build_summary(
    *,
    root: Path,
    output: Path,
    loop_output: Path,
    fine_tuning_output: Path,
    args: argparse.Namespace,
    started_at: str,
    ended_at: str,
    steps: list[StepResult],
    artifacts: LoopArtifacts,
    severity_counts: Counter[str],
) -> dict[str, Any]:
    state = artifacts.state
    dataset = state.get("dataset", {}) if isinstance(state.get("dataset"), dict) else {}
    manifest = state.get("manifest", {}) if isinstance(state.get("manifest"), dict) else {}
    runtime = state.get("runtime", {}) if isinstance(state.get("runtime"), dict) else {}
    testflight = state.get("testFlight", {}) if isinstance(state.get("testFlight"), dict) else {}
    failed_steps = [step for step in steps if not step.passed and not step.skipped]
    hard_gaps = [gap for gap in artifacts.gaps if str(gap.get("severity")) in {"critical", "error"}]
    return {
        "schemaVersion": "1.0.0",
        "startedAt": started_at,
        "endedAt": ended_at,
        "root": str(root),
        "output": str(output),
        "loopOutput": str(loop_output),
        "fineTuningOutput": str(fine_tuning_output),
        "passed": not failed_steps and not hard_gaps,
        "releaseBakeRequested": bool(args.release_bake),
        "manifest": manifest,
        "dataset": dataset,
        "runtime": runtime,
        "testFlight": testflight,
        "gaps": {
            "total": len(artifacts.gaps),
            "bySeverity": dict(sorted(severity_counts.items())),
            "hard": len(hard_gaps),
        },
        "nextPrompts": len(artifacts.next_prompts),
        "testFlightScenarios": len(artifacts.testflight_scenarios),
        "adapterRuntimeManifest": {
            "mode": artifacts.adapter_runtime_manifest.get("mode"),
            "sharedBaseModelID": artifacts.adapter_runtime_manifest.get("sharedBaseModelID"),
            "adapterCount": len(artifacts.adapter_runtime_manifest.get("adapters", []) or []),
            "releaseBakePolicy": artifacts.adapter_runtime_manifest.get("releaseBakePolicy", {}),
        },
        "releaseBakeManifest": {
            "mode": artifacts.release_bake_manifest.get("mode"),
            "skipped": artifacts.release_bake_manifest.get("skipped"),
            "releaseBakeRequested": artifacts.release_bake_manifest.get("release_bake_requested"),
            "agentCount": len(artifacts.release_bake_manifest.get("agents", {}) or {}),
        },
        "steps": [step.to_dict() for step in steps],
    }


def build_html_dashboard(summary: dict[str, Any], steps: list[StepResult], artifacts: LoopArtifacts, pipeline_svg: str) -> str:
    dataset = summary.get("dataset", {}) if isinstance(summary.get("dataset"), dict) else {}
    manifest = summary.get("manifest", {}) if isinstance(summary.get("manifest"), dict) else {}
    runtime = summary.get("runtime", {}) if isinstance(summary.get("runtime"), dict) else {}
    gaps = summary.get("gaps", {}) if isinstance(summary.get("gaps"), dict) else {}
    families = dataset.get("families", {}) if isinstance(dataset.get("families"), dict) else {}
    agent_ft = dataset.get("agentFineTuning", {}) if isinstance(dataset.get("agentFineTuning"), dict) else {}
    release = summary.get("releaseBakeManifest", {}) if isinstance(summary.get("releaseBakeManifest"), dict) else {}
    adapter = summary.get("adapterRuntimeManifest", {}) if isinstance(summary.get("adapterRuntimeManifest"), dict) else {}
    status_class = "pass" if summary.get("passed") else "fail"
    generated_at = html.escape(summary.get("endedAt", now_iso()))

    cards = [
        ("status", "PASS" if summary.get("passed") else "NEEDS WORK", status_class),
        ("tools", manifest.get("toolCount", 0), "neutral"),
        ("intents", manifest.get("intentCount", 0), "neutral"),
        ("model slots", manifest.get("modelSlotCount", 0), "neutral"),
        ("dataset records", dataset.get("recordCount", 0), "neutral"),
        ("gaps", gaps.get("total", 0), "fail" if gaps.get("hard", 0) else "warn" if gaps.get("total", 0) else "pass"),
        ("next prompts", summary.get("nextPrompts", 0), "neutral"),
        ("TestFlight scenarios", summary.get("testFlightScenarios", 0), "neutral"),
    ]
    card_html = "\n".join(
        f'<section class="card {klass}"><span>{html.escape(str(label))}</span><strong>{html.escape(str(value))}</strong></section>'
        for label, value, klass in cards
    )

    step_rows = "\n".join(
        "<tr>"
        f"<td>{html.escape(step.name)}</td>"
        f"<td><span class='pill {html.escape(step.status)}'>{html.escape(step.status)}</span></td>"
        f"<td>{step.duration_seconds:.1f}s</td>"
        f"<td><code>{html.escape(shlex.join(step.command)) if step.command else 'internal'}</code></td>"
        "</tr>"
        for step in steps
    )

    gap_rows = "\n".join(
        "<tr>"
        f"<td><span class='pill {html.escape(str(gap.get('severity', 'unknown')))}'>{html.escape(str(gap.get('severity', 'unknown')))}</span></td>"
        f"<td>{html.escape(str(gap.get('category', '')))}</td>"
        f"<td>{html.escape(str(gap.get('title', '')))}</td>"
        f"<td>{html.escape(str(gap.get('recommendedAction', '')))}</td>"
        "</tr>"
        for gap in artifacts.gaps[:80]
    ) or "<tr><td colspan='4'>No gaps detected.</td></tr>"

    prompt_rows = "\n".join(
        "<tr>"
        f"<td>{html.escape(str(prompt.get('taskType', '')))}</td>"
        f"<td>{html.escape(str(prompt.get('priority', '')))}</td>"
        f"<td><code>{html.escape(str(prompt.get('id', '')))}</code></td>"
        "</tr>"
        for prompt in artifacts.next_prompts[:80]
    ) or "<tr><td colspan='3'>No next prompts generated.</td></tr>"

    command_tails = "\n".join(
        f"<details><summary>{html.escape(step.name)} · {html.escape(step.status)}</summary><pre>{html.escape(step.stdout_tail or step.stderr_tail or '')}</pre></details>"
        for step in steps
        if step.stdout_tail or step.stderr_tail
    )

    dataset_chart = svg_bar_chart(families, "Dataset family records")
    severity_chart = svg_bar_chart(gaps.get("bySeverity", {}) if isinstance(gaps.get("bySeverity"), dict) else {}, "Gap severities")
    ft_chart = svg_bar_chart({agent: sum(int(v) for v in counts.values() if isinstance(v, int)) for agent, counts in agent_ft.items() if isinstance(counts, dict)}, "Agent fine-tuning records")

    release_json = html.escape(json.dumps(release, ensure_ascii=False, indent=2, sort_keys=True))
    adapter_json = html.escape(json.dumps(adapter, ensure_ascii=False, indent=2, sort_keys=True))
    runtime_json = html.escape(json.dumps(runtime, ensure_ascii=False, indent=2, sort_keys=True))

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Lumen Visual Improve-Loop</title>
<style>
:root {{
  color-scheme: dark;
  --bg: #090b10;
  --panel: #111722;
  --panel2: #151d2b;
  --text: #e8eefc;
  --muted: #8ea0bd;
  --green: #44d483;
  --yellow: #ffcc66;
  --red: #ff6b7a;
  --blue: #73a7ff;
  --cyan: #64e3ff;
  --line: #27364f;
}}
* {{ box-sizing: border-box; }}
body {{ margin: 0; background: radial-gradient(circle at top left, #17233a 0, var(--bg) 42rem); color: var(--text); font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
header {{ padding: 2rem; border-bottom: 1px solid var(--line); background: rgba(9,11,16,.78); position: sticky; top: 0; backdrop-filter: blur(18px); z-index: 2; }}
h1 {{ margin: 0; font-size: clamp(1.8rem, 3vw, 3rem); letter-spacing: -0.04em; }}
header p {{ margin: .5rem 0 0; color: var(--muted); }}
main {{ padding: 2rem; display: grid; gap: 1.25rem; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 1rem; }}
.card {{ background: linear-gradient(180deg, var(--panel), var(--panel2)); border: 1px solid var(--line); border-radius: 18px; padding: 1rem; min-height: 105px; box-shadow: 0 18px 40px rgba(0,0,0,.24); }}
.card span {{ display: block; color: var(--muted); text-transform: uppercase; font-size: .74rem; letter-spacing: .12em; }}
.card strong {{ display: block; margin-top: .65rem; font-size: 1.9rem; letter-spacing: -0.04em; }}
.card.pass strong {{ color: var(--green); }}
.card.warn strong {{ color: var(--yellow); }}
.card.fail strong {{ color: var(--red); }}
.panel {{ background: rgba(17,23,34,.84); border: 1px solid var(--line); border-radius: 20px; padding: 1.25rem; overflow: auto; box-shadow: 0 18px 40px rgba(0,0,0,.20); }}
.panel h2 {{ margin: 0 0 1rem; font-size: 1.15rem; letter-spacing: -0.02em; }}
table {{ width: 100%; border-collapse: collapse; }}
th, td {{ padding: .72rem .65rem; border-bottom: 1px solid var(--line); text-align: left; vertical-align: top; }}
th {{ color: var(--muted); font-weight: 600; font-size: .82rem; text-transform: uppercase; letter-spacing: .08em; }}
code, pre {{ font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace; }}
code {{ color: #bad2ff; }}
pre {{ max-height: 24rem; overflow: auto; padding: 1rem; background: #080b12; border: 1px solid var(--line); border-radius: 12px; color: #d9e6ff; white-space: pre-wrap; }}
details {{ margin: .75rem 0; }}
summary {{ cursor: pointer; color: var(--cyan); }}
.pill {{ display: inline-flex; align-items: center; padding: .2rem .55rem; border-radius: 999px; font-size: .8rem; font-weight: 700; border: 1px solid currentColor; }}
.pill.passed, .pill.pass {{ color: var(--green); }}
.pill.warning, .pill.warn, .pill.skipped {{ color: var(--yellow); }}
.pill.failed, .pill.critical, .pill.error, .pill.fail {{ color: var(--red); }}
.two {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(340px, 1fr)); gap: 1.25rem; }}
svg text {{ font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
.footer {{ color: var(--muted); font-size: .86rem; }}
</style>
</head>
<body>
<header>
<h1>Lumen Visual Improve-Loop</h1>
<p>Generated at {generated_at}. Adapter-first by default. Release bake is explicit.</p>
</header>
<main>
<section class="grid">{card_html}</section>
<section class="panel"><h2>Pipeline</h2>{pipeline_svg}</section>
<section class="two">
  <section class="panel"><h2>Dataset records</h2>{dataset_chart}</section>
  <section class="panel"><h2>Gap severities</h2>{severity_chart}</section>
</section>
<section class="panel"><h2>Agent fine-tuning</h2>{ft_chart}</section>
<section class="panel"><h2>Steps</h2><table><thead><tr><th>Step</th><th>Status</th><th>Time</th><th>Command</th></tr></thead><tbody>{step_rows}</tbody></table></section>
<section class="panel"><h2>Gaps</h2><table><thead><tr><th>Severity</th><th>Category</th><th>Title</th><th>Action</th></tr></thead><tbody>{gap_rows}</tbody></table></section>
<section class="panel"><h2>Next action prompts</h2><table><thead><tr><th>Task</th><th>Priority</th><th>ID</th></tr></thead><tbody>{prompt_rows}</tbody></table></section>
<section class="two">
  <section class="panel"><h2>Adapter runtime manifest</h2><pre>{adapter_json}</pre></section>
  <section class="panel"><h2>Release-bake manifest</h2><pre>{release_json}</pre></section>
</section>
<section class="panel"><h2>Runtime summary</h2><pre>{runtime_json}</pre></section>
<section class="panel"><h2>Command output tails</h2>{command_tails or '<p>No command output captured.</p>'}</section>
<p class="footer">Generated by <code>tools/run_visual_improve_loop.py</code>.</p>
</main>
</body>
</html>
"""


def build_pipeline_svg(steps: list[StepResult], artifacts: LoopArtifacts) -> str:
    width = 1180
    row_height = 82
    height = 120 + max(1, len(steps)) * row_height
    boxes: list[str] = []
    for index, step in enumerate(steps):
        y = 70 + index * row_height
        color = "#44d483" if step.passed else "#ff6b7a"
        if step.skipped:
            color = "#ffcc66"
        label = svg_escape(step.name)
        status = svg_escape(step.status)
        command = svg_escape(shorten(shlex.join(step.command), 94) if step.command else "internal")
        boxes.append(f"<rect x='42' y='{y}' width='1090' height='58' rx='16' fill='#111722' stroke='{color}' stroke-width='2'/>")
        boxes.append(f"<circle cx='72' cy='{y + 29}' r='10' fill='{color}'/>")
        boxes.append(f"<text x='96' y='{y + 24}' fill='#e8eefc' font-size='17' font-weight='700'>{index + 1}. {label}</text>")
        boxes.append(f"<text x='96' y='{y + 45}' fill='#8ea0bd' font-size='12'>{status} · {step.duration_seconds:.1f}s · {command}</text>")
        if index < len(steps) - 1:
            boxes.append(f"<path d='M72 {y + 58} L72 {y + row_height}' stroke='#27364f' stroke-width='3'/>")
    hard_gaps = sum(1 for gap in artifacts.gaps if str(gap.get("severity")) in {"critical", "error"})
    title_color = "#44d483" if hard_gaps == 0 and all(step.passed for step in steps) else "#ff6b7a"
    return (
        f"<svg viewBox='0 0 {width} {height}' width='100%' role='img' aria-label='Lumen improve-loop pipeline' xmlns='http://www.w3.org/2000/svg'>"
        "<rect width='100%' height='100%' rx='20' fill='#090b10'/>"
        f"<text x='42' y='38' fill='{title_color}' font-size='25' font-weight='800'>Improve-loop pipeline</text>"
        f"<text x='42' y='60' fill='#8ea0bd' font-size='13'>Hard gaps: {hard_gaps} · Total gaps: {len(artifacts.gaps)} · Scenarios: {len(artifacts.testflight_scenarios)}</text>"
        + "".join(boxes)
        + "</svg>"
    )


def svg_bar_chart(data: dict[str, Any], title: str) -> str:
    clean: dict[str, float] = {}
    for key, value in data.items():
        try:
            clean[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    if not clean:
        return "<p>No data.</p>"
    items = sorted(clean.items(), key=lambda item: item[1], reverse=True)[:28]
    width = 920
    left = 250
    bar_width = 600
    row_height = 28
    height = 54 + len(items) * row_height
    max_value = max(value for _, value in items) or 1.0
    rows: list[str] = []
    for index, (label, value) in enumerate(items):
        y = 42 + index * row_height
        w = max(2, int((value / max_value) * bar_width))
        rows.append(f"<text x='12' y='{y + 16}' fill='#8ea0bd' font-size='12'>{svg_escape(shorten(label, 34))}</text>")
        rows.append(f"<rect x='{left}' y='{y}' width='{bar_width}' height='18' rx='9' fill='#0b1019' stroke='#27364f'/>")
        rows.append(f"<rect x='{left}' y='{y}' width='{w}' height='18' rx='9' fill='#73a7ff'/>")
        rows.append(f"<text x='{left + bar_width + 14}' y='{y + 15}' fill='#e8eefc' font-size='12'>{int(value) if value.is_integer() else round(value, 2)}</text>")
    return (
        f"<svg viewBox='0 0 {width} {height}' width='100%' role='img' aria-label='{svg_escape(title)}' xmlns='http://www.w3.org/2000/svg'>"
        f"<text x='12' y='24' fill='#e8eefc' font-size='16' font-weight='700'>{svg_escape(title)}</text>"
        + "".join(rows)
        + "</svg>"
    )


def print_terminal_summary(console: Console, artifacts: LoopArtifacts, steps: list[StepResult], dashboard_files: dict[str, Path]) -> None:
    hard_gaps = [gap for gap in artifacts.gaps if str(gap.get("severity")) in {"critical", "error"}]
    failed_steps = [step for step in steps if not step.passed and not step.skipped]
    console.title("Summary")
    if not failed_steps and not hard_gaps:
        console.ok("loop passed with no hard gaps")
    else:
        if failed_steps:
            console.fail(f"failed steps: {len(failed_steps)}")
        if hard_gaps:
            console.warn(f"hard gaps: {len(hard_gaps)}")
    console.info(f"total gaps: {len(artifacts.gaps)}")
    console.info(f"next prompts: {len(artifacts.next_prompts)}")
    console.info(f"TestFlight scenarios: {len(artifacts.testflight_scenarios)}")
    console.info(f"HTML: {dashboard_files['html']}")


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def shorten(value: str, max_len: int) -> str:
    normalized = re.sub(r"\s+", " ", value).strip()
    if len(normalized) <= max_len:
        return normalized
    return normalized[: max_len - 1] + "…"


def svg_escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


if __name__ == "__main__":
    raise SystemExit(main())
