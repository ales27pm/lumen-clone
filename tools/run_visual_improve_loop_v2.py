#!/usr/bin/env python3
"""Hardened visual runner for the Lumen improvement loop.

This is the repo-rooted version of the visual improve-loop orchestrator. Every
relative path is resolved against --root, not against the shell's current working
directory. The runner keeps the normal loop adapter-first, writes a visual HTML
report, emits a pipeline SVG, and makes GGUF release bake explicit.
"""
# pylint: disable=line-too-long
# cspell:words GGUF gguf timespec PYTHONUNBUFFERED toolcalls popleft Segoe minmax Menlo Consolas

from __future__ import annotations

import argparse
import html
import json
import os
import re
import shlex
import subprocess
import sys
import time
import webbrowser
from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

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


@dataclass
class StepResult:
    """Execution outcome for a single pipeline step."""

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
        """Return whether the step finished successfully."""
        return self.returncode == 0

    @property
    def status(self) -> str:
        """Return a user-facing step status string."""
        if self.skipped:
            return "skipped"
        return "passed" if self.passed else "failed"

    def to_dict(self) -> dict[str, Any]:
        """Serialize this result for summary JSON output."""
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


@dataclass
class LoopArtifacts:
    """Loaded improve-loop artifacts consumed by the dashboard."""

    state: dict[str, Any]
    gaps: list[dict[str, Any]]
    next_prompts: list[dict[str, Any]]
    testflight_scenarios: list[dict[str, Any]]
    release_bake_manifest: dict[str, Any]
    adapter_runtime_manifest: dict[str, Any]


class Console:
    """Minimal console printer with optional quiet mode."""

    def __init__(self, *, quiet: bool = False) -> None:
        """Create a console wrapper."""
        self.quiet = quiet

    def line(self, message: str = "") -> None:
        """Print a line when output is enabled."""
        if not self.quiet:
            print(message)

    def step(self, index: int, total: int, name: str) -> None:
        """Print a formatted step header."""
        self.line(f"\n[{index:02d}/{total:02d}] {name}")

    def ok(self, message: str) -> None:
        """Print a success message."""
        self.line(f"✓ {message}")

    def warn(self, message: str) -> None:
        """Print a warning message."""
        self.line(f"⚠ {message}")

    def fail(self, message: str) -> None:
        """Print an error message."""
        self.line(f"✗ {message}")

    def info(self, message: str) -> None:
        """Print an informational message."""
        self.line(f"• {message}")


def now_iso() -> str:
    """Return the current UTC time formatted as an ISO timestamp."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def rooted_path(root: Path, value: Path) -> Path:
    """Resolve a CLI path against the repo root when it is relative."""
    value = Path(value)
    return value.resolve() if value.is_absolute() else (root / value).resolve()


def split_shell(command: str | None) -> list[str]:
    """Split a shell command string into argv tokens."""
    if not command or not command.strip():
        return []
    return shlex.split(command)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the visual improve-loop runner."""
    parser = argparse.ArgumentParser(
        description="Run the Lumen improve-loop with repo-rooted paths and visual HTML/SVG output.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--root", type=Path, default=Path("."), help="Repository root.")
    parser.add_argument("--output", type=Path, default=Path("generated/agent_manifest"), help="Agent manifest output directory.")
    parser.add_argument("--loop-output", type=Path, default=Path("generated/agent_improvement_loop"), help="Improve-loop output directory.")
    parser.add_argument("--fine-tuning-output", type=Path, default=Path("generated/fine_tuning"), help="Per-agent fine-tuning output directory.")
    parser.add_argument("--dashboard-output", type=Path, default=Path("generated/visual_improve_loop"), help="Visual report output directory.")
    parser.add_argument("--runtime-audit", action="append", type=Path, default=[], help="Runtime audit JSON file or directory. Can be passed more than once.")
    parser.add_argument("--no-auto-discover-runtime-audit", action="store_true", help="Disable runtime-audit auto-discovery.")
    parser.add_argument("--test-command", default=None, help="Override test command.")
    parser.add_argument("--skip-tests", action="store_true", help="Skip default pytest step.")
    parser.add_argument("--build-command", default=None, help="Optional build command passed into improve-loop.")
    parser.add_argument("--train-command", default=None, help="Optional training command passed into improve-loop.")
    parser.add_argument("--dry-run-commands", action="store_true", help="Record build/train commands without executing them in improve-loop.")
    parser.add_argument("--release-bake", action="store_true", help="Explicitly run optional GGUF release bake.")
    parser.add_argument("--skip-release-bake-existing", action="store_true", help="Reuse existing release-baked GGUF files.")
    parser.add_argument("--release-bake-python", default=sys.executable, help="Python interpreter for export_gguf.py.")
    parser.add_argument("--config-dir", type=Path, default=Path("tools/fine_tuning/unsloth/configs"), help="Unsloth config directory.")
    parser.add_argument("--agents", default=DEFAULT_AGENTS, help="Comma-separated agents.")
    parser.add_argument("--quantization", default="q4_k_m", help="GGUF quantization.")
    parser.add_argument("--release-bake-output-root", type=Path, default=Path("models/gguf_release_bake"), help="Release-bake GGUF output root.")
    parser.add_argument("--release-bake-manifest-output", type=Path, default=Path("generated/fine_tuning/release_bake_gguf_manifest.json"), help="Release-bake manifest output file.")
    parser.add_argument("--hf-repo-id", default=None, help="Optional Hugging Face repo id for upload.")
    parser.add_argument("--hf-private", action="store_true", help="Create HF repo as private.")
    parser.add_argument("--skip-upload", action="store_true", help="Skip HF upload.")
    parser.add_argument("--strict", action=argparse.BooleanOptionalAction, default=True, help="Run improve-loop in strict mode.")
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True, help="Use deterministic generation.")
    parser.add_argument("--pretty", action=argparse.BooleanOptionalAction, default=True, help="Write pretty manifest output.")
    parser.add_argument("--generate-system-prompts", action=argparse.BooleanOptionalAction, default=True, help="Generate fleet prompts/cross-model artifacts.")
    parser.add_argument("--generate-agent-fine-tuning", action=argparse.BooleanOptionalAction, default=True, help="Generate per-agent datasets.")
    parser.add_argument("--app-run-mode", default="testflight", help="Live runtime mode passed to improve-loop.")
    parser.add_argument("--testflight-build-label", default=None, help="Build/version label for runbook.")
    parser.add_argument("--require-testflight-runtime-audit", action="store_true", help="Treat missing TestFlight audit as hard gap.")
    parser.add_argument("--testflight-scenario-limit", type=int, default=120, help="Maximum TestFlight scenarios.")
    parser.add_argument("--fail-on-gaps", action="store_true", help="Exit non-zero on critical/error gaps.")
    parser.add_argument("--keep-going", action="store_true", help="Continue after failed commands.")
    parser.add_argument("--open-dashboard", action="store_true", help="Open generated dashboard.")
    parser.add_argument("--quiet-commands", action="store_true", help="Suppress live command output.")
    parser.add_argument("--tail-chars", type=int, default=16000, help="Command output tail chars to keep.")
    return parser.parse_args(argv)


def build_env(root: Path) -> dict[str, str]:
    """Build process environment variables for child commands."""
    env = os.environ.copy()
    crawler_root = str((root / "tools" / "lumen_manifest_crawler").resolve())
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = crawler_root if not existing else f"{crawler_root}{os.pathsep}{existing}"
    env.setdefault("PYTHONUNBUFFERED", "1")
    return env


def _append_optional_improve_flags(
    improve: list[str], args: argparse.Namespace, runtime_audits: list[Path]
) -> None:
    """Append optional improve-loop arguments based on CLI flags."""
    if args.dry_run_commands:
        improve.append("--dry-run-commands")
    if args.require_testflight_runtime_audit:
        improve.append("--require-testflight-runtime-audit")
    if args.testflight_build_label:
        improve.extend(["--testflight-build-label", args.testflight_build_label])
    if args.build_command:
        improve.extend(["--build-command", args.build_command])
    if args.train_command:
        improve.extend(["--train-command", args.train_command])
    for audit in runtime_audits:
        improve.extend(["--runtime-audit", str(audit)])


def _build_improve_command(
    args: argparse.Namespace,
    root: Path,
    output: Path,
    loop_output: Path,
    fine_tuning_output: Path,
    runtime_audits: list[Path],
) -> list[str]:
    """Build the improve-loop command argv."""
    improve = [
        sys.executable,
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
        "--strict" if args.strict else "--no-strict",
        "--deterministic" if args.deterministic else "--non-deterministic",
        "--pretty" if args.pretty else "--no-pretty",
        (
            "--generate-system-prompts"
            if args.generate_system_prompts
            else "--no-generate-system-prompts"
        ),
        (
            "--generate-agent-fine-tuning"
            if args.generate_agent_fine_tuning
            else "--no-generate-agent-fine-tuning"
        ),
    ]
    _append_optional_improve_flags(improve, args, runtime_audits)
    return improve


def _build_release_export_command(
    args: argparse.Namespace, root: Path, release_manifest: Path
) -> tuple[str, list[str]]:
    """Build the release-bake/export command and step name."""
    export = [
        args.release_bake_python,
        str(root / "tools" / "fine_tuning" / "unsloth" / "export_gguf.py"),
        "--config-dir",
        str(rooted_path(root, args.config_dir)),
        "--agents",
        args.agents,
        "--quantization",
        args.quantization,
        "--output-root",
        str(rooted_path(root, args.release_bake_output_root)),
        "--manifest-output",
        str(release_manifest),
    ]
    if not args.release_bake:
        return "adapter-first release-bake manifest", export

    export.append("--release-bake")
    if args.skip_release_bake_existing:
        export.append("--skip-existing")
    if args.hf_repo_id:
        export.extend(["--hf-repo-id", args.hf_repo_id])
    if args.hf_private:
        export.append("--hf-private")
    if args.skip_upload:
        export.append("--skip-upload")
    return "optional GGUF release bake", export


def build_command_queue(
    args: argparse.Namespace,
    root: Path,
    output: Path,
    loop_output: Path,
    fine_tuning_output: Path,
    runtime_audits: list[Path],
) -> list[dict[str, Any]]:
    """Construct the ordered command queue for the run."""
    commands: list[dict[str, Any]] = []

    if not args.skip_tests:
        default_test = [sys.executable, "-m", "pytest", "tools/lumen_manifest_crawler/tests"]
        commands.append(
            {
                "name": "test manifest crawler",
                "command": split_shell(args.test_command) or default_test,
            }
        )

    improve = _build_improve_command(
        args, root, output, loop_output, fine_tuning_output, runtime_audits
    )
    commands.append({"name": "improve-loop generation", "command": improve})

    release_manifest = rooted_path(root, args.release_bake_manifest_output)
    name, export = _build_release_export_command(args, root, release_manifest)
    commands.append({"name": name, "command": export})
    return commands


def is_runtime_audit_candidate(path: Path) -> bool:
    """Return whether a JSON file appears to be a runtime audit payload."""
    lowered = path.name.lower()
    if any(token in lowered for token in RUNTIME_AUDIT_EXCLUDE_HINTS):
        return False
    if not any(token in lowered for token in RUNTIME_AUDIT_NAME_HINTS):
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False
    if not isinstance(payload, dict):
        return False
    keys = {str(key).lower() for key in payload}
    if {"failures", "traces", "runtime", "events", "toolcalls", "tool_calls"}.intersection(keys):
        return True
    sample = json.dumps(payload, ensure_ascii=False)[:20000].lower()
    return "runtime" in sample or "trace" in sample or "agent grounding" in sample


def find_runtime_audit_json(path: Path) -> list[Path]:
    """Find runtime-audit JSON files from a file or directory path."""
    if path.is_file():
        return [path] if is_runtime_audit_candidate(path) else []
    if not path.exists():
        return []
    return sorted(candidate for candidate in path.rglob("*.json") if is_runtime_audit_candidate(candidate))


def collect_runtime_audits(args: argparse.Namespace, root: Path, console: Console) -> list[Path]:
    """Collect explicit and auto-discovered runtime-audit inputs."""
    found: list[Path] = []
    for raw in args.runtime_audit:
        path = rooted_path(root, raw)
        if path.exists():
            found.extend(find_runtime_audit_json(path) if path.is_dir() else [path])
        else:
            console.warn(f"runtime audit path not found: {path}")
    if not args.no_auto_discover_runtime_audit:
        for search_root in (
            root / "generated" / "runtime_audits",
            root / "generated" / "runtime_audit",
            root / "generated" / "agent_improvement_loop" / "runtime_audits",
            root / "generated" / "testflight_exports",
            root / "exports",
        ):
            found.extend(find_runtime_audit_json(search_root))
    deduped: list[Path] = []
    seen: set[Path] = set()
    for path in found:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            deduped.append(resolved)
    if deduped:
        console.ok(f"runtime audit inputs: {len(deduped)}")
    else:
        console.warn("no runtime audit JSON detected; TestFlight handoff artifacts will be generated")
    return deduped


def run_preflight(root: Path) -> StepResult:
    """Check required repository files before executing commands."""
    start = time.perf_counter()
    started_at = now_iso()
    required = [
        root / "tools" / "lumen_manifest_crawler" / "lumen_manifest_crawler" / "cli.py",
        root / "tools" / "lumen_manifest_crawler" / "lumen_manifest_crawler" / "improvement_loop.py",
        root / "tools" / "fine_tuning" / "unsloth" / "export_gguf.py",
        root / "tools" / "fine_tuning" / "unsloth" / "configs",
        root / "ios" / "Lumen",
    ]
    missing = [path for path in required if not path.exists()]
    return StepResult(
        name="preflight",
        command=[],
        cwd=str(root),
        started_at=started_at,
        ended_at=now_iso(),
        duration_seconds=time.perf_counter() - start,
        returncode=0 if not missing else 2,
        stdout_tail="\n".join(str(path) for path in required if path.exists()),
        stderr_tail="\n".join(f"missing {path}" for path in missing),
    )


def run_command(name: str, command: list[str], cwd: Path, env: dict[str, str], tail_chars: int, quiet: bool) -> StepResult:
    """Run a command and capture combined output with bounded tail size."""
    start = time.perf_counter()
    started_at = now_iso()
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
        return StepResult(name, command, str(cwd), started_at, now_iso(), time.perf_counter() - start, 127, stderr_tail=str(exc))
    assert process.stdout is not None
    for line in process.stdout:
        tail.append(line)
        while sum(len(item) for item in tail) > tail_chars and tail:
            tail.popleft()
        if not quiet:
            print("    " + line.rstrip())
    code = process.wait()
    return StepResult(name, command, str(cwd), started_at, now_iso(), time.perf_counter() - start, code, stdout_tail="".join(tail))


def read_json(path: Path) -> Any:
    """Read JSON from disk and return an error envelope on failure."""
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return {"_readError": str(exc), "_path": str(path)}


def read_jsonl(path: Path) -> list[Any]:
    """Read a JSONL file and return parsed rows plus parse errors."""
    if not path.exists():
        return []
    out: list[Any] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError as exc:
            out.append({"_readError": str(exc), "_line": line_number, "_path": str(path)})
    return out


def read_artifacts(loop_output: Path, fine_tuning_output: Path, release_bake_manifest: Path) -> LoopArtifacts:
    """Load improve-loop artifacts used by summary and dashboard rendering."""
    gaps_payload = read_json(loop_output / "loop_gaps.json")
    gaps = gaps_payload.get("gaps", []) if isinstance(gaps_payload, dict) else []
    return LoopArtifacts(
        state=read_json(loop_output / "loop_state.json") if isinstance(read_json(loop_output / "loop_state.json"), dict) else {},
        gaps=[gap for gap in gaps if isinstance(gap, dict)],
        next_prompts=[item for item in read_jsonl(loop_output / "next_action_prompts.jsonl") if isinstance(item, dict)],
        testflight_scenarios=[item for item in read_jsonl(loop_output / "testflight_scenarios.jsonl") if isinstance(item, dict)],
        release_bake_manifest=read_json(release_bake_manifest) if isinstance(read_json(release_bake_manifest), dict) else {},
        adapter_runtime_manifest=read_json(fine_tuning_output / "adapter_runtime_manifest.json") if isinstance(read_json(fine_tuning_output / "adapter_runtime_manifest.json"), dict) else {},
    )


def escape(value: Any) -> str:
    """HTML-escape any value for safe inclusion in generated markup."""
    return html.escape(str(value), quote=True)


def shorten(value: str, max_len: int = 120) -> str:
    """Collapse whitespace and truncate text for compact display."""
    value = re.sub(r"\s+", " ", value).strip()
    return value if len(value) <= max_len else value[: max_len - 1] + "…"


def normalize_html_attribute_quotes(markup: str) -> str:
    """Normalize single-quoted tag attributes to double quotes for HTMLHint."""
    tag_pattern = re.compile(r"<[^>]+>")
    attr_pattern = re.compile(r"([A-Za-z_:][\w:.-]*)='([^']*)'")

    def _normalize_tag(tag_match: re.Match[str]) -> str:
        tag = tag_match.group(0)
        return attr_pattern.sub(r'\1="\2"', tag)

    return tag_pattern.sub(_normalize_tag, markup)


def svg_bar_chart(data: dict[str, Any], title: str) -> str:
    """Render a compact horizontal bar chart as inline SVG."""
    numeric: dict[str, float] = {}
    for key, value in data.items():
        try:
            numeric[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    if not numeric:
        return "<p>No data.</p>"
    items = sorted(numeric.items(), key=lambda item: item[1], reverse=True)[:24]
    width = 880
    left = 245
    bar_width = 520
    row_h = 28
    height = 52 + row_h * len(items)
    max_v = max(value for _, value in items) or 1.0
    rows: list[str] = []
    for idx, (label, value) in enumerate(items):
        y = 38 + idx * row_h
        w = max(2, int((value / max_v) * bar_width))
        rows.append(f"<text x='12' y='{y + 15}' fill='#8ea0bd' font-size='12'>{escape(shorten(label, 34))}</text>")
        rows.append(f"<rect x='{left}' y='{y}' width='{bar_width}' height='18' rx='9' fill='#101827'/>")
        rows.append(f"<rect x='{left}' y='{y}' width='{w}' height='18' rx='9' fill='#73a7ff'/>")
        rows.append(f"<text x='{left + bar_width + 12}' y='{y + 15}' fill='#e8eefc' font-size='12'>{escape(int(value) if value.is_integer() else round(value, 2))}</text>")
    svg = (
        f"<svg viewBox='0 0 {width} {height}' width='100%' "
        f"xmlns='http://www.w3.org/2000/svg'><text x='12' y='23' fill='#e8eefc' "
        f"font-size='16' font-weight='700'>{escape(title)}</text>{''.join(rows)}</svg>"
    )
    return normalize_html_attribute_quotes(svg)


def pipeline_svg(steps: list[StepResult], artifacts: LoopArtifacts) -> str:
    """Render the command pipeline and outcomes as inline SVG."""
    row_h = 78
    width = 1120
    height = 90 + max(1, len(steps)) * row_h
    hard = sum(1 for gap in artifacts.gaps if str(gap.get("severity")) in {"critical", "error"})
    rows: list[str] = []
    for idx, step in enumerate(steps):
        y = 58 + idx * row_h
        colour = "#44d483" if step.passed else "#ff6b7a"
        rows.append(f"<rect x='32' y='{y}' width='1056' height='56' rx='15' fill='#111722' stroke='{colour}' stroke-width='2'/>")
        rows.append(f"<text x='55' y='{y + 24}' fill='#e8eefc' font-size='16' font-weight='700'>{idx + 1}. {escape(step.name)}</text>")
        rows.append(f"<text x='55' y='{y + 44}' fill='#8ea0bd' font-size='12'>{escape(step.status)} · {step.duration_seconds:.1f}s · {escape(shorten(shlex.join(step.command), 100) if step.command else 'internal')}</text>")
    svg = (
        f"<svg viewBox='0 0 {width} {height}' width='100%' "
        f"xmlns='http://www.w3.org/2000/svg'><rect width='100%' height='100%' "
        f"rx='20' fill='#090b10'/><text x='32' y='34' fill='#e8eefc' font-size='24' "
        f"font-weight='800'>Improve-loop pipeline</text><text x='32' y='53' "
        f"fill='#8ea0bd' font-size='13'>hard gaps: {hard} · total gaps: "
        f"{len(artifacts.gaps)} · scenarios: {len(artifacts.testflight_scenarios)}"
        f"</text>{''.join(rows)}</svg>"
    )
    return normalize_html_attribute_quotes(svg)


def build_summary(root: Path, output: Path, loop_output: Path, fine_tuning_output: Path, args: argparse.Namespace, started_at: str, ended_at: str, steps: list[StepResult], artifacts: LoopArtifacts) -> dict[str, Any]:
    """Build the machine-readable summary payload for the run."""
    state = artifacts.state
    dataset = state.get("dataset", {}) if isinstance(state.get("dataset"), dict) else {}
    manifest = state.get("manifest", {}) if isinstance(state.get("manifest"), dict) else {}
    runtime = state.get("runtime", {}) if isinstance(state.get("runtime"), dict) else {}
    testflight = state.get("testFlight", {}) if isinstance(state.get("testFlight"), dict) else {}
    severity = Counter(str(gap.get("severity") or "unknown") for gap in artifacts.gaps)
    hard_gaps = [gap for gap in artifacts.gaps if str(gap.get("severity")) in {"critical", "error"}]
    failed_steps = [step for step in steps if not step.passed and not step.skipped]
    return {
        "schemaVersion": "2.0.0",
        "startedAt": started_at,
        "endedAt": ended_at,
        "root": str(root),
        "output": str(output),
        "loopOutput": str(loop_output),
        "fineTuningOutput": str(fine_tuning_output),
        "passed": not hard_gaps and not failed_steps,
        "releaseBakeRequested": bool(args.release_bake),
        "manifest": manifest,
        "dataset": dataset,
        "runtime": runtime,
        "testFlight": testflight,
        "gaps": {"total": len(artifacts.gaps), "hard": len(hard_gaps), "bySeverity": dict(sorted(severity.items()))},
        "nextPrompts": len(artifacts.next_prompts),
        "testFlightScenarios": len(artifacts.testflight_scenarios),
        "adapterRuntimeManifest": {
            "mode": artifacts.adapter_runtime_manifest.get("mode"),
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


def build_html(summary: dict[str, Any], steps: list[StepResult], artifacts: LoopArtifacts) -> str:
    """Build the visual HTML dashboard document."""
    dataset = summary.get("dataset", {}) if isinstance(summary.get("dataset"), dict) else {}
    manifest = summary.get("manifest", {}) if isinstance(summary.get("manifest"), dict) else {}
    gaps = summary.get("gaps", {}) if isinstance(summary.get("gaps"), dict) else {}
    families = dataset.get("families", {}) if isinstance(dataset.get("families"), dict) else {}
    agent_ft = dataset.get("agentFineTuning", {}) if isinstance(dataset.get("agentFineTuning"), dict) else {}
    agent_counts = {agent: sum(int(value) for value in counts.values() if isinstance(value, int)) for agent, counts in agent_ft.items() if isinstance(counts, dict)}
    cards = [
        ("status", "PASS" if summary.get("passed") else "NEEDS WORK"),
        ("tools", manifest.get("toolCount", 0)),
        ("intents", manifest.get("intentCount", 0)),
        ("dataset records", dataset.get("recordCount", 0)),
        ("gaps", gaps.get("total", 0)),
        ("TestFlight scenarios", summary.get("testFlightScenarios", 0)),
        ("next prompts", summary.get("nextPrompts", 0)),
    ]
    card_html = "".join(f"<section class='card'><span>{escape(label)}</span><strong>{escape(value)}</strong></section>" for label, value in cards)
    step_rows = "".join(
        f"<tr><td>{escape(step.name)}</td><td>{escape(step.status)}</td><td>{step.duration_seconds:.1f}s</td><td><code>{escape(shlex.join(step.command) if step.command else 'internal')}</code></td></tr>"
        for step in steps
    )
    gap_rows = "".join(
        f"<tr><td>{escape(gap.get('severity', 'unknown'))}</td><td>{escape(gap.get('category', ''))}</td><td>{escape(gap.get('title', ''))}</td><td>{escape(gap.get('recommendedAction', ''))}</td></tr>"
        for gap in artifacts.gaps[:120]
    ) or "<tr><td colspan='4'>No gaps detected.</td></tr>"
    prompt_rows = "".join(
        f"<tr><td>{escape(prompt.get('taskType', ''))}</td><td>{escape(prompt.get('priority', ''))}</td><td><code>{escape(prompt.get('id', ''))}</code></td></tr>"
        for prompt in artifacts.next_prompts[:120]
    ) or "<tr><td colspan='3'>No next prompts.</td></tr>"
    tails = "".join(f"<details><summary>{escape(step.name)} · {escape(step.status)}</summary><pre>{escape(step.stdout_tail or step.stderr_tail)}</pre></details>" for step in steps if step.stdout_tail or step.stderr_tail)
    html_doc = f"""<!doctype html>
<html lang='en'>
<head>
<meta charset='utf-8'>
<meta name='viewport' content='width=device-width, initial-scale=1'>
<title>Lumen Visual Improve-Loop v2</title>
<style>
:root {{ color-scheme: dark; --bg:#090b10; --panel:#111722; --line:#27364f; --text:#e8eefc; --muted:#8ea0bd; --blue:#73a7ff; --red:#ff6b7a; --green:#44d483; }}
body {{ margin:0; background:radial-gradient(circle at top left,#17233a,var(--bg) 42rem); color:var(--text); font-family:ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif; }}
header, main {{ padding:2rem; }}
header {{ border-bottom:1px solid var(--line); }}
h1 {{ margin:0; font-size:clamp(1.8rem,3vw,3rem); letter-spacing:-.04em; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(165px,1fr)); gap:1rem; }}
.card, .panel {{ background:rgba(17,23,34,.88); border:1px solid var(--line); border-radius:18px; padding:1rem; box-shadow:0 18px 38px rgba(0,0,0,.22); }}
.card span {{ display:block; color:var(--muted); text-transform:uppercase; font-size:.72rem; letter-spacing:.11em; }}
.card strong {{ display:block; margin-top:.5rem; font-size:1.65rem; }}
main {{ display:grid; gap:1.25rem; }}
table {{ width:100%; border-collapse:collapse; }}
th,td {{ padding:.65rem; border-bottom:1px solid var(--line); vertical-align:top; text-align:left; }}
th {{ color:var(--muted); text-transform:uppercase; font-size:.78rem; }}
pre,code {{ font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; }}
pre {{ white-space:pre-wrap; max-height:24rem; overflow:auto; background:#070b12; border:1px solid var(--line); border-radius:12px; padding:1rem; }}
.two {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(340px,1fr)); gap:1.25rem; }}
</style>
</head>
<body>
<header><h1>Lumen Visual Improve-Loop v2</h1><p>Repo-rooted, adapter-first, release-bake explicit.</p></header>
<main>
<section class='grid'>{card_html}</section>
<section class='panel'><h2>Pipeline</h2>{pipeline_svg(steps, artifacts)}</section>
<section class='two'><section class='panel'><h2>Dataset records</h2>{svg_bar_chart(families, 'Dataset family records')}</section><section class='panel'><h2>Gap severities</h2>{svg_bar_chart(gaps.get('bySeverity', {}) if isinstance(gaps.get('bySeverity'), dict) else {}, 'Gap severities')}</section></section>
<section class='panel'><h2>Agent fine-tuning records</h2>{svg_bar_chart(agent_counts, 'Agent fine-tuning records')}</section>
<section class='panel'><h2>Steps</h2><table><thead><tr><th>Step</th><th>Status</th><th>Time</th><th>Command</th></tr></thead><tbody>{step_rows}</tbody></table></section>
<section class='panel'><h2>Gaps</h2><table><thead><tr><th>Severity</th><th>Category</th><th>Title</th><th>Action</th></tr></thead><tbody>{gap_rows}</tbody></table></section>
<section class='panel'><h2>Next action prompts</h2><table><thead><tr><th>Task</th><th>Priority</th><th>ID</th></tr></thead><tbody>{prompt_rows}</tbody></table></section>
<section class='two'><section class='panel'><h2>Adapter runtime manifest</h2><pre>{escape(json.dumps(summary.get('adapterRuntimeManifest', {}), ensure_ascii=False, indent=2, sort_keys=True))}</pre></section><section class='panel'><h2>Release bake manifest</h2><pre>{escape(json.dumps(summary.get('releaseBakeManifest', {}), ensure_ascii=False, indent=2, sort_keys=True))}</pre></section></section>
<section class='panel'><h2>Command output tails</h2>{tails or '<p>No command output captured.</p>'}</section>
</main>
</body>
</html>
"""
    return normalize_html_attribute_quotes(html_doc)


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
    """Write summary JSON, dashboard HTML, and pipeline SVG to disk."""
    dashboard_dir.mkdir(parents=True, exist_ok=True)
    summary = build_summary(root, output, loop_output, fine_tuning_output, args, started_at, ended_at, steps, artifacts)
    html_doc = build_html(summary, steps, artifacts)
    svg_doc = pipeline_svg(steps, artifacts)
    summary_path = dashboard_dir / "visual_improve_loop_summary.json"
    html_path = dashboard_dir / "index.html"
    svg_path = dashboard_dir / "pipeline.svg"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    html_path.write_text(html_doc, encoding="utf-8")
    svg_path.write_text(svg_doc, encoding="utf-8")
    return {"summary": summary_path, "html": html_path, "svg": svg_path}


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for the visual improve-loop runner."""
    args = parse_args(argv)
    root = args.root.resolve()
    output = rooted_path(root, args.output)
    loop_output = rooted_path(root, args.loop_output)
    fine_tuning_output = rooted_path(root, args.fine_tuning_output)
    dashboard_output = rooted_path(root, args.dashboard_output)
    release_manifest = rooted_path(root, args.release_bake_manifest_output)
    console = Console(quiet=False)
    started_at = now_iso()

    for path in (output, loop_output, fine_tuning_output, dashboard_output, release_manifest.parent):
        path.mkdir(parents=True, exist_ok=True)

    console.info(f"repo: {root}")
    console.info(f"manifest output: {output}")
    console.info(f"loop output: {loop_output}")
    console.info(f"fine-tuning output: {fine_tuning_output}")
    console.info(f"dashboard output: {dashboard_output}")

    steps: list[StepResult] = []
    preflight = run_preflight(root)
    steps.append(preflight)
    if not preflight.passed and not args.keep_going:
        artifacts = read_artifacts(loop_output, fine_tuning_output, release_manifest)
        write_visual_outputs(root=root, dashboard_dir=dashboard_output, output=output, loop_output=loop_output, fine_tuning_output=fine_tuning_output, args=args, started_at=started_at, ended_at=now_iso(), steps=steps, artifacts=artifacts)
        return 2

    runtime_audits = collect_runtime_audits(args, root, console)
    commands = build_command_queue(args, root, output, loop_output, fine_tuning_output, runtime_audits)
    total = len(commands) + 2
    for idx, spec in enumerate(commands, start=2):
        console.step(idx, total, spec["name"])
        result = run_command(spec["name"], spec["command"], root, build_env(root), args.tail_chars, args.quiet_commands)
        steps.append(result)
        if result.passed:
            console.ok(f"{result.name}: passed")
        else:
            console.fail(f"{result.name}: failed with {result.returncode}")
            if not args.keep_going:
                break

    console.step(total, total, "visual dashboard")
    artifacts = read_artifacts(loop_output, fine_tuning_output, release_manifest)
    files = write_visual_outputs(root=root, dashboard_dir=dashboard_output, output=output, loop_output=loop_output, fine_tuning_output=fine_tuning_output, args=args, started_at=started_at, ended_at=now_iso(), steps=steps, artifacts=artifacts)
    console.ok(f"dashboard: {files['html']}")
    console.ok(f"summary: {files['summary']}")
    console.ok(f"pipeline svg: {files['svg']}")
    if args.open_dashboard:
        webbrowser.open(files["html"].as_uri())

    hard_gaps = [gap for gap in artifacts.gaps if str(gap.get("severity")) in {"critical", "error"}]
    failed_steps = [step for step in steps if not step.passed and not step.skipped]
    if failed_steps:
        return 1
    if args.fail_on_gaps and hard_gaps:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
