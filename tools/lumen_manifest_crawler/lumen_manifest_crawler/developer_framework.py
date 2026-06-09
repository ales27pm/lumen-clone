"""Developer framework orchestration for Lumen.

This module turns the developer improve framework into executable, whitelisted
local jobs and a consolidated status snapshot. It intentionally reuses existing
scripts instead of duplicating crawler, training, visual, or Hugging Face logic.
"""

from __future__ import annotations

import html
import ipaddress
import json
import os
import platform
import secrets
import shlex
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from lumen_manifest_crawler.dataset.runtime_ingest import load_runtime_audit_reports


class FrameworkEnvironment(str, Enum):
    AUTO = "auto"
    MACOS = "macos"
    UBUNTU = "ubuntu"


class EvidenceLayer(str, Enum):
    STATIC_SOURCE = "static_source"
    LOCAL_VALIDATION = "local_validation"
    SIMULATOR_VALIDATION = "simulator_validation"
    DEVICE_RUNTIME = "device_runtime"
    LIVE_E2E = "live_e2e"
    TRAINING_FEEDBACK = "training_feedback"


@dataclass(frozen=True)
class FrameworkJob:
    id: str
    title: str
    environment: FrameworkEnvironment
    evidence_layer: EvidenceLayer
    command: tuple[str, ...]
    description: str
    outputs: tuple[str, ...] = ()
    requires_confirmation: bool = False

    def output_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "environment": self.environment.value,
            "evidenceLayer": self.evidence_layer.value,
            "command": list(self.command),
            "description": self.description,
            "outputs": list(self.outputs),
            "requiresConfirmation": self.requires_confirmation,
        }


@dataclass
class FrameworkJobState:
    job_id: str | None = None
    status: str = "idle"
    started_at: float | None = None
    ended_at: float | None = None
    returncode: int | None = None
    command: list[str] = field(default_factory=list)
    log: deque[str] = field(default_factory=lambda: deque(maxlen=2000))

    def output_dict(self) -> dict[str, Any]:
        return {
            "jobID": self.job_id,
            "status": self.status,
            "startedAt": self.started_at,
            "endedAt": self.ended_at,
            "durationSeconds": None if self.started_at is None else round((self.ended_at or time.time()) - self.started_at, 2),
            "returncode": self.returncode,
            "command": self.command,
            "log": list(self.log),
        }


JobDefinition = Mapping[str, Any]

UBUNTU_TRAINING_JOB_IDS = (
    "ubuntu-preflight",
    "train-adapters",
    "convert-adapters",
    "hf-resolve",
    "hf-upload-dry-run",
)

AGENT_ADAPTER_ROLES = ("cortex", "executor", "mouth", "mimicry", "rem", "fleet")


def resolve_environment(value: FrameworkEnvironment | str = FrameworkEnvironment.AUTO) -> FrameworkEnvironment:
    raw = value.value if isinstance(value, FrameworkEnvironment) else str(value)
    if raw != FrameworkEnvironment.AUTO.value:
        return FrameworkEnvironment(raw)
    system = platform.system().casefold()
    if system == "darwin":
        return FrameworkEnvironment.MACOS
    return FrameworkEnvironment.UBUNTU if system == "linux" else FrameworkEnvironment.MACOS


def _make_jobs(root: Path, env: FrameworkEnvironment, definitions: Sequence[JobDefinition]) -> list[FrameworkJob]:
    py = sys.executable
    jobs: list[FrameworkJob] = []
    for definition in definitions:
        raw_command = tuple(definition["command"])
        command = (py, *raw_command[1:]) if raw_command and raw_command[0] == "PYTHON" else raw_command
        jobs.append(
            FrameworkJob(
                id=str(definition["id"]),
                title=str(definition["title"]),
                environment=env,
                evidence_layer=definition["evidence_layer"],
                command=tuple(str(part) for part in command),
                description=str(definition["description"]),
                outputs=tuple(str(output) for output in definition.get("outputs", ())),
                requires_confirmation=bool(definition.get("requires_confirmation", False)),
            )
        )
    return jobs


def build_framework_jobs(root: Path, environment: FrameworkEnvironment | str = FrameworkEnvironment.AUTO) -> list[FrameworkJob]:
    env = resolve_environment(environment)
    common_defs: list[JobDefinition] = [
        {
            "id": "status",
            "title": "Framework status",
            "evidence_layer": EvidenceLayer.LOCAL_VALIDATION,
            "command": ("PYTHON", "-m", "lumen_manifest_crawler", "framework", "status", "--root", str(root)),
            "description": "Print consolidated framework state.",
        },
        {
            "id": "ingest-runtime",
            "title": "Ingest runtime exports",
            "evidence_layer": EvidenceLayer.DEVICE_RUNTIME,
            "command": (
                "PYTHON",
                "-m",
                "lumen_manifest_crawler",
                "improve-loop",
                "--root",
                str(root),
                "--runtime-audit",
                str(root / "exports"),
            ),
            "description": "Run improve-loop with repo exports as runtime audit input.",
            "outputs": ("generated/agent_improvement_loop/loop_state.json",),
        },
    ]

    if env == FrameworkEnvironment.UBUNTU:
        env_defs: list[JobDefinition] = [
            {
                "id": "ubuntu-preflight",
                "title": "Ubuntu training preflight",
                "evidence_layer": EvidenceLayer.LOCAL_VALIDATION,
                "command": ("PYTHON", "tools/lumen_terminal_improve_loop.py", "--mode", "preflight", "--dry-run", "--skip-pytest"),
                "description": "Check adapter runtime invariants and Qwen3 training config readiness.",
            },
            {
                "id": "train-adapters",
                "title": "Train role LoRA adapters",
                "evidence_layer": EvidenceLayer.TRAINING_FEEDBACK,
                "command": ("PYTHON", "tools/lumen_terminal_improve_loop.py", "--mode", "train", "--resume", "--assistant-only-loss"),
                "description": "Train role adapters from generated fine-tuning datasets.",
                "outputs": ("models/lora_qwen3_bootstrap",),
                "requires_confirmation": True,
            },
            {
                "id": "convert-adapters",
                "title": "Convert LoRA adapters to GGUF",
                "evidence_layer": EvidenceLayer.TRAINING_FEEDBACK,
                "command": ("PYTHON", "tools/lumen_terminal_improve_loop.py", "--mode", "convert", "--resume", "--base-model-id", "Qwen/Qwen3-1.7B"),
                "description": "Convert trained LoRA adapters to GGUF with an explicit base.",
                "outputs": ("models/lora_qwen3_gguf",),
                "requires_confirmation": True,
            },
            {
                "id": "hf-resolve",
                "title": "Resolve Hugging Face artifact manifest",
                "evidence_layer": EvidenceLayer.TRAINING_FEEDBACK,
                "command": ("PYTHON", "tools/hf_artifacts/publish_hf_artifacts.py", "--skip-upload"),
                "description": "Write resolved HF artifact manifest without uploading.",
                "outputs": ("generated/hf_artifacts/lumen_hf_artifact_manifest.resolved.json",),
            },
            {
                "id": "hf-upload-dry-run",
                "title": "Dry-run Hugging Face upload",
                "evidence_layer": EvidenceLayer.TRAINING_FEEDBACK,
                "command": ("PYTHON", "tools/hf_artifacts/publish_hf_artifacts.py", "--dry-run"),
                "description": "Validate and print HF uploads without uploading.",
            },
        ]
        return _make_jobs(root, env, common_defs + env_defs)

    env_defs = [
        {
            "id": "adapter-invariants",
            "title": "Adapter invariant check",
            "evidence_layer": EvidenceLayer.LOCAL_VALIDATION,
            "command": ("PYTHON", "tools/check_adapter_runtime_invariants.py"),
            "description": "Verify Qwen3 adapter-first runtime and training invariants.",
        },
        {
            "id": "build-readiness",
            "title": "iOS build readiness",
            "evidence_layer": EvidenceLayer.LOCAL_VALIDATION,
            "command": ("scripts/check-ios-build-readiness.sh",),
            "description": "Run static iOS readiness checks and Xcode availability reporting.",
        },
        {
            "id": "improve-loop",
            "title": "Generate manifest and improve-loop artifacts",
            "evidence_layer": EvidenceLayer.STATIC_SOURCE,
            "command": (
                "PYTHON",
                "-m",
                "lumen_manifest_crawler",
                "improve-loop",
                "--root",
                str(root),
                "--output",
                str(root / "generated/agent_manifest"),
                "--loop-output",
                str(root / "generated/agent_improvement_loop"),
            ),
            "description": "Generate manifest, datasets, gaps, prompts, and TestFlight runbook.",
            "outputs": ("generated/agent_manifest", "generated/agent_improvement_loop"),
        },
        {
            "id": "visual-dashboard",
            "title": "Generate visual dashboard",
            "evidence_layer": EvidenceLayer.LOCAL_VALIDATION,
            "command": ("PYTHON", "tools/run_visual_improve_loop_v2.py", "--root", str(root), "--skip-tests"),
            "description": "Run visual improve-loop dashboard generation.",
            "outputs": ("generated/visual_improve_loop/index.html",),
        },
    ]
    return _make_jobs(root, env, common_defs + env_defs)


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _read_jsonl(path: Path, limit: int = 50) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if len(records) >= limit:
                    break
                line = line.strip()
                if not line:
                    continue
                value = json.loads(line)
                if isinstance(value, dict):
                    records.append(value)
    except (OSError, json.JSONDecodeError):
        return []
    return records


def adapter_runtime_contract() -> dict[str, Any]:
    return {
        "schemaVersion": "lumen.agent_adapter_runtime/1.0.0",
        "modelFamily": "qwen3",
        "runtimeShape": "one_shared_chat_base_plus_role_lora_adapters",
        "sharedChatBase": {
            "repoID": "ales27pm/lumen-qwen3-bootstrap-gguf",
            "fileName": "lumen-qwen3-fast-shared-q4_k_m.gguf",
            "baseModelID": "Qwen/Qwen3-1.7B",
            "loadPolicy": "load_once",
        },
        "embeddingModel": {
            "repoID": "Qwen/Qwen3-Embedding-0.6B-GGUF",
            "fileName": "Qwen3-Embedding-0.6B-Q8_0.gguf",
            "uses": ["source_map", "tool_schema", "memory", "rag", "repair_retrieval"],
        },
        "adapterRepoID": "ales27pm/lumen-qwen3-bootstrap-adapters-gguf",
        "roles": [
            {
                "id": role,
                "adapterFile": f"lumen-{role}-lora.gguf",
                "trainingOutputDir": f"models/lora/{role}",
                "convertedOutput": f"models/lora_qwen3_gguf/lumen-{role}-lora.gguf",
                "promptBinding": "systemPrompt",
            }
            for role in AGENT_ADAPTER_ROLES
        ],
        "workflow": [
            "compile_role_datasets",
            "train_lora_adapters",
            "validate_role_evals",
            "convert_lora_to_gguf",
            "upload_hf_artifacts",
            "ship_testflight",
            "export_runtime_traces",
            "ingest_gaps_and_repairs",
        ],
        "invariants": [
            "do_not_train_or_ship_six_role_baked_full_ggufs_by_default",
            "role_switches_must_not_unload_the_shared_chat_base",
            "adapter_activation_must_clear_previous_lora_adapters",
            "adapter_failures_must_be_visible_in_runtime_traces",
            "live_e2e_remains_the_only_scenario_pass_fail_owner",
        ],
        "promotionGates": [
            "manifest_only_tool_use",
            "strict_json_validity",
            "approval_boundary_correctness",
            "sentinel_suppression",
            "runtime_trace_presence",
            "latency_and_memory_budget",
            "testflight_live_e2e",
        ],
    }


def load_framework_snapshot(root: Path, environment: FrameworkEnvironment | str = FrameworkEnvironment.AUTO) -> dict[str, Any]:
    root = root.resolve()
    env = resolve_environment(environment)
    loop_output = root / "generated/agent_improvement_loop"
    visual_output = root / "generated/visual_improve_loop"
    hf_resolved = root / "generated/hf_artifacts/lumen_hf_artifact_manifest.resolved.json"
    state = _read_json(loop_output / "loop_state.json") or {}
    gaps_payload = _read_json(loop_output / "loop_gaps.json") or {}
    gaps = gaps_payload.get("gaps") if isinstance(gaps_payload, dict) else []
    gaps = gaps if isinstance(gaps, list) else []
    next_prompts = _read_jsonl(loop_output / "next_action_prompts.jsonl", limit=80)
    scenarios = _read_jsonl(loop_output / "testflight_scenarios.jsonl", limit=80)
    visual_summary = _read_json(visual_output / "visual_improve_loop_summary.json") or {}
    hf_manifest = _read_json(hf_resolved) or {}
    jobs = build_framework_jobs(root, env)
    return {
        "schemaVersion": "lumen.developer_framework/1.0.0",
        "root": str(root),
        "environment": env.value,
        "authoritativeLiveLayer": EvidenceLayer.LIVE_E2E.value,
        "evidenceLayers": [
            {
                "id": layer.value,
                "ownsScenarioPassFail": layer == EvidenceLayer.LIVE_E2E,
            }
            for layer in EvidenceLayer
        ],
        "loopState": state,
        "gaps": gaps[:80],
        "gapCount": len(gaps),
        "nextActionPrompts": next_prompts,
        "testflightScenarios": scenarios,
        "visualSummary": visual_summary,
        "hfManifest": hf_manifest,
        "adapterRuntime": adapter_runtime_contract(),
        "availableJobs": [job.output_dict() for job in jobs],
    }


def analyze_reports(root: Path, paths: Sequence[Path]) -> dict[str, Any]:
    root = root.resolve()
    resolved = [path if path.is_absolute() else root / path for path in paths]
    reports = load_runtime_audit_reports(resolved)
    failures = [
        failure
        for report in reports
        for failure in report.get("failures", [])
        if isinstance(failure, dict)
    ]
    plain_findings: list[dict[str, Any]] = []
    for path in _iter_report_files(resolved):
        plain_findings.extend(_scan_plain_text_findings(path))
    return {
        "schemaVersion": "lumen.developer_framework.report_analysis/1.0.0",
        "reportCount": len(reports),
        "runtimeFailureCount": len(failures),
        "runtimeFailures": failures[:100],
        "plainFindings": plain_findings,
    }


def _scan_plain_text_findings(path: Path) -> list[dict[str, Any]]:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    lower = text.casefold()
    findings: list[dict[str, Any]] = []
    if "traceback" in lower:
        findings.append({"severity": "error", "type": "python_traceback", "source": str(path)})
    if "xcodebuild" in lower and ("error:" in lower or "build failed" in lower):
        findings.append({"severity": "error", "type": "xcodebuild_failure", "source": str(path)})
    if _looks_like_hf_upload_failure(lower):
        findings.append({"severity": "warning", "type": "hf_upload_failure", "source": str(path)})
    if _should_scan_plain_no_model_evidence(path) and ("no model loaded" in lower or "routing-only checks completed" in lower):
        findings.append({"severity": "error", "type": "invalid_live_e2e_no_model", "source": str(path)})
    return findings


def _should_scan_plain_no_model_evidence(path: Path) -> bool:
    # JSON evidence is normalized structurally by load_runtime_audit_reports().
    # Whole-file substring scans are only appropriate for text logs; JSON
    # exports can legitimately carry policy notes that describe invalid
    # evidence phrases without any scenario having emitted them.
    return path.suffix.casefold() not in {".json"}


def _looks_like_hf_upload_failure(lower_text: str) -> bool:
    mentions_hf = "hugging face" in lower_text or "huggingface.co" in lower_text
    mentions_upload = "upload" in lower_text or "push_to_hub" in lower_text
    has_failure = (
        "error:" in lower_text
        or "[error" in lower_text
        or "upload failed" in lower_text
        or "failed to upload" in lower_text
    )
    return mentions_hf and mentions_upload and has_failure


def _iter_report_files(paths: Iterable[Path]) -> Iterable[Path]:
    for path in paths:
        if path.is_dir():
            yield from (candidate for candidate in path.rglob("*") if candidate.is_file())
        elif path.is_file():
            yield path


def _framework_job_env(root: Path) -> dict[str, str]:
    env = os.environ.copy()
    pythonpath_entries = [
        str((root / "tools/lumen_manifest_crawler").resolve()),
        str(root.resolve()),
    ]
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_entries + ([existing] if existing else []))
    env.setdefault("PYTHONUNBUFFERED", "1")
    return env


class FrameworkJobRunner:
    def __init__(self, root: Path, environment: FrameworkEnvironment) -> None:
        self.root = root.resolve()
        self.environment = environment
        self.lock = threading.Lock()
        self.state = FrameworkJobState()

    def start(self, job_id: str) -> tuple[bool, str]:
        jobs = {job.id: job for job in build_framework_jobs(self.root, self.environment)}
        job = jobs.get(job_id)
        if job is None:
            return False, f"Unknown framework job: {job_id}"
        with self.lock:
            if self.state.status == "running":
                return False, "A job is already running."
            self.state = FrameworkJobState(job_id=job.id, status="running", started_at=time.time(), command=list(job.command))
            thread = threading.Thread(target=self._run, args=(job,), daemon=True)
            thread.start()
            return True, "started"

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return self.state.output_dict()

    def _append(self, line: str) -> None:
        with self.lock:
            self.state.log.append(line.rstrip())

    def _run(self, job: FrameworkJob) -> None:
        env = _framework_job_env(self.root)
        self._append("$ " + shlex.join(job.command))
        try:
            # Security: job.command comes only from build_framework_jobs(), selected by
            # whitelisted job id, and is executed with shell=False argument vectors.
            process = subprocess.Popen(
                list(job.command),
                cwd=self.root,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                errors="replace",
            )
            assert process.stdout is not None
            for line in process.stdout:
                self._append(line)
            returncode = process.wait()
        except OSError as exc:
            self._append(f"ERROR: {exc}")
            returncode = 127
        with self.lock:
            self.state.returncode = returncode
            self.state.ended_at = time.time()
            self.state.status = "passed" if returncode == 0 else "failed"


def run_framework_job(root: Path, job_id: str, environment: FrameworkEnvironment | str = FrameworkEnvironment.AUTO) -> int:
    root = root.resolve()
    env = resolve_environment(environment)
    jobs = {job.id: job for job in build_framework_jobs(root, env)}
    job = jobs.get(job_id)
    if job is None:
        raise ValueError(f"Unknown framework job: {job_id}")
    # Security: the external job id is resolved to the internal whitelist above;
    # subprocess.run receives a shell=False argv list, not a shell command string.
    completed = subprocess.run(list(job.command), cwd=root, check=False, env=_framework_job_env(root))
    return int(completed.returncode)


def serve_framework(
    root: Path,
    host: str,
    port: int,
    environment: FrameworkEnvironment | str = FrameworkEnvironment.AUTO,
    *,
    open_browser: bool = False,
    allow_remote: bool = False,
) -> int:
    if not _is_loopback_host(host):
        if not allow_remote:
            print(
                "Refusing to bind Lumen Developer Framework to a non-loopback host. "
                "Use --allow-remote only on trusted networks."
            )
            return 2
        print(
            "WARNING: serving Lumen Developer Framework on a non-loopback host. "
            "Local developer jobs and logs may be reachable from the network."
        )
    env = resolve_environment(environment)
    runner = FrameworkJobRunner(root, env)
    csrf_token = secrets.token_urlsafe(32)

    class Handler(BaseHTTPRequestHandler):
        server_version = "LumenDeveloperFrameworkHTTP/1.0"

        def do_GET(self) -> None:  # noqa: N802
            if self.path in {"/", "/index.html"}:
                self._send_html(_index_html(root, env, csrf_token))
                return
            if self.path == "/status.json":
                payload = load_framework_snapshot(root, env)
                payload["job"] = runner.snapshot()
                self._send_json(payload)
                return
            if self.path == "/jobs.json":
                self._send_json({"jobs": [job.output_dict() for job in build_framework_jobs(root, env)]})
                return
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")

        def do_POST(self) -> None:  # noqa: N802
            prefix = "/run/"
            if self.path.startswith(prefix):
                if not secrets.compare_digest(self.headers.get("X-CSRF-Token", ""), csrf_token):
                    self._send_json({"ok": False, "message": "Invalid CSRF token."}, HTTPStatus.FORBIDDEN)
                    return
                job_id = self.path[len(prefix):]
                ok, message = runner.start(job_id)
                self._send_json({"ok": ok, "message": message, "job": runner.snapshot()}, HTTPStatus.ACCEPTED if ok else HTTPStatus.BAD_REQUEST)
                return
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")

        def log_message(self, fmt: str, *args: Any) -> None:
            sys.stderr.write("[lumen-framework] " + fmt % args + "\n")

        def _send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
            data = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _send_html(self, body: str, status: HTTPStatus = HTTPStatus.OK) -> None:
            data = body.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    server = ThreadingHTTPServer((host, port), Handler)
    url = f"http://{host}:{port}/"
    print(f"Serving Lumen Developer Framework at {url}")
    if open_browser:
        import webbrowser

        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server.")
    finally:
        server.server_close()
    return 0


def _is_loopback_host(host: str) -> bool:
    normalized = host.strip().casefold()
    if normalized == "localhost":
        return True
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        return False


_INDEX_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Lumen Developer Framework</title>
<style>
:root {{ color-scheme: dark; --bg:#0b0d10; --panel:#151922; --line:#2b3342; --text:#edf2f7; --muted:#97a3b6; --accent:#6fb1ff; --green:#4bd17b; --red:#ff6675; }}
body {{ margin:0; background:var(--bg); color:var(--text); font-family:ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif; }}
main {{ max-width:1240px; margin:0 auto; padding:20px; display:grid; gap:14px; }}
section {{ border-top:1px solid var(--line); padding-top:12px; }}
.grid {{ display:grid; grid-template-columns: 1.1fr .9fr; gap:14px; }}
.panel {{ background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:14px; }}
button {{ border:1px solid var(--line); background:#202635; color:var(--text); border-radius:7px; padding:8px 10px; cursor:pointer; margin:3px; }}
button:hover {{ border-color:var(--accent); }}
pre {{ white-space:pre-wrap; max-height:360px; overflow:auto; background:#07090d; border:1px solid var(--line); border-radius:7px; padding:10px; }}
.muted {{ color:var(--muted); }}
.ok {{ color:var(--green); }}
.bad {{ color:var(--red); }}
table {{ width:100%; border-collapse:collapse; }}
td, th {{ border-bottom:1px solid var(--line); padding:7px; text-align:left; vertical-align:top; }}
</style>
</head>
<body>
<main>
<header>
<p class="muted">root: {root} · environment: {env}</p>
<h1>Lumen Developer Framework</h1>
</header>
<div class="grid">
<div class="panel">
<h2>Whitelisted Jobs</h2>
<div id="jobs"></div>
</div>
<div class="panel">
<h2>Status</h2>
<div id="summary" class="muted">Loading…</div>
<pre id="log"></pre>
</div>
</div>
<section class="panel">
<h2>Gaps</h2>
<pre id="gaps"></pre>
</section>
<section class="panel">
<h2>Evidence Layers</h2>
<div id="evidence"></div>
</section>
<section class="panel">
<h2>Agent Adapter Runtime</h2>
<div id="adapter"></div>
</section>
</main>
<script>
const csrfToken = {csrf_token_json};
async function runJob(id) {{
  const res = await fetch('/run/' + encodeURIComponent(id), {{ method: 'POST', headers: {{ 'X-CSRF-Token': csrfToken }} }});
  const payload = await res.json();
  if (!payload.ok) alert(payload.message || 'Job failed to start');
  await refresh();
}}
function esc(value) {{
  return String(value ?? '').replace(/[&<>"']/g, c => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));
}}
async function refresh() {{
  const [statusRes, jobsRes] = await Promise.all([fetch('/status.json'), fetch('/jobs.json')]);
  const status = await statusRes.json();
  const jobsPayload = await jobsRes.json();
  const job = status.job || {{}};
  document.getElementById('summary').innerHTML = `<strong>${{esc(job.status || 'idle')}}</strong> · ${{esc(job.jobID || 'no job')}} · gaps=${{status.gapCount || 0}} · live=${{esc(status.authoritativeLiveLayer)}}`;
  document.getElementById('log').textContent = (job.log || []).join('\\n');
  document.getElementById('jobs').innerHTML = `<table><thead><tr><th>Job</th><th>Layer</th><th></th></tr></thead><tbody>${{(jobsPayload.jobs || []).map(j => `<tr><td><strong>${{esc(j.title)}}</strong><br><span class="muted">${{esc(j.description)}}</span><br><code>${{esc((j.command || []).join(' '))}}</code></td><td>${{esc(j.evidenceLayer)}}</td><td><button onclick="runJob('${{esc(j.id)}}')">Run</button></td></tr>`).join('')}}</tbody></table>`;
  document.getElementById('gaps').textContent = JSON.stringify((status.gaps || []).slice(0, 20), null, 2);
  document.getElementById('evidence').innerHTML = `<table><tbody>${{(status.evidenceLayers || []).map(l => `<tr><td>${{esc(l.id)}}</td><td>${{l.ownsScenarioPassFail ? '<span class="ok">scenario pass/fail owner</span>' : '<span class="muted">diagnostic</span>'}}</td></tr>`).join('')}}</tbody></table>`;
  const adapter = status.adapterRuntime || {{}};
  const roles = adapter.roles || [];
  document.getElementById('adapter').innerHTML = `<p><strong>${{esc(adapter.runtimeShape || 'unknown')}}</strong><br><span class="muted">${{esc(adapter.sharedChatBase?.repoID || '')}} / ${{esc(adapter.sharedChatBase?.fileName || '')}}</span></p><table><thead><tr><th>Role</th><th>Adapter</th><th>Training Output</th></tr></thead><tbody>${{roles.map(r => `<tr><td>${{esc(r.id)}}</td><td>${{esc(r.adapterFile)}}</td><td><code>${{esc(r.trainingOutputDir)}}</code></td></tr>`).join('')}}</tbody></table><p class="muted">Gates: ${{esc((adapter.promotionGates || []).join(', '))}}</p>`;
}}
setInterval(refresh, 1500);
refresh();
</script>
</body>
</html>"""


def _index_html(root: Path, env: FrameworkEnvironment, csrf_token: str = "") -> str:
    return _INDEX_TEMPLATE.format(
        root=html.escape(str(root.resolve())),
        env=html.escape(env.value),
        csrf_token_json=json.dumps(csrf_token),
    )
