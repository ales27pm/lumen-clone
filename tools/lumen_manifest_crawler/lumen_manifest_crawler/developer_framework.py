"""Developer framework orchestration for Lumen.

This module turns the developer improve framework into executable, whitelisted
local jobs and a consolidated status snapshot. It intentionally reuses existing
scripts instead of duplicating crawler, training, visual, or Hugging Face logic.
"""

from __future__ import annotations

import html
import json
import os
import platform
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
from typing import Any, Iterable, Sequence

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


def resolve_environment(value: FrameworkEnvironment | str = FrameworkEnvironment.AUTO) -> FrameworkEnvironment:
    raw = value.value if isinstance(value, FrameworkEnvironment) else str(value)
    if raw != FrameworkEnvironment.AUTO.value:
        return FrameworkEnvironment(raw)
    system = platform.system().casefold()
    if system == "darwin":
        return FrameworkEnvironment.MACOS
    return FrameworkEnvironment.UBUNTU if system == "linux" else FrameworkEnvironment.MACOS


def build_framework_jobs(root: Path, environment: FrameworkEnvironment | str = FrameworkEnvironment.AUTO) -> list[FrameworkJob]:
    env = resolve_environment(environment)
    py = sys.executable
    common: list[FrameworkJob] = [
        FrameworkJob(
            id="status",
            title="Framework status",
            environment=env,
            evidence_layer=EvidenceLayer.LOCAL_VALIDATION,
            command=(py, "-m", "lumen_manifest_crawler", "framework", "status", "--root", str(root)),
            description="Print consolidated framework state.",
        ),
        FrameworkJob(
            id="ingest-runtime",
            title="Ingest runtime exports",
            environment=env,
            evidence_layer=EvidenceLayer.DEVICE_RUNTIME,
            command=(
                py,
                "-m",
                "lumen_manifest_crawler",
                "improve-loop",
                "--root",
                str(root),
                "--runtime-audit",
                str(root / "exports"),
            ),
            description="Run improve-loop with repo exports as runtime audit input.",
            outputs=("generated/agent_improvement_loop/loop_state.json",),
        ),
    ]

    if env == FrameworkEnvironment.UBUNTU:
        return common + [
            FrameworkJob(
                id="ubuntu-preflight",
                title="Ubuntu training preflight",
                environment=env,
                evidence_layer=EvidenceLayer.LOCAL_VALIDATION,
                command=(py, "tools/lumen_terminal_improve_loop.py", "--mode", "preflight", "--dry-run", "--skip-pytest"),
                description="Check adapter runtime invariants and Qwen3 training config readiness.",
            ),
            FrameworkJob(
                id="train-adapters",
                title="Train role LoRA adapters",
                environment=env,
                evidence_layer=EvidenceLayer.TRAINING_FEEDBACK,
                command=(py, "tools/lumen_terminal_improve_loop.py", "--mode", "train", "--resume", "--assistant-only-loss"),
                description="Train role adapters from generated fine-tuning datasets.",
                outputs=("models/lora_qwen3_bootstrap",),
                requires_confirmation=True,
            ),
            FrameworkJob(
                id="convert-adapters",
                title="Convert LoRA adapters to GGUF",
                environment=env,
                evidence_layer=EvidenceLayer.TRAINING_FEEDBACK,
                command=(py, "tools/lumen_terminal_improve_loop.py", "--mode", "convert", "--resume", "--base-model-id", "Qwen/Qwen3-1.7B"),
                description="Convert trained LoRA adapters to GGUF with an explicit base.",
                outputs=("models/lora_qwen3_gguf",),
                requires_confirmation=True,
            ),
            FrameworkJob(
                id="hf-resolve",
                title="Resolve Hugging Face artifact manifest",
                environment=env,
                evidence_layer=EvidenceLayer.TRAINING_FEEDBACK,
                command=(py, "tools/hf_artifacts/publish_hf_artifacts.py", "--skip-upload"),
                description="Write resolved HF artifact manifest without uploading.",
                outputs=("generated/hf_artifacts/lumen_hf_artifact_manifest.resolved.json",),
            ),
            FrameworkJob(
                id="hf-upload-dry-run",
                title="Dry-run Hugging Face upload",
                environment=env,
                evidence_layer=EvidenceLayer.TRAINING_FEEDBACK,
                command=(py, "tools/hf_artifacts/publish_hf_artifacts.py", "--dry-run"),
                description="Validate and print HF uploads without uploading.",
            ),
        ]

    return common + [
        FrameworkJob(
            id="adapter-invariants",
            title="Adapter invariant check",
            environment=env,
            evidence_layer=EvidenceLayer.LOCAL_VALIDATION,
            command=(py, "tools/check_adapter_runtime_invariants.py"),
            description="Verify Qwen3 adapter-first runtime and training invariants.",
        ),
        FrameworkJob(
            id="build-readiness",
            title="iOS build readiness",
            environment=env,
            evidence_layer=EvidenceLayer.LOCAL_VALIDATION,
            command=("scripts/check-ios-build-readiness.sh",),
            description="Run static iOS readiness checks and Xcode availability reporting.",
        ),
        FrameworkJob(
            id="improve-loop",
            title="Generate manifest and improve-loop artifacts",
            environment=env,
            evidence_layer=EvidenceLayer.STATIC_SOURCE,
            command=(
                py,
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
            description="Generate manifest, datasets, gaps, prompts, and TestFlight runbook.",
            outputs=("generated/agent_manifest", "generated/agent_improvement_loop"),
        ),
        FrameworkJob(
            id="visual-dashboard",
            title="Generate visual dashboard",
            environment=env,
            evidence_layer=EvidenceLayer.LOCAL_VALIDATION,
            command=(py, "tools/run_visual_improve_loop_v2.py", "--root", str(root), "--skip-tests"),
            description="Run visual improve-loop dashboard generation.",
            outputs=("generated/visual_improve_loop/index.html",),
        ),
    ]


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
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        lower = text.casefold()
        if "traceback" in lower:
            plain_findings.append({"severity": "error", "type": "python_traceback", "source": str(path)})
        if "xcodebuild" in lower and ("error:" in lower or "build failed" in lower):
            plain_findings.append({"severity": "error", "type": "xcodebuild_failure", "source": str(path)})
        if "hugging face" in lower and ("error" in lower or "failed" in lower):
            plain_findings.append({"severity": "warning", "type": "hf_upload_failure", "source": str(path)})
        if "no model loaded" in lower or "routing-only checks completed" in lower:
            plain_findings.append({"severity": "error", "type": "invalid_live_e2e_no_model", "source": str(path)})
    return {
        "schemaVersion": "lumen.developer_framework.report_analysis/1.0.0",
        "reportCount": len(reports),
        "runtimeFailureCount": len(failures),
        "runtimeFailures": failures[:100],
        "plainFindings": plain_findings,
    }


def _iter_report_files(paths: Iterable[Path]) -> Iterable[Path]:
    for path in paths:
        if path.is_dir():
            yield from (candidate for candidate in path.rglob("*") if candidate.is_file())
        elif path.is_file():
            yield path


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
        env = os.environ.copy()
        crawler_root = str((self.root / "tools/lumen_manifest_crawler").resolve())
        env["PYTHONPATH"] = crawler_root if not env.get("PYTHONPATH") else f"{crawler_root}{os.pathsep}{env['PYTHONPATH']}"
        env.setdefault("PYTHONUNBUFFERED", "1")
        self._append("$ " + shlex.join(job.command))
        try:
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
    completed = subprocess.run(list(job.command), cwd=root, check=False)
    return int(completed.returncode)


def serve_framework(root: Path, host: str, port: int, environment: FrameworkEnvironment | str = FrameworkEnvironment.AUTO, *, open_browser: bool = False) -> int:
    env = resolve_environment(environment)
    runner = FrameworkJobRunner(root, env)

    class Handler(BaseHTTPRequestHandler):
        server_version = "LumenDeveloperFrameworkHTTP/1.0"

        def do_GET(self) -> None:  # noqa: N802
            if self.path in {"/", "/index.html"}:
                self._send_html(_index_html(root, env))
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


def _index_html(root: Path, env: FrameworkEnvironment) -> str:
    escaped_root = html.escape(str(root.resolve()))
    escaped_env = html.escape(env.value)
    return f"""<!doctype html>
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
<p class="muted">root: {escaped_root} · environment: {escaped_env}</p>
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
</main>
<script>
async function runJob(id) {{
  const res = await fetch('/run/' + encodeURIComponent(id), {{ method: 'POST' }});
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
}}
setInterval(refresh, 1500);
refresh();
</script>
</body>
</html>"""
