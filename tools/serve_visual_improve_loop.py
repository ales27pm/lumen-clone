#!/usr/bin/env python3
"""Local web control panel for the Lumen visual improve-loop.

This is intentionally localhost-first and command-whitelisted. The browser can
trigger only the commands supplied at server startup: the visual improve-loop and
an optional training command. It does not accept arbitrary shell commands from
HTTP requests.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import shlex
import subprocess
import sys
import threading
import time
import webbrowser
from collections import deque
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Sequence

MAX_LOG_LINES = 2000


@dataclass(slots=True)
class JobState:
    name: str | None = None
    command: list[str] = field(default_factory=list)
    status: str = "idle"
    started_at: float | None = None
    ended_at: float | None = None
    returncode: int | None = None
    log: deque[str] = field(default_factory=lambda: deque(maxlen=MAX_LOG_LINES))

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "command": self.command,
            "status": self.status,
            "startedAt": self.started_at,
            "endedAt": self.ended_at,
            "durationSeconds": None if self.started_at is None else round((self.ended_at or time.time()) - self.started_at, 2),
            "returncode": self.returncode,
            "log": list(self.log),
        }


class JobRunner:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.lock = threading.Lock()
        self.job = JobState()

    def start(self, name: str, command: list[str]) -> tuple[bool, str]:
        with self.lock:
            if self.job.status == "running":
                return False, "A job is already running."
            self.job = JobState(name=name, command=command, status="running", started_at=time.time())
            thread = threading.Thread(target=self._run, args=(command,), daemon=True)
            thread.start()
            return True, "started"

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return self.job.as_dict()

    def _append(self, line: str) -> None:
        with self.lock:
            self.job.log.append(line.rstrip())

    def _run(self, command: list[str]) -> None:
        env = os.environ.copy()
        crawler_root = str((self.root / "tools" / "lumen_manifest_crawler").resolve())
        env["PYTHONPATH"] = crawler_root if not env.get("PYTHONPATH") else f"{crawler_root}{os.pathsep}{env['PYTHONPATH']}"
        env.setdefault("PYTHONUNBUFFERED", "1")
        self._append("$ " + shlex.join(command))
        try:
            process = subprocess.Popen(
                command,
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
            self.job.returncode = returncode
            self.job.ended_at = time.time()
            self.job.status = "passed" if returncode == 0 else "failed"


@dataclass(frozen=True)
class ServerConfig:
    root: Path
    host: str
    port: int
    visual_command: list[str]
    train_command: list[str]
    dashboard_path: Path
    open_browser: bool


class ImproveLoopHandler(BaseHTTPRequestHandler):
    server_version = "LumenImproveLoopHTTP/1.0"
    runner: JobRunner
    config: ServerConfig

    def do_GET(self) -> None:  # noqa: N802
        if self.path in {"/", "/index.html"}:
            self._send_html(self._index_html())
            return
        if self.path == "/status.json":
            self._send_json(self.runner.snapshot())
            return
        if self.path == "/dashboard-summary.json":
            self._serve_dashboard_summary()
            return
        if self.path == "/dashboard":
            self._serve_dashboard()
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def do_POST(self) -> None:  # noqa: N802
        if self.path == "/run-loop":
            ok, message = self.runner.start("visual-improve-loop", self.config.visual_command)
            self._send_json({"ok": ok, "message": message, "job": self.runner.snapshot()}, HTTPStatus.ACCEPTED if ok else HTTPStatus.CONFLICT)
            return
        if self.path == "/run-train":
            if not self.config.train_command:
                self._send_json({"ok": False, "message": "No --train-command was configured for this server."}, HTTPStatus.BAD_REQUEST)
                return
            ok, message = self.runner.start("training", self.config.train_command)
            self._send_json({"ok": ok, "message": message, "job": self.runner.snapshot()}, HTTPStatus.ACCEPTED if ok else HTTPStatus.CONFLICT)
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write("[lumen-loop-server] " + fmt % args + "\n")

    def _serve_dashboard(self) -> None:
        dashboard = self.config.dashboard_path
        if not dashboard.exists():
            self._send_html("<h1>Dashboard not generated yet</h1><p>Run the improve-loop first.</p>", HTTPStatus.NOT_FOUND)
            return
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(dashboard.read_bytes())

    def _serve_dashboard_summary(self) -> None:
        summary_path = self.config.dashboard_path.parent / "visual_improve_loop_summary.json"
        if not summary_path.exists():
            self._send_json({"ok": False, "message": "Dashboard summary not generated yet."}, HTTPStatus.NOT_FOUND)
            return
        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            self._send_json({"ok": False, "message": f"Invalid summary JSON: {exc}"}, HTTPStatus.INTERNAL_SERVER_ERROR)
            return
        if isinstance(payload, dict):
            payload["ok"] = True
            self._send_json(payload)
            return
        self._send_json({"ok": True, "data": payload})

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

    def _index_html(self) -> str:
        visual = html.escape(shlex.join(self.config.visual_command))
        train = html.escape(shlex.join(self.config.train_command)) if self.config.train_command else "Not configured"
        return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Lumen Improve-Loop Control</title>
<style>
:root {{ color-scheme: dark; --bg:#080b11; --panel:#111827; --line:#2b3b55; --text:#e8eefc; --muted:#8ea0bd; --accent:#73a7ff; --green:#44d483; --red:#ff6b7a; }}
body {{ margin:0; background:radial-gradient(circle at top left,#17233a,var(--bg) 42rem); color:var(--text); font-family:ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif; }}
main {{ max-width:1100px; margin:0 auto; padding:2rem; display:grid; gap:1rem; }}
.panel {{ background:rgba(17,24,39,.88); border:1px solid var(--line); border-radius:18px; padding:1rem; box-shadow:0 18px 38px rgba(0,0,0,.25); }}
h1 {{ letter-spacing:-.04em; font-size:clamp(2rem,4vw,3.3rem); margin:.2rem 0; }}
button, a.button {{ appearance:none; border:1px solid var(--line); background:#17233a; color:var(--text); border-radius:12px; padding:.8rem 1rem; font-weight:700; cursor:pointer; text-decoration:none; display:inline-flex; gap:.5rem; align-items:center; margin:.25rem .25rem .25rem 0; }}
button:hover, a.button:hover {{ border-color:var(--accent); }}
code, pre {{ font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; }}
pre {{ white-space:pre-wrap; max-height:34rem; overflow:auto; background:#060910; border:1px solid var(--line); border-radius:14px; padding:1rem; }}
.muted {{ color:var(--muted); }}
.status-passed {{ color:var(--green); }}
.status-failed {{ color:var(--red); }}
</style>
</head>
<body>
<main>
<header>
<p class="muted">Localhost control panel</p>
<h1>Lumen Improve-Loop Control</h1>
<p class="muted">Runs only preconfigured server-start commands. No arbitrary browser shell input.</p>
</header>
<section class="panel">
<h2>Actions</h2>
<button onclick="post('/run-loop')">Run visual improve-loop + embedding state</button>
<button onclick="post('/run-train')">Run fine-tuning command</button>
<a class="button" href="/dashboard" target="_blank">Open generated dashboard</a>
</section>
<section class="panel">
<h2>Configured commands</h2>
<p class="muted">Visual loop</p><pre>{visual}</pre>
<p class="muted">Training</p><pre>{train}</pre>
</section>
<section class="panel">
<h2>Status</h2>
<div id="summary" class="muted">Loading…</div>
<pre id="log"></pre>
</section>
<section class="panel">
<h2>Live Dashboard</h2>
<p class="muted">Real-time progress from the generated visual improve-loop summary.</p>
<div id="dashboard-overview" class="muted">Waiting for summary…</div>
<canvas id="step-chart" width="1024" height="280" style="width:100%;height:280px;background:#060910;border:1px solid var(--line);border-radius:14px;"></canvas>
<pre id="step-details"></pre>
</section>
</main>
<script>
async function post(path) {{
  const res = await fetch(path, {{method:'POST'}});
  const data = await res.json();
  await refresh();
  if (!data.ok) alert(data.message || 'Action failed');
}}
async function refresh() {{
  const res = await fetch('/status.json');
  const data = await res.json();
  const statusClass = data.status === 'passed' ? 'status-passed' : data.status === 'failed' ? 'status-failed' : '';
  document.getElementById('summary').innerHTML = `<strong class="${{statusClass}}">${{data.status}}</strong> · ${{data.name || 'no job'}} · ${{data.durationSeconds ?? 0}}s · rc=${{data.returncode ?? ''}}`;
  document.getElementById('log').textContent = (data.log || []).join('\n');
  await refreshDashboard();
}}
function renderChart(steps) {{
  const canvas = document.getElementById('step-chart');
  const ctx = canvas.getContext('2d');
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = '#060910';
  ctx.fillRect(0, 0, w, h);
  if (!steps.length) return;

  const margin = 36;
  const chartW = w - margin * 2;
  const chartH = h - margin * 2;
  const maxDuration = Math.max(...steps.map(s => Number(s.durationSeconds || 0)), 1);
  const barGap = 12;
  const barW = Math.max((chartW - (steps.length - 1) * barGap) / steps.length, 8);

  ctx.strokeStyle = '#2b3b55';
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {{
    const y = margin + chartH * (i / 4);
    ctx.beginPath();
    ctx.moveTo(margin, y);
    ctx.lineTo(margin + chartW, y);
    ctx.stroke();
  }}

  steps.forEach((step, idx) => {{
    const dur = Number(step.durationSeconds || 0);
    const barH = (dur / maxDuration) * chartH;
    const x = margin + idx * (barW + barGap);
    const y = margin + chartH - barH;
    const status = normalizeStatus(step.status);
    ctx.fillStyle = status === 'success' ? '#44d483' : status === 'failed' ? '#ff6b7a' : '#73a7ff';
    ctx.fillRect(x, y, barW, Math.max(barH, 2));
  }});
}}

async function refreshDashboard() {{
  try {{
    const res = await fetch('/dashboard-summary.json');
    const payload = await res.json();
    if (!res.ok) {{
      document.getElementById('dashboard-overview').textContent = payload.message || 'Dashboard summary not generated yet.';
      document.getElementById('step-details').textContent = '';
      renderChart([]);
      return;
    }}
    const steps = Array.isArray(payload.steps) ? payload.steps : [];
    const total = steps.length;
    const success = steps.filter(s => normalizeStatus(s.status) === 'success').length;
    const failed = steps.filter(s => normalizeStatus(s.status) === 'failed').length;
    const running = steps.filter(s => normalizeStatus(s.status) === 'running').length;
    document.getElementById('dashboard-overview').textContent = `Steps: ${{total}} · success: ${{success}} · failed: ${{failed}} · running: ${{running}}`;
    document.getElementById('step-details').textContent = steps
      .map((s, i) => `${{i + 1}}. ${{s.name || 'step'}} · status=${{normalizeStatus(s.status)}} · duration=${{s.durationSeconds ?? 0}}s`)
      .join('\n');
    renderChart(steps);
  }} catch (err) {{
    console.error('Failed to refresh dashboard', err);
    document.getElementById('dashboard-overview').textContent = 'Dashboard summary not generated yet.';
    document.getElementById('step-details').textContent = '';
    renderChart([]);
    return;
  }}
}}
function normalizeStatus(status) {{
  const value = String(status || '').toLowerCase();
  if (value === 'passed') return 'success';
  if (value === 'in_progress') return 'running';
  return value || 'unknown';
}}
setInterval(refresh, 1500);
refresh();
</script>
</body>
</html>"""


def build_visual_command(args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        str((args.root / args.visual_runner).resolve() if not args.visual_runner.is_absolute() else args.visual_runner.resolve()),
        "--root",
        str(args.root.resolve()),
        "--output",
        str(args.output),
        "--loop-output",
        str(args.loop_output),
        "--fine-tuning-output",
        str(args.fine_tuning_output),
        "--dashboard-output",
        str(args.dashboard_output),
    ]
    if args.skip_tests:
        command.append("--skip-tests")
    if args.quiet_commands:
        command.append("--quiet-commands")
    if args.require_testflight_runtime_audit:
        command.append("--require-testflight-runtime-audit")
    if args.fail_on_gaps:
        command.append("--fail-on-gaps")
    for runtime_audit in args.runtime_audit:
        command.extend(["--runtime-audit", str(runtime_audit)])
    if args.build_command:
        command.extend(["--build-command", args.build_command])
    if args.train_command:
        command.extend(["--train-command", args.train_command])
    if args.dry_run_commands:
        command.append("--dry-run-commands")
    return command


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve a local web page for running the Lumen visual improve-loop and configured fine-tuning command.")
    parser.add_argument("--root", type=Path, default=Path("."), help="Repository root.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host. Keep this localhost unless you understand the risk.")
    parser.add_argument("--port", type=int, default=8765, help="Bind port.")
    parser.add_argument("--visual-runner", type=Path, default=Path("tools/run_visual_improve_loop_with_embedding_state.py"), help="Visual improve-loop runner path relative to --root.")
    parser.add_argument("--output", type=Path, default=Path("generated/agent_manifest"), help="Manifest output path passed to visual runner.")
    parser.add_argument("--loop-output", type=Path, default=Path("generated/agent_improvement_loop"), help="Loop output path passed to visual runner.")
    parser.add_argument("--fine-tuning-output", type=Path, default=Path("generated/fine_tuning"), help="Fine-tuning output path passed to visual runner.")
    parser.add_argument("--dashboard-output", type=Path, default=Path("generated/visual_improve_loop"), help="Dashboard output path passed to visual runner.")
    parser.add_argument("--runtime-audit", action="append", default=[], help="Runtime audit JSON file or directory passed to visual runner.")
    parser.add_argument("--build-command", default=None, help="Optional build command passed into the visual runner.")
    parser.add_argument("--train-command", default=None, help="Optional fine-tuning command. The page can run this exact command only.")
    parser.add_argument("--skip-tests", action="store_true", help="Pass --skip-tests to visual runner.")
    parser.add_argument("--quiet-commands", action="store_true", help="Pass --quiet-commands to visual runner.")
    parser.add_argument("--dry-run-commands", action="store_true", help="Pass --dry-run-commands to visual runner.")
    parser.add_argument("--require-testflight-runtime-audit", action="store_true", help="Pass --require-testflight-runtime-audit to visual runner.")
    parser.add_argument("--fail-on-gaps", action="store_true", help="Pass --fail-on-gaps to visual runner.")
    parser.add_argument("--open", action="store_true", help="Open the local web page in the browser.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.root.resolve()
    visual_command = build_visual_command(args)
    train_command = shlex.split(args.train_command) if args.train_command else []
    dashboard_path = (root / args.dashboard_output / "index.html").resolve() if not args.dashboard_output.is_absolute() else (args.dashboard_output / "index.html").resolve()

    config = ServerConfig(
        root=root,
        host=args.host,
        port=args.port,
        visual_command=visual_command,
        train_command=train_command,
        dashboard_path=dashboard_path,
        open_browser=args.open,
    )
    runner = JobRunner(root)

    class Handler(ImproveLoopHandler):
        pass

    Handler.runner = runner
    Handler.config = config

    server = ThreadingHTTPServer((config.host, config.port), Handler)
    url = f"http://{config.host}:{config.port}/"
    print(f"Serving Lumen improve-loop control at {url}")
    print("Visual command:", shlex.join(config.visual_command))
    if config.train_command:
        print("Training command:", shlex.join(config.train_command))
    else:
        print("Training command: not configured")
    if config.open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
