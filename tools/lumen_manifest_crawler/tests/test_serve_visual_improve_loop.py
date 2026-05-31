from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _load_server() -> ModuleType:
    module_path = _repo_root() / "tools" / "serve_visual_improve_loop.py"
    module_name = "lumen_serve_visual_improve_loop"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_build_visual_command_includes_runtime_audits_and_flags(tmp_path: Path) -> None:
    server = _load_server()
    root = tmp_path / "repo"
    root.mkdir()
    audit = root / "exports" / "audit.json"
    audit.parent.mkdir()
    audit.write_text("{}", encoding="utf-8")

    args = server.parse_args(
        [
            "--root",
            str(root),
            "--runtime-audit",
            str(audit),
            "--skip-tests",
            "--quiet-commands",
            "--dry-run-commands",
            "--fail-on-gaps",
        ]
    )

    command = server.build_visual_command(args)
    assert command.count("--dashboard-output") == 1
    assert "--runtime-audit" in command
    assert str(audit) in command
    assert "--skip-tests" in command
    assert "--quiet-commands" in command
    assert "--dry-run-commands" in command
    assert "--fail-on-gaps" in command


def test_dashboard_summary_route_returns_payload(tmp_path: Path) -> None:
    server = _load_server()
    dashboard_dir = tmp_path / "generated" / "visual_improve_loop"
    dashboard_dir.mkdir(parents=True)
    summary_path = dashboard_dir / "visual_improve_loop_summary.json"
    summary_path.write_text(json.dumps({"steps": [{"name": "compile", "status": "success", "durationSeconds": 1.2}]}), encoding="utf-8")

    config = server.ServerConfig(
        root=tmp_path,
        host="127.0.0.1",
        port=8765,
        visual_command=["python", "runner.py"],
        train_command=[],
        dashboard_path=dashboard_dir / "index.html",
        open_browser=False,
    )

    class Sink:
        def __init__(self) -> None:
            self.payload: tuple[dict, object] | None = None

        def _send_json(self, payload, status=server.HTTPStatus.OK):
            self.payload = (payload, status)

    sink = Sink()
    sink.config = config
    server.ImproveLoopHandler._serve_dashboard_summary(sink)
    assert sink.payload is not None
    payload, status = sink.payload
    assert status == server.HTTPStatus.OK
    assert payload["ok"] is True
    assert payload["steps"][0]["name"] == "compile"


def test_dashboard_summary_route_missing_file_returns_not_found(tmp_path: Path) -> None:
    server = _load_server()
    dashboard_dir = tmp_path / "generated" / "visual_improve_loop"
    dashboard_dir.mkdir(parents=True)

    config = server.ServerConfig(
        root=tmp_path,
        host="127.0.0.1",
        port=8765,
        visual_command=["python", "runner.py"],
        train_command=[],
        dashboard_path=dashboard_dir / "index.html",
        open_browser=False,
    )

    class Sink:
        def __init__(self) -> None:
            self.payload: tuple[dict, object] | None = None

        def _send_json(self, payload, status=server.HTTPStatus.OK):
            self.payload = (payload, status)

    sink = Sink()
    sink.config = config
    server.ImproveLoopHandler._serve_dashboard_summary(sink)
    assert sink.payload is not None
    payload, status = sink.payload
    assert status == server.HTTPStatus.NOT_FOUND
    assert payload["ok"] is False
    assert "not generated yet" in payload["message"].lower()


def test_dashboard_summary_route_invalid_json_returns_error(tmp_path: Path) -> None:
    server = _load_server()
    dashboard_dir = tmp_path / "generated" / "visual_improve_loop"
    dashboard_dir.mkdir(parents=True)
    summary_path = dashboard_dir / "visual_improve_loop_summary.json"
    summary_path.write_text("{ invalid", encoding="utf-8")

    config = server.ServerConfig(
        root=tmp_path,
        host="127.0.0.1",
        port=8765,
        visual_command=["python", "runner.py"],
        train_command=[],
        dashboard_path=dashboard_dir / "index.html",
        open_browser=False,
    )

    class Sink:
        def __init__(self) -> None:
            self.payload: tuple[dict, object] | None = None

        def _send_json(self, payload, status=server.HTTPStatus.OK):
            self.payload = (payload, status)

    sink = Sink()
    sink.config = config
    server.ImproveLoopHandler._serve_dashboard_summary(sink)
    assert sink.payload is not None
    payload, status = sink.payload
    assert status == server.HTTPStatus.INTERNAL_SERVER_ERROR
    assert payload["ok"] is False
    assert "invalid summary json" in payload["message"].lower()
