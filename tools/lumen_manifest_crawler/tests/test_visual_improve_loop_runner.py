from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _load_runner() -> ModuleType:
    module_path = _repo_root() / "tools" / "run_visual_improve_loop_v2.py"
    module_name = "lumen_visual_improve_loop_v2"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_relative_outputs_are_resolved_against_invocation_root(tmp_path: Path) -> None:
    runner = _load_runner()
    root = tmp_path / "repo"
    root.mkdir()

    args = runner.parse_args([
        "--root",
        str(root),
        "--output",
        "generated/agent_manifest",
        "--loop-output",
        "generated/agent_improvement_loop",
        "--fine-tuning-output",
        "generated/fine_tuning",
        "--dashboard-output",
        "generated/visual_improve_loop",
        "--release-bake-manifest-output",
        "generated/fine_tuning/release_bake_gguf_manifest.json",
        "--skip-tests",
    ])

    assert runner.rooted_path(root, args.output) == root / "generated" / "agent_manifest"
    assert runner.rooted_path(root, args.loop_output) == root / "generated" / "agent_improvement_loop"
    assert runner.rooted_path(root, args.fine_tuning_output) == root / "generated" / "fine_tuning"
    assert runner.rooted_path(root, args.dashboard_output) == root / "generated" / "visual_improve_loop"
    assert runner.rooted_path(root, args.release_bake_manifest_output) == root / "generated" / "fine_tuning" / "release_bake_gguf_manifest.json"


def test_command_queue_uses_repo_rooted_outputs(tmp_path: Path) -> None:
    runner = _load_runner()
    root = tmp_path / "repo"
    root.mkdir()

    args = runner.parse_args([
        "--root",
        str(root),
        "--skip-tests",
        "--no-auto-discover-runtime-audit",
    ])
    output = runner.rooted_path(root, args.output)
    loop_output = runner.rooted_path(root, args.loop_output)
    fine_tuning_output = runner.rooted_path(root, args.fine_tuning_output)

    commands = runner.build_command_queue(args, root, output, loop_output, fine_tuning_output, [])
    improve = next(command for command in commands if command["name"] == "improve-loop generation")["command"]
    release_manifest = next(command for command in commands if command["name"] == "adapter-first release-bake manifest")["command"]

    assert str(output) in improve
    assert str(loop_output) in improve
    assert str(fine_tuning_output) in improve
    assert str(root / "generated" / "fine_tuning" / "release_bake_gguf_manifest.json") in release_manifest


def test_release_bake_command_requires_explicit_flag(tmp_path: Path) -> None:
    runner = _load_runner()
    root = tmp_path / "repo"
    root.mkdir()

    default_args = runner.parse_args(["--root", str(root), "--skip-tests"])
    default_commands = runner.build_command_queue(
        default_args,
        root,
        runner.rooted_path(root, default_args.output),
        runner.rooted_path(root, default_args.loop_output),
        runner.rooted_path(root, default_args.fine_tuning_output),
        [],
    )
    default_release = default_commands[-1]["command"]
    assert "--release-bake" not in default_release

    bake_args = runner.parse_args(["--root", str(root), "--skip-tests", "--release-bake"])
    bake_commands = runner.build_command_queue(
        bake_args,
        root,
        runner.rooted_path(root, bake_args.output),
        runner.rooted_path(root, bake_args.loop_output),
        runner.rooted_path(root, bake_args.fine_tuning_output),
        [],
    )
    bake_release = bake_commands[-1]["command"]
    assert "--release-bake" in bake_release


def test_runtime_audit_discovery_accepts_realistic_export_and_rejects_loop_state(tmp_path: Path) -> None:
    runner = _load_runner()
    export_dir = tmp_path / "exports"
    export_dir.mkdir()
    good = export_dir / "lumen-in-app-dataset-testflight.json"
    good.write_text(json.dumps({"runtime": {"build": "TestFlight"}, "traces": [{"id": "t1"}], "failures": []}), encoding="utf-8")
    bad = export_dir / "loop_state.json"
    bad.write_text(json.dumps({"runtime": {"build": "TestFlight"}, "traces": [{"id": "t2"}]}), encoding="utf-8")

    found = runner.find_runtime_audit_json(export_dir)

    assert good in found
    assert bad not in found


def test_dashboard_outputs_escape_dynamic_content(tmp_path: Path) -> None:
    runner = _load_runner()
    root = tmp_path / "repo"
    dashboard = tmp_path / "dashboard"
    artifacts = runner.LoopArtifacts(
        state={"manifest": {"toolCount": 1}, "dataset": {"families": {"x<script>": 2}}, "runtime": {}},
        gaps=[{"severity": "error", "category": "x", "title": "<script>alert(1)</script>", "recommendedAction": "escape it"}],
        next_prompts=[{"taskType": "x", "priority": "high", "id": "prompt<script>"}],
        testflight_scenarios=[],
        release_bake_manifest={"mode": "adapter_first"},
        adapter_runtime_manifest={"mode": "adapter_first", "adapters": []},
    )
    args = runner.parse_args(["--root", str(root), "--skip-tests"])

    outputs = runner.write_visual_outputs(
        root=root,
        dashboard_dir=dashboard,
        output=root / "generated" / "agent_manifest",
        loop_output=root / "generated" / "agent_improvement_loop",
        fine_tuning_output=root / "generated" / "fine_tuning",
        args=args,
        started_at="2026-05-03T00:00:00+00:00",
        ended_at="2026-05-03T00:00:01+00:00",
        steps=[],
        artifacts=artifacts,
    )

    html = outputs["html"].read_text(encoding="utf-8")
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html
