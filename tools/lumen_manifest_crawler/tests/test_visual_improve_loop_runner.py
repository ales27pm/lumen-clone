from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest


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


def test_verify_now_requires_explicit_non_deterministic_device_debug_mode() -> None:
    runner = _load_runner()

    with pytest.raises(SystemExit):
        runner.parse_args([
            "--verify-runtime-audit-now",
            "--runtime-audit-expected-build-number",
            "204",
        ])


def test_command_queue_uses_repo_rooted_outputs(tmp_path: Path) -> None:
    runner = _load_runner()
    root = tmp_path / "repo"
    root.mkdir()

    args = runner.parse_args([
        "--root",
        str(root),
        "--skip-tests",
        "--no-auto-discover-runtime-audit",
        "--runtime-audit-reference-time",
        "2026-08-10T09:35:43Z",
        "--runtime-audit-max-age-seconds",
        "7200",
        "--verify-runtime-audit-now",
        "--no-deterministic",
        "--app-run-mode",
        "device-debug",
        "--runtime-audit-expected-build-number",
        "20260810044413",
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
    cross_model_index = improve.index("--cross-model-train-dir") + 1
    assert improve[cross_model_index] == str(root / "generated" / "cross_model_training")
    reference_index = improve.index("--runtime-audit-reference-time") + 1
    assert improve[reference_index] == "2026-08-10T09:35:43Z"
    max_age_index = improve.index("--runtime-audit-max-age-seconds") + 1
    assert improve[max_age_index] == "7200"
    assert "--verify-runtime-audit-now" in improve
    build_index = improve.index("--runtime-audit-expected-build-number") + 1
    assert improve[build_index] == "20260810044413"
    assert "--non-deterministic" in improve
    assert improve[improve.index("--app-run-mode") + 1] == "device-debug"
    assert str(root / "generated" / "fine_tuning" / "release_bake_gguf_manifest.json") in release_manifest
    config_dir_index = release_manifest.index("--config-dir") + 1
    assert release_manifest[config_dir_index] == str(fine_tuning_output)


def test_app_run_mode_typo_fails_closed() -> None:
    runner = _load_runner()

    with pytest.raises(SystemExit):
        runner.parse_args([
            "--app-run-mode",
            "testfilght",
            "--require-testflight-runtime-audit",
        ])


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
    assert default_release[:3] == [
        default_args.release_bake_python,
        "-m",
        "tools.fine_tuning.unsloth.export_gguf",
    ]
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
    assert "--config-dir" not in bake_release

    prepared_dir = root / "prepared-run" / "configs"
    explicit_args = runner.parse_args([
        "--root",
        str(root),
        "--skip-tests",
        "--release-bake",
        "--config-dir",
        str(prepared_dir),
    ])
    explicit_commands = runner.build_command_queue(
        explicit_args,
        root,
        runner.rooted_path(root, explicit_args.output),
        runner.rooted_path(root, explicit_args.loop_output),
        runner.rooted_path(root, explicit_args.fine_tuning_output),
        [],
    )
    explicit_release = explicit_commands[-1]["command"]
    config_dir_index = explicit_release.index("--config-dir") + 1
    assert explicit_release[config_dir_index] == str(prepared_dir)


def test_runtime_audit_discovery_accepts_realistic_export_and_rejects_loop_state(tmp_path: Path) -> None:
    runner = _load_runner()
    export_dir = tmp_path / "exports"
    export_dir.mkdir()
    good = export_dir / "lumen-testflight-agent-grounding-testflight.json"
    good.write_text(json.dumps({"runtime": {"build": "TestFlight"}, "traces": [{"id": "t1"}], "failures": []}), encoding="utf-8")
    bad = export_dir / "loop_state.json"
    bad.write_text(json.dumps({"runtime": {"build": "TestFlight"}, "traces": [{"id": "t2"}]}), encoding="utf-8")

    found = runner.find_runtime_audit_json(export_dir)

    assert good in found
    assert bad not in found


def test_runtime_audit_discovery_accepts_live_e2e_report_names(tmp_path: Path) -> None:
    runner = _load_runner()
    export_dir = tmp_path / "exports"
    export_dir.mkdir()
    report = export_dir / "lumen-live-e2e-report-2026-06-08T23-26-36Z.json"
    report.write_text(
        json.dumps(
            {
                "schemaVersion": "1.0.0",
                "exportPolicy": {
                    "format": "live-e2e-test-report-json",
                    "sourceLayer": "e2eTestReport",
                    "ownsLiveE2EScenarios": True,
                },
                "payload": {"passed": 1, "failed": 0, "results": []},
            }
        ),
        encoding="utf-8",
    )

    found = runner.find_runtime_audit_json(export_dir)

    assert report in found


def test_runtime_audit_discovery_accepts_plain_latest_e2e_report(tmp_path: Path) -> None:
    runner = _load_runner()
    report = tmp_path / "latest-e2e-report.json"
    report.write_text(
        json.dumps({"id": "report-1", "passed": 1, "failed": 0, "results": []}),
        encoding="utf-8",
    )

    assert runner.find_runtime_audit_json(tmp_path) == [report]


def test_dashboard_outputs_escape_dynamic_content(tmp_path: Path) -> None:
    runner = _load_runner()
    root = tmp_path / "repo"
    dashboard = tmp_path / "dashboard"
    audit = tmp_path / "private" / "diagnostic-name.json"
    audit.parent.mkdir(parents=True)
    audit_bytes = b'{"privacy":"redacted"}\n'
    audit.write_bytes(audit_bytes)
    artifacts = runner.LoopArtifacts(
        state={
            "manifest": {"toolCount": 1},
            "dataset": {"families": {"x<script>": 2}},
            "runtime": {
                "/Users/alex/Client Secret/report.json": "private key canary",
            },
            "testFlight": {
                "status": "historical-runtime-audit-ingested",
                "proofStatus": "historical-unverified",
                "currentRuntimeAuditProvided": False,
            },
        },
        gaps=[{"severity": "error", "category": "x", "title": "<script>alert(1)</script>   \nnext line", "recommendedAction": "escape it"}],
        next_prompts=[{"taskType": "x", "priority": "high", "id": "prompt<script>"}],
        testflight_scenarios=[],
        release_bake_manifest={"mode": "adapter_first"},
        adapter_runtime_manifest={"mode": "adapter_first", "adapters": []},
    )
    args = runner.parse_args(["--root", str(root), "--skip-tests"])

    step = runner.StepResult(
        name="test output",
        command=[
            runner.sys.executable,
            "pytest",
            str(root),
            str(root / "generated" / "agent_manifest"),
            str(audit),
        ],
        cwd=str(root),
        started_at="2026-05-03T00:00:00+00:00",
        ended_at="2026-05-03T00:00:01+00:00",
        duration_seconds=1.0,
        returncode=1,
        stdout_tail=(
            f"source line {root}    \n    \ninput {audit}\t \n"
            "/Volumes/AlexisSecret/model.gguf /usr/local/private-tool\n"
            "/Users/alex/Client Secret/model.gguf\n"
            "[report](</Users/alex/Client Secret/report.json>)\n"
            "file://localhost/Users/alex/private.json\n"
        ),
    )
    outputs = runner.write_visual_outputs(
        root=root,
        dashboard_dir=dashboard,
        output=root / "generated" / "agent_manifest",
        loop_output=root / "generated" / "agent_improvement_loop",
        fine_tuning_output=root / "generated" / "fine_tuning",
        args=args,
        started_at="2026-05-03T00:00:00+00:00",
        ended_at="2026-05-03T00:00:01+00:00",
        steps=[step],
        artifacts=artifacts,
        runtime_audits=[audit],
    )

    html = outputs["html"].read_text(encoding="utf-8")
    summary = json.loads(outputs["summary"].read_text(encoding="utf-8"))
    assert summary["schemaVersion"] == "2.1.0"
    assert summary["testFlight"]["proofStatus"] == "historical-unverified"
    assert "Pipeline PASS does not imply current device or TestFlight proof." in html
    assert "historical-unverified" in html
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html
    assert all(line == line.rstrip() for line in html.splitlines())
    expected_ref = f"runtime-audit-sha256-{runner.hashlib.sha256(audit_bytes).hexdigest()}"
    for path in outputs.values():
        contents = path.read_text(encoding="utf-8")
        assert str(root) not in contents
        assert str(audit) not in contents
        assert "diagnostic-name.json" not in contents
        assert "AlexisSecret" not in contents
        assert "/usr/local/" not in contents
        assert "Client Secret" not in contents
        assert "file:" not in contents.casefold()
        assert str(Path(runner.sys.executable).resolve()) not in contents
    assert summary["root"] == "."
    assert summary["output"] == "generated/agent_manifest"
    assert "./generated/agent_manifest" in json.dumps(summary, sort_keys=True)
    assert expected_ref in json.dumps(summary, sort_keys=True)
