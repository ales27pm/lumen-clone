"""Failed qualification must prevent configured build/training side effects."""

import sys

import pytest

from lumen_manifest_crawler import improvement_loop as loop
from lumen_manifest_crawler.manifest import AgentBehaviorManifest, ValidationFailure, ValidationReport


@pytest.mark.parametrize("failure", ["validation", "pre_generation_tests", None])
@pytest.mark.parametrize("dry_run", [False, True])
def test_commands_respect_validation_and_pretest_gate(monkeypatch, tmp_path, failure, dry_run):
    manifest = AgentBehaviorManifest()
    report = ValidationReport(
        passed=failure != "validation",
        failures=[ValidationFailure(code="invalid_target", message="Invalid target")]
        if failure == "validation" else [],
    )
    monkeypatch.setattr(loop, "generate_manifest", lambda root: manifest)
    monkeypatch.setattr(loop, "generate_all_datasets", lambda *args, **kwargs: {})
    monkeypatch.setattr(loop, "validate_manifest", lambda *args, **kwargs: report)
    monkeypatch.setattr(loop, "write_outputs", lambda *args, **kwargs: None)
    build_marker, train_marker = tmp_path / "built", tmp_path / "trained"

    def marker_command(path):
        return (sys.executable, "-c", "from pathlib import Path; import sys; Path(sys.argv[1]).touch()", str(path))

    result = loop.run_agent_improvement_loop(loop.AgentImprovementLoopConfig(
        root=tmp_path, output=tmp_path / "output", loop_output=tmp_path / "loop",
        generate_system_prompts=False, generate_agent_fine_tuning=False,
        test_command=(sys.executable, "-c", "raise SystemExit(7)")
        if failure == "pre_generation_tests" else (),
        build_command=marker_command(build_marker), train_command=marker_command(train_marker),
        dry_run_commands=dry_run,
    ))
    # A dry-run pretest was never executed and has no observed failure.
    blocked = failure == "validation" or (failure == "pre_generation_tests" and not dry_run)
    for name, marker in (("build_for_testflight", build_marker), ("train", train_marker)):
        command = next(command for command in result.command_results if command.name == name)
        assert marker.exists() is (not blocked and not dry_run)
        assert command.passed is (not blocked)
        if blocked:
            assert "not executed" in command.stderr_tail
    if blocked:
        assert result.passed is False
