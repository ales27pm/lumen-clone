"""Training validation must survive CLI output filtering, dry runs and caching."""

import json
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from lumen_manifest_crawler import cli
from lumen_manifest_crawler.manifest import AgentBehaviorManifest, ValidationFailure, ValidationReport


@pytest.fixture
def generation_boundary(monkeypatch, tmp_path):
    state = SimpleNamespace(
        failure=None, manifest_failure=None, validations=[], writes=[],
        output=tmp_path / "output", root=tmp_path,
    )
    manifest = AgentBehaviorManifest()
    datasets = {agent: object() for agent in ("cortex", "executor", "mouth", "mimicry", "rem", "fleet")}
    monkeypatch.setattr(cli, "generate_manifest", lambda root: manifest)
    monkeypatch.setattr(cli, "_manifest_fingerprint", lambda manifest: "unchanged")
    monkeypatch.setattr(cli, "load_runtime_audit_reports", lambda paths: [])
    monkeypatch.setattr(cli, "generate_all_datasets", lambda *args, **kwargs: {"dataset_manifest": []})
    monkeypatch.setattr(cli, "validate_manifest", lambda *args, **kwargs: ValidationReport(
        passed=state.manifest_failure is None,
        failures=[state.manifest_failure] if state.manifest_failure else [],
    ))
    monkeypatch.setattr(cli, "generate_fleet_artifacts", lambda manifest: SimpleNamespace(
        system_prompts=[], cross_model_training=[], orchestration_evals=[],
    ))
    monkeypatch.setattr(cli, "compile_agent_fine_tuning_datasets", lambda *args, **kwargs: datasets)

    def validate(manifest, materialized, **kwargs):
        state.validations.append(set(materialized))
        return ([state.failure] if state.failure else []) + [
            ValidationFailure(code="missing_agent_dataset", message=f"Missing {agent}")
            for agent in datasets if agent not in materialized
        ]

    def write(output, manifest, report, records, **kwargs):
        output.mkdir(parents=True, exist_ok=True)
        (output / "validation.json").write_text(report.model_dump_json(), encoding="utf-8")
        state.writes.append((report, kwargs.get("fine_tuning_datasets")))

    monkeypatch.setattr(cli, "validate_agent_fine_tuning_datasets", validate)
    monkeypatch.setattr(cli, "write_outputs", write)
    monkeypatch.setattr(cli, "_diff_directories", lambda *args: {"changed": False})
    monkeypatch.setattr(cli.tempfile, "mkdtemp", lambda **kwargs: str(tmp_path / "preview"))

    def run(*options, training=True):
        return CliRunner().invoke(cli.app, [
            "generate", "--root", str(tmp_path), "--output", str(state.output),
            *(["--generate-agent-fine-tuning"] if training else []), *options,
        ])

    state.run = run
    return state


@pytest.mark.parametrize("options", [[], ["--no-fail-on-validation"], ["--dry-run"]])
def test_training_failures_are_reflected_in_serialized_report(generation_boundary, options):
    state = generation_boundary
    state.failure = ValidationFailure(code="executor_non_json_output", message="Invalid target")
    result = state.run(*options)
    assert result.exit_code == (0 if "--no-fail-on-validation" in options else 1)
    target = state.root / "preview" / "agent_manifest" if "--dry-run" in options else state.output
    report = json.loads((target / "validation.json").read_text())
    assert report["passed"] is False
    assert report["failures"][0]["code"] == "executor_non_json_output"


@pytest.mark.parametrize("flag", ["--dry-run", "--diff"])
@pytest.mark.parametrize("source", ["manifest_failure", "failure"])
def test_preview_honors_validation_exit_policy(generation_boundary, flag, source):
    state = generation_boundary
    setattr(state, source, ValidationFailure(code="invalid_fixture", message="Invalid fixture"))
    result = state.run(flag)
    assert result.exit_code == 1
    assert not state.output.exists()
    assert json.loads(result.stdout)["changed"] is False
    assert state.run(flag, "--no-fail-on-validation").exit_code == 0


def test_agent_filter_validates_complete_fleet_then_selects_output(generation_boundary):
    state = generation_boundary
    result = state.run("--agent-filter", "executor")
    assert result.exit_code == 0, result.stdout
    assert state.validations == [{"cortex", "executor", "mouth", "mimicry", "rem", "fleet"}]
    assert set(state.writes[0][1]) == {"executor"}


@pytest.mark.parametrize("selection", ["unknown", ", ", "executor,unknown"])
def test_invalid_agent_filter_is_rejected_before_writing(generation_boundary, selection):
    state = generation_boundary
    assert state.run("--agent-filter", selection).exit_code != 0
    assert state.writes == []


def test_agent_filter_cannot_be_silently_ignored_without_training(generation_boundary):
    state = generation_boundary
    assert state.run("--agent-filter", "executor", training=False).exit_code != 0
    assert state.writes == []


def test_incremental_manifest_cannot_skip_requested_training_validation(generation_boundary, monkeypatch):
    state = generation_boundary
    monkeypatch.setattr(cli, "_incremental_outputs_are_current", lambda *args, **kwargs: True)
    result = state.run("--incremental")
    assert result.exit_code == 0, result.stdout
    assert len(state.validations) == 1
    assert len(state.writes) == 1


@pytest.mark.parametrize("options", [[], ["--dry-run"], ["--diff"], ["--no-fail-on-validation"]])
def test_valid_generation_preserves_success(generation_boundary, options):
    state = generation_boundary
    assert state.run(*options).exit_code == 0
    assert state.writes[0][0].passed is True
    assert state.writes[0][0].failures == []


@pytest.mark.parametrize("cached", [
    None, "not JSON", "[]", '{"passed":false,"failures":[],"warnings":[]}',
    '{"passed":true,"failures":[{}],"warnings":[]}',
    '{"passed":false,"passed":true,"failures":[],"warnings":[]}',
])
def test_incremental_cannot_turn_missing_or_failed_receipt_into_success(generation_boundary, monkeypatch, cached):
    state = generation_boundary
    state.output.mkdir()
    if cached is not None:
        (state.output / "manifest_validation_report.json").write_text(cached)
    state.manifest_failure = ValidationFailure(code="invalid_fixture", message="Current input invalid")
    monkeypatch.setattr(cli, "_incremental_outputs_are_current", lambda *args, **kwargs: True)
    assert state.run("--incremental", training=False).exit_code == 1
    assert state.writes[0][0].passed is False


@pytest.mark.parametrize("flag", ["--strict", "--fail-on-change"])
def test_incremental_cannot_bypass_requested_checks(generation_boundary, monkeypatch, flag):
    state = generation_boundary
    state.output.mkdir()
    (state.output / "manifest_validation_report.json").write_text(
        '{"passed":true,"failures":[],"warnings":[]}'
    )
    monkeypatch.setattr(cli, "_incremental_outputs_are_current", lambda *args, **kwargs: True)
    monkeypatch.setattr(cli, "_has_git_changes", lambda root: True)
    if flag == "--strict":
        state.manifest_failure = ValidationFailure(code="promoted_warning", message="Strict policy failed")
    assert state.run("--incremental", flag, training=False).exit_code == 1
    assert len(state.writes) == 1


def test_successful_incremental_receipt_remains_reusable(generation_boundary, monkeypatch):
    state = generation_boundary
    state.output.mkdir()
    (state.output / "manifest_validation_report.json").write_text(
        '{"passed":true,"failures":[],"warnings":[]}'
    )
    monkeypatch.setattr(cli, "_incremental_outputs_are_current", lambda *args, **kwargs: True)
    assert state.run("--incremental", training=False).exit_code == 0
    assert state.writes == []
