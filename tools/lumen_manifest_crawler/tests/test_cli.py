import json
from pathlib import Path
from types import SimpleNamespace

from lumen_manifest_crawler import cli
from lumen_manifest_crawler.cli import (
    _incremental_outputs_are_current,
    _should_generate_full_fleet_artifacts,
)
from lumen_manifest_crawler.output.writer import CROSS_MODEL_ARTIFACT_FILENAMES


def _run_stubbed_generation(
    monkeypatch,
    *,
    root: Path,
    output: Path,
    fine_tuning_output: Path,
    dry_run: bool,
    cross_model_train_dir: Path | None = None,
) -> tuple[list[tuple[Path, Path]], list[str], Path | None]:
    manifest = SimpleNamespace(
        tools=[],
        intents=[],
        fleet=SimpleNamespace(slots=[]),
    )
    report = SimpleNamespace(failures=[], warnings=[])
    fleet_artifacts = SimpleNamespace(
        system_prompts=[],
        cross_model_training=[],
        orchestration_evals=[],
    )
    writes: list[tuple[Path, Path]] = []
    printed: list[str] = []
    dry_run_root = root / "dry-run" if dry_run else None

    monkeypatch.setattr(cli, "generate_manifest", lambda _root: manifest)
    monkeypatch.setattr(cli, "_manifest_fingerprint", lambda _manifest: "fingerprint")
    monkeypatch.setattr(cli, "load_runtime_audit_reports", lambda _paths: [])
    monkeypatch.setattr(
        cli,
        "generate_all_datasets",
        lambda *_args, **_kwargs: {"dataset_manifest": []},
    )
    monkeypatch.setattr(cli, "validate_manifest", lambda *_args, **_kwargs: report)
    monkeypatch.setattr(cli, "generate_fleet_artifacts", lambda _manifest: fleet_artifacts)
    monkeypatch.setattr(
        cli,
        "compile_agent_fine_tuning_datasets",
        lambda *_args, **_kwargs: {"cortex": object()},
    )
    monkeypatch.setattr(
        cli,
        "validate_agent_fine_tuning_datasets",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(cli.console, "print", lambda value: printed.append(str(value)))

    if dry_run_root is not None:
        def fake_mkdtemp(*, prefix: str) -> str:
            assert prefix == "lumen-manifest-dry-run-"
            dry_run_root.mkdir()
            return str(dry_run_root)

        monkeypatch.setattr(cli.tempfile, "mkdtemp", fake_mkdtemp)

    def fake_write_outputs(
        output_dir: Path,
        *_args,
        fine_tuning_datasets=None,
        fine_tuning_output_dir: Path | None = None,
        **_kwargs,
    ) -> None:
        assert fine_tuning_datasets is not None
        assert fine_tuning_output_dir is not None
        output_dir.mkdir(parents=True, exist_ok=True)
        fine_tuning_output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "generated.marker").write_text("manifest\n", encoding="utf-8")
        (fine_tuning_output_dir / "generated.marker").write_text("fine-tuning\n", encoding="utf-8")
        target_cross_model = _kwargs.get("cross_model_train_dir")
        if target_cross_model is not None:
            target_cross_model.mkdir(parents=True, exist_ok=True)
            (target_cross_model / "generated.marker").write_text(
                "cross-model\n",
                encoding="utf-8",
            )
        writes.append((output_dir, fine_tuning_output_dir))

    monkeypatch.setattr(cli, "write_outputs", fake_write_outputs)
    cli.generate(
        root=root,
        output=output,
        pretty=False,
        runtime_audit=None,
        deterministic=True,
        generate_system_prompts=False,
        export_md=False,
        cross_model_train_dir=cross_model_train_dir,
        dry_run=dry_run,
        diff=False,
        incremental=False,
        strict=False,
        fail_on_change=False,
        fail_on_validation=True,
        generate_agent_fine_tuning=True,
        fine_tuning_output=fine_tuning_output,
        unsloth_output=None,
        agent_filter=None,
    )
    return writes, printed, dry_run_root


def test_agent_fine_tuning_requires_complete_fleet_artifacts() -> None:
    assert _should_generate_full_fleet_artifacts(
        generate_system_prompts=False,
        cross_model_train_dir=None,
        generate_agent_fine_tuning=True,
    )


def test_fleet_artifacts_are_skipped_when_no_output_consumes_them() -> None:
    assert not _should_generate_full_fleet_artifacts(
        generate_system_prompts=False,
        cross_model_train_dir=None,
        generate_agent_fine_tuning=False,
    )


def test_explicit_fleet_outputs_still_require_complete_artifacts() -> None:
    assert _should_generate_full_fleet_artifacts(
        generate_system_prompts=True,
        cross_model_train_dir=None,
        generate_agent_fine_tuning=False,
    )
    assert _should_generate_full_fleet_artifacts(
        generate_system_prompts=False,
        cross_model_train_dir=Path("cross-model"),
        generate_agent_fine_tuning=False,
    )


def test_incremental_generation_requires_complete_matching_external_mirror(
    tmp_path: Path,
) -> None:
    output = tmp_path / "agent_manifest"
    mirror = tmp_path / "cross_model_training"
    nested = output / "cross_model_training"
    nested.mkdir(parents=True)
    mirror.mkdir()
    fingerprint = "f" * 64
    source_integrity = {
        "baseCommit": "a" * 40,
        "workingTreeDigest": "b" * 64,
        "dirtyState": False,
    }
    (output / "AgentBehaviorManifest.incremental.sha256").write_text(
        fingerprint + "\n",
        encoding="utf-8",
    )
    (output / "dataset_manifest.json").write_text("{}\n", encoding="utf-8")
    (output / "AgentBehaviorManifest.json").write_text(
        json.dumps({"sourceIntegrity": source_integrity}) + "\n",
        encoding="utf-8",
    )
    for filename in CROSS_MODEL_ARTIFACT_FILENAMES:
        if filename == "orchestration_evals.jsonl":
            content = json.dumps(
                {
                    "metadata": {
                        "manifestCommit": source_integrity["baseCommit"],
                        "sourceIntegrity": source_integrity,
                    }
                },
                sort_keys=True,
            ) + "\n"
        elif filename.endswith(".jsonl"):
            content = "{}\n"
        else:
            content = "header\n"
        (nested / filename).write_text(content, encoding="utf-8")
        (mirror / filename).write_text(content, encoding="utf-8")

    assert _incremental_outputs_are_current(
        output,
        fingerprint,
        cross_model_train_dir=mirror,
        require_cross_model_artifacts=True,
    )

    assert _incremental_outputs_are_current(
        output,
        fingerprint,
        cross_model_train_dir=None,
        require_cross_model_artifacts=True,
    )

    (mirror / "train_sft_cross.jsonl").write_text(
        "stale\n",
        encoding="utf-8",
    )
    assert not _incremental_outputs_are_current(
        output,
        fingerprint,
        cross_model_train_dir=mirror,
        require_cross_model_artifacts=True,
    )

    (mirror / "train_sft_cross.jsonl").write_text(
        "{}\n",
        encoding="utf-8",
    )
    (nested / "unexpected.txt").write_text("unexpected\n", encoding="utf-8")
    assert not _incremental_outputs_are_current(
        output,
        fingerprint,
        cross_model_train_dir=mirror,
        require_cross_model_artifacts=True,
    )
    (nested / "unexpected.txt").unlink()

    stale_integrity = {**source_integrity, "baseCommit": "c" * 40}
    (output / "AgentBehaviorManifest.json").write_text(
        json.dumps({"sourceIntegrity": stale_integrity}) + "\n",
        encoding="utf-8",
    )
    assert not _incremental_outputs_are_current(
        output,
        fingerprint,
        cross_model_train_dir=mirror,
        require_cross_model_artifacts=True,
    )
    assert not _incremental_outputs_are_current(
        output,
        fingerprint,
        cross_model_train_dir=None,
        require_cross_model_artifacts=True,
    )


def test_generate_dry_run_isolates_explicit_fine_tuning_output(
    tmp_path: Path,
    monkeypatch,
) -> None:
    output = tmp_path / "manifest"
    fine_tuning_output = tmp_path / "fine-tuning"
    output.mkdir()
    fine_tuning_output.mkdir()
    (output / "generated.marker").write_text("manifest\n", encoding="utf-8")
    (fine_tuning_output / "sentinel.txt").write_text(
        "original fine-tuning\n",
        encoding="utf-8",
    )

    writes, printed, dry_run_root = _run_stubbed_generation(
        monkeypatch,
        root=tmp_path,
        output=output,
        fine_tuning_output=fine_tuning_output,
        dry_run=True,
    )

    assert dry_run_root is not None
    assert writes == [
        (dry_run_root / "agent_manifest", dry_run_root / "fine_tuning")
    ]
    assert (output / "generated.marker").read_text(encoding="utf-8") == "manifest\n"
    assert (fine_tuning_output / "sentinel.txt").read_text(encoding="utf-8") == (
        "original fine-tuning\n"
    )
    assert not (fine_tuning_output / "generated.marker").exists()

    diff_report = json.loads(printed[-1])
    assert diff_report["added"] == []
    assert diff_report["removed"] == []
    assert diff_report["modified"] == []
    assert diff_report["fine_tuning"]["existingDir"] == str(fine_tuning_output)
    assert diff_report["fine_tuning"]["generatedDir"] == str(
        dry_run_root / "fine_tuning"
    )
    assert diff_report["fine_tuning"]["changed"] is True
    assert diff_report["changed"] is True


def test_generate_non_dry_uses_explicit_fine_tuning_output(
    tmp_path: Path,
    monkeypatch,
) -> None:
    output = tmp_path / "manifest"
    fine_tuning_output = tmp_path / "fine-tuning"

    writes, _, dry_run_root = _run_stubbed_generation(
        monkeypatch,
        root=tmp_path,
        output=output,
        fine_tuning_output=fine_tuning_output,
        dry_run=False,
    )

    assert dry_run_root is None
    assert writes == [(output, fine_tuning_output)]
    assert (output / "generated.marker").read_text(encoding="utf-8") == "manifest\n"
    assert (fine_tuning_output / "generated.marker").read_text(encoding="utf-8") == (
        "fine-tuning\n"
    )


def test_generate_dry_run_propagates_external_cross_model_diff(
    tmp_path: Path,
    monkeypatch,
) -> None:
    output = tmp_path / "manifest"
    fine_tuning_output = tmp_path / "fine-tuning"
    cross_model_output = tmp_path / "cross-model"
    output.mkdir()
    fine_tuning_output.mkdir()
    cross_model_output.mkdir()
    (output / "generated.marker").write_text("manifest\n", encoding="utf-8")
    (fine_tuning_output / "generated.marker").write_text(
        "fine-tuning\n",
        encoding="utf-8",
    )

    _, printed, dry_run_root = _run_stubbed_generation(
        monkeypatch,
        root=tmp_path,
        output=output,
        fine_tuning_output=fine_tuning_output,
        dry_run=True,
        cross_model_train_dir=cross_model_output,
    )

    assert dry_run_root is not None
    diff_report = json.loads(printed[-1])
    assert diff_report["cross_model_training"]["changed"] is True
    assert diff_report["cross_model_training"]["added"] == [
        "generated.marker"
    ]
    assert diff_report["changed"] is True
