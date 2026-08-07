from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest


_CROSS_MODEL_FILENAMES = (
    "cross_model_training.jsonl",
    "cross_model_training_index.csv",
    "dpo_train_cross.jsonl",
    "dpo_val_cross.jsonl",
    "orchestration_evals.jsonl",
    "train_sft_cross.jsonl",
    "val_sft_cross.jsonl",
)

_MINIMAL_MANIFEST_FILES = (
    "AgentBehaviorManifest.json",
    "AgentBehaviorManifest.md",
    "fleet_system_prompts.json",
    "manifest_validation_report.json",
    "runtime_grounding_bundle.json",
    "runtime_grounding_prompt.md",
)

_SOURCE_INTEGRITY = {
    "baseCommit": "a" * 40,
    "workingTreeDigest": "b" * 64,
    "dirtyState": False,
}


def _resource_tree(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    repository = tmp_path / "repository"
    project = repository / "ios"
    project.mkdir(parents=True)
    manifest = repository / "generated" / "agent_manifest"
    dataset = manifest / "dataset"
    dataset.mkdir(parents=True)
    for filename in _MINIMAL_MANIFEST_FILES:
        (manifest / filename).write_text(f"{filename}\n", encoding="utf-8")
    for filename in ("codebase_home_corpus.jsonl", "codebase_home_sft.jsonl"):
        (dataset / filename).write_text("{}\n", encoding="utf-8")

    primary = repository / "generated" / "cross_model_training"
    nested = manifest / "cross_model_training"
    primary.mkdir(parents=True)
    nested.mkdir(parents=True)
    for filename in _CROSS_MODEL_FILENAMES:
        content = f"canonical {filename}\n"
        (primary / filename).write_text(content, encoding="utf-8")
        (nested / filename).write_text(content, encoding="utf-8")
    return repository, project, primary, nested


def _run_copy_script(
    *,
    repository: Path,
    project: Path,
    destination: Path,
    mode: str | None = "minimal",
    configuration: str | None = None,
) -> subprocess.CompletedProcess[str]:
    script = (
        Path(__file__).resolve().parents[3]
        / "ios"
        / "Lumen"
        / "Scripts"
        / "copy_agent_grounding_resources.sh"
    )
    environment: dict[str, str] = {
        **os.environ,
        "PROJECT_DIR": str(project),
        "TARGET_BUILD_DIR": str(destination),
        "UNLOCALIZED_RESOURCES_FOLDER_PATH": "Resources",
    }
    if mode is not None:
        environment["AGENT_GROUNDING_RESOURCE_MODE"] = mode
    else:
        environment.pop("AGENT_GROUNDING_RESOURCE_MODE", None)
    if configuration is not None:
        environment["CONFIGURATION"] = configuration
    return subprocess.run(
        ["sh", str(script)],
        cwd=repository,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )


def _run_generated_artifact_check(
    repository: Path,
) -> subprocess.CompletedProcess[str]:
    script = (
        Path(__file__).resolve().parents[3]
        / "scripts"
        / "check-generated-jsonl-artifacts.py"
    )
    return subprocess.run(
        ["python3", str(script)],
        cwd=repository,
        capture_output=True,
        text=True,
        check=False,
    )


def _write_valid_cross_model_directory(
    directory: Path,
    *,
    source_integrity: dict[str, object] = _SOURCE_INTEGRITY,
) -> None:
    directory.mkdir(parents=True)
    for filename in _CROSS_MODEL_FILENAMES:
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
        else:
            content = "{}\n" if filename.endswith(".jsonl") else "header\n"
        (directory / filename).write_text(content, encoding="utf-8")


def _write_manifest(
    repository: Path,
    source_integrity: dict[str, object],
) -> None:
    path = repository / "generated" / "agent_manifest" / "AgentBehaviorManifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"sourceIntegrity": source_integrity}) + "\n",
        encoding="utf-8",
    )


def test_minimal_resource_copy_excludes_developer_training_corpora(
    tmp_path: Path,
) -> None:
    repository, project, _, _ = _resource_tree(tmp_path)
    destination = tmp_path / "build"

    completed = _run_copy_script(
        repository=repository,
        project=project,
        destination=destination,
    )

    assert completed.returncode == 0, completed.stderr
    copied_manifest = (
        destination
        / "Resources"
        / "AgentGrounding"
        / "agent_manifest"
        / "AgentBehaviorManifest.json"
    )
    grounding_root = destination / "Resources" / "AgentGrounding"
    assert copied_manifest.read_text(encoding="utf-8") == "AgentBehaviorManifest.json\n"
    assert not (grounding_root / "agent_manifest" / "dataset").exists()
    assert list((grounding_root / "cross_model_training").iterdir()) == []


def test_release_defaults_to_minimal_runtime_resources(tmp_path: Path) -> None:
    repository, project, _, _ = _resource_tree(tmp_path)
    destination = tmp_path / "build"

    completed = _run_copy_script(
        repository=repository,
        project=project,
        destination=destination,
        mode=None,
        configuration="Release",
    )

    assert completed.returncode == 0, completed.stderr
    grounding_root = destination / "Resources" / "AgentGrounding"
    assert (grounding_root / "agent_manifest" / "runtime_grounding_bundle.json").is_file()
    assert not (grounding_root / "agent_manifest" / "dataset").exists()
    assert list((grounding_root / "cross_model_training").iterdir()) == []


def test_resource_copy_rejects_divergent_cross_model_mirrors(
    tmp_path: Path,
) -> None:
    repository, project, _, nested = _resource_tree(tmp_path)
    (nested / "train_sft_cross.jsonl").write_text(
        "stale train_sft_cross.jsonl\n",
        encoding="utf-8",
    )

    completed = _run_copy_script(
        repository=repository,
        project=project,
        destination=tmp_path / "build",
        mode="full",
    )

    assert completed.returncode != 0
    assert (
        "Cross-model resource mirrors diverged for train_sft_cross.jsonl"
        in completed.stderr
    )


@pytest.mark.parametrize("present_mirror", ["primary", "nested", "neither"])
def test_generated_artifact_gate_requires_both_cross_model_mirrors(
    tmp_path: Path,
    present_mirror: str,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    directory = (
        repository / "generated" / "cross_model_training"
        if present_mirror == "primary"
        else repository
        / "generated"
        / "agent_manifest"
        / "cross_model_training"
    )
    if present_mirror != "neither":
        _write_valid_cross_model_directory(directory)

    completed = _run_generated_artifact_check(repository)

    assert completed.returncode != 0
    assert "both required mirror directories must exist" in completed.stdout


def test_generated_artifact_gate_rejects_unexpected_mirror_entries(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    primary = repository / "generated" / "cross_model_training"
    nested = (
        repository
        / "generated"
        / "agent_manifest"
        / "cross_model_training"
    )
    _write_valid_cross_model_directory(primary)
    _write_valid_cross_model_directory(nested)
    _write_manifest(repository, _SOURCE_INTEGRITY)
    (nested / "stale-copy.jsonl").write_text("{}\n", encoding="utf-8")

    completed = _run_generated_artifact_check(repository)

    assert completed.returncode != 0
    assert "unexpected mirror entries" in completed.stdout


def test_generated_artifact_gate_accepts_current_identical_mirrors(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    primary = repository / "generated" / "cross_model_training"
    nested = (
        repository
        / "generated"
        / "agent_manifest"
        / "cross_model_training"
    )
    _write_valid_cross_model_directory(primary)
    _write_valid_cross_model_directory(nested)
    _write_manifest(repository, _SOURCE_INTEGRITY)

    completed = _run_generated_artifact_check(repository)

    assert completed.returncode == 0, completed.stdout


def test_generated_artifact_gate_rejects_identical_stale_lineage(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    primary = repository / "generated" / "cross_model_training"
    nested = (
        repository
        / "generated"
        / "agent_manifest"
        / "cross_model_training"
    )
    _write_valid_cross_model_directory(primary)
    _write_valid_cross_model_directory(nested)
    _write_manifest(
        repository,
        {**_SOURCE_INTEGRITY, "baseCommit": "c" * 40},
    )

    completed = _run_generated_artifact_check(repository)

    assert completed.returncode != 0
    assert "row lineage does not match" in completed.stdout
