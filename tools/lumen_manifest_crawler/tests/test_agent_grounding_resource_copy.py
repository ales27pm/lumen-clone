from __future__ import annotations

import hashlib
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
    "AgentBehaviorManifest.sha256",
    "AgentBehaviorManifest.md",
    "fleet_system_prompts.json",
    "manifest_validation_report.json",
    "runtime_grounding_bundle.json",
    "runtime_grounding_prompt.md",
)

_SOURCE_FILE_RELATIVE_PATH = "ios/Lumen/Models/SyntheticGroundingSource.swift"
_SOURCE_FILE_BYTES = b"struct SyntheticGroundingSource {}\n"
_SOURCE_INTEGRITY = {
    "baseCommit": "a" * 40,
    "workingTreeDigest": "b" * 64,
    "dirtyState": False,
    "files": [
        {
            "path": _SOURCE_FILE_RELATIVE_PATH,
            "sha256": hashlib.sha256(_SOURCE_FILE_BYTES).hexdigest(),
        }
    ],
}


def _write_resource_manifest(
    repository: Path,
    source_integrity: dict[str, object],
) -> None:
    manifest_bytes = (
        json.dumps({"sourceIntegrity": source_integrity}, sort_keys=True) + "\n"
    ).encode("utf-8")
    manifest = repository / "generated" / "agent_manifest"
    (manifest / "AgentBehaviorManifest.json").write_bytes(manifest_bytes)
    (manifest / "AgentBehaviorManifest.sha256").write_text(
        hashlib.sha256(manifest_bytes).hexdigest() + "\n",
        encoding="utf-8",
    )
    ios_mirror = repository / "ios" / "Lumen" / "AgentBehaviorManifest.json"
    ios_mirror.parent.mkdir(parents=True, exist_ok=True)
    ios_mirror.write_bytes(manifest_bytes)


def _resource_tree(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    repository = tmp_path / "repository"
    project = repository / "ios"
    project.mkdir(parents=True)
    manifest = repository / "generated" / "agent_manifest"
    dataset = manifest / "dataset"
    dataset.mkdir(parents=True)

    source_file = repository / _SOURCE_FILE_RELATIVE_PATH
    source_file.parent.mkdir(parents=True)
    source_file.write_bytes(_SOURCE_FILE_BYTES)
    _write_resource_manifest(repository, _SOURCE_INTEGRITY)
    (manifest / "AgentBehaviorManifest.md").write_text(
        "# Synthetic manifest fixture\n",
        encoding="utf-8",
    )
    (manifest / "fleet_system_prompts.json").write_text(
        "{}\n",
        encoding="utf-8",
    )
    (manifest / "manifest_validation_report.json").write_text(
        json.dumps({"passed": True, "failures": [], "warnings": []}) + "\n",
        encoding="utf-8",
    )
    (manifest / "runtime_grounding_bundle.json").write_text(
        "{}\n",
        encoding="utf-8",
    )
    (manifest / "runtime_grounding_prompt.md").write_text(
        "Synthetic runtime grounding prompt.\n",
        encoding="utf-8",
    )

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
    else:
        environment.pop("CONFIGURATION", None)
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
            row_integrity = {
                key: source_integrity[key]
                for key in ("baseCommit", "workingTreeDigest", "dirtyState")
            }
            content = json.dumps(
                {
                    "metadata": {
                        "manifestCommit": source_integrity["baseCommit"],
                        "sourceIntegrity": row_integrity,
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
    source_manifest = (
        repository
        / "generated"
        / "agent_manifest"
        / "AgentBehaviorManifest.json"
    )
    assert copied_manifest.read_bytes() == source_manifest.read_bytes()
    copied_hash = grounding_root / "agent_manifest" / "AgentBehaviorManifest.sha256"
    assert copied_hash.is_file()
    assert copied_hash.read_text(encoding="utf-8").strip() == hashlib.sha256(
        copied_manifest.read_bytes()
    ).hexdigest()
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


@pytest.mark.parametrize("missing_filename", _MINIMAL_MANIFEST_FILES)
def test_minimal_resource_copy_fails_closed_when_runtime_resource_is_missing(
    tmp_path: Path,
    missing_filename: str,
) -> None:
    repository, project, _, _ = _resource_tree(tmp_path)
    (
        repository
        / "generated"
        / "agent_manifest"
        / missing_filename
    ).unlink()

    completed = _run_copy_script(
        repository=repository,
        project=project,
        destination=tmp_path / "build",
        mode="minimal",
        configuration="Release",
    )

    assert completed.returncode != 0
    assert "[missing_required_resource]" in completed.stderr
    assert missing_filename in completed.stderr


@pytest.mark.parametrize(
    ("report", "expected_error"),
    (
        ("not-json\n", "manifest_validation_report_invalid"),
        (
            {"passed": False, "failures": [], "warnings": []},
            "manifest_validation_report_not_passed",
        ),
        (
            {"passed": True, "failures": [{"code": "fixture"}], "warnings": []},
            "manifest_validation_report_failures",
        ),
        (
            {"passed": True, "failures": [], "warnings": [{"code": "fixture"}]},
            "manifest_validation_report_warnings",
        ),
    ),
)
def test_release_resource_copy_rejects_invalid_validation_report(
    tmp_path: Path,
    report: object,
    expected_error: str,
) -> None:
    repository, project, _, _ = _resource_tree(tmp_path)
    report_path = (
        repository
        / "generated"
        / "agent_manifest"
        / "manifest_validation_report.json"
    )
    if isinstance(report, str):
        report_path.write_text(report, encoding="utf-8")
    else:
        report_path.write_text(json.dumps(report) + "\n", encoding="utf-8")

    completed = _run_copy_script(
        repository=repository,
        project=project,
        destination=tmp_path / "build",
        mode=None,
        configuration="Release",
    )

    assert completed.returncode != 0
    assert f"[{expected_error}]" in completed.stderr


def test_release_resource_copy_rejects_stale_ios_manifest_mirror(
    tmp_path: Path,
) -> None:
    repository, project, _, _ = _resource_tree(tmp_path)
    (project / "Lumen" / "AgentBehaviorManifest.json").write_text(
        '{"sourceIntegrity":{"baseCommit":"stale"}}\n',
        encoding="utf-8",
    )

    completed = _run_copy_script(
        repository=repository,
        project=project,
        destination=tmp_path / "build",
        mode=None,
        configuration="Release",
    )

    assert completed.returncode != 0
    assert "[manifest_mirror_diverged]" in completed.stderr


def test_release_resource_copy_rejects_corrupt_manifest_hash_sidecar(
    tmp_path: Path,
) -> None:
    repository, project, _, _ = _resource_tree(tmp_path)
    hash_path = (
        repository
        / "generated"
        / "agent_manifest"
        / "AgentBehaviorManifest.sha256"
    )
    hash_path.write_text("0" * 64 + "\n", encoding="utf-8")

    completed = _run_copy_script(
        repository=repository,
        project=project,
        destination=tmp_path / "build",
        mode=None,
        configuration="Release",
    )

    assert completed.returncode != 0
    assert "[manifest_hash_mismatch]" in completed.stderr


def test_release_resource_copy_rejects_missing_manifest_source_file(
    tmp_path: Path,
) -> None:
    repository, project, _, _ = _resource_tree(tmp_path)
    (repository / _SOURCE_FILE_RELATIVE_PATH).unlink()

    completed = _run_copy_script(
        repository=repository,
        project=project,
        destination=tmp_path / "build",
        mode=None,
        configuration="Release",
    )

    assert completed.returncode != 0
    assert "[manifest_source_integrity_missing]" in completed.stderr
    assert _SOURCE_FILE_RELATIVE_PATH in completed.stderr


def test_release_resource_copy_rejects_manifest_source_path_traversal(
    tmp_path: Path,
) -> None:
    repository, project, _, _ = _resource_tree(tmp_path)
    outside_file = repository.parent / "outside.swift"
    outside_file.write_bytes(b"outside repository\n")
    traversal_integrity = {
        **_SOURCE_INTEGRITY,
        "files": [
            {
                "path": "../outside.swift",
                "sha256": hashlib.sha256(outside_file.read_bytes()).hexdigest(),
            }
        ],
    }
    _write_resource_manifest(repository, traversal_integrity)

    completed = _run_copy_script(
        repository=repository,
        project=project,
        destination=tmp_path / "build",
        mode=None,
        configuration="Release",
    )

    assert completed.returncode != 0
    assert "[manifest_source_integrity_path_traversal]" in completed.stderr
    assert "../outside.swift" in completed.stderr


def test_release_resource_copy_rejects_manifest_source_hash_mismatch(
    tmp_path: Path,
) -> None:
    repository, project, _, _ = _resource_tree(tmp_path)
    (repository / _SOURCE_FILE_RELATIVE_PATH).write_bytes(
        b"struct SyntheticGroundingSource { let stale = true }\n"
    )

    completed = _run_copy_script(
        repository=repository,
        project=project,
        destination=tmp_path / "build",
        mode=None,
        configuration="Release",
    )

    assert completed.returncode != 0
    assert "[manifest_source_integrity_mismatch]" in completed.stderr
    assert _SOURCE_FILE_RELATIVE_PATH in completed.stderr


def test_release_resource_copy_rejects_skip_mode(tmp_path: Path) -> None:
    repository, project, _, _ = _resource_tree(tmp_path)

    completed = _run_copy_script(
        repository=repository,
        project=project,
        destination=tmp_path / "build",
        mode="skip",
        configuration="Release",
    )

    assert completed.returncode != 0
    assert "[resource_mode_skip]" in completed.stderr


def test_debug_full_mode_emits_typed_runtime_fallback_diagnostic(
    tmp_path: Path,
) -> None:
    repository, project, _, _ = _resource_tree(tmp_path)
    (
        repository
        / "generated"
        / "agent_manifest"
        / "runtime_grounding_bundle.json"
    ).unlink()
    destination = tmp_path / "build"

    completed = _run_copy_script(
        repository=repository,
        project=project,
        destination=destination,
        mode="full",
        configuration="Debug",
    )

    assert completed.returncode == 0, completed.stderr
    assert (
        "DIAGNOSTIC code=missing_required_resource severity=warning "
        "action=runtime-fallback"
    ) in completed.stdout
    grounding_root = destination / "Resources" / "AgentGrounding"
    assert list((grounding_root / "agent_manifest").iterdir()) == []
    assert list((grounding_root / "cross_model_training").iterdir()) == []


def test_debug_full_mode_source_mismatch_uses_typed_runtime_fallback(
    tmp_path: Path,
) -> None:
    repository, project, _, _ = _resource_tree(tmp_path)
    (repository / _SOURCE_FILE_RELATIVE_PATH).write_bytes(b"stale source\n")
    destination = tmp_path / "build"

    completed = _run_copy_script(
        repository=repository,
        project=project,
        destination=destination,
        mode="full",
        configuration="Debug",
    )

    assert completed.returncode == 0, completed.stderr
    assert (
        "DIAGNOSTIC code=manifest_source_integrity_mismatch severity=warning "
        "action=runtime-fallback"
    ) in completed.stdout
    grounding_root = destination / "Resources" / "AgentGrounding"
    assert list((grounding_root / "agent_manifest").iterdir()) == []
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
