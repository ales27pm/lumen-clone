from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[4]
MODULE_PATH = ROOT / "tools/fine_tuning/unsloth/training_lineage.py"
SPEC = importlib.util.spec_from_file_location("training_lineage_under_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
training_lineage = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(training_lineage)


def test_repository_code_bundle_is_phase_specific_and_self_verifying() -> None:
    bundle = training_lineage.repository_training_code_bundle(ROOT)

    assert set(bundle["phases"]) == {"sft", "dpo", "orpo"}
    assert bundle["phases"]["sft"]["trainingCodeSHA256"] != bundle["phases"]["dpo"]["trainingCodeSHA256"]
    assert any(
        entry["path"] == "lumen_training/train_dpo.py"
        for entry in bundle["phases"]["dpo"]["files"]
    )
    assert any(
        entry["path"] == "lumen_training/train_sft.py"
        for entry in bundle["phases"]["sft"]["files"]
    )
    assert any(
        entry["path"]
        == "lumen_manifest_crawler/dataset/public_adapter_eval_sources.json"
        for entry in bundle["phases"]["sft"]["files"]
    )
    policy = bundle["phases"]["sft"]["closurePolicy"]
    assert policy["coveredLogicalPaths"] == [
        "app.py",
        "lumen_manifest_crawler",
        "lumen_training",
        "requirements.txt",
    ]
    assert policy["rejectUnlistedBehaviorFiles"] is True
    assert training_lineage.verify_training_code_bundle(bundle) == bundle["trainingCodeSHA256"]


def test_deployed_code_mutation_fails_manifest_verification(tmp_path: Path) -> None:
    deployed = tmp_path / "deployed"
    deployed.mkdir()
    (deployed / "trainer.py").write_text("print('one')\n", encoding="utf-8")
    manifest = training_lineage.build_training_code_manifest(
        phase="sft",
        files={"trainer.py": deployed / "trainer.py"},
    )
    training_lineage.verify_training_code_manifest(manifest, root=deployed)

    (deployed / "trainer.py").write_text("print('two')\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Deployed training-code drift"):
        training_lineage.verify_training_code_manifest(manifest, root=deployed)


def test_unlisted_behavior_file_fails_bidirectional_closure_verification(
    tmp_path: Path,
) -> None:
    deployed = tmp_path / "deployed"
    package = deployed / "package"
    package.mkdir(parents=True)
    (package / "declared.py").write_text("VALUE = 1\n", encoding="utf-8")
    manifest = training_lineage.build_training_code_manifest(
        phase="sft",
        files={"package/declared.py": package / "declared.py"},
        closure_policy={
            "includedExtensions": [".py"],
            "coveredLogicalPaths": ["package"],
            "excludedLogicalPaths": [],
            "excludedDirectoryNames": ["__pycache__"],
            "coverDeployedRoot": False,
            "rejectUnlistedBehaviorFiles": True,
        },
    )
    training_lineage.verify_training_code_manifest(manifest, root=deployed)

    (package / "unlisted.py").write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unlisted behavior-affecting"):
        training_lineage.verify_training_code_manifest(manifest, root=deployed)


def test_requirements_mutation_and_dependency_drift_fail_closed(tmp_path: Path) -> None:
    source = ROOT / "tools/hf_zerogpu/space_template/requirements.txt"
    requirements = tmp_path / "requirements.txt"
    requirements.write_bytes(source.read_bytes())
    lock = training_lineage.build_training_dependency_lock(requirements)
    training_lineage.verify_training_dependency_lock(
        lock,
        requirements_path=requirements,
    )

    requirements.write_text(
        requirements.read_text(encoding="utf-8") + "unexpected-package==1.0.0\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="dependency set drifted"):
        training_lineage.verify_training_dependency_lock(
            lock,
            requirements_path=requirements,
        )

    requirements.write_text(
        source.read_text(encoding="utf-8").replace(
            "trl==0.24.0",
            "trl==0.25.0",
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="version for trl drifted"):
        training_lineage.build_training_dependency_lock(requirements)

    installed = dict(lock["packageVersions"])
    installed["trl"] = "0.25.0"
    with pytest.raises(ValueError, match="Installed controlled package versions drifted"):
        training_lineage.verify_training_dependency_lock(
            lock,
            installed_versions=installed,
        )
    with pytest.raises(ValueError, match="Runtime Python version drifted"):
        training_lineage.verify_training_dependency_lock(
            lock,
            runtime_python_version="3.11",
        )
    with pytest.raises(ValueError, match="Runtime CUDA version drifted"):
        training_lineage.verify_training_dependency_lock(
            lock,
            runtime_cuda_version="12.9",
        )


def test_dependency_lock_covers_every_direct_requirement() -> None:
    requirements = ROOT / "tools/hf_zerogpu/space_template/requirements.txt"
    lock = training_lineage.build_training_dependency_lock(requirements)
    names = {
        training_lineage._requirement_name(line)
        for line in requirements.read_text(encoding="utf-8").splitlines()
        if training_lineage._requirement_name(line) is not None
    }

    assert names == {*lock["packageVersions"], "unsloth"}
    assert lock["requirementsSHA256"] == training_lineage.file_sha256(requirements)
    assert lock["trainingDependencyLockSHA256"] == training_lineage.canonical_sha256(
        {
            key: value
            for key, value in lock.items()
            if key != "trainingDependencyLockSHA256"
        }
    )


@pytest.mark.parametrize(
    ("kind", "revision"),
    [
        ("unresolved", None),
        ("git", "main"),
        ("huggingface_space", "a" * 39),
    ],
)
def test_runtime_source_requires_immutable_revision(kind: object, revision: object) -> None:
    with pytest.raises(ValueError):
        training_lineage.validate_runtime_source(kind=kind, revision=revision)


def test_runtime_source_accepts_git_and_space_commits() -> None:
    assert training_lineage.validate_runtime_source(
        kind="git",
        revision="a" * 40,
    ) == ("git", "a" * 40)
    assert training_lineage.validate_runtime_source(
        kind="huggingface_space",
        revision="b" * 40,
    ) == ("huggingface_space", "b" * 40)


def test_space_runtime_source_repository_head_remains_unverified() -> None:
    revision = "a" * 40
    audit = {
        "runtimeSourceKind": "huggingface_space",
        "runtimeSourceRevision": revision,
        "expectedRuntimeSourceRevision": revision,
        "observedRepositoryRevision": revision,
        "observedRuntimeRevision": None,
        "runtimeSourceBindingStatus": "operator_declared_unverified",
        "runtimeSourceBindingMethod": "huggingface_repository_head_supplemental",
    }

    assert training_lineage.validate_runtime_source_audit(audit) == audit


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("runtimeSourceBindingStatus", "verified"),
        ("runtimeSourceBindingMethod", "self_declared_verified"),
        ("observedRuntimeRevision", "a" * 40),
        ("observedRepositoryRevision", "b" * 40),
    ],
)
def test_space_runtime_source_rejects_overstated_or_mismatched_evidence(
    field: str,
    value: object,
) -> None:
    revision = "a" * 40
    audit = {
        "runtimeSourceKind": "huggingface_space",
        "runtimeSourceRevision": revision,
        "expectedRuntimeSourceRevision": revision,
        "observedRepositoryRevision": revision,
        "observedRuntimeRevision": None,
        "runtimeSourceBindingStatus": "operator_declared_unverified",
        "runtimeSourceBindingMethod": "huggingface_repository_head_supplemental",
    }
    audit[field] = value

    with pytest.raises(ValueError):
        training_lineage.validate_runtime_source_audit(audit)


def test_local_runtime_source_requires_independently_observed_matching_head() -> None:
    revision = "c" * 40
    audit = {
        "runtimeSourceKind": "git",
        "runtimeSourceRevision": revision,
        "expectedRuntimeSourceRevision": revision,
        "observedRepositoryRevision": revision,
        "observedRuntimeRevision": revision,
        "runtimeSourceBindingStatus": "local_checkout_observed",
        "runtimeSourceBindingMethod": "git_head_plus_training_code_manifest",
    }

    assert training_lineage.validate_runtime_source_audit(
        audit,
        observed_local_revision=revision,
    ) == audit
    with pytest.raises(ValueError, match="independently observed HEAD"):
        training_lineage.validate_runtime_source_audit(audit)
    with pytest.raises(ValueError, match="does not match"):
        training_lineage.validate_runtime_source_audit(
            audit,
            observed_local_revision="d" * 40,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("observedRepositoryRevision", "d" * 40),
        ("observedRuntimeRevision", None),
        ("runtimeSourceBindingStatus", "operator_declared_unverified"),
        ("runtimeSourceBindingMethod", "operator_declared_only"),
    ],
)
def test_local_runtime_source_rejects_incomplete_or_self_declared_evidence(
    field: str,
    value: object,
) -> None:
    revision = "c" * 40
    audit = {
        "runtimeSourceKind": "git",
        "runtimeSourceRevision": revision,
        "expectedRuntimeSourceRevision": revision,
        "observedRepositoryRevision": revision,
        "observedRuntimeRevision": revision,
        "runtimeSourceBindingStatus": "local_checkout_observed",
        "runtimeSourceBindingMethod": "git_head_plus_training_code_manifest",
    }
    audit[field] = value

    with pytest.raises(ValueError):
        training_lineage.validate_runtime_source_audit(
            audit,
            observed_local_revision=revision,
        )


def test_ubuntu_launcher_records_complete_local_git_source_audit() -> None:
    launcher = (ROOT / "scripts/ubuntu_train_lumen_adapters_aio.sh").read_text(
        encoding="utf-8"
    )

    assert '"expectedRuntimeSourceRevision": runtime_source_revision' in launcher
    assert '"observedRepositoryRevision": runtime_source_revision' in launcher
    assert '"observedRuntimeRevision": runtime_source_revision' in launcher
    assert '"runtimeSourceBindingStatus": "local_checkout_observed"' in launcher
    assert (
        '"runtimeSourceBindingMethod": "git_head_plus_training_code_manifest"'
        in launcher
    )
    assert "cfg.update(local_runtime_source)" in launcher
    assert "**local_runtime_source" in launcher
