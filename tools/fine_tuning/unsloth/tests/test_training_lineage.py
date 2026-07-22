from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
from collections import namedtuple
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[4]
MODULE_PATH = ROOT / "tools/fine_tuning/unsloth/training_lineage.py"
SPEC = importlib.util.spec_from_file_location("training_lineage_under_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
training_lineage = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(training_lineage)


class _InstalledDistribution:
    def __init__(
        self,
        root: Path,
        *,
        name: str,
        version: str,
        direct_url: dict[str, object] | None = None,
    ) -> None:
        self.root = root
        self.metadata = {"Name": name}
        self.version = version
        self._direct_url = direct_url
        package = root / name.replace("-", "_")
        package.mkdir(parents=True)
        init_path = package / "__init__.py"
        init_path.write_text(
            f"__version__ = {version!r}\n",
            encoding="utf-8",
        )
        digest = base64.urlsafe_b64encode(
            hashlib.sha256(init_path.read_bytes()).digest()
        ).decode("ascii").rstrip("=")
        dist_info = root / f"{name.replace('-', '_')}-{version}.dist-info"
        dist_info.mkdir()
        self.record_path = dist_info / "RECORD"
        self.record_path.write_text(
            f"{package.name}/__init__.py,sha256={digest},{init_path.stat().st_size}\n"
            f"{dist_info.name}/RECORD,,\n",
            encoding="utf-8",
        )

    def read_text(self, filename: str) -> str | None:
        if filename == "RECORD":
            return self.record_path.read_text(encoding="utf-8")
        if filename == "INSTALLER":
            return "uv\n"
        if filename == "direct_url.json" and self._direct_url is not None:
            return json.dumps(self._direct_url)
        return None

    def locate_file(self, filename: str) -> Path:
        return self.root / filename

    @property
    def files(self) -> list[str]:
        return [
            row.split(",", 1)[0]
            for row in self.record_path.read_text(encoding="utf-8").splitlines()
            if row and not row.split(",", 1)[0].endswith("/RECORD")
        ]


def _resign_resolved_environment(environment: dict[str, object]) -> None:
    distributions = environment["distributions"]
    assert isinstance(distributions, list)
    for entry in distributions:
        assert isinstance(entry, dict)
        unsigned = {
            key: value
            for key, value in entry.items()
            if key != "distributionSHA256"
        }
        entry["distributionSHA256"] = training_lineage.canonical_sha256(unsigned)
    payload = {
        key: value
        for key, value in environment.items()
        if key != "resolvedTrainingEnvironmentSHA256"
    }
    environment["resolvedTrainingEnvironmentSHA256"] = (
        training_lineage.canonical_sha256(payload)
    )


def _tokenizer_snapshot_fixture(
    tmp_path: Path,
) -> tuple[Path, dict[str, object], dict[str, bytes]]:
    revision = "7" * 40
    model_root = tmp_path / "models--example--model"
    blobs = model_root / "blobs"
    snapshot = model_root / "snapshots" / revision
    blobs.mkdir(parents=True)
    snapshot.mkdir(parents=True)
    payloads = {
        "config.json": b'{"model_type":"qwen3"}\n',
        "merges.txt": b"a b\nc d\n",
        "tokenizer.json": b'{"version":"1.0"}\n',
        "tokenizer_config.json": b'{"chat_template":"test"}\n',
        "vocab.json": b'{"a":0,"b":1}\n',
    }
    files: list[dict[str, object]] = []
    for filename, payload in payloads.items():
        digest = hashlib.sha256(payload).hexdigest()
        (blobs / digest).write_bytes(payload)
        (snapshot / filename).symlink_to(Path("../../blobs") / digest)
        files.append(
            {
                "path": filename,
                "sizeBytes": len(payload),
                "sha256": digest,
                "huggingFaceBlobID": digest,
            }
        )
    closure = training_lineage.canonical_base_model_tokenizer_closure(
        base_model_id="example/model",
        base_model_revision=revision,
        files=files,
    )
    contract: dict[str, object] = {
        "base_model_id": "example/model",
        "base_model_name": "example/model",
        "base_model_revision": revision,
        "tokenizer_files": closure["files"],
        "tokenizer_digest": next(
            item["sha256"]
            for item in closure["files"]
            if item["path"] == "tokenizer.json"
        ),
        "tokenizer_closure_sha256": training_lineage.canonical_sha256(
            closure
        ),
    }
    return snapshot, contract, payloads


def _private_tokenizer_snapshot(
    tmp_path: Path,
    payloads: dict[str, bytes],
) -> Path:
    snapshot = tmp_path / "private-tokenizer"
    snapshot.mkdir(mode=0o700)
    for filename, payload in payloads.items():
        path = snapshot / filename
        path.write_bytes(payload)
        path.chmod(0o400)
    return snapshot


@pytest.mark.parametrize(
    "filename",
    ("config.json", "merges.txt", "tokenizer_config.json", "vocab.json"),
)
def test_hugging_face_tokenizer_snapshot_rejects_each_non_tokenizer_mutation(
    tmp_path: Path,
    filename: str,
) -> None:
    snapshot, contract, _ = _tokenizer_snapshot_fixture(tmp_path)
    training_lineage.verify_base_model_tokenizer_snapshot(snapshot, **contract)

    target = (snapshot / filename).resolve(strict=True)
    target.chmod(0o600)
    target.write_bytes(target.read_bytes() + b"tampered")

    with pytest.raises(ValueError, match=filename):
        training_lineage.verify_base_model_tokenizer_snapshot(
            snapshot,
            **contract,
        )


def test_private_tokenizer_snapshot_binds_regular_files_and_stability_signatures(
    tmp_path: Path,
) -> None:
    _, contract, payloads = _tokenizer_snapshot_fixture(tmp_path)
    private = _private_tokenizer_snapshot(tmp_path, payloads)

    verification = training_lineage.verify_private_base_model_tokenizer_snapshot(
        private,
        **contract,
    )

    assert verification["snapshotPath"] == str(private.resolve())
    assert [
        item["path"] for item in verification["fileStabilitySignatures"]
    ] == list(training_lineage.BASE_MODEL_TOKENIZER_REQUIRED_PATHS)
    assert verification["snapshotVerificationSHA256"] == (
        training_lineage.canonical_sha256(
            {
                key: value
                for key, value in verification.items()
                if key != "snapshotVerificationSHA256"
            }
        )
    )


def test_private_conversion_snapshot_retains_tokenizer_and_model_signatures(
    tmp_path: Path,
) -> None:
    _, tokenizer_contract, payloads = _tokenizer_snapshot_fixture(tmp_path)
    conversion = tmp_path / "private-conversion"
    conversion.mkdir(mode=0o700)
    for filename, payload in payloads.items():
        path = conversion / filename
        path.write_bytes(payload)
        path.chmod(0o400)
    shard_payloads = {
        "model-00001-of-00002.safetensors": b"first-shard",
        "model-00002-of-00002.safetensors": b"second-shard",
    }
    shard_root = tmp_path / "weight-blobs"
    shard_root.mkdir()
    generation_payload = b'{"do_sample":false}\n'
    generation_path = conversion / "generation_config.json"
    generation_path.write_bytes(generation_payload)
    generation_path.chmod(0o400)
    generation_config_file = {
        "path": "generation_config.json",
        "sizeBytes": len(generation_payload),
        "sha256": hashlib.sha256(generation_payload).hexdigest(),
        "huggingFaceBlobID": hashlib.sha256(generation_payload).hexdigest(),
    }
    shards: list[dict[str, object]] = []
    for filename, payload in shard_payloads.items():
        target = conversion / filename
        target.write_bytes(payload)
        target.chmod(0o400)
        shards.append(
            {
                "filename": filename,
                "size": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    shards.sort(key=lambda item: str(item["filename"]))
    index_payload = json.dumps(
        {
            "weight_map": {
                "a": shards[0]["filename"],
                "b": shards[1]["filename"],
            }
        },
        sort_keys=True,
    ).encode("utf-8")
    index_path = conversion / "model.safetensors.index.json"
    index_path.write_bytes(index_payload)
    index_path.chmod(0o400)
    artifact_digest = training_lineage.canonical_sha256(
        {
            "schemaVersion": "lumen.base-model-weight-shards/1.0.0",
            "shards": shards,
        }
    )
    index_digest = hashlib.sha256(index_payload).hexdigest()
    index_binding = training_lineage.canonical_sha256(
        {
            "schemaVersion": "lumen.base-model-index-shard-binding/1.0.0",
            "indexDigest": index_digest,
            "referencedShardNames": [item["filename"] for item in shards],
            "shardContractDigest": artifact_digest,
        }
    )

    verification = (
        training_lineage.verify_private_base_model_conversion_snapshot(
            conversion,
            **tokenizer_contract,
            model_index_digest=index_digest,
            index_referenced_shard_names=[
                item["filename"] for item in shards
            ],
            index_shard_binding_sha256=index_binding,
            model_artifact_digest=artifact_digest,
            weight_shards=shards,
            generation_config_file=generation_config_file,
        )
    )

    assert verification["snapshotPath"] == str(conversion.resolve())
    assert len(verification["modelFileStabilitySignatures"]) == 4
    assert verification["tokenizerSnapshotVerification"][
        "fileStabilitySignatures"
    ]


def test_private_runtime_snapshot_fails_before_copy_when_space_is_insufficient(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, contract, payloads = _tokenizer_snapshot_fixture(tmp_path)
    private = _private_tokenizer_snapshot(tmp_path, payloads)
    source = tmp_path / "source-model"
    source.mkdir()
    DiskUsage = namedtuple("DiskUsage", ("total", "used", "free"))
    monkeypatch.setattr(
        training_lineage.shutil,
        "disk_usage",
        lambda _path: DiskUsage(1, 1, 0),
    )

    with pytest.raises(ValueError, match="Insufficient free space"):
        training_lineage.create_private_base_model_runtime_snapshot(
            source_snapshot_dir=source,
            private_tokenizer_snapshot_dir=private,
            destination=tmp_path / "private-model",
            **contract,
            generation_config_file={
                "path": "generation_config.json",
                "sizeBytes": 1,
                "sha256": "1" * 64,
                "huggingFaceBlobID": "1" * 64,
            },
            model_index_digest="2" * 64,
            index_referenced_shard_names=["model.safetensors"],
            index_shard_binding_sha256="3" * 64,
            model_artifact_digest="4" * 64,
            weight_shards=[
                {
                    "filename": "model.safetensors",
                    "size": 1,
                    "sha256": "5" * 64,
                }
            ],
        )


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
        == "lumen_manifest_crawler/dataset/chat_template_contract.py"
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


def test_space_configuration_is_canonical_and_bound_to_runtime_front_matter(
    tmp_path: Path,
) -> None:
    source = ROOT / "tools/hf_zerogpu/space_template/README.md"
    readme = tmp_path / "README.md"
    readme.write_bytes(source.read_bytes())

    configuration = training_lineage.build_space_configuration(readme)

    assert configuration == {
        "schemaVersion": "lumen.zerogpu.space-configuration/1.0.0",
        "sdk": "gradio",
        "appFile": "app.py",
        "pythonVersion": "3.10",
        "suggestedHardware": None,
        "spaceConfigurationSHA256": training_lineage.canonical_sha256(
            {
                "schemaVersion": "lumen.zerogpu.space-configuration/1.0.0",
                "sdk": "gradio",
                "appFile": "app.py",
                "pythonVersion": "3.10",
                "suggestedHardware": None,
            }
        ),
    }
    assert training_lineage.verify_space_configuration(
        configuration,
        readme_path=readme,
    ) == configuration["spaceConfigurationSHA256"]


@pytest.mark.parametrize(
    ("source", "replacement"),
    [
        ("sdk: gradio", "sdk: static"),
        ("app_file: app.py", "app_file: alternate.py"),
        ('python_version: "3.10"', 'python_version: "3.11"'),
    ],
)
def test_space_configuration_mutation_fails_runtime_verification(
    tmp_path: Path,
    source: str,
    replacement: str,
) -> None:
    template = ROOT / "tools/hf_zerogpu/space_template/README.md"
    readme = tmp_path / "README.md"
    readme.write_bytes(template.read_bytes())
    configuration = training_lineage.build_space_configuration(readme)
    readme.write_text(
        readme.read_text(encoding="utf-8").replace(source, replacement),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="runtime configuration drifted"):
        training_lineage.verify_space_configuration(
            configuration,
            readme_path=readme,
        )


def test_space_configuration_rejects_uncontrolled_runtime_metadata(
    tmp_path: Path,
) -> None:
    template = ROOT / "tools/hf_zerogpu/space_template/README.md"
    readme = tmp_path / "README.md"
    readme.write_text(
        template.read_text(encoding="utf-8").replace(
            'python_version: "3.10"',
            'python_version: "3.10"\nsuggested_hardware: zero-a10g',
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="Unsupported Space README front-matter field",
    ):
        training_lineage.build_space_configuration(readme)


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


def test_dependency_lock_accepts_only_its_exact_cuda_wheel_local_tag() -> None:
    requirements = ROOT / "tools/hf_zerogpu/space_template/requirements.txt"
    lock = training_lineage.build_training_dependency_lock(requirements)
    installed = dict(lock["packageVersions"])
    for name in ("torch", "torchvision", "torchaudio"):
        installed[name] += "+cu128"

    assert training_lineage.verify_training_dependency_lock(
        lock,
        installed_versions=installed,
    ) == lock["trainingDependencyLockSHA256"]

    installed["torch"] = "2.9.1+cu129"
    with pytest.raises(
        ValueError,
        match="Installed controlled package versions drifted",
    ):
        training_lineage.verify_training_dependency_lock(
            lock,
            installed_versions=installed,
        )

    installed["torch"] = "2.9.1+cu128"
    installed["trl"] = "0.24.0+cu128"
    with pytest.raises(
        ValueError,
        match="Installed controlled package versions drifted",
    ):
        training_lineage.verify_training_dependency_lock(
            lock,
            installed_versions=installed,
        )

    installed["trl"] = "0.24.0"
    installed["torch"] = "2.9.1+cpu"
    with pytest.raises(
        ValueError,
        match="Installed controlled package versions drifted",
    ):
        training_lineage.verify_training_dependency_lock(
            lock,
            installed_versions=installed,
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
    assert {
        name: lock["packageVersions"][name]
        for name in ("torch", "torchvision", "torchaudio")
    } == {
        "torch": "2.9.1",
        "torchvision": "0.24.1",
        "torchaudio": "2.9.1",
    }
    assert {
        name: lock["packageVersions"][name]
        for name in ("transformers", "gradio", "trackio", "huggingface_hub")
    } == {
        "transformers": "4.57.6",
        "gradio": "6.17.3",
        "trackio": "0.20.2",
        "huggingface_hub": "0.36.2",
    }
    assert lock["requirementsSHA256"] == training_lineage.file_sha256(requirements)
    assert lock["trainingDependencyLockSHA256"] == training_lineage.canonical_sha256(
        {
            key: value
            for key, value in lock.items()
            if key != "trainingDependencyLockSHA256"
        }
    )


def test_resolved_environment_attests_transitive_distribution_content(
    tmp_path: Path,
) -> None:
    direct = _InstalledDistribution(
        tmp_path,
        name="direct-package",
        version="1.0.0",
    )
    transitive = _InstalledDistribution(
        tmp_path,
        name="transitive-package",
        version="2.0.0",
        direct_url={
            "url": "https://github.com/example/transitive-package.git",
            "vcs_info": {
                "vcs": "git",
                "commit_id": "a" * 40,
                "requested_revision": "a" * 40,
            },
        },
    )
    resolved = training_lineage.build_resolved_training_environment(
        [transitive, direct]
    )

    assert [item["name"] for item in resolved["distributions"]] == [
        "direct-package",
        "transitive-package",
    ]
    assert resolved["distributions"][1]["directURL"]["vcs_info"][
        "commit_id"
    ] == "a" * 40
    assert training_lineage.verify_resolved_training_environment(
        resolved,
        distributions=[direct, transitive],
        verify_installed=True,
    ) == resolved["resolvedTrainingEnvironmentSHA256"]

    transitive_package = tmp_path / "transitive_package" / "__init__.py"
    transitive_package.write_text("__version__ = '2.0.1'\n", encoding="utf-8")
    with pytest.raises(ValueError, match="RECORD (?:size|content) mismatch"):
        training_lineage.verify_resolved_training_environment(
            resolved,
            distributions=[direct, transitive],
            verify_installed=True,
        )


def test_resolved_environment_snapshot_records_scan_metrics_and_authenticates_cache(
    tmp_path: Path,
) -> None:
    distribution = _InstalledDistribution(
        tmp_path,
        name="cached-package",
        version="1.0.0",
    )

    resolved, scan = training_lineage.build_resolved_training_environment_snapshot(
        [distribution]
    )

    assert scan["distributionCount"] == 1
    assert scan["installedFileCount"] == 1
    assert scan["totalHashedBytes"] > 0
    assert scan["durationMilliseconds"] >= 0
    key = b"k" * 32
    attestation = training_lineage.sign_resolved_training_environment_cache(
        resolved,
        scan,
        key=key,
        startup_id="a" * 32,
    )
    assert training_lineage.verify_resolved_training_environment_cache(
        resolved,
        attestation,
        key=key,
    ) == scan

    tampered = dict(attestation)
    tampered_scan = dict(scan)
    tampered_scan["totalHashedBytes"] += 1
    tampered["scan"] = tampered_scan
    with pytest.raises(ValueError, match="not authentic"):
        training_lineage.verify_resolved_training_environment_cache(
            resolved,
            tampered,
            key=key,
        )


def _rewrite_distribution_record(
    distribution: _InstalledDistribution,
    source_paths: list[Path],
) -> None:
    rows: list[str] = []
    for path in source_paths:
        digest = base64.urlsafe_b64encode(
            hashlib.sha256(path.read_bytes()).digest()
        ).decode("ascii").rstrip("=")
        rows.append(
            f"{path.relative_to(distribution.root).as_posix()},"
            f"sha256={digest},{path.stat().st_size}\n"
        )
    rows.append(
        f"{distribution.record_path.parent.name}/RECORD,,\n"
    )
    distribution.record_path.write_text("".join(rows), encoding="utf-8")


def test_installed_python_callable_identity_reconstructs_record_bound_code(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    distribution = _InstalledDistribution(
        tmp_path,
        name="callable-package",
        version="1.2.3",
    )
    init_path = tmp_path / "callable_package" / "__init__.py"
    source_path = tmp_path / "callable_package" / "loss_utils.py"
    source_path.write_text(
        "def selected_callable(value):\n    return value + 1\n",
        encoding="utf-8",
    )
    _rewrite_distribution_record(distribution, [init_path, source_path])
    resolved = training_lineage.build_resolved_training_environment(
        [distribution]
    )
    monkeypatch.setattr(
        training_lineage.importlib_metadata,
        "distribution",
        lambda name: distribution,
    )

    identity = (
        training_lineage.installed_distribution_python_callable_identity(
            distribution_name="callable-package",
            source_logical_path="callable_package/loss_utils.py",
            callable_name="selected_callable",
            resolved_environment=resolved,
        )
    )

    assert identity["distributionName"] == "callable-package"
    assert identity["distributionVersion"] == "1.2.3"
    assert identity["sourceFileSHA256"] == hashlib.sha256(
        source_path.read_bytes()
    ).hexdigest()
    assert len(identity["codeSHA256"]) == 64
    assert len(identity["installedCallableIdentitySHA256"]) == 64

    source_path.write_text(
        "def selected_callable(value):\n    return value + 2\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="RECORD content mismatch"):
        training_lineage.installed_distribution_python_callable_identity(
            distribution_name="callable-package",
            source_logical_path="callable_package/loss_utils.py",
            callable_name="selected_callable",
            resolved_environment=resolved,
        )


@pytest.mark.parametrize(
    "source",
    [
        "def another_callable():\n    return 1\n",
        (
            "def selected_callable():\n    return 1\n\n"
            "def outer():\n"
            "    def selected_callable():\n"
            "        return 2\n"
            "    return selected_callable()\n"
        ),
    ],
)
def test_installed_python_callable_identity_rejects_missing_or_ambiguous_code(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source: str,
) -> None:
    distribution = _InstalledDistribution(
        tmp_path,
        name="callable-package",
        version="1.2.3",
    )
    init_path = tmp_path / "callable_package" / "__init__.py"
    source_path = tmp_path / "callable_package" / "loss_utils.py"
    source_path.write_text(source, encoding="utf-8")
    _rewrite_distribution_record(distribution, [init_path, source_path])
    resolved = training_lineage.build_resolved_training_environment(
        [distribution]
    )
    monkeypatch.setattr(
        training_lineage.importlib_metadata,
        "distribution",
        lambda name: distribution,
    )

    with pytest.raises(ValueError, match="missing or ambiguous"):
        training_lineage.installed_distribution_python_callable_identity(
            distribution_name="callable-package",
            source_logical_path="callable_package/loss_utils.py",
            callable_name="selected_callable",
            resolved_environment=resolved,
        )


def test_resolved_environment_rejects_duplicate_or_secret_bearing_provenance(
    tmp_path: Path,
) -> None:
    first = _InstalledDistribution(tmp_path / "one", name="duplicate", version="1")
    second = _InstalledDistribution(tmp_path / "two", name="duplicate", version="2")
    with pytest.raises(ValueError, match="duplicate distributions"):
        training_lineage.build_resolved_training_environment([first, second])

    unsafe = _InstalledDistribution(
        tmp_path / "unsafe",
        name="unsafe",
        version="1",
        direct_url={"url": "https://token@example.com/private.git"},
    )
    with pytest.raises(ValueError, match="secret-safe"):
        training_lineage.build_resolved_training_environment([unsafe])


@pytest.mark.parametrize(
    "direct_url",
    [
        {"url": "https://example.com/latest.whl"},
        {
            "url": "https://github.com/example/project.git",
            "vcs_info": {"vcs": "git", "commit_id": "main"},
        },
        {
            "url": "https://example.com/project.whl",
            "archive_info": {},
        },
        {
            "url": "https://example.com/source",
            "dir_info": {"editable": False},
        },
        {
            "url": "https://github.com/example/project.git",
            "vcs_info": {"vcs": "git", "commit_id": "a" * 40},
            "subdirectory": "../outside",
        },
    ],
)
def test_resolved_environment_rejects_mutable_direct_url_provenance(
    tmp_path: Path,
    direct_url: dict[str, object],
) -> None:
    distribution = _InstalledDistribution(
        tmp_path,
        name="unsafe-provenance",
        version="1",
        direct_url=direct_url,
    )

    with pytest.raises(ValueError):
        training_lineage.build_resolved_training_environment([distribution])


@pytest.mark.parametrize(
    ("replacement", "error_match"),
    [
        (
            {
                "directURL": {
                    "url": "https://token@example.com/private.git",
                    "vcs_info": {"vcs": "git", "commit_id": "a" * 40},
                }
            },
            "secret-safe",
        ),
        (
            {
                "directURL": {
                    "url": "https://github.com/example/project.git",
                    "vcs_info": {
                        "vcs": "git",
                        "commit_id": "a" * 40,
                        "requested_revision": "main",
                    },
                }
            },
            "VCS provenance",
        ),
        (
            {
                "directURL": {
                    "url": "https://example.com/source",
                    "dir_info": {"editable": True},
                }
            },
            "directory provenance",
        ),
        (
            {
                "directURL": {
                    "url": "https://example.com/project.whl",
                    "archive_info": {},
                }
            },
            "archive lacks",
        ),
        (
            {
                "directURL": {
                    "url": "https://github.com/example/project.git",
                    "vcs_info": {"vcs": "git", "commit_id": "a" * 40},
                    "unexpected": "value",
                }
            },
            "unsupported direct-url",
        ),
        ({"installer": "uv; injected"}, "invalid installer"),
    ],
    ids=[
        "credentials",
        "mutable-vcs-revision",
        "editable-directory",
        "missing-archive-hash",
        "extra-direct-url-key",
        "invalid-installer",
    ],
)
def test_loaded_resolved_environment_rejects_resigned_unsafe_provenance(
    tmp_path: Path,
    replacement: dict[str, object],
    error_match: str,
) -> None:
    distribution = _InstalledDistribution(
        tmp_path,
        name="loaded-provenance",
        version="1",
        direct_url={
            "url": "https://github.com/example/project.git",
            "vcs_info": {"vcs": "git", "commit_id": "a" * 40},
        },
    )
    resolved = training_lineage.build_resolved_training_environment([distribution])
    entry = resolved["distributions"][0]
    assert isinstance(entry, dict)
    entry.update(replacement)
    _resign_resolved_environment(resolved)

    with pytest.raises(ValueError, match=error_match):
        training_lineage.verify_resolved_training_environment(resolved)


def test_resolved_environment_rejects_unhashed_behavior_and_parent_record_paths(
    tmp_path: Path,
) -> None:
    unhashed = _InstalledDistribution(
        tmp_path / "unhashed",
        name="unhashed",
        version="1",
    )
    behavior = unhashed.root / "unhashed" / "behavior.py"
    behavior.write_text("VALUE = 1\n", encoding="utf-8")
    unhashed.record_path.write_text(
        unhashed.record_path.read_text(encoding="utf-8")
        + "unhashed/behavior.py,,\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unattested RECORD file"):
        training_lineage.build_resolved_training_environment([unhashed])

    escaping = _InstalledDistribution(
        tmp_path / "escaping",
        name="escaping",
        version="1",
    )
    escaping.record_path.write_text("../outside.py,sha256=" + "a" * 43 + ",1\n")
    with pytest.raises(ValueError, match="unsafe RECORD"):
        training_lineage.build_resolved_training_environment([escaping])


def test_resolved_environment_accepts_hashed_venv_entrypoint_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment_root = tmp_path / "venv"
    site_packages = environment_root / "lib/python3.12/site-packages"
    distribution = _InstalledDistribution(
        site_packages,
        name="entrypoint-package",
        version="1",
    )
    entrypoint = environment_root / "bin/entrypoint-package"
    entrypoint.parent.mkdir(parents=True)
    entrypoint.write_text("#!/usr/bin/env python\n", encoding="utf-8")
    digest = base64.urlsafe_b64encode(
        hashlib.sha256(entrypoint.read_bytes()).digest()
    ).decode("ascii").rstrip("=")
    distribution.record_path.write_text(
        distribution.record_path.read_text(encoding="utf-8")
        + f"../../../bin/entrypoint-package,sha256={digest},{entrypoint.stat().st_size}\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(training_lineage.sys, "prefix", str(environment_root))

    resolved = training_lineage.build_resolved_training_environment([distribution])

    assert resolved["distributions"][0]["installedFileCount"] == 2


@pytest.mark.parametrize("attested_first", [False, True])
def test_resolved_environment_hashes_pip_duplicate_generated_bytecode(
    tmp_path: Path,
    attested_first: bool,
) -> None:
    distribution = _InstalledDistribution(
        tmp_path,
        name="numpy",
        version="2.2.6",
    )
    source_path = "numpy/generated.py"
    source = distribution.root / source_path
    source.write_text("VALUE = 1\n", encoding="utf-8")
    source_digest = base64.urlsafe_b64encode(
        hashlib.sha256(source.read_bytes()).digest()
    ).decode("ascii").rstrip("=")
    logical_path = "numpy/__pycache__/generated.cpython-310.pyc"
    installed_bytecode = distribution.root / logical_path
    installed_bytecode.parent.mkdir(parents=True, exist_ok=True)
    installed_bytecode.write_bytes(b"pip-generated-bytecode")
    wheel_bytecode = b"wheel-supplied-bytecode"
    wheel_digest = base64.urlsafe_b64encode(
        hashlib.sha256(wheel_bytecode).digest()
    ).decode("ascii").rstrip("=")
    attested_row = (
        f"{logical_path},sha256={wheel_digest},{len(wheel_bytecode)}\n"
    )
    generated_row = f"{logical_path},,\n"
    duplicate_rows = (
        attested_row + generated_row
        if attested_first
        else generated_row + attested_row
    )
    distribution.record_path.write_text(
        distribution.record_path.read_text(encoding="utf-8")
        + f"{source_path},sha256={source_digest},{source.stat().st_size}\n"
        + duplicate_rows,
        encoding="utf-8",
    )

    resolved = training_lineage.build_resolved_training_environment([distribution])
    original_digest = resolved["resolvedTrainingEnvironmentSHA256"]

    assert resolved["distributions"][0]["installedFileCount"] == 3

    installed_bytecode.write_bytes(b"different-generated-bytecode")
    mutated = training_lineage.build_resolved_training_environment([distribution])
    assert mutated["resolvedTrainingEnvironmentSHA256"] != original_digest


@pytest.mark.parametrize("duplicate_kind", ["unattested", "attested"])
def test_resolved_environment_rejects_other_duplicate_bytecode_rows(
    tmp_path: Path,
    duplicate_kind: str,
) -> None:
    distribution = _InstalledDistribution(
        tmp_path,
        name="duplicate-bytecode",
        version="1",
    )
    logical_path = (
        "duplicate_bytecode/__pycache__/generated.cpython-310.pyc"
    )
    bytecode = distribution.root / logical_path
    bytecode.parent.mkdir(parents=True, exist_ok=True)
    bytecode.write_bytes(b"bytecode")
    digest = base64.urlsafe_b64encode(
        hashlib.sha256(bytecode.read_bytes()).digest()
    ).decode("ascii").rstrip("=")
    if duplicate_kind == "attested":
        row = f"{logical_path},sha256={digest},{bytecode.stat().st_size}\n"
    else:
        row = f"{logical_path},,\n"
    distribution.record_path.write_text(
        distribution.record_path.read_text(encoding="utf-8") + row + row,
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unsafe RECORD"):
        training_lineage.build_resolved_training_environment([distribution])


def test_resolved_environment_rejects_canonical_bytecode_path_aliases(
    tmp_path: Path,
) -> None:
    distribution = _InstalledDistribution(
        tmp_path,
        name="aliased-bytecode",
        version="1",
    )
    package = distribution.root / "aliased_bytecode"
    source = package / "generated.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    source_digest = base64.urlsafe_b64encode(
        hashlib.sha256(source.read_bytes()).digest()
    ).decode("ascii").rstrip("=")
    bytecode = package / "__pycache__/generated.cpython-310.pyc"
    bytecode.parent.mkdir(parents=True)
    bytecode.write_bytes(b"bytecode")
    distribution.record_path.write_text(
        distribution.record_path.read_text(encoding="utf-8")
        + f"aliased_bytecode/generated.py,sha256={source_digest},{source.stat().st_size}\n"
        + "aliased_bytecode/__pycache__/generated.cpython-310.pyc,,\n"
        + "aliased_bytecode/__pycache__/./generated.cpython-310.pyc,,\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unsafe RECORD"):
        training_lineage.build_resolved_training_environment([distribution])


def test_resolved_environment_rejects_excluded_bytecode_outside_distribution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment_root = tmp_path / "venv"
    site_packages = environment_root / "lib/python3.10/site-packages"
    distribution = _InstalledDistribution(
        site_packages,
        name="escaping-bytecode",
        version="1",
    )
    escaping_path = "../../../outside/__pycache__/evil.cpython-310.pyc"
    distribution.record_path.write_text(
        distribution.record_path.read_text(encoding="utf-8")
        + f"{escaping_path},,\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(training_lineage.sys, "prefix", str(environment_root))

    with pytest.raises(ValueError, match="unsafe generated bytecode"):
        training_lineage.build_resolved_training_environment([distribution])


def test_resolved_environment_rejects_sourceless_excluded_bytecode(
    tmp_path: Path,
) -> None:
    distribution = _InstalledDistribution(
        tmp_path,
        name="sourceless-bytecode",
        version="1",
    )
    logical_path = (
        "sourceless_bytecode/__pycache__/hidden.cpython-310.pyc"
    )
    bytecode = distribution.root / logical_path
    bytecode.parent.mkdir(parents=True, exist_ok=True)
    bytecode.write_bytes(b"hidden-bytecode")
    distribution.record_path.write_text(
        distribution.record_path.read_text(encoding="utf-8")
        + f"{logical_path},,\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="without an attested source"):
        training_lineage.build_resolved_training_environment([distribution])


@pytest.mark.parametrize("optimization", ["opt-1", "opt-2"])
def test_resolved_environment_hashes_optimized_generated_bytecode(
    tmp_path: Path,
    optimization: str,
) -> None:
    distribution = _InstalledDistribution(
        tmp_path,
        name="optimized-bytecode",
        version="1",
    )
    source = distribution.root / "optimized_bytecode/generated.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    source_digest = base64.urlsafe_b64encode(
        hashlib.sha256(source.read_bytes()).digest()
    ).decode("ascii").rstrip("=")
    logical_path = (
        "optimized_bytecode/__pycache__/"
        f"generated.cpython-310.{optimization}.pyc"
    )
    bytecode = distribution.root / logical_path
    bytecode.parent.mkdir(parents=True, exist_ok=True)
    bytecode.write_bytes(b"optimized-bytecode")
    distribution.record_path.write_text(
        distribution.record_path.read_text(encoding="utf-8")
        + "optimized_bytecode/generated.py,"
        + f"sha256={source_digest},{source.stat().st_size}\n"
        + f"{logical_path},,\n",
        encoding="utf-8",
    )

    resolved = training_lineage.build_resolved_training_environment([distribution])
    assert resolved["distributions"][0]["installedFileCount"] == 3


def test_resolved_environment_accepts_only_contained_absolute_record_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment_root = tmp_path / "venv"
    site_packages = environment_root / "lib/python3.10/site-packages"
    distribution = _InstalledDistribution(
        site_packages,
        name="absolute-entrypoint",
        version="1",
    )
    entrypoint = environment_root / "bin/absolute-entrypoint"
    entrypoint.parent.mkdir(parents=True)
    entrypoint.write_text("#!/usr/bin/env python\n", encoding="utf-8")
    digest = base64.urlsafe_b64encode(
        hashlib.sha256(entrypoint.read_bytes()).digest()
    ).decode("ascii").rstrip("=")
    distribution.record_path.write_text(
        distribution.record_path.read_text(encoding="utf-8")
        + f"{entrypoint},sha256={digest},{entrypoint.stat().st_size}\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(training_lineage.sys, "prefix", str(environment_root))

    resolved = training_lineage.build_resolved_training_environment([distribution])
    assert resolved["distributions"][0]["installedFileCount"] == 2

    outside = tmp_path / "outside-entrypoint"
    outside.write_text("outside\n", encoding="utf-8")
    outside_digest = base64.urlsafe_b64encode(
        hashlib.sha256(outside.read_bytes()).digest()
    ).decode("ascii").rstrip("=")
    distribution.record_path.write_text(
        distribution.record_path.read_text(encoding="utf-8")
        + f"{outside},sha256={outside_digest},{outside.stat().st_size}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unsafe RECORD"):
        training_lineage.build_resolved_training_environment([distribution])


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
    launcher = (ROOT / "tools/fine_tuning/unsloth/ubuntu_pipeline.py").read_text(
        encoding="utf-8"
    )

    assert '"expectedRuntimeSourceRevision": revision' in launcher
    assert '"observedRepositoryRevision": revision' in launcher
    assert '"observedRuntimeRevision": revision' in launcher
    assert '"runtimeSourceBindingStatus": "verified_clean_snapshot"' in launcher
    assert (
        '"git_clean_worktree_plus_ubuntu_orchestration_manifest"'
        in launcher
    )
    assert "config.update(runtime_source)" in launcher
    assert "**runtime_source" in launcher


def test_attested_local_runtime_source_accepts_only_the_exact_binding_pair() -> None:
    revision = "e" * 40
    audit = {
        "runtimeSourceKind": "git",
        "runtimeSourceRevision": revision,
        "expectedRuntimeSourceRevision": revision,
        "observedRepositoryRevision": revision,
        "observedRuntimeRevision": revision,
        "runtimeSourceBindingStatus": "verified_clean_snapshot",
        "runtimeSourceBindingMethod": (
            "git_clean_worktree_plus_ubuntu_orchestration_manifest"
        ),
    }

    assert training_lineage.validate_runtime_source_audit(
        audit,
        observed_local_revision=revision,
    ) == audit
    audit["runtimeSourceBindingMethod"] = "git_head_plus_training_code_manifest"
    with pytest.raises(ValueError):
        training_lineage.validate_runtime_source_audit(
            audit,
            observed_local_revision=revision,
        )
