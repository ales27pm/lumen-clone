from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from lumen_manifest_crawler.dataset.optimization_policy import (
    expected_optimization_step_policy,
)
from tools.fine_tuning.unsloth import train_sft


class _DummyComponent:
    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        self.kwargs = dict(_kwargs)

    def __enter__(self) -> _DummyComponent:
        return self

    def __exit__(self, *_args: Any) -> None:
        return None

    def click(self, **_kwargs: Any) -> None:
        self.click_kwargs = dict(_kwargs)
        return None

    def queue(self) -> _DummyComponent:
        return self

    def launch(self) -> None:
        return None


def _load_app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    source = Path(__file__).resolve().parents[2] / "hf_zerogpu" / "space_template" / "app.py"
    app_root = tmp_path / "space"
    app_root.mkdir()
    shutil.copy2(source, app_root / "app.py")
    shutil.copy2(
        Path(__file__).resolve().parents[2]
        / "hf_zerogpu"
        / "space_template"
        / "README.md",
        app_root / "README.md",
    )
    unsloth_root = Path(__file__).resolve().parents[2] / "fine_tuning" / "unsloth"
    shutil.copy2(unsloth_root / "training_lineage.py", app_root / "training_lineage.py")
    shutil.copy2(unsloth_root / "adapter_artifact.py", app_root / "adapter_artifact.py")
    shutil.copy2(
        Path(__file__).resolve().parents[2] / "hf_zerogpu" / "space_template" / "requirements.txt",
        app_root / "requirements.txt",
    )
    spec = importlib.util.spec_from_file_location(
        "lumen_training_lineage_fixture",
        unsloth_root / "training_lineage.py",
    )
    assert spec is not None and spec.loader is not None
    lineage = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lineage)
    distribution_payload = {
        "name": "synthetic-runtime",
        "version": "1.0.0",
        "directURL": None,
        "installer": "test",
        "recordSHA256": "1" * 64,
        "installedFileCount": 1,
        "installedContentSHA256": "2" * 64,
    }
    distribution = {
        **distribution_payload,
        "distributionSHA256": lineage.canonical_sha256(distribution_payload),
    }
    resolved_payload = {
        "schemaVersion": "lumen.resolved-training-environment/1.0.0",
        "recordPolicy": {
            "hashAlgorithm": "sha256",
            "verifyDeclaredFileHashes": True,
            "excludeUnhashedSelfRecord": True,
            "hashUnattestedGeneratedBytecode": True,
            "hashRegeneratedBytecodePairs": True,
            "requireAttestedSourceForGeneratedBytecode": True,
            "rejectOtherUnhashedFiles": True,
        },
        "distributions": [distribution],
    }
    resolved_environment = {
        **resolved_payload,
        "resolvedTrainingEnvironmentSHA256": lineage.canonical_sha256(
            resolved_payload
        ),
    }
    resolved_scan = {
        "schemaVersion": "lumen.resolved-training-environment-cache/1.0.0",
        "resolvedTrainingEnvironmentSHA256": resolved_environment[
            "resolvedTrainingEnvironmentSHA256"
        ],
        "durationMilliseconds": 7,
        "distributionCount": 1,
        "installedFileCount": 1,
        "totalHashedBytes": 128,
    }
    monkeypatch.setattr(
        lineage,
        "build_resolved_training_environment_snapshot",
        lambda: (resolved_environment, resolved_scan),
    )
    monkeypatch.setitem(sys.modules, "training_lineage", lineage)
    space_configuration = lineage.build_space_configuration(app_root / "README.md")
    (app_root / "lumen_zero_gpu_defaults.json").write_text(
        json.dumps(
            {
                "dataset_repo": "user/datasets",
                "dataset_path_in_repo": "runs/test/fine_tuning",
                "adapter_repo": "user/adapters",
                "run_id": "test",
                "agents": ["executor"],
                "container_image_digest": "sha256:" + "c" * 64,
                "container_image_digest_source": "operator_declared",
                "runtime_image_binding_status": "manual_validation_required",
                "runtime_image_binding_verified": False,
                "dataset_revision": "a" * 40,
                "spaceConfiguration": space_configuration,
                "spaceConfigurationSHA256": space_configuration[
                    "spaceConfigurationSHA256"
                ],
            }
        ),
        encoding="utf-8",
    )

    gradio = ModuleType("gradio")
    for name in ("Blocks", "Row", "Textbox", "Number", "Dropdown", "Checkbox", "JSON", "Button"):
        setattr(gradio, name, _DummyComponent)
    gradio.Markdown = lambda *_args, **_kwargs: None
    spaces = ModuleType("spaces")
    spaces.GPU = lambda **_kwargs: (lambda function: function)
    hub = ModuleType("huggingface_hub")
    hub.HfApi = object
    hub.snapshot_download = lambda **_kwargs: ""
    monkeypatch.setitem(sys.modules, "gradio", gradio)
    monkeypatch.setitem(sys.modules, "spaces", spaces)
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    monkeypatch.syspath_prepend(str(app_root))

    spec = importlib.util.spec_from_file_location("lumen_zerogpu_test_app", app_root / "app.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    generation_payload = b'{"do_sample":false}\n'
    generation_digest = __import__("hashlib").sha256(
        generation_payload
    ).hexdigest()
    module.DEFAULT_BASE_MODEL_GENERATION_CONFIG_FILE = {
        "path": "generation_config.json",
        "sizeBytes": len(generation_payload),
        "sha256": generation_digest,
        "huggingFaceBlobID": generation_digest,
    }
    module.TEST_GENERATION_CONFIG_PAYLOAD = generation_payload
    module.TEST_RESOLVED_TRAINING_ENVIRONMENT = resolved_environment
    module.TEST_RESOLVED_TRAINING_ENVIRONMENT_SCAN = resolved_scan
    return module


def test_runtime_source_repository_head_is_supplemental_and_unverified(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_app(tmp_path, monkeypatch)

    class FakeApi:
        def __init__(self, *, token: str | None) -> None:
            assert token == "fine-grained-token"

        def space_info(self, **_kwargs: Any) -> SimpleNamespace:
            return SimpleNamespace(sha="4" * 40)

    monkeypatch.setenv("LUMEN_ZERO_GPU_HUB_TOKEN", "fine-grained-token")
    monkeypatch.setattr(module, "HfApi", FakeApi)
    module.DEFAULTS["space_repo"] = "user/space"

    lineage = module._resolve_runtime_source_binding(
        kind="huggingface_space",
        expected_revision="4" * 40,
    )

    assert lineage == {
        "runtimeSourceKind": "huggingface_space",
        "runtimeSourceRevision": "4" * 40,
        "expectedRuntimeSourceRevision": "4" * 40,
        "observedRepositoryRevision": "4" * 40,
        "observedRuntimeRevision": None,
        "runtimeSourceBindingStatus": "operator_declared_unverified",
        "runtimeSourceBindingMethod": "huggingface_repository_head_supplemental",
    }


def test_runtime_source_rejects_malformed_or_mismatched_expected_revision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_app(tmp_path, monkeypatch)
    instantiated = False

    class FakeApi:
        def __init__(self, *, token: str | None) -> None:
            nonlocal instantiated
            instantiated = True

        def space_info(self, **_kwargs: Any) -> SimpleNamespace:
            return SimpleNamespace(sha="5" * 40)

    monkeypatch.setattr(module, "HfApi", FakeApi)
    module.DEFAULTS["space_repo"] = "user/space"

    with pytest.raises(ValueError, match="full lowercase commit SHA"):
        module._resolve_runtime_source_binding(
            kind="huggingface_space",
            expected_revision="main",
        )
    assert instantiated is False

    with pytest.raises(ValueError, match="does not match the expected revision"):
        module._resolve_runtime_source_binding(
            kind="huggingface_space",
            expected_revision="4" * 40,
        )


def test_absent_runtime_source_observation_never_becomes_verified(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_app(tmp_path, monkeypatch)

    class UnavailableApi:
        def __init__(self, *, token: str | None) -> None:
            pass

        def space_info(self, **_kwargs: Any) -> SimpleNamespace:
            raise RuntimeError("offline")

    monkeypatch.setattr(module, "HfApi", UnavailableApi)
    module.DEFAULTS["space_repo"] = "user/space"
    lineage = module._resolve_runtime_source_binding(
        kind="huggingface_space",
        expected_revision="4" * 40,
    )
    assert lineage["observedRepositoryRevision"] is None
    assert lineage["observedRuntimeRevision"] is None
    assert lineage["runtimeSourceBindingStatus"] == "operator_declared_unverified"
    assert lineage["runtimeSourceBindingMethod"] == "operator_declared_only"


def test_runtime_preflight_rejects_space_front_matter_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_app(tmp_path, monkeypatch)
    code_digest = "1" * 64
    dependency_digest = "2" * 64
    requirements_digest = "3" * 64
    module.DEFAULTS.update(
        {
            "trainingCodeManifest": {"phase": "sft"},
            "trainingCodeSHA256": code_digest,
            "trainingDependencyLock": {
                "requirementsSHA256": requirements_digest,
            },
            "trainingDependencyLockSHA256": dependency_digest,
            "requirementsSHA256": requirements_digest,
        }
    )
    monkeypatch.setattr(
        module,
        "verify_training_code_manifest",
        lambda *_args, **_kwargs: code_digest,
    )
    monkeypatch.setattr(
        module,
        "installed_controlled_package_versions",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        module,
        "verify_training_dependency_lock",
        lambda *_args, **_kwargs: dependency_digest,
    )
    monkeypatch.setattr(module, "_installed_unsloth_revision", lambda: "a" * 40)
    monkeypatch.setattr(
        module,
        "_resolve_runtime_source_binding",
        lambda **_kwargs: {
            "runtimeSourceKind": "huggingface_space",
            "runtimeSourceRevision": "4" * 40,
            "expectedRuntimeSourceRevision": "4" * 40,
            "observedRepositoryRevision": None,
            "observedRuntimeRevision": None,
            "runtimeSourceBindingStatus": "operator_declared_unverified",
            "runtimeSourceBindingMethod": "operator_declared_only",
        },
    )
    torch = ModuleType("torch")
    torch.version = SimpleNamespace(cuda="12.8")
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setenv("LUMEN_ZERO_GPU_EXPECTED_RUNTIME_SOURCE_REVISION", "4" * 40)

    assert module._verify_runtime_lineage()["spaceConfigurationSHA256"] == (
        module.DEFAULTS["spaceConfigurationSHA256"]
    )

    readme = module.APP_ROOT / "README.md"
    readme.write_text(
        readme.read_text(encoding="utf-8").replace(
            "app_file: app.py",
            "app_file: alternate.py",
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="runtime configuration drifted"):
        module._verify_runtime_lineage()


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def test_resolve_run_workspace_rejects_unsafe_identifiers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_app(tmp_path, monkeypatch)
    work_root = tmp_path / "work"
    monkeypatch.setenv("LUMEN_ZERO_GPU_WORKDIR", str(work_root))

    for run_id in ("../../escape", "/tmp/escape", "nested/run", ".hidden"):
        with pytest.raises(ValueError, match="run_id contains unsupported characters"):
            module._resolve_run_workspace(run_id, "internal_only")

    qualified_run_id, run_root = module._resolve_run_workspace(
        "audit-2026.07.13",
        "internal_only",
    )
    assert qualified_run_id == "audit-2026.07.13-internal_only"
    assert run_root == work_root.resolve() / qualified_run_id


def test_resolve_run_workspace_rejects_symlink_escape(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_app(tmp_path, monkeypatch)
    work_root = tmp_path / "work"
    outside = tmp_path / "outside"
    work_root.mkdir()
    outside.mkdir()
    (work_root / "linked-internal_only").symlink_to(outside, target_is_directory=True)
    monkeypatch.setenv("LUMEN_ZERO_GPU_WORKDIR", str(work_root))

    with pytest.raises(ValueError, match="run_id escapes the ZeroGPU work directory"):
        module._resolve_run_workspace("linked", "internal_only")


def test_training_endpoint_rejects_absolute_run_id_before_deletion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_app(tmp_path, monkeypatch)
    work_root = tmp_path / "work"
    outside = tmp_path / "outside-internal_only"
    outside.mkdir()
    sentinel = outside / "keep.txt"
    sentinel.write_text("keep", encoding="utf-8")
    monkeypatch.setenv("LUMEN_ZERO_GPU_WORKDIR", str(work_root))
    with pytest.raises(ValueError, match="run_id contains unsupported characters"):
        module._resolve_run_workspace(str(outside), "internal_only")
    admin_token = "Lumen-Admin-Token-0123456789-ABCDEF"
    monkeypatch.setenv("LUMEN_ZERO_GPU_ADMIN_TOKEN", admin_token)
    monkeypatch.setenv(
        "LUMEN_ZERO_GPU_HUB_TOKEN",
        "hf_fine_grained_repository_token",
    )
    result = module.train_lumen_adapters(
        str(outside),
        "executor",
        "",
        42,
        True,
        False,
        False,
        False,
        "large",
        "internal_only",
        True,
        False,
        request=SimpleNamespace(headers={"x-lumen-admin-token": admin_token}),
    )
    assert result["error_code"] == "training_failed"
    assert str(outside) not in json.dumps(result)
    assert sentinel.read_text(encoding="utf-8") == "keep"


def _write_variant_fixture(module: Any, root: Path) -> tuple[Path, dict[str, Any]]:
    agent_root = root / "executor"
    variant_root = agent_root / "experiments" / "internal_plus_public_optimized"
    variant_root.mkdir(parents=True, exist_ok=True)
    lanes = {
        "train_sft": [
            {"messages": [{"role": "assistant", "content": f"ok-{index}"}]}
            for index in range(96)
        ],
        "val_sft": [{"messages": [{"role": "assistant", "content": "valid"}]}],
        "train_dpo": [
            {"prompt": f"x-{index}", "chosen": "y", "rejected": "z"}
            for index in range(16)
        ],
        "val_dpo": [],
    }
    for lane, records in lanes.items():
        _write_jsonl(variant_root / f"{lane}.jsonl", records)
    shard_payloads = {
        "model-00001-of-00002.safetensors": b"first-shard",
        "model-00002-of-00002.safetensors": b"second-shard",
    }
    weight_shards = [
        {
            "filename": "model-00001-of-00002.safetensors",
            "size": len(shard_payloads["model-00001-of-00002.safetensors"]),
            "sha256": __import__("hashlib").sha256(
                shard_payloads["model-00001-of-00002.safetensors"]
            ).hexdigest(),
        },
        {
            "filename": "model-00002-of-00002.safetensors",
            "size": len(shard_payloads["model-00002-of-00002.safetensors"]),
            "sha256": __import__("hashlib").sha256(
                shard_payloads["model-00002-of-00002.safetensors"]
            ).hexdigest(),
        },
    ]
    index_payload = json.dumps(
        {
            "weight_map": {
                "a": "model-00001-of-00002.safetensors",
                "b": "model-00002-of-00002.safetensors",
            }
        },
        sort_keys=True,
    ).encode("utf-8")
    tokenizer_payloads = {
        "config.json": b'{"model_type":"qwen3"}\n',
        "merges.txt": b"a b\n",
        "tokenizer.json": b'{"version":"1.0"}\n',
        "tokenizer_config.json": b'{"chat_template":"test"}\n',
        "vocab.json": b'{"a":0}\n',
    }
    tokenizer_files = [
        {
            "path": filename,
            "sizeBytes": len(payload),
            "sha256": __import__("hashlib").sha256(payload).hexdigest(),
            "huggingFaceBlobID": __import__("hashlib").sha256(payload).hexdigest(),
        }
        for filename, payload in tokenizer_payloads.items()
    ]
    tokenizer_files.sort(key=lambda item: item["path"])
    tokenizer_digest = next(
        item["sha256"]
        for item in tokenizer_files
        if item["path"] == "tokenizer.json"
    )
    tokenizer_closure_sha256 = module._canonical_sha256(
        {
            "schemaVersion": "lumen.base-model-tokenizer-closure/1.0.0",
            "baseModelID": "Qwen/Qwen3-1.7B",
            "baseModelRevision": "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
            "files": tokenizer_files,
        }
    )
    base_policy = expected_optimization_step_policy(
        agent="executor",
        sft_train_record_count=128,
        dpo_train_record_count=32,
    )
    config = {
        "agent": "executor",
        "baseModelID": "Qwen/Qwen3-1.7B",
        "base_model_name": "Qwen/Qwen3-1.7B",
        "baseModelRevision": "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
        "baseModelIndexDigest": __import__("hashlib").sha256(
            index_payload
        ).hexdigest(),
        "baseModelIndexReferencedShardNames": [
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
        ],
        "baseModelArtifactDigest": module._canonical_sha256(
            {
                "schemaVersion": "lumen.base-model-weight-shards/1.0.0",
                "shards": weight_shards,
            }
        ),
        "baseModelWeightShards": weight_shards,
        "baseModelTokenizerDigest": tokenizer_digest,
        "baseModelTokenizerFiles": tokenizer_files,
        "baseModelTokenizerClosureSHA256": tokenizer_closure_sha256,
        "max_seq_length": 128,
        "load_in_4bit": True,
        "lora_r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0,
        "learning_rate": 0.0002,
        "seed": 42,
        "batch_size": 2,
        "gradient_accumulation_steps": 8,
        "num_train_epochs": base_policy["sft"]["selectedEpochs"],
        "dpo_num_train_epochs": base_policy["dpo"]["selectedEpochs"],
        "optimizationStepPolicy": base_policy,
        "warmup_steps": 0,
        "merge_adapters_by_default": False,
        "release_bake_enabled_by_default": False,
        "trainingCodeManifest": {"phase": "sft"},
        "trainingCodeSHA256": "1" * 64,
        "trainingDependencyLock": {"schema": "lock"},
        "trainingDependencyLockSHA256": "2" * 64,
        "requirementsSHA256": "3" * 64,
    }
    config["baseModelIndexShardBindingSHA256"] = module._canonical_sha256(
        {
            "schemaVersion": "lumen.base-model-index-shard-binding/1.0.0",
            "indexDigest": config["baseModelIndexDigest"],
            "referencedShardNames": config["baseModelIndexReferencedShardNames"],
            "shardContractDigest": config["baseModelArtifactDigest"],
        }
    )
    environment_lock = {
        "schemaVersion": "lumen.adapter-training-environment-lock/1.1.0",
        "pythonVersion": "3.10",
        "cudaVersion": "12.8",
        "packageVersions": {"torch": "2.9.1"},
        "unslothRevision": "935474c20aabc2aadb1da17338959c7c6f9bdafe",
        "llamaCppRevision": "34558825a27f4d74dcfd7a91bfde4464baa2a30a",
        "baseTokenizerSHA256": tokenizer_digest,
        "baseTokenizerClosureSHA256": tokenizer_closure_sha256,
        "containerImageDigestPolicy": "operator_declared_manual_runtime_verification",
    }
    config["trainingEnvironmentLock"] = environment_lock
    (agent_root / "unsloth_config.json").write_text(json.dumps(config), encoding="utf-8")
    variant_policy = expected_optimization_step_policy(
        agent="executor",
        sft_train_record_count=len(lanes["train_sft"]),
        dpo_train_record_count=len(lanes["train_dpo"]),
    )
    controlled_config = dict(config)
    controlled_config["optimizationStepPolicy"] = variant_policy
    controlled_config["num_train_epochs"] = variant_policy["sft"][
        "selectedEpochs"
    ]
    controlled_config["dpo_num_train_epochs"] = variant_policy["dpo"][
        "selectedEpochs"
    ]
    manifest = {
        "schemaVersion": module.EXPERIMENT_VARIANT_SCHEMA_VERSION,
        "agent": "executor",
        "variant": "internal_plus_public_optimized",
        "baseModelID": "Qwen/Qwen3-1.7B",
        "baseModelRevision": config["baseModelRevision"],
        "baseModelIndexDigest": config["baseModelIndexDigest"],
        "baseModelIndexReferencedShardNames": config["baseModelIndexReferencedShardNames"],
        "baseModelIndexShardBindingSHA256": config["baseModelIndexShardBindingSHA256"],
        "baseModelArtifactDigest": config["baseModelArtifactDigest"],
        "baseModelWeightShards": config["baseModelWeightShards"],
        "baseModelTokenizerDigest": config["baseModelTokenizerDigest"],
        "baseModelTokenizerFiles": config["baseModelTokenizerFiles"],
        "baseModelTokenizerClosureSHA256": config[
            "baseModelTokenizerClosureSHA256"
        ],
        "trainingEnvironmentLock": environment_lock,
        "trainingEnvironmentLockSHA256": module._canonical_sha256(environment_lock),
        "trainingEnvironmentSHA256": None,
        "seed": 42,
        "controlledTrainingConfig": controlled_config,
        "trainingConfigSHA256": module._canonical_sha256(controlled_config),
        "trainingConfigInvariantSHA256": module._canonical_sha256(
            module._normalized_invariant_training_config(
                controlled_config,
                agent="executor",
                sft_train_record_count=len(lanes["train_sft"]),
                dpo_train_record_count=len(lanes["train_dpo"]),
            )
        ),
        "trainingCorpusSHA256": module._canonical_sha256(
            [*lanes["train_sft"], *lanes["val_sft"], *lanes["train_dpo"], *lanes["val_dpo"]]
        ),
        "datasets": {
            "trainSFT": {"count": len(lanes["train_sft"]), "sha256": module._canonical_sha256(lanes["train_sft"])},
            "validationSFT": {"count": 1, "sha256": module._canonical_sha256(lanes["val_sft"])},
            "trainDPO": {"count": len(lanes["train_dpo"]), "sha256": module._canonical_sha256(lanes["train_dpo"])},
            "validationDPO": {"count": 0, "sha256": module._canonical_sha256(lanes["val_dpo"])},
        },
    }
    manifest["variantManifestSHA256"] = module._canonical_sha256(manifest)
    (variant_root / "variant_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (root / "adapter_runtime_manifest.json").write_text(
        json.dumps(
            {
                "sharedBaseModelID": "Qwen/Qwen3-1.7B",
                "adapters": [{"agent": "executor", "baseModelID": "Qwen/Qwen3-1.7B"}],
            }
        ),
        encoding="utf-8",
    )
    run_root = (
        root.parent.parent
        if root.name == "fine_tuning" and root.parent.name == "generated"
        else root.parent / "run"
    )
    private_snapshot = run_root / module.PRIVATE_TOKENIZER_SNAPSHOT_DIRNAME
    private_snapshot.mkdir(mode=0o700, parents=True, exist_ok=True)
    private_snapshot.chmod(0o700)
    for filename, payload in tokenizer_payloads.items():
        target = private_snapshot / filename
        if target.exists():
            target.chmod(0o600)
        target.write_bytes(payload)
        target.chmod(0o400)
    runtime_snapshot = (
        run_root / module.PRIVATE_BASE_MODEL_RUNTIME_SNAPSHOT_DIRNAME
    )
    runtime_snapshot.mkdir(mode=0o700, parents=True, exist_ok=True)
    runtime_snapshot.chmod(0o700)
    for filename, payload in {
        **tokenizer_payloads,
        "generation_config.json": module.TEST_GENERATION_CONFIG_PAYLOAD,
        "model.safetensors.index.json": index_payload,
        **shard_payloads,
    }.items():
        target = runtime_snapshot / filename
        if target.exists():
            target.chmod(0o600)
        target.write_bytes(payload)
        target.chmod(0o400)
    return variant_root, manifest


def test_prepare_configs_selects_and_attests_optimized_variant(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_app(tmp_path, monkeypatch)
    source_root = tmp_path / "datasets"
    source_root.mkdir()
    variant_root, manifest = _write_variant_fixture(module, source_root)

    prepared = module._prepare_configs(
        source_root=source_root,
        run_root=tmp_path / "run",
        agents=["executor"],
        base_model_override="",
        seed=42,
        variant="internal_plus_public_optimized",
    )

    config = json.loads(Path(prepared[0]["config"]).read_text(encoding="utf-8"))
    assert config["dataset_dir"] == str(variant_root)
    assert config["variantManifestSHA256"] == manifest["variantManifestSHA256"]
    assert config["variantAttestation"]["schema"] == (
        module.TRAINING_VARIANT_ATTESTATION_SCHEMA
    )
    assert config["variantAttestation"]["trainingCorpusSHA256"] == manifest["trainingCorpusSHA256"]
    assert config["num_train_epochs"] == manifest["controlledTrainingConfig"][
        "num_train_epochs"
    ]
    assert config["optimizationStepPolicy"] == manifest[
        "controlledTrainingConfig"
    ]["optimizationStepPolicy"]
    assert config["variantAttestation"]["effectiveTrainingConfigSHA256"] == manifest[
        "trainingConfigSHA256"
    ]
    assert config["variantAttestation"]["trainingConfigInvariantSHA256"] == manifest[
        "trainingConfigInvariantSHA256"
    ]
    assert config["trainingContainerImageDigest"] == "sha256:" + "c" * 64
    assert config["trainingContainerImageDigestSource"] == "operator_declared"
    assert config["trainingRuntimeImageBindingStatus"] == "manual_validation_required"
    assert config["trainingRuntimeImageBindingVerified"] is False
    assert config["trainingEnvironmentSHA256"] == module._canonical_sha256(
        {
            "schemaVersion": "lumen.adapter-training-environment/1.0.0",
            "containerImageDigest": "sha256:" + "c" * 64,
            "containerImageDigestSource": "operator_declared",
            "runtimeImageBindingStatus": "manual_validation_required",
                "runtimeImageBindingVerified": False,
                "effectiveSeed": 42,
                "environmentLock": manifest["trainingEnvironmentLock"],
                "zeroGPUSize": None,
                "zeroGPUDurationSeconds": None,
                "observedAccelerator": None,
            }
        )
    assert config["variantAttestation"]["baseModelRevision"] == manifest["baseModelRevision"]
    assert config["variantAttestation"]["trainingEnvironmentSHA256"] == config["trainingEnvironmentSHA256"]
    assert config["variantAttestation"]["runtimeImageBindingStatus"] == "manual_validation_required"
    assert config["variantAttestation"]["runtimeImageBindingVerified"] is False


def test_prepare_configs_replaces_unresolved_runtime_audit_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_app(tmp_path, monkeypatch)
    run_root = tmp_path / "run"
    source_root = run_root / "generated/fine_tuning"
    source_root.mkdir(parents=True)
    _write_variant_fixture(module, source_root)
    config_path = source_root / "executor" / "unsloth_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["runtimeSourceKind"] = "unresolved"
    config["runtimeSourceRevision"] = None
    config_path.write_text(json.dumps(config), encoding="utf-8")
    runtime = {
        "trainingCodeManifest": {"phase": "sft"},
        "trainingCodeSHA256": "1" * 64,
        "trainingDependencyLock": {"schema": "lock"},
        "trainingDependencyLockSHA256": "2" * 64,
        "requirementsSHA256": "3" * 64,
        "resolvedTrainingEnvironment": module.TEST_RESOLVED_TRAINING_ENVIRONMENT,
            "resolvedTrainingEnvironmentSHA256": module.TEST_RESOLVED_TRAINING_ENVIRONMENT[
                "resolvedTrainingEnvironmentSHA256"
            ],
            "resolvedTrainingEnvironmentCacheAttestation": module._STARTUP_ENVIRONMENT_ATTESTATION,
            "resolvedTrainingEnvironmentScanAudit": module.TEST_RESOLVED_TRAINING_ENVIRONMENT_SCAN,
            "zeroGPUSize": "large",
            "zeroGPUDurationSeconds": 1200,
            "observedAccelerator": _test_accelerator(),
            "spaceConfigurationSHA256": module.DEFAULTS[
                "spaceConfigurationSHA256"
            ],
        "runtimeSourceKind": "huggingface_space",
        "runtimeSourceRevision": "4" * 40,
        "expectedRuntimeSourceRevision": "4" * 40,
        "observedRepositoryRevision": "4" * 40,
        "observedRuntimeRevision": None,
        "runtimeSourceBindingStatus": "operator_declared_unverified",
        "runtimeSourceBindingMethod": "huggingface_repository_head_supplemental",
    }
    lineage = module._build_run_resume_lineage(
        run_id="run",
        run_root=run_root,
        source_root=source_root,
        dataset_repo="user/dataset",
        dataset_revision="5" * 40,
        dataset_path="runs/test/fine_tuning",
        agents=["executor"],
        variant="internal_plus_public_optimized",
        seed=42,
        assistant_only_loss=True,
        runtime_lineage=runtime,
    )
    prepared = module._prepare_configs(
        source_root=source_root,
        run_root=run_root,
        agents=["executor"],
        base_model_override="",
        seed=42,
        variant="internal_plus_public_optimized",
        run_lineage=lineage,
        runtime_lineage=runtime,
    )

    resolved = json.loads(Path(prepared[0]["config"]).read_text(encoding="utf-8"))
    verified_lineage, verified_agent = train_sft._agent_resume_lineage(
        resolved
    )
    assert verified_lineage == lineage
    assert verified_agent["trainingConfigInvariantSHA256"] == resolved[
        "variantAttestation"
    ]["trainingConfigInvariantSHA256"]
    train_sft._validate_run_resume_config(
        resolved,
        cfg_path=Path(prepared[0]["config"]).resolve(),
        assistant_only_loss=True,
    )
    assert resolved["runtimeSourceKind"] == "huggingface_space"
    assert resolved["runtimeSourceRevision"] == "4" * 40
    assert resolved["expectedRuntimeSourceRevision"] == "4" * 40
    assert resolved["observedRepositoryRevision"] == "4" * 40
    assert resolved["observedRuntimeRevision"] is None
    assert resolved["runtimeSourceBindingStatus"] == "operator_declared_unverified"
    assert (
        resolved["runtimeSourceBindingMethod"]
        == "huggingface_repository_head_supplemental"
    )
    assert resolved["trainingCodeSHA256"] == "1" * 64
    assert resolved["spaceConfigurationSHA256"] == module.DEFAULTS[
        "spaceConfigurationSHA256"
    ]


def test_trained_adapter_rejects_tampered_or_substituted_finalized_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_app(tmp_path, monkeypatch)
    source_root = tmp_path / "datasets"
    source_root.mkdir()
    _write_variant_fixture(module, source_root)
    item = module._prepare_configs(
        source_root=source_root,
        run_root=tmp_path / "run",
        agents=["executor"],
        base_model_override="",
        seed=42,
        variant="internal_plus_public_optimized",
    )[0]
    attestation = item["variantAttestation"]
    config = json.loads(Path(item["config"]).read_text(encoding="utf-8"))
    training_environment = {
        "schemaVersion": "lumen.adapter-training-environment/1.0.0",
        "containerImageDigest": config["trainingContainerImageDigest"],
        "containerImageDigestSource": config["trainingContainerImageDigestSource"],
        "runtimeImageBindingStatus": config["trainingRuntimeImageBindingStatus"],
        "runtimeImageBindingVerified": config["trainingRuntimeImageBindingVerified"],
        "effectiveSeed": config["seed"],
        "environmentLock": config["trainingEnvironmentLock"],
        "zeroGPUSize": item.get("zeroGPUSize"),
        "zeroGPUDurationSeconds": item.get("zeroGPUDurationSeconds"),
        "observedAccelerator": item.get("observedAccelerator"),
    }
    finalized = {
        "agent": item["agent"],
        "variant": item["variant"],
        "sourceVariantManifestSHA256": item["variantManifestSHA256"],
        "baseModelID": item["base_model_name"],
        "baseModelRevision": item["baseModelRevision"],
        "baseModelIndexDigest": item["baseModelIndexDigest"],
        "baseModelIndexReferencedShardNames": item["baseModelIndexReferencedShardNames"],
        "baseModelIndexShardBindingSHA256": item["baseModelIndexShardBindingSHA256"],
        "baseModelArtifactDigest": item["baseModelArtifactDigest"],
        "baseModelWeightShards": item["baseModelWeightShards"],
        "baseModelTokenizerDigest": item["baseModelTokenizerDigest"],
        "baseModelTokenizerFiles": item["baseModelTokenizerFiles"],
        "baseModelTokenizerClosureSHA256": item[
            "baseModelTokenizerClosureSHA256"
        ],
        "trainingEnvironmentSHA256": item["trainingEnvironmentSHA256"],
        "trainingEnvironment": training_environment,
        "zeroGPUSize": item.get("zeroGPUSize"),
        "zeroGPUDurationSeconds": item.get("zeroGPUDurationSeconds"),
        "observedAccelerator": item.get("observedAccelerator"),
        "trainingCorpusSHA256": attestation["trainingCorpusSHA256"],
        "trainingConfigSHA256": attestation["effectiveTrainingConfigSHA256"],
        "trainingConfigInvariantSHA256": attestation[
            "trainingConfigInvariantSHA256"
        ],
        "datasets": {
            name: {"sha256": digest}
            for name, digest in attestation["laneHashes"].items()
        },
        "artifact": {
            "status": "trained",
            "trainingPhase": "sft",
            "parentSFTAdapterSHA256": None,
            "adapterSHA256": "a" * 64,
            "adapterManifestSHA256": "a" * 64,
        },
    }
    finalized["variantManifestSHA256"] = module._canonical_sha256(finalized)
    manifest_path = Path(item["finalized_variant_manifest"])

    module._verify_finalized_variant_lineage(item, finalized, manifest_path)

    tampered = dict(finalized)
    tampered["agent"] = "cortex"
    with pytest.raises(ValueError, match="integrity check failed"):
        module._verify_finalized_variant_lineage(item, tampered, manifest_path)

    substituted = dict(tampered)
    substituted.pop("variantManifestSHA256")
    substituted["variantManifestSHA256"] = module._canonical_sha256(substituted)
    with pytest.raises(ValueError, match="identity or source lineage mismatch"):
        module._verify_finalized_variant_lineage(item, substituted, manifest_path)

    missing_digest = json.loads(json.dumps(finalized))
    missing_digest["artifact"].pop("adapterSHA256")
    missing_digest.pop("variantManifestSHA256")
    missing_digest["variantManifestSHA256"] = module._canonical_sha256(
        missing_digest
    )
    with pytest.raises(ValueError, match="valid SFT adapter lineage"):
        module._verify_finalized_variant_lineage(
            item, missing_digest, manifest_path
        )

    attestation_drift = json.loads(json.dumps(finalized))
    attestation_drift["datasets"]["trainSFT"]["sha256"] = "f" * 64
    attestation_drift.pop("variantManifestSHA256")
    attestation_drift["variantManifestSHA256"] = module._canonical_sha256(
        attestation_drift
    )
    with pytest.raises(ValueError, match="prepared attestation"):
        module._verify_finalized_variant_lineage(
            item, attestation_drift, manifest_path
        )


@pytest.mark.parametrize("private", [True, False])
def test_adapter_upload_requires_matching_repository_visibility(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    private: bool,
) -> None:
    module = _load_app(tmp_path, monkeypatch)
    calls: list[str] = []

    class FakeApi:
        def __init__(self, *, token: str) -> None:
            assert token == "fine-grained-token"

        def model_info(self, **_kwargs: Any) -> SimpleNamespace:
            calls.append("model_info")
            return SimpleNamespace(private=private)

    monkeypatch.setattr(module, "HfApi", FakeApi)
    monkeypatch.setenv(
        "LUMEN_ZERO_GPU_PRIVATE_ADAPTERS",
        "1" if private else "0",
    )

    assert module._upload_outputs(
        tmp_path,
        [],
        "user/adapters",
        "test-run",
        "fine-grained-token",
        False,
    ) == {}
    assert calls == ["model_info"]


def test_adapter_upload_rejects_visibility_drift_before_upload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_app(tmp_path, monkeypatch)
    calls: list[str] = []

    class FakeApi:
        def __init__(self, *, token: str) -> None:
            assert token == "fine-grained-token"

        def model_info(self, **_kwargs: Any) -> SimpleNamespace:
            calls.append("model_info")
            return SimpleNamespace(private=False)

        def upload_folder(self, **_kwargs: Any) -> None:
            calls.append("upload_folder")

        def upload_file(self, **_kwargs: Any) -> None:
            calls.append("upload_file")

    monkeypatch.setattr(module, "HfApi", FakeApi)
    monkeypatch.setenv("LUMEN_ZERO_GPU_PRIVATE_ADAPTERS", "1")

    with pytest.raises(RuntimeError, match="visibility postcondition failed"):
        module._upload_outputs(
            tmp_path,
            [{"agent": "executor"}],
            "user/adapters",
            "test-run",
            "fine-grained-token",
            False,
        )
    assert calls == ["model_info"]


def test_variant_dataset_rejects_tampered_lane_and_control_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_app(tmp_path, monkeypatch)
    source_root = tmp_path / "datasets"
    source_root.mkdir()
    variant_root, _ = _write_variant_fixture(module, source_root)
    (variant_root / "train_sft.jsonl").write_text(
        json.dumps({"messages": [{"role": "assistant", "content": "tampered"}]}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="dataset (?:count|hash) mismatch"):
        module._variant_dataset(
            source_root / "executor",
            agent="executor",
            variant="internal_plus_public_optimized",
        )

    _write_variant_fixture(module, source_root)
    config_path = source_root / "executor" / "unsloth_config.json"
    drifted_config = json.loads(config_path.read_text(encoding="utf-8"))
    drifted_config["max_train_records"] = 1
    config_path.write_text(json.dumps(drifted_config), encoding="utf-8")
    with pytest.raises(ValueError, match="optimization-step policy is invalid"):
        module._prepare_configs(
            source_root=source_root,
            run_root=tmp_path / "run",
            agents=["executor"],
            base_model_override="",
            seed=42,
            variant="internal_plus_public_optimized",
        )
    _write_variant_fixture(module, source_root)
    with pytest.raises(ValueError, match="Seed override"):
        module._prepare_configs(
            source_root=source_root,
            run_root=tmp_path / "run",
            agents=["executor"],
            base_model_override="",
            seed=7,
            variant="internal_plus_public_optimized",
        )


def test_variant_config_rejects_rehashed_policy_and_nonvariant_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_app(tmp_path, monkeypatch)
    source_root = tmp_path / "datasets"
    source_root.mkdir()
    variant_root, _ = _write_variant_fixture(module, source_root)
    manifest_path = variant_root / "variant_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop("variantManifestSHA256")
    controlled = manifest["controlledTrainingConfig"]
    controlled["optimizationStepPolicy"]["sft"][
        "minimumEffectiveSteps"
    ] += 1
    manifest["trainingConfigSHA256"] = module._canonical_sha256(controlled)
    manifest["variantManifestSHA256"] = module._canonical_sha256(manifest)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="optimization policy is invalid"):
        module._variant_dataset(
            source_root / "executor",
            agent="executor",
            variant="internal_plus_public_optimized",
        )

    variant_root, _ = _write_variant_fixture(module, source_root)
    manifest_path = variant_root / "variant_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop("variantManifestSHA256")
    controlled = manifest["controlledTrainingConfig"]
    controlled["learning_rate"] = 0.9
    manifest["trainingConfigSHA256"] = module._canonical_sha256(controlled)
    manifest["trainingConfigInvariantSHA256"] = module._canonical_sha256(
        module._normalized_invariant_training_config(
            controlled,
            agent="executor",
            sft_train_record_count=manifest["datasets"]["trainSFT"]["count"],
            dpo_train_record_count=manifest["datasets"]["trainDPO"]["count"],
        )
    )
    manifest["variantManifestSHA256"] = module._canonical_sha256(manifest)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="optimization-step policy is invalid"):
        module._prepare_configs(
            source_root=source_root,
            run_root=tmp_path / "run",
            agents=["executor"],
            base_model_override="",
            seed=42,
            variant="internal_plus_public_optimized",
        )


def test_variant_dataset_rejects_rehashed_wrong_schema(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_app(tmp_path, monkeypatch)
    source_root = tmp_path / "datasets"
    source_root.mkdir()
    variant_root, _ = _write_variant_fixture(module, source_root)
    manifest_path = variant_root / "variant_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop("variantManifestSHA256")
    manifest["schemaVersion"] = "lumen.adapter-experiment-variant/1.2.0"
    manifest["variantManifestSHA256"] = module._canonical_sha256(manifest)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="schema is unsupported"):
        module._variant_dataset(
            source_root / "executor",
            agent="executor",
            variant="internal_plus_public_optimized",
        )


@pytest.mark.parametrize(
    ("stored_seed", "requested_seed"),
    [(True, 1), (42.0, 42)],
)
def test_prepare_configs_rejects_non_integer_seed_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stored_seed: object,
    requested_seed: int,
) -> None:
    module = _load_app(tmp_path, monkeypatch)
    source_root = tmp_path / "datasets"
    source_root.mkdir()
    variant_root, _ = _write_variant_fixture(module, source_root)
    config_path = source_root / "executor" / "unsloth_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["seed"] = stored_seed
    config_path.write_text(json.dumps(config), encoding="utf-8")
    manifest_path = variant_root / "variant_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop("variantManifestSHA256")
    manifest["seed"] = stored_seed
    controlled = manifest["controlledTrainingConfig"]
    controlled["seed"] = stored_seed
    manifest["trainingConfigSHA256"] = module._canonical_sha256(controlled)
    manifest["trainingConfigInvariantSHA256"] = module._canonical_sha256(
        module._normalized_invariant_training_config(
            controlled,
            agent="executor",
            sft_train_record_count=manifest["datasets"]["trainSFT"]["count"],
            dpo_train_record_count=manifest["datasets"]["trainDPO"]["count"],
        )
    )
    manifest["variantManifestSHA256"] = module._canonical_sha256(manifest)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="seed contract is invalid"):
        module._prepare_configs(
            source_root=source_root,
            run_root=tmp_path / "run",
            agents=["executor"],
            base_model_override="",
            seed=requested_seed,
            variant="internal_plus_public_optimized",
        )


@pytest.mark.parametrize("controlled_seed", [True, 1.0])
def test_training_attestation_rejects_python_numeric_equality(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    controlled_seed: object,
) -> None:
    module = _load_app(tmp_path, monkeypatch)
    controlled = {"seed": controlled_seed}
    manifest = {
        "datasets": {},
        "controlledTrainingConfig": controlled,
        "trainingConfigSHA256": module._canonical_sha256(controlled),
    }

    with pytest.raises(ValueError, match="drifted from the controlled variant"):
        module._training_attestation({"seed": 1}, manifest)


def test_training_attestation_rejects_declared_exact_hash_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_app(tmp_path, monkeypatch)
    controlled = {"seed": 42}
    manifest = {
        "datasets": {},
        "controlledTrainingConfig": controlled,
        "trainingConfigSHA256": "0" * 64,
    }

    with pytest.raises(ValueError, match="drifted from the controlled variant"):
        module._training_attestation(dict(controlled), manifest)


def _authorized_request(token: str) -> SimpleNamespace:
    return SimpleNamespace(headers={"x-lumen-admin-token": token})


def _test_accelerator() -> dict[str, Any]:
    return {
        "bindingStatus": "runtime_observed_unverified",
        "backend": "cuda",
        "deviceCount": 1,
        "devices": [
            {
                "index": 0,
                "name": "Synthetic CUDA",
                "totalMemoryBytes": 24 * 1024 * 1024 * 1024,
                "computeCapability": [8, 0],
            }
        ],
    }


def test_space_browser_surface_is_api_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_app(tmp_path, monkeypatch)

    assert module.run.kwargs["visible"] is False
    assert module.run.click_kwargs["api_visibility"] == "undocumented"
    assert module.output.kwargs["visible"] is False
    for component in (
        module.run_id,
        module.agents,
        module.base_model,
        module.seed,
        module.gpu_size,
        module.experiment_variant,
        module.assistant_loss,
        module.resume,
        module.convert,
        module.upload,
        module.confirm_variant,
        module.destructive_reset,
    ):
        assert component.kwargs["visible"] is False


def test_startup_environment_cache_is_reused_without_package_rescan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_app(tmp_path, monkeypatch)
    monkeypatch.setattr(
        module,
        "build_resolved_training_environment_snapshot",
        lambda: pytest.fail("request preflight must reuse the startup scan"),
    )

    first = module._verified_startup_environment_cache()
    second = module._verified_startup_environment_cache()

    assert first == second
    assert first[1]["distributionCount"] == 1
    assert first[1]["totalHashedBytes"] == 128
    child_environment = module._startup_environment_child_variable()
    assert json.loads(
        child_environment[
            "LUMEN_ZERO_GPU_RESOLVED_ENVIRONMENT_CACHE_ATTESTATION"
        ]
    ) == first[2]
    assert len(
        child_environment[
            "LUMEN_ZERO_GPU_RESOLVED_ENVIRONMENT_CACHE_HMAC_KEY"
        ]
    ) == 64


def test_training_endpoint_rejects_gpu_contract_drift_before_allocation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_app(tmp_path, monkeypatch)
    admin_token = "Lumen-Admin-Token-0123456789-ABCDEF"
    monkeypatch.setenv("LUMEN_ZERO_GPU_ADMIN_TOKEN", admin_token)
    monkeypatch.setenv("LUMEN_ZERO_GPU_HUB_TOKEN", "hf_repository_token")
    calls: list[str] = []
    monkeypatch.setattr(
        module,
        "_train_lumen_adapters_gpu",
        lambda *_args: calls.append("gpu") or {"ok": True},
    )

    response = module.train_lumen_adapters(
        "run",
        "executor",
        "",
        42,
        True,
        False,
        False,
        False,
        "xlarge",
        request=_authorized_request(admin_token),
    )

    assert response["error_code"] == "training_failed"
    assert calls == []


def test_training_endpoint_authorizes_before_gpu_or_filesystem(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_app(tmp_path, monkeypatch)
    work_root = tmp_path / "work"
    admin_token = "Lumen-Admin-Token-0123456789-ABCDEF"
    monkeypatch.setenv("LUMEN_ZERO_GPU_WORKDIR", str(work_root))
    calls: list[str] = []
    monkeypatch.setattr(
        module,
        "_train_lumen_adapters_gpu",
        lambda *_args: calls.append("gpu") or {"ok": True},
    )
    monkeypatch.setattr(
        module,
        "HfApi",
        lambda *_args, **_kwargs: pytest.fail("HfApi must not be instantiated"),
    )

    missing_configuration = module.train_lumen_adapters(
        "run", "executor", "", 42, True, False, False, False, "large"
    )
    assert missing_configuration["error_code"] == "authorization_not_configured"
    assert calls == []
    assert not work_root.exists()
    monkeypatch.setenv("LUMEN_ZERO_GPU_ADMIN_TOKEN", admin_token)

    missing = module.train_lumen_adapters(
        "run", "executor", "", 42, True, False, False, False, "large"
    )
    wrong = module.train_lumen_adapters(
        "run",
        "executor",
        "",
        42,
        True,
        False,
        False,
        False,
        "large",
        request=_authorized_request("Wrong-Admin-Token-0123456789-ABCDEF"),
    )
    assert missing["error_code"] == "unauthorized"
    assert wrong["error_code"] == "unauthorized"
    assert calls == []
    assert not work_root.exists()

    missing_repository_token = module.train_lumen_adapters(
        "run",
        "executor",
        "",
        42,
        True,
        False,
        False,
        False,
        "large",
        request=_authorized_request(admin_token),
    )
    assert (
        missing_repository_token["error_code"]
        == "repository_authorization_not_configured"
    )
    assert calls == []
    assert not work_root.exists()

    monkeypatch.setenv("LUMEN_ZERO_GPU_HUB_TOKEN", "hf_fine_grained_repository_token")
    accepted = module.train_lumen_adapters(
        "run",
        "executor",
        "",
        42,
        True,
        False,
        False,
        False,
        "large",
        request=_authorized_request(admin_token),
    )
    assert accepted == {"ok": True}
    assert calls == ["gpu"]


def test_training_endpoint_rejects_concurrency_and_sanitizes_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_app(tmp_path, monkeypatch)
    work_root = tmp_path / "sensitive-work"
    admin_token = "Lumen-Admin-Token-0123456789-ABCDEF"
    hub_token = "hf_secret_repository_token"
    monkeypatch.setenv("LUMEN_ZERO_GPU_WORKDIR", str(work_root))
    monkeypatch.setenv("LUMEN_ZERO_GPU_ADMIN_TOKEN", admin_token)
    monkeypatch.setenv("LUMEN_ZERO_GPU_HUB_TOKEN", hub_token)
    request = _authorized_request(admin_token)

    with module._exclusive_training_operation():
        conflict = module.train_lumen_adapters(
            "run", "executor", "", 42, True, False, False, False, "large", request=request
        )
    assert conflict["error_code"] == "training_already_active"

    monkeypatch.setattr(
        module,
        "_train_lumen_adapters_gpu",
        lambda *_args: (_ for _ in ()).throw(
            RuntimeError(f"secret={hub_token} path={work_root}")
        ),
    )
    failed = module.train_lumen_adapters(
        "run", "executor", "", 42, True, False, False, False, "large", request=request
    )
    rendered = json.dumps(failed)
    assert failed["error_code"] == "training_failed"
    assert "traceback" not in rendered.casefold()
    assert hub_token not in rendered
    assert str(work_root) not in rendered


def _write_resume_fixture(
    module: Any,
    *,
    work_root: Path,
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    run_id = "resume-test-internal_plus_public_optimized"
    run_root = work_root / run_id
    source_root = run_root / "generated" / "fine_tuning"
    source_root.mkdir(parents=True)
    _write_variant_fixture(module, source_root)
    runtime = {
        "trainingCodeManifest": {"phase": "sft"},
        "trainingCodeSHA256": "1" * 64,
        "trainingDependencyLock": {"schema": "lock"},
        "trainingDependencyLockSHA256": "2" * 64,
        "requirementsSHA256": "3" * 64,
        "resolvedTrainingEnvironment": module.TEST_RESOLVED_TRAINING_ENVIRONMENT,
        "resolvedTrainingEnvironmentSHA256": module.TEST_RESOLVED_TRAINING_ENVIRONMENT[
            "resolvedTrainingEnvironmentSHA256"
        ],
        "resolvedTrainingEnvironmentCacheAttestation": module._STARTUP_ENVIRONMENT_ATTESTATION,
        "resolvedTrainingEnvironmentScanAudit": module.TEST_RESOLVED_TRAINING_ENVIRONMENT_SCAN,
        "zeroGPUSize": "large",
        "zeroGPUDurationSeconds": 1200,
        "observedAccelerator": _test_accelerator(),
        "spaceConfigurationSHA256": module.DEFAULTS[
            "spaceConfigurationSHA256"
        ],
        "runtimeSourceKind": "huggingface_space",
        "runtimeSourceRevision": "4" * 40,
        "expectedRuntimeSourceRevision": "4" * 40,
        "observedRepositoryRevision": "4" * 40,
        "observedRuntimeRevision": None,
        "runtimeSourceBindingStatus": "operator_declared_unverified",
        "runtimeSourceBindingMethod": "huggingface_repository_head_supplemental",
    }
    lineage = module._build_run_resume_lineage(
        run_id=run_id,
        run_root=run_root,
        source_root=source_root,
        dataset_repo="user/dataset",
        dataset_revision="5" * 40,
        dataset_path="runs/test/fine_tuning",
        agents=["executor"],
        variant="internal_plus_public_optimized",
        seed=42,
        assistant_only_loss=True,
        runtime_lineage=runtime,
    )
    prepared = module._prepare_configs(
        source_root=source_root,
        run_root=run_root,
        agents=["executor"],
        base_model_override="",
        seed=42,
        variant="internal_plus_public_optimized",
        run_lineage=lineage,
        runtime_lineage=runtime,
    )
    module._write_fresh_run_contract(
        run_root=run_root,
        run_lineage=lineage,
        prepared=prepared,
        resolved_environment_scan_audit=runtime[
            "resolvedTrainingEnvironmentScanAudit"
        ],
    )
    checkpoint = run_root / "training" / "executor" / "checkpoint-1"
    checkpoint.mkdir(parents=True)
    (checkpoint / "trainer_state.json").write_text("{}\n", encoding="utf-8")
    checkpoint_digest = module._checkpoint_directory_manifest(checkpoint)[
        "checkpointSHA256"
    ]
    checkpoint_path = Path(lineage["agents"][0]["checkpointLineagePath"])
    record = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    record["checkpoints"] = [
        {"path": "checkpoint-1", "checkpointSHA256": checkpoint_digest}
    ]
    record = module._self_hashed(record, field="checkpointLineageSHA256")
    module._atomic_write_json(checkpoint_path, record)
    return run_root, lineage, runtime


def test_resume_contract_accepts_unchanged_lineage_without_snapshot_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_app(tmp_path, monkeypatch)
    work_root = tmp_path / "work"
    run_root, lineage, runtime = _write_resume_fixture(module, work_root=work_root)
    monkeypatch.setenv("LUMEN_ZERO_GPU_WORKDIR", str(work_root))
    monkeypatch.setenv("LUMEN_ZERO_GPU_DATASET_REPO", "user/dataset")
    monkeypatch.setenv("LUMEN_ZERO_GPU_DATASET_REVISION", "5" * 40)
    monkeypatch.setenv("LUMEN_ZERO_GPU_DATASET_PATH", "runs/test/fine_tuning")
    monkeypatch.setenv("LUMEN_ZERO_GPU_HUB_TOKEN", "fine-grained-token")
    module.DEFAULTS.update(
        {
            "dataset_repo": "user/dataset",
            "dataset_revision": "5" * 40,
            "dataset_path_in_repo": "runs/test/fine_tuning",
        }
    )
    monkeypatch.setattr(module, "_verify_runtime_lineage", lambda: runtime)
    monkeypatch.setattr(module, "_observed_accelerator", _test_accelerator)
    monkeypatch.setattr(
        module,
        "_copy_dataset_snapshot",
        lambda *_args, **_kwargs: pytest.fail("resume must not replace the snapshot"),
    )
    original_rmtree = module.shutil.rmtree
    monkeypatch.setattr(
        module.shutil,
        "rmtree",
        lambda *_args, **_kwargs: pytest.fail("resume must not recursively delete"),
    )
    monkeypatch.setattr(module, "_run", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        module,
        "_verify_trained_adapter",
        lambda item: (
            Path(item["adapter_dir"]),
            {"artifact": {"adapterSHA256": "a" * 64}, "variantManifestSHA256": "b" * 64},
        ),
    )
    try:
        result = module._train_lumen_adapters_gpu(
            "resume-test",
            "executor",
            "",
            42,
            True,
            True,
            False,
            False,
            "large",
            "internal_plus_public_optimized",
            True,
            False,
        )
    finally:
        monkeypatch.setattr(module.shutil, "rmtree", original_rmtree)
    assert result["ok"] is True
    assert result["runResumeLineageSHA256"] == lineage["runResumeLineageSHA256"]
    assert result["requirementsSHA256"] == lineage["requirementsSHA256"]
    assert run_root.is_dir()


def test_resume_lineage_ignores_new_startup_scan_timing_and_cache_signature(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_app(tmp_path, monkeypatch)
    run_root, lineage, runtime = _write_resume_fixture(
        module,
        work_root=tmp_path / "work",
    )
    restarted_runtime = json.loads(json.dumps(runtime))
    restarted_runtime["resolvedTrainingEnvironmentScanAudit"][
        "durationMilliseconds"
    ] += 19
    restarted_runtime["resolvedTrainingEnvironmentCacheAttestation"][
        "startupID"
    ] = "f" * 32
    restarted_runtime["resolvedTrainingEnvironmentCacheAttestation"][
        "cacheHMACSHA256"
    ] = "e" * 64
    source_root = run_root / "generated" / "fine_tuning"
    expected = module._build_run_resume_lineage(
        run_id=lineage["runID"],
        run_root=run_root,
        source_root=source_root,
        dataset_repo=lineage["datasetRepository"],
        dataset_revision=lineage["datasetRevision"],
        dataset_path=lineage["datasetPath"],
        agents=lineage["selectedAgents"],
        variant=lineage["experimentVariant"],
        seed=lineage["seed"],
        assistant_only_loss=lineage["assistantOnlyLoss"],
        runtime_lineage=restarted_runtime,
    )

    assert expected == lineage
    manifest_path, prepared = module._load_resume_contract(
        run_root=run_root,
        expected_lineage=expected,
    )
    assert manifest_path == run_root / module.RUN_MANIFEST_NAME
    assert [item["agent"] for item in prepared] == ["executor"]


@pytest.mark.parametrize(
    "field,replacement",
    [
        ("datasetRevision", "6" * 40),
        ("experimentVariant", "internal_only"),
        ("seed", 7),
        ("assistantOnlyLoss", False),
        ("trainingCodeSHA256", "7" * 64),
        ("trainingDependencyLockSHA256", "8" * 64),
        ("requirementsSHA256", "9" * 64),
        ("resolvedTrainingEnvironmentSHA256", "e" * 64),
        ("spaceConfigurationSHA256", "f" * 64),
        ("runtimeSourceRevision", "a" * 40),
        ("expectedRuntimeSourceRevision", "a" * 40),
        ("observedRepositoryRevision", "a" * 40),
        ("runtimeSourceBindingStatus", "verified"),
        ("runtimeSourceBindingMethod", "self_declared"),
    ],
)
def test_resume_contract_rejects_top_level_lineage_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    replacement: Any,
) -> None:
    module = _load_app(tmp_path, monkeypatch)
    run_root, lineage, _ = _write_resume_fixture(module, work_root=tmp_path / "work-run")
    drifted = json.loads(json.dumps(lineage))
    drifted[field] = replacement
    unsigned = dict(drifted)
    unsigned.pop("runResumeLineageSHA256")
    drifted["runResumeLineageSHA256"] = module._canonical_sha256(unsigned)
    with pytest.raises(ValueError, match="Resume lineage"):
        module._load_resume_contract(run_root=run_root, expected_lineage=drifted)


def test_resume_contract_rejects_agent_lane_base_and_manifest_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_app(tmp_path, monkeypatch)
    run_root, lineage, _ = _write_resume_fixture(module, work_root=tmp_path / "work")
    for mutate in (
        lambda value: value["agents"][0]["laneHashes"].update({"trainSFT": "9" * 64}),
        lambda value: value["agents"][0].update({"sourceVariantManifestSHA256": "a" * 64}),
        lambda value: value["agents"][0].update({"baseModelRevision": "b" * 40}),
        lambda value: value["agents"][0].update({"baseModelArtifactDigest": "c" * 64}),
        lambda value: value["agents"][0].update({"baseModelIndexShardBindingSHA256": "d" * 64}),
        lambda value: value["agents"][0].update({"trainingEnvironmentLockSHA256": "e" * 64}),
    ):
        drifted = json.loads(json.dumps(lineage))
        mutate(drifted)
        unsigned = dict(drifted)
        unsigned.pop("runResumeLineageSHA256")
        drifted["runResumeLineageSHA256"] = module._canonical_sha256(unsigned)
        with pytest.raises(ValueError, match="Resume lineage"):
            module._load_resume_contract(run_root=run_root, expected_lineage=drifted)


def test_resume_contract_requires_run_and_checkpoint_manifests(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_app(tmp_path, monkeypatch)
    run_root, lineage, _ = _write_resume_fixture(module, work_root=tmp_path / "work")
    run_manifest = run_root / module.RUN_MANIFEST_NAME
    run_manifest.unlink()
    with pytest.raises(FileNotFoundError, match="lineage manifest"):
        module._load_resume_contract(run_root=run_root, expected_lineage=lineage)

    run_root, lineage, _ = _write_resume_fixture(
        module, work_root=tmp_path / "work-checkpoint"
    )
    checkpoint_manifest = Path(lineage["agents"][0]["checkpointLineagePath"])
    checkpoint_manifest.unlink()
    with pytest.raises(FileNotFoundError, match="lineage manifest"):
        module._load_resume_contract(run_root=run_root, expected_lineage=lineage)


def test_resume_reads_run_manifest_before_runtime_or_snapshot_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_app(tmp_path, monkeypatch)
    work_root = tmp_path / "work"
    run_root = work_root / "missing-manifest-internal_plus_public_optimized"
    run_root.mkdir(parents=True)
    monkeypatch.setenv("LUMEN_ZERO_GPU_WORKDIR", str(work_root))
    monkeypatch.setattr(
        module,
        "_verify_runtime_lineage",
        lambda: pytest.fail("runtime lineage must not be read before the run manifest"),
    )

    with pytest.raises(FileNotFoundError, match="lineage manifest"):
        module._train_lumen_adapters_gpu(
            "missing-manifest",
            "executor",
            "",
            42,
            True,
            True,
            False,
            False,
            "large",
            "internal_plus_public_optimized",
            True,
            False,
        )
