from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


AGENTS = ("cortex", "executor", "mouth", "mimicry", "rem", "fleet")


def _load_export_module() -> ModuleType:
    repo_root = Path(__file__).resolve().parents[3]
    module_path = repo_root / "tools" / "fine_tuning" / "unsloth" / "export_gguf.py"
    spec = importlib.util.spec_from_file_location("lumen_unsloth_export_gguf", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_config(
    tmp_path: Path,
    *,
    agent: str = "cortex",
    merge_by_default: bool = False,
    prepared_release_bake: bool = False,
) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    tokenizer_files = [
        {
            "path": filename,
            "sizeBytes": 1,
            "sha256": "e" * 64,
            "huggingFaceBlobID": "1" * 40,
        }
        for filename in (
            "config.json",
            "merges.txt",
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.json",
        )
    ]
    config = {
        "agent": agent,
        "base_model_name": "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit",
        "baseModelID": "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit",
        "baseModelRevision": "a" * 40,
        "baseModelIndexDigest": "b" * 64,
        "baseModelIndexReferencedShardNames": ["model.safetensors"],
        "baseModelIndexShardBindingSHA256": "f" * 64,
        "baseModelArtifactDigest": "c" * 64,
        "baseModelWeightShards": [
            {"filename": "model.safetensors", "size": 1, "sha256": "d" * 64}
        ],
        "baseModelTokenizerDigest": "e" * 64,
        "baseModelTokenizerFiles": tokenizer_files,
        "max_seq_length": 4096,
        "load_in_4bit": True,
        "output_dir": f"{tmp_path}/models/lora/{agent}",
        "adapter_output_dir": f"{tmp_path}/models/lora/{agent}",
        "artifact_mode": "adapter_first",
        "default_export_artifact": "lora_adapter",
        "merge_adapters_by_default": merge_by_default,
        "release_bake_enabled_by_default": False,
        "gguf_output_dir": f"{tmp_path}/models/gguf_release_bake/{agent}_merged_gguf",
    }
    config["baseModelTokenizerClosureSHA256"] = hashlib.sha256(
        json.dumps(
            {
                "schemaVersion": "lumen.base-model-tokenizer-closure/1.0.0",
                "baseModelID": config["baseModelID"],
                "baseModelRevision": config["baseModelRevision"],
                "files": tokenizer_files,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    if prepared_release_bake:
        config.update(
            {
                "adapter_training_phase": "sft_dpo",
                "baseModelGenerationConfigFile": {
                    "path": "generation_config.json",
                    "sizeBytes": 1,
                    "sha256": "9" * 64,
                    "huggingFaceBlobID": "8" * 40,
                },
                "baseModelTokenizerSnapshotPath": str(
                    tmp_path / "tokenizer_snapshot"
                ),
                "baseModelTokenizerSnapshotVerification": {
                    "snapshotPath": str(tmp_path / "tokenizer_snapshot"),
                },
                "baseModelRuntimeSnapshotPath": str(tmp_path / "runtime_snapshot"),
                "baseModelRuntimeSnapshotVerification": {
                    "snapshotPath": str(tmp_path / "runtime_snapshot"),
                },
                "bf16": True,
                "finalized_variant_manifest": str(
                    tmp_path / "finalized_variant_manifest.json"
                ),
                "fp16": False,
                "parent_sft_adapter_sha256": "7" * 64,
                "trainingEnvironmentSHA256": "6" * 64,
                "variant": "internal_plus_public_optimized",
                "variantAttestation": {"schema": "fixture/1.0.0"},
                "variantManifestSHA256": "5" * 64,
            }
        )
    path = tmp_path / f"{agent}.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def test_load_config_requires_adapter_first_merge_disabled(tmp_path: Path) -> None:
    export_gguf = _load_export_module()
    config_path = _write_config(tmp_path, merge_by_default=False)

    cfg = export_gguf.load_config(config_path)

    assert cfg["artifact_mode"] == "adapter_first"
    assert cfg["merge_adapters_by_default"] is False
    assert cfg["release_bake_enabled_by_default"] is False


def test_load_config_rejects_merge_by_default(tmp_path: Path) -> None:
    export_gguf = _load_export_module()
    config_path = _write_config(tmp_path, merge_by_default=True)

    try:
        export_gguf.load_config(config_path)
    except ValueError as error:
        assert "merge_adapters_by_default=false" in str(error)
    else:
        raise AssertionError("load_config should reject configs that merge adapters by default")


def test_load_config_rejects_legacy_config_without_base_lineage(tmp_path: Path) -> None:
    export_gguf = _load_export_module()
    config_path = _write_config(tmp_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config.pop("baseModelIndexDigest")
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="baseModelIndexDigest"):
        export_gguf.load_config(config_path)


def test_load_config_rejects_split_base_model_identity(tmp_path: Path) -> None:
    export_gguf = _load_export_module()
    config_path = _write_config(tmp_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["baseModelID"] = "example/different-model"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="baseModelID must exactly match base_model_name",
    ):
        export_gguf.load_config(config_path)


def test_gather_configs_prefers_generated_nested_configs(tmp_path: Path) -> None:
    export_gguf = _load_export_module()
    config_path = _write_config(tmp_path)
    generated = tmp_path / "generated" / "cortex"
    generated.mkdir(parents=True)
    (generated / "unsloth_config.json").write_text(
        config_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    configs = export_gguf.gather_configs([], str(tmp_path / "generated"), ["cortex"])

    assert [config["agent"] for config in configs] == ["cortex"]


def test_checked_in_generated_configs_load_for_adapter_first_default() -> None:
    export_gguf = _load_export_module()
    repo_root = Path(__file__).resolve().parents[3]
    config_dir = repo_root / "generated" / "fine_tuning"

    configs = export_gguf.gather_configs(
        [],
        str(config_dir),
        list(AGENTS),
    )

    assert [config["agent"] for config in configs] == list(AGENTS)
    assert all("baseModelRuntimeSnapshotPath" not in config for config in configs)


def test_release_bake_default_resolves_prepared_run_configs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    export_gguf = _load_export_module()
    monkeypatch.setenv("LUMEN_AIO_RUN_ROOT", str(tmp_path / "run"))

    resolved = export_gguf._resolve_config_dir(
        config_paths=[],
        config_dir=None,
        release_bake=True,
    )

    assert resolved == str(tmp_path / "run" / "configs")


def test_explicit_config_defines_default_selected_agents(tmp_path: Path) -> None:
    export_gguf = _load_export_module()
    config_path = _write_config(
        tmp_path,
        agent="fleet",
        prepared_release_bake=True,
    )

    selected_agents = export_gguf._selected_agents(
        agents_arg=None,
        config_paths=[str(config_path)],
    )
    configs = export_gguf.gather_configs(
        [str(config_path)],
        None,
        selected_agents,
        require_release_bake_lineage=True,
    )

    assert selected_agents == ["fleet"]
    assert [config["agent"] for config in configs] == ["fleet"]


def test_release_bake_config_records_canonical_source_path(tmp_path: Path) -> None:
    export_gguf = _load_export_module()
    config_path = _write_config(tmp_path, prepared_release_bake=True)

    config = export_gguf.load_config(
        config_path,
        require_release_bake_lineage=True,
    )

    assert config[export_gguf.CONFIG_SOURCE_PATH_KEY] == str(config_path.resolve())


def test_release_bake_without_prepared_source_fails_before_discovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    export_gguf = _load_export_module()
    monkeypatch.delenv("LUMEN_AIO_RUN_ROOT", raising=False)

    with pytest.raises(ValueError, match="will not guess a recent run"):
        export_gguf._resolve_config_dir(
            config_paths=[],
            config_dir=None,
            release_bake=True,
        )


def test_release_bake_directory_requires_and_selects_final_config(
    tmp_path: Path,
) -> None:
    export_gguf = _load_export_module()
    prepared_path = _write_config(tmp_path, prepared_release_bake=True)
    final_path = tmp_path / "cortex.final.json"
    final_path.write_text(prepared_path.read_text(encoding="utf-8"), encoding="utf-8")
    _write_config(tmp_path, prepared_release_bake=False)

    configs = export_gguf.gather_configs(
        [],
        str(tmp_path),
        ["cortex"],
        require_release_bake_lineage=True,
    )

    assert configs[0]["adapter_training_phase"] == "sft_dpo"
    assert configs[0]["parent_sft_adapter_sha256"] == "7" * 64


def test_release_bake_directory_never_falls_back_to_sft_config(
    tmp_path: Path,
) -> None:
    export_gguf = _load_export_module()
    _write_config(tmp_path)

    with pytest.raises(FileNotFoundError, match="will not fall back to cortex.json"):
        export_gguf.gather_configs(
            [],
            str(tmp_path),
            ["cortex"],
            require_release_bake_lineage=True,
        )


def test_release_bake_rejects_static_generated_config(tmp_path: Path) -> None:
    export_gguf = _load_export_module()
    config_path = _write_config(tmp_path)

    with pytest.raises(ValueError, match="prepared <agent>.final.json"):
        export_gguf.load_config(
            config_path,
            require_release_bake_lineage=True,
        )


def test_release_bake_dpo_config_requires_parent_sft_digest(tmp_path: Path) -> None:
    export_gguf = _load_export_module()
    config_path = _write_config(tmp_path, prepared_release_bake=True)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config.pop("parent_sft_adapter_sha256")
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="requires parent_sft_adapter_sha256"):
        export_gguf.load_config(
            config_path,
            require_release_bake_lineage=True,
        )


def test_release_bake_skipped_manifest_is_adapter_first(tmp_path: Path) -> None:
    export_gguf = _load_export_module()
    config_path = _write_config(tmp_path, merge_by_default=False)
    cfg = export_gguf.load_config(config_path)

    class Args:
        manifest_output = str(tmp_path / "release_bake_gguf_manifest.json")

    manifest = export_gguf._release_bake_skipped_manifest([cfg], Args())

    assert manifest["mode"] == "adapter_first"
    assert manifest["release_bake_requested"] is False
    assert manifest["skipped"] is True
    assert "--release-bake" in manifest["reason"]
    assert manifest["agents"]["cortex"]["merge_adapters_by_default"] is False
    assert manifest["agents"]["cortex"]["release_bake_enabled_by_default"] is False


def test_static_agent_configs_disable_default_release_bake() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    config_dir = repo_root / "tools" / "fine_tuning" / "unsloth" / "configs"

    for agent in AGENTS:
        cfg = json.loads((config_dir / f"{agent}.json").read_text(encoding="utf-8"))
        assert cfg["agent"] == agent
        assert cfg["artifact_mode"] == "adapter_first"
        assert cfg["default_export_artifact"] == "lora_adapter"
        assert cfg["merge_adapters_by_default"] is False
        assert cfg["release_bake_enabled_by_default"] is False
        assert "lora" in cfg["output_dir"].lower()
        assert "release_bake" in cfg["gguf_output_dir"].lower()


def test_skip_existing_requires_matching_current_lineage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    export_gguf = _load_export_module()
    config_path = _write_config(tmp_path, prepared_release_bake=True)
    cfg = export_gguf.load_config(config_path)
    output_root = tmp_path / "explicit_release_bake_root"
    output_dir = output_root / "cortex_release_bake_gguf"
    assert output_dir != Path(cfg["gguf_output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / "lumen-cortex-release-bake-q4_k_m.gguf"
    target.write_bytes(b"GGUF-current")
    lineage = {
        "adapterSHA256": "1" * 64,
        "adapterTrainingPhase": "sft",
        "finalizedVariantManifestSHA256": "2" * 64,
        "sourceVariantManifestSHA256": "3" * 64,
        "trainingConfigSHA256": "5" * 64,
        "trainingConfigInvariantSHA256": "6" * 64,
        "baseModelRevision": cfg["baseModelRevision"],
        "baseModelIndexDigest": cfg["baseModelIndexDigest"],
        "baseModelIndexReferencedShardNames": cfg["baseModelIndexReferencedShardNames"],
        "baseModelIndexShardBindingSHA256": cfg["baseModelIndexShardBindingSHA256"],
        "baseModelArtifactDigest": cfg["baseModelArtifactDigest"],
        "baseModelTokenizerDigest": cfg["baseModelTokenizerDigest"],
        "trainingEnvironmentSHA256": "4" * 64,
    }
    calls: list[str] = []

    def verify_lineage(_cfg: dict[str, object]) -> dict[str, object]:
        calls.append("verified")
        return lineage

    monkeypatch.setattr(export_gguf, "_verified_release_bake_lineage", verify_lineage)
    runtime_path = Path(cfg["baseModelRuntimeSnapshotPath"])
    runtime_path.mkdir()
    runtime_verification = cfg["baseModelRuntimeSnapshotVerification"]
    runtime_evidence = {
        "baseModelTokenizerDigest": cfg["baseModelTokenizerDigest"],
        "baseModelTokenizerFiles": cfg["baseModelTokenizerFiles"],
        "baseModelTokenizerClosureSHA256": cfg[
            "baseModelTokenizerClosureSHA256"
        ],
        "baseModelGenerationConfigFile": cfg["baseModelGenerationConfigFile"],
        "baseModelTokenizerSnapshotPath": cfg["baseModelTokenizerSnapshotPath"],
        "baseModelTokenizerSnapshotVerification": cfg[
            "baseModelTokenizerSnapshotVerification"
        ],
        "baseModelRuntimeSnapshotPath": str(runtime_path),
        "baseModelRuntimeSnapshotVerification": runtime_verification,
        "runtimeModelBinding": {"binding": "model"},
        "runtimeTokenizerBinding": {"binding": "tokenizer"},
    }
    monkeypatch.setattr(
        export_gguf,
        "_verified_private_runtime_model_snapshot",
        lambda _cfg: (runtime_path, runtime_verification),
    )
    monkeypatch.setattr(
        export_gguf,
        "_runtime_tokenizer_evidence",
        lambda _cfg, **_kwargs: runtime_evidence,
    )
    report = {
        "agent": "cortex",
        "mode": "optional_release_bake",
        "quantization": "q4_k_m",
        "adapter_dir": str(Path(cfg["output_dir"]).resolve()),
        "gguf_output_dir": str(output_dir.resolve()),
        "gguf_file": target.name,
        "gguf_path": str(target.resolve()),
        "size_bytes": target.stat().st_size,
        "sha256": export_gguf.sha256sum(target),
        "base_model_name": cfg["base_model_name"],
        **runtime_evidence,
        **lineage,
    }
    (output_dir / "gguf_release_bake_report.json").write_text(
        json.dumps(report),
        encoding="utf-8",
    )

    summary = export_gguf.existing_summary_for_agent(
        cfg,
        output_root=output_root,
        quantization_override=None,
    )

    assert calls == ["verified"]
    assert summary is not None and summary["reused_existing"] is True

    target.write_bytes(b"GGUF-stale")
    with pytest.raises(ValueError, match="does not match current (size_bytes|sha256)"):
        export_gguf.existing_summary_for_agent(
            cfg,
            output_root=output_root,
            quantization_override=None,
        )


def test_release_bake_requires_verified_full_run_qualification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    export_gguf = _load_export_module()
    from tools.fine_tuning.unsloth import ubuntu_pipeline

    run_root = tmp_path / "prepared-run"
    config_dir = run_root / "configs"
    source = _write_config(
        config_dir,
        agent="cortex",
        prepared_release_bake=True,
    )
    final_path = config_dir / "cortex.final.json"
    final_path.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    config = export_gguf.load_config(
        final_path,
        require_release_bake_lineage=True,
    )
    (run_root / "aio_run_manifest.json").write_text(
        json.dumps({"agents": [{"agent": "cortex"}]}),
        encoding="utf-8",
    )
    adapter_sha = "4" * 64
    summary = {
        "status": "complete_without_gguf",
        "evaluationStatus": "quality_gate_passed",
        "evaluationScope": "full",
        "qualification": "quality_gate_passed",
        "promotionEligible": True,
        "preferenceTraining": True,
        "summarySHA256": "3" * 64,
        "agents": {
            "cortex": {"finalPhase": {"adapterSHA256": adapter_sha}},
        },
    }
    monkeypatch.setattr(
        ubuntu_pipeline,
        "_verified_completed_summary",
        lambda _run_root, _agents: summary,
    )

    evidence = export_gguf._verified_release_bake_qualification(
        [config],
        {"cortex": {"adapterSHA256": adapter_sha}},
    )

    assert evidence["sourceRunSummarySHA256"] == "3" * 64
    assert evidence["sourceRunEvaluationStatus"] == "quality_gate_passed"
    summary["evaluationScope"] = "smoke"
    with pytest.raises(ValueError, match="verified full evaluation"):
        export_gguf._verified_release_bake_qualification(
            [config],
            {"cortex": {"adapterSHA256": adapter_sha}},
        )


def test_release_bake_rejects_direct_upload_before_output_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    export_gguf = _load_export_module()
    config_dir = tmp_path / "prepared-run" / "configs"
    source = _write_config(config_dir, prepared_release_bake=True)
    final_path = config_dir / "cortex.final.json"
    final_path.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    output_root = tmp_path / "release-bake-output"
    monkeypatch.setattr(
        export_gguf,
        "parse_args",
        lambda: SimpleNamespace(
            release_bake=True,
            config=[str(final_path)],
            config_dir=None,
            agents=None,
            output_root=str(output_root),
            manifest_output=str(tmp_path / "manifest.json"),
            hf_repo_id="example/repo",
            hf_private=True,
            skip_upload=False,
            quantization=None,
            max_memory_usage=None,
            skip_existing=False,
        ),
    )

    with pytest.raises(ValueError, match="Direct Hugging Face upload is unsupported"):
        export_gguf.main()

    assert not output_root.exists()


def test_release_bake_rejects_rehashed_training_config_invariant_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    export_gguf = _load_export_module()
    adapter_dir = tmp_path / "models/lora/cortex"
    adapter_dir.mkdir(parents=True)
    finalized_path = tmp_path / "finalized_variant_manifest.json"
    training_environment = {"schemaVersion": "fixture/1.0.0"}
    training_environment_sha = export_gguf._canonical_sha256(
        training_environment
    )
    lane_hashes = {
        "trainSFT": "1" * 64,
        "validationSFT": "2" * 64,
        "trainDPO": "3" * 64,
        "validationDPO": "4" * 64,
    }
    source_manifest_sha = "5" * 64
    attestation = {
        "schema": export_gguf.TRAINING_VARIANT_ATTESTATION_SCHEMA,
        "variant": "internal_plus_public_optimized",
        "variantManifestSHA256": source_manifest_sha,
        "trainingEnvironmentSHA256": training_environment_sha,
        "trainingCorpusSHA256": "6" * 64,
        "effectiveTrainingConfigSHA256": "7" * 64,
        "trainingConfigInvariantSHA256": "8" * 64,
        "laneHashes": lane_hashes,
    }
    config = {
        "agent": "cortex",
        "variant": "internal_plus_public_optimized",
        "variantManifestSHA256": source_manifest_sha,
        "adapter_output_dir": str(adapter_dir),
        "output_dir": str(tmp_path / "training/cortex"),
        "finalized_variant_manifest": str(finalized_path),
        "base_model_name": "Qwen/Qwen3-1.7B",
        "baseModelID": "Qwen/Qwen3-1.7B",
        "baseModelRevision": "a" * 40,
        "baseModelIndexDigest": "9" * 64,
        "baseModelIndexReferencedShardNames": ["model.safetensors"],
        "baseModelIndexShardBindingSHA256": "a" * 64,
        "baseModelArtifactDigest": "b" * 64,
        "baseModelWeightShards": [
            {"filename": "model.safetensors", "sha256": "c" * 64}
        ],
        "baseModelTokenizerDigest": "d" * 64,
        "baseModelTokenizerFiles": [],
        "baseModelTokenizerClosureSHA256": "e" * 64,
        "trainingEnvironmentSHA256": training_environment_sha,
        "variantAttestation": attestation,
    }
    finalized = {
        "agent": config["agent"],
        "variant": config["variant"],
        "sourceVariantManifestSHA256": source_manifest_sha,
        **{
            field: config[field]
            for field in (
                "baseModelRevision",
                "baseModelIndexDigest",
                "baseModelIndexReferencedShardNames",
                "baseModelIndexShardBindingSHA256",
                "baseModelArtifactDigest",
                "baseModelWeightShards",
                "baseModelTokenizerDigest",
                "baseModelTokenizerFiles",
                "baseModelTokenizerClosureSHA256",
            )
        },
        "baseModelID": config["baseModelID"],
        "trainingEnvironment": training_environment,
        "trainingEnvironmentSHA256": training_environment_sha,
        "trainingCorpusSHA256": attestation["trainingCorpusSHA256"],
        "trainingConfigSHA256": attestation[
            "effectiveTrainingConfigSHA256"
        ],
        "trainingConfigInvariantSHA256": "f" * 64,
        "datasets": {
            name: {"sha256": digest}
            for name, digest in lane_hashes.items()
        },
    }
    finalized["variantManifestSHA256"] = export_gguf._canonical_sha256(
        finalized
    )
    finalized_path.write_text(json.dumps(finalized), encoding="utf-8")
    monkeypatch.setattr(export_gguf, "_verify_base_model_lineage", lambda _cfg: None)

    with pytest.raises(
        ValueError,
        match="does not match the prepared training attestation",
    ):
        export_gguf._verified_release_bake_lineage(config)

    attestation.pop("trainingConfigInvariantSHA256")
    finalized.pop("trainingConfigInvariantSHA256")
    finalized.pop("variantManifestSHA256")
    finalized["variantManifestSHA256"] = export_gguf._canonical_sha256(
        finalized
    )
    finalized_path.write_text(json.dumps(finalized), encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="lacks a valid trainingConfigInvariantSHA256",
    ):
        export_gguf._verified_release_bake_lineage(config)

    attestation["trainingConfigInvariantSHA256"] = "8" * 64
    attestation["schema"] = "lumen.training-variant-attestation/1.2.0"

    with pytest.raises(
        ValueError,
        match="lacks a variant training attestation",
    ):
        export_gguf._verified_release_bake_lineage(config)


def test_ubuntu_post_training_gate_requires_finalized_source_and_artifact_lineage() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    script = (repo_root / "tools/fine_tuning/unsloth/ubuntu_pipeline.py").read_text(
        encoding="utf-8"
    )

    assert "Container image digest must be sha256" in script
    assert '"sourceVariantManifestSHA256": config.get("variantManifestSHA256")' in script
    assert "_shared_finalized_variant_validator" in script
    assert "canonical_sha256(unsigned) != digest" in script
    assert 'artifact.get("trainingPhase") != "sft"' in script
    assert "expected_adapter_sha256=str(artifact.get" in script


def test_release_bake_loads_pinned_base_before_verified_adapter() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    source = (
        repo_root / "tools" / "fine_tuning" / "unsloth" / "export_gguf.py"
    ).read_text(encoding="utf-8")

    assert "model_name=str(runtime_tokenizer_snapshot_path)" in source
    assert "tokenizer_name=str(runtime_tokenizer_snapshot_path)" in source
    assert "local_files_only=True" in source
    assert 'revision=cfg["baseModelRevision"]' in source
    assert "PeftModel.from_pretrained(model, str(adapter_dir), is_trainable=False)" in source


def test_dpo_release_bake_requires_external_parent_sft_digest() -> None:
    export_gguf = _load_export_module()
    parent = "a" * 64
    artifact = {"parentSFTAdapterSHA256": parent}

    assert export_gguf._expected_adapter_training_lineage(
        {
            "adapter_training_phase": "sft_dpo",
            "parent_sft_adapter_sha256": parent,
        },
        artifact,
    ) == ("sft_dpo", parent)
    with pytest.raises(ValueError, match="configured parent SFT digest"):
        export_gguf._expected_adapter_training_lineage(
            {"adapter_training_phase": "sft_dpo"}, artifact
        )
    with pytest.raises(ValueError, match="configured parent SFT digest"):
        export_gguf._expected_adapter_training_lineage(
            {
                "adapter_training_phase": "sft_dpo",
                "parent_sft_adapter_sha256": "b" * 64,
            },
            artifact,
        )
