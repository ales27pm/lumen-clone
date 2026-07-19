from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from tools.fine_tuning.unsloth import runtime_binding_smoke_gate as gate


REPO_ROOT = Path(__file__).resolve().parents[4]
_MODEL_CONFIG_PAYLOAD = (
    json.dumps(
        {
            "attention_bias": False,
            "max_position_embeddings": 40_960,
            "model_type": "qwen3",
            "num_hidden_layers": 28,
            "tie_word_embeddings": True,
            "vocab_size": 151_936,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    + "\n"
).encode("utf-8")


def _runtime_config(run_root: Path, agent: str, *, max_seq_length: int = 4096) -> dict[str, Any]:
    runtime_snapshot = run_root / "training" / "base_model_runtime_snapshot"
    tokenizer_snapshot = run_root / "training" / "global_tokenizer_snapshot"
    snapshot_verification = {
        "snapshotPath": str(runtime_snapshot),
        "snapshotVerificationSHA256": "a" * 64,
    }
    tokenizer_verification = {
        "snapshotPath": str(tokenizer_snapshot),
        "snapshotVerificationSHA256": "b" * 64,
    }
    return {
        "agent": agent,
        "output_dir": str(run_root / "training" / agent),
        "adapter_output_dir": str(run_root / "models" / "lora" / agent),
        "dataset_dir": str(run_root / "generated" / agent),
        "variant": "internal_plus_public_optimized",
        "variantManifestSHA256": "c" * 64,
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.0,
        "learning_rate": 2e-4,
        "batch_size": 1,
        "gradient_accumulation_steps": 8,
        "num_train_epochs": 2,
        "warmup_steps": 10,
        "bf16": False,
        "fp16": True,
        "base_model_name": "example/model",
        "baseModelID": "example/model",
        "baseModelRevision": "d" * 40,
        "baseModelIndexDigest": "e" * 64,
        "baseModelIndexReferencedShardNames": ["model-00001.safetensors"],
        "baseModelIndexShardBindingSHA256": "f" * 64,
        "baseModelArtifactDigest": "0" * 64,
        "baseModelWeightShards": [
            {
                "path": "model-00001.safetensors",
                "sizeBytes": 1,
                "sha256": "1" * 64,
            }
        ],
        "baseModelGenerationConfigFile": {
            "path": "generation_config.json",
            "sizeBytes": 1,
            "sha256": "2" * 64,
            "huggingFaceBlobID": "3" * 40,
        },
        "baseModelTokenizerDigest": "4" * 64,
        "baseModelTokenizerFiles": [
            {
                "path": "config.json",
                "sizeBytes": len(_MODEL_CONFIG_PAYLOAD),
                "sha256": hashlib.sha256(_MODEL_CONFIG_PAYLOAD).hexdigest(),
            },
            {"path": "tokenizer.json", "sizeBytes": 1, "sha256": "4" * 64},
        ],
        "baseModelTokenizerClosureSHA256": "5" * 64,
        "baseModelTokenizerSnapshotPath": str(tokenizer_snapshot),
        "baseModelTokenizerSnapshotVerification": tokenizer_verification,
        "baseModelRuntimeSnapshotPath": str(runtime_snapshot),
        "baseModelRuntimeSnapshotVerification": snapshot_verification,
        "chatTemplateContract": {"schemaVersion": "example.chat/1.0.0"},
        "max_seq_length": max_seq_length,
        "load_in_4bit": True,
        "seed": 42,
        "trainingEnvironmentLock": {"schemaVersion": "example.environment/1.0.0"},
        "trainingContainerImageDigest": "sha256:" + "6" * 64,
        "trainingContainerImageDigestSource": "operator_declared",
        "trainingRuntimeImageBindingStatus": "manual_validation_required",
        "trainingRuntimeImageBindingVerified": False,
        "trainingEnvironmentSHA256": "7" * 64,
        "trainingCodeManifest": {"phase": "sft", "files": []},
        "trainingCodeSHA256": "8" * 64,
        "trainingDependencyLock": {"schemaVersion": "example.dependencies/1.0.0"},
        "trainingDependencyLockSHA256": "9" * 64,
        "requirementsSHA256": "a" * 64,
        "resolvedTrainingEnvironment": {"schemaVersion": "example.resolved/1.0.0"},
        "resolvedTrainingEnvironmentSHA256": "b" * 64,
        "resolvedTrainingEnvironmentCacheAttestation": {
            "schemaVersion": "example.cache-attestation/1.0.0"
        },
        "resolvedTrainingEnvironmentScanAudit": {"distributionCount": 1},
        "spaceConfigurationSHA256": None,
        "zeroGPUSize": None,
        "zeroGPUDurationSeconds": None,
        "observedAccelerator": {"backend": "cuda"},
        "runtimeSourceKind": "git",
        "runtimeSourceRevision": "c" * 40,
        "expectedRuntimeSourceRevision": "c" * 40,
        "observedRepositoryRevision": "c" * 40,
        "observedRuntimeRevision": "c" * 40,
        "runtimeSourceBindingStatus": "verified",
        "runtimeSourceBindingMethod": (
            "git_clean_worktree_plus_ubuntu_orchestration_manifest"
        ),
        "workingTreeDigest": "d" * 64,
        "ubuntuOrchestrationCodeSHA256": "e" * 64,
        "ubuntuSourceIntegritySHA256": "f" * 64,
        "ubuntuSourceIntegrity": {"sourceIntegritySHA256": "f" * 64},
    }


def _write_configs(
    tmp_path: Path,
    *,
    max_lengths: dict[str, int],
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, dict[str, dict[str, Any]]]:
    run_root = tmp_path / "run"
    configs_dir = run_root / "configs"
    training_dir = run_root / "training"
    configs_dir.mkdir(parents=True)
    training_dir.mkdir()
    runtime_snapshot = training_dir / "base_model_runtime_snapshot"
    tokenizer_snapshot = training_dir / "global_tokenizer_snapshot"
    runtime_snapshot.mkdir()
    tokenizer_snapshot.mkdir()
    (runtime_snapshot / "config.json").write_bytes(_MODEL_CONFIG_PAYLOAD)
    for directory in (
        run_root,
        configs_dir,
        training_dir,
        runtime_snapshot,
        tokenizer_snapshot,
    ):
        directory.chmod(0o700)
    configs: dict[str, dict[str, Any]] = {}
    for agent, max_length in max_lengths.items():
        config = _runtime_config(run_root, agent, max_seq_length=max_length)
        path = configs_dir / f"{agent}.json"
        path.write_text(json.dumps(config, sort_keys=True) + "\n", encoding="utf-8")
        configs[agent] = config
    monkeypatch.setattr(
        gate,
        "load_config",
        lambda path: json.loads(path.read_text(encoding="utf-8")),
    )
    return run_root, configs


def _smoke_for(config: dict[str, Any]) -> dict[str, Any]:
    snapshot_digest = config["baseModelRuntimeSnapshotVerification"][
        "snapshotVerificationSHA256"
    ]
    compute_dtype = "bfloat16" if config["bf16"] else "float16"
    projection_names = gate._expected_qwen_projection_names(28)
    parameter_names = gate._expected_qwen_parameter_names(
        28,
        tied_embeddings=True,
    )
    outer_quantization_config = {
        "load_in_4bit": True,
        "load_in_8bit": False,
        "quant_method": "bitsandbytes",
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_use_double_quant": True,
        "bnb_4bit_compute_dtype": compute_dtype,
        "llm_int8_enable_fp32_cpu_offload": False,
        "llm_int8_has_fp16_weight": False,
        "llm_int8_threshold": 6.0,
        "llm_int8_skip_modules": None,
    }
    representative_target = {
        "moduleName": projection_names[0],
        "moduleClass": "bitsandbytes.nn.modules.Linear4bit",
        "parameterClass": "bitsandbytes.nn.modules.Params4bit",
        "deviceType": "cuda",
        "storageDType": "uint8",
        "computeDType": compute_dtype,
        "quantType": "nf4",
        "doubleQuantized": True,
        "bnbQuantized": True,
        "requiresGrad": False,
        "quantStatePresent": True,
        "quantStateClass": "bitsandbytes.functional.QuantState",
        "quantStateQuantType": "nf4",
        "quantStateNested": True,
        "quantStateBlocksize": 64,
        "quantStateDType": compute_dtype,
        "quantStateNestedBlocksize": 256,
    }
    placement_unsigned = {
        "schemaVersion": gate.PARAMETER_PLACEMENT_SCHEMA,
        "status": "passed",
        "totalParameterCount": len(parameter_names),
        "cudaParameterCount": len(parameter_names),
        "deviceTypeCounts": {"cuda": len(parameter_names)},
        "parameterNamesSHA256": gate.canonical_sha256(parameter_names),
        "allParametersOnCUDA": True,
    }
    parameter_placement = {
        **placement_unsigned,
        gate.PARAMETER_PLACEMENT_HASH_FIELD: gate.canonical_sha256(
            placement_unsigned
        ),
    }
    forward_unsigned = {
        "schemaVersion": gate.FORWARD_PROBE_SCHEMA,
        "status": "passed",
        "fixedInputSHA256": gate.canonical_sha256(gate._FIXED_FORWARD_INPUT),
        "batchSize": 1,
        "tokenCount": 4,
        "logitsShape": [1, 4, 151_936],
        "logitsDType": compute_dtype,
        "logitsDeviceType": "cuda",
        "allFinite": True,
        "requiresGrad": False,
        "useCache": False,
    }
    forward_probe = {
        **forward_unsigned,
        gate.FORWARD_PROBE_HASH_FIELD: gate.canonical_sha256(forward_unsigned),
    }
    runtime_load_materialization = {
        "requestedMaxSequenceLength": config["max_seq_length"],
        "runtimeMaxSequenceLength": config["max_seq_length"],
        "requestedComputeDType": compute_dtype,
        "runtimeIsLoadedIn4Bit": True,
        "runtimeIsQuantized": True,
        "runtimeQuantizationMethod": "bitsandbytes",
        "quantizerClass": (
            "transformers.quantizers.quantizer_bnb_4bit.Bnb4BitHfQuantizer"
        ),
        "activeQuantizationConfigClass": (
            "transformers.utils.quantization_config.BitsAndBytesConfig"
        ),
        "outerQuantizationConfig": outer_quantization_config,
        "activeQuantizationConfig": {
            **outer_quantization_config,
            "llm_int8_skip_modules": gate._BNB_ACTIVE_SKIP_MODULES,
        },
        "expectedTargetModuleCount": len(projection_names),
        "materializedTargetModuleCount": len(projection_names),
        "targetModuleNamesSHA256": gate.canonical_sha256(projection_names),
        "materializedTargetModulesSHA256": gate.canonical_sha256(
            [
                {**representative_target, "moduleName": module_name}
                for module_name in projection_names
            ]
        ),
        "representativeMaterializedTarget": representative_target,
        "parameterPlacement": parameter_placement,
        "forwardKernelProbe": forward_probe,
    }
    model_unsigned = {
        "schemaVersion": gate.RUNTIME_MODEL_BINDING_SCHEMA,
        "baseModelID": config["baseModelID"],
        "baseModelRevision": config["baseModelRevision"],
        "baseModelIndexDigest": config["baseModelIndexDigest"],
        "baseModelIndexShardBindingSHA256": config[
            "baseModelIndexShardBindingSHA256"
        ],
        "baseModelArtifactDigest": config["baseModelArtifactDigest"],
        "baseModelTokenizerClosureSHA256": config[
            "baseModelTokenizerClosureSHA256"
        ],
        "baseModelGenerationConfigFile": config["baseModelGenerationConfigFile"],
        "runtimeSnapshotPath": config["baseModelRuntimeSnapshotPath"],
        "runtimeSnapshotVerificationSHA256": snapshot_digest,
        "runtimeLoadMaterialization": runtime_load_materialization,
        "localFilesOnly": True,
    }
    model_binding = {
        **model_unsigned,
        "runtimeModelBindingSHA256": gate.canonical_sha256(model_unsigned),
    }
    tokenizer_unsigned = {
        "schemaVersion": "lumen.runtime-tokenizer-binding/1.1.0",
        "baseModelID": config["baseModelID"],
        "baseModelRevision": config["baseModelRevision"],
        "runtimeSnapshotPath": config["baseModelRuntimeSnapshotPath"],
        "runtimeSnapshotVerificationSHA256": snapshot_digest,
    }
    tokenizer_binding = {
        **tokenizer_unsigned,
        "runtimeTokenizerBindingSHA256": gate.canonical_sha256(tokenizer_unsigned),
    }
    unsigned = {
        "schemaVersion": gate.SMOKE_SCHEMA,
        "baseModelTokenizerDigest": config["baseModelTokenizerDigest"],
        "baseModelTokenizerFiles": config["baseModelTokenizerFiles"],
        "baseModelTokenizerClosureSHA256": config[
            "baseModelTokenizerClosureSHA256"
        ],
        "baseModelGenerationConfigFile": config["baseModelGenerationConfigFile"],
        "baseModelTokenizerSnapshotPath": config[
            "baseModelTokenizerSnapshotPath"
        ],
        "baseModelTokenizerSnapshotVerification": config[
            "baseModelTokenizerSnapshotVerification"
        ],
        "baseModelRuntimeSnapshotPath": config["baseModelRuntimeSnapshotPath"],
        "baseModelRuntimeSnapshotVerification": config[
            "baseModelRuntimeSnapshotVerification"
        ],
        "runtimeModelBinding": model_binding,
        "runtimeTokenizerBinding": tokenizer_binding,
    }
    return {**unsigned, gate.SMOKE_HASH_FIELD: gate.canonical_sha256(unsigned)}


def test_gate_groups_identical_load_contracts_and_reverifies_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_root, configs = _write_configs(
        tmp_path,
        max_lengths={"fleet": 4096, "cortex": 4096},
        monkeypatch=monkeypatch,
    )
    calls: list[str] = []

    def invoke(path: Path) -> dict[str, Any]:
        calls.append(path.stem)
        return _smoke_for(configs[path.stem])

    monkeypatch.setattr(gate, "_invoke_train_sft_smoke", invoke)
    with pytest.raises(RuntimeError, match="Missing runtime-binding"):
        gate.verify_existing_report(run_root, ("fleet", "cortex"))
    report, reused = gate.run_gate(run_root, ("fleet", "cortex"))

    assert reused is False
    assert calls == [report["contracts"][0]["representativeAgent"]]
    assert report["distinctRuntimeLoadContractCount"] == 1
    assert report["contracts"][0]["agents"] == ["fleet", "cortex"]
    report_path = run_root / "training" / gate.REPORT_FILENAME
    assert report_path.stat().st_mode & 0o777 == 0o600
    unsigned = dict(report)
    declared = unsigned.pop(gate.REPORT_HASH_FIELD)
    assert declared == gate.canonical_sha256(unsigned)

    monkeypatch.setattr(
        gate,
        "_invoke_train_sft_smoke",
        lambda _path: pytest.fail("a verified report must not rerun the loader"),
    )
    reverified, reused = gate.run_gate(run_root, ("fleet", "cortex"))
    assert reused is True
    assert reverified == report
    assert gate.verify_existing_report(run_root, ("fleet", "cortex")) == report


def test_gate_runs_once_per_distinct_max_sequence_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_root, configs = _write_configs(
        tmp_path,
        max_lengths={"fleet": 4096, "cortex": 2048},
        monkeypatch=monkeypatch,
    )
    calls: list[str] = []

    def invoke(path: Path) -> dict[str, Any]:
        calls.append(path.stem)
        return _smoke_for(configs[path.stem])

    monkeypatch.setattr(gate, "_invoke_train_sft_smoke", invoke)
    report, reused = gate.run_gate(run_root, ("fleet", "cortex"))

    assert reused is False
    assert sorted(calls) == ["cortex", "fleet"]
    assert report["distinctRuntimeLoadContractCount"] == 2
    assert sorted(record["agents"] for record in report["contracts"]) == [
        ["cortex"],
        ["fleet"],
    ]


def test_gate_groups_on_every_consumed_environment_attestation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_root, configs = _write_configs(
        tmp_path,
        max_lengths={"fleet": 4096, "cortex": 4096},
        monkeypatch=monkeypatch,
    )
    configs["cortex"]["spaceConfigurationSHA256"] = "7" * 64
    cortex_path = run_root / "configs" / "cortex.json"
    cortex_path.write_text(
        json.dumps(configs["cortex"], sort_keys=True) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        gate,
        "_invoke_train_sft_smoke",
        lambda path: _smoke_for(configs[path.stem]),
    )

    report, reused = gate.run_gate(run_root, ("fleet", "cortex"))

    assert reused is False
    assert report["distinctRuntimeLoadContractCount"] == 2


def test_public_materialization_verifier_reconstructs_nested_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _run_root, configs = _write_configs(
        tmp_path,
        max_lengths={"cortex": 4096},
        monkeypatch=monkeypatch,
    )
    config = configs["cortex"]
    materialization = _smoke_for(config)["runtimeModelBinding"][
        "runtimeLoadMaterialization"
    ]
    bound_config = json.loads(_MODEL_CONFIG_PAYLOAD)

    verified = gate.verify_runtime_load_materialization_evidence(
        materialization,
        config,
        bound_config,
    )
    assert verified == materialization

    tampered_forward = json.loads(json.dumps(materialization))
    forward = tampered_forward["forwardKernelProbe"]
    forward["allFinite"] = False
    forward_unsigned = dict(forward)
    forward_unsigned.pop(gate.FORWARD_PROBE_HASH_FIELD)
    forward[gate.FORWARD_PROBE_HASH_FIELD] = gate.canonical_sha256(
        forward_unsigned
    )
    with pytest.raises(RuntimeError, match="forward probe drifted"):
        gate.verify_runtime_load_materialization_evidence(
            tampered_forward,
            config,
            bound_config,
        )

    tampered_inventory = json.loads(json.dumps(materialization))
    tampered_inventory["targetModuleNamesSHA256"] = "0" * 64
    with pytest.raises(RuntimeError, match="NF4 materialization drifted"):
        gate.verify_runtime_load_materialization_evidence(
            tampered_inventory,
            config,
            bound_config,
        )

    tampered_placement = json.loads(json.dumps(materialization))
    placement = tampered_placement["parameterPlacement"]
    placement["deviceTypeCounts"] = {"cuda": 309, "cpu": 1}
    placement_unsigned = dict(placement)
    placement_unsigned.pop(gate.PARAMETER_PLACEMENT_HASH_FIELD)
    placement[gate.PARAMETER_PLACEMENT_HASH_FIELD] = gate.canonical_sha256(
        placement_unsigned
    )
    with pytest.raises(RuntimeError, match="parameter placement drifted"):
        gate.verify_runtime_load_materialization_evidence(
            tampered_placement,
            config,
            bound_config,
        )


def test_pinned_qwen_runtime_inventory_accounts_for_tied_lm_head() -> None:
    projection_names = gate._expected_qwen_projection_names(28)
    runtime_parameter_names = gate._expected_qwen_parameter_names(
        28,
        tied_embeddings=True,
    )
    serialized_weight_names = gate._expected_qwen_parameter_names(
        28,
        tied_embeddings=False,
    )

    assert len(projection_names) == 196
    assert len(runtime_parameter_names) == 310
    assert len(serialized_weight_names) == 311
    assert "model.embed_tokens.weight" in runtime_parameter_names
    assert "lm_head.weight" not in runtime_parameter_names
    assert serialized_weight_names == sorted(
        {*runtime_parameter_names, "lm_head.weight"}
    )


@pytest.mark.parametrize(
    "nested_field",
    ("forwardKernelProbe", "parameterPlacement"),
)
def test_materialization_verifier_rejects_rehashed_nested_extra_keys(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    nested_field: str,
) -> None:
    _run_root, configs = _write_configs(
        tmp_path,
        max_lengths={"cortex": 4096},
        monkeypatch=monkeypatch,
    )
    config = configs["cortex"]
    materialization = _smoke_for(config)["runtimeModelBinding"][
        "runtimeLoadMaterialization"
    ]
    tampered = json.loads(json.dumps(materialization))
    nested = tampered[nested_field]
    nested["unboundExtension"] = "rehashed-but-untrusted"
    hash_field = (
        gate.FORWARD_PROBE_HASH_FIELD
        if nested_field == "forwardKernelProbe"
        else gate.PARAMETER_PLACEMENT_HASH_FIELD
    )
    unsigned = dict(nested)
    unsigned.pop(hash_field)
    nested[hash_field] = gate.canonical_sha256(unsigned)

    with pytest.raises(RuntimeError, match="drifted from its reconstructed contract"):
        gate.verify_runtime_load_materialization_evidence(
            tampered,
            config,
            json.loads(_MODEL_CONFIG_PAYLOAD),
        )


def test_materialization_verifier_rejects_extra_keys_and_reconstructed_digest_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _run_root, configs = _write_configs(
        tmp_path,
        max_lengths={"cortex": 4096},
        monkeypatch=monkeypatch,
    )
    config = configs["cortex"]
    materialization = _smoke_for(config)["runtimeModelBinding"][
        "runtimeLoadMaterialization"
    ]

    tampered_extra = json.loads(json.dumps(materialization))
    tampered_extra["unboundExtension"] = {"status": "passed"}
    with pytest.raises(RuntimeError, match="NF4 materialization drifted"):
        gate.verify_runtime_load_materialization_evidence(
            tampered_extra,
            config,
            json.loads(_MODEL_CONFIG_PAYLOAD),
        )

    tampered_digest = json.loads(json.dumps(materialization))
    tampered_digest["materializedTargetModulesSHA256"] = "6" * 64
    with pytest.raises(RuntimeError, match="NF4 materialization drifted"):
        gate.verify_runtime_load_materialization_evidence(
            tampered_digest,
            config,
            json.loads(_MODEL_CONFIG_PAYLOAD),
        )


def test_materialization_verifier_rejects_json_type_alias_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _run_root, configs = _write_configs(
        tmp_path,
        max_lengths={"cortex": 4096},
        monkeypatch=monkeypatch,
    )
    config = configs["cortex"]
    materialization = _smoke_for(config)["runtimeModelBinding"][
        "runtimeLoadMaterialization"
    ]
    tampered = json.loads(json.dumps(materialization))
    tampered["runtimeIsLoadedIn4Bit"] = 1

    with pytest.raises(RuntimeError, match="NF4 materialization drifted"):
        gate.verify_runtime_load_materialization_evidence(
            tampered,
            config,
            json.loads(_MODEL_CONFIG_PAYLOAD),
        )


@pytest.mark.parametrize(
    ("config_field", "tampered_value"),
    (
        ("num_hidden_layers", 28.0),
        ("tie_word_embeddings", False),
        ("vocab_size", 151_935),
    ),
)
def test_materialization_verifier_requires_the_pinned_qwen_topology(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    config_field: str,
    tampered_value: Any,
) -> None:
    _run_root, configs = _write_configs(
        tmp_path,
        max_lengths={"cortex": 4096},
        monkeypatch=monkeypatch,
    )
    config = configs["cortex"]
    materialization = _smoke_for(config)["runtimeModelBinding"][
        "runtimeLoadMaterialization"
    ]
    bound_config = json.loads(_MODEL_CONFIG_PAYLOAD)
    bound_config[config_field] = tampered_value

    with pytest.raises(RuntimeError, match="pinned Qwen3-1.7B topology"):
        gate.verify_runtime_load_materialization_evidence(
            materialization,
            config,
            bound_config,
        )


def test_materialization_verifier_requires_explicit_4bit_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _run_root, configs = _write_configs(
        tmp_path,
        max_lengths={"cortex": 4096},
        monkeypatch=monkeypatch,
    )
    config = configs["cortex"]
    materialization = _smoke_for(config)["runtimeModelBinding"][
        "runtimeLoadMaterialization"
    ]
    config["load_in_4bit"] = 1

    with pytest.raises(RuntimeError, match="precision/context contract"):
        gate.verify_runtime_load_materialization_evidence(
            materialization,
            config,
            json.loads(_MODEL_CONFIG_PAYLOAD),
        )


def test_gate_rejects_tampered_report_instead_of_replacing_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_root, configs = _write_configs(
        tmp_path,
        max_lengths={"cortex": 4096},
        monkeypatch=monkeypatch,
    )
    monkeypatch.setattr(
        gate,
        "_invoke_train_sft_smoke",
        lambda path: _smoke_for(configs[path.stem]),
    )
    gate.run_gate(run_root, ("cortex",))
    report_path = run_root / "training" / gate.REPORT_FILENAME
    tampered = json.loads(report_path.read_text(encoding="utf-8"))
    tampered["status"] = "failed"
    report_path.write_text(json.dumps(tampered) + "\n", encoding="utf-8")
    report_path.chmod(0o600)
    before = report_path.read_bytes()

    with pytest.raises(RuntimeError, match="self-hash"):
        gate.run_gate(run_root, ("cortex",))
    assert report_path.read_bytes() == before


def test_gate_rejects_a_prepared_config_change_on_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_root, configs = _write_configs(
        tmp_path,
        max_lengths={"cortex": 4096},
        monkeypatch=monkeypatch,
    )
    monkeypatch.setattr(
        gate,
        "_invoke_train_sft_smoke",
        lambda path: _smoke_for(configs[path.stem]),
    )
    gate.run_gate(run_root, ("cortex",))
    config_path = run_root / "configs" / "cortex.json"
    changed = json.loads(config_path.read_text(encoding="utf-8"))
    changed["max_seq_length"] = 2048
    config_path.write_text(json.dumps(changed, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="prepared config|prepared run|contract"):
        gate.run_gate(run_root, ("cortex",))


def test_aio_launcher_orders_real_smoke_after_prepare_only_before_converter_and_training() -> None:
    launcher = (
        REPO_ROOT / "scripts" / "ubuntu_train_lumen_adapters_aio.sh"
    ).read_text(encoding="utf-8")
    prepare_only = launcher.index('if [[ "$PREPARE_ONLY" == "1" ]]')
    smoke = launcher.index(
        "tools.fine_tuning.unsloth.runtime_binding_smoke_gate"
    )
    converter = launcher.index('CONVERTER_REPO=""')
    training = launcher.index('for agent in "${AGENTS[@]}"; do')

    assert prepare_only < smoke < converter < training
    invocation = launcher[smoke : launcher.index(")\n", smoke)]
    assert "--run-root" in invocation
    assert "--agents" in invocation
    assert ">" not in invocation
    assert "--runtime-binding-smoke" in gate._invoke_train_sft_smoke.__code__.co_consts
