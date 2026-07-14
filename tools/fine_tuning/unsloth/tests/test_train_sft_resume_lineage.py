from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.fine_tuning.unsloth import train_sft


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _fixture(tmp_path: Path) -> tuple[dict[str, object], Path, Path]:
    run_root = tmp_path / "run"
    dataset_dir = (
        run_root
        / "generated/fine_tuning/executor/experiments/internal_plus_public_optimized"
    )
    dataset_dir.mkdir(parents=True)
    dataset_files = {
        "train_sft.jsonl": '{"messages":[]}\n',
        "val_sft.jsonl": "",
        "train_dpo.jsonl": "",
        "val_dpo.jsonl": "",
    }
    for name, content in dataset_files.items():
        (dataset_dir / name).write_text(content, encoding="utf-8")

    config_path = run_root / "configs/executor.json"
    checkpoint_lineage_path = run_root / "checkpoint_lineage/executor.json"
    output_dir = run_root / "training/executor"
    adapter_dir = run_root / "models/lora_qwen3_bootstrap/executor"
    lane_hashes = {
        "trainSFT": "1" * 64,
        "validationSFT": "2" * 64,
        "trainDPO": "3" * 64,
        "validationDPO": "4" * 64,
    }
    attestation = {
        "laneHashes": lane_hashes,
        "trainingCorpusSHA256": "5" * 64,
        "effectiveTrainingConfigSHA256": "6" * 64,
        "trainingEnvironmentLockSHA256": "7" * 64,
    }
    config: dict[str, object] = {
        "agent": "executor",
        "baseModelID": "Qwen/Qwen3-1.7B",
        "base_model_name": "Qwen/Qwen3-1.7B",
        "baseModelRevision": "a" * 40,
        "baseModelIndexDigest": "8" * 64,
        "baseModelIndexShardBindingSHA256": "9" * 64,
        "baseModelArtifactDigest": "a" * 64,
        "baseModelWeightShards": [
            {"filename": "model.safetensors", "size": 1, "sha256": "b" * 64}
        ],
        "baseModelTokenizerDigest": "c" * 64,
        "variant": "internal_plus_public_optimized",
        "variantManifestSHA256": "d" * 64,
        "variantAttestation": attestation,
        "seed": 42,
        "trainingCodeSHA256": "e" * 64,
        "trainingDependencyLockSHA256": "f" * 64,
        "requirementsSHA256": "0" * 64,
        "resolvedTrainingEnvironment": {"schema": "synthetic"},
        "resolvedTrainingEnvironmentSHA256": "2" * 64,
        "spaceConfigurationSHA256": "1" * 64,
        "runtimeSourceKind": "huggingface_space",
        "runtimeSourceRevision": "b" * 40,
        "expectedRuntimeSourceRevision": "b" * 40,
        "observedRepositoryRevision": "b" * 40,
        "observedRuntimeRevision": None,
        "runtimeSourceBindingStatus": "operator_declared_unverified",
        "runtimeSourceBindingMethod": "huggingface_repository_head_supplemental",
        "dataset_dir": str(dataset_dir),
        "output_dir": str(output_dir),
        "adapter_output_dir": str(adapter_dir),
        "checkpointLineagePath": str(checkpoint_lineage_path),
    }
    agent_lineage = {
        "agent": "executor",
        "sourceVariantManifestSHA256": config["variantManifestSHA256"],
        "laneHashes": lane_hashes,
        "datasetFileSHA256": {
            name: train_sft._hash_file(dataset_dir / name) for name in sorted(dataset_files)
        },
        "trainingCorpusSHA256": attestation["trainingCorpusSHA256"],
        "controlledTrainingConfigSHA256": attestation[
            "effectiveTrainingConfigSHA256"
        ],
        "baseModelID": config["baseModelID"],
        "baseModelRevision": config["baseModelRevision"],
        "baseModelIndexDigest": config["baseModelIndexDigest"],
        "baseModelIndexShardBindingSHA256": config[
            "baseModelIndexShardBindingSHA256"
        ],
        "baseModelArtifactDigest": config["baseModelArtifactDigest"],
        "baseModelWeightShards": config["baseModelWeightShards"],
        "baseModelTokenizerDigest": config["baseModelTokenizerDigest"],
        "seed": 42,
        "trainingEnvironmentLockSHA256": attestation[
            "trainingEnvironmentLockSHA256"
        ],
        "configPath": str(config_path),
        "checkpointLineagePath": str(checkpoint_lineage_path),
        "checkpointRoot": str(output_dir),
        "outputDirectory": str(output_dir),
        "adapterOutputDirectory": str(adapter_dir),
    }
    run_payload = {
        "schema": train_sft.RUN_RESUME_LINEAGE_SCHEMA,
        "runID": "run-internal_plus_public_optimized",
        "datasetRepository": "user/dataset",
        "datasetRevision": "c" * 40,
        "datasetPath": "runs/run/fine_tuning",
        "localDatasetSnapshot": str(run_root / "generated/fine_tuning"),
        "selectedAgents": ["executor"],
        "experimentVariant": config["variant"],
        "seed": 42,
        "trainingCodeSHA256": config["trainingCodeSHA256"],
        "trainingDependencyLockSHA256": config[
            "trainingDependencyLockSHA256"
        ],
        "requirementsSHA256": config["requirementsSHA256"],
        "resolvedTrainingEnvironment": config[
            "resolvedTrainingEnvironment"
        ],
        "resolvedTrainingEnvironmentSHA256": config[
            "resolvedTrainingEnvironmentSHA256"
        ],
        "spaceConfigurationSHA256": config["spaceConfigurationSHA256"],
        "runtimeSourceKind": config["runtimeSourceKind"],
        "runtimeSourceRevision": config["runtimeSourceRevision"],
        "expectedRuntimeSourceRevision": config[
            "expectedRuntimeSourceRevision"
        ],
        "observedRepositoryRevision": config[
            "observedRepositoryRevision"
        ],
        "observedRuntimeRevision": config["observedRuntimeRevision"],
        "runtimeSourceBindingStatus": config[
            "runtimeSourceBindingStatus"
        ],
        "runtimeSourceBindingMethod": config[
            "runtimeSourceBindingMethod"
        ],
        "assistantOnlyLoss": False,
        "agents": [agent_lineage],
    }
    run_lineage = {
        **run_payload,
        "runResumeLineageSHA256": train_sft._canonical_sha256(run_payload),
    }
    config["runResumeLineage"] = run_lineage
    config["runResumeLineageSHA256"] = run_lineage["runResumeLineageSHA256"]
    _write_json(config_path, config)

    checkpoint_payload = {
        "schema": train_sft.CHECKPOINT_LINEAGE_SCHEMA,
        "agent": "executor",
        "runResumeLineageSHA256": run_lineage["runResumeLineageSHA256"],
        "configSHA256": train_sft._hash_file(config_path),
        "datasetFileSHA256": agent_lineage["datasetFileSHA256"],
        "laneHashes": lane_hashes,
        "resolvedTrainingEnvironmentSHA256": run_lineage[
            "resolvedTrainingEnvironmentSHA256"
        ],
        "spaceConfigurationSHA256": run_lineage[
            "spaceConfigurationSHA256"
        ],
        "runtimeSourceBinding": {
            field: run_lineage[field]
            for field in (
                "runtimeSourceKind",
                "runtimeSourceRevision",
                "expectedRuntimeSourceRevision",
                "observedRepositoryRevision",
                "observedRuntimeRevision",
                "runtimeSourceBindingStatus",
                "runtimeSourceBindingMethod",
            )
        },
        "checkpointRoot": str(output_dir),
        "outputDirectory": str(output_dir),
        "checkpoints": [],
    }
    checkpoint_record = {
        **checkpoint_payload,
        "checkpointLineageSHA256": train_sft._canonical_sha256(checkpoint_payload),
    }
    _write_json(checkpoint_lineage_path, checkpoint_record)
    return config, config_path, checkpoint_lineage_path


def test_fresh_run_accepts_empty_self_hashed_checkpoint_lineage(tmp_path: Path) -> None:
    config, config_path, lineage_path = _fixture(tmp_path)

    checkpoint, record = train_sft._validate_checkpoint_lineage(
        config,
        cfg_path=config_path,
        require_checkpoint=False,
    )

    assert checkpoint is None
    assert record == lineage_path


def test_resume_selects_latest_cryptographically_bound_checkpoint(tmp_path: Path) -> None:
    config, config_path, lineage_path = _fixture(tmp_path)
    output_dir = Path(str(config["output_dir"]))
    for step in (10, 20):
        checkpoint = output_dir / f"checkpoint-{step}"
        checkpoint.mkdir(parents=True)
        (checkpoint / "trainer_state.json").write_text(
            json.dumps({"global_step": step}), encoding="utf-8"
        )
        train_sft._record_checkpoint(lineage_path, checkpoint)

    checkpoint, record = train_sft._validate_checkpoint_lineage(
        config,
        cfg_path=config_path,
        require_checkpoint=True,
    )

    assert checkpoint == output_dir / "checkpoint-20"
    assert record == lineage_path


def test_resume_rejects_snapshot_drift_before_checkpoint_use(tmp_path: Path) -> None:
    config, config_path, _ = _fixture(tmp_path)
    (Path(str(config["dataset_dir"])) / "train_sft.jsonl").write_text(
        '{"messages":[{"role":"user","content":"changed"}]}\n',
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="Dataset snapshot drifted"):
        train_sft._validate_checkpoint_lineage(
            config,
            cfg_path=config_path,
            require_checkpoint=True,
        )


def test_resume_rejects_missing_checkpoint_lineage(tmp_path: Path) -> None:
    config, config_path, lineage_path = _fixture(tmp_path)
    lineage_path.unlink()

    with pytest.raises(RuntimeError, match="Missing required lineage manifest"):
        train_sft._validate_checkpoint_lineage(
            config,
            cfg_path=config_path,
            require_checkpoint=True,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("seed", 7),
        ("trainingCodeSHA256", "1" * 64),
        ("trainingDependencyLockSHA256", "2" * 64),
        ("resolvedTrainingEnvironmentSHA256", "4" * 64),
        ("spaceConfigurationSHA256", "3" * 64),
        ("expectedRuntimeSourceRevision", "d" * 40),
        ("observedRepositoryRevision", "e" * 40),
        ("runtimeSourceBindingStatus", "verified"),
        ("runtimeSourceBindingMethod", "self_declared"),
        ("baseModelRevision", "d" * 40),
        ("variantManifestSHA256", "3" * 64),
    ],
)
def test_resume_rejects_config_lineage_drift(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    config, config_path, _ = _fixture(tmp_path)
    config[field] = value

    with pytest.raises(RuntimeError, match="drifted"):
        train_sft._validate_checkpoint_lineage(
            config,
            cfg_path=config_path,
            require_checkpoint=True,
        )
