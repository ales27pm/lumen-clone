from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


AGENTS = ("cortex", "executor", "mouth", "mimicry", "rem", "fleet")
VARIANTS = (
    "internal_only",
    "internal_plus_public_baseline",
    "internal_plus_public_optimized",
)
DATASET_FILES = (
    "train_sft.jsonl",
    "val_sft.jsonl",
    "train_dpo.jsonl",
    "val_dpo.jsonl",
)
RUNTIME_SOURCE_FIELDS = (
    "runtimeSourceKind",
    "runtimeSourceRevision",
    "expectedRuntimeSourceRevision",
    "observedRepositoryRevision",
    "observedRuntimeRevision",
    "runtimeSourceBindingStatus",
    "runtimeSourceBindingMethod",
)
RUNTIME_CONFIG_FIELDS = {
    "trainingContainerImageDigest",
    "trainingContainerImageDigestSource",
    "trainingRuntimeImageBindingStatus",
    "trainingRuntimeImageBindingVerified",
    "trainingEnvironmentSHA256",
    "resolvedTrainingEnvironment",
    "resolvedTrainingEnvironmentSHA256",
    "resolvedTrainingEnvironmentScanAudit",
    "resolvedTrainingEnvironmentCacheAttestation",
    "spaceConfigurationSHA256",
    "zeroGPUSize",
    "zeroGPUDurationSeconds",
    "observedAccelerator",
    *RUNTIME_SOURCE_FIELDS,
}
NON_CONTROLLED_CONFIG_FIELDS = {
    "adapterExport",
    "adapter_gguf_output_path",
    "adapter_output_dir",
    "dataset_dir",
    "dpo_output_dir",
    "gguf_output_dir",
    "gguf_repo_id",
    "mergeExport",
    "output_dir",
    "variant",
    "variantAttestation",
    "variantManifestSHA256",
    *RUNTIME_CONFIG_FIELDS,
}


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Unable to read JSON object: {path}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected a JSON object: {path}")
    return value


def write_object(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise RuntimeError(f"Unable to read dataset: {path}") from exc
    for lineno, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Invalid JSON at {path}:{lineno}") from exc
        if not isinstance(value, dict):
            raise RuntimeError(f"Expected a JSON object at {path}:{lineno}")
        rows.append(value)
    return rows


def parse_agents(value: str) -> tuple[str, ...]:
    raw = value.split(",")
    if not raw or any(not item.strip() for item in raw):
        raise RuntimeError("Agent list must be a comma-separated list without empty entries")
    agents = tuple(item.strip().lower() for item in raw)
    unknown = sorted(set(agents) - set(AGENTS))
    if unknown:
        raise RuntimeError(f"Unsupported agents: {', '.join(unknown)}")
    if len(set(agents)) != len(agents):
        raise RuntimeError("Agent list must not contain duplicates")
    return agents


def require_variant(value: str) -> str:
    if value not in VARIANTS:
        raise RuntimeError(
            f"Unsupported experiment variant: {value}. Expected one of: {', '.join(VARIANTS)}"
        )
    return value


def require_container_digest(value: str) -> str:
    if re.fullmatch(r"sha256:[0-9a-f]{64}", value) is None:
        raise RuntimeError(
            "Container image digest must be sha256 followed by 64 lowercase hex characters"
        )
    return value


def git_head(root: Path) -> str:
    try:
        value = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("Unable to observe the repository Git revision") from exc
    if re.fullmatch(r"[0-9a-f]{40}", value) is None:
        raise RuntimeError("Repository HEAD is not an immutable Git commit SHA")
    return value


def local_runtime_source(root: Path) -> dict[str, Any]:
    revision = git_head(root)
    return {
        "runtimeSourceKind": "git",
        "runtimeSourceRevision": revision,
        "expectedRuntimeSourceRevision": revision,
        "observedRepositoryRevision": revision,
        "observedRuntimeRevision": revision,
        "runtimeSourceBindingStatus": "local_checkout_observed",
        "runtimeSourceBindingMethod": "git_head_plus_training_code_manifest",
    }


def validate_run_root(run_root: Path, *, allowed_parent: Path) -> Path:
    if run_root.is_symlink():
        raise RuntimeError(f"Run root must not be a symlink: {run_root}")
    resolved = run_root.expanduser().resolve()
    parent = allowed_parent.expanduser().resolve()
    if resolved == parent or parent not in resolved.parents:
        raise RuntimeError(
            f"Run root must be a child of the allowed output parent {parent}: {resolved}"
        )
    if resolved == Path("/") or len(resolved.parts) < 3:
        raise RuntimeError(f"Unsafe run root: {resolved}")
    return resolved


def _require_dataset_contract(
    manifest: Mapping[str, Any],
    key: str,
    records: Sequence[Mapping[str, Any]],
    *,
    manifest_path: Path,
) -> None:
    datasets = manifest.get("datasets")
    contract = datasets.get(key) if isinstance(datasets, Mapping) else None
    if not isinstance(contract, Mapping):
        raise RuntimeError(f"Variant manifest is missing datasets.{key}: {manifest_path}")
    if type(contract.get("count")) is not int or contract["count"] != len(records):
        raise RuntimeError(f"Dataset count mismatch for datasets.{key}: {manifest_path}")
    if contract.get("sha256") != canonical_sha256(list(records)):
        raise RuntimeError(f"Dataset digest mismatch for datasets.{key}: {manifest_path}")


def validate_variant(
    source_root: Path,
    *,
    agent: str,
    variant: str,
    seed: int,
    base_model_override: str,
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    agent_root = source_root / agent
    variant_root = agent_root / "experiments" / variant
    for filename in DATASET_FILES:
        if not (variant_root / filename).is_file():
            raise RuntimeError(f"Missing controlled dataset: {variant_root / filename}")
    manifest_path = variant_root / "variant_manifest.json"
    config_path = agent_root / "unsloth_config.json"
    manifest = read_object(manifest_path)
    config = read_object(config_path)
    if manifest.get("agent") != agent or manifest.get("variant") != variant:
        raise RuntimeError(f"Variant manifest identity mismatch: {manifest_path}")
    declared_manifest_digest = manifest.get("variantManifestSHA256")
    unsigned_manifest = dict(manifest)
    unsigned_manifest.pop("variantManifestSHA256", None)
    if (
        re.fullmatch(r"[0-9a-f]{64}", str(declared_manifest_digest or "")) is None
        or canonical_sha256(unsigned_manifest) != declared_manifest_digest
    ):
        raise RuntimeError(f"Variant manifest integrity check failed: {manifest_path}")

    lanes = {
        filename.removesuffix(".jsonl"): read_jsonl(variant_root / filename)
        for filename in DATASET_FILES
    }
    _require_dataset_contract(manifest, "trainSFT", lanes["train_sft"], manifest_path=manifest_path)
    _require_dataset_contract(
        manifest,
        "validationSFT",
        lanes["val_sft"],
        manifest_path=manifest_path,
    )
    datasets = manifest.get("datasets")
    if isinstance(datasets, Mapping) and (
        "trainDPO" in datasets or "validationDPO" in datasets
    ):
        _require_dataset_contract(
            manifest,
            "trainDPO",
            lanes["train_dpo"],
            manifest_path=manifest_path,
        )
        _require_dataset_contract(
            manifest,
            "validationDPO",
            lanes["val_dpo"],
            manifest_path=manifest_path,
        )
    else:
        _require_dataset_contract(
            manifest,
            "dpo",
            [*lanes["train_dpo"], *lanes["val_dpo"]],
            manifest_path=manifest_path,
        )
    training_corpus = [
        *lanes["train_sft"],
        *lanes["val_sft"],
        *lanes["train_dpo"],
        *lanes["val_dpo"],
    ]
    if manifest.get("trainingCorpusSHA256") != canonical_sha256(training_corpus):
        raise RuntimeError(f"Training-corpus digest mismatch: {manifest_path}")

    controlled = manifest.get("controlledTrainingConfig")
    if not isinstance(controlled, Mapping):
        raise RuntimeError(f"Variant manifest lacks controlledTrainingConfig: {manifest_path}")
    unexpected = set(config) - set(controlled) - NON_CONTROLLED_CONFIG_FIELDS
    if (
        manifest.get("trainingConfigSHA256") != canonical_sha256(dict(controlled))
        or any(config.get(key) != value for key, value in controlled.items())
        or unexpected
    ):
        detail = f" unexpected fields: {', '.join(sorted(unexpected))}" if unexpected else ""
        raise RuntimeError(f"Generated config drifted from the controlled variant:{detail} {config_path}")
    if config.get("agent") != agent:
        raise RuntimeError(f"Generated config agent mismatch: {config_path}")
    if manifest.get("seed") != seed:
        raise RuntimeError(f"Seed {seed} would break the controlled variant for {agent}")
    base_model = base_model_override or str(config.get("base_model_name") or "")
    if not base_model or manifest.get("baseModelID") != base_model:
        raise RuntimeError(f"Base model would break the controlled variant for {agent}")
    for split in (lanes["train_sft"], lanes["val_sft"]):
        for row in split:
            messages = row.get("messages")
            if not isinstance(messages, list):
                raise RuntimeError(f"Invalid messages array in {agent}/{variant}")
            assistant = next(
                (
                    message.get("content", "")
                    for message in messages
                    if isinstance(message, Mapping)
                    and message.get("role") == "assistant"
                ),
                "",
            )
            if not str(assistant).strip() or str(assistant).strip().lower() in {"null", "none"}:
                raise RuntimeError(f"Empty/null assistant output in {agent}/{variant}")
    return config, manifest, variant_root


def static_preflight(
    *,
    root: Path,
    dataset_source: Path,
    agents: Sequence[str],
    variant: str,
    seed: int,
    base_model_override: str,
    container_digest: str,
    run_root: Path,
    allowed_parent: Path,
    expected_run_id: str | None = None,
) -> dict[str, Any]:
    require_variant(variant)
    require_container_digest(container_digest)
    resolved_run_root = validate_run_root(run_root, allowed_parent=allowed_parent)
    if expected_run_id is not None:
        if (
            re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,255}", expected_run_id)
            is None
            or resolved_run_root != allowed_parent.expanduser().resolve() / expected_run_id
        ):
            raise RuntimeError("Run root must be the exact expected child of its allowed parent")
    if not dataset_source.is_dir():
        raise RuntimeError(f"Dataset source does not exist: {dataset_source}")
    runtime_manifest = read_object(dataset_source / "adapter_runtime_manifest.json")
    read_object(
        root / "generated" / "agent_manifest" / "AgentBehaviorManifest.json"
    )
    checked: list[dict[str, Any]] = []
    for agent in agents:
        _, manifest, _ = validate_variant(
            dataset_source,
            agent=agent,
            variant=variant,
            seed=seed,
            base_model_override=base_model_override,
        )
        checked.append(
            {
                "agent": agent,
                "variantManifestSHA256": manifest["variantManifestSHA256"],
                "trainingCorpusSHA256": manifest["trainingCorpusSHA256"],
            }
        )
    return {
        "schema": "lumen.ubuntu-training-static-preflight/1.0.0",
        "status": "static_ready",
        "trainingReady": False,
        "unchecked": ["python_environment", "cuda_runtime", "accelerator", "network"],
        "variant": variant,
        "agents": checked,
        "runRoot": str(run_root.resolve()),
        "adapterRepoID": runtime_manifest.get("adapterRepoID"),
    }


def _runtime_lineage(
    *,
    root: Path,
    source_config: Mapping[str, Any],
    container_digest: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    from tools.fine_tuning.unsloth.train_sft import (
        _training_environment,
        _training_runtime_lineage,
    )

    config = dict(source_config)
    config.update(local_runtime_source(root))
    config["trainingContainerImageDigest"] = container_digest
    config["trainingContainerImageDigestSource"] = "operator_declared"
    config["trainingRuntimeImageBindingStatus"] = "manual_validation_required"
    config["trainingRuntimeImageBindingVerified"] = False
    config["trainingEnvironmentSHA256"] = None
    config["resolvedTrainingEnvironment"] = None
    config["resolvedTrainingEnvironmentSHA256"] = None
    config["resolvedTrainingEnvironmentScanAudit"] = None
    config["spaceConfigurationSHA256"] = None
    config["zeroGPUSize"] = None
    config["zeroGPUDurationSeconds"] = None
    config["observedAccelerator"] = None
    lineage = _training_runtime_lineage(config, phase="sft")
    config.update(
        {
            key: lineage[key]
            for key in (
                "resolvedTrainingEnvironment",
                "resolvedTrainingEnvironmentSHA256",
                "resolvedTrainingEnvironmentScanAudit",
                "spaceConfigurationSHA256",
                "zeroGPUSize",
                "zeroGPUDurationSeconds",
                "observedAccelerator",
            )
        }
    )
    environment = _training_environment(config, runtime_lineage=lineage)
    return lineage, environment


def runtime_preflight(
    *,
    root: Path,
    dataset_source: Path,
    agent: str,
    variant: str,
    seed: int,
    base_model_override: str,
    container_digest: str,
) -> dict[str, Any]:
    config, _, _ = validate_variant(
        dataset_source,
        agent=agent,
        variant=variant,
        seed=seed,
        base_model_override=base_model_override,
    )
    lineage, environment = _runtime_lineage(
        root=root,
        source_config=config,
        container_digest=container_digest,
    )
    accelerator = lineage.get("observedAccelerator")
    if not isinstance(accelerator, Mapping) or accelerator.get("backend") != "cuda":
        raise RuntimeError("Runtime preflight did not observe a CUDA accelerator")
    return {
        "schema": "lumen.ubuntu-training-runtime-preflight/1.0.0",
        "status": "training_ready",
        "trainingReady": True,
        "pythonVersion": environment["environmentLock"]["pythonVersion"],
        "cudaVersion": environment["environmentLock"]["cudaVersion"],
        "observedAccelerator": accelerator,
        "resolvedTrainingEnvironmentSHA256": lineage[
            "resolvedTrainingEnvironmentSHA256"
        ],
        "trainingEnvironmentSHA256": environment["trainingEnvironmentSHA256"],
        "runtimeSourceRevision": lineage["runtimeSourceRevision"],
    }


def _training_attestation(
    config: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    controlled = manifest.get("controlledTrainingConfig")
    if not isinstance(controlled, Mapping):
        raise RuntimeError("Variant manifest lacks controlled training config")
    effective = {key: config.get(key) for key in controlled}
    if effective != dict(controlled):
        raise RuntimeError("Effective training config drifted from the controlled variant")
    datasets = manifest.get("datasets")
    if not isinstance(datasets, Mapping):
        raise RuntimeError("Variant manifest lacks dataset lineage")
    return {
        "schema": "lumen.training-variant-attestation/1.0.0",
        "variant": manifest["variant"],
        "variantManifestSHA256": manifest["variantManifestSHA256"],
        "trainingCorpusSHA256": manifest["trainingCorpusSHA256"],
        "laneHashes": {
            name: contract["sha256"]
            for name, contract in sorted(datasets.items())
            if isinstance(contract, Mapping) and isinstance(contract.get("sha256"), str)
        },
        "effectiveTrainingConfigSHA256": canonical_sha256(effective),
        "baseModelRevision": manifest["baseModelRevision"],
        "baseModelIndexDigest": manifest["baseModelIndexDigest"],
        "baseModelIndexReferencedShardNames": manifest[
            "baseModelIndexReferencedShardNames"
        ],
        "baseModelIndexShardBindingSHA256": manifest[
            "baseModelIndexShardBindingSHA256"
        ],
        "baseModelArtifactDigest": manifest["baseModelArtifactDigest"],
        "baseModelWeightShards": manifest["baseModelWeightShards"],
        "baseModelTokenizerDigest": manifest["baseModelTokenizerDigest"],
        "trainingEnvironmentLockSHA256": manifest[
            "trainingEnvironmentLockSHA256"
        ],
        "trainingEnvironmentSHA256": config["trainingEnvironmentSHA256"],
        "trainingCodeSHA256": config["trainingCodeSHA256"],
        "trainingDependencyLockSHA256": config[
            "trainingDependencyLockSHA256"
        ],
        "requirementsSHA256": config["requirementsSHA256"],
        "runtimeImageBindingStatus": config[
            "trainingRuntimeImageBindingStatus"
        ],
        "runtimeImageBindingVerified": config[
            "trainingRuntimeImageBindingVerified"
        ],
        **{field: config[field] for field in RUNTIME_SOURCE_FIELDS},
    }


def prepare_run(
    *,
    root: Path,
    dataset_source: Path,
    run_root: Path,
    agents: Sequence[str],
    variant: str,
    seed: int,
    base_model_override: str,
    container_digest: str,
) -> dict[str, Any]:
    if run_root.exists():
        raise RuntimeError(f"Run root already exists: {run_root}")
    source_config, _, _ = validate_variant(
        dataset_source,
        agent=agents[0],
        variant=variant,
        seed=seed,
        base_model_override=base_model_override,
    )
    runtime_lineage, runtime_environment = _runtime_lineage(
        root=root,
        source_config=source_config,
        container_digest=container_digest,
    )
    run_root.mkdir(parents=True)
    snapshot_root = run_root / "generated" / "fine_tuning"
    shutil.copytree(dataset_source, snapshot_root)
    behavior_manifest_source = (
        root / "generated" / "agent_manifest" / "AgentBehaviorManifest.json"
    )
    if not behavior_manifest_source.is_file():
        raise RuntimeError(
            f"Missing frozen behavior manifest: {behavior_manifest_source}"
        )
    behavior_manifest_snapshot = (
        run_root
        / "generated"
        / "agent_manifest"
        / "AgentBehaviorManifest.json"
    )
    behavior_manifest_snapshot.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(behavior_manifest_source, behavior_manifest_snapshot)
    for directory in (
        run_root / "configs",
        run_root / "logs",
        run_root / "training",
        run_root / "models" / "lora_qwen3_bootstrap",
        run_root / "models" / "lora_qwen3_dpo",
        run_root / "models" / "lora_qwen3_gguf",
        run_root / "evaluation",
    ):
        directory.mkdir(parents=True, exist_ok=True)

    runtime_manifest = read_object(snapshot_root / "adapter_runtime_manifest.json")
    base_by_agent = {
        item["agent"]: item.get("baseModelID")
        for item in runtime_manifest.get("adapters", [])
        if isinstance(item, Mapping) and isinstance(item.get("agent"), str)
    }
    prepared: list[dict[str, Any]] = []
    runtime_source = local_runtime_source(root)
    for agent in agents:
        config, manifest, variant_root = validate_variant(
            snapshot_root,
            agent=agent,
            variant=variant,
            seed=seed,
            base_model_override=base_model_override,
        )
        for field in (
            "trainingEnvironmentLock",
            "trainingCodeManifest",
            "trainingCodeSHA256",
            "trainingDependencyLock",
            "trainingDependencyLockSHA256",
            "requirementsSHA256",
        ):
            if config.get(field) != source_config.get(field):
                raise RuntimeError(f"Shared training lineage differs for {agent}: {field}")
        base_model = (
            base_model_override
            or str(base_by_agent.get(agent) or "")
            or str(config.get("base_model_name") or "")
        )
        adapter_dir = run_root / "models" / "lora_qwen3_bootstrap" / agent
        preference_adapter_dir = run_root / "models" / "lora_qwen3_dpo" / agent
        training_dir = run_root / "training" / agent
        gguf_path = run_root / "models" / "lora_qwen3_gguf" / f"lumen-{agent}-lora.gguf"
        config["base_model_name"] = base_model
        config["baseModelID"] = base_model
        config["trainingContainerImageDigest"] = container_digest
        config["trainingContainerImageDigestSource"] = "operator_declared"
        config["trainingRuntimeImageBindingStatus"] = "manual_validation_required"
        config["trainingRuntimeImageBindingVerified"] = False
        config.update(runtime_source)
        for key in (
            "resolvedTrainingEnvironment",
            "resolvedTrainingEnvironmentSHA256",
            "resolvedTrainingEnvironmentScanAudit",
            "spaceConfigurationSHA256",
            "zeroGPUSize",
            "zeroGPUDurationSeconds",
            "observedAccelerator",
        ):
            config[key] = runtime_lineage[key]
        config["trainingEnvironmentSHA256"] = None
        from tools.fine_tuning.unsloth.train_sft import _training_environment

        environment = _training_environment(config, runtime_lineage=runtime_lineage)
        if environment["trainingEnvironmentSHA256"] != runtime_environment[
            "trainingEnvironmentSHA256"
        ]:
            raise RuntimeError(f"Training environment differs across agent configs: {agent}")
        config["trainingEnvironmentSHA256"] = environment[
            "trainingEnvironmentSHA256"
        ]
        config["dataset_dir"] = str(variant_root)
        config["variant"] = variant
        config["variantManifestSHA256"] = manifest["variantManifestSHA256"]
        config["output_dir"] = str(training_dir)
        config["adapter_output_dir"] = str(adapter_dir)
        config["dpo_output_dir"] = str(preference_adapter_dir)
        config["adapter_gguf_output_path"] = str(gguf_path)
        config["gguf_output_dir"] = str(
            run_root
            / "models"
            / "gguf_release_bake_qwen3_bootstrap"
            / f"{agent}_merged_gguf"
        )
        config["seed"] = seed
        config["merge_adapters_by_default"] = False
        config["release_bake_enabled_by_default"] = False
        export = config.setdefault("adapterExport", {})
        if not isinstance(export, dict):
            raise RuntimeError(f"adapterExport must be an object for {agent}")
        export.update(
            {
                "trainBaseModelWeights": False,
                "mergeAdaptersByDefault": False,
                "adapterArtifact": str(adapter_dir),
                "adapterDirectory": str(adapter_dir),
                "adapterGGUFArtifact": str(gguf_path),
            }
        )
        config["variantAttestation"] = _training_attestation(config, manifest)
        config_path = run_root / "configs" / f"{agent}.json"
        write_object(config_path, config)
        prepared.append(
            {
                "agent": agent,
                "config": str(config_path),
                "configSHA256": file_sha256(config_path),
                "datasetDir": str(variant_root),
                "variantManifestSHA256": manifest["variantManifestSHA256"],
                "sftAdapterDir": str(adapter_dir),
                "sftFinalizedVariantManifest": str(
                    training_dir / "finalized_variant_manifest.json"
                ),
                "preferenceTrainer": config.get("preference_trainer", "dpo"),
                "preferenceAdapterDir": str(preference_adapter_dir),
                "preferenceFinalizedVariantManifest": str(
                    training_dir / "dpo" / "finalized_variant_manifest.json"
                ),
                "adapterGGUF": str(gguf_path),
            }
        )
    run_manifest: dict[str, Any] = {
        "schema": "lumen.ubuntu-training-run/2.0.0",
        "runID": run_root.name,
        "runRoot": str(run_root),
        "adapterFirst": True,
        "trainBaseModelWeights": False,
        "freshRun": True,
        "variant": variant,
        "seed": seed,
        "containerImageDigest": container_digest,
        "sourceDatasetRoot": str(dataset_source.resolve()),
        "snapshotDatasetRoot": str(snapshot_root),
        "adapterRepoID": runtime_manifest.get("adapterRepoID"),
        "adapterRuntimeManifestFileSHA256": file_sha256(
            snapshot_root / "adapter_runtime_manifest.json"
        ),
        "behaviorManifest": str(behavior_manifest_snapshot),
        "behaviorManifestFileSHA256": file_sha256(behavior_manifest_snapshot),
        "trainingEnvironment": runtime_environment,
        **runtime_source,
        "agents": prepared,
    }
    run_manifest["runManifestSHA256"] = canonical_sha256(run_manifest)
    write_object(run_root / "aio_run_manifest.json", run_manifest)
    write_object(run_root / "training_environment.json", runtime_environment)
    return run_manifest


def _verified_run_manifest(run_root: Path) -> dict[str, Any]:
    manifest_path = run_root / "aio_run_manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise RuntimeError(f"Missing regular Lumen run manifest: {manifest_path}")
    manifest = read_object(manifest_path)
    declared = manifest.get("runManifestSHA256")
    unsigned = dict(manifest)
    unsigned.pop("runManifestSHA256", None)
    if canonical_sha256(unsigned) != declared:
        raise RuntimeError("Prepared run manifest integrity check failed")
    manifest_agents = manifest.get("agents")
    if (
        manifest.get("schema") != "lumen.ubuntu-training-run/2.0.0"
        or manifest.get("adapterFirst") is not True
        or manifest.get("trainBaseModelWeights") is not False
        or manifest.get("runID") != run_root.name
        or manifest.get("runRoot") != str(run_root)
        or not isinstance(manifest_agents, list)
        or not manifest_agents
        or any(
            not isinstance(item, Mapping) or item.get("agent") not in AGENTS
            for item in manifest_agents
        )
        or len({item["agent"] for item in manifest_agents}) != len(manifest_agents)
        or re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9._-]{0,255}",
            str(manifest.get("runID") or ""),
        )
        is None
    ):
        raise RuntimeError("Prepared run manifest does not own this exact run root")
    return manifest


def verify_owned_run(run_root: Path, *, variant: str) -> dict[str, Any]:
    manifest = _verified_run_manifest(run_root)
    if manifest.get("variant") != variant:
        raise RuntimeError("Existing run directory belongs to another experiment variant")
    return {
        "status": "owned_run",
        "runID": manifest["runID"],
        "variant": variant,
        "runManifestSHA256": manifest["runManifestSHA256"],
    }


def _reject_managed_symlinks(run_root: Path) -> None:
    for name in (
        "configs",
        "logs",
        "training",
        "models",
        "evaluation",
        "generated",
    ):
        managed = run_root / name
        if managed.is_symlink() or (managed.exists() and not managed.is_dir()):
            raise RuntimeError(f"Managed run path is not a regular directory: {managed}")
        if managed.is_dir():
            for entry in managed.rglob("*"):
                if entry.is_symlink():
                    raise RuntimeError(f"Managed run path contains a symlink: {entry}")


def _expected_agent_paths(run_root: Path, agent: str) -> dict[str, Path]:
    return {
        "config": run_root / "configs" / f"{agent}.json",
        "dataset_dir": (
            run_root
            / "generated"
            / "fine_tuning"
            / agent
            / "experiments"
        ),
        "output_dir": run_root / "training" / agent,
        "adapter_output_dir": run_root / "models" / "lora_qwen3_bootstrap" / agent,
        "dpo_output_dir": run_root / "models" / "lora_qwen3_dpo" / agent,
        "adapter_gguf_output_path": (
            run_root / "models" / "lora_qwen3_gguf" / f"lumen-{agent}-lora.gguf"
        ),
        "gguf_output_dir": (
            run_root
            / "models"
            / "gguf_release_bake_qwen3_bootstrap"
            / f"{agent}_merged_gguf"
        ),
    }


def validate_prepared_runtime(
    *,
    root: Path,
    run_root: Path,
    agents: Sequence[str],
    variant: str,
    container_digest: str,
) -> dict[str, Any]:
    manifest = _verified_run_manifest(run_root)
    _reject_managed_symlinks(run_root)
    manifest_agents = manifest.get("agents")
    if (
        manifest.get("variant") != variant
        or manifest.get("containerImageDigest") != container_digest
        or not isinstance(manifest_agents, list)
        or any(not isinstance(item, Mapping) for item in manifest_agents)
        or [item.get("agent") for item in manifest_agents] != list(agents)
    ):
        raise RuntimeError("Resume request does not match the prepared run manifest")
    prepared_by_agent = {str(item["agent"]): item for item in manifest_agents}
    current_runtime_source = local_runtime_source(root)
    if any(
        manifest.get(field) != current_runtime_source[field]
        for field in RUNTIME_SOURCE_FIELDS
    ):
        raise RuntimeError("Resume source revision drifted from the prepared run")
    snapshot_root = run_root / "generated" / "fine_tuning"
    runtime_manifest_path = snapshot_root / "adapter_runtime_manifest.json"
    training_environment_path = run_root / "training_environment.json"
    if (
        runtime_manifest_path.is_symlink()
        or not runtime_manifest_path.is_file()
        or file_sha256(runtime_manifest_path)
        != manifest.get("adapterRuntimeManifestFileSHA256")
        or read_object(runtime_manifest_path).get("adapterRepoID")
        != manifest.get("adapterRepoID")
    ):
        raise RuntimeError("Prepared adapter runtime manifest drifted")
    if (
        training_environment_path.is_symlink()
        or not training_environment_path.is_file()
        or read_object(training_environment_path) != manifest.get("trainingEnvironment")
    ):
        raise RuntimeError("Prepared training environment record drifted")
    seed = manifest.get("seed")
    if type(seed) is not int:
        raise RuntimeError("Prepared run manifest has an invalid seed")
    for agent in agents:
        paths = _expected_agent_paths(run_root, agent)
        config_path = paths["config"]
        prepared_entry = prepared_by_agent[agent]
        if (
            config_path.is_symlink()
            or not config_path.is_file()
            or prepared_entry.get("config") != str(config_path)
            or prepared_entry.get("configSHA256") != file_sha256(config_path)
        ):
            raise RuntimeError(f"Prepared config integrity check failed for {agent}")
        prepared_config = read_object(config_path)
        _, pending_manifest, variant_root = validate_variant(
            snapshot_root,
            agent=agent,
            variant=variant,
            seed=seed,
            base_model_override=str(prepared_config.get("base_model_name") or ""),
        )
        expected_paths = {
            **paths,
            "dataset_dir": variant_root,
        }
        path_drift = [
            field
            for field, expected in expected_paths.items()
            if field != "config"
            and prepared_config.get(field) != str(expected)
        ]
        export = prepared_config.get("adapterExport")
        if not isinstance(export, Mapping):
            path_drift.append("adapterExport")
        elif (
            export.get("adapterArtifact") != str(paths["adapter_output_dir"])
            or export.get("adapterDirectory") != str(paths["adapter_output_dir"])
            or export.get("adapterGGUFArtifact")
            != str(paths["adapter_gguf_output_path"])
            or export.get("trainBaseModelWeights") is not False
            or export.get("mergeAdaptersByDefault") is not False
        ):
            path_drift.append("adapterExport")
        if (
            prepared_config.get("dataset_dir") != str(variant_root)
            or prepared_config.get("variantManifestSHA256")
            != pending_manifest.get("variantManifestSHA256")
            or prepared_config.get("trainingContainerImageDigest")
            != container_digest
            or not isinstance(manifest.get("trainingEnvironment"), Mapping)
            or prepared_config.get("trainingEnvironmentSHA256")
            != manifest["trainingEnvironment"].get("trainingEnvironmentSHA256")
            or any(
                prepared_config.get(field) != current_runtime_source[field]
                for field in RUNTIME_SOURCE_FIELDS
            )
            or path_drift
        ):
            detail = f": {', '.join(path_drift)}" if path_drift else ""
            raise RuntimeError(
                f"Prepared config or dataset snapshot drifted for {agent}{detail}"
            )
    config = read_object(run_root / "configs" / f"{agents[0]}.json")
    lineage, environment = _runtime_lineage(
        root=root,
        source_config=config,
        container_digest=container_digest,
    )
    if environment["trainingEnvironmentSHA256"] != config.get(
        "trainingEnvironmentSHA256"
    ):
        raise RuntimeError("Current runtime drifted from the prepared training environment")
    return {
        "status": "resume_ready",
        "trainingEnvironmentSHA256": environment["trainingEnvironmentSHA256"],
        "observedAccelerator": lineage["observedAccelerator"],
    }


def verify_gguf(run_root: Path, agent: str) -> dict[str, Any]:
    summary = read_object(run_root / "aio_summary.json")
    expected_summary_sha = summary.get("summarySHA256")
    unsigned = dict(summary)
    unsigned.pop("summarySHA256", None)
    if canonical_sha256(unsigned) != expected_summary_sha:
        raise RuntimeError("Existing Ubuntu training summary integrity check failed")
    agent_summary = (summary.get("agents") or {}).get(agent)
    if not isinstance(agent_summary, Mapping):
        raise RuntimeError(f"Existing summary lacks agent {agent}")
    path = run_root / "models" / "lora_qwen3_gguf" / f"lumen-{agent}-lora.gguf"
    expected_digest = agent_summary.get("adapterGGUFSHA256")
    expected_size = agent_summary.get("adapterGGUFSizeBytes")
    if (
        not path.is_file()
        or type(expected_size) is not int
        or expected_size <= 0
        or path.stat().st_size != expected_size
        or re.fullmatch(r"[0-9a-f]{64}", str(expected_digest or "")) is None
        or file_sha256(path) != expected_digest
    ):
        raise RuntimeError(f"Existing GGUF does not match the completed summary: {path}")
    return {
        "agent": agent,
        "adapterGGUF": str(path),
        "adapterGGUFSHA256": expected_digest,
        "adapterGGUFSizeBytes": expected_size,
    }


def _verify_manifest_integrity(path: Path) -> dict[str, Any]:
    value = read_object(path)
    digest = value.get("variantManifestSHA256")
    unsigned = dict(value)
    unsigned.pop("variantManifestSHA256", None)
    if (
        re.fullmatch(r"[0-9a-f]{64}", str(digest or "")) is None
        or canonical_sha256(unsigned) != digest
    ):
        raise RuntimeError(f"Finalized variant manifest integrity check failed: {path}")
    return value


def _verify_training_report(
    path: Path,
    *,
    phase: str,
    expected: Mapping[str, Any],
) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"Missing regular {phase} training report: {path}")
    report = read_object(path)
    drifted = [
        field for field, value in expected.items() if report.get(field) != value
    ]
    if (
        drifted
        or not isinstance(report.get("metrics"), Mapping)
        or not isinstance(report.get("evaluation_metrics"), Mapping)
        or type(report.get("train_records")) is not int
        or report["train_records"] <= 0
        or type(report.get("val_records")) is not int
        or report["val_records"] <= 0
    ):
        detail = f": {', '.join(drifted)}" if drifted else ""
        raise RuntimeError(f"{phase} training report failed lineage validation{detail}")
    return report


def _verify_finalized_variant_binding(
    config: Mapping[str, Any],
    finalized: Mapping[str, Any],
    *,
    agent: str,
    phase: str,
    expected_training_code_sha256: Any,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    from tools.fine_tuning.unsloth.train_dpo import (
        _shared_finalized_variant_validator,
    )

    validator = _shared_finalized_variant_validator()
    if not validator(
        finalized,
        agent=agent,
        expected_variant=str(config.get("variant") or ""),
        require_trained_artifact=True,
    ):
        raise RuntimeError(f"{phase} finalized variant manifest is structurally invalid")

    attestation = config.get("variantAttestation")
    artifact = finalized.get("artifact")
    datasets = finalized.get("datasets")
    if (
        not isinstance(attestation, Mapping)
        or not isinstance(artifact, Mapping)
        or not isinstance(datasets, Mapping)
    ):
        raise RuntimeError(f"{phase} finalized manifest lacks controlled lineage")

    lane_hashes = {
        name: contract.get("sha256")
        for name, contract in sorted(datasets.items())
        if isinstance(contract, Mapping) and isinstance(contract.get("sha256"), str)
    }
    expected_static = {
        "agent": agent,
        "variant": config.get("variant"),
        "sourceVariantManifestSHA256": config.get("variantManifestSHA256"),
        "baseModelID": config.get("baseModelID", config.get("base_model_name")),
        "trainingCorpusSHA256": attestation.get("trainingCorpusSHA256"),
        "trainingConfigSHA256": attestation.get(
            "effectiveTrainingConfigSHA256"
        ),
        "baseModelRevision": attestation.get("baseModelRevision"),
        "baseModelIndexDigest": attestation.get("baseModelIndexDigest"),
        "baseModelIndexReferencedShardNames": attestation.get(
            "baseModelIndexReferencedShardNames"
        ),
        "baseModelIndexShardBindingSHA256": attestation.get(
            "baseModelIndexShardBindingSHA256"
        ),
        "baseModelArtifactDigest": attestation.get("baseModelArtifactDigest"),
        "baseModelWeightShards": attestation.get("baseModelWeightShards"),
        "baseModelTokenizerDigest": attestation.get("baseModelTokenizerDigest"),
        "trainingEnvironmentLockSHA256": attestation.get(
            "trainingEnvironmentLockSHA256"
        ),
    }
    drifted = [
        field
        for field, expected in expected_static.items()
        if finalized.get(field) != expected
    ]
    if drifted:
        raise RuntimeError(
            f"{phase} finalized manifest drifted from controlled lineage: "
            + ", ".join(drifted)
        )
    if lane_hashes != attestation.get("laneHashes"):
        raise RuntimeError(f"{phase} finalized dataset lane hashes drifted")

    expected_runtime = {
        "trainingCodeSHA256": expected_training_code_sha256,
        "trainingDependencyLockSHA256": config.get(
            "trainingDependencyLockSHA256"
        ),
        "requirementsSHA256": config.get("requirementsSHA256"),
        "resolvedTrainingEnvironment": config.get("resolvedTrainingEnvironment"),
        "resolvedTrainingEnvironmentSHA256": config.get(
            "resolvedTrainingEnvironmentSHA256"
        ),
        "spaceConfigurationSHA256": config.get("spaceConfigurationSHA256"),
        "zeroGPUSize": config.get("zeroGPUSize"),
        "zeroGPUDurationSeconds": config.get("zeroGPUDurationSeconds"),
        "observedAccelerator": config.get("observedAccelerator"),
        **{field: config.get(field) for field in RUNTIME_SOURCE_FIELDS},
    }
    runtime_drifted = [
        field
        for field, expected in expected_runtime.items()
        if finalized.get(field) != expected
    ]
    if runtime_drifted:
        raise RuntimeError(
            f"{phase} finalized runtime lineage drifted: "
            + ", ".join(runtime_drifted)
        )

    environment = finalized.get("trainingEnvironment")
    if not isinstance(environment, Mapping):
        raise RuntimeError(f"{phase} finalized manifest lacks training environment")
    declared_environment_sha = finalized.get("trainingEnvironmentSHA256")
    if (
        canonical_sha256(dict(environment)) != declared_environment_sha
        or environment.get("containerImageDigest")
        != config.get("trainingContainerImageDigest")
        or environment.get("environmentLock") != config.get("trainingEnvironmentLock")
        or environment.get("effectiveSeed") != config.get("seed")
        or any(
            environment.get(field) != expected
            for field, expected in expected_runtime.items()
            if field
            in {
                "trainingCodeSHA256",
                "trainingDependencyLockSHA256",
                "requirementsSHA256",
                "resolvedTrainingEnvironment",
                "resolvedTrainingEnvironmentSHA256",
                "zeroGPUSize",
                "zeroGPUDurationSeconds",
                "observedAccelerator",
            }
        )
    ):
        raise RuntimeError(f"{phase} training environment binding failed")

    artifact_expected = {
        "adapterManifestSHA256": artifact.get("adapterSHA256"),
        "effectiveSeed": config.get("seed"),
        "trainingCodeSHA256": expected_training_code_sha256,
        "trainingDependencyLockSHA256": config.get(
            "trainingDependencyLockSHA256"
        ),
        "requirementsSHA256": config.get("requirementsSHA256"),
        "resolvedTrainingEnvironmentSHA256": config.get(
            "resolvedTrainingEnvironmentSHA256"
        ),
        "spaceConfigurationSHA256": config.get("spaceConfigurationSHA256"),
        "zeroGPUSize": config.get("zeroGPUSize"),
        "zeroGPUDurationSeconds": config.get("zeroGPUDurationSeconds"),
        "observedAccelerator": config.get("observedAccelerator"),
        **{field: config.get(field) for field in RUNTIME_SOURCE_FIELDS},
    }
    artifact_drifted = [
        field
        for field, expected in artifact_expected.items()
        if artifact.get(field) != expected
    ]
    if artifact_drifted:
        raise RuntimeError(
            f"{phase} adapter artifact lineage drifted: "
            + ", ".join(artifact_drifted)
        )
    return artifact, attestation


def verify_sft(run_root: Path, agent: str) -> dict[str, Any]:
    from tools.fine_tuning.unsloth.adapter_artifact import verify_adapter_artifact

    config = read_object(run_root / "configs" / f"{agent}.json")
    adapter_dir = run_root / "models" / "lora_qwen3_bootstrap" / agent
    finalized_path = run_root / "training" / agent / "finalized_variant_manifest.json"
    report_path = run_root / "training" / agent / "training_report.json"
    finalized = _verify_manifest_integrity(finalized_path)
    artifact, attestation = _verify_finalized_variant_binding(
        config,
        finalized,
        agent=agent,
        phase="SFT",
        expected_training_code_sha256=config.get("trainingCodeSHA256"),
    )
    if (
        artifact.get("status") != "trained"
        or artifact.get("trainingPhase") != "sft"
        or artifact.get("parentSFTAdapterSHA256") is not None
        or artifact.get("referenceSFTAdapterSHA256") is not None
        or artifact.get("preferenceTrainer") is not None
    ):
        raise RuntimeError(f"SFT identity or lineage mismatch: {finalized_path}")
    if finalized.get("trainingEnvironmentSHA256") != attestation.get(
        "trainingEnvironmentSHA256"
    ):
        raise RuntimeError("SFT training environment drifted from its attestation")
    adapter_manifest = verify_adapter_artifact(
        adapter_dir,
        expected_adapter_sha256=str(artifact.get("adapterSHA256") or ""),
        expected_training_phase="sft",
    )
    _verify_training_report(
        report_path,
        phase="SFT",
        expected={
            "schema": "lumen.train_sft.manifest/1.0.0",
            "agent": agent,
            "trainingPhase": "sft",
            "seed": config.get("seed"),
            "config_sha256": file_sha256(
                run_root / "configs" / f"{agent}.json"
            ),
            "output_dir": str(run_root / "training" / agent),
            "adapter_output_dir": str(adapter_dir),
            "adapterSHA256": adapter_manifest["adapterSHA256"],
            "finalizedVariantManifestSHA256": finalized[
                "variantManifestSHA256"
            ],
            "trainingEnvironmentSHA256": finalized[
                "trainingEnvironmentSHA256"
            ],
        },
    )
    return {
        "phase": "sft",
        "adapterSHA256": adapter_manifest["adapterSHA256"],
        "finalizedVariantManifestSHA256": finalized["variantManifestSHA256"],
        "report": str(report_path),
    }


def verify_preference(run_root: Path, agent: str) -> dict[str, Any]:
    from tools.fine_tuning.unsloth.adapter_artifact import verify_adapter_artifact

    config = read_object(run_root / "configs" / f"{agent}.json")
    sft = verify_sft(run_root, agent)
    adapter_dir = run_root / "models" / "lora_qwen3_dpo" / agent
    finalized_path = run_root / "training" / agent / "dpo" / "finalized_variant_manifest.json"
    report_path = run_root / "training" / agent / "dpo" / "dpo_report.json"
    finalized = _verify_manifest_integrity(finalized_path)
    trainer = str(config.get("preference_trainer", "dpo")).lower()
    phase_digests = config.get("trainingCodeSHA256ByPhase")
    if not isinstance(phase_digests, Mapping):
        raise RuntimeError(f"Preference manifest lacks lineage: {finalized_path}")
    artifact, _ = _verify_finalized_variant_binding(
        config,
        finalized,
        agent=agent,
        phase=trainer.upper(),
        expected_training_code_sha256=phase_digests.get(trainer),
    )
    expected_reference = sft["adapterSHA256"] if trainer == "dpo" else None
    if (
        artifact.get("status") != "trained"
        or artifact.get("trainingPhase") != "sft_dpo"
        or artifact.get("parentSFTAdapterSHA256") != sft["adapterSHA256"]
        or artifact.get("referenceSFTAdapterSHA256") != expected_reference
        or artifact.get("preferenceTrainer") != trainer
    ):
        raise RuntimeError(f"Preference identity or parent lineage mismatch: {finalized_path}")
    adapter_manifest = verify_adapter_artifact(
        adapter_dir,
        expected_adapter_sha256=str(artifact.get("adapterSHA256") or ""),
        expected_training_phase="sft_dpo",
        expected_parent_sft_adapter_sha256=sft["adapterSHA256"],
    )
    _verify_training_report(
        report_path,
        phase=trainer.upper(),
        expected={
            "agent": agent,
            "trainer": "ORPOTrainer" if trainer == "orpo" else "DPOTrainer",
            "training_phase": "sft_dpo",
            "seed": config.get("seed"),
            "variantManifestSHA256": config.get("variantManifestSHA256"),
            "output_dir": str(run_root / "training" / agent / "dpo"),
            "adapter_output_dir": str(adapter_dir),
            "adapterSHA256": adapter_manifest["adapterSHA256"],
            "parent_sft_adapter_sha256": sft["adapterSHA256"],
            "reference_sft_adapter_sha256": expected_reference,
            "finalized_variant_manifest_sha256": finalized[
                "variantManifestSHA256"
            ],
            "trainingEnvironmentSHA256": finalized[
                "trainingEnvironmentSHA256"
            ],
        },
    )
    return {
        "phase": trainer,
        "adapterSHA256": adapter_manifest["adapterSHA256"],
        "parentSFTAdapterSHA256": sft["adapterSHA256"],
        "finalizedVariantManifestSHA256": finalized["variantManifestSHA256"],
        "report": str(report_path),
    }


def write_final_config(run_root: Path, agent: str) -> dict[str, Any]:
    preference = verify_preference(run_root, agent)
    run_manifest = read_object(run_root / "aio_run_manifest.json")
    declared_run_sha = run_manifest.get("runManifestSHA256")
    unsigned_run = dict(run_manifest)
    unsigned_run.pop("runManifestSHA256", None)
    if canonical_sha256(unsigned_run) != declared_run_sha:
        raise RuntimeError("Prepared run manifest integrity check failed")
    behavior_manifest_path = (
        run_root
        / "generated"
        / "agent_manifest"
        / "AgentBehaviorManifest.json"
    )
    behavior_file_sha = file_sha256(behavior_manifest_path)
    if behavior_file_sha != run_manifest.get("behaviorManifestFileSHA256"):
        raise RuntimeError("Frozen behavior manifest drifted from the prepared run")
    config = read_object(run_root / "configs" / f"{agent}.json")
    finalized_path = (
        run_root / "training" / agent / "dpo" / "finalized_variant_manifest.json"
    )
    finalized = _verify_manifest_integrity(finalized_path)
    artifact = finalized.get("artifact")
    if not isinstance(artifact, Mapping):
        raise RuntimeError("Preference finalized manifest lacks adapter lineage")
    trainer = str(config.get("preference_trainer", "dpo")).lower()
    phase_manifests = config.get("trainingCodeManifestsByPhase")
    phase_digests = config.get("trainingCodeSHA256ByPhase")
    if not isinstance(phase_manifests, Mapping) or not isinstance(
        phase_digests, Mapping
    ):
        raise RuntimeError("Prepared config lacks phase-specific training code lineage")
    attestation = config.get("variantAttestation")
    if not isinstance(attestation, Mapping):
        raise RuntimeError("Prepared config lacks variant attestation")
    final_attestation = dict(attestation)
    final_attestation["trainingEnvironmentSHA256"] = finalized[
        "trainingEnvironmentSHA256"
    ]
    final_attestation["trainingCodeSHA256"] = phase_digests[trainer]
    for field in RUNTIME_SOURCE_FIELDS:
        final_attestation[field] = finalized[field]
    config.update(
        {
            "adapter_output_dir": str(
                run_root / "models" / "lora_qwen3_dpo" / agent
            ),
            "output_dir": str(run_root / "training" / agent / "dpo"),
            "finalized_variant_manifest": str(finalized_path),
            "adapter_training_phase": "sft_dpo",
            "parent_sft_adapter_sha256": preference["parentSFTAdapterSHA256"],
            "trainingEnvironmentSHA256": finalized[
                "trainingEnvironmentSHA256"
            ],
            "trainingCodeManifest": phase_manifests[trainer],
            "trainingCodeSHA256": phase_digests[trainer],
            "resolvedTrainingEnvironment": finalized[
                "resolvedTrainingEnvironment"
            ],
            "resolvedTrainingEnvironmentSHA256": finalized[
                "resolvedTrainingEnvironmentSHA256"
            ],
            "observedAccelerator": finalized.get("observedAccelerator"),
            "zeroGPUSize": finalized.get("zeroGPUSize"),
            "zeroGPUDurationSeconds": finalized.get("zeroGPUDurationSeconds"),
            "variantAttestation": final_attestation,
            "behaviorManifestFileSHA256": behavior_file_sha,
        }
    )
    export = config.get("adapterExport")
    if not isinstance(export, dict):
        raise RuntimeError("Prepared config lacks adapter export lineage")
    export.update(
        {
            "adapterArtifact": str(
                run_root / "models" / "lora_qwen3_dpo" / agent
            ),
            "adapterDirectory": str(
                run_root / "models" / "lora_qwen3_dpo" / agent
            ),
            "adapterGGUFArtifact": str(
                run_root
                / "models"
                / "lora_qwen3_gguf"
                / f"lumen-{agent}-lora.gguf"
            ),
            "trainBaseModelWeights": False,
            "mergeAdaptersByDefault": False,
        }
    )
    for field in RUNTIME_SOURCE_FIELDS:
        config[field] = finalized[field]
    final_path = run_root / "configs" / f"{agent}.final.json"
    write_object(final_path, config)
    return {
        "agent": agent,
        "config": str(final_path),
        "adapterSHA256": preference["adapterSHA256"],
        "trainingEnvironmentSHA256": finalized["trainingEnvironmentSHA256"],
    }


def clean_phase(run_root: Path, agent: str, phase: str) -> None:
    if phase not in {"sft", "preference"}:
        raise RuntimeError(f"Unsupported cleanup phase: {phase}")
    manifest = _verified_run_manifest(run_root)
    manifest_agents = manifest.get("agents")
    if (
        not isinstance(manifest_agents, list)
        or agent not in {
            item.get("agent")
            for item in manifest_agents
            if isinstance(item, Mapping)
        }
    ):
        raise RuntimeError(f"Agent is not owned by this prepared run: {agent}")
    _reject_managed_symlinks(run_root)
    targets: list[Path] = [
        run_root / "models" / "lora_qwen3_gguf" / f"lumen-{agent}-lora.gguf",
        run_root / "evaluation" / agent,
    ]
    if phase == "sft":
        targets.extend(
            [
                run_root / "training" / agent,
                run_root / "models" / "lora_qwen3_bootstrap" / agent,
                run_root / "models" / "lora_qwen3_dpo" / agent,
            ]
        )
    else:
        targets.extend(
            [
                run_root / "training" / agent / "dpo",
                run_root / "models" / "lora_qwen3_dpo" / agent,
            ]
        )
    for target in targets:
        if run_root not in target.parents or target.is_symlink():
            raise RuntimeError(f"Refusing unsafe phase cleanup target: {target}")
        if target.is_dir():
            shutil.rmtree(target)
        elif target.exists():
            target.unlink()


def _verify_evaluation_outputs(
    run_root: Path,
    agent: str,
    *,
    final_phase: Mapping[str, Any],
) -> dict[str, Any]:
    evaluation_dir = run_root / "evaluation" / agent
    expected_names = {
        "candidate_outputs.jsonl",
        "evaluation_report.json",
        "evaluation_run_manifest.json",
    }
    if evaluation_dir.is_symlink() or not evaluation_dir.is_dir():
        raise RuntimeError(f"Missing regular evaluation directory: {evaluation_dir}")
    entries = list(evaluation_dir.iterdir())
    if (
        {entry.name for entry in entries} != expected_names
        or any(entry.is_symlink() or not entry.is_file() for entry in entries)
    ):
        raise RuntimeError(
            f"Evaluation directory must contain exactly the verified evidence trio: {evaluation_dir}"
        )
    candidate_path = evaluation_dir / "candidate_outputs.jsonl"
    report_path = evaluation_dir / "evaluation_report.json"
    run_path = evaluation_dir / "evaluation_run_manifest.json"
    if candidate_path.stat().st_size == 0:
        raise RuntimeError(f"Candidate output evidence is empty: {candidate_path}")
    evaluation_run = read_object(run_path)
    run_digest = evaluation_run.get("runManifestSHA256")
    unsigned_run = dict(evaluation_run)
    unsigned_run.pop("runManifestSHA256", None)
    report = read_object(report_path)
    report_digest = report.get("reportSHA256")
    unsigned_report = dict(report)
    unsigned_report.pop("reportSHA256", None)
    if (
        canonical_sha256(unsigned_run) != run_digest
        or canonical_sha256(unsigned_report) != report_digest
        or evaluation_run.get("agent") != agent
        or evaluation_run.get("adapterSHA256") != final_phase.get("adapterSHA256")
        or evaluation_run.get("finalizedVariantManifestSHA256")
        != final_phase.get("finalizedVariantManifestSHA256")
        or evaluation_run.get("candidateOutputsFileSHA256")
        != file_sha256(candidate_path)
        or evaluation_run.get("evaluationReportFileSHA256")
        != file_sha256(report_path)
        or evaluation_run.get("evaluationReportSHA256") != report_digest
        or evaluation_run.get("candidateOutputsSHA256")
        != report.get("candidateOutputsSHA256")
    ):
        raise RuntimeError(f"Evaluation evidence lineage failed verification: {evaluation_dir}")
    if evaluation_run.get("status") not in {
        "quality_gate_passed",
        "smoke_complete",
    }:
        raise RuntimeError(f"Evaluation did not pass or complete a smoke run: {run_path}")
    if (
        evaluation_run.get("completeEvaluation") is True
        and evaluation_run.get("qualityGatePassed") is not True
    ):
        raise RuntimeError(f"Full evaluation quality gate failed: {run_path}")
    return evaluation_run


def write_summary(
    *,
    run_root: Path,
    agents: Sequence[str],
    variant: str,
    preference: bool,
    require_gguf: bool,
    require_evaluation: bool,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "schema": "lumen.ubuntu-training-summary/2.0.0",
        "status": "pending_verification",
        "variant": variant,
        "runRoot": str(run_root),
        "preferenceTraining": preference,
        "agents": {},
    }
    for agent in agents:
        sft = verify_sft(run_root, agent)
        final_phase = verify_preference(run_root, agent) if preference else sft
        gguf = run_root / "models" / "lora_qwen3_gguf" / f"lumen-{agent}-lora.gguf"
        if require_gguf and (not gguf.is_file() or gguf.stat().st_size == 0):
            raise RuntimeError(f"Missing required GGUF adapter: {gguf}")
        evaluation_dir = run_root / "evaluation" / agent
        evaluation = evaluation_dir / "evaluation_report.json"
        evaluation_run = evaluation_dir / "evaluation_run_manifest.json"
        if require_evaluation and (
            not evaluation.is_file()
            or evaluation.stat().st_size == 0
            or not evaluation_run.is_file()
        ):
            raise RuntimeError(f"Missing required evaluation report: {evaluation}")
        evaluation_status: dict[str, Any] | None = None
        if evaluation_run.is_file():
            evaluation_status = _verify_evaluation_outputs(
                run_root,
                agent,
                final_phase=final_phase,
            )
        summary["agents"][agent] = {
            "sft": sft,
            "finalPhase": final_phase,
            "adapterGGUF": str(gguf),
            "adapterGGUFExists": gguf.is_file(),
            "adapterGGUFSHA256": file_sha256(gguf) if gguf.is_file() else None,
            "adapterGGUFSizeBytes": gguf.stat().st_size if gguf.is_file() else 0,
            "evaluationReport": str(evaluation),
            "evaluationReportExists": evaluation.is_file(),
            "evaluation": evaluation_status,
        }
    evaluations = [
        item["evaluation"]
        for item in summary["agents"].values()
        if isinstance(item.get("evaluation"), Mapping)
    ]
    if len(evaluations) == len(agents) and all(
        item.get("status") == "quality_gate_passed" for item in evaluations
    ):
        summary["status"] = "complete"
    elif evaluations:
        summary["status"] = "smoke_complete"
    else:
        summary["status"] = "training_complete_without_full_evaluation"
    summary["summarySHA256"] = canonical_sha256(summary)
    write_object(run_root / "aio_summary.json", summary)
    return summary


def _verified_completed_summary(
    run_root: Path,
    agents: Sequence[str],
) -> dict[str, Any]:
    summary_path = run_root / "aio_summary.json"
    if summary_path.is_symlink() or not summary_path.is_file():
        raise RuntimeError(f"Missing regular completed summary: {summary_path}")
    summary = read_object(summary_path)
    declared = summary.get("summarySHA256")
    unsigned = dict(summary)
    unsigned.pop("summarySHA256", None)
    summary_agents = summary.get("agents")
    if (
        canonical_sha256(unsigned) != declared
        or summary.get("schema") != "lumen.ubuntu-training-summary/2.0.0"
        or summary.get("runRoot") != str(run_root)
        or summary.get("preferenceTraining") is not True
        or summary.get("status")
        not in {
            "complete",
            "smoke_complete",
            "training_complete_without_full_evaluation",
        }
        or not isinstance(summary_agents, Mapping)
        or set(summary_agents) != set(agents)
    ):
        raise RuntimeError("Completed Ubuntu training summary failed verification")
    evaluation_statuses: list[str] = []
    for agent in agents:
        item = summary_agents.get(agent)
        if not isinstance(item, Mapping):
            raise RuntimeError(f"Completed summary lacks agent {agent}")
        final_phase = verify_preference(run_root, agent)
        if item.get("finalPhase") != final_phase:
            raise RuntimeError(f"Completed summary adapter lineage drifted for {agent}")
        evaluation = item.get("evaluation")
        if evaluation is not None:
            verified_evaluation = _verify_evaluation_outputs(
                run_root,
                agent,
                final_phase=final_phase,
            )
            if evaluation != verified_evaluation:
                raise RuntimeError(f"Completed summary evaluation drifted for {agent}")
            evaluation_statuses.append(str(verified_evaluation.get("status")))
        elif item.get("evaluationReportExists") is not False:
            raise RuntimeError(f"Completed summary evaluation flag drifted for {agent}")
        gguf = run_root / "models" / "lora_qwen3_gguf" / f"lumen-{agent}-lora.gguf"
        if item.get("adapterGGUFExists") is True:
            if (
                gguf.is_symlink()
                or not gguf.is_file()
                or gguf.stat().st_size != item.get("adapterGGUFSizeBytes")
                or file_sha256(gguf) != item.get("adapterGGUFSHA256")
            ):
                raise RuntimeError(f"Completed summary GGUF drifted for {agent}")
        elif gguf.exists():
            raise RuntimeError(f"Unbound GGUF exists for {agent}")
    expected_status = (
        "complete"
        if len(evaluation_statuses) == len(agents)
        and set(evaluation_statuses) == {"quality_gate_passed"}
        else "smoke_complete"
        if evaluation_statuses
        else "training_complete_without_full_evaluation"
    )
    if summary.get("status") != expected_status:
        raise RuntimeError("Completed summary overstates its evaluation status")
    return summary


def upload_run(
    *,
    run_root: Path,
    agents: Sequence[str],
    run_id: str,
    private: bool,
    include_gguf: bool,
    token_file: Path,
) -> dict[str, Any]:
    try:
        from huggingface_hub import CommitOperationAdd, HfApi
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required for upload") from exc
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,255}", run_id) is None:
        raise RuntimeError("Upload run ID is unsafe")
    run_manifest = _verified_run_manifest(run_root)
    if run_manifest.get("runID") != run_id:
        raise RuntimeError("Upload run ID does not match the prepared run")
    manifest_agents = run_manifest.get("agents")
    if [item.get("agent") for item in manifest_agents] != list(agents):
        raise RuntimeError("Upload agents do not match the prepared run")
    summary = _verified_completed_summary(run_root, agents)
    repo_id = str(run_manifest.get("adapterRepoID") or "").strip()
    runtime_manifest_path = (
        run_root / "generated" / "fine_tuning" / "adapter_runtime_manifest.json"
    )
    if (
        not repo_id
        or file_sha256(runtime_manifest_path)
        != run_manifest.get("adapterRuntimeManifestFileSHA256")
        or read_object(runtime_manifest_path).get("adapterRepoID") != repo_id
    ):
        raise RuntimeError("Prepared adapter repository destination drifted")
    if read_object(run_root / "training_environment.json") != run_manifest.get(
        "trainingEnvironment"
    ):
        raise RuntimeError("Prepared training environment record drifted")

    local_files: list[tuple[Path, str]] = []
    for agent in agents:
        preference = verify_preference(run_root, agent)
        adapter_dir = run_root / "models" / "lora_qwen3_dpo" / agent
        adapter_manifest = read_object(adapter_dir / "adapter_artifact_manifest.json")
        artifact_files = adapter_manifest.get("files")
        if not isinstance(artifact_files, list):
            raise RuntimeError(f"Adapter upload manifest is invalid for {agent}")
        for item in artifact_files:
            if not isinstance(item, Mapping) or not isinstance(item.get("path"), str):
                raise RuntimeError(f"Adapter upload manifest is invalid for {agent}")
            local_files.append(
                (
                    adapter_dir / item["path"],
                    f"runs/{run_id}/adapters/{agent}/{item['path']}",
                )
            )
        local_files.append(
            (
                adapter_dir / "adapter_artifact_manifest.json",
                f"runs/{run_id}/adapters/{agent}/adapter_artifact_manifest.json",
            )
        )
        local_files.append(
            (
                run_root
                / "training"
                / agent
                / "dpo"
                / "finalized_variant_manifest.json",
                f"runs/{run_id}/manifests/{agent}/variant_manifest.json",
            )
        )
        evaluation = summary["agents"][agent].get("evaluation")
        if isinstance(evaluation, Mapping):
            for filename in (
                "candidate_outputs.jsonl",
                "evaluation_report.json",
                "evaluation_run_manifest.json",
            ):
                local_files.append(
                    (
                        run_root / "evaluation" / agent / filename,
                        f"runs/{run_id}/evaluation/{agent}/{filename}",
                    )
                )
        if include_gguf:
            if summary["agents"][agent].get("adapterGGUFExists") is not True:
                raise RuntimeError(f"Upload requires a verified GGUF for {agent}")
            local_files.append(
                (
                    run_root
                    / "models"
                    / "lora_qwen3_gguf"
                    / f"lumen-{agent}-lora.gguf",
                    f"runs/{run_id}/gguf/lumen-{agent}-lora.gguf",
                )
            )
        if preference.get("adapterSHA256") != adapter_manifest.get("adapterSHA256"):
            raise RuntimeError(f"Upload adapter lineage drifted for {agent}")
    for filename in (
        "aio_run_manifest.json",
        "aio_summary.json",
        "training_environment.json",
    ):
        local_files.append((run_root / filename, f"runs/{run_id}/{filename}"))
    remote_paths = [remote for _, remote in local_files]
    if len(set(remote_paths)) != len(remote_paths):
        raise RuntimeError("Upload file contract contains duplicate remote paths")
    for local_path, _ in local_files:
        if local_path.is_symlink() or not local_path.is_file():
            raise RuntimeError(f"Upload input is not a regular verified file: {local_path}")
    receipt_path = run_root / "upload_receipts.json"
    if receipt_path.exists() or receipt_path.is_symlink():
        raise RuntimeError(f"Upload receipt path already exists: {receipt_path}")

    if token_file.is_symlink() or not token_file.is_file():
        raise RuntimeError("Upload token path is not a regular mounted secret")
    token = token_file.read_text(encoding="utf-8").strip()
    if not token or "\n" in token or "\r" in token:
        raise RuntimeError("Upload token file is empty or malformed")
    api = HfApi(token=token)
    identity = api.whoami()
    if not isinstance(identity, Mapping) or not identity.get("name"):
        raise RuntimeError("Hugging Face authentication preflight failed")
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
    info = api.repo_info(repo_id=repo_id, repo_type="model")
    if bool(info.private) != private:
        raise RuntimeError("Remote repository visibility does not match the requested policy")
    existing_files = api.list_repo_files(repo_id=repo_id, repo_type="model")
    prefix = f"runs/{run_id}/"
    if any(path.startswith(prefix) for path in existing_files):
        raise RuntimeError(f"Remote run prefix already exists: {prefix}")
    parent_revision = getattr(info, "sha", None)
    if parent_revision is not None and re.fullmatch(
        r"[0-9a-f]{40}", str(parent_revision)
    ) is None:
        raise RuntimeError("Remote parent revision is not immutable")
    commit = api.create_commit(
        repo_id=repo_id,
        repo_type="model",
        operations=[
            CommitOperationAdd(path_in_repo=remote, path_or_fileobj=str(local))
            for local, remote in local_files
        ],
        commit_message=f"Upload verified Lumen training run {run_id}",
        parent_commit=parent_revision,
    )
    commit_oid = getattr(commit, "oid", None)
    if re.fullmatch(r"[0-9a-f]{40}", str(commit_oid or "")) is None:
        raise RuntimeError("Hugging Face upload did not return an immutable commit OID")
    final_info = api.repo_info(repo_id=repo_id, repo_type="model")
    final_revision = getattr(final_info, "sha", None)
    if final_revision != commit_oid or bool(final_info.private) != private:
        raise RuntimeError("Remote upload head or visibility failed post-commit verification")
    result: dict[str, Any] = {
        "schema": "lumen.ubuntu-training-upload/1.0.0",
        "repository": repo_id,
        "private": bool(final_info.private),
        "headRevision": final_revision,
        "parentRevision": parent_revision,
        "runID": run_id,
        "uploadedFileCount": len(local_files),
        "uploadedPaths": remote_paths,
        "commitOID": commit_oid,
    }
    result["uploadSHA256"] = canonical_sha256(result)
    write_object(receipt_path, result)
    return result


def _common_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--dataset-source", type=Path, required=True)
    parser.add_argument("--agents", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--base-model", default="")
    parser.add_argument("--container-digest", required=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fail-closed Ubuntu training pipeline helper")
    subparsers = parser.add_subparsers(dest="command", required=True)
    static = subparsers.add_parser("static-preflight")
    _common_parser(static)
    static.add_argument("--run-root", type=Path, required=True)
    static.add_argument("--allowed-run-parent", type=Path, required=True)
    static.add_argument("--run-id")
    runtime = subparsers.add_parser("runtime-preflight")
    _common_parser(runtime)
    prepare = subparsers.add_parser("prepare")
    _common_parser(prepare)
    prepare.add_argument("--run-root", type=Path, required=True)
    validate = subparsers.add_parser("validate-prepared-runtime")
    validate.add_argument("--root", type=Path, required=True)
    validate.add_argument("--run-root", type=Path, required=True)
    validate.add_argument("--agents", required=True)
    validate.add_argument("--variant", required=True)
    validate.add_argument("--container-digest", required=True)
    owned = subparsers.add_parser("verify-owned-run")
    owned.add_argument("--run-root", type=Path, required=True)
    owned.add_argument("--variant", required=True)
    verify = subparsers.add_parser("verify-phase")
    verify.add_argument("--run-root", type=Path, required=True)
    verify.add_argument("--agent", choices=AGENTS, required=True)
    verify.add_argument("--phase", choices=("sft", "preference"), required=True)
    verify_gguf_parser = subparsers.add_parser("verify-gguf")
    verify_gguf_parser.add_argument("--run-root", type=Path, required=True)
    verify_gguf_parser.add_argument("--agent", choices=AGENTS, required=True)
    clean = subparsers.add_parser("clean-phase")
    clean.add_argument("--run-root", type=Path, required=True)
    clean.add_argument("--agent", choices=AGENTS, required=True)
    clean.add_argument("--phase", choices=("sft", "preference"), required=True)
    final_config = subparsers.add_parser("write-final-config")
    final_config.add_argument("--run-root", type=Path, required=True)
    final_config.add_argument("--agent", choices=AGENTS, required=True)
    summary = subparsers.add_parser("write-summary")
    summary.add_argument("--run-root", type=Path, required=True)
    summary.add_argument("--agents", required=True)
    summary.add_argument("--variant", required=True)
    summary.add_argument("--preference", action="store_true")
    summary.add_argument("--require-gguf", action="store_true")
    summary.add_argument("--require-evaluation", action="store_true")
    upload = subparsers.add_parser("upload")
    upload.add_argument("--run-root", type=Path, required=True)
    upload.add_argument("--agents", required=True)
    upload.add_argument("--run-id", required=True)
    upload.add_argument("--public", action="store_true")
    upload.add_argument("--include-gguf", action="store_true")
    upload.add_argument("--token-file", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if hasattr(args, "agents"):
        agents = parse_agents(args.agents)
    resolved_run_root: Path | None = None
    if hasattr(args, "run_root") and args.command != "static-preflight":
        raw_run_root = args.run_root.expanduser()
        if raw_run_root.is_symlink():
            raise RuntimeError(f"Run root must not be a symlink: {raw_run_root}")
        resolved_run_root = raw_run_root.resolve()
    if args.command == "static-preflight":
        result = static_preflight(
            root=args.root.resolve(),
            dataset_source=args.dataset_source.resolve(),
            agents=agents,
            variant=args.variant,
            seed=args.seed,
            base_model_override=args.base_model,
            container_digest=args.container_digest,
            run_root=args.run_root,
            allowed_parent=args.allowed_run_parent,
            expected_run_id=args.run_id,
        )
    elif args.command == "runtime-preflight":
        result = runtime_preflight(
            root=args.root.resolve(),
            dataset_source=args.dataset_source.resolve(),
            agent=agents[0],
            variant=args.variant,
            seed=args.seed,
            base_model_override=args.base_model,
            container_digest=args.container_digest,
        )
    elif args.command == "prepare":
        result = prepare_run(
            root=args.root.resolve(),
            dataset_source=args.dataset_source.resolve(),
            run_root=resolved_run_root,
            agents=agents,
            variant=args.variant,
            seed=args.seed,
            base_model_override=args.base_model,
            container_digest=args.container_digest,
        )
    elif args.command == "validate-prepared-runtime":
        result = validate_prepared_runtime(
            root=args.root.resolve(),
            run_root=resolved_run_root,
            agents=agents,
            variant=args.variant,
            container_digest=args.container_digest,
        )
    elif args.command == "verify-owned-run":
        result = verify_owned_run(
            resolved_run_root,
            variant=args.variant,
        )
    elif args.command == "verify-phase":
        result = (
            verify_sft(resolved_run_root, args.agent)
            if args.phase == "sft"
            else verify_preference(resolved_run_root, args.agent)
        )
    elif args.command == "verify-gguf":
        result = verify_gguf(resolved_run_root, args.agent)
    elif args.command == "clean-phase":
        clean_phase(resolved_run_root, args.agent, args.phase)
        result = {"status": "cleaned", "agent": args.agent, "phase": args.phase}
    elif args.command == "write-final-config":
        result = write_final_config(resolved_run_root, args.agent)
    elif args.command == "write-summary":
        result = write_summary(
            run_root=resolved_run_root,
            agents=agents,
            variant=args.variant,
            preference=args.preference,
            require_gguf=args.require_gguf,
            require_evaluation=args.require_evaluation,
        )
    elif args.command == "upload":
        result = upload_run(
            run_root=resolved_run_root,
            agents=agents,
            run_id=args.run_id,
            private=not args.public,
            include_gguf=args.include_gguf,
            token_file=args.token_file,
        )
    else:  # pragma: no cover - argparse enforces the command set.
        raise RuntimeError(f"Unsupported command: {args.command}")
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"ubuntu pipeline error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
