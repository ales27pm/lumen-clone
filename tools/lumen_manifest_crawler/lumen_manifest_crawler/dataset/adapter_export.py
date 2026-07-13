from __future__ import annotations

import re
from typing import Any

from lumen_manifest_crawler.dataset.adapter_evaluation import (
    DEFAULT_BASE_MODEL_ARTIFACT_DIGEST,
    DEFAULT_BASE_MODEL_REVISION,
    DEFAULT_BASE_MODEL_TOKENIZER_DIGEST,
    EXPERIMENT_VARIANTS,
    promotion_contract,
)

ADAPTER_EXPORT_SCHEMA_VERSION = "1.2.0"
DEFAULT_AGENT_BASE_MODEL_ID = "Qwen/Qwen3-1.7B"
DEFAULT_LORA_OUTPUT_ROOT = "models/lora_qwen3_bootstrap"
DEFAULT_ADAPTER_GGUF_OUTPUT_ROOT = "models/lora_qwen3_gguf"
DEFAULT_RELEASE_BAKE_OUTPUT_ROOT = "models/gguf_release_bake_qwen3_bootstrap"
DEFAULT_ADAPTER_REPO_ID = "ales27pm/lumen-qwen3-bootstrap-adapters-gguf"
DEFAULT_SHARED_BASE_REPO_ID = "ales27pm/lumen-qwen3-bootstrap-gguf"
DEFAULT_SHARED_BASE_FILE_NAME = "lumen-qwen3-fast-shared-q4_k_m.gguf"


def adapter_artifact_name(agent: str) -> str:
    slug = re.sub(r"[^a-z0-9_.-]+", "-", agent.strip().casefold()).strip("-._")
    return f"lumen-{slug or 'agent'}-lora.gguf"


def adapter_output_dir(agent: str) -> str:
    return f"{DEFAULT_LORA_OUTPUT_ROOT}/{agent}"


def adapter_gguf_output_path(agent: str) -> str:
    return f"{DEFAULT_ADAPTER_GGUF_OUTPUT_ROOT}/{adapter_artifact_name(agent)}"


def adapter_artifact_path(agent: str) -> str:
    # Unsloth/PEFT saves LoRA adapters as a directory containing adapter weights,
    # tokenizer/config metadata, and trainer state. The runtime contract must point
    # at that real adapter directory until the explicit LoRA->GGUF conversion stage.
    return adapter_output_dir(agent)


def release_bake_output_dir(agent: str) -> str:
    return f"{DEFAULT_RELEASE_BAKE_OUTPUT_ROOT}/{agent}_merged_gguf"


def base_model_id_from_config(config: dict[str, Any] | None) -> str:
    config = config or {}
    for key in (
        "baseModelID",
        "base_model_id",
        "base_model",
        "base_model_name",
        "model_name",
        "modelName",
    ):
        value = config.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return DEFAULT_AGENT_BASE_MODEL_ID


def augment_unsloth_config_for_adapter_export(agent: str, config: dict[str, Any] | None) -> dict[str, Any]:
    """Return an adapter-first training/export config without mutating the input config.

    The improvement loop should train and keep role adapters as the default artifact.
    Merged full-model/GGUF export remains possible, but only as an explicit release-bake
    step after an adapter passes role-specific eval gates.
    """
    out = dict(config or {})
    base_model_id = base_model_id_from_config(out)
    out.setdefault("baseModelID", base_model_id)
    out.setdefault("baseModelRevision", DEFAULT_BASE_MODEL_REVISION)
    out.setdefault("baseModelArtifactDigest", DEFAULT_BASE_MODEL_ARTIFACT_DIGEST)
    out.setdefault("baseModelTokenizerDigest", DEFAULT_BASE_MODEL_TOKENIZER_DIGEST)
    out["artifactMode"] = "adapter_first"
    out["defaultExportArtifact"] = "lora_adapter"
    out["artifact_mode"] = "adapter_first"
    out["default_export_artifact"] = "lora_adapter"
    out["merge_adapters_by_default"] = False
    out["release_bake_enabled_by_default"] = False
    out["adapter_output_dir"] = adapter_output_dir(agent)
    out["output_dir"] = adapter_output_dir(agent)
    out["adapter_gguf_output_path"] = adapter_gguf_output_path(agent)
    out["gguf_output_dir"] = release_bake_output_dir(agent)
    out["adapterExport"] = {
        "enabled": True,
        "agent": agent,
        "adapterID": f"lumen-{agent}-adapter",
        "adapterArtifact": adapter_artifact_path(agent),
        "adapterDirectory": adapter_output_dir(agent),
        "adapterGGUFArtifact": adapter_gguf_output_path(agent),
        "adapterRepoID": DEFAULT_ADAPTER_REPO_ID,
        "baseModelID": base_model_id,
        "baseModelRevision": out["baseModelRevision"],
        "baseModelArtifactDigest": out["baseModelArtifactDigest"],
        "baseModelTokenizerDigest": out["baseModelTokenizerDigest"],
        "sharedBaseRepoID": DEFAULT_SHARED_BASE_REPO_ID,
        "sharedBaseFileName": DEFAULT_SHARED_BASE_FILE_NAME,
        "trainBaseModelWeights": False,
        "saveAdapterByDefault": True,
        "mergeAdaptersByDefault": False,
        "rollbackUnit": "adapter",
        "trainingPhases": {
            "sft": "enabled",
            "dpo": "generated_not_trained",
        },
    }
    out["mergeExport"] = {
        "enabledByDefault": False,
        "phase": "optional_release_bake",
        "allowManualExport": True,
        "requiresPassingEvalGates": True,
        "reason": "Keep one shared base model plus role adapters during iterative training; merge only for release/runtime backends that cannot load adapters dynamically.",
    }
    return out


def agent_adapter_export_plan(agent: str, dataset_card: dict[str, Any], unsloth_config: dict[str, Any] | None) -> dict[str, Any]:
    base_model_id = base_model_id_from_config(unsloth_config)
    config = unsloth_config or {}
    return {
        "schemaVersion": ADAPTER_EXPORT_SCHEMA_VERSION,
        "mode": "adapter_first",
        "agent": agent,
        "baseModelID": base_model_id,
        "baseModelRevision": config.get("baseModelRevision", DEFAULT_BASE_MODEL_REVISION),
        "baseModelArtifactDigest": config.get("baseModelArtifactDigest", DEFAULT_BASE_MODEL_ARTIFACT_DIGEST),
        "baseModelTokenizerDigest": config.get("baseModelTokenizerDigest", DEFAULT_BASE_MODEL_TOKENIZER_DIGEST),
        "sharedBaseRepoID": DEFAULT_SHARED_BASE_REPO_ID,
        "sharedBaseFileName": DEFAULT_SHARED_BASE_FILE_NAME,
        "adapterRepoID": DEFAULT_ADAPTER_REPO_ID,
        "adapterID": f"lumen-{agent}-adapter",
        "adapterArtifact": adapter_artifact_path(agent),
        "adapterDirectory": adapter_output_dir(agent),
        "adapterGGUFArtifact": adapter_gguf_output_path(agent),
        "systemPrompt": dataset_card.get("systemPrompt"),
        "datasetCard": {
            "manifestCommit": dataset_card.get("manifestCommit"),
            "recordCounts": dataset_card.get("recordCounts", {}),
            "sourceFamilies": dataset_card.get("sourceFamilies", []),
            "taskTypes": dataset_card.get("taskTypes", []),
            "evaluation": dataset_card.get("evaluation", {}),
            "preferenceTraining": dataset_card.get("preferenceTraining", {}),
        },
        "runtimeBinding": {
            "loadBaseModelOnce": True,
            "selectAdapterByAgentSlot": True,
            "agentSlot": agent,
            "promptBinding": "systemPrompt",
            "fallbackToBaselineAdapter": True,
        },
        "exportPolicy": {
            "defaultArtifact": "adapter",
            "mergeAdaptersByDefault": False,
            "mergedExportPhase": "optional_release_bake",
            "publishMergedArtifactByDefault": False,
            "allowMergedExport": True,
            "requiresPassingEvalGatesBeforeMerge": True,
            "rollbackUnit": "adapter",
        },
        "experimentPolicy": {
            "requiredVariants": list(EXPERIMENT_VARIANTS),
            "promotionContract": promotion_contract(),
            "runtimePointerPolicy": "unchanged_until_promoted",
            "experimentManifestSHA256": (dataset_card.get("experimentPolicy") or {}).get("experimentManifestSHA256"),
        },
        "expectedArtifacts": {
            "adapterDirectory": adapter_output_dir(agent),
            "adapterGGUF": adapter_gguf_output_path(agent),
            "trainSFT": "train_sft.jsonl",
            "validationSFT": "val_sft.jsonl",
            "trainDPO": "train_dpo.jsonl",
            "validationDPO": "val_dpo.jsonl",
            "eval": "eval.jsonl",
            "datasetCard": "dataset_card.json",
            "trainingConfig": "unsloth_config.json",
            "evaluationFingerprints": "evaluation_fingerprints.json",
            "contaminationReport": "contamination_report.json",
            "experimentManifest": "experiment_manifest.json",
            "experimentRoot": "experiments",
            "variantPathTemplate": "experiments/{variant}",
            "variantManifestPathTemplate": "experiments/{variant}/variant_manifest.json",
        },
    }


def adapter_runtime_manifest(datasets: dict[str, Any]) -> dict[str, Any]:
    adapters: list[dict[str, Any]] = []
    base_model_ids: set[str] = set()
    for agent, dataset in sorted(datasets.items()):
        unsloth_config = getattr(dataset, "unsloth_config", {}) or {}
        dataset_card = getattr(dataset, "dataset_card", {}) or {}
        base_model_id = base_model_id_from_config(unsloth_config)
        base_model_ids.add(base_model_id)
        adapters.append(
            {
                "agent": agent,
                "adapterID": f"lumen-{agent}-adapter",
                "adapterArtifact": adapter_artifact_path(agent),
                "adapterDirectory": adapter_output_dir(agent),
                "adapterGGUFArtifact": adapter_gguf_output_path(agent),
                "adapterRepoID": DEFAULT_ADAPTER_REPO_ID,
                "baseModelID": base_model_id,
                "baseModelRevision": unsloth_config.get("baseModelRevision", DEFAULT_BASE_MODEL_REVISION),
                "baseModelArtifactDigest": unsloth_config.get("baseModelArtifactDigest", DEFAULT_BASE_MODEL_ARTIFACT_DIGEST),
                "baseModelTokenizerDigest": unsloth_config.get("baseModelTokenizerDigest", DEFAULT_BASE_MODEL_TOKENIZER_DIGEST),
                "systemPrompt": dataset_card.get("systemPrompt"),
                "recordCounts": dataset_card.get("recordCounts", {}),
                "evaluation": dataset_card.get("evaluation", {}),
                "preferenceTraining": dataset_card.get("preferenceTraining", {}),
                "experimentPolicy": dataset_card.get("experimentPolicy", {}),
            }
        )

    shared_base_model_id = next(iter(base_model_ids)) if len(base_model_ids) == 1 else None
    return {
        "schemaVersion": ADAPTER_EXPORT_SCHEMA_VERSION,
        "mode": "adapter_first",
        "sharedBaseModelID": shared_base_model_id,
        "sharedBaseModelRevision": DEFAULT_BASE_MODEL_REVISION,
        "sharedBaseModelArtifactDigest": DEFAULT_BASE_MODEL_ARTIFACT_DIGEST,
        "sharedBaseModelTokenizerDigest": DEFAULT_BASE_MODEL_TOKENIZER_DIGEST,
        "sharedBaseRepoID": DEFAULT_SHARED_BASE_REPO_ID,
        "sharedBaseFileName": DEFAULT_SHARED_BASE_FILE_NAME,
        "adapterRepoID": DEFAULT_ADAPTER_REPO_ID,
        "baseModelIDs": sorted(base_model_ids),
        "runtimeStrategy": {
            "loadBaseModelOnce": True,
            "selectAdapterByAgentSlot": True,
            "mergeAdaptersByDefault": False,
            "mergedExportPhase": "optional_release_bake",
            "fallbackUnit": "adapter",
        },
        "adapters": adapters,
        "releaseBakePolicy": {
            "enabledByDefault": False,
            "manualOnly": True,
            "requiresPassingEvalGates": True,
            "allowedWhenRuntimeCannotLoadAdapters": True,
        },
    }
