from __future__ import annotations

import re
from typing import Any

ADAPTER_EXPORT_SCHEMA_VERSION = "1.0.0"
DEFAULT_AGENT_BASE_MODEL_ID = "Qwen/Qwen3-1.7B"
DEFAULT_LORA_OUTPUT_ROOT = "models/lora"
DEFAULT_ADAPTER_DIR = DEFAULT_LORA_OUTPUT_ROOT
DEFAULT_RELEASE_BAKE_OUTPUT_ROOT = "models/gguf_release_bake"


def adapter_artifact_name(agent: str) -> str:
    slug = re.sub(r"[^a-z0-9_.-]+", "-", agent.strip().casefold()).strip("-._")
    return f"{slug or 'agent'}.lora"


def adapter_output_dir(agent: str) -> str:
    return f"{DEFAULT_LORA_OUTPUT_ROOT}/{agent}"


def adapter_artifact_path(agent: str) -> str:
    # Unsloth/PEFT saves LoRA adapters as a directory containing adapter weights,
    # tokenizer/config metadata, and trainer state. The runtime contract must point
    # at that real adapter directory, not at a synthetic standalone .lora filename.
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
    out["artifactMode"] = "adapter_first"
    out["defaultExportArtifact"] = "lora_adapter"
    out["artifact_mode"] = "adapter_first"
    out["default_export_artifact"] = "lora_adapter"
    out["merge_adapters_by_default"] = False
    out["release_bake_enabled_by_default"] = False
    out["adapter_output_dir"] = adapter_output_dir(agent)
    out["output_dir"] = adapter_output_dir(agent)
    out["gguf_output_dir"] = release_bake_output_dir(agent)
    out["adapterExport"] = {
        "enabled": True,
        "agent": agent,
        "adapterID": f"lumen-{agent}-adapter",
        "adapterArtifact": adapter_artifact_path(agent),
        "adapterDirectory": adapter_output_dir(agent),
        "baseModelID": base_model_id,
        "trainBaseModelWeights": False,
        "saveAdapterByDefault": True,
        "mergeAdaptersByDefault": False,
        "rollbackUnit": "adapter",
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
    return {
        "schemaVersion": ADAPTER_EXPORT_SCHEMA_VERSION,
        "mode": "adapter_first",
        "agent": agent,
        "baseModelID": base_model_id,
        "adapterID": f"lumen-{agent}-adapter",
        "adapterArtifact": adapter_artifact_path(agent),
        "adapterDirectory": adapter_output_dir(agent),
        "systemPrompt": dataset_card.get("systemPrompt"),
        "datasetCard": {
            "manifestCommit": dataset_card.get("manifestCommit"),
            "recordCounts": dataset_card.get("recordCounts", {}),
            "sourceFamilies": dataset_card.get("sourceFamilies", []),
            "taskTypes": dataset_card.get("taskTypes", []),
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
        "expectedArtifacts": {
            "adapterDirectory": adapter_output_dir(agent),
            "trainSFT": "train_sft.jsonl",
            "validationSFT": "val_sft.jsonl",
            "trainDPO": "train_dpo.jsonl",
            "validationDPO": "val_dpo.jsonl",
            "eval": "eval.jsonl",
            "datasetCard": "dataset_card.json",
            "trainingConfig": "unsloth_config.json",
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
                "baseModelID": base_model_id,
                "systemPrompt": dataset_card.get("systemPrompt"),
                "recordCounts": dataset_card.get("recordCounts", {}),
            }
        )

    shared_base_model_id = next(iter(base_model_ids)) if len(base_model_ids) == 1 else None
    return {
        "schemaVersion": ADAPTER_EXPORT_SCHEMA_VERSION,
        "mode": "adapter_first",
        "sharedBaseModelID": shared_base_model_id,
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
