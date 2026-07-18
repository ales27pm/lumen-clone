from __future__ import annotations

import re
from typing import Any

from lumen_manifest_crawler.dataset.adapter_evaluation import (
    DEFAULT_BASE_MODEL_ARTIFACT_DIGEST,
    DEFAULT_BASE_MODEL_INDEX_DIGEST,
    DEFAULT_BASE_MODEL_INDEX_REFERENCED_SHARD_NAMES,
    DEFAULT_BASE_MODEL_INDEX_SHARD_BINDING_SHA256,
    DEFAULT_BASE_MODEL_REVISION,
    DEFAULT_BASE_MODEL_TOKENIZER_DIGEST,
    DEFAULT_BASE_MODEL_WEIGHT_SHARDS,
    EXPERIMENT_VARIANTS,
    RUNTIME_SOURCE_AUDIT_FIELDS,
    default_training_lineage_contract,
    promotion_contract,
)
from lumen_manifest_crawler.runtime_prompt_contract import (
    RUNTIME_PROMPT_COMPOSER_POLICY_ID,
    RUNTIME_PROMPT_COMPOSER_POLICY_SHA256,
    SHIPPED_RUNTIME_QUALIFICATION_SCHEMA_VERSION,
    prompt_sha256,
)

ADAPTER_EXPORT_SCHEMA_VERSION = "1.5.0"
DEFAULT_AGENT_BASE_MODEL_ID = "Qwen/Qwen3-1.7B"
DEFAULT_LORA_OUTPUT_ROOT = "models/lora_qwen3_bootstrap"
DEFAULT_TRAINING_OUTPUT_ROOT = "models/training_runs_qwen3_bootstrap"
DEFAULT_DPO_OUTPUT_ROOT = "models/lora_qwen3_dpo"
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


def training_output_dir(agent: str) -> str:
    return f"{DEFAULT_TRAINING_OUTPUT_ROOT}/{agent}"


def dpo_output_dir(agent: str) -> str:
    return f"{DEFAULT_DPO_OUTPUT_ROOT}/{agent}"


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
    out.setdefault("baseModelIndexDigest", DEFAULT_BASE_MODEL_INDEX_DIGEST)
    out.setdefault(
        "baseModelIndexReferencedShardNames",
        list(DEFAULT_BASE_MODEL_INDEX_REFERENCED_SHARD_NAMES),
    )
    out.setdefault(
        "baseModelIndexShardBindingSHA256",
        DEFAULT_BASE_MODEL_INDEX_SHARD_BINDING_SHA256,
    )
    out.setdefault("baseModelArtifactDigest", DEFAULT_BASE_MODEL_ARTIFACT_DIGEST)
    out.setdefault("baseModelWeightShards", [dict(item) for item in DEFAULT_BASE_MODEL_WEIGHT_SHARDS])
    out.setdefault("baseModelTokenizerDigest", DEFAULT_BASE_MODEL_TOKENIZER_DIGEST)
    for key, value in default_training_lineage_contract().items():
        out.setdefault(key, value)
    out["artifactMode"] = "adapter_first"
    out["defaultExportArtifact"] = "lora_adapter"
    out["artifact_mode"] = "adapter_first"
    out["default_export_artifact"] = "lora_adapter"
    out["merge_adapters_by_default"] = False
    out["release_bake_enabled_by_default"] = False
    out["adapter_output_dir"] = adapter_output_dir(agent)
    out["output_dir"] = training_output_dir(agent)
    out["dpo_output_dir"] = dpo_output_dir(agent)
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
        "baseModelIndexDigest": out["baseModelIndexDigest"],
        "baseModelIndexReferencedShardNames": out["baseModelIndexReferencedShardNames"],
        "baseModelIndexShardBindingSHA256": out["baseModelIndexShardBindingSHA256"],
        "baseModelArtifactDigest": out["baseModelArtifactDigest"],
        "baseModelWeightShards": out["baseModelWeightShards"],
        "baseModelTokenizerDigest": out["baseModelTokenizerDigest"],
        "trainingCodeSHA256": out["trainingCodeSHA256"],
        "trainingCodeSHA256ByPhase": dict(out["trainingCodeSHA256ByPhase"]),
        "trainingCodeBundleSHA256": out["trainingCodeBundleSHA256"],
        "trainingDependencyLockSHA256": out["trainingDependencyLockSHA256"],
        "requirementsSHA256": out["requirementsSHA256"],
        **{field: out[field] for field in RUNTIME_SOURCE_AUDIT_FIELDS},
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
        "baseModelIndexDigest": config.get("baseModelIndexDigest", DEFAULT_BASE_MODEL_INDEX_DIGEST),
        "baseModelIndexReferencedShardNames": config.get(
            "baseModelIndexReferencedShardNames",
            list(DEFAULT_BASE_MODEL_INDEX_REFERENCED_SHARD_NAMES),
        ),
        "baseModelIndexShardBindingSHA256": config.get(
            "baseModelIndexShardBindingSHA256",
            DEFAULT_BASE_MODEL_INDEX_SHARD_BINDING_SHA256,
        ),
        "baseModelArtifactDigest": config.get("baseModelArtifactDigest", DEFAULT_BASE_MODEL_ARTIFACT_DIGEST),
        "baseModelWeightShards": config.get("baseModelWeightShards", DEFAULT_BASE_MODEL_WEIGHT_SHARDS),
        "baseModelTokenizerDigest": config.get("baseModelTokenizerDigest", DEFAULT_BASE_MODEL_TOKENIZER_DIGEST),
        "trainingCodeSHA256": config.get("trainingCodeSHA256"),
        "trainingCodeSHA256ByPhase": dict(
            config.get("trainingCodeSHA256ByPhase") or {}
        ),
        "trainingCodeBundleSHA256": config.get("trainingCodeBundleSHA256"),
        "trainingDependencyLockSHA256": config.get("trainingDependencyLockSHA256"),
        "requirementsSHA256": config.get("requirementsSHA256"),
        **{field: config.get(field) for field in RUNTIME_SOURCE_AUDIT_FIELDS},
        "sharedBaseRepoID": DEFAULT_SHARED_BASE_REPO_ID,
        "sharedBaseFileName": DEFAULT_SHARED_BASE_FILE_NAME,
        "adapterRepoID": DEFAULT_ADAPTER_REPO_ID,
        "adapterID": f"lumen-{agent}-adapter",
        "adapterArtifact": adapter_artifact_path(agent),
        "adapterDirectory": adapter_output_dir(agent),
        "adapterGGUFArtifact": adapter_gguf_output_path(agent),
        "systemPrompt": dataset_card.get("systemPrompt"),
        "datasetCard": {
            "sourceIntegrity": dataset_card.get("sourceIntegrity"),
            # Compatibility for existing adapter-plan consumers.
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
            "controlledVariables": list(
                (dataset_card.get("experimentPolicy") or {}).get(
                    "controlledVariables", []
                )
            ),
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
    training_code_digests: set[str] = set()
    training_code_phase_digests: list[dict[str, str] | None] = []
    training_code_bundle_digests: set[str] = set()
    dependency_lock_digests: set[str] = set()
    requirements_digests: set[str] = set()
    for agent, dataset in sorted(datasets.items()):
        unsloth_config = getattr(dataset, "unsloth_config", {}) or {}
        dataset_card = getattr(dataset, "dataset_card", {}) or {}
        base_model_id = base_model_id_from_config(unsloth_config)
        base_model_ids.add(base_model_id)
        for values, key in (
            (training_code_digests, "trainingCodeSHA256"),
            (training_code_bundle_digests, "trainingCodeBundleSHA256"),
            (dependency_lock_digests, "trainingDependencyLockSHA256"),
            (requirements_digests, "requirementsSHA256"),
        ):
            value = unsloth_config.get(key)
            if isinstance(value, str) and value:
                values.add(value)
        phase_digests = unsloth_config.get("trainingCodeSHA256ByPhase")
        training_code_phase_digests.append(
            dict(phase_digests) if isinstance(phase_digests, dict) else None
        )
        system_prompt = dataset_card.get("systemPrompt")
        role_contract_prompt = (
            system_prompt.strip()
            if isinstance(system_prompt, str) and system_prompt.strip()
            else None
        )
        offline_frozen_evaluation = dataset_card.get("evaluation", {})
        runtime_prompt_contract = {
            "schemaVersion": SHIPPED_RUNTIME_QUALIFICATION_SCHEMA_VERSION,
            "composerPolicyID": RUNTIME_PROMPT_COMPOSER_POLICY_ID,
            "composerPolicySHA256": RUNTIME_PROMPT_COMPOSER_POLICY_SHA256,
            "roleContractPromptSHA256": (
                prompt_sha256(role_contract_prompt)
                if role_contract_prompt is not None
                else None
            ),
            "roleContractPromptCharacterCount": (
                len(role_contract_prompt)
                if role_contract_prompt is not None
                else None
            ),
            "privacy": "hashes_and_character_counts_only",
            "requiredObservedEvidence": [
                "componentPromptSHA256",
                "sourcePromptSHA256",
                "effectivePromptSHA256",
                "composerPolicySHA256",
            ],
        }
        qualification_reasons = [
            "offline_frozen_evaluation_is_not_shipped_runtime_evidence",
            "missing_component_prompt_sha256",
            "missing_source_prompt_sha256",
            "missing_effective_prompt_sha256",
            "missing_observed_composer_policy_sha256",
        ]
        if role_contract_prompt is None:
            qualification_reasons.append("missing_role_contract_prompt_sha256")
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
                "baseModelIndexDigest": unsloth_config.get("baseModelIndexDigest", DEFAULT_BASE_MODEL_INDEX_DIGEST),
                "baseModelIndexReferencedShardNames": unsloth_config.get(
                    "baseModelIndexReferencedShardNames",
                    list(DEFAULT_BASE_MODEL_INDEX_REFERENCED_SHARD_NAMES),
                ),
                "baseModelIndexShardBindingSHA256": unsloth_config.get(
                    "baseModelIndexShardBindingSHA256",
                    DEFAULT_BASE_MODEL_INDEX_SHARD_BINDING_SHA256,
                ),
                "baseModelArtifactDigest": unsloth_config.get("baseModelArtifactDigest", DEFAULT_BASE_MODEL_ARTIFACT_DIGEST),
                "baseModelWeightShards": unsloth_config.get("baseModelWeightShards", DEFAULT_BASE_MODEL_WEIGHT_SHARDS),
                "baseModelTokenizerDigest": unsloth_config.get("baseModelTokenizerDigest", DEFAULT_BASE_MODEL_TOKENIZER_DIGEST),
                "trainingCodeSHA256": unsloth_config.get("trainingCodeSHA256"),
                "trainingCodeSHA256ByPhase": dict(
                    unsloth_config.get("trainingCodeSHA256ByPhase") or {}
                ),
                "trainingCodeBundleSHA256": unsloth_config.get("trainingCodeBundleSHA256"),
                "trainingDependencyLockSHA256": unsloth_config.get("trainingDependencyLockSHA256"),
                "requirementsSHA256": unsloth_config.get("requirementsSHA256"),
                **{
                    field: unsloth_config.get(field)
                    for field in RUNTIME_SOURCE_AUDIT_FIELDS
                },
                "systemPrompt": system_prompt,
                "runtimePromptContract": runtime_prompt_contract,
                "recordCounts": dataset_card.get("recordCounts", {}),
                "offlineFrozenEvaluation": {
                    "scope": "offline_frozen_adapter_suite",
                    "evidenceType": "frozen_suite_contract",
                    "executionStatus": "not_executed_by_runtime_manifest_exporter",
                    "qualifiesShippedRuntime": False,
                    "contract": offline_frozen_evaluation,
                },
                "shippedRuntimeQualification": {
                    "schemaVersion": SHIPPED_RUNTIME_QUALIFICATION_SCHEMA_VERSION,
                    "qualified": False,
                    "status": "unqualified_missing_runtime_evidence",
                    "reasonCodes": sorted(qualification_reasons),
                    "observedEvidence": {
                        "componentPromptSHA256": None,
                        "sourcePromptSHA256": None,
                        "effectivePromptSHA256": None,
                        "composerPolicySHA256": None,
                    },
                },
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
        "sharedBaseModelIndexDigest": DEFAULT_BASE_MODEL_INDEX_DIGEST,
        "sharedBaseModelIndexReferencedShardNames": list(
            DEFAULT_BASE_MODEL_INDEX_REFERENCED_SHARD_NAMES
        ),
        "sharedBaseModelIndexShardBindingSHA256": (
            DEFAULT_BASE_MODEL_INDEX_SHARD_BINDING_SHA256
        ),
        "sharedBaseModelArtifactDigest": DEFAULT_BASE_MODEL_ARTIFACT_DIGEST,
        "sharedBaseModelWeightShards": DEFAULT_BASE_MODEL_WEIGHT_SHARDS,
        "sharedBaseModelTokenizerDigest": DEFAULT_BASE_MODEL_TOKENIZER_DIGEST,
        "sharedTrainingCodeSHA256": (
            next(iter(training_code_digests))
            if len(training_code_digests) == 1
            else None
        ),
        "sharedTrainingCodeSHA256ByPhase": (
            training_code_phase_digests[0]
            if training_code_phase_digests
            and training_code_phase_digests[0] is not None
            and all(
                value == training_code_phase_digests[0]
                for value in training_code_phase_digests[1:]
            )
            else None
        ),
        "sharedTrainingCodeBundleSHA256": (
            next(iter(training_code_bundle_digests))
            if len(training_code_bundle_digests) == 1
            else None
        ),
        "sharedTrainingDependencyLockSHA256": (
            next(iter(dependency_lock_digests))
            if len(dependency_lock_digests) == 1
            else None
        ),
        "sharedRequirementsSHA256": (
            next(iter(requirements_digests))
            if len(requirements_digests) == 1
            else None
        ),
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
        "runtimeQualificationPolicy": {
            "schemaVersion": SHIPPED_RUNTIME_QUALIFICATION_SCHEMA_VERSION,
            "offlineFrozenEvaluationQualifiesShippedRuntime": False,
            "requiresExactComponentPromptSHA256": True,
            "requiresExactSourcePromptSHA256": True,
            "requiresExactEffectivePromptSHA256": True,
            "requiresExactComposerPolicySHA256": True,
            "missingOrMismatchedEvidence": "unqualified",
            "privacy": "hashes_and_character_counts_only",
        },
        "adapters": adapters,
        "releaseBakePolicy": {
            "enabledByDefault": False,
            "manualOnly": True,
            "requiresPassingEvalGates": True,
            "allowedWhenRuntimeCannotLoadAdapters": True,
        },
    }
