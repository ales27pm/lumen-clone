from __future__ import annotations

from types import SimpleNamespace

from lumen_manifest_crawler.dataset.adapter_evaluation import (
    RUNTIME_SOURCE_AUDIT_FIELDS,
)
from lumen_manifest_crawler.dataset.adapter_export import (
    ADAPTER_EXPORT_SCHEMA_VERSION,
    adapter_runtime_manifest,
    agent_adapter_export_plan,
    augment_unsloth_config_for_adapter_export,
)
from lumen_manifest_crawler.runtime_prompt_contract import (
    RUNTIME_PROMPT_COMPOSER_POLICY_SHA256,
    prompt_sha256,
)


def _dataset(config: dict) -> SimpleNamespace:
    return SimpleNamespace(
        unsloth_config=config,
        dataset_card={"systemPrompt": "You are the executor."},
    )


def test_export_surfaces_preserve_phase_code_and_runtime_source_audit() -> None:
    config = augment_unsloth_config_for_adapter_export("executor", {})
    plan = agent_adapter_export_plan("executor", {}, config)
    runtime = adapter_runtime_manifest({"executor": _dataset(config)})
    runtime_adapter = runtime["adapters"][0]

    assert ADAPTER_EXPORT_SCHEMA_VERSION == "1.5.0"
    assert runtime["schemaVersion"] == ADAPTER_EXPORT_SCHEMA_VERSION
    assert runtime["sharedTrainingCodeSHA256ByPhase"] == config[
        "trainingCodeSHA256ByPhase"
    ]

    for exported in (config["adapterExport"], plan, runtime_adapter):
        assert exported["trainingCodeSHA256"] == config["trainingCodeSHA256"]
        assert exported["trainingCodeSHA256ByPhase"] == config[
            "trainingCodeSHA256ByPhase"
        ]
        assert exported["trainingCodeBundleSHA256"] == config[
            "trainingCodeBundleSHA256"
        ]
        for field in RUNTIME_SOURCE_AUDIT_FIELDS:
            assert exported[field] == config[field]


def test_runtime_prompt_composer_policy_hash_is_cross_language_stable() -> None:
    assert RUNTIME_PROMPT_COMPOSER_POLICY_SHA256 == (
        "ef2ad84aa40487da2cc2f9432f333e880f979f79fd5de2ac9abfe0b6212b6540"
    )


def test_runtime_manifest_separates_offline_eval_from_shipped_prompt_qualification() -> None:
    config = augment_unsloth_config_for_adapter_export("executor", {})
    dataset = _dataset(config)
    dataset.dataset_card["evaluation"] = {
        "schemaVersion": "lumen.adapter-eval/1.0.0",
        "frozenEvaluationSHA256": "a" * 64,
        "recordCount": 3,
    }

    runtime = adapter_runtime_manifest({"executor": dataset})
    adapter = runtime["adapters"][0]

    assert "evaluation" not in adapter
    assert adapter["offlineFrozenEvaluation"] == {
        "scope": "offline_frozen_adapter_suite",
        "evidenceType": "frozen_suite_contract",
        "executionStatus": "not_executed_by_runtime_manifest_exporter",
        "qualifiesShippedRuntime": False,
        "contract": dataset.dataset_card["evaluation"],
    }
    prompt_contract = adapter["runtimePromptContract"]
    assert prompt_contract["roleContractPromptSHA256"] == prompt_sha256(
        "You are the executor."
    )
    assert prompt_contract["composerPolicySHA256"] == (
        RUNTIME_PROMPT_COMPOSER_POLICY_SHA256
    )
    assert "systemPrompt" not in prompt_contract
    assert adapter["shippedRuntimeQualification"]["qualified"] is False
    assert adapter["shippedRuntimeQualification"]["status"] == (
        "unqualified_missing_runtime_evidence"
    )
    assert "missing_effective_prompt_sha256" in adapter[
        "shippedRuntimeQualification"
    ]["reasonCodes"]
    assert runtime["runtimeQualificationPolicy"][
        "offlineFrozenEvaluationQualifiesShippedRuntime"
    ] is False


def test_runtime_manifest_does_not_claim_shared_phase_code_after_drift() -> None:
    executor = augment_unsloth_config_for_adapter_export("executor", {})
    cortex = augment_unsloth_config_for_adapter_export("cortex", {})
    cortex["trainingCodeSHA256ByPhase"] = {
        **cortex["trainingCodeSHA256ByPhase"],
        "dpo": "f" * 64,
    }

    runtime = adapter_runtime_manifest(
        {
            "executor": _dataset(executor),
            "cortex": _dataset(cortex),
        }
    )

    assert runtime["sharedTrainingCodeSHA256ByPhase"] is None
    by_agent = {item["agent"]: item for item in runtime["adapters"]}
    assert by_agent["executor"]["trainingCodeSHA256ByPhase"] == executor[
        "trainingCodeSHA256ByPhase"
    ]
    assert by_agent["cortex"]["trainingCodeSHA256ByPhase"] == cortex[
        "trainingCodeSHA256ByPhase"
    ]
