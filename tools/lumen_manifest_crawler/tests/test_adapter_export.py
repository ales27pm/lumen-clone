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

    assert ADAPTER_EXPORT_SCHEMA_VERSION == "1.4.0"
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
