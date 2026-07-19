from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any


OPTIMIZATION_POLICY_SCHEMA_VERSION = "lumen.adapter-effective-steps/1.0.0"
EXPERIMENT_VARIANT_SCHEMA_VERSION = "lumen.adapter-experiment-variant/1.3.0"
TRAINING_CONFIG_INVARIANT_SCHEMA_VERSION = (
    "lumen.adapter-training-config-invariant/1.0.0"
)
NON_TRAINING_CONFIG_FIELDS = frozenset(
    {
        "adapterExport",
        "adapter_gguf_output_path",
        "adapter_output_dir",
        "dataset_dir",
        "dpo_output_dir",
        "gguf_output_dir",
        "gguf_repo_id",
        "mergeExport",
        "output_dir",
        "runtimeSourceKind",
        "runtimeSourceRevision",
        "expectedRuntimeSourceRevision",
        "observedRepositoryRevision",
        "observedRuntimeRevision",
        "runtimeSourceBindingStatus",
        "runtimeSourceBindingMethod",
        "workingTreeDigest",
        "ubuntuOrchestrationCodeSHA256",
        "ubuntuSourceIntegritySHA256",
        "ubuntuSourceIntegrity",
        "spaceConfigurationSHA256",
        "zeroGPUSize",
        "zeroGPUDurationSeconds",
        "observedAccelerator",
    }
)
VARIANT_SPECIFIC_TRAINING_CONFIG_FIELDS = frozenset(
    {
        "optimizationStepPolicy",
        "num_train_epochs",
        "dpo_num_train_epochs",
    }
)
VARIANT_DERIVED_TRAINING_CONFIG_PATHS = (
    "dpo_num_train_epochs",
    "num_train_epochs",
    "optimizationStepPolicy.dpo.effectiveStepsPerEpoch",
    "optimizationStepPolicy.dpo.projectedEffectiveSteps",
    "optimizationStepPolicy.dpo.selectedEpochs",
    "optimizationStepPolicy.dpo.trainRecordCount",
    "optimizationStepPolicy.sft.effectiveStepsPerEpoch",
    "optimizationStepPolicy.sft.projectedEffectiveSteps",
    "optimizationStepPolicy.sft.selectedEpochs",
    "optimizationStepPolicy.sft.trainRecordCount",
)
_DERIVED_POSITIVE_INTEGER = {
    "normalization": "variant_derived_positive_integer"
}
_AGENTS = frozenset({"cortex", "executor", "mouth", "mimicry", "rem", "fleet"})
_MINIMUM_EFFECTIVE_STEPS_BY_LANE = {
    "sft": {
        "executor": 40,
        "mouth": 24,
        "mimicry": 20,
        "rem": 20,
        "fleet": 24,
    },
    "dpo": {
        "executor": 8,
        "mouth": 9,
        "mimicry": 8,
        "rem": 8,
        "fleet": 8,
    },
}
_GRADIENT_ACCUMULATION_STEPS = {
    "cortex": 16,
    "executor": 8,
    "mouth": 4,
    "mimicry": 2,
    "rem": 4,
    "fleet": 8,
}
_MAXIMUM_NON_CORTEX_EPOCHS = 8
_POLICY_KEYS = frozenset(
    {
        "schemaVersion",
        "mode",
        "batchSize",
        "gradientAccumulationSteps",
        "sft",
        "dpo",
        "maximumEpochs",
    }
)
_LANE_KEYS = frozenset(
    {
        "trainRecordCount",
        "baseEpochs",
        "selectedEpochs",
        "effectiveStepsPerEpoch",
        "minimumEffectiveSteps",
        "projectedEffectiveSteps",
        "minimumSatisfied",
    }
)


def _positive_integer(value: Any) -> bool:
    return type(value) is int and value > 0


def expected_optimization_step_policy(
    *,
    agent: str,
    sft_train_record_count: int,
    dpo_train_record_count: int,
) -> dict[str, Any]:
    if agent not in _AGENTS:
        raise ValueError(f"Unsupported adapter optimization agent: {agent}")
    if not _positive_integer(sft_train_record_count):
        raise ValueError("SFT optimization requires a positive training-record count")
    if not _positive_integer(dpo_train_record_count):
        raise ValueError("DPO optimization requires a positive training-record count")

    batch_size = 1 if agent in {"cortex", "fleet"} else 2
    gradient_accumulation_steps = _GRADIENT_ACCUMULATION_STEPS[agent]
    high_reasoning = agent in {"cortex", "executor", "rem"}
    base_epochs = {
        "sft": (
            3
            if agent in {"cortex", "fleet"}
            else 2
            if high_reasoning
            else 1
        ),
        "dpo": (
            1
            if agent == "cortex"
            else 2
            if high_reasoning
            else 1
        ),
    }
    maximum_epochs = None if agent == "cortex" else _MAXIMUM_NON_CORTEX_EPOCHS
    lanes: dict[str, dict[str, Any]] = {}
    for lane, record_count in (
        ("sft", sft_train_record_count),
        ("dpo", dpo_train_record_count),
    ):
        micro_batches = (record_count + batch_size - 1) // batch_size
        steps_per_epoch = (
            micro_batches + gradient_accumulation_steps - 1
        ) // gradient_accumulation_steps
        minimum_steps = (
            None
            if agent == "cortex"
            else _MINIMUM_EFFECTIVE_STEPS_BY_LANE[lane][agent]
        )
        if agent == "cortex":
            selected_epochs = base_epochs[lane]
        else:
            assert minimum_steps is not None
            selected_epochs = max(
                base_epochs[lane],
                (minimum_steps + steps_per_epoch - 1) // steps_per_epoch,
            )
            if selected_epochs > _MAXIMUM_NON_CORTEX_EPOCHS:
                raise ValueError(
                    f"{agent} {lane} optimization lane exceeds the epoch cap"
                )
        projected_steps = steps_per_epoch * selected_epochs
        lanes[lane] = {
            "trainRecordCount": record_count,
            "baseEpochs": base_epochs[lane],
            "selectedEpochs": selected_epochs,
            "effectiveStepsPerEpoch": steps_per_epoch,
            "minimumEffectiveSteps": minimum_steps,
            "projectedEffectiveSteps": projected_steps,
            "minimumSatisfied": (
                True
                if minimum_steps is None
                else projected_steps >= minimum_steps
            ),
        }
    return {
        "schemaVersion": OPTIMIZATION_POLICY_SCHEMA_VERSION,
        "mode": (
            "cortex_empirical_fixed"
            if agent == "cortex"
            else "non_cortex_minimum_effective_steps"
        ),
        "batchSize": batch_size,
        "gradientAccumulationSteps": gradient_accumulation_steps,
        "sft": lanes["sft"],
        "dpo": lanes["dpo"],
        "maximumEpochs": maximum_epochs,
    }


def _validated_policy(
    config: Mapping[str, Any],
    *,
    agent: str | None,
    sft_train_record_count: int | None,
    dpo_train_record_count: int | None,
) -> dict[str, Any]:
    configured_agent = config.get("agent")
    resolved_agent = agent or configured_agent
    if resolved_agent not in _AGENTS or (
        configured_agent is not None and configured_agent != resolved_agent
    ):
        raise ValueError("Training config optimization agent is invalid")
    policy = config.get("optimizationStepPolicy")
    if not isinstance(policy, Mapping) or set(policy) != _POLICY_KEYS:
        raise ValueError("Optimization-step policy has invalid top-level fields")
    if not _positive_integer(policy.get("batchSize")) or not _positive_integer(
        policy.get("gradientAccumulationSteps")
    ):
        raise ValueError("Optimization-step policy has invalid batch geometry")
    maximum_epochs = policy.get("maximumEpochs")
    if maximum_epochs is not None and not _positive_integer(maximum_epochs):
        raise ValueError("Optimization-step policy has an invalid epoch cap")
    if not _positive_integer(config.get("batch_size")) or not _positive_integer(
        config.get("gradient_accumulation_steps")
    ):
        raise ValueError("Training config has invalid batch geometry")
    if (
        config["batch_size"] != policy["batchSize"]
        or config["gradient_accumulation_steps"]
        != policy["gradientAccumulationSteps"]
    ):
        raise ValueError("Training config and optimization policy batch geometry differ")
    for lane in ("sft", "dpo"):
        lane_policy = policy.get(lane)
        if not isinstance(lane_policy, Mapping) or set(lane_policy) != _LANE_KEYS:
            raise ValueError(f"Optimization-step policy has invalid {lane} fields")
        if not all(
            _positive_integer(lane_policy.get(field))
            for field in (
                "trainRecordCount",
                "baseEpochs",
                "selectedEpochs",
                "effectiveStepsPerEpoch",
                "projectedEffectiveSteps",
            )
        ):
            raise ValueError(f"Optimization-step policy has invalid {lane} integers")
        minimum = lane_policy.get("minimumEffectiveSteps")
        if minimum is not None and not _positive_integer(minimum):
            raise ValueError(f"Optimization-step policy has invalid {lane} minimum")
        if lane_policy.get("minimumSatisfied") is not True:
            raise ValueError(f"Optimization-step policy has unsatisfied {lane} minimum")
    if type(config.get("num_train_epochs")) is not int or not _positive_integer(
        config.get("num_train_epochs")
    ):
        raise ValueError("SFT epoch count must be a positive integer")
    if type(config.get("dpo_num_train_epochs")) is not int or not _positive_integer(
        config.get("dpo_num_train_epochs")
    ):
        raise ValueError("DPO epoch count must be a positive integer")

    expected = expected_optimization_step_policy(
        agent=resolved_agent,
        sft_train_record_count=(
            sft_train_record_count
            if sft_train_record_count is not None
            else policy["sft"]["trainRecordCount"]
        ),
        dpo_train_record_count=(
            dpo_train_record_count
            if dpo_train_record_count is not None
            else policy["dpo"]["trainRecordCount"]
        ),
    )
    if dict(policy) != expected:
        raise ValueError("Optimization-step policy does not match its training lanes")
    if (
        config["num_train_epochs"] != expected["sft"]["selectedEpochs"]
        or config["dpo_num_train_epochs"]
        != expected["dpo"]["selectedEpochs"]
    ):
        raise ValueError("Training epochs do not match the optimization-step policy")
    return expected


def invariant_training_config(
    config: Mapping[str, Any],
    *,
    agent: str | None = None,
    sft_train_record_count: int | None = None,
    dpo_train_record_count: int | None = None,
) -> dict[str, Any]:
    """Return a domain-separated config with only derived integers normalized."""

    present = VARIANT_SPECIFIC_TRAINING_CONFIG_FIELDS & set(config)
    if present and present != VARIANT_SPECIFIC_TRAINING_CONFIG_FIELDS:
        missing = sorted(VARIANT_SPECIFIC_TRAINING_CONFIG_FIELDS - present)
        raise ValueError(
            "Controlled training config has partial variant optimizer state: "
            + ", ".join(missing)
        )
    normalized = dict(sorted(config.items()))
    derived_paths: tuple[str, ...] = ()
    if present:
        policy = _validated_policy(
            config,
            agent=agent,
            sft_train_record_count=sft_train_record_count,
            dpo_train_record_count=dpo_train_record_count,
        )
        normalized_policy = dict(policy)
        for lane in ("sft", "dpo"):
            normalized_lane = dict(policy[lane])
            for field in (
                "trainRecordCount",
                "selectedEpochs",
                "effectiveStepsPerEpoch",
                "projectedEffectiveSteps",
            ):
                normalized_lane[field] = dict(_DERIVED_POSITIVE_INTEGER)
            normalized_policy[lane] = normalized_lane
        normalized["optimizationStepPolicy"] = normalized_policy
        normalized["num_train_epochs"] = dict(_DERIVED_POSITIVE_INTEGER)
        normalized["dpo_num_train_epochs"] = dict(_DERIVED_POSITIVE_INTEGER)
        derived_paths = VARIANT_DERIVED_TRAINING_CONFIG_PATHS
    return {
        "schemaVersion": TRAINING_CONFIG_INVARIANT_SCHEMA_VERSION,
        "variantDerivedFieldPaths": list(derived_paths),
        "normalizedConfig": normalized,
    }


def effective_variant_training_config(
    *,
    agent: str,
    base_config: Mapping[str, Any],
    controlled_config: Mapping[str, Any],
    noncontrolled_fields: set[str] | frozenset[str],
    sft_train_record_count: int,
    dpo_train_record_count: int,
) -> dict[str, Any]:
    """Validate and overlay the exact three dataset-derived config fields."""

    base_controlled_keys = set(base_config) - set(noncontrolled_fields)
    if set(controlled_config) != base_controlled_keys:
        raise ValueError(
            "Variant controlled training config fields drifted from the base config"
        )

    def canonical(value: Any) -> str:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )

    for key in sorted(
        base_controlled_keys - VARIANT_SPECIFIC_TRAINING_CONFIG_FIELDS
    ):
        if canonical(controlled_config[key]) != canonical(base_config[key]):
            raise ValueError(
                f"Variant controlled training config changed non-variant field: {key}"
            )
    base_controlled = {
        key: base_config[key]
        for key in sorted(base_controlled_keys)
    }
    base_invariant = invariant_training_config(base_controlled, agent=agent)
    controlled_invariant = invariant_training_config(
        controlled_config,
        agent=agent,
        sft_train_record_count=sft_train_record_count,
        dpo_train_record_count=dpo_train_record_count,
    )
    if canonical(base_invariant) != canonical(controlled_invariant):
        raise ValueError(
            "Variant invariant training config differs from the base config"
        )
    effective = dict(base_config)
    for field in VARIANT_SPECIFIC_TRAINING_CONFIG_FIELDS:
        effective[field] = controlled_config[field]
    return effective
