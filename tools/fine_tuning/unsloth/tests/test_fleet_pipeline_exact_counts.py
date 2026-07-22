from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from lumen_manifest_crawler.dataset.fine_tuning import (
    FineTuningDatasetConfig,
    _fleet_loss_share_contract,
)
from tools.fine_tuning.unsloth import train_sft, ubuntu_pipeline


SPECIFICATION = [
    ("native", 55),
    ("policy_bridge", 5),
    ("supplemental_a", 5),
    ("supplemental_b", 5),
    ("public", 30),
]

DPO_SPECIFICATION = [
    ("native", 55),
    ("primary", 4),
    ("supplemental_a", 3),
    ("supplemental_b", 3),
    ("public", 4),
]

_CALLABLE_IDENTITY_UNSIGNED = {
    "schemaVersion": "lumen.installed-python-callable-identity/1.0.0",
    "resolvedTrainingEnvironmentSHA256": "e" * 64,
    "distributionName": "unsloth-zoo",
    "distributionVersion": "2026.7.2",
    "distributionSHA256": "2" * 64,
    "sourceLogicalPath": "unsloth_zoo/loss_utils.py",
    "sourceFileSize": 1234,
    "sourceFileSHA256": "3" * 64,
    "callableName": "_unsloth_get_batch_samples",
    "callableQualname": "_unsloth_get_batch_samples",
    "callableFirstLineNumber": 99,
    "codeSHA256": "b" * 64,
}
_CALLABLE_IDENTITY = {
    **_CALLABLE_IDENTITY_UNSIGNED,
    "installedCallableIdentitySHA256": ubuntu_pipeline.canonical_sha256(
        _CALLABLE_IDENTITY_UNSIGNED
    ),
}


def _metadata(kind: str) -> dict[str, Any]:
    if kind == "native":
        return {
            "sourceFamily": "fleet_orchestration_native",
            "taskType": "fleet_orchestration_event_graph",
        }
    if kind == "primary":
        return {
            "sourceFamily": "adapter_ultra_specific",
            "taskType": "delegation_protocol",
        }
    if kind == "policy_bridge":
        return {
            "sourceFamily": "adapter_ultra_specific",
            "taskType": "fleet_contract_event_graph_vocabulary",
        }
    if kind == "supplemental_a":
        return {
            "sourceFamily": "codebase_home_sft",
            "taskType": "codebase_home_grounding",
        }
    if kind == "supplemental_b":
        return {
            "sourceFamily": "self_model_sft",
            "taskType": "self_model_grounded_answer",
        }
    if kind == "public":
        return {
            "sourceFamily": "public_adapter_corpus_fixture",
            "taskType": "public_capability_delegation",
            "publicCorpus": {"sourceArtifactSHA256": "c" * 64},
        }
    raise AssertionError(kind)


def _rows() -> list[dict[str, Any]]:
    return [
        {
            "messages": [
                {"role": "user", "content": f"request {index}"},
                {
                    "role": "assistant",
                    "content": " ".join(
                        f"answer_{index}_{word}" for word in range(token_count)
                    ),
                },
            ],
            "metadata": _metadata(kind),
        }
        for index, (kind, token_count) in enumerate(SPECIFICATION)
    ]


def _dpo_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, (kind, token_count) in enumerate(DPO_SPECIFICATION):
        metadata = _metadata(kind)
        if kind == "native":
            metadata = {
                "sourceFamily": "fleet_orchestration_native",
                "taskType": "fleet_orchestration_event_graph_preference",
            }
        rows.append(
            {
                "prompt": [{"role": "user", "content": f"request {index}"}],
                "chosen": {
                    "role": "assistant",
                    "content": " ".join(
                        f"chosen_{index}_{word}"
                        for word in range(token_count)
                    ),
                },
                "rejected": {"role": "assistant", "content": "rejected"},
                "metadata": metadata,
            }
        )
    return rows


class _ExactMaskTokenizer:
    chat_template = "{% generation %}"
    eos_token_id = 151_645

    def __call__(
        self,
        value: str,
        *,
        add_special_tokens: bool,
    ) -> dict[str, list[int]]:
        assert add_special_tokens is False
        return {"input_ids": list(range(len(value.split())))}

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        return_dict: bool = False,
        return_assistant_tokens_mask: bool = False,
        enable_thinking: bool,
    ) -> Any:
        del add_generation_prompt, return_assistant_tokens_mask, enable_thinking
        input_ids: list[int] = []
        assistant_masks: list[int] = []
        for message in messages:
            token_count = len(message["content"].split())
            input_ids.extend(
                range(len(input_ids), len(input_ids) + token_count)
            )
            assistant_masks.extend(
                [1 if message["role"] == "assistant" else 0] * token_count
            )
        if tokenize and return_dict:
            return {
                "input_ids": input_ids,
                "attention_mask": [1] * len(input_ids),
                "assistant_masks": assistant_masks,
            }
        if tokenize:
            return input_ids
        return "rendered"


class _ExactTextTokenizer:
    eos_token_id = 151_645

    def __call__(
        self,
        value: str,
        *,
        add_special_tokens: bool,
    ) -> dict[str, list[int]]:
        assert add_special_tokens is False
        return {"input_ids": list(range(len(value.split())))}


def _render_preference(
    row: dict[str, Any],
    *,
    tokenizer: Any,
) -> dict[str, str]:
    assert isinstance(tokenizer, _ExactTextTokenizer)

    def text(value: Any) -> str:
        if isinstance(value, list):
            return " ".join(
                str(message.get("content") or "")
                for message in value
                if isinstance(message, dict)
            )
        return str(value)

    return {
        "prompt": text(row["prompt"]),
        "chosen": text(row["chosen"]),
        "rejected": text(row["rejected"]),
    }


def _config() -> dict[str, Any]:
    contract = _fleet_loss_share_contract(FineTuningDatasetConfig())
    tokenizer = contract["tokenizer"]
    return {
        "agent": "fleet",
        "seed": 42,
        "batch_size": 1,
        "gradient_accumulation_steps": 8,
        "num_train_epochs": 4,
        "packing": False,
        "optimizationStepPolicy": {
            "sft": {"trainRecordCount": 5},
        },
        "base_model_name": tokenizer["baseModelID"],
        "baseModelRevision": tokenizer["baseModelRevision"],
        "baseModelTokenizerDigest": tokenizer["tokenizerSHA256"],
        "baseModelTokenizerClosureSHA256": tokenizer[
            "tokenizerClosureSHA256"
        ],
        "trainingCodeSHA256": "d" * 64,
        "resolvedTrainingEnvironment": {"fixture": True},
        "resolvedTrainingEnvironmentSHA256": "e" * 64,
        "trainingEnvironmentSHA256": "f" * 64,
        "trainingContainerImageDigest": f"sha256:{'1' * 64}",
        "fleetLossShareContract": contract,
    }


def _evidence(
    rows: list[dict[str, Any]],
    counts: list[int],
    config: dict[str, Any],
) -> dict[str, Any]:
    target_rows = list(zip(rows, counts))
    return train_sft._build_fleet_loss_share_evidence(
        contract_value=config["fleetLossShareContract"],
        lane="sft",
        split_target_rows={
            "train": target_rows,
            "validation": target_rows,
        },
        config=config,
    )


def _dpo_evidence(
    rows: list[dict[str, Any]],
    counts: list[int],
    config: dict[str, Any],
) -> dict[str, Any]:
    target_rows = list(zip(rows, counts))
    return train_sft._build_fleet_loss_share_evidence(
        contract_value=config["fleetLossShareContract"],
        lane="dpo",
        split_target_rows={
            "train": target_rows,
            "validation": target_rows,
        },
        config=config,
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def test_runtime_rejects_self_consistent_rewritten_fleet_counts_and_schedule(
    tmp_path: Path,
) -> None:
    config = _config()
    rows = _rows()
    dataset_dir = tmp_path / "fleet"
    _write_jsonl(dataset_dir / "train_sft.jsonl", rows)
    _write_jsonl(dataset_dir / "val_sft.jsonl", rows)

    # These fabricated counts still satisfy every aggregate cap and are used to
    # regenerate a perfectly self-consistent optimizer-window schedule.
    fabricated = _evidence(rows, [50, 10, 5, 5, 30], config)
    assert ubuntu_pipeline._verify_fleet_loss_share_evidence(
        value=fabricated,
        config=config,
        phase="sft",
        dataset_dir=dataset_dir,
    ) == fabricated

    with pytest.raises(
        RuntimeError,
        match="Fleet sft train exact-token row evidence drifted",
    ):
        ubuntu_pipeline._verify_fleet_loss_share_evidence(
            value=fabricated,
            config=config,
            phase="sft",
            dataset_dir=dataset_dir,
            tokenizer=_ExactMaskTokenizer(),
            require_exact_tokenizer_counts=True,
        )


def test_runtime_exactly_retokenizes_fleet_dpo_chosen_completions(
    tmp_path: Path,
) -> None:
    config = _config()
    rows = _dpo_rows()
    dataset_dir = tmp_path / "fleet-dpo"
    _write_jsonl(dataset_dir / "train_dpo.jsonl", rows)
    _write_jsonl(dataset_dir / "val_dpo.jsonl", rows)

    # Exact chosen counts are [56, 5, 4, 4, 5] after TRL's appended EOS.
    fabricated = _dpo_evidence(rows, [55, 5, 4, 4, 5], config)
    assert ubuntu_pipeline._verify_fleet_loss_share_evidence(
        value=fabricated,
        config=config,
        phase="preference",
        dataset_dir=dataset_dir,
    ) == fabricated

    with pytest.raises(
        RuntimeError,
        match="Fleet dpo train exact-token row evidence drifted",
    ):
        ubuntu_pipeline._verify_fleet_loss_share_evidence(
            value=fabricated,
            config=config,
            phase="preference",
            dataset_dir=dataset_dir,
            tokenizer=_ExactTextTokenizer(),
            preference_renderer=_render_preference,
            require_exact_tokenizer_counts=True,
        )


def test_fleet_verifier_applies_the_same_sft_record_limits_as_training(
    tmp_path: Path,
) -> None:
    config = _config()
    config.update({"max_train_records": 5, "max_val_records": 5})
    selected_rows = _rows()
    full_rows = selected_rows + copy.deepcopy(selected_rows)
    evidence = _evidence(
        selected_rows,
        [token_count for _, token_count in SPECIFICATION],
        config,
    )
    dataset_dir = tmp_path / "fleet-limited"
    _write_jsonl(dataset_dir / "train_sft.jsonl", full_rows)
    _write_jsonl(dataset_dir / "val_sft.jsonl", full_rows)

    assert ubuntu_pipeline._verify_fleet_loss_share_evidence(
        value=evidence,
        config=config,
        phase="sft",
        dataset_dir=dataset_dir,
        tokenizer=_ExactMaskTokenizer(),
        require_exact_tokenizer_counts=True,
    ) == evidence

    unlimited = dict(config)
    unlimited.pop("max_train_records")
    unlimited.pop("max_val_records")
    with pytest.raises(RuntimeError, match="evidence row count drifted"):
        ubuntu_pipeline._verify_fleet_loss_share_evidence(
            value=evidence,
            config=unlimited,
            phase="sft",
            dataset_dir=dataset_dir,
        )


def _runtime_normalization_fixture() -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    config = _config()
    scheduled_indices_sha256 = "a" * 64
    target_count = 80
    preflight = {
        "fleetLossShareEvidence": {
            "splits": {
                "train": {
                    "records": 8,
                    "optimizerWindowSchedule": {
                        "datasetRecordsPerEpoch": 8,
                        "epochs": [
                            {
                                "firstOptimizerWindowRecordIndicesSHA256": (
                                    scheduled_indices_sha256
                                ),
                                "firstOptimizerWindowTargetTokenCount": (
                                    target_count
                                ),
                            }
                        ],
                    },
                }
            }
        }
    }
    unsigned = {
        "schemaVersion": (
            "lumen.fleet-sft-runtime-loss-normalization/1.2.0"
        ),
        "status": "passed",
        "enforcementPhase": "post_trainer_init_pre_optimizer",
        "countingRule": (
            "sum_shifted_non_ignored_labels_intersect_shifted_attention_mask"
        ),
        "trainingCodeSHA256": config["trainingCodeSHA256"],
        "resolvedTrainingEnvironmentSHA256": config[
            "resolvedTrainingEnvironmentSHA256"
        ],
        "trainingEnvironmentSHA256": config[
            "trainingEnvironmentSHA256"
        ],
        "trainingContainerImageDigest": config[
            "trainingContainerImageDigest"
        ],
        "trainerClass": "__main__._FleetSFTTrainer",
        "modelClass": "peft.peft_model.PeftModelForCausalLM",
        "baseModelClass": (
            "transformers.models.qwen3.modeling_qwen3.Qwen3ForCausalLM"
        ),
        "baseModelType": "qwen3",
        "getBatchSamples": {
            "module": "unsloth_zoo.loss_utils",
            "name": "_unsloth_get_batch_samples",
            "installedCallableIdentity": copy.deepcopy(
                _CALLABLE_IDENTITY
            ),
        },
        "modelAcceptsLossKwargs": True,
        "lossType": "nll",
        "packing": False,
        "sftConfigPaddingFree": False,
        "trainerArgsPaddingFree": False,
        "trainerPaddingFree": False,
        "dataCollatorPaddingFree": False,
        "batchCollationMode": "padded_attention_mask",
        "packedSequenceLengthsPresent": False,
        "positionIDsPresent": False,
        "attentionMaskPresent": True,
        "observedMicroBatchSizes": [1] * 8,
        "worldSize": 1,
        "perDeviceTrainBatchSize": 1,
        "trainerGradientAccumulationSteps": 8,
        "acceleratorGradientAccumulationSteps": 1,
        "optimizerWindowRecordCapacity": 8,
        "optimizerWindowMicroBatchCount": 8,
        "scheduledRowIndicesSHA256": scheduled_indices_sha256,
        "expectedMicroBatchTargetTokenCounts": [10] * 8,
        "observedMicroBatchTargetTokenCounts": [10] * 8,
        "expectedTargetTokenCount": target_count,
        "reconstructedTargetTokenCount": target_count,
        "reportedNumItemsInBatch": target_count,
        "samplerStateSHA256": "c" * 64,
        "samplerStatePreserved": True,
        "rngStateRestored": True,
        "preOptimizerStateVerified": True,
    }
    evidence = {
        **unsigned,
        "runtimeLossNormalizationSHA256": (
            ubuntu_pipeline.canonical_sha256(unsigned)
        ),
    }
    return config, preflight, {
        "fleetRuntimeLossNormalization": evidence
    }


def _patch_callable_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        ubuntu_pipeline,
        "installed_distribution_python_callable_identity",
        lambda **kwargs: copy.deepcopy(_CALLABLE_IDENTITY),
    )


def test_pipeline_binds_runtime_normalization_to_attested_first_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_callable_identity(monkeypatch)
    config, preflight, report = _runtime_normalization_fixture()

    observed = ubuntu_pipeline._verify_fleet_sft_runtime_loss_normalization(
        report=report,
        config=config,
        token_length_preflight=preflight,
    )

    assert observed == report["fleetRuntimeLossNormalization"]


def test_pipeline_schedule_reconstruction_requires_batch_size_one() -> None:
    config = _config()
    config["batch_size"] = 2

    with pytest.raises(RuntimeError, match="controls are invalid"):
        ubuntu_pipeline._pipeline_fleet_sft_optimizer_window_schedule(
            row_token_evidence=[
                {
                    "rowIndex": 0,
                    "targetTokenCount": 10,
                    "sourceRowSHA256": "a" * 64,
                },
                {
                    "rowIndex": 1,
                    "targetTokenCount": 10,
                    "sourceRowSHA256": "b" * 64,
                },
            ],
            config=config,
            schedule_contract={},
            minimum_basis_points=5_000,
            maximum_basis_points=6_000,
        )


def test_pipeline_reconstructs_family_round_robin_schedule(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    row_evidence = [
        {
            "rowIndex": row_index,
            "sourceRowSHA256": ubuntu_pipeline.canonical_sha256(
                {"rowIndex": row_index, "targetTokenCount": token_count}
            ),
            "sourceFamily": (
                "fleet_orchestration_native"
                if row_index < 10
                else "adapter_ultra_specific"
            ),
            "taskType": (
                "fleet_orchestration_event_graph"
                if row_index < 10
                else "delegation_protocol"
            ),
            "category": "behavioral_primary",
            "targetTokenCount": token_count,
        }
        for row_index, token_count in enumerate(
            [51] * 10 + [4, 5, 6, 7, 8, 9, 10] * 10
        )
    ]
    contract = config["fleetLossShareContract"]
    schedule_contract = contract["sftOptimizerWindowScheduleContract"]
    band = contract["optimizerFamilyShareBands"]["lanes"]["sft"]

    trainer_schedule, trainer_epoch_orders = (
        train_sft._build_fleet_sft_optimizer_window_schedule(
            row_token_evidence=row_evidence,
            config=config,
            schedule_contract=schedule_contract,
            minimum_basis_points=band["minimumBasisPoints"],
            maximum_basis_points=band["maximumBasisPoints"],
        )
    )
    original_canonical_sha256 = ubuntu_pipeline.canonical_sha256
    observed_key_sets: dict[str, set[frozenset[str]]] = {}
    ranked_roles = {
        "native_source_record",
        "non_native_source_record",
        "window_record",
    }

    def capture_rank_payload(value: Any) -> str:
        if isinstance(value, dict) and value.get("role") in ranked_roles:
            role = str(value["role"])
            observed_key_sets.setdefault(role, set()).add(
                frozenset(value)
            )
        return original_canonical_sha256(value)

    monkeypatch.setattr(
        ubuntu_pipeline,
        "canonical_sha256",
        capture_rank_payload,
    )
    verifier_schedule = (
        ubuntu_pipeline._pipeline_fleet_sft_optimizer_window_schedule(
            row_token_evidence=row_evidence,
            config=config,
            schedule_contract=schedule_contract,
            minimum_basis_points=band["minimumBasisPoints"],
            maximum_basis_points=band["maximumBasisPoints"],
        )
    )

    assert verifier_schedule == trainer_schedule
    assert observed_key_sets == {
        "native_source_record": {
            frozenset(
                {"algorithm", "seed", "role", "sourceRowSHA256"}
            )
        },
        "non_native_source_record": {
            frozenset(
                {"algorithm", "seed", "role", "sourceRowSHA256"}
            )
        },
        "window_record": {
            frozenset(
                {
                    "algorithm",
                    "seed",
                    "epochIndex",
                    "candidateIndex",
                    "role",
                    "windowIndex",
                    "sourceRowSHA256",
                }
            )
        },
    }

    permuted_evidence = [
        {
            **copy.deepcopy(row_evidence[source_index]),
            "rowIndex": row_index,
        }
        for row_index, source_index in enumerate(
            reversed(range(len(row_evidence)))
        )
    ]
    permuted_trainer_schedule, permuted_epoch_orders = (
        train_sft._build_fleet_sft_optimizer_window_schedule(
            row_token_evidence=permuted_evidence,
            config=config,
            schedule_contract=schedule_contract,
            minimum_basis_points=band["minimumBasisPoints"],
            maximum_basis_points=band["maximumBasisPoints"],
        )
    )
    permuted_verifier_schedule = (
        ubuntu_pipeline._pipeline_fleet_sft_optimizer_window_schedule(
            row_token_evidence=permuted_evidence,
            config=config,
            schedule_contract=schedule_contract,
            minimum_basis_points=band["minimumBasisPoints"],
            maximum_basis_points=band["maximumBasisPoints"],
        )
    )

    assert permuted_verifier_schedule == permuted_trainer_schedule
    assert [
        [row_evidence[row_index]["sourceRowSHA256"] for row_index in order]
        for order in trainer_epoch_orders
    ] == [
        [
            permuted_evidence[row_index]["sourceRowSHA256"]
            for row_index in order
        ]
        for order in permuted_epoch_orders
    ]


def test_pipeline_runtime_verifier_requires_batch_size_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_callable_identity(monkeypatch)
    config, preflight, report = _runtime_normalization_fixture()
    config["batch_size"] = 2
    train_split = preflight["fleetLossShareEvidence"]["splits"]["train"]
    train_split["records"] = 16
    train_split["optimizerWindowSchedule"]["datasetRecordsPerEpoch"] = 16
    evidence = report["fleetRuntimeLossNormalization"]
    evidence["perDeviceTrainBatchSize"] = 2
    evidence["optimizerWindowRecordCapacity"] = 16
    evidence["observedMicroBatchSizes"] = [2] * 8
    unsigned = dict(evidence)
    unsigned.pop("runtimeLossNormalizationSHA256")
    evidence["runtimeLossNormalizationSHA256"] = (
        ubuntu_pipeline.canonical_sha256(unsigned)
    )

    with pytest.raises(RuntimeError, match="failed verification"):
        ubuntu_pipeline._verify_fleet_sft_runtime_loss_normalization(
            report=report,
            config=config,
            token_length_preflight=preflight,
        )


def test_pipeline_rejects_self_hashed_runtime_denominator_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_callable_identity(monkeypatch)
    config, preflight, report = _runtime_normalization_fixture()
    forged = copy.deepcopy(report["fleetRuntimeLossNormalization"])
    forged["observedMicroBatchTargetTokenCounts"] = [9, 11, *([10] * 6)]
    unsigned = dict(forged)
    unsigned.pop("runtimeLossNormalizationSHA256")
    forged["runtimeLossNormalizationSHA256"] = (
        ubuntu_pipeline.canonical_sha256(unsigned)
    )

    with pytest.raises(RuntimeError, match="failed verification"):
        ubuntu_pipeline._verify_fleet_sft_runtime_loss_normalization(
            report={"fleetRuntimeLossNormalization": forged},
            config=config,
            token_length_preflight=preflight,
        )


@pytest.mark.parametrize(
    ("field_path", "replacement"),
    [
        (("getBatchSamples", "installedCallableIdentity", "codeSHA256"), "9" * 64),
        (("modelClass",), "forged.Model"),
        (("resolvedTrainingEnvironmentSHA256",), "8" * 64),
        (("trainingContainerImageDigest",), f"sha256:{'7' * 64}"),
        (("sftConfigPaddingFree",), True),
        (("trainerArgsPaddingFree",), True),
        (("trainerPaddingFree",), True),
        (("dataCollatorPaddingFree",), True),
        (("batchCollationMode",), "padding_free"),
        (("packedSequenceLengthsPresent",), True),
        (("positionIDsPresent",), True),
        (("attentionMaskPresent",), False),
        (("observedMicroBatchSizes",), [2] * 8),
    ],
)
def test_pipeline_rejects_self_hashed_runtime_identity_forgery(
    monkeypatch: pytest.MonkeyPatch,
    field_path: tuple[str, ...],
    replacement: Any,
) -> None:
    _patch_callable_identity(monkeypatch)
    config, preflight, report = _runtime_normalization_fixture()
    forged = copy.deepcopy(report["fleetRuntimeLossNormalization"])
    cursor: dict[str, Any] = forged
    for field in field_path[:-1]:
        cursor = cursor[field]
    cursor[field_path[-1]] = replacement
    if field_path[0] == "getBatchSamples":
        installed_identity = forged["getBatchSamples"][
            "installedCallableIdentity"
        ]
        installed_unsigned = dict(installed_identity)
        installed_unsigned.pop("installedCallableIdentitySHA256")
        installed_identity["installedCallableIdentitySHA256"] = (
            ubuntu_pipeline.canonical_sha256(installed_unsigned)
        )
    unsigned = dict(forged)
    unsigned.pop("runtimeLossNormalizationSHA256")
    forged["runtimeLossNormalizationSHA256"] = (
        ubuntu_pipeline.canonical_sha256(unsigned)
    )

    with pytest.raises(RuntimeError, match="failed verification"):
        ubuntu_pipeline._verify_fleet_sft_runtime_loss_normalization(
            report={"fleetRuntimeLossNormalization": forged},
            config=config,
            token_length_preflight=preflight,
        )


def test_pipeline_forbids_fleet_runtime_evidence_for_other_agents() -> None:
    config, preflight, report = _runtime_normalization_fixture()
    config["agent"] = "cortex"

    with pytest.raises(RuntimeError, match="Non-Fleet"):
        ubuntu_pipeline._verify_fleet_sft_runtime_loss_normalization(
            report=report,
            config=config,
            token_length_preflight=preflight,
        )
