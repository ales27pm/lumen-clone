from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest

from tools.fine_tuning.unsloth import train_sft


def _epoch_orders() -> list[list[int]]:
    return [
        [0, 1, 2, 3],
        [1, 3, 0, 2],
        [2, 0, 3, 1],
        [3, 2, 1, 0],
    ]


def test_resume_wrapper_cannot_reset_trainer_selected_epoch() -> None:
    sampler = train_sft._FleetEpochStratifiedSampler(_epoch_orders())

    # Transformers selects the checkpoint epoch first. The replacement
    # DataLoaderShard built by Accelerate.skip_first_batches then begins at
    # iteration zero and reaches the same sampler with set_epoch(0).
    sampler.set_epoch(2)
    sampler.set_epoch(0)

    assert list(sampler) == _epoch_orders()[2]
    assert sampler.audit_state() == {
        "configuredEpochCount": 4,
        "activeEpoch": 2,
        "setEpochRequestCount": 2,
        "acceptedEpochTransitionCount": 1,
        "idempotentEpochRequestCount": 0,
        "suppressedLowerEpochResetCount": 1,
        "lastRequestedEpoch": 0,
        "lastSuppressedLowerEpoch": 0,
    }

    sampler.set_epoch(3)
    assert list(sampler) == _epoch_orders()[3]
    assert sampler.audit_state()["activeEpoch"] == 3
    assert sampler.audit_state()["acceptedEpochTransitionCount"] == 2


def test_sampler_accepts_monotonic_epochs_and_audits_idempotent_calls() -> None:
    sampler = train_sft._FleetEpochStratifiedSampler(_epoch_orders())

    sampler.set_epoch(0)
    sampler.set_epoch(1)
    sampler.set_epoch(1)
    sampler.set_epoch(3)

    assert list(sampler) == _epoch_orders()[3]
    assert sampler.audit_state() == {
        "configuredEpochCount": 4,
        "activeEpoch": 3,
        "setEpochRequestCount": 4,
        "acceptedEpochTransitionCount": 2,
        "idempotentEpochRequestCount": 2,
        "suppressedLowerEpochResetCount": 0,
        "lastRequestedEpoch": 3,
        "lastSuppressedLowerEpoch": None,
    }


@pytest.mark.parametrize("epoch", [-1, 4, 5, True, 1.0, "1"])
def test_sampler_fails_closed_on_invalid_epoch(epoch: object) -> None:
    sampler = train_sft._FleetEpochStratifiedSampler(_epoch_orders())
    before = sampler.audit_state()

    with pytest.raises(RuntimeError, match="invalid epoch"):
        sampler.set_epoch(epoch)  # type: ignore[arg-type]

    assert sampler.audit_state() == before
    assert list(sampler) == _epoch_orders()[0]


def test_fleet_trainer_args_require_complete_single_process_batches() -> None:
    valid = SimpleNamespace(
        world_size=1,
        ignore_data_skip=False,
        dataloader_drop_last=False,
        per_device_train_batch_size=1,
        packing=False,
        padding_free=False,
    )
    train_sft._validate_fleet_sft_trainer_args(valid)

    for field, invalid_value in (
        ("world_size", 2),
        ("world_size", True),
        ("ignore_data_skip", True),
        ("dataloader_drop_last", True),
        ("per_device_train_batch_size", 2),
        ("per_device_train_batch_size", True),
        ("packing", True),
        ("padding_free", True),
    ):
        invalid = SimpleNamespace(**vars(valid))
        setattr(invalid, field, invalid_value)
        with pytest.raises(RuntimeError, match="dataloader_drop_last=False"):
            train_sft._validate_fleet_sft_trainer_args(invalid)


def test_sft_config_explicitly_disables_drop_last() -> None:
    main_source = inspect.getsource(train_sft.main)
    explicit_policy = "dataloader_drop_last=False,"

    assert explicit_policy in main_source
    assert main_source.index(explicit_policy) < main_source.index(
        "SFTConfig(**sft_kwargs)"
    )


def test_fleet_sft_batching_policy_is_fleet_only() -> None:
    fleet_kwargs: dict[str, object] = {"packing": False}
    non_fleet_kwargs: dict[str, object] = {"packing": False}

    train_sft._apply_fleet_sft_batching_policy(fleet_kwargs, enabled=True)
    train_sft._apply_fleet_sft_batching_policy(
        non_fleet_kwargs,
        enabled=False,
    )

    assert fleet_kwargs == {"packing": False, "padding_free": False}
    assert non_fleet_kwargs == {"packing": False}


def test_fleet_schedule_requires_one_record_per_micro_batch() -> None:
    with pytest.raises(RuntimeError, match="batch_size=1"):
        train_sft._fleet_sft_schedule_controls(
            {
                "seed": 3407,
                "batch_size": 2,
                "gradient_accumulation_steps": 4,
                "num_train_epochs": 3,
                "packing": False,
            }
        )
