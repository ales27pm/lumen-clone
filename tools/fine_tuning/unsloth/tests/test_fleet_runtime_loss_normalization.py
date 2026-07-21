from __future__ import annotations

from types import MethodType, SimpleNamespace
from typing import Any

import pytest

from tools.fine_tuning.unsloth import train_sft


class _FakeTensor:
    def __init__(self, value: Any) -> None:
        self.value = value

    def detach(self) -> _FakeTensor:
        return self

    def cpu(self) -> _FakeTensor:
        return self

    def tolist(self) -> Any:
        return self.value


class _FakeScalar(_FakeTensor):
    def numel(self) -> int:
        return 1

    def item(self) -> Any:
        return self.value


class _FakeBatchIterator:
    def __init__(self, batches: list[dict[str, Any]]) -> None:
        self._iterator = iter(batches)
        self.workers_shutdown = False

    def __iter__(self) -> _FakeBatchIterator:
        return self

    def __next__(self) -> dict[str, Any]:
        return next(self._iterator)

    def _shutdown_workers(self) -> None:
        self.workers_shutdown = True


class _FakeBaseModel:
    config = SimpleNamespace(model_type="qwen3")


class _FakePeftModel:
    def get_base_model(self) -> _FakeBaseModel:
        return _FakeBaseModel()


def _unsloth_get_batch_samples(
    trainer: Any,
    epoch_iterator: Any,
    num_batches: int,
    device: Any,
) -> tuple[list[dict[str, Any]], Any]:
    del device
    batches = [next(epoch_iterator) for _ in range(num_batches)]
    if trainer.return_none_denominator:
        denominator: Any = None
    else:
        denominator = _FakeScalar(
            sum(
                _independent_shifted_target_count(batch)
                for batch in batches
            )
            + trainer.denominator_delta
        )
    return batches, denominator


_unsloth_get_batch_samples.__module__ = "unsloth_zoo.loss_utils"


def _wrong_get_batch_samples(
    trainer: Any,
    epoch_iterator: Any,
    num_batches: int,
    device: Any,
) -> tuple[list[dict[str, Any]], _FakeScalar]:
    del trainer, device
    batches = [next(epoch_iterator) for _ in range(num_batches)]
    return batches, _FakeScalar(1)


class _FakeTrainer:
    def __init__(
        self,
        *,
        sampler: train_sft._FleetEpochStratifiedSampler,
        batches: list[dict[str, Any]],
    ) -> None:
        self.args = SimpleNamespace(
            gradient_accumulation_steps=2,
            per_device_train_batch_size=1,
            world_size=1,
            packing=False,
            padding_free=False,
            dataloader_drop_last=False,
            loss_type="nll",
            device="cpu",
        )
        self.accelerator = SimpleNamespace(
            gradient_accumulation_steps=1,
        )
        self.compute_loss_func = None
        self.model_accepts_loss_kwargs = True
        self.optimizer = None
        self.lr_scheduler = None
        self.padding_free = False
        self.data_collator = SimpleNamespace(padding_free=False)
        self.model = _FakePeftModel()
        self.train_dataset = [object() for _ in batches]
        self._sampler = sampler
        self._batches = batches
        self.denominator_delta = 0
        self.return_none_denominator = False
        self.get_batch_samples = MethodType(
            _unsloth_get_batch_samples,
            self,
        )

    def get_train_dataloader(self) -> _FakeBatchIterator:
        # Reproduce the pinned Trainer/Accelerate epoch-zero request. The
        # attestation must undo even this idempotent audit-state mutation.
        self._sampler.set_epoch(0)
        return _FakeBatchIterator(
            [self._batches[row_index] for row_index in self._sampler]
        )


def _batches() -> list[dict[str, Any]]:
    return [
        {
            "labels": _FakeTensor([[91, 11, 12, -100]]),
            "attention_mask": _FakeTensor([[1, 1, 1, 1]]),
        },
        {
            "labels": _FakeTensor([[92, 21, 22, 23]]),
            "attention_mask": _FakeTensor([[1, 1, 0, 1]]),
        },
        {
            "labels": _FakeTensor([[93, 31, -100]]),
            "attention_mask": _FakeTensor([[1, 1, 1]]),
        },
        {
            "labels": _FakeTensor([[94, 41, 42, 43]]),
            "attention_mask": _FakeTensor([[1, 1, 1, 1]]),
        },
    ]


def _independent_shifted_target_count(batch: dict[str, Any]) -> int:
    labels = batch["labels"].value
    attention = batch["attention_mask"].value
    return sum(
        1
        for label_row, attention_row in zip(labels, attention)
        for target, attended in zip(label_row[1:], attention_row[1:])
        if target != -100 and attended != 0
    )


def _row_evidence(batches: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "rowIndex": row_index,
            "targetTokenCount": _independent_shifted_target_count(batch),
        }
        for row_index, batch in enumerate(batches)
    ]


def _config() -> dict[str, Any]:
    return {
        "seed": 3407,
        "batch_size": 1,
        "gradient_accumulation_steps": 2,
        "num_train_epochs": 2,
        "packing": False,
        "trainingCodeSHA256": "d" * 64,
        "resolvedTrainingEnvironment": {"fixture": True},
        "resolvedTrainingEnvironmentSHA256": "e" * 64,
        "trainingEnvironmentSHA256": "f" * 64,
        "trainingContainerImageDigest": f"sha256:{'1' * 64}",
    }


def _epoch_orders() -> list[list[int]]:
    return [[0, 1, 2, 3], [2, 3, 0, 1]]


def _patch_stable_rng(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        train_sft,
        "_capture_fleet_runtime_rng_state",
        lambda: {"state": "stable"},
    )
    monkeypatch.setattr(
        train_sft,
        "_restore_fleet_runtime_rng_state",
        lambda state: None,
    )
    monkeypatch.setattr(
        train_sft,
        "_fleet_runtime_rng_states_equal",
        lambda left, right: left == right,
    )


def _runtime_fixture(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[
    _FakeTrainer,
    train_sft._FleetEpochStratifiedSampler,
    list[dict[str, Any]],
]:
    _patch_stable_rng(monkeypatch)
    monkeypatch.setattr(
        train_sft,
        "FLEET_SFT_TRAINER_CLASS",
        f"{_FakeTrainer.__module__}.{_FakeTrainer.__name__}",
    )
    monkeypatch.setattr(
        train_sft,
        "FLEET_SFT_MODEL_CLASS",
        f"{_FakePeftModel.__module__}.{_FakePeftModel.__name__}",
    )
    monkeypatch.setattr(
        train_sft,
        "FLEET_SFT_BASE_MODEL_CLASS",
        f"{_FakeBaseModel.__module__}.{_FakeBaseModel.__name__}",
    )
    live_code = _unsloth_get_batch_samples.__code__
    callable_identity = {
        "schemaVersion": "lumen.installed-python-callable-identity/1.0.0",
        "resolvedTrainingEnvironmentSHA256": "e" * 64,
        "distributionName": "unsloth-zoo",
        "distributionVersion": "2026.7.2",
        "distributionSHA256": "2" * 64,
        "sourceLogicalPath": "unsloth_zoo/loss_utils.py",
        "sourceFileSize": 1234,
        "sourceFileSHA256": "3" * 64,
        "callableName": "_unsloth_get_batch_samples",
        "callableQualname": _unsloth_get_batch_samples.__qualname__,
        "callableFirstLineNumber": live_code.co_firstlineno,
        "codeSHA256": train_sft._runtime_callable_code_sha256(
            _unsloth_get_batch_samples,
            label="fixture",
        ),
        "installedCallableIdentitySHA256": "4" * 64,
    }
    monkeypatch.setattr(
        train_sft,
        "installed_distribution_python_callable_identity",
        lambda **kwargs: dict(callable_identity),
    )
    batches = _batches()
    sampler = train_sft._FleetEpochStratifiedSampler(_epoch_orders())
    trainer = _FakeTrainer(sampler=sampler, batches=batches)
    return trainer, sampler, _row_evidence(batches)


def _attest(
    trainer: _FakeTrainer,
    sampler: train_sft._FleetEpochStratifiedSampler,
    row_evidence: list[dict[str, Any]],
) -> dict[str, Any]:
    return train_sft._attest_fleet_sft_runtime_loss_normalization(
        trainer,
        assistant_only_loss=True,
        sft_config_padding_free=False,
        config=_config(),
        row_token_evidence=row_evidence,
        epoch_orders=_epoch_orders(),
        fleet_sampler=sampler,
    )


def test_shifted_target_count_intersects_attention_mask() -> None:
    tokenized = {
        "labels": [77, -100, 7, 8, 9],
        "attention_mask": [1, 1, 1, 0, 1],
    }

    assert train_sft._shifted_sft_target_token_count(tokenized) == 2
    assert (
        train_sft._runtime_shifted_sft_target_token_count(
            {
                "labels": _FakeTensor([tokenized["labels"]]),
                "attention_mask": _FakeTensor(
                    [tokenized["attention_mask"]]
                ),
            }
        )
        == 2
    )


def test_runtime_attestation_matches_full_window_and_restores_sampler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer, sampler, row_evidence = _runtime_fixture(monkeypatch)
    sampler_before = sampler.audit_state()

    evidence = _attest(trainer, sampler, row_evidence)

    assert evidence["status"] == "passed"
    assert evidence["expectedMicroBatchTargetTokenCounts"] == [2, 2]
    assert evidence["observedMicroBatchTargetTokenCounts"] == [2, 2]
    assert evidence["expectedTargetTokenCount"] == 4
    assert evidence["reconstructedTargetTokenCount"] == 4
    assert evidence["reportedNumItemsInBatch"] == 4
    assert evidence["samplerStatePreserved"] is True
    assert evidence["rngStateRestored"] is True
    assert sampler.audit_state() == sampler_before
    assert (
        len(
            evidence["getBatchSamples"]["installedCallableIdentity"][
                "codeSHA256"
            ]
        )
        == 64
    )
    assert evidence["trainingCodeSHA256"] == "d" * 64
    assert evidence["resolvedTrainingEnvironmentSHA256"] == "e" * 64
    assert len(evidence["runtimeLossNormalizationSHA256"]) == 64


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("model_accepts_loss_kwargs", False),
        ("compute_loss_func", object()),
        ("optimizer", object()),
        ("padding_free", True),
    ],
)
def test_runtime_attestation_rejects_unpinned_trainer_state(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: Any,
) -> None:
    trainer, sampler, row_evidence = _runtime_fixture(monkeypatch)
    setattr(trainer, field, value)

    with pytest.raises(RuntimeError, match="pinned token-normalized"):
        _attest(trainer, sampler, row_evidence)


@pytest.mark.parametrize(
    ("owner", "field", "value"),
    [
        ("args", "padding_free", True),
        ("data_collator", "padding_free", True),
        ("data_collator", "padding_free", None),
    ],
)
def test_runtime_attestation_rejects_nested_padding_state_drift(
    monkeypatch: pytest.MonkeyPatch,
    owner: str,
    field: str,
    value: Any,
) -> None:
    trainer, sampler, row_evidence = _runtime_fixture(monkeypatch)
    setattr(getattr(trainer, owner), field, value)

    with pytest.raises(RuntimeError, match="pinned token-normalized"):
        _attest(trainer, sampler, row_evidence)


def test_runtime_attestation_rejects_preinit_sft_config_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer, sampler, row_evidence = _runtime_fixture(monkeypatch)

    with pytest.raises(RuntimeError, match="pinned token-normalized"):
        train_sft._attest_fleet_sft_runtime_loss_normalization(
            trainer,
            assistant_only_loss=True,
            sft_config_padding_free=True,
            config=_config(),
            row_token_evidence=row_evidence,
            epoch_orders=_epoch_orders(),
            fleet_sampler=sampler,
        )


def test_runtime_attestation_rejects_wrong_batch_sampling_implementation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer, sampler, row_evidence = _runtime_fixture(monkeypatch)
    trainer.get_batch_samples = MethodType(
        _wrong_get_batch_samples,
        trainer,
    )

    with pytest.raises(RuntimeError, match="pinned token-normalized"):
        _attest(trainer, sampler, row_evidence)


def test_runtime_attestation_rejects_installed_callable_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer, sampler, row_evidence = _runtime_fixture(monkeypatch)
    original = train_sft.installed_distribution_python_callable_identity

    def drifted(**kwargs: Any) -> dict[str, Any]:
        identity = original(**kwargs)
        identity["codeSHA256"] = "9" * 64
        return identity

    monkeypatch.setattr(
        train_sft,
        "installed_distribution_python_callable_identity",
        drifted,
    )

    with pytest.raises(RuntimeError, match="drifted from the installed wheel"):
        _attest(trainer, sampler, row_evidence)


@pytest.mark.parametrize("denominator_delta", [-1, 1])
def test_runtime_attestation_rejects_wrong_window_denominator(
    monkeypatch: pytest.MonkeyPatch,
    denominator_delta: int,
) -> None:
    trainer, sampler, row_evidence = _runtime_fixture(monkeypatch)
    trainer.denominator_delta = denominator_delta

    with pytest.raises(RuntimeError, match="differs from the scheduled"):
        _attest(trainer, sampler, row_evidence)


def test_runtime_attestation_rejects_missing_window_denominator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer, sampler, row_evidence = _runtime_fixture(monkeypatch)
    trainer.return_none_denominator = True

    with pytest.raises(RuntimeError, match="positive integer"):
        _attest(trainer, sampler, row_evidence)


def test_runtime_attestation_rejects_batch_schedule_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer, sampler, row_evidence = _runtime_fixture(monkeypatch)
    row_evidence[0]["targetTokenCount"] += 1

    with pytest.raises(RuntimeError, match="differs from the scheduled"):
        _attest(trainer, sampler, row_evidence)


def test_runtime_attestation_rejects_rng_state_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer, sampler, row_evidence = _runtime_fixture(monkeypatch)
    states = iter(({"state": "before"}, {"state": "after"}))
    monkeypatch.setattr(
        train_sft,
        "_capture_fleet_runtime_rng_state",
        lambda: next(states),
    )

    with pytest.raises(RuntimeError, match="changed training state"):
        _attest(trainer, sampler, row_evidence)

    assert sampler.audit_state()["setEpochRequestCount"] == 0


@pytest.mark.parametrize(
    "tokenized",
    [
        {"labels": [1, 2], "attention_mask": [1]},
        {"labels": [1, "2"], "attention_mask": [1, 1]},
        {"labels": [1, 2], "attention_mask": [1, True]},
    ],
)
def test_shifted_target_count_rejects_malformed_rows(
    tokenized: dict[str, Any],
) -> None:
    with pytest.raises(RuntimeError, match="malformed labels"):
        train_sft._shifted_sft_target_token_count(tokenized)


def test_runtime_target_count_rejects_packing() -> None:
    with pytest.raises(RuntimeError, match="forbids padding-free or packed"):
        train_sft._runtime_shifted_sft_target_token_count(
            {
                "labels": _FakeTensor([[1, 2]]),
                "attention_mask": _FakeTensor([[1, 1]]),
                "packed_seq_lengths": _FakeTensor([2]),
            }
        )


def test_runtime_target_count_rejects_padding_free_position_ids() -> None:
    with pytest.raises(RuntimeError, match="forbids padding-free or packed"):
        train_sft._runtime_shifted_sft_target_token_count(
            {
                "labels": _FakeTensor([[1, 2]]),
                "attention_mask": _FakeTensor([[1, 1]]),
                "position_ids": _FakeTensor([[0, 1]]),
            }
        )


def test_runtime_target_count_requires_padded_attention_mask() -> None:
    with pytest.raises(RuntimeError, match="lacks attention_mask"):
        train_sft._runtime_shifted_sft_target_token_count(
            {"labels": _FakeTensor([[1, 2]])}
        )


def test_runtime_target_count_rejects_batch_size_drift() -> None:
    with pytest.raises(RuntimeError, match="batch size differs"):
        train_sft._runtime_shifted_sft_target_token_count(
            {
                "labels": _FakeTensor([[1, 2], [3, 4]]),
                "attention_mask": _FakeTensor([[1, 1], [1, 1]]),
            },
            expected_batch_size=1,
        )
