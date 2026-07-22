from __future__ import annotations

import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.fine_tuning.unsloth import train_sft, ubuntu_pipeline


FP16_PRECISION = {
    "schemaVersion": train_sft.TRAINING_PRECISION_SCHEMA,
    "bf16": False,
    "fp16": True,
    "dtype": "float16",
}


def _completed_fixture() -> tuple[object, object, object, dict[str, float]]:
    trainer = SimpleNamespace(
        state=SimpleNamespace(global_step=12, max_steps=12, epoch=3.0)
    )
    training_args = SimpleNamespace(
        num_train_epochs=3.0,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        world_size=1,
        bf16=False,
        fp16=True,
    )
    train_result = SimpleNamespace(
        global_step=12,
        metrics={
            "train_loss": 0.25,
            "train_runtime": 4.0,
            "epoch": 3.0,
        },
    )
    evaluation_metrics = {
        "eval_loss": 0.2,
        "eval_runtime": 1.0,
        "epoch": 3.0,
    }
    return trainer, training_args, train_result, evaluation_metrics


@pytest.mark.parametrize(
    "config",
    (
        {},
        {"bf16": False},
        {"bf16": "false", "fp16": True},
        {"bf16": False, "fp16": False},
        {"bf16": True, "fp16": True},
    ),
)
def test_training_precision_requires_explicit_mutually_exclusive_booleans(
    config: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        train_sft._resolve_training_precision(config)


def test_training_completion_requires_exact_terminal_step_and_epoch() -> None:
    trainer, training_args, train_result, evaluation_metrics = _completed_fixture()

    evidence = train_sft._verified_training_completion_evidence(
        trainer,
        training_args,
        train_result,
        evaluation_metrics,
        has_eval_dataset=True,
        train_record_count=16,
        expected_precision=FP16_PRECISION,
    )

    assert evidence == {
        "schema": train_sft.TRAINING_COMPLETION_EVIDENCE_SCHEMA,
        "status": "completed",
        "globalStep": 12,
        "maxSteps": 12,
        "expectedMaxSteps": 12,
        "trainResultGlobalStep": 12,
        "configuredNumTrainEpochs": 3.0,
        "observedEpoch": 3.0,
        "trainRecordCount": 16,
        "perDeviceTrainBatchSize": 2,
        "gradientAccumulationSteps": 2,
        "worldSize": 1,
        "trainDataloaderBatchCount": 8,
        "updateStepsPerEpoch": 4,
        "trainMetricsVerified": True,
        "evaluationMetricsVerified": True,
        "resolvedPrecision": FP16_PRECISION,
    }

    trainer.state.global_step = 11
    with pytest.raises(RuntimeError, match="terminal global step"):
        train_sft._verified_training_completion_evidence(
            trainer,
            training_args,
            train_result,
            evaluation_metrics,
            has_eval_dataset=True,
            train_record_count=16,
            expected_precision=FP16_PRECISION,
        )

    trainer.state.global_step = 12
    training_args.fp16 = False
    with pytest.raises(RuntimeError, match="precision"):
        train_sft._verified_training_completion_evidence(
            trainer,
            training_args,
            train_result,
            evaluation_metrics,
            has_eval_dataset=True,
            train_record_count=16,
            expected_precision=FP16_PRECISION,
        )


def test_training_completion_reconstructs_expected_steps_independently() -> None:
    trainer, training_args, train_result, evaluation_metrics = _completed_fixture()
    trainer.state.global_step = 9
    trainer.state.max_steps = 9
    train_result.global_step = 9

    with pytest.raises(RuntimeError, match="terminal step drifted"):
        train_sft._verified_training_completion_evidence(
            trainer,
            training_args,
            train_result,
            evaluation_metrics,
            has_eval_dataset=True,
            train_record_count=16,
            expected_precision=FP16_PRECISION,
        )


@pytest.mark.parametrize(
    ("target", "replacement", "message"),
    (
        ("train", math.nan, "finite numeric metric"),
        ("evaluation", math.inf, "finite numeric metric"),
        ("epoch", 2.5, "configured epoch count"),
    ),
)
def test_training_completion_rejects_false_success_evidence(
    target: str,
    replacement: float,
    message: str,
) -> None:
    trainer, training_args, train_result, evaluation_metrics = _completed_fixture()
    if target == "train":
        train_result.metrics["train_loss"] = replacement
    elif target == "evaluation":
        evaluation_metrics["eval_loss"] = replacement
    else:
        trainer.state.epoch = replacement

    with pytest.raises(RuntimeError, match=message):
        train_sft._verified_training_completion_evidence(
            trainer,
            training_args,
            train_result,
            evaluation_metrics,
            has_eval_dataset=True,
            train_record_count=16,
            expected_precision=FP16_PRECISION,
        )


def test_atomic_json_install_fsyncs_file_and_parent_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fsynced: list[int] = []
    real_fsync = os.fsync

    def record_fsync(fd: int) -> None:
        fsynced.append(fd)
        real_fsync(fd)

    monkeypatch.setattr(train_sft.os, "fsync", record_fsync)
    output = tmp_path / "lineage.json"

    train_sft._write_json_atomic(output, {"status": "bound"})

    assert output.is_file()
    assert len(fsynced) == 2


def test_pipeline_reconstructs_completion_and_finite_metric_evidence(
    tmp_path: Path,
) -> None:
    report_path = tmp_path / "training_report.json"
    report = {
        "agent": "cortex",
        "train_records": 10,
        "val_records": 2,
        "metrics": {"train_loss": 0.25, "epoch": 3.0},
        "evaluation_metrics": {"eval_loss": 0.2, "epoch": 3.0},
        "precision": FP16_PRECISION,
        "trainingCompletion": {
            "schema": train_sft.TRAINING_COMPLETION_EVIDENCE_SCHEMA,
            "status": "completed",
            "globalStep": 12,
            "maxSteps": 12,
            "expectedMaxSteps": 12,
            "trainResultGlobalStep": 12,
            "configuredNumTrainEpochs": 3.0,
            "observedEpoch": 3.0,
            "trainRecordCount": 10,
            "perDeviceTrainBatchSize": 1,
            "gradientAccumulationSteps": 3,
            "worldSize": 1,
            "trainDataloaderBatchCount": 10,
            "updateStepsPerEpoch": 4,
            "trainMetricsVerified": True,
            "evaluationMetricsVerified": True,
            "resolvedPrecision": FP16_PRECISION,
        },
    }
    report_path.write_text(json.dumps(report) + "\n", encoding="utf-8")

    verified = ubuntu_pipeline._verify_training_report(
        report_path,
        phase="SFT",
        expected={"agent": "cortex"},
        configured_num_train_epochs=3.0,
        per_device_train_batch_size=1,
        configured_gradient_accumulation_steps=3,
        expected_precision=FP16_PRECISION,
    )
    assert verified == report

    report["trainingCompletion"]["maxSteps"] = 13
    report_path.write_text(json.dumps(report) + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="false or incomplete"):
        ubuntu_pipeline._verify_training_report(
            report_path,
            phase="SFT",
            expected={"agent": "cortex"},
            configured_num_train_epochs=3.0,
            per_device_train_batch_size=1,
            configured_gradient_accumulation_steps=3,
            expected_precision=FP16_PRECISION,
        )

    report["trainingCompletion"]["maxSteps"] = 12
    report["precision"] = {
        "schemaVersion": train_sft.TRAINING_PRECISION_SCHEMA,
        "bf16": True,
        "fp16": False,
        "dtype": "bfloat16",
    }
    report_path.write_text(json.dumps(report) + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="false or incomplete"):
        ubuntu_pipeline._verify_training_report(
            report_path,
            phase="SFT",
            expected={"agent": "cortex"},
            configured_num_train_epochs=3.0,
            per_device_train_batch_size=1,
            configured_gradient_accumulation_steps=3,
            expected_precision=FP16_PRECISION,
        )
