from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.fine_tuning.unsloth import train_dpo


PARENT_SFT_SHA256 = "a" * 64


def _minimal_reference_evidence() -> dict[str, object]:
    chosen = ["bf800000"]
    rejected = ["c0000000"]
    source_rows = ["1" * 64]
    split = {
        "rowCount": 1,
        "sourceRowSHA256": source_rows,
        "sourceRowsSHA256": train_dpo._canonical_sha256(source_rows),
        "refChosenLogpsIEEE754Binary32": chosen,
        "refRejectedLogpsIEEE754Binary32": rejected,
        "referenceLogProbPairsSHA256": train_dpo._canonical_sha256(
            [{"chosen": chosen[0], "rejected": rejected[0]}]
        ),
    }
    payload: dict[str, object] = {
        "schema": train_dpo.DPO_REFERENCE_LOG_PROB_EVIDENCE_SCHEMA,
        "agent": "cortex",
        "preferenceTrainer": "dpo",
        "parentSFTAdapterSHA256": PARENT_SFT_SHA256,
        "sourceVariantManifestSHA256": "2" * 64,
        "configSHA256": "3" * 64,
        "datasetFileSHA256": {
            "train_dpo.jsonl": "4" * 64,
            "val_dpo.jsonl": "5" * 64,
            "variant_manifest.json": "6" * 64,
        },
        "trainingCodeSHA256": "7" * 64,
        "resolvedTrainingEnvironmentSHA256": "8" * 64,
        "columns": list(train_dpo.DPO_REFERENCE_LOG_PROB_COLUMNS),
        "splits": {"train": dict(split), "validation": dict(split)},
    }
    return train_dpo._self_hashed_reference_log_prob_evidence(payload)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _fixture(tmp_path: Path) -> tuple[dict[str, object], Path, Path]:
    run_root = tmp_path / "run"
    dataset_dir = (
        run_root
        / "generated/fine_tuning/cortex/experiments/internal_plus_public_optimized"
    )
    dataset_dir.mkdir(parents=True)
    (dataset_dir / "train_dpo.jsonl").write_text(
        '{"prompt":[],"chosen":{},"rejected":{}}\n',
        encoding="utf-8",
    )
    (dataset_dir / "val_dpo.jsonl").write_text("", encoding="utf-8")
    _write_json(
        dataset_dir / "variant_manifest.json",
        {"variantManifestSHA256": "b" * 64},
    )
    config_path = run_root / "configs/cortex.json"
    lineage_path = run_root / "checkpoint_lineage/cortex.preference.json"
    config: dict[str, object] = {
        "agent": "cortex",
        "preference_trainer": "dpo",
        "base_model_name": "Qwen/Qwen3-1.7B",
        "variantManifestSHA256": "b" * 64,
        "dataset_dir": str(dataset_dir),
        "output_dir": str(run_root / "training/cortex"),
        "preferenceCheckpointLineagePath": str(lineage_path),
        "preferenceTokenLengthPreflightPath": str(
            run_root / "training/cortex/dpo/token_length_preflight.json"
        ),
        "trainingCodeSHA256ByPhase": {
            "sft": "c" * 64,
            "dpo": "d" * 64,
            "orpo": "e" * 64,
        },
        "resolvedTrainingEnvironmentSHA256": "f" * 64,
        "bf16": False,
        "fp16": True,
        "save_total_limit": 2,
    }
    _write_json(config_path, config)
    _write_json(
        lineage_path,
        train_dpo._initial_preference_checkpoint_lineage(
            config,
            cfg_path=config_path,
        ),
    )
    return config, config_path, lineage_path


def _complete_checkpoint(
    root: Path,
    step: int,
    *,
    nested_reference: bool = False,
    include_scaler: bool = True,
) -> Path:
    checkpoint = root / f"checkpoint-{step}"
    checkpoint.mkdir(parents=True)
    _write_json(
        checkpoint / "adapter_config.json",
        {"base_model_name_or_path": "Qwen/Qwen3-1.7B"},
    )
    _write_json(checkpoint / "trainer_state.json", {"global_step": step})
    for filename in (
        "adapter_model.safetensors",
        "optimizer.pt",
        "rng_state.pth",
        "scheduler.pt",
        "training_args.bin",
    ):
        (checkpoint / filename).write_bytes(f"{filename}:{step}".encode("utf-8"))
    if include_scaler:
        (checkpoint / train_dpo.CHECKPOINT_SCALER_FILENAME).write_bytes(
            f"scaler.pt:{step}".encode("utf-8")
        )
    if nested_reference:
        _write_json(
            checkpoint
            / train_dpo.REFERENCE_ADAPTER_NAME
            / "adapter_config.json",
            {"base_model_name_or_path": "Qwen/Qwen3-1.7B"},
        )
    return checkpoint


def _bind_fresh(
    config: dict[str, object],
    config_path: Path,
) -> tuple[Path | None, Path | None, list[Path]]:
    result = train_dpo._bind_and_validate_preference_checkpoint_lineage(
        config,
        cfg_path=config_path,
        preference_trainer="dpo",
        parent_sft_adapter_sha256=PARENT_SFT_SHA256,
        require_checkpoint=False,
    )
    lineage_path = Path(str(config["preferenceCheckpointLineagePath"]))
    record = train_dpo._read_preference_checkpoint_lineage(lineage_path)
    updated = dict(record)
    updated["referenceLogProbEvidenceSHA256"] = _minimal_reference_evidence()[
        "referenceLogProbEvidenceSHA256"
    ]
    _write_json(
        lineage_path,
        train_dpo._self_hashed_preference_checkpoint_record(updated),
    )
    return result


def test_preference_checkpoint_policy_is_bounded_and_periodic() -> None:
    assert train_dpo._preference_checkpoint_policy({}) == (5, 2)
    assert train_dpo._preference_checkpoint_policy(
        {"preference_checkpoint_save_steps": 3, "save_total_limit": 4}
    ) == (3, 4)
    with pytest.raises(ValueError, match="positive integer"):
        train_dpo._preference_checkpoint_policy(
            {"preference_checkpoint_save_steps": True}
        )
    with pytest.raises(ValueError, match="at least two"):
        train_dpo._preference_checkpoint_policy({"save_total_limit": 1})


class _WhitespaceTokenizer:
    eos_token_id = 151_645

    def __call__(self, value: str, *, add_special_tokens: bool) -> dict[str, list[int]]:
        assert add_special_tokens is False
        return {"input_ids": list(range(len(value.split())))}


def _render_length_fixture(
    row: dict[str, object],
    *,
    tokenizer: object,
) -> dict[str, str]:
    assert isinstance(tokenizer, _WhitespaceTokenizer)
    return {
        "prompt": str(row["prompt"]),
        "chosen": str(row["chosen"]),
        "rejected": str(row["rejected"]),
    }


def test_token_length_preflight_records_exact_distribution_and_margins() -> None:
    report = train_dpo._preflight_preference_token_lengths(
        {
            "train": [
                {
                    "prompt": "p p",
                    "chosen": "c c c",
                    "rejected": "r",
                },
                {
                    "prompt": "p p p p",
                    "chosen": "c",
                    "rejected": "r r",
                },
            ],
            "validation": [
                {"prompt": "p", "chosen": "c", "rejected": "r"},
            ],
        },
        tokenizer=_WhitespaceTokenizer(),
        render_preference=_render_length_fixture,
        max_prompt_length=68,
        max_sequence_length=135,
    )

    assert report["schemaVersion"] == (
        train_dpo.PREFERENCE_TOKEN_LENGTH_PREFLIGHT_SCHEMA
    )
    assert report["addSpecialTokens"] is False
    assert report["completionTokenizationPolicy"] == (
        train_dpo.DPO_COMPLETION_TOKENIZATION_POLICY
    )
    assert report["appendedEOSTokenID"] == _WhitespaceTokenizer.eos_token_id
    assert report["percentileMethod"] == "nearest_rank"
    assert report["records"] == 3
    assert report["promptTokens"] == {"min": 1, "p50": 2, "p95": 4, "max": 4}
    assert report["chosenTotalTokens"] == {
        "min": 3,
        "p50": 6,
        "p95": 6,
        "max": 6,
    }
    assert report["rejectedTotalTokens"] == {
        "min": 3,
        "p50": 4,
        "p95": 7,
        "max": 7,
    }
    assert report["maximumTotalTokens"] == {
        "min": 3,
        "p50": 6,
        "p95": 7,
        "max": 7,
    }
    assert report["minimumPromptMarginTokens"] == 64
    assert report["minimumSequenceMarginTokens"] == 128
    assert report["smallestPromptMarginTokens"] == 64
    assert report["smallestSequenceMarginTokens"] == 128
    assert report["truncationRequired"] is False
    assert report["splits"]["train"]["records"] == 2
    assert report["splits"]["validation"]["records"] == 1


class _QwenRenderedEOSTokenizer:
    eos_token = "<|im_end|>"
    eos_token_id = 151_645

    def __call__(self, value: str, *, add_special_tokens: bool) -> dict[str, list[int]]:
        assert add_special_tokens is False
        if value == "prompt":
            return {"input_ids": [41]}
        if value.endswith(f"{self.eos_token}\n"):
            # Qwen's rendered assistant completion already carries im_end plus
            # its trailing newline before TRL appends another EOS ID.
            return {"input_ids": [42, self.eos_token_id, 198]}
        raise AssertionError(value)


def test_token_length_preflight_counts_trl_eos_after_qwen_rendered_eos() -> None:
    tokenizer = _QwenRenderedEOSTokenizer()
    completion = f"answer{tokenizer.eos_token}\n"
    report = train_dpo._preflight_preference_token_lengths(
        {
            "train": [
                {
                    "prompt": "prompt",
                    "chosen": completion,
                    "rejected": completion,
                }
            ],
            "validation": [],
        },
        tokenizer=tokenizer,
        render_preference=lambda row, *, tokenizer: row,
        max_prompt_length=65,
        max_sequence_length=133,
    )

    # Prompt tokenization is unchanged. Completion accounting mirrors TRL's
    # three rendered IDs plus its unconditional one-ID EOS suffix.
    assert report["promptTokens"] == {"min": 1, "p50": 1, "p95": 1, "max": 1}
    assert report["chosenCompletionTokens"] == {
        "min": 4,
        "p50": 4,
        "p95": 4,
        "max": 4,
    }
    assert report["chosenTotalTokens"] == {
        "min": 5,
        "p50": 5,
        "p95": 5,
        "max": 5,
    }
    assert report["appendedEOSTokenID"] == tokenizer.eos_token_id
    assert report["completionTokenizationPolicy"][
        "appendedEOSTokensPerCompletion"
    ] == 1


@pytest.mark.parametrize(
    ("row", "message"),
    [
        (
            {"prompt": "p p p p p p", "chosen": "c", "rejected": "r"},
            "prompt uses 6 tokens",
        ),
        (
            {"prompt": "p p p p", "chosen": "c c c c c", "rejected": "r"},
            "prompt plus completion uses 10 tokens",
        ),
    ],
)
def test_token_length_preflight_rejects_any_truncation(
    row: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(RuntimeError, match=message):
        train_dpo._preflight_preference_token_lengths(
            {"train": [row], "validation": []},
            tokenizer=_WhitespaceTokenizer(),
            render_preference=_render_length_fixture,
            max_prompt_length=5,
            max_sequence_length=8,
        )


def test_token_length_preflight_rejects_tight_non_truncating_margin() -> None:
    with pytest.raises(RuntimeError, match="smallest exact-tokenizer margin is 63"):
        train_dpo._preflight_preference_token_lengths(
            {
                "train": [
                    {
                        "prompt": "p p p p",
                        "chosen": "c",
                        "rejected": "r",
                    }
                ]
            },
            tokenizer=_WhitespaceTokenizer(),
            render_preference=_render_length_fixture,
            max_prompt_length=67,
            max_sequence_length=133,
        )


def test_preference_preflight_evidence_is_bound_into_checkpoint_lineage(
    tmp_path: Path,
) -> None:
    config, config_path, lineage_path = _fixture(tmp_path)
    _bind_fresh(config, config_path)
    preflight = train_dpo._preflight_preference_token_lengths(
        {
            "train": [{"prompt": "p", "chosen": "c", "rejected": "r"}],
            "validation": [],
        },
        tokenizer=_WhitespaceTokenizer(),
        render_preference=_render_length_fixture,
        max_prompt_length=65,
        max_sequence_length=131,
    )

    evidence = train_dpo._bind_preference_token_length_preflight(
        config,
        cfg_path=config_path,
        preflight=preflight,
    )

    evidence_path = Path(str(config["preferenceTokenLengthPreflightPath"]))
    assert json.loads(evidence_path.read_text()) == evidence
    assert evidence["trainingCodeSHA256"] == config["trainingCodeSHA256ByPhase"][
        "dpo"
    ]
    record = train_dpo._read_preference_checkpoint_lineage(lineage_path)
    assert record["tokenLengthPreflightSHA256"] == evidence["preflightSHA256"]
    evidence_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="evidence drifted"):
        train_dpo._bind_preference_token_length_preflight(
            config,
            cfg_path=config_path,
            preflight=preflight,
        )


def test_fresh_preference_run_binds_exact_sft_parent(tmp_path: Path) -> None:
    config, config_path, lineage_path = _fixture(tmp_path)

    checkpoint, record_path, unbound = _bind_fresh(config, config_path)

    assert checkpoint is None
    assert record_path == lineage_path
    assert unbound == []
    record = train_dpo._read_preference_checkpoint_lineage(lineage_path)
    assert record["parentSFTAdapterSHA256"] == PARENT_SFT_SHA256
    assert record["referenceSFTAdapterSHA256"] == PARENT_SFT_SHA256
    assert record["saveStrategy"] == "steps"
    assert record["saveSteps"] == 5


def test_resume_selects_latest_complete_bound_policy_checkpoint(
    tmp_path: Path,
) -> None:
    config, config_path, lineage_path = _fixture(tmp_path)
    _bind_fresh(config, config_path)
    checkpoint_root = Path(str(config["output_dir"])) / "dpo"
    for step in (5, 10):
        checkpoint = _complete_checkpoint(checkpoint_root, step)
        train_dpo._record_preference_checkpoint(lineage_path, checkpoint)

    selected, record_path, unbound = (
        train_dpo._bind_and_validate_preference_checkpoint_lineage(
            config,
            cfg_path=config_path,
            preference_trainer="dpo",
            parent_sft_adapter_sha256=PARENT_SFT_SHA256,
            require_checkpoint=True,
        )
    )

    assert selected == checkpoint_root / "checkpoint-10"
    assert record_path == lineage_path
    assert unbound == []


def test_fp16_preference_resume_rejects_missing_scaler_state(tmp_path: Path) -> None:
    config, config_path, lineage_path = _fixture(tmp_path)
    _bind_fresh(config, config_path)
    checkpoint = _complete_checkpoint(
        Path(str(config["output_dir"])) / "dpo",
        5,
    )
    train_dpo._record_preference_checkpoint(lineage_path, checkpoint)
    (checkpoint / train_dpo.CHECKPOINT_SCALER_FILENAME).unlink()

    with pytest.raises(RuntimeError, match=r"missing scaler\.pt"):
        train_dpo._bind_and_validate_preference_checkpoint_lineage(
            config,
            cfg_path=config_path,
            preference_trainer="dpo",
            parent_sft_adapter_sha256=PARENT_SFT_SHA256,
            require_checkpoint=True,
        )


def test_fp16_preference_resume_rejects_tampered_scaler_state(tmp_path: Path) -> None:
    config, config_path, lineage_path = _fixture(tmp_path)
    _bind_fresh(config, config_path)
    checkpoint = _complete_checkpoint(
        Path(str(config["output_dir"])) / "dpo",
        5,
    )
    train_dpo._record_preference_checkpoint(lineage_path, checkpoint)
    (checkpoint / train_dpo.CHECKPOINT_SCALER_FILENAME).write_bytes(b"tampered")

    with pytest.raises(RuntimeError, match="drifted from lineage"):
        train_dpo._bind_and_validate_preference_checkpoint_lineage(
            config,
            cfg_path=config_path,
            preference_trainer="dpo",
            parent_sft_adapter_sha256=PARENT_SFT_SHA256,
            require_checkpoint=True,
        )


def test_bf16_preference_resume_does_not_require_scaler_state(tmp_path: Path) -> None:
    config, config_path, lineage_path = _fixture(tmp_path)
    config["bf16"] = True
    config["fp16"] = False
    _write_json(config_path, config)
    _write_json(
        lineage_path,
        train_dpo._initial_preference_checkpoint_lineage(
            config,
            cfg_path=config_path,
        ),
    )
    _bind_fresh(config, config_path)
    checkpoint = _complete_checkpoint(
        Path(str(config["output_dir"])) / "dpo",
        5,
        include_scaler=False,
    )
    train_dpo._record_preference_checkpoint(lineage_path, checkpoint)

    selected, _, _ = train_dpo._bind_and_validate_preference_checkpoint_lineage(
        config,
        cfg_path=config_path,
        preference_trainer="dpo",
        parent_sft_adapter_sha256=PARENT_SFT_SHA256,
        require_checkpoint=True,
    )

    assert selected == checkpoint
    record = train_dpo._read_preference_checkpoint_lineage(lineage_path)
    assert record["checkpointScalerState"] == {
        "schemaVersion": train_dpo.CHECKPOINT_SCALER_STATE_SCHEMA,
        "filename": "scaler.pt",
        "required": False,
        "requirement": "required_for_fp16_cuda_native_amp",
        "transformersVersion": "4.57.6",
        "accelerateVersion": "1.14.0",
    }


def test_resume_survives_rotation_callback_crash_window(tmp_path: Path) -> None:
    config, config_path, lineage_path = _fixture(tmp_path)
    _bind_fresh(config, config_path)
    checkpoint_root = Path(str(config["output_dir"])) / "dpo"
    checkpoint5 = _complete_checkpoint(checkpoint_root, 5)
    train_dpo._record_preference_checkpoint(lineage_path, checkpoint5)
    checkpoint10 = _complete_checkpoint(checkpoint_root, 10)
    train_dpo._record_preference_checkpoint(lineage_path, checkpoint10)

    # Transformers rotates before it invokes on_save. Model that narrow crash
    # window: checkpoint-5 vanished and checkpoint-15 is not yet lineage-bound.
    import shutil

    shutil.rmtree(checkpoint5)
    checkpoint15 = _complete_checkpoint(checkpoint_root, 15)

    selected, _, unbound = train_dpo._bind_and_validate_preference_checkpoint_lineage(
        config,
        cfg_path=config_path,
        preference_trainer="dpo",
        parent_sft_adapter_sha256=PARENT_SFT_SHA256,
        require_checkpoint=True,
    )

    assert selected == checkpoint10
    assert unbound == [checkpoint15]
    train_dpo._prune_unbound_preference_checkpoints(unbound)
    assert not checkpoint15.exists()
    assert checkpoint10.is_dir()


def test_resume_prunes_partial_rotated_older_and_unbound_newer(
    tmp_path: Path,
) -> None:
    config, config_path, lineage_path = _fixture(tmp_path)
    _bind_fresh(config, config_path)
    checkpoint_root = Path(str(config["output_dir"])) / "dpo"
    checkpoint5 = _complete_checkpoint(checkpoint_root, 5)
    train_dpo._record_preference_checkpoint(lineage_path, checkpoint5)
    checkpoint10 = _complete_checkpoint(checkpoint_root, 10)
    train_dpo._record_preference_checkpoint(lineage_path, checkpoint10)

    # Signed state is [N-2, N-1]. Rotation partially deletes N-2 and writes N,
    # then the process dies before on_save can bind N.
    (checkpoint5 / "optimizer.pt").unlink()
    checkpoint15 = _complete_checkpoint(checkpoint_root, 15)

    selected, _, discardable = (
        train_dpo._bind_and_validate_preference_checkpoint_lineage(
            config,
            cfg_path=config_path,
            preference_trainer="dpo",
            parent_sft_adapter_sha256=PARENT_SFT_SHA256,
            require_checkpoint=True,
        )
    )

    assert selected == checkpoint10
    assert discardable == [checkpoint5, checkpoint15]
    train_dpo._prune_unbound_preference_checkpoints(discardable)
    assert not checkpoint5.exists()
    assert checkpoint10.is_dir()
    assert not checkpoint15.exists()


def test_resume_rejects_complete_hash_drift_in_older_checkpoint(
    tmp_path: Path,
) -> None:
    config, config_path, lineage_path = _fixture(tmp_path)
    _bind_fresh(config, config_path)
    checkpoint_root = Path(str(config["output_dir"])) / "dpo"
    checkpoint5 = _complete_checkpoint(checkpoint_root, 5)
    train_dpo._record_preference_checkpoint(lineage_path, checkpoint5)
    train_dpo._record_preference_checkpoint(
        lineage_path,
        _complete_checkpoint(checkpoint_root, 10),
    )
    (checkpoint5 / "optimizer.pt").write_bytes(b"complete-but-drifted")

    with pytest.raises(RuntimeError, match="drifted from lineage"):
        train_dpo._bind_and_validate_preference_checkpoint_lineage(
            config,
            cfg_path=config_path,
            preference_trainer="dpo",
            parent_sft_adapter_sha256=PARENT_SFT_SHA256,
            require_checkpoint=True,
        )


def test_resume_rejects_partial_newest_bound_checkpoint(tmp_path: Path) -> None:
    config, config_path, lineage_path = _fixture(tmp_path)
    _bind_fresh(config, config_path)
    checkpoint_root = Path(str(config["output_dir"])) / "dpo"
    train_dpo._record_preference_checkpoint(
        lineage_path,
        _complete_checkpoint(checkpoint_root, 5),
    )
    checkpoint10 = _complete_checkpoint(checkpoint_root, 10)
    train_dpo._record_preference_checkpoint(lineage_path, checkpoint10)
    (checkpoint10 / "optimizer.pt").unlink()

    with pytest.raises(RuntimeError, match="checkpoint is incomplete"):
        train_dpo._bind_and_validate_preference_checkpoint_lineage(
            config,
            cfg_path=config_path,
            preference_trainer="dpo",
            parent_sft_adapter_sha256=PARENT_SFT_SHA256,
            require_checkpoint=True,
        )


def test_resume_rejects_symlink_inside_stale_partial(tmp_path: Path) -> None:
    config, config_path, lineage_path = _fixture(tmp_path)
    _bind_fresh(config, config_path)
    checkpoint_root = Path(str(config["output_dir"])) / "dpo"
    checkpoint5 = _complete_checkpoint(checkpoint_root, 5)
    train_dpo._record_preference_checkpoint(lineage_path, checkpoint5)
    train_dpo._record_preference_checkpoint(
        lineage_path,
        _complete_checkpoint(checkpoint_root, 10),
    )
    (checkpoint5 / "optimizer.pt").unlink()
    (checkpoint5 / "unsafe-link").symlink_to(tmp_path / "outside")

    with pytest.raises(RuntimeError, match="contains a symlink"):
        train_dpo._bind_and_validate_preference_checkpoint_lineage(
            config,
            cfg_path=config_path,
            preference_trainer="dpo",
            parent_sft_adapter_sha256=PARENT_SFT_SHA256,
            require_checkpoint=True,
        )


def test_resume_discards_unbound_first_checkpoint_and_restarts_fresh(
    tmp_path: Path,
) -> None:
    config, config_path, _ = _fixture(tmp_path)
    _bind_fresh(config, config_path)
    checkpoint = _complete_checkpoint(
        Path(str(config["output_dir"])) / "dpo",
        5,
    )

    selected, _, unbound = train_dpo._bind_and_validate_preference_checkpoint_lineage(
        config,
        cfg_path=config_path,
        preference_trainer="dpo",
        parent_sft_adapter_sha256=PARENT_SFT_SHA256,
        require_checkpoint=True,
    )

    assert selected is None
    assert unbound == [checkpoint]
    train_dpo._prune_unbound_preference_checkpoints(unbound)
    assert not checkpoint.exists()


def test_resume_rejects_checkpoint_content_drift(tmp_path: Path) -> None:
    config, config_path, lineage_path = _fixture(tmp_path)
    _bind_fresh(config, config_path)
    checkpoint_root = Path(str(config["output_dir"])) / "dpo"
    checkpoint = _complete_checkpoint(checkpoint_root, 5)
    train_dpo._record_preference_checkpoint(lineage_path, checkpoint)
    (checkpoint / "optimizer.pt").write_bytes(b"drift")

    with pytest.raises(RuntimeError, match="drifted from lineage"):
        train_dpo._bind_and_validate_preference_checkpoint_lineage(
            config,
            cfg_path=config_path,
            preference_trainer="dpo",
            parent_sft_adapter_sha256=PARENT_SFT_SHA256,
            require_checkpoint=True,
        )


def test_resume_rejects_frozen_reference_adapter_in_checkpoint(
    tmp_path: Path,
) -> None:
    config, config_path, lineage_path = _fixture(tmp_path)
    _bind_fresh(config, config_path)
    checkpoint = _complete_checkpoint(
        Path(str(config["output_dir"])) / "dpo",
        5,
        nested_reference=True,
    )

    with pytest.raises(RuntimeError, match="must not persist"):
        train_dpo._record_preference_checkpoint(lineage_path, checkpoint)


@pytest.mark.parametrize("drift", ("config", "dataset", "parent"))
def test_resume_rejects_lineage_drift(tmp_path: Path, drift: str) -> None:
    config, config_path, lineage_path = _fixture(tmp_path)
    _bind_fresh(config, config_path)
    checkpoint = _complete_checkpoint(
        Path(str(config["output_dir"])) / "dpo",
        5,
    )
    train_dpo._record_preference_checkpoint(lineage_path, checkpoint)
    parent = PARENT_SFT_SHA256
    if drift == "config":
        config["save_total_limit"] = 3
    elif drift == "dataset":
        (Path(str(config["dataset_dir"])) / "train_dpo.jsonl").write_text(
            "changed\n",
            encoding="utf-8",
        )
    else:
        parent = "9" * 64

    with pytest.raises(RuntimeError, match="drifted"):
        train_dpo._bind_and_validate_preference_checkpoint_lineage(
            config,
            cfg_path=config_path,
            preference_trainer="dpo",
            parent_sft_adapter_sha256=parent,
            require_checkpoint=True,
        )


def test_reference_adapter_is_deleted_only_after_frozen_precompute() -> None:
    events: list[str] = []

    class Model:
        def __init__(self) -> None:
            self.peft_config = {
                train_dpo.POLICY_ADAPTER_NAME: object(),
                train_dpo.REFERENCE_ADAPTER_NAME: object(),
            }

        def set_adapter(self, name: str) -> None:
            events.append(f"set:{name}")

        def delete_adapter(self, name: str) -> None:
            events.append(f"delete:{name}")
            del self.peft_config[name]

    class Trainer:
        model = Model()
        eval_dataset = object()
        _precomputed_train_ref_log_probs = True
        _precomputed_eval_ref_log_probs = True

    result = train_dpo._drop_precomputed_reference_adapter(
        Trainer(),
        reference_log_prob_evidence=_minimal_reference_evidence(),
    )

    assert result == {
        "referenceAdapterRemovedAfterPrecompute": True,
        "checkpointAdapterNames": [train_dpo.POLICY_ADAPTER_NAME],
    }
    assert events == [
        f"set:{train_dpo.POLICY_ADAPTER_NAME}",
        f"delete:{train_dpo.REFERENCE_ADAPTER_NAME}",
        f"set:{train_dpo.POLICY_ADAPTER_NAME}",
    ]


def test_reference_adapter_removal_fails_before_eval_precompute() -> None:
    class Trainer:
        model = object()
        eval_dataset = object()
        _precomputed_train_ref_log_probs = True
        _precomputed_eval_ref_log_probs = False

    with pytest.raises(RuntimeError, match="before all reference"):
        train_dpo._drop_precomputed_reference_adapter(
            Trainer(),
            reference_log_prob_evidence=_minimal_reference_evidence(),
        )
