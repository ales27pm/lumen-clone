from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.fine_tuning.unsloth import train_sft


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def test_checkpoint_scaler_contract_matches_pinned_training_stack() -> None:
    repo_root = Path(__file__).resolve().parents[4]
    requirement_lines = {
        line.strip()
        for line in (
            repo_root / "tools/hf_zerogpu/space_template/requirements.txt"
        ).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }

    assert (
        f"transformers=={train_sft.CHECKPOINT_SCALER_TRANSFORMERS_VERSION}"
        in requirement_lines
    )
    assert (
        f"accelerate=={train_sft.CHECKPOINT_SCALER_ACCELERATE_VERSION}"
        in requirement_lines
    )


def _fixture(tmp_path: Path) -> tuple[dict[str, object], Path, Path]:
    run_root = tmp_path / "run"
    dataset_dir = run_root / "generated/fine_tuning/executor/experiments/optimized"
    dataset_dir.mkdir(parents=True)
    (dataset_dir / "train_sft.jsonl").write_text('{"messages":[]}\n', encoding="utf-8")
    (dataset_dir / "val_sft.jsonl").write_text('{"messages":[]}\n', encoding="utf-8")
    _write_json(dataset_dir / "variant_manifest.json", {"digest": "variant"})
    config_path = run_root / "configs/executor.json"
    lineage_path = run_root / "checkpoint_lineage/executor.sft.json"
    config: dict[str, object] = {
        "agent": "executor",
        "base_model_name": "Qwen/Qwen3-1.7B",
        "variantManifestSHA256": "a" * 64,
        "dataset_dir": str(dataset_dir),
        "output_dir": str(run_root / "training/executor"),
        "adapter_output_dir": str(run_root / "models/lora/executor"),
        "sftCheckpointLineagePath": str(lineage_path),
        "sftTokenLengthPreflightPath": str(
            run_root / "training/executor/sft_token_length_preflight.json"
        ),
        "trainingCodeSHA256ByPhase": {"sft": "b" * 64},
        "resolvedTrainingEnvironmentSHA256": "c" * 64,
        "bf16": False,
        "fp16": True,
        "save_total_limit": 2,
    }
    _write_json(config_path, config)
    _write_json(
        lineage_path,
        train_sft._initial_sft_checkpoint_lineage(config, cfg_path=config_path),
    )
    return config, config_path, lineage_path


def _complete_checkpoint(
    root: Path,
    step: int,
    *,
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
        (checkpoint / filename).write_bytes(f"{filename}:{step}".encode())
    if include_scaler:
        (checkpoint / train_sft.CHECKPOINT_SCALER_FILENAME).write_bytes(
            f"scaler.pt:{step}".encode()
        )
    return checkpoint


def test_sft_checkpoint_policy_is_periodic_and_retains_recovery_fallback() -> None:
    assert train_sft._sft_checkpoint_policy({}) == (10, 2)
    assert train_sft._sft_checkpoint_policy(
        {"sft_checkpoint_save_steps": 4, "save_total_limit": 3}
    ) == (4, 3)
    with pytest.raises(ValueError, match="positive integer"):
        train_sft._sft_checkpoint_policy({"sft_checkpoint_save_steps": True})
    with pytest.raises(ValueError, match="at least two"):
        train_sft._sft_checkpoint_policy({"save_total_limit": 1})


def test_sft_resume_selects_latest_bound_complete_checkpoint(tmp_path: Path) -> None:
    config, config_path, lineage_path = _fixture(tmp_path)
    fresh, _, _ = train_sft._bind_and_validate_sft_checkpoint_lineage(
        config,
        cfg_path=config_path,
        assistant_only_loss=True,
        require_checkpoint=False,
    )
    assert fresh is None
    root = Path(str(config["output_dir"]))
    for step in (10, 20):
        train_sft._record_sft_checkpoint(
            lineage_path,
            _complete_checkpoint(root, step),
        )

    selected, _, unbound = train_sft._bind_and_validate_sft_checkpoint_lineage(
        config,
        cfg_path=config_path,
        assistant_only_loss=True,
        require_checkpoint=True,
    )

    assert selected == root / "checkpoint-20"
    assert unbound == []


def test_sft_fp16_resume_rejects_missing_scaler_state(tmp_path: Path) -> None:
    config, config_path, lineage_path = _fixture(tmp_path)
    train_sft._bind_and_validate_sft_checkpoint_lineage(
        config,
        cfg_path=config_path,
        assistant_only_loss=True,
        require_checkpoint=False,
    )
    checkpoint = _complete_checkpoint(Path(str(config["output_dir"])), 10)
    train_sft._record_sft_checkpoint(lineage_path, checkpoint)
    (checkpoint / train_sft.CHECKPOINT_SCALER_FILENAME).unlink()

    with pytest.raises(RuntimeError, match=r"missing scaler\.pt"):
        train_sft._bind_and_validate_sft_checkpoint_lineage(
            config,
            cfg_path=config_path,
            assistant_only_loss=True,
            require_checkpoint=True,
        )


def test_sft_fp16_resume_rejects_tampered_scaler_state(tmp_path: Path) -> None:
    config, config_path, lineage_path = _fixture(tmp_path)
    train_sft._bind_and_validate_sft_checkpoint_lineage(
        config,
        cfg_path=config_path,
        assistant_only_loss=True,
        require_checkpoint=False,
    )
    checkpoint = _complete_checkpoint(Path(str(config["output_dir"])), 10)
    train_sft._record_sft_checkpoint(lineage_path, checkpoint)
    (checkpoint / train_sft.CHECKPOINT_SCALER_FILENAME).write_bytes(b"tampered")

    with pytest.raises(RuntimeError, match="contents drifted"):
        train_sft._bind_and_validate_sft_checkpoint_lineage(
            config,
            cfg_path=config_path,
            assistant_only_loss=True,
            require_checkpoint=True,
        )


def test_sft_bf16_resume_does_not_require_scaler_state(tmp_path: Path) -> None:
    config, config_path, lineage_path = _fixture(tmp_path)
    config["bf16"] = True
    config["fp16"] = False
    _write_json(config_path, config)
    _write_json(
        lineage_path,
        train_sft._initial_sft_checkpoint_lineage(config, cfg_path=config_path),
    )
    train_sft._bind_and_validate_sft_checkpoint_lineage(
        config,
        cfg_path=config_path,
        assistant_only_loss=True,
        require_checkpoint=False,
    )
    checkpoint = _complete_checkpoint(
        Path(str(config["output_dir"])),
        10,
        include_scaler=False,
    )
    train_sft._record_sft_checkpoint(lineage_path, checkpoint)

    selected, _, _ = train_sft._bind_and_validate_sft_checkpoint_lineage(
        config,
        cfg_path=config_path,
        assistant_only_loss=True,
        require_checkpoint=True,
    )

    assert selected == checkpoint
    record = train_sft._read_sft_checkpoint_lineage(lineage_path)
    assert record["checkpointScalerState"] == {
        "schemaVersion": train_sft.CHECKPOINT_SCALER_STATE_SCHEMA,
        "filename": "scaler.pt",
        "required": False,
        "requirement": "required_for_fp16_cuda_native_amp",
        "transformersVersion": "4.57.6",
        "accelerateVersion": "1.14.0",
    }


def test_sft_resume_prunes_partial_rotated_older_and_unbound_newer(
    tmp_path: Path,
) -> None:
    config, config_path, lineage_path = _fixture(tmp_path)
    train_sft._bind_and_validate_sft_checkpoint_lineage(
        config,
        cfg_path=config_path,
        assistant_only_loss=True,
        require_checkpoint=False,
    )
    root = Path(str(config["output_dir"]))
    checkpoint10 = _complete_checkpoint(root, 10)
    train_sft._record_sft_checkpoint(lineage_path, checkpoint10)
    checkpoint20 = _complete_checkpoint(root, 20)
    train_sft._record_sft_checkpoint(lineage_path, checkpoint20)

    # Rotation can be interrupted after partially deleting N-2 and fully
    # writing N, but before on_save replaces the signed [N-2, N-1] record.
    (checkpoint10 / "optimizer.pt").unlink()
    checkpoint30 = _complete_checkpoint(root, 30)

    selected, _, discardable = train_sft._bind_and_validate_sft_checkpoint_lineage(
        config,
        cfg_path=config_path,
        assistant_only_loss=True,
        require_checkpoint=True,
    )

    assert selected == checkpoint20
    assert discardable == [checkpoint10, checkpoint30]
    train_sft._prune_unbound_sft_checkpoints(discardable)
    assert not checkpoint10.exists()
    assert checkpoint20.is_dir()
    assert not checkpoint30.exists()


def test_sft_resume_rejects_complete_hash_drift_in_older_checkpoint(
    tmp_path: Path,
) -> None:
    config, config_path, lineage_path = _fixture(tmp_path)
    train_sft._bind_and_validate_sft_checkpoint_lineage(
        config,
        cfg_path=config_path,
        assistant_only_loss=True,
        require_checkpoint=False,
    )
    root = Path(str(config["output_dir"]))
    checkpoint10 = _complete_checkpoint(root, 10)
    train_sft._record_sft_checkpoint(lineage_path, checkpoint10)
    train_sft._record_sft_checkpoint(lineage_path, _complete_checkpoint(root, 20))
    (checkpoint10 / "optimizer.pt").write_bytes(b"complete-but-drifted")

    with pytest.raises(RuntimeError, match="contents drifted"):
        train_sft._bind_and_validate_sft_checkpoint_lineage(
            config,
            cfg_path=config_path,
            assistant_only_loss=True,
            require_checkpoint=True,
        )


def test_sft_resume_rejects_partial_newest_bound_checkpoint(tmp_path: Path) -> None:
    config, config_path, lineage_path = _fixture(tmp_path)
    train_sft._bind_and_validate_sft_checkpoint_lineage(
        config,
        cfg_path=config_path,
        assistant_only_loss=True,
        require_checkpoint=False,
    )
    root = Path(str(config["output_dir"]))
    checkpoint10 = _complete_checkpoint(root, 10)
    train_sft._record_sft_checkpoint(lineage_path, checkpoint10)
    checkpoint20 = _complete_checkpoint(root, 20)
    train_sft._record_sft_checkpoint(lineage_path, checkpoint20)
    (checkpoint20 / "optimizer.pt").unlink()

    with pytest.raises(RuntimeError, match="checkpoint is incomplete"):
        train_sft._bind_and_validate_sft_checkpoint_lineage(
            config,
            cfg_path=config_path,
            assistant_only_loss=True,
            require_checkpoint=True,
        )


def test_sft_resume_rejects_symlink_inside_stale_partial(tmp_path: Path) -> None:
    config, config_path, lineage_path = _fixture(tmp_path)
    train_sft._bind_and_validate_sft_checkpoint_lineage(
        config,
        cfg_path=config_path,
        assistant_only_loss=True,
        require_checkpoint=False,
    )
    root = Path(str(config["output_dir"]))
    checkpoint10 = _complete_checkpoint(root, 10)
    train_sft._record_sft_checkpoint(lineage_path, checkpoint10)
    train_sft._record_sft_checkpoint(lineage_path, _complete_checkpoint(root, 20))
    (checkpoint10 / "optimizer.pt").unlink()
    (checkpoint10 / "unsafe-link").symlink_to(tmp_path / "outside")

    with pytest.raises(RuntimeError, match="contains a symlink"):
        train_sft._bind_and_validate_sft_checkpoint_lineage(
            config,
            cfg_path=config_path,
            assistant_only_loss=True,
            require_checkpoint=True,
        )


def test_sft_resume_never_trusts_unbound_first_checkpoint(tmp_path: Path) -> None:
    config, config_path, _ = _fixture(tmp_path)
    train_sft._bind_and_validate_sft_checkpoint_lineage(
        config,
        cfg_path=config_path,
        assistant_only_loss=True,
        require_checkpoint=False,
    )
    checkpoint = _complete_checkpoint(Path(str(config["output_dir"])), 10)

    selected, _, unbound = train_sft._bind_and_validate_sft_checkpoint_lineage(
        config,
        cfg_path=config_path,
        assistant_only_loss=True,
        require_checkpoint=True,
    )

    assert selected is None
    assert unbound == [checkpoint]
    train_sft._prune_unbound_sft_checkpoints(unbound)
    assert not checkpoint.exists()


def test_sft_resume_rejects_assistant_loss_and_checkpoint_drift(tmp_path: Path) -> None:
    config, config_path, lineage_path = _fixture(tmp_path)
    train_sft._bind_and_validate_sft_checkpoint_lineage(
        config,
        cfg_path=config_path,
        assistant_only_loss=True,
        require_checkpoint=False,
    )
    checkpoint = _complete_checkpoint(Path(str(config["output_dir"])), 10)
    train_sft._record_sft_checkpoint(lineage_path, checkpoint)

    with pytest.raises(RuntimeError, match="assistant-only-loss setting drifted"):
        train_sft._bind_and_validate_sft_checkpoint_lineage(
            config,
            cfg_path=config_path,
            assistant_only_loss=False,
            require_checkpoint=True,
        )
    (checkpoint / "optimizer.pt").write_bytes(b"drift")
    with pytest.raises(RuntimeError, match="contents drifted"):
        train_sft._bind_and_validate_sft_checkpoint_lineage(
            config,
            cfg_path=config_path,
            assistant_only_loss=True,
            require_checkpoint=True,
        )


class _MaskTokenizer:
    chat_template = "{% generation %}"

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        return_dict: bool = False,
        return_assistant_tokens_mask: bool = False,
        enable_thinking: bool,
    ) -> object:
        del add_generation_prompt, return_assistant_tokens_mask, enable_thinking
        tokens: list[int] = []
        masks: list[int] = []
        for message in messages:
            count = len(message["content"].split()) + 1
            tokens.extend(range(len(tokens), len(tokens) + count))
            masks.extend([1 if message["role"] == "assistant" else 0] * count)
        if tokenize and return_dict:
            return {
                "input_ids": tokens,
                "attention_mask": [1] * len(tokens),
                "assistant_masks": masks,
            }
        if tokenize:
            return tokens
        return "rendered"


def _sft_record(user_words: int, assistant_words: int) -> dict[str, object]:
    return {
        "messages": [
            {"role": "user", "content": "u " * user_words},
            {"role": "assistant", "content": "a " * assistant_words},
        ]
    }


def test_sft_token_preflight_records_total_and_target_distributions() -> None:
    report = train_sft._preflight_sft_token_lengths(
        {
            "train": ([_sft_record(1, 2), _sft_record(3, 1)], Path("train.jsonl")),
            "validation": ([_sft_record(1, 1)], Path("val.jsonl")),
        },
        tokenizer=_MaskTokenizer(),
        max_sequence_length=135,
    )

    assert report["totalTokens"] == {"min": 5, "p50": 6, "p95": 7, "max": 7}
    assert report["assistantTargetTokens"] == {
        "min": 2,
        "p50": 2,
        "p95": 3,
        "max": 3,
    }
    assert report["smallestSequenceMarginTokens"] == 128
    assert report["truncationRequired"] is False


def test_sft_token_preflight_and_row_builder_forbid_truncation() -> None:
    record = _sft_record(3, 2)  # 8 rendered tokens including the controlled template prefix
    with pytest.raises(RuntimeError, match="uses 8 tokens.*exceeding"):
        train_sft._preflight_sft_token_lengths(
            {"train": ([record], Path("train.jsonl"))},
            tokenizer=_MaskTokenizer(),
            max_sequence_length=6,
        )
    with pytest.raises(RuntimeError, match="SFT truncation is forbidden"):
        train_sft.build_sft_rows(
            [record],
            tokenizer=_MaskTokenizer(),
            assistant_only_loss=True,
            path=Path("train.jsonl"),
            max_seq_length=6,
        )


def test_sft_preflight_evidence_is_bound_into_checkpoint_lineage(
    tmp_path: Path,
) -> None:
    config, config_path, lineage_path = _fixture(tmp_path)
    train_sft._bind_and_validate_sft_checkpoint_lineage(
        config,
        cfg_path=config_path,
        assistant_only_loss=True,
        require_checkpoint=False,
    )
    preflight = train_sft._preflight_sft_token_lengths(
        {"train": ([_sft_record(1, 1)], Path("train.jsonl"))},
        tokenizer=_MaskTokenizer(),
        max_sequence_length=133,
    )

    evidence = train_sft._bind_sft_token_length_preflight(
        config,
        cfg_path=config_path,
        preflight=preflight,
    )

    evidence_path = Path(str(config["sftTokenLengthPreflightPath"]))
    assert json.loads(evidence_path.read_text()) == evidence
    assert evidence["trainingCodeSHA256"] == config["trainingCodeSHA256ByPhase"][
        "sft"
    ]
    record = train_sft._read_sft_checkpoint_lineage(lineage_path)
    assert record["tokenLengthPreflightSHA256"] == evidence["preflightSHA256"]
