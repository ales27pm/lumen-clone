from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.fine_tuning.unsloth import train_dpo, ubuntu_pipeline


PARENT_SFT_SHA256 = "a" * 64


class _Dataset:
    def __init__(self, columns: dict[str, list[object]]) -> None:
        if not columns or len({len(values) for values in columns.values()}) != 1:
            raise ValueError("test dataset columns must have one shared row count")
        self._columns = {name: list(values) for name, values in columns.items()}
        self.column_names = list(columns)

    def __len__(self) -> int:
        return len(next(iter(self._columns.values())))

    def __getitem__(self, name: str) -> list[object]:
        return list(self._columns[name])

    def add_column(self, name: str, values: list[object]) -> "_Dataset":
        if name in self._columns:
            raise ValueError("duplicate test column")
        return _Dataset({**self._columns, name: list(values)})


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _fixture(
    tmp_path: Path,
) -> tuple[dict[str, object], Path, Path, Path, dict[str, list[dict[str, object]]]]:
    run_root = tmp_path / "run"
    dataset_dir = run_root / "generated/fine_tuning/cortex/variant"
    dataset_dir.mkdir(parents=True)
    raw_rows = {
        split: {
            "prompt": [{"role": "user", "content": split}],
            "chosen": {"role": "assistant", "content": "yes"},
            "rejected": {"role": "assistant", "content": "no"},
        }
        for split in ("train", "validation")
    }
    (dataset_dir / "train_dpo.jsonl").write_text(
        json.dumps(raw_rows["train"]) + "\n",
        encoding="utf-8",
    )
    (dataset_dir / "val_dpo.jsonl").write_text(
        json.dumps(raw_rows["validation"]) + "\n",
        encoding="utf-8",
    )
    _write_json(dataset_dir / "variant_manifest.json", {"status": "bound"})
    config_path = run_root / "configs/cortex.json"
    lineage_path = run_root / "checkpoint_lineage/cortex.preference.json"
    output_dir = run_root / "training/cortex/dpo"
    config: dict[str, object] = {
        "agent": "cortex",
        "preference_trainer": "dpo",
        "base_model_name": "Qwen/Qwen3-1.7B",
        "variantManifestSHA256": "b" * 64,
        "dataset_dir": str(dataset_dir),
        "output_dir": str(run_root / "training/cortex"),
        "preferenceCheckpointLineagePath": str(lineage_path),
        "preferenceTokenLengthPreflightPath": str(
            output_dir / "token_length_preflight.json"
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
    train_dpo._bind_and_validate_preference_checkpoint_lineage(
        config,
        cfg_path=config_path,
        preference_trainer="dpo",
        parent_sft_adapter_sha256=PARENT_SFT_SHA256,
        require_checkpoint=False,
    )
    source_rows = {
        split: [train_dpo.row_to_preference(raw_rows[split])]
        for split in ("train", "validation")
    }
    return config, config_path, lineage_path, output_dir, source_rows


def _trainer_with_reference_columns() -> SimpleNamespace:
    return SimpleNamespace(
        train_dataset=_Dataset(
            {
                "prompt": ["train"],
                "ref_chosen_logps": [-1.25],
                "ref_rejected_logps": [-2.5],
            }
        ),
        eval_dataset=_Dataset(
            {
                "prompt": ["validation"],
                "ref_chosen_logps": [-0.75],
                "ref_rejected_logps": [-3.0],
            }
        ),
    )


def test_reference_evidence_binds_exact_columns_and_restores_on_resume(
    tmp_path: Path,
) -> None:
    config, config_path, lineage_path, output_dir, source_rows = _fixture(tmp_path)
    evidence_path = output_dir / train_dpo.DPO_REFERENCE_LOG_PROB_EVIDENCE_FILENAME
    evidence = train_dpo._build_reference_log_prob_evidence(
        _trainer_with_reference_columns(),
        config,
        cfg_path=config_path,
        parent_sft_adapter_sha256=PARENT_SFT_SHA256,
        source_rows_by_split=source_rows,
    )
    train_dpo._write_json_atomic(evidence_path, evidence)
    train_dpo._bind_reference_log_prob_evidence(
        lineage_path,
        evidence_path,
        evidence,
    )

    resumed = SimpleNamespace(
        train_dataset=_Dataset({"prompt": ["train"]}),
        eval_dataset=_Dataset({"prompt": ["validation"]}),
        _precomputed_train_ref_log_probs=False,
        _precomputed_eval_ref_log_probs=False,
    )
    precomputed, restored, reused = train_dpo._prepare_reference_log_prob_evidence(
        resumed,
        SimpleNamespace(),
        config,
        cfg_path=config_path,
        parent_sft_adapter_sha256=PARENT_SFT_SHA256,
        source_rows_by_split=source_rows,
        evidence_path=evidence_path,
        checkpoint_lineage_path=lineage_path,
        reuse_existing=True,
    )

    assert precomputed == {"train": True, "evaluation": True}
    assert reused is True
    assert restored == evidence
    assert resumed.train_dataset["ref_chosen_logps"] == [-1.25]
    assert resumed.eval_dataset["ref_rejected_logps"] == [-3.0]
    record = train_dpo._read_preference_checkpoint_lineage(lineage_path)
    assert (
        record["referenceLogProbEvidenceSHA256"]
        == evidence["referenceLogProbEvidenceSHA256"]
    )


def test_reference_evidence_rejects_nonfinite_and_source_drift(tmp_path: Path) -> None:
    config, config_path, lineage_path, output_dir, source_rows = _fixture(tmp_path)
    evidence = train_dpo._build_reference_log_prob_evidence(
        _trainer_with_reference_columns(),
        config,
        cfg_path=config_path,
        parent_sft_adapter_sha256=PARENT_SFT_SHA256,
        source_rows_by_split=source_rows,
    )
    tampered = json.loads(json.dumps(evidence))
    tampered["splits"]["train"]["refChosenLogpsIEEE754Binary32"][0] = "7f800000"
    with pytest.raises(RuntimeError, match="non-finite"):
        train_dpo.verify_reference_log_prob_evidence(tampered)

    evidence_path = output_dir / train_dpo.DPO_REFERENCE_LOG_PROB_EVIDENCE_FILENAME
    train_dpo._write_json_atomic(evidence_path, evidence)
    train_dpo._bind_reference_log_prob_evidence(lineage_path, evidence_path, evidence)
    drifted_rows = {**source_rows, "train": [{"prompt": "changed"}]}
    with pytest.raises(RuntimeError, match="sourceRowSHA256 drifted"):
        train_dpo._prepare_reference_log_prob_evidence(
            SimpleNamespace(
                train_dataset=_Dataset({"prompt": ["train"]}),
                eval_dataset=_Dataset({"prompt": ["validation"]}),
            ),
            SimpleNamespace(),
            config,
            cfg_path=config_path,
            parent_sft_adapter_sha256=PARENT_SFT_SHA256,
            source_rows_by_split=drifted_rows,
            evidence_path=evidence_path,
            checkpoint_lineage_path=lineage_path,
            reuse_existing=True,
        )


def test_reference_adapter_removal_requires_qualified_evidence(
    tmp_path: Path,
) -> None:
    config, config_path, _, _, source_rows = _fixture(tmp_path)
    trainer = _trainer_with_reference_columns()

    class Model:
        def __init__(self) -> None:
            self.peft_config = {
                train_dpo.POLICY_ADAPTER_NAME: object(),
                train_dpo.REFERENCE_ADAPTER_NAME: object(),
            }

        def set_adapter(self, _name: str) -> None:
            pass

        def delete_adapter(self, name: str) -> None:
            del self.peft_config[name]

    trainer.model = Model()
    trainer._precomputed_train_ref_log_probs = True
    trainer._precomputed_eval_ref_log_probs = True
    evidence = train_dpo._build_reference_log_prob_evidence(
        trainer,
        config,
        cfg_path=config_path,
        parent_sft_adapter_sha256=PARENT_SFT_SHA256,
        source_rows_by_split=source_rows,
    )

    contract = train_dpo._drop_precomputed_reference_adapter(
        trainer,
        reference_log_prob_evidence=evidence,
    )

    assert contract == {
        "referenceAdapterRemovedAfterPrecompute": True,
        "checkpointAdapterNames": [train_dpo.POLICY_ADAPTER_NAME],
    }


def test_pipeline_independently_reconstructs_reference_evidence(
    tmp_path: Path,
) -> None:
    config, config_path, lineage_path, output_dir, source_rows = _fixture(tmp_path)
    evidence_path = output_dir / train_dpo.DPO_REFERENCE_LOG_PROB_EVIDENCE_FILENAME
    evidence = train_dpo._build_reference_log_prob_evidence(
        _trainer_with_reference_columns(),
        config,
        cfg_path=config_path,
        parent_sft_adapter_sha256=PARENT_SFT_SHA256,
        source_rows_by_split=source_rows,
    )
    train_dpo._write_json_atomic(evidence_path, evidence)
    train_dpo._bind_reference_log_prob_evidence(
        lineage_path,
        evidence_path,
        evidence,
    )
    report = {
        "train_records": 1,
        "val_records": 1,
        "reference_log_probs_precomputed": {"train": True, "evaluation": True},
        "checkpoint_adapter_contract": {
            "referenceAdapterRemovedAfterPrecompute": True,
            "checkpointAdapterNames": [train_dpo.POLICY_ADAPTER_NAME],
        },
        "reference_log_prob_evidence": {
            "path": str(evidence_path),
            "referenceLogProbEvidenceSHA256": evidence[
                "referenceLogProbEvidenceSHA256"
            ],
            "fileSHA256": ubuntu_pipeline.file_sha256(evidence_path),
            "reusedFromCheckpointLineage": False,
            "trainRowCount": 1,
            "validationRowCount": 1,
        },
    }

    verified = ubuntu_pipeline._verify_dpo_reference_log_prob_report(
        run_root=tmp_path / "run",
        agent="cortex",
        config=config,
        report=report,
        parent_sft_adapter_sha256=PARENT_SFT_SHA256,
    )

    assert (
        verified["referenceLogProbEvidenceSHA256"]
        == evidence["referenceLogProbEvidenceSHA256"]
    )
