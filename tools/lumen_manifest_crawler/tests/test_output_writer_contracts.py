from __future__ import annotations

import csv
from pathlib import Path

from lumen_manifest_crawler.dataset.fine_tuning import AgentFineTuningDataset
from lumen_manifest_crawler.output.writer import _write_dataset_index, _write_fine_tuning_outputs


def _dataset(*, variants: tuple[str, ...]) -> AgentFineTuningDataset:
    variant_artifacts = {
        variant: {
            "train_sft": [],
            "val_sft": [],
            "train_dpo": [],
            "val_dpo": [],
            "contamination_report": {},
            "variant_manifest": {"variant": variant},
        }
        for variant in variants
    }
    return AgentFineTuningDataset(
        agent="executor",
        train_sft=[],
        val_sft=[],
        train_dpo=[],
        val_dpo=[],
        eval=[],
        dataset_card={"systemPrompt": "Executor"},
        unsloth_config={"baseModelID": "Qwen/Qwen3-1.7B"},
        contamination_report={},
        experiment_variants=variant_artifacts,
        experiment_manifest={},
    )


def test_dataset_index_omits_families_without_dataset_files(tmp_path: Path) -> None:
    index_path = tmp_path / "dataset_index.csv"

    _write_dataset_index(
        index_path,
        {
            "dataset_manifest": [{"schema": "test"}],
            "empty_family": [],
            "present_family": [{"split": "train", "agentRole": "executor", "taskType": "tool"}],
        },
    )

    with index_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["family"] for row in rows] == ["present_family"]


def test_fine_tuning_writer_removes_stale_experiment_variants(tmp_path: Path) -> None:
    root = tmp_path / "fine_tuning"
    experiments = root / "executor" / "experiments"
    stale = experiments / "stale_variant"
    stale.mkdir(parents=True)
    (stale / "variant_manifest.json").write_text("{}", encoding="utf-8")

    _write_fine_tuning_outputs(root, {"executor": _dataset(variants=("current_variant",))})

    assert not stale.exists()
    assert (experiments / "current_variant" / "variant_manifest.json").is_file()
