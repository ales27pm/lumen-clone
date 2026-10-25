"""Dataset names must not escape the output roots selected by the caller."""

from __future__ import annotations

from pathlib import Path

import pytest

from lumen_manifest_crawler.dataset.fine_tuning import AgentFineTuningDataset
from lumen_manifest_crawler.manifest import AgentBehaviorManifest, ValidationReport
from lumen_manifest_crawler.output import writer


def dataset(*, variants: tuple[str, ...] = ()) -> AgentFineTuningDataset:
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
        experiment_manifest={},
        experiment_variants={
            name: {
                "train_sft": [],
                "val_sft": [],
                "train_dpo": [],
                "val_dpo": [],
                "contamination_report": {},
                "variant_manifest": {"variant": "synthetic"},
            }
            for name in variants
        },
    )


@pytest.mark.parametrize(
    "name", ("", ".", "..", "a/b", "a\\b", "a\0b", None, 1, True, "\ud800", "\udc80")
)
@pytest.mark.parametrize("kind", ("agent", "variant", "dataset_family"))
def test_invalid_component_is_rejected_before_locks_or_any_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: object, kind: str
) -> None:
    def unexpected_write(*args: object, **kwargs: object) -> None:
        pytest.fail("invalid output names must be rejected before locks or writing")

    monkeypatch.setattr(writer, "_generation_locks", unexpected_write)
    monkeypatch.setattr(writer, "_write_outputs_unlocked", unexpected_write)
    datasets = {name: []} if kind == "dataset_family" else {}
    fine_tuning = (
        {name: dataset()}
        if kind == "agent"
        else {"executor": dataset(variants=(name,))} if kind == "variant" else None
    )
    with pytest.raises(ValueError) as error:
        writer.write_outputs(
            tmp_path / "manifest",
            AgentBehaviorManifest(),
            ValidationReport(passed=False),
            datasets,  # type: ignore[arg-type]
            pretty=False,
            fine_tuning_datasets=fine_tuning,  # type: ignore[arg-type]
            fine_tuning_output_dir=tmp_path / "fine-tuning",
        )
    assert str(error.value) == "Output names must be nonempty path components"
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("kind", ("agent", "variant"))
def test_fine_tuning_rejection_precedes_outside_write_and_stale_deletion(
    tmp_path: Path, kind: str
) -> None:
    root = tmp_path / "requested-output"
    escaped = tmp_path / "escaped"
    sentinel = escaped / "experiments" / "stale" / "sentinel.txt"
    sentinel.parent.mkdir(parents=True)
    sentinel.write_bytes(b"synthetic sentinel\n")
    fine_tuning = (
        {"../escaped": dataset()}
        if kind == "agent"
        else {"executor": dataset(variants=("../../../escaped",))}
    )

    with pytest.raises(ValueError, match="^Output names must be nonempty path components$"):
        writer._write_fine_tuning_outputs(root, fine_tuning)

    assert sentinel.read_bytes() == b"synthetic sentinel\n"
    assert not (escaped / "train_sft.jsonl").exists()
    assert not root.exists()


def test_family_rejection_precedes_outside_unlink_and_other_publication(tmp_path: Path) -> None:
    root = tmp_path / "requested-output"
    sentinel = tmp_path / "outside.jsonl"
    sentinel.write_bytes(b"synthetic sentinel\n")

    with pytest.raises(ValueError, match="^Output names must be nonempty path components$"):
        writer.write_outputs(
            root,
            AgentBehaviorManifest(),
            ValidationReport(passed=False),
            {"../../outside": []},
            pretty=False,
        )

    assert sentinel.read_bytes() == b"synthetic sentinel\n"
    assert not root.exists()


def test_all_names_are_checked_before_valid_earlier_datasets_are_written(tmp_path: Path) -> None:
    root = tmp_path / "fine-tuning"
    with pytest.raises(ValueError, match="^Output names must be nonempty path components$"):
        writer._write_fine_tuning_outputs(
            root,
            {"a-valid": dataset(), "z-invalid": dataset(variants=("../escape",))},
        )
    assert not root.exists()


@pytest.mark.parametrize(
    "name", ("executor", "version-2", "v1.2", ".hidden", "étude", "two words", "😀")
)
def test_ordinary_single_component_names_remain_supported(tmp_path: Path, name: str) -> None:
    root = tmp_path / "fine-tuning"
    writer._write_fine_tuning_outputs(root, {name: dataset(variants=(name,))})
    assert (root / name / "train_sft.jsonl").is_file()
    assert (root / name / "experiments" / name / "variant_manifest.json").is_file()


def test_canonical_dataset_aliases_remain_supported(tmp_path: Path) -> None:
    root = tmp_path / "manifest"
    writer.write_outputs(
        root,
        AgentBehaviorManifest(),
        ValidationReport(passed=True),
        {name: [] for name in (*writer.CANONICAL_DATASET_ALIASES, "dataset_manifest")},
        pretty=False,
    )
    assert (root / "AgentBehaviorManifest.json").is_file()
