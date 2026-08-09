from __future__ import annotations

import csv
import json
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from lumen_manifest_crawler.dataset.fine_tuning import AgentFineTuningDataset
from lumen_manifest_crawler.output import writer as writer_module
from lumen_manifest_crawler.output.writer import (
    ConcurrentGenerationError,
    _generation_lock,
    _generation_lock_path,
    _write_dataset_index,
    _write_fine_tuning_outputs,
    _write_jsonl,
)


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


def test_generation_lock_rejects_same_output_but_not_distinct_output(
    tmp_path: Path,
) -> None:
    first_output = tmp_path / "first" / "agent_manifest"
    second_output = tmp_path / "second" / "agent_manifest"
    child_script = """
import sys
from pathlib import Path
from lumen_manifest_crawler.output.writer import _generation_lock

with _generation_lock(Path(sys.argv[1])):
    print("locked", flush=True)
    sys.stdin.readline()
"""
    process = subprocess.Popen(
        [sys.executable, "-c", child_script, str(first_output)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert process.stdout is not None
    assert process.stdin is not None
    try:
        assert process.stdout.readline().strip() == "locked"

        with pytest.raises(
            ConcurrentGenerationError,
            match="already writing.*output tree",
        ):
            with _generation_lock(first_output):
                pytest.fail("same-output lock acquisition must fail fast")

        with _generation_lock(second_output):
            pass

        lock_path = _generation_lock_path(first_output)
        assert first_output not in lock_path.parents
        assert first_output.parent not in lock_path.parents
    finally:
        process.stdin.write("release\n")
        process.stdin.flush()
        process.stdin.close()
        return_code = process.wait(timeout=10)
        stderr = process.stderr.read() if process.stderr is not None else ""
    assert return_code == 0, stderr

    with _generation_lock(first_output):
        pass


def test_write_outputs_holds_locks_across_every_owned_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "agent_manifest"
    cross_model_output = tmp_path / "cross_model_training"
    fine_tuning_output = tmp_path / "fine_tuning"
    writer_was_called = False

    def assert_locked_writer(*args: object, **kwargs: object) -> None:
        nonlocal writer_was_called
        writer_was_called = True
        assert args[0] == output
        assert kwargs["cross_model_train_dir"] == cross_model_output
        assert kwargs["fine_tuning_output_dir"] == fine_tuning_output
        for owned_output in (output, cross_model_output, fine_tuning_output):
            with pytest.raises(ConcurrentGenerationError):
                with _generation_lock(owned_output):
                    pytest.fail(f"the owning writer must keep {owned_output} locked")

    monkeypatch.setattr(writer_module, "_write_outputs_unlocked", assert_locked_writer)

    writer_module.write_outputs(
        output,
        object(),  # type: ignore[arg-type]
        object(),  # type: ignore[arg-type]
        {},
        pretty=False,
        cross_model_train_dir=cross_model_output,
        fine_tuning_datasets={},
        fine_tuning_output_dir=fine_tuning_output,
    )

    assert writer_was_called


@pytest.mark.parametrize(
    "shared_tree",
    ("cross_model_train_dir", "fine_tuning_output_dir"),
)
def test_write_outputs_rejects_shared_secondary_tree_with_distinct_manifests(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    shared_tree: str,
) -> None:
    first_output = tmp_path / "first_manifest"
    second_output = tmp_path / "second_manifest"
    shared_output = tmp_path / "shared_secondary"
    writer_calls: list[Path] = []

    first_kwargs = {
        "cross_model_train_dir": tmp_path / "first_cross_model",
        "fine_tuning_output_dir": tmp_path / "first_fine_tuning",
    }
    second_kwargs = {
        "cross_model_train_dir": tmp_path / "second_cross_model",
        "fine_tuning_output_dir": tmp_path / "second_fine_tuning",
    }
    first_kwargs[shared_tree] = shared_output
    second_kwargs[shared_tree] = shared_output

    def first_writer(*args: object, **kwargs: object) -> None:
        writer_calls.append(args[0])  # type: ignore[arg-type]
        if args[0] == first_output:
            with pytest.raises(ConcurrentGenerationError):
                writer_module.write_outputs(
                    second_output,
                    object(),  # type: ignore[arg-type]
                    object(),  # type: ignore[arg-type]
                    {},
                    pretty=False,
                    fine_tuning_datasets={},
                    **second_kwargs,  # type: ignore[arg-type]
                )
        else:  # pragma: no cover - reached only if the secondary lock regresses
            return

    monkeypatch.setattr(writer_module, "_write_outputs_unlocked", first_writer)

    writer_module.write_outputs(
        first_output,
        object(),  # type: ignore[arg-type]
        object(),  # type: ignore[arg-type]
        {},
        pretty=False,
        fine_tuning_datasets={},
        **first_kwargs,  # type: ignore[arg-type]
    )

    assert writer_calls == [first_output]


def test_jsonl_replacement_never_exposes_partial_or_interleaved_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "dataset.jsonl"
    old_bytes = b'{"generation": "old"}\n'
    records = [
        {"generation": "new", "ordinal": ordinal, "payload": "x" * 4096}
        for ordinal in range(4)
    ]
    expected_bytes = "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
        for record in records
    ).encode("utf-8")
    path.write_bytes(old_bytes)
    path.chmod(0o640)

    second_record_started = threading.Event()
    allow_writer_to_finish = threading.Event()
    original_dumps = json.dumps
    dump_count = 0

    def blocking_dumps(*args: object, **kwargs: object) -> str:
        nonlocal dump_count
        dump_count += 1
        if dump_count == 2:
            second_record_started.set()
            assert allow_writer_to_finish.wait(timeout=10)
        return original_dumps(*args, **kwargs)

    monkeypatch.setattr(writer_module.json, "dumps", blocking_dumps)
    writer_errors: list[BaseException] = []

    def write_records() -> None:
        try:
            _write_jsonl(path, records)
        except BaseException as error:  # pragma: no cover - asserted below
            writer_errors.append(error)

    writer = threading.Thread(target=write_records)
    writer.start()
    assert second_record_started.wait(timeout=10)

    observed = {path.read_bytes()}
    allow_writer_to_finish.set()
    while writer.is_alive():
        observed.add(path.read_bytes())
    writer.join(timeout=10)
    observed.add(path.read_bytes())

    assert not writer_errors
    assert not writer.is_alive()
    assert observed <= {old_bytes, expected_bytes}
    assert old_bytes in observed
    assert path.read_bytes() == expected_bytes
    assert path.stat().st_mode & 0o777 == 0o640
    assert not list(tmp_path.glob(f".{path.name}.*.tmp"))


def test_new_jsonl_uses_deterministic_artifact_mode(tmp_path: Path) -> None:
    path = tmp_path / "new.jsonl"

    _write_jsonl(path, [{"value": "deterministic"}])

    assert path.read_bytes() == b'{"value": "deterministic"}\n'
    assert path.stat().st_mode & 0o777 == 0o644
