from pathlib import Path

from lumen_manifest_crawler.crawler import generate_manifest
from lumen_manifest_crawler.dataset import generate_all_datasets


def _first_compiled_record(datasets: dict[str, list[dict]]) -> dict:
    return datasets["train_sft"][0]


def test_deterministic_datasets_omit_commit_lineage_fields():
    manifest = generate_manifest(Path(".").resolve())
    datasets = generate_all_datasets(manifest, deterministic=True)

    dataset_manifest = datasets["dataset_manifest"][0]
    first_train_record = _first_compiled_record(datasets)

    assert dataset_manifest["manifest"]["commit"] is None
    assert first_train_record["metadata"]["manifestCommit"] is None
    assert first_train_record["grounding"]["sourceIntegrityCommit"] is None


def test_non_deterministic_datasets_include_commit_lineage_fields():
    manifest = generate_manifest(Path(".").resolve())
    datasets = generate_all_datasets(manifest, deterministic=False)

    dataset_manifest = datasets["dataset_manifest"][0]
    first_train_record = _first_compiled_record(datasets)

    assert dataset_manifest["manifest"]["commit"] == manifest.sourceIntegrity.commit
    assert first_train_record["metadata"]["manifestCommit"] == manifest.sourceIntegrity.commit
    assert first_train_record["grounding"]["sourceIntegrityCommit"] == manifest.sourceIntegrity.commit
