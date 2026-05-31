from __future__ import annotations

import json
from pathlib import Path

from lumen_manifest_crawler.crawler import generate_manifest
from lumen_manifest_crawler.dataset import generate_all_datasets
from lumen_manifest_crawler.dataset.embedding import EMBEDDING_MODEL_ID, compile_embedding_datasets
from lumen_manifest_crawler.output.writer import write_outputs
from lumen_manifest_crawler.validators import validate_manifest


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def test_embedding_dataset_families_are_generated() -> None:
    manifest = generate_manifest(_repo_root())
    datasets = generate_all_datasets(manifest)

    required = {
        "embedding_corpus",
        "embedding_train_pairs",
        "embedding_val_pairs",
        "embedding_train_triplets",
        "embedding_val_triplets",
        "embedding_hard_negatives",
        "embedding_eval_retrieval",
        "embedding_dataset_card",
    }

    assert required.issubset(datasets.keys())
    for family in required:
        assert datasets[family], f"{family} should not be empty"

    card = datasets["embedding_dataset_card"][0]
    assert card["model"] == EMBEDDING_MODEL_ID
    assert card["task"] == "retrieval_similarity_ranking"
    assert card["counts"]["corpus"] == len(datasets["embedding_corpus"])
    assert card["counts"]["hardNegatives"] == len(datasets["embedding_hard_negatives"])


def test_embedding_records_are_retrieval_not_chat_sft() -> None:
    manifest = generate_manifest(_repo_root())
    datasets = generate_all_datasets(manifest)

    for record in datasets["embedding_train_pairs"][:50]:
        assert "query" in record
        assert "documentID" in record
        assert "messages" not in record
        assert record["label"] == 1.0

    for record in datasets["embedding_train_triplets"][:50]:
        assert "query" in record
        assert "positiveDocumentID" in record
        assert "negativeDocumentID" in record
        assert record["positiveDocumentID"] != record["negativeDocumentID"]
        assert "messages" not in record


def test_embedding_corpus_contains_core_lumen_object_types() -> None:
    manifest = generate_manifest(_repo_root())
    datasets = generate_all_datasets(manifest)
    object_types = {record["objectType"] for record in datasets["embedding_corpus"]}

    assert "tool_schema" in object_types
    assert "intent" in object_types
    assert "routing_rule" in object_types
    assert "fleet_slot" in object_types
    assert "source_code_map_entry" in object_types


def test_embedding_dedicated_output_directory_is_written(tmp_path: Path) -> None:
    manifest = generate_manifest(_repo_root())
    datasets = generate_all_datasets(manifest)
    report = validate_manifest(manifest, datasets)
    output = tmp_path / "agent_manifest"

    write_outputs(output, manifest, report, datasets, pretty=True)

    embedding_dir = output / "embedding"
    expected_files = {
        "corpus.jsonl",
        "train_pairs.jsonl",
        "val_pairs.jsonl",
        "train_triplets.jsonl",
        "val_triplets.jsonl",
        "hard_negatives.jsonl",
        "eval_retrieval.jsonl",
        "dataset_card.json",
    }
    assert embedding_dir.exists()
    assert expected_files.issubset({path.name for path in embedding_dir.iterdir()})

    card = json.loads((embedding_dir / "dataset_card.json").read_text(encoding="utf-8"))
    assert card["model"] == EMBEDDING_MODEL_ID
    assert card["counts"]["corpus"] == sum(1 for _ in (embedding_dir / "corpus.jsonl").open(encoding="utf-8"))


def test_embedding_compile_is_deterministic() -> None:
    manifest = generate_manifest(_repo_root())
    datasets = generate_all_datasets(manifest)

    first = compile_embedding_datasets(manifest, datasets)
    second = compile_embedding_datasets(manifest, datasets)

    assert first.dataset_card == second.dataset_card
    assert first.corpus == second.corpus
    assert first.train_pairs == second.train_pairs
    assert first.train_triplets == second.train_triplets
