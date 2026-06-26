from __future__ import annotations

import json
from pathlib import Path

from lumen_manifest_crawler.crawler import generate_manifest
from lumen_manifest_crawler.dataset import generate_all_datasets
from lumen_manifest_crawler.dataset.reranker import RERANKER_MODEL_ID, compile_reranker_datasets
from lumen_manifest_crawler.output.writer import write_outputs
from lumen_manifest_crawler.validators import validate_manifest


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def test_reranker_dataset_families_are_generated() -> None:
    manifest = generate_manifest(_repo_root())
    datasets = generate_all_datasets(manifest)

    required = {
        "reranker_train_pairs",
        "reranker_val_pairs",
        "reranker_hard_negative_pairs",
        "reranker_eval_reranking",
        "reranker_dataset_card",
    }

    assert required.issubset(datasets.keys())
    for family in required:
        assert datasets[family], f"{family} should not be empty"

    card = datasets["reranker_dataset_card"][0]
    assert card["model"] == RERANKER_MODEL_ID
    assert card["task"] == "candidate_reranking"
    assert card["counts"]["trainPairs"] == len(datasets["reranker_train_pairs"])
    assert card["counts"]["hardNegativePairs"] == len(datasets["reranker_hard_negative_pairs"])


def test_reranker_records_are_pairwise_not_chat_sft() -> None:
    manifest = generate_manifest(_repo_root())
    datasets = generate_all_datasets(manifest)

    for record in datasets["reranker_train_pairs"][:50]:
        assert record["task"] == "pairwise_reranking"
        assert "query" in record
        assert "positiveDocumentID" in record
        assert "negativeDocumentID" in record
        assert record["positiveDocumentID"] != record["negativeDocumentID"]
        assert record["documents"][0]["relevance"] == 1.0
        assert record["documents"][1]["relevance"] == 0.0
        assert "messages" not in record

    for record in datasets["reranker_eval_reranking"][:50]:
        assert record["task"] == "candidate_reranking_eval"
        assert "candidateDocuments" in record
        assert any(candidate["relevance"] == 1.0 for candidate in record["candidateDocuments"])
        assert any(candidate["relevance"] == 0.0 for candidate in record["candidateDocuments"])
        assert "messages" not in record


def test_reranker_dedicated_output_directory_is_written(tmp_path: Path) -> None:
    manifest = generate_manifest(_repo_root())
    datasets = generate_all_datasets(manifest)
    report = validate_manifest(manifest, datasets)
    output = tmp_path / "agent_manifest"

    write_outputs(output, manifest, report, datasets, pretty=True)

    reranker_dir = output / "reranker"
    expected_files = {
        "train_pairs.jsonl",
        "val_pairs.jsonl",
        "hard_negative_pairs.jsonl",
        "eval_reranking.jsonl",
        "dataset_card.json",
    }
    assert reranker_dir.exists()
    assert expected_files.issubset({path.name for path in reranker_dir.iterdir()})

    card = json.loads((reranker_dir / "dataset_card.json").read_text(encoding="utf-8"))
    assert card["model"] == RERANKER_MODEL_ID
    assert card["counts"]["trainPairs"] == sum(1 for _ in (reranker_dir / "train_pairs.jsonl").open(encoding="utf-8"))


def test_reranker_compile_is_deterministic() -> None:
    manifest = generate_manifest(_repo_root())
    datasets = generate_all_datasets(manifest)
    embedding_only = {key: value for key, value in datasets.items() if key.startswith("embedding_")}

    first = compile_reranker_datasets(embedding_only)
    second = compile_reranker_datasets(embedding_only)

    assert first.dataset_card == second.dataset_card
    assert first.train_pairs == second.train_pairs
    assert first.val_pairs == second.val_pairs
    assert first.hard_negative_pairs == second.hard_negative_pairs
