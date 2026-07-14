from __future__ import annotations

import json
from pathlib import Path

import pytest

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


def test_embedding_tool_schema_carries_runtime_contract() -> None:
    manifest = generate_manifest(_repo_root())
    datasets = generate_all_datasets(manifest)
    tool = next(item for item in manifest.tools if item.id == "calendar.create")
    record = next(item for item in datasets["embedding_corpus"] if item["objectType"] == "tool_schema" and item["objectID"] == tool.id)
    metadata = record["metadata"]

    assert f"Permission kind: {tool.permissionKind}" in record["text"]
    assert f"Confirmation mode: {tool.confirmationMode}" in record["text"]
    assert metadata["requiresApproval"] is tool.requiresApproval
    assert metadata["permissionKey"] == tool.permissionKey
    assert metadata["permissionKind"] == tool.permissionKind
    assert metadata["confirmationMode"] == tool.confirmationMode


@pytest.mark.slow
def test_embedding_corpus_contains_codebase_home_when_root_is_provided() -> None:
    repo_root = _repo_root()
    manifest = generate_manifest(repo_root)
    datasets = generate_all_datasets(manifest, root=repo_root)
    object_types = {record["objectType"] for record in datasets["embedding_corpus"]}
    codebase_records = [record for record in datasets["codebase_home_corpus"] if record.get("path") != "."]

    assert "codebase_home_module" in object_types
    assert codebase_records
    assert any(record["path"].endswith("AgentService.swift") for record in codebase_records)
    assert any(record["path"].startswith("tools/lumen_manifest_crawler/") for record in codebase_records)
    assert any(pair["family"] == "codebase_home_corpus_retrieval" for pair in datasets["embedding_train_pairs"] + datasets["embedding_val_pairs"])


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

    alias = output / "dataset" / "embedding_corpus.jsonl"
    assert alias.is_symlink()
    assert alias.readlink() == Path("../embedding/corpus.jsonl")
    assert alias.resolve() == (embedding_dir / "corpus.jsonl").resolve()

    card = json.loads((embedding_dir / "dataset_card.json").read_text(encoding="utf-8"))
    assert card["model"] == EMBEDDING_MODEL_ID
    assert card["counts"]["corpus"] == sum(1 for _ in (embedding_dir / "corpus.jsonl").open(encoding="utf-8"))


@pytest.mark.slow
def test_runtime_grounding_bundle_is_written_for_build_injection(tmp_path: Path) -> None:
    repo_root = _repo_root()
    manifest = generate_manifest(repo_root)
    datasets = generate_all_datasets(manifest, root=repo_root)
    report = validate_manifest(manifest, datasets)
    output = tmp_path / "agent_manifest"

    write_outputs(output, manifest, report, datasets, pretty=True)

    bundle_path = output / "runtime_grounding_bundle.json"
    prompt_path = output / "runtime_grounding_prompt.md"
    assert bundle_path.exists()
    assert prompt_path.exists()

    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    assert bundle["artifactKind"] == "agent_grounding_runtime_bundle"
    assert bundle["sourceIntegrity"] == manifest.sourceIntegrity.lineage_dict()
    assert bundle["injectionPolicy"]["target"] == "AgentGroundingPromptComposer"
    assert bundle["codebaseHome"]["recordCount"] == len(datasets["codebase_home_corpus"])
    assert bundle["codebaseHome"]["selectedFiles"]
    assert "Bundled source grounding" not in prompt_path.read_text(encoding="utf-8")
    assert "Lumen Runtime Grounding Bundle" in prompt_path.read_text(encoding="utf-8")


def test_embedding_compile_is_deterministic() -> None:
    manifest = generate_manifest(_repo_root())
    datasets = generate_all_datasets(manifest)

    first = compile_embedding_datasets(manifest, datasets)
    second = compile_embedding_datasets(manifest, datasets)

    assert first.dataset_card == second.dataset_card
    assert first.corpus == second.corpus
    assert first.train_pairs == second.train_pairs
    assert first.train_triplets == second.train_triplets
