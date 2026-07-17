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


def _normalized_query(value: object) -> str:
    return " ".join(str(value or "").strip().split()).casefold()


def _known_positive_documents(datasets: dict[str, list[dict]]) -> dict[str, set[str]]:
    known: dict[str, set[str]] = {}
    for family in ("embedding_train_pairs", "embedding_val_pairs"):
        for record in datasets[family]:
            known.setdefault(_normalized_query(record["query"]), set()).add(record["documentID"])
    for record in datasets["embedding_eval_retrieval"]:
        known.setdefault(_normalized_query(record["query"]), set()).update(record["positiveDocumentIDs"])
    return known


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


def test_embedding_train_validation_and_eval_are_held_out_by_query_and_document() -> None:
    manifest = generate_manifest(_repo_root())
    datasets = generate_all_datasets(manifest)

    train_pairs = datasets["embedding_train_pairs"]
    val_pairs = datasets["embedding_val_pairs"]
    eval_records = datasets["embedding_eval_retrieval"]
    assert train_pairs and val_pairs and eval_records

    train_queries = {_normalized_query(record["query"]) for record in train_pairs}
    val_queries = {_normalized_query(record["query"]) for record in val_pairs}
    eval_queries = {_normalized_query(record["query"]) for record in eval_records}
    assert train_queries.isdisjoint(val_queries)
    assert train_queries.isdisjoint(eval_queries)
    assert val_queries.isdisjoint(eval_queries)

    train_documents = {record["documentID"] for record in train_pairs}
    train_documents.update(
        record["negativeDocumentID"]
        for record in datasets["embedding_hard_negatives"]
        if record["split"] == "train"
    )
    val_documents = {record["documentID"] for record in val_pairs}
    val_documents.update(
        record["negativeDocumentID"]
        for record in datasets["embedding_hard_negatives"]
        if record["split"] == "validation"
    )
    eval_documents = {
        document_id
        for record in eval_records
        for document_id in record["positiveDocumentIDs"]
    }
    eval_documents.update(
        document_id
        for record in eval_records
        for document_id in record["hardNegativeDocumentIDs"]
    )
    assert train_documents.isdisjoint(val_documents)
    assert train_documents.isdisjoint(eval_documents)
    assert val_documents.isdisjoint(eval_documents)

    train_val_pairs = {
        (_normalized_query(record["query"]), record["documentID"])
        for record in train_pairs + val_pairs
    }
    eval_pairs = {
        (_normalized_query(record["query"]), document_id)
        for record in eval_records
        for document_id in record["positiveDocumentIDs"]
    }
    assert train_val_pairs.isdisjoint(eval_pairs)


def test_eval_scenario_documents_and_connected_queries_are_evaluation_only() -> None:
    manifest = generate_manifest(_repo_root())
    datasets = generate_all_datasets(manifest)

    eval_scenario_document_ids = {
        record["id"]
        for record in datasets["embedding_corpus"]
        if record["objectType"] == "eval_scenario"
    }
    assert eval_scenario_document_ids

    for record in datasets["embedding_train_pairs"] + datasets["embedding_val_pairs"]:
        assert record["documentID"] not in eval_scenario_document_ids
        assert record.get("metadata", {}).get("sourceFamily") != "eval_scenarios"

    for record in datasets["embedding_train_triplets"] + datasets["embedding_val_triplets"]:
        assert record["positiveDocumentID"] not in eval_scenario_document_ids
        assert record["negativeDocumentID"] not in eval_scenario_document_ids

    for record in datasets["embedding_hard_negatives"]:
        assert record["positiveDocumentID"] not in eval_scenario_document_ids
        assert record["negativeDocumentID"] not in eval_scenario_document_ids

    eval_scenario_retrieval = [
        record
        for record in datasets["embedding_eval_retrieval"]
        if record["family"] == "eval_scenarios_retrieval"
    ]
    assert eval_scenario_retrieval
    assert all(
        set(record["positiveDocumentIDs"]) & eval_scenario_document_ids
        for record in eval_scenario_retrieval
    )

    eval_queries = {
        _normalized_query(record["query"])
        for record in eval_scenario_retrieval
    }
    train_validation_queries = {
        _normalized_query(record["query"])
        for record in datasets["embedding_train_pairs"] + datasets["embedding_val_pairs"]
    }
    assert eval_queries.isdisjoint(train_validation_queries)

    card = datasets["embedding_dataset_card"][0]
    assert card["evaluationIsolation"] == {
        "sourceFamilies": ["eval_scenarios"],
        "policy": "evaluation_only_with_connected_query_groups",
        "contentPolicy": "normalized_eval_user_segment_documents_are_evaluation_only",
    }


def test_eval_scenario_query_collision_reserves_the_entire_connected_group() -> None:
    manifest = generate_manifest(_repo_root())
    shared_query = "Find the deliberately reserved collision record."
    compiled = compile_embedding_datasets(
        manifest,
        {
            "tool_schema_cards": [
                {
                    "id": "ordinary-source-record",
                    "query": shared_query,
                    "summary": "An otherwise trainable source record.",
                }
            ],
            "eval_scenarios": [
                {
                    "id": "reserved-evaluation-record",
                    "messages": [{"role": "user", "content": shared_query}],
                }
            ],
        },
    )
    datasets = compiled.as_dataset_families()

    training_queries = {
        _normalized_query(record["query"])
        for record in datasets["embedding_train_pairs"] + datasets["embedding_val_pairs"]
    }
    assert _normalized_query(shared_query) not in training_queries

    matching_eval = next(
        record
        for record in datasets["embedding_eval_retrieval"]
        if _normalized_query(record["query"]) == _normalized_query(shared_query)
    )
    positive_documents = set(matching_eval["positiveDocumentIDs"])
    source_families = {
        record["metadata"].get("sourceFamily")
        for record in datasets["embedding_corpus"]
        if record["id"] in positive_documents
    }
    assert source_families == {"tool_schema_cards", "eval_scenarios"}


def test_eval_prompt_inside_source_document_is_never_used_for_training() -> None:
    manifest = generate_manifest(_repo_root())
    heldout_prompt = "Email Jordan Patel directly."
    compiled = compile_embedding_datasets(
        manifest,
        {
            "eval_scenarios": [
                {
                    "id": "reserved-evaluation-record",
                    "messages": [{"role": "user", "content": heldout_prompt}],
                }
            ],
            "codebase_home_chunks": [
                {
                    "id": "source-containing-heldout-text",
                    "path": "tests/eval_definitions.py",
                    "module": "tests",
                    "language": "python",
                    "sha256": "source-sha",
                    "chunkSHA256": "chunk-sha",
                    "lineStart": 1,
                    "lineEnd": 2,
                    "text": f'PROMPT = "{heldout_prompt}"',
                }
            ],
        },
    )
    datasets = compiled.as_dataset_families()
    source_document = next(
        record
        for record in datasets["embedding_corpus"]
        if record["objectID"] == "source-containing-heldout-text"
    )
    assert source_document["metadata"]["evaluationOnly"] is True

    training_document_ids = {
        record["documentID"]
        for record in datasets["embedding_train_pairs"] + datasets["embedding_val_pairs"]
    }
    training_document_ids.update(
        record["negativeDocumentID"]
        for record in datasets["embedding_hard_negatives"]
    )
    assert source_document["id"] not in training_document_ids

    evaluation_document_ids = {
        document_id
        for record in datasets["embedding_eval_retrieval"]
        for document_id in record["positiveDocumentIDs"]
    }
    assert source_document["id"] in evaluation_document_ids


def test_embedding_never_labels_a_known_query_positive_as_a_hard_negative() -> None:
    manifest = generate_manifest(_repo_root())
    compiled = compile_embedding_datasets(
        manifest,
        {
            "tool_schema_cards": [
                {"id": "positive-a", "query": "Find this shared contract", "summary": "First valid answer"},
                {"id": "positive-b", "query": "  FIND this shared contract  ", "summary": "Second valid answer"},
            ]
        },
    )
    datasets = compiled.as_dataset_families()
    known_positives = _known_positive_documents(datasets)
    shared_query = _normalized_query("Find this shared contract")
    assert len(known_positives[shared_query]) == 2

    for record in datasets["embedding_hard_negatives"]:
        query = _normalized_query(record["query"])
        assert record["negativeDocumentID"] not in known_positives[query]
    for record in datasets["embedding_eval_retrieval"]:
        query = _normalized_query(record["query"])
        assert set(record["hardNegativeDocumentIDs"]).isdisjoint(known_positives[query])
