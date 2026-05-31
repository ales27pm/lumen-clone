from __future__ import annotations

import json
from pathlib import Path

from tools.augment_loop_state_embedding import augment


def _write_jsonl(path: Path, count: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for index in range(count):
            handle.write(json.dumps({"id": index}) + "\n")


def test_augment_adds_embedding_summary_to_loop_and_visual_state(tmp_path: Path) -> None:
    loop_state = tmp_path / "generated" / "agent_improvement_loop" / "loop_state.json"
    visual_state = tmp_path / "generated" / "visual_improve_loop" / "visual_improve_loop_summary.json"
    embedding_dir = tmp_path / "generated" / "agent_manifest" / "embedding"

    loop_state.parent.mkdir(parents=True, exist_ok=True)
    visual_state.parent.mkdir(parents=True, exist_ok=True)
    loop_state.write_text(json.dumps({"schemaVersion": "1.1.0", "dataset": {"families": {}}}), encoding="utf-8")
    visual_state.write_text(json.dumps({"schemaVersion": "2.0.0", "dataset": {"recordCount": 12}}), encoding="utf-8")
    (embedding_dir).mkdir(parents=True, exist_ok=True)
    (embedding_dir / "dataset_card.json").write_text(
        json.dumps({"model": "Qwen/Qwen3-Embedding-0.6B", "teacherModel": "Qwen/Qwen3-Embedding-4B", "task": "retrieval_similarity_ranking"}),
        encoding="utf-8",
    )
    _write_jsonl(embedding_dir / "corpus.jsonl", 3)
    _write_jsonl(embedding_dir / "train_pairs.jsonl", 4)
    _write_jsonl(embedding_dir / "val_pairs.jsonl", 1)
    _write_jsonl(embedding_dir / "train_triplets.jsonl", 2)
    _write_jsonl(embedding_dir / "val_triplets.jsonl", 1)
    _write_jsonl(embedding_dir / "hard_negatives.jsonl", 5)
    _write_jsonl(embedding_dir / "eval_retrieval.jsonl", 6)

    state = augment(loop_state, embedding_dir, visual_state)
    visual = json.loads(visual_state.read_text(encoding="utf-8"))

    assert state["embedding"]["corpusCount"] == 3
    assert state["embedding"]["pairCount"] == 5
    assert state["embedding"]["tripletCount"] == 3
    assert state["embedding"]["hardNegativeCount"] == 5
    assert state["embedding"]["evalCount"] == 6
    assert state["embedding"]["generated"] is True
    assert state["dataset"]["embedding"] == state["embedding"]
    assert visual["embedding"] == state["embedding"]
    assert visual["dataset"]["embedding"] == state["embedding"]
