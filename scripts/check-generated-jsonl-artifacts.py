#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path


GENERATED_ROOTS = (
    Path("generated"),
    Path("tools/lumen_manifest_crawler/generated"),
)

CANONICAL_FINE_TUNING_ROOT = Path("generated/fine_tuning")
STALE_NESTED_FINE_TUNING_ROOT = Path("generated/agent_manifest/fine_tuning")

CANONICAL_DATASET_ALIASES = {
    "dataset/embedding_corpus.jsonl": "../embedding/corpus.jsonl",
    "dataset/embedding_train_pairs.jsonl": "../embedding/train_pairs.jsonl",
    "dataset/embedding_val_pairs.jsonl": "../embedding/val_pairs.jsonl",
    "dataset/embedding_train_triplets.jsonl": "../embedding/train_triplets.jsonl",
    "dataset/embedding_val_triplets.jsonl": "../embedding/val_triplets.jsonl",
    "dataset/embedding_hard_negatives.jsonl": "../embedding/hard_negatives.jsonl",
    "dataset/embedding_eval_retrieval.jsonl": "../embedding/eval_retrieval.jsonl",
    "dataset/reranker_train_pairs.jsonl": "../reranker/train_pairs.jsonl",
    "dataset/reranker_val_pairs.jsonl": "../reranker/val_pairs.jsonl",
    "dataset/reranker_hard_negative_pairs.jsonl": "../reranker/hard_negative_pairs.jsonl",
    "dataset/reranker_eval_reranking.jsonl": "../reranker/eval_reranking.jsonl",
}


def main() -> int:
    zero_byte_files: list[Path] = []
    alias_errors: list[str] = []
    stale_nested_fine_tuning_files: list[Path] = []
    for root in GENERATED_ROOTS:
        if not root.exists():
            continue
        zero_byte_files.extend(
            path
            for path in root.rglob("*.jsonl")
            if path.is_file() and path.stat().st_size == 0
        )
        agent_manifest = root / "agent_manifest" if root.name == "generated" else root
        if not agent_manifest.exists():
            continue
        for alias_relative, target_relative in CANONICAL_DATASET_ALIASES.items():
            alias = agent_manifest / alias_relative
            if not alias.exists() and not alias.is_symlink():
                continue
            if not alias.is_symlink():
                alias_errors.append(f"{alias} should be a symlink to {target_relative}")
                continue
            actual_target = alias.readlink()
            if actual_target.as_posix() != target_relative:
                alias_errors.append(f"{alias} points to {actual_target}, expected {target_relative}")
                continue
            if not alias.resolve().is_file():
                alias_errors.append(f"{alias} points to missing canonical artifact {target_relative}")
    if CANONICAL_FINE_TUNING_ROOT.exists() and STALE_NESTED_FINE_TUNING_ROOT.exists():
        stale_nested_fine_tuning_files.extend(
            path
            for path in STALE_NESTED_FINE_TUNING_ROOT.rglob("*")
            if path.is_file()
        )

    if zero_byte_files:
        print("error: zero-byte generated JSONL artifacts found:")
        for path in sorted(zero_byte_files):
            print(f"  {path}")
        print("Regenerate the owning artifact or delete the empty placeholder.")
        return 1
    if alias_errors:
        print("error: generated dataset alias validation failed:")
        for error in alias_errors:
            print(f"  {error}")
        print("Regenerate artifacts with the manifest crawler so aliases do not duplicate large JSONL payloads.")
        return 1
    if stale_nested_fine_tuning_files:
        print("error: stale nested fine-tuning artifacts found:")
        for path in sorted(stale_nested_fine_tuning_files):
            print(f"  {path}")
        print("Use generated/fine_tuning as the checked-in fine-tuning artifact root.")
        return 1

    print("generated JSONL artifact validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
