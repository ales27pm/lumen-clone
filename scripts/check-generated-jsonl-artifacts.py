#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


GENERATED_ROOTS = (
    Path("generated"),
    Path("tools/lumen_manifest_crawler/generated"),
)

CANONICAL_FINE_TUNING_ROOT = Path("generated/fine_tuning")
STALE_NESTED_FINE_TUNING_ROOT = Path("generated/agent_manifest/fine_tuning")

CROSS_MODEL_PRIMARY_ROOT = Path("generated/cross_model_training")
CROSS_MODEL_NESTED_ROOT = Path(
    "generated/agent_manifest/cross_model_training"
)
CROSS_MODEL_FILENAMES = (
    "cross_model_training.jsonl",
    "cross_model_training_index.csv",
    "dpo_train_cross.jsonl",
    "dpo_val_cross.jsonl",
    "orchestration_evals.jsonl",
    "train_sft_cross.jsonl",
    "val_sft_cross.jsonl",
)

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


def _cross_model_lineage_errors(
    directory: Path,
    manifest_path: Path,
) -> list[str]:
    errors: list[str] = []
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"cannot read current manifest lineage from {manifest_path}: {exc}"]
    source_integrity = manifest.get("sourceIntegrity")
    if not isinstance(source_integrity, dict):
        return [f"current manifest has no sourceIntegrity object: {manifest_path}"]
    expected_integrity = {
        key: source_integrity.get(key)
        for key in ("baseCommit", "workingTreeDigest", "dirtyState")
    }
    if (
        not isinstance(expected_integrity["baseCommit"], str)
        or not isinstance(expected_integrity["workingTreeDigest"], str)
        or not isinstance(expected_integrity["dirtyState"], bool)
    ):
        return [f"current manifest sourceIntegrity is malformed: {manifest_path}"]

    orchestration_rows = 0
    for filename in CROSS_MODEL_FILENAMES:
        if not filename.endswith(".jsonl"):
            continue
        path = directory / filename
        if not path.is_file():
            continue
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue
            metadata = row.get("metadata")
            if filename == "orchestration_evals.jsonl":
                orchestration_rows += 1
                if not isinstance(metadata, dict):
                    errors.append(
                        f"{path}:{line_number}: orchestration row has no metadata"
                    )
                    continue
            if not isinstance(metadata, dict):
                continue
            manifest_commit = metadata.get("manifestCommit")
            row_integrity = metadata.get("sourceIntegrity")
            has_lineage = manifest_commit is not None or row_integrity is not None
            if not has_lineage:
                if filename == "orchestration_evals.jsonl":
                    errors.append(
                        f"{path}:{line_number}: orchestration row has no lineage"
                    )
                continue
            if (
                manifest_commit != expected_integrity["baseCommit"]
                or row_integrity != expected_integrity
            ):
                errors.append(
                    f"{path}:{line_number}: row lineage does not match "
                    f"{manifest_path}"
                )
    if orchestration_rows == 0:
        errors.append(
            f"{directory / 'orchestration_evals.jsonl'} has no lineage-bound rows"
        )
    return errors


def main() -> int:
    zero_byte_files: list[Path] = []
    invalid_jsonl_rows: list[str] = []
    alias_errors: list[str] = []
    cross_model_mirror_errors: list[str] = []
    stale_nested_fine_tuning_files: list[Path] = []
    for root in GENERATED_ROOTS:
        if not root.exists():
            continue
        zero_byte_files.extend(
            path
            for path in root.rglob("*.jsonl")
            if path.is_file() and path.stat().st_size == 0
        )
        for path in root.rglob("*.jsonl"):
            if not path.is_file() or path.stat().st_size == 0:
                continue
            with path.open(encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        invalid_jsonl_rows.append(
                            f"{path}:{line_number}: blank JSONL row"
                        )
                        continue
                    try:
                        value = json.loads(line)
                    except json.JSONDecodeError as exc:
                        invalid_jsonl_rows.append(
                            f"{path}:{line_number}: {exc.msg}"
                        )
                        continue
                    if not isinstance(value, dict):
                        invalid_jsonl_rows.append(
                            f"{path}:{line_number}: row must be a JSON object"
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
    primary_exists = CROSS_MODEL_PRIMARY_ROOT.is_dir()
    nested_exists = CROSS_MODEL_NESTED_ROOT.is_dir()
    if not primary_exists or not nested_exists:
        cross_model_mirror_errors.append(
            "both required mirror directories must exist: "
            f"{CROSS_MODEL_PRIMARY_ROOT} and {CROSS_MODEL_NESTED_ROOT}"
        )
    else:
        expected = set(CROSS_MODEL_FILENAMES)
        for directory in (
            CROSS_MODEL_PRIMARY_ROOT,
            CROSS_MODEL_NESTED_ROOT,
        ):
            entries = list(directory.iterdir())
            names = {entry.name for entry in entries}
            if names != expected:
                cross_model_mirror_errors.append(
                    f"unexpected mirror entries in {directory}: "
                    f"expected {sorted(expected)}, got {sorted(names)}"
                )
            unsafe = sorted(
                entry.name
                for entry in entries
                if entry.is_symlink() or not entry.is_file()
            )
            if unsafe:
                cross_model_mirror_errors.append(
                    f"mirror entries must be regular files in {directory}: "
                    f"{unsafe}"
                )
        for filename in CROSS_MODEL_FILENAMES:
            primary = CROSS_MODEL_PRIMARY_ROOT / filename
            nested = CROSS_MODEL_NESTED_ROOT / filename
            if not primary.is_file() or not nested.is_file():
                cross_model_mirror_errors.append(
                    f"missing mirrored artifact: {primary} or {nested}"
                )
                continue
            if primary.read_bytes() != nested.read_bytes():
                cross_model_mirror_errors.append(
                    f"mirrored artifacts differ: {primary} != {nested}"
                )
        cross_model_mirror_errors.extend(
            _cross_model_lineage_errors(
                CROSS_MODEL_PRIMARY_ROOT,
                Path("generated/agent_manifest/AgentBehaviorManifest.json"),
            )
        )

    if zero_byte_files:
        print("error: zero-byte generated JSONL artifacts found:")
        for path in sorted(zero_byte_files):
            print(f"  {path}")
        print("Regenerate the owning artifact or delete the empty placeholder.")
        return 1
    if invalid_jsonl_rows:
        print("error: invalid generated JSONL rows found:")
        for error in invalid_jsonl_rows[:100]:
            print(f"  {error}")
        if len(invalid_jsonl_rows) > 100:
            print(f"  ... {len(invalid_jsonl_rows) - 100} additional errors")
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
    if cross_model_mirror_errors:
        print("error: cross-model artifact mirrors are inconsistent:")
        for error in cross_model_mirror_errors:
            print(f"  {error}")
        print(
            "Regenerate with --cross-model-train-dir so the canonical nested "
            "copy and external mirror are refreshed together."
        )
        return 1

    print("generated JSONL artifact validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
