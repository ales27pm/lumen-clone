"""Dataset generation entrypoints for role records and compiled artifacts."""
# pylint: disable=missing-module-docstring,missing-function-docstring,line-too-long

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from lumen_manifest_crawler.dataset.codebase_home import generate_codebase_home_records
from lumen_manifest_crawler.dataset.cortex import generate_cortex_records
from lumen_manifest_crawler.dataset.compiler import (
    DatasetCompilerConfig,
    _records_hash,
    compile_state_of_art_datasets,
    finalize_dataset_manifest,
)
from lumen_manifest_crawler.dataset.embedding import compile_embedding_datasets
from lumen_manifest_crawler.dataset.executor import (
    generate_approval_boundary_records,
    generate_executor_records,
    generate_negative_samples,
)
from lumen_manifest_crawler.dataset.mimicry import generate_mimicry_records
from lumen_manifest_crawler.dataset.mouth import generate_mouth_records
from lumen_manifest_crawler.dataset.public_adapter_corpus import (
    load_public_adapter_corpus,
    lumen_contract_from_manifest,
)
from lumen_manifest_crawler.dataset.rem import generate_rem_records
from lumen_manifest_crawler.dataset.reranker import compile_reranker_datasets
from lumen_manifest_crawler.dataset.runtime_ingest import load_runtime_audit_reports
from lumen_manifest_crawler.manifest import AgentBehaviorManifest


PUBLIC_ADAPTER_CORPUS_RELATIVE_PATH = Path("datasets/public_adapter_corpus")


def generate_role_datasets(manifest: AgentBehaviorManifest) -> dict[str, list[dict[str, Any]]]:
    """Generate the primary per-role dataset families from the manifest."""
    return {
        "cortex_routing": generate_cortex_records(manifest),
        "executor_tool_calls": generate_executor_records(manifest),
        "mouth_responses": generate_mouth_records(manifest),
        "mimicry_style": generate_mimicry_records(manifest),
        "rem_reflection": generate_rem_records(manifest),
        "negative_samples": generate_negative_samples(manifest),
        "approval_boundary_samples": generate_approval_boundary_records(manifest),
    }


def generate_all_datasets(
    manifest: AgentBehaviorManifest,
    *,
    root: Path | None = None,
    runtime_audit_paths: list[Path] | None = None,
    runtime_audit_reports: list[dict[str, Any]] | None = None,
    deterministic: bool = True,
) -> dict[str, list[dict[str, Any]]]:
    """Compile role datasets plus normalized runtime-audit-derived datasets."""
    role_records = generate_role_datasets(manifest)
    runtime_audit_reports = (
        runtime_audit_reports
        if runtime_audit_reports is not None
        else load_runtime_audit_reports(runtime_audit_paths)
    )
    compiled = compile_state_of_art_datasets(
        manifest,
        role_records,
        runtime_audit_reports=runtime_audit_reports,
        config=DatasetCompilerConfig(deterministic=deterministic),
    )
    codebase_home_records = generate_codebase_home_records(root) if root is not None else {}
    _isolate_evaluation_source_records(
        codebase_home_records,
        compiled.records.get("eval_scenarios", []),
    )
    public_records, public_snapshot = _load_public_adapter_corpus_families(root, manifest)
    _augment_dataset_manifest_with_public_corpus(compiled.manifest, public_records, public_snapshot)
    datasets: dict[str, list[dict[str, Any]]] = {
        **role_records,
        **compiled.records,
        **codebase_home_records,
        **public_records,
    }
    embedding = compile_embedding_datasets(manifest, datasets)
    embedding_families = embedding.as_dataset_families()
    reranker = compile_reranker_datasets({
        **datasets,
        **embedding_families,
    })
    complete_datasets = {
        **datasets,
        **embedding_families,
        **reranker.as_dataset_families(),
    }
    complete_manifest = finalize_dataset_manifest(compiled.manifest, complete_datasets)
    return {**complete_datasets, "dataset_manifest": [complete_manifest]}


def _isolate_evaluation_source_records(
    codebase_home_records: dict[str, list[dict[str, Any]]],
    evaluation_records: list[dict[str, Any]],
) -> None:
    """Keep tracked eval definitions available as corpus, never training text."""

    heldout_segments = _normalized_evaluation_user_segments(evaluation_records)
    if not heldout_segments or not codebase_home_records:
        return

    sensitive_chunk_count = 0
    for record in codebase_home_records.get("codebase_home_chunks", []):
        matches = _contained_evaluation_segments(
            str(record.get("text") or ""),
            heldout_segments,
        )
        if not matches:
            continue
        sensitive_chunk_count += 1
        metadata = dict(record.get("metadata") or {})
        metadata.update(
            {
                "evaluationOnly": True,
                "evaluationIsolationReason": "contains_normalized_eval_user_segment",
                "evaluationSegmentSHA256": [
                    hashlib.sha256(segment.encode("utf-8")).hexdigest()
                    for segment in matches
                ],
            }
        )
        record["metadata"] = metadata

    excluded_sft_count = 0
    for family in ("codebase_home_sft", "codebase_home_chunk_sft"):
        retained: list[dict[str, Any]] = []
        for record in codebase_home_records.get(family, []):
            serialized = json.dumps(
                record,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            if _contained_evaluation_segments(serialized, heldout_segments):
                excluded_sft_count += 1
                continue
            retained.append(record)
        codebase_home_records[family] = retained

    overview = next(
        (
            record
            for record in codebase_home_records.get("codebase_home_corpus", [])
            if record.get("path") == "."
        ),
        None,
    )
    if overview is not None:
        metadata = dict(overview.get("metadata") or {})
        metadata.update(
            {
                "evaluationIsolationPolicy": "eval_user_segments_corpus_only",
                "evaluationSensitiveChunkCount": sensitive_chunk_count,
                "evaluationSensitiveSFTExcludedCount": excluded_sft_count,
            }
        )
        overview["metadata"] = metadata


def _normalized_evaluation_user_segments(
    records: list[dict[str, Any]],
) -> tuple[str, ...]:
    segments: set[str] = set()
    for record in records:
        messages = record.get("messages")
        if not isinstance(messages, list):
            continue
        for message in messages:
            if not isinstance(message, dict) or message.get("role") != "user":
                continue
            normalized = _normalize_evaluation_text(str(message.get("content") or ""))
            if normalized:
                segments.add(normalized)
    return tuple(sorted(segments))


def _contained_evaluation_segments(
    value: str,
    normalized_segments: tuple[str, ...],
) -> tuple[str, ...]:
    normalized_value = _normalize_evaluation_text(value)
    if not normalized_value:
        return ()
    padded_value = f" {normalized_value} "
    return tuple(
        segment
        for segment in normalized_segments
        if f" {segment} " in padded_value
    )


def _normalize_evaluation_text(value: str) -> str:
    return " ".join(re.findall(r"\w+", value.casefold(), flags=re.UNICODE))


def _load_public_adapter_corpus_families(
    root: Path | None,
    manifest: AgentBehaviorManifest,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any] | None]:
    if root is None:
        return {}, None
    snapshot_dir = root.resolve() / PUBLIC_ADAPTER_CORPUS_RELATIVE_PATH
    manifest_path = snapshot_dir / "manifest.json"
    records_path = snapshot_dir / "records.jsonl"
    if not manifest_path.exists() and not records_path.exists():
        return {}, None
    grouped = load_public_adapter_corpus(
        snapshot_dir,
        lumen_contract=lumen_contract_from_manifest(manifest),
    )
    snapshot = json.loads(manifest_path.read_text(encoding="utf-8"))
    families = {
        f"public_adapter_corpus_{agent}": records
        for agent, records in sorted(grouped.items())
        if records
    }
    return families, snapshot


def _augment_dataset_manifest_with_public_corpus(
    manifest: dict[str, Any],
    families: dict[str, list[dict[str, Any]]],
    snapshot: dict[str, Any] | None,
) -> None:
    if not families or snapshot is None:
        return
    counts = manifest.setdefault("counts", {})
    hashes = manifest.setdefault("hashes", {})
    for family, records in sorted(families.items()):
        counts[family] = len(records)
        hashes[family] = _records_hash(records)

    sources = manifest.setdefault("sources", {})
    raw_families = set(sources.get("rawDatasetFamilies") or [])
    raw_families.update(families)
    sources["rawDatasetFamilies"] = sorted(raw_families)
    sources["publicAdapterCorpus"] = {
        "schema": snapshot.get("schema"),
        "selectionPolicyVersion": snapshot.get("selectionPolicyVersion"),
        "recordCount": snapshot.get("recordCount"),
        "recordsSHA256": snapshot.get("recordsSHA256"),
        "countsByAgent": snapshot.get("countsByAgent"),
        "countsBySource": snapshot.get("countsBySource"),
        "countsByTaskType": snapshot.get("countsByTaskType"),
        "qualityScoreSummaryByAgent": snapshot.get("qualityScoreSummaryByAgent"),
        "lumenContractSHA256": snapshot.get("lumenContractSHA256"),
        "partitionPolicy": snapshot.get("partitionPolicy"),
        "sourceManifestSHA256": snapshot.get("sourceManifestSHA256"),
    }
    manifest.setdefault("trainingPolicy", {})["publicCorpusPolicy"] = (
        "pinned permissive-license sources only; ML sources restricted to explicit train partitions; "
        "non-ML reference corpora explicitly labeled as reference_corpus; "
        "opaque source groups, content hashes, PII screening, and adapter-exact routing required"
    )
