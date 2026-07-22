from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

RERANKER_MODEL_ID = "Qwen/Qwen3-Reranker-0.6B"
RERANKER_TEACHER_MODEL_ID = "Qwen/Qwen3-Reranker-4B"
RERANKER_DATASET_SCHEMA_VERSION = "1.0.0"


@dataclass(frozen=True)
class RerankerDatasets:
    train_pairs: list[dict[str, Any]]
    val_pairs: list[dict[str, Any]]
    hard_negative_pairs: list[dict[str, Any]]
    eval_reranking: list[dict[str, Any]]
    dataset_card: dict[str, Any]

    def as_dataset_families(self) -> dict[str, list[dict[str, Any]]]:
        return {
            "reranker_train_pairs": self.train_pairs,
            "reranker_val_pairs": self.val_pairs,
            "reranker_hard_negative_pairs": self.hard_negative_pairs,
            "reranker_eval_reranking": self.eval_reranking,
            "reranker_dataset_card": [self.dataset_card],
        }


def compile_reranker_datasets(datasets: dict[str, list[dict[str, Any]]]) -> RerankerDatasets:
    """Build pairwise reranker artifacts from embedding corpus and negatives.

    The reranker dataset is intentionally not chat/SFT data. It trains candidate
    ordering from query, positive document, and hard-negative document triples.
    """
    corpus = [record for record in datasets.get("embedding_corpus", []) if isinstance(record, dict)]
    docs = {str(record.get("id")): record for record in corpus if record.get("id")}
    training_docs = {
        document_id: record
        for document_id, record in docs.items()
        if not _is_evaluation_only(record)
    }
    hard_negatives = [
        record for record in datasets.get("embedding_hard_negatives", [])
        if isinstance(record, dict)
    ]
    hard_negative_index = _index_hard_negatives(hard_negatives)
    known_positive_ids = _known_positive_documents(datasets)

    train_pairs = _compile_pair_split(
        datasets.get("embedding_train_pairs", []),
        training_docs,
        hard_negative_index,
        known_positive_ids,
    )
    val_pairs = _compile_pair_split(
        datasets.get("embedding_val_pairs", []),
        training_docs,
        hard_negative_index,
        known_positive_ids,
    )
    hard_negative_pairs = _compile_hard_negative_pairs(
        hard_negatives,
        training_docs,
        known_positive_ids,
    )
    eval_reranking = _compile_eval_records(
        datasets.get("embedding_eval_retrieval", []),
        docs,
        known_positive_ids,
    )

    dataset_card = {
        "schemaVersion": RERANKER_DATASET_SCHEMA_VERSION,
        "model": RERANKER_MODEL_ID,
        "teacherModel": RERANKER_TEACHER_MODEL_ID,
        "task": "candidate_reranking",
        "sourceFamilies": [
            "embedding_corpus",
            "embedding_train_pairs",
            "embedding_val_pairs",
            "embedding_hard_negatives",
            "embedding_eval_retrieval",
        ],
        "nonGoals": [
            "Do not train the reranker on chat SFT records.",
            "Do not use evaluation-only corpus documents as positive or negative training candidates.",
            "Do not use generated private runtime payloads as document text.",
            "Do not promote a reranker without hard-negative accuracy evidence.",
        ],
        "evaluationIsolation": {
            "policy": "exclude_evaluation_only_documents_from_all_training_candidates",
        },
        "counts": {
            "trainPairs": len(train_pairs),
            "valPairs": len(val_pairs),
            "hardNegativePairs": len(hard_negative_pairs),
            "evalReranking": len(eval_reranking),
        },
        "promotionMetrics": {
            "rerankedRecallAt1Minimum": 0.78,
            "rerankedNdcgAt5Minimum": 0.84,
            "hardNegativePairAccuracyMinimum": 0.88,
            "toolSchemaRerankAccuracyMinimum": 0.92,
            "runtimeRepairRerankAccuracyMinimum": 0.82,
            "rerankerHealthCheckPassRate": 1.0,
        },
        "families": sorted(
            {
                str(record.get("family"))
                for record in train_pairs + val_pairs + hard_negative_pairs
                if record.get("family")
            }
        ),
    }

    return RerankerDatasets(
        train_pairs=train_pairs,
        val_pairs=val_pairs,
        hard_negative_pairs=hard_negative_pairs,
        eval_reranking=eval_reranking,
        dataset_card=dataset_card,
    )


def _compile_pair_split(
    pairs: list[dict[str, Any]],
    docs: dict[str, dict[str, Any]],
    hard_negative_index: dict[tuple[str, str], list[dict[str, Any]]],
    known_positive_ids: dict[str, set[str]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for pair in pairs:
        if not isinstance(pair, dict):
            continue
        positive_id = str(pair.get("documentID") or "")
        positive = docs.get(positive_id)
        if positive is None:
            continue
        query = _clean(str(pair.get("query") or ""))
        if not query:
            continue
        negative_id = _negative_for_pair(
            pair,
            docs,
            hard_negative_index,
            known_positive_ids,
        )
        negative = docs.get(negative_id) if negative_id else None
        if negative is None:
            continue
        records.append(_pair_record(query, positive, negative, pair))
    return records


def _compile_hard_negative_pairs(
    hard_negatives: list[dict[str, Any]],
    docs: dict[str, dict[str, Any]],
    known_positive_ids: dict[str, set[str]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for negative in hard_negatives:
        query = _clean(str(negative.get("query") or ""))
        positive = docs.get(str(negative.get("positiveDocumentID") or ""))
        negative_id = str(negative.get("negativeDocumentID") or "")
        negative_doc = docs.get(negative_id)
        if not query or positive is None or negative_doc is None:
            continue
        if negative_id in known_positive_ids.get(_normalize_query(query), set()):
            continue
        records.append(_pair_record(
            query,
            positive,
            negative_doc,
            negative,
            record_kind="hard_negative_pair",
            reason=str(negative.get("reason") or ""),
        ))
    return records


def _compile_eval_records(
    evals: list[dict[str, Any]],
    docs: dict[str, dict[str, Any]],
    known_positive_ids: dict[str, set[str]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for item in evals:
        if not isinstance(item, dict):
            continue
        query = _clean(str(item.get("query") or ""))
        positive_ids = [str(value) for value in item.get("positiveDocumentIDs", []) if str(value) in docs]
        query_positives = known_positive_ids.get(_normalize_query(query), set())
        negative_ids = [
            str(value)
            for value in item.get("hardNegativeDocumentIDs", [])
            if str(value) in docs and str(value) not in query_positives
        ]
        if not query or not positive_ids or not negative_ids:
            continue
        candidates = [
            _candidate(docs[doc_id], 1.0)
            for doc_id in positive_ids
        ] + [
            _candidate(docs[doc_id], 0.0)
            for doc_id in negative_ids
        ]
        records.append({
            "id": _stable_id("reranker_eval", item.get("id"), query),
            "schemaVersion": RERANKER_DATASET_SCHEMA_VERSION,
            "task": "candidate_reranking_eval",
            "query": query,
            "positiveDocumentIDs": positive_ids,
            "candidateDocuments": candidates,
            "family": item.get("family"),
            "split": "evaluation",
            "groupID": item.get("groupID"),
            "metrics": ["reranked_recall@1", "reranked_ndcg@5", "hard_negative_pair_accuracy"],
            "targets": {
                "rerankedRecallAt1": 0.78,
                "rerankedNdcgAt5": 0.84,
                "hardNegativePairAccuracy": 0.88,
            },
            "metadata": item.get("metadata") or {},
        })
    return records


def _pair_record(
    query: str,
    positive: dict[str, Any],
    negative: dict[str, Any],
    source: dict[str, Any],
    *,
    record_kind: str = "pairwise_reranking",
    reason: str = "",
) -> dict[str, Any]:
    split = source.get("split")
    record = {
        "id": _stable_id("reranker_pair", record_kind, query, positive.get("id"), negative.get("id")),
        "schemaVersion": RERANKER_DATASET_SCHEMA_VERSION,
        "task": record_kind,
        "query": query,
        "positiveDocumentID": positive.get("id"),
        "positiveText": _document_text(positive),
        "negativeDocumentID": negative.get("id"),
        "negativeText": _document_text(negative),
        "documents": [
            _candidate(positive, 1.0),
            _candidate(negative, 0.0),
        ],
        "label": 1.0,
        "family": source.get("family"),
        "metadata": {
            **(source.get("metadata") or {}),
            "positiveObjectType": positive.get("objectType"),
            "negativeObjectType": negative.get("objectType"),
            "hardNegativeReason": reason or source.get("reason") or "",
        },
    }
    if split is not None:
        record["split"] = split
    return record


def _candidate(document: dict[str, Any], relevance: float) -> dict[str, Any]:
    return {
        "documentID": document.get("id"),
        "title": document.get("title"),
        "text": _document_text(document),
        "objectType": document.get("objectType"),
        "relevance": relevance,
    }


def _document_text(document: dict[str, Any]) -> str:
    title = _clean(str(document.get("title") or ""))
    text = _clean(str(document.get("text") or ""))
    if title and text and title not in text:
        return f"{title}\n{text}"
    return text or title


def _index_hard_negatives(records: list[dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    index: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for record in sorted(records, key=lambda item: str(item.get("id") or "")):
        key = (_normalize_query(str(record.get("query") or "")), str(record.get("positiveDocumentID") or ""))
        index.setdefault(key, []).append(record)
    return index


def _known_positive_documents(datasets: dict[str, list[dict[str, Any]]]) -> dict[str, set[str]]:
    known: dict[str, set[str]] = {}
    for family in ("embedding_train_pairs", "embedding_val_pairs"):
        for record in datasets.get(family, []):
            if not isinstance(record, dict):
                continue
            query = _normalize_query(str(record.get("query") or ""))
            document_id = str(record.get("documentID") or "")
            if query and document_id:
                known.setdefault(query, set()).add(document_id)
    for record in datasets.get("embedding_eval_retrieval", []):
        if not isinstance(record, dict):
            continue
        query = _normalize_query(str(record.get("query") or ""))
        if not query:
            continue
        positive_ids = {
            str(document_id)
            for document_id in record.get("positiveDocumentIDs", [])
            if str(document_id)
        }
        known.setdefault(query, set()).update(positive_ids)
    return known


def _negative_for_pair(
    pair: dict[str, Any],
    docs: dict[str, dict[str, Any]],
    hard_negative_index: dict[tuple[str, str], list[dict[str, Any]]],
    known_positive_ids: dict[str, set[str]],
) -> str | None:
    query = _clean(str(pair.get("query") or ""))
    normalized_query = _normalize_query(query)
    positive_id = str(pair.get("documentID") or "")
    query_positives = known_positive_ids.get(normalized_query, set())
    hard_candidates = hard_negative_index.get((normalized_query, positive_id), [])
    for candidate in hard_candidates:
        negative_id = str(candidate.get("negativeDocumentID") or "")
        if negative_id in docs and negative_id != positive_id and negative_id not in query_positives:
            return negative_id

    positive = docs.get(positive_id)
    if positive is None:
        return None
    positive_type = str(positive.get("objectType") or "")
    candidates = [
        doc for doc in docs.values()
        if doc.get("id") != positive_id
        and str(doc.get("id") or "") not in query_positives
        and not _is_evaluation_only(doc)
        and doc.get("objectType") == positive_type
    ] or [
        doc for doc in docs.values()
        if doc.get("id") != positive_id
        and str(doc.get("id") or "") not in query_positives
        and not _is_evaluation_only(doc)
    ]
    if not candidates:
        return None
    return str(sorted(candidates, key=lambda doc: str(doc.get("id") or ""))[0].get("id"))


def _is_evaluation_only(document: dict[str, Any]) -> bool:
    metadata = document.get("metadata")
    return isinstance(metadata, dict) and metadata.get("evaluationOnly") is True


def _clean(value: str) -> str:
    return " ".join(str(value or "").strip().split())


def _normalize_query(value: str) -> str:
    return _clean(value).casefold()


def _stable_id(*parts: Any) -> str:
    payload = json.dumps(parts, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]
