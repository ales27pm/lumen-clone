from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any

from lumen_manifest_crawler.manifest import AgentBehaviorManifest

EMBEDDING_MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"
EMBEDDING_TEACHER_MODEL_ID = "Qwen/Qwen3-Embedding-4B"
EMBEDDING_DATASET_SCHEMA_VERSION = "1.0.0"


@dataclass(frozen=True)
class EmbeddingDatasets:
    corpus: list[dict[str, Any]]
    train_pairs: list[dict[str, Any]]
    val_pairs: list[dict[str, Any]]
    train_triplets: list[dict[str, Any]]
    val_triplets: list[dict[str, Any]]
    hard_negatives: list[dict[str, Any]]
    eval_retrieval: list[dict[str, Any]]
    dataset_card: dict[str, Any]

    def as_dataset_families(self) -> dict[str, list[dict[str, Any]]]:
        return {
            "embedding_corpus": self.corpus,
            "embedding_train_pairs": self.train_pairs,
            "embedding_val_pairs": self.val_pairs,
            "embedding_train_triplets": self.train_triplets,
            "embedding_val_triplets": self.val_triplets,
            "embedding_hard_negatives": self.hard_negatives,
            "embedding_eval_retrieval": self.eval_retrieval,
            "embedding_dataset_card": [self.dataset_card],
        }


def compile_embedding_datasets(
    manifest: AgentBehaviorManifest,
    datasets: dict[str, list[dict[str, Any]]],
) -> EmbeddingDatasets:
    """Build retrieval/ranking datasets for the dedicated embedding model.

    This is intentionally not SFT/chat data. It creates corpus records, positive
    query-document pairs, triplets, hard negatives, and retrieval evals from the
    manifest, source map, fleet roles, tool schemas, runtime repairs, and evals.
    """
    corpus: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []
    hard_negatives: list[dict[str, Any]] = []
    evaluation_user_segments = _evaluation_user_segments(
        datasets.get("eval_scenarios", [])
    )
    evaluation_only_document_ids: set[str] = set()

    def add_doc(object_type: str, object_id: str, title: str, text: str, metadata: dict[str, Any] | None = None) -> str:
        doc_id = _stable_id("doc", object_type, object_id)
        if not text.strip():
            text_value = title
        else:
            text_value = text.strip()
        doc_metadata = dict(metadata or {})
        evaluation_matches = _contained_evaluation_segments(
            f"{title}\n{text_value}",
            evaluation_user_segments,
        )
        if evaluation_matches:
            evaluation_only_document_ids.add(doc_id)
            doc_metadata.update(
                {
                    "evaluationOnly": True,
                    "evaluationIsolationReason": "contains_normalized_eval_user_segment",
                    "evaluationSegmentSHA256": [
                        hashlib.sha256(segment.encode("utf-8")).hexdigest()
                        for segment in evaluation_matches
                    ],
                }
            )
        corpus.append({
            "id": doc_id,
            "objectType": object_type,
            "objectID": object_id,
            "title": title.strip() or object_id,
            "text": text_value,
            "metadata": doc_metadata,
        })
        return doc_id

    def add_pair(
        query: str,
        doc_id: str,
        family: str,
        metadata: dict[str, Any] | None = None,
        *,
        evaluation_only: bool = False,
    ) -> None:
        cleaned = _clean(query)
        if not cleaned:
            return
        pair = {
            "id": _stable_id("pair", family, cleaned, doc_id),
            "query": cleaned,
            "documentID": doc_id,
            "label": 1.0,
            "family": family,
            "metadata": metadata or {},
        }
        if evaluation_only or doc_id in evaluation_only_document_ids:
            pair["_evaluationOnly"] = True
        pairs.append(pair)

    tool_doc_ids: dict[str, str] = {}
    for tool in sorted(manifest.tools, key=lambda item: item.id):
        arg_lines = [
            f"{arg.name}: {arg.type}; required={arg.required}; {arg.description or ''}".strip()
            for arg in tool.arguments
        ]
        text = "\n".join([
            f"Tool ID: {tool.id}",
            f"Display name: {tool.displayName or tool.id}",
            f"Description: {tool.description or 'No explicit description.'}",
            f"Requires approval: {tool.requiresApproval}",
            f"Permission key: {tool.permissionKey or 'none'}",
            f"Permission kind: {tool.permissionKind or 'none'}",
            f"Confirmation mode: {tool.confirmationMode or 'none'}",
            "Arguments:",
            *arg_lines,
            f"Source: {tool.source or tool.inferredSource or 'unknown'}",
        ])
        tool_metadata = {
            "toolID": tool.id,
            "requiresApproval": tool.requiresApproval,
            "permissionKey": tool.permissionKey,
            "permissionKind": tool.permissionKind,
            "confirmationMode": tool.confirmationMode,
            "source": tool.source,
        }
        doc_id = add_doc(
            "tool_schema",
            tool.id,
            f"Tool schema: {tool.id}",
            text,
            tool_metadata,
        )
        tool_doc_ids[tool.id] = doc_id
        add_pair(f"Which tool should handle {tool.displayName or tool.id}?", doc_id, "natural_query_to_tool_schema", tool_metadata)
        add_pair(f"Find the schema and arguments for `{tool.id}`.", doc_id, "tool_id_to_tool_contract", tool_metadata)
        add_pair(f"When is `{tool.id}` allowed and what arguments does it require?", doc_id, "tool_contract_query", tool_metadata)

    intent_doc_ids: dict[str, str] = {}
    for intent in sorted(manifest.intents, key=lambda item: item.id):
        text = "\n".join([
            f"Intent: {intent.id}",
            f"Allowed tool IDs: {', '.join(intent.allowedToolIDs) or 'none'}",
            f"Source: {intent.source or 'unknown'}",
        ])
        doc_id = add_doc("intent", intent.id, f"Intent: {intent.id}", text, {"allowedToolIDs": intent.allowedToolIDs})
        intent_doc_ids[intent.id] = doc_id
        add_pair(f"What tools are allowed for the `{intent.id}` intent?", doc_id, "natural_query_to_intent_rule", {"intent": intent.id})
        add_pair(f"Route a user request with intent `{intent.id}`.", doc_id, "routing_rule_query", {"intent": intent.id})

    for entry in sorted(manifest.routingMatrix, key=lambda item: item.intent):
        text = "\n".join([
            f"Routing rule for intent: {entry.intent}",
            f"Allowed tools: {', '.join(entry.allowedTools) or 'none'}",
            f"Forbidden tools: {', '.join(entry.forbiddenTools) or 'none'}",
        ])
        doc_id = add_doc("routing_rule", entry.intent, f"Routing rule: {entry.intent}", text, {"allowedTools": entry.allowedTools, "forbiddenTools": entry.forbiddenTools})
        add_pair(f"Which tool can Cortex select for `{entry.intent}`?", doc_id, "natural_query_to_routing_rule", {"intent": entry.intent})
        add_pair(f"Approval and forbidden-tool boundary for `{entry.intent}`.", doc_id, "permission_boundary_query", {"intent": entry.intent})

    for slot in sorted(manifest.fleet.slots, key=lambda item: item.id):
        text = "\n".join([
            f"Fleet slot: {slot.id}",
            f"Role: {slot.role}",
            f"Model family: {slot.modelFamily or 'unknown'}",
            "Responsibilities:",
            *slot.responsibilities,
            f"Source: {slot.source or 'unknown'}",
        ])
        doc_id = add_doc("fleet_slot", slot.id, f"Fleet slot: {slot.id}", text, {"role": slot.role, "modelFamily": slot.modelFamily})
        add_pair(f"What is the role of the `{slot.id}` agent?", doc_id, "agent_role_question_to_fleet_slot", {"slot": slot.id})
        add_pair(f"How should other agents interact with `{slot.id}`?", doc_id, "peer_boundary_query", {"slot": slot.id})

    for scope in sorted(set(manifest.memory.scopes)):
        doc_id = add_doc("memory_scope", scope, f"Memory scope: {scope}", f"Memory scope `{scope}` defines retrieval/storage boundary for Lumen memory and RAG records.", {"scope": scope})
        add_pair(f"Which memory scope should store or retrieve `{scope}` information?", doc_id, "memory_rag_query_to_scope", {"scope": scope})

    for freshness in sorted(manifest.memory.freshnessClasses, key=lambda item: item.id):
        text = f"Freshness class `{freshness.id}` has ttlSeconds={freshness.ttlSeconds} durable={freshness.durable}. Source: {freshness.source or 'unknown'}."
        doc_id = add_doc("memory_scope", freshness.id, f"Freshness class: {freshness.id}", text, {"ttlSeconds": freshness.ttlSeconds, "durable": freshness.durable})
        add_pair(f"Should `{freshness.id}` memory be considered durable or time-limited?", doc_id, "memory_freshness_query", {"freshnessClass": freshness.id})

    for file_hash in sorted(manifest.sourceIntegrity.files, key=lambda item: item.path):
        doc_id = add_doc(
            "source_code_map_entry",
            file_hash.path,
            f"Source file: {file_hash.path}",
            f"Source file `{file_hash.path}` is part of the manifest source map with sha256 `{file_hash.sha256}`.",
            {"path": file_hash.path, "sha256": file_hash.sha256},
        )
        add_pair(f"Where is `{file_hash.path}` represented in the source map?", doc_id, "code_domain_query_to_source_map", {"path": file_hash.path})
        add_pair(f"Find the code file related to {file_hash.path.split('/')[-1]}", doc_id, "source_file_name_query", {"path": file_hash.path})

    for family in ("tool_schema_cards", "manifest_grounding_cards", "runtime_audit_repairs", "eval_scenarios", "codebase_home_corpus", "codebase_home_chunks"):
        limit = None if family in {"codebase_home_corpus", "codebase_home_chunks"} else 300
        records = datasets.get(family, [])
        selected_records = records if limit is None else records[:limit]
        for index, record in enumerate(selected_records):
            doc_id = _record_to_corpus(add_doc, family, index, record)
            query = _query_for_dataset_record(family, record)
            if query:
                add_pair(
                    query,
                    doc_id,
                    f"{family}_retrieval",
                    {"sourceFamily": family},
                    evaluation_only=family == "eval_scenarios",
                )

    train_pairs, val_pairs, eval_pairs = _split_pair_groups(pairs)
    known_positive_ids = _known_positive_documents(pairs)
    split_document_ids = {
        "train": {str(pair["documentID"]) for pair in train_pairs},
        "validation": {str(pair["documentID"]) for pair in val_pairs},
        "evaluation": {str(pair["documentID"]) for pair in eval_pairs},
    }

    doc_by_id = {doc["id"]: doc for doc in corpus}
    doc_tokens_by_id = {
        doc_id: _tokens(str(doc.get("text") or "") + " " + str(doc.get("title") or ""))
        for doc_id, doc in doc_by_id.items()
    }
    for pair in train_pairs + val_pairs:
        negative_id = _select_hard_negative(
            pair,
            doc_by_id,
            doc_tokens_by_id,
            known_positive_ids,
            split_document_ids[pair["split"]],
        )
        if not negative_id:
            continue
        hard_negatives.append({
            "id": _stable_id("hard_negative", pair["id"], negative_id),
            "query": pair["query"],
            "positiveDocumentID": pair["documentID"],
            "negativeDocumentID": negative_id,
            "family": pair["family"],
            "split": pair["split"],
            "groupID": pair["groupID"],
            "reason": _negative_reason(doc_by_id[pair["documentID"]], doc_by_id[negative_id]),
            "metadata": pair.get("metadata", {}),
        })

    evals = _build_eval_records(
        eval_pairs,
        doc_by_id,
        doc_tokens_by_id,
        known_positive_ids,
        split_document_ids["evaluation"],
    )

    triplets = [
        {
            "id": _stable_id("triplet", item["id"]),
            "query": item["query"],
            "positiveDocumentID": item["positiveDocumentID"],
            "negativeDocumentID": item["negativeDocumentID"],
            "family": item["family"],
            "split": item["split"],
            "groupID": item["groupID"],
            "metadata": item.get("metadata", {}),
        }
        for item in hard_negatives
    ]

    train_triplets = [item for item in triplets if item.get("split") == "train"]
    val_triplets = [item for item in triplets if item.get("split") == "validation"]
    dataset_card = {
        "schemaVersion": EMBEDDING_DATASET_SCHEMA_VERSION,
        "model": EMBEDDING_MODEL_ID,
        "teacherModel": EMBEDDING_TEACHER_MODEL_ID,
        "task": "retrieval_similarity_ranking",
        "nonGoals": [
            "Do not train embedding model on chat SFT records.",
            "Do not place eval-scenario documents, eval-containing source documents, or connected query groups in training or validation artifacts.",
            "Do not expose raw private runtime state or hidden reasoning.",
            "Do not treat static scenario checks as live E2E model evidence.",
        ],
        "evaluationIsolation": {
            "sourceFamilies": ["eval_scenarios"],
            "policy": "evaluation_only_with_connected_query_groups",
            "contentPolicy": "normalized_eval_user_segment_documents_are_evaluation_only",
        },
        "counts": {
            "corpus": len(corpus),
            "trainPairs": len(train_pairs),
            "valPairs": len(val_pairs),
            "trainTriplets": len(train_triplets),
            "valTriplets": len(val_triplets),
            "hardNegatives": len(hard_negatives),
            "evalRetrieval": len(evals),
        },
        "promotionMetrics": {
            "recallAt1Minimum": 0.72,
            "recallAt5Minimum": 0.90,
            "mrrMinimum": 0.78,
            "ndcgAt5Minimum": 0.82,
            "hardNegativeAccuracyMinimum": 0.85,
            "toolRetrievalAccuracyMinimum": 0.90,
            "sourceMapRetrievalAccuracyMinimum": 0.80,
            "runtimeRepairRetrievalAccuracyMinimum": 0.78,
            "embeddingHealthCheckPassRate": 1.0,
        },
        "families": sorted({record.get("family") for record in pairs if record.get("family")}),
    }

    return EmbeddingDatasets(
        corpus=corpus,
        train_pairs=train_pairs,
        val_pairs=val_pairs,
        train_triplets=train_triplets,
        val_triplets=val_triplets,
        hard_negatives=hard_negatives,
        eval_retrieval=evals,
        dataset_card=dataset_card,
    )


def _record_to_corpus(add_doc: Any, family: str, index: int, record: dict[str, Any]) -> str:
    record_id = str(record.get("id") or _stable_id(family, index, record))
    object_type = {
        "tool_schema_cards": "tool_schema",
        "manifest_grounding_cards": "manifest_grounding_card",
        "runtime_audit_repairs": "repair_sample",
        "eval_scenarios": "eval_scenario",
        "codebase_home_corpus": "codebase_home_module",
        "codebase_home_chunks": "codebase_home_source_chunk",
    }.get(family, family)
    title = str(record.get("title") or record.get("path") or record.get("taskType") or record.get("type") or f"{family}:{record_id}")
    if family == "codebase_home_corpus":
        text = "\n".join(
            [
                f"Path: {record.get('path')}",
                f"Module: {record.get('module')}",
                f"Language: {record.get('language')}",
                f"Responsibility: {record.get('responsibility')}",
                f"Symbols: {', '.join(record.get('symbols') or [])}",
                f"Imports: {', '.join(record.get('imports') or [])}",
                f"Evidence:\n{record.get('evidenceSnippet') or ''}",
            ]
        )
    elif family == "codebase_home_chunks":
        text = "\n".join(
            [
                f"Path: {record.get('path')}",
                f"Module: {record.get('module')}",
                f"Language: {record.get('language')}",
                f"Source hash: {record.get('sha256')}",
                f"Chunk hash: {record.get('chunkSHA256')}",
                f"Lines: {record.get('lineStart')}-{record.get('lineEnd')}",
                f"Source:\n{record.get('text') or ''}",
            ]
        )
    else:
        text = json.dumps(record, ensure_ascii=False, sort_keys=True)
    return add_doc(object_type, record_id, title, text, {"sourceFamily": family})


def _query_for_dataset_record(family: str, record: dict[str, Any]) -> str:
    if family == "runtime_audit_repairs":
        return str(record.get("promptPrefix") or record.get("scenario") or record.get("lesson") or "Find the repair sample for this runtime failure.")
    if family == "eval_scenarios":
        messages = record.get("messages") if isinstance(record.get("messages"), list) else []
        for message in messages:
            if isinstance(message, dict) and message.get("role") == "user":
                return str(message.get("content") or "")
        return str(record.get("prompt") or "Find the eval scenario for this agent behaviour.")
    if family == "tool_schema_cards":
        return str(record.get("query") or record.get("toolID") or record.get("id") or "Find the matching tool schema.")
    if family == "manifest_grounding_cards":
        return str(record.get("query") or record.get("summary") or record.get("id") or "Find the manifest grounding card.")
    if family == "codebase_home_corpus":
        path = str(record.get("path") or "")
        module = str(record.get("module") or "")
        symbols = record.get("symbols") if isinstance(record.get("symbols"), list) else []
        if path == ".":
            return "What are the main modules in Lumen's app codebase home?"
        if symbols:
            return f"Where is `{symbols[0]}` implemented in the Lumen codebase?"
        return f"Which file owns `{module}` behavior in Lumen?"
    if family == "codebase_home_chunks":
        path = str(record.get("path") or "")
        line_start = record.get("lineStart")
        line_end = record.get("lineEnd")
        return f"What source text does `{path}` contain on lines {line_start}-{line_end}?"
    return ""


def _select_hard_negative(
    pair: dict[str, Any],
    docs: dict[str, dict[str, Any]],
    doc_tokens_by_id: dict[str, set[str]],
    known_positive_ids: dict[str, set[str]],
    allowed_document_ids: set[str],
) -> str | None:
    positive = docs.get(str(pair.get("documentID")))
    if not positive:
        return None
    query_tokens = _tokens(str(pair.get("query") or ""))
    excluded_ids = known_positive_ids.get(_normalize_query(str(pair.get("query") or "")), set())
    positive_type = str(positive.get("objectType") or "")
    candidates = [
        doc
        for doc in docs.values()
        if doc["id"] != positive["id"]
        and str(doc["id"]) not in excluded_ids
        and str(doc["id"]) in allowed_document_ids
    ]
    same_type = [doc for doc in candidates if doc.get("objectType") == positive_type]
    pool = same_type or candidates
    if not pool:
        return None
    ranked = sorted(
        pool,
        key=lambda doc: (
            -len(query_tokens.intersection(doc_tokens_by_id.get(str(doc.get("id")), set()))),
            str(doc.get("id")),
        ),
    )
    return str(ranked[0]["id"])


def _negative_reason(positive: dict[str, Any], negative: dict[str, Any]) -> str:
    if positive.get("objectType") == negative.get("objectType"):
        return f"same object type `{positive.get('objectType')}` but wrong object ID"
    return f"similar retrieval surface but wrong object type `{negative.get('objectType')}`"


def _known_positive_documents(records: list[dict[str, Any]]) -> dict[str, set[str]]:
    known: dict[str, set[str]] = {}
    for record in records:
        query = _normalize_query(str(record.get("query") or ""))
        document_id = str(record.get("documentID") or "")
        if query and document_id:
            known.setdefault(query, set()).add(document_id)
    return known


def _split_pair_groups(
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Split connected query/document groups into train, validation, and eval.

    A connected component keeps every normalized query and every positive
    document in exactly one split, including cases where query variants point
    at the same document or one query has multiple valid positive documents.
    If any pair in a component is marked evaluation-only, the entire connected
    component is reserved for evaluation so the same query cannot leak through
    a different source family.
    """
    if not records:
        return [], [], []

    parent: dict[str, str] = {}

    def find(node: str) -> str:
        parent.setdefault(node, node)
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        if left_root < right_root:
            parent[right_root] = left_root
        else:
            parent[left_root] = right_root

    normalized_records: list[tuple[dict[str, Any], str, str]] = []
    for record in records:
        query = _normalize_query(str(record.get("query") or ""))
        document_id = str(record.get("documentID") or "")
        if not query or not document_id:
            continue
        query_node = f"query:{query}"
        document_node = f"document:{document_id}"
        union(query_node, document_node)
        normalized_records.append((record, query_node, document_node))

    grouped: dict[str, list[dict[str, Any]]] = {}
    for record, query_node, _ in normalized_records:
        grouped.setdefault(find(query_node), []).append(record)

    ordered_groups: list[tuple[str, list[dict[str, Any]]]] = []
    for records_in_group in grouped.values():
        record_ids = sorted(str(record.get("id") or "") for record in records_in_group)
        group_id = _stable_id("pair_group", record_ids)
        ordered_groups.append((group_id, records_in_group))
    ordered_groups.sort(key=lambda item: _stable_id("pair_group_order", item[0]))

    assignments: dict[str, list[tuple[str, list[dict[str, Any]]]]] = {
        "train": [],
        "validation": [],
        "evaluation": [],
    }
    evaluation_only_groups = [
        (group_id, records_in_group)
        for group_id, records_in_group in ordered_groups
        if any(record.get("_evaluationOnly") is True for record in records_in_group)
    ]
    splittable_groups = [
        (group_id, records_in_group)
        for group_id, records_in_group in ordered_groups
        if not any(record.get("_evaluationOnly") is True for record in records_in_group)
    ]
    for group_id, records_in_group in splittable_groups:
        bucket = int(_stable_id("pair_group_split", group_id)[:8], 16) % 10
        split = "evaluation" if bucket == 0 else "validation" if bucket == 1 else "train"
        assignments[split].append((group_id, records_in_group))

    if len(splittable_groups) >= 3:
        _ensure_nonempty_group_splits(assignments)
    assignments["evaluation"].extend(evaluation_only_groups)

    def records_for(split: str) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        for group_id, records_in_group in assignments[split]:
            output.extend(
                {
                    **{
                        key: value
                        for key, value in record.items()
                        if key != "_evaluationOnly"
                    },
                    "split": split,
                    "groupID": group_id,
                }
                for record in sorted(records_in_group, key=lambda item: str(item.get("id") or ""))
            )
        return sorted(output, key=lambda item: str(item.get("id") or ""))

    return records_for("train"), records_for("validation"), records_for("evaluation")


def _ensure_nonempty_group_splits(
    assignments: dict[str, list[tuple[str, list[dict[str, Any]]]]],
) -> None:
    for missing_split in ("train", "validation", "evaluation"):
        if assignments[missing_split]:
            continue
        donors = [split for split, groups in assignments.items() if len(groups) > 1]
        if not donors:
            return
        donor = sorted(donors, key=lambda split: (-len(assignments[split]), split))[0]
        moved = sorted(assignments[donor], key=lambda item: item[0])[0]
        assignments[donor].remove(moved)
        assignments[missing_split].append(moved)


def _build_eval_records(
    pairs: list[dict[str, Any]],
    docs: dict[str, dict[str, Any]],
    doc_tokens_by_id: dict[str, set[str]],
    known_positive_ids: dict[str, set[str]],
    allowed_document_ids: set[str],
) -> list[dict[str, Any]]:
    by_query: dict[str, list[dict[str, Any]]] = {}
    for pair in pairs:
        query = _normalize_query(str(pair.get("query") or ""))
        if query:
            by_query.setdefault(query, []).append(pair)

    evals: list[dict[str, Any]] = []
    for normalized_query, query_pairs in sorted(by_query.items()):
        ordered_pairs = sorted(query_pairs, key=lambda item: str(item.get("id") or ""))
        representative = ordered_pairs[0]
        positive_ids = sorted(
            {
                str(pair.get("documentID") or "")
                for pair in ordered_pairs
                if str(pair.get("documentID") or "") in docs
            }
        )
        if not positive_ids:
            continue
        negative_id = _select_hard_negative(
            representative,
            docs,
            doc_tokens_by_id,
            known_positive_ids,
            allowed_document_ids,
        )
        if not negative_id:
            continue
        evals.append({
            "id": _stable_id("eval_retrieval", normalized_query, positive_ids),
            "query": representative["query"],
            "positiveDocumentIDs": positive_ids,
            "hardNegativeDocumentIDs": [negative_id],
            "family": representative["family"],
            "split": "evaluation",
            "groupID": representative["groupID"],
            "metrics": ["recall@1", "recall@5", "mrr", "ndcg@5", "hard_negative_accuracy"],
            "metadata": representative.get("metadata", {}),
        })
    return evals


def _tokens(value: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9_.-]+", value.casefold()) if len(token) > 1}


def _clean(value: str) -> str:
    return " ".join(str(value or "").strip().split())


def _normalize_query(value: str) -> str:
    return _clean(value).casefold()


def _evaluation_user_segments(
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


def _stable_id(*parts: Any) -> str:
    payload = json.dumps(parts, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]
