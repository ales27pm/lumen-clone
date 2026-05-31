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
    evals: list[dict[str, Any]] = []

    def add_doc(object_type: str, object_id: str, title: str, text: str, metadata: dict[str, Any] | None = None) -> str:
        doc_id = _stable_id("doc", object_type, object_id)
        if not text.strip():
            text_value = title
        else:
            text_value = text.strip()
        corpus.append({
            "id": doc_id,
            "objectType": object_type,
            "objectID": object_id,
            "title": title.strip() or object_id,
            "text": text_value,
            "metadata": metadata or {},
        })
        return doc_id

    def add_pair(query: str, doc_id: str, family: str, metadata: dict[str, Any] | None = None) -> None:
        cleaned = _clean(query)
        if not cleaned:
            return
        pairs.append({
            "id": _stable_id("pair", family, cleaned, doc_id),
            "query": cleaned,
            "documentID": doc_id,
            "label": 1.0,
            "family": family,
            "metadata": metadata or {},
        })

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
            "Arguments:",
            *arg_lines,
            f"Source: {tool.source or tool.inferredSource or 'unknown'}",
        ])
        doc_id = add_doc(
            "tool_schema",
            tool.id,
            f"Tool schema: {tool.id}",
            text,
            {"requiresApproval": tool.requiresApproval, "permissionKey": tool.permissionKey, "source": tool.source},
        )
        tool_doc_ids[tool.id] = doc_id
        add_pair(f"Which tool should handle {tool.displayName or tool.id}?", doc_id, "natural_query_to_tool_schema", {"toolID": tool.id})
        add_pair(f"Find the schema and arguments for `{tool.id}`.", doc_id, "tool_id_to_tool_contract", {"toolID": tool.id})
        add_pair(f"When is `{tool.id}` allowed and what arguments does it require?", doc_id, "tool_contract_query", {"toolID": tool.id})

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

    for family in ("tool_schema_cards", "manifest_grounding_cards", "runtime_audit_repairs", "eval_scenarios"):
        for index, record in enumerate(datasets.get(family, [])[:300]):
            doc_id = _record_to_corpus(add_doc, family, index, record)
            query = _query_for_dataset_record(family, record)
            if query:
                add_pair(query, doc_id, f"{family}_retrieval", {"sourceFamily": family})

    doc_by_id = {doc["id"]: doc for doc in corpus}
    for pair in pairs:
        negative_id = _select_hard_negative(pair, doc_by_id)
        if not negative_id:
            continue
        hard_negatives.append({
            "id": _stable_id("hard_negative", pair["id"], negative_id),
            "query": pair["query"],
            "positiveDocumentID": pair["documentID"],
            "negativeDocumentID": negative_id,
            "family": pair["family"],
            "reason": _negative_reason(doc_by_id[pair["documentID"]], doc_by_id[negative_id]),
            "metadata": pair.get("metadata", {}),
        })
        evals.append({
            "id": _stable_id("eval_retrieval", pair["id"]),
            "query": pair["query"],
            "positiveDocumentIDs": [pair["documentID"]],
            "hardNegativeDocumentIDs": [negative_id],
            "family": pair["family"],
            "metrics": ["recall@1", "recall@5", "mrr", "ndcg@5", "hard_negative_accuracy"],
            "metadata": pair.get("metadata", {}),
        })

    triplets = [
        {
            "id": _stable_id("triplet", item["id"]),
            "query": item["query"],
            "positiveDocumentID": item["positiveDocumentID"],
            "negativeDocumentID": item["negativeDocumentID"],
            "family": item["family"],
            "metadata": item.get("metadata", {}),
        }
        for item in hard_negatives
    ]

    train_pairs, val_pairs = _split(pairs)
    train_triplets, val_triplets = _split(triplets)
    dataset_card = {
        "schemaVersion": EMBEDDING_DATASET_SCHEMA_VERSION,
        "model": EMBEDDING_MODEL_ID,
        "teacherModel": EMBEDDING_TEACHER_MODEL_ID,
        "task": "retrieval_similarity_ranking",
        "nonGoals": [
            "Do not train embedding model on chat SFT records.",
            "Do not expose raw private runtime state or hidden reasoning.",
            "Do not treat static scenario checks as live E2E model evidence.",
        ],
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
    }.get(family, family)
    title = str(record.get("title") or record.get("taskType") or record.get("type") or f"{family}:{record_id}")
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
    return ""


def _select_hard_negative(pair: dict[str, Any], docs: dict[str, dict[str, Any]]) -> str | None:
    positive = docs.get(str(pair.get("documentID")))
    if not positive:
        return None
    query_tokens = _tokens(str(pair.get("query") or ""))
    positive_type = str(positive.get("objectType") or "")
    candidates = [doc for doc in docs.values() if doc["id"] != positive["id"]]
    same_type = [doc for doc in candidates if doc.get("objectType") == positive_type]
    pool = same_type or candidates
    if not pool:
        return None
    ranked = sorted(
        pool,
        key=lambda doc: (
            -len(query_tokens.intersection(_tokens(str(doc.get("text") or "") + " " + str(doc.get("title") or "")))),
            str(doc.get("id")),
        ),
    )
    return str(ranked[0]["id"])


def _negative_reason(positive: dict[str, Any], negative: dict[str, Any]) -> str:
    if positive.get("objectType") == negative.get("objectType"):
        return f"same object type `{positive.get('objectType')}` but wrong object ID"
    return f"similar retrieval surface but wrong object type `{negative.get('objectType')}`"


def _split(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train: list[dict[str, Any]] = []
    val: list[dict[str, Any]] = []
    for record in sorted(records, key=lambda item: str(item.get("id") or "")):
        target = val if int(_stable_id("split", record.get("id"))[:8], 16) % 10 == 0 else train
        cloned = {**record, "split": "validation" if target is val else "train"}
        target.append(cloned)
    if records and not val and len(train) > 1:
        val.append({**train.pop(0), "split": "validation"})
    return train, val


def _tokens(value: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9_.-]+", value.casefold()) if len(token) > 1}


def _clean(value: str) -> str:
    return " ".join(str(value or "").strip().split())


def _stable_id(*parts: Any) -> str:
    payload = json.dumps(parts, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]
