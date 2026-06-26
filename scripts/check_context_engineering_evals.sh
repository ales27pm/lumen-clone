#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="$ROOT/evals/context_regression.yaml"

required_files=(
  "$CONFIG"
  "$ROOT/evals/context_regression/memory_recall.jsonl"
  "$ROOT/evals/context_regression/rag_local_qa.jsonl"
  "$ROOT/evals/context_regression/code_assist_long_context.jsonl"
  "$ROOT/evals/context_regression/tool_routing_guardrails.jsonl"
)

required_terms=(
  "answer_relevance"
  "faithfulness"
  "context_precision"
  "context_recall"
  "tool_call_precision"
  "p95_latency_ms"
  "prompt_input_tokens"
  "citation_hit_rate"
  "pii_leakage_score"
  "context_query_expansion_rate"
  "memory_tier_coverage"
  "contextQueryExpanded"
  "memoryTierCounts"
)

for file in "${required_files[@]}"; do
  if [[ ! -s "$file" ]]; then
    echo "missing or empty: ${file#$ROOT/}" >&2
    exit 1
  fi
done

for term in "${required_terms[@]}"; do
  if ! grep -q "$term" "$CONFIG"; then
    echo "missing config term: $term" >&2
    exit 1
  fi
done

for dataset in "${required_files[@]:1}"; do
  if ! python3 -m json.tool "$dataset" >/dev/null 2>&1; then
    while IFS= read -r line; do
      [[ -z "$line" ]] && continue
      python3 -m json.tool <<<"$line" >/dev/null
    done < "$dataset"
  fi
done

echo "context engineering eval artifacts ok"
