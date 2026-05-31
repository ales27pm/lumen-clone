# Lumen App Plan

This document tracks strategic implementation work for Lumen that should survive beyond one generated dataset cycle.

## Qwen3-first model migration plan

### Goal

Move Lumen toward a Qwen3-first model stack while preserving the current Qwen2.5-based agent models as the baseline until the app-specific evaluation loop proves that Qwen3 is better for Lumen's real runtime behaviour.

The target is not a blind model swap. The target is a measured migration where each model role improves on:

- manifest obedience;
- tool-routing accuracy;
- strict JSON/tool-call discipline;
- source-code and fleet self-awareness;
- runtime repair quality;
- RAG/memory retrieval quality;
- TestFlight/E2E behaviour;
- latency, memory, battery, and thermal behaviour.

### Recommended baseline

Keep the current Qwen2.5 1.5B-style agent models as the benchmark baseline until Qwen3 candidates beat them on Lumen-specific evals.

### Immediate model addition

Add the official Qwen3 embedding model as the default embedding candidate:

```text
Qwen/Qwen3-Embedding-0.6B
```

Use it for:

- source-code manifest retrieval;
- tool schema retrieval;
- routing rule retrieval;
- fleet/peer role retrieval;
- memory and RAG retrieval;
- runtime audit failure clustering;
- E2E failure to repair-sample retrieval;
- source map and code-domain retrieval.

Reasoning: Qwen3-Embedding-0.6B is official, embedding-native, Qwen-family, sentence-transformers compatible, and much more appropriate than low-adoption Qwen2.5 community embedding conversions for Lumen's core retrieval layer.

### Embedding model configuration and override policy

The embedding model must be configurable and reversible. Do not hard-code the Qwen3 embedding model as an irreversible runtime dependency.

Planned configuration keys:

```json
{
  "embedding": {
    "defaultModelID": "Qwen/Qwen3-Embedding-0.6B",
    "fallbackModelID": "current-baseline-embedding-model",
    "teacherModelID": "Qwen/Qwen3-Embedding-4B",
    "provider": "local",
    "quantization": "runtime-selected",
    "enableQwen3EmbeddingCandidate": true,
    "allowRuntimeFallback": true
  }
}
```

Required override layers, from highest to lowest priority:

1. explicit developer/test override in the app settings or debug configuration;
2. environment/build flag used by CI/TestFlight experiments;
3. persisted app configuration bundled with the model manifest;
4. generated dataset/model card default;
5. compiled fallback default.

Suggested environment or build flags:

```text
LUMEN_EMBEDDING_MODEL_ID=Qwen/Qwen3-Embedding-0.6B
LUMEN_EMBEDDING_FALLBACK_MODEL_ID=<current-baseline-embedding-model>
LUMEN_ENABLE_QWEN3_EMBEDDING=1
LUMEN_FORCE_BASELINE_EMBEDDING=0
```

Rollback behaviour:

- if Qwen3 embedding evals regress below the migration gates, set `LUMEN_FORCE_BASELINE_EMBEDDING=1` for the next build/run;
- if the Qwen3 model fails to load, produces invalid vector dimensions, or fails health checks, automatically fall back to `fallbackModelID` when `allowRuntimeFallback=true`;
- if fallback happens, the in-app dataset export must include `embeddingModelID`, `embeddingFallbackModelID`, `usedEmbeddingFallback`, and the health-check failure reason;
- the next improvement-loop cycle must ingest that fallback signal and add a gap if Qwen3 fallback occurred in TestFlight.

Required runtime export fields:

```json
{
  "embeddingModelID": "Qwen/Qwen3-Embedding-0.6B",
  "embeddingFallbackModelID": "current-baseline-embedding-model",
  "usedEmbeddingFallback": false,
  "embeddingVectorDimension": 0,
  "embeddingHealthCheckPassed": true,
  "embeddingEvalSummary": {
    "recallAt1": 0.0,
    "recallAt5": 0.0,
    "mrr": 0.0,
    "ndcgAt5": 0.0,
    "hardNegativeAccuracy": 0.0
  }
}
```

### Optional benchmark model

Track this as a heavier local/server benchmark, not the default mobile/runtime embedding model:

```text
Qwen/Qwen3-Embedding-4B
```

Use it to compare retrieval quality, generate teacher labels, or benchmark whether larger embedding capacity improves Lumen's retrieval tasks enough to justify the cost.

### Role-specific model selection

Use this as the initial model bill of materials. The app should keep model IDs configurable and should treat every Qwen3 agent model as a candidate until it beats the current baseline on Lumen-specific evals.

| Lumen role | Default/candidate model | Fallback | Teacher/benchmark | Notes |
|---|---|---|---|---|
| Embedding / RAG / memory retrieval | `Qwen/Qwen3-Embedding-0.6B` | current baseline embedding model | `Qwen/Qwen3-Embedding-4B` | First Qwen3 model to add. Use for source map, tool schema, memory/RAG, runtime repair, and E2E failure retrieval. |
| Reranker | `Qwen/Qwen3-Reranker-0.6B` *(candidate; disabled by default)* | embedding-only retrieval | `Qwen/Qwen3-Reranker-4B` | Rerank top-k retrieval results only. Do not run for every low-risk query if latency is tight. Enable only after reranker promotion gates pass; until then use embedding-only retrieval by default. |
| Cortex / router | `Qwen/Qwen3-1.7B` | current Qwen2.5 1.5B-style Cortex baseline | `Qwen/Qwen3-Coder-30B-A3B-Instruct` for offline dataset generation | Best first chat/agent migration because routing accuracy is measurable. |
| Executor / tool JSON | `Qwen/Qwen3-1.7B` fine-tuned for strict JSON | current Executor baseline | `Qwen/Qwen3-Coder-30B-A3B-Instruct` | Promote only after strict JSON validity and manifest-only tool gates pass. |
| REM / repair agent | `Qwen/Qwen3-1.7B` | current REM baseline | `Qwen/Qwen3-Coder-30B-A3B-Instruct` | Good early migration candidate because runtime repair quality is structured and measurable. |
| Mouth / final response | `Qwen/Qwen3-1.7B` | current Mouth baseline | current strongest local/server text model | Migrate later. Requires perfect sentinel suppression and strong user-facing quality. |
| Mimicry / style | `Qwen/Qwen3-0.6B` first, `Qwen/Qwen3-1.7B` if quality drops | current Mimicry baseline | `Qwen/Qwen3-1.7B` | Style adaptation may not need 1.7B. Promote 0.6B only if facts and safety boundaries remain stable. |
| Fleet / self-awareness | `Qwen/Qwen3-1.7B` | current Fleet baseline | `Qwen/Qwen3-Coder-30B-A3B-Instruct` | Use the same base as Cortex/REM for coherent source-map and peer-boundary knowledge. |
| Vision / screenshot/photo understanding | `Qwen/Qwen3-VL-2B-Instruct` | Apple/native or disabled vision path | `Qwen/Qwen3-VL-8B-Instruct` | Future candidate after text/RAG loop is stable. Use for screenshots, photos, visual memory, and document/image inspection. |
| Multimodal embedding | `Qwen/Qwen3-VL-Embedding-2B` | text-only embedding path | `Qwen/Qwen3-VL-Embedding-8B` | Future visual retrieval layer for photos, screenshots, and image-backed memory. |
| Multimodal reranker | `Qwen/Qwen3-VL-Reranker-2B` | text-only reranker or no reranker | `Qwen/Qwen3-VL-Reranker-8B` | Add only after multimodal corpus and eval records exist. |
| Voice ASR | Apple native speech first; experimental `Qwen/Qwen3-ASR-1.7B` | Apple native speech | hosted/local ASR benchmark | Keep Apple native as production default unless offline Qwen ASR wins on latency and reliability. |
| Voice TTS | Apple native speech first; experimental `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` | Apple native speech | hosted/local TTS benchmark | Treat Qwen TTS as experimental; production iOS path should remain native until proven. |
| Offline code/dataset teacher | `Qwen/Qwen3-Coder-30B-A3B-Instruct` | current local teacher model | `Qwen/Qwen3-Coder-30B-A3B-Instruct` GGUF variants | Not an iPhone runtime model. Use on local GPU/server for code-aware dataset generation, hard negatives, repair samples, and eval review. |

Recommended app/runtime stack:

```json
{
  "agentBaseline": "current Qwen2.5 1.5B-style model",
  "agentCandidate": "Qwen/Qwen3-1.7B",
  "styleCandidate": "Qwen/Qwen3-0.6B",
  "embeddingDefault": "Qwen/Qwen3-Embedding-0.6B",
  "embeddingTeacher": "Qwen/Qwen3-Embedding-4B",
  "rerankerDefault": "Qwen/Qwen3-Reranker-0.6B",
  "rerankerEnabledByDefault": false,
  "rerankerTeacher": "Qwen/Qwen3-Reranker-4B",
  "visionCandidate": "Qwen/Qwen3-VL-2B-Instruct",
  "visionEmbeddingCandidate": "Qwen/Qwen3-VL-Embedding-2B",
  "visionRerankerCandidate": "Qwen/Qwen3-VL-Reranker-2B",
  "codeDatasetTeacher": "Qwen/Qwen3-Coder-30B-A3B-Instruct"
}
```

Recommended on-device strategy:

- do not ship six independent full 1.7B role models by default;
- prefer one `Qwen/Qwen3-1.7B` agent base with role-specific prompts, adapters, or LoRA variants;
- keep `Qwen/Qwen3-Embedding-0.6B` as the dedicated retrieval model;
- add `Qwen/Qwen3-Reranker-0.6B` only if top-k reranking improves quality enough to justify latency;
- use `Qwen/Qwen3-0.6B` for Mimicry or low-cost fallback roles if it passes evals;
- keep vision, ASR, and TTS model paths experimental until the core text/RAG loop is stable.

Recommended local training/teacher stack:

```json
{
  "mainFineTuneBase": "Qwen/Qwen3-1.7B",
  "embeddingFineTuneBase": "Qwen/Qwen3-Embedding-0.6B",
  "rerankerFineTuneBase": "Qwen/Qwen3-Reranker-0.6B",
  "embeddingTeacher": "Qwen/Qwen3-Embedding-4B",
  "rerankerTeacher": "Qwen/Qwen3-Reranker-4B",
  "codeAndDatasetTeacher": "Qwen/Qwen3-Coder-30B-A3B-Instruct"
}
```

The offline teacher should generate or review:

- source-map summaries;
- code-aware eval prompts;
- hard negatives for similar tools and intents;
- runtime repair samples;
- E2E failure classification;
- strict JSON examples for Executor;
- fleet/self-awareness records;
- dataset quality reports.

Before any teacher-generated or teacher-reviewed record becomes training or eval data, the loop must apply the same sanitization boundary as the Non-goals section: automated PII/redaction filters, deterministic removal of hidden/internal reasoning traces, and stripping or redaction of raw log lines, trace identifiers, device identifiers, account identifiers, private file-system paths, and transient runtime state. Residual metadata is allowed only when it is needed for analysis and must be explicit, minimal, non-secret, and documented in the dataset card.

### Candidate migration order

Do not migrate every role at once. Add Qwen3 candidates in this order:

1. **Embedding** — safest and highest-confidence improvement.
2. **Reranker** — high precision gain, but only after embedding corpus/evals exist.
3. **Cortex** — routing and tool-selection accuracy are easy to measure.
4. **REM** — runtime repair quality is structured and can be evaluated without high user-facing risk.
5. **Executor** — migrate only after strict JSON/tool-call evals pass.
6. **Mouth** — migrate after sentinel suppression, failure summaries, and user-facing quality are stable.
7. **Mimicry** — migrate after style adaptation preserves facts and safety boundaries.
8. **Fleet/self-awareness** — migrate after source-map and peer-boundary evals are stable.
9. **Vision / multimodal retrieval** — migrate only after text/RAG loop stability.
10. **Voice** — keep Apple native first; test Qwen ASR/TTS experimentally.

### Required generated embedding datasets

Add a first-class embedding dataset generator to the improvement loop. The embedding model should not be trained on chat SFT records. It needs retrieval, similarity, ranking, and hard-negative data.

Planned output directory:

```text
generated/agent_manifest/embedding/
├── corpus.jsonl
├── train_pairs.jsonl
├── val_pairs.jsonl
├── train_triplets.jsonl
├── val_triplets.jsonl
├── hard_negatives.jsonl
├── eval_retrieval.jsonl
└── dataset_card.json
```

Required corpus object types:

- tool schema;
- intent;
- routing rule;
- source-file summary;
- fleet slot;
- peer boundary;
- memory scope;
- runtime failure;
- repair sample;
- eval scenario;
- manifest grounding card;
- source-code map entry.

Required pair/triplet families:

- natural user query → relevant tool schema;
- natural user query → relevant intent/routing rule;
- runtime failure → repair sample;
- E2E failure → corrected runtime repair;
- tool ID → source file and tool contract;
- source file → extracted manifest concept;
- agent role question → fleet slot / peer boundary;
- memory/RAG query → relevant memory or RAG chunk;
- permission/approval request → boundary rule;
- code-domain query → source-code map entry.

Required hard-negative families:

- `mail.draft` vs `outlook.draft.create`;
- `web.search` vs `web.fetch`;
- `memory.save` vs `memory.recall`;
- `calendar.create` vs `reminders.create`;
- `maps.search` vs `maps.directions`;
- `trigger.create` vs `reminders.create`;
- tool schema vs similarly named but wrong tool;
- runtime repair sample vs unrelated repair sample;
- fleet peer role vs wrong peer role.

### Required reranker datasets

Add a reranker dataset once embedding evals exist. The reranker should learn pairwise query/document scoring, not general chat behaviour.

Planned output directory:

```text
generated/agent_manifest/reranker/
├── train_pairs.jsonl
├── val_pairs.jsonl
├── hard_negative_pairs.jsonl
├── eval_reranking.jsonl
└── dataset_card.json
```

Required reranker examples:

- query + correct tool schema vs similar wrong tool schema;
- query + correct source-map entry vs unrelated source file;
- runtime failure + correct repair sample vs unrelated repair sample;
- E2E failure + corrected output vs fixture-like or fabricated correction;
- memory/RAG query + correct chunk vs stale or wrong chunk;
- fleet role question + correct peer boundary vs wrong peer role.

### Required retrieval eval metrics

The loop should emit retrieval evals and track:

- Recall@1;
- Recall@5;
- MRR;
- nDCG;
- hard-negative accuracy;
- tool-retrieval accuracy;
- source-map retrieval accuracy;
- runtime-repair retrieval accuracy.

Initial numeric targets for the first Qwen3 embedding promotion gate:

| Metric | Minimum target | Preferred target | Notes |
|---|---:|---:|---|
| Recall@1 | >= 0.72 | >= 0.80 | Measured over `eval_retrieval.jsonl`; strict top hit. |
| Recall@5 | >= 0.90 | >= 0.95 | Required for RAG usability. |
| MRR | >= 0.78 | >= 0.85 | Penalizes correct hits buried too low. |
| nDCG@5 | >= 0.82 | >= 0.90 | Use graded relevance when multiple positives exist. |
| hard-negative accuracy | >= 0.85 | >= 0.92 | Must distinguish similar tools/intents. |
| tool-retrieval accuracy | >= 0.90 | >= 0.95 | Tool queries must retrieve the correct tool schema. |
| source-map retrieval accuracy | >= 0.80 | >= 0.88 | Code/source awareness retrieval. |
| runtime-repair retrieval accuracy | >= 0.78 | >= 0.86 | Failed runtime traces should retrieve the right repair family. |
| embedding health-check pass rate | 100% | 100% | Vector dimension, non-empty embeddings, no NaN/Inf. |

Promotion rule: Qwen3-Embedding-0.6B may become the default only if it meets every minimum target and is not worse than the baseline by more than 2 percentage points on any required metric. If it beats the baseline by at least 3 percentage points on Recall@5 or hard-negative accuracy without latency regression, promote it for the next TestFlight cycle.

Rollback rule: roll back to the configured fallback embedding model if any required metric drops below minimum, if hard-negative accuracy regresses by more than 5 percentage points, or if TestFlight runtime fallback occurs more than once in a release candidate cycle.

### Required reranker eval metrics

Initial numeric targets for `Qwen/Qwen3-Reranker-0.6B`:

| Metric | Minimum target | Preferred target | Notes |
|---|---:|---:|---|
| reranked Recall@1 | >= 0.82 | >= 0.90 | Measured after embedding top-k retrieval. |
| reranked nDCG@5 | >= 0.88 | >= 0.94 | Should improve over embedding-only ranking. |
| hard-negative pair accuracy | >= 0.88 | >= 0.95 | Critical for similar tools/intents. |
| top-5 reorder win rate vs embedding-only | >= 0.60 | >= 0.70 | Reranker must improve enough to justify latency. |
| P95 rerank latency regression | <= +15% | <= +8% | Compared with embedding-only retrieval. |

Promotion rule: enable reranking by default only if it improves Recall@1 by at least 5 percentage points or hard-negative accuracy by at least 4 percentage points without exceeding latency gates.

Rollback rule: disable reranking and use embedding-only retrieval if latency exceeds the budget or if reranked nDCG@5 is worse than embedding-only for two consecutive TestFlight cycles.

### Required behaviour eval metrics

Agent model migration must also use numeric gates. Initial targets:

| Metric | Minimum target | Preferred target | Applies to |
|---|---:|---:|---|
| manifest-only tool use | >= 0.98 | 1.00 | Cortex, Executor |
| hallucinated-tool rejection | >= 0.98 | 1.00 | Cortex, Executor |
| required-argument handling | >= 0.92 | >= 0.97 | Cortex, Executor |
| strict JSON validity | >= 0.98 | 1.00 | Executor |
| approval/permission boundary accuracy | >= 0.95 | >= 0.99 | Cortex, Executor, Mouth |
| sentinel suppression | 1.00 | 1.00 | Mouth, all user-facing paths |
| runtime repair usefulness | >= 0.85 | >= 0.92 | REM |
| fleet peer-boundary accuracy | >= 0.95 | >= 0.99 | Fleet/self-awareness records |
| TestFlight/E2E pass rate | >= 0.90 | >= 0.95 | Full app loop |
| crash-free TestFlight scenario run | 100% | 100% | Full app loop |
| P95 local response latency regression | <= +10% | <= +5% | Compared with current baseline |
| peak memory regression | <= +10% | <= +5% | Compared with current baseline |

Promotion rule: a Qwen3 candidate may replace a role only if it meets every minimum target, has zero sentinel leaks, has zero crash regressions in the TestFlight scenario batch, and is no worse than the baseline by more than 2 percentage points on any role-critical metric.

Rollback rule: immediately revert the role to the baseline model if TestFlight/E2E pass rate drops below 0.90, if any sentinel leak appears, if manifest-only tool use drops below 0.98, or if latency/memory regressions exceed the allowed thresholds.

### Improvement-loop integration

Add the embedding generator to:

- `tools/lumen_manifest_crawler/lumen_manifest_crawler/dataset/embedding.py`;
- `tools/lumen_manifest_crawler/lumen_manifest_crawler/dataset/__init__.py`;
- `tools/lumen_manifest_crawler/lumen_manifest_crawler/output/writer.py`;
- `tools/lumen_manifest_crawler/lumen_manifest_crawler/improvement_loop.py`;
- validators/tests;
- generated dataset cards and loop summaries.

Add the reranker generator to:

- `tools/lumen_manifest_crawler/lumen_manifest_crawler/dataset/reranker.py`;
- `tools/lumen_manifest_crawler/lumen_manifest_crawler/dataset/__init__.py`;
- `tools/lumen_manifest_crawler/lumen_manifest_crawler/output/writer.py`;
- `tools/lumen_manifest_crawler/lumen_manifest_crawler/improvement_loop.py`;
- validators/tests;
- generated dataset cards and loop summaries.

The loop summary should include:

```json
{
  "embedding": {
    "model": "Qwen/Qwen3-Embedding-0.6B",
    "fallbackModel": "current-baseline-embedding-model",
    "usedFallback": false,
    "corpusCount": 0,
    "pairCount": 0,
    "tripletCount": 0,
    "hardNegativeCount": 0,
    "evalCount": 0,
    "metrics": {
      "recallAt1": 0.0,
      "recallAt5": 0.0,
      "mrr": 0.0,
      "ndcgAt5": 0.0,
      "hardNegativeAccuracy": 0.0,
      "toolRetrievalAccuracy": 0.0,
      "sourceMapRetrievalAccuracy": 0.0,
      "runtimeRepairRetrievalAccuracy": 0.0
    }
  },
  "reranker": {
    "model": "Qwen/Qwen3-Reranker-0.6B",
    "fallbackMode": "embedding-only",
    "enabledByDefault": false,
    "pairCount": 0,
    "hardNegativePairCount": 0,
    "evalCount": 0,
    "metrics": {
      "rerankedRecallAt1": 0.0,
      "rerankedNdcgAt5": 0.0,
      "hardNegativePairAccuracy": 0.0,
      "top5ReorderWinRate": 0.0,
      "p95RerankLatencyRegression": 0.0
    }
  }
}
```

Counts should become non-zero once implementation is complete.

### Migration gates

A Qwen3 candidate may replace an existing model role only if it beats or matches the current Qwen2.5 baseline on:

- manifest-only tool use;
- hallucinated-tool rejection;
- required argument handling;
- approval/permission boundary behaviour;
- sentinel suppression;
- runtime repair quality;
- TestFlight/E2E pass rate;
- latency/memory budget;
- deterministic dataset validation.

Use the numeric targets in the retrieval, reranker, and behaviour metrics sections as the initial objective gates. Revise targets upward after the first stable baseline is measured.

### Non-goals

- Do not switch every agent model to Qwen3 without an A/B run.
- Do not run all role models as separate full models on-device by default.
- Do not use a normal chat model as an embedding model unless it is adapted and evaluated as an embedding model.
- Do not train the embedding model on conversational SFT records as if it were a chat agent.
- Do not train the reranker on conversational SFT records as if it were a chat agent.
- Do not expose raw private runtime state or hidden reasoning in embedding corpus records.
- Do not remove the baseline/fallback model until Qwen3 passes two consecutive TestFlight improvement-loop cycles.
- Do not make vision, ASR, or TTS Qwen models production defaults until text/RAG stability is proven.

### Decision

Adopt a Qwen3-first strategy, starting with:

```text
Embedding default: Qwen/Qwen3-Embedding-0.6B
Embedding fallback: current baseline embedding model until Qwen3 passes promotion gates
Reranker candidate: Qwen/Qwen3-Reranker-0.6B, disabled by default until it beats embedding-only retrieval
Agent candidate: Qwen/Qwen3-1.7B
Lightweight style/fallback candidate: Qwen/Qwen3-0.6B
Agent baseline: current Qwen2.5 1.5B-style models
Agent migration: staged Qwen3 candidates, promoted only by Lumen-specific evals
Vision candidate: Qwen/Qwen3-VL-2B-Instruct, future phase
Code/dataset teacher: Qwen/Qwen3-Coder-30B-A3B-Instruct, offline only
```
