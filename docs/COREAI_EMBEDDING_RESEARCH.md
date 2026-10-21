# Core AI embeddings for long-term memory

Research checked on 2026-10-20. This document describes a concrete candidate and its integration requirements. It does not enable a new embedding backend, install model weights, or establish iPhone qualification.

## Recommendation and current boundary

Keep one Gemma generation model for the app and a separate embedding runtime. Evaluate the existing **EmbeddingGemma 300M Core AI export** as the first Core AI retrieval candidate. Its published graph returns a sentence vector directly through the system Core AI API; it does not need another chat model, agent, sampling loop, or KV cache. Preserve the current functional embedding backend until the candidate passes the checks below.

An embedding model does not store or remember user facts. SwiftData retains memory contents and metadata; embedding vectors support semantic retrieval of those records. Retrieved text can then ground the single Gemma generator. Google's [EmbeddingGemma model card](https://ai.google.dev/gemma/docs/embeddinggemma/model_card) describes a multilingual retrieval model, with 768-dimensional vectors and a native 2K-token input limit. **The particular exported graph investigated here accepts only 256 tokens.** Its task, tokenizer, precision, and exported shape remain part of its identity.

## Apple support versus the community export

Apple documents the general path: an `.aimodel` is specialized with `AIModel`, a named `InferenceFunction` is loaded, and `NDArray` inputs produce named tensor outputs. This is a real Core AI integration path, independent of Core ML, MLX, or a GGUF backend. See [Apple's integration guide](https://developer.apple.com/documentation/coreai/integrating-on-device-ai-models-in-your-app-with-core-ai) and [InferenceFunction](https://developer.apple.com/documentation/coreai/inferencefunction).

The pinned Apple package, `apple/coreai-models@9db747612f972001279f02adb04d3ef982310d0e`, does not supply an EmbeddingGemma exporter or a sentence-embedding Swift service. Its [RoBERTa export](https://github.com/apple/coreai-models/blob/9db747612f972001279f02adb04d3ef982310d0e/models/roberta/export.py) returns `last_hidden_state` without a retrieval pooling/projection pipeline. That is not the sentence-vector contract required here.

Gemma's generative `load_embeddings`/gather functions and PLE sidecar are also not retrieval embeddings. They gather token-indexed rows for decoder inputs; PLE supplies a per-token, per-layer table. Neither computes a trained, normalized representation of an arbitrary memory or query. See Apple's [embedding primitives](https://github.com/apple/coreai-models/blob/9db747612f972001279f02adb04d3ef982310d0e/python/src/coreai_models/primitives/ios/embedding.py) and [PLE reader](https://github.com/apple/coreai-models/blob/9db747612f972001279f02adb04d3ef982310d0e/swift/Sources/CoreAILanguageModels/Assets/PerLayerEmbeddings.swift).

The candidate is a **community conversion**, maintained by the author of [CoreAI Model Zoo](https://github.com/john-rocky/coreai-model-zoo), of Google's `google/embeddinggemma-300m`. It is not an Apple-published embedding model. The canonical converted repository is [mlboydaisuke/embeddinggemma-300m-CoreAI](https://huggingface.co/mlboydaisuke/embeddinggemma-300m-CoreAI/tree/74f6d1fe2a566dde8ae2d13fae4f5f73e8a16055); `coreai-community/embeddinggemma-300m-CoreAI` explicitly identifies itself as a mirror.

## Immutable artifact and tensor contract

Converted repository revision: **`74f6d1fe2a566dde8ae2d13fae4f5f73e8a16055`**. Import the complete `model/` directory, not only the `.aimodel`. It contains one static FP32 graph, local tokenizer files, and a public parity fixture. The export executes the full sentence-transformers chain: transformer, masked mean pooling, dense projection, and L2 normalization. See the [conversion source](https://github.com/john-rocky/coreai-model-zoo/blob/1a3cf2ad83f0c5a658a3c38a917be030260e971f/conversion/export_embeddinggemma.py) and [recorded export recipe](https://github.com/john-rocky/coreai-model-zoo/blob/1a3cf2ad83f0c5a658a3c38a917be030260e971f/models/qwen3-embedding/recipe.toml).

| Function `main` value | Shape | Scalar type | Meaning |
| --- | --- | --- | --- |
| `input_ids` | `[1, 256]` | Int32 | Token IDs, right padded with ID `0` |
| `attention_mask` | `[1, 256]` | Int32 | `1` for real tokens, `0` for padding |
| `embedding` | `[1, 768]` | Float32 | One L2-normalized sentence vector |

The host must prepend the correct task prefix **before ordinary tokenizer encoding**:

| Input purpose | Prefix |
| --- | --- |
| Search query | `task: search result \| query: ` |
| Memory/document | `title: none \| text: ` |

Use `AutoTokenizer.from(modelFolder:)` and `encode(text:)`, with special-token behavior verified against the reference tokenizer. Do not use a chat template or a network-enabled pretrained initializer. The complete prefixed input, including tokenizer special tokens, must fit the 256-token grid. Reject an oversized query explicitly and chunk documents with tokenizer-aware boundaries; do not silently discard a memory's tail. The author's current helper truncates, so it should not be copied unchanged. These prefixes and tensor operations are visible in [TextEmbedder.swift](https://github.com/john-rocky/coreai-kit/blob/87838d6efd9f31540377e3764e7baefaefd86472/Sources/CoreAIKitEmbeddings/TextEmbedder.swift).

### Published byte identities

Paths below are relative to `model/`. Small-file SHA-256 values were calculated from the immutable revision's bytes. The two large-file digests and sizes were read from its Hugging Face LFS listing; the large payloads were **not downloaded or independently hashed** during this research.

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `embeddinggemma-300m_float32_static.aimodel/main.mlirb` | 1,244,157,504 | `5aa5a79d29c12a1cc654b1e6268c6d6185924e5a82229bb4ae2f3bcabb745297` |
| `embeddinggemma-300m_float32_static.aimodel/main.hash` | 32 | `4fd5481c6594a3408d026a5ac64f3bf30a6f05fbacf818bd1aaa4111ca933762` |
| `embeddinggemma-300m_float32_static.aimodel/metadata.json` | 340 | `05f24b6a4d0d2a919c93b56f9182b3ab8cf3c0f270ee5045bb2bca3021554af7` |
| `tokenizer/tokenizer.json` | 33,385,272 | `bcb89366c4787d27671d021cdc006defcd1cb53b02b04213ae8dd47c009c7f71` |
| `tokenizer/tokenizer_config.json` | 705 | `3620f952639a0bda00fa78ea42e04825588823088e5e132bcbb45a8ddb59e6a0` |
| `reference.json` | 1,616 | `f926c52ccc827c25c5e59541d173852c96fcc541ede7f93da0fb1730435925a2` |

The raw 32-byte `main.hash` contains the same digest declared for `main.mlirb`. This is consistency evidence, not an independent signature of the producer. The [immutable listing API](https://huggingface.co/api/models/mlboydaisuke/embeddinggemma-300m-CoreAI/tree/74f6d1fe2a566dde8ae2d13fae4f5f73e8a16055?recursive=true&expand=true) supplies the file inventory.

These six files total **1,277,545,469 bytes**, approximately 1.278 GB. This is storage size, not peak resident memory. Account separately for tokenizer memory, specialization buffers, graph intermediates, the Gemma generator, and other app allocations. The author's [export card](https://huggingface.co/mlboydaisuke/embeddinggemma-300m-CoreAI/blob/74f6d1fe2a566dde8ae2d13fae4f5f73e8a16055/README.md) reports NaN vectors for both full-FP16 and mixed-precision attempts and ships FP32. A stale `ModelID` comment saying FP16 and an older DocChat size estimate must not override the actual published files and recipe.

Weights follow the [Gemma terms](https://ai.google.dev/gemma/terms), not the converter's BSD-3-Clause license. This research does not accept a model license or authorize redistribution. The export revision pins published bytes; the recipe does **not** record an immutable revision of the original Google checkpoint. Resolve that source-lineage gap before a qualified release or reproduce a conversion from a separately pinned source.

## Swift integration contract

A future actor-owned embedding service can use the already linked system framework directly:

```swift
let model = try await AIModel(
    contentsOf: assetURL,
    options: SpecializationOptions(preferredComputeUnitKind: .gpu)
)
guard let function = try model.loadFunction(named: "main") else {
    // Throw a typed missing-function error.
    return
}
// Validate function.descriptor before allocating or running inputs.
var outputs = try await function.run(inputs: [
    "input_ids": idsArray,
    "attention_mask": maskArray
])
guard let vectorArray = outputs.remove("embedding")?.ndArray else {
    // Throw a typed missing-output error.
    return
}
let view = vectorArray.view(as: Float.self)
// Require contiguousElements, exactly 768 finite values, and a valid unit norm.
```

Validate exact input/output names, shapes, scalar types, and an empty state set using the function descriptor. Create the two Int32 `NDArray`s and fill their mutable views from the locally encoded, padded arrays. Retain model/function resources under one owner and serialize work to bound intermediate allocation. Check cancellation before and after inference and drain it before unloading; a cancelled vector must not be persisted. The API pattern is documented by [Apple](https://developer.apple.com/documentation/coreai/integrating-on-device-ai-models-in-your-app-with-core-ai) and implemented in the author's [GraphModel.swift](https://github.com/john-rocky/coreai-kit/blob/87838d6efd9f31540377e3764e7baefaefd86472/Sources/CoreAIKitVision/GraphModel.swift).

GPU is a specialization preference, not evidence of placement. No ANE use is established here. Linking CoreAIKit solely for this graph is unnecessary and would introduce an additional dependency graph; its local graph implementation serves as API evidence, not a requirement to adopt its automatic downloader or forked LLM runtime.

## Required persistence and input migration

Current [MemoryItem](../ios/Lumen/Models/MemoryItem.swift) stores an embedding array without its model identity, dimension, task format, or backend revision. [MemoryStore](../ios/Lumen/Services/MemoryStore.swift) currently obtains vectors without retaining that identity. A replacement backend must first add a reviewed SwiftData compatibility/migration plan and re-embed existing memories. Equal dimensions alone do not make two embedding spaces compatible.

[RAGChunk](../ios/Lumen/Models/RAGChunk.swift) already records model identifier, vector dimension, and format version; [RAGStore](../ios/Lumen/Services/RAGStore.swift) detects stale identities and requires reindexing. Preserve that distinction and rebuild derived vector indexes when the backend changes. Retain source memory/document text throughout the migration.

[SemanticEmbeddingText](../ios/Lumen/Assistant/SemanticEmbeddingText.swift) currently emits `search_query:` / `search_document:` with format version 3. Those prefixes cannot simply be wrapped inside EmbeddingGemma's prompts. Introduce an explicit query/document purpose at the embedding boundary, retain bounded metadata/content as data, then apply the selected model's prompt recipe exactly once. Include artifact digest, tokenizer identity, prefix/format version, pooling/normalization contract, and output dimension in the persisted embedding identity.

## Acceptance criteria before activation

1. Import the complete pinned package through a local, confined, streamed-integrity boundary. Reject missing files, symlinks, changed bytes, wrong tensor contracts, and unavailable resources. Do not load it at startup or download a missing tokenizer implicitly.
2. Resolve upstream checkpoint lineage and distribution terms. Record artifact identity separately from runtime qualification.
3. Compare local Swift token IDs and masks against the reference tokenizer, including accents, French text, emoji, special-token boundaries, and the 256-token limit. Preserve explicit query/document behavior and fail oversize queries instead of truncating silently.
4. On the actual target iPhone/iOS 27.2 app, execute finite-vector, dimension, norm, determinism, reference-cosine, and retrieval-ranking checks. Add a fixed French/multilingual memory corpus with close distractors, missing facts, conflicting revisions, and a regression comparison against the current embedding backend. Measure ranking preservation, not just cosine similarity.
5. Measure cold specialization, warm latency, cancellation, scene changes, memory pressure, thermal behavior, and peak memory with the single Gemma generator. No shared-residency or ANE claim follows from a model file or compute preference.
6. Complete memory identity migration and RAG reindexing before allowing cross-model semantic search; test interrupted/failed reindex recovery and preserve typed lexical fallback versus embedding failure.

The author's [opt-in smoke XCTest](https://github.com/john-rocky/coreai-kit/blob/87838d6efd9f31540377e3764e7baefaefd86472/Tests/CoreAIKitTests/TextEmbedderSmokeTests.swift) checks four public examples, cosine agreement and matching-document rank. It is useful preparation, not proof of this app's execution. The [published inventory](https://github.com/john-rocky/coreai-model-zoo/blob/1a3cf2ad83f0c5a658a3c38a917be030260e971f/models/_INVENTORY.md) labels the candidate's tier-1 check as metadata/tokenizer comparison without device execution, and its tier-0 runtime load is deferred. No fresh iPhone/iOS 27.2 qualification was found for this exact artifact during this review. Keep that gap explicit.
