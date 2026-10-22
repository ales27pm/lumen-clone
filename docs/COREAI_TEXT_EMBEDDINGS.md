# CoreAI text embeddings

The optional CoreAI embedding provider runs **EmbeddingGemma 2's text encoder** separately from the one Gemma chat generator. Embedding creates sentence retrieval vectors; it does not store memories, generate replies, invoke tools, or create a principal/sub-agent model fleet. The existing GGUF embedding provider remains the default. Selecting CoreAI is explicit and fails closed when unsupported, unprepared, missing, or corrupt; it does not fall back to vectors from another embedding space.

This implementation targets physical iOS devices on **iOS 27.1 or later**, preserving Lumen's deployment target. Apple's generic [AIModel](https://developer.apple.com/documentation/coreai/aimodel), [InferenceFunction](https://developer.apple.com/documentation/coreai/inferencefunction), and [NDArray](https://developer.apple.com/documentation/coreai/ndarray) APIs are documented on iOS from 27.0. Their availability establishes the API boundary, not this export's device compatibility or Neural Engine placement.

## Immutable sources and integration choice

- Model conversion: [EverPaper/embeddinggemma-2-coreai at 305fc2a96790d9343f38038aeca211dfca56d9a5](https://huggingface.co/EverPaper/embeddinggemma-2-coreai/tree/305fc2a96790d9343f38038aeca211dfca56d9a5), export version 1.2.
- Original checkpoint: [Google EmbeddingGemma 2 at 914f7f89142e33e77833254d9c9b90c3cef7303b](https://huggingface.co/google/embeddinggemma-2/tree/914f7f89142e33e77833254d9c9b90c3cef7303b). Export attribution names its `model.safetensors` SHA256 as `197a32965d4b1105faf060417baa899e193fb73cd401f42ec9295234d5553d79`.
- Exact tensor contract: pinned [model/config.json](https://huggingface.co/EverPaper/embeddinggemma-2-coreai/blob/305fc2a96790d9343f38038aeca211dfca56d9a5/model/config.json), [Swift Embedder.swift](https://huggingface.co/EverPaper/embeddinggemma-2-coreai/blob/305fc2a96790d9343f38038aeca211dfca56d9a5/swift/Sources/EmbeddingGemma2CoreAI/Embedder.swift), and [export/eg2_text_mm.py](https://huggingface.co/EverPaper/embeddinggemma-2-coreai/blob/305fc2a96790d9343f38038aeca211dfca56d9a5/export/eg2_text_mm.py).

The author's Swift package is inside the Hugging Face repository under `swift/`; the pinned model card links no separate GitHub Swift package. Its [Package.swift](https://huggingface.co/EverPaper/embeddinggemma-2-coreai/blob/305fc2a96790d9343f38038aeca211dfca56d9a5/swift/Package.swift) declares macOS 27, Swift tools 6.0, no third-party package dependencies, and includes image/audio support. Lumen uses Apple's framework directly and vendors only the author's two Foundation-only tokenizer source files. It adds no Swift package dependency and does not copy image/audio preprocessing, CLI, cache deletion, or prewarm behavior.

## Local package import

Obtain the nine files below from the immutable export revision through the user's own download workflow. **Assemble a new clean folder containing exactly this tree**, then select that folder in Lumen's CoreAI text embedding settings:

```text
EmbeddingGemma2-Text/
  LICENSE
  NOTICE
  model/
    MODEL-ATTRIBUTION.json
    config.json
    embeddinggemma-2-text-mm.aimodel/
      main.hash
      main.mlirb
      metadata.json
    tokenizer/
      tokenizer.json
      tokenizer_config.json
```

A complete Hugging Face checkout contains additional files and is rejected. A client download using `--local-dir` can also create `.cache/huggingface` metadata. The importer ignores only those download receipts and `.DS_Store` in the source inventory, validates their paths without following symlinks, and never copies them. The installed package still contains exactly nine files. The clean-folder assembly below also works. Do not copy `.cache`, README, Swift/Python/export/test directories, or the vision/audio assets. Empty directory scaffolding is not another model file; symlinks and nonregular files are rejected.

The importer copies the package into owned Application Support storage and streams SHA256 verification during import and preparation. The exact byte/hash inventory is pinned in `CoreAITextEmbeddingModelStore.pinnedFiles`, matching the author's [CHECKSUMS.sha256](https://huggingface.co/EverPaper/embeddinggemma-2-coreai/blob/305fc2a96790d9343f38038aeca211dfca56d9a5/CHECKSUMS.sha256):

| Relative path | Bytes | SHA256 |
| --- | ---: | --- |
| `LICENSE` | 11,358 | `cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30` |
| `NOTICE` | 21,485 | `b9ac552f2eb418f79dc23cc325a7b8c38ae05c1fcc8c1880014bae281a0963a6` |
| `model/MODEL-ATTRIBUTION.json` | 5,166 | `7951556c4559c8609cf67715acc53e58d8c70f7f62242494d7858406b1bbe2f7` |
| `model/config.json` | 4,643 | `0204c8d4532f6930709c3c8eb7f7b25b54e4be42aa7b8c6c397fc17e872fb7bf` |
| `model/embeddinggemma-2-text-mm.aimodel/main.hash` | 32 | `ea089710e57b22972d6ab4d4b960b385394b81f333ba2f75b4fb4aac1d238e6f` |
| `model/embeddinggemma-2-text-mm.aimodel/main.mlirb` | 632,206,233 | `0b51638f2d17794825de21def6ab485d968fdf412853ea209617623a51f1bbde` |
| `model/embeddinggemma-2-text-mm.aimodel/metadata.json` | 431 | `8984123cfdb32ba8cf526a1f3a41571205475aff40b3f1d65494f476f115f8a6` |
| `model/tokenizer/tokenizer.json` | 32,170,510 | `4d777ef5bdc1aa36227abdfb77c3e49e7b9c892d16e1b6bda41c393504828be4` |
| `model/tokenizer/tokenizer_config.json` | 1,599 | `17bd5d6e9364ca49a534e1502076593317c298d4a663623091ed45388f004874` |

`main.hash` contains the 32 binary bytes of the `main.mlirb` SHA256. The graph's reported size/hash was checked against both the pinned published inventory and Hugging Face LFS metadata during implementation; **model weights were not downloaded or executed for this work**. The importer verifies actual local graph bytes before use. Metadata, notices, tokenizer source and public test fixture bytes were independently read and checked. A re-export with different graph bytes is a different artifact and is deliberately rejected.

The shared [package safeguards](COREAI_GEMMA.md) apply to this encoder too: bounded progressive inventory, pinned sizes and aggregate quota checked before copying, chunk-scoped memory, cancellable settings operations, and recovery of interrupted installations before a new import. Recovery preserves the active installation, source text, source model folders and Apple caches. The standalone native regression command documented there exercises both real stores with synthetic temporary packages; it does not qualify this encoder on iPhone.

## Inference and text contract

`CoreAITextEmbeddingService.shared` owns one independent encoder, tokenizer and function set behind its actor. Explicit foreground preparation verifies the package and constructs `AIModel(contentsOf:options:)`, requesting `.neuralEngine` as a preference. All seven functions are loaded during preparation; no inference probe or prewarm is run. `embed(_:)` requires an already prepared model, so indexing, memory maintenance, background work and metadata discovery cannot load it accidentally.

Explicit document import/reindex, note saving and manual memory/bookmark actions use `ForegroundEmbeddingWork` to prepare the selected encoder through `ModelLoader` before calling the stores. They do not require a preceding chat. Inactive or cancelled actions do not begin indexing, and the action tasks participate in app cancellation. The settings panel reads the scene's typed Observation environment. A failed or cancelled attempt to load a GGUF embedding model restores the previously selected embedding provider; restoration errors remain visible.

The function is `mm_S` for the smallest fitting `S` in **32, 64, 128, 288, 512, 1024, 2048**. Before admission the provider verifies all function names, input/output shapes and scalar types, and absence of recurrent states.

| Tensor | Type and shape | Values |
| --- | --- | --- |
| `input_ids` | int32 `[1,S]` | BOS 2, purpose-prefix tokens plus raw text, EOS 1, right padding 0 |
| `attention_mask` | float16 `[1,S]` | 1 for real tokens including BOS/EOS; 0 for padding |
| `mm_embeds` | float16 `[1,S,512]` | Every value zero |
| `mm_mask` | float16 `[1,S]` | Every value zero |
| Output `embedding` | float16 `[1,768]` | Masked mean pooling, projection and L2 normalization already in the graph |

Raw caller text has an explicit `EmbeddingPurpose`. The provider adds exactly the model's prefix; it never applies a generative chat template:

| Purpose | Prefix |
| --- | --- |
| Retrieval query | `task: search result \| query: ` |
| Retrieval document | `title: {title or none} \| text: ` |
| Semantic similarity | `task: sentence similarity \| query: ` |
| Code query | `task: code retrieval \| query: ` |

The legacy `search_query:` / `search_document:` formatted request compatibility case is rejected by this provider. Production memory/RAG callers pass raw bodies and explicit purpose. Input size is bounded before tokenization. More than **2048 total tokens**, including prefix and BOS/EOS, returns a typed overflow error; Lumen does not silently truncate. Chunk documents before embedding. Google's original model supports 8192 tokens, while this pinned export supports 2048.

The default dimension is **256**. Callers may explicitly request 128, 256, 512 or 768; any other dimension is rejected. Native float16 output is widened to Double, checked for shape, contiguity and finite values, sliced to its Matryoshka prefix, and L2-normalized with Double accumulation. Empty, zero, nonfinite or malformed results are errors. Persisted identity includes the immutable export revision and format-4 processing recipe, plus MRL dimension. Queries and documents share that identity. Changing any source artifact, prefix/tokenization/normalization recipe, or dimension requires a different identity and a fresh source-based index.

The app cancellation bus registers active preparation and embedding tasks. Cancellation is cooperative: admission, model replacement and unload stay closed until the awaited framework operation actually drains. Model functions/buffers are released afterward. The provider never deletes Apple's caches or invokes the author's process-temp cleanup helper.

Import, removal and preference changes check cancellation before admission and again after draining the model, before modifying files or preferences. Each provider selection has an exact process-local revision, including repeated choices of the same provider. A failed legacy selection may restore only its own revision, and persisted model IDs are restored only while they still belong to that attempt. This prevents a late failure from undoing a newer user selection. Consistency rollback remains allowed after the initiating task was cancelled; ordinary cancelled mutations cannot commit.

## Tokenizer provenance and tests

The unmodified Apache-2.0 source files are in `ios/Lumen/Services/LLM/CoreAI/EmbeddingGemma2Tokenizer/` alongside the upstream license and notice:

| Source | SHA256 |
| --- | --- |
| `GemmaTokenizer.swift` | `4f5ef8a454a84937e503ba99c9f05e237347178933a442f2352a2c5db0ad87eb` |
| `ExactJSON.swift` | `5d73797991ed17635d4aedfbd6c59df19024cf2e8f3c2c7293e6457c23c430b4` |

The tokenizer preserves exact UTF8 byte keys, Unicode scalar boundaries, byte fallback, special token matching, BPE rank/tie order and leading U+FEFF string values. Replacing its parsing with canonical-equivalence String keys or a tokenizer with different normalization can change vector identity.

`CoreAITextEmbeddingPolicyTests` covers purpose prefixes, total input budget, fixed buckets, dimensions, normalization and invalid vectors, config compatibility, and exact Unicode/JSON behavior. The compact public tokenizer fixture contains 11 unchanged expected sequences selected from the pinned author's `tests/tokenizer-cases.json` (SHA256 `0797f417a0fa96270dc2f439478000fe3b3a235eb1e83b3054cf69674e323ae9`). It preserves original vocabulary IDs, all byte/special tokens, and every source BPE merge relevant to those selected inputs. The extraction provenance is embedded in the fixture. Independent fixture preparation verified all 11 sequences against the author's expected IDs. The test vocabulary is not a production tokenizer and cannot pass the importer's pinned full-tokenizer hash.

## Precision, persistence and qualification boundary

Google warns against naive float16 execution. The export implements max-absolute-value rescaling before every RMSNorm and final L2 norm to avoid squaring out-of-range activations; it also uses finite attention masks. These changes are recorded in the pinned [MODEL-ATTRIBUTION.json](https://huggingface.co/EverPaper/embeddinggemma-2-coreai/blob/305fc2a96790d9343f38038aeca211dfca56d9a5/model/MODEL-ATTRIBUTION.json) and [NOTICE](https://huggingface.co/EverPaper/embeddinggemma-2-coreai/blob/305fc2a96790d9343f38038aeca211dfca56d9a5/NOTICE). The custom export is not equivalent to blindly casting the Google checkpoint to float16. Preserve the pinned graph and validate finite unit vectors.

The author's runtime/precision evidence is from **one M5 Mac with macOS 27**, including text reference comparisons. The package declares macOS, and its graph was exported with coreai-torch 0.4.3/coreai-core 1.0.0b3. No iPhone, iOS specialization, ANE placement, thermals, memory pressure, first-load latency, retrieval quality, cancellation timing, or signed-app proof has been established here. A compute preference is not placement evidence. The graph is about 632 MB plus about 32 MB tokenizer on disk; runtime specialization and resident memory are additional and must be measured alongside the Gemma generator on target devices. The author's Mac observations include substantial first specialization latency; they are not iPhone timing predictions.

Memory content, saved document chunk text and conversations remain distinct from derived vectors. The explicit fresh-index action prepares and verifies the selected model first, then clears vector arrays, embedding metadata and both application vector caches in pages of at most 32 source rows. After reset completes, it reconstructs memories and saved document chunks from source text in pages of at most 16 embedding attempts. No previous vector or parallel index is retained, converted or reused. Model files and Apple's compilation caches are outside the reset. See [EMBEDDING_INDEX_REBUILD.md](EMBEDDING_INDEX_REBUILD.md).

The durable reset-pending flag blocks semantic reloads and embedding writes while clearing is incomplete. The reset rotates an index generation so delayed inference from before the reset cannot repopulate it. Model identity, dimension and processing version still guard every new result. A provider or generation change rejects an old continuation.

Memory enumeration is database-paged by stable creation date and UUID, with a batch-size-plus-one lookahead; it does not fetch and sort the entire collection or calculate an exact remaining count. Results expose bounded source rows scanned and `hasMore`. Additional per-ID verification reads are limited to one source row and reject edits or deletion saved in another context while inference was suspended. An oversized or invalid input is skipped and its cursor advances, preserving source text without a replacement vector. Cancellation, resource/runtime errors, identity changes and save failures stop at a retryable source.

Editing memory text clears its vector and metadata and invalidates the semantic cache. Write-budget denial is a storage failure, and a failed insertion is removed without rolling back unrelated edits. Pending captures are drained exclusively and merged by ID with the live queue after inference, preserving concurrent additions. Provider cancellation also terminates staged document replacement immediately, even when the caller task itself was not cancelled; the visible source corpus is retained. Queue diagnostics persist stable error categories rather than error descriptions.

Before qualifying this provider on a device, compile it with the actual iOS SDK, run the native reset/rebuild and SQLite UUID pagination tests, import and hash the nine-file package, compare the author's pinned text references across every exported bucket and supported MRL dimension, verify typed overflow/cancellation and query/document identity isolation, exercise resource pressure alongside Gemma chat, and measure retrieval quality, memory, latency and actual placement. Xcode compilation, XCTest execution and physical-iPhone qualification remain unverified here. Historical-schema migration and old-index retention are not acceptance requirements for this fresh-index contract. Static checks and public tokenizer fixtures do not substitute for native runs.

Model, tokenizer and EverPaper's own source are declared Apache-2.0 in the pinned primary sources; upstream notices and modification attribution are retained. The source also directs deployments to the [Gemma Prohibited Use Policy](https://ai.google.dev/gemma/prohibited_use_policy). This is a community conversion, without Google or Apple endorsement.
