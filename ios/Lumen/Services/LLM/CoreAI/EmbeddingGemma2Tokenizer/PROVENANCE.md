These two Swift files are unmodified from EverPaper/embeddinggemma-2-coreai,
revision 305fc2a96790d9343f38038aeca211dfca56d9a5:
https://huggingface.co/EverPaper/embeddinggemma-2-coreai/tree/305fc2a96790d9343f38038aeca211dfca56d9a5/swift/Sources/EmbeddingGemma2CoreAI

GemmaTokenizer.swift SHA256 4f5ef8a454a84937e503ba99c9f05e237347178933a442f2352a2c5db0ad87eb
ExactJSON.swift SHA256 5d73797991ed17635d4aedfbd6c59df19024cf2e8f3c2c7293e6457c23c430b4

Copyright 2026 Everpaper. Apache License 2.0; see LICENSE.txt.
NOTICE.txt is the complete pinned upstream modification/third-party notice.
The original relative LICENSE references in the unmodified source headers refer
to the upstream layout. This directory retains that license as LICENSE.txt.

Only the exact byte-preserving Foundation tokenizer/parser is reused. The full
macOS Swift package, multimodal implementation and cache/temp cleanup are not
included. Model weights and the full tokenizer data are user-imported resources,
not bundled here. See docs/COREAI_TEXT_EMBEDDINGS.md for the immutable input ABI,
license/artifact identity, fixture extraction and device-validation limits.
