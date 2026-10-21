# Memory embedding storage compatibility

`MemoryItem` gains three optional persisted fields: `embeddingFormatVersion: Int?`,
`embeddingModelIdentifier: String?`, and `embeddingDimension: Int?`. Each defaults
to `nil`, including the new initializer parameters. The existing memory ID,
content, source, creation date, kind, pin, topic, expiry, freshness, and vector
fields keep their names and types. This is an additive schema change; it does
not delete memories, clear vectors, or rewrite their provenance.

`LumenApp.makeContainer` continues to open an unversioned `Schema` containing
`Conversation`, `ChatMessage`, `MemoryItem`, `StoredModel`, `RAGChunk`, and
`Trigger`. There is no new production `SchemaMigrationPlan` in this change.
The intended upgrade uses SwiftData's inferred migration when opening an
existing store with the additive optional fields. Compatibility must be tested
against real existing stores; optional declarations alone are not evidence of a
successful upgrade. Apple's [schema migration guidance](https://developer.apple.com/videos/play/wwdc2025/291/)
describes explicit versioned schemas and migration plans for evolving models.

## Existing vectors and retrieval

A legacy vector with missing metadata has unknown identity. Its length, the
currently selected provider, the chat model, and the tokenizer do not establish
which model or text format produced it. Migration and reads leave the three
fields `nil`. No code should stamp an old vector with the current identity
without generating its replacement.

Semantic retrieval accepts only vectors with compatible format, model identity,
and dimension. Missing metadata, mismatched identity, malformed vectors, or
incompatible dimensions exclude a row from the semantic index. They do not
delete the memory or its text: lexical retrieval remains available subject to
the existing memory eligibility rules, including expiry and other filters.
Switching the generative model does not by itself identify an embedding model.

Editing memory text immediately clears its embedding identity and invalidates
the in-memory semantic cache. The source text and previous vector are retained,
but semantic retrieval excludes that vector until an explicit rebuild encodes
the edited text. A rebuild also checks that a row still belongs to its context
and that its source fields did not change while inference was suspended; a
deleted or edited row cannot receive the old in-flight result.

Re-embedding is a separate, explicit foreground operation with a bounded batch
and cancellation/resource checks. It must not run as part of app launch,
container migration, passive diagnostics, or background maintenance. A
successful update saves the replacement vector and its actual format, model
identity, and dimension together. Failed or cancelled work preserves existing
memory content and provenance and does not mark an old vector as current.
An interrupted batch can be retried by selecting remaining incompatible rows.

A new memory is reported as stored only after its context save succeeds. A disk
budget deferral or save failure removes only that operation's pending insert,
preserves unrelated context edits, and leaves a queued capture available for
retry. No whole-context rollback is used for this cleanup.
Queue draining admits one promotion pass per queue file and merges completed
capture IDs into the live queue after inference. Captures enqueued during an
await, failed captures, and cancelled/unattempted captures remain available;
overlapping drains cannot promote the same snapshot twice.

## Regression coverage and evidence

`ios/LumenTests/MemoryEmbeddingCompatibilityTests.swift` creates a real temporary
on-disk SwiftData store from a frozen, test-local pre-change `MemoryItem` shape.
It uses the same leaf entity name and all ten old persisted attributes and
defaults. It deliberately constructs an unversioned `Schema`, then opens the
same store URL with the production `MemoryItem` type, without a custom migration
plan, deleting the store, exporting/importing rows, or rewriting a fixture.
The test explicitly disables CloudKit through Apple's documented
[`ModelConfiguration` initializer](https://developer.apple.com/documentation/swiftdata/modelconfiguration/init%28_%3Aschema%3Aurl%3Aallowssave%3Acloudkitdatabase%3A%29).

The assertions cover populated/pinned, sparse/vectorless, and expired memories;
all old values and row IDs survive; new metadata is `nil` after the first
reopen. A known synthetic replacement vector and explicit metadata are saved
for one row, and a second reopen verifies persistence while untouched rows stay
unknown. Two initializer tests check legacy call compatibility and explicit
metadata assignment. These are storage tests, not provider inference or live
re-embedding evidence.

The focused fixture contains only `MemoryItem`. It does not prove upgrade
compatibility for the app's complete six-entity schema, relationships, every
historical release store, CloudKit synchronization, or downgrade to an older
binary. Before promoting an upgrade, an Apple executor must also open a copy of
a store produced by the prior shipped build with the full production schema,
verify memory and conversation preservation, reopen after saving new metadata,
and retain the producing build/commit and test evidence. Never reset or delete
the original store to make migration pass. An opening failure is a persistence
failure requiring investigation, not permission to replace the user's store.

On the Linux implementation host, SwiftData/Xcode compilation and these XCTest
cases have **not been executed**. On an Apple executor, run the repository's
stable `build-for-testing` checkpoint first, then execute the focused class:

```bash
xcodebuild -project ios/Lumen.xcodeproj \
  -scheme Lumen \
  -destination 'platform=iOS Simulator,name=Lumen Focused Test iPhone' \
  test \
  CODE_SIGNING_ALLOWED=NO \
  -only-testing:LumenTests/MemoryEmbeddingCompatibilityTests
```

A successful compile establishes compilation only. An executed passing focused
test establishes the synthetic `MemoryItem` store upgrade described above;
full-schema historical-store and physical-device validation remain separate.
