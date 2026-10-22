# Fresh embedding index from source content

The embedding contract is a fresh derived index. Source memories, saved document
chunk text, conversations and their provenance are retained. Previous vectors
are discarded; they are never converted, copied into another index, assigned a
new identity, or kept in a parallel compatibility index. This change introduces
no persisted schema field or model-conversion migration.

## Explicit foreground workflow

In Settings or Models, start a fresh index only after the selected local encoder
has been prepared and verified. `EmbeddingSourceRebuild` uses its own
`ModelContext` in the app's existing container so maintenance does not save
unrelated pending UI edits. It progresses through distinct phases:

1. Clear vector arrays and their format/model/dimension metadata from memories
   and saved document chunks, at most 32 source rows per explicit reset page.
   Invalidate both application vector caches. Source fields remain unchanged.
2. Once every reset page has saved, rebuild memories from source text in pages
   of at most 16 embedding attempts.
3. Rebuild saved document chunks from their retained text and source metadata,
   also at most 16 attempts per page.
4. Search only vectors produced in the new selected embedding space.

Each button action advances one page. The reset does not remove the SwiftData
store, source files, model packages, tokenizers or Apple's specialization and
compilation caches. It does not perform inference or load a model by itself.
Source content remains available to lexical retrieval under existing eligibility
rules while vector data is unavailable.

A durable reset-pending flag blocks semantic cache reloads and embedding writes
until the complete reset has saved. Cancellation, resource denial or save failure
leave that flag set. Restart the explicit fresh-index workflow after relaunch or
an abandoned reset; its continuation cursors are process-local, not a second
persistent index. A newer reset invalidates older sessions. Captures whose
inference started before reset cannot save their delayed vectors after reset,
even if the selected model identity is unchanged.

## Bounded reconstruction and identity

`MemoryStore.rebuildEmbeddings` uses SwiftData keyset pagination ordered by
`createdAt` and `id`, with a database predicate after the last completed source.
`fetchLimit` is the clamped batch size (1–32) plus one lookahead row;
`includePendingChanges` is false. It does not fetch, filter or sort the complete
collection, calculate an exact remaining count, or skip sources because an
existing vector matches. The normal UI limit is 16, so the source enumeration
reads at most 17 rows. A bounded one-row persisted-source check after inference
also rejects edits/deletion saved through another context.

Results report `scanned` for enumerated source rows including lookahead,
`attempted`, `rebuilt`, `skipped` and `hasMore`; `scanned` does not count the
separate per-ID verification reads. Continuation carries the completed source's
stable ordering keys, the selected model/format/dimension identity and the index
generation. Vector mutation or deletion of that completed source does not shift
the next page. A provider or generation change rejects the continuation instead
of silently restarting it in another space.

Oversized or invalid inputs advance the cursor and remain without replacement
vectors, preserving their source text. Cancellation, a runtime/resource error,
a changed/deleted source or a persistence failure stops before advancing the
failed source, allowing a retry. A successful save stores the vector and its
actual identity together. Source edits immediately clear their derived vectors
and metadata. An interrupted reconstruction uses the already saved new vectors;
no previous-space fallback is available.

Model identity, output dimension and processing version remain mandatory. They
prevent mixing results if selection changes during inference, and prevent
query/document comparisons across incompatible spaces. Their purpose is
integrity of the new index, not compatibility with previous vectors.

New memory insertions still report success only after saving. A denied write
budget or failed save removes only that operation's pending insertion, preserves
unrelated context changes and leaves queued captures available for retry. Queue
draining merges completed IDs into the live queue after inference, retaining
captures added concurrently and cancelled or failed work.

## Native validation still required

The regression suites include source preservation, bounded reset/rebuild pages,
identity changes, interrupted work, failed saves and reset-versus-inference
races. `EmbeddingConsumerIdentityTests` adds an on-disk SQLite test with equal
creation timestamps, deterministic UUID ordering, cursor-row deletion, vector
mutation and multiple fresh contexts. That test is intended to verify actual
SwiftData predicate/sort support; a source parser or Python guard cannot prove
that the target SDK compiles the UUID comparison or translates it correctly.

On the Linux implementation host, Xcode compilation and these XCTest cases have
**not been executed**. Use the repository's stable `build-for-testing` checkpoint,
then execute the affected tests on the Apple executor. Native database behavior,
CoreAI numerical parity, retrieval quality and physical-iPhone memory, latency,
thermal behavior and compute placement remain separate qualification steps.
Opening/migrating historical schemas and retaining an old parallel index are
outside this fresh-index acceptance contract. Do not delete source records or a
whole persistence store to make an embedding check pass.
