import Foundation
import SwiftData

/// Explicit foreground maintenance: verify the selected encoder first, discard
/// derived vectors in pages, then encode source rows in pages. No old index is
/// retained, copied, converted, or assigned a new identity.
@MainActor
final class EmbeddingSourceRebuild {
    enum Phase: Equatable { case reset, memory, rag, complete }
    private(set) var phase = Phase.reset
    private(set) var message = ""
    let metadata: RAGEmbeddingIndexMetadata
    private let container: ModelContainer
    private let resetGeneration: UUID
    private var resetCursor: EmbeddingVectorReset.Cursor?
    private var memoryCursor: MemoryStore.RebuildCursor?
    private var ragCursor: RAGEmbeddingRebuilder.Cursor?

    /// Call only after foreground preparation has returned valid metadata. The
    /// dedicated context shares the app's container and never saves UI edits.
    init(container: ModelContainer, prepared metadata: RAGEmbeddingIndexMetadata) {
        self.metadata = metadata
        self.container = container
        resetGeneration = EmbeddingIndexLifecycle.beginReset()
    }

    func advance(matching prepared: RAGEmbeddingIndexMetadata) async {
        guard prepared == metadata else {
            message = "The selected embedding space changed. Start a fresh index with the selected model."
            return
        }
        guard !Task.isCancelled else { message = "Paused. Source contents are preserved."; return }
        guard phase == .reset || EmbeddingIndexLifecycle.captureWrite() == resetGeneration else {
            message = "Another reset replaced this session. Start a fresh index."
            return
        }
        // A fresh context for each page sees persisted source edits and cannot
        // retain dirty restored fields after a failed save or alter UI edits.
        let context = ModelContext(container)
        context.autosaveEnabled = false
        switch phase {
        case .reset:
            guard EmbeddingIndexLifecycle.ownsReset(resetGeneration) else {
                message = "Another reset replaced this session. Start a fresh index."
                return
            }
            let result = EmbeddingVectorReset.resetBatch(context: context, after: resetCursor, limit: 32)
            resetCursor = result.nextCursor
            message = "Cleared vectors for \(result.memoryCleared) memories and \(result.ragCleared) document chunks; read \(result.scanned) source rows."
            if result.complete, EmbeddingIndexLifecycle.finishReset(resetGeneration) {
                phase = .memory
                message += " Reset complete. Continue to rebuild from source text."
            } else if result.diagnostic != nil {
                message += " Reset paused. Semantic search remains paused until reset completes."
            }
        case .memory:
            let result = await MemoryStore.rebuildEmbeddings(
                context: context, matching: metadata, limit: 16, after: memoryCursor
            )
            memoryCursor = result.nextCursor
            message = "Rebuilt \(result.rebuilt) memories; skipped \(result.skipped); read \(result.scanned) source rows."
            let onlySkipped = result.diagnostic == "memory_inputs_skipped:\(result.skipped)"
            if result.diagnostic != nil && !onlySkipped {
                message += " Batch paused. Source contents are preserved."
            } else if !result.hasMore {
                phase = .rag
                message += " Memory scan complete. Continue with document source text."
            }
        case .rag:
            let result = await RAGEmbeddingRebuilder.rebuildBatch(
                context: context, matching: metadata, after: ragCursor, limit: 16
            )
            ragCursor = result.nextCursor
            message = "Rebuilt \(result.rebuilt) document chunks; skipped \(result.skipped); read \(result.scanned) source rows."
            if result.diagnostic != nil {
                message += " Batch paused. Source contents are preserved."
            } else if !result.hasMore {
                phase = .complete
                message += " Source scan complete. Skipped inputs have no replacement vectors."
            }
        case .complete:
            message = "Source scan complete. Only the new embedding space is searchable."
        }
    }
}
