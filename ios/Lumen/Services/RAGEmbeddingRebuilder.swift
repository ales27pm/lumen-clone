import Foundation
import SwiftData
import SwiftUI

/// Rebuilds the persisted source text (including notes and extracted photo text)
/// without deleting or reimporting source rows or constructing a parallel index.
@MainActor
enum RAGEmbeddingRebuilder {
    struct Cursor: Sendable, Equatable {
        let createdAt: Date
        let id: UUID
        let metadata: RAGEmbeddingIndexMetadata
        let generation: UUID
    }

    struct BatchResult: Sendable, Equatable {
        let scanned: Int
        let rebuilt: Int
        let skipped: Int
        let hasMore: Bool
        let nextCursor: Cursor?
        let diagnostic: String?
    }

    static func pageDescriptor(after cursor: Cursor?, limit: Int) -> FetchDescriptor<RAGChunk> {
        let prefix = RAGChunk.replacementStagingSourceTypePrefix
        let predicate: Predicate<RAGChunk>
        if let cursor {
            let date = cursor.createdAt
            let id = cursor.id
            predicate = #Predicate<RAGChunk> {
                !$0.sourceType.starts(with: prefix)
                    && ($0.createdAt > date || ($0.createdAt == date && $0.id > id))
            }
        } else {
            predicate = #Predicate<RAGChunk> { !$0.sourceType.starts(with: prefix) }
        }
        var descriptor = FetchDescriptor<RAGChunk>(predicate: predicate, sortBy: [
            SortDescriptor(\RAGChunk.createdAt), SortDescriptor(\RAGChunk.id)
        ])
        descriptor.includePendingChanges = false
        descriptor.fetchLimit = min(32, max(1, limit)) + 1
        return descriptor
    }

    static func rebuildBatch(
        context: ModelContext,
        matching metadata: RAGEmbeddingIndexMetadata,
        after cursor: Cursor? = nil,
        limit: Int = 16,
        embed: (EmbeddingRequest) async throws -> EmbeddingRuntimeResult = { request in
            try await AssistantKernel.runEmbeddingWithIdentity(
                text: request.text, purpose: request.purpose, dimensions: request.dimensions
            )
        },
        save: ((ModelContext) throws -> Void)? = nil,
        writeBudget: DiskWriteBudget = .shared
    ) async -> BatchResult {
        let generation = EmbeddingIndexLifecycle.captureWrite()
        var position = cursor
        var rebuilt = 0
        var skipped = 0
        var scanned = 0
        func stopped(_ diagnostic: String) -> BatchResult {
            BatchResult(scanned: scanned, rebuilt: rebuilt, skipped: skipped,
                hasMore: true, nextCursor: position, diagnostic: diagnostic)
        }
        guard let generation, EmbeddingIndexLifecycle.permitsWrite(generation) else {
            return stopped("embedding_index_reset_pending")
        }
        guard cursor == nil || (cursor?.metadata == metadata && cursor?.generation == generation),
              metadata.formatVersion > 0, metadata.dimension > 0, !metadata.modelIdentifier.isEmpty,
              metadata.modelIdentifier != RAGEmbeddingMetadata.unidentifiedModelIdentifier else {
            return stopped("embedding_rebuild_session_changed")
        }
        guard !context.hasChanges else { return stopped("embedding_rebuild_unsaved_changes") }
        guard !Task.isCancelled else { return stopped("cancelled") }
        guard foregroundAllowed() else { return stopped("foreground_embedding_budget_required") }
        let pageLimit = min(32, max(1, limit))
        let rows: [RAGChunk]
        do { rows = try context.fetch(pageDescriptor(after: cursor, limit: pageLimit)) }
        catch { return stopped("rag_fetch_failed:\(RuntimeMetricErrorSanitizer.code(for: error))") }
        scanned = rows.count
        for row in rows.prefix(pageLimit) {
            guard !Task.isCancelled else { return stopped("cancelled") }
            guard foregroundAllowed() else { return stopped("foreground_embedding_budget_required") }
            guard EmbeddingIndexLifecycle.permitsWrite(generation) else { return stopped("embedding_rebuild_session_changed") }
            let source = Source(row)
            let next = Cursor(createdAt: row.createdAt, id: row.id, metadata: metadata, generation: generation)
            let result: EmbeddingRuntimeResult
            do {
                result = try await embed(EmbeddingRequest(
                    text: SemanticEmbeddingText.documentBody(
                        content: source.content, sourceName: source.name,
                        sourceType: source.type, chunkIndex: source.chunkIndex
                    ),
                    purpose: .retrievalDocument(title: source.name), dimensions: metadata.dimension
                ))
            } catch is CancellationError { return stopped("cancelled") }
            catch let error as CoreAITextEmbeddingError where error == .invalidRequest || error == .inputTooLong {
                guard !Task.isCancelled, foregroundAllowed(), EmbeddingIndexLifecycle.permitsWrite(generation),
                      source.matches(row, in: context) else {
                    return stopped("embedding_rebuild_session_changed")
                }
                if let diagnostic = source.persistedDiagnostic(in: context.container) { return stopped(diagnostic) }
                skipped += 1
                position = next
                continue
            } catch { return stopped("embedding_failed:\(RuntimeMetricErrorSanitizer.code(for: error))") }
            guard !Task.isCancelled else { return stopped("cancelled") }
            guard foregroundAllowed() else { return stopped("foreground_embedding_budget_required") }
            guard EmbeddingIndexLifecycle.permitsWrite(generation) else { return stopped("embedding_rebuild_session_changed") }
            guard source.matches(row, in: context) else {
                return stopped("rag_source_changed_during_rebuild")
            }
            if let diagnostic = source.persistedDiagnostic(in: context.container) { return stopped(diagnostic) }
            guard !context.hasChanges else { return stopped("embedding_rebuild_unsaved_changes") }
            guard result.modelIdentifier == metadata.modelIdentifier, result.formatVersion == metadata.formatVersion,
                  result.vector.count == metadata.dimension, result.vector.allSatisfy(\.isFinite) else {
                return stopped("embedding_identity_changed_during_rebuild")
            }
            guard let reservation = writeBudget.reserveWrite(bytes: 64 * 1024, category: .rag) else {
                return stopped("disk_write_budget_denied")
            }
            let previous = (row.embedding, row.embeddingFormatVersion, row.embeddingModelIdentifier, row.embeddingDimension)
            row.embedding = result.vector
            row.embeddingFormatVersion = metadata.formatVersion
            row.embeddingModelIdentifier = metadata.modelIdentifier
            row.embeddingDimension = metadata.dimension
            do {
                if let save { try save(context) } else { try context.save() }
                writeBudget.commitReservedWrite(reservation)
            } catch {
                writeBudget.releaseReservedWrite(reservation)
                (row.embedding, row.embeddingFormatVersion, row.embeddingModelIdentifier, row.embeddingDimension) = previous
                return stopped("persist_failed:\(RuntimeMetricErrorSanitizer.code(for: error))")
            }
            RAGVectorIndex.shared.invalidate()
            rebuilt += 1
            position = next
        }
        let more = rows.count > pageLimit
        return BatchResult(scanned: scanned, rebuilt: rebuilt, skipped: skipped,
            hasMore: more, nextCursor: more ? position : nil, diagnostic: nil)
    }

    private static func foregroundAllowed() -> Bool {
        ResourceBudgetGate.diagnosticSnapshot().scenePhase == .active
            && ResourceBudgetGate.budgetDenialReason(policy: .embedding, reason: "rag.rebuild") == nil
    }

    private struct Source {
        let id: UUID
        let date: Date
        let content: String
        let type: String
        let name: String
        let ref: String?
        let chunkIndex: Int

        init(_ row: RAGChunk) {
            id = row.id; date = row.createdAt; content = row.content
            type = row.sourceType; name = row.sourceName; ref = row.sourceRef; chunkIndex = row.chunkIndex
        }

        func persistedDiagnostic(in container: ModelContainer) -> String? {
            let readContext = ModelContext(container)
            readContext.autosaveEnabled = false
            let sourceID = id
            var descriptor = FetchDescriptor<RAGChunk>(predicate: #Predicate { $0.id == sourceID })
            descriptor.fetchLimit = 1
            descriptor.includePendingChanges = false
            do {
                guard let row = try readContext.fetch(descriptor).first else {
                    return "rag_source_changed_during_rebuild"
                }
                return matches(row, in: readContext) ? nil : "rag_source_changed_during_rebuild"
            } catch {
                return "rag_source_fetch_failed:\(RuntimeMetricErrorSanitizer.code(for: error))"
            }
        }

        func matches(_ row: RAGChunk, in context: ModelContext) -> Bool {
            row.modelContext === context && !row.isDeleted && row.id == id && row.createdAt == date
                && row.content == content && row.sourceType == type && row.sourceName == name
                && row.sourceRef == ref && row.chunkIndex == chunkIndex
        }
    }
}
