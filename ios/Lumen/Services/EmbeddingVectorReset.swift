import Foundation
import SwiftData

/// Explicitly discards derived vectors while leaving their source records intact.
/// The caller prepares/verifies the selected model and excludes embedding writers
/// for the entire reset phase before entering this paginated operation.
@MainActor
enum EmbeddingVectorReset {
    nonisolated enum Phase: String, Sendable, Equatable {
        case memory
        case rag
    }

    nonisolated struct Position: Sendable, Equatable {
        let createdAt: Date
        let id: UUID
    }

    nonisolated struct Cursor: Sendable, Equatable {
        let phase: Phase
        let position: Position?
    }

    nonisolated struct BatchResult: Sendable, Equatable {
        let scanned: Int
        let memoryCleared: Int
        let ragCleared: Int
        let nextCursor: Cursor?
        let complete: Bool
        let mode: String
        let diagnostic: String?
    }

    static let maximumBatchSize = 64

    /// Exactly one entity page is fetched per call, with one lookahead row.
    /// A clean operation context prevents saving or rolling back unrelated edits.
    /// The cursor advances only after the current page has durably saved.
    static func resetBatch(
        context: ModelContext,
        after cursor: Cursor? = nil,
        limit: Int = 32,
        save: ((ModelContext) throws -> Void)? = nil,
        writeBudget: DiskWriteBudget = .shared
    ) -> BatchResult {
        let current = cursor ?? Cursor(phase: .memory, position: nil)
        func stopped(_ mode: String, _ diagnostic: String) -> BatchResult {
            BatchResult(
                scanned: 0, memoryCleared: 0, ragCleared: 0,
                nextCursor: current, complete: false, mode: mode, diagnostic: diagnostic
            )
        }
        guard !Task.isCancelled else { return stopped("cancelled", "cancelled") }
        guard !context.hasChanges else { return stopped("failed", "reset_context_has_unsaved_changes") }
        guard ResourceBudgetGate.budgetDenialReason(policy: .foregroundInteractive, reason: "embedding.vectorReset") == nil else {
            return stopped("skipped", "foreground_resource_budget_required")
        }
        let pageSize = min(maximumBatchSize, max(1, limit))
        switch current.phase {
        case .memory:
            return resetMemoryPage(context: context, cursor: current, limit: pageSize, save: save, writeBudget: writeBudget)
        case .rag:
            return resetRAGPage(context: context, cursor: current, limit: pageSize, save: save, writeBudget: writeBudget)
        }
    }

    static func memoryDescriptor(after position: Position?, limit: Int) -> FetchDescriptor<MemoryItem> {
        let predicate: Predicate<MemoryItem>?
        if let position {
            let date = position.createdAt
            let id = position.id
            predicate = #Predicate<MemoryItem> { item in
                item.createdAt > date || (item.createdAt == date && item.id > id)
            }
        } else {
            predicate = nil
        }
        var descriptor = FetchDescriptor<MemoryItem>(
            predicate: predicate,
            sortBy: [SortDescriptor(\MemoryItem.createdAt), SortDescriptor(\MemoryItem.id)]
        )
        descriptor.includePendingChanges = false
        descriptor.fetchLimit = min(maximumBatchSize, max(1, limit)) + 1
        return descriptor
    }

    static func ragDescriptor(after position: Position?, limit: Int) -> FetchDescriptor<RAGChunk> {
        let predicate: Predicate<RAGChunk>?
        if let position {
            let date = position.createdAt
            let id = position.id
            predicate = #Predicate<RAGChunk> { chunk in
                chunk.createdAt > date || (chunk.createdAt == date && chunk.id > id)
            }
        } else {
            predicate = nil
        }
        var descriptor = FetchDescriptor<RAGChunk>(
            predicate: predicate,
            sortBy: [SortDescriptor(\RAGChunk.createdAt), SortDescriptor(\RAGChunk.id)]
        )
        // Staging rows are derived data as well. Clear their vectors without
        // deleting or promoting them; rebuild separately excludes staging text.
        descriptor.includePendingChanges = false
        descriptor.fetchLimit = min(maximumBatchSize, max(1, limit)) + 1
        return descriptor
    }

    private static func resetMemoryPage(
        context: ModelContext, cursor: Cursor, limit: Int,
        save: ((ModelContext) throws -> Void)?, writeBudget: DiskWriteBudget
    ) -> BatchResult {
        let fetched: [MemoryItem]
        do {
            fetched = try context.fetch(memoryDescriptor(after: cursor.position, limit: limit))
        } catch {
            return failed(cursor: cursor, diagnostic: "memory_reset_fetch_failed:\(RuntimeMetricErrorSanitizer.code(for: error))")
        }
        let page = Array(fetched.prefix(limit))
        let snapshots = page.map { MemorySnapshot(item: $0) }
        let changed = snapshots.filter(\.hasDerivedData)
        let outcome = persistPage(
            context: context, count: changed.count,
            estimatedBytes: estimatedBytes(payloads: changed.map(\.estimatedBytes)),
            category: .memory, save: save, writeBudget: writeBudget,
            mutate: { changed.forEach { $0.clear() } },
            restore: { changed.forEach { $0.restore() } }
        )
        if let outcome { return failed(cursor: cursor, mode: outcome.mode, diagnostic: outcome.diagnostic) }
        let next: Cursor
        if fetched.count > limit, let last = page.last {
            next = Cursor(phase: .memory, position: Position(createdAt: last.createdAt, id: last.id))
        } else {
            next = Cursor(phase: .rag, position: nil)
        }
        invalidateCaches()
        return BatchResult(scanned: fetched.count, memoryCleared: changed.count, ragCleared: 0, nextCursor: next, complete: false, mode: "resetting", diagnostic: nil)
    }

    private static func resetRAGPage(
        context: ModelContext, cursor: Cursor, limit: Int,
        save: ((ModelContext) throws -> Void)?, writeBudget: DiskWriteBudget
    ) -> BatchResult {
        let fetched: [RAGChunk]
        do {
            fetched = try context.fetch(ragDescriptor(after: cursor.position, limit: limit))
        } catch {
            return failed(cursor: cursor, diagnostic: "rag_reset_fetch_failed:\(RuntimeMetricErrorSanitizer.code(for: error))")
        }
        let page = Array(fetched.prefix(limit))
        let snapshots = page.map { RAGSnapshot(chunk: $0) }
        let changed = snapshots.filter(\.hasDerivedData)
        let outcome = persistPage(
            context: context, count: changed.count,
            estimatedBytes: estimatedBytes(payloads: changed.map(\.estimatedBytes)),
            category: .rag, save: save, writeBudget: writeBudget,
            mutate: { changed.forEach { $0.clear() } },
            restore: { changed.forEach { $0.restore() } }
        )
        if let outcome { return failed(cursor: cursor, mode: outcome.mode, diagnostic: outcome.diagnostic) }
        let next: Cursor?
        if fetched.count > limit, let last = page.last {
            next = Cursor(phase: .rag, position: Position(createdAt: last.createdAt, id: last.id))
        } else {
            next = nil
        }
        invalidateCaches()
        return BatchResult(scanned: fetched.count, memoryCleared: 0, ragCleared: changed.count, nextCursor: next, complete: next == nil, mode: next == nil ? "reset" : "resetting", diagnostic: nil)
    }

    private static func persistPage(
        context: ModelContext, count: Int, estimatedBytes: Int, category: DiskWriteCategory,
        save: ((ModelContext) throws -> Void)?, writeBudget: DiskWriteBudget,
        mutate: () -> Void, restore: () -> Void
    ) -> (mode: String, diagnostic: String)? {
        guard !Task.isCancelled else { return ("cancelled", "cancelled") }
        guard ResourceBudgetGate.budgetDenialReason(policy: .foregroundInteractive, reason: "embedding.vectorReset") == nil else {
            return ("skipped", "foreground_resource_budget_required")
        }
        guard count > 0 else { return nil }
        guard let reservation = writeBudget.reserveWrite(bytes: estimatedBytes, category: category) else {
            return ("skipped", "disk_write_budget_denied")
        }
        guard !Task.isCancelled else {
            writeBudget.releaseReservedWrite(reservation)
            return ("cancelled", "cancelled")
        }
        mutate()
        guard !Task.isCancelled else {
            restore()
            writeBudget.releaseReservedWrite(reservation)
            return ("cancelled", "cancelled")
        }
        do {
            if let save { try save(context) } else { try context.save() }
            writeBudget.commitReservedWrite(reservation)
            return nil
        } catch {
            restore()
            writeBudget.releaseReservedWrite(reservation)
            return ("failed", "vector_reset_persist_failed:\(RuntimeMetricErrorSanitizer.code(for: error))")
        }
    }

    private static func failed(cursor: Cursor, mode: String = "failed", diagnostic: String) -> BatchResult {
        BatchResult(scanned: 0, memoryCleared: 0, ragCleared: 0, nextCursor: cursor, complete: false, mode: mode, diagnostic: diagnostic)
    }

    private static func invalidateCaches() {
        MemoryVectorIndex.shared.invalidate()
        RAGVectorIndex.shared.invalidate()
    }

    private static func estimatedBytes(payloads: [Int]) -> Int {
        payloads.reduce(4 * 1024) { total, payload in
            let sum = total.addingReportingOverflow(payload)
            return sum.overflow ? Int.max : sum.partialValue
        }
    }

    private struct MemorySnapshot {
        let item: MemoryItem
        let vector: [Double]
        let format: Int?
        let model: String?
        let dimension: Int?

        init(item: MemoryItem) {
            self.item = item
            vector = item.embedding
            format = item.embeddingFormatVersion
            model = item.embeddingModelIdentifier
            dimension = item.embeddingDimension
        }

        var hasDerivedData: Bool { !vector.isEmpty || format != nil || model != nil || dimension != nil }
        var estimatedBytes: Int { 512 + item.content.utf8.count + vector.count * MemoryLayout<Double>.stride + (model?.utf8.count ?? 0) }

        func clear() {
            item.embedding = []
            item.embeddingFormatVersion = nil
            item.embeddingModelIdentifier = nil
            item.embeddingDimension = nil
        }

        func restore() {
            item.embedding = vector
            item.embeddingFormatVersion = format
            item.embeddingModelIdentifier = model
            item.embeddingDimension = dimension
        }
    }

    private struct RAGSnapshot {
        let chunk: RAGChunk
        let vector: [Double]
        let format: Int
        let model: String
        let dimension: Int

        init(chunk: RAGChunk) {
            self.chunk = chunk
            vector = chunk.embedding
            format = chunk.embeddingFormatVersion
            model = chunk.embeddingModelIdentifier
            dimension = chunk.embeddingDimension
        }

        var hasDerivedData: Bool { !vector.isEmpty || format != 0 || !model.isEmpty || dimension != 0 }
        var estimatedBytes: Int { 512 + chunk.content.utf8.count + vector.count * MemoryLayout<Double>.stride + model.utf8.count }

        func clear() {
            chunk.embedding = []
            chunk.embeddingFormatVersion = 0
            chunk.embeddingModelIdentifier = ""
            chunk.embeddingDimension = 0
        }

        func restore() {
            chunk.embedding = vector
            chunk.embeddingFormatVersion = format
            chunk.embeddingModelIdentifier = model
            chunk.embeddingDimension = dimension
        }
    }
}
