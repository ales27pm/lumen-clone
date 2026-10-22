import Foundation
import SwiftData
import SwiftUI
import XCTest
@testable import Lumen

@MainActor
final class EmbeddingVectorResetTests: XCTestCase {
    private let oldMetadata = RAGEmbeddingIndexMetadata(formatVersion: 3, modelIdentifier: "test:discarded-space", dimension: 2)

    override func tearDown() async throws {
        MemoryVectorIndex.shared.invalidate()
        RAGVectorIndex.shared.invalidate()
        ResourceBudgetGate.testSnapshotOverride = nil
        try await super.tearDown()
    }

    func testResetPagesPreserveEverySourceFieldAndInvalidateBothCaches() throws {
        allowForeground()
        let container = try makeContainer()
        let context = ModelContext(container)
        context.autosaveEnabled = false
        let memories = (1...5).reversed().map { memory(index: $0) }
        let chunks = (1...4).reversed().map { chunk(index: $0) }
        memories.forEach { context.insert($0) }
        chunks.forEach { context.insert($0) }
        try context.save()
        let memorySources = memories.map(MemorySource.init)
        let chunkSources = chunks.map(ChunkSource.init)
        XCTAssertEqual(MemoryVectorIndex.shared.ensureLoaded(context: context, metadata: oldMetadata).loadedCount, 5)
        XCTAssertEqual(RAGVectorIndex.shared.ensureLoaded(context: context, metadata: oldMetadata).loadedCount, 4)

        var cursor: EmbeddingVectorReset.Cursor?
        var clearedMemories = 0
        var clearedChunks = 0
        var complete = false
        for _ in 0..<10 {
            let result = EmbeddingVectorReset.resetBatch(context: context, after: cursor, limit: 2, writeBudget: generousBudget())
            XCTAssertNil(result.diagnostic)
            XCTAssertLessThanOrEqual(result.scanned, 3)
            clearedMemories += result.memoryCleared
            clearedChunks += result.ragCleared
            XCTAssertTrue(MemoryVectorIndex.shared.search(query: [1, 0], topK: 10).isEmpty)
            XCTAssertEqual(RAGVectorIndex.shared.count, 0)
            cursor = result.nextCursor
            if result.complete { complete = true; break }
        }
        XCTAssertTrue(complete)
        XCTAssertNil(cursor)
        XCTAssertEqual(clearedMemories, 5)
        XCTAssertEqual(clearedChunks, 4)
        XCTAssertEqual(try context.fetchCount(FetchDescriptor<MemoryItem>()), 5)
        XCTAssertEqual(try context.fetchCount(FetchDescriptor<RAGChunk>()), 4)
        for item in memories {
            XCTAssertTrue(item.embedding.isEmpty)
            XCTAssertNil(item.embeddingFormatVersion)
            XCTAssertNil(item.embeddingModelIdentifier)
            XCTAssertNil(item.embeddingDimension)
            XCTAssertEqual(MemorySource(item), memorySources.first { $0.id == item.id })
        }
        for item in chunks {
            XCTAssertTrue(item.embedding.isEmpty)
            XCTAssertEqual(item.embeddingFormatVersion, 0)
            XCTAssertEqual(item.embeddingModelIdentifier, "")
            XCTAssertEqual(item.embeddingDimension, 0)
            XCTAssertEqual(ChunkSource(item), chunkSources.first { $0.id == item.id })
        }

        // Only newly computed vectors are eligible after the reset. Matching the
        // old dimensionality alone does not make an old embedding reusable.
        let metadata = RAGEmbeddingIndexMetadata(formatVersion: 4, modelIdentifier: "test:fresh-space", dimension: 2)
        let freshMemory = try XCTUnwrap(memories.first)
        freshMemory.embedding = [0, 1]
        freshMemory.embeddingFormatVersion = metadata.formatVersion
        freshMemory.embeddingModelIdentifier = metadata.modelIdentifier
        freshMemory.embeddingDimension = metadata.dimension
        let freshChunk = try XCTUnwrap(chunks.first)
        freshChunk.embedding = [0, 1]
        freshChunk.embeddingFormatVersion = metadata.formatVersion
        freshChunk.embeddingModelIdentifier = metadata.modelIdentifier
        freshChunk.embeddingDimension = metadata.dimension
        try context.save()
        XCTAssertEqual(MemoryVectorIndex.shared.ensureLoaded(context: context, metadata: metadata).loadedCount, 1)
        XCTAssertEqual(RAGVectorIndex.shared.ensureLoaded(context: context, metadata: metadata).loadedCount, 1)
        XCTAssertEqual(MemoryVectorIndex.shared.search(query: [0, 1], topK: 5).map(\.id), [freshMemory.persistentModelID])
        XCTAssertEqual(RAGVectorIndex.shared.search(query: [0, 1], topK: 5).map(\.id), [freshChunk.persistentModelID])
    }

    func testFileBackedEqualTimestampPagesResetAllRowsAndRetainSourceAndModelFiles() throws {
        allowForeground()
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("EmbeddingReset-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let sourceURL = directory.appendingPathComponent("source.txt")
        let modelURL = directory.appendingPathComponent("model.aimodel")
        let compiledURL = directory.appendingPathComponent("apple-specialization-cache.fixture")
        let payload = Data("synthetic source bytes".utf8)
        try payload.write(to: sourceURL)
        try payload.write(to: modelURL)
        try payload.write(to: compiledURL)
        let schema = Schema([MemoryItem.self, RAGChunk.self])
        let config = ModelConfiguration(schema: schema, url: directory.appendingPathComponent("sources.store"))
        try autoreleasepool {
            let container = try ModelContainer(for: schema, configurations: [config])
            let context = ModelContext(container)
            context.autosaveEnabled = false
            for index in (1...5).reversed() {
                context.insert(memory(index: index))
                let item = chunk(index: index)
                item.sourceRef = sourceURL.path
                context.insert(item)
            }
            try context.save()
            var cursor: EmbeddingVectorReset.Cursor?
            var complete = false
            for _ in 0..<12 {
                let result = EmbeddingVectorReset.resetBatch(context: context, after: cursor, limit: 2, writeBudget: generousBudget())
                XCTAssertNil(result.diagnostic)
                cursor = result.nextCursor
                if result.complete { complete = true; break }
            }
            XCTAssertTrue(complete)
        }
        try autoreleasepool {
            let container = try ModelContainer(for: schema, configurations: [config])
            let context = ModelContext(container)
            let memories = try context.fetch(FetchDescriptor<MemoryItem>())
            let chunks = try context.fetch(FetchDescriptor<RAGChunk>())
            XCTAssertEqual(memories.count, 5)
            XCTAssertEqual(chunks.count, 5)
            XCTAssertTrue(memories.allSatisfy { $0.embedding.isEmpty && $0.embeddingModelIdentifier == nil })
            XCTAssertTrue(chunks.allSatisfy { $0.embedding.isEmpty && $0.embeddingModelIdentifier.isEmpty && $0.sourceRef == sourceURL.path })
        }
        XCTAssertEqual(try Data(contentsOf: sourceURL), payload)
        XCTAssertEqual(try Data(contentsOf: modelURL), payload)
        XCTAssertEqual(try Data(contentsOf: compiledURL), payload)
    }

    func testSaveFailureRestoresVectorsAndIdentityAndReleasesWriteReservation() throws {
        allowForeground()
        let context = ModelContext(try makeContainer())
        context.autosaveEnabled = false
        let item = memory(index: 1)
        context.insert(item)
        try context.save()
        let budget = generousBudget()
        enum SyntheticFailure: Error { case save }
        var saves = 0
        let result = EmbeddingVectorReset.resetBatch(context: context, limit: 1, save: { _ in
            saves += 1
            XCTAssertTrue(item.embedding.isEmpty)
            throw SyntheticFailure.save
        }, writeBudget: budget)
        XCTAssertEqual(saves, 1)
        XCTAssertEqual(result.mode, "failed")
        XCTAssertFalse(result.complete)
        XCTAssertEqual(result.nextCursor, EmbeddingVectorReset.Cursor(phase: .memory, position: nil))
        XCTAssertEqual(item.embedding, [1, 0])
        XCTAssertEqual(item.embeddingModelIdentifier, oldMetadata.modelIdentifier)
        XCTAssertEqual(item.embeddingFormatVersion, oldMetadata.formatVersion)
        XCTAssertEqual(item.embeddingDimension, oldMetadata.dimension)
        XCTAssertEqual(budget.snapshot().bytes1Minute, 0)
        XCTAssertNotNil(budget.reserveWrite(bytes: 2_000_000, category: .memory))
    }

    func testCancelledBatchDoesNotMutateSourcesOrAdvanceCursor() async throws {
        allowForeground()
        let context = ModelContext(try makeContainer())
        context.autosaveEnabled = false
        let item = memory(index: 1)
        context.insert(item)
        try context.save()
        let result = await Task { @MainActor in
            withUnsafeCurrentTask { $0?.cancel() }
            return EmbeddingVectorReset.resetBatch(context: context, writeBudget: generousBudget())
        }.value
        XCTAssertEqual(result.mode, "cancelled")
        XCTAssertFalse(result.complete)
        XCTAssertEqual(result.scanned, 0)
        XCTAssertEqual(item.embedding, [1, 0])
        XCTAssertEqual(item.content, "Synthetic source 1")
    }

    func testDirtyContextAndDiskOrForegroundDenialCannotClearVectors() throws {
        allowForeground()
        let context = ModelContext(try makeContainer())
        context.autosaveEnabled = false
        let item = memory(index: 1)
        context.insert(item)
        let dirty = EmbeddingVectorReset.resetBatch(context: context, writeBudget: generousBudget())
        XCTAssertEqual(dirty.diagnostic, "reset_context_has_unsaved_changes")
        XCTAssertEqual(item.embedding, [1, 0])
        try context.save()
        let denied = EmbeddingVectorReset.resetBatch(context: context, writeBudget: DiskWriteBudget(oneMinuteLimit: 0, fifteenMinuteLimit: 0, dayLimit: 0))
        XCTAssertEqual(denied.diagnostic, "disk_write_budget_denied")
        XCTAssertEqual(item.embedding, [1, 0])
        ResourceBudgetGate.testSnapshotOverride = .init(scenePhase: .background, lowPowerModeEnabled: false, thermalState: .nominal, recentMemoryWarningCount: 0, lastMemoryWarningAt: nil)
        let background = EmbeddingVectorReset.resetBatch(context: context, writeBudget: generousBudget())
        XCTAssertEqual(background.diagnostic, "foreground_resource_budget_required")
        XCTAssertEqual(item.embedding, [1, 0])
    }

    func testDescriptorsBoundEveryFetchIncludingLookaheadAndSortByStablePair() {
        let position = EmbeddingVectorReset.Position(createdAt: Date(timeIntervalSince1970: 100), id: UUID())
        let memories = EmbeddingVectorReset.memoryDescriptor(after: position, limit: 16)
        let chunks = EmbeddingVectorReset.ragDescriptor(after: position, limit: .max)
        XCTAssertEqual(memories.fetchLimit, 17)
        XCTAssertEqual(memories.fetchOffset, 0)
        XCTAssertEqual(memories.sortBy, [SortDescriptor(\MemoryItem.createdAt), SortDescriptor(\MemoryItem.id)])
        XCTAssertEqual(chunks.fetchLimit, EmbeddingVectorReset.maximumBatchSize + 1)
        XCTAssertEqual(chunks.fetchOffset, 0)
        XCTAssertEqual(chunks.sortBy, [SortDescriptor(\RAGChunk.createdAt), SortDescriptor(\RAGChunk.id)])
        XCTAssertNotNil(memories.predicate)
        XCTAssertNotNil(chunks.predicate)
    }

    private func makeContainer() throws -> ModelContainer {
        try ModelContainer(for: MemoryItem.self, RAGChunk.self, configurations: ModelConfiguration(isStoredInMemoryOnly: true))
    }

    private func generousBudget() -> DiskWriteBudget {
        DiskWriteBudget(oneMinuteLimit: 2_000_000, fifteenMinuteLimit: 20_000_000, dayLimit: 100_000_000)
    }

    private func allowForeground() {
        ResourceBudgetGate.testSnapshotOverride = .init(scenePhase: .active, lowPowerModeEnabled: false, thermalState: .nominal, recentMemoryWarningCount: 0, lastMemoryWarningAt: nil)
    }

    private func memory(index: Int) -> MemoryItem {
        let item = MemoryItem(content: "Synthetic source \(index)", kind: .project, source: "manual", embedding: [1, 0], topic: "Synthetic topic", expiresAt: Date(timeIntervalSince1970: 900), freshnessClass: .durable, embeddingFormatVersion: oldMetadata.formatVersion, embeddingModelIdentifier: oldMetadata.modelIdentifier, embeddingDimension: oldMetadata.dimension)
        item.id = uuid(index)
        item.createdAt = Date(timeIntervalSince1970: 100)
        item.isPinned = true
        return item
    }

    private func chunk(index: Int) -> RAGChunk {
        let item = RAGChunk(content: "Synthetic chunk \(index)", sourceType: .note, sourceName: "Synthetic note", sourceRef: "source-\(index)", chunkIndex: index, embedding: [1, 0], embeddingFormatVersion: oldMetadata.formatVersion, embeddingModelIdentifier: oldMetadata.modelIdentifier, embeddingDimension: oldMetadata.dimension)
        item.id = uuid(index)
        item.createdAt = Date(timeIntervalSince1970: 100)
        return item
    }

    private func uuid(_ index: Int) -> UUID {
        UUID(uuidString: "00000000-0000-0000-0000-\(String(format: "%012d", index))")!
    }

    private struct MemorySource: Equatable {
        let id: UUID
        let content: String
        let source: String
        let createdAt: Date
        let kind: String
        let isPinned: Bool
        let topic: String?
        let expiresAt: Date?
        let freshnessClass: String?

        init(_ item: MemoryItem) {
            id = item.id; content = item.content; source = item.source; createdAt = item.createdAt
            kind = item.kind; isPinned = item.isPinned; topic = item.topic
            expiresAt = item.expiresAt; freshnessClass = item.freshnessClass
        }
    }

    private struct ChunkSource: Equatable {
        let id: UUID
        let content: String
        let sourceType: String
        let sourceName: String
        let sourceRef: String?
        let chunkIndex: Int
        let createdAt: Date

        init(_ item: RAGChunk) {
            id = item.id; content = item.content; sourceType = item.sourceType; sourceName = item.sourceName
            sourceRef = item.sourceRef; chunkIndex = item.chunkIndex; createdAt = item.createdAt
        }
    }
}
