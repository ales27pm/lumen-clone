import Foundation
import SwiftData
import SwiftUI
import XCTest
@testable import Lumen

@MainActor
final class EmbeddingConsumerIdentityTests: XCTestCase {
    private let metadata = RAGEmbeddingIndexMetadata(formatVersion: 4, modelIdentifier: "coreai:test-space:dimension2", dimension: 2)

    override func tearDown() async throws {
        MemoryVectorIndex.shared.invalidate()
        RAGVectorIndex.shared.invalidate()
        ResourceBudgetGate.testSnapshotOverride = nil
        try await super.tearDown()
    }

    func testRawBodiesPreserveLegacyProductionInputWithoutNestedTaskPrefixes() {
        let body = SemanticEmbeddingText.documentBody(content: "line one\nline two", sourceName: "Notes", sourceType: "note", chunkIndex: 2)
        XCTAssertEqual("search_document:\n" + body, SemanticEmbeddingText.document(content: "line one\nline two", sourceName: "Notes", sourceType: "note", chunkIndex: 2))
        let memory = SemanticEmbeddingText.memoryDocumentBody(content: "Concise answers", kind: .preference, source: "manual", topic: "style")
        XCTAssertEqual("search_document:\n" + memory, SemanticEmbeddingText.memoryDocument(content: "Concise answers", kind: .preference, source: "manual", topic: "style"))
        XCTAssertEqual("search_query: " + SemanticEmbeddingText.queryBody("  term\nmore "), SemanticEmbeddingText.query("  term\nmore "))
        XCTAssertFalse(body.hasPrefix("search_document:"))
        XCTAssertFalse(memory.hasPrefix("search_document:"))
        let request = EmbeddingRequest(text: body, purpose: .retrievalDocument(title: "Notes"), dimensions: 256)
        XCTAssertEqual(request.purpose, .retrievalDocument(title: "Notes"))
        XCTAssertEqual(request.dimensions, 256)
        XCTAssertEqual(EmbeddingRequest(text: "legacy").purpose, .legacyFormatted)
    }

    func testMemoryIndexRejectsUnknownAndDifferentSameDimensionSpacesAndReloadsOnSelectionChange() throws {
        let container = try memoryContainer()
        let context = ModelContext(container)
        let current = identifiedMemory("current")
        let unknown = MemoryItem(content: "legacy", embedding: [1, 0])
        let otherModel = identifiedMemory("other model")
        otherModel.embeddingModelIdentifier = "other-space"
        let otherFormat = identifiedMemory("other format")
        otherFormat.embeddingFormatVersion = 3
        let wrongDimension = identifiedMemory("invalid dimension")
        wrongDimension.embeddingDimension = 3
        let invalid = identifiedMemory("invalid vector")
        for memory in [current, unknown, otherModel, otherFormat, wrongDimension, invalid] { context.insert(memory) }
        try context.save()
        invalid.embedding = [.nan, 0]
        MemoryVectorIndex.shared.invalidate()

        let load = MemoryVectorIndex.shared.ensureLoadedForTests(metadata: metadata) {
            try context.fetch(FetchDescriptor<MemoryItem>())
        }
        XCTAssertEqual(load.loadedCount, 1)
        XCTAssertEqual(load.diagnostic, "memory_reindex_required")
        XCTAssertEqual(MemoryVectorIndex.shared.search(query: [1, 0], topK: 10).map(\.id), [current.persistentModelID])
        XCTAssertEqual(MemoryVectorIndex.shared.ensureLoaded(context: context, metadata: metadata).diagnostic, "memory_reindex_required")
        let other = RAGEmbeddingIndexMetadata(formatVersion: 4, modelIdentifier: "other-space", dimension: 2)
        XCTAssertEqual(MemoryVectorIndex.shared.ensureLoaded(context: context, metadata: other).loadedCount, 1)
        XCTAssertEqual(MemoryVectorIndex.shared.search(query: [1, 0], topK: 10).map(\.id), [otherModel.persistentModelID])
        XCTAssertFalse(MemoryVectorIndex.shared.append(id: current.persistentModelID, isPinned: false, vector: current.embedding, metadata: metadata))
        XCTAssertEqual(MemoryStore.lexicalRecall(query: "legacy", context: context, limit: 5).map(\.id), [unknown.id])
    }

    func testRAGIndexingUsesExplicitDocumentPurposeAndPersistsProviderFormat() async throws {
        allowForeground()
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("synthetic.txt")
        try Data("A synthetic local note about actor isolation.".utf8).write(to: url)
        let container = try ModelContainer(for: RAGChunk.self, configurations: ModelConfiguration(isStoredInMemoryOnly: true))
        let context = ModelContext(container)
        var requests: [EmbeddingRequest] = []
        let result = await RAGStore.indexFileWithDiagnostics(url: url, context: context, embed: { request in
            requests.append(request)
            return EmbeddingRuntimeResult(vector: [1, 0], modelIdentifier: self.metadata.modelIdentifier, formatVersion: 4)
        }, save: { context, _, _ in try context.save() })

        XCTAssertEqual(result.mode, .indexed)
        XCTAssertEqual(requests.count, 1)
        XCTAssertEqual(requests.first?.purpose, .retrievalDocument(title: "synthetic.txt"))
        XCTAssertFalse(requests.first?.text.hasPrefix("search_document:") ?? true)
        let chunk = try XCTUnwrap(context.fetch(FetchDescriptor<RAGChunk>()).first)
        XCTAssertEqual(chunk.embeddingFormatVersion, 4)
        XCTAssertEqual(chunk.embeddingModelIdentifier, metadata.modelIdentifier)
        XCTAssertEqual(chunk.embeddingDimension, 2)
        RAGVectorIndex.shared.invalidate()
        let legacy = RAGVectorIndex.shared.ensureLoaded(context: context, formatVersion: 3, modelIdentifier: metadata.modelIdentifier, dimension: 2)
        XCTAssertEqual(legacy.loadedCount, 0)
        XCTAssertEqual(RAGVectorIndex.shared.ensureLoaded(context: context, metadata: metadata).loadedCount, 1)
    }

    func testBoundedRebuildPreservesSourcesAndReencodesEveryPageWithoutReusingVectors() async throws {
        allowForeground()
        let context = ModelContext(try memoryContainer())
        let source = MemoryItem(content: "Synthetic source", kind: .preference, source: "manual", topic: "style",
            expiresAt: Date(timeIntervalSince1970: 2_000_000_000), freshnessClass: .durable)
        source.isPinned = true
        source.createdAt = Date(timeIntervalSince1970: 1_600_000_000)
        let deferred = MemoryItem(content: "Later row")
        deferred.createdAt = source.createdAt.addingTimeInterval(1)
        // Even an already matching vector is not reused by a fresh source pass.
        let current = identifiedMemory("Previously encoded source")
        current.createdAt = source.createdAt.addingTimeInterval(2)
        for item in [source, deferred, current] { context.insert(item) }
        try context.save()
        let originalID = source.id
        var requests: [EmbeddingRequest] = []
        let embed: (EmbeddingRequest) async throws -> EmbeddingRuntimeResult = { request in
            requests.append(request)
            return EmbeddingRuntimeResult(vector: [0, 1], modelIdentifier: self.metadata.modelIdentifier, formatVersion: 4)
        }

        let first = await MemoryStore.rebuildEmbeddings(context: context, matching: metadata, limit: 1, embed: embed, writeBudget: DiskWriteBudget())
        XCTAssertEqual(first.scanned, 2)
        XCTAssertEqual(first.attempted, 1)
        XCTAssertEqual(first.rebuilt, 1)
        XCTAssertTrue(first.hasMore)
        XCTAssertEqual(first.mode, "partial")
        XCTAssertEqual(requests.first?.purpose, .retrievalDocument(title: "style"))
        XCTAssertFalse(requests.first?.text.hasPrefix("search_document:") ?? true)
        XCTAssertEqual(source.id, originalID)
        XCTAssertEqual(source.content, "Synthetic source")
        XCTAssertEqual(source.source, "manual")
        XCTAssertEqual(source.memoryKind, .preference)
        XCTAssertEqual(source.topic, "style")
        XCTAssertEqual(source.expiresAt, Date(timeIntervalSince1970: 2_000_000_000))
        XCTAssertEqual(source.freshnessClass, MemoryFreshnessClass.durable.rawValue)
        XCTAssertTrue(source.isPinned)
        XCTAssertEqual(source.createdAt, Date(timeIntervalSince1970: 1_600_000_000))
        XCTAssertTrue(MemoryVectorIndex.matches(source, metadata: metadata))
        XCTAssertNil(deferred.embeddingModelIdentifier)
        XCTAssertTrue(deferred.embedding.isEmpty)
        let cursor = try XCTUnwrap(first.nextCursor)
        XCTAssertEqual(cursor.id, source.id)

        let second = await MemoryStore.rebuildEmbeddings(context: context, matching: metadata, limit: 1, after: cursor, embed: embed, writeBudget: DiskWriteBudget())
        XCTAssertEqual(second.scanned, 2)
        XCTAssertEqual(second.attempted, 1)
        XCTAssertTrue(second.hasMore)
        XCTAssertEqual(second.nextCursor?.id, deferred.id)
        let third = await MemoryStore.rebuildEmbeddings(context: context, matching: metadata, limit: 1, after: second.nextCursor, embed: embed, writeBudget: DiskWriteBudget())
        XCTAssertEqual(third.scanned, 1)
        XCTAssertEqual(third.rebuilt, 1)
        XCTAssertFalse(third.hasMore)
        XCTAssertNil(third.nextCursor)
        XCTAssertEqual(requests.count, 3)
        XCTAssertTrue(requests[1].text.contains("Later row"))
        XCTAssertTrue(requests[2].text.contains("Previously encoded source"))
        XCTAssertEqual(current.embedding, [0, 1])
        XCTAssertEqual(try context.fetchCount(FetchDescriptor<MemoryItem>()), 3)
    }

    func testRebuildFailureCancellationAndProviderChangeLeaveSourcesUnindexed() async throws {
        allowForeground()
        let container = try memoryContainer()
        let context = ModelContext(container)
        let memory = MemoryItem(content: "Synthetic retained memory", embedding: [])
        context.insert(memory)
        try context.save()
        enum Failure: Error { case injected }

        let failed = await MemoryStore.rebuildEmbeddings(context: context, matching: metadata, embed: { _ in throw Failure.injected })
        XCTAssertEqual(failed.rebuilt, 0)
        XCTAssertEqual(failed.mode, "failed")
        let cancelled = await MemoryStore.rebuildEmbeddings(context: context, matching: metadata, embed: { _ in throw CancellationError() })
        XCTAssertEqual(cancelled.mode, "cancelled")
        let changed = await MemoryStore.rebuildEmbeddings(context: context, matching: metadata, embed: { _ in
            EmbeddingRuntimeResult(vector: [1, 0], modelIdentifier: "different-same-dimension-space", formatVersion: 4)
        })
        XCTAssertEqual(changed.diagnostic, "embedding_identity_changed_during_reindex")
        XCTAssertTrue(memory.embedding.isEmpty)
        XCTAssertNil(memory.embeddingModelIdentifier)
        XCTAssertEqual(try context.fetchCount(FetchDescriptor<MemoryItem>()), 1)
    }

    func testEditingRebuiltMemoryInvalidatesItsVectorAndRebuildsTheNewText() async throws {
        allowForeground()
        let container = try memoryContainer()
        let context = ModelContext(container)
        let memory = MemoryItem(content: "Synthetic original preference", embedding: [])
        context.insert(memory)
        try context.save()
        let originalID = memory.id
        var requests: [EmbeddingRequest] = []
        let embed: (EmbeddingRequest) async throws -> EmbeddingRuntimeResult = { request in
            requests.append(request)
            return EmbeddingRuntimeResult(vector: requests.count == 1 ? [1, 0] : [0, 1], modelIdentifier: self.metadata.modelIdentifier, formatVersion: 4)
        }

        let first = await MemoryStore.rebuildEmbeddings(context: context, matching: metadata, embed: embed, writeBudget: DiskWriteBudget())
        XCTAssertEqual(first.rebuilt, 1)
        XCTAssertEqual(MemoryVectorIndex.shared.ensureLoaded(context: context, metadata: metadata).loadedCount, 1)
        MemoryStore.updateContent(memory.content, for: memory)
        XCTAssertTrue(MemoryVectorIndex.matches(memory, metadata: metadata))

        MemoryStore.updateContent("Synthetic edited preference", for: memory)
        XCTAssertTrue(memory.embedding.isEmpty)
        XCTAssertNil(memory.embeddingFormatVersion)
        XCTAssertNil(memory.embeddingModelIdentifier)
        XCTAssertNil(memory.embeddingDimension)
        XCTAssertFalse(MemoryVectorIndex.matches(memory, metadata: metadata))
        XCTAssertEqual(MemoryVectorIndex.shared.ensureLoaded(context: context, metadata: metadata).loadedCount, 0)
        XCTAssertEqual(MemoryStore.lexicalRecall(query: "edited", context: context, limit: 5).map(\.id), [originalID])
        try context.save()
        let verificationContext = ModelContext(container)
        let persisted = try XCTUnwrap(verificationContext.fetch(FetchDescriptor<MemoryItem>()).first)
        XCTAssertEqual(persisted.content, "Synthetic edited preference")
        XCTAssertNil(persisted.embeddingModelIdentifier)

        let second = await MemoryStore.rebuildEmbeddings(context: context, matching: metadata, embed: embed, writeBudget: DiskWriteBudget())
        XCTAssertEqual(second.attempted, 1)
        XCTAssertEqual(second.rebuilt, 1)
        XCTAssertEqual(requests.count, 2)
        XCTAssertTrue(requests[1].text.contains("Synthetic edited preference"))
        XCTAssertFalse(requests[1].text.contains("Synthetic original preference"))
        XCTAssertEqual(memory.id, originalID)
        XCTAssertEqual(memory.embedding, [0, 1])
        XCTAssertTrue(MemoryVectorIndex.matches(memory, metadata: metadata))
        XCTAssertEqual(MemoryVectorIndex.shared.ensureLoaded(context: context, metadata: metadata).loadedCount, 1)
    }

    func testEditingMemoryDuringInferenceDoesNotInstallTheOldTextVector() async throws {
        allowForeground()
        let context = ModelContext(try memoryContainer())
        let memory = MemoryItem(content: "Synthetic original", embedding: [])
        context.insert(memory)
        try context.save()

        let result = await MemoryStore.rebuildEmbeddings(context: context, matching: metadata, embed: { _ in
            MemoryStore.updateContent("Synthetic edited while encoding", for: memory)
            return EmbeddingRuntimeResult(vector: [1, 0], modelIdentifier: self.metadata.modelIdentifier, formatVersion: 4)
        }, writeBudget: DiskWriteBudget())

        XCTAssertEqual(result.rebuilt, 0)
        XCTAssertEqual(result.diagnostic, "memory_changed_during_reindex")
        XCTAssertEqual(memory.content, "Synthetic edited while encoding")
        XCTAssertTrue(memory.embedding.isEmpty)
        XCTAssertNil(memory.embeddingModelIdentifier)
    }

    func testDeletionDuringInferenceDoesNotTouchTheDeletedRow() async throws {
        allowForeground()
        let context = ModelContext(try memoryContainer())
        let memory = MemoryItem(content: "Synthetic row deleted while encoding", embedding: [])
        context.insert(memory)
        try context.save()
        var saves = 0

        let result = await MemoryStore.rebuildEmbeddings(context: context, matching: metadata, embed: { _ in
            context.delete(memory)
            try context.save()
            return EmbeddingRuntimeResult(vector: [1, 0], modelIdentifier: self.metadata.modelIdentifier, formatVersion: 4)
        }, save: { _ in saves += 1 }, writeBudget: DiskWriteBudget())

        XCTAssertEqual(result.rebuilt, 0)
        XCTAssertEqual(result.diagnostic, "memory_deleted_during_reindex")
        XCTAssertEqual(saves, 0)
        XCTAssertEqual(try context.fetchCount(FetchDescriptor<MemoryItem>()), 0)
    }

    func testDeletionOfLaterCandidateDuringInferenceStopsBeforeReadingIt() async throws {
        allowForeground()
        let context = ModelContext(try memoryContainer())
        let first = MemoryItem(content: "Synthetic first row", embedding: [])
        first.createdAt = Date(timeIntervalSince1970: 1)
        let second = MemoryItem(content: "Synthetic later deleted row", embedding: [])
        second.createdAt = Date(timeIntervalSince1970: 2)
        for item in [first, second] { context.insert(item) }
        try context.save()
        var calls = 0

        let result = await MemoryStore.rebuildEmbeddings(context: context, matching: metadata, embed: { _ in
            calls += 1
            context.delete(second)
            try context.save()
            return EmbeddingRuntimeResult(vector: [1, 0], modelIdentifier: self.metadata.modelIdentifier, formatVersion: 4)
        }, writeBudget: DiskWriteBudget())

        XCTAssertEqual(result.rebuilt, 1)
        XCTAssertEqual(result.mode, "partial")
        XCTAssertEqual(result.diagnostic, "memory_deleted_during_reindex")
        XCTAssertEqual(calls, 1)
        XCTAssertTrue(MemoryVectorIndex.matches(first, metadata: metadata))
        XCTAssertEqual(try context.fetchCount(FetchDescriptor<MemoryItem>()), 1)
    }

    func testDeferredRememberRetainsItsQueueCaptureAndUnrelatedContextEdits() async throws {
        let container = try memoryContainer()
        let context = ModelContext(container)
        context.autosaveEnabled = false
        let existing = MemoryItem(content: "Synthetic existing memory")
        context.insert(existing)
        try context.save()
        existing.topic = "Unrelated pending edit"
        let queueURL = FileManager.default.temporaryDirectory.appendingPathComponent("MemoryRememberDeferred-\(UUID().uuidString).json")
        defer { try? FileManager.default.removeItem(at: queueURL) }
        let capture = try MemoryCaptureQueue.enqueue(content: "Synthetic queued new memory", fileURL: queueURL)
        let deniedBudget = DiskWriteBudget(oneMinuteLimit: 1)

        let result = await MemoryCaptureQueue.drain(fileURL: queueURL) { pending in
            _ = try await MemoryStore.rememberResult(pending.content, kind: pending.kind, source: pending.source, topic: pending.topic, context: context, embed: { _ in
                EmbeddingRuntimeResult(vector: [1, 0], modelIdentifier: self.metadata.modelIdentifier, formatVersion: 4)
            }, writeBudget: deniedBudget)
        }

        XCTAssertEqual(result.promoted, 0)
        XCTAssertEqual(result.remaining, 1)
        XCTAssertEqual(try MemoryCaptureQueue.loadPending(fileURL: queueURL).map(\.id), [capture.id])
        XCTAssertEqual(try context.fetch(FetchDescriptor<MemoryItem>()).map(\.id), [existing.id])
        XCTAssertEqual(existing.topic, "Unrelated pending edit")
        try context.save()
        let verificationContext = ModelContext(container)
        let persisted = try verificationContext.fetch(FetchDescriptor<MemoryItem>())
        XCTAssertEqual(persisted.count, 1)
        XCTAssertEqual(persisted.first?.topic, "Unrelated pending edit")
    }

    func testRememberSaveFailureRemovesOnlyItsNewRowAndReleasesWriteBudget() async throws {
        let container = try memoryContainer()
        let context = ModelContext(container)
        context.autosaveEnabled = false
        let existing = MemoryItem(content: "Synthetic prior memory")
        context.insert(existing)
        try context.save()
        existing.topic = "Preserved pending edit"
        let budget = DiskWriteBudget(oneMinuteLimit: 65_536)
        enum Failure: Error { case save }

        do {
            _ = try await MemoryStore.rememberResult("Synthetic failed new row", kind: .fact, source: "test", topic: nil, context: context, embed: { _ in
                EmbeddingRuntimeResult(vector: [1, 0], modelIdentifier: self.metadata.modelIdentifier, formatVersion: 4)
            }, save: { _ in throw Failure.save }, writeBudget: budget)
            XCTFail("A failed save must not report a stored memory")
        } catch Failure.save {
            // Expected persistence failure.
        }

        XCTAssertEqual(try context.fetch(FetchDescriptor<MemoryItem>()).map(\.id), [existing.id])
        XCTAssertEqual(existing.topic, "Preserved pending edit")
        XCTAssertTrue(budget.canWrite(bytes: 65_536, category: .memory))
        try context.save()
        let verificationContext = ModelContext(container)
        XCTAssertEqual(try verificationContext.fetchCount(FetchDescriptor<MemoryItem>()), 1)
    }

    func testRebuildSkipsOversizedOldestMemoryAndRebuildsTheNextRow() async throws {
        allowForeground()
        let context = ModelContext(try memoryContainer())
        let oversized = MemoryItem(content: "Synthetic oversized source memory", embedding: [])
        oversized.createdAt = Date(timeIntervalSince1970: 1)
        let valid = MemoryItem(content: "Synthetic valid later memory", embedding: [])
        valid.createdAt = Date(timeIntervalSince1970: 2)
        for item in [oversized, valid] { context.insert(item) }
        try context.save()

        let result = await MemoryStore.rebuildEmbeddings(context: context, matching: metadata, embed: { request in
            if request.text.contains("oversized") { throw CoreAITextEmbeddingError.inputTooLong }
            return EmbeddingRuntimeResult(vector: [1, 0], modelIdentifier: self.metadata.modelIdentifier, formatVersion: 4)
        }, writeBudget: DiskWriteBudget())

        XCTAssertEqual(result.attempted, 2)
        XCTAssertEqual(result.rebuilt, 1)
        XCTAssertEqual(result.skipped, 1)
        XCTAssertFalse(result.hasMore)
        XCTAssertEqual(result.diagnostic, "memory_inputs_skipped:1")
        XCTAssertNil(result.nextCursor)
        XCTAssertTrue(oversized.embedding.isEmpty)
        XCTAssertNil(oversized.embeddingModelIdentifier)
        XCTAssertTrue(MemoryVectorIndex.matches(valid, metadata: metadata))
        XCTAssertEqual(try context.fetchCount(FetchDescriptor<MemoryItem>()), 2)
    }

    func testRebuildContinuationAdvancesPastAnEntireSkippedBatch() async throws {
        allowForeground()
        let context = ModelContext(try memoryContainer())
        let oversized = (0..<16).map { index in
            let item = MemoryItem(content: "Synthetic oversized memory \(index)", embedding: [])
            item.createdAt = Date(timeIntervalSince1970: Double(index))
            context.insert(item)
            return item
        }
        let valid = MemoryItem(content: "Synthetic valid memory after the first batch", embedding: [])
        valid.createdAt = Date(timeIntervalSince1970: 16)
        context.insert(valid)
        try context.save()
        var calls = 0
        let embed: (EmbeddingRequest) async throws -> EmbeddingRuntimeResult = { request in
            calls += 1
            if request.text.contains("oversized") { throw CoreAITextEmbeddingError.invalidRequest }
            return EmbeddingRuntimeResult(vector: [1, 0], modelIdentifier: self.metadata.modelIdentifier, formatVersion: 4)
        }

        let first = await MemoryStore.rebuildEmbeddings(context: context, matching: metadata, limit: 16, embed: embed, writeBudget: DiskWriteBudget())
        XCTAssertEqual(first.attempted, 16)
        XCTAssertEqual(first.skipped, 16)
        XCTAssertEqual(first.rebuilt, 0)
        XCTAssertEqual(first.scanned, 17)
        XCTAssertTrue(first.hasMore)
        let cursor = try XCTUnwrap(first.nextCursor)
        XCTAssertEqual(cursor.id, oversized[15].id)

        let second = await MemoryStore.rebuildEmbeddings(context: context, matching: metadata, limit: 16, after: cursor, embed: embed, writeBudget: DiskWriteBudget())
        XCTAssertEqual(second.attempted, 1)
        XCTAssertEqual(second.rebuilt, 1)
        XCTAssertEqual(second.skipped, 0)
        XCTAssertEqual(second.scanned, 1)
        XCTAssertFalse(second.hasMore)
        XCTAssertNil(second.nextCursor)
        XCTAssertEqual(calls, 17)
        XCTAssertTrue(oversized.allSatisfy { $0.embedding.isEmpty && $0.embeddingModelIdentifier == nil })
        XCTAssertTrue(MemoryVectorIndex.matches(valid, metadata: metadata))
    }

    func testRebuildContinuationRetriesTerminalFailureWithoutRepeatingCompletedRows() async throws {
        allowForeground()
        let context = ModelContext(try memoryContainer())
        let first = MemoryItem(content: "Synthetic skipped memory", embedding: [])
        first.createdAt = Date(timeIntervalSince1970: 1)
        let retry = MemoryItem(content: "Synthetic retry memory", embedding: [])
        retry.createdAt = Date(timeIntervalSince1970: 2)
        for item in [first, retry] { context.insert(item) }
        try context.save()
        let stopped = await MemoryStore.rebuildEmbeddings(context: context, matching: metadata, embed: { request in
            if request.text.contains("skipped") { throw CoreAITextEmbeddingError.inputTooLong }
            throw CoreAITextEmbeddingError.resourceDenied
        })

        XCTAssertEqual(stopped.attempted, 2)
        XCTAssertEqual(stopped.skipped, 1)
        XCTAssertEqual(stopped.rebuilt, 0)
        XCTAssertEqual(stopped.nextCursor?.id, first.id)
        XCTAssertTrue(retry.embedding.isEmpty)
        var requests: [EmbeddingRequest] = []
        let resumed = await MemoryStore.rebuildEmbeddings(context: context, matching: metadata, after: stopped.nextCursor, embed: { request in
            requests.append(request)
            return EmbeddingRuntimeResult(vector: [1, 0], modelIdentifier: self.metadata.modelIdentifier, formatVersion: 4)
        }, writeBudget: DiskWriteBudget())
        XCTAssertEqual(resumed.rebuilt, 1)
        XCTAssertEqual(requests.count, 1)
        XCTAssertTrue(requests[0].text.contains("retry"))
        XCTAssertNil(resumed.nextCursor)
    }

    func testRebuildRejectsContinuationFromAnotherEmbeddingSpaceWithoutRestarting() async throws {
        allowForeground()
        let context = ModelContext(try memoryContainer())
        let memory = MemoryItem(content: "Synthetic oldest memory", embedding: [])
        memory.createdAt = Date(timeIntervalSince1970: 1)
        context.insert(memory)
        try context.save()
        let other = RAGEmbeddingIndexMetadata(formatVersion: 4, modelIdentifier: "other-space", dimension: 2)
        let cursor = MemoryStore.RebuildCursor(createdAt: Date(timeIntervalSince1970: 100), id: UUID(), metadata: other,
            generation: try XCTUnwrap(EmbeddingIndexLifecycle.captureWrite()))
        let result = await MemoryStore.rebuildEmbeddings(context: context, matching: metadata, after: cursor, embed: { _ in
            EmbeddingRuntimeResult(vector: [1, 0], modelIdentifier: self.metadata.modelIdentifier, formatVersion: 4)
        }, writeBudget: DiskWriteBudget())
        XCTAssertEqual(result.scanned, 0)
        XCTAssertEqual(result.attempted, 0)
        XCTAssertEqual(result.rebuilt, 0)
        XCTAssertEqual(result.diagnostic, "embedding_identity_changed_during_reindex")
        XCTAssertNil(result.nextCursor)
        XCTAssertTrue(memory.embedding.isEmpty)
        XCTAssertNil(memory.embeddingModelIdentifier)
    }

    func testRebuildPersistenceFailureRestoresOriginalRowAndReleasesWriteReservation() async throws {
        allowForeground()
        let container = try memoryContainer()
        let context = ModelContext(container)
        let memory = MemoryItem(content: "Synthetic saved memory", embedding: [])
        context.insert(memory)
        try context.save()
        let budget = DiskWriteBudget(oneMinuteLimit: 65_536)
        enum Failure: Error { case save }
        let result = await MemoryStore.rebuildEmbeddings(context: context, matching: metadata, embed: { _ in
            EmbeddingRuntimeResult(vector: [1, 0], modelIdentifier: self.metadata.modelIdentifier, formatVersion: 4)
        }, save: { _ in throw Failure.save }, writeBudget: budget)

        XCTAssertEqual(result.rebuilt, 0)
        XCTAssertEqual(result.mode, "failed")
        XCTAssertTrue(result.diagnostic?.hasPrefix("persist_failed:") == true)
        XCTAssertTrue(memory.embedding.isEmpty)
        XCTAssertNil(memory.embeddingModelIdentifier)
        XCTAssertTrue(budget.canWrite(bytes: 65_536, category: .memory))
    }

    func testBackgroundRebuildDoesNotInvokeProviderOrChangeRows() async throws {
        ResourceBudgetGate.testSnapshotOverride = .init(scenePhase: .background, lowPowerModeEnabled: false, thermalState: .nominal, recentMemoryWarningCount: 0, lastMemoryWarningAt: nil)
        let container = try memoryContainer()
        let context = ModelContext(container)
        let memory = MemoryItem(content: "Synthetic background memory", embedding: [])
        context.insert(memory)
        try context.save()
        var calls = 0
        let result = await MemoryStore.rebuildEmbeddings(context: context, matching: metadata, embed: { _ in
            calls += 1
            return EmbeddingRuntimeResult(vector: [1, 0], modelIdentifier: self.metadata.modelIdentifier, formatVersion: 4)
        })
        XCTAssertEqual(calls, 0)
        XCTAssertEqual(result.rebuilt, 0)
        XCTAssertEqual(result.mode, "skipped")
        XCTAssertTrue(memory.embedding.isEmpty)
    }

    func testRebuildDescriptorLimitsDatabaseReadsToOneBoundedPageAndLookahead() {
        XCTAssertEqual(MemoryStore.rebuildPageDescriptor(after: nil, limit: 0).fetchLimit, 2)
        XCTAssertEqual(MemoryStore.rebuildPageDescriptor(after: nil, limit: 16).fetchLimit, 17)
        XCTAssertEqual(MemoryStore.rebuildPageDescriptor(after: nil, limit: 100_000).fetchLimit, 33)
        XCTAssertNil(MemoryStore.rebuildPageDescriptor(after: nil, limit: 16).fetchOffset)
        XCTAssertFalse(MemoryStore.rebuildPageDescriptor(after: nil, limit: 16).includePendingChanges)
        XCTAssertEqual(MemoryStore.rebuildPageDescriptor(after: nil, limit: 16).sortBy.count, 2)
    }

    func testSQLiteRebuildPaginatesEqualDatesAfterVectorChangesAndCursorRowDeletion() async throws {
        allowForeground()
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("MemoryFreshIndexPages-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let schema = Schema([MemoryItem.self])
        let configuration = ModelConfiguration("MemoryFreshIndexPages", schema: schema,
            url: directory.appendingPathComponent("memories.store"), cloudKitDatabase: .none)
        let container = try ModelContainer(for: schema, configurations: [configuration])
        let equalDate = Date(timeIntervalSince1970: 1_600_000_000)
        let ids = (0..<6).map { UUID(uuidString: "00000000-0000-4000-8000-00000000000\($0)")! }
        do {
            let context = ModelContext(container)
            context.autosaveEnabled = false
            for index in [3, 1, 5, 0, 4, 2] {
                let source = MemoryItem(content: "Synthetic page source \(index)", source: "test", topic: "pages")
                source.id = ids[index]
                source.createdAt = equalDate
                context.insert(source)
            }
            try context.save()
        }
        var cursor: MemoryStore.RebuildCursor?
        var encoded: [String] = []
        for page in 0..<3 {
            // A fresh context prevents in-memory fixture sorting from hiding an
            // unsupported UUID sort/predicate in the native SQLite store.
            let context = ModelContext(container)
            context.autosaveEnabled = false
            let result = await MemoryStore.rebuildEmbeddings(context: context, matching: metadata, limit: 2, after: cursor, embed: { request in
                encoded.append(request.text)
                return EmbeddingRuntimeResult(vector: [1, 0], modelIdentifier: self.metadata.modelIdentifier, formatVersion: 4)
            }, writeBudget: DiskWriteBudget())
            XCTAssertEqual(result.scanned, page < 2 ? 3 : 2)
            XCTAssertEqual(result.attempted, 2)
            XCTAssertEqual(result.rebuilt, 2)
            XCTAssertEqual(result.hasMore, page < 2)
            cursor = result.nextCursor
            if page == 0 {
                let firstID = ids[0]
                let cursorID = ids[1]
                let first = try XCTUnwrap(context.fetch(FetchDescriptor<MemoryItem>(predicate: #Predicate { $0.id == firstID })).first)
                first.embedding = [0, 1]
                let cursorRow = try XCTUnwrap(context.fetch(FetchDescriptor<MemoryItem>(predicate: #Predicate { $0.id == cursorID })).first)
                context.delete(cursorRow)
                try context.save()
            }
        }
        XCTAssertNil(cursor)
        XCTAssertEqual(encoded.count, ids.count)
        for index in 0..<6 {
            XCTAssertTrue(encoded[index].contains("Synthetic page source \(index)"))
        }
        let verificationContext = ModelContext(container)
        XCTAssertEqual(try verificationContext.fetchCount(FetchDescriptor<MemoryItem>()), 5)
        for source in try verificationContext.fetch(FetchDescriptor<MemoryItem>()) {
            XCTAssertEqual(source.source, "test")
            XCTAssertEqual(source.topic, "pages")
            XCTAssertEqual(source.createdAt, equalDate)
            XCTAssertTrue(source.content.hasPrefix("Synthetic page source "))
        }
    }

    func testRebuildLimitClampsInferenceAndHydrationWithoutCountingTheStore() async throws {
        allowForeground()
        let context = ModelContext(try memoryContainer())
        for index in 0..<40 {
            let source = MemoryItem(content: "Synthetic bounded source \(index)")
            source.createdAt = Date(timeIntervalSince1970: Double(index))
            context.insert(source)
        }
        try context.save()
        var calls = 0
        let result = await MemoryStore.rebuildEmbeddings(context: context, matching: metadata, limit: 10_000, embed: { _ in
            calls += 1
            return EmbeddingRuntimeResult(vector: [1, 0], modelIdentifier: self.metadata.modelIdentifier, formatVersion: 4)
        }, writeBudget: DiskWriteBudget(oneMinuteLimit: 10_000_000))
        XCTAssertEqual(result.scanned, 33)
        XCTAssertEqual(result.attempted, 32)
        XCTAssertEqual(result.rebuilt, 32)
        XCTAssertEqual(calls, 32)
        XCTAssertTrue(result.hasMore)
        XCTAssertNotNil(result.nextCursor)
    }

    func testRememberRejectsDelayedVectorAfterResetEvenWhenResetHasFinished() async throws {
        allowForeground()
        let context = ModelContext(try memoryContainer())
        let source = MemoryItem(content: "Synthetic existing source")
        context.insert(source)
        try context.save()
        do {
            _ = try await MemoryStore.rememberResult("Synthetic delayed capture", kind: .fact, source: "test", topic: nil,
                context: context, embed: { _ in
                    let reset = EmbeddingIndexLifecycle.beginReset()
                    XCTAssertTrue(EmbeddingIndexLifecycle.finishReset(reset))
                    return EmbeddingRuntimeResult(vector: [1, 0], modelIdentifier: self.metadata.modelIdentifier, formatVersion: 4)
                }, writeBudget: DiskWriteBudget())
            XCTFail("A pre-reset inference must not write into the new index")
        } catch MemoryStore.PersistenceFailure.embeddingIndexReset {
            // The source capture remains the caller's responsibility to retry.
        }
        XCTAssertEqual(try context.fetch(FetchDescriptor<MemoryItem>()).map(\.id), [source.id])
        XCTAssertTrue(source.embedding.isEmpty)
    }

    func testRebuildRejectsDelayedVectorAcrossResetAndLeavesItsSourceUnindexed() async throws {
        allowForeground()
        let context = ModelContext(try memoryContainer())
        let source = MemoryItem(content: "Synthetic retained rebuild source")
        context.insert(source)
        try context.save()
        var saves = 0
        let result = await MemoryStore.rebuildEmbeddings(context: context, matching: metadata, embed: { _ in
            let reset = EmbeddingIndexLifecycle.beginReset()
            XCTAssertTrue(EmbeddingIndexLifecycle.finishReset(reset))
            return EmbeddingRuntimeResult(vector: [1, 0], modelIdentifier: self.metadata.modelIdentifier, formatVersion: 4)
        }, save: { _ in saves += 1 }, writeBudget: DiskWriteBudget())
        XCTAssertEqual(result.rebuilt, 0)
        XCTAssertEqual(result.diagnostic, "embedding_index_reset")
        XCTAssertTrue(result.hasMore)
        XCTAssertEqual(saves, 0)
        XCTAssertEqual(source.content, "Synthetic retained rebuild source")
        XCTAssertTrue(source.embedding.isEmpty)
        XCTAssertNil(source.embeddingModelIdentifier)
    }

    func testRebuildRejectsSameModelCursorFromBeforeResetWithoutFetchingOrEncoding() async throws {
        allowForeground()
        let context = ModelContext(try memoryContainer())
        for index in 0..<2 {
            let source = MemoryItem(content: "Synthetic reset continuation source \(index)")
            source.createdAt = Date(timeIntervalSince1970: Double(index))
            context.insert(source)
        }
        try context.save()
        let first = await MemoryStore.rebuildEmbeddings(context: context, matching: metadata, limit: 1, embed: { _ in
            EmbeddingRuntimeResult(vector: [1, 0], modelIdentifier: self.metadata.modelIdentifier, formatVersion: 4)
        }, writeBudget: DiskWriteBudget())
        let cursor = try XCTUnwrap(first.nextCursor)
        let reset = EmbeddingIndexLifecycle.beginReset()
        XCTAssertTrue(EmbeddingIndexLifecycle.finishReset(reset))
        var calls = 0
        let next = await MemoryStore.rebuildEmbeddings(context: context, matching: metadata, limit: 1, after: cursor, embed: { _ in
            calls += 1
            return EmbeddingRuntimeResult(vector: [1, 0], modelIdentifier: self.metadata.modelIdentifier, formatVersion: 4)
        }, writeBudget: DiskWriteBudget())
        XCTAssertEqual(next.scanned, 0)
        XCTAssertEqual(next.attempted, 0)
        XCTAssertEqual(calls, 0)
        XCTAssertEqual(next.diagnostic, "embedding_index_reset")
    }

    func testPendingResetBlocksRememberAndRebuildBeforeProviderCalls() async throws {
        allowForeground()
        let context = ModelContext(try memoryContainer())
        let source = MemoryItem(content: "Synthetic pending reset source")
        context.insert(source)
        try context.save()
        let reset = EmbeddingIndexLifecycle.beginReset()
        defer { _ = EmbeddingIndexLifecycle.finishReset(reset) }
        var calls = 0
        let embed: (EmbeddingRequest) async throws -> EmbeddingRuntimeResult = { _ in
            calls += 1
            return EmbeddingRuntimeResult(vector: [1, 0], modelIdentifier: self.metadata.modelIdentifier, formatVersion: 4)
        }
        do {
            _ = try await MemoryStore.rememberResult("Synthetic deferred capture", kind: .fact, source: "test", topic: nil,
                context: context, embed: embed, writeBudget: DiskWriteBudget())
            XCTFail("Remember must fail while vector reset is pending")
        } catch MemoryStore.PersistenceFailure.embeddingIndexReset {}
        let rebuilt = await MemoryStore.rebuildEmbeddings(context: context, matching: metadata, embed: embed, writeBudget: DiskWriteBudget())
        XCTAssertEqual(rebuilt.scanned, 0)
        XCTAssertEqual(rebuilt.attempted, 0)
        XCTAssertEqual(rebuilt.diagnostic, "embedding_index_reset")
        XCTAssertEqual(calls, 0)
        XCTAssertTrue(source.embedding.isEmpty)
    }

    func testSavedCrossContextSourceEditDuringRebuildDoesNotInstallStaleVector() async throws {
        allowForeground()
        let container = try memoryContainer()
        let context = ModelContext(container)
        context.autosaveEnabled = false
        let source = MemoryItem(content: "Synthetic source before external edit")
        context.insert(source)
        try context.save()
        let sourceID = source.id
        var saves = 0
        let result = await MemoryStore.rebuildEmbeddings(context: context, matching: metadata, embed: { _ in
            let otherContext = ModelContext(container)
            otherContext.autosaveEnabled = false
            let edited = try XCTUnwrap(otherContext.fetch(FetchDescriptor<MemoryItem>(predicate: #Predicate { $0.id == sourceID })).first)
            MemoryStore.updateContent("Synthetic saved external edit", for: edited)
            try otherContext.save()
            return EmbeddingRuntimeResult(vector: [1, 0], modelIdentifier: self.metadata.modelIdentifier, formatVersion: 4)
        }, save: { _ in saves += 1 }, writeBudget: DiskWriteBudget())
        XCTAssertEqual(result.rebuilt, 0)
        XCTAssertEqual(result.diagnostic, "memory_changed_during_reindex")
        XCTAssertEqual(saves, 0)
        let persisted = try XCTUnwrap(ModelContext(container).fetch(FetchDescriptor<MemoryItem>()).first)
        XCTAssertEqual(persisted.content, "Synthetic saved external edit")
        XCTAssertTrue(persisted.embedding.isEmpty)
    }

    func testSavedCrossContextDeletionDuringRebuildDoesNotResurrectSource() async throws {
        allowForeground()
        let container = try memoryContainer()
        let context = ModelContext(container)
        context.autosaveEnabled = false
        let source = MemoryItem(content: "Synthetic source deleted externally")
        context.insert(source)
        try context.save()
        let sourceID = source.id
        var saves = 0
        let result = await MemoryStore.rebuildEmbeddings(context: context, matching: metadata, embed: { _ in
            let otherContext = ModelContext(container)
            otherContext.autosaveEnabled = false
            let deleted = try XCTUnwrap(otherContext.fetch(FetchDescriptor<MemoryItem>(predicate: #Predicate { $0.id == sourceID })).first)
            otherContext.delete(deleted)
            try otherContext.save()
            return EmbeddingRuntimeResult(vector: [1, 0], modelIdentifier: self.metadata.modelIdentifier, formatVersion: 4)
        }, save: { _ in saves += 1 }, writeBudget: DiskWriteBudget())
        XCTAssertEqual(result.rebuilt, 0)
        XCTAssertEqual(result.diagnostic, "memory_deleted_during_reindex")
        XCTAssertEqual(saves, 0)
        XCTAssertEqual(try ModelContext(container).fetchCount(FetchDescriptor<MemoryItem>()), 0)
    }

    func testEditingCachedVectorlessMemoryAfterAnotherContextRebuildClearsPersistedVector() async throws {
        allowForeground()
        let container = try memoryContainer()
        let presentationContext = ModelContext(container)
        presentationContext.autosaveEnabled = false
        let presented = MemoryItem(content: "Synthetic source cached before rebuild")
        presentationContext.insert(presented)
        try presentationContext.save()
        XCTAssertTrue(presented.embedding.isEmpty)
        XCTAssertNil(presented.embeddingModelIdentifier)

        let rebuildContext = ModelContext(container)
        rebuildContext.autosaveEnabled = false
        let result = await MemoryStore.rebuildEmbeddings(
            context: rebuildContext, matching: metadata,
            embed: { _ in EmbeddingRuntimeResult(vector: [1, 0], modelIdentifier: self.metadata.modelIdentifier, formatVersion: self.metadata.formatVersion) },
            writeBudget: DiskWriteBudget()
        )
        XCTAssertEqual(result.rebuilt, 1)

        // The presentation context may still hold the vectorless snapshot from
        // before another context saved the embedding. Editing it must discard
        // the persisted derived vector as well as the local cached fields.
        MemoryStore.updateContent("Synthetic edited source after rebuild", for: presented)
        try presentationContext.save()
        let readContext = ModelContext(container)
        let persisted = try XCTUnwrap(readContext.fetch(FetchDescriptor<MemoryItem>()).first)
        XCTAssertEqual(persisted.content, "Synthetic edited source after rebuild")
        XCTAssertTrue(persisted.embedding.isEmpty)
        XCTAssertNil(persisted.embeddingFormatVersion)
        XCTAssertNil(persisted.embeddingModelIdentifier)
        XCTAssertNil(persisted.embeddingDimension)
        MemoryVectorIndex.shared.invalidate()
        XCTAssertEqual(MemoryVectorIndex.shared.ensureLoaded(context: presentationContext, metadata: metadata).loadedCount, 0)
    }

    private func memoryContainer() throws -> ModelContainer {
        try ModelContainer(for: MemoryItem.self, configurations: ModelConfiguration(isStoredInMemoryOnly: true))
    }

    private func identifiedMemory(_ content: String) -> MemoryItem {
        MemoryItem(content: content, embedding: [1, 0], embeddingFormatVersion: metadata.formatVersion, embeddingModelIdentifier: metadata.modelIdentifier, embeddingDimension: metadata.dimension)
    }

    private func allowForeground() {
        ResourceBudgetGate.testSnapshotOverride = .init(scenePhase: .active, lowPowerModeEnabled: false, thermalState: .nominal, recentMemoryWarningCount: 0, lastMemoryWarningAt: nil)
    }
}
