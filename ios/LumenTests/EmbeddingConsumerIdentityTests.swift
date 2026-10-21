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

        let load = MemoryVectorIndex.shared.ensureLoaded(context: context, metadata: metadata)
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

    func testBoundedReembeddingPreservesMemoryIdentityAndSkipsCurrentRows() async throws {
        allowForeground()
        let container = try memoryContainer()
        let context = ModelContext(container)
        let legacy = MemoryItem(content: "Synthetic legacy", kind: .preference, embedding: [0, 1], topic: "style")
        legacy.isPinned = true
        legacy.createdAt = Date(timeIntervalSince1970: 1_600_000_000)
        let deferred = MemoryItem(content: "Later row", embedding: [0, 1])
        let current = identifiedMemory("already current")
        for item in [legacy, deferred, current] { context.insert(item) }
        try context.save()
        let originalID = legacy.id
        var requests: [EmbeddingRequest] = []

        let result = await MemoryStore.reembedStaleEmbeddings(context: context, matching: metadata, limit: 1, embed: { request in
            requests.append(request)
            return EmbeddingRuntimeResult(vector: [1, 0], modelIdentifier: self.metadata.modelIdentifier, formatVersion: 4)
        }, writeBudget: DiskWriteBudget())

        XCTAssertEqual(result.attempted, 1)
        XCTAssertEqual(result.rebuilt, 1)
        XCTAssertEqual(result.remaining, 1)
        XCTAssertEqual(result.mode, "partial")
        XCTAssertEqual(requests.first?.purpose, .retrievalDocument(title: "style"))
        XCTAssertFalse(requests.first?.text.hasPrefix("search_document:") ?? true)
        XCTAssertEqual(legacy.id, originalID)
        XCTAssertEqual(legacy.content, "Synthetic legacy")
        XCTAssertTrue(legacy.isPinned)
        XCTAssertEqual(legacy.createdAt, Date(timeIntervalSince1970: 1_600_000_000))
        XCTAssertTrue(MemoryVectorIndex.matches(legacy, metadata: metadata))
        XCTAssertNil(deferred.embeddingModelIdentifier)
        XCTAssertEqual(deferred.embedding, [0, 1])
        XCTAssertEqual(try context.fetchCount(FetchDescriptor<MemoryItem>()), 3)

        let cursor = try XCTUnwrap(result.nextCursor)
        XCTAssertEqual(cursor.id, legacy.id)
        // The cursor remains valid after its successful row leaves the stale set.
        let resumed = await MemoryStore.reembedStaleEmbeddings(context: context, matching: metadata, limit: 1, after: cursor, embed: { request in
            XCTAssertTrue(request.text.contains("Later row"))
            return EmbeddingRuntimeResult(vector: [1, 0], modelIdentifier: self.metadata.modelIdentifier, formatVersion: 4)
        }, writeBudget: DiskWriteBudget())
        XCTAssertEqual(resumed.attempted, 1)
        XCTAssertEqual(resumed.rebuilt, 1)
        XCTAssertEqual(resumed.remaining, 0)
        XCTAssertNil(resumed.nextCursor)
        XCTAssertTrue(MemoryVectorIndex.matches(deferred, metadata: metadata))
    }

    func testReembeddingFailureCancellationAndProviderChangeKeepOldVectors() async throws {
        allowForeground()
        let container = try memoryContainer()
        let context = ModelContext(container)
        let memory = MemoryItem(content: "Synthetic retained memory", embedding: [0, 1])
        context.insert(memory)
        try context.save()
        enum Failure: Error { case injected }

        let failed = await MemoryStore.reembedStaleEmbeddings(context: context, matching: metadata, embed: { _ in throw Failure.injected })
        XCTAssertEqual(failed.rebuilt, 0)
        XCTAssertEqual(failed.mode, "failed")
        let cancelled = await MemoryStore.reembedStaleEmbeddings(context: context, matching: metadata, embed: { _ in throw CancellationError() })
        XCTAssertEqual(cancelled.mode, "cancelled")
        let changed = await MemoryStore.reembedStaleEmbeddings(context: context, matching: metadata, embed: { _ in
            EmbeddingRuntimeResult(vector: [1, 0], modelIdentifier: "different-same-dimension-space", formatVersion: 4)
        })
        XCTAssertEqual(changed.diagnostic, "embedding_identity_changed_during_reindex")
        XCTAssertEqual(memory.embedding, [0, 1])
        XCTAssertNil(memory.embeddingModelIdentifier)
        XCTAssertEqual(try context.fetchCount(FetchDescriptor<MemoryItem>()), 1)
    }

    func testEditingRebuiltMemoryInvalidatesItsVectorAndRebuildsTheNewText() async throws {
        allowForeground()
        let container = try memoryContainer()
        let context = ModelContext(container)
        let memory = MemoryItem(content: "Synthetic original preference", embedding: [0, 1])
        context.insert(memory)
        try context.save()
        let originalID = memory.id
        var requests: [EmbeddingRequest] = []
        let embed: (EmbeddingRequest) async throws -> EmbeddingRuntimeResult = { request in
            requests.append(request)
            return EmbeddingRuntimeResult(vector: requests.count == 1 ? [1, 0] : [0, 1], modelIdentifier: self.metadata.modelIdentifier, formatVersion: 4)
        }

        let first = await MemoryStore.reembedStaleEmbeddings(context: context, matching: metadata, embed: embed, writeBudget: DiskWriteBudget())
        XCTAssertEqual(first.rebuilt, 1)
        XCTAssertEqual(MemoryVectorIndex.shared.ensureLoaded(context: context, metadata: metadata).loadedCount, 1)
        MemoryStore.updateContent(memory.content, for: memory)
        XCTAssertTrue(MemoryVectorIndex.matches(memory, metadata: metadata))

        MemoryStore.updateContent("Synthetic edited preference", for: memory)
        XCTAssertEqual(memory.embedding, [1, 0])
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

        let second = await MemoryStore.reembedStaleEmbeddings(context: context, matching: metadata, embed: embed, writeBudget: DiskWriteBudget())
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
        let memory = MemoryItem(content: "Synthetic original", embedding: [0, 1])
        context.insert(memory)
        try context.save()

        let result = await MemoryStore.reembedStaleEmbeddings(context: context, matching: metadata, embed: { _ in
            MemoryStore.updateContent("Synthetic edited while encoding", for: memory)
            return EmbeddingRuntimeResult(vector: [1, 0], modelIdentifier: self.metadata.modelIdentifier, formatVersion: 4)
        }, writeBudget: DiskWriteBudget())

        XCTAssertEqual(result.rebuilt, 0)
        XCTAssertEqual(result.diagnostic, "memory_changed_during_reindex")
        XCTAssertEqual(memory.content, "Synthetic edited while encoding")
        XCTAssertEqual(memory.embedding, [0, 1])
        XCTAssertNil(memory.embeddingModelIdentifier)
    }

    func testDeletionDuringInferenceDoesNotTouchTheDeletedRow() async throws {
        allowForeground()
        let context = ModelContext(try memoryContainer())
        let memory = MemoryItem(content: "Synthetic row deleted while encoding", embedding: [0, 1])
        context.insert(memory)
        try context.save()
        var saves = 0

        let result = await MemoryStore.reembedStaleEmbeddings(context: context, matching: metadata, embed: { _ in
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
        let first = MemoryItem(content: "Synthetic first row", embedding: [0, 1])
        first.createdAt = Date(timeIntervalSince1970: 1)
        let second = MemoryItem(content: "Synthetic later deleted row", embedding: [0, 1])
        second.createdAt = Date(timeIntervalSince1970: 2)
        for item in [first, second] { context.insert(item) }
        try context.save()
        var calls = 0

        let result = await MemoryStore.reembedStaleEmbeddings(context: context, matching: metadata, embed: { _ in
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

    func testReembeddingSkipsOversizedOldestMemoryAndRebuildsTheNextRow() async throws {
        allowForeground()
        let context = ModelContext(try memoryContainer())
        let oversized = MemoryItem(content: "Synthetic oversized legacy memory", embedding: [0, 1])
        oversized.createdAt = Date(timeIntervalSince1970: 1)
        let valid = MemoryItem(content: "Synthetic valid later memory", embedding: [0, 1])
        valid.createdAt = Date(timeIntervalSince1970: 2)
        for item in [oversized, valid] { context.insert(item) }
        try context.save()

        let result = await MemoryStore.reembedStaleEmbeddings(context: context, matching: metadata, embed: { request in
            if request.text.contains("oversized") { throw CoreAITextEmbeddingError.inputTooLong }
            return EmbeddingRuntimeResult(vector: [1, 0], modelIdentifier: self.metadata.modelIdentifier, formatVersion: 4)
        }, writeBudget: DiskWriteBudget())

        XCTAssertEqual(result.attempted, 2)
        XCTAssertEqual(result.rebuilt, 1)
        XCTAssertEqual(result.skipped, 1)
        XCTAssertEqual(result.remaining, 1)
        XCTAssertEqual(result.diagnostic, "memory_inputs_skipped:1")
        XCTAssertNil(result.nextCursor)
        XCTAssertEqual(oversized.embedding, [0, 1])
        XCTAssertNil(oversized.embeddingModelIdentifier)
        XCTAssertTrue(MemoryVectorIndex.matches(valid, metadata: metadata))
        XCTAssertEqual(try context.fetchCount(FetchDescriptor<MemoryItem>()), 2)
    }

    func testReembeddingContinuationAdvancesPastAnEntireSkippedBatch() async throws {
        allowForeground()
        let context = ModelContext(try memoryContainer())
        let oversized = (0..<16).map { index in
            let item = MemoryItem(content: "Synthetic oversized memory \(index)", embedding: [0, 1])
            item.createdAt = Date(timeIntervalSince1970: Double(index))
            context.insert(item)
            return item
        }
        let valid = MemoryItem(content: "Synthetic valid memory after the first batch", embedding: [0, 1])
        valid.createdAt = Date(timeIntervalSince1970: 16)
        context.insert(valid)
        try context.save()
        var calls = 0
        let embed: (EmbeddingRequest) async throws -> EmbeddingRuntimeResult = { request in
            calls += 1
            if request.text.contains("oversized") { throw CoreAITextEmbeddingError.invalidRequest }
            return EmbeddingRuntimeResult(vector: [1, 0], modelIdentifier: self.metadata.modelIdentifier, formatVersion: 4)
        }

        let first = await MemoryStore.reembedStaleEmbeddings(context: context, matching: metadata, limit: 16, embed: embed, writeBudget: DiskWriteBudget())
        XCTAssertEqual(first.attempted, 16)
        XCTAssertEqual(first.skipped, 16)
        XCTAssertEqual(first.rebuilt, 0)
        XCTAssertEqual(first.remaining, 17)
        let cursor = try XCTUnwrap(first.nextCursor)
        XCTAssertEqual(cursor.id, oversized[15].id)

        let second = await MemoryStore.reembedStaleEmbeddings(context: context, matching: metadata, limit: 16, after: cursor, embed: embed, writeBudget: DiskWriteBudget())
        XCTAssertEqual(second.attempted, 1)
        XCTAssertEqual(second.rebuilt, 1)
        XCTAssertEqual(second.skipped, 0)
        XCTAssertEqual(second.remaining, 16)
        XCTAssertNil(second.nextCursor)
        XCTAssertEqual(calls, 17)
        XCTAssertTrue(oversized.allSatisfy { $0.embedding == [0, 1] && $0.embeddingModelIdentifier == nil })
        XCTAssertTrue(MemoryVectorIndex.matches(valid, metadata: metadata))
    }

    func testReembeddingContinuationRetriesTerminalFailureWithoutRepeatingCompletedRows() async throws {
        allowForeground()
        let context = ModelContext(try memoryContainer())
        let first = MemoryItem(content: "Synthetic skipped memory", embedding: [0, 1])
        first.createdAt = Date(timeIntervalSince1970: 1)
        let retry = MemoryItem(content: "Synthetic retry memory", embedding: [0, 1])
        retry.createdAt = Date(timeIntervalSince1970: 2)
        for item in [first, retry] { context.insert(item) }
        try context.save()
        let stopped = await MemoryStore.reembedStaleEmbeddings(context: context, matching: metadata, embed: { request in
            if request.text.contains("skipped") { throw CoreAITextEmbeddingError.inputTooLong }
            throw CoreAITextEmbeddingError.resourceDenied
        })

        XCTAssertEqual(stopped.attempted, 2)
        XCTAssertEqual(stopped.skipped, 1)
        XCTAssertEqual(stopped.rebuilt, 0)
        XCTAssertEqual(stopped.nextCursor?.id, first.id)
        XCTAssertEqual(retry.embedding, [0, 1])
        var requests: [EmbeddingRequest] = []
        let resumed = await MemoryStore.reembedStaleEmbeddings(context: context, matching: metadata, after: stopped.nextCursor, embed: { request in
            requests.append(request)
            return EmbeddingRuntimeResult(vector: [1, 0], modelIdentifier: self.metadata.modelIdentifier, formatVersion: 4)
        }, writeBudget: DiskWriteBudget())
        XCTAssertEqual(resumed.rebuilt, 1)
        XCTAssertEqual(requests.count, 1)
        XCTAssertTrue(requests[0].text.contains("retry"))
        XCTAssertNil(resumed.nextCursor)
    }

    func testReembeddingIgnoresContinuationFromAnotherEmbeddingSpace() async throws {
        allowForeground()
        let context = ModelContext(try memoryContainer())
        let memory = MemoryItem(content: "Synthetic oldest memory", embedding: [0, 1])
        memory.createdAt = Date(timeIntervalSince1970: 1)
        context.insert(memory)
        try context.save()
        let other = RAGEmbeddingIndexMetadata(formatVersion: 4, modelIdentifier: "other-space", dimension: 2)
        let cursor = MemoryStore.ReembeddingCursor(createdAt: Date(timeIntervalSince1970: 100), id: UUID(), metadata: other)
        let result = await MemoryStore.reembedStaleEmbeddings(context: context, matching: metadata, after: cursor, embed: { _ in
            EmbeddingRuntimeResult(vector: [1, 0], modelIdentifier: self.metadata.modelIdentifier, formatVersion: 4)
        }, writeBudget: DiskWriteBudget())
        XCTAssertEqual(result.attempted, 1)
        XCTAssertEqual(result.rebuilt, 1)
        XCTAssertNil(result.nextCursor)
        XCTAssertTrue(MemoryVectorIndex.matches(memory, metadata: metadata))
    }

    func testReembeddingPersistenceFailureRestoresOriginalRowAndReleasesWriteReservation() async throws {
        allowForeground()
        let container = try memoryContainer()
        let context = ModelContext(container)
        let memory = MemoryItem(content: "Synthetic saved memory", embedding: [0, 1])
        context.insert(memory)
        try context.save()
        let budget = DiskWriteBudget(oneMinuteLimit: 65_536)
        enum Failure: Error { case save }
        let result = await MemoryStore.reembedStaleEmbeddings(context: context, matching: metadata, embed: { _ in
            EmbeddingRuntimeResult(vector: [1, 0], modelIdentifier: self.metadata.modelIdentifier, formatVersion: 4)
        }, save: { _ in throw Failure.save }, writeBudget: budget)

        XCTAssertEqual(result.rebuilt, 0)
        XCTAssertEqual(result.mode, "failed")
        XCTAssertTrue(result.diagnostic?.hasPrefix("persist_failed:") == true)
        XCTAssertEqual(memory.embedding, [0, 1])
        XCTAssertNil(memory.embeddingModelIdentifier)
        XCTAssertTrue(budget.canWrite(bytes: 65_536, category: .memory))
    }

    func testBackgroundReembeddingDoesNotInvokeProviderOrChangeRows() async throws {
        ResourceBudgetGate.testSnapshotOverride = .init(scenePhase: .background, lowPowerModeEnabled: false, thermalState: .nominal, recentMemoryWarningCount: 0, lastMemoryWarningAt: nil)
        let container = try memoryContainer()
        let context = ModelContext(container)
        let memory = MemoryItem(content: "Synthetic background memory", embedding: [0, 1])
        context.insert(memory)
        try context.save()
        var calls = 0
        let result = await MemoryStore.reembedStaleEmbeddings(context: context, matching: metadata, embed: { _ in
            calls += 1
            return EmbeddingRuntimeResult(vector: [1, 0], modelIdentifier: self.metadata.modelIdentifier, formatVersion: 4)
        })
        XCTAssertEqual(calls, 0)
        XCTAssertEqual(result.rebuilt, 0)
        XCTAssertEqual(result.mode, "skipped")
        XCTAssertEqual(memory.embedding, [0, 1])
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
