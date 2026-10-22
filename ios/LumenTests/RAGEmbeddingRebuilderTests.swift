import Foundation
import SwiftData
import SwiftUI
import XCTest
@testable import Lumen

@MainActor
final class RAGEmbeddingRebuilderTests: XCTestCase {
    private let metadata = RAGEmbeddingIndexMetadata(formatVersion: 4, modelIdentifier: "test:source-rebuild", dimension: 2)

    override func tearDown() async throws {
        MemoryVectorIndex.shared.invalidate()
        RAGVectorIndex.shared.invalidate()
        ResourceBudgetGate.testSnapshotOverride = nil
        try await super.tearDown()
    }

    func testBoundedPagesRebuildPersistedNoteFileAndPhotoTextWithoutStagingOrSourceLoss() async throws {
        allowForeground()
        let context = ModelContext(try makeContainer())
        context.autosaveEnabled = false
        let rows = [chunk(index: 3, kind: .photo), chunk(index: 1, kind: .note), chunk(index: 2, kind: .file)]
        let staged = chunk(index: 4, kind: .note)
        staged.sourceType = RAGChunk.replacementStagingSourceType(id: UUID(), kind: .note)
        for row in rows + [staged] { context.insert(row) }
        try context.save()
        let sources = rows.map(SourceSnapshot.init)
        var requests: [EmbeddingRequest] = []
        let first = await RAGEmbeddingRebuilder.rebuildBatch(context: context, matching: metadata, limit: 2, embed: { request in
            requests.append(request)
            return self.vector()
        }, writeBudget: generousBudget())
        XCTAssertNil(first.diagnostic)
        XCTAssertEqual(first.rebuilt, 2)
        XCTAssertEqual(first.scanned, 3)
        XCTAssertTrue(first.hasMore)
        let cursor = try XCTUnwrap(first.nextCursor)
        XCTAssertEqual(cursor.id, uuid(2))
        let second = await RAGEmbeddingRebuilder.rebuildBatch(context: context, matching: metadata, after: cursor, limit: 2, embed: { request in
            requests.append(request)
            return self.vector()
        }, writeBudget: generousBudget())
        XCTAssertNil(second.diagnostic)
        XCTAssertEqual(second.rebuilt, 1)
        XCTAssertFalse(second.hasMore)
        XCTAssertNil(second.nextCursor)
        XCTAssertEqual(requests.count, 3)
        XCTAssertTrue(requests.allSatisfy { $0.purpose == .retrievalDocument(title: "Synthetic source") && $0.dimensions == 2 })
        XCTAssertTrue(requests.contains { $0.text.contains("Synthetic source body 1") })
        XCTAssertTrue(requests.contains { $0.text.contains("Synthetic source body 3") })
        XCTAssertFalse(requests.contains { $0.text.contains("Synthetic source body 4") })
        XCTAssertTrue(staged.embedding.isEmpty)
        XCTAssertEqual(try context.fetchCount(FetchDescriptor<RAGChunk>()), 4)
        for row in rows {
            XCTAssertEqual(SourceSnapshot(row), sources.first { $0.id == row.id })
            XCTAssertEqual(row.embedding, [1, 0])
            XCTAssertEqual(row.embeddingModelIdentifier, metadata.modelIdentifier)
            XCTAssertEqual(row.embeddingFormatVersion, metadata.formatVersion)
            XCTAssertEqual(row.embeddingDimension, metadata.dimension)
        }
        XCTAssertNil(rows.first { $0.kind == .note }?.sourceRef)
    }

    func testFileBackedUUIDTiesVisitEverySourceExactlyOnceAcrossPages() async throws {
        allowForeground()
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("RAGSourcePages-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let schema = Schema([RAGChunk.self])
        let config = ModelConfiguration(schema: schema, url: directory.appendingPathComponent("sources.store"))
        let container = try ModelContainer(for: schema, configurations: [config])
        let context = ModelContext(container)
        context.autosaveEnabled = false
        for index in (1...9).reversed() { context.insert(chunk(index: index, kind: .note)) }
        try context.save()
        var visited: [String] = []
        var cursor: RAGEmbeddingRebuilder.Cursor?
        var complete = false
        for _ in 0..<10 {
            let result = await RAGEmbeddingRebuilder.rebuildBatch(context: context, matching: metadata, after: cursor, limit: 2, embed: { request in
                visited.append(request.text)
                return self.vector()
            }, writeBudget: generousBudget())
            XCTAssertNil(result.diagnostic)
            XCTAssertLessThanOrEqual(result.scanned, 3)
            XCTAssertLessThanOrEqual(result.rebuilt, 2)
            cursor = result.nextCursor
            if !result.hasMore { complete = true; break }
        }
        XCTAssertTrue(complete)
        XCTAssertEqual(visited.count, 9)
        XCTAssertEqual(Set(visited).count, 9)
        // A distinct context confirms committed vector state rather than only
        // checking the rebuilt live instances in the writer's identity map.
        let readContext = ModelContext(container)
        let saved = try readContext.fetch(FetchDescriptor<RAGChunk>())
        XCTAssertEqual(saved.count, 9)
        XCTAssertTrue(saved.allSatisfy { $0.embedding == [1, 0] && $0.embeddingModelIdentifier == metadata.modelIdentifier })
    }

    func testSourceEditAndDeletionDuringInferenceCannotPersistLateVectors() async throws {
        allowForeground()
        for deleting in [false, true] {
            let context = ModelContext(try makeContainer())
            context.autosaveEnabled = false
            let row = chunk(index: 1, kind: .note)
            context.insert(row)
            try context.save()
            var saves = 0
            let result = await RAGEmbeddingRebuilder.rebuildBatch(context: context, matching: metadata, embed: { _ in
                if deleting { context.delete(row) } else { row.content = "Edited source retained" }
                try context.save()
                return self.vector()
            }, save: { _ in saves += 1 }, writeBudget: generousBudget())
            XCTAssertEqual(result.diagnostic, "rag_source_changed_during_rebuild")
            XCTAssertEqual(result.rebuilt, 0)
            XCTAssertEqual(saves, 0)
            if deleting {
                XCTAssertEqual(try context.fetchCount(FetchDescriptor<RAGChunk>()), 0)
            } else {
                XCTAssertEqual(row.content, "Edited source retained")
                XCTAssertTrue(row.embedding.isEmpty)
            }
        }
    }

    func testChangedProviderAndResetEpochRejectLateResultsWithoutSaving() async throws {
        allowForeground()
        let context = ModelContext(try makeContainer())
        context.autosaveEnabled = false
        let row = chunk(index: 1, kind: .note)
        context.insert(row)
        try context.save()
        var saves = 0
        let providerChanged = await RAGEmbeddingRebuilder.rebuildBatch(context: context, matching: metadata, embed: { _ in
            EmbeddingRuntimeResult(vector: [1, 0], modelIdentifier: "test:another-space", formatVersion: 4)
        }, save: { _ in saves += 1 }, writeBudget: generousBudget())
        XCTAssertEqual(providerChanged.diagnostic, "embedding_identity_changed_during_rebuild")
        XCTAssertTrue(row.embedding.isEmpty)
        var resetToken: UUID?
        let reset = await RAGEmbeddingRebuilder.rebuildBatch(context: context, matching: metadata, embed: { _ in
            resetToken = EmbeddingIndexLifecycle.beginReset()
            return self.vector()
        }, save: { _ in saves += 1 }, writeBudget: generousBudget())
        if let resetToken { XCTAssertTrue(EmbeddingIndexLifecycle.finishReset(resetToken)) }
        XCTAssertEqual(reset.diagnostic, "embedding_rebuild_session_changed")
        XCTAssertEqual(reset.rebuilt, 0)
        XCTAssertEqual(saves, 0)
        XCTAssertTrue(row.embedding.isEmpty)
    }

    func testSourceChangedThroughAnotherContextCannotReceiveAStaleVector() async throws {
        allowForeground()
        for deleting in [false, true] {
            let container = try makeContainer()
            let writer = ModelContext(container)
            writer.autosaveEnabled = false
            let row = chunk(index: 1, kind: .note)
            writer.insert(row)
            try writer.save()
            var saves = 0
            let result = await RAGEmbeddingRebuilder.rebuildBatch(context: writer, matching: metadata, embed: { _ in
                let editor = ModelContext(container)
                editor.autosaveEnabled = false
                let id = row.id
                var descriptor = FetchDescriptor<RAGChunk>(predicate: #Predicate { $0.id == id })
                descriptor.fetchLimit = 1
                let edited = try XCTUnwrap(try editor.fetch(descriptor).first)
                if deleting { editor.delete(edited) } else { edited.content = "Edited in another context" }
                try editor.save()
                return self.vector()
            }, save: { _ in saves += 1 }, writeBudget: generousBudget())
            XCTAssertEqual(result.diagnostic, "rag_source_changed_during_rebuild")
            XCTAssertEqual(result.rebuilt, 0)
            XCTAssertEqual(saves, 0)
            let verifier = ModelContext(container)
            let saved = try verifier.fetch(FetchDescriptor<RAGChunk>())
            if deleting {
                XCTAssertTrue(saved.isEmpty)
            } else {
                XCTAssertEqual(saved.first?.content, "Edited in another context")
                XCTAssertTrue(saved.first?.embedding.isEmpty == true)
            }
        }
    }

    func testCancellationAfterInferenceCannotPersistLateVector() async throws {
        allowForeground()
        let context = ModelContext(try makeContainer())
        context.autosaveEnabled = false
        let row = chunk(index: 1, kind: .note)
        context.insert(row)
        try context.save()
        let result = await Task { @MainActor in
            await RAGEmbeddingRebuilder.rebuildBatch(context: context, matching: metadata, embed: { _ in
                withUnsafeCurrentTask { $0?.cancel() }
                return self.vector()
            }, writeBudget: self.generousBudget())
        }.value
        XCTAssertEqual(result.diagnostic, "cancelled")
        XCTAssertEqual(result.rebuilt, 0)
        XCTAssertTrue(row.embedding.isEmpty)
        XCTAssertEqual(row.embeddingModelIdentifier, "")
    }

    func testSaveFailureRestoresOnlyVectorFieldsAndReleasesReservation() async throws {
        allowForeground()
        let context = ModelContext(try makeContainer())
        context.autosaveEnabled = false
        let row = chunk(index: 1, kind: .note)
        context.insert(row)
        try context.save()
        let source = SourceSnapshot(row)
        let budget = generousBudget()
        enum SyntheticFailure: Error { case save }
        let result = await RAGEmbeddingRebuilder.rebuildBatch(context: context, matching: metadata, embed: { _ in self.vector() }, save: { _ in
            XCTAssertEqual(row.embedding, [1, 0])
            throw SyntheticFailure.save
        }, writeBudget: budget)
        XCTAssertTrue(result.diagnostic?.hasPrefix("persist_failed:") == true)
        XCTAssertEqual(result.rebuilt, 0)
        XCTAssertTrue(row.embedding.isEmpty)
        XCTAssertEqual(row.embeddingModelIdentifier, "")
        XCTAssertEqual(row.embeddingFormatVersion, 0)
        XCTAssertEqual(row.embeddingDimension, 0)
        XCTAssertEqual(SourceSnapshot(row), source)
        XCTAssertEqual(budget.snapshot().bytes1Minute, 0)
        XCTAssertNotNil(budget.reserveWrite(bytes: 2_000_000, category: .rag))
    }

    func testOversizedSourceAdvancesCursorAndDoesNotBlockLaterSource() async throws {
        allowForeground()
        let context = ModelContext(try makeContainer())
        context.autosaveEnabled = false
        for index in 1...3 { context.insert(chunk(index: index, kind: .note)) }
        try context.save()
        var calls = 0
        let result = await RAGEmbeddingRebuilder.rebuildBatch(context: context, matching: metadata, limit: 2, embed: { _ in
            calls += 1
            if calls == 1 { throw CoreAITextEmbeddingError.inputTooLong }
            return self.vector()
        }, writeBudget: generousBudget())
        XCTAssertNil(result.diagnostic)
        XCTAssertEqual(result.skipped, 1)
        XCTAssertEqual(result.rebuilt, 1)
        XCTAssertEqual(result.nextCursor?.id, uuid(2))
        XCTAssertTrue(result.hasMore)
        XCTAssertEqual(calls, 2)
    }

    func testUnrelatedEditWhileAwaitingInferenceIsNotImplicitlySavedByRebuild() async throws {
        allowForeground()
        let context = ModelContext(try makeContainer())
        context.autosaveEnabled = false
        let row = chunk(index: 1, kind: .note)
        context.insert(row)
        try context.save()
        var saves = 0
        let result = await RAGEmbeddingRebuilder.rebuildBatch(context: context, matching: metadata, embed: { _ in
            context.insert(self.chunk(index: 2, kind: .file))
            return self.vector()
        }, save: { _ in saves += 1 }, writeBudget: generousBudget())
        XCTAssertEqual(result.diagnostic, "embedding_rebuild_unsaved_changes")
        XCTAssertEqual(result.rebuilt, 0)
        XCTAssertEqual(saves, 0)
        XCTAssertTrue(row.embedding.isEmpty)
        XCTAssertTrue(context.hasChanges)
    }

    func testResetDisablesLoadedCachesAndInvalidatesAllPreviouslyCapturedWriters() throws {
        let context = ModelContext(try ModelContainer(for: MemoryItem.self, RAGChunk.self, configurations: ModelConfiguration(isStoredInMemoryOnly: true)))
        let memory = MemoryItem(content: "Synthetic memory", embedding: [1, 0], embeddingFormatVersion: metadata.formatVersion, embeddingModelIdentifier: metadata.modelIdentifier, embeddingDimension: metadata.dimension)
        let rag = chunk(index: 1, kind: .note)
        rag.embedding = [1, 0]; rag.embeddingFormatVersion = metadata.formatVersion
        rag.embeddingModelIdentifier = metadata.modelIdentifier; rag.embeddingDimension = metadata.dimension
        context.insert(memory); context.insert(rag)
        try context.save()
        XCTAssertEqual(MemoryVectorIndex.shared.ensureLoaded(context: context, metadata: metadata).loadedCount, 1)
        XCTAssertEqual(RAGVectorIndex.shared.ensureLoaded(context: context, metadata: metadata).loadedCount, 1)
        let writer = try XCTUnwrap(EmbeddingIndexLifecycle.captureWrite())
        let token = EmbeddingIndexLifecycle.beginReset()
        defer { _ = EmbeddingIndexLifecycle.finishReset(token) }
        XCTAssertFalse(EmbeddingIndexLifecycle.permitsWrite(writer))
        XCTAssertNil(EmbeddingIndexLifecycle.captureWrite())
        XCTAssertFalse(EmbeddingIndexLifecycle.semanticReadsAllowed)
        XCTAssertTrue(MemoryVectorIndex.shared.search(query: [1, 0], topK: 10).isEmpty)
        XCTAssertTrue(RAGVectorIndex.shared.search(query: [1, 0], topK: 10).isEmpty)
        XCTAssertEqual(MemoryVectorIndex.shared.ensureLoaded(context: context, metadata: metadata).loadedCount, 0)
        XCTAssertEqual(RAGVectorIndex.shared.ensureLoaded(context: context, metadata: metadata).loadedCount, 0)
        XCTAssertFalse(EmbeddingIndexLifecycle.finishReset(UUID()))
        XCTAssertTrue(EmbeddingIndexLifecycle.resetPending)
    }

    private func makeContainer() throws -> ModelContainer {
        try ModelContainer(for: RAGChunk.self, configurations: ModelConfiguration(isStoredInMemoryOnly: true))
    }

    private func vector() -> EmbeddingRuntimeResult {
        EmbeddingRuntimeResult(vector: [1, 0], modelIdentifier: metadata.modelIdentifier, formatVersion: metadata.formatVersion)
    }

    private func generousBudget() -> DiskWriteBudget {
        DiskWriteBudget(oneMinuteLimit: 2_000_000, fifteenMinuteLimit: 20_000_000, dayLimit: 100_000_000)
    }

    private func allowForeground() {
        ResourceBudgetGate.testSnapshotOverride = .init(scenePhase: .active, lowPowerModeEnabled: false, thermalState: .nominal, recentMemoryWarningCount: 0, lastMemoryWarningAt: nil)
    }

    private func uuid(_ index: Int) -> UUID {
        UUID(uuidString: "00000000-0000-0000-0000-\(String(format: "%012d", index))")!
    }

    private func chunk(index: Int, kind: RAGSourceType) -> RAGChunk {
        let row = RAGChunk(content: "Synthetic source body \(index)", sourceType: kind, sourceName: "Synthetic source", sourceRef: kind == .note ? nil : "fixture-source-\(index)", chunkIndex: index, embedding: [], embeddingFormatVersion: 0, embeddingModelIdentifier: "", embeddingDimension: 0)
        row.id = uuid(index)
        row.createdAt = Date(timeIntervalSince1970: 100)
        return row
    }

    private struct SourceSnapshot: Equatable {
        let id: UUID
        let createdAt: Date
        let content: String
        let type: String
        let name: String
        let ref: String?
        let chunkIndex: Int

        init(_ row: RAGChunk) {
            id = row.id; createdAt = row.createdAt; content = row.content
            type = row.sourceType; name = row.sourceName; ref = row.sourceRef; chunkIndex = row.chunkIndex
        }
    }
}
