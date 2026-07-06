import XCTest
import SwiftData
import SwiftUI
@testable import Lumen

final class PersistenceAuditTests: XCTestCase {
    private struct SaveError: Error {}

    @MainActor
    func testMemoryStoreFailedSaveSurfacesFailure() {
        let ok = MemoryStore.auditPersistence(operation: "test", scope: "MemoryItem") {}
        XCTAssertTrue(ok)

        let failed = MemoryStore.auditPersistence(operation: "test", scope: "MemoryItem") {
            throw SaveError()
        }
        XCTAssertFalse(failed)
    }

    @MainActor
    func testRAGStoreFailedSaveSurfacesFailure() {
        let failed = RAGStore.auditPersistence(operation: "test", scope: "RAGChunk") {
            throw SaveError()
        }
        XCTAssertFalse(failed)
        let ok = RAGStore.auditPersistence(operation: "test", scope: "RAGChunk") {}
        XCTAssertTrue(ok)
    }

    @MainActor
    func testRAGStorePersistAndAppendVectorsDoesNotAppendOnFailedSave() {
        struct TestSaveError: Error {}
        let container = try! ModelContainer(for: RAGChunk.self, configurations: ModelConfiguration(isStoredInMemoryOnly: true))
        let context = ModelContext(container)
        RAGVectorIndex.shared.invalidate()
        RAGVectorIndex.shared.ensureLoaded(context: context)

        let chunk = RAGChunk(content: "test", sourceType: .note, sourceName: "n", sourceRef: nil, chunkIndex: 0, embedding: [0.1, 0.2])
        context.insert(chunk)
        var pending: [(id: PersistentIdentifier, bucket: String, vector: [Double])] = [
            (id: chunk.persistentModelID, bucket: RAGSourceType.note.rawValue, vector: [0.1, 0.2])
        ]

        let failed = RAGStore.persistAndAppendVectors(
            context: context,
            operation: "test",
            pending: &pending
        ) { _, _, _ in
            throw TestSaveError()
        }
        XCTAssertNil(failed)
        XCTAssertEqual(pending.count, 1)
        XCTAssertEqual(RAGVectorIndex.shared.count, 0)
    }

    @MainActor
    func testRAGStoreResolvedVectorCandidatesSkipsStaleIdentifiers() throws {
        let container = try ModelContainer(for: RAGChunk.self, configurations: ModelConfiguration(isStoredInMemoryOnly: true))
        let context = ModelContext(container)

        let stale = RAGChunk(content: "stale", sourceType: .file, sourceName: "stale", embedding: [1, 0])
        context.insert(stale)
        try context.save()
        let staleID = stale.persistentModelID
        context.delete(stale)
        try context.save()

        let live = RAGChunk(content: "live", sourceType: .file, sourceName: "live", embedding: [1, 0])
        context.insert(live)
        try context.save()

        let resolved = RAGStore.resolvedVectorCandidates(
            vectorHits: [
                (id: staleID, score: 0.99),
                (id: live.persistentModelID, score: 0.75)
            ],
            context: context
        )

        XCTAssertEqual(resolved.count, 1)
        let resolvedCandidate = try XCTUnwrap(resolved.first)
        XCTAssertEqual(resolvedCandidate.0.sourceName, "live")
        XCTAssertEqual(resolvedCandidate.1, 0.75, accuracy: 0.001)
    }

    @MainActor
    func testMemoryExtractionReportsSkippedMaintenanceBudget() async throws {
        ResourceBudgetGate.testSnapshotOverride = .init(
            scenePhase: .active,
            lowPowerModeEnabled: false,
            thermalState: .serious,
            recentMemoryWarningCount: 0,
            lastMemoryWarningAt: nil
        )
        defer { ResourceBudgetGate.testSnapshotOverride = nil }

        let container = try ModelContainer(for: MemoryItem.self, configurations: ModelConfiguration(isStoredInMemoryOnly: true))
        let context = ModelContext(container)

        let result = await MemoryStore.extractAndStore(
            userText: "I love deterministic diagnostics.",
            assistantText: "",
            context: context
        )

        XCTAssertEqual(result.attempted, 0)
        XCTAssertEqual(result.stored, 0)
        XCTAssertEqual(result.failed, 0)
        XCTAssertEqual(result.skipped, 1)
        XCTAssertEqual(result.diagnostics, ["memory_extract_skipped"])
    }

    @MainActor
    func testRAGIndexFileReportsReadFailure() async throws {
        let container = try ModelContainer(for: RAGChunk.self, configurations: ModelConfiguration(isStoredInMemoryOnly: true))
        let context = ModelContext(container)
        let missingURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("missing-rag-\(UUID().uuidString).txt")

        let result = await RAGStore.indexFileWithDiagnostics(url: missingURL, context: context)

        XCTAssertEqual(result.indexedCount, 0)
        XCTAssertEqual(result.mode, .failed)
        XCTAssertEqual(result.diagnostic, "file_read_failed")
    }

    @MainActor
    func testTriggerSchedulerFailedSaveSurfacesFailure() {
        let failed = TriggerScheduler.shared.auditPersistence(operation: "test", scope: "Trigger") {
            throw SaveError()
        }
        XCTAssertFalse(failed)
        let ok = TriggerScheduler.shared.auditPersistence(operation: "test", scope: "Trigger") {}
        XCTAssertTrue(ok)
    }

    @MainActor
    func testModelLaunchBootstrapFailedSaveSurfacesFailure() {
        let failed = ModelLaunchBootstrap.auditPersistence(operation: "test", scope: "StoredModel") {
            throw SaveError()
        }
        XCTAssertFalse(failed)
        let ok = ModelLaunchBootstrap.auditPersistence(operation: "test", scope: "StoredModel") {}
        XCTAssertTrue(ok)
    }
}
