import XCTest
import SwiftData
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
