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
    func testMemoryRememberWithDiagnosticsSkipsEmptyContent() async throws {
        let container = try ModelContainer(for: MemoryItem.self, configurations: ModelConfiguration(isStoredInMemoryOnly: true))
        let context = ModelContext(container)

        let result = await MemoryStore.rememberWithDiagnostics("   \n", kind: .fact, source: "test", context: context)

        XCTAssertEqual(result.mode, "skipped")
        XCTAssertEqual(result.diagnostic, "empty_content")
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
        let metadata = RAGEmbeddingIndexMetadata(
            formatVersion: SemanticEmbeddingText.formatVersion,
            modelIdentifier: RAGEmbeddingMetadata.unidentifiedModelIdentifier,
            dimension: 2
        )
        RAGVectorIndex.shared.ensureLoaded(context: context, metadata: metadata)

        let chunk = RAGChunk(content: "test", sourceType: .note, sourceName: "n", sourceRef: nil, chunkIndex: 0, embedding: [0.1, 0.2])
        var pending: [RAGStore.PendingVector] = [
            RAGStore.PendingVector(
                chunk: chunk,
                bucket: RAGSourceType.note.rawValue,
                vector: [0.1, 0.2],
                metadata: metadata
            )
        ]

        let failed = RAGStore.persistAndAppendVectors(
            context: context,
            operation: "test",
            pending: &pending
        ) { _, _, _ in
            throw TestSaveError()
        }
        XCTAssertNil(failed)
        XCTAssertTrue(pending.isEmpty)
        XCTAssertEqual(RAGVectorIndex.shared.count, 0)
        XCTAssertTrue((try? context.fetch(FetchDescriptor<RAGChunk>()))?.isEmpty == true)
    }

    @MainActor
    func testRAGStoreBudgetDeniedSavePreservesPendingWithoutAppending() {
        let container = try! ModelContainer(for: RAGChunk.self, configurations: ModelConfiguration(isStoredInMemoryOnly: true))
        let context = ModelContext(container)
        RAGVectorIndex.shared.invalidate()
        defer { RAGVectorIndex.shared.invalidate() }
        let metadata = RAGEmbeddingIndexMetadata(
            formatVersion: SemanticEmbeddingText.formatVersion,
            modelIdentifier: "llama:sha256:model-a",
            dimension: 2
        )
        _ = RAGVectorIndex.shared.ensureLoaded(context: context, metadata: metadata)
        let chunk = RAGChunk(
            content: "retryable",
            sourceType: .note,
            sourceName: "n",
            embedding: [0.1, 0.2],
            embeddingFormatVersion: metadata.formatVersion,
            embeddingModelIdentifier: metadata.modelIdentifier,
            embeddingDimension: metadata.dimension
        )
        var pending = [RAGStore.PendingVector(
            chunk: chunk,
            bucket: RAGSourceType.note.rawValue,
            vector: chunk.embedding,
            metadata: metadata
        )]

        let result = RAGStore.persistAndAppendVectors(
            context: context,
            operation: "budget-denied",
            pending: &pending,
            save: { _, _, _ in throw RAGStore.PersistenceError.diskWriteBudgetDenied }
        )

        XCTAssertNil(result)
        XCTAssertEqual(pending.count, 1)
        XCTAssertEqual(RAGVectorIndex.shared.count, 0)
        XCTAssertTrue((try? context.fetch(FetchDescriptor<RAGChunk>()))?.isEmpty == true)
        try? context.save()
        XCTAssertTrue((try? context.fetch(FetchDescriptor<RAGChunk>()))?.isEmpty == true)

        let retry = RAGStore.persistAndAppendVectors(
            context: context,
            operation: "budget-retry",
            pending: &pending,
            save: { context, _, _ in try context.save() }
        )
        XCTAssertEqual(retry?.persistedCount, 1)
        XCTAssertTrue(pending.isEmpty)
        XCTAssertEqual(RAGVectorIndex.shared.count, 1)
    }

    @MainActor
    func testRAGStoreEarlyExitFlushesQueuedVectorsAsPartialResult() throws {
        let container = try ModelContainer(for: RAGChunk.self, configurations: ModelConfiguration(isStoredInMemoryOnly: true))
        let context = ModelContext(container)
        RAGVectorIndex.shared.invalidate()
        defer { RAGVectorIndex.shared.invalidate() }
        let metadata = RAGEmbeddingIndexMetadata(
            formatVersion: SemanticEmbeddingText.formatVersion,
            modelIdentifier: "llama:sha256:model-a",
            dimension: 2
        )
        _ = RAGVectorIndex.shared.ensureLoaded(context: context, metadata: metadata)
        let chunk = RAGChunk(
            content: "completed before later embedding failure",
            sourceType: .note,
            sourceName: "n",
            embedding: [0.1, 0.2],
            embeddingFormatVersion: metadata.formatVersion,
            embeddingModelIdentifier: metadata.modelIdentifier,
            embeddingDimension: metadata.dimension
        )
        var pending = [RAGStore.PendingVector(
            chunk: chunk,
            bucket: RAGSourceType.note.rawValue,
            vector: chunk.embedding,
            metadata: metadata
        )]

        let result = RAGStore.persistPendingVectorsForEarlyExit(
            context: context,
            operation: "embedding-failure",
            diagnostic: "embedding_failed:test",
            pending: &pending,
            save: { context, _, _ in try context.save() }
        )

        XCTAssertEqual(result.indexedCount, 1)
        XCTAssertEqual(result.mode, .partial)
        XCTAssertEqual(result.diagnostic, "embedding_failed:test")
        XCTAssertTrue(pending.isEmpty)
        XCTAssertEqual(try context.fetch(FetchDescriptor<RAGChunk>()).count, 1)
        XCTAssertEqual(RAGVectorIndex.shared.count, 1)
    }

    @MainActor
    func testRAGFileEmbeddingFailureFlushesCurrentPendingBatch() async throws {
        struct EmbeddingFailure: Error {}

        let container = try ModelContainer(for: RAGChunk.self, configurations: ModelConfiguration(isStoredInMemoryOnly: true))
        let context = ModelContext(container)
        let fileURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("rag-partial-\(UUID().uuidString).txt")
        try String(repeating: "a", count: RAGStore.chunkSize + 1).write(to: fileURL, atomically: true, encoding: .utf8)
        defer { try? FileManager.default.removeItem(at: fileURL) }

        RAGVectorIndex.shared.invalidate()
        defer { RAGVectorIndex.shared.invalidate() }
        ResourceBudgetGate.testSnapshotOverride = .init(
            scenePhase: .active,
            lowPowerModeEnabled: false,
            thermalState: .nominal,
            recentMemoryWarningCount: 0,
            lastMemoryWarningAt: nil
        )
        defer { ResourceBudgetGate.testSnapshotOverride = nil }

        var embeddingCallCount = 0
        let result = await RAGStore.indexFileWithDiagnostics(
            url: fileURL,
            context: context,
            embed: { _ in
                embeddingCallCount += 1
                guard embeddingCallCount == 1 else { throw EmbeddingFailure() }
                return EmbeddingRuntimeResult(vector: [0.1, 0.2], modelIdentifier: "llama:sha256:model-a")
            },
            save: { context, _, _ in try context.save() }
        )

        XCTAssertEqual(embeddingCallCount, 2)
        XCTAssertEqual(result.indexedCount, 1)
        XCTAssertEqual(result.mode, .partial)
        XCTAssertTrue(result.diagnostic?.hasPrefix("embedding_failed:") == true)
        XCTAssertEqual(try context.fetch(FetchDescriptor<RAGChunk>()).count, 1)
        XCTAssertEqual(RAGVectorIndex.shared.count, 1)
    }

    @MainActor
    func testRAGStorePersistAndAppendVectorsReloadsAfterMetadataRejection() throws {
        let container = try ModelContainer(for: RAGChunk.self, configurations: ModelConfiguration(isStoredInMemoryOnly: true))
        let context = ModelContext(container)
        let loadedMetadata = RAGEmbeddingIndexMetadata(
            formatVersion: SemanticEmbeddingText.formatVersion,
            modelIdentifier: "llama:sha256:model-a",
            dimension: 2
        )
        let persistedMetadata = RAGEmbeddingIndexMetadata(
            formatVersion: SemanticEmbeddingText.formatVersion,
            modelIdentifier: "llama:sha256:model-b",
            dimension: 2
        )
        RAGVectorIndex.shared.invalidate()
        defer { RAGVectorIndex.shared.invalidate() }
        _ = RAGVectorIndex.shared.ensureLoaded(context: context, metadata: loadedMetadata)

        let chunk = RAGChunk(
            content: "model-b row",
            sourceType: .note,
            sourceName: "n",
            embedding: [0.1, 0.2],
            embeddingFormatVersion: persistedMetadata.formatVersion,
            embeddingModelIdentifier: persistedMetadata.modelIdentifier,
            embeddingDimension: persistedMetadata.dimension
        )
        var pending = [RAGStore.PendingVector(
            chunk: chunk,
            bucket: RAGSourceType.note.rawValue,
            vector: chunk.embedding,
            metadata: persistedMetadata
        )]

        let result = try XCTUnwrap(RAGStore.persistAndAppendVectors(
            context: context,
            operation: "metadata-rejection",
            pending: &pending,
            save: { context, _, _ in try context.save() }
        ))

        XCTAssertEqual(result.persistedCount, 1)
        XCTAssertEqual(result.indexState, .reloaded)
        XCTAssertTrue(pending.isEmpty)
        XCTAssertEqual(RAGVectorIndex.shared.count, 1)
    }

    @MainActor
    func testEmbeddingIdentityChangeDiscardsStagedChunksWithoutFalsePersistence() throws {
        let container = try ModelContainer(for: RAGChunk.self, configurations: ModelConfiguration(isStoredInMemoryOnly: true))
        let context = ModelContext(container)
        let metadata = RAGEmbeddingIndexMetadata(
            formatVersion: SemanticEmbeddingText.formatVersion,
            modelIdentifier: "llama:sha256:model-a",
            dimension: 2
        )
        let chunk = RAGChunk(
            content: "staged",
            sourceType: .note,
            sourceName: "n",
            embedding: [0.1, 0.2],
            embeddingModelIdentifier: metadata.modelIdentifier
        )
        var pending = [RAGStore.PendingVector(
            chunk: chunk,
            bucket: RAGSourceType.note.rawValue,
            vector: chunk.embedding,
            metadata: metadata
        )]

        var activeMetadata: RAGEmbeddingIndexMetadata? = metadata
        let changedMetadata = RAGEmbeddingIndexMetadata(
            formatVersion: metadata.formatVersion,
            modelIdentifier: "llama:sha256:model-b",
            dimension: metadata.dimension
        )

        XCTAssertFalse(RAGStore.prepareEmbeddingMetadata(
            changedMetadata,
            active: &activeMetadata,
            pending: &pending,
            context: context
        ))
        XCTAssertTrue(pending.isEmpty)
        try context.save()
        XCTAssertTrue(try context.fetch(FetchDescriptor<RAGChunk>()).isEmpty)
    }

    @MainActor
    func testRAGVectorIndexLoadFailureIsDiagnosticAndRetryable() {
        RAGVectorIndex.shared.invalidate()

        let failed = RAGVectorIndex.shared.ensureLoadedForTests {
            throw SaveError()
        }

        XCTAssertEqual(failed.mode, "failed")
        XCTAssertEqual(failed.loadedCount, 0)
        XCTAssertTrue(failed.diagnostic?.hasPrefix("rag_vector_index_fetch_failed:") == true)
        XCTAssertEqual(RAGVectorIndex.shared.count, 0)
    }

    @MainActor
    func testMemoryVectorIndexLoadFailureIsDiagnosticAndRetryable() {
        MemoryVectorIndex.shared.invalidate()

        let failed = MemoryVectorIndex.shared.ensureLoadedForTests {
            throw SaveError()
        }

        XCTAssertEqual(failed.mode, "failed")
        XCTAssertEqual(failed.loadedCount, 0)
        XCTAssertTrue(failed.diagnostic?.hasPrefix("memory_vector_index_fetch_failed:") == true)
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
            userText: "I prefer deterministic diagnostics.",
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
        XCTAssertTrue(result.diagnostic?.hasPrefix("file_read_failed:") == true)
        XCTAssertFalse(result.diagnostic?.contains(missingURL.path) == true)
    }

    @MainActor
    func testRAGFileExtractionDistinguishesReadFailureFromDecodeFailure() {
        let rawPath = "/private/raw/rag/secret.rtf"
        let url = URL(fileURLWithPath: rawPath)
        let readFailure = RAGStore.extractFileTextWithDiagnosticsForTests(
            url: url,
            readData: { _ in
                throw NSError(domain: rawPath, code: 7)
            },
            attributedString: { _, _ in NSAttributedString(string: "unused") },
            pdfText: { _ in "unused" }
        )

        XCTAssertEqual(readFailure.mode, .failed)
        XCTAssertTrue(readFailure.diagnostic?.hasPrefix("rtf_read_failed:") == true)
        XCTAssertFalse(readFailure.diagnostic?.contains(rawPath) == true)

        let decodeFailure = RAGStore.extractFileTextWithDiagnosticsForTests(
            url: url,
            readData: { _ in Data("{\\rtf1 malformed".utf8) },
            attributedString: { _, _ in
                throw NSError(domain: rawPath, code: 8)
            },
            pdfText: { _ in "unused" }
        )

        XCTAssertEqual(decodeFailure.mode, .failed)
        XCTAssertTrue(decodeFailure.diagnostic?.hasPrefix("rtf_decode_failed:") == true)
        XCTAssertFalse(decodeFailure.diagnostic?.contains(rawPath) == true)
        XCTAssertNotEqual(readFailure.diagnostic, decodeFailure.diagnostic)
    }

    @MainActor
    func testRAGFileExtractionDistinguishesPDFOpenFailure() {
        let rawPath = "/private/raw/rag/secret.pdf"
        let url = URL(fileURLWithPath: rawPath)
        let result = RAGStore.extractFileTextWithDiagnosticsForTests(
            url: url,
            readData: { _ in Data() },
            attributedString: { _, _ in NSAttributedString(string: "unused") },
            pdfText: { _ in
                throw NSError(domain: rawPath, code: 11)
            }
        )

        XCTAssertEqual(result.mode, .failed)
        XCTAssertTrue(result.diagnostic?.hasPrefix("pdf_open_failed:") == true)
        XCTAssertFalse(result.diagnostic?.contains(rawPath) == true)
    }

    @MainActor
    func testRAGCountsAndChunksDiagnosticAPIsDistinguishLoadedEmptyStore() throws {
        let container = try ModelContainer(for: RAGChunk.self, configurations: ModelConfiguration(isStoredInMemoryOnly: true))
        let context = ModelContext(container)

        let counts = RAGStore.countsWithDiagnostics(context: context)
        let chunks = RAGStore.chunksWithDiagnostics(for: .note, context: context)

        XCTAssertEqual(counts.mode, "loaded")
        XCTAssertNil(counts.diagnostic)
        XCTAssertTrue(counts.counts.isEmpty)
        XCTAssertEqual(chunks.mode, "loaded")
        XCTAssertNil(chunks.diagnostic)
        XCTAssertTrue(chunks.chunks.isEmpty)
    }

    @MainActor
    func testMemoryRecallNormalizationPreservesEmptyQueryDiagnostic() async throws {
        let container = try ModelContainer(for: MemoryItem.self, configurations: ModelConfiguration(isStoredInMemoryOnly: true))
        let context = ModelContext(container)
        let routing = IntentRoutingDecision(intent: .chat, allowedToolIDs: [], requiresClarification: false, clarificationPrompt: nil)

        let result = await MemoryRecall.recallAndNormalizeWithDiagnostics(query: "   ", routing: routing, context: context, limit: 8)

        XCTAssertTrue(result.items.isEmpty)
        XCTAssertEqual(result.mode, "empty_query")
        XCTAssertEqual(result.diagnostic, "empty_query")
    }

    @MainActor
    func testMemoryExportFailureDoesNotReturnEmptyArraySuccess() {
        let result = MemoryStore.exportJSONWithDiagnosticsForTests {
            throw SaveError()
        }

        XCTAssertNil(result.json)
        XCTAssertEqual(result.mode, "failed")
        XCTAssertTrue(result.diagnostic?.hasPrefix("export_failed:") == true)
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
    func testTriggerPersistenceFailureDoesNotRenderAsNoResult() {
        let error = NSError(domain: "SwiftData", code: 9, userInfo: [NSLocalizedDescriptionKey: "raw trigger database path"])
        let rendered = TriggerScheduler.triggerPersistenceFailureMessage(error: error)

        XCTAssertTrue(rendered.contains("persistence save failed"))
        XCTAssertFalse(rendered.contains("No result"))
        XCTAssertFalse(rendered.contains("raw trigger database path"))
    }

    @MainActor
    func testTriggerFetchFailureIsExplicitAndSanitized() {
        let error = NSError(domain: "SwiftData", code: 14, userInfo: [NSLocalizedDescriptionKey: "raw trigger fetch database path"])
        let rendered = TriggerScheduler.triggerFetchFailureMessage(error: error)

        XCTAssertTrue(rendered.contains("Trigger fetch failed"))
        XCTAssertFalse(rendered.contains("No result"))
        XCTAssertFalse(rendered.contains("No scheduled runs"))
        XCTAssertFalse(rendered.contains("raw trigger fetch database path"))
    }

    @MainActor
    func testTriggerToolPersistenceMessagesAreExplicitAndSanitized() {
        let error = NSError(domain: "SwiftData", code: 11, userInfo: [NSLocalizedDescriptionKey: "raw trigger tool database path"])
        let fetch = TriggerTools.triggerFetchFailureMessage(error: error)
        let save = TriggerTools.triggerSaveFailureMessage(operation: "create", error: error)

        XCTAssertTrue(fetch.contains("Trigger fetch failed"))
        XCTAssertTrue(save.contains("persistence save failed"))
        XCTAssertFalse(fetch.contains("No scheduled runs"))
        XCTAssertFalse(save.contains("Scheduled"))
        XCTAssertFalse(fetch.contains("raw trigger tool database path"))
        XCTAssertFalse(save.contains("raw trigger tool database path"))
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

    @MainActor
    func testModelCatalogFetchFailureMessageIsExplicitAndSanitized() {
        let error = NSError(domain: "SwiftData", code: 10, userInfo: [NSLocalizedDescriptionKey: "raw model database path"])
        let rendered = ModelLaunchBootstrap.modelCatalogFetchFailureMessage(error: error)

        XCTAssertTrue(rendered.contains("Model catalog fetch failed"))
        XCTAssertFalse(rendered.contains("0 /"))
        XCTAssertFalse(rendered.contains("raw model database path"))
    }

    @MainActor
    func testRemCycleModelCatalogFetchFailureIsDiagnosticAndSanitized() {
        let error = NSError(domain: "SwiftData", code: 12, userInfo: [NSLocalizedDescriptionKey: "raw rem model database path"])
        let snapshot = RemCycleService.storedModelCatalogSnapshotForTests {
            throw error
        }

        XCTAssertTrue(snapshot.stored.isEmpty)
        XCTAssertTrue(snapshot.diagnostic?.contains("Model catalog fetch failed") == true)
        XCTAssertFalse(snapshot.diagnostic?.contains("0 /") == true)
        XCTAssertFalse(snapshot.diagnostic?.contains("raw rem model database path") == true)

        let report = RemCycleReport(
            id: UUID(),
            createdAt: Date(),
            reason: "test",
            runnableV1: false,
            missingSlots: [],
            assignedSlots: [],
            storedModelCount: snapshot.stored.count,
            modelCatalogDiagnostic: snapshot.diagnostic,
            activeChatModelID: nil,
            activeEmbeddingModelID: nil,
            parseFailureSummary: "",
            parseNoiseSummary: ""
        )

        XCTAssertEqual(report.modelCatalogDiagnostic, snapshot.diagnostic)
    }

    @MainActor
    func testModelLoadSnapshotFetchFailureIsDiagnosticAndSanitized() {
        let error = NSError(domain: "SwiftData", code: 13, userInfo: [NSLocalizedDescriptionKey: "raw settings model database path"])
        let result = ModelLoader.modelLoadSnapshotForTests(appState: AppState()) {
            throw error
        }

        XCTAssertNil(result.snapshot)
        XCTAssertFalse(result.isReady)
        XCTAssertTrue(result.diagnostic?.contains("Model catalog fetch failed") == true)
        XCTAssertFalse(result.diagnostic?.contains("0 /") == true)
        XCTAssertFalse(result.diagnostic?.contains("raw settings model database path") == true)
    }
}
