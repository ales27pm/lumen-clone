import XCTest
import SwiftData
import SwiftUI
import Photos
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
    func testEmbeddingIdentityChangePersistsStagedChunksBeforeInvalidatingIndex() throws {
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

        var activeMetadata: RAGEmbeddingIndexMetadata? = metadata
        RAGVectorIndex.shared.invalidate()
        defer { RAGVectorIndex.shared.invalidate() }
        _ = RAGVectorIndex.shared.ensureLoaded(context: context, metadata: metadata)
        let changedMetadata = RAGEmbeddingIndexMetadata(
            formatVersion: metadata.formatVersion,
            modelIdentifier: "llama:sha256:model-b",
            dimension: metadata.dimension
        )
        var saveCallCount = 0

        let result = try XCTUnwrap(RAGStore.prepareEmbeddingMetadata(
            changedMetadata,
            active: &activeMetadata,
            pending: &pending,
            context: context,
            identityChangeOperation: "test.embeddingIdentityChanged",
            previouslyPersistedCount: 2,
            save: { context, _, _ in
                saveCallCount += 1
                try context.save()
            }
        ))
        XCTAssertEqual(result.indexedCount, 3)
        XCTAssertEqual(result.mode, .partial)
        XCTAssertEqual(result.diagnostic, "embedding_identity_changed_during_index")
        XCTAssertEqual(saveCallCount, 1)
        XCTAssertEqual(activeMetadata, metadata)
        XCTAssertTrue(pending.isEmpty)
        XCTAssertEqual(try context.fetch(FetchDescriptor<RAGChunk>()).count, 1)
        XCTAssertEqual(RAGVectorIndex.shared.count, 0)
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
    func testEmptyImportedFileReindexRemovesStaleDocumentCorpus() async throws {
        let container = try ModelContainer(for: RAGChunk.self, configurations: ModelConfiguration(isStoredInMemoryOnly: true))
        let context = ModelContext(container)
        let metadata = RAGEmbeddingIndexMetadata(
            formatVersion: SemanticEmbeddingText.formatVersion,
            modelIdentifier: "llama:sha256:model-a",
            dimension: 2
        )
        context.insert(RAGChunk(
            content: "existing architecture",
            sourceType: .file,
            sourceName: "architecture.txt",
            embedding: [1, 0],
            embeddingModelIdentifier: metadata.modelIdentifier
        ))
        context.insert(RAGChunk(
            content: "existing design",
            sourceType: .pdf,
            sourceName: "design.pdf",
            embedding: [1, 0],
            embeddingModelIdentifier: metadata.modelIdentifier
        ))
        try context.save()
        RAGVectorIndex.shared.invalidate()
        defer { RAGVectorIndex.shared.invalidate() }
        XCTAssertEqual(RAGVectorIndex.shared.ensureLoaded(context: context, metadata: metadata).loadedCount, 2)

        let result = await RAGStore.indexImportedFilesWithDiagnostics(
            context: context,
            importedFilesResult: FileStore.ImportedFilesResult(
                directory: URL(fileURLWithPath: "/tmp/imports", isDirectory: true),
                files: [],
                mode: "loaded",
                diagnostic: "empty_imports"
            )
        )

        XCTAssertEqual(result.mode, .cleared)
        XCTAssertEqual(result.diagnostic, "empty_imports")
        XCTAssertTrue(try context.fetch(FetchDescriptor<RAGChunk>()).isEmpty)
        XCTAssertTrue(RAGStore.lexicalSearch(
            query: "architecture",
            context: context,
            sourceTypes: [.file, .pdf],
            limit: 5
        ).isEmpty)
        XCTAssertTrue(RAGVectorIndex.shared.search(
            query: [1, 0],
            topK: 5,
            allowedBuckets: [RAGSourceType.file.rawValue, RAGSourceType.pdf.rawValue]
        ).isEmpty)
    }

    @MainActor
    func testAlreadyEmptyImportedFileReindexBypassesWriteBudgetAsNoOp() async throws {
        let container = try ModelContainer(for: RAGChunk.self, configurations: ModelConfiguration(isStoredInMemoryOnly: true))
        let context = ModelContext(container)
        DiskWriteBudget.shared.setGenerationActive(true)
        defer { DiskWriteBudget.shared.setGenerationActive(false) }

        let result = await RAGStore.indexImportedFilesWithDiagnostics(
            context: context,
            importedFilesResult: FileStore.ImportedFilesResult(
                directory: URL(fileURLWithPath: "/tmp/imports", isDirectory: true),
                files: [],
                mode: "loaded",
                diagnostic: "empty_imports"
            )
        )

        XCTAssertEqual(result.mode, .cleared)
        XCTAssertEqual(result.diagnostic, "empty_imports")
        XCTAssertTrue(try context.fetch(FetchDescriptor<RAGChunk>()).isEmpty)
    }

    @MainActor
    func testEmptyImportedFileReindexCleansHiddenReplacementOrphans() async throws {
        let container = try ModelContainer(for: RAGChunk.self, configurations: ModelConfiguration(isStoredInMemoryOnly: true))
        let context = ModelContext(container)
        let orphan = RAGChunk(content: "orphan", sourceType: .file, sourceName: "orphan.txt")
        orphan.sourceType = RAGChunk.replacementStagingSourceType(id: UUID(), kind: .file)
        context.insert(orphan)
        try context.save()
        ResourceBudgetGate.testSnapshotOverride = .init(
            scenePhase: .active,
            lowPowerModeEnabled: false,
            thermalState: .nominal,
            recentMemoryWarningCount: 0,
            lastMemoryWarningAt: nil
        )
        RAGStore.testCPUWatchdogDegradedOverride = false
        defer {
            ResourceBudgetGate.testSnapshotOverride = nil
            RAGStore.testCPUWatchdogDegradedOverride = nil
        }

        let result = await RAGStore.indexImportedFilesWithDiagnostics(
            context: context,
            importedFilesResult: .init(
                directory: URL(fileURLWithPath: "/tmp/imports", isDirectory: true),
                files: [],
                mode: "loaded",
                diagnostic: "empty_imports"
            )
        )

        XCTAssertEqual(result.mode, .cleared)
        XCTAssertEqual(result.diagnostic, "empty_imports")
        XCTAssertTrue(try context.fetch(FetchDescriptor<RAGChunk>()).isEmpty)
    }

    @MainActor
    func testEmptyImportedFileReindexDefersHiddenOrphanCleanupWithoutLosingClearedResult() async throws {
        let container = try ModelContainer(for: RAGChunk.self, configurations: ModelConfiguration(isStoredInMemoryOnly: true))
        let context = ModelContext(container)
        let orphan = RAGChunk(content: "orphan", sourceType: .file, sourceName: "orphan.txt")
        orphan.sourceType = RAGChunk.replacementStagingSourceType(id: UUID(), kind: .file)
        context.insert(orphan)
        try context.save()
        let budget = DiskWriteBudget(oneMinuteLimit: 1_000, fifteenMinuteLimit: 2_000, dayLimit: 3_000)
        budget.setGenerationActive(true)
        ResourceBudgetGate.testSnapshotOverride = .init(
            scenePhase: .active,
            lowPowerModeEnabled: false,
            thermalState: .nominal,
            recentMemoryWarningCount: 0,
            lastMemoryWarningAt: nil
        )
        RAGStore.testCPUWatchdogDegradedOverride = false
        defer {
            budget.setGenerationActive(false)
            ResourceBudgetGate.testSnapshotOverride = nil
            RAGStore.testCPUWatchdogDegradedOverride = nil
        }

        let result = await RAGStore.indexImportedFilesWithDiagnostics(
            context: context,
            importedFilesResult: .init(
                directory: URL(fileURLWithPath: "/tmp/imports", isDirectory: true),
                files: [],
                mode: "loaded",
                diagnostic: "empty_imports"
            ),
            writeBudget: budget
        )

        XCTAssertEqual(result.mode, .cleared)
        XCTAssertEqual(
            result.diagnostic,
            "empty_imports;replacement_staging_cleanup_deferred:disk_write_budget_denied"
        )
        XCTAssertEqual(try context.fetch(FetchDescriptor<RAGChunk>()).count, 1)
        XCTAssertTrue(try context.fetch(FetchDescriptor<RAGChunk>()).allSatisfy(\.isReplacementStaging))
    }

    @MainActor
    func testFailedImportedFileEnumerationPreservesExistingDocumentCorpus() async throws {
        let container = try ModelContainer(for: RAGChunk.self, configurations: ModelConfiguration(isStoredInMemoryOnly: true))
        let context = ModelContext(container)
        context.insert(RAGChunk(content: "existing architecture", sourceType: .file, sourceName: "architecture.txt"))
        try context.save()

        let result = await RAGStore.indexImportedFilesWithDiagnostics(
            context: context,
            importedFilesResult: FileStore.ImportedFilesResult(
                directory: URL(fileURLWithPath: "/tmp/imports", isDirectory: true),
                files: [],
                mode: "failed",
                diagnostic: "imports_list_failed:test"
            )
        )

        XCTAssertEqual(result.mode, .failed)
        XCTAssertEqual(result.diagnostic, "imports_list_failed:test")
        XCTAssertEqual(try context.fetch(FetchDescriptor<RAGChunk>()).map(\.sourceName), ["architecture.txt"])
    }

    @MainActor
    func testImportedFileReplacementUsesBoundedStagingThenCommitsNewCorpus() async throws {
        let container = try ModelContainer(for: RAGChunk.self, configurations: ModelConfiguration(isStoredInMemoryOnly: true))
        let context = ModelContext(container)
        let metadata = RAGEmbeddingIndexMetadata(
            formatVersion: SemanticEmbeddingText.formatVersion,
            modelIdentifier: "llama:sha256:model-a",
            dimension: 2
        )
        context.insert(RAGChunk(
            content: "existing architecture",
            sourceType: .file,
            sourceName: "architecture.txt",
            embedding: [1, 0],
            embeddingModelIdentifier: metadata.modelIdentifier
        ))
        context.insert(RAGChunk(content: "keep this note", sourceType: .note, sourceName: "note"))
        try context.save()

        let fileURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("rag-replacement-\(UUID().uuidString).txt")
        try "replacement design".write(to: fileURL, atomically: true, encoding: .utf8)
        defer { try? FileManager.default.removeItem(at: fileURL) }

        RAGVectorIndex.shared.invalidate()
        defer { RAGVectorIndex.shared.invalidate() }
        XCTAssertEqual(RAGVectorIndex.shared.ensureLoaded(context: context, metadata: metadata).loadedCount, 1)
        var saveCallCount = 0
        var saveOperations: [String] = []

        let result = await RAGStore.indexImportedFilesWithDiagnostics(
            context: context,
            importedFilesResult: FileStore.ImportedFilesResult(
                directory: fileURL.deletingLastPathComponent(),
                files: [fileURL],
                mode: "loaded",
                diagnostic: nil
            ),
            embed: { _ in
                EmbeddingRuntimeResult(vector: [0, 1], modelIdentifier: metadata.modelIdentifier)
            },
            save: { context, operation, _ in
                saveCallCount += 1
                saveOperations.append(operation)
                try context.save()
            }
        )

        XCTAssertEqual(result.mode, .indexed)
        XCTAssertEqual(result.indexedCount, 1)
        XCTAssertNil(result.diagnostic)
        XCTAssertEqual(saveCallCount, 2)
        XCTAssertEqual(saveOperations.filter { $0 == "indexImportedFiles.stage" }.count, 1)
        let chunks = try context.fetch(FetchDescriptor<RAGChunk>())
        XCTAssertEqual(Set(chunks.map(\.sourceName)), Set([fileURL.lastPathComponent, "note"]))
        XCTAssertFalse(chunks.contains { $0.sourceName == "architecture.txt" })
        XCTAssertEqual(RAGVectorIndex.shared.count, 1)
    }

    @MainActor
    func testImportedFileReplacementPromotesAndDeletesEveryPageBeyondBatchSize() async throws {
        let container = try ModelContainer(for: RAGChunk.self, configurations: ModelConfiguration(isStoredInMemoryOnly: true))
        let context = ModelContext(container)
        let metadata = RAGEmbeddingIndexMetadata(
            formatVersion: SemanticEmbeddingText.formatVersion,
            modelIdentifier: "llama:sha256:model-a",
            dimension: 2
        )
        for index in 0..<70 {
            context.insert(RAGChunk(
                content: "old \(index)",
                sourceType: .file,
                sourceName: "old-\(index).txt",
                embedding: [1, 0],
                embeddingModelIdentifier: metadata.modelIdentifier
            ))
        }
        try context.save()
        let fileURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("rag-paged-replacement-\(UUID().uuidString).txt")
        try String(repeating: "replacement ", count: 4_000).write(to: fileURL, atomically: true, encoding: .utf8)
        defer { try? FileManager.default.removeItem(at: fileURL) }
        ResourceBudgetGate.testSnapshotOverride = .init(
            scenePhase: .active,
            lowPowerModeEnabled: false,
            thermalState: .nominal,
            recentMemoryWarningCount: 0,
            lastMemoryWarningAt: nil
        )
        defer { ResourceBudgetGate.testSnapshotOverride = nil }

        var saveOperations: [String] = []
        let result = await RAGStore.indexImportedFilesWithDiagnostics(
            context: context,
            importedFilesResult: .init(
                directory: fileURL.deletingLastPathComponent(),
                files: [fileURL],
                mode: "loaded",
                diagnostic: nil
            ),
            embed: { _ in
                EmbeddingRuntimeResult(vector: [0, 1], modelIdentifier: metadata.modelIdentifier)
            },
            save: { context, operation, _ in
                saveOperations.append(operation)
                try context.save()
            }
        )

        XCTAssertEqual(result.mode, .indexed)
        XCTAssertGreaterThan(result.indexedCount, 64)
        XCTAssertGreaterThanOrEqual(saveOperations.filter { $0 == "indexImportedFiles.stage" }.count, 3)
        let chunks = try context.fetch(FetchDescriptor<RAGChunk>())
        XCTAssertEqual(chunks.count, result.indexedCount)
        XCTAssertTrue(chunks.allSatisfy { $0.sourceName == fileURL.lastPathComponent })
        XCTAssertFalse(chunks.contains { $0.isReplacementStaging })
    }

    @MainActor
    func testImportedFileReplacementAccountsSmallBatchesByPayloadInsteadOfFixedCharge() async throws {
        let container = try ModelContainer(for: RAGChunk.self, configurations: ModelConfiguration(isStoredInMemoryOnly: true))
        let context = ModelContext(container)
        context.insert(RAGChunk(content: "existing architecture", sourceType: .file, sourceName: "old.txt"))
        try context.save()
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("rag-small-batch-budget-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let files = try (0..<12).map { index in
            let url = directory.appendingPathComponent("small-\(index).txt")
            try "Lumen module \(index)".write(to: url, atomically: true, encoding: .utf8)
            return url
        }
        let budget = DiskWriteBudget(
            oneMinuteLimit: 400 * 1024,
            fifteenMinuteLimit: 800 * 1024,
            dayLimit: 2 * 1024 * 1024
        )

        let result = await RAGStore.indexImportedFilesWithDiagnostics(
            context: context,
            importedFilesResult: .init(
                directory: directory,
                files: files,
                mode: "loaded",
                diagnostic: nil
            ),
            embed: { _ in
                EmbeddingRuntimeResult(vector: [0, 1], modelIdentifier: "llama:sha256:model-a")
            },
            writeBudget: budget
        )

        XCTAssertEqual(result.mode, .indexed)
        XCTAssertEqual(result.indexedCount, 12)
        XCTAssertLessThan(budget.snapshot().bytes1Minute, 400 * 1024)
        let chunks = try context.fetch(FetchDescriptor<RAGChunk>())
        XCTAssertEqual(chunks.count, 12)
        XCTAssertFalse(chunks.contains { $0.isReplacementStaging })
    }

    @MainActor
    func testGenerationStartingAfterFirstStageUsesReservedCleanupAndPreservesCorpus() async throws {
        let container = try ModelContainer(for: RAGChunk.self, configurations: ModelConfiguration(isStoredInMemoryOnly: true))
        let context = ModelContext(container)
        context.insert(RAGChunk(content: "existing architecture", sourceType: .file, sourceName: "old.txt"))
        try context.save()
        let fileURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("rag-generation-after-stage-\(UUID().uuidString).txt")
        try String(repeating: "replacement module ", count: 4_000).write(to: fileURL, atomically: true, encoding: .utf8)
        defer { try? FileManager.default.removeItem(at: fileURL) }
        let budget = DiskWriteBudget(
            oneMinuteLimit: 10 * 1024 * 1024,
            fifteenMinuteLimit: 20 * 1024 * 1024,
            dayLimit: 40 * 1024 * 1024
        )
        RAGStore.testCPUWatchdogDegradedOverride = false
        defer {
            RAGStore.testCPUWatchdogDegradedOverride = nil
            budget.setGenerationActive(false)
        }
        var embeddingCallCount = 0

        let result = await RAGStore.indexImportedFilesWithDiagnostics(
            context: context,
            importedFilesResult: .init(
                directory: fileURL.deletingLastPathComponent(),
                files: [fileURL],
                mode: "loaded",
                diagnostic: nil
            ),
            embed: { _ in
                embeddingCallCount += 1
                if embeddingCallCount == 33 {
                    budget.setGenerationActive(true)
                }
                return EmbeddingRuntimeResult(vector: [0, 1], modelIdentifier: "llama:sha256:model-a")
            },
            writeBudget: budget
        )

        XCTAssertGreaterThanOrEqual(embeddingCallCount, 64)
        XCTAssertEqual(result.mode, .skipped)
        XCTAssertTrue(result.diagnostic?.hasPrefix("cleanup_deferred:disk_write_budget_denied") == true)
        XCTAssertFalse(result.diagnostic?.contains("replacement_staging_cleanup_") == true)
        let chunks = try context.fetch(FetchDescriptor<RAGChunk>())
        XCTAssertEqual(chunks.map(\.sourceName), ["old.txt"])
        XCTAssertFalse(chunks.contains { $0.isReplacementStaging })
    }

    @MainActor
    func testImportedFileEmbeddingFailurePreservesExistingCorpusAndVectorIndex() async throws {
        struct EmbeddingFailure: Error {}

        let container = try ModelContainer(for: RAGChunk.self, configurations: ModelConfiguration(isStoredInMemoryOnly: true))
        let context = ModelContext(container)
        let metadata = RAGEmbeddingIndexMetadata(
            formatVersion: SemanticEmbeddingText.formatVersion,
            modelIdentifier: "llama:sha256:model-a",
            dimension: 2
        )
        context.insert(RAGChunk(
            content: "existing architecture",
            sourceType: .file,
            sourceName: "architecture.txt",
            embedding: [1, 0],
            embeddingModelIdentifier: metadata.modelIdentifier
        ))
        try context.save()
        let fileURLs = [
            FileManager.default.temporaryDirectory
                .appendingPathComponent("rag-staged-before-failure-\(UUID().uuidString).txt"),
            FileManager.default.temporaryDirectory
                .appendingPathComponent("rag-embedding-failure-\(UUID().uuidString).txt")
        ]
        try "staged replacement".write(to: fileURLs[0], atomically: true, encoding: .utf8)
        try "failing replacement".write(to: fileURLs[1], atomically: true, encoding: .utf8)
        defer { fileURLs.forEach { try? FileManager.default.removeItem(at: $0) } }

        RAGVectorIndex.shared.invalidate()
        defer { RAGVectorIndex.shared.invalidate() }
        XCTAssertEqual(RAGVectorIndex.shared.ensureLoaded(context: context, metadata: metadata).loadedCount, 1)
        var embeddingCallCount = 0

        let result = await RAGStore.indexImportedFilesWithDiagnostics(
            context: context,
            importedFilesResult: FileStore.ImportedFilesResult(
                directory: fileURLs[0].deletingLastPathComponent(),
                files: fileURLs,
                mode: "loaded",
                diagnostic: nil
            ),
            embed: { _ in
                embeddingCallCount += 1
                guard embeddingCallCount == 1 else { throw EmbeddingFailure() }
                return EmbeddingRuntimeResult(vector: [0, 1], modelIdentifier: metadata.modelIdentifier)
            }
        )

        XCTAssertEqual(embeddingCallCount, 2)
        XCTAssertEqual(result.mode, .failed)
        XCTAssertEqual(result.indexedCount, 0)
        XCTAssertTrue(result.diagnostic?.hasPrefix("source_failures=1:embedding_failed:") == true)
        XCTAssertEqual(try context.fetch(FetchDescriptor<RAGChunk>()).map(\.sourceName), ["architecture.txt"])
        XCTAssertEqual(RAGVectorIndex.shared.count, 1)
    }

    @MainActor
    func testImportedFileRebuildCollectsSourceFailuresAndContinuesBeforeRollback() async throws {
        let container = try ModelContainer(for: RAGChunk.self, configurations: ModelConfiguration(isStoredInMemoryOnly: true))
        let context = ModelContext(container)
        context.insert(RAGChunk(content: "existing architecture", sourceType: .file, sourceName: "architecture.txt"))
        try context.save()

        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("rag-source-failures-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let emptyURL = directory.appendingPathComponent("empty.txt")
        let missingURL = directory.appendingPathComponent("missing.txt")
        let validURL = directory.appendingPathComponent("valid.txt")
        try "".write(to: emptyURL, atomically: true, encoding: .utf8)
        try "replacement design".write(to: validURL, atomically: true, encoding: .utf8)
        var embeddingCallCount = 0

        let result = await RAGStore.indexImportedFilesWithDiagnostics(
            context: context,
            importedFilesResult: FileStore.ImportedFilesResult(
                directory: directory,
                files: [emptyURL, missingURL, validURL],
                mode: "loaded",
                diagnostic: nil
            ),
            embed: { _ in
                embeddingCallCount += 1
                return EmbeddingRuntimeResult(vector: [0, 1], modelIdentifier: "llama:sha256:model-a")
            }
        )

        XCTAssertEqual(embeddingCallCount, 1)
        XCTAssertEqual(result.mode, .failed)
        XCTAssertEqual(result.indexedCount, 0)
        XCTAssertTrue(result.diagnostic?.hasPrefix("source_failures=2:") == true)
        XCTAssertTrue(result.diagnostic?.contains("source_modes=skipped:1,failed:1") == true)
        XCTAssertTrue(result.diagnostic?.contains("|skipped|empty_text") == true)
        XCTAssertTrue(result.diagnostic?.contains("|failed|file_read_failed:") == true)
        let chunks = try context.fetch(FetchDescriptor<RAGChunk>())
        XCTAssertEqual(chunks.map(\.sourceName), ["architecture.txt"])
        XCTAssertFalse(chunks.contains { $0.isReplacementStaging })
    }

    @MainActor
    func testImportedFileRebuildCleansEveryEncodedOrphanPageBeforeStaging() async throws {
        let container = try ModelContainer(for: RAGChunk.self, configurations: ModelConfiguration(isStoredInMemoryOnly: true))
        let context = ModelContext(container)
        for index in 0..<70 {
            let orphan = RAGChunk(
                content: "orphan \(index)",
                sourceType: .file,
                sourceName: "orphan-\(index).txt"
            )
            orphan.sourceType = RAGChunk.replacementStagingSourceType(id: UUID(), kind: .file)
            context.insert(orphan)
        }
        try context.save()

        let fileURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("rag-orphan-recovery-\(UUID().uuidString).txt")
        try "replacement architecture".write(to: fileURL, atomically: true, encoding: .utf8)
        defer { try? FileManager.default.removeItem(at: fileURL) }

        let result = await RAGStore.indexImportedFilesWithDiagnostics(
            context: context,
            importedFilesResult: .init(
                directory: fileURL.deletingLastPathComponent(),
                files: [fileURL],
                mode: "loaded",
                diagnostic: nil
            ),
            embed: { _ in
                EmbeddingRuntimeResult(vector: [0, 1], modelIdentifier: "llama:sha256:model-a")
            },
            save: { context, _, _ in try context.save() }
        )

        XCTAssertEqual(result.mode, .indexed)
        let chunks = try context.fetch(FetchDescriptor<RAGChunk>())
        XCTAssertEqual(chunks.count, result.indexedCount)
        XCTAssertTrue(chunks.allSatisfy { $0.sourceName == fileURL.lastPathComponent })
        XCTAssertFalse(chunks.contains { $0.isReplacementStaging })
    }

    @MainActor
    func testImportedFileFailureDiscardsEveryStagingPageAndPreservesOldCorpus() async throws {
        let container = try ModelContainer(for: RAGChunk.self, configurations: ModelConfiguration(isStoredInMemoryOnly: true))
        let context = ModelContext(container)
        context.insert(RAGChunk(content: "existing architecture", sourceType: .file, sourceName: "architecture.txt"))
        try context.save()
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("rag-paged-discard-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let largeURL = directory.appendingPathComponent("large.txt")
        let missingURL = directory.appendingPathComponent("missing.txt")
        try String(repeating: "replacement ", count: 4_000).write(to: largeURL, atomically: true, encoding: .utf8)
        ResourceBudgetGate.testSnapshotOverride = .init(
            scenePhase: .active,
            lowPowerModeEnabled: false,
            thermalState: .nominal,
            recentMemoryWarningCount: 0,
            lastMemoryWarningAt: nil
        )
        defer { ResourceBudgetGate.testSnapshotOverride = nil }
        var embeddingCallCount = 0

        let result = await RAGStore.indexImportedFilesWithDiagnostics(
            context: context,
            importedFilesResult: .init(
                directory: directory,
                files: [largeURL, missingURL],
                mode: "loaded",
                diagnostic: nil
            ),
            embed: { _ in
                embeddingCallCount += 1
                return EmbeddingRuntimeResult(vector: [0, 1], modelIdentifier: "llama:sha256:model-a")
            },
            save: { context, _, _ in try context.save() }
        )

        XCTAssertGreaterThan(embeddingCallCount, 64)
        XCTAssertEqual(result.mode, .failed)
        let chunks = try context.fetch(FetchDescriptor<RAGChunk>())
        XCTAssertEqual(chunks.map(\.sourceName), ["architecture.txt"])
        XCTAssertFalse(chunks.contains { $0.isReplacementStaging })
    }

    @MainActor
    func testImportedFileReplacementSaveFailureRollsBackCorpusAndVectorIndex() async throws {
        let container = try ModelContainer(for: RAGChunk.self, configurations: ModelConfiguration(isStoredInMemoryOnly: true))
        let context = ModelContext(container)
        let metadata = RAGEmbeddingIndexMetadata(
            formatVersion: SemanticEmbeddingText.formatVersion,
            modelIdentifier: "llama:sha256:model-a",
            dimension: 2
        )
        context.insert(RAGChunk(
            content: "existing architecture",
            sourceType: .file,
            sourceName: "architecture.txt",
            embedding: [1, 0],
            embeddingModelIdentifier: metadata.modelIdentifier
        ))
        try context.save()
        let fileURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("rag-save-failure-\(UUID().uuidString).txt")
        try "replacement design".write(to: fileURL, atomically: true, encoding: .utf8)
        defer { try? FileManager.default.removeItem(at: fileURL) }

        RAGVectorIndex.shared.invalidate()
        defer { RAGVectorIndex.shared.invalidate() }
        XCTAssertEqual(RAGVectorIndex.shared.ensureLoaded(context: context, metadata: metadata).loadedCount, 1)

        var saveCallCount = 0
        let result = await RAGStore.indexImportedFilesWithDiagnostics(
            context: context,
            importedFilesResult: FileStore.ImportedFilesResult(
                directory: fileURL.deletingLastPathComponent(),
                files: [fileURL],
                mode: "loaded",
                diagnostic: nil
            ),
            embed: { _ in
                EmbeddingRuntimeResult(vector: [0, 1], modelIdentifier: metadata.modelIdentifier)
            },
            save: { context, _, _ in
                saveCallCount += 1
                if saveCallCount == 2 { throw SaveError() }
                try context.save()
            }
        )

        XCTAssertEqual(result.mode, .failed)
        XCTAssertEqual(result.indexedCount, 0)
        XCTAssertTrue(result.diagnostic?.hasPrefix("replacement_persist_failed:") == true)
        XCTAssertEqual(saveCallCount, 3)
        XCTAssertEqual(try context.fetch(FetchDescriptor<RAGChunk>()).map(\.sourceName), ["architecture.txt"])
        XCTAssertFalse(try context.fetch(FetchDescriptor<RAGChunk>()).contains { $0.isReplacementStaging })
        XCTAssertEqual(RAGVectorIndex.shared.count, 1)
    }

    @MainActor
    func testCancelledImportedFileReplacementPreservesExistingCorpus() async throws {
        let container = try ModelContainer(for: RAGChunk.self, configurations: ModelConfiguration(isStoredInMemoryOnly: true))
        let context = ModelContext(container)
        let metadata = RAGEmbeddingIndexMetadata(
            formatVersion: SemanticEmbeddingText.formatVersion,
            modelIdentifier: "llama:sha256:model-a",
            dimension: 2
        )
        context.insert(RAGChunk(
            content: "existing architecture",
            sourceType: .file,
            sourceName: "architecture.txt",
            embedding: [1, 0],
            embeddingModelIdentifier: metadata.modelIdentifier
        ))
        try context.save()
        let fileURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("rag-cancelled-\(UUID().uuidString).txt")
        try String(repeating: "replacement design ", count: 4_000).write(to: fileURL, atomically: true, encoding: .utf8)
        defer { try? FileManager.default.removeItem(at: fileURL) }
        RAGVectorIndex.shared.invalidate()
        defer { RAGVectorIndex.shared.invalidate() }
        XCTAssertEqual(RAGVectorIndex.shared.ensureLoaded(context: context, metadata: metadata).loadedCount, 1)
        var embeddingCallCount = 0
        var saveOperations: [String] = []

        let task = Task { @MainActor () -> RAGStore.IndexResult in
            return await RAGStore.indexImportedFilesWithDiagnostics(
                context: context,
                importedFilesResult: FileStore.ImportedFilesResult(
                    directory: fileURL.deletingLastPathComponent(),
                    files: [fileURL],
                    mode: "loaded",
                    diagnostic: nil
                ),
                embed: { _ in
                    embeddingCallCount += 1
                    if embeddingCallCount == 33 {
                        withUnsafeCurrentTask { $0?.cancel() }
                    }
                    return EmbeddingRuntimeResult(vector: [0, 1], modelIdentifier: metadata.modelIdentifier)
                },
                save: { context, operation, _ in
                    saveOperations.append(operation)
                    try context.save()
                }
            )
        }
        let result = await task.value

        XCTAssertEqual(result.mode, .skipped)
        XCTAssertEqual(result.indexedCount, 0)
        XCTAssertEqual(result.diagnostic, "cancelled")
        XCTAssertEqual(embeddingCallCount, 33)
        XCTAssertGreaterThanOrEqual(saveOperations.filter { $0 == "indexImportedFiles.stage" }.count, 1)
        XCTAssertTrue(saveOperations.contains("indexImportedFiles.discard"))
        XCTAssertEqual(try context.fetch(FetchDescriptor<RAGChunk>()).map(\.sourceName), ["architecture.txt"])
        XCTAssertFalse(try context.fetch(FetchDescriptor<RAGChunk>()).contains { $0.isReplacementStaging })
        XCTAssertEqual(RAGVectorIndex.shared.count, 1)
    }

    @MainActor
    func testBulkReplacementLeaseRejectsReentrantSingleFileImportAndWipe() async throws {
        let container = try ModelContainer(for: RAGChunk.self, configurations: ModelConfiguration(isStoredInMemoryOnly: true))
        let context = ModelContext(container)
        context.insert(RAGChunk(content: "existing architecture", sourceType: .file, sourceName: "old.txt"))
        try context.save()

        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("rag-reentrant-mutation-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let bulkURL = directory.appendingPathComponent("bulk.txt")
        let nestedURL = directory.appendingPathComponent("nested.txt")
        try "bulk replacement".write(to: bulkURL, atomically: true, encoding: .utf8)
        try "nested import".write(to: nestedURL, atomically: true, encoding: .utf8)

        var nestedResult: RAGStore.IndexResult?
        var wipeError: RAGStore.PersistenceError?
        let result = await RAGStore.indexImportedFilesWithDiagnostics(
            context: context,
            importedFilesResult: .init(
                directory: directory,
                files: [bulkURL],
                mode: "loaded",
                diagnostic: nil
            ),
            embed: { _ in
                nestedResult = await RAGStore.indexFileWithDiagnostics(
                    url: nestedURL,
                    context: context,
                    embed: { _ in
                        XCTFail("The rejected nested import must not embed")
                        return EmbeddingRuntimeResult(vector: [1, 0], modelIdentifier: "llama:sha256:model-a")
                    }
                )
                do {
                    try RAGStore.wipe(.file, context: context)
                } catch let error as RAGStore.PersistenceError {
                    wipeError = error
                }
                return EmbeddingRuntimeResult(vector: [0, 1], modelIdentifier: "llama:sha256:model-a")
            },
            save: { context, _, _ in try context.save() }
        )

        XCTAssertEqual(result.mode, .indexed)
        XCTAssertEqual(nestedResult?.mode, .skipped)
        XCTAssertEqual(nestedResult?.diagnostic, "replacement_already_in_progress")
        XCTAssertEqual(wipeError, .replacementAlreadyInProgress)
        let chunks = try context.fetch(FetchDescriptor<RAGChunk>())
        XCTAssertEqual(chunks.map(\.sourceName), [bulkURL.lastPathComponent])
        XCTAssertFalse(chunks.contains { $0.isReplacementStaging })
    }

    @MainActor
    func testCPUDegradationBeforeCommitRollsBackStagingAndPreservesCorpus() async throws {
        let container = try ModelContainer(for: RAGChunk.self, configurations: ModelConfiguration(isStoredInMemoryOnly: true))
        let context = ModelContext(container)
        context.insert(RAGChunk(content: "existing architecture", sourceType: .file, sourceName: "old.txt"))
        try context.save()
        let fileURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("rag-cpu-transition-\(UUID().uuidString).txt")
        try "replacement architecture".write(to: fileURL, atomically: true, encoding: .utf8)
        defer { try? FileManager.default.removeItem(at: fileURL) }
        RAGStore.testCPUWatchdogDegradedOverride = false
        defer { RAGStore.testCPUWatchdogDegradedOverride = nil }

        var saveOperations: [String] = []
        let result = await RAGStore.indexImportedFilesWithDiagnostics(
            context: context,
            importedFilesResult: .init(
                directory: fileURL.deletingLastPathComponent(),
                files: [fileURL],
                mode: "loaded",
                diagnostic: nil
            ),
            embed: { _ in
                EmbeddingRuntimeResult(vector: [0, 1], modelIdentifier: "llama:sha256:model-a")
            },
            save: { context, operation, _ in
                saveOperations.append(operation)
                try context.save()
                if operation == "indexImportedFiles.stage" {
                    RAGStore.testCPUWatchdogDegradedOverride = true
                }
            }
        )

        XCTAssertEqual(result.mode, .skipped)
        XCTAssertEqual(result.diagnostic, "cpu_watchdog_degraded")
        XCTAssertTrue(saveOperations.contains("indexImportedFiles.stage"))
        XCTAssertTrue(saveOperations.contains("indexImportedFiles.discard"))
        XCTAssertEqual(try context.fetch(FetchDescriptor<RAGChunk>()).map(\.sourceName), ["old.txt"])
        XCTAssertFalse(try context.fetch(FetchDescriptor<RAGChunk>()).contains { $0.isReplacementStaging })
    }

    @MainActor
    func testEmptyPhotoReindexRemovesStalePhotoCorpus() async throws {
        let container = try ModelContainer(for: RAGChunk.self, configurations: ModelConfiguration(isStoredInMemoryOnly: true))
        let context = ModelContext(container)
        let metadata = RAGEmbeddingIndexMetadata(
            formatVersion: SemanticEmbeddingText.formatVersion,
            modelIdentifier: "llama:sha256:model-a",
            dimension: 2
        )
        context.insert(RAGChunk(
            content: "Photos January: coffee shop",
            sourceType: .photo,
            sourceName: "Photos 2026-01",
            embedding: [1, 0],
            embeddingModelIdentifier: metadata.modelIdentifier
        ))
        try context.save()
        RAGVectorIndex.shared.invalidate()
        defer { RAGVectorIndex.shared.invalidate() }
        XCTAssertEqual(RAGVectorIndex.shared.ensureLoaded(context: context, metadata: metadata).loadedCount, 1)

        let result = await RAGStore.indexPhotosWithDiagnostics(
            context: context,
            photoLibrarySnapshot: .init(authorizationStatus: .authorized, assets: [])
        )

        XCTAssertEqual(result.mode, .cleared)
        XCTAssertEqual(result.diagnostic, "empty_photo_library")
        XCTAssertTrue(try context.fetch(FetchDescriptor<RAGChunk>()).isEmpty)
        XCTAssertTrue(RAGStore.lexicalSearch(
            query: "coffee",
            context: context,
            sourceTypes: [.photo],
            limit: 5
        ).isEmpty)
        XCTAssertTrue(RAGVectorIndex.shared.search(
            query: [1, 0],
            topK: 5,
            allowedBuckets: [RAGSourceType.photo.rawValue]
        ).isEmpty)
    }

    @MainActor
    func testAlreadyEmptyPhotoReindexBypassesWriteBudgetAsNoOp() async throws {
        let container = try ModelContainer(for: RAGChunk.self, configurations: ModelConfiguration(isStoredInMemoryOnly: true))
        let context = ModelContext(container)
        DiskWriteBudget.shared.setGenerationActive(true)
        defer { DiskWriteBudget.shared.setGenerationActive(false) }

        let result = await RAGStore.indexPhotosWithDiagnostics(
            context: context,
            photoLibrarySnapshot: .init(authorizationStatus: .authorized, assets: [])
        )

        XCTAssertEqual(result.mode, .cleared)
        XCTAssertEqual(result.diagnostic, "empty_photo_library")
        XCTAssertTrue(try context.fetch(FetchDescriptor<RAGChunk>()).isEmpty)
    }

    @MainActor
    func testNonEmptyPhotoReplacementStagesAndCommitsSummary() async throws {
        let container = try ModelContainer(for: RAGChunk.self, configurations: ModelConfiguration(isStoredInMemoryOnly: true))
        let context = ModelContext(container)
        context.insert(RAGChunk(content: "old photos", sourceType: .photo, sourceName: "Photos 2025-12"))
        try context.save()
        var saveOperations: [String] = []

        let result = await RAGStore.indexPhotosWithDiagnostics(
            context: context,
            photoLibrarySnapshot: .init(authorizationStatus: .authorized, assets: []),
            photoIndexSummaries: [.init(month: "2026-08", content: "Photos (2026-08): 3 items.")],
            embed: { _ in
                EmbeddingRuntimeResult(vector: [0, 1], modelIdentifier: "llama:sha256:model-a")
            },
            save: { context, operation, _ in
                saveOperations.append(operation)
                try context.save()
            }
        )

        XCTAssertEqual(result, .init(indexedCount: 1, mode: .indexed, diagnostic: nil))
        XCTAssertEqual(saveOperations, ["indexPhotos.stage", "indexPhotos.replace"])
        let chunks = try context.fetch(FetchDescriptor<RAGChunk>())
        XCTAssertEqual(chunks.map(\.sourceName), ["Photos 2026-08"])
        XCTAssertFalse(chunks.contains { $0.isReplacementStaging })
    }

    @MainActor
    func testNonEmptyPhotoReplacementCommitFailureDiscardsStageAndPreservesCorpus() async throws {
        let container = try ModelContainer(for: RAGChunk.self, configurations: ModelConfiguration(isStoredInMemoryOnly: true))
        let context = ModelContext(container)
        context.insert(RAGChunk(content: "old photos", sourceType: .photo, sourceName: "Photos 2025-12"))
        try context.save()
        var saveOperations: [String] = []

        let result = await RAGStore.indexPhotosWithDiagnostics(
            context: context,
            photoLibrarySnapshot: .init(authorizationStatus: .authorized, assets: []),
            photoIndexSummaries: [.init(month: "2026-08", content: "Photos (2026-08): 3 items.")],
            embed: { _ in
                EmbeddingRuntimeResult(vector: [0, 1], modelIdentifier: "llama:sha256:model-a")
            },
            save: { context, operation, _ in
                saveOperations.append(operation)
                if operation == "indexPhotos.replace" { throw SaveError() }
                try context.save()
            }
        )

        XCTAssertEqual(result.mode, .failed)
        XCTAssertTrue(result.diagnostic?.hasPrefix("replacement_persist_failed:") == true)
        XCTAssertEqual(saveOperations, ["indexPhotos.stage", "indexPhotos.replace", "indexPhotos.discard"])
        let chunks = try context.fetch(FetchDescriptor<RAGChunk>())
        XCTAssertEqual(chunks.map(\.sourceName), ["Photos 2025-12"])
        XCTAssertFalse(chunks.contains { $0.isReplacementStaging })
    }

    @MainActor
    func testReplacementStagingChunksStayHiddenFromLiveCorpusViews() throws {
        let container = try ModelContainer(for: RAGChunk.self, configurations: ModelConfiguration(isStoredInMemoryOnly: true))
        let context = ModelContext(container)
        let metadata = RAGEmbeddingIndexMetadata(
            formatVersion: SemanticEmbeddingText.formatVersion,
            modelIdentifier: "llama:sha256:model-a",
            dimension: 2
        )
        let active = RAGChunk(
            content: "active architecture",
            sourceType: .file,
            sourceName: "active.txt",
            embedding: [1, 0],
            embeddingModelIdentifier: metadata.modelIdentifier
        )
        let staged = RAGChunk(
            content: "hidden replacement",
            sourceType: .file,
            sourceName: "staged.txt",
            embedding: [0, 1],
            embeddingModelIdentifier: metadata.modelIdentifier
        )
        staged.sourceType = RAGChunk.replacementStagingSourceType(id: UUID(), kind: .file)
        context.insert(active)
        context.insert(staged)
        try context.save()
        RAGVectorIndex.shared.invalidate()
        defer { RAGVectorIndex.shared.invalidate() }

        XCTAssertEqual(RAGStore.counts(context: context)[.file], 1)
        XCTAssertEqual(RAGStore.chunks(for: .file, context: context).map(\.sourceName), ["active.txt"])
        XCTAssertTrue(RAGStore.lexicalSearch(
            query: "hidden",
            context: context,
            sourceTypes: [.file],
            limit: 5
        ).isEmpty)
        XCTAssertEqual(RAGVectorIndex.shared.ensureLoaded(context: context, metadata: metadata).loadedCount, 1)
    }

    @MainActor
    func testDeniedPhotoAuthorizationPreservesExistingPhotoCorpus() async throws {
        let container = try ModelContainer(for: RAGChunk.self, configurations: ModelConfiguration(isStoredInMemoryOnly: true))
        let context = ModelContext(container)
        context.insert(RAGChunk(content: "existing photos", sourceType: .photo, sourceName: "Photos 2026-01"))
        try context.save()

        let result = await RAGStore.indexPhotosWithDiagnostics(
            context: context,
            photoLibrarySnapshot: .init(authorizationStatus: .denied, assets: [])
        )

        XCTAssertEqual(result.mode, .failed)
        XCTAssertEqual(result.diagnostic, "photos_permission_denied:denied")
        XCTAssertEqual(try context.fetch(FetchDescriptor<RAGChunk>()).map(\.sourceName), ["Photos 2026-01"])
    }

    @MainActor
    func testEmptyPhotoReplacementSaveFailureRollsBackExistingPhotoCorpus() async throws {
        let container = try ModelContainer(for: RAGChunk.self, configurations: ModelConfiguration(isStoredInMemoryOnly: true))
        let context = ModelContext(container)
        context.insert(RAGChunk(content: "existing photos", sourceType: .photo, sourceName: "Photos 2026-01"))
        try context.save()

        let result = await RAGStore.indexPhotosWithDiagnostics(
            context: context,
            photoLibrarySnapshot: .init(authorizationStatus: .authorized, assets: []),
            save: { _, _, _ in throw SaveError() }
        )

        XCTAssertEqual(result.mode, .failed)
        XCTAssertEqual(result.indexedCount, 0)
        XCTAssertTrue(result.diagnostic?.hasPrefix("replacement_persist_failed:") == true)
        XCTAssertEqual(try context.fetch(FetchDescriptor<RAGChunk>()).map(\.sourceName), ["Photos 2026-01"])
    }

    @MainActor
    func testEmptyPhotoReplacementBudgetDenialPreservesExistingPhotoCorpus() async throws {
        let container = try ModelContainer(for: RAGChunk.self, configurations: ModelConfiguration(isStoredInMemoryOnly: true))
        let context = ModelContext(container)
        context.insert(RAGChunk(content: "existing photos", sourceType: .photo, sourceName: "Photos 2026-01"))
        try context.save()
        DiskWriteBudget.shared.setGenerationActive(true)
        defer { DiskWriteBudget.shared.setGenerationActive(false) }

        let result = await RAGStore.indexPhotosWithDiagnostics(
            context: context,
            photoLibrarySnapshot: .init(authorizationStatus: .authorized, assets: [])
        )

        XCTAssertEqual(result.mode, .skipped)
        XCTAssertEqual(result.indexedCount, 0)
        XCTAssertEqual(result.diagnostic, "cleanup_deferred:disk_write_budget_denied")
        XCTAssertEqual(try context.fetch(FetchDescriptor<RAGChunk>()).map(\.sourceName), ["Photos 2026-01"])
    }

    @MainActor
    func testImportedFileCleanupBudgetDenialIsDeferredAndPreservesCorpus() async throws {
        let container = try ModelContainer(for: RAGChunk.self, configurations: ModelConfiguration(isStoredInMemoryOnly: true))
        let context = ModelContext(container)
        context.insert(RAGChunk(content: "existing architecture", sourceType: .file, sourceName: "architecture.txt"))
        try context.save()
        let fileURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("rag-budget-denied-\(UUID().uuidString).txt")
        try "replacement design".write(to: fileURL, atomically: true, encoding: .utf8)
        defer { try? FileManager.default.removeItem(at: fileURL) }
        DiskWriteBudget.shared.setGenerationActive(true)
        defer { DiskWriteBudget.shared.setGenerationActive(false) }

        var embedWasCalled = false

        let result = await RAGStore.indexImportedFilesWithDiagnostics(
            context: context,
            importedFilesResult: FileStore.ImportedFilesResult(
                directory: fileURL.deletingLastPathComponent(),
                files: [fileURL],
                mode: "loaded",
                diagnostic: nil
            ),
            embed: { _ in
                embedWasCalled = true
                return EmbeddingRuntimeResult(vector: [0, 1], modelIdentifier: "llama:sha256:model-a")
            }
        )

        XCTAssertEqual(result.mode, .skipped)
        XCTAssertEqual(result.diagnostic, "cleanup_deferred:disk_write_budget_denied")
        XCTAssertTrue(embedWasCalled)
        XCTAssertEqual(try context.fetch(FetchDescriptor<RAGChunk>()).map(\.sourceName), ["architecture.txt"])
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
