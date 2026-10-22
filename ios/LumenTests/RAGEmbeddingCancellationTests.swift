import Foundation
import Photos
import SwiftData
import SwiftUI
import XCTest
@testable import Lumen

@MainActor
final class RAGEmbeddingCancellationTests: XCTestCase {
    private let metadata = RAGEmbeddingIndexMetadata(
        formatVersion: 4, modelIdentifier: "coreai:synthetic-cancellation-space", dimension: 2
    )

    override func tearDown() async throws {
        RAGVectorIndex.shared.invalidate()
        ResourceBudgetGate.testSnapshotOverride = nil
        RAGStore.testCPUWatchdogDegradedOverride = nil
        try await super.tearDown()
    }

    func testProviderCancellationStopsBulkReplacementWithoutCancellingItsCaller() async throws {
        allowForeground()
        let container = try ragContainer()
        let context = ModelContext(container)
        let original = try insertOriginal(type: .file, context: context)
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let first = directory.appendingPathComponent("first.txt")
        let second = directory.appendingPathComponent("must-not-be-processed.txt")
        try String(repeating: "Synthetic replacement text. ", count: 4_000)
            .write(to: first, atomically: true, encoding: .utf8)
        try "Synthetic second source".write(to: second, atomically: true, encoding: .utf8)
        var calls = 0
        var progress: [Double] = []
        var saves: [String] = []
        XCTAssertFalse(Task.isCancelled)

        let result = await RAGStore.indexImportedFilesWithDiagnostics(
            context: context,
            progress: { progress.append($0) },
            importedFilesResult: .init(directory: directory, files: [first, second], mode: "loaded", diagnostic: nil),
            embed: { request in
                calls += 1
                XCTAssertEqual(request.purpose, .retrievalDocument(title: "first.txt"))
                // Service replacement cancels its inner inference task. Its
                // CancellationError can propagate to an uncancelled importer.
                if calls == 33 { throw CancellationError() }
                return self.embedding()
            },
            save: { context, operation, _ in
                saves.append(operation)
                try context.save()
            }
        )

        XCTAssertFalse(Task.isCancelled)
        XCTAssertEqual(calls, 33)
        XCTAssertTrue(progress.isEmpty)
        XCTAssertEqual(result.mode, .skipped)
        XCTAssertEqual(result.diagnostic, "cancelled")
        XCTAssertEqual(result.indexedCount, 0)
        XCTAssertEqual(saves, ["indexImportedFiles.stage", "indexImportedFiles.discard"])
        try assertOnlyOriginalRemains(original, context: context)
    }

    func testAlreadyCancelledBulkImportStopsBeforeReadingItsFirstSource() async throws {
        allowForeground()
        let container = try ragContainer()
        let context = ModelContext(container)
        let original = try insertOriginal(type: .file, context: context)
        let missing = FileManager.default.temporaryDirectory
            .appendingPathComponent("missing-\(UUID().uuidString).txt")
        var calls = 0

        let task = Task { @MainActor () -> RAGStore.IndexResult in
            withUnsafeCurrentTask { $0?.cancel() }
            return await RAGStore.indexImportedFilesWithDiagnostics(
                context: context,
                importedFilesResult: .init(
                    directory: missing.deletingLastPathComponent(), files: [missing], mode: "loaded", diagnostic: nil
                ),
                embed: { _ in calls += 1; return self.embedding() },
                save: { context, _, _ in try context.save() }
            )
        }
        let result = await task.value

        // Reading the deliberately missing path would produce a source read
        // failure. Cancellation must win before extraction starts.
        XCTAssertEqual(result.mode, .skipped)
        XCTAssertEqual(result.diagnostic, "cancelled")
        XCTAssertEqual(calls, 0)
        try assertOnlyOriginalRemains(original, context: context)
    }

    func testProviderCancellationDiscardsStagedPhotosAndPreservesCancellationCategory() async throws {
        allowForeground()
        let container = try ragContainer()
        let context = ModelContext(container)
        let original = try insertOriginal(type: .photo, context: context)
        var calls = 0
        var saves: [String] = []

        let result = await RAGStore.indexPhotosWithDiagnostics(
            context: context,
            photoLibrarySnapshot: .init(authorizationStatus: .authorized, assets: []),
            photoIndexSummaries: [
                .init(month: "2026-01", content: "Synthetic first photo summary"),
                .init(month: "2026-02", content: "Synthetic cancelled photo summary"),
                .init(month: "2026-03", content: "Synthetic unattempted photo summary")
            ],
            embed: { _ in
                calls += 1
                if calls == 2 { throw CancellationError() }
                return self.embedding()
            },
            save: { context, operation, _ in
                saves.append(operation)
                try context.save()
            }
        )

        XCTAssertFalse(Task.isCancelled)
        XCTAssertEqual(calls, 2)
        XCTAssertEqual(result.mode, .skipped)
        XCTAssertEqual(result.diagnostic, "cancelled")
        XCTAssertEqual(result.indexedCount, 0)
        XCTAssertEqual(saves, ["indexPhotos.stage", "indexPhotos.discard"])
        try assertOnlyOriginalRemains(original, context: context)
    }

    func testSameModelResetRejectsSlowBulkProducerAndDiscardsOnlyItsStaging() async throws {
        allowForeground()
        let container = try ragContainer()
        let context = ModelContext(container)
        let original = try insertOriginal(type: .file, context: context)
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let file = directory.appendingPathComponent("synthetic.txt")
        try String(repeating: "Synthetic source text. ", count: 4_000)
            .write(to: file, atomically: true, encoding: .utf8)
        var calls = 0
        var saves: [String] = []

        let result = await RAGStore.indexImportedFilesWithDiagnostics(
            context: context,
            importedFilesResult: .init(directory: directory, files: [file], mode: "loaded", diagnostic: nil),
            embed: { _ in
                calls += 1
                if calls == 33 {
                    // A complete reset interval can pass while inference is
                    // suspended, without changing the provider's identity.
                    let resetEpoch = EmbeddingIndexLifecycle.beginReset()
                    XCTAssertTrue(EmbeddingIndexLifecycle.finishReset(resetEpoch))
                }
                return self.embedding()
            },
            save: { context, operation, _ in saves.append(operation); try context.save() }
        )

        XCTAssertEqual(calls, 33)
        XCTAssertEqual(result.mode, .skipped)
        XCTAssertEqual(result.diagnostic, "embedding_index_reset")
        XCTAssertEqual(result.indexedCount, 0)
        XCTAssertEqual(saves, ["indexImportedFiles.stage", "indexImportedFiles.discard"])
        let chunks = try context.fetch(FetchDescriptor<RAGChunk>())
        XCTAssertEqual(chunks.map(\.id), [original])
        XCTAssertEqual(chunks.first?.content, "Synthetic retained source")
        XCTAssertFalse(chunks.contains(where: \.isReplacementStaging))
        XCTAssertEqual(RAGVectorIndex.shared.count, 0)
    }

    func testSameModelResetRejectsPendingSingleFileVectors() async throws {
        allowForeground()
        let container = try ragContainer()
        let context = ModelContext(container)
        let original = try insertOriginal(type: .file, context: context)
        let file = FileManager.default.temporaryDirectory.appendingPathComponent("epoch-\(UUID().uuidString).txt")
        try "Synthetic new source".write(to: file, atomically: true, encoding: .utf8)
        defer { try? FileManager.default.removeItem(at: file) }
        var saves = 0

        let result = await RAGStore.indexFileWithDiagnostics(
            url: file, context: context,
            embed: { _ in
                let resetEpoch = EmbeddingIndexLifecycle.beginReset()
                XCTAssertTrue(EmbeddingIndexLifecycle.finishReset(resetEpoch))
                return self.embedding()
            },
            save: { context, _, _ in saves += 1; try context.save() }
        )

        XCTAssertEqual(result.mode, .skipped)
        XCTAssertEqual(result.diagnostic, "embedding_index_reset")
        XCTAssertEqual(saves, 0)
        XCTAssertEqual(try context.fetch(FetchDescriptor<RAGChunk>()).map(\.id), [original])
    }

    func testSameModelResetRejectsNoteProducerBeforeSaving() async throws {
        allowForeground()
        let container = try ragContainer()
        let context = ModelContext(container)
        let original = try insertOriginal(type: .note, context: context)

        let result = await RAGStore.indexNoteWithDiagnostics(
            title: "Synthetic note", body: "Synthetic new note text", context: context,
            embed: { _ in
                let resetEpoch = EmbeddingIndexLifecycle.beginReset()
                XCTAssertTrue(EmbeddingIndexLifecycle.finishReset(resetEpoch))
                return self.embedding()
            }
        )

        XCTAssertEqual(result.mode, .skipped)
        XCTAssertEqual(result.diagnostic, "embedding_index_reset")
        XCTAssertEqual(try context.fetch(FetchDescriptor<RAGChunk>()).map(\.id), [original])
    }

    func testSameModelResetRejectsRAGIndexerProducerBeforeSaving() async throws {
        let container = try ragContainer()
        let context = ModelContext(container)
        let original = try insertOriginal(type: .note, context: context)

        do {
            _ = try await RAGIndexer().indexText(
                source: .init(id: "synthetic", type: "note", title: "Synthetic", ref: nil),
                title: "Synthetic", text: "Synthetic new source", metadata: [:], context: context,
                embed: { _ in
                    let resetEpoch = EmbeddingIndexLifecycle.beginReset()
                    XCTAssertTrue(EmbeddingIndexLifecycle.finishReset(resetEpoch))
                    return self.embedding()
                }
            )
            XCTFail("An obsolete producer must not insert vectors after reset")
        } catch {
            XCTAssertEqual(error as? RAGIndexingError, .embeddingIndexReset)
        }
        XCTAssertEqual(try context.fetch(FetchDescriptor<RAGChunk>()).map(\.id), [original])
    }

    func testPendingResetRejectsImportBeforeExtractionOrEmbedding() async throws {
        let container = try ragContainer()
        let context = ModelContext(container)
        let resetEpoch = EmbeddingIndexLifecycle.beginReset()
        defer { XCTAssertTrue(EmbeddingIndexLifecycle.finishReset(resetEpoch)) }
        var calls = 0

        let result = await RAGStore.indexFileWithDiagnostics(
            url: FileManager.default.temporaryDirectory.appendingPathComponent("missing-\(UUID().uuidString).txt"),
            context: context, embed: { _ in calls += 1; return self.embedding() }
        )

        XCTAssertEqual(result.mode, .skipped)
        XCTAssertEqual(result.diagnostic, "embedding_index_reset")
        XCTAssertEqual(calls, 0)
        XCTAssertEqual(try context.fetchCount(FetchDescriptor<RAGChunk>()), 0)
    }

    func testSameModelResetDiscardsStagedPhotoProducer() async throws {
        allowForeground()
        let container = try ragContainer()
        let context = ModelContext(container)
        let original = try insertOriginal(type: .photo, context: context)
        var calls = 0
        var saves: [String] = []

        let result = await RAGStore.indexPhotosWithDiagnostics(
            context: context,
            photoLibrarySnapshot: .init(authorizationStatus: .authorized, assets: []),
            photoIndexSummaries: [
                .init(month: "2026-01", content: "Synthetic first photo summary"),
                .init(month: "2026-02", content: "Synthetic obsolete photo summary")
            ],
            embed: { _ in
                calls += 1
                if calls == 2 {
                    let resetEpoch = EmbeddingIndexLifecycle.beginReset()
                    XCTAssertTrue(EmbeddingIndexLifecycle.finishReset(resetEpoch))
                }
                return self.embedding()
            },
            save: { context, operation, _ in saves.append(operation); try context.save() }
        )

        XCTAssertEqual(calls, 2)
        XCTAssertEqual(result.mode, .skipped)
        XCTAssertEqual(result.diagnostic, "embedding_index_reset")
        XCTAssertEqual(saves, ["indexPhotos.stage", "indexPhotos.discard"])
        let chunks = try context.fetch(FetchDescriptor<RAGChunk>())
        XCTAssertEqual(chunks.map(\.id), [original])
        XCTAssertFalse(chunks.contains(where: \.isReplacementStaging))
    }

    func testResetDuringFailedFileInferenceCannotFlushItsEarlierPendingVectors() async throws {
        allowForeground()
        let container = try ragContainer()
        let context = ModelContext(container)
        let original = try insertOriginal(type: .file, context: context)
        let file = FileManager.default.temporaryDirectory.appendingPathComponent("epoch-error-\(UUID().uuidString).txt")
        try String(repeating: "Synthetic new source. ", count: 200)
            .write(to: file, atomically: true, encoding: .utf8)
        defer { try? FileManager.default.removeItem(at: file) }
        var calls = 0
        var saves = 0

        let result = await RAGStore.indexFileWithDiagnostics(
            url: file, context: context,
            embed: { _ in
                calls += 1
                if calls == 2 {
                    let resetEpoch = EmbeddingIndexLifecycle.beginReset()
                    XCTAssertTrue(EmbeddingIndexLifecycle.finishReset(resetEpoch))
                    throw CoreAITextEmbeddingError.inferenceFailed
                }
                return self.embedding()
            },
            save: { context, _, _ in saves += 1; try context.save() }
        )

        XCTAssertEqual(calls, 2)
        XCTAssertEqual(result.mode, .skipped)
        XCTAssertEqual(result.diagnostic, "embedding_index_reset")
        XCTAssertEqual(saves, 0)
        XCTAssertEqual(try context.fetch(FetchDescriptor<RAGChunk>()).map(\.id), [original])
    }

    private func embedding() -> EmbeddingRuntimeResult {
        EmbeddingRuntimeResult(vector: [0, 1], modelIdentifier: metadata.modelIdentifier, formatVersion: metadata.formatVersion)
    }

    private func ragContainer() throws -> ModelContainer {
        try ModelContainer(for: RAGChunk.self, configurations: ModelConfiguration(isStoredInMemoryOnly: true))
    }

    private func insertOriginal(type: RAGSourceType, context: ModelContext) throws -> UUID {
        let original = RAGChunk(
            content: "Synthetic retained source", sourceType: type, sourceName: "original",
            embedding: [1, 0], embeddingFormatVersion: metadata.formatVersion,
            embeddingModelIdentifier: metadata.modelIdentifier, embeddingDimension: metadata.dimension
        )
        context.insert(original)
        try context.save()
        RAGVectorIndex.shared.invalidate()
        XCTAssertEqual(RAGVectorIndex.shared.ensureLoaded(context: context, metadata: metadata).loadedCount, 1)
        return original.id
    }

    private func assertOnlyOriginalRemains(_ original: UUID, context: ModelContext) throws {
        let chunks = try context.fetch(FetchDescriptor<RAGChunk>())
        XCTAssertEqual(chunks.map(\.id), [original])
        XCTAssertFalse(chunks.contains(where: \.isReplacementStaging))
        XCTAssertEqual(chunks.first?.embedding, [1, 0])
        XCTAssertEqual(RAGVectorIndex.shared.count, 1)
    }

    private func allowForeground() {
        RAGStore.testCPUWatchdogDegradedOverride = false
        ResourceBudgetGate.testSnapshotOverride = .init(
            scenePhase: .active, lowPowerModeEnabled: false, thermalState: .nominal,
            recentMemoryWarningCount: 0, lastMemoryWarningAt: nil
        )
    }
}
