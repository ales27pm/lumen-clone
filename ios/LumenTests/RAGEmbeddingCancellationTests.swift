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
