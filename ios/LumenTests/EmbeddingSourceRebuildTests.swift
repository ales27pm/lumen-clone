import Foundation
import SwiftData
import SwiftUI
import XCTest
@testable import Lumen

@MainActor
final class EmbeddingSourceRebuildTests: XCTestCase {
    private let metadata = RAGEmbeddingIndexMetadata(
        formatVersion: 4, modelIdentifier: "coreai:synthetic-source-rebuild", dimension: 2
    )

    override func tearDown() async throws {
        // Each test owns its reset flag; cleanup must also retire any test epoch.
        let epoch = EmbeddingIndexLifecycle.beginReset()
        _ = EmbeddingIndexLifecycle.finishReset(epoch)
        ResourceBudgetGate.testSnapshotOverride = nil
        try await super.tearDown()
    }

    func testEmptyContainerAdvancesThroughResetMemoryAndRAGWithoutInference() async throws {
        allowForeground()
        let container = try sourceContainer()
        let rebuild = EmbeddingSourceRebuild(container: container, prepared: metadata)
        XCTAssertTrue(EmbeddingIndexLifecycle.resetPending)
        XCTAssertFalse(EmbeddingIndexLifecycle.semanticReadsAllowed)

        await rebuild.advance(matching: metadata)
        XCTAssertEqual(rebuild.phase, .reset)
        XCTAssertTrue(EmbeddingIndexLifecycle.resetPending)
        await rebuild.advance(matching: metadata)
        XCTAssertEqual(rebuild.phase, .memory)
        XCTAssertFalse(EmbeddingIndexLifecycle.resetPending)
        await rebuild.advance(matching: metadata)
        XCTAssertEqual(rebuild.phase, .rag)
        await rebuild.advance(matching: metadata)
        XCTAssertEqual(rebuild.phase, .complete)
        XCTAssertTrue(EmbeddingIndexLifecycle.semanticReadsAllowed)
        XCTAssertEqual(try ModelContext(container).fetchCount(FetchDescriptor<MemoryItem>()), 0)
        XCTAssertEqual(try ModelContext(container).fetchCount(FetchDescriptor<RAGChunk>()), 0)
    }

    func testDifferentPreparedIdentityCannotAdvanceOrReleaseReset() async throws {
        allowForeground()
        let rebuild = EmbeddingSourceRebuild(container: try sourceContainer(), prepared: metadata)
        let different = RAGEmbeddingIndexMetadata(
            formatVersion: 4, modelIdentifier: "coreai:other-source-rebuild", dimension: 2
        )
        await rebuild.advance(matching: different)
        XCTAssertEqual(rebuild.phase, .reset)
        XCTAssertTrue(EmbeddingIndexLifecycle.resetPending)
        XCTAssertTrue(rebuild.message.contains("selected embedding space changed"))
    }

    func testSupersededResetSessionCannotAdvanceOrReleaseNewerReset() async throws {
        allowForeground()
        let rebuild = EmbeddingSourceRebuild(container: try sourceContainer(), prepared: metadata)
        let newerEpoch = EmbeddingIndexLifecycle.beginReset()
        await rebuild.advance(matching: metadata)
        XCTAssertEqual(rebuild.phase, .reset)
        XCTAssertTrue(EmbeddingIndexLifecycle.ownsReset(newerEpoch))
        XCTAssertTrue(rebuild.message.contains("Another reset replaced this session"))
    }

    func testCompletedResetSessionCannotResumeAfterNewerSameModelReset() async throws {
        allowForeground()
        let rebuild = EmbeddingSourceRebuild(container: try sourceContainer(), prepared: metadata)
        await rebuild.advance(matching: metadata)
        await rebuild.advance(matching: metadata)
        XCTAssertEqual(rebuild.phase, .memory)

        let newerEpoch = EmbeddingIndexLifecycle.beginReset()
        XCTAssertTrue(EmbeddingIndexLifecycle.finishReset(newerEpoch))
        await rebuild.advance(matching: metadata)
        XCTAssertEqual(rebuild.phase, .memory)
        XCTAssertTrue(rebuild.message.contains("Another reset replaced this session"))
        XCTAssertFalse(EmbeddingIndexLifecycle.resetPending)
    }

    func testDedicatedResetContextsNeverSaveUnrelatedPresentationEdits() async throws {
        allowForeground()
        let container = try sourceContainer()
        let presentationContext = ModelContext(container)
        presentationContext.autosaveEnabled = false
        let conversationSource = MemoryItem(content: "Synthetic pending presentation edit")
        presentationContext.insert(conversationSource)
        XCTAssertTrue(presentationContext.hasChanges)
        let rebuild = EmbeddingSourceRebuild(container: container, prepared: metadata)

        await rebuild.advance(matching: metadata)
        await rebuild.advance(matching: metadata)
        await rebuild.advance(matching: metadata)
        await rebuild.advance(matching: metadata)

        XCTAssertEqual(rebuild.phase, .complete)
        XCTAssertTrue(presentationContext.hasChanges)
        XCTAssertTrue(conversationSource.modelContext === presentationContext)
        XCTAssertEqual(conversationSource.content, "Synthetic pending presentation edit")
        XCTAssertEqual(try ModelContext(container).fetchCount(FetchDescriptor<MemoryItem>()), 0)
    }

    private func sourceContainer() throws -> ModelContainer {
        try ModelContainer(for: MemoryItem.self, RAGChunk.self,
            configurations: ModelConfiguration(isStoredInMemoryOnly: true))
    }

    private func allowForeground() {
        ResourceBudgetGate.testSnapshotOverride = .init(
            scenePhase: .active, lowPowerModeEnabled: false, thermalState: .nominal,
            recentMemoryWarningCount: 0, lastMemoryWarningAt: nil
        )
    }
}
