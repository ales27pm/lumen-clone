import Foundation
import SwiftUI
import XCTest
@testable import Lumen

@MainActor
final class GemmaMemoryEmbeddingPolicyTests: XCTestCase {
    override func tearDown() async throws {
        ResourceBudgetGate.testSnapshotOverride = nil
        try await super.tearDown()
    }

    func testMissingSelectedEmbeddingIsExplicitAndDoesNotChangeChatRuntime() async {
        allowForeground()
        let appState = AppState()
        appState.activeEmbeddingModelID = UUID().uuidString
        let before = await AppLlamaService.shared.loadedChatPath
        await ModelLoader.prepareSelectedMemoryEmbedding(
            snapshot: ModelLoadSnapshot(appState: appState, stored: []), intent: .userChat
        )
        XCTAssertEqual(ModelLoader.memoryEmbeddingReadiness, "selected_model_missing")
        let after = await AppLlamaService.shared.loadedChatPath
        XCTAssertEqual(after, before)
    }

    func testMissingEmbeddingFileDoesNotSubstituteAnotherInstalledModel() async {
        allowForeground()
        let selected = StoredModel(
            name: "Synthetic embedding", repoId: "synthetic/local", fileName: "embedding.gguf",
            sizeBytes: 1, quantization: "Q4", parameters: "synthetic", role: .embedding,
            localPath: FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString).path
        )
        let appState = AppState()
        appState.activeEmbeddingModelID = selected.id.uuidString
        let before = await AppLlamaService.shared.loadedEmbedPath
        await ModelLoader.prepareSelectedMemoryEmbedding(
            snapshot: ModelLoadSnapshot(appState: appState, stored: [selected]), intent: .userVoice
        )
        XCTAssertEqual(ModelLoader.memoryEmbeddingReadiness, "selected_file_missing")
        let after = await AppLlamaService.shared.loadedEmbedPath
        XCTAssertEqual(after, before)
        XCTAssertEqual(appState.activeEmbeddingModelID, selected.id.uuidString)
    }

    func testMaintenanceCannotLoadEmbeddingOrGemmaThroughForegroundPreparation() async {
        allowForeground()
        let appState = AppState()
        for intent in [ModelLoadIntent.appStartup, .diagnostics, .background] {
            await ModelLoader.prepareSelectedMemoryEmbedding(
                snapshot: ModelLoadSnapshot(appState: appState, stored: []), intent: intent
            )
            XCTAssertEqual(ModelLoader.memoryEmbeddingReadiness, "resource_denied")
        }
    }

    private func allowForeground() {
        ResourceBudgetGate.testSnapshotOverride = .init(
            scenePhase: .active, lowPowerModeEnabled: false, thermalState: .nominal,
            recentMemoryWarningCount: 0, lastMemoryWarningAt: nil
        )
    }
}
