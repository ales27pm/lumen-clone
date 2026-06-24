import XCTest
import SwiftUI
@testable import Lumen

@MainActor
final class BackgroundTaskNoModelLoadTests: XCTestCase {
    override func tearDown() async throws {
        ResourceBudgetGate.testSnapshotOverride = nil
        try await super.tearDown()
    }

    func testBackgroundTaskCannotStartModelLoad() {
        ResourceBudgetGate.testSnapshotOverride = .init(scenePhase: .background, lowPowerModeEnabled: false, thermalState: .nominal, recentMemoryWarningCount: 0, lastMemoryWarningAt: nil)
        XCTAssertFalse(ModelLoader.canStartModelLoad(intent: .background))
        XCTAssertFalse(ResourceBudgetGate.allowsHeavyModelWork(reason: ModelLoadIntent.background.rawValue))
    }

    func testBackgroundSafeMemoryRecallCanRunWithoutLoadedModel() async {
        let assessment = await BackgroundToolBridgePolicy.assess(
            prompt: "what do you remember about my workshop preferences",
            modelContext: nil
        )

        XCTAssertEqual(assessment.status, .runnable)
        XCTAssertTrue(assessment.canRunWithoutLoadedTextRuntime)
        XCTAssertTrue(assessment.availableToolIDs.contains("memory.recall"))
    }

    func testNetworkPromptCannotRunWithoutLoadedModelInBackground() async {
        let assessment = await BackgroundToolBridgePolicy.assess(
            prompt: "search the web for the latest iOS background task docs",
            modelContext: nil
        )

        XCTAssertEqual(assessment.status, .noBackgroundSafeRoutedTools)
        XCTAssertFalse(assessment.canRunWithoutLoadedTextRuntime)
        XCTAssertTrue(assessment.routedToolIDs.contains("web.search"))
        XCTAssertTrue(assessment.availableToolIDs.isEmpty)
    }

    func testForegroundClarificationPromptCannotRunAsBackgroundToolOnly() async {
        let assessment = await BackgroundToolBridgePolicy.assess(
            prompt: "send email",
            modelContext: nil
        )

        XCTAssertEqual(assessment.status, .clarificationRequired)
        XCTAssertFalse(assessment.canRunWithoutLoadedTextRuntime)
        XCTAssertTrue(assessment.skipMessage.contains("foreground clarification"))
    }
}
