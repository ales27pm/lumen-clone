import XCTest
import SwiftUI
import SwiftData
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

    func testSelfImprovementPolicyNeverAllowsModelLoadingAndUsesZeroTokens() {
        let decision = BackgroundTaskPolicy.decide(.init(
            taskKind: .selfImprovement,
            lowPowerMode: false,
            thermalState: .nominal,
            isForeground: false,
            backgroundAgentsEnabled: true,
            requiresNetwork: false,
            estimatedCost: 2
        ))

        XCTAssertTrue(decision.allow)
        XCTAssertFalse(decision.allowModelLoading)
        XCTAssertEqual(decision.maxTokens, 0)
        XCTAssertEqual(decision.maxSteps, 3)
    }

    func testBackgroundProcessingNeverAuthorizesModelLoad() {
        ResourceBudgetGate.testSnapshotOverride = .init(scenePhase: .background, lowPowerModeEnabled: false, thermalState: .nominal, recentMemoryWarningCount: 0, lastMemoryWarningAt: nil)

        XCTAssertFalse(ModelLoader.canStartModelLoad(intent: .background))
        XCTAssertFalse(BackgroundTaskPolicy.decide(.init(
            taskKind: .selfImprovement,
            lowPowerMode: false,
            thermalState: .nominal,
            isForeground: false,
            backgroundAgentsEnabled: true,
            requiresNetwork: false,
            estimatedCost: 2
        )).allowModelLoading)
    }

    func testSelfImprovementMissingContextSkipsSafely() async throws {
        ResourceBudgetGate.testSnapshotOverride = .init(scenePhase: .background, lowPowerModeEnabled: false, thermalState: .nominal, recentMemoryWarningCount: 0, lastMemoryWarningAt: nil)
        let store = metricsStore()
        let loop = SelfImprovementLoop(metricsStore: store, config: .init(cooldownSeconds: 0))

        let outcome = await loop.run(trigger: .backgroundProcessing, container: nil)
        let metric = try await store.recentMetrics(limit: 1).last

        XCTAssertEqual(outcome, .skipped("shared_container_unavailable"))
        XCTAssertEqual(metric?.errorCode, "shared_container_unavailable")
        XCTAssertTrue(metric?.success == true)
        XCTAssertTrue(metric?.policySummary.contains("shared_container_unavailable") == true)
    }

    func testRuntimeSelfImprovementStaysLocalOnlyAndDoesNotLoadOrTrain() async throws {
        ResourceBudgetGate.testSnapshotOverride = .init(scenePhase: .background, lowPowerModeEnabled: false, thermalState: .nominal, recentMemoryWarningCount: 0, lastMemoryWarningAt: nil)
        let store = metricsStore()
        let loop = SelfImprovementLoop(metricsStore: store, config: .init(cooldownSeconds: 0))

        let outcome = await loop.run(trigger: .backgroundProcessing, container: try inMemoryContainer())

        guard case .applied(let summary) = outcome else {
            XCTFail("Expected local maintenance summary, got \(outcome)")
            return
        }
        let lower = summary.lowercased()
        XCTAssertFalse(lower.contains("upload"))
        XCTAssertFalse(lower.contains("train"))
        XCTAssertFalse(lower.contains("weight"))
        XCTAssertFalse(lower.contains("dataset"))
        XCTAssertFalse(ModelLoader.canStartModelLoad(intent: .background))
    }

    func testBackgroundSafeMemoryRecallCanRunWithoutLoadedModel() async {
        let assessment = await BackgroundToolExecutionPolicy.assess(
            prompt: "what do you remember about my workshop preferences",
            modelContext: nil
        )

        XCTAssertEqual(assessment.status, .runnable)
        XCTAssertTrue(assessment.canRunWithoutLoadedTextRuntime)
        XCTAssertTrue(assessment.availableToolIDs.contains("memory.recall"))
    }

    func testNetworkPromptCannotRunWithoutLoadedModelInBackground() async {
        let assessment = await BackgroundToolExecutionPolicy.assess(
            prompt: "search the web for the latest iOS background task docs",
            modelContext: nil
        )

        XCTAssertEqual(assessment.status, .noBackgroundSafeRoutedTools)
        XCTAssertFalse(assessment.canRunWithoutLoadedTextRuntime)
        XCTAssertTrue(assessment.routedToolIDs.contains("web.search"))
        XCTAssertTrue(assessment.availableToolIDs.isEmpty)
    }

    func testForegroundClarificationPromptCannotRunAsBackgroundToolOnly() async {
        let assessment = await BackgroundToolExecutionPolicy.assess(
            prompt: "send email",
            modelContext: nil
        )

        XCTAssertEqual(assessment.status, .clarificationRequired)
        XCTAssertFalse(assessment.canRunWithoutLoadedTextRuntime)
        XCTAssertTrue(assessment.skipMessage.contains("foreground clarification"))
    }

    private func metricsStore() -> RuntimeMetricsStore {
        let url = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("background-no-model-load-\(UUID().uuidString).jsonl")
        return RuntimeMetricsStore(fileURL: url)
    }

    private func inMemoryContainer() throws -> ModelContainer {
        let schema = Schema([
            Conversation.self,
            ChatMessage.self,
            MemoryItem.self,
            StoredModel.self,
            RAGChunk.self,
            Trigger.self,
        ])
        let config = ModelConfiguration(schema: schema, isStoredInMemoryOnly: true)
        return try ModelContainer(for: schema, configurations: [config])
    }
}
