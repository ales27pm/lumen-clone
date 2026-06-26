import XCTest
import SwiftUI
@testable import Lumen

@MainActor
final class ResourceBudgetGateTests: XCTestCase {
    override func tearDown() async throws {
        ResourceBudgetGate.testSnapshotOverride = nil
        try await super.tearDown()
    }

    func testInactiveAndBackgroundDoNotCancelRuntimeWork() {
        XCTAssertFalse(ResourceBudgetGate.shouldCancelForScenePhase(.inactive))
        XCTAssertFalse(ResourceBudgetGate.shouldCancelForScenePhase(.background))
        XCTAssertFalse(ResourceBudgetGate.shouldCancelForScenePhase(.active))
    }

    func testLowPowerDoesNotDenyHeavyWork() {
        ResourceBudgetGate.testSnapshotOverride = .init(scenePhase: .active, lowPowerModeEnabled: true, thermalState: .nominal, recentMemoryWarningCount: 0, lastMemoryWarningAt: nil)
        XCTAssertTrue(ResourceBudgetGate.allowsHeavyModelWork(reason: ModelLoadIntent.diagnostics.rawValue))
        XCTAssertTrue(ResourceBudgetGate.allowsHeavyModelWork(reason: ModelLoadIntent.userChat.rawValue))
    }

    func testForegroundInteractiveBudgetReasonsCoverNominalSeriousBackgroundAndLowPower() {
        let nominal = ResourceBudgetGate.Snapshot(scenePhase: .active, lowPowerModeEnabled: false, thermalState: .nominal, recentMemoryWarningCount: 0, lastMemoryWarningAt: nil)
        XCTAssertNil(ResourceBudgetGate.heavyModelWorkDenialReason(snapshot: nominal, reason: "strict-live-training.executor-preflight"))

        let serious = ResourceBudgetGate.Snapshot(scenePhase: .active, lowPowerModeEnabled: false, thermalState: .serious, recentMemoryWarningCount: 0, lastMemoryWarningAt: nil)
        XCTAssertEqual(
            ResourceBudgetGate.heavyModelWorkDenialReason(snapshot: serious, reason: "strict-live-training.executor-preflight"),
            "strict-live-training.executor-preflight: thermalState=serious; device thermal state serious; cool device and retry"
        )

        let background = ResourceBudgetGate.Snapshot(scenePhase: .background, lowPowerModeEnabled: false, thermalState: .nominal, recentMemoryWarningCount: 0, lastMemoryWarningAt: nil)
        XCTAssertEqual(
            ResourceBudgetGate.heavyModelWorkDenialReason(snapshot: background, reason: "strict-live-training.executor-preflight"),
            "strict-live-training.executor-preflight: scenePhase=background"
        )

        let lowPower = ResourceBudgetGate.Snapshot(scenePhase: .active, lowPowerModeEnabled: true, thermalState: .nominal, recentMemoryWarningCount: 0, lastMemoryWarningAt: nil)
        XCTAssertNil(ResourceBudgetGate.heavyModelWorkDenialReason(snapshot: lowPower, reason: "strict-live-training.executor-preflight"))
    }

    func testSeriousAndCriticalThermalDenyHeavyWork() {
        ResourceBudgetGate.testSnapshotOverride = .init(scenePhase: .active, lowPowerModeEnabled: false, thermalState: .serious, recentMemoryWarningCount: 0, lastMemoryWarningAt: nil)
        XCTAssertFalse(ResourceBudgetGate.allowsHeavyModelWork(reason: ModelLoadIntent.userChat.rawValue))
        ResourceBudgetGate.testSnapshotOverride = .init(scenePhase: .active, lowPowerModeEnabled: false, thermalState: .critical, recentMemoryWarningCount: 0, lastMemoryWarningAt: nil)
        XCTAssertFalse(ResourceBudgetGate.allowsHeavyModelWork(reason: ModelLoadIntent.userVoice.rawValue))
    }

    func testStaleMemoryWarningAllowsLaterExplicitUserWork() {
        let staleWarning = Date().addingTimeInterval(-(MemoryPressureMonitor.modelLoadSuppressionInterval + 1))
        ResourceBudgetGate.testSnapshotOverride = .init(scenePhase: .active, lowPowerModeEnabled: false, thermalState: .nominal, recentMemoryWarningCount: 1, lastMemoryWarningAt: staleWarning)
        XCTAssertTrue(ResourceBudgetGate.allowsHeavyModelWork(reason: ModelLoadIntent.userChat.rawValue))
    }

    func testFreshMemoryWarningAllowsOnlyExplicitLoadedContinuation() {
        let snapshot = ResourceBudgetGate.Snapshot(scenePhase: .active, lowPowerModeEnabled: false, thermalState: .nominal, recentMemoryWarningCount: 1, lastMemoryWarningAt: Date())

        XCTAssertFalse(ResourceBudgetGate.allowsHeavyModelWork(snapshot: snapshot, reason: ModelLoadIntent.userChat.rawValue))
        XCTAssertTrue(ResourceBudgetGate.allowsLoadedForegroundContinuationAfterMemoryPressure(snapshot: snapshot, reason: ModelLoadIntent.userChat.rawValue))
    }

    func testMemoryPressureContinuationStillDeniesUnsafeThermalState() {
        let snapshot = ResourceBudgetGate.Snapshot(scenePhase: .active, lowPowerModeEnabled: false, thermalState: .serious, recentMemoryWarningCount: 1, lastMemoryWarningAt: Date())

        XCTAssertFalse(ResourceBudgetGate.allowsLoadedForegroundContinuationAfterMemoryPressure(snapshot: snapshot, reason: ModelLoadIntent.userChat.rawValue))
    }

    func testGenerateRequestCappedReasoningPreservesMemoryPressureContinuation() {
        let request = GenerateRequest(
            systemPrompt: "sys",
            history: [],
            userMessage: "user",
            temperature: 0,
            topP: 1,
            repetitionPenalty: 1,
            maxTokens: 999,
            modelName: "agent-json",
            relevantMemories: [],
            responseFormat: .constrainedJSON(schema: AgentService.structuredAgentResponseSchema),
            developerTraceModeEnabled: true,
            reasoningCaptureEnabled: true,
            allowsMemoryPressureContinuation: true
        )

        let capped = request.cappedForDeveloperReasoning()

        XCTAssertEqual(capped.maxTokens, 768)
        XCTAssertTrue(capped.allowsMemoryPressureContinuation)
        XCTAssertEqual(capped.responseFormat, request.responseFormat)
    }

    func testMemoryPressureMonitorAgesOutWarningCount() {
        let metricsURL = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let monitor = MemoryPressureMonitor(metricsStore: RuntimeMetricsStore(fileURL: metricsURL), notificationCenter: NotificationCenter())
        let staleWarning = Date().addingTimeInterval(-(MemoryPressureMonitor.modelLoadSuppressionInterval + 1))
        monitor.recordWarningForTesting(count: 1, at: staleWarning)

        XCTAssertEqual(monitor.recentWarningCount(), 0)
        XCTAssertEqual(monitor.warningCount, 0)
        XCTAssertNil(monitor.lastWarningAt)
    }

    func testUnknownAndMemoryWarningDenyHeavyWork() {
        ResourceBudgetGate.testSnapshotOverride = .init(scenePhase: nil, lowPowerModeEnabled: false, thermalState: .nominal, recentMemoryWarningCount: 0, lastMemoryWarningAt: nil)
        XCTAssertFalse(ResourceBudgetGate.allowsHeavyModelWork(reason: ModelLoadIntent.userChat.rawValue))
        ResourceBudgetGate.testSnapshotOverride = .init(scenePhase: .active, lowPowerModeEnabled: false, thermalState: .unknown, recentMemoryWarningCount: 0, lastMemoryWarningAt: nil)
        XCTAssertFalse(ResourceBudgetGate.allowsHeavyModelWork(reason: ModelLoadIntent.userChat.rawValue))
        ResourceBudgetGate.testSnapshotOverride = .init(scenePhase: .active, lowPowerModeEnabled: false, thermalState: .nominal, recentMemoryWarningCount: 1, lastMemoryWarningAt: Date())
        XCTAssertFalse(ResourceBudgetGate.allowsHeavyModelWork(reason: ModelLoadIntent.userChat.rawValue))
    }

    func testSlotBudgetPolicyDecisionKeepsForegroundInteractiveStrict() {
        let snapshot = ResourceBudgetGate.Snapshot(
            scenePhase: .background,
            lowPowerModeEnabled: false,
            thermalState: .nominal,
            recentMemoryWarningCount: 0,
            lastMemoryWarningAt: nil
        )

        let decision = ResourceBudgetGate.decision(
            policy: .foregroundInteractive,
            snapshot: snapshot,
            reason: "slot-runtime.executor"
        )

        XCTAssertFalse(decision.allowed)
        XCTAssertEqual(decision.policy, .foregroundInteractive)
        XCTAssertEqual(decision.denialReason, "slot-runtime.executor: scenePhase=background")
    }

    func testMaintenanceIdlePolicyDeniesLowPowerBackgroundWork() {
        let snapshot = ResourceBudgetGate.Snapshot(
            scenePhase: .background,
            lowPowerModeEnabled: true,
            thermalState: .nominal,
            recentMemoryWarningCount: 0,
            lastMemoryWarningAt: nil
        )

        let decision = ResourceBudgetGate.decision(
            policy: .maintenanceIdle,
            snapshot: snapshot,
            reason: "slot-runtime.rem"
        )

        XCTAssertFalse(decision.allowed)
        XCTAssertEqual(decision.denialReason, "slot-runtime.rem: lowPowerMode=true")
    }

    func testEmbeddingPolicyAllowsForegroundLowPowerButDeniesBackgroundLowPower() {
        let foreground = ResourceBudgetGate.Snapshot(
            scenePhase: .active,
            lowPowerModeEnabled: true,
            thermalState: .nominal,
            recentMemoryWarningCount: 0,
            lastMemoryWarningAt: nil
        )
        let background = ResourceBudgetGate.Snapshot(
            scenePhase: .background,
            lowPowerModeEnabled: true,
            thermalState: .nominal,
            recentMemoryWarningCount: 0,
            lastMemoryWarningAt: nil
        )

        XCTAssertTrue(ResourceBudgetGate.allowsWork(policy: .embedding, snapshot: foreground, reason: "slot-runtime.embedding"))
        XCTAssertEqual(
            ResourceBudgetGate.budgetDenialReason(policy: .embedding, snapshot: background, reason: "slot-runtime.embedding"),
            "slot-runtime.embedding: lowPowerMode=true"
        )
    }
}
