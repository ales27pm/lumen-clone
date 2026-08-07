import XCTest
import SwiftUI
@testable import Lumen

final class BackgroundTaskPolicyTests: XCTestCase {
    func testStartupRegistersAndSchedulesWithoutAuthorizationOperation() {
        XCTAssertEqual(BackgroundStartupPolicy.operations, [
            .registerTasks,
            .scheduleTasks,
        ])
    }

    func testCriticalThermalDenied() {
        let d = BackgroundTaskPolicy.decide(.init(taskKind: .triggerScan, lowPowerMode: false, thermalState: .critical, isForeground: false, backgroundAgentsEnabled: true, requiresNetwork: false, estimatedCost: 1))
        XCTAssertFalse(d.allow)
    }

    func testSeriousThermalDenied() {
        let d = BackgroundTaskPolicy.decide(.init(taskKind: .triggerScan, lowPowerMode: false, thermalState: .serious, isForeground: false, backgroundAgentsEnabled: true, requiresNetwork: false, estimatedCost: 1))
        XCTAssertFalse(d.allow)
    }

    func testNetworkRequiredDeniedWhenNetworkCannotBeAllowed() {
        let d = BackgroundTaskPolicy.decide(.init(taskKind: .ragMaintenance, lowPowerMode: true, thermalState: .nominal, isForeground: false, backgroundAgentsEnabled: true, requiresNetwork: true, estimatedCost: 1))
        XCTAssertFalse(d.allow)
        XCTAssertFalse(d.allowNetwork)
    }

    func testBackgroundTriggerScanAllowedWithoutModelLoading() {
        let d = BackgroundTaskPolicy.decide(.init(taskKind: .triggerScan, lowPowerMode: false, thermalState: .nominal, isForeground: false, backgroundAgentsEnabled: true, requiresNetwork: false, estimatedCost: 1))
        XCTAssertTrue(d.allow)
        XCTAssertFalse(d.allowModelLoading)
        XCTAssertEqual(d.maxSteps, 2)
    }

    func testBackgroundMemoryMaintenanceAllowedWithoutModelLoading() {
        let d = BackgroundTaskPolicy.decide(.init(taskKind: .memoryConsolidation, lowPowerMode: false, thermalState: .nominal, isForeground: false, backgroundAgentsEnabled: true, requiresNetwork: false, estimatedCost: 2))
        XCTAssertTrue(d.allow)
        XCTAssertFalse(d.allowModelLoading)
        XCTAssertEqual(d.maxSteps, 1)
    }

    func testBackgroundSelfImprovementAllowedWithoutModelLoading() {
        let d = BackgroundTaskPolicy.decide(.init(taskKind: .selfImprovement, lowPowerMode: false, thermalState: .nominal, isForeground: false, backgroundAgentsEnabled: true, requiresNetwork: false, estimatedCost: 2))
        XCTAssertTrue(d.allow)
        XCTAssertFalse(d.allowModelLoading)
        XCTAssertEqual(d.maxSteps, 3)
        XCTAssertEqual(d.maxTokens, 0)
    }

    func testBackgroundAgentsDisabledDeniesSelfImprovement() {
        let d = BackgroundTaskPolicy.decide(.init(taskKind: .selfImprovement, lowPowerMode: false, thermalState: .nominal, isForeground: false, backgroundAgentsEnabled: false, requiresNetwork: false, estimatedCost: 2))
        XCTAssertFalse(d.allow)
        XCTAssertFalse(d.allowModelLoading)
        XCTAssertEqual(d.maxTokens, 0)
        XCTAssertEqual(d.denyReason, "background agents disabled")
    }

    func testLowPowerDeniesBackgroundSelfImprovement() {
        let d = BackgroundTaskPolicy.decide(.init(taskKind: .selfImprovement, lowPowerMode: true, thermalState: .nominal, isForeground: false, backgroundAgentsEnabled: true, requiresNetwork: false, estimatedCost: 2))
        XCTAssertFalse(d.allow)
        XCTAssertFalse(d.allowModelLoading)
        XCTAssertEqual(d.maxTokens, 0)
        XCTAssertEqual(d.denyReason, "low power background mode")
    }

    func testLowPowerDeniesBackgroundMaintenanceEvenWithoutNetwork() {
        let d = BackgroundTaskPolicy.decide(.init(taskKind: .memoryConsolidation, lowPowerMode: true, thermalState: .nominal, isForeground: false, backgroundAgentsEnabled: true, requiresNetwork: false, estimatedCost: 2))
        XCTAssertFalse(d.allow)
        XCTAssertEqual(d.denyReason, "low power background mode")
    }

    func testLowPowerAllowsModelHousekeepingWithoutModelLoading() {
        let d = BackgroundTaskPolicy.decide(.init(taskKind: .modelHousekeeping, lowPowerMode: true, thermalState: .nominal, isForeground: false, backgroundAgentsEnabled: true, requiresNetwork: false, estimatedCost: 0))
        XCTAssertTrue(d.allow)
        XCTAssertFalse(d.allowModelLoading)
    }

    func testForegroundLowCostWorkCanAllowModelLoading() {
        let d = BackgroundTaskPolicy.decide(.init(taskKind: .triggerScan, lowPowerMode: false, thermalState: .nominal, isForeground: true, backgroundAgentsEnabled: true, requiresNetwork: false, estimatedCost: 1))
        XCTAssertTrue(d.allow)
        XCTAssertTrue(d.allowModelLoading)
    }

    @MainActor
    func testBackgroundMemoryMaintenanceRecordsPolicySkipMetric() async throws {
        #if DEBUG
        let fileURL = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent("background-metrics-\(UUID().uuidString).jsonl")
        let store = RuntimeMetricsStore(fileURL: fileURL)
        let orchestrator = BackgroundOrchestrator(metricsStore: store)
        ResourceBudgetGate.testSnapshotOverride = .init(
            scenePhase: .background,
            lowPowerModeEnabled: true,
            thermalState: .nominal,
            recentMemoryWarningCount: 0,
            lastMemoryWarningAt: nil
        )
        defer {
            ResourceBudgetGate.testSnapshotOverride = nil
            try? FileManager.default.removeItem(at: fileURL)
        }

        try await orchestrator.runMemoryConsolidationIfAllowed()

        let metric = try await store.recentMetrics(limit: 1).last
        XCTAssertEqual(metric?.runtimeName, "background")
        XCTAssertEqual(metric?.taskKind, BackgroundTaskKind.memoryConsolidation.rawValue)
        XCTAssertEqual(metric?.success, true)
        XCTAssertEqual(metric?.errorCode, "background_policy_denied")
        XCTAssertTrue(metric?.policySummary.contains("low power background mode") == true)
        #endif
    }

    @MainActor
    func testProcessingMaintenanceRunsLocalTasksBeforeHousekeeping() async throws {
        #if DEBUG
        let fileURL = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent("background-processing-\(UUID().uuidString).jsonl")
        let store = RuntimeMetricsStore(fileURL: fileURL)
        var housekeepingDidRun = false
        let orchestrator = BackgroundOrchestrator(metricsStore: store, modelHousekeeping: {
            housekeepingDidRun = true
            return FleetRuntimeCleanupResult(unloadedSlots: [])
        })
        ResourceBudgetGate.testSnapshotOverride = .init(
            scenePhase: .background,
            lowPowerModeEnabled: true,
            thermalState: .nominal,
            recentMemoryWarningCount: 0,
            lastMemoryWarningAt: nil
        )
        defer {
            ResourceBudgetGate.testSnapshotOverride = nil
            try? FileManager.default.removeItem(at: fileURL)
        }

        try await orchestrator.runProcessingMaintenance(until: Date().addingTimeInterval(1))

        let metrics = try await store.recentMetrics(limit: 4)
        XCTAssertEqual(metrics.map(\.taskKind), [
            BackgroundTaskKind.selfImprovement.rawValue,
            BackgroundTaskKind.memoryConsolidation.rawValue,
            BackgroundTaskKind.ragMaintenance.rawValue,
            BackgroundTaskKind.modelHousekeeping.rawValue
        ])
        XCTAssertEqual(metrics[0].errorCode, "background_policy_denied")
        XCTAssertEqual(metrics[1].errorCode, "background_policy_denied")
        XCTAssertEqual(metrics[2].errorCode, "background_policy_denied")
        XCTAssertNil(metrics[3].errorCode)
        XCTAssertTrue(metrics[3].success)
        XCTAssertEqual(metrics[3].policySummary, "optional chat slot cleanup; unloaded=none")
        XCTAssertTrue(housekeepingDidRun)
        #endif
    }

    @MainActor
    func testModelHousekeepingRecordsUnloadedSlotsMetric() async throws {
        #if DEBUG
        let fileURL = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent("background-model-housekeeping-\(UUID().uuidString).jsonl")
        let store = RuntimeMetricsStore(fileURL: fileURL)
        let orchestrator = BackgroundOrchestrator(metricsStore: store, modelHousekeeping: {
            FleetRuntimeCleanupResult(unloadedSlots: [.rem, .mimicry])
        })
        ResourceBudgetGate.testSnapshotOverride = .init(
            scenePhase: .background,
            lowPowerModeEnabled: false,
            thermalState: .nominal,
            recentMemoryWarningCount: 0,
            lastMemoryWarningAt: nil
        )
        defer {
            ResourceBudgetGate.testSnapshotOverride = nil
            try? FileManager.default.removeItem(at: fileURL)
        }

        try await orchestrator.runModelHousekeepingIfAllowed()

        let metric = try await store.recentMetrics(limit: 1).last
        XCTAssertEqual(metric?.runtimeName, "background")
        XCTAssertEqual(metric?.taskKind, BackgroundTaskKind.modelHousekeeping.rawValue)
        XCTAssertEqual(metric?.success, true)
        XCTAssertNil(metric?.errorCode)
        XCTAssertEqual(metric?.policySummary, "optional chat slot cleanup; unloaded=mimicry,rem")
        XCTAssertNotNil(metric?.latencyMs)
        #endif
    }

    @MainActor
    func testModelHousekeepingRunsWhenAgentModeIsDisabled() async throws {
        #if DEBUG
        let agentModeKey = "agentModeEnabled"
        let savedAgentMode = UserDefaults.standard.object(forKey: agentModeKey)
        let fileURL = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent("background-model-housekeeping-agent-disabled-\(UUID().uuidString).jsonl")
        let store = RuntimeMetricsStore(fileURL: fileURL)
        var housekeepingDidRun = false
        let orchestrator = BackgroundOrchestrator(metricsStore: store, modelHousekeeping: {
            housekeepingDidRun = true
            return FleetRuntimeCleanupResult(unloadedSlots: [])
        })
        UserDefaults.standard.set(false, forKey: agentModeKey)
        ResourceBudgetGate.testSnapshotOverride = .init(
            scenePhase: .background,
            lowPowerModeEnabled: false,
            thermalState: .nominal,
            recentMemoryWarningCount: 0,
            lastMemoryWarningAt: nil
        )
        defer {
            if let savedAgentMode {
                UserDefaults.standard.set(savedAgentMode, forKey: agentModeKey)
            } else {
                UserDefaults.standard.removeObject(forKey: agentModeKey)
            }
            ResourceBudgetGate.testSnapshotOverride = nil
            try? FileManager.default.removeItem(at: fileURL)
        }

        try await orchestrator.runModelHousekeepingIfAllowed()

        let metric = try await store.recentMetrics(limit: 1).last
        XCTAssertTrue(housekeepingDidRun)
        XCTAssertEqual(metric?.taskKind, BackgroundTaskKind.modelHousekeeping.rawValue)
        XCTAssertNil(metric?.errorCode)
        XCTAssertTrue(metric?.success == true)
        #endif
    }

    @MainActor
    func testProcessingMaintenanceRecordsDeadlineSkip() async throws {
        #if DEBUG
        let fileURL = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent("background-deadline-\(UUID().uuidString).jsonl")
        let store = RuntimeMetricsStore(fileURL: fileURL)
        let orchestrator = BackgroundOrchestrator(metricsStore: store)
        defer { try? FileManager.default.removeItem(at: fileURL) }

        try await orchestrator.runProcessingMaintenance(until: Date().addingTimeInterval(-1))

        let metric = try await store.recentMetrics(limit: 1).last
        XCTAssertEqual(metric?.taskKind, BackgroundTaskKind.selfImprovement.rawValue)
        XCTAssertEqual(metric?.errorCode, "deadline_exceeded")
        #endif
    }

    @MainActor
    func testAppRefreshRecordsMissingContainerMetric() async throws {
        #if DEBUG
        let savedContainer = SharedContainer.shared
        let fileURL = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent("background-refresh-missing-container-\(UUID().uuidString).jsonl")
        let store = RuntimeMetricsStore(fileURL: fileURL)
        let orchestrator = BackgroundOrchestrator(metricsStore: store)
        SharedContainer.shared = nil
        defer {
            SharedContainer.shared = savedContainer
            try? FileManager.default.removeItem(at: fileURL)
        }

        await orchestrator.handleAppRefresh()

        let metric = try await store.recentMetrics(limit: 1).last
        XCTAssertEqual(metric?.runtimeName, "background")
        XCTAssertEqual(metric?.taskKind, BackgroundTaskKind.triggerScan.rawValue)
        XCTAssertEqual(metric?.success, false)
        XCTAssertEqual(metric?.errorCode, "shared_container_unavailable")
        #endif
    }

    @MainActor
    func testProcessingRecordsMissingContainerMetric() async throws {
        #if DEBUG
        let savedContainer = SharedContainer.shared
        let fileURL = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent("background-processing-missing-container-\(UUID().uuidString).jsonl")
        let store = RuntimeMetricsStore(fileURL: fileURL)
        let orchestrator = BackgroundOrchestrator(metricsStore: store)
        SharedContainer.shared = nil
        defer {
            SharedContainer.shared = savedContainer
            try? FileManager.default.removeItem(at: fileURL)
        }

        await orchestrator.handleProcessing()

        let metric = try await store.recentMetrics(limit: 1).last
        XCTAssertEqual(metric?.runtimeName, "background")
        XCTAssertEqual(metric?.taskKind, BackgroundTaskKind.triggerScan.rawValue)
        XCTAssertEqual(metric?.success, false)
        XCTAssertEqual(metric?.errorCode, "shared_container_unavailable")
        #endif
    }
}
