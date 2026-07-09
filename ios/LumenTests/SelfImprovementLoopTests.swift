import XCTest
import SwiftUI
@testable import Lumen

final class SelfImprovementLoopTests: XCTestCase {
    @MainActor
    func testConcurrentRunsCoalesce() async throws {
        #if DEBUG
        ResourceBudgetGate.testSnapshotOverride = nominalSnapshot()
        defer { ResourceBudgetGate.testSnapshotOverride = nil }
        let store = metricsStore()
        var runCount = 0
        let loop = SelfImprovementLoop(metricsStore: store, config: .init(cooldownSeconds: 0), maintenance: { _, _, _ in
            runCount += 1
            try await Task.sleep(nanoseconds: 50_000_000)
            return .applied("test_applied")
        })

        let first = Task { @MainActor in await loop.run(trigger: .test, context: nil) }
        try await Task.sleep(nanoseconds: 5_000_000)
        let second = await loop.run(trigger: .test, context: nil)
        let firstOutcome = await first.value

        XCTAssertEqual(firstOutcome, .applied("test_applied"))
        XCTAssertEqual(second, .skipped("already_running"))
        XCTAssertEqual(runCount, 1)
        #endif
    }

    @MainActor
    func testCooldownWindowSkipsSubsequentRun() async throws {
        #if DEBUG
        ResourceBudgetGate.testSnapshotOverride = nominalSnapshot()
        defer { ResourceBudgetGate.testSnapshotOverride = nil }
        let store = metricsStore()
        var now = Date(timeIntervalSince1970: 100)
        let loop = SelfImprovementLoop(metricsStore: store, config: .init(cooldownSeconds: 60), now: { now }, maintenance: { _, _, _ in
            .applied("test_applied")
        })

        let first = await loop.run(trigger: .test, context: nil)
        now = Date(timeIntervalSince1970: 120)
        let second = await loop.run(trigger: .test, context: nil)

        XCTAssertEqual(first, .applied("test_applied"))
        XCTAssertEqual(second, .skipped("cooldown_active"))
        #endif
    }

    @MainActor
    func testCircuitBreakerOpensAfterThresholdFailures() async throws {
        #if DEBUG
        ResourceBudgetGate.testSnapshotOverride = nominalSnapshot()
        defer { ResourceBudgetGate.testSnapshotOverride = nil }
        let store = metricsStore()
        let loop = SelfImprovementLoop(
            metricsStore: store,
            config: .init(cooldownSeconds: 0, failureThreshold: 2, circuitOpenSeconds: 600),
            maintenance: { _, _, _ in throw TestFailure() }
        )

        let first = await loop.run(trigger: .test, context: nil)
        let second = await loop.run(trigger: .test, context: nil)
        let third = await loop.run(trigger: .test, context: nil)

        XCTAssertEqual(first, .failed("TestFailure"))
        XCTAssertEqual(second, .failed("TestFailure"))
        XCTAssertEqual(third, .skipped("circuit_open"))
        #endif
    }

    @MainActor
    func testCancellationRecordsCancelledOutcomeAndAllowsFutureRun() async throws {
        #if DEBUG
        ResourceBudgetGate.testSnapshotOverride = nominalSnapshot()
        defer { ResourceBudgetGate.testSnapshotOverride = nil }
        let store = metricsStore()
        var shouldCancel = true
        let loop = SelfImprovementLoop(metricsStore: store, config: .init(cooldownSeconds: 0), maintenance: { _, _, _ in
            if shouldCancel {
                throw CancellationError()
            }
            return .applied("recovered")
        })

        let cancelled = await loop.run(trigger: .test, context: nil)
        shouldCancel = false
        let recovered = await loop.run(trigger: .test, context: nil)

        XCTAssertEqual(cancelled, .cancelled)
        XCTAssertEqual(recovered, .applied("recovered"))
        #endif
    }

    @MainActor
    func testFailureMetricUsesErrorCodeWithoutRawPromptPayload() async throws {
        #if DEBUG
        ResourceBudgetGate.testSnapshotOverride = nominalSnapshot()
        defer { ResourceBudgetGate.testSnapshotOverride = nil }
        let store = metricsStore()
        let loop = SelfImprovementLoop(metricsStore: store, config: .init(cooldownSeconds: 0), maintenance: { _, _, _ in
            throw RawPromptFailure()
        })

        let outcome = await loop.run(trigger: .test, context: nil)
        let metrics = try await store.recentMetrics(limit: 1)

        XCTAssertEqual(outcome, .failed("RawPromptFailure"))
        XCTAssertEqual(metrics.last?.runtimeName, "selfImprovement")
        XCTAssertEqual(metrics.last?.errorCode, "RawPromptFailure")
        XCTAssertFalse(metrics.last?.policySummary.contains("secret raw prompt") == true)
        #endif
    }

    private func metricsStore() -> RuntimeMetricsStore {
        let url = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("self-improvement-\(UUID().uuidString).jsonl")
        return RuntimeMetricsStore(fileURL: url)
    }

    private func nominalSnapshot() -> ResourceBudgetGate.Snapshot {
        .init(
            scenePhase: .active,
            lowPowerModeEnabled: false,
            thermalState: .nominal,
            recentMemoryWarningCount: 0,
            lastMemoryWarningAt: nil
        )
    }

    private struct TestFailure: Error {}

    private struct RawPromptFailure: Error, CustomStringConvertible {
        var description: String { "secret raw prompt should not be logged" }
    }
}
