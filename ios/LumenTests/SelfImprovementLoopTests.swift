import XCTest
import SwiftUI
import SwiftData
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
        let cancellationProbe = TestCancellationProbe()
        let loop = SelfImprovementLoop(metricsStore: store, config: .init(cooldownSeconds: 0), maintenance: { _, _, _ in
            if await cancellationProbe.consumeShouldCancel() {
                throw CancellationError()
            }
            return .applied("recovered")
        })

        let cancelled = await loop.run(trigger: .test, context: nil)
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
        XCTAssertEqual(metrics.last?.taskKind, BackgroundTaskKind.selfImprovement.rawValue)
        XCTAssertEqual(metrics.last?.errorCode, "RawPromptFailure")
        XCTAssertFalse(metrics.last?.policySummary.contains("secret raw prompt") == true)
        #endif
    }

    @MainActor
    func testDeadlineExceededAfterMaintenanceIsSkippedWithoutFailureMetric() async throws {
        #if DEBUG
        ResourceBudgetGate.testSnapshotOverride = nominalSnapshot()
        defer { ResourceBudgetGate.testSnapshotOverride = nil }
        let store = metricsStore()
        var now = Date(timeIntervalSince1970: 100)
        let loop = SelfImprovementLoop(metricsStore: store, config: .init(cooldownSeconds: 0, maxRunDurationSeconds: 5), now: { now }, maintenance: { _, _, _ in
            now = Date(timeIntervalSince1970: 106)
            return .applied("finished_after_deadline")
        })

        let outcome = await loop.run(trigger: .test, context: nil)
        let metrics = try await store.recentMetrics(limit: 1)

        XCTAssertEqual(outcome, .skipped("deadline_expired"))
        XCTAssertEqual(metrics.last?.taskKind, BackgroundTaskKind.selfImprovement.rawValue)
        XCTAssertEqual(metrics.last?.success, true)
        XCTAssertEqual(metrics.last?.errorCode, "deadline_expired")
        XCTAssertTrue(metrics.last?.policySummary.contains("skipped: deadline_expired") == true)
        #endif
    }

    @MainActor
    func testBackgroundMaintenanceDoesNotRepeatMemoryOrRAGMaintenance() async throws {
        #if DEBUG
        ResourceBudgetGate.testSnapshotOverride = nominalSnapshot()
        defer { ResourceBudgetGate.testSnapshotOverride = nil }
        let store = metricsStore()
        let context = try ModelContext(inMemoryContainer())
        let loop = SelfImprovementLoop(metricsStore: store, config: .init(cooldownSeconds: 0))

        let outcome = await loop.run(trigger: .backgroundProcessing, context: context)

        guard case .applied(let summary) = outcome else {
            XCTFail("Expected applied background maintenance, got \(outcome)")
            return
        }
        XCTAssertTrue(summary.contains("memory=already_run"))
        XCTAssertTrue(summary.contains("rag=already_run"))
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

    private struct TestFailure: Error {}

    private struct RawPromptFailure: Error, CustomStringConvertible {
        var description: String { "secret raw prompt should not be logged" }
    }

    private actor TestCancellationProbe {
        private var shouldCancel = true

        func consumeShouldCancel() -> Bool {
            defer { shouldCancel = false }
            return shouldCancel
        }
    }
}
