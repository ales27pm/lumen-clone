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
        let maintenanceProbe = TestMaintenanceProbe()
        let loop = SelfImprovementLoop(metricsStore: store, config: .init(cooldownSeconds: 0), maintenance: { _, _, _, _ in
            await maintenanceProbe.run()
        })

        let first = Task { @MainActor in await loop.run(trigger: .test, container: nil) }
        await maintenanceProbe.waitUntilStarted()
        let second = await loop.run(trigger: .test, container: nil)
        await maintenanceProbe.release()
        let firstOutcome = await first.value
        let runCount = await maintenanceProbe.runCount()

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
        let loop = SelfImprovementLoop(metricsStore: store, config: .init(cooldownSeconds: 60), now: { now }, maintenance: { _, _, _, _ in
            .applied("test_applied")
        })

        let first = await loop.run(trigger: .test, container: nil)
        now = Date(timeIntervalSince1970: 120)
        let second = await loop.run(trigger: .test, container: nil)

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
            maintenance: { _, _, _, _ in throw TestFailure() }
        )

        let first = await loop.run(trigger: .test, container: nil)
        let second = await loop.run(trigger: .test, container: nil)
        let third = await loop.run(trigger: .test, container: nil)

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
        let loop = SelfImprovementLoop(metricsStore: store, config: .init(cooldownSeconds: 0), maintenance: { _, _, _, _ in
            if await cancellationProbe.consumeShouldCancel() {
                throw CancellationError()
            }
            return .applied("recovered")
        })

        let cancelled = await loop.run(trigger: .test, container: nil)
        let recovered = await loop.run(trigger: .test, container: nil)

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
        let loop = SelfImprovementLoop(metricsStore: store, config: .init(cooldownSeconds: 0), maintenance: { _, _, _, _ in
            throw RawPromptFailure()
        })

        let outcome = await loop.run(trigger: .test, container: nil)
        let metrics = try await store.recentMetrics(limit: 1)

        XCTAssertEqual(outcome, .failed("RawPromptFailure"))
        XCTAssertEqual(metrics.last?.runtimeName, "selfImprovement")
        XCTAssertEqual(metrics.last?.taskKind, BackgroundTaskKind.selfImprovement.rawValue)
        XCTAssertEqual(metrics.last?.errorCode, "rawpromptfailure")
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
        let loop = SelfImprovementLoop(metricsStore: store, config: .init(cooldownSeconds: 0, maxRunDurationSeconds: 5), now: { now }, maintenance: { _, _, _, _ in
            now = Date(timeIntervalSince1970: 106)
            return .applied("finished_after_deadline")
        })

        let outcome = await loop.run(trigger: .test, container: nil)
        let metrics = try await store.recentMetrics(limit: 1)

        XCTAssertEqual(outcome, .skipped("deadline_exceeded"))
        XCTAssertEqual(metrics.last?.taskKind, BackgroundTaskKind.selfImprovement.rawValue)
        XCTAssertEqual(metrics.last?.success, true)
        XCTAssertEqual(metrics.last?.errorCode, "deadline_exceeded")
        XCTAssertTrue(metrics.last?.policySummary.contains("skipped: deadline_exceeded") == true)
        #endif
    }

    @MainActor
    func testDeadlineExceededBeforeStartSkipsWithoutMaintenance() async throws {
        #if DEBUG
        ResourceBudgetGate.testSnapshotOverride = nominalSnapshot()
        defer { ResourceBudgetGate.testSnapshotOverride = nil }
        let store = metricsStore()
        var didRunMaintenance = false
        let now = Date(timeIntervalSince1970: 100)
        let loop = SelfImprovementLoop(metricsStore: store, config: .init(cooldownSeconds: 0), now: { now }, maintenance: { _, _, _, _ in
            didRunMaintenance = true
            return .applied("should_not_run")
        })

        let outcome = await loop.run(trigger: .test, container: nil, deadline: Date(timeIntervalSince1970: 99))
        let metrics = try await store.recentMetrics(limit: 1)

        XCTAssertEqual(outcome, .skipped("deadline_exceeded"))
        XCTAssertFalse(didRunMaintenance)
        XCTAssertEqual(metrics.last?.errorCode, "deadline_exceeded")
        #endif
    }

    @MainActor
    func testConfiguredMaxDurationIsPassedToMaintenanceWithoutExternalDeadline() async throws {
        #if DEBUG
        ResourceBudgetGate.testSnapshotOverride = nominalSnapshot()
        defer { ResourceBudgetGate.testSnapshotOverride = nil }
        let store = metricsStore()
        let now = Date(timeIntervalSince1970: 100)
        var receivedDeadline: Date?
        let loop = SelfImprovementLoop(metricsStore: store, config: .init(cooldownSeconds: 0, maxRunDurationSeconds: 5), now: { now }, maintenance: { _, _, deadline, _ in
            receivedDeadline = deadline
            return .applied("bounded")
        })

        let outcome = await loop.run(trigger: .test, container: nil)

        XCTAssertEqual(outcome, .applied("bounded"))
        XCTAssertEqual(receivedDeadline, Date(timeIntervalSince1970: 105))
        #endif
    }

    @MainActor
    func testBackgroundAgentsDisabledDeniesSelfImprovement() async throws {
        #if DEBUG
        ResourceBudgetGate.testSnapshotOverride = nominalSnapshot()
        defer { ResourceBudgetGate.testSnapshotOverride = nil }
        let store = metricsStore()
        let loop = SelfImprovementLoop(
            metricsStore: store,
            config: .init(cooldownSeconds: 0),
            backgroundAgentsEnabled: { false },
            maintenance: { _, _, _, _ in .applied("should_not_run") }
        )

        let outcome = await loop.run(trigger: .backgroundProcessing, container: nil)
        let metrics = try await store.recentMetrics(limit: 1)

        XCTAssertEqual(outcome, .skipped("policy_denied"))
        XCTAssertEqual(metrics.last?.errorCode, "policy_denied")
        XCTAssertTrue(metrics.last?.policySummary.contains("policy_denied") == true)
        #endif
    }

    @MainActor
    func testMetricSummaryRedactsRawDetails() async throws {
        #if DEBUG
        ResourceBudgetGate.testSnapshotOverride = nominalSnapshot()
        defer { ResourceBudgetGate.testSnapshotOverride = nil }
        let store = metricsStore()
        let loop = SelfImprovementLoop(metricsStore: store, config: .init(cooldownSeconds: 0), maintenance: { _, _, _, _ in
            .applied("prompt=secret raw prompt path=/Users/example/private/file.txt")
        })

        let outcome = await loop.run(trigger: .test, container: nil)
        let metrics = try await store.recentMetrics(limit: 1)

        XCTAssertEqual(outcome, .applied("prompt=secret raw prompt path=/Users/example/private/file.txt"))
        XCTAssertFalse(metrics.last?.policySummary.contains("secret raw prompt") == true)
        XCTAssertFalse(metrics.last?.policySummary.contains("/Users/example") == true)
        #endif
    }

    @MainActor
    func testBackgroundMaintenanceDoesNotRepeatMemoryOrRAGMaintenance() async throws {
        #if DEBUG
        ResourceBudgetGate.testSnapshotOverride = nominalSnapshot()
        defer { ResourceBudgetGate.testSnapshotOverride = nil }
        let store = metricsStore()
        let container = try inMemoryContainer()
        let loop = SelfImprovementLoop(metricsStore: store, config: .init(cooldownSeconds: 0))

        let outcome = await loop.run(trigger: .backgroundProcessing, container: container, maintenanceMode: .snapshotOnly)

        guard case .applied(let summary) = outcome else {
            XCTFail("Expected applied background maintenance, got \(outcome)")
            return
        }
        XCTAssertTrue(summary.contains("memory=skipped_snapshot_only"))
        XCTAssertTrue(summary.contains("rag=skipped_snapshot_only"))
        #endif
    }

    @MainActor
    func testAppLaunchWithContainerRunsSnapshotMaintenance() async throws {
        #if DEBUG
        ResourceBudgetGate.testSnapshotOverride = nominalSnapshot()
        defer { ResourceBudgetGate.testSnapshotOverride = nil }
        let store = metricsStore()
        let container = try inMemoryContainer()
        let loop = SelfImprovementLoop(metricsStore: store, config: .init(cooldownSeconds: 0))

        let outcome = await loop.run(trigger: .appLaunch, container: container, maintenanceMode: .snapshotOnly)

        guard case .applied(let summary) = outcome else {
            XCTFail("Expected app launch snapshot maintenance, got \(outcome)")
            return
        }
        XCTAssertTrue(summary.contains("memory=skipped_snapshot_only"))
        XCTAssertTrue(summary.contains("rag=skipped_snapshot_only"))
        #endif
    }

    @MainActor
    func testDefaultMaintenanceUsesInjectedClockForDeadlineChecks() async throws {
        #if DEBUG
        ResourceBudgetGate.testSnapshotOverride = nominalSnapshot()
        defer { ResourceBudgetGate.testSnapshotOverride = nil }
        let store = metricsStore()
        let container = try inMemoryContainer()
        let now = Date(timeIntervalSince1970: 100)
        let loop = SelfImprovementLoop(
            metricsStore: store,
            config: .init(cooldownSeconds: 0, maxRunDurationSeconds: 5),
            now: { now }
        )

        let outcome = await loop.run(trigger: .test, container: container, maintenanceMode: .snapshotOnly)

        guard case .applied(let summary) = outcome else {
            XCTFail("Expected injected-clock maintenance to apply, got \(outcome)")
            return
        }
        XCTAssertTrue(summary.contains("metrics=compact"))
        #endif
    }

    @MainActor
    func testAppLaunchWithoutContextSkipsHeavyMaintenance() async throws {
        #if DEBUG
        ResourceBudgetGate.testSnapshotOverride = nominalSnapshot()
        defer { ResourceBudgetGate.testSnapshotOverride = nil }
        let store = metricsStore()
        let loop = SelfImprovementLoop(metricsStore: store, config: .init(cooldownSeconds: 0))

        let outcome = await loop.run(trigger: .appLaunch, container: nil)
        let metrics = try await store.recentMetrics(limit: 1)

        XCTAssertEqual(outcome, .skipped("shared_container_unavailable"))
        XCTAssertEqual(metrics.last?.taskKind, BackgroundTaskKind.selfImprovement.rawValue)
        XCTAssertEqual(metrics.last?.success, true)
        XCTAssertTrue(metrics.last?.policySummary.contains("shared_container_unavailable") == true)
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

    private actor TestMaintenanceProbe {
        private var count = 0
        private var started = false
        private var startedContinuation: CheckedContinuation<Void, Never>?
        private var releaseContinuation: CheckedContinuation<Void, Never>?

        func run() async -> SelfImprovementMaintenanceResult {
            count += 1
            started = true
            startedContinuation?.resume()
            startedContinuation = nil
            await withCheckedContinuation { continuation in
                releaseContinuation = continuation
            }
            return .applied("test_applied")
        }

        func waitUntilStarted() async {
            guard !started else { return }
            await withCheckedContinuation { continuation in
                startedContinuation = continuation
            }
        }

        func release() {
            releaseContinuation?.resume()
            releaseContinuation = nil
        }

        func runCount() -> Int {
            count
        }
    }
}
