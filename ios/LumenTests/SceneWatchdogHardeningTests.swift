import XCTest
import SwiftUI
@testable import Lumen

@MainActor
final class SceneWatchdogHardeningTests: XCTestCase {
    func testSceneTransitionCoordinatorReturnsQuickly() {
        let start = ProcessInfo.processInfo.systemUptime
        SceneTransitionCoordinator.shared.handleScenePhaseChange(.background)
        let elapsed = ProcessInfo.processInfo.systemUptime - start
        XCTAssertLessThan(elapsed, 0.1)
    }

    func testCancellationBusCancelsSceneSensitiveTasksSynchronously() async {
        let cancelled = XCTestExpectation(description: "task cancelled")
        let task = Task<Void, Never> {
            while !Task.isCancelled { await Task.yield() }
            cancelled.fulfill()
        }
        AppCancellationBus.shared.register(task, category: .chatGeneration)
        AppCancellationBus.shared.cancelAllSceneSensitive()
        await fulfillment(of: [cancelled], timeout: 1)
    }

    func testDeferredMaintenanceQueueDoesNotRunWhileBackgrounded() async throws {
        let queue = DeferredMaintenanceQueue.shared
        queue.updateScenePhase(.background)
        let ran = XCTestExpectation(description: "job should not run")
        ran.isInverted = true
        queue.enqueue(DeferredMaintenanceJob(key: "test-background", category: .diagnostics, staleAfter: 5, maxRuntime: 1) {
            ran.fulfill()
        })
        await fulfillment(of: [ran], timeout: 0.4)
    }

    func testDeferredMaintenanceQueueWaitsAfterForegroundReactivation() async throws {
        let queue = DeferredMaintenanceQueue.shared
        queue.updateScenePhase(.active)
        let ran = XCTestExpectation(description: "job should wait")
        ran.isInverted = true
        queue.enqueue(DeferredMaintenanceJob(key: "test-foreground-grace", category: .diagnostics, staleAfter: 5, maxRuntime: 1) {
            ran.fulfill()
        })
        await fulfillment(of: [ran], timeout: 1)
    }

    func testCPUWatchdogGuardDegradesLongCategory() throws {
        let guardrail = CPUWatchdogGuard(window: 10, degradeThreshold: 0.01)
        let token = guardrail.begin(category: .diagnostics)
        Thread.sleep(forTimeInterval: 0.02)
        guardrail.end(token: token)
        XCTAssertTrue(guardrail.shouldDegrade(category: .diagnostics))
    }

    func testDiskWriteBudgetDefersRepeatedLargeWrites() {
        let budget = DiskWriteBudget(oneMinuteLimit: 1_000, fifteenMinuteLimit: 2_000, dayLimit: 3_000)
        budget.recordWrite(bytes: 900, category: .diagnostics)
        XCTAssertTrue(budget.shouldDefer(bytes: 200, category: .diagnostics))
        XCTAssertFalse(budget.canWrite(bytes: 200, category: .diagnostics))
    }

    func testDiagnosticsOpeningUsesCachedSnapshot() {
        let provider = DiagnosticsProvider()
        _ = provider.cachedSnapshot()
        XCTAssertEqual(provider.explicitCollectionCount, 0)
    }
}
