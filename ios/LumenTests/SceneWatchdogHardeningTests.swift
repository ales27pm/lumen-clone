import XCTest
import SwiftUI
import UIKit
@testable import Lumen

@MainActor
final class SceneWatchdogHardeningTests: XCTestCase {
    override func setUp() async throws {
        try await super.setUp()
        #if DEBUG
        SceneTransitionCoordinator.shared.resetForTesting()
        await MainActor.run { DeferredMaintenanceQueue.shared.resetForTesting() }
        AppCancellationBus.shared.resetForTesting()
        #endif
    }

    override func tearDown() async throws {
        #if DEBUG
        SceneTransitionCoordinator.shared.resetForTesting()
        await MainActor.run { DeferredMaintenanceQueue.shared.resetForTesting() }
        AppCancellationBus.shared.resetForTesting()
        #endif
        try await super.tearDown()
    }

    func testSceneTransitionCoordinatorReturnsQuickly() {
        let start = ProcessInfo.processInfo.systemUptime
        SceneTransitionCoordinator.shared.handleScenePhaseChange(.background)
        let elapsed = ProcessInfo.processInfo.systemUptime - start
        XCTAssertLessThan(elapsed, 0.1)
    }

    func testSceneTransitionDoesNotRunCancellationCallbacksWhenPolicyAllowsBackground() async {
        let cancelled = XCTestExpectation(description: "scene transition should not run cancellation callbacks")
        cancelled.isInverted = true
        let id = AppCancellationBus.shared.registerCancellation({
            Thread.sleep(forTimeInterval: 0.2)
            cancelled.fulfill()
        }, category: .chatGeneration)

        let start = ProcessInfo.processInfo.systemUptime
        SceneTransitionCoordinator.shared.handleScenePhaseChange(.background)
        let elapsed = ProcessInfo.processInfo.systemUptime - start

        XCTAssertLessThan(elapsed, 0.05)
        await fulfillment(of: [cancelled], timeout: 0.4)
        AppCancellationBus.shared.unregister(id, category: .chatGeneration)
    }

    func testSceneTransitionDoesNotSynchronouslyRunDiagnosticObservers() async {
        let observerStarted = XCTestExpectation(description: "diagnostic observer ran")
        let observerID = PersistentRuntimeDiagnosticsObserver.shared.addObserver { signal in
            guard signal.kind == .sceneTransition else { return }
            observerStarted.fulfill()
            Thread.sleep(forTimeInterval: 0.2)
        }
        defer { PersistentRuntimeDiagnosticsObserver.shared.removeObserver(observerID) }

        let start = ProcessInfo.processInfo.systemUptime
        SceneTransitionCoordinator.shared.handleScenePhaseChange(.background)
        let elapsed = ProcessInfo.processInfo.systemUptime - start

        XCTAssertLessThan(elapsed, 0.05)
        await fulfillment(of: [observerStarted], timeout: 1)
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

    func testApplicationWillTerminateDoesNotSynchronouslyCancelSceneSensitiveWork() async {
        let cancelled = XCTestExpectation(description: "termination should not run cancellation callbacks")
        cancelled.isInverted = true
        let id = AppCancellationBus.shared.registerCancellation({
            cancelled.fulfill()
        }, category: .chatGeneration)

        let delegate = LumenAppDelegate()
        let start = ProcessInfo.processInfo.systemUptime
        delegate.applicationWillTerminate(UIApplication.shared)
        let elapsed = ProcessInfo.processInfo.systemUptime - start

        XCTAssertLessThan(elapsed, 0.05)
        XCTAssertEqual(AppCancellationBus.shared.lastCancellationReason, "will-terminate")
        await fulfillment(of: [cancelled], timeout: 0.2)
        AppCancellationBus.shared.unregister(id, category: .chatGeneration)
    }

    func testApplicationDidBecomeActiveRefreshesResourceBudgetGate() async {
        let delegate = LumenAppDelegate()
        delegate.applicationWillResignActive(UIApplication.shared)
        await Task.yield()
        XCTAssertEqual(ResourceBudgetGate.diagnosticSnapshot().scenePhase, .inactive)

        delegate.applicationDidBecomeActive(UIApplication.shared)
        await Task.yield()
        XCTAssertEqual(ResourceBudgetGate.diagnosticSnapshot().scenePhase, .active)
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
        await fulfillment(of: [ran], timeout: 3)
    }

    func testCPUWatchdogGuardDegradesRecordedWorkOnly() throws {
        let guardrail = CPUWatchdogGuard(window: 10, degradeThreshold: 0.01)
        guardrail.recordWork(category: .diagnostics, duration: 0.02)
        XCTAssertTrue(guardrail.shouldDegrade(category: .diagnostics))
    }

    func testCPUWatchdogGuardDoesNotDegradeActiveWallTime() throws {
        let guardrail = CPUWatchdogGuard(window: 10, degradeThreshold: 0.01)
        let token = guardrail.begin(category: .chatGeneration)
        Thread.sleep(forTimeInterval: 0.02)
        XCTAssertFalse(guardrail.shouldDegrade(category: .chatGeneration))
        XCTAssertEqual(guardrail.currentSnapshot().activeCountsByCategory[.chatGeneration], 1)
        guardrail.end(token: token)
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

    #if DEBUG
    func testDeferredMaintenanceQueueDrainsAfterForegroundGrace() async {
        let queue = DeferredMaintenanceQueue.shared
        let ran = XCTestExpectation(description: "job ran after grace")
        queue.updateScenePhase(.background)
        queue.enqueue(DeferredMaintenanceJob(key: "test-drain-after-grace", category: .diagnostics, staleAfter: 5, maxRuntime: 1) {
            ran.fulfill()
        })
        queue.updateScenePhase(.active)
        queue.forceForegroundGraceElapsedForTesting()
        await fulfillment(of: [ran], timeout: 1)
    }

    func testDeferredMaintenanceQueueActiveGateCanBeCleared() async {
        let queue = DeferredMaintenanceQueue.shared
        let blocked = XCTestExpectation(description: "job blocked while active")
        blocked.isInverted = true
        queue.updateScenePhase(.active)
        queue.forceForegroundGraceElapsedForTesting()
        queue.setChatOrVoiceActive(true)
        queue.enqueue(DeferredMaintenanceJob(key: "test-active-gate", category: .diagnostics, staleAfter: 5, maxRuntime: 1) {
            blocked.fulfill()
        })
        await fulfillment(of: [blocked], timeout: 0.2)

        let ran = XCTestExpectation(description: "job ran after active cleared")
        queue.enqueue(DeferredMaintenanceJob(key: "test-active-gate", category: .diagnostics, staleAfter: 5, maxRuntime: 1) {
            ran.fulfill()
        })
        queue.setChatOrVoiceActive(false)
        await fulfillment(of: [ran], timeout: 1)
    }
    #endif
}
