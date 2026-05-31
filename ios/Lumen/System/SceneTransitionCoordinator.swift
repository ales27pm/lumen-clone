import Foundation
import SwiftUI
import OSLog

@MainActor
final class SceneTransitionCoordinator {
    static let shared = SceneTransitionCoordinator()

    private let logger = Logger(subsystem: "ai.lumen.app", category: "scene")
    private(set) var currentPhase: ScenePhase = .active
    private var lastTransitionReason: String?

    private init() {}

    func handleScenePhaseChange(_ phase: ScenePhase) {
        measure("scenePhase.\(String(describing: phase))") {
            currentPhase = phase
            ResourceBudgetGate.recordScenePhase(phase)
            DeferredMaintenanceQueue.shared.updateScenePhase(phase)
            if ResourceBudgetGate.shouldCancelForScenePhase(phase) {
                cancelSceneSensitive(reason: "scene-phase-\(phase)")
            }
        }
    }

    func handleWillResignActive() {
        measure("willResignActive") {
            currentPhase = .inactive
            ResourceBudgetGate.recordScenePhase(.inactive)
            DeferredMaintenanceQueue.shared.updateScenePhase(.inactive)
            cancelSceneSensitive(reason: "will-resign-active")
        }
    }

    func handleDidEnterBackground() {
        measure("didEnterBackground") {
            currentPhase = .background
            ResourceBudgetGate.recordScenePhase(.background)
            DeferredMaintenanceQueue.shared.updateScenePhase(.background)
            cancelSceneSensitive(reason: "did-enter-background")
            DeferredMaintenanceQueue.shared.enqueue(
                DeferredMaintenanceJob(key: "scene-background-coalesced-persistence", category: .persistence, staleAfter: 15 * 60, maxRuntime: 2) {}
            )
        }
    }

    func requestForegroundActivation() {
        measure("foregroundActivation") {
            currentPhase = .active
            ResourceBudgetGate.recordScenePhase(.active)
            DeferredMaintenanceQueue.shared.updateScenePhase(.active)
        }
    }

    private func cancelSceneSensitive(reason: String) {
        lastTransitionReason = reason
        AppCancellationBus.shared.markCancellationRequested(reason)
        AppCancellationBus.shared.cancelAllSceneSensitive()
        VoiceService.shared.stopListening()
        VoiceService.shared.stopSpeaking()
        ModelLoader.cancelActiveLoads()
    }

    private func measure(_ operation: String, _ work: () -> Void) {
        let start = ProcessInfo.processInfo.systemUptime
        work()
        let elapsed = (ProcessInfo.processInfo.systemUptime - start) * 1000
        if elapsed > 100 {
            logger.fault("scene_transition_slow operation=\(operation, privacy: .public) elapsed_ms=\(elapsed, privacy: .public)")
            assertionFailure("Scene transition exceeded 100 ms: \(operation) \(elapsed) ms")
        } else if elapsed > 50 {
            logger.warning("scene_transition_near_budget operation=\(operation, privacy: .public) elapsed_ms=\(elapsed, privacy: .public)")
        }
    }
}
