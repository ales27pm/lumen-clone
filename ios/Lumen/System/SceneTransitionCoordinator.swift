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
            PersistentRuntimeDiagnosticsObserver.shared.emit(.init(kind: .sceneTransition, values: ["phase": Self.phaseName(phase), "cancellationRequested": String(ResourceBudgetGate.shouldCancelForScenePhase(phase))]))
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
            PersistentRuntimeDiagnosticsObserver.shared.emit(.init(kind: .sceneTransition, values: ["phase": "inactive", "cancellationRequested": "true"]))
            cancelSceneSensitive(reason: "will-resign-active")
        }
    }

    func handleDidEnterBackground() {
        measure("didEnterBackground") {
            currentPhase = .background
            ResourceBudgetGate.recordScenePhase(.background)
            DeferredMaintenanceQueue.shared.updateScenePhase(.background)
            PersistentRuntimeDiagnosticsObserver.shared.emit(.init(kind: .sceneTransition, values: ["phase": "background", "cancellationRequested": "true"]))
            cancelSceneSensitive(reason: "did-enter-background")
        }
    }

    func requestForegroundActivation() {
        measure("foregroundActivation") {
            currentPhase = .active
            ResourceBudgetGate.recordScenePhase(.active)
            DeferredMaintenanceQueue.shared.updateScenePhase(.active)
            PersistentRuntimeDiagnosticsObserver.shared.emit(.init(kind: .sceneTransition, values: ["phase": "active", "cancellationRequested": "false"]))
        }
    }

    private func cancelSceneSensitive(reason: String) {
        lastTransitionReason = reason
        AppCancellationBus.shared.markCancellationRequested(reason)
        AppCancellationBus.shared.cancelAllSceneSensitive()
        VoiceService.shared.stopListening()
        VoiceService.shared.stopSpeaking()
        Task.detached(priority: .userInitiated) {
            await AppLlamaService.shared.cancelActiveGeneration(reason: reason)
        }
        ModelLoader.cancelActiveLoads()
        let phaseName = currentPhaseName
        for delayMs in [500, 1000, 2000] {
            Task.detached(priority: .utility) {
                try? await Task.sleep(nanoseconds: UInt64(delayMs) * 1_000_000)
                PersistentRuntimeDiagnosticsObserver.shared.emit(.init(kind: .sceneTransition, values: [
                    "phase": phaseName,
                    "checkMs": String(delayMs),
                    "generationActive": String(DiskWriteBudget.shared.isGenerationActive()),
                    "cancellationReason": AppCancellationBus.shared.lastCancellationReason ?? reason
                ]))
            }
        }
    }

    private var currentPhaseName: String { Self.phaseName(currentPhase) }

    private static func phaseName(_ phase: ScenePhase) -> String {
        switch phase {
        case .active: return "active"
        case .inactive: return "inactive"
        case .background: return "background"
        @unknown default: return "unknown"
        }
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
