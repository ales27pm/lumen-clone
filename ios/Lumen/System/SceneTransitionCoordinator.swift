import Foundation
import SwiftUI
import OSLog

@MainActor
final class SceneTransitionCoordinator {
    static let shared = SceneTransitionCoordinator()

    private let logger = Logger(subsystem: "ai.lumen.app", category: "scene")
    private(set) var currentPhase: ScenePhase = .active
    private var lastTransitionReason: String?
    private var lastIssuedCancellationAt: Date?
    private var deferredCleanupTask: Task<Void, Never>?

    private init() {}

    func handleScenePhaseChange(_ phase: ScenePhase) {
        guard phase != currentPhase else { return }
        measure("scenePhase.\(String(describing: phase))") {
            currentPhase = phase
            ResourceBudgetGate.recordScenePhase(phase)
            DeferredMaintenanceQueue.shared.updateScenePhase(phase)
            PersistentRuntimeDiagnosticsObserver.shared.emit(.init(kind: .sceneTransition, values: ["phase": Self.phaseName(phase), "cancellationRequested": String(ResourceBudgetGate.shouldCancelForScenePhase(phase))]))
            if phase == .active {
                cancelDeferredCleanup()
            } else if ResourceBudgetGate.shouldCancelForScenePhase(phase) {
                cancelSceneSensitive(reason: "scene-phase-\(Self.phaseName(phase))")
            }
        }
    }

    func handleWillResignActive() {
        guard currentPhase != .inactive else { return }
        measure("willResignActive") {
            currentPhase = .inactive
            ResourceBudgetGate.recordScenePhase(.inactive)
            DeferredMaintenanceQueue.shared.updateScenePhase(.inactive)
            PersistentRuntimeDiagnosticsObserver.shared.emit(.init(kind: .sceneTransition, values: ["phase": "inactive", "cancellationRequested": "true"]))
            cancelSceneSensitive(reason: "will-resign-active")
        }
    }

    func handleDidEnterBackground() {
        guard currentPhase != .background else { return }
        measure("didEnterBackground") {
            currentPhase = .background
            ResourceBudgetGate.recordScenePhase(.background)
            DeferredMaintenanceQueue.shared.updateScenePhase(.background)
            PersistentRuntimeDiagnosticsObserver.shared.emit(.init(kind: .sceneTransition, values: ["phase": "background", "cancellationRequested": "true"]))
            cancelSceneSensitive(reason: "did-enter-background")
        }
    }

    func requestForegroundActivation() {
        cancelDeferredCleanup()
        measure("foregroundActivation") {
            currentPhase = .active
            ResourceBudgetGate.recordScenePhase(.active)
            DeferredMaintenanceQueue.shared.updateScenePhase(.active)
            PersistentRuntimeDiagnosticsObserver.shared.emit(.init(kind: .sceneTransition, values: ["phase": "active", "cancellationRequested": "false"]))
        }
    }

    private func cancelDeferredCleanup() {
        deferredCleanupTask?.cancel()
        deferredCleanupTask = nil
        lastIssuedCancellationAt = nil
    }

    private func cancelSceneSensitive(reason: String) {
        guard shouldIssueCancellation(reason: reason) else { return }
        lastTransitionReason = reason
        AppCancellationBus.shared.markCancellationRequested(reason)
        AppCancellationBus.shared.cancelAllSceneSensitiveDeferred(priority: .utility)
        Task.detached(priority: .userInitiated) {
            await AppLlamaService.shared.cancelActiveGeneration(reason: reason)
        }
        scheduleDeferredSceneCleanup(reason: reason)
        scheduleCancellationDiagnostics(reason: reason, phaseName: currentPhaseName)
    }

    private func shouldIssueCancellation(reason: String) -> Bool {
        let now = Date()
        if let lastIssuedCancellationAt,
           now.timeIntervalSince(lastIssuedCancellationAt) < 1.5 {
            return false
        }
        lastIssuedCancellationAt = now
        return true
    }

    private func scheduleDeferredSceneCleanup(reason: String) {
        deferredCleanupTask?.cancel()
        deferredCleanupTask = Task { @MainActor in
            await Task.yield()
            guard !Task.isCancelled, currentPhase != .active else { return }
            ModelLoader.cancelActiveLoads()
            VoiceService.shared.stopListening()
            VoiceService.shared.stopSpeaking()
            PersistentRuntimeDiagnosticsObserver.shared.emit(.init(kind: .sceneTransition, values: [
                "phase": currentPhaseName,
                "deferredCleanup": "complete",
                "cancellationReason": reason
            ]))
            deferredCleanupTask = nil
        }
    }

    private func scheduleCancellationDiagnostics(reason: String, phaseName: String) {
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

    #if DEBUG
    func resetForTesting() {
        currentPhase = .active
        lastTransitionReason = nil
        lastIssuedCancellationAt = nil
        cancelDeferredCleanup()
    }
    #endif
}
