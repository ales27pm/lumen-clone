import Foundation
import SwiftUI
import UIKit

@MainActor
enum ResourceBudgetGate {
    struct Snapshot: Equatable {
        let scenePhase: ScenePhase?
        let lowPowerModeEnabled: Bool?
        let thermalState: DeviceThermalState?
        let recentMemoryWarningCount: Int?
        let lastMemoryWarningAt: Date?

        @MainActor
        static var current: Snapshot {
            Snapshot(
                scenePhase: inferredScenePhase(),
                lowPowerModeEnabled: ProcessInfo.processInfo.isLowPowerModeEnabled,
                thermalState: .from(processThermalState: ProcessInfo.processInfo.thermalState),
                recentMemoryWarningCount: MemoryPressureMonitor.shared.recentWarningCount(),
                lastMemoryWarningAt: MemoryPressureMonitor.shared.lastWarningAt
            )
        }
    }

    #if DEBUG
    static var testSnapshotOverride: Snapshot?

    static func setDiagnosticSnapshotOverride(_ snapshot: Snapshot?) {
        testSnapshotOverride = snapshot
    }

    static func clearDiagnosticSnapshotOverride() {
        testSnapshotOverride = nil
    }
    #endif

    private static var lastScenePhase: ScenePhase?

    static func recordScenePhase(_ phase: ScenePhase) {
        lastScenePhase = phase
    }

    static func diagnosticSnapshot() -> Snapshot {
        currentSnapshot()
    }

    static func allowsHeavyModelWork(reason: String) -> Bool {
        allowsHeavyModelWork(snapshot: currentSnapshot(), reason: reason)
    }

    static func allowsHeavyModelWork(snapshot: Snapshot, reason: String) -> Bool {
        guard !hasRecentMemoryWarning(snapshot) else { return false }
        guard let thermal = snapshot.thermalState else { return false }
        guard thermal != .serious, thermal != .critical, thermal != .unknown else { return false }
        guard snapshot.lowPowerModeEnabled != nil else { return false }
        return true
    }

    static func allowsForegroundModelLoad(reason: String) -> Bool {
        return allowsHeavyModelWork(reason: reason)
    }

    static func shouldCancelForScenePhase(_ phase: ScenePhase) -> Bool {
        false
    }

    static func allowsMaintenance(reason: String) -> Bool {
        let snapshot = currentSnapshot()
        guard !hasRecentMemoryWarning(snapshot) else { return false }
        if let thermal = snapshot.thermalState, thermal == .serious || thermal == .critical || thermal == .unknown { return false }
        return true
    }

    private static func currentSnapshot() -> Snapshot {
        #if DEBUG
        if let testSnapshotOverride { return testSnapshotOverride }
        #endif
        var snapshot = Snapshot.current
        if snapshot.scenePhase == nil, let lastScenePhase {
            snapshot = Snapshot(
                scenePhase: lastScenePhase,
                lowPowerModeEnabled: snapshot.lowPowerModeEnabled,
                thermalState: snapshot.thermalState,
                recentMemoryWarningCount: snapshot.recentMemoryWarningCount,
                lastMemoryWarningAt: snapshot.lastMemoryWarningAt
            )
        }
        return snapshot
    }

    private static func hasRecentMemoryWarning(_ snapshot: Snapshot) -> Bool {
        guard let warnings = snapshot.recentMemoryWarningCount else { return true }
        guard warnings > 0 else { return false }
        guard let lastWarningAt = snapshot.lastMemoryWarningAt else { return true }
        return Date().timeIntervalSince(lastWarningAt) < MemoryPressureMonitor.modelLoadSuppressionInterval
    }

    private static func inferredScenePhase() -> ScenePhase? {
        if let lastScenePhase { return lastScenePhase }
        switch UIApplication.shared.applicationState {
        case .active: return .active
        case .inactive: return .inactive
        case .background: return .background
        @unknown default: return nil
        }
    }
}

@MainActor
enum RuntimeLifecycleCanceller {
    private static var lastCancellationAt = Date.distantPast

    static func cancelForSceneTransition(reason: String = "scene-transition") {
        let now = Date()
        guard now.timeIntervalSince(lastCancellationAt) >= 1.5 else { return }
        lastCancellationAt = now
        AppCancellationBus.shared.markCancellationRequested(reason)
        AppCancellationBus.shared.cancelAllSceneSensitiveDeferred(priority: .utility)
        ModelLoader.cancelActiveLoads()
        Task.detached(priority: .userInitiated) {
            await AppLlamaService.shared.cancelActiveGeneration(reason: reason)
        }
        Task { @MainActor in
            await Task.yield()
            guard ResourceBudgetGate.diagnosticSnapshot().scenePhase != .active else { return }
            VoiceService.shared.stopListening()
            VoiceService.shared.stopSpeaking()
        }
        DeferredMaintenanceQueue.shared.enqueue(
            DeferredMaintenanceJob(key: "runtime-cleanup-optional-chat-slots", category: .persistence, staleAfter: 10 * 60, maxRuntime: 2) {
                await MainActor.run { FleetRuntimeCleanup.unloadOptionalChatSlots() }
            }
        )
    }
}
