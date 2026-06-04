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
    #endif

    private static var lastScenePhase: ScenePhase?

    static func recordScenePhase(_ phase: ScenePhase) {
        lastScenePhase = phase
    }

    static func allowsHeavyModelWork(reason: String) -> Bool {
        let snapshot = currentSnapshot()
        guard snapshot.scenePhase == .active else { return false }
        guard !hasRecentMemoryWarning(snapshot) else { return false }
        guard let thermal = snapshot.thermalState else { return false }
        guard thermal != .serious, thermal != .critical, thermal != .unknown else { return false }
        guard let lowPower = snapshot.lowPowerModeEnabled else { return false }
        if lowPower && !isExplicitUserTurn(reason) { return false }
        return true
    }

    static func allowsForegroundModelLoad(reason: String) -> Bool {
        guard isExplicitUserTurn(reason) else { return false }
        return allowsHeavyModelWork(reason: reason)
    }

    static func shouldCancelForScenePhase(_ phase: ScenePhase) -> Bool {
        phase == .inactive || phase == .background
    }

    static func allowsMaintenance(reason: String) -> Bool {
        let snapshot = currentSnapshot()
        guard snapshot.scenePhase == .active else { return false }
        guard !hasRecentMemoryWarning(snapshot) else { return false }
        if let thermal = snapshot.thermalState, thermal == .serious || thermal == .critical || thermal == .unknown { return false }
        if snapshot.lowPowerModeEnabled == true { return false }
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

    private static func isExplicitUserTurn(_ reason: String) -> Bool {
        let normalized = reason.lowercased()
        return normalized.contains(ModelLoadIntent.userChat.rawValue.lowercased()) || normalized.contains(ModelLoadIntent.userVoice.rawValue.lowercased())
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
    static func cancelForSceneTransition(reason: String = "scene-transition") {
        AppCancellationBus.shared.markCancellationRequested(reason)
        AppCancellationBus.shared.cancelAllSceneSensitive()
        ModelLoader.cancelActiveLoads()
        VoiceService.shared.stopListening()
        VoiceService.shared.stopSpeaking()
        Task.detached(priority: .userInitiated) {
            await AppLlamaService.shared.cancelActiveGeneration(reason: reason)
        }
        DeferredMaintenanceQueue.shared.enqueue(
            DeferredMaintenanceJob(key: "runtime-cleanup-optional-chat-slots", category: .persistence, staleAfter: 10 * 60, maxRuntime: 2) {
                await MainActor.run { FleetRuntimeCleanup.unloadOptionalChatSlots() }
            }
        )
    }
}
