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

        static var current: Snapshot {
            Snapshot(
                scenePhase: inferredScenePhase(),
                lowPowerModeEnabled: ProcessInfo.processInfo.isLowPowerModeEnabled,
                thermalState: .from(processThermalState: ProcessInfo.processInfo.thermalState),
                recentMemoryWarningCount: MemoryPressureMonitor.shared.warningCount
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
        guard let warnings = snapshot.recentMemoryWarningCount, warnings == 0 else { return false }
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
                recentMemoryWarningCount: snapshot.recentMemoryWarningCount
            )
        }
        return snapshot
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
        ModelLoader.cancelActiveLoads()
        VoiceService.shared.stopListening()
        VoiceService.shared.stopSpeaking()
        Task { @MainActor in FleetRuntimeCleanup.unloadOptionalChatSlots() }
    }
}
