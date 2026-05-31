import Foundation

enum BackgroundTaskKind: String, Sendable {
    case triggerScan
    case memoryConsolidation
    case ragMaintenance
    case modelHousekeeping
}

struct BackgroundTaskPolicyInput: Sendable, Equatable {
    let taskKind: BackgroundTaskKind
    let lowPowerMode: Bool
    let thermalState: DeviceThermalState
    let isForeground: Bool
    let backgroundAgentsEnabled: Bool
    let requiresNetwork: Bool
    let estimatedCost: Int
}

struct BackgroundTaskDecision: Sendable, Equatable {
    let allow: Bool
    let denyReason: String?
    let maxSteps: Int
    let maxTokens: Int
    let allowModelLoading: Bool
    let allowNetwork: Bool
}

enum BackgroundTaskPolicy {
    static func decide(_ input: BackgroundTaskPolicyInput) -> BackgroundTaskDecision {
        guard input.backgroundAgentsEnabled else {
            return .init(allow: false, denyReason: "background agents disabled", maxSteps: 0, maxTokens: 0, allowModelLoading: false, allowNetwork: false)
        }
        if input.thermalState == .serious || input.thermalState == .critical {
            return .init(allow: false, denyReason: "thermal state disallows background work", maxSteps: 0, maxTokens: 0, allowModelLoading: false, allowNetwork: false)
        }
        let isLowPowerBackground = input.lowPowerMode && !input.isForeground
        if isLowPowerBackground && input.taskKind != .triggerScan {
            return .init(allow: false, denyReason: "low power background mode", maxSteps: 0, maxTokens: 0, allowModelLoading: false, allowNetwork: false)
        }

        let allowModelLoading = !(!input.isForeground && input.estimatedCost > 5)
        let allowNetwork = input.requiresNetwork && !isLowPowerBackground
        if input.requiresNetwork && !allowNetwork {
            return .init(allow: false, denyReason: "network unavailable for required background task", maxSteps: 0, maxTokens: 0, allowModelLoading: false, allowNetwork: false)
        }
        return .init(allow: true, denyReason: nil, maxSteps: input.taskKind == .triggerScan ? 2 : 1, maxTokens: 256, allowModelLoading: allowModelLoading, allowNetwork: allowNetwork)
    }
}
