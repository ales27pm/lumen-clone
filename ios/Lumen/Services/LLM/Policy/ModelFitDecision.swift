import Foundation

enum ModelFitDecision: Sendable, Codable, Equatable {
    case allowed(ModelFitReport)
    case downgraded(ModelFitReport)
    case rejected(ModelFitReport)

    var isAllowed: Bool {
        switch self {
        case .allowed, .downgraded:
            return true
        case .rejected:
            return false
        }
    }

    var isRejected: Bool {
        if case .rejected = self {
            return true
        }
        return false
    }

    var report: ModelFitReport {
        switch self {
        case .allowed(let report), .downgraded(let report), .rejected(let report):
            return report
        }
    }
}

struct ModelFitReport: Sendable, Codable, Equatable {
    let model: LocalLLMModel
    let snapshot: DeviceCapabilitySnapshot
    let requestedProfile: InferenceProfile
    let selectedProfile: InferenceProfile
    let requestedBudget: InferenceBudget
    let selectedBudget: InferenceBudget
    let memoryEstimate: ModelMemoryEstimate
    let reasons: [ModelFitReason]
}

enum ModelFitReason: String, Sendable, Codable, Equatable, CaseIterable {
    case modelFitsMemoryBudget
    case modelExceedsMemoryBudget
    case modelExceedsDeviceTier
    case simulatorDisablesMetal
    case metalUnavailable
    case thermalPressureTooHigh
    case lowPowerModeDowngrade
    case backgroundExecutionBlocked
    case contextReduced
    case completionBudgetReduced
    case gpuDisabled
    case tinyIntentAlwaysAllowed
    case remoteDoesNotNeedLocalModelMemory
    case unknownModelSize
}
