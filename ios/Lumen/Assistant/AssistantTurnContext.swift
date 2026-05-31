import Foundation

enum AssistantTaskKind: Sendable, Equatable {
    case chat
    case agentPlan
    case toolDecision
    case embedding
    case summarization
    case memoryExtraction
    case safetyClassification
    case speechCommandParsing
    case backgroundTrigger
    case remConsolidation
}

struct AssistantTurnContext: Sendable, Equatable {
    let task: AssistantTaskKind
    let input: String
    let isForeground: Bool
    let lowPowerMode: Bool
    let thermalState: ProcessInfo.ThermalState
    let prefersFoundationModels: Bool

    init(
        task: AssistantTaskKind,
        input: String,
        isForeground: Bool,
        lowPowerMode: Bool,
        thermalState: ProcessInfo.ThermalState,
        prefersFoundationModels: Bool = true
    ) {
        self.task = task
        self.input = input
        self.isForeground = isForeground
        self.lowPowerMode = lowPowerMode
        self.thermalState = thermalState
        self.prefersFoundationModels = prefersFoundationModels
    }
}
