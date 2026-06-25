import Foundation

struct ComputePolicyInput: Sendable, Equatable {
    let task: AssistantTaskKind
    let isForeground: Bool
    let lowPowerMode: Bool
    let thermalState: DeviceThermalState
    let allowHeavyRuntime: Bool

    init(
        task: AssistantTaskKind,
        isForeground: Bool,
        lowPowerMode: Bool,
        thermalState: DeviceThermalState,
        allowHeavyRuntime: Bool = true
    ) {
        self.task = task
        self.isForeground = isForeground
        self.lowPowerMode = lowPowerMode
        self.thermalState = thermalState
        self.allowHeavyRuntime = allowHeavyRuntime
    }
}

struct ComputeDecision: Sendable, Equatable {
    let maxTokens: Int
    let allowHeavyRuntime: Bool
    let budgetPolicy: LumenSlotBudgetPolicy
    let denialReason: String?
}

enum ComputePolicy {
    static func decide(for input: ComputePolicyInput) -> ComputeDecision {
        let budgetPolicy = budgetPolicy(for: input.task)
        let thermalLimited = input.thermalState == .serious || input.thermalState == .critical || input.thermalState == .unknown
        if !input.allowHeavyRuntime {
            return ComputeDecision(maxTokens: 512, allowHeavyRuntime: false, budgetPolicy: budgetPolicy, denialReason: "\(budgetPolicy.rawValue): heavyRuntime=false")
        }
        if thermalLimited {
            return ComputeDecision(maxTokens: 512, allowHeavyRuntime: false, budgetPolicy: budgetPolicy, denialReason: "\(budgetPolicy.rawValue): thermalState=\(input.thermalState.rawValue)")
        }

        switch budgetPolicy {
        case .foregroundInteractive:
            guard input.isForeground else {
                return ComputeDecision(maxTokens: 256, allowHeavyRuntime: false, budgetPolicy: budgetPolicy, denialReason: "\(budgetPolicy.rawValue): scenePhase=background")
            }
            return ComputeDecision(maxTokens: input.lowPowerMode ? 512 : 1024, allowHeavyRuntime: true, budgetPolicy: budgetPolicy, denialReason: nil)
        case .maintenanceIdle:
            guard !input.lowPowerMode else {
                return ComputeDecision(maxTokens: 512, allowHeavyRuntime: false, budgetPolicy: budgetPolicy, denialReason: "\(budgetPolicy.rawValue): lowPowerMode=true")
            }
            return ComputeDecision(maxTokens: input.isForeground ? 768 : 512, allowHeavyRuntime: true, budgetPolicy: budgetPolicy, denialReason: nil)
        case .embedding:
            if input.lowPowerMode && !input.isForeground {
                return ComputeDecision(maxTokens: 256, allowHeavyRuntime: false, budgetPolicy: budgetPolicy, denialReason: "\(budgetPolicy.rawValue): lowPowerMode=true")
            }
            return ComputeDecision(maxTokens: input.lowPowerMode ? 256 : 512, allowHeavyRuntime: true, budgetPolicy: budgetPolicy, denialReason: nil)
        }
    }

    static func decide(for context: AssistantTurnContext) -> ComputeDecision {
        let input = ComputePolicyInput(
            task: context.task,
            isForeground: context.isForeground,
            lowPowerMode: context.lowPowerMode,
            thermalState: .from(processThermalState: context.thermalState),
            allowHeavyRuntime: context.allowHeavyRuntime
        )
        return decide(for: input)
    }

    private static func budgetPolicy(for task: AssistantTaskKind) -> LumenSlotBudgetPolicy {
        switch task {
        case .embedding, .safetyClassification:
            return .embedding
        case .backgroundTrigger, .remConsolidation:
            return .maintenanceIdle
        case .chat, .agentPlan, .toolDecision, .summarization, .memoryExtraction, .speechCommandParsing:
            return .foregroundInteractive
        }
    }
}
