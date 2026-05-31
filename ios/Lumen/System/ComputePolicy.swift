import Foundation

struct ComputePolicyInput: Sendable, Equatable {
    let task: AssistantTaskKind
    let isForeground: Bool
    let lowPowerMode: Bool
    let thermalState: DeviceThermalState
}

struct ComputeDecision: Sendable, Equatable {
    let maxTokens: Int
    let allowHeavyRuntime: Bool
}

enum ComputePolicy {
    static func decide(for input: ComputePolicyInput) -> ComputeDecision {
        let thermalLimited = input.thermalState == .serious || input.thermalState == .critical
        if !input.isForeground {
            return ComputeDecision(maxTokens: 256, allowHeavyRuntime: false)
        }
        if thermalLimited || input.lowPowerMode {
            return ComputeDecision(maxTokens: 512, allowHeavyRuntime: false)
        }
        return ComputeDecision(maxTokens: 1024, allowHeavyRuntime: true)
    }

    static func decide(for context: AssistantTurnContext) -> ComputeDecision {
        let input = ComputePolicyInput(
            task: context.task,
            isForeground: context.isForeground,
            lowPowerMode: context.lowPowerMode,
            thermalState: .from(processThermalState: context.thermalState)
        )
        return decide(for: input)
    }
}
