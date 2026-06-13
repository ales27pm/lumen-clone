import Foundation
import SwiftData

struct AgentKernelRunOptions: @unchecked Sendable {
    enum GroundingMode: Sendable, Equatable {
        case foregroundChat
        case headlessTrigger
        case slotAgent
        case rolePipeline
    }

    var modelContext: ModelContext?
    var conversationID: UUID?
    var turnID: UUID?
    var groundingMode: GroundingMode
    var allowDegradedGrounding: Bool
    var preventDoubleGrounding: Bool
    var diagnosticsEnabled: Bool

    static var `default`: AgentKernelRunOptions {
        .init(
            modelContext: nil,
            conversationID: nil,
            turnID: nil,
            groundingMode: .foregroundChat,
            allowDegradedGrounding: true,
            preventDoubleGrounding: true,
            diagnosticsEnabled: false
        )
    }
}

extension AgentKernelRunOptions {
    init(_ legacy: LegacyAgentRunOptions) {
        self.init(
            modelContext: legacy.modelContext,
            conversationID: legacy.conversationID,
            turnID: legacy.turnID,
            groundingMode: .init(legacy.groundingMode),
            allowDegradedGrounding: legacy.allowDegradedGrounding,
            preventDoubleGrounding: legacy.preventDoubleGrounding,
            diagnosticsEnabled: legacy.diagnosticsEnabled
        )
    }
}

extension AgentKernelRunOptions.GroundingMode {
    init(_ legacy: LegacyAgentRunOptions.GroundingMode) {
        switch legacy {
        case .foregroundChat: self = .foregroundChat
        case .headlessTrigger: self = .headlessTrigger
        case .slotAgent: self = .slotAgent
        case .rolePipeline: self = .rolePipeline
        }
    }

    var legacyGroundingMode: LegacyAgentRunOptions.GroundingMode {
        switch self {
        case .foregroundChat: return .foregroundChat
        case .headlessTrigger: return .headlessTrigger
        case .slotAgent: return .slotAgent
        case .rolePipeline: return .rolePipeline
        }
    }
}

extension LegacyAgentRunOptions {
    init(_ kernel: AgentKernelRunOptions) {
        self.init(
            modelContext: kernel.modelContext,
            conversationID: kernel.conversationID,
            turnID: kernel.turnID,
            groundingMode: kernel.groundingMode.legacyGroundingMode,
            allowDegradedGrounding: kernel.allowDegradedGrounding,
            preventDoubleGrounding: kernel.preventDoubleGrounding,
            diagnosticsEnabled: kernel.diagnosticsEnabled
        )
    }
}
