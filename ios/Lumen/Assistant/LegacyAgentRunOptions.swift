import Foundation
import SwiftData

nonisolated struct LegacyAgentRunOptions: @unchecked Sendable {
    nonisolated enum GroundingMode: Sendable, Equatable { case foregroundChat, headlessTrigger, slotAgent, rolePipeline }

    var modelContext: ModelContext?
    var conversationID: UUID?
    var turnID: UUID?
    var groundingMode: GroundingMode
    var allowDegradedGrounding: Bool
    var preventDoubleGrounding: Bool
    var diagnosticsEnabled: Bool
    var allowDeterministicCompatibility: Bool = true

    static var `default`: LegacyAgentRunOptions {
        .init(modelContext: nil, conversationID: nil, turnID: nil, groundingMode: .foregroundChat, allowDegradedGrounding: true, preventDoubleGrounding: true, diagnosticsEnabled: false)
    }
}
