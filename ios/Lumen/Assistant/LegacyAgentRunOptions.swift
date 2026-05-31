import Foundation
import SwiftData

struct LegacyAgentRunOptions {
    enum GroundingMode: Sendable, Equatable { case foregroundChat, headlessTrigger, slotAgent, rolePipeline }

    var modelContext: ModelContext?
    var conversationID: UUID?
    var turnID: UUID?
    var groundingMode: GroundingMode
    var allowDegradedGrounding: Bool
    var preventDoubleGrounding: Bool
    var diagnosticsEnabled: Bool

    static var `default`: LegacyAgentRunOptions {
        .init(modelContext: nil, conversationID: nil, turnID: nil, groundingMode: .foregroundChat, allowDegradedGrounding: true, preventDoubleGrounding: true, diagnosticsEnabled: false)
    }
}
