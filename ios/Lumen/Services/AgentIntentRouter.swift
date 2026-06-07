import Foundation

/// Compatibility facade for older agent-routing tests/call sites.
///
/// `IntentRouter` is the single source of truth for classification and tool
/// scoping. This wrapper deliberately contains no independent enum, scoring
/// table, or tool matrix so production routing and deterministic compatibility
/// routing cannot drift.
nonisolated enum AgentIntentRouter {
    typealias Intent = UserIntent

    struct Decision: Sendable, Equatable {
        let intent: Intent
        let confidence: Int
        let reason: String
        let allowedToolIDs: Set<String>
        let requiresUserApproval: Bool
        let shouldAskClarification: Bool
        let clarificationQuestion: String?
        let alternatives: [Intent]

        var allowsTools: Bool { !allowedToolIDs.isEmpty }
    }

    static func decide(userMessage: String, attachments: [ChatAttachment] = []) -> Decision {
        let routing = IntentRouter.classify(userMessage)
        let allowed = routing.allowedToolIDs
        let approvalRequired = allowed.contains { ToolRouteGuard.requiresUserApproval($0) }
        let attachmentReason = attachments.isEmpty ? "" : "; attachments present, but routing still delegates to IntentRouter"
        return Decision(
            intent: routing.intent,
            confidence: 100,
            reason: "delegated to IntentRouter.classify\(attachmentReason)",
            allowedToolIDs: allowed,
            requiresUserApproval: approvalRequired,
            shouldAskClarification: routing.requiresClarification,
            clarificationQuestion: routing.clarificationPrompt,
            alternatives: []
        )
    }

    static func filteredTools(from enabledTools: [ToolDefinition], userMessage: String, attachments: [ChatAttachment] = []) -> [ToolDefinition] {
        let decision = decide(userMessage: userMessage, attachments: attachments)
        guard decision.allowsTools, !decision.shouldAskClarification else { return [] }
        return enabledTools.filter { decision.allowedToolIDs.contains(ToolRouteGuard.canonicalToolID($0.id)) }
    }

    static func routingSystemNote(for decision: Decision) -> String {
        if decision.shouldAskClarification {
            return "\n\nRouting: ask one concise clarification question before using tools."
        }
        if decision.allowedToolIDs.isEmpty {
            return "\n\nRouting: answer directly; no tool is available for this turn."
        }
        let tools = decision.allowedToolIDs.sorted().joined(separator: ", ")
        return "\n\nRouting: IntentRouter selected \(decision.intent.rawValue). Allowed tools: \(tools)."
    }

    static func allowedToolIDs(for intent: Intent) -> Set<String> {
        IntentRouter.allowedToolIDs(for: intent)
    }
}
