import Foundation

/// Compatibility facade for older agent-routing tests/call sites.
///
/// `IntentRouter` is the single source of truth for classification and tool
/// scoping. This wrapper deliberately contains no independent enum, scoring
/// table, or tool matrix so production routing and deterministic compatibility
/// routing cannot drift.
///
/// Attachments are already injected into the assembled prompt/context before
/// agent execution. This wrapper does not reinterpret attachments as a separate
/// routing signal; callers that need attachment-specific behavior must express
/// that request in the user message before calling `decide`.
nonisolated enum AgentIntentRouter {
    typealias Intent = UserIntent

    struct Decision: Sendable, Equatable {
        let intent: Intent
        /// Compatibility score only. `IntentRouter` does not expose model/scoring
        /// confidence, so downstream diagnostics must not treat this as a real
        /// classifier probability.
        let confidence: Int
        let confidenceSource: String
        let reason: String
        let allowedToolIDs: Set<String>
        let requiresUserApproval: Bool
        let shouldAskClarification: Bool
        let clarificationQuestion: String?
        let alternatives: [Intent]
        let attachmentsWerePresent: Bool
        let attachmentsAffectRouting: Bool

        var allowsTools: Bool { !allowedToolIDs.isEmpty }
        var compatibilityIntentName: String { shouldAskClarification ? "clarify" : intent.rawValue }
    }

    static func decide(userMessage: String, attachments: [ChatAttachment] = []) -> Decision {
        let classification = IntentClarificationPolicy.apply(DeterministicIntentFallback.classify(userMessage), to: userMessage)
        return decision(
            routing: classification.asRoutingDecision(),
            alternatives: classification.alternatives.map(\.intent),
            reasonPrefix: "delegated to deterministic semantic routing",
            attachments: attachments
        )
    }

    static func decideSemantic(userMessage: String, attachments: [ChatAttachment] = []) async -> Decision {
        let classification = await IntentClassifierService.shared.classify(userMessage)
        return decision(
            routing: classification.asRoutingDecision(),
            alternatives: classification.alternatives.map(\.intent),
            reasonPrefix: "delegated to IntentClassifierService",
            attachments: attachments
        )
    }

    private static func decision(
        routing: IntentRoutingDecision,
        alternatives: [Intent],
        reasonPrefix: String,
        attachments: [ChatAttachment]
    ) -> Decision {
        let allowed = routing.allowedToolIDs
        let approvalRequired = allowed.contains { ToolRouteGuard.requiresUserApproval($0) }
        let confidence = compatibilityConfidence(for: routing, attachmentsWerePresent: !attachments.isEmpty)
        let attachmentReason = attachments.isEmpty ? "" : "; attachments are prompt context only and do not alter routing"
        return Decision(
            intent: routing.intent,
            confidence: confidence.value,
            confidenceSource: confidence.source,
            reason: "\(reasonPrefix)\(attachmentReason)",
            allowedToolIDs: allowed,
            requiresUserApproval: approvalRequired,
            shouldAskClarification: routing.requiresClarification,
            clarificationQuestion: routing.clarificationPrompt,
            alternatives: Array(alternatives.reduce(into: [Intent]()) { acc, item in
                if !acc.contains(item) {
                    acc.append(item)
                }
            }.prefix(5)),
            attachmentsWerePresent: !attachments.isEmpty,
            attachmentsAffectRouting: false
        )
    }

    static func filteredTools(from enabledTools: [ToolDefinition], userMessage: String, attachments: [ChatAttachment] = []) -> [ToolDefinition] {
        let decision = decide(userMessage: userMessage, attachments: attachments)
        guard decision.allowsTools, !decision.shouldAskClarification else { return [] }
        return enabledTools.filter { decision.allowedToolIDs.contains(ToolRouteGuard.canonicalToolID($0.id)) }
    }

    static func routingSystemNote(for decision: Decision) -> String {
        if decision.shouldAskClarification {
            return "\n\nRouting: ask one concise clarification question before using tools. Compatibility intent: \(decision.compatibilityIntentName)."
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

    private static func compatibilityConfidence(for routing: IntentRoutingDecision, attachmentsWerePresent: Bool) -> (value: Int, source: String) {
        if routing.requiresClarification {
            return (65, "compatibility:clarification-required")
        }
        if routing.intent == .unknown {
            return (0, "compatibility:unknown-intent")
        }
        if attachmentsWerePresent && routing.allowedToolIDs.isEmpty {
            return (92, "compatibility:attachment-context-direct-answer")
        }
        if !routing.allowedToolIDs.isEmpty {
            return (90, "compatibility:intentrouter-tool-scope")
        }
        return (82, "compatibility:intentrouter-direct-answer")
    }
}
