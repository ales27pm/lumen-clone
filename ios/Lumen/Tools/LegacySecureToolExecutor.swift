import Foundation
import SwiftData

@MainActor
enum LegacySecureToolExecutor {
    private static let readOnlyAllowlist: Set<String> = ["weather", "maps.search", "rag.search", "memory.recall", "files.read", "trigger.list"]

    static func execute(_ toolID: String, arguments: [String: String], approval: ToolExecutionApproval = .autonomous, conversationID: UUID? = nil, turnID: UUID? = nil, modelContext: ModelContext? = nil, isBackground: Bool = false) async -> String {
        await execute(toolID: toolID, arguments: AgentJSONArguments(stringDictionary: arguments), approval: approval, conversationID: conversationID, turnID: turnID, modelContext: modelContext, isBackground: isBackground)
    }

    static func execute(_ toolID: String, arguments: AgentJSONArguments, approval: ToolExecutionApproval = .autonomous, conversationID: UUID? = nil, turnID: UUID? = nil, modelContext: ModelContext? = nil, isBackground: Bool = false) async -> String {
        await execute(toolID: toolID, arguments: arguments, approval: approval, conversationID: conversationID, turnID: turnID, modelContext: modelContext, isBackground: isBackground)
    }

    static func execute(toolID: String, arguments: AgentJSONArguments, approval: ToolExecutionApproval = .autonomous, conversationID: UUID? = nil, turnID: UUID? = nil, modelContext: ModelContext? = nil, isBackground: Bool = false) async -> String {
        let canonical = ToolRouteGuard.canonicalToolID(toolID)
        let mappedSecureID: String? = {
            switch canonical {
            case "rag.search": return "rag.search.secure"
            case "memory.recall": return "memory.search"
            case "contacts.search": return "contacts.lookup"
            case "location.current": return "location.snapshot"
            default: return nil
            }
        }()
        if let mappedSecureID {
            let invocation = ToolInvocation(id: UUID(), toolID: mappedSecureID, arguments: arguments.stringCoerced, source: isBackground ? .backgroundTrigger : .modelProposed, conversationID: conversationID, turnID: turnID, createdAt: Date())
            let ctx = ToolExecutionContext(isForeground: !isBackground, appState: nil, modelContext: modelContext, permissionRegistry: .shared, metricsStore: .shared)
            let result = await SecureToolRegistry.shared.execute(invocation, context: ctx)
            return result.modelText
        }

        let lower = canonical.lowercased()
        if lower.contains("delete") || lower.contains("send") || lower.contains("open") || lower.contains("call") || lower.contains("mail") || lower.contains("message") || lower.contains("web") {
            return "Tool denied by legacy secure policy. Open the app to approve this action."
        }
        guard readOnlyAllowlist.contains(canonical) else {
            return "Tool unavailable pending secure migration."
        }
        return await ToolExecutor.shared.execute(canonical, arguments: arguments, approval: approval)
    }
}
