import Foundation

nonisolated enum ApprovalSource: String, Sendable, Codable {
    case uiConfirmation
}

nonisolated struct ToolApprovalState: Sendable, Codable, Hashable {
    let pendingActionID: UUID
    let toolID: String
    let arguments: AgentJSONArguments
    let approvedAt: Date?
    let source: ApprovalSource

    var executionApproval: ToolExecutionApproval {
        guard source == .uiConfirmation else { return .autonomous }
        return approvedAt == nil ? .pending : .userApproved
    }
}

nonisolated enum ExecutorActionKind: String, Sendable, Codable {
    case executeTool
    case requestApproval
    case requestPermission
    case clarification
    case reject
}

nonisolated struct ExecutorPendingApproval: Sendable, Codable, Hashable {
    let pendingActionID: UUID
    let toolID: String
    let arguments: AgentJSONArguments
    let confirmationMessage: String
    let reason: String
}

@MainActor
final class ToolApprovalQueue {
    static let shared = ToolApprovalQueue()
    private static let maxPending = 256
    private var pendingByID: [UUID: ExecutorPendingApproval] = [:]
    private var insertionOrder: [UUID] = []
    private init() {}

    func enqueue(toolID: String, toolName: String, arguments: [String: String]) -> ExecutorPendingApproval {
        let pending = ExecutorPendingApproval(
            pendingActionID: UUID(),
            toolID: toolID,
            arguments: AgentJSONArguments(stringDictionary: arguments),
            confirmationMessage: "Approve \(toolName) with arguments: \(arguments)",
            reason: "requiresApproval"
        )
        if pendingByID[pending.pendingActionID] == nil {
            insertionOrder.append(pending.pendingActionID)
        }
        pendingByID[pending.pendingActionID] = pending
        pruneIfNeeded()
        return pending
    }

    func resolve(_ pendingActionID: UUID) -> ExecutorPendingApproval? { pendingByID[pendingActionID] }
    func clear(_ pendingActionID: UUID) {
        pendingByID.removeValue(forKey: pendingActionID)
        insertionOrder.removeAll { $0 == pendingActionID }
    }
    func consume(_ pendingActionID: UUID) -> ExecutorPendingApproval? {
        let resolved = pendingByID.removeValue(forKey: pendingActionID)
        insertionOrder.removeAll { $0 == pendingActionID }
        return resolved
    }

    private func pruneIfNeeded() {
        while pendingByID.count > Self.maxPending, let oldest = insertionOrder.first {
            insertionOrder.removeFirst()
            pendingByID.removeValue(forKey: oldest)
        }
    }
}

nonisolated enum ApprovalBoundaryFormatter {
    static func approvalMessage(for pending: ExecutorPendingApproval) -> String {
        "Approval required before running \(pending.toolID). Pending action id: \(pending.pendingActionID.uuidString)."
    }
}
