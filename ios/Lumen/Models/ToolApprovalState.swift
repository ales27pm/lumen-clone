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

nonisolated enum ToolApprovalPayloadVerificationError: String, Error, Sendable, Equatable {
    case missingPendingActionID
    case malformedPendingActionID
    case expiredOrMismatchedPendingAction

    var userMessage: String {
        switch self {
        case .missingPendingActionID, .malformedPendingActionID:
            return "Approval request cannot be verified. Ask Lumen to prepare the action again."
        case .expiredOrMismatchedPendingAction:
            return "Approval request expired or does not match this tool. Ask Lumen to prepare the action again."
        }
    }
}

nonisolated struct ExecutorPendingApproval: Sendable, Codable, Hashable {
    let pendingActionID: UUID
    let toolID: String
    let arguments: AgentJSONArguments
    let confirmationMessage: String
    let reason: String
    let createdAt: Date
    let expiresAt: Date

    func isExpired(at now: Date = Date()) -> Bool {
        now >= expiresAt
    }
}

@MainActor
final class ToolApprovalQueue {
    static let shared = ToolApprovalQueue()
    private static let maxPending = 256
    private static let pendingLifetime: TimeInterval = 10 * 60
    private var pendingByID: [UUID: ExecutorPendingApproval] = [:]
    private var insertionOrder: [UUID] = []
    private init() {}

    func enqueue(
        toolID: String,
        toolName: String,
        arguments: [String: String],
        createdAt: Date = Date(),
        expiresAt: Date? = nil
    ) -> ExecutorPendingApproval {
        let resolvedExpiresAt = expiresAt ?? createdAt.addingTimeInterval(Self.pendingLifetime)
        let canonicalToolID = ToolRouteGuard.canonicalToolID(toolID)
        let pending = ExecutorPendingApproval(
            pendingActionID: UUID(),
            toolID: canonicalToolID,
            arguments: AgentJSONArguments(stringDictionary: arguments),
            confirmationMessage: "Approve \(toolName) with arguments: \(arguments)",
            reason: "requiresApproval",
            createdAt: createdAt,
            expiresAt: resolvedExpiresAt
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
    private func consume(_ pendingActionID: UUID) -> ExecutorPendingApproval? {
        let resolved = pendingByID.removeValue(forKey: pendingActionID)
        insertionOrder.removeAll { $0 == pendingActionID }
        return resolved
    }
    func consume(_ pendingActionID: UUID, matchingToolID expectedToolID: String) -> ExecutorPendingApproval? {
        guard let pending = pendingByID[pendingActionID] else { return nil }
        guard !pending.isExpired() else {
            clear(pendingActionID)
            return nil
        }
        guard pending.toolID == expectedToolID else {
            clear(pendingActionID)
            return nil
        }
        return consume(pendingActionID)
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

@MainActor
enum ChatApprovalBoundaryMapper {
    static func pendingToolMessage(
        for step: AgentStep,
        queue: ToolApprovalQueue? = nil
    ) -> ChatMessage? {
        guard step.kind == .approvalBoundary,
              let rawToolID = step.toolID else {
            return nil
        }

        let approvalQueue = queue ?? .shared
        let toolID = ToolRouteGuard.canonicalToolID(rawToolID)
        let toolArgs = step.toolArgs ?? [:]
        let pending = approvalQueue.enqueue(
            toolID: toolID,
            toolName: ToolRegistry.find(id: toolID)?.name ?? toolID,
            arguments: toolArgs
        )
        return ChatMessage(
            role: .tool,
            content: ToolApprovalPayloadCodec.serialize(
                ToolApprovalPayloadCodec.displayArguments(
                    for: pending,
                    visibleArguments: toolArgs
                )
            ),
            toolName: toolID,
            toolStatus: .pendingApproval,
            toolResult: nil
        )
    }
}

nonisolated enum ToolApprovalPayloadCodec {
    static let pendingActionIDKey = "pendingActionID"
    static let legacyPendingActionIDKey = "pending_action_id"

    static func displayArguments(for pending: ExecutorPendingApproval, visibleArguments: [String: String]) -> [String: String] {
        var args = visibleArguments
        args[pendingActionIDKey] = pending.pendingActionID.uuidString
        return args
    }

    static func serialize(_ args: [String: String]) -> String {
        args.keys.sorted()
            .map { key in "\(key): \(args[key] ?? "")" }
            .joined(separator: ", ")
    }

    static func parseLooseArguments(_ string: String) -> [String: String] {
        var out: [String: String] = [:]
        for pair in string.components(separatedBy: ",") {
            let parts = pair.split(separator: ":", maxSplits: 1).map { $0.trimmingCharacters(in: .whitespaces) }
            if parts.count == 2 { out[parts[0]] = parts[1] }
        }
        return out
    }

    static func pendingActionID(from args: [String: String]) -> UUID? {
        let raw = args[pendingActionIDKey] ?? args[legacyPendingActionIDKey]
        return raw.flatMap(UUID.init(uuidString:))
    }

    @MainActor
    static func consumePendingApproval(
        from args: [String: String],
        matchingToolID expectedToolID: String
    ) -> Result<ExecutorPendingApproval, ToolApprovalPayloadVerificationError> {
        consumePendingApproval(from: args, matchingToolID: expectedToolID, queue: .shared)
    }

    @MainActor
    static func consumePendingApproval(
        from args: [String: String],
        matchingToolID expectedToolID: String,
        queue: ToolApprovalQueue
    ) -> Result<ExecutorPendingApproval, ToolApprovalPayloadVerificationError> {
        guard containsPendingActionIDField(args) else {
            return .failure(.missingPendingActionID)
        }
        guard let pendingID = pendingActionID(from: args) else {
            return .failure(.malformedPendingActionID)
        }
        guard let pending = queue.consume(pendingID, matchingToolID: expectedToolID) else {
            return .failure(.expiredOrMismatchedPendingAction)
        }
        return .success(pending)
    }

    static func containsPendingActionIDField(_ args: [String: String]) -> Bool {
        args[pendingActionIDKey] != nil || args[legacyPendingActionIDKey] != nil
    }

    static func removingPendingActionIDFields(_ args: [String: String]) -> [String: String] {
        var copy = args
        copy.removeValue(forKey: pendingActionIDKey)
        copy.removeValue(forKey: legacyPendingActionIDKey)
        return copy
    }
}
