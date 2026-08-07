import Foundation
import OSLog
import SwiftData

enum ToolApprovalPersistenceOperation: String, Sendable {
    case verificationDenied = "approval.verification-denied"
    case policyDenied = "approval.policy-denied"
    case userDenied = "approval.user-denied"
    case runningClaim = "approval.running-claim"
    case terminalOutcome = "approval.terminal-outcome"
    case failureMarker = "approval.failure-marker"
}

enum ToolApprovalPersistenceConsequence: String, Sendable {
    case actionNotRun = "action-not-run"
    case actionBlocked = "action-blocked"
    case terminalStateUncertain = "terminal-state-uncertain"
}

struct ToolApprovalPersistenceFailure: Error, Equatable, Sendable {
    let operation: ToolApprovalPersistenceOperation
    let errorCode: String
    let consequence: ToolApprovalPersistenceConsequence

    var userMessage: String {
        switch consequence {
        case .actionNotRun:
            return "The action was not run because Lumen could not save its approval state. Ask Lumen to prepare it again. Diagnostic: \(errorCode)."
        case .actionBlocked:
            return "The action remains blocked, but Lumen could not save that status. It will not run. Diagnostic: \(errorCode)."
        case .terminalStateUncertain:
            return "The action may have completed, but Lumen could not save its result. It will not run again. Diagnostic: \(errorCode)."
        }
    }
}

@MainActor
enum ToolApprovalPersistenceCoordinator {
    enum PersistenceOutcome: Equatable {
        case persisted
        case failed(ToolApprovalPersistenceFailure)
    }

    enum ClaimOutcome: Equatable {
        case claimed(ExecutorPendingApproval)
        case rejected
        case blocked(ToolApprovalPersistenceFailure)
    }

    private static let logger = Logger(subsystem: "ai.lumen.app", category: "tool-approval-persistence")

    static func claimForExecution(
        message: ChatMessage,
        toolID: String,
        payloadArguments: [String: String],
        policyAllowed: Bool,
        policyDeniedResult: String,
        queue: ToolApprovalQueue? = nil,
        context: ModelContext
    ) -> ClaimOutcome {
        claimForExecution(
            message: message,
            toolID: toolID,
            payloadArguments: payloadArguments,
            policyAllowed: policyAllowed,
            policyDeniedResult: policyDeniedResult,
            queue: queue,
            save: { try context.save() }
        )
    }

    static func claimForExecution(
        message: ChatMessage,
        toolID: String,
        payloadArguments: [String: String],
        policyAllowed: Bool,
        policyDeniedResult: String,
        queue: ToolApprovalQueue? = nil,
        save: () throws -> Void
    ) -> ClaimOutcome {
        let queue = queue ?? .shared
        guard message.status == .pendingApproval else {
            return .rejected
        }

        let verification = ToolApprovalPayloadCodec.verifyPendingApproval(
            from: payloadArguments,
            matchingToolID: toolID,
            queue: queue
        )
        let pending: ExecutorPendingApproval
        switch verification {
        case .success(let verified):
            pending = verified
        case .failure(let error):
            let denial = persistDenied(
                message: message,
                result: error.userMessage,
                pendingActionID: ToolApprovalPayloadCodec.pendingActionID(from: payloadArguments),
                operation: .verificationDenied,
                queue: queue,
                save: save
            )
            return claimOutcome(for: denial)
        }

        guard policyAllowed else {
            let denial = persistDenied(
                message: message,
                result: policyDeniedResult,
                pendingActionID: pending.pendingActionID,
                operation: .policyDenied,
                queue: queue,
                save: save
            )
            return claimOutcome(for: denial)
        }

        let claim = persist(
            message: message,
            status: .running,
            result: nil,
            operation: .runningClaim,
            consequence: .actionNotRun,
            save: save
        )
        guard case .persisted = claim else {
            queue.clear(pending.pendingActionID)
            guard case .failed(let failure) = claim else { return .rejected }
            persistFailureMarker(message: message, failure: failure, save: save)
            return .blocked(failure)
        }

        guard let consumed = queue.consume(pending.pendingActionID, matchingToolID: toolID) else {
            let failure = makeFailure(
                operation: .runningClaim,
                errorCode: "approval_claim_lost",
                consequence: .actionNotRun
            )
            persistFailureMarker(message: message, failure: failure, save: save)
            return .blocked(failure)
        }
        return .claimed(consumed)
    }

    static func executeIfClaimed<Result>(
        _ claim: ClaimOutcome,
        execute: (ExecutorPendingApproval) async -> Result
    ) async -> Result? {
        guard case .claimed(let pending) = claim else { return nil }
        return await execute(pending)
    }

    @discardableResult
    static func persistDenied(
        message: ChatMessage,
        result: String,
        pendingActionID: UUID?,
        operation: ToolApprovalPersistenceOperation = .userDenied,
        queue: ToolApprovalQueue? = nil,
        context: ModelContext
    ) -> PersistenceOutcome {
        persistDenied(
            message: message,
            result: result,
            pendingActionID: pendingActionID,
            operation: operation,
            queue: queue,
            save: { try context.save() }
        )
    }

    @discardableResult
    static func persistDenied(
        message: ChatMessage,
        result: String,
        pendingActionID: UUID?,
        operation: ToolApprovalPersistenceOperation = .userDenied,
        queue: ToolApprovalQueue? = nil,
        save: () throws -> Void
    ) -> PersistenceOutcome {
        let queue = queue ?? .shared
        let outcome = persist(
            message: message,
            status: .denied,
            result: result,
            operation: operation,
            consequence: .actionBlocked,
            save: save
        )
        if let pendingActionID {
            queue.clear(pendingActionID)
        }
        if case .failed(let failure) = outcome {
            persistFailureMarker(message: message, failure: failure, save: save)
        }
        return outcome
    }

    @discardableResult
    static func persistTerminal(
        message: ChatMessage,
        status: ToolStatus,
        result: String,
        context: ModelContext
    ) -> PersistenceOutcome {
        persistTerminal(
            message: message,
            status: status,
            result: result,
            save: { try context.save() }
        )
    }

    @discardableResult
    static func persistTerminal(
        message: ChatMessage,
        status: ToolStatus,
        result: String,
        save: () throws -> Void
    ) -> PersistenceOutcome {
        let safeTerminalStatus: ToolStatus
        switch status {
        case .completed, .denied, .failed:
            safeTerminalStatus = status
        case .pendingApproval, .running:
            safeTerminalStatus = .failed
        }

        let outcome = persist(
            message: message,
            status: safeTerminalStatus,
            result: result,
            operation: .terminalOutcome,
            consequence: .terminalStateUncertain,
            save: save
        )
        if case .failed(let failure) = outcome {
            persistFailureMarker(message: message, failure: failure, save: save)
        }
        return outcome
    }

    private static func claimOutcome(for outcome: PersistenceOutcome) -> ClaimOutcome {
        switch outcome {
        case .persisted:
            return .rejected
        case .failed(let failure):
            return .blocked(failure)
        }
    }

    private static func persist(
        message: ChatMessage,
        status: ToolStatus,
        result: String?,
        operation: ToolApprovalPersistenceOperation,
        consequence: ToolApprovalPersistenceConsequence,
        save: () throws -> Void
    ) -> PersistenceOutcome {
        message.toolStatus = status.rawValue
        message.toolResult = result
        do {
            try save()
            return .persisted
        } catch {
            return .failed(makeFailure(error, operation: operation, consequence: consequence))
        }
    }

    private static func persistFailureMarker(
        message: ChatMessage,
        failure: ToolApprovalPersistenceFailure,
        save: () throws -> Void
    ) {
        message.toolStatus = ToolStatus.failed.rawValue
        message.toolResult = failure.userMessage
        do {
            try save()
        } catch {
            _ = makeFailure(error, operation: .failureMarker, consequence: failure.consequence)
        }
    }

    private static func makeFailure(
        _ error: Error,
        operation: ToolApprovalPersistenceOperation,
        consequence: ToolApprovalPersistenceConsequence
    ) -> ToolApprovalPersistenceFailure {
        makeFailure(
            operation: operation,
            errorCode: RuntimeMetricErrorSanitizer.code(for: error),
            consequence: consequence
        )
    }

    private static func makeFailure(
        operation: ToolApprovalPersistenceOperation,
        errorCode: String,
        consequence: ToolApprovalPersistenceConsequence
    ) -> ToolApprovalPersistenceFailure {
        let safeErrorCode = PersistentRuntimeDiagnosticsRedactor.safeCode(errorCode)
        let failure = ToolApprovalPersistenceFailure(
            operation: operation,
            errorCode: safeErrorCode,
            consequence: consequence
        )
        logger.error(
            "save_failed operation=\(operation.rawValue, privacy: .public) error_code=\(safeErrorCode, privacy: .public) consequence=\(consequence.rawValue, privacy: .public)"
        )
        PersistentRuntimeDiagnosticsObserver.shared.emit(.init(kind: .fallbackUsed, values: [
            "source": "tool-approval-persistence",
            "operation": operation.rawValue,
            "fallbackbehavior": "retain-non-replayable-state",
            "reason": "swiftdata-save-failed",
            "consequence": consequence.rawValue,
            "errorcode": safeErrorCode
        ]))
        return failure
    }
}
