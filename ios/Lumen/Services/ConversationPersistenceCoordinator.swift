import Foundation
import OSLog
import SwiftData

nonisolated struct ConversationPersistenceFailure: Equatable, Identifiable, Sendable {
    let operation: String
    let errorCode: String
    let estimatedBytes: Int

    var id: String { "\(operation):\(errorCode)" }

    var userMessage: String {
        "Your conversation is still visible, but it could not be saved. Retry before closing the app. Diagnostic: \(errorCode)."
    }
}

@MainActor
enum ConversationPersistenceCoordinator {
    enum Outcome: Equatable {
        case saved
        case deferred
        case failed(ConversationPersistenceFailure)
    }

    typealias FailureHandler = @MainActor @Sendable (ConversationPersistenceFailure) -> Void

    private static let logger = Logger(subsystem: "ai.lumen.app", category: "persistence")

    @discardableResult
    static func saveOrDefer(
        context: ModelContext,
        estimatedBytes: Int,
        operation: String,
        scope: String = "Conversation",
        deferredKey: String,
        deferredCategory: DeferredMaintenanceCategory = .conversation,
        onFailure: @escaping FailureHandler
    ) -> Outcome {
        let byteCount = max(0, estimatedBytes)
        guard !DiskWriteBudget.shared.shouldDefer(bytes: byteCount, category: .conversation) else {
            enqueueDeferredSave(
                context: context,
                estimatedBytes: byteCount,
                operation: operation,
                scope: scope,
                deferredKey: deferredKey,
                deferredCategory: deferredCategory,
                onFailure: onFailure
            )
            return .deferred
        }

        let outcome = attemptSave(
            estimatedBytes: byteCount,
            operation: operation,
            scope: scope,
            save: { try context.save() }
        )
        notifyFailure(in: outcome, using: onFailure)
        return outcome
    }

    static func enqueueDeferredSave(
        context: ModelContext,
        estimatedBytes: Int,
        operation: String,
        scope: String = "Conversation",
        deferredKey: String,
        deferredCategory: DeferredMaintenanceCategory,
        onFailure: @escaping FailureHandler
    ) {
        enqueueDeferredSave(
            estimatedBytes: estimatedBytes,
            operation: operation,
            scope: scope,
            deferredKey: deferredKey,
            deferredCategory: deferredCategory,
            budget: .shared,
            queue: .shared,
            save: { try context.save() },
            onFailure: onFailure
        )
    }

    static func enqueueDeferredSave(
        estimatedBytes: Int,
        operation: String,
        scope: String = "Conversation",
        deferredKey: String,
        deferredCategory: DeferredMaintenanceCategory,
        budget: DiskWriteBudget,
        queue: DeferredMaintenanceQueue,
        save: @escaping @MainActor @Sendable () throws -> Void,
        onFailure: @escaping FailureHandler
    ) {
        let byteCount = max(0, estimatedBytes)
        queue.enqueue(DeferredMaintenanceJob(
            key: deferredKey,
            category: deferredCategory,
            staleAfter: 10 * 60,
            maxRuntime: 2
        ) { @MainActor in
            let outcome = attemptSave(
                estimatedBytes: byteCount,
                operation: operation,
                scope: scope,
                budget: budget,
                save: save
            )
            notifyFailure(in: outcome, using: onFailure)
        })
    }

    @discardableResult
    static func attemptSave(
        estimatedBytes: Int,
        operation: String,
        scope: String = "Conversation",
        budget: DiskWriteBudget = .shared,
        save: () throws -> Void
    ) -> Outcome {
        let byteCount = max(0, estimatedBytes)
        do {
            try save()
            budget.recordWrite(bytes: byteCount, category: .conversation)
            return .saved
        } catch {
            let failure = makeFailure(
                error,
                operation: operation,
                scope: scope,
                estimatedBytes: byteCount
            )
            return .failed(failure)
        }
    }

    private static func notifyFailure(in outcome: Outcome, using handler: FailureHandler) {
        guard case .failed(let failure) = outcome else { return }
        handler(failure)
    }

    private static func makeFailure(
        _ error: Error,
        operation: String,
        scope: String,
        estimatedBytes: Int
    ) -> ConversationPersistenceFailure {
        let safeOperation = PersistentRuntimeDiagnosticsRedactor.safeCode(operation)
        let safeScope = PersistentRuntimeDiagnosticsRedactor.safeCode(scope)
        let errorCode = PersistentRuntimeDiagnosticsRedactor.safeCode(
            RuntimeMetricErrorSanitizer.code(for: error)
        )
        let failure = ConversationPersistenceFailure(
            operation: safeOperation,
            errorCode: errorCode,
            estimatedBytes: estimatedBytes
        )

        logger.error(
            "save_failed operation=\(safeOperation, privacy: .public) scope=\(safeScope, privacy: .public) error_code=\(errorCode, privacy: .public)"
        )
        PersistentRuntimeDiagnosticsObserver.shared.emit(.init(kind: .fallbackUsed, values: [
            "source": "conversation-persistence",
            "operation": safeOperation,
            "scope": safeScope,
            "primarybehavior": "save-conversation",
            "fallbackbehavior": "retain-dirty-context",
            "reason": "swiftdata-save-failed",
            "consequence": "persistence-pending-user-retry",
            "errorcode": errorCode,
            "estimatedbytes": String(estimatedBytes)
        ]))
        return failure
    }
}
