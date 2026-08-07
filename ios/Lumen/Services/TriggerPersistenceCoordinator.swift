import Foundation
import OSLog
import SwiftData

nonisolated enum TriggerPersistenceOperation: String, Equatable, Sendable {
    case create
    case update
    case pause
    case resume
    case delete
    case run
    case recovery

    var userAction: String {
        switch self {
        case .create: "create this trigger"
        case .update: "save these trigger changes"
        case .pause: "pause this trigger"
        case .resume: "resume this trigger"
        case .delete: "delete this trigger"
        case .run: "save the completed trigger run"
        case .recovery: "resolve suspended trigger execution"
        }
    }
}

nonisolated struct TriggerPersistenceFailure: Error, Equatable, Identifiable, Sendable {
    let operation: TriggerPersistenceOperation
    let errorCode: String
    let autonomousExecutionSuspended: Bool

    var id: String { "\(operation.rawValue):\(errorCode)" }

    var alertTitle: String {
        switch operation {
        case .create: "Trigger not created"
        case .update: "Trigger not updated"
        case .pause: "Trigger not paused"
        case .resume: "Trigger not resumed"
        case .delete: "Trigger not deleted"
        case .run: "Trigger run not saved"
        case .recovery: "Trigger recovery not saved"
        }
    }

    var userMessage: String {
        let draftMessage = switch operation {
        case .create, .update:
            "Your draft remains open."
        default:
            "Your saved trigger was preserved."
        }
        let safetyMessage = autonomousExecutionSuspended
            ? "Automatic trigger execution is suspended to prevent an unintended run."
            : "Automatic trigger execution was not changed."
        return "Lumen could not \(operation.userAction). \(draftMessage) \(safetyMessage) Retry when storage is available. Diagnostic: \(errorCode)."
    }
}

@MainActor
enum TriggerPersistenceCoordinator {
    enum Outcome: Equatable {
        case saved
        case failed(TriggerPersistenceFailure)
    }

    private static let logger = Logger(subsystem: "ai.lumen.app", category: "persistence")

    /// Commits one user-visible trigger mutation. The caller supplies a narrow
    /// rollback that restores only the affected trigger, avoiding a broad
    /// ModelContext rollback that could discard unrelated pending user work.
    @discardableResult
    static func attempt(
        operation: TriggerPersistenceOperation,
        save: () throws -> Void,
        restore: () -> Void,
        onSaved: () -> Void,
        onFailure: () -> Bool
    ) -> Outcome {
        do {
            try save()
            onSaved()
            return .saved
        } catch {
            restore()
            let autonomousExecutionSuspended = onFailure()
            return .failed(makeFailure(
                error,
                operation: operation,
                autonomousExecutionSuspended: autonomousExecutionSuspended
            ))
        }
    }

    /// Deletes through an isolated context so a failed save can be rolled back
    /// without invalidating the view's model instance or discarding unrelated
    /// pending work in the environment context.
    @discardableResult
    static func delete(
        triggerID: UUID,
        container: ModelContainer,
        save: (ModelContext) throws -> Void = { try $0.save() },
        onSaved: () -> Void,
        onFailure: () -> Bool
    ) -> Outcome {
        let deletionContext = ModelContext(container)
        let trigger: Trigger
        do {
            var descriptor = FetchDescriptor<Trigger>(
                predicate: #Predicate { $0.id == triggerID }
            )
            descriptor.fetchLimit = 1
            guard let storedTrigger = try deletionContext.fetch(descriptor).first else {
                onSaved()
                return .saved
            }
            trigger = storedTrigger
        } catch {
            let autonomousExecutionSuspended = onFailure()
            return .failed(makeFailure(
                error,
                operation: .delete,
                autonomousExecutionSuspended: autonomousExecutionSuspended
            ))
        }

        deletionContext.delete(trigger)
        return attempt(
            operation: .delete,
            save: { try save(deletionContext) },
            restore: { deletionContext.rollback() },
            onSaved: onSaved,
            onFailure: onFailure
        )
    }

    /// Pauses every stored trigger referenced by a durable suspension token and
    /// commits all changes in one save. Missing trigger IDs are deliberately part
    /// of the successful recovery set, but callers must not clear their tokens
    /// until `onSaved` runs.
    @discardableResult
    static func resolveSuspensions(
        triggerIDs: Set<UUID>,
        container: ModelContainer,
        save: (ModelContext) throws -> Void = { try $0.save() },
        onSaved: () -> Void,
        onFailure: () -> Bool
    ) -> Outcome {
        let recoveryContext = ModelContext(container)
        let storedTriggers: [Trigger]
        do {
            storedTriggers = try recoveryContext.fetch(FetchDescriptor<Trigger>())
                .filter { triggerIDs.contains($0.id) }
        } catch {
            let autonomousExecutionSuspended = onFailure()
            return .failed(makeFailure(
                error,
                operation: .recovery,
                autonomousExecutionSuspended: autonomousExecutionSuspended
            ))
        }

        for trigger in storedTriggers {
            trigger.isPaused = true
            trigger.nextFireAt = nil
        }

        return attempt(
            operation: .recovery,
            save: { try save(recoveryContext) },
            restore: { recoveryContext.rollback() },
            onSaved: onSaved,
            onFailure: onFailure
        )
    }

    static func makeFailure(
        _ error: Error,
        operation: TriggerPersistenceOperation,
        autonomousExecutionSuspended: Bool
    ) -> TriggerPersistenceFailure {
        let errorCode = PersistentRuntimeDiagnosticsRedactor.safeCode(
            RuntimeMetricErrorSanitizer.code(for: error)
        )
        let failure = TriggerPersistenceFailure(
            operation: operation,
            errorCode: errorCode,
            autonomousExecutionSuspended: autonomousExecutionSuspended
        )
        logger.error(
            "trigger_save_failed operation=\(operation.rawValue, privacy: .public) error_code=\(errorCode, privacy: .public) autonomous_execution_suspended=\(autonomousExecutionSuspended, privacy: .public)"
        )
        return failure
    }
}
