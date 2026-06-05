import Foundation

actor AgentRunCoordinator {
    static let shared = AgentRunCoordinator()

    private var activeTask: Task<PersistentDiagnosticRunRecord, Never>?
    private let store: PersistentRuntimeDiagnosticsStore

    init(store: PersistentRuntimeDiagnosticsStore = .shared) {
        self.store = store
    }

    func cancelActive(reason: String) async {
        AppCancellationBus.shared.markCancellationRequested(reason)
        AppCancellationBus.shared.cancel(.chatGeneration)
        activeTask?.cancel()
        activeTask = nil
    }

    func run(
        record: PersistentDiagnosticRunRecord,
        cancellationReason: String,
        operation: @escaping @Sendable (PersistentDiagnosticRunRecord) async throws -> PersistentDiagnosticRunRecord
    ) async -> PersistentDiagnosticRunRecord {
        await cancelActive(reason: "diagnostics-replaced-by-new-agent-run")
        var startingRecord = record
        startingRecord.status = .running
        let task = Task { () -> PersistentDiagnosticRunRecord in
            await withTaskCancellationHandler {
                do {
                    try Task.checkCancellation()
                    var completed = try await operation(startingRecord)
                    try Task.checkCancellation()
                    if completed.status == .running {
                        completed.status = .passed
                        completed.finishedAt = Date()
                        completed.events.append(PersistentDiagnosticEvent(code: "agent_run_passed", message: "Agent run completed"))
                    }
                    await store.appendRunUpdate(completed)
                    return completed
                } catch is CancellationError {
                    var cancelled = startingRecord
                    cancelled.status = .cancelled
                    cancelled.finishedAt = Date()
                    cancelled.metrics.didCancel = true
                    cancelled.metrics.cancellationReason = cancellationReason
                    cancelled.events.append(PersistentDiagnosticEvent(code: "agent_run_cancelled", message: "Agent run cancelled", values: ["reason": cancellationReason]))
                    await store.appendRunUpdate(cancelled)
                    return cancelled
                } catch {
                    var failed = startingRecord
                    failed.status = .failed
                    failed.finishedAt = Date()
                    let code = RuntimeMetricErrorSanitizer.code(for: error)
                    failed.metrics.errorCodes.append(code)
                    failed.failureSummary = code
                    failed.events.append(PersistentDiagnosticEvent(code: "agent_run_failed", message: "Agent run failed", values: ["errorCode": code]))
                    await store.appendRunUpdate(failed)
                    return failed
                }
            } onCancel: {
                AppCancellationBus.shared.markCancellationRequested(cancellationReason)
                AppCancellationBus.shared.cancel(.chatGeneration)
            }
        }
        activeTask = task
        let result = await task.value
        activeTask = nil
        return result
    }
}
