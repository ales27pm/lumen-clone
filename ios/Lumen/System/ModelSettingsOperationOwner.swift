import Foundation
import Observation

/// Owns a settings mutation until its service has drained its workers. Views may
/// detach immediately, but cancellation never admits overlapping replacement work
/// or delivers an old result to a later appearance of the same settings section.
@MainActor
@Observable
final class ModelSettingsOperationOwner {
    private(set) var isBusy = false
    @ObservationIgnored private let cancellationBus: AppCancellationBus
    @ObservationIgnored private var task: Task<Void, Never>?
    @ObservationIgnored private var registration: UUID?
    @ObservationIgnored private var operationID: UUID?
    @ObservationIgnored private var appearanceID: UUID?

    init(cancellationBus: AppCancellationBus = .shared) {
        self.cancellationBus = cancellationBus
    }

    var isActive: Bool { appearanceID != nil }

    func activate() {
        if appearanceID == nil { appearanceID = UUID() }
    }

    func deactivate() {
        appearanceID = nil
        cancel()
    }

    func cancel() {
        task?.cancel()
    }

    @discardableResult
    func start<Value: Sendable>(
        operation: @escaping @MainActor () async throws -> Value,
        completion: @escaping @MainActor (Result<Value, Error>) -> Void
    ) -> Task<Void, Never>? {
        guard let appearanceID, !isBusy else { return nil }
        let id = UUID()
        operationID = id
        isBusy = true
        let work = Task { @MainActor [self] in
            let result: Result<Value, Error>
            do {
                try Task.checkCancellation()
                let value = try await operation()
                // A service may finish its atomic commit while cancellation is
                // requested. Never turn that late return into UI success.
                try Task.checkCancellation()
                result = .success(value)
            } catch {
                result = .failure(Task.isCancelled ? CancellationError() : error)
            }
            guard operationID == id else { return }
            if let registration {
                cancellationBus.unregister(registration, category: .modelMaintenance)
            }
            registration = nil
            task = nil
            operationID = nil
            isBusy = false
            if self.appearanceID == appearanceID { completion(result) }
        }
        task = work
        registration = cancellationBus.registerCancellation({ work.cancel() }, category: .modelMaintenance)
        return work
    }
}
