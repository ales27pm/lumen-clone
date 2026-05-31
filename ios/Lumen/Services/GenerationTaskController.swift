import Foundation

@MainActor
final class GenerationTaskController<Key: Hashable> {
    private var tasks: [Key: Task<Void, Never>] = [:]
    private var activeRequestIDs: [Key: UUID] = [:]
    private var startedKeys: Set<Key> = []

    func startupIfNeeded(for key: Key, _ action: @escaping () -> Void) {
        guard !startedKeys.contains(key) else { return }
        startedKeys.insert(key)
        action()
    }

    func begin(for key: Key, task: Task<Void, Never>, requestID: UUID = UUID()) -> UUID {
        tasks[key]?.cancel()
        activeRequestIDs[key] = requestID
        tasks[key] = task
        assertSingleActiveGeneration(for: key)
        return requestID
    }

    func cancel(for key: Key) {
        tasks[key]?.cancel()
        tasks[key] = nil
        activeRequestIDs[key] = nil
    }

    func isCurrent(_ requestID: UUID, for key: Key) -> Bool {
        activeRequestIDs[key] == requestID
    }

    func clearIfCurrent(_ requestID: UUID, for key: Key) {
        guard activeRequestIDs[key] == requestID else { return }
        tasks[key] = nil
        activeRequestIDs[key] = nil
    }

    func hasActiveGeneration(for key: Key) -> Bool {
        tasks[key] != nil && activeRequestIDs[key] != nil
    }

    /// Audit assertion: every conversation key must map to at most one active
    /// generation task and request ID at any time.
    func assertSingleActiveGeneration(for key: Key) {
        let taskExists = tasks[key] != nil
        let requestExists = activeRequestIDs[key] != nil
        assert(taskExists == requestExists, "GenerationTaskController invariant violation: task/request mismatch for key \(String(describing: key)).")
    }
}
