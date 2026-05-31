import Foundation
import OSLog

nonisolated enum AppCancellationCategory: String, CaseIterable, Sendable {
    case chatGeneration
    case voiceRecognition
    case tts
    case modelLoad
    case diagnostics
    case memoryMaintenance
    case ragIndexing
    case triggerScan
    case persistenceFlush
    case backgroundWork

    var isSceneSensitive: Bool {
        switch self {
        case .chatGeneration, .voiceRecognition, .tts, .modelLoad, .diagnostics,
             .memoryMaintenance, .ragIndexing, .triggerScan, .persistenceFlush, .backgroundWork:
            return true
        }
    }
}

nonisolated final class AppCancellationBus: @unchecked Sendable {
    static let shared = AppCancellationBus()

    private let lock = NSLock()
    private var tasks: [AppCancellationCategory: [UUID: @Sendable () -> Void]] = [:]
    private var cancellationRequestedReason: String?
    private let logger = Logger(subsystem: "ai.lumen.app", category: "cancellation")

    private init() {}

    @discardableResult
    func register(_ task: Task<Void, Never>, category: AppCancellationCategory) -> UUID {
        let id = registerCancellation({ task.cancel() }, category: category)
        Task.detached { [weak self] in
            await task.value
            self?.unregister(id, category: category)
        }
        return id
    }

    @discardableResult
    func registerThrowing(_ task: Task<Void, Error>, category: AppCancellationCategory) -> UUID {
        let id = registerCancellation({ task.cancel() }, category: category)
        Task.detached { [weak self] in
            _ = try? await task.value
            self?.unregister(id, category: category)
        }
        return id
    }

    func cancel(_ category: AppCancellationCategory) {
        let cancellers: [@Sendable () -> Void]
        lock.lock()
        let categoryTasks = tasks[category] ?? [:]
        cancellers = Array(categoryTasks.values)
        tasks[category] = nil
        lock.unlock()
        cancellers.forEach { $0() }
    }

    func cancelAllSceneSensitive() {
        let cancellers: [@Sendable () -> Void]
        lock.lock()
        cancellers = AppCancellationCategory.allCases
            .filter(\.isSceneSensitive)
            .flatMap { category in Array((tasks[category] ?? [:]).values) }
        for category in AppCancellationCategory.allCases where category.isSceneSensitive {
            tasks[category] = nil
        }
        lock.unlock()
        cancellers.forEach { $0() }
    }

    func markCancellationRequested(_ reason: String) {
        lock.lock()
        cancellationRequestedReason = reason
        lock.unlock()
        logger.info("cancellation_requested reason=\(reason, privacy: .public)")
    }

    var lastCancellationReason: String? {
        lock.lock()
        defer { lock.unlock() }
        return cancellationRequestedReason
    }

    @discardableResult
    func registerCancellation(_ cancel: @escaping @Sendable () -> Void, category: AppCancellationCategory) -> UUID {
        let id = UUID()
        lock.lock()
        tasks[category, default: [:]][id] = cancel
        lock.unlock()
        return id
    }

    func unregister(_ id: UUID, category: AppCancellationCategory) {
        lock.lock()
        tasks[category]?[id] = nil
        if tasks[category]?.isEmpty == true { tasks[category] = nil }
        lock.unlock()
    }
}
