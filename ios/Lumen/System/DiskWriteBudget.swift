import Foundation

nonisolated enum DiskWriteCategory: String, CaseIterable, Sendable {
    case conversation
    case memory
    case rag
    case diagnostics
    case logs
    case modelMetadata
    case triggers
}

nonisolated struct DiskWriteBudgetSnapshot: Equatable, Sendable {
    let bytes1Minute: Int64
    let bytes15Minutes: Int64
    let bytes24Hours: Int64
    let bytesByCategory24Hours: [DiskWriteCategory: Int64]
}

nonisolated final class DiskWriteGenerationLease: @unchecked Sendable {
    private let lock = NSLock()
    private var ended = false

    fileprivate init() {}

    func end() {
        lock.lock()
        let shouldEnd = !ended
        ended = true
        lock.unlock()
        if shouldEnd { DiskWriteBudget.shared.endGenerationLease() }
    }
}

nonisolated final class DiskWriteBudget: @unchecked Sendable {
    static let shared = DiskWriteBudget()

    private struct Event {
        let at: TimeInterval
        let bytes: Int64
        let category: DiskWriteCategory
    }

    private let lock = NSLock()
    private var events: [Event] = []
    private let oneMinuteLimit: Int64
    private let fifteenMinuteLimit: Int64
    private let dayLimit: Int64
    private var generationActive = false
    private var activeGenerationLeaseCount = 0

    init(oneMinuteLimit: Int64 = 1_500_000, fifteenMinuteLimit: Int64 = 18_000_000, dayLimit: Int64 = 450_000_000) {
        self.oneMinuteLimit = oneMinuteLimit
        self.fifteenMinuteLimit = fifteenMinuteLimit
        self.dayLimit = dayLimit
    }

    func canWrite(bytes: Int, category: DiskWriteCategory) -> Bool {
        !shouldDefer(bytes: bytes, category: category)
    }

    func setGenerationActive(_ active: Bool) {
        lock.lock()
        generationActive = active
        if !active { activeGenerationLeaseCount = 0 }
        lock.unlock()
    }

    func beginGeneration() -> DiskWriteGenerationLease {
        lock.lock()
        activeGenerationLeaseCount += 1
        generationActive = true
        lock.unlock()
        return DiskWriteGenerationLease()
    }

    fileprivate func endGenerationLease() {
        lock.lock()
        activeGenerationLeaseCount = max(0, activeGenerationLeaseCount - 1)
        if activeGenerationLeaseCount == 0 { generationActive = false }
        lock.unlock()
    }

    func isGenerationActive() -> Bool {
        lock.lock()
        defer { lock.unlock() }
        return generationActive || activeGenerationLeaseCount > 0
    }

    func shouldDefer(bytes: Int, category: DiskWriteCategory) -> Bool {
        lock.lock()
        let active = generationActive || activeGenerationLeaseCount > 0
        lock.unlock()
        let generationBlockedCategories: Set<DiskWriteCategory> = [.diagnostics, .logs, .memory, .rag, .triggers]
        if active, generationBlockedCategories.contains(category) {
            return true
        }

        let requested = Int64(max(0, bytes))
        let now = ProcessInfo.processInfo.systemUptime
        lock.lock()
        prune(now: now)
        let one = total(since: now - 60) + requested
        let fifteen = total(since: now - 15 * 60) + requested
        let day = total(since: now - 24 * 60 * 60) + requested
        lock.unlock()
        return one > oneMinuteLimit || fifteen > fifteenMinuteLimit || day > dayLimit
    }

    func recordWrite(bytes: Int, category: DiskWriteCategory) {
        let count = Int64(max(0, bytes))
        guard count > 0 else { return }
        let now = ProcessInfo.processInfo.systemUptime
        lock.lock()
        prune(now: now)
        events.append(Event(at: now, bytes: count, category: category))
        lock.unlock()
    }

    func snapshot() -> DiskWriteBudgetSnapshot {
        let now = ProcessInfo.processInfo.systemUptime
        lock.lock()
        prune(now: now)
        var byCategory: [DiskWriteCategory: Int64] = [:]
        for category in DiskWriteCategory.allCases {
            byCategory[category] = events.filter { $0.category == category }.reduce(0) { $0 + $1.bytes }
        }
        let snap = DiskWriteBudgetSnapshot(
            bytes1Minute: total(since: now - 60),
            bytes15Minutes: total(since: now - 15 * 60),
            bytes24Hours: total(since: now - 24 * 60 * 60),
            bytesByCategory24Hours: byCategory
        )
        lock.unlock()
        return snap
    }

    private func prune(now: TimeInterval) {
        events.removeAll { $0.at < now - 24 * 60 * 60 }
    }

    private func total(since cutoff: TimeInterval) -> Int64 {
        events.filter { $0.at >= cutoff }.reduce(0) { $0 + $1.bytes }
    }
}
