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

nonisolated struct DiskWriteBudgetReservation: Equatable, Hashable, Sendable {
    fileprivate let id: UUID
    fileprivate let bytes: Int64
    fileprivate let category: DiskWriteCategory
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
    private var activeReservations: [UUID: DiskWriteBudgetReservation] = [:]
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

    func reserveWrite(bytes: Int, category: DiskWriteCategory) -> DiskWriteBudgetReservation? {
        let requested = Int64(max(0, bytes))
        let now = ProcessInfo.processInfo.systemUptime
        lock.lock()
        defer { lock.unlock() }
        guard !isGenerationBlocked(category: category) else { return nil }
        prune(now: now)
        guard !wouldExceedLimits(requested: requested, now: now) else { return nil }

        let reservation = DiskWriteBudgetReservation(
            id: UUID(),
            bytes: requested,
            category: category
        )
        activeReservations[reservation.id] = reservation
        return reservation
    }

    @discardableResult
    func commitReservedWrite(_ reservation: DiskWriteBudgetReservation) -> Bool {
        let now = ProcessInfo.processInfo.systemUptime
        lock.lock()
        defer { lock.unlock() }
        guard activeReservations.removeValue(forKey: reservation.id) == reservation else {
            return false
        }
        prune(now: now)
        if reservation.bytes > 0 {
            events.append(Event(at: now, bytes: reservation.bytes, category: reservation.category))
        }
        return true
    }

    @discardableResult
    func releaseReservedWrite(_ reservation: DiskWriteBudgetReservation) -> Bool {
        lock.lock()
        defer { lock.unlock() }
        return activeReservations.removeValue(forKey: reservation.id) == reservation
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
        let requested = Int64(max(0, bytes))
        let now = ProcessInfo.processInfo.systemUptime
        lock.lock()
        defer { lock.unlock() }
        guard !isGenerationBlocked(category: category) else { return true }
        prune(now: now)
        return wouldExceedLimits(requested: requested, now: now)
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
            byCategory[category] = saturatedSum(events.lazy.filter { $0.category == category }.map(\.bytes))
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
        saturatedSum(events.lazy.filter { $0.at >= cutoff }.map(\.bytes))
    }

    private func isGenerationBlocked(category: DiskWriteCategory) -> Bool {
        let generationBlockedCategories: Set<DiskWriteCategory> = [.diagnostics, .logs, .memory, .rag, .triggers]
        return (generationActive || activeGenerationLeaseCount > 0)
            && generationBlockedCategories.contains(category)
    }

    private func wouldExceedLimits(requested: Int64, now: TimeInterval) -> Bool {
        guard let reserved = checkedSum(activeReservations.values.lazy.map(\.bytes)),
              let one = checkedSum([total(since: now - 60), reserved, requested]),
              let fifteen = checkedSum([total(since: now - 15 * 60), reserved, requested]),
              let day = checkedSum([total(since: now - 24 * 60 * 60), reserved, requested]) else {
            return true
        }
        return one > oneMinuteLimit || fifteen > fifteenMinuteLimit || day > dayLimit
    }

    private func saturatedSum<S: Sequence>(_ values: S) -> Int64 where S.Element == Int64 {
        checkedSum(values) ?? .max
    }

    private func checkedSum<S: Sequence>(_ values: S) -> Int64? where S.Element == Int64 {
        var sum: Int64 = 0
        for value in values {
            let result = sum.addingReportingOverflow(value)
            guard !result.overflow else { return nil }
            sum = result.partialValue
        }
        return sum
    }
}
