import Foundation

nonisolated enum CPUWatchdogCategory: String, CaseIterable, Sendable {
    case chatGeneration
    case voice
    case modelLoad
    case diagnostics
    case memory
    case rag
    case triggers
    case persistence
}

nonisolated struct CPUWatchdogToken: Sendable {
    fileprivate let id: UUID
    fileprivate let category: CPUWatchdogCategory
    fileprivate let startedAt: TimeInterval
}

nonisolated struct CPUWatchdogSnapshot: Equatable, Sendable {
    let totalsByCategory: [CPUWatchdogCategory: TimeInterval]
    let activeCountsByCategory: [CPUWatchdogCategory: Int]
    let degradedCategories: Set<CPUWatchdogCategory>
}

nonisolated final class CPUWatchdogGuard: @unchecked Sendable {
    static let shared = CPUWatchdogGuard()

    private struct Sample {
        let category: CPUWatchdogCategory
        let start: TimeInterval
        let end: TimeInterval
    }

    private let lock = NSLock()
    private var active: [UUID: CPUWatchdogToken] = [:]
    private var samples: [Sample] = []
    private let window: TimeInterval
    private let degradeThreshold: TimeInterval

    init(window: TimeInterval = 120, degradeThreshold: TimeInterval = 45) {
        self.window = window
        self.degradeThreshold = degradeThreshold
    }

    func begin(category: CPUWatchdogCategory) -> CPUWatchdogToken {
        let now = ProcessInfo.processInfo.systemUptime
        let token = CPUWatchdogToken(id: UUID(), category: category, startedAt: now)
        lock.lock()
        prune(now: now)
        active[token.id] = token
        lock.unlock()
        return token
    }

    func end(token: CPUWatchdogToken) {
        let now = ProcessInfo.processInfo.systemUptime
        lock.lock()
        if active.removeValue(forKey: token.id) != nil {
            samples.append(Sample(category: token.category, start: token.startedAt, end: now))
        }
        prune(now: now)
        lock.unlock()
    }

    func shouldDegrade(category: CPUWatchdogCategory) -> Bool {
        let now = ProcessInfo.processInfo.systemUptime
        lock.lock()
        prune(now: now)
        let total = rollingTotal(category: category, now: now)
        lock.unlock()
        return total >= degradeThreshold
    }

    func currentSnapshot() -> CPUWatchdogSnapshot {
        let now = ProcessInfo.processInfo.systemUptime
        lock.lock()
        prune(now: now)
        var totals: [CPUWatchdogCategory: TimeInterval] = [:]
        var activeCounts: [CPUWatchdogCategory: Int] = [:]
        for category in CPUWatchdogCategory.allCases {
            totals[category] = rollingTotal(category: category, now: now)
            activeCounts[category] = active.values.filter { $0.category == category }.count
        }
        let degraded = Set(CPUWatchdogCategory.allCases.filter { (totals[$0] ?? 0) >= degradeThreshold })
        lock.unlock()
        return CPUWatchdogSnapshot(totalsByCategory: totals, activeCountsByCategory: activeCounts, degradedCategories: degraded)
    }

    private func prune(now: TimeInterval) {
        let cutoff = now - window
        samples.removeAll { $0.end < cutoff }
    }

    private func rollingTotal(category: CPUWatchdogCategory, now: TimeInterval) -> TimeInterval {
        let cutoff = now - window
        let completed = samples.filter { $0.category == category }.reduce(0) { partial, sample in
            partial + max(0, sample.end - max(sample.start, cutoff))
        }
        let running = active.values.filter { $0.category == category }.reduce(0) { partial, token in
            partial + max(0, now - max(token.startedAt, cutoff))
        }
        return completed + running
    }
}
