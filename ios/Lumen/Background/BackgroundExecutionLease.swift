import Foundation

actor BackgroundExecutionLease {
    struct Lease: Sendable, Equatable {
        let category: String
        let reason: String
        let acquiredAt: Date
        let expiresAt: Date
    }

    private var leases: [String: Lease] = [:]

    func acquire(category: String, reason: String, ttl: TimeInterval = 300, now: Date = Date()) -> Bool {
        cleanupExpired(now: now)
        guard leases[category] == nil else { return false }
        leases[category] = Lease(category: category, reason: reason, acquiredAt: now, expiresAt: now.addingTimeInterval(ttl))
        return true
    }

    func release(category: String) {
        leases.removeValue(forKey: category)
    }

    func activeLease(category: String, now: Date = Date()) -> Lease? {
        cleanupExpired(now: now)
        return leases[category]
    }

    private func cleanupExpired(now: Date) {
        leases = leases.filter { $0.value.expiresAt > now }
    }
}
