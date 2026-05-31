import Foundation
import CryptoKit

actor LegacyGroundingCache {
    struct Key: Hashable {
        let conversationID: UUID?
        let turnID: UUID?
        let userDigest: String
        let background: Bool
        let lowPowerMode: Bool
        let thermalState: DeviceThermalState
    }
    private struct Entry { let bundle: LegacyGroundingBundle; let expiresAt: Date }
    private var store: [Key: Entry] = [:]
    private let ttl: TimeInterval
    init(ttl: TimeInterval = 120) { self.ttl = ttl }

    static func digest(_ text: String) -> String {
        SHA256.hash(data: Data(text.utf8)).map { String(format: "%02x", $0) }.joined()
    }

    func get(_ key: Key, now: Date = Date()) -> LegacyGroundingBundle? {
        pruneExpired(now: now)
        guard let e = store[key], e.expiresAt > now else { store.removeValue(forKey: key); return nil }
        return e.bundle
    }
    func put(_ key: Key, bundle: LegacyGroundingBundle, now: Date = Date()) {
        pruneExpired(now: now)
        store[key] = Entry(bundle: bundle, expiresAt: now.addingTimeInterval(ttl))
    }
    func invalidate(conversationID: UUID?) { store = store.filter { $0.key.conversationID != conversationID } }

    private func pruneExpired(now: Date) {
        store = store.filter { $0.value.expiresAt > now }
    }
}
