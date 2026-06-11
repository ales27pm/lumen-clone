import Foundation
import CryptoKit

private final class RuntimeFallbackEmissionGate: @unchecked Sendable {
    private let lock = NSLock()
    private let duplicateWindowSeconds: TimeInterval
    private var lastSignature: String?
    private var lastEmissionDate: Date?

    init(duplicateWindowSeconds: TimeInterval = 2) {
        self.duplicateWindowSeconds = duplicateWindowSeconds
    }

    func shouldEmit(signature: String, at date: Date = Date()) -> Bool {
        lock.lock()
        defer { lock.unlock() }

        if signature == lastSignature,
           let lastEmissionDate,
           date.timeIntervalSince(lastEmissionDate) <= duplicateWindowSeconds {
            return false
        }

        lastSignature = signature
        lastEmissionDate = date
        return true
    }
}

nonisolated enum RuntimeFallbackLogger {
    private static let emissionGate = RuntimeFallbackEmissionGate()

    static func record(
        source: String,
        primaryBehavior: String,
        fallbackBehavior: String,
        reason: String,
        consequence: String,
        values: [String: String] = [:]
    ) {
        var payload = LumenTrainedModelRuntimeRegistry.selected.traceValues
        payload["schemaVersion"] = "lumen.runtime_fallback/1.0.0"
        payload["source"] = source
        payload["primaryBehavior"] = primaryBehavior
        payload["fallbackBehavior"] = fallbackBehavior
        payload["reason"] = reason
        payload["consequence"] = consequence
        values.forEach { key, value in
            payload[key] = value
        }

        let signature = fallbackSignature(source: source, fallbackBehavior: fallbackBehavior, reason: reason, values: payload)
        guard emissionGate.shouldEmit(signature: signature) else { return }

        PersistentRuntimeDiagnosticsObserver.shared.emit(.init(kind: .fallbackUsed, values: payload))
    }

    static func promptHash(_ text: String) -> String {
        SHA256.hash(data: Data(text.utf8)).map { String(format: "%02x", $0) }.joined()
    }

    private static func fallbackSignature(
        source: String,
        fallbackBehavior: String,
        reason: String,
        values: [String: String]
    ) -> String {
        let stableValues = values
            .filter { key, _ in key.lowercased() != "createdat" }
            .sorted { $0.key < $1.key }
            .map { "\($0.key)=\($0.value)" }
            .joined(separator: "|")
        return [source, fallbackBehavior, reason, stableValues].joined(separator: "|")
    }
}
