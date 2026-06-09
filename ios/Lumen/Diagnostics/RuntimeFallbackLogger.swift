import Foundation
import CryptoKit

nonisolated enum RuntimeFallbackLogger {
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
        PersistentRuntimeDiagnosticsObserver.shared.emit(.init(kind: .fallbackUsed, values: payload))
    }

    static func promptHash(_ text: String) -> String {
        SHA256.hash(data: Data(text.utf8)).map { String(format: "%02x", $0) }.joined()
    }
}
