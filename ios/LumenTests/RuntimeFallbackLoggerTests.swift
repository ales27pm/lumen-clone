import Foundation
import Testing
@testable import Lumen

@Suite(.serialized)
struct RuntimeFallbackLoggerTests {
    @Test func runtimeFallbackLoggerSuppressesImmediateDuplicateSignals() async throws {
        let source = "test-source-\(UUID().uuidString)"
        let reason = "test-reason-\(UUID().uuidString)"
        let received = RuntimeFallbackSignalCollector(source: source, reason: reason)
        let observerID = PersistentRuntimeDiagnosticsObserver.shared.addObserver { signal in
            received.recordIfMatching(signal)
        }
        defer { PersistentRuntimeDiagnosticsObserver.shared.removeObserver(observerID) }

        RuntimeFallbackLogger.record(
            source: source,
            primaryBehavior: "primary",
            fallbackBehavior: "fallback",
            reason: reason,
            consequence: "consequence",
            values: ["rawChars": "0", "removedArtifacts": "emptyAfterSanitization"]
        )
        RuntimeFallbackLogger.record(
            source: source,
            primaryBehavior: "primary",
            fallbackBehavior: "fallback",
            reason: reason,
            consequence: "consequence",
            values: ["rawChars": "0", "removedArtifacts": "emptyAfterSanitization"]
        )

        #expect(received.count == 1)
    }

    @Test func runtimeFallbackLoggerAllowsDistinctSignals() async throws {
        let source = "test-source-\(UUID().uuidString)"
        let reason = "test-reason-\(UUID().uuidString)"
        let received = RuntimeFallbackSignalCollector(source: source, reason: reason)
        let observerID = PersistentRuntimeDiagnosticsObserver.shared.addObserver { signal in
            received.recordIfMatching(signal)
        }
        defer { PersistentRuntimeDiagnosticsObserver.shared.removeObserver(observerID) }

        RuntimeFallbackLogger.record(
            source: source,
            primaryBehavior: "primary",
            fallbackBehavior: "fallback",
            reason: reason,
            consequence: "consequence",
            values: ["rawChars": "0", "removedArtifacts": "emptyAfterSanitization"]
        )
        RuntimeFallbackLogger.record(
            source: source,
            primaryBehavior: "primary",
            fallbackBehavior: "fallback",
            reason: reason,
            consequence: "consequence",
            values: ["rawChars": "1", "removedArtifacts": "emptyAfterSanitization"]
        )

        #expect(received.count == 2)
    }
}

private final class RuntimeFallbackSignalCollector: @unchecked Sendable {
    private let source: String
    private let reason: String
    private let lock = NSLock()
    private var received: [PersistentRuntimeDiagnosticSignal] = []

    init(source: String, reason: String) {
        self.source = source
        self.reason = reason
    }

    func recordIfMatching(_ signal: PersistentRuntimeDiagnosticSignal) {
        guard signal.kind == .fallbackUsed,
              signal.values["source"] == source,
              signal.values["reason"] == reason else { return }
        lock.lock()
        received.append(signal)
        lock.unlock()
    }

    var count: Int {
        lock.lock()
        defer { lock.unlock() }
        return received.count
    }
}
