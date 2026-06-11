import Foundation
import Testing
@testable import Lumen

@Suite(.serialized)
struct RuntimeFallbackLoggerTests {
    @Test func runtimeFallbackLoggerSuppressesImmediateDuplicateSignals() async throws {
        let source = "test-source-\(UUID().uuidString)"
        let reason = "test-reason-\(UUID().uuidString)"
        var received: [PersistentRuntimeDiagnosticSignal] = []
        let observerID = PersistentRuntimeDiagnosticsObserver.shared.addObserver { signal in
            if signal.kind == .fallbackUsed,
               signal.values["source"] == source,
               signal.values["reason"] == reason {
                received.append(signal)
            }
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
        var received: [PersistentRuntimeDiagnosticSignal] = []
        let observerID = PersistentRuntimeDiagnosticsObserver.shared.addObserver { signal in
            if signal.kind == .fallbackUsed,
               signal.values["source"] == source,
               signal.values["reason"] == reason {
                received.append(signal)
            }
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
