import Foundation
import Testing
@testable import Lumen

@Suite(.serialized)
struct RuntimeFallbackLoggerTests {
    @Test func runtimeFallbackLoggerSuppressesImmediateDuplicateSignals() async throws {
        RuntimeFallbackLogger.resetForTesting()
        let source = "test-source-\(UUID().uuidString)"
        let reason = "test-reason-\(UUID().uuidString)"

        let firstEmission = RuntimeFallbackLogger.record(
            source: source,
            primaryBehavior: "primary",
            fallbackBehavior: "fallback",
            reason: reason,
            consequence: "consequence",
            values: ["rawChars": "0", "removedArtifacts": "emptyAfterSanitization"]
        )
        let duplicateEmission = RuntimeFallbackLogger.record(
            source: source,
            primaryBehavior: "primary",
            fallbackBehavior: "fallback",
            reason: reason,
            consequence: "consequence",
            values: ["rawChars": "0", "removedArtifacts": "emptyAfterSanitization"]
        )

        #expect(firstEmission)
        #expect(!duplicateEmission)
    }

    @Test func runtimeFallbackLoggerAllowsDistinctSignals() async throws {
        RuntimeFallbackLogger.resetForTesting()
        let source = "test-source-\(UUID().uuidString)"
        let reason = "test-reason-\(UUID().uuidString)"

        let firstEmission = RuntimeFallbackLogger.record(
            source: source,
            primaryBehavior: "primary",
            fallbackBehavior: "fallback",
            reason: reason,
            consequence: "consequence",
            values: ["rawChars": "0", "removedArtifacts": "emptyAfterSanitization"]
        )
        let distinctEmission = RuntimeFallbackLogger.record(
            source: source,
            primaryBehavior: "primary",
            fallbackBehavior: "fallback",
            reason: reason,
            consequence: "consequence",
            values: ["rawChars": "1", "removedArtifacts": "emptyAfterSanitization"]
        )

        #expect(firstEmission)
        #expect(distinctEmission)
    }
}
