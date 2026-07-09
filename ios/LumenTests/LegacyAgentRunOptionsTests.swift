import XCTest
@testable import Lumen

final class LegacyAgentRunOptionsTests: XCTestCase {
    func testDefaults() {
        let o = LegacyAgentRunOptions.default
        XCTAssertNil(o.modelContext)
        XCTAssertNil(o.conversationID)
        XCTAssertNil(o.turnID)
        XCTAssertEqual(o.groundingMode, .foregroundChat)
        XCTAssertTrue(o.allowDegradedGrounding)
        XCTAssertTrue(o.preventDoubleGrounding)
        XCTAssertFalse(o.diagnosticsEnabled)
    }

    func testDeterministicCompatibilityExecutionIsDebugOnly() {
        let o = LegacyAgentRunOptions.default
        #if DEBUG
        XCTAssertTrue(o.allowsDeterministicCompatibilityExecution)
        XCTAssertTrue(o.allowsParseFailureDeterministicRecoveryExecution)
        #else
        XCTAssertFalse(o.allowsDeterministicCompatibilityExecution)
        XCTAssertFalse(o.allowsParseFailureDeterministicRecoveryExecution)
        #endif
    }
}
