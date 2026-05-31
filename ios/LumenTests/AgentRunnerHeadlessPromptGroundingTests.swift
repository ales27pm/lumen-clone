import XCTest
@testable import Lumen

final class AgentRunnerHeadlessPromptGroundingTests: XCTestCase {
    func testPolicyProfileStricterHeadless() {
        XCTAssertLessThan(LegacyPromptInjectionPolicy.headlessTrigger.memoryMax, LegacyPromptInjectionPolicy.foregroundChat.memoryMax)
        XCTAssertTrue(LegacyPromptInjectionPolicy.headlessTrigger.backgroundSafeToolsOnly)
    }
}
