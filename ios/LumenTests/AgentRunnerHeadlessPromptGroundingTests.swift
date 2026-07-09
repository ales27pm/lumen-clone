import XCTest
@testable import Lumen

final class AgentRunnerHeadlessPromptGroundingTests: XCTestCase {
    func testPolicyProfileStricterHeadless() {
        XCTAssertLessThan(LegacyPromptInjectionPolicy.headlessTrigger.memoryMax, LegacyPromptInjectionPolicy.foregroundChat.memoryMax)
        XCTAssertTrue(LegacyPromptInjectionPolicy.headlessTrigger.backgroundSafeToolsOnly)
    }

    @MainActor
    func testStoredModelFetchFailureDoesNotRenderAsEmptyFleet() {
        let error = NSError(domain: "SwiftData", code: 7, userInfo: [NSLocalizedDescriptionKey: "raw database path"])
        let rendered = HeadlessAgentKernelRunner.storedModelFetchFailureMessage(error: error)

        XCTAssertTrue(rendered.contains("model catalog fetch failed"))
        XCTAssertFalse(rendered.contains("local model not loaded"))
        XCTAssertFalse(rendered.contains("raw database path"))
    }
}
