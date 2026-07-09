import XCTest
@testable import Lumen

final class HeadlessGroundingPolicyTests: XCTestCase {
    @MainActor func testBackgroundToolsFiltered() async throws {
        let routing = IntentRoutingDecision(
            intent: .webSearch,
            allowedToolIDs: ["open.url"],
            requiresClarification: false,
            clarificationPrompt: nil
        )

        let assessment = await BackgroundToolExecutionPolicy.assess(
            prompt: "open https://example.com",
            routing: routing,
            modelContext: nil
        )

        XCTAssertEqual(assessment.status, .noBackgroundSafeRoutedTools)
        XCTAssertFalse(assessment.availableTools.contains { $0.id == "open.url" })
    }
}
