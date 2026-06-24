import XCTest
@testable import Lumen

final class LumenIntentPolicyTests: XCTestCase {
    func testSensitiveActionRequiresOpenApp() {
        XCTAssertTrue(LumenIntentPolicy.requiresOpenAppForSensitiveAction("calendar.read"))
    }

    func testHeadlessPromptAllowsLocalMemoryRecall() {
        let decision = LumenIntentPolicy.headlessPromptDecision(for: "what do you remember about my workshop preferences")

        XCTAssertFalse(decision.requiresOpenApp)
        XCTAssertNil(decision.reason)
    }

    func testHeadlessPromptRequiresOpenAppForNetwork() {
        let decision = LumenIntentPolicy.headlessPromptDecision(for: "search the web for the latest iOS background task docs")

        XCTAssertTrue(decision.requiresOpenApp)
        XCTAssertEqual(decision.reason, "trigger may require external network access")
    }

    func testHeadlessPromptRequiresOpenAppForApprovedAction() {
        let decision = LumenIntentPolicy.headlessPromptDecision(for: "send email to Alex with the jobsite update")

        XCTAssertTrue(decision.requiresOpenApp)
        XCTAssertEqual(decision.reason, "trigger may require approved actions")
    }

    func testHeadlessPromptRequiresOpenAppForProtectedPersonalData() {
        let decision = LumenIntentPolicy.headlessPromptDecision(for: "search my photos for receipts from yesterday")

        XCTAssertTrue(decision.requiresOpenApp)
        XCTAssertEqual(decision.reason, "trigger may read protected personal data")
    }

    func testHeadlessPromptRequiresOpenAppForExplicitSensitiveToolID() {
        let decision = LumenIntentPolicy.headlessPromptDecision(for: "run outlook.message.delete for the latest message")

        XCTAssertTrue(decision.requiresOpenApp)
        XCTAssertEqual(decision.reason, "trigger may require approved actions")
    }
}
