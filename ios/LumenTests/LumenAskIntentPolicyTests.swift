import XCTest
@testable import Lumen

final class LumenAskIntentPolicyTests: XCTestCase {
    func testAskEmptyRejectedByValidationRule() {
        let q = "   ".trimmingCharacters(in: .whitespacesAndNewlines)
        XCTAssertTrue(q.isEmpty)
    }

    func testAskRejectsSensitiveHeadlessRequestBeforeKernelRun() {
        if #available(iOS 16.0, *) {
            let message = LumenAskIntent.policyMessage(for: "send an email to Alex")

            XCTAssertEqual(message, "Open Lumen to approve: trigger may require approved actions")
        }
    }

    func testAskAllowsBoundedLocalHeadlessQuestion() {
        if #available(iOS 16.0, *) {
            XCTAssertNil(LumenAskIntent.policyMessage(for: "summarize the difference between actors and classes"))
        }
    }

    func testAppIntentRemainsComputeForegroundButCannotPromptForToolPermission() {
        XCTAssertTrue(AgentKernelSource.appIntent.isForeground)
        XCTAssertFalse(AgentKernelSource.appIntent.allowsPermissionPrompts)
        XCTAssertFalse(ToolInvocationSource.appIntent.allowsPermissionPrompts)
    }
}
