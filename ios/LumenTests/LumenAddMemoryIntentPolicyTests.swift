import XCTest
@testable import Lumen

final class LumenAddMemoryIntentPolicyTests: XCTestCase {
    func testCredentialLikeRejectedByPolicyEntrypoint() {
        if #available(iOS 16.0, *) {
            let message = LumenAddMemoryIntent.policyMessage(for: "my password is 123")
            XCTAssertEqual(message, "Memory rejected: credential-like content is not allowed.")
        }
    }

    func testSensitiveMemoryRequiresOpenApp() {
        if #available(iOS 16.0, *) {
            let message = LumenAddMemoryIntent.policyMessage(for: "medical detail")
            XCTAssertTrue(message?.contains("Open Lumen to approve") == true)
        }
    }
}
