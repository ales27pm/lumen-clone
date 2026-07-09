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

    func testQueuedMessageDescribesLocalLaterIndexing() {
        if #available(iOS 16.0, *) {
            let message = LumenAddMemoryIntent.queuedMessage(pendingCount: 2, retryCount: 0)
            XCTAssertEqual(message, "Memory captured locally for later indexing. Pending captures: 2.")
        }
    }

    func testQueuedMessageDoesNotInventPendingCountWhenUnknown() {
        if #available(iOS 16.0, *) {
            let message = LumenAddMemoryIntent.queuedMessage(
                pendingCount: nil,
                pendingDiagnostic: "pending_count_failed:decode_failed",
                retryCount: 1
            )
            XCTAssertEqual(
                message,
                "Memory captured locally for later indexing. Pending captures: unknown. Diagnostic: pending_count_failed:decode_failed. Retry count: 1."
            )
        }
    }

    func testSavedMessageIncludesPromotedPendingCaptures() {
        if #available(iOS 16.0, *) {
            let message = LumenAddMemoryIntent.savedMessage(drained: 2)
            XCTAssertEqual(message, "Memory saved. Also indexed 2 pending memory captures.")
        }
    }
}
