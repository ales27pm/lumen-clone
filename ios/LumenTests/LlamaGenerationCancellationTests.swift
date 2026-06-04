import XCTest
@testable import Lumen

final class LlamaGenerationCancellationTests: XCTestCase {
    func testCancellationTokenPreservesReasonAndThrowsCancellationError() {
        let token = LlamaGenerationCancellationToken()

        XCTAssertFalse(token.isCancelled)
        token.cancel(reason: "scene-transition-test")

        XCTAssertTrue(token.isCancelled)
        XCTAssertEqual(token.reason, "scene-transition-test")
        XCTAssertThrowsError(try token.checkCancellation()) { error in
            XCTAssertTrue(error is CancellationError)
        }
    }
}
