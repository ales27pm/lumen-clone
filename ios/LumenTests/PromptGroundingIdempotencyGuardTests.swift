import XCTest
@testable import Lumen

final class PromptGroundingIdempotencyGuardTests: XCTestCase {
    func testDetectAndStripGeneratedMarker() {
        let text = "hello\n\n" + PromptGroundingIdempotencyGuard.marker + "\n[LOCAL MEMORY]\na"
        let stripped = PromptGroundingIdempotencyGuard.stripExistingGrounding(from: text)
        XCTAssertTrue(stripped.stripped)
        XCTAssertEqual(stripped.text, "hello")
    }

    func testAmbiguousSingleHeaderNotStripped() {
        let text = "User wrote: [LOCAL MEMORY] keep this"
        let stripped = PromptGroundingIdempotencyGuard.stripExistingGrounding(from: text)
        XCTAssertFalse(stripped.stripped)
        XCTAssertTrue(stripped.ambiguous)
    }

    func testEarliestKnownHeaderStrippedWhenGeneratedHeadersExist() {
        let text = "hello\n[AVAILABLE LOCAL TOOLS]\ntool\n[LOCAL MEMORY]\nmemory"
        let stripped = PromptGroundingIdempotencyGuard.stripExistingGrounding(from: text)
        XCTAssertTrue(stripped.stripped)
        XCTAssertEqual(stripped.text, "hello")
    }
}
