import XCTest
@testable import Lumen

final class LegacyGroundingResultTests: XCTestCase {
    func testConstructs() {
        let r = LegacyGroundingResult(systemPrompt: "s", userMessage: "u", grounding: nil, sections: [], bridgedTools: [], degradedReasons: ["x"], metricsSummary: "m", truncationOccurred: false)
        XCTAssertEqual(r.degradedReasons.first, "x")
    }
}
