import XCTest
@testable import Lumen

final class SafeToolOutputLimiterTests: XCTestCase {
    func testNeverExceedsMax() {
        let r = ToolResult(invocationID: UUID(), status: .success, displayText: String(repeating: "a", count: 20), modelText: String(repeating: "b", count: 20), structuredPayload: nil, privacyLevel: .low, metricsSummary: "", errorCode: nil)
        for max in 0...12 {
            let out = SafeToolOutputLimiter.limit(result: r, maxOutput: max)
            XCTAssertLessThanOrEqual(out.displayText.count, max)
            XCTAssertLessThanOrEqual(out.modelText.count, max)
        }
    }
}
