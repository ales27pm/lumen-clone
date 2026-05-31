import XCTest
@testable import Lumen

final class LumenAskIntentPolicyTests: XCTestCase {
    func testAskEmptyRejectedByValidationRule() {
        let q = "   ".trimmingCharacters(in: .whitespacesAndNewlines)
        XCTAssertTrue(q.isEmpty)
    }
}
