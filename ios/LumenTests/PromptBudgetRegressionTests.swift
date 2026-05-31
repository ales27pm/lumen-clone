import XCTest
@testable import Lumen

final class PromptBudgetRegressionTests: XCTestCase {
    func testAllocatorBounded() {
        let b = ContextBudgetAllocator.allocate(maxChars: 2000)
        XCTAssertLessThanOrEqual(b.system + b.history + b.memories + b.rag + b.tools + b.runtime, 2000)
    }
}
