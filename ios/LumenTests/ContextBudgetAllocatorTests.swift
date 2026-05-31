import XCTest
@testable import Lumen

final class ContextBudgetAllocatorTests: XCTestCase {
    func testAllocationWithinBudget() {
        let s = ContextBudgetAllocator.allocate(maxChars: 4000)
        XCTAssertLessThanOrEqual(s.system + s.history + s.memories + s.rag + s.tools + s.runtime, 4000)
    }
}
