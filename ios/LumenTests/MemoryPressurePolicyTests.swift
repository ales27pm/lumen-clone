import XCTest
@testable import Lumen

final class MemoryPressurePolicyTests: XCTestCase {
    func testUnloadPriority() {
        XCTAssertEqual(MemoryPressureUnloadPolicy.slotPriority, [.mimicry, .rem, .executor, .cortex, .mouth])
    }
}
