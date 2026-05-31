import XCTest
@testable import Lumen

final class VoiceCommandRouterTests: XCTestCase {
    func testTypeAccessible() {
        XCTAssertNotNil(VoiceCommandRouter.self)
    }
}
