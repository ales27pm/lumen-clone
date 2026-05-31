import XCTest
@testable import Lumen

final class LumenIntentResultRendererTests: XCTestCase {
    func testOpenAppMessage() {
        XCTAssertTrue(LumenIntentResultRenderer.openAppRequired("calendar").contains("Open Lumen"))
    }
}
