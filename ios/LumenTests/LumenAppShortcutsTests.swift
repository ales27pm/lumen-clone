import XCTest
@testable import Lumen

final class LumenAppShortcutsTests: XCTestCase {
    func testRendererBoundedOutput() {
        let rendered = LumenIntentResultRenderer.degraded("x")
        XCTAssertFalse(rendered.isEmpty)
    }
}
