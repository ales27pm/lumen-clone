import XCTest
@testable import Lumen

@MainActor
final class ToolSecurityDiagnosticsTests: XCTestCase {
    func testToolRowsHaveCategories() async {
        let snap = await DiagnosticsProvider().collect()
        XCTAssertFalse(snap.tools.tools.isEmpty)
        XCTAssertFalse(snap.tools.tools.contains { $0.category.isEmpty })
    }
}
