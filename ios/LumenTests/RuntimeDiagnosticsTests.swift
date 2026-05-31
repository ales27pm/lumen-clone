import XCTest
@testable import Lumen

@MainActor
final class RuntimeDiagnosticsTests: XCTestCase {
    func testMetricSummariesBounded() async {
        let snap = await DiagnosticsProvider().collect()
        XCTAssertLessThanOrEqual(snap.runtime.recentMetricSummaries.count, 5)
    }
}
