import XCTest
@testable import Lumen

final class PrivacyReportTests: XCTestCase {
    func testPrivacyReportNoRawContentFields() {
        let report = PrivacyReportSnapshot(localOnlyMode: true, networkAccessState: "denied", recentToolCategories: ["readOnly"], appIntentLimitations: ["x"])
        XCTAssertFalse(report.recentToolCategories.contains { $0.contains("message") })
    }
}
