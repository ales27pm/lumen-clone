import XCTest
@testable import Lumen

@MainActor
final class BackgroundDiagnosticsTests: XCTestCase {
    func testCachedSnapshotContainsEntitlementWarningsField() {
        let snap = DiagnosticsProvider().cachedSnapshot()
        XCTAssertNotNil(snap.background.entitlementWarnings)
    }

    func testCachedSnapshotExposesContinuedProcessingDiagnostics() {
        let snap = DiagnosticsProvider().cachedSnapshot()
        XCTAssertEqual(
            snap.background.continuedProcessingRegistrationIdentifier,
            TriggerScheduler.continuedProcessingRegistrationIdentifier
        )
        XCTAssertTrue(snap.background.continuedProcessingRegistrationIdentifier.contains("*"))
        XCTAssertEqual(snap.background.continuedProcessingExpectedEntitlementValue, "true")
    }
}
