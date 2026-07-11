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
        XCTAssertNil(snap.background.continuedProcessingLastSubmittedIdentifier)
        XCTAssertEqual(
            snap.background.continuedProcessingRegistrationErrorDomain == nil,
            snap.background.continuedProcessingRegistrationErrorCode == nil
        )
        XCTAssertNil(snap.background.continuedProcessingSubmitErrorDomain)
        XCTAssertNil(snap.background.continuedProcessingSubmitErrorCode)
        if #available(iOS 26.0, *) {
            XCTAssertEqual(snap.background.continuedProcessingRegisteredBeforeLaunchCompletion, true)
        } else {
            XCTAssertNil(snap.background.continuedProcessingRegisteredBeforeLaunchCompletion)
        }
        XCTAssertEqual(snap.background.continuedProcessingExpectedEntitlementValue, "true")
    }
}
