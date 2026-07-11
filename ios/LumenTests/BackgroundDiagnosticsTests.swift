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
        XCTAssertNil(snap.background.continuedProcessingRegistrationErrorDomain)
        XCTAssertNil(snap.background.continuedProcessingRegistrationErrorCode)
        XCTAssertNil(snap.background.continuedProcessingSubmitErrorDomain)
        XCTAssertNil(snap.background.continuedProcessingSubmitErrorCode)
        XCTAssertNil(snap.background.continuedProcessingRegisteredBeforeLaunchCompletion)
        XCTAssertEqual(snap.background.continuedProcessingExpectedEntitlementValue, "true")
    }
}
