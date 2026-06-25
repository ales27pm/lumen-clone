import XCTest
@testable import Lumen

final class BackgroundEntitlementValidatorTests: XCTestCase {
    func testMissingKeysProducesWarnings() {
        let warnings = BackgroundEntitlementValidator.validate(infoDictionary: [:])
        XCTAssertFalse(warnings.isEmpty)
    }

    func testContinuedProcessingPlistPatternAndConcreteSubmissionIdentifierDoNotConflate() {
        let token = "submission-token"
        let concreteIdentifier = TriggerScheduler.continuedProcessingIdentifier(for: token)

        XCTAssertEqual(
            TriggerScheduler.continuedProcessingIdentifierPattern,
            "com.27pm.lumenclone.agent.continued-processing.*"
        )
        XCTAssertEqual(
            TriggerScheduler.continuedProcessingRegistrationIdentifier,
            TriggerScheduler.continuedProcessingIdentifierPattern
        )
        XCTAssertEqual(
            concreteIdentifier,
            "com.27pm.lumenclone.agent.continued-processing.\(token)"
        )
        XCTAssertFalse(concreteIdentifier.contains("*"))
        XCTAssertTrue(TriggerScheduler.continuedProcessingRegistrationIdentifier.contains("*"))
    }
}
