import XCTest
@testable import Lumen

final class BackgroundEntitlementValidatorTests: XCTestCase {
    func testMissingKeysProducesWarnings() {
        let warnings = BackgroundEntitlementValidator.validate(infoDictionary: [:])
        XCTAssertFalse(warnings.isEmpty)
    }

    func testAlarmKitUsageDescriptionIsRequiredForRuntimeAvailability() {
        let warnings = BackgroundEntitlementValidator.validate(infoDictionary: [
            "BGTaskSchedulerPermittedIdentifiers": Array(BackgroundEntitlementValidator.requiredTaskIDs),
            "NSMicrophoneUsageDescription": "Voice input",
            "NSSpeechRecognitionUsageDescription": "Speech input",
            "NSLocationAlwaysAndWhenInUseUsageDescription": "Background location",
            "NSCalendarsFullAccessUsageDescription": "Calendar access",
            "NSContactsUsageDescription": "Contact lookup",
            "UIBackgroundModes": ["audio", "fetch", "location", "processing"]
        ])

        XCTAssertTrue(warnings.contains { warning in
            warning.code == "missing_usage_description"
                && warning.message.contains("NSAlarmKitUsageDescription")
        })
    }

    func testCompleteRuntimeUsageDescriptionsDoNotWarnForAlarmKit() {
        let warnings = BackgroundEntitlementValidator.validate(infoDictionary: [
            "BGTaskSchedulerPermittedIdentifiers": Array(BackgroundEntitlementValidator.requiredTaskIDs),
            "NSMicrophoneUsageDescription": "Voice input",
            "NSSpeechRecognitionUsageDescription": "Speech input",
            "NSLocationAlwaysAndWhenInUseUsageDescription": "Background location",
            "NSCalendarsFullAccessUsageDescription": "Calendar access",
            "NSContactsUsageDescription": "Contact lookup",
            "NSAlarmKitUsageDescription": "Alarm scheduling",
            "UIBackgroundModes": ["audio", "fetch", "location", "processing"]
        ])

        XCTAssertFalse(warnings.contains { $0.message.contains("NSAlarmKitUsageDescription") })
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
