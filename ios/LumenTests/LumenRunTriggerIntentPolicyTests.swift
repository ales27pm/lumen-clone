import XCTest
@testable import Lumen

final class LumenRunTriggerIntentPolicyTests: XCTestCase {
    func testSensitivePromptRequiresOpenApp() {
        XCTAssertTrue(LumenIntentPolicy.requiresOpenAppForSensitiveAction("calendar sync"))
    }

    func testRunTriggerIntentDoesNotCollapseEmptyResultToGenericNoResult() {
        if #available(iOS 16.0, *) {
            let rendered = LumenRunTriggerIntent.renderedTriggerResult(nil)

            XCTAssertTrue(rendered.contains("trigger returned empty result"))
            XCTAssertFalse(rendered.contains("No result"))
        }
    }

    func testRunTriggerIntentTrimsAndBoundsReturnedText() {
        if #available(iOS 16.0, *) {
            let rendered = LumenRunTriggerIntent.renderedTriggerResult("  " + String(repeating: "x", count: 520) + "  ")

            XCTAssertEqual(rendered.count, 500)
            XCTAssertFalse(rendered.contains("  "))
        }
    }
}
