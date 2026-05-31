import XCTest
@testable import Lumen

final class BackgroundEntitlementValidatorTests: XCTestCase {
    func testMissingKeysProducesWarnings() {
        let warnings = BackgroundEntitlementValidator.validate(infoDictionary: [:])
        XCTAssertFalse(warnings.isEmpty)
    }
}
