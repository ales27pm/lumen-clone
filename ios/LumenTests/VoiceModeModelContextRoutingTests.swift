import XCTest
@testable import Lumen

final class VoiceModeModelContextRoutingTests: XCTestCase {
    func testLegacyRunOptionsCarriesModelContextField() {
        let opts = LegacyAgentRunOptions.default
        XCTAssertNil(opts.modelContext)
    }
}
