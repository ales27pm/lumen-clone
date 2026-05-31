import XCTest
@testable import Lumen

final class LegacySecureToolExecutorTests: XCTestCase {
    @MainActor func testUnknownSensitiveDenied() async {
        let out = await LegacySecureToolExecutor.execute(toolID: "web.fetch", arguments: AgentJSONArguments(stringDictionary: ["url":"https://example.com"]))
        XCTAssertTrue(out.lowercased().contains("denied") || out.lowercased().contains("approve"))
    }
}
