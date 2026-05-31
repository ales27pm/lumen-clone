import XCTest
@testable import Lumen

final class LegacyExternalToolPolicyTests: XCTestCase {
    @MainActor func testSensitiveLegacyDenied() async {
        let out = await LegacySecureToolExecutor.execute(toolID: "open.url", arguments: AgentJSONArguments(stringDictionary: ["url":"https://x.com"]))
        XCTAssertTrue(out.lowercased().contains("denied"))
        XCTAssertFalse(out.lowercased().contains("approve"))
    }
}
