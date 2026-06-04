import XCTest
@testable import Lumen

final class RolePipelineSmokeTestTracePolicyTests: XCTestCase {
    func testSmokeTestPathRecordsTraceOrReportsCompatibilityModeOnly() {
        let expectation = RolePipelineAgentService.smokeTestTraceExpectation

        if expectation.recordsAgentBehaviorTrace {
            XCTAssertEqual(expectation.mode, .realModelTrace)
            XCTAssertNil(expectation.compatibilityModeNotice)
        } else {
            XCTAssertEqual(expectation.mode, .compatibilityDiagnosticsOnly)
            let notice = expectation.compatibilityModeNotice ?? ""
            XCTAssertTrue(
                notice.localizedCaseInsensitiveContains("compatibility-mode only"),
                "Diagnostics-only smoke tests must explicitly report that they are compatibility-mode only."
            )
            XCTAssertTrue(
                notice.localizedCaseInsensitiveContains("does not run AppLlamaService")
                    || notice.localizedCaseInsensitiveContains("does not record AgentBehaviorTrace"),
                "Diagnostics-only smoke tests must not imply that they run the real model trace path."
            )
        }
    }
}
