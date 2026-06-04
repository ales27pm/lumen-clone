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

    func testSmokeTestDisplayStringsFollowMode() {
        let diagnostics = RolePipelineSmokeTestTraceExpectation(
            mode: .compatibilityDiagnosticsOnly,
            compatibilityModeNotice: "Compatibility-mode only."
        )
        XCTAssertEqual(diagnostics.buttonTitle, "Run Compatibility Diagnostics Smoke Test")
        XCTAssertEqual(diagnostics.runningTitle, "Running Diagnostics Smoke Test…")
        XCTAssertTrue(diagnostics.footerSentence.contains("does not run AppLlamaService"))
        XCTAssertEqual(diagnostics.promptPrefix, "Compatibility diagnostics smoke test")

        let realModel = RolePipelineSmokeTestTraceExpectation(
            mode: .realModelTrace,
            compatibilityModeNotice: nil
        )
        XCTAssertEqual(realModel.buttonTitle, "Run Real Model Trace Smoke Test")
        XCTAssertEqual(realModel.runningTitle, "Running Real Model Trace Smoke Test…")
        XCTAssertTrue(realModel.footerSentence.contains("record AgentBehaviorTrace"))
        XCTAssertEqual(realModel.promptPrefix, "Real-model trace smoke test")
    }

    func testRealModelTraceSummaryWarnsWhenExpectedTraceIsMissing() {
        let expectation = RolePipelineSmokeTestTraceExpectation(
            mode: .realModelTrace,
            compatibilityModeNotice: nil
        )

        XCTAssertTrue(
            expectation.completedSummary(recordedTraceCount: 0, tailChanged: false, producedOutput: true)
                .contains("no AgentBehaviorTrace was recorded")
        )
        XCTAssertTrue(
            expectation.completedSummary(recordedTraceCount: 2, tailChanged: false, producedOutput: true)
                .contains("recorded 2 trace(s)")
        )
    }

    func testDiagnosticsOnlySummarySurfacesUnexpectedTraceRecording() {
        let expectation = RolePipelineSmokeTestTraceExpectation(
            mode: .compatibilityDiagnosticsOnly,
            compatibilityModeNotice: "Compatibility-mode only."
        )

        XCTAssertTrue(
            expectation.completedSummary(recordedTraceCount: 1, tailChanged: false, producedOutput: true)
                .contains("unexpectedly recorded 1 AgentBehaviorTrace")
        )
        XCTAssertTrue(
            expectation.completedSummary(recordedTraceCount: 0, tailChanged: false, producedOutput: true)
                .contains("Compatibility-mode only")
        )
    }
}
