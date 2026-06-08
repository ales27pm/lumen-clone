import XCTest
@testable import Lumen

final class DeveloperFrameworkTests: XCTestCase {
    func testOnlyLiveE2EOwnsScenarioPassFail() {
        for layer in DeveloperEvidenceLayer.allCases {
            if layer == .e2eTestReport {
                XCTAssertTrue(layer.ownsLiveE2EScenarios)
                XCTAssertEqual(layer.sourceLayer, "e2eTestReport")
            } else {
                XCTAssertFalse(layer.ownsLiveE2EScenarios)
            }
        }
    }

    func testEvidenceLayerBaselineIncludesAllLayers() {
        let baseline = DeveloperEvidenceLayerStatus.baseline()

        XCTAssertEqual(baseline.count, DeveloperEvidenceLayer.allCases.count)
        XCTAssertTrue(baseline.contains { $0.layer == .agentGroundingRuntimeAudit })
        XCTAssertTrue(baseline.contains { $0.layer == .agentBehaviorTraceRecorder })
        XCTAssertTrue(baseline.contains { $0.layer == .e2eTestReport && $0.status == "live owner" })
    }

    func testWorkflowActionsExposeExpectedExports() {
        XCTAssertEqual(DeveloperWorkflowAction.exportRuntimeAudit.title, "Export runtime audit package")
        XCTAssertEqual(DeveloperWorkflowAction.exportLiveE2E.systemImage, "arrow.up.doc")
        XCTAssertEqual(DeveloperWorkflowAction.exportRecentTraces.title, "Export recent runtime traces")
    }
}
