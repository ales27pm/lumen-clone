import XCTest
@testable import Lumen

final class DeveloperFrameworkTests: XCTestCase {
    private struct LiveE2EExportFixture: Codable {
        let passed: Bool
        let failed: Int
        let results: [String]
    }

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

    func testLiveE2EExportEnvelopeMatchesImproveLoopIngestionContract() throws {
        let result = try EvidenceLayerExporter.writeLayer(
            payload: LiveE2EExportFixture(passed: false, failed: 1, results: ["fixture"]),
            filePrefix: "lumen-live-e2e-report",
            format: "live-e2e-test-report-json",
            sourceLayer: "e2eTestReport",
            ownsLiveE2EScenarios: true,
            includesDeterministicStaticScenarios: false,
            privacy: "fixture",
            notes: ["fixture"]
        )

        let data = try Data(contentsOf: result.url)
        let object = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        let policy = try XCTUnwrap(object["exportPolicy"] as? [String: Any])
        let payload = try XCTUnwrap(object["payload"] as? [String: Any])

        XCTAssertEqual(policy["format"] as? String, "live-e2e-test-report-json")
        XCTAssertEqual(policy["sourceLayer"] as? String, "e2eTestReport")
        XCTAssertEqual(policy["ownsLiveE2EScenarios"] as? Bool, true)
        XCTAssertEqual(policy["includesDeterministicStaticScenarios"] as? Bool, false)
        XCTAssertEqual(payload["failed"] as? Int, 1)
        XCTAssertTrue(result.url.lastPathComponent.hasPrefix("lumen-live-e2e-report-"))
        XCTAssertEqual(result.url.pathComponents.suffix(2).first, "LumenEvidenceLayerExports")
    }
}
