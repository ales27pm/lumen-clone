import XCTest
@testable import Lumen

final class DeveloperFrameworkTests: XCTestCase {

    // MARK: - Existing tests

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

    // MARK: - DeveloperConsoleTab

    func testConsoleTabTitles() {
        XCTAssertEqual(DeveloperConsoleTab.overview.title, "Overview")
        XCTAssertEqual(DeveloperConsoleTab.evidence.title, "Evidence")
        XCTAssertEqual(DeveloperConsoleTab.workflows.title, "Workflow")
        XCTAssertEqual(DeveloperConsoleTab.reports.title, "Reports")
        XCTAssertEqual(DeveloperConsoleTab.privacy.title, "Privacy")
    }

    func testConsoleTabIDsMatchRawValues() {
        for tab in DeveloperConsoleTab.allCases {
            XCTAssertEqual(tab.id, tab.rawValue)
        }
    }

    func testConsoleTabAllCasesHasExpectedCount() {
        XCTAssertEqual(DeveloperConsoleTab.allCases.count, 5)
    }

    // MARK: - DeveloperEvidenceLayer properties

    func testEvidenceLayerTitles() {
        XCTAssertEqual(DeveloperEvidenceLayer.agentGroundingRuntimeAudit.title, "Runtime Audit Package")
        XCTAssertEqual(DeveloperEvidenceLayer.runtimeManifestAudit.title, "Runtime Registry")
        XCTAssertEqual(DeveloperEvidenceLayer.agentModelBehaviorAuditor.title, "Model Behaviour")
        XCTAssertEqual(DeveloperEvidenceLayer.runtimeScenarioRunnerStaticChecks.title, "Static Scenarios")
        XCTAssertEqual(DeveloperEvidenceLayer.agentBehaviorTraceRecorder.title, "Runtime Traces")
        XCTAssertEqual(DeveloperEvidenceLayer.e2eTestReport.title, "Live E2E")
    }

    func testEvidenceLayerSystemImages() {
        XCTAssertEqual(DeveloperEvidenceLayer.agentGroundingRuntimeAudit.systemImage, "checkmark.seal.text.page")
        XCTAssertEqual(DeveloperEvidenceLayer.runtimeManifestAudit.systemImage, "list.bullet.rectangle")
        XCTAssertEqual(DeveloperEvidenceLayer.agentModelBehaviorAuditor.systemImage, "brain.head.profile")
        XCTAssertEqual(DeveloperEvidenceLayer.runtimeScenarioRunnerStaticChecks.systemImage, "checklist.checked")
        XCTAssertEqual(DeveloperEvidenceLayer.agentBehaviorTraceRecorder.systemImage, "waveform.path.ecg")
        XCTAssertEqual(DeveloperEvidenceLayer.e2eTestReport.systemImage, "testtube.2")
    }

    func testEvidenceLayerSourceLayersAreStable() {
        XCTAssertEqual(DeveloperEvidenceLayer.agentGroundingRuntimeAudit.sourceLayer, "agentGroundingRuntimeAudit")
        XCTAssertEqual(DeveloperEvidenceLayer.runtimeManifestAudit.sourceLayer, "runtimeManifestAudit")
        XCTAssertEqual(DeveloperEvidenceLayer.agentModelBehaviorAuditor.sourceLayer, "agentModelBehaviorAuditor")
        // Note the dot-separated format for this case
        XCTAssertEqual(DeveloperEvidenceLayer.runtimeScenarioRunnerStaticChecks.sourceLayer, "runtimeScenarioRunner.staticChecks")
        XCTAssertEqual(DeveloperEvidenceLayer.agentBehaviorTraceRecorder.sourceLayer, "agentBehaviorTraceRecorder")
        XCTAssertEqual(DeveloperEvidenceLayer.e2eTestReport.sourceLayer, "e2eTestReport")
    }

    func testEvidenceLayerTrustRole() {
        XCTAssertEqual(DeveloperEvidenceLayer.e2eTestReport.trustRole, "Scenario pass/fail owner")
        for layer in DeveloperEvidenceLayer.allCases where layer != .e2eTestReport {
            XCTAssertEqual(layer.trustRole, "Diagnostic evidence")
        }
    }

    func testEvidenceLayerPrivacySummariesAreNonEmpty() {
        for layer in DeveloperEvidenceLayer.allCases {
            XCTAssertFalse(layer.privacySummary.isEmpty, "\(layer) has empty privacySummary")
        }
    }

    func testEvidenceLayerNextActionsAreNonEmpty() {
        for layer in DeveloperEvidenceLayer.allCases {
            XCTAssertFalse(layer.nextAction.isEmpty, "\(layer) has empty nextAction")
        }
    }

    func testEvidenceLayerIDsMatchRawValues() {
        for layer in DeveloperEvidenceLayer.allCases {
            XCTAssertEqual(layer.id, layer.rawValue)
        }
    }

    func testEvidenceLayerAllCasesHasExpectedCount() {
        XCTAssertEqual(DeveloperEvidenceLayer.allCases.count, 6)
    }

    // MARK: - DeveloperEvidenceLayerStatus

    func testBaselineStatusForNonLiveLayersIsDiagnostic() {
        let baseline = DeveloperEvidenceLayerStatus.baseline()
        for entry in baseline where entry.layer != .e2eTestReport {
            XCTAssertEqual(entry.status, "diagnostic", "\(entry.layer) should have 'diagnostic' status")
        }
    }

    func testBaselineCountIsNilForAllLayers() {
        let baseline = DeveloperEvidenceLayerStatus.baseline()
        for entry in baseline {
            XCTAssertNil(entry.count, "\(entry.layer) baseline count should be nil")
        }
    }

    func testBaselineIsBlockingIsFalseForAllLayers() {
        let baseline = DeveloperEvidenceLayerStatus.baseline()
        for entry in baseline {
            XCTAssertFalse(entry.isBlocking, "\(entry.layer) baseline isBlocking should be false")
        }
    }

    func testBaselineDetailMatchesLayerNextAction() {
        let baseline = DeveloperEvidenceLayerStatus.baseline()
        for entry in baseline {
            XCTAssertEqual(entry.detail, entry.layer.nextAction, "\(entry.layer) detail should equal nextAction")
        }
    }

    func testEvidenceLayerStatusIDMatchesLayerID() {
        let status = DeveloperEvidenceLayerStatus(
            layer: .agentGroundingRuntimeAudit,
            status: "diagnostic",
            detail: "detail",
            count: nil,
            isBlocking: false
        )
        XCTAssertEqual(status.id, DeveloperEvidenceLayer.agentGroundingRuntimeAudit.id)
    }

    // MARK: - DeveloperWorkflowAction

    func testWorkflowActionTitlesAreComplete() {
        XCTAssertEqual(DeveloperWorkflowAction.collectDiagnostics.title, "Collect diagnostics")
        XCTAssertEqual(DeveloperWorkflowAction.runAgentGrounding.title, "Run Agent Grounding audit")
        XCTAssertEqual(DeveloperWorkflowAction.runLiveTraceSmoke.title, "Run live trace smoke test")
        XCTAssertEqual(DeveloperWorkflowAction.runE2EStandard.title, "Run standard E2E suite")
        XCTAssertEqual(DeveloperWorkflowAction.runE2ETraining.title, "Run training E2E suite")
        XCTAssertEqual(DeveloperWorkflowAction.runPersistentDiagnostics.title, "Run persistent diagnostics")
    }

    func testWorkflowActionSystemImagesAreComplete() {
        XCTAssertEqual(DeveloperWorkflowAction.collectDiagnostics.systemImage, "waveform.path.ecg")
        XCTAssertEqual(DeveloperWorkflowAction.runAgentGrounding.systemImage, "checkmark.seal.text.page")
        XCTAssertEqual(DeveloperWorkflowAction.runLiveTraceSmoke.systemImage, "bolt.heart")
        XCTAssertEqual(DeveloperWorkflowAction.runE2EStandard.systemImage, "testtube.2")
        XCTAssertEqual(DeveloperWorkflowAction.runE2ETraining.systemImage, "graduationcap")
        XCTAssertEqual(DeveloperWorkflowAction.runPersistentDiagnostics.systemImage, "repeat.circle")
        XCTAssertEqual(DeveloperWorkflowAction.exportRuntimeAudit.systemImage, "square.and.arrow.up")
    }

    func testWorkflowActionIDsMatchRawValues() {
        for action in DeveloperWorkflowAction.allCases {
            XCTAssertEqual(action.id, action.rawValue)
        }
    }

    func testWorkflowActionAllCasesHasExpectedCount() {
        XCTAssertEqual(DeveloperWorkflowAction.allCases.count, 9)
    }

    func testWorkflowActionTitlesAndSystemImagesAreNonEmpty() {
        for action in DeveloperWorkflowAction.allCases {
            XCTAssertFalse(action.title.isEmpty, "\(action) has empty title")
            XCTAssertFalse(action.systemImage.isEmpty, "\(action) has empty systemImage")
        }
    }

    // MARK: - DeveloperFinding.Severity

    func testFindingSeveritySystemImages() {
        XCTAssertEqual(DeveloperFinding.Severity.info.systemImage, "info.circle")
        XCTAssertEqual(DeveloperFinding.Severity.warning.systemImage, "exclamationmark.triangle")
        XCTAssertEqual(DeveloperFinding.Severity.error.systemImage, "xmark.octagon")
    }

    func testFindingSeverityRawValues() {
        XCTAssertEqual(DeveloperFinding.Severity.info.rawValue, "info")
        XCTAssertEqual(DeveloperFinding.Severity.warning.rawValue, "warning")
        XCTAssertEqual(DeveloperFinding.Severity.error.rawValue, "error")
    }

    // MARK: - DeveloperFinding

    func testFindingHasUniqueIDs() {
        let f1 = DeveloperFinding(severity: .info, title: "T1", detail: "D1")
        let f2 = DeveloperFinding(severity: .info, title: "T1", detail: "D1")
        XCTAssertNotEqual(f1.id, f2.id)
    }

    func testFindingEquatableComparesAllStoredProperties() {
        let f1 = DeveloperFinding(severity: .warning, title: "Title", detail: "Detail")
        let f2 = DeveloperFinding(severity: .warning, title: "Title", detail: "Detail")
        // UUID is let with default value, each instance gets a unique UUID.
        // Two separate findings are NOT equal because IDs differ.
        XCTAssertNotEqual(f1, f2)
    }

    // MARK: - Regression: static scenario layer must not own live E2E

    func testStaticScenariosLayerDoesNotOwnLiveE2E() {
        XCTAssertFalse(DeveloperEvidenceLayer.runtimeScenarioRunnerStaticChecks.ownsLiveE2EScenarios)
    }

    // MARK: - Regression: source layer dot notation is preserved

    func testRuntimeScenarioRunnerStaticChecksSourceLayerUsesDotseparation() {
        let sourceLayer = DeveloperEvidenceLayer.runtimeScenarioRunnerStaticChecks.sourceLayer
        XCTAssertTrue(sourceLayer.contains("."), "sourceLayer should use dot-separated notation for sub-category")
    }
}
