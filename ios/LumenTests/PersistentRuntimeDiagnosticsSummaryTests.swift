import XCTest
@testable import Lumen

final class PersistentRuntimeDiagnosticsSummaryTests: XCTestCase {
    func testSummaryIncludesLatestRemediationAndLocalPrivacyState() {
        var state = PersistentDiagnosticState()
        state.status.passedCount = 2
        state.status.failedCount = 1
        state.status.skippedCount = 1

        var older = PersistentDiagnosticRunRecord(
            campaignID: UUID(),
            scenario: .plainFastPrompt,
            startedAt: Date(timeIntervalSince1970: 10),
            status: .passed
        )
        older.finishedAt = Date(timeIntervalSince1970: 11)

        var latest = PersistentDiagnosticRunRecord(
            campaignID: UUID(),
            scenario: .thermalResourceGate,
            startedAt: Date(timeIntervalSince1970: 20),
            status: .failed,
            failureSummary: "resource_gate_policy_failed"
        )
        latest.finishedAt = Date(timeIntervalSince1970: 21)
        latest.remediationProposals = [
            PersistentDiagnosticRemediationProposal(
                id: "fix-resource-gate-policy-matrix",
                title: "Fix the resource-gate policy matrix",
                rationale: "Simulated resource states did not match expected policy.",
                action: "Inspect ResourceBudgetGate and ModelLoader policy checks.",
                severity: .critical
            )
        ]
        state.records = [older, latest]

        let text = PersistentRuntimeDiagnosticsSummaryRenderer.render(
            state: state,
            campaign: PersistentDiagnosticCampaign(enabled: true, runContinuously: true),
            snapshot: Self.snapshot(),
            includeRemediation: true
        )

        XCTAssertTrue(text.contains("campaign=continuous"))
        XCTAssertTrue(text.contains("passed=2, failed=1, skipped=1"))
        XCTAssertTrue(text.contains("Privacy: localOnly=true; network=unknown"))
        XCTAssertTrue(text.contains("Latest: Thermal resource gate failed"))
        XCTAssertTrue(text.contains("Remediation: Fix the resource-gate policy matrix"))
        XCTAssertLessThanOrEqual(text.count, PersistentRuntimeDiagnosticsSummaryRenderer.defaultMaxCharacters)
    }

    func testSummaryDoesNotClaimRunsWhenStateIsEmpty() {
        let text = PersistentRuntimeDiagnosticsSummaryRenderer.render(
            state: nil,
            campaign: nil,
            snapshot: Self.snapshot(),
            includeRemediation: true
        )

        XCTAssertTrue(text.contains("campaign=not configured"))
        XCTAssertTrue(text.contains("No persistent diagnostic runs recorded yet."))
        XCTAssertFalse(text.contains("Remediation:"))
    }

    private static func snapshot() -> DiagnosticsSnapshot {
        DiagnosticsSnapshot(
            runtime: RuntimeDiagnosticsSnapshot(
                foundationModelsAvailable: false,
                foundationModelsStatus: "unavailable",
                coreMLAvailable: true,
                coreMLStatus: "available",
                metalAvailable: true,
                lowPowerModeEnabled: false,
                thermalState: "nominal",
                memoryWarningCount: 0,
                recentMetricSummaries: []
            ),
            permissions: PermissionDiagnosticsSnapshot(domains: []),
            tools: ToolSecuritySnapshot(tools: []),
            background: BackgroundDiagnosticsSnapshot(
                permittedIdentifiers: [],
                entitlementWarnings: [],
                entitlementStates: [],
                backgroundGPUSupported: false,
                continuedProcessingStatus: "cached",
                availableMemoryBytes: 0,
                energyKit: EnergyKitCapabilitySnapshot(frameworkAvailable: false, expectedEntitlementConfigured: false, status: "test", venueCount: nil),
                storeKit: StoreKitCapabilitySnapshot(frameworkAvailable: true, status: "test", environment: "test")
            ),
            grounding: GroundingDiagnosticsSnapshot(contextSource: "test", degradedReasons: [], sectionCounts: [:], doubleGroundingNormalized: true),
            privacy: PrivacyReportSnapshot(localOnlyMode: true, networkAccessState: "unknown", recentToolCategories: [], appIntentLimitations: [])
        )
    }
}
