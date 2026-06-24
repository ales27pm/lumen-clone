import Foundation
#if canImport(AppIntents)
import AppIntents

@available(iOS 16.0, *)
struct LumenDiagnosticsIntent: AppIntent {
    static var title: LocalizedStringResource = "Check Lumen Diagnostics"
    static var description = IntentDescription("Show local runtime status and remediation hints without starting model work.")
    static var openAppWhenRun = false

    @Parameter(title: "Include Remediation", default: true) var includeRemediation: Bool

    @MainActor
    func perform() async throws -> some IntentResult & ReturnsValue<String> {
        let state = await PersistentRuntimeDiagnosticsStore.shared.loadState()
        let campaign = await PersistentRuntimeDiagnosticsStore.shared.loadCampaign()
        let snapshot = DiagnosticsProvider().cachedSnapshot()
        let summary = PersistentRuntimeDiagnosticsSummaryRenderer.render(
            state: state,
            campaign: campaign,
            snapshot: snapshot,
            includeRemediation: includeRemediation
        )
        return .result(value: summary)
    }
}
#endif
