import Foundation

struct DiagnosticsSnapshot: Sendable {
    let build: BuildDiagnosticsSnapshot
    let runtime: RuntimeDiagnosticsSnapshot
    let permissions: PermissionDiagnosticsSnapshot
    let tools: ToolSecuritySnapshot
    let background: BackgroundDiagnosticsSnapshot
    let grounding: GroundingDiagnosticsSnapshot
    let privacy: PrivacyReportSnapshot
}
