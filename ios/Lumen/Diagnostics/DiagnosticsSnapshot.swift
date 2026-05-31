import Foundation

struct DiagnosticsSnapshot: Sendable {
    let runtime: RuntimeDiagnosticsSnapshot
    let permissions: PermissionDiagnosticsSnapshot
    let tools: ToolSecuritySnapshot
    let background: BackgroundDiagnosticsSnapshot
    let grounding: GroundingDiagnosticsSnapshot
    let privacy: PrivacyReportSnapshot
}
