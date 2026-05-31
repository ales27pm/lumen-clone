import SwiftUI

struct DiagnosticsView: View {
    @State private var snapshot: DiagnosticsSnapshot?
    @State private var provider = DiagnosticsProvider()

    var body: some View {
        Group {
            if let snapshot {
                List {
                    NavigationLink("Runtime") { RuntimeDashboardView(runtime: snapshot.runtime) }
                    NavigationLink("Permissions") { PermissionSnapshotView(snapshot: snapshot.permissions) }
                    NavigationLink("Tools") { ToolSecurityView(tools: snapshot.tools) }
                    NavigationLink("Background") { BackgroundDiagnosticsView(background: snapshot.background) }
                    NavigationLink("Grounding") { GroundingDiagnosticsView(grounding: snapshot.grounding) }
                    NavigationLink("Privacy") { PrivacyReportView(privacy: snapshot.privacy) }
                }
            } else {
                ProgressView("Loading diagnostics…")
            }
        }
        .navigationTitle("Diagnostics")
        .task { snapshot = await provider.collect() }
    }
}

struct PermissionSnapshotView: View {
    let snapshot: PermissionDiagnosticsSnapshot
    var body: some View {
        List(snapshot.domains, id: \.domain) { row in
            HStack { Text(row.domain); Spacer(); Text(row.state).foregroundStyle(.secondary) }
        }.navigationTitle("Permissions")
    }
}
