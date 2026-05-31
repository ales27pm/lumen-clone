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
                ContentUnavailableView("Diagnostics unavailable", systemImage: "waveform.path.ecg")
            }
        }
        .navigationTitle("Diagnostics")
        .toolbar {
            ToolbarItem(placement: .topBarTrailing) {
                Button("Refresh") {
                    Task { snapshot = await provider.collect() }
                }
            }
        }
        .onAppear { snapshot = provider.cachedSnapshot() }
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
