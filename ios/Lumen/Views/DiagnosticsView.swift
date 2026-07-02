import SwiftUI

struct DiagnosticsView: View {
    @State private var snapshot: DiagnosticsSnapshot?
    @State private var provider = DiagnosticsProvider()

    var body: some View {
        Group {
            if let snapshot {
                List {
                    NavigationLink("Build") { BuildDiagnosticsView(build: snapshot.build) }
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

struct BuildDiagnosticsView: View {
    let build: BuildDiagnosticsSnapshot

    var body: some View {
        List {
            row("Bundle ID", build.bundleIdentifier)
            row("Bundle version", build.bundleVersion)
            row("Build source", build.buildSourceIdentifier)
            row("Git SHA", build.gitSHA)
            row("Configuration", build.configuration)
            row("Scheme", build.scheme)
            row("NSAlarmKitUsageDescription", build.alarmKitUsageDescription ?? "Missing")
        }
        .navigationTitle("Build")
    }

    private func row(_ title: String, _ value: String) -> some View {
        HStack(alignment: .firstTextBaseline) {
            Text(title)
            Spacer(minLength: 16)
            Text(value)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.trailing)
        }
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
