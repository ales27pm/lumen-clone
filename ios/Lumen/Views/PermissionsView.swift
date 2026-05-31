import SwiftUI

struct PermissionsView: View {
    @State private var center = PermissionsCenter.shared
    @Environment(\.scenePhase) private var scenePhase

    var body: some View {
        List {
            Section {
                ForEach(PermissionKind.allCases) { kind in
                    row(for: kind)
                }
            } footer: {
                Text("Lumen requests permissions on demand when you use a feature. You can also pre-grant them here. Denied permissions must be changed in the iOS Settings app. AlarmKit also requires the AlarmKit entitlement in the signed provisioning profile.")
            }

            Section {
                Button {
                    center.openSystemSettings()
                } label: {
                    Label("Open iOS Settings", systemImage: "arrow.up.forward.app")
                }
            }
        }
        .navigationTitle("Permissions")
        .onAppear { center.refreshAll() }
        .onChange(of: scenePhase) { _, phase in
            if phase == .active { center.refreshAll() }
        }
    }

    @ViewBuilder
    private func row(for kind: PermissionKind) -> some View {
        let state = center.state(kind)
        Button {
            Task { await center.request(kind) }
        } label: {
            HStack(alignment: .center, spacing: 12) {
                Image(systemName: kind.systemImage)
                    .font(.system(size: 17, weight: .semibold))
                    .foregroundStyle(Color(state.tint))
                    .frame(width: 28)

                VStack(alignment: .leading, spacing: 3) {
                    Text(kind.title)
                        .font(.body)
                        .foregroundStyle(.primary)
                    Text(kind.rationale)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .lineLimit(2)
                    if let message = center.lastRequestMessage(kind), !message.isEmpty {
                        Text(message)
                            .font(.caption2)
                            .foregroundStyle(message.lowercased().contains("failed") || message.lowercased().contains("not granted") ? .orange : .secondary)
                            .lineLimit(4)
                            .textSelection(.enabled)
                    }
                }

                Spacer(minLength: 8)

                HStack(spacing: 6) {
                    Image(systemName: state.systemImage)
                        .foregroundStyle(Color(state.tint))
                    Text(actionLabel(for: state))
                        .font(.footnote.weight(.medium))
                        .foregroundStyle(.secondary)
                }
            }
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .disabled(state == .unavailable)
    }

    private func actionLabel(for state: PermissionState) -> String {
        switch state {
        case .notDetermined: return "Request"
        case .granted: return "Allowed"
        case .limited: return "Limited"
        case .denied: return "Denied"
        case .restricted: return "Restricted"
        case .unavailable: return "Unavailable"
        }
    }
}
