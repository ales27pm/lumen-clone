import SwiftUI

struct BackgroundDiagnosticsView: View {
    let background: BackgroundDiagnosticsSnapshot
    var body: some View {
        List {
            Section("Permitted Identifiers") { ForEach(background.permittedIdentifiers, id: \.self) { Text($0) } }
            Section("Runtime Capabilities") {
                Text("Background GPU: \(background.backgroundGPUSupported ? "Supported" : "Unavailable")")
                Text("Continued processing: \(background.continuedProcessingStatus)")
                Text("Available memory: \(Self.formatBytes(background.availableMemoryBytes))")
                Text("EnergyKit: \(background.energyKit.status)")
                if let venueCount = background.energyKit.venueCount {
                    Text("Energy venues: \(venueCount)")
                }
                Text("StoreKit: \(background.storeKit.status)")
                Text("StoreKit environment: \(background.storeKit.environment)")
            }
            Section("Entitlements") {
                ForEach(0..<background.entitlementStates.count, id: \.self) { index in
                    EntitlementStateRow(state: background.entitlementStates[index])
                }
            }
            Section("Entitlement Warnings") { ForEach(background.entitlementWarnings, id: \.self) { Text($0).foregroundStyle(.orange) } }
        }.navigationTitle("Background")
    }

    private static func formatBytes(_ bytes: UInt64) -> String {
        ByteCountFormatter.string(fromByteCount: Int64(bytes), countStyle: .memory)
    }
}

private struct EntitlementStateRow: View {
    let state: RuntimeEntitlementState

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(state.displayName)
            Text("\(state.key): \(state.valueDescription)")
                .font(.caption.monospaced())
                .foregroundStyle(state.enabled ? Color.secondary : Color.orange)
        }
    }
}
