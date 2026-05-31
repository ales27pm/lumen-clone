import SwiftUI

struct BackgroundDiagnosticsView: View {
    let background: BackgroundDiagnosticsSnapshot
    var body: some View {
        List {
            Section("Permitted Identifiers") { ForEach(background.permittedIdentifiers, id: \.self) { Text($0) } }
            Section("Entitlement Warnings") { ForEach(background.entitlementWarnings, id: \.self) { Text($0).foregroundStyle(.orange) } }
        }.navigationTitle("Background")
    }
}
