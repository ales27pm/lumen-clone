import SwiftUI

struct PrivacyReportView: View {
    let privacy: PrivacyReportSnapshot
    var body: some View {
        List {
            Text("Network tools: \(networkToolsStatus)")
            Text("Network access: \(privacy.networkAccessState)")
            Section("Tool categories") { ForEach(privacy.recentToolCategories, id: \.self) { Text($0) } }
            Section("AppIntent limitations") { ForEach(privacy.appIntentLimitations, id: \.self) { Text($0) } }
        }.navigationTitle("Privacy")
    }

    private var networkToolsStatus: String {
        privacy.networkToolsEnabled.map { $0 ? "Enabled" : "Disabled" } ?? "Unknown"
    }
}
