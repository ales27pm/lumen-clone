import SwiftUI

struct PrivacyReportView: View {
    let privacy: PrivacyReportSnapshot
    var body: some View {
        List {
            Text("Local-only mode: \(privacy.localOnlyMode ? "On" : "Off")")
            Text("Network access: \(privacy.networkAccessState)")
            Section("Tool categories") { ForEach(privacy.recentToolCategories, id: \.self) { Text($0) } }
            Section("AppIntent limitations") { ForEach(privacy.appIntentLimitations, id: \.self) { Text($0) } }
        }.navigationTitle("Privacy")
    }
}
