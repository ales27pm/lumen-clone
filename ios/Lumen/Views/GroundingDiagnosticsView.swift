import SwiftUI

struct GroundingDiagnosticsView: View {
    let grounding: GroundingDiagnosticsSnapshot
    var body: some View {
        List {
            Text("Context source: \(grounding.contextSource)")
            Text("Double-grounding normalized: \(grounding.doubleGroundingNormalized ? "yes" : "no")")
            Section("Degraded Reasons") { ForEach(grounding.degradedReasons, id: \.self) { Text($0) } }
        }.navigationTitle("Grounding")
    }
}
