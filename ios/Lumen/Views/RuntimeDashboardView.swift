import SwiftUI

struct RuntimeDashboardView: View {
    let runtime: RuntimeDiagnosticsSnapshot
    var body: some View {
        List {
            Section("Availability") {
                Text("FoundationModels: \(runtime.foundationModelsAvailable ? "Available" : "Unavailable") - \(runtime.foundationModelsStatus)")
                Text("CoreML embeddings: \(runtime.coreMLAvailable ? "Available" : "Unavailable") - \(runtime.coreMLStatus)")
                Text("Metal: \(runtime.metalAvailable ? "Available" : "Unavailable")")
            }
            Section("Runtime Capabilities") {
                ForEach(runtime.runtimeCapabilityRows) { row in
                    VStack(alignment: .leading, spacing: 4) {
                        Text(row.kind.rawValue)
                            .font(.subheadline.weight(.semibold))
                        Text("generation: \(row.generationSelectable ? "selectable" : row.generationSupported ? "supported but unavailable" : "not supported")")
                        Text("embeddings: \(row.embeddingSelectable ? "selectable" : row.embeddingSupported ? "supported but unavailable" : "not supported")")
                        Text(row.status)
                            .foregroundStyle(.secondary)
                    }
                    .font(.caption)
                }
            }
            Section("Policy") {
                Text("Low Power: \(runtime.lowPowerModeEnabled ? "On" : "Off")")
                Text("Thermal: \(runtime.thermalState)")
                Text("Memory warnings: \(runtime.memoryWarningCount)")
            }
            Section("Recent Metrics") {
                ForEach(runtime.recentMetricSummaries, id: \.self) { Text($0).font(.caption.monospaced()) }
            }
        }.navigationTitle("Runtime")
    }
}
