import SwiftUI

struct RuntimeDashboardView: View {
    let runtime: RuntimeDiagnosticsSnapshot
    var body: some View {
        List {
            Section("Availability") {
                Text("FoundationModels: \(runtime.foundationModelsAvailable ? "Available" : "Unavailable")")
                Text("CoreML: \(runtime.coreMLAvailable ? "Available" : "Unavailable")")
                Text("Metal: \(runtime.metalAvailable ? "Available" : "Unavailable")")
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
