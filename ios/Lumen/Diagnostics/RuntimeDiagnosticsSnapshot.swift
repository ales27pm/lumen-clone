import Foundation

struct RuntimeDiagnosticsSnapshot: Sendable {
    let foundationModelsAvailable: Bool
    let coreMLAvailable: Bool
    let metalAvailable: Bool
    let lowPowerModeEnabled: Bool
    let thermalState: String
    let memoryWarningCount: Int
    let recentMetricSummaries: [String]
}
