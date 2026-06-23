import Foundation

struct RuntimeDiagnosticsSnapshot: Sendable {
    let foundationModelsAvailable: Bool
    let foundationModelsStatus: String
    let coreMLAvailable: Bool
    let coreMLStatus: String
    let metalAvailable: Bool
    let lowPowerModeEnabled: Bool
    let thermalState: String
    let memoryWarningCount: Int
    let recentMetricSummaries: [String]
}
