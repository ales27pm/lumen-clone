import Foundation

actor ToolMetricsRecorder {
    let store: RuntimeMetricsStore
    init(store: RuntimeMetricsStore = .shared) { self.store = store }

    @discardableResult
    func record(toolID: ToolID, status: ToolResultStatus, success: Bool, errorCode: String? = nil, memoryWarningCount: Int = 0) async -> Bool {
        let metric = RuntimeMetric(timestamp: Date(), runtimeName: "tool", taskKind: toolID, modelIDHash: nil, policySummary: status.rawValue, latencyMs: nil, success: success, errorCode: errorCode, thermalState: .from(processThermalState: ProcessInfo.processInfo.thermalState), lowPowerMode: ProcessInfo.processInfo.isLowPowerModeEnabled, memoryWarningCount: memoryWarningCount)
        do {
            try await store.appendMetric(metric)
            return true
        } catch {
            return false
        }
    }
}
