import Foundation
import SwiftData
import BackgroundTasks

@MainActor
final class BackgroundOrchestrator {
    static let shared = BackgroundOrchestrator()

    private let lease = BackgroundExecutionLease()
    private let metrics = RuntimeMetricsStore.shared

    func register() {
        TriggerScheduler.shared.registerTasks()
    }

    func schedule() {
        TriggerScheduler.shared.scheduleBackgroundRefresh()
    }

    func handleAppRefresh() async {
        guard let container = SharedContainer.shared else { return }
        let context = ModelContext(container)
        await runTriggerScan(context: context)
    }

    func handleProcessing() async {
        guard let container = SharedContainer.shared else { return }
        let context = ModelContext(container)
        await runTriggerScan(context: context)
        await runMemoryConsolidationIfAllowed()
        await runRAGMaintenanceIfAllowed()
        await runModelHousekeepingIfAllowed()
    }

    func runTriggerScan(context: ModelContext) async {
        let acquired = await lease.acquire(category: "triggerScan", reason: "background trigger scan")
        guard acquired else { return }
        await TriggerScheduler.shared.fireDueTriggers(context: context, settings: SettingsSnapshot.loadFromDisk())
        try? await metrics.appendMetric(RuntimeMetric(timestamp: Date(), runtimeName: "background", taskKind: "triggerScan", modelIDHash: nil, policySummary: "trigger scheduler fireDueTriggers", latencyMs: nil, success: true, errorCode: nil, thermalState: .from(processThermalState: ProcessInfo.processInfo.thermalState), lowPowerMode: ProcessInfo.processInfo.isLowPowerModeEnabled, memoryWarningCount: MemoryPressureMonitor.shared.warningCount))
        await lease.release(category: "triggerScan")
    }

    func runMemoryConsolidationIfAllowed() async {
        guard let container = SharedContainer.shared else { return }
        let context = ModelContext(container)
        await MemoryConsolidator.consolidate(context: context, metricsStore: metrics)
    }

    func runRAGMaintenanceIfAllowed() async {
        guard let container = SharedContainer.shared else { return }
        let context = ModelContext(container)
        let result = await RAGEngine().maintenance(context: context)
        try? await metrics.appendMetric(RuntimeMetric(timestamp: Date(), runtimeName: "background", taskKind: "ragMaintenance", modelIDHash: nil, policySummary: result.metricSummary, latencyMs: nil, success: result.success, errorCode: result.success ? nil : "maintenance_failed", thermalState: .from(processThermalState: ProcessInfo.processInfo.thermalState), lowPowerMode: ProcessInfo.processInfo.isLowPowerModeEnabled, memoryWarningCount: MemoryPressureMonitor.shared.warningCount))
    }

    func runModelHousekeepingIfAllowed() async {
        try? await metrics.appendMetric(RuntimeMetric(timestamp: Date(), runtimeName: "background", taskKind: "modelHousekeeping", modelIDHash: nil, policySummary: "not available in current runtime", latencyMs: nil, success: false, errorCode: "not_available", thermalState: .from(processThermalState: ProcessInfo.processInfo.thermalState), lowPowerMode: ProcessInfo.processInfo.isLowPowerModeEnabled, memoryWarningCount: 0))
    }
}
