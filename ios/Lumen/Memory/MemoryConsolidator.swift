import Foundation
import SwiftData

@MainActor
enum MemoryConsolidator {
    static func consolidate(context: ModelContext, metricsStore: RuntimeMetricsStore = .shared) async {
        let all = (try? context.fetch(FetchDescriptor<MemoryItem>())) ?? []
        var seen = Set<String>()
        var deleted = 0
        for item in all.sorted(by: { $0.createdAt > $1.createdAt }) {
            let key = item.content.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
            if seen.contains(key), !item.isPinned { context.delete(item); deleted += 1 } else { seen.insert(key) }
        }
        do {
            try context.save()
            try? await metricsStore.appendMetric(.init(timestamp: Date(), runtimeName: "memory", taskKind: "consolidation", modelIDHash: nil, policySummary: "dedupe deleted=\(deleted)", latencyMs: nil, success: true, errorCode: nil, thermalState: .from(processThermalState: ProcessInfo.processInfo.thermalState), lowPowerMode: ProcessInfo.processInfo.isLowPowerModeEnabled, memoryWarningCount: MemoryPressureMonitor.shared.warningCount))
        } catch {
            try? await metricsStore.appendMetric(.init(timestamp: Date(), runtimeName: "memory", taskKind: "consolidation", modelIDHash: nil, policySummary: "dedupe", latencyMs: nil, success: false, errorCode: RuntimeMetricErrorSanitizer.code(for: error), thermalState: .from(processThermalState: ProcessInfo.processInfo.thermalState), lowPowerMode: ProcessInfo.processInfo.isLowPowerModeEnabled, memoryWarningCount: MemoryPressureMonitor.shared.warningCount))
        }
    }
}
