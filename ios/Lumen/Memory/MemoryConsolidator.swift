import Foundation
import SwiftData

@MainActor
enum MemoryConsolidator {
    static func consolidate(
        context: ModelContext,
        metricsStore: RuntimeMetricsStore = .shared,
        promoteQueuedCaptures: Bool = false
    ) async {
        let drain = promoteQueuedCaptures
            ? await MemoryCaptureQueue.drain(context: context, allowPromotion: true)
            : nil
        let all: [MemoryItem]
        do {
            all = try context.fetch(FetchDescriptor<MemoryItem>())
        } catch {
            try? await metricsStore.appendMetric(.init(timestamp: Date(), runtimeName: "memory", taskKind: "consolidation", modelIDHash: nil, policySummary: policySummary(deleted: 0, drain: drain), latencyMs: nil, success: false, errorCode: RuntimeMetricErrorSanitizer.code(for: error), thermalState: .from(processThermalState: ProcessInfo.processInfo.thermalState), lowPowerMode: ProcessInfo.processInfo.isLowPowerModeEnabled, memoryWarningCount: MemoryPressureMonitor.shared.warningCount))
            return
        }
        var seen = Set<String>()
        var deleted = 0
        for item in all.sorted(by: { $0.createdAt > $1.createdAt }) {
            let key = item.content.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
            if seen.contains(key), !item.isPinned { context.delete(item); deleted += 1 } else { seen.insert(key) }
        }
        do {
            try context.save()
            try? await metricsStore.appendMetric(.init(timestamp: Date(), runtimeName: "memory", taskKind: "consolidation", modelIDHash: nil, policySummary: policySummary(deleted: deleted, drain: drain), latencyMs: nil, success: true, errorCode: nil, thermalState: .from(processThermalState: ProcessInfo.processInfo.thermalState), lowPowerMode: ProcessInfo.processInfo.isLowPowerModeEnabled, memoryWarningCount: MemoryPressureMonitor.shared.warningCount))
        } catch {
            try? await metricsStore.appendMetric(.init(timestamp: Date(), runtimeName: "memory", taskKind: "consolidation", modelIDHash: nil, policySummary: policySummary(deleted: deleted, drain: drain), latencyMs: nil, success: false, errorCode: RuntimeMetricErrorSanitizer.code(for: error), thermalState: .from(processThermalState: ProcessInfo.processInfo.thermalState), lowPowerMode: ProcessInfo.processInfo.isLowPowerModeEnabled, memoryWarningCount: MemoryPressureMonitor.shared.warningCount))
        }
    }

    private static func policySummary(deleted: Int, drain: MemoryCaptureDrainResult?) -> String {
        var parts = ["dedupe deleted=\(deleted)"]
        if let drain {
            if let skippedReason = drain.skippedReason {
                parts.append("queued skipped=\(skippedReason) remaining=\(drain.remainingDescription)")
            } else {
                parts.append("queued attempted=\(drain.attempted) promoted=\(drain.promoted) remaining=\(drain.remainingDescription)")
            }
        }
        return parts.joined(separator: "; ")
    }
}
