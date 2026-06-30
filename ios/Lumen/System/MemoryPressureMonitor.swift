import Foundation
import UIKit

enum MemoryPressureUnloadPolicy {
    static let slotPriority: [LumenModelSlot] = [.mimicry, .rem, .executor, .cortex, .mouth]
}

@MainActor
final class MemoryPressureMonitor {
    static let shared = MemoryPressureMonitor()
    static let modelLoadSuppressionInterval: TimeInterval = 120
    private static let duplicateWarningCoalescingInterval: TimeInterval = 1
    private(set) var warningCount: Int = 0
    private(set) var lastWarningAt: Date?
    private let metricsStore: RuntimeMetricsStore
    private let notificationCenter: NotificationCenter
    private var observerToken: NSObjectProtocol?

    init(metricsStore: RuntimeMetricsStore = .shared, notificationCenter: NotificationCenter = .default) {
        self.metricsStore = metricsStore
        self.notificationCenter = notificationCenter
        observerToken = notificationCenter.addObserver(
            forName: UIApplication.didReceiveMemoryWarningNotification,
            object: nil,
            queue: .main
        ) { [weak self] _ in
            Task { @MainActor [weak self] in
                await self?.handleWarning()
            }
        }
    }

    deinit {
        if let observerToken {
            notificationCenter.removeObserver(observerToken)
        }
    }

    /// Counts memory warnings within a time window.
    /// - Parameter interval: The time window for counting warnings. Defaults to `modelLoadSuppressionInterval`.
    /// - Returns: The count of recent memory warnings, or `0` if the time window has expired or no warnings have occurred.
    func recentWarningCount(now: Date = Date(), within interval: TimeInterval? = nil) -> Int {
        let interval = interval ?? Self.modelLoadSuppressionInterval
        guard let lastWarningAt else { return 0 }
        guard now.timeIntervalSince(lastWarningAt) < interval else {
            warningCount = 0
            self.lastWarningAt = nil
            return 0
        }
        return warningCount
    }

    #if DEBUG
    func recordWarningForTesting(count: Int = 1, at date: Date?) {
        warningCount = count
        lastWarningAt = date
    }
    #endif

    func handleWarning(now: Date = Date()) async {
        if let lastWarningAt, now.timeIntervalSince(lastWarningAt) < Self.duplicateWarningCoalescingInterval {
            return
        }
        warningCount += 1
        lastWarningAt = now
        ModelLoader.cancelActiveLoads()
        await AppLlamaService.shared.cancelActiveGeneration(reason: "memory-warning")
        let cleanup = await FleetRuntimeCleanup.unloadNonCoreChatSlotsNow()
        let metric = RuntimeMetric(
            timestamp: now,
            runtimeName: "system",
            taskKind: "memoryPressure",
            modelIDHash: nil,
            policySummary: "non-core slot cleanup unloaded=\(cleanup.unloadedSlotSummary)",
            latencyMs: nil,
            success: true,
            errorCode: nil,
            thermalState: .from(processThermalState: ProcessInfo.processInfo.thermalState),
            lowPowerMode: ProcessInfo.processInfo.isLowPowerModeEnabled,
            memoryWarningCount: warningCount
        )
        _ = try? await metricsStore.appendMetric(metric)
    }
}
