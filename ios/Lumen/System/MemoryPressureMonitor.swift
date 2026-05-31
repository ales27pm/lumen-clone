import Foundation
import UIKit

enum MemoryPressureUnloadPolicy {
    static let slotPriority: [LumenModelSlot] = [.mimicry, .rem, .executor, .cortex, .mouth]
}

@MainActor
final class MemoryPressureMonitor {
    static let shared = MemoryPressureMonitor()
    static let modelLoadSuppressionInterval: TimeInterval = 120
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
            Task { @MainActor in
                await self?.handleWarning()
            }
        }
    }

    deinit {
        if let observerToken {
            notificationCenter.removeObserver(observerToken)
        }
    }

    func recentWarningCount(now: Date = Date(), within interval: TimeInterval = MemoryPressureMonitor.modelLoadSuppressionInterval) -> Int {
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

    func handleWarning() async {
        warningCount += 1
        lastWarningAt = Date()
        ModelLoader.cancelActiveLoads()
        FleetRuntimeCleanup.unloadOptionalChatSlots()
        let metric = RuntimeMetric(
            timestamp: Date(),
            runtimeName: "system",
            taskKind: "memoryPressure",
            modelIDHash: nil,
            policySummary: "optional slot cleanup",
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
