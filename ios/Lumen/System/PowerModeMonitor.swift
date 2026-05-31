import Foundation
import Combine

struct PowerModeSnapshot: Sendable, Equatable {
    let lowPowerModeEnabled: Bool
}

@MainActor
final class PowerModeMonitor: ObservableObject {
    @Published private(set) var snapshot: PowerModeSnapshot
    private let notificationCenter: NotificationCenter
    private var observerToken: NSObjectProtocol?

    init(notificationCenter: NotificationCenter = .default, processInfo: ProcessInfo = .processInfo) {
        self.notificationCenter = notificationCenter
        self.snapshot = PowerModeSnapshot(lowPowerModeEnabled: processInfo.isLowPowerModeEnabled)

        observerToken = notificationCenter.addObserver(
            forName: Notification.Name.NSProcessInfoPowerStateDidChange,
            object: nil,
            queue: .main
        ) { [weak self] _ in
            self?.snapshot = PowerModeSnapshot(lowPowerModeEnabled: ProcessInfo.processInfo.isLowPowerModeEnabled)
        }
    }

    deinit {
        if let observerToken {
            notificationCenter.removeObserver(observerToken)
        }
    }
}
