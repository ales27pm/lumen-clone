import Foundation
import Combine

@MainActor
final class ThermalStateMonitor: ObservableObject {
    @Published private(set) var currentState: DeviceThermalState
    private let notificationCenter: NotificationCenter
    private var observerToken: NSObjectProtocol?
    private var continuation: AsyncStream<DeviceThermalState>.Continuation?
    let updates: AsyncStream<DeviceThermalState>

    init(notificationCenter: NotificationCenter = .default, processInfo: ProcessInfo = .processInfo) {
        self.notificationCenter = notificationCenter
        self.currentState = .from(processThermalState: processInfo.thermalState)
        var streamContinuation: AsyncStream<DeviceThermalState>.Continuation?
        self.updates = AsyncStream { streamContinuation = $0 }
        self.continuation = streamContinuation

        observerToken = notificationCenter.addObserver(
            forName: ProcessInfo.thermalStateDidChangeNotification,
            object: nil,
            queue: .main
        ) { [weak self] _ in
            guard let self else { return }
            let state = DeviceThermalState.from(processThermalState: ProcessInfo.processInfo.thermalState)
            self.currentState = state
            self.continuation?.yield(state)
        }
    }

    deinit {
        if let observerToken {
            notificationCenter.removeObserver(observerToken)
        }
    }
}
