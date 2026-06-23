import UIKit

@MainActor
final class BackgroundRuntimeContinuation {
    private var identifier: UIBackgroundTaskIdentifier = .invalid
    private var continuedProcessingLease: BackgroundContinuedProcessingLease?
    private var ended = false

    private init(identifier: UIBackgroundTaskIdentifier, continuedProcessingLease: BackgroundContinuedProcessingLease?) {
        self.identifier = identifier
        self.continuedProcessingLease = continuedProcessingLease
    }

    static func begin(name: String) -> BackgroundRuntimeContinuation? {
        var taskID: UIBackgroundTaskIdentifier = .invalid
        let continuedLease = BackgroundContinuedProcessingCoordinator.shared.begin(
            title: "Lumen",
            subtitle: name,
            prefersGPU: true
        )
        taskID = UIApplication.shared.beginBackgroundTask(withName: name) {
            guard taskID != .invalid else { return }
            UIApplication.shared.endBackgroundTask(taskID)
            taskID = .invalid
            continuedLease?.complete(success: false)
        }
        guard taskID != .invalid else { return nil }
        return BackgroundRuntimeContinuation(identifier: taskID, continuedProcessingLease: continuedLease)
    }

    func end() {
        guard !ended, identifier != .invalid else { return }
        ended = true
        continuedProcessingLease?.complete(success: true)
        continuedProcessingLease = nil
        UIApplication.shared.endBackgroundTask(identifier)
        identifier = .invalid
    }
}
