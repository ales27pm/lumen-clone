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

    static func begin(name: String, allowsContinuedProcessing: Bool = false) -> BackgroundRuntimeContinuation? {
        var taskID: UIBackgroundTaskIdentifier = .invalid
        var continuation: BackgroundRuntimeContinuation?
        let continuedLease = allowsContinuedProcessing
            ? BackgroundContinuedProcessingCoordinator.shared.begin(
                title: "Lumen",
                subtitle: name,
                prefersGPU: true
            )
            : nil
        taskID = UIApplication.shared.beginBackgroundTask(withName: name) {
            if let continuation {
                continuation.finish(success: false)
            } else if taskID != .invalid {
                continuedLease?.complete(success: false)
                UIApplication.shared.endBackgroundTask(taskID)
                taskID = .invalid
            }
        }
        guard taskID != .invalid else {
            continuedLease?.complete(success: false)
            return nil
        }
        let runtimeContinuation = BackgroundRuntimeContinuation(identifier: taskID, continuedProcessingLease: continuedLease)
        continuation = runtimeContinuation
        return runtimeContinuation
    }

    func end(success: Bool = true) {
        finish(success: success)
    }

    private func finish(success: Bool) {
        guard !ended, identifier != .invalid else { return }
        ended = true
        continuedProcessingLease?.complete(success: success)
        continuedProcessingLease = nil
        UIApplication.shared.endBackgroundTask(identifier)
        identifier = .invalid
    }
}
