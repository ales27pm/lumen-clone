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
        let continuedLease = allowsContinuedProcessing
            ? BackgroundContinuedProcessingCoordinator.shared.begin(
                title: "Lumen",
                subtitle: name,
                prefersGPU: true
            )
            : nil

        let runtimeContinuation = BackgroundRuntimeContinuation(identifier: .invalid, continuedProcessingLease: continuedLease)
        let taskID = UIApplication.shared.beginBackgroundTask(withName: name) { [runtimeContinuation] in
            MainActor.assumeIsolated {
                runtimeContinuation.finish(success: false)
            }
        }
        guard taskID != .invalid else {
            continuedLease?.complete(success: false)
            return nil
        }
        runtimeContinuation.identifier = taskID
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
