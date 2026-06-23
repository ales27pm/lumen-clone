import UIKit

@MainActor
final class BackgroundRuntimeContinuation {
    private var identifier: UIBackgroundTaskIdentifier = .invalid
    private var ended = false

    private init(identifier: UIBackgroundTaskIdentifier) {
        self.identifier = identifier
    }

    static func begin(name: String) -> BackgroundRuntimeContinuation? {
        var taskID: UIBackgroundTaskIdentifier = .invalid
        taskID = UIApplication.shared.beginBackgroundTask(withName: name) {
            guard taskID != .invalid else { return }
            UIApplication.shared.endBackgroundTask(taskID)
            taskID = .invalid
        }
        guard taskID != .invalid else { return nil }
        return BackgroundRuntimeContinuation(identifier: taskID)
    }

    func end() {
        guard !ended, identifier != .invalid else { return }
        ended = true
        UIApplication.shared.endBackgroundTask(identifier)
        identifier = .invalid
    }
}
