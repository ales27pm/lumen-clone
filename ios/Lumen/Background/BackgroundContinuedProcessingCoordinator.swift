import BackgroundTasks
import Foundation
import OSLog

@MainActor
final class BackgroundContinuedProcessingCoordinator {
    static let shared = BackgroundContinuedProcessingCoordinator()

    private let logger = Logger(subsystem: "ai.lumen.app", category: "background")
    private var activeTasks: [String: BGTask] = [:]
    private var pendingCompletions: [String: Bool] = [:]
    private var registered = false
    private(set) var lastSubmissionStatus: String = "not_requested"

    private init() {}

    var gpuSupported: Bool {
        guard #available(iOS 26.0, *) else { return false }
        return BGTaskScheduler.supportedResources.contains(.gpu)
    }

    func registerIfAvailable() {
        guard #available(iOS 26.0, *), !registered else { return }
        registered = BGTaskScheduler.shared.register(forTaskWithIdentifier: TriggerScheduler.continuedProcessingIdentifier, using: nil) { task in
            Task { @MainActor in
                self.attach(task)
            }
        }
        if !registered {
            lastSubmissionStatus = "registration_failed"
        }
    }

    func begin(title: String, subtitle: String, prefersGPU: Bool = true) -> BackgroundContinuedProcessingLease? {
        guard #available(iOS 26.0, *) else {
            lastSubmissionStatus = "unavailable_before_ios_26"
            return nil
        }
        registerIfAvailable()
        guard registered else { return nil }

        let submissionToken = UUID().uuidString
        let identifier = TriggerScheduler.continuedProcessingIdentifier(for: submissionToken)
        let request = BGContinuedProcessingTaskRequest(
            identifier: identifier,
            title: title,
            subtitle: subtitle
        )
        request.strategy = .queue
        if prefersGPU, BGTaskScheduler.supportedResources.contains(.gpu) {
            request.requiredResources = .gpu
        }

        do {
            try BGTaskScheduler.shared.submit(request)
            lastSubmissionStatus = request.requiredResources.contains(.gpu) ? "submitted_gpu" : "submitted_default"
            return BackgroundContinuedProcessingLease(identifier: request.identifier, submissionToken: submissionToken)
        } catch {
            lastSubmissionStatus = "submit_failed:\(RuntimeMetricErrorSanitizer.code(for: error))"
            logger.warning("continued_processing_submit_failed error=\(String(describing: error), privacy: .public)")
            return nil
        }
    }

    func complete(identifier: String, submissionToken: String, success: Bool) {
        guard identifier == TriggerScheduler.continuedProcessingIdentifier(for: submissionToken) else { return }
        if let task = activeTasks.removeValue(forKey: identifier) {
            if #available(iOS 26.0, *), let continuedTask = task as? BGContinuedProcessingTask {
                continuedTask.progress.completedUnitCount = continuedTask.progress.totalUnitCount
            }
            task.setTaskCompleted(success: success)
            pendingCompletions[identifier] = nil
            lastSubmissionStatus = success ? "completed" : "completed_unsuccessful"
        } else {
            pendingCompletions[identifier] = success
            BGTaskScheduler.shared.cancel(taskRequestWithIdentifier: identifier)
            lastSubmissionStatus = success ? "cancelled_before_launch_after_success" : "cancelled_before_launch_after_failure"
        }
    }

    private func attach(_ task: BGTask) {
        guard task.identifier.hasPrefix(TriggerScheduler.continuedProcessingIdentifierPrefix) else {
            task.setTaskCompleted(success: false)
            return
        }
        activeTasks[task.identifier] = task
        task.expirationHandler = { [weak self] in
            Task { @MainActor in
                self?.completeAttachedTask(identifier: task.identifier, success: false)
            }
        }
        if #available(iOS 26.0, *), let continuedTask = task as? BGContinuedProcessingTask {
            continuedTask.progress.totalUnitCount = 100
            continuedTask.progress.completedUnitCount = 5
            continuedTask.updateTitle("Lumen", subtitle: "Continuing local model work")
        }
        if let pendingCompletion = pendingCompletions.removeValue(forKey: task.identifier) {
            completeAttachedTask(identifier: task.identifier, success: pendingCompletion)
        }
    }

    private func completeAttachedTask(identifier: String, success: Bool) {
        guard let task = activeTasks.removeValue(forKey: identifier) else { return }
        if #available(iOS 26.0, *), let continuedTask = task as? BGContinuedProcessingTask {
            continuedTask.progress.completedUnitCount = continuedTask.progress.totalUnitCount
        }
        task.setTaskCompleted(success: success)
        pendingCompletions[identifier] = nil
        lastSubmissionStatus = success ? "completed" : "completed_unsuccessful"
    }
}

struct BackgroundContinuedProcessingLease {
    let identifier: String
    let submissionToken: String

    func complete(success: Bool = true) {
        Task { @MainActor in
            BackgroundContinuedProcessingCoordinator.shared.complete(identifier: identifier, submissionToken: submissionToken, success: success)
        }
    }
}
