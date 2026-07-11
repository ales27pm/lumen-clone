import BackgroundTasks
import Foundation
import OSLog

@MainActor
final class BackgroundContinuedProcessingCoordinator {
    static let shared = BackgroundContinuedProcessingCoordinator(registrar: SystemBackgroundTaskRegistrar.shared)

    private let logger = Logger(subsystem: "ai.lumen.app", category: "background")
    private var activeTasks: [String: BGTask] = [:]
    private var pendingCompletions: [String: Bool] = [:]
    private var registeredTaskIdentifiers: Set<String> = []
    private(set) var lastSubmissionStatus: String = "not_requested"
    private(set) var lastSubmittedIdentifier: String?
    private(set) var lastRegistrationIdentifier: String = TriggerScheduler.continuedProcessingRegistrationIdentifier
    private(set) var lastRegistrationErrorDomain: String?
    private(set) var lastRegistrationErrorCode: Int?
    private(set) var lastSubmitErrorDomain: String?
    private(set) var lastSubmitErrorCode: Int?
    private(set) var lastRegistrationBeforeAppLaunchCompletion: Bool?
    private var appLaunchCompleted = false
    private let registrar: any BackgroundTaskRegistering

    init(registrar: any BackgroundTaskRegistering) {
        self.registrar = registrar
    }

    var gpuSupported: Bool {
        guard #available(iOS 26.0, *) else { return false }
        return BGTaskScheduler.supportedResources.contains(.gpu)
    }

    func begin(title: String, subtitle: String, prefersGPU: Bool = true) -> BackgroundContinuedProcessingLease? {
        guard #available(iOS 26.0, *) else {
            lastSubmissionStatus = "unavailable_before_ios_26"
            return nil
        }

        let submissionToken = UUID().uuidString
        let identifier = TriggerScheduler.continuedProcessingIdentifier(for: submissionToken)
        lastSubmittedIdentifier = identifier
        guard registerHandlerIfNeeded().succeeded else { return nil }
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
            lastSubmitErrorDomain = nil
            lastSubmitErrorCode = nil
            return BackgroundContinuedProcessingLease(identifier: request.identifier, submissionToken: submissionToken)
        } catch {
            lastSubmissionStatus = "submit_failed:\(RuntimeMetricErrorSanitizer.code(for: error))"
            let nsError = error as NSError
            lastSubmitErrorDomain = nsError.domain
            lastSubmitErrorCode = nsError.code
            logger.warning("continued_processing_submit_failed error_code=\(RuntimeMetricErrorSanitizer.code(for: error), privacy: .public)")
            return nil
        }
    }

    func markApplicationLaunchCompleted() {
        appLaunchCompleted = true
    }

    @discardableResult
    func registerHandlerBeforeApplicationLaunchCompletion() -> BackgroundTaskRegistrationOutcome {
        registerHandlerIfNeeded()
    }

    @discardableResult
    private func registerHandlerIfNeeded() -> BackgroundTaskRegistrationOutcome {
        let identifier = TriggerScheduler.continuedProcessingRegistrationIdentifier
        guard #available(iOS 26.0, *) else {
            return BackgroundTaskRegistrationOutcome(
                identifier: identifier,
                succeeded: false,
                beforeApplicationLaunchCompletion: !appLaunchCompleted,
                errorDomain: "BGTaskScheduler.unavailable",
                errorCode: nil
            )
        }
        lastRegistrationIdentifier = identifier
        guard !registeredTaskIdentifiers.contains(identifier) else {
            return BackgroundTaskRegistrationOutcome(identifier: identifier, succeeded: true, beforeApplicationLaunchCompletion: !appLaunchCompleted, errorDomain: nil, errorCode: nil)
        }
        lastRegistrationBeforeAppLaunchCompletion = !appLaunchCompleted
        let registered = registrar.register(identifier: identifier) { task in
            Task { @MainActor in
                self.attach(task)
            }
        }
        if registered {
            registeredTaskIdentifiers.insert(identifier)
            lastRegistrationErrorDomain = nil
            lastRegistrationErrorCode = nil
        } else {
            lastSubmissionStatus = "registration_failed"
            lastRegistrationErrorDomain = "BGTaskScheduler.register"
            lastRegistrationErrorCode = 0
        }
        return BackgroundTaskRegistrationOutcome(
            identifier: identifier,
            succeeded: registered,
            beforeApplicationLaunchCompletion: !appLaunchCompleted,
            errorDomain: lastRegistrationErrorDomain,
            errorCode: lastRegistrationErrorCode
        )
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
