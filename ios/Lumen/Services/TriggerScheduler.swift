import Foundation
import SwiftData
import BackgroundTasks
import UserNotifications
import UIKit
import OSLog

@MainActor
protocol BackgroundTaskRegistering: AnyObject {
    func register(identifier: String, handler: @escaping (BGTask) -> Void) -> Bool
}

@MainActor
final class SystemBackgroundTaskRegistrar: BackgroundTaskRegistering {
    static let shared = SystemBackgroundTaskRegistrar()

    func register(identifier: String, handler: @escaping (BGTask) -> Void) -> Bool {
        BGTaskScheduler.shared.register(forTaskWithIdentifier: identifier, using: nil, launchHandler: handler)
    }
}

@MainActor
protocol BackgroundTaskSubmitting: AnyObject {
    func submit(_ request: BGTaskRequest) throws
    func cancelAllTaskRequests()
}

@MainActor
final class SystemBackgroundTaskSubmitter: BackgroundTaskSubmitting {
    static let shared = SystemBackgroundTaskSubmitter()

    func submit(_ request: BGTaskRequest) throws {
        try BGTaskScheduler.shared.submit(request)
    }

    func cancelAllTaskRequests() {
        BGTaskScheduler.shared.cancelAllTaskRequests()
    }
}

@MainActor
protocol TriggerExecutionSafetyStoring: AnyObject {
    var autonomousExecutionSuspensionTokens: Set<String> { get set }
}

@MainActor
final class UserDefaultsTriggerExecutionSafetyStore: TriggerExecutionSafetyStoring {
    private static let suspensionKey = "trigger.autonomousExecutionSuspensionTokens"
    private let defaults: UserDefaults

    init(defaults: UserDefaults = .standard) {
        self.defaults = defaults
    }

    var autonomousExecutionSuspensionTokens: Set<String> {
        get { Set(defaults.stringArray(forKey: Self.suspensionKey) ?? []) }
        set { defaults.set(newValue.sorted(), forKey: Self.suspensionKey) }
    }
}

struct BackgroundTaskRegistrationOutcome: Equatable, Sendable {
    let identifier: String
    let succeeded: Bool
    let beforeApplicationLaunchCompletion: Bool
    let errorDomain: String?
    let errorCode: Int?
}

nonisolated enum TriggerRunOutcome: Equatable, Sendable {
    case completed(String)
    case deferred(TriggerExecutionIssue)
    case failed(TriggerExecutionIssue)
    case cancelled(TriggerExecutionIssue)
    case blocked(TriggerExecutionIssue)
    case persistenceFailed(TriggerPersistenceFailure)

    var renderedText: String {
        switch self {
        case .completed(let text):
            text
        case .deferred(let issue), .failed(let issue), .cancelled(let issue), .blocked(let issue):
            issue.message
        case .persistenceFailed(let failure):
            failure.userMessage
        }
    }
}

nonisolated enum TriggerExecutionIssueCategory: String, Hashable, Sendable {
    case transientUnavailable
    case userInteractionRequired
    case unsupportedSchedule
    case executionFailure
    case cancellation
    case persistenceFailure
}

nonisolated struct TriggerExecutionIssue: Equatable, Sendable {
    let category: TriggerExecutionIssueCategory
    let code: String
    let message: String

    init(
        category: TriggerExecutionIssueCategory = .transientUnavailable,
        code: String,
        message: String
    ) {
        self.category = category
        self.code = code
        self.message = message
    }
}

nonisolated struct TriggerScanOutcome: Equatable, Sendable {
    nonisolated enum Severity: String, Equatable, Sendable {
        case completed
        case deferred
        case failed
    }

    let completedCount: Int
    let deferredIssues: [TriggerExecutionIssue]
    let failedIssues: [TriggerExecutionIssue]

    static var completed: Self {
        Self(completedCount: 0, deferredIssues: [], failedIssues: [])
    }

    static func deferred(_ issue: TriggerExecutionIssue) -> Self {
        Self(completedCount: 0, deferredIssues: [issue], failedIssues: [])
    }

    static func failed(_ issue: TriggerExecutionIssue) -> Self {
        Self(completedCount: 0, deferredIssues: [], failedIssues: [issue])
    }

    static func aggregate(
        completedCount: Int,
        deferredIssues: [TriggerExecutionIssue],
        failedIssues: [TriggerExecutionIssue]
    ) -> Self {
        Self(
            completedCount: completedCount,
            deferredIssues: deferredIssues,
            failedIssues: failedIssues
        )
    }

    var severity: Severity {
        if !failedIssues.isEmpty { return .failed }
        if !deferredIssues.isEmpty { return .deferred }
        return .completed
    }

    var backgroundTaskSucceeded: Bool {
        severity == .completed
    }

    var issue: TriggerExecutionIssue? {
        failedIssues.first ?? deferredIssues.first
    }
}

@MainActor
final class TriggerScheduler {
    static let shared = TriggerScheduler(
        registrar: SystemBackgroundTaskRegistrar.shared,
        submitter: SystemBackgroundTaskSubmitter.shared,
        executionSafetyStore: UserDefaultsTriggerExecutionSafetyStore()
    )

    nonisolated static let refreshIdentifier = "com.27pm.lumenclone.agent.refresh"
    nonisolated static let processIdentifier = "com.27pm.lumenclone.agent.process"
    nonisolated static let continuedProcessingIdentifierPrefix = "com.27pm.lumenclone.agent.continued-processing."
    nonisolated static let continuedProcessingIdentifierPattern = "\(continuedProcessingIdentifierPrefix)*"
    nonisolated static let continuedProcessingRegistrationIdentifier = continuedProcessingIdentifierPattern
    nonisolated static let notificationCategory = "LumenAgent"

    nonisolated static func continuedProcessingIdentifier(for submissionToken: String) -> String {
        "\(continuedProcessingIdentifierPrefix)\(submissionToken)"
    }

    nonisolated static func persistenceSafetyToken(
        operation: TriggerPersistenceOperation,
        triggerID: UUID
    ) -> String {
        "\(operation.rawValue).\(triggerID.uuidString.lowercased())"
    }

    nonisolated static func triggerID(fromPersistenceSafetyToken token: String) -> UUID? {
        let components = token.split(separator: ".", omittingEmptySubsequences: false)
        guard components.count == 2,
              TriggerPersistenceOperation(rawValue: String(components[0])) != nil else {
            return nil
        }
        return UUID(uuidString: String(components[1]))
    }

    private let registrar: any BackgroundTaskRegistering
    private let submitter: any BackgroundTaskSubmitting
    private let executionSafetyStore: any TriggerExecutionSafetyStoring
    private let headlessRun: @MainActor (String, SettingsSnapshot, ModelContext, Int) async -> HeadlessAgentRunResult
    private var registeredTaskIdentifiers: Set<String> = []
    private(set) var lastRegistrationOutcomes: [BackgroundTaskRegistrationOutcome] = []
    private var isRunning = false
    var lastPermissionGranted: Bool?

    /// Optional hook so UI can observe permission changes when `requestPermission`
    /// is called from background entry points.
    var onPermissionResult: (@MainActor (Bool) -> Void)?

    private let logger = Logger(subsystem: "ai.lumen.app", category: "persistence")

    init(
        registrar: any BackgroundTaskRegistering,
        submitter: any BackgroundTaskSubmitting,
        executionSafetyStore: any TriggerExecutionSafetyStoring,
        headlessRun: @escaping @MainActor (String, SettingsSnapshot, ModelContext, Int) async -> HeadlessAgentRunResult = { prompt, settings, context, maxSteps in
            await HeadlessAgentKernelRunner.runWithOutcome(
                prompt: prompt,
                settings: settings,
                context: context,
                maxSteps: maxSteps,
                source: .trigger
            )
        }
    ) {
        self.registrar = registrar
        self.submitter = submitter
        self.executionSafetyStore = executionSafetyStore
        self.headlessRun = headlessRun
    }

    convenience init(registrar: any BackgroundTaskRegistering) {
        self.init(
            registrar: registrar,
            submitter: SystemBackgroundTaskSubmitter.shared,
            executionSafetyStore: UserDefaultsTriggerExecutionSafetyStore()
        )
    }

    var isAutonomousExecutionSuspended: Bool {
        !executionSafetyStore.autonomousExecutionSuspensionTokens.isEmpty
    }

    var autonomouslySuspendedTriggerIDs: Set<UUID> {
        Set(executionSafetyStore.autonomousExecutionSuspensionTokens.compactMap {
            Self.triggerID(fromPersistenceSafetyToken: $0)
        })
    }

    private func persist(_ context: ModelContext, operation: String, scope: String) throws {
        do { try context.save() } catch {
            logger.error("persist_failed op=\(operation, privacy: .public) scope=\(scope, privacy: .public) error_code=\(RuntimeMetricErrorSanitizer.code(for: error), privacy: .public)")
            throw error
        }
    }

    func auditPersistence(operation: String, scope: String, save: () throws -> Void) -> Bool {
        do {
            try save()
            return true
        } catch {
            logger.error("persist_failed op=\(operation, privacy: .public) scope=\(scope, privacy: .public) error_code=\(RuntimeMetricErrorSanitizer.code(for: error), privacy: .public)")
            return false
        }
    }

    @discardableResult
    func registerTasks(beforeApplicationLaunchCompletion: Bool = false) -> [BackgroundTaskRegistrationOutcome] {
        var outcomes: [BackgroundTaskRegistrationOutcome] = []
        if !registeredTaskIdentifiers.contains(Self.refreshIdentifier) {
            let succeeded = registrar.register(identifier: Self.refreshIdentifier) { task in
            guard let refresh = task as? BGAppRefreshTask else { task.setTaskCompleted(success: false); return }
            Task { @MainActor in await BackgroundOrchestrator.shared.handleAppRefresh(task: refresh) }
            }
            outcomes.append(Self.registrationOutcome(identifier: Self.refreshIdentifier, succeeded: succeeded, beforeApplicationLaunchCompletion: beforeApplicationLaunchCompletion))
            if succeeded { registeredTaskIdentifiers.insert(Self.refreshIdentifier) }
        }
        if !registeredTaskIdentifiers.contains(Self.processIdentifier) {
            let succeeded = registrar.register(identifier: Self.processIdentifier) { task in
            guard let proc = task as? BGProcessingTask else { task.setTaskCompleted(success: false); return }
            Task { @MainActor in await BackgroundOrchestrator.shared.handleProcessing(task: proc) }
            }
            outcomes.append(Self.registrationOutcome(identifier: Self.processIdentifier, succeeded: succeeded, beforeApplicationLaunchCompletion: beforeApplicationLaunchCompletion))
            if succeeded { registeredTaskIdentifiers.insert(Self.processIdentifier) }
        }
        if !outcomes.isEmpty { lastRegistrationOutcomes = outcomes }
        let center = UNUserNotificationCenter.current()
        let category = UNNotificationCategory(identifier: Self.notificationCategory, actions: [], intentIdentifiers: [], options: [])
        center.setNotificationCategories([category])
        return outcomes
    }

    private nonisolated static func registrationOutcome(
        identifier: String,
        succeeded: Bool,
        beforeApplicationLaunchCompletion: Bool
    ) -> BackgroundTaskRegistrationOutcome {
        BackgroundTaskRegistrationOutcome(
            identifier: identifier,
            succeeded: succeeded,
            beforeApplicationLaunchCompletion: beforeApplicationLaunchCompletion,
            errorDomain: succeeded ? nil : "BGTaskScheduler.register",
            errorCode: succeeded ? nil : 0
        )
    }

    @discardableResult
    func requestPermission() async -> Bool {
        let granted = (try? await UNUserNotificationCenter.current().requestAuthorization(options: [.alert, .sound, .badge])) ?? false
        lastPermissionGranted = granted
        onPermissionResult?(granted)
        return granted
    }

    @discardableResult
    func scheduleBackgroundRefresh() -> Bool {
        guard !isAutonomousExecutionSuspended else {
            logger.warning("background_schedule_blocked reason=trigger_persistence_safety_interlock")
            return false
        }
        let req = BGAppRefreshTaskRequest(identifier: Self.refreshIdentifier)
        req.earliestBeginDate = Date().addingTimeInterval(15 * 60)
        let proc = BGProcessingTaskRequest(identifier: Self.processIdentifier)
        proc.earliestBeginDate = Date().addingTimeInterval(30 * 60)
        proc.requiresNetworkConnectivity = true
        proc.requiresExternalPower = false

        do {
            try submitter.submit(req)
            try submitter.submit(proc)
            return true
        } catch {
            logger.warning("background_schedule_failed error_code=\(RuntimeMetricErrorSanitizer.code(for: error), privacy: .public)")
            return false
        }
    }

    /// A SwiftData failure can leave a user's intended pause/delete uncommitted.
    /// Persist the safety interlock outside SwiftData and cancel pending app
    /// background work so a later launch cannot silently run the stale trigger.
    func suspendAutonomousExecutionAfterPersistenceFailure(token: String) {
        var tokens = executionSafetyStore.autonomousExecutionSuspensionTokens
        tokens.insert(PersistentRuntimeDiagnosticsRedactor.safeCode(token))
        executionSafetyStore.autonomousExecutionSuspensionTokens = tokens
        submitter.cancelAllTaskRequests()
        logger.error("autonomous_trigger_execution_suspended reason=persistence_failure")
    }

    /// Only an explicit trigger mutation that has just saved successfully may
    /// clear the interlock. A successful mutation supersedes every failed
    /// operation for that same trigger, but never affects another trigger.
    func resumeAutonomousExecutionAfterSuccessfulPersistence(triggerID: UUID) {
        let tokens = Set(executionSafetyStore.autonomousExecutionSuspensionTokens.filter { token in
            Self.triggerID(fromPersistenceSafetyToken: token) != triggerID
        })
        executionSafetyStore.autonomousExecutionSuspensionTokens = tokens
        if tokens.isEmpty {
            logger.notice("autonomous_trigger_execution_resumed reason=matching_trigger_save_succeeded")
        }
    }

    /// Resolves persisted safety tokens without executing any trigger prompt.
    /// Every referenced trigger is first persisted in a paused state. Tokens for
    /// both present and already-deleted triggers are cleared only after that one
    /// save succeeds.
    @discardableResult
    func resolveAutonomousExecutionSuspension(
        container: ModelContainer,
        save: (ModelContext) throws -> Void = { try $0.save() }
    ) -> TriggerPersistenceCoordinator.Outcome {
        let triggerIDs = autonomouslySuspendedTriggerIDs
        let outcome = TriggerPersistenceCoordinator.resolveSuspensions(
            triggerIDs: triggerIDs,
            container: container,
            save: save,
            onSaved: { [self] in
                for triggerID in triggerIDs {
                    resumeAutonomousExecutionAfterSuccessfulPersistence(triggerID: triggerID)
                }
                if !isAutonomousExecutionSuspended {
                    scheduleBackgroundRefresh()
                }
            },
            onFailure: { [self] in isAutonomousExecutionSuspended }
        )
        return outcome
    }

    // MARK: - Firing

    @discardableResult
    func fireDueTriggers(context: ModelContext, appState: AppState) async -> TriggerScanOutcome {
        await fireDueTriggers(context: context, settings: appState.snapshot)
    }

    @discardableResult
    func fireDueTriggers(context: ModelContext, settings: SettingsSnapshot) async -> TriggerScanOutcome {
        guard !isAutonomousExecutionSuspended else {
            return .deferred(.init(
                category: .persistenceFailure,
                code: "trigger_persistence_safety_interlock",
                message: Self.autonomousExecutionSuspendedMessage
            ))
        }
        guard !isRunning else {
            return .deferred(.init(
                code: "trigger_scan_already_running",
                message: "Trigger scan deferred because another scan is already active."
            ))
        }
        let deadline = Date().addingTimeInterval(4.5)
        isRunning = true
        defer { isRunning = false }

        let now = Date()
        let all: [Trigger]
        do {
            all = try context.fetch(FetchDescriptor<Trigger>())
        } catch {
            return .failed(.init(
                category: .executionFailure,
                code: "trigger_fetch_failed",
                message: Self.triggerFetchFailureMessage(error: error)
            ))
        }

        var completedCount = 0
        var deferredIssues: [TriggerExecutionIssue] = []
        var failedIssues: [TriggerExecutionIssue] = []
        var mustSkipFinalPersistence = false

        scan: for t in all where !t.isPaused {
            if isAutonomousExecutionSuspended {
                deferredIssues.append(.init(
                    category: .persistenceFailure,
                    code: "trigger_persistence_safety_interlock",
                    message: Self.autonomousExecutionSuspendedMessage
                ))
                mustSkipFinalPersistence = true
                break scan
            }
            guard Date() < deadline else {
                deferredIssues.append(.init(
                    code: "trigger_scan_deadline_exceeded",
                    message: "Trigger scan deferred because its execution deadline expired."
                ))
                break scan
            }

            if let unavailability = t.kind.creationUnavailability {
                // Legacy records may predate the fail-closed creation contract.
                // Quarantine them without touching EventKit or requesting access,
                // then continue so independent due triggers are not starved.
                t.isPaused = true
                t.nextFireAt = nil
                failedIssues.append(.init(
                    category: .unsupportedSchedule,
                    code: unavailability.rawValue,
                    message: unavailability.message
                ))
                continue scan
            }

            if let next = t.nextFireAt ?? t.computeNextFire(from: now), next <= now.addingTimeInterval(30) {
                let outcome = await runTriggerWithPersistenceOutcome(
                    t,
                    context: context,
                    settings: settings,
                    notify: true
                )
                switch outcome {
                case .completed:
                    completedCount += 1
                case .deferred(let issue), .cancelled(let issue):
                    deferredIssues.append(issue)
                case .failed(let issue), .blocked(let issue):
                    failedIssues.append(issue)
                case .persistenceFailed(let failure):
                    // A generic save could commit run-state without clearing the
                    // durable retry token. Stop this scan and leave the safety
                    // interlock authoritative until explicit recovery.
                    failedIssues.append(.init(
                        category: .persistenceFailure,
                        code: failure.errorCode,
                        message: failure.userMessage
                    ))
                    mustSkipFinalPersistence = true
                    break scan
                }
            } else if t.nextFireAt == nil {
                t.nextFireAt = t.computeNextFire(from: now)
            }
        }

        if !mustSkipFinalPersistence {
            do {
                try persist(context, operation: "fireDueTriggers", scope: "Trigger")
            } catch {
                failedIssues.append(.init(
                    category: .persistenceFailure,
                    code: "trigger_scan_persistence_failed",
                    message: Self.triggerPersistenceFailureMessage(error: error)
                ))
            }
        }

        return .aggregate(
            completedCount: completedCount,
            deferredIssues: deferredIssues,
            failedIssues: failedIssues
        )
    }

    @discardableResult
    func runTrigger(_ trigger: Trigger, context: ModelContext, appState: AppState, notify: Bool) async -> String? {
        await runTriggerWithPersistenceOutcome(
            trigger,
            context: context,
            settings: appState.snapshot,
            notify: notify
        ).renderedText
    }

    @discardableResult
    func runTrigger(_ trigger: Trigger, context: ModelContext, settings: SettingsSnapshot, notify: Bool) async -> String? {
        await runTriggerWithPersistenceOutcome(
            trigger,
            context: context,
            settings: settings,
            notify: notify
        ).renderedText
    }

    func runTriggerWithPersistenceOutcome(
        _ trigger: Trigger,
        context: ModelContext,
        appState: AppState,
        notify: Bool
    ) async -> TriggerRunOutcome {
        await runTriggerWithPersistenceOutcome(
            trigger,
            context: context,
            settings: appState.snapshot,
            notify: notify
        )
    }

    func runTriggerWithPersistenceOutcome(
        _ trigger: Trigger,
        context: ModelContext,
        settings: SettingsSnapshot,
        notify: Bool
    ) async -> TriggerRunOutcome {
        if notify, isAutonomousExecutionSuspended {
            return .blocked(.init(
                category: .persistenceFailure,
                code: "trigger_persistence_safety_interlock",
                message: Self.autonomousExecutionSuspendedMessage
            ))
        }
        let result = await headlessRun(
            trigger.prompt,
            settings,
            context,
            min(settings.maxAgentSteps, 3)
        )
        switch result.status {
        case .completed:
            break
        case .deferred:
            return .deferred(.init(
                category: Self.issueCategory(for: result, fallback: .transientUnavailable),
                code: result.code ?? "headless_trigger_deferred",
                message: result.text
            ))
        case .blocked:
            let issue = TriggerExecutionIssue(
                category: Self.issueCategory(for: result, fallback: .userInteractionRequired),
                code: result.code ?? "headless_trigger_requires_user_interaction",
                message: result.text
            )
            trigger.isPaused = true
            trigger.nextFireAt = nil
            if let failure = persistRunState(
                triggerID: trigger.id,
                save: { try persist(context, operation: "blockTrigger", scope: "Trigger") }
            ) {
                return .persistenceFailed(failure)
            }
            return .blocked(issue)
        case .failed:
            return .failed(.init(
                category: Self.issueCategory(for: result, fallback: .executionFailure),
                code: result.code ?? "headless_trigger_failed",
                message: result.text
            ))
        case .cancelled:
            return .cancelled(.init(
                category: Self.issueCategory(for: result, fallback: .cancellation),
                code: result.code ?? "headless_trigger_cancelled",
                message: result.text
            ))
        }
        trigger.lastRunAt = Date()
        trigger.lastResult = result.text
        updateNextFireAfterRun(for: trigger)
        if let failure = persistRunState(
            triggerID: trigger.id,
            save: { try persist(context, operation: "runTrigger", scope: "Trigger") }
        ) {
            return .persistenceFailed(failure)
        }

        if notify {
            await postNotification(trigger: trigger, body: result.text)
        }
        return .completed(result.text)
    }

    /// Commits the state transition after the headless run has already occurred.
    /// A failure persists a safety token outside SwiftData before returning so
    /// stale next-fire metadata cannot cause an autonomous replay on relaunch.
    func persistRunState(
        triggerID: UUID,
        save: () throws -> Void
    ) -> TriggerPersistenceFailure? {
        let safetyToken = Self.persistenceSafetyToken(operation: .run, triggerID: triggerID)
        do {
            try save()
            resumeAutonomousExecutionAfterSuccessfulPersistence(triggerID: triggerID)
            return nil
        } catch {
            suspendAutonomousExecutionAfterPersistenceFailure(token: safetyToken)
            return TriggerPersistenceCoordinator.makeFailure(
                error,
                operation: .run,
                autonomousExecutionSuspended: true
            )
        }
    }

    static func triggerPersistenceFailureMessage(error: Error) -> String {
        let errorCode = RuntimeMetricErrorSanitizer.code(for: error)
        return "Trigger failed: persistence save failed (\(errorCode))."
    }

    static let autonomousExecutionSuspendedMessage = "Trigger execution is suspended until trigger changes can be saved safely."

    static func triggerFetchFailureMessage(error: Error) -> String {
        let errorCode = RuntimeMetricErrorSanitizer.code(for: error)
        return "Trigger fetch failed (\(errorCode))."
    }

    private nonisolated static func issueCategory(
        for result: HeadlessAgentRunResult,
        fallback: TriggerExecutionIssueCategory
    ) -> TriggerExecutionIssueCategory {
        guard let category = result.issueCategory else { return fallback }
        switch category {
        case .transientUnavailable:
            return .transientUnavailable
        case .userInteractionRequired:
            return .userInteractionRequired
        case .executionFailure:
            return .executionFailure
        case .cancellation:
            return .cancellation
        }
    }

    private func updateNextFireAfterRun(for trigger: Trigger) {
        switch trigger.kind {
        case .once:
            trigger.isPaused = true
            trigger.nextFireAt = nil
        default:
            trigger.nextFireAt = trigger.computeNextFire(from: Date())
        }
    }

    private func postNotification(trigger: Trigger, body: String) async {
        let content = UNMutableNotificationContent()
        content.title = trigger.title.isEmpty ? "Lumen" : trigger.title
        content.body = String(body.prefix(240))
        content.sound = .default
        content.categoryIdentifier = Self.notificationCategory
        content.userInfo = ["triggerID": trigger.id.uuidString]
        let req = UNNotificationRequest(identifier: trigger.id.uuidString, content: content, trigger: nil)
        try? await UNUserNotificationCenter.current().add(req)
    }

    // MARK: - Local scheduling (user-facing, best-effort while app is alive or via background refresh)

    @discardableResult
    func refreshNextFireTimes(context: ModelContext) -> String? {
        let all: [Trigger]
        do {
            all = try context.fetch(FetchDescriptor<Trigger>())
        } catch {
            return Self.triggerFetchFailureMessage(error: error)
        }
        let now = Date()
        for t in all {
            t.nextFireAt = t.isPaused ? nil : t.computeNextFire(from: now)
        }
        do {
            try persist(context, operation: "refreshNextFireTimes", scope: "Trigger")
        } catch {
            return Self.triggerPersistenceFailureMessage(error: error)
        }
        return nil
    }

}
