import Foundation
import SwiftData
import BackgroundTasks
import UserNotifications
import EventKit
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

struct BackgroundTaskRegistrationOutcome: Equatable, Sendable {
    let identifier: String
    let succeeded: Bool
    let beforeApplicationLaunchCompletion: Bool
    let errorDomain: String?
    let errorCode: Int?
}

@MainActor
final class TriggerScheduler {
    static let shared = TriggerScheduler(registrar: SystemBackgroundTaskRegistrar.shared)

    nonisolated static let refreshIdentifier = "com.27pm.lumenclone.agent.refresh"
    nonisolated static let processIdentifier = "com.27pm.lumenclone.agent.process"
    nonisolated static let continuedProcessingIdentifierPrefix = "com.27pm.lumenclone.agent.continued-processing."
    nonisolated static let continuedProcessingIdentifierPattern = "\(continuedProcessingIdentifierPrefix)*"
    nonisolated static let continuedProcessingRegistrationIdentifier = continuedProcessingIdentifierPattern
    nonisolated static let notificationCategory = "LumenAgent"

    nonisolated static func continuedProcessingIdentifier(for submissionToken: String) -> String {
        "\(continuedProcessingIdentifierPrefix)\(submissionToken)"
    }

    private let registrar: any BackgroundTaskRegistering
    private var registeredTaskIdentifiers: Set<String> = []
    private(set) var lastRegistrationOutcomes: [BackgroundTaskRegistrationOutcome] = []
    private var isRunning = false
    var lastPermissionGranted: Bool?

    /// Optional hook so UI can observe permission changes when `requestPermission`
    /// is called from background entry points.
    var onPermissionResult: (@MainActor (Bool) -> Void)?

    private let logger = Logger(subsystem: "ai.lumen.app", category: "persistence")

    init(registrar: any BackgroundTaskRegistering) {
        self.registrar = registrar
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

    func scheduleBackgroundRefresh() {
        let req = BGAppRefreshTaskRequest(identifier: Self.refreshIdentifier)
        req.earliestBeginDate = Date().addingTimeInterval(15 * 60)
        try? BGTaskScheduler.shared.submit(req)

        let proc = BGProcessingTaskRequest(identifier: Self.processIdentifier)
        proc.earliestBeginDate = Date().addingTimeInterval(30 * 60)
        proc.requiresNetworkConnectivity = true
        proc.requiresExternalPower = false
        try? BGTaskScheduler.shared.submit(proc)
    }

    // MARK: - Firing

    @discardableResult
    func fireDueTriggers(context: ModelContext, appState: AppState) async -> String? {
        await fireDueTriggers(context: context, settings: appState.snapshot)
    }

    @discardableResult
    func fireDueTriggers(context: ModelContext, settings: SettingsSnapshot) async -> String? {
        guard !isRunning else { return nil }
        let deadline = Date().addingTimeInterval(4.5)
        isRunning = true
        defer { isRunning = false }

        let now = Date()
        let all: [Trigger]
        do {
            all = try context.fetch(FetchDescriptor<Trigger>())
        } catch {
            return Self.triggerFetchFailureMessage(error: error)
        }
        for t in all where !t.isPaused {
            guard Date() < deadline else { break }
            if let next = t.nextFireAt ?? t.computeNextFire(from: now), next <= now.addingTimeInterval(30) {
                _ = await runTrigger(t, context: context, settings: settings, notify: true)
            } else if t.nextFireAt == nil {
                t.nextFireAt = t.computeNextFire(from: now)
            }
        }
        do {
            try persist(context, operation: "fireDueTriggers", scope: "Trigger")
        } catch {
            return Self.triggerPersistenceFailureMessage(error: error)
        }
        return nil
    }

    @discardableResult
    func runTrigger(_ trigger: Trigger, context: ModelContext, appState: AppState, notify: Bool) async -> String? {
        await runTrigger(trigger, context: context, settings: appState.snapshot, notify: notify)
    }

    @discardableResult
    func runTrigger(_ trigger: Trigger, context: ModelContext, settings: SettingsSnapshot, notify: Bool) async -> String? {
        let result = await HeadlessAgentKernelRunner.run(prompt: trigger.prompt, settings: settings, context: context, maxSteps: min(settings.maxAgentSteps, 3), source: .trigger)
        trigger.lastRunAt = Date()
        trigger.lastResult = result.text
        updateNextFireAfterRun(for: trigger)
        do {
            try persist(context, operation: "runTrigger", scope: "Trigger")
        } catch {
            return Self.triggerPersistenceFailureMessage(error: error)
        }

        if notify {
            await postNotification(trigger: trigger, body: result.text)
        }
        return result.text
    }

    static func triggerPersistenceFailureMessage(error: Error) -> String {
        let errorCode = RuntimeMetricErrorSanitizer.code(for: error)
        return "Trigger failed: persistence save failed (\(errorCode))."
    }

    static func triggerFetchFailureMessage(error: Error) -> String {
        let errorCode = RuntimeMetricErrorSanitizer.code(for: error)
        return "Trigger fetch failed (\(errorCode))."
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

    // MARK: - Calendar helpers

    func minutesUntilNextEvent() async -> Int? {
        let store = EKEventStore()
        let granted = (try? await store.requestFullAccessToEvents()) ?? false
        guard granted else { return nil }
        let now = Date()
        let end = now.addingTimeInterval(24 * 3600)
        let pred = store.predicateForEvents(withStart: now, end: end, calendars: nil)
        let events = store.events(matching: pred).filter { $0.startDate > now }.sorted { $0.startDate < $1.startDate }
        guard let next = events.first else { return nil }
        return Int(next.startDate.timeIntervalSince(now) / 60)
    }
}
