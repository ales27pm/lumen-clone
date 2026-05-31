import Foundation
import SwiftData
import BackgroundTasks
import UserNotifications
import EventKit
import UIKit
import OSLog

@MainActor
final class TriggerScheduler {
    static let shared = TriggerScheduler()

    static let refreshIdentifier = "com.27pm.lumen.agent.refresh"
    static let processIdentifier = "com.27pm.lumen.agent.process"
    static let notificationCategory = "LumenAgent"

    private var registered = false
    private var isRunning = false
    var lastPermissionGranted: Bool?

    /// Optional hook so UI can observe permission changes when `requestPermission`
    /// is called from background entry points.
    var onPermissionResult: (@MainActor (Bool) -> Void)?

    private let logger = Logger(subsystem: "ai.lumen.app", category: "persistence")

    private func persist(_ context: ModelContext, operation: String, scope: String) throws {
        do { try context.save() } catch {
            logger.error("persist_failed op=\(operation, privacy: .public) scope=\(scope, privacy: .public) error=\(String(describing: error), privacy: .public)")
            throw error
        }
    }

    func auditPersistence(operation: String, scope: String, save: () throws -> Void) -> Bool {
        do {
            try save()
            return true
        } catch {
            logger.error("persist_failed op=\(operation, privacy: .public) scope=\(scope, privacy: .public) error=\(String(describing: error), privacy: .public)")
            return false
        }
    }

    func registerTasks() {
        guard !registered else { return }
        registered = true
        BGTaskScheduler.shared.register(forTaskWithIdentifier: Self.refreshIdentifier, using: nil) { task in
            guard let refresh = task as? BGAppRefreshTask else { task.setTaskCompleted(success: false); return }
            Task { @MainActor in await self.handleRefresh(task: refresh) }
        }
        BGTaskScheduler.shared.register(forTaskWithIdentifier: Self.processIdentifier, using: nil) { task in
            guard let proc = task as? BGProcessingTask else { task.setTaskCompleted(success: false); return }
            Task { @MainActor in await self.handleRefresh(task: proc) }
        }
        let center = UNUserNotificationCenter.current()
        let category = UNNotificationCategory(identifier: Self.notificationCategory, actions: [], intentIdentifiers: [], options: [])
        center.setNotificationCategories([category])
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

    private func handleRefresh(task: BGTask) async {
        scheduleBackgroundRefresh()
        task.expirationHandler = { task.setTaskCompleted(success: false) }
        guard let container = SharedContainer.shared else { task.setTaskCompleted(success: true); return }
        let context = ModelContext(container)
        // Load settings fresh from disk so the background task never runs against
        // stale in-memory state from a previous foreground session.
        let snapshot = SettingsSnapshot.loadFromDisk()
        await fireDueTriggers(context: context, settings: snapshot)
        task.setTaskCompleted(success: true)
    }

    // MARK: - Firing

    func fireDueTriggers(context: ModelContext, appState: AppState) async {
        await fireDueTriggers(context: context, settings: appState.snapshot)
    }

    func fireDueTriggers(context: ModelContext, settings: SettingsSnapshot) async {
        guard !isRunning else { return }
        isRunning = true
        defer { isRunning = false }

        let now = Date()
        guard let all = try? context.fetch(FetchDescriptor<Trigger>()) else { return }
        for t in all where !t.isPaused {
            if let next = t.nextFireAt ?? t.computeNextFire(from: now), next <= now.addingTimeInterval(30) {
                _ = await runTrigger(t, context: context, settings: settings, notify: true)
            } else if t.nextFireAt == nil {
                t.nextFireAt = t.computeNextFire(from: now)
            }
        }
        do { try persist(context, operation: "fireDueTriggers", scope: "Trigger") } catch { return }
    }

    @discardableResult
    func runTrigger(_ trigger: Trigger, context: ModelContext, appState: AppState, notify: Bool) async -> String? {
        await runTrigger(trigger, context: context, settings: appState.snapshot, notify: notify)
    }

    @discardableResult
    func runTrigger(_ trigger: Trigger, context: ModelContext, settings: SettingsSnapshot, notify: Bool) async -> String? {
        let result = await AgentRunner.runHeadless(prompt: trigger.prompt, settings: settings, context: context, maxSteps: min(settings.maxAgentSteps, 3))
        trigger.lastRunAt = Date()
        trigger.lastResult = result.text
        switch trigger.kind {
        case .once:
            trigger.isPaused = true
            trigger.nextFireAt = nil
        default:
            trigger.nextFireAt = trigger.computeNextFire(from: Date())
        }
        do { try persist(context, operation: "runTrigger", scope: "Trigger") } catch { return nil }

        if notify {
            await postNotification(trigger: trigger, body: result.text)
        }
        return result.text
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

    func refreshNextFireTimes(context: ModelContext) {
        guard let all = try? context.fetch(FetchDescriptor<Trigger>()) else { return }
        let now = Date()
        for t in all {
            t.nextFireAt = t.isPaused ? nil : t.computeNextFire(from: now)
        }
do { try persist(context, operation: "refreshNextFireTimes", scope: "Trigger") } catch { return }
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
