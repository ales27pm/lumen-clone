import Foundation
import SwiftUI
import OSLog

nonisolated enum DeferredMaintenanceCategory: String, Sendable {
    case conversation
    case memory
    case rag
    case diagnostics
    case triggers
    case persistence
    case voice
}

struct DeferredMaintenanceJob: Sendable {
    let key: String
    let category: DeferredMaintenanceCategory
    let enqueuedAt: Date
    let staleAfter: TimeInterval
    let maxRuntime: TimeInterval
    let operation: @Sendable () async -> Void

    init(
        key: String,
        category: DeferredMaintenanceCategory,
        staleAfter: TimeInterval = 10 * 60,
        maxRuntime: TimeInterval = 5,
        operation: @escaping @Sendable () async -> Void
    ) {
        self.key = key
        self.category = category
        self.enqueuedAt = Date()
        self.staleAfter = staleAfter
        self.maxRuntime = maxRuntime
        self.operation = operation
    }
}

@MainActor
final class DeferredMaintenanceQueue {
    static let shared = DeferredMaintenanceQueue()

    private let foregroundGraceDuration: TimeInterval
    private var jobs: [String: DeferredMaintenanceJob] = [:]
    private var scenePhase: ScenePhase = .active
    private var lastForegroundActivation = Date.distantPast
    private var chatOrVoiceActive = false
    private var drainTask: Task<Void, Never>?
    private let logger = Logger(subsystem: "ai.lumen.app", category: "maintenance")

    private init(foregroundGraceDuration: TimeInterval = 3) {
        self.foregroundGraceDuration = foregroundGraceDuration
    }

    func enqueue(_ job: DeferredMaintenanceJob) {
        jobs[job.key] = job
        scheduleDrainIfEligible()
    }

    func updateScenePhase(_ phase: ScenePhase) {
        scenePhase = phase
        if phase == .active { lastForegroundActivation = Date() }
        if phase != .active { cancelScheduledDrain() }
        scheduleDrainIfEligible()
    }

    func setChatOrVoiceActive(_ active: Bool) {
        chatOrVoiceActive = active
        scheduleDrainIfEligible()
    }

    func canRunNow(now: Date = Date()) -> Bool {
        scenePhase == .active
        && ResourceBudgetGate.allowsMaintenance(reason: "deferred-maintenance")
        && now.timeIntervalSince(lastForegroundActivation) >= foregroundGraceDuration
        && !chatOrVoiceActive
    }

    func pendingCount() -> Int { jobs.count }

    func resetForTesting() {
        cancelScheduledDrain()
        jobs.removeAll()
        scenePhase = .active
        lastForegroundActivation = Date.distantPast
        chatOrVoiceActive = false
    }

    private func scheduleDrainIfEligible(now: Date = Date()) {
        guard !jobs.isEmpty else {
            cancelScheduledDrain()
            return
        }
        guard drainTask == nil else { return }
        guard scenePhase == .active, !chatOrVoiceActive else { return }
        guard ResourceBudgetGate.allowsMaintenance(reason: "deferred-maintenance") else { return }

        let graceRemaining = max(0, foregroundGraceDuration - now.timeIntervalSince(lastForegroundActivation))
        drainTask = Task { @MainActor [weak self] in
            if graceRemaining > 0 {
                let ns = UInt64(graceRemaining * 1_000_000_000)
                try? await Task.sleep(nanoseconds: ns)
            }
            guard let self else { return }
            self.drainTask = nil
            await self.drainEligibleJobs()
            if !self.jobs.isEmpty { self.scheduleDrainIfEligible() }
        }
    }

    private func cancelScheduledDrain() {
        drainTask?.cancel()
        drainTask = nil
    }

    private func drainEligibleJobs() async {
        while canRunNow(), let job = nextJob() {
            if Date().timeIntervalSince(job.enqueuedAt) > job.staleAfter {
                jobs[job.key] = nil
                continue
            }
            jobs[job.key] = nil
            await run(job)
        }
    }

    private func nextJob() -> DeferredMaintenanceJob? {
        jobs.values.sorted { $0.enqueuedAt < $1.enqueuedAt }.first
    }

    private func run(_ job: DeferredMaintenanceJob) async {
        logger.info("maintenance_start key=\(job.key, privacy: .public) category=\(job.category.rawValue, privacy: .public)")
        await withTaskGroup(of: Void.self) { group in
            group.addTask { await job.operation() }
            group.addTask {
                let ns = UInt64(max(0.1, job.maxRuntime) * 1_000_000_000)
                try? await Task.sleep(nanoseconds: ns)
            }
            await group.next()
            group.cancelAll()
        }
    }
}
