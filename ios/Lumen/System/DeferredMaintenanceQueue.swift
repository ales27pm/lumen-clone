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

    private var jobs: [String: DeferredMaintenanceJob] = [:]
    private var scenePhase: ScenePhase = .active
    private var lastForegroundActivation = Date.distantPast
    private var chatOrVoiceActive = false
    private var drainTask: Task<Void, Never>?
    private let logger = Logger(subsystem: "ai.lumen.app", category: "maintenance")

    private init() {}

    func enqueue(_ job: DeferredMaintenanceJob) {
        jobs[job.key] = job
        scheduleDrainIfEligible()
    }

    func updateScenePhase(_ phase: ScenePhase) {
        scenePhase = phase
        if phase == .active { lastForegroundActivation = Date() }
        scheduleDrainIfEligible()
    }

    func setChatOrVoiceActive(_ active: Bool) {
        chatOrVoiceActive = active
        scheduleDrainIfEligible()
    }

    func canRunNow(now: Date = Date()) -> Bool {
        scenePhase == .active
        && ResourceBudgetGate.allowsMaintenance(reason: "deferred-maintenance")
        && now.timeIntervalSince(lastForegroundActivation) >= 3
        && !chatOrVoiceActive
    }

    func pendingCount() -> Int { jobs.count }

    private func scheduleDrainIfEligible() {
        guard drainTask == nil, canRunNow() else { return }
        drainTask = Task { @MainActor in
            defer { drainTask = nil }
            await drainEligibleJobs()
            if !jobs.isEmpty { scheduleDrainIfEligible() }
        }
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
