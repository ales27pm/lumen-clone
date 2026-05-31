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
    private var chatOrVoiceActiveSetAt: Date?
    private var drainTask: Task<Void, Never>?
    private var delayedDrainTask: Task<Void, Never>?
    private var activeLeaseTask: Task<Void, Never>?
    private let logger = Logger(subsystem: "ai.lumen.app", category: "maintenance")
    private let foregroundGrace: TimeInterval = 3
    private let activeLeaseTimeout: TimeInterval = 10 * 60

    private init() {}

    func enqueue(_ job: DeferredMaintenanceJob) {
        jobs[job.key] = job
        scheduleDrainIfEligible()
    }

    func updateScenePhase(_ phase: ScenePhase) {
        scenePhase = phase
        if phase == .active {
            lastForegroundActivation = Date()
        } else {
            delayedDrainTask?.cancel()
            delayedDrainTask = nil
            drainTask?.cancel()
            drainTask = nil
        }
        scheduleDrainIfEligible()
    }

    func setChatOrVoiceActive(_ active: Bool) {
        chatOrVoiceActive = active
        chatOrVoiceActiveSetAt = active ? Date() : nil
        if active {
            scheduleActiveLeaseTimeout()
        } else {
            activeLeaseTask?.cancel()
            activeLeaseTask = nil
        }
        scheduleDrainIfEligible()
    }

    func canRunNow(now: Date = Date()) -> Bool {
        scenePhase == .active
        && ResourceBudgetGate.allowsMaintenance(reason: "deferred-maintenance")
        && now.timeIntervalSince(lastForegroundActivation) >= foregroundGrace
        && !isChatOrVoiceGateActive(now: now)
    }

    func pendingCount() -> Int { jobs.count }

    private func scheduleDrainIfEligible() {
        guard !jobs.isEmpty else { return }
        guard drainTask == nil else { return }
        delayedDrainTask?.cancel()
        delayedDrainTask = nil

        if canRunNow() {
            drainTask = Task { @MainActor in
                defer { drainTask = nil }
                await drainEligibleJobs()
                if !jobs.isEmpty { scheduleDrainIfEligible() }
            }
            return
        }

        scheduleDelayedForegroundDrainIfNeeded()
    }

    private func scheduleDelayedForegroundDrainIfNeeded(now: Date = Date()) {
        guard scenePhase == .active, delayedDrainTask == nil else { return }
        guard !isChatOrVoiceGateActive(now: now) else { return }
        let remainingGrace = foregroundGrace - now.timeIntervalSince(lastForegroundActivation)
        guard remainingGrace > 0 else { return }
        delayedDrainTask = Task { @MainActor in
            let nanoseconds = UInt64(max(0.05, remainingGrace) * 1_000_000_000)
            try? await Task.sleep(nanoseconds: nanoseconds)
            guard !Task.isCancelled else { return }
            delayedDrainTask = nil
            scheduleDrainIfEligible()
        }
    }

    private func scheduleActiveLeaseTimeout() {
        activeLeaseTask?.cancel()
        activeLeaseTask = Task { @MainActor in
            try? await Task.sleep(nanoseconds: UInt64(activeLeaseTimeout * 1_000_000_000))
            guard !Task.isCancelled else { return }
            if let chatOrVoiceActiveSetAt, Date().timeIntervalSince(chatOrVoiceActiveSetAt) >= activeLeaseTimeout {
                chatOrVoiceActive = false
                self.chatOrVoiceActiveSetAt = nil
                activeLeaseTask = nil
                scheduleDrainIfEligible()
            }
        }
    }

    private func drainEligibleJobs() async {
        while !Task.isCancelled, canRunNow(), let job = nextJob() {
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

    private func isChatOrVoiceGateActive(now: Date) -> Bool {
        guard chatOrVoiceActive else { return false }
        guard let chatOrVoiceActiveSetAt else { return true }
        if now.timeIntervalSince(chatOrVoiceActiveSetAt) >= activeLeaseTimeout {
            chatOrVoiceActive = false
            self.chatOrVoiceActiveSetAt = nil
            activeLeaseTask?.cancel()
            activeLeaseTask = nil
            return false
        }
        return true
    }

    #if DEBUG
    func forceForegroundGraceElapsedForTesting() {
        lastForegroundActivation = Date().addingTimeInterval(-foregroundGrace)
        scheduleDrainIfEligible()
    }

    func resetForTesting() {
        jobs.removeAll()
        scenePhase = .active
        lastForegroundActivation = Date.distantPast
        chatOrVoiceActive = false
        chatOrVoiceActiveSetAt = nil
        drainTask?.cancel()
        drainTask = nil
        delayedDrainTask?.cancel()
        delayedDrainTask = nil
        activeLeaseTask?.cancel()
        activeLeaseTask = nil
    }
    #endif
}
