import Foundation
import OSLog
import SwiftUI

nonisolated final class AgentGroundingCancellationToken: @unchecked Sendable {
    private let lock = NSLock()
    private var cancelled = false

    func cancel() {
        lock.lock()
        cancelled = true
        lock.unlock()
    }

    var isCancelled: Bool {
        lock.lock()
        defer { lock.unlock() }
        return cancelled
    }

    func checkCancellation() throws {
        if isCancelled || Task.isCancelled { throw CancellationError() }
    }
}

nonisolated struct AgentGroundingMetrics: Sendable {
    var sectionCount: Int = 0
    var toolCount: Int = 0
    var memoryCount: Int = 0
    var promptChars: Int = 0
}

@MainActor
enum AgentGroundingInstrumentation {
    private static let log = OSLog(subsystem: "ai.lumen.app", category: "agent-grounding")

    static func mark(_ stage: String, metrics: AgentGroundingMetrics = .init(), elapsedMs: Double? = nil) {
        let snapshot = ResourceBudgetGate.diagnosticSnapshot()
        let elapsedText = elapsedMs.map { String(format: "%.1f", $0) } ?? "-"
        let message = "stage=\(stage) elapsed_ms=\(elapsedText) sections=\(metrics.sectionCount) tools=\(metrics.toolCount) memories=\(metrics.memoryCount) prompt_chars=\(metrics.promptChars) scene=\(sceneText(snapshot.scenePhase)) thermal=\(snapshot.thermalState?.rawValue ?? "unknown")"
        os_log("%{public}@", log: log, type: .info, message)
    }

    static func elapsedMs(since start: TimeInterval) -> Double {
        (ProcessInfo.processInfo.systemUptime - start) * 1000
    }

    private static func sceneText(_ phase: ScenePhase?) -> String {
        switch phase {
        case .active: return "active"
        case .inactive: return "inactive"
        case .background: return "background"
        case nil: return "unknown"
        @unknown default: return "unknown"
        }
    }
}
