import Foundation
import SwiftUI

actor LifecycleProbeController {
    struct ProbeResult: Sendable, Equatable {
        let record: PersistentDiagnosticRunRecord
        let shouldPersist: Bool
    }

    private var activeRecord: PersistentDiagnosticRunRecord?
    private var transitions: [String] = []
    private var sawInactiveOrBackground = false

    func arm(record: PersistentDiagnosticRunRecord) -> PersistentDiagnosticRunRecord {
        activeRecord = record
        transitions = []
        sawInactiveOrBackground = false
        var armed = record
        armed.events.append(PersistentDiagnosticEvent(code: "lifecycle_probe_armed", message: "Lifecycle probe armed"))
        activeRecord = armed
        return armed
    }

    func record(phase: ScenePhase) -> ProbeResult? {
        guard var record = activeRecord else { return nil }
        let name = Self.phaseName(phase)
        transitions.append(name)
        record.events.append(PersistentDiagnosticEvent(code: "lifecycle_transition", message: "Lifecycle transition", values: ["phase": name]))
        if phase == .inactive || phase == .background {
            sawInactiveOrBackground = true
            record.metrics.appBecameInactiveOrBackgroundDuringRun = true
            activeRecord = record
            return nil
        }
        guard phase == .active else {
            activeRecord = record
            return nil
        }
        record.finishedAt = Date()
        record.metrics.cancellationReason = AppCancellationBus.shared.lastCancellationReason
        record.events.append(PersistentDiagnosticEvent(code: sawInactiveOrBackground ? "lifecycle_probe_passed" : "lifecycle_probe_skipped", message: "Lifecycle probe finalized", values: ["transitions": transitions.joined(separator: ",")]))
        record.status = sawInactiveOrBackground ? .passed : .skipped
        if !sawInactiveOrBackground { record.failureSummary = nil }
        activeRecord = nil
        return ProbeResult(record: record, shouldPersist: true)
    }

    func finalizeWithoutTransition() -> PersistentDiagnosticRunRecord? {
        guard var record = activeRecord else { return nil }
        record.finishedAt = Date()
        record.status = sawInactiveOrBackground ? .passed : .skipped
        record.metrics.appBecameInactiveOrBackgroundDuringRun = sawInactiveOrBackground
        record.events.append(PersistentDiagnosticEvent(code: sawInactiveOrBackground ? "lifecycle_probe_passed" : "lifecycle_probe_skipped", message: "Lifecycle probe finalized", values: ["transitions": transitions.joined(separator: ",")]))
        activeRecord = nil
        return record
    }

    private static func phaseName(_ phase: ScenePhase) -> String {
        switch phase {
        case .active: return "active"
        case .inactive: return "inactive"
        case .background: return "background"
        @unknown default: return "unknown"
        }
    }
}
