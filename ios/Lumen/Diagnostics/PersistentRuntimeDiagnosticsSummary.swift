import Foundation

nonisolated enum PersistentRuntimeDiagnosticsSummaryRenderer {
    static let defaultMaxCharacters = 700

    static func render(
        state: PersistentDiagnosticState?,
        campaign: PersistentDiagnosticCampaign?,
        snapshot: DiagnosticsSnapshot?,
        pendingMemoryCaptureCount: Int? = nil,
        includeRemediation: Bool = true,
        maxCharacters: Int = defaultMaxCharacters
    ) -> String {
        var lines: [String] = []
        let status = state?.status ?? PersistentDiagnosticRunnerStatus()
        let campaignState: String
        if let campaign {
            campaignState = campaign.enabled ? (campaign.runContinuously ? "continuous" : "enabled") : "disabled"
        } else {
            campaignState = "not configured"
        }

        let runState: String
        if status.isRunning {
            runState = status.isPaused ? "paused" : "running"
        } else {
            runState = "idle"
        }
        lines.append("Lumen diagnostics: \(runState); campaign=\(campaignState); passed=\(status.passedCount), failed=\(status.failedCount), skipped=\(status.skippedCount).")

        if let snapshot {
            lines.append("Runtime: FoundationModels=\(availability(snapshot.runtime.foundationModelsAvailable)); CoreML=\(availability(snapshot.runtime.coreMLAvailable)); thermal=\(snapshot.runtime.thermalState); lowPower=\(snapshot.runtime.lowPowerModeEnabled).")
            lines.append("Privacy: localOnly=\(snapshot.privacy.localOnlyMode); network=\(snapshot.privacy.networkAccessState).")
        }

        if let pendingMemoryCaptureCount {
            if pendingMemoryCaptureCount > 0 {
                lines.append("Memory capture queue: \(pendingMemoryCaptureCount) pending local capture\(pendingMemoryCaptureCount == 1 ? "" : "s") awaiting indexing.")
                if includeRemediation {
                    lines.append("Memory remediation: open Lumen with a local embedding runtime available to promote pending captures.")
                }
            } else {
                lines.append("Memory capture queue: clear.")
            }
        }

        if let latest = latestRecord(in: state) {
            lines.append("Latest: \(latest.scenario.displayName) \(latest.status.rawValue).")
            if let summary = latest.failureSummary, !summary.isEmpty {
                lines.append("Failure: \(summary).")
            }
            if includeRemediation, let proposal = latest.remediationProposals?.first {
                lines.append("Remediation: \(proposal.title) - \(proposal.action)")
            }
        } else if let lastRemediation = status.lastRemediationSummary, !lastRemediation.isEmpty, includeRemediation {
            lines.append("Remediation: \(lastRemediation).")
        } else {
            lines.append("No persistent diagnostic runs recorded yet.")
        }

        return bounded(lines.joined(separator: "\n"), maxCharacters: maxCharacters)
    }

    private static func latestRecord(in state: PersistentDiagnosticState?) -> PersistentDiagnosticRunRecord? {
        state?.records.max { lhs, rhs in
            (lhs.finishedAt ?? lhs.startedAt) < (rhs.finishedAt ?? rhs.startedAt)
        }
    }

    private static func availability(_ value: Bool) -> String {
        value ? "available" : "unavailable"
    }

    private static func bounded(_ text: String, maxCharacters: Int) -> String {
        let limit = max(80, maxCharacters)
        guard text.count > limit else { return text }
        let suffix = "\n... truncated"
        let prefixCount = max(0, limit - suffix.count)
        return String(text.prefix(prefixCount)).trimmingCharacters(in: .whitespacesAndNewlines) + suffix
    }
}
