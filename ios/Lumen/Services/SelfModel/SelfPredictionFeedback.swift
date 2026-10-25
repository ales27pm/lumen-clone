import Foundation

/// A bounded projection of host-recorded tool results. It does not estimate a
/// capability, diagnose a cause, or replace execution-time permission checks.
nonisolated enum SelfPredictionFeedback {
    static let maximumAge: TimeInterval = 24 * 60 * 60
    static let maximumTools = 3
    static let maximumOutcomesPerTool = 8

    nonisolated enum State: Sendable, Equatable {
        case available(Snapshot)
        case unavailable
    }

    nonisolated struct Snapshot: Sendable, Equatable {
        let runtime: String
        let generatedAt: Date
        let entries: [ToolSummary]
        /// Recent predictions without an outcome; this does not mean running.
        let pendingCount: Int
        let cancelledCount: Int
        let staleCount: Int
        let futureCount: Int
        let omittedToolCount: Int
    }

    nonisolated struct ToolSummary: Sendable, Equatable {
        let toolID: String
        /// Counts describe at most the latest eight eligible results per tool.
        let completedCount: Int
        let successCount: Int
        let lastStatus: SelfPredictionLedger.OutcomeStatus
        let lastObservedAt: Date
        let lastSeq: Int
        let transition: StatusTransition?
    }

    nonisolated struct StatusTransition: Sendable, Equatable {
        let from: SelfPredictionLedger.OutcomeStatus
        let to: SelfPredictionLedger.OutcomeStatus
    }

    /// Input is joined by invocation ID inside the ledger before projection.
    nonisolated struct Record: Sendable {
        let runtime: String
        let toolID: String
        let predictionDate: Date
        let outcome: Outcome?
    }

    nonisolated struct Outcome: Sendable {
        let seq: Int
        let timestamp: Date
        let status: SelfPredictionLedger.OutcomeStatus
        let cancelled: Bool
    }

    static func project(
        records: [Record], runtime: String, toolIDs: Set<String>, now: Date
    ) -> Snapshot {
        var pendingCount = 0
        var cancelledCount = 0
        var staleCount = 0
        var futureCount = 0
        var byTool: [String: [Outcome]] = [:]

        for record in records where record.runtime == runtime && toolIDs.contains(record.toolID) {
            let timestamp = record.outcome?.timestamp ?? record.predictionDate
            let age = now.timeIntervalSince(timestamp)
            // Freshness is assessed before outcome kind so exclusion counts are
            // disjoint. Clock rollback cannot turn a future timestamp into proof.
            guard age.isFinite, age <= maximumAge else {
                staleCount += 1
                continue
            }
            guard age >= 0 else {
                futureCount += 1
                continue
            }
            guard let outcome = record.outcome else {
                pendingCount += 1
                continue
            }
            guard !outcome.cancelled else {
                cancelledCount += 1
                continue
            }
            byTool[record.toolID, default: []].append(outcome)
        }

        let summaries = byTool.map { (toolID, outcomes) -> ToolSummary in
            let window = outcomes.sorted { $0.seq < $1.seq }.suffix(maximumOutcomesPerTool)
            // Each group was created by an eligible outcome, so it is non-empty.
            let last = window[window.index(before: window.endIndex)]
            let previous = window.count > 1 ? window[window.index(window.endIndex, offsetBy: -2)] : nil
            let transition = previous.flatMap { previous in
                previous.status == last.status ? nil : StatusTransition(from: previous.status, to: last.status)
            }
            return ToolSummary(
                toolID: toolID, completedCount: window.count,
                successCount: window.filter { $0.status == .success }.count,
                lastStatus: last.status, lastObservedAt: last.timestamp,
                lastSeq: last.seq, transition: transition
            )
        }.sorted {
            $0.lastSeq == $1.lastSeq ? $0.toolID < $1.toolID : $0.lastSeq > $1.lastSeq
        }

        return Snapshot(
            runtime: runtime, generatedAt: now, entries: Array(summaries.prefix(maximumTools)),
            pendingCount: pendingCount, cancelledCount: cancelledCount,
            staleCount: staleCount, futureCount: futureCount,
            omittedToolCount: max(0, summaries.count - maximumTools)
        )
    }

    /// The entire returned value is valid JSON. Whole tool entries are omitted
    /// when necessary; facts, identifiers and status transitions are never cut.
    static func render(_ state: State, maxCharacters: Int = 640) -> String {
        guard maxCharacters > 0 else { return "" }
        switch state {
        case .unavailable:
            return encode(Rendered(state: "unavailable"), maxCharacters: maxCharacters) ?? ""
        case .available(let snapshot):
            guard snapshot.generatedAt.timeIntervalSince1970.isFinite,
                  snapshot.entries.allSatisfy({ entry in
                      let age = snapshot.generatedAt.timeIntervalSince(entry.lastObservedAt)
                      return age.isFinite && age >= 0 && age <= maximumAge
                  }) else { return "" }
            var retained = snapshot.entries
            while true {
                let payload = Rendered(
                    state: "available", runtime: snapshot.runtime,
                    windowHours: 24, outcomesPerTool: maximumOutcomesPerTool,
                    asOf: snapshot.generatedAt,
                    excluded: .init(pending: snapshot.pendingCount, cancelled: snapshot.cancelledCount,
                                    stale: snapshot.staleCount, future: snapshot.futureCount),
                    tools: retained.map { entry in
                        .init(tool: entry.toolID, n: entry.completedCount, success: entry.successCount,
                              last: entry.lastStatus.rawValue, seq: entry.lastSeq,
                              ageSeconds: Int(max(0, snapshot.generatedAt.timeIntervalSince(entry.lastObservedAt))),
                              change: entry.transition.map { "\($0.from.rawValue)->\($0.to.rawValue)" })
                    },
                    omittedTools: snapshot.omittedToolCount + snapshot.entries.count - retained.count
                )
                if let rendered = encode(payload, maxCharacters: maxCharacters) { return rendered }
                guard !retained.isEmpty else { return "" }
                retained.removeLast()
            }
        }
    }

    private nonisolated struct Rendered: Encodable {
        let scope = "historical_tool_results_not_current_capability"
        let state: String
        var runtime: String? = nil
        var windowHours: Int? = nil
        var outcomesPerTool: Int? = nil
        var asOf: Date? = nil
        var excluded: Excluded? = nil
        var tools: [RenderedTool]? = nil
        var omittedTools: Int? = nil
    }

    private nonisolated struct Excluded: Encodable {
        let pending: Int
        let cancelled: Int
        let stale: Int
        let future: Int
    }

    private nonisolated struct RenderedTool: Encodable {
        let tool: String
        let n: Int
        let success: Int
        let last: String
        let seq: Int
        let ageSeconds: Int
        let change: String?
    }

    private static func encode(_ value: Rendered, maxCharacters: Int) -> String? {
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        guard let data = try? encoder.encode(value),
              let text = String(data: data, encoding: .utf8), text.count <= maxCharacters else { return nil }
        return text
    }
}
