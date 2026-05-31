import Foundation
import SwiftData

@MainActor
enum RemCycleService {
    private static let minimumInterval: TimeInterval = 6 * 60 * 60
    private static var lastRunDate: Date?

    static func runIfDue(context: ModelContext, appState: AppState, reason: String) async {
        let now = Date()
        if let lastRunDate, now.timeIntervalSince(lastRunDate) < minimumInterval {
            return
        }
        lastRunDate = now
        await run(context: context, appState: appState, reason: reason, createdAt: now)
    }

    static func run(context: ModelContext, appState: AppState, reason: String, createdAt: Date = Date()) async {
        let stored = (try? context.fetch(FetchDescriptor<StoredModel>())) ?? []
        let fleet = LumenModelFleetResolver.resolveV1(appState: appState, storedModels: stored)
        // Detached tasks are intentional here so parse-summary file IO does not inherit
        // the @MainActor isolation of this service. The loaders read immutable snapshots
        // and return Sendable Strings without touching SwiftData/AppState/UI state.
        let parseSummary = await Task.detached(priority: .utility) {
            AgentParseFailureSummaryLoader.developerText(topN: 5)
        }.value
        let noiseSummary = await Task.detached(priority: .utility) {
            AgentParseNoiseSummaryLoader.developerText(topN: 5)
        }.value

        let report = RemCycleReport(
            id: UUID(),
            createdAt: createdAt,
            reason: reason,
            runnableV1: fleet.isRunnableV1,
            missingSlots: fleet.missingSlots.map(\.rawValue),
            assignedSlots: fleet.assignments.keys.map(\.rawValue).sorted(),
            storedModelCount: stored.count,
            activeChatModelID: appState.activeChatModelID,
            activeEmbeddingModelID: appState.activeEmbeddingModelID,
            parseFailureSummary: parseSummary,
            parseNoiseSummary: noiseSummary
        )

        write(report)

        do {
            try await MemoryCascade.condenseIfNeeded(context: context)
        } catch {
            // REM diagnostics are opportunistic and must never block runtime.
        }
    }

    private static func write(_ report: RemCycleReport) {
        do {
            let directory = try reportsDirectory()
            let url = directory.appendingPathComponent("rem-cycles.jsonl", isDirectory: false)
            let encoder = JSONEncoder()
            encoder.dateEncodingStrategy = .iso8601
            let data = try encoder.encode(report)
            var line = data
            line.append(0x0A)

            if FileManager.default.fileExists(atPath: url.path(percentEncoded: false)) {
                let handle = try FileHandle(forWritingTo: url)
                defer { try? handle.close() }
                try handle.seekToEnd()
                try handle.write(contentsOf: line)
            } else {
                try line.write(to: url, options: [.atomic])
            }
        } catch {
            // REM diagnostics are opportunistic and must never block runtime.
        }
    }

    static func reportsDirectory() throws -> URL {
        let base = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first
            ?? FileManager.default.temporaryDirectory
        let directory = base
            .appendingPathComponent("Diagnostics", isDirectory: true)
            .appendingPathComponent("REM", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        return directory
    }
}

nonisolated struct RemCycleReport: Codable, Sendable {
    let id: UUID
    let createdAt: Date
    let reason: String
    let runnableV1: Bool
    let missingSlots: [String]
    let assignedSlots: [String]
    let storedModelCount: Int
    let activeChatModelID: String?
    let activeEmbeddingModelID: String?
    let parseFailureSummary: String
    let parseNoiseSummary: String
}
