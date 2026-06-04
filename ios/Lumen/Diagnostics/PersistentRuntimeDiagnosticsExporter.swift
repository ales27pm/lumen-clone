import Foundation
import UIKit

actor PersistentRuntimeDiagnosticsExporter {
    static let shared = PersistentRuntimeDiagnosticsExporter()

    private let store: PersistentRuntimeDiagnosticsStore
    private let fileManager: FileManager

    init(store: PersistentRuntimeDiagnosticsStore = .shared, fileManager: FileManager = .default) {
        self.store = store
        self.fileManager = fileManager
    }

    func export() async throws -> URL {
        let directory = fileManager.temporaryDirectory.appendingPathComponent("PersistentRuntimeDiagnosticsExport", isDirectory: true)
        try? fileManager.removeItem(at: directory)
        try fileManager.createDirectory(at: directory, withIntermediateDirectories: true)
        let packageURL = directory.appendingPathComponent("persistent-runtime-diagnostics-export.json")
        let payload = PersistentRuntimeDiagnosticsExportPayload(
            exportedAt: Date(),
            appVersion: Bundle.main.persistentDiagnosticsAppVersionSummary,
            sourceCommit: Self.sourceCommit(),
            deviceModel: UIDevice.current.model,
            systemName: UIDevice.current.systemName,
            systemVersion: UIDevice.current.systemVersion,
            campaign: await store.loadCampaign(),
            state: await store.loadState(),
            ndjson: String(data: await store.readLogDataForExport(), encoding: .utf8) ?? ""
        )
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601
        let data = try encoder.encode(payload.redacted())
        try data.write(to: packageURL, options: [.atomic])
        return packageURL
    }

    private static func sourceCommit() -> String? {
        Bundle.main.object(forInfoDictionaryKey: "GitCommit") as? String
    }
}

nonisolated struct PersistentRuntimeDiagnosticsExportPayload: Codable, Sendable {
    var exportedAt: Date
    var appVersion: String
    var sourceCommit: String?
    var deviceModel: String
    var systemName: String
    var systemVersion: String
    var campaign: PersistentDiagnosticCampaign?
    var state: PersistentDiagnosticState?
    var ndjson: String

    func redacted() -> Self {
        var copy = self
        copy.ndjson = ndjson
            .split(separator: "\n", omittingEmptySubsequences: false)
            .map { PersistentRuntimeDiagnosticsRedactor.redactWithoutTruncating(String($0)) }
            .joined(separator: "\n")
        if var state = copy.state {
            state.records = state.records.map { record in
                var mutable = record
                mutable.events = mutable.events.map { PersistentDiagnosticEvent(code: $0.code, message: $0.message, values: $0.values) }
                mutable.failureSummary = mutable.failureSummary.map(PersistentRuntimeDiagnosticsRedactor.redact)
                return mutable
            }
            copy.state = state
        }
        return copy
    }
}
