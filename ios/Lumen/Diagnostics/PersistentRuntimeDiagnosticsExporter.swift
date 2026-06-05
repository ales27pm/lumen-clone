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

    func export(includeFullHistory: Bool = false) async throws -> URL {
        let directory = fileManager.temporaryDirectory.appendingPathComponent(
            "PersistentRuntimeDiagnosticsExport",
            isDirectory: true
        )

        try? fileManager.removeItem(at: directory)
        try fileManager.createDirectory(at: directory, withIntermediateDirectories: true)

        let packageURL = directory.appendingPathComponent("persistent-runtime-diagnostics-export.json")

        let campaign = await store.loadCampaign()
        let state = Self.exportState(await store.loadState(), includeFullHistory: includeFullHistory)
        let logData = await store.readLogDataForExport(full: includeFullHistory)
        let ndjson = String(data: logData, encoding: .utf8) ?? ""
        let metricKitPayloads = await Self.metricKitPayloads()

        let device = await MainActor.run {
            (
                appVersion: Bundle.main.persistentDiagnosticsAppVersionSummary,
                deviceModel: UIDevice.current.model,
                systemName: UIDevice.current.systemName,
                systemVersion: UIDevice.current.systemVersion
            )
        }

        let payload = PersistentRuntimeDiagnosticsExportPayload(
            exportedAt: Date(),
            appVersion: device.appVersion,
            sourceCommit: Self.sourceCommit(),
            deviceModel: device.deviceModel,
            systemName: device.systemName,
            systemVersion: device.systemVersion,
            campaign: campaign,
            state: state,
            ndjson: ndjson,
            metricKitPayloads: metricKitPayloads
        )

        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601

        let data = try encoder.encode(payload.redacted())
        try data.write(to: packageURL, options: [.atomic])

        return packageURL
    }

    private static func exportState(_ state: PersistentDiagnosticState?, includeFullHistory: Bool) -> PersistentDiagnosticState? {
        guard var state = state else { return nil }
        guard !includeFullHistory else { return state }
        if state.records.count > 500 { state.records.removeFirst(state.records.count - 500) }
        state.trimCompletedRunIDs()
        return state
    }

    private static func metricKitPayloads() async -> [PersistentMetricKitExportPayload] {
        let urls = await MetricKitDiagnosticsStore.shared.exportSummaryPayloadURLs()
        return urls.compactMap { url in
            guard let text = try? String(contentsOf: url) else { return nil }
            return PersistentMetricKitExportPayload(fileName: url.lastPathComponent, json: text)
        }
    }

    private static func sourceCommit() -> String? {
        Bundle.main.object(forInfoDictionaryKey: "GitCommit") as? String
    }
}

nonisolated struct PersistentMetricKitExportPayload: Codable, Sendable {
    var fileName: String
    var json: String
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
    var metricKitPayloads: [PersistentMetricKitExportPayload]

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
