import Foundation
import UIKit

actor PersistentRuntimeDiagnosticsStore {
    static let shared = PersistentRuntimeDiagnosticsStore()

    let directoryURL: URL
    let campaignURL: URL
    let stateURL: URL
    let logURL: URL
    let rotatedLogURL: URL
    private let fileManager: FileManager
    private let encoder: JSONEncoder
    private let decoder: JSONDecoder
    private var pendingEntries: [PersistentDiagnosticLogEntry] = []
    private var ringEntries: [PersistentDiagnosticLogEntry] = []
    private var lastBatchWriteAt = Date()
    private var scheduledFlushTask: Task<Void, Never>?
    private let batchInterval: TimeInterval = 15
    private let batchSize = 50
    private let maxStoredRecords = 500
    private let maxLogBytes = 1 * 1024 * 1024
    private let defaultExportLineLimit = 500
    private let defaultExportByteLimit = 1 * 1024 * 1024

    init(directoryURL: URL? = nil, fileManager: FileManager = .default) {
        self.fileManager = fileManager
        let base = directoryURL ?? fileManager.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
            .appendingPathComponent("Diagnostics", isDirectory: true)
            .appendingPathComponent("PersistentRuntimeDiagnostics", isDirectory: true)
        self.directoryURL = base
        self.campaignURL = base.appendingPathComponent("persistent-runtime-diagnostics-campaign.json")
        self.stateURL = base.appendingPathComponent("persistent-runtime-diagnostics-state.json")
        self.logURL = base.appendingPathComponent("persistent-runtime-diagnostics.jsonl")
        self.rotatedLogURL = base.appendingPathComponent("persistent-runtime-diagnostics.1.jsonl")
        self.encoder = JSONEncoder()
        self.encoder.outputFormatting = [.sortedKeys]
        self.encoder.dateEncodingStrategy = .iso8601
        self.decoder = JSONDecoder()
        self.decoder.dateDecodingStrategy = .iso8601
    }

    func loadCampaign() async -> PersistentDiagnosticCampaign? {
        guard let data = try? Data(contentsOf: campaignURL) else { return nil }
        return try? decoder.decode(PersistentDiagnosticCampaign.self, from: data)
    }

    func saveCampaign(_ campaign: PersistentDiagnosticCampaign) async throws {
        try ensureDirectory()
        let data = try encoder.encode(campaign)
        guard DiskWriteBudget.shared.canWrite(bytes: data.count, category: .diagnostics) else { return }
        try data.write(to: campaignURL, options: [.atomic])
        DiskWriteBudget.shared.recordWrite(bytes: data.count, category: .diagnostics)
    }

    func loadState() async -> PersistentDiagnosticState? {
        guard let data = try? Data(contentsOf: stateURL) else { return nil }
        return try? decoder.decode(PersistentDiagnosticState.self, from: data)
    }

    func saveState(_ state: PersistentDiagnosticState) async throws {
        try ensureDirectory()
        let data = try encoder.encode(trimmedState(state))
        guard DiskWriteBudget.shared.canWrite(bytes: data.count, category: .diagnostics) else { return }
        try data.write(to: stateURL, options: [.atomic])
        DiskWriteBudget.shared.recordWrite(bytes: data.count, category: .diagnostics)
    }

    func appendEvent(_ event: PersistentDiagnosticEvent, recordID: UUID? = nil, campaignID: UUID? = nil) async {
        let entry = PersistentDiagnosticLogEntry(kind: "event", recordID: recordID, campaignID: campaignID, event: event, record: nil)
        await append(entry)
    }

    func appendRunUpdate(_ record: PersistentDiagnosticRunRecord) async {
        let entry = PersistentDiagnosticLogEntry(kind: "run", recordID: record.id, campaignID: record.campaignID, event: nil, record: record)
        await append(entry)
    }

    func flushBufferedIfPossible() async {
        guard !DiskWriteBudget.shared.isGenerationActive(), !pendingEntries.isEmpty else { return }
        await flushPending(force: true)
    }

    func clearLogs() async throws {
        try? fileManager.removeItem(at: logURL)
        try? fileManager.removeItem(at: rotatedLogURL)
        pendingEntries = []
        ringEntries = []
        scheduledFlushTask?.cancel()
        scheduledFlushTask = nil
        lastBatchWriteAt = Date()
    }

    func readLogDataForExport(full: Bool = false) async -> Data {
        var lines: [String] = []
        let persistedEntries = persistedLogEntries(includeRotated: full)
        let exportEntries = persistedEntries + pendingEntries
        for entry in exportEntries {
            if let data = try? encoder.encode(entry), let line = String(data: data, encoding: .utf8) {
                lines.append(line)
            }
        }
        if !full, lines.count > defaultExportLineLimit { lines.removeFirst(lines.count - defaultExportLineLimit) }
        let data = lines.joined(separator: "\n").data(using: .utf8) ?? Data()
        return full ? data : boundedExportData(data)
    }

    func markUnfinishedRunInterrupted(launchUUID: UUID, startupAt: Date) async throws -> PersistentDiagnosticRunRecord? {
        var state = await loadState() ?? PersistentDiagnosticState()
        guard let activeRunID = state.activeRunID,
              !state.completedRunIDs.contains(activeRunID),
              let scenario = state.activeScenario,
              let startedAt = state.activeStartedAt else { return nil }
        var metrics = PersistentDiagnosticMetrics()
        metrics.cancellationReason = state.cleanCancellationBeforeTermination ? "clean_cancel_before_termination" : "interrupted_or_terminated"
        let statusText = state.cleanCancellationBeforeTermination ? "clean_cancel_before_termination" : "interrupted_or_terminated"
        var record = PersistentDiagnosticRunRecord(id: activeRunID, campaignID: state.activeCampaignID ?? UUID(), scenario: scenario, startedAt: startedAt, status: .interrupted, metrics: metrics)
        record.finishedAt = startupAt
        record.failureSummary = statusText
        record.events.append(PersistentDiagnosticEvent(code: "crash_resume", message: statusText, values: [
            "previousLaunchUUID": state.activeLaunchUUID?.uuidString ?? "unknown",
            "currentLaunchUUID": launchUUID.uuidString
        ]))
        let proposals = PersistentDiagnosticRemediationAdvisor.proposals(for: record, status: .interrupted, code: "crash_resume")
        if !proposals.isEmpty {
            record.remediationProposals = proposals
            record.events.append(PersistentDiagnosticEvent(
                code: "diagnostic_remediation_proposal",
                message: proposals.map(\.title).joined(separator: "; "),
                values: [
                    "ids": proposals.map(\.id).joined(separator: ","),
                    "severity": proposals.map(\.severity.rawValue).sorted().joined(separator: ",")
                ]
            ))
        }
        state.status.lastCrashResumeStatus = statusText
        state.status.lastRemediationSummary = record.remediationProposals?.first?.title
        state.records.append(record)
        if state.records.count > maxStoredRecords { state.records.removeFirst(state.records.count - maxStoredRecords) }
        state.markRunCompleted(record.id)
        state.activeRunID = nil
        state.activeCampaignID = nil
        state.activeScenario = nil
        state.activeStartedAt = nil
        state.cleanCancellationBeforeTermination = false
        try await saveState(state)
        await appendRunUpdate(record)
        return record
    }

    private func append(_ entry: PersistentDiagnosticLogEntry) async {
        ringEntries.append(entry)
        if ringEntries.count > maxStoredRecords { ringEntries.removeFirst(ringEntries.count - maxStoredRecords) }
        pendingEntries.append(entry)
        if pendingEntries.count > maxStoredRecords { pendingEntries.removeFirst(pendingEntries.count - maxStoredRecords) }
        await flushPending(force: false)
        scheduleFlushIfNeeded()
    }

    private func flushPending(force: Bool) async {
        guard !pendingEntries.isEmpty else { return }
        let elapsed = Date().timeIntervalSince(lastBatchWriteAt)
        guard force || pendingEntries.count >= batchSize || elapsed >= batchInterval else { return }
        guard !DiskWriteBudget.shared.isGenerationActive() else { return }
        guard let data = encodeBatchJSON(pendingEntries) else { return }
        guard DiskWriteBudget.shared.canWrite(bytes: data.count, category: .diagnostics) else { return }
        do {
            try ensureDirectory()
            try rotateIfNeeded(incomingBytes: data.count)
            if fileManager.fileExists(atPath: logURL.path) {
                let handle = try FileHandle(forWritingTo: logURL)
                try handle.seekToEnd()
                try handle.write(contentsOf: data)
                try handle.close()
            } else {
                try data.write(to: logURL, options: [.atomic])
            }
            DiskWriteBudget.shared.recordWrite(bytes: data.count, category: .diagnostics)
            pendingEntries = []
            scheduledFlushTask?.cancel()
            scheduledFlushTask = nil
            lastBatchWriteAt = Date()
        } catch {
            if pendingEntries.count > maxStoredRecords { pendingEntries.removeFirst(pendingEntries.count - maxStoredRecords) }
        }
    }

    private func encodeBatchJSON(_ entries: [PersistentDiagnosticLogEntry]) -> Data? {
        guard !entries.isEmpty else { return nil }
        guard let data = try? encoder.encode(entries) else { return nil }
        var output = data
        output.append("\n".data(using: .utf8) ?? Data())
        return output
    }

    private func persistedLogEntries(includeRotated: Bool) -> [PersistentDiagnosticLogEntry] {
        var entries: [PersistentDiagnosticLogEntry] = []
        let urls = (includeRotated ? [rotatedLogURL] : []) + [logURL]
        for url in urls {
            guard let text = try? String(contentsOf: url, encoding: .utf8) else { continue }
            for line in text.split(separator: "\n") {
                guard let data = String(line).data(using: .utf8) else { continue }
                if let batch = try? decoder.decode([PersistentDiagnosticLogEntry].self, from: data) {
                    entries.append(contentsOf: batch)
                } else if let legacyEntry = try? decoder.decode(PersistentDiagnosticLogEntry.self, from: data) {
                    entries.append(legacyEntry)
                }
            }
        }
        return entries
    }

    private func scheduleFlushIfNeeded() {
        guard !pendingEntries.isEmpty, scheduledFlushTask == nil else { return }
        let delay = UInt64(batchInterval * 1_000_000_000)
        scheduledFlushTask = Task { [weak self] in
            try? await Task.sleep(nanoseconds: delay)
            await self?.flushBufferedIfPossible()
        }
    }

    private func rotateIfNeeded(incomingBytes: Int) throws {
        let currentBytes = (try? fileManager.attributesOfItem(atPath: logURL.path)[.size] as? NSNumber)?.intValue ?? 0
        guard currentBytes + incomingBytes > maxLogBytes else { return }
        try? fileManager.removeItem(at: rotatedLogURL)
        if fileManager.fileExists(atPath: logURL.path) {
            try fileManager.moveItem(at: logURL, to: rotatedLogURL)
        }
    }

    private func ensureDirectory() throws {
        try fileManager.createDirectory(at: directoryURL, withIntermediateDirectories: true)
    }

    private func trimmedState(_ state: PersistentDiagnosticState) -> PersistentDiagnosticState {
        var copy = state
        if copy.records.count > maxStoredRecords { copy.records.removeFirst(copy.records.count - maxStoredRecords) }
        copy.trimCompletedRunIDs()
        return copy
    }

    private func boundedExportData(_ data: Data) -> Data {
        guard !data.isEmpty else { return data }
        let text = String(data: data, encoding: .utf8) ?? ""
        var lines = text.split(separator: "\n", omittingEmptySubsequences: false).map(String.init)
        if lines.count > defaultExportLineLimit {
            lines.removeFirst(lines.count - defaultExportLineLimit)
        }
        var output = lines.joined(separator: "\n")
        var encoded = output.data(using: .utf8) ?? Data()
        if encoded.count > defaultExportByteLimit {
            encoded = Data(encoded.suffix(defaultExportByteLimit))
            output = String(data: encoded, encoding: .utf8) ?? ""
            if let newline = output.firstIndex(of: "\n") {
                output = String(output[output.index(after: newline)...])
            }
            encoded = output.data(using: .utf8) ?? encoded
        }
        return encoded
    }
}

nonisolated struct PersistentDiagnosticLogEntry: Codable, Sendable {
    let kind: String
    let at: Date
    let recordID: UUID?
    let campaignID: UUID?
    let event: PersistentDiagnosticEvent?
    let record: PersistentDiagnosticRunRecord?

    init(kind: String, recordID: UUID?, campaignID: UUID?, event: PersistentDiagnosticEvent?, record: PersistentDiagnosticRunRecord?) {
        self.kind = kind
        self.at = Date()
        self.recordID = recordID
        self.campaignID = campaignID
        self.event = event
        self.record = record
    }
}
