import Foundation

/// Host-authored, content-free evidence for structured tool invocations only.
/// A successful begin is a durable acknowledgement, not a claim that the tool ran.
actor SelfPredictionLedger {
    static let shared = SelfPredictionLedger()
    nonisolated static let fileName = "self-prediction-ledger-v1.jsonl"
    nonisolated static let schemaVersion = "self_prediction_ledger.v1"
    nonisolated static let coverageScope = "structured_tool_invocations_stored_window"

    nonisolated enum ProbabilityOrigin: String, Codable, Sendable {
        case model, missing, invalid
        case hostRepair = "host_repair"
    }

    nonisolated enum OutcomeStatus: String, Codable, Sendable {
        case success, failed, denied, requiresApproval, unavailable
    }

    nonisolated struct Context: Sendable {
        let runID: UUID
        let toolID: String
        let runtime: String
        let probabilityOrigin: ProbabilityOrigin
        let probability: Double?

        init(runID: UUID, toolID: String, runtime: String,
             probabilityOrigin: ProbabilityOrigin, probability: Double? = nil) {
            self.runID = runID
            self.toolID = toolID
            self.runtime = runtime
            self.probabilityOrigin = probabilityOrigin
            self.probability = probability
        }
    }

    nonisolated enum Failure: String, Error, LocalizedError, Sendable, Equatable {
        case invalidConfiguration, invalidPrediction, duplicatePrediction
        case unknownPrediction, duplicateOutcome, capacityExceeded, corruptFile, storageFailure

        var errorDescription: String? {
            switch self {
            case .invalidConfiguration: return "The prediction ledger configuration is invalid."
            case .invalidPrediction: return "The prediction metadata is invalid."
            case .duplicatePrediction: return "This prediction is already recorded."
            case .unknownPrediction: return "No recorded prediction matches this outcome."
            case .duplicateOutcome: return "This outcome is already recorded."
            case .capacityExceeded: return "The prediction ledger has reached its storage limit."
            case .corruptFile: return "The prediction ledger cannot be validated."
            case .storageFailure: return "The prediction ledger could not be saved or read."
            }
        }
    }

    nonisolated struct Summary: Sendable {
        let eventCount: Int
        let predictionCount: Int
        let outcomeCount: Int
        let pendingCount: Int
        /// False does not guarantee space for arbitrary metadata; begin checks exact bytes.
        let isAtCapacity: Bool
    }

    nonisolated struct ExportSnapshot: Sendable {
        let data: Data
        let summary: Summary
    }

    private nonisolated enum Kind: String, Codable, Sendable {
        case prediction, outcome
    }

    /// The optional fields form two disjoint, strictly validated wire shapes.
    private nonisolated struct Event: Codable, Sendable {
        let kind: Kind
        let seq: Int
        let id: UUID
        let timestamp: Date
        let runID: UUID?
        let toolID: String?
        let runtime: String?
        let probabilityOrigin: ProbabilityOrigin?
        let probability: Double?
        let status: OutcomeStatus?
        let toolBodyStarted: Bool?
        let cancelled: Bool?

        static func prediction(context: Context, id: UUID, seq: Int, timestamp: Date) -> Event {
            Event(kind: .prediction, seq: seq, id: id, timestamp: timestamp,
                  runID: context.runID, toolID: context.toolID, runtime: context.runtime,
                  probabilityOrigin: context.probabilityOrigin, probability: context.probability,
                  status: nil, toolBodyStarted: nil, cancelled: nil)
        }

        static func outcome(id: UUID, seq: Int, timestamp: Date, status: OutcomeStatus,
                            toolBodyStarted: Bool, cancelled: Bool) -> Event {
            Event(kind: .outcome, seq: seq, id: id, timestamp: timestamp,
                  runID: nil, toolID: nil, runtime: nil, probabilityOrigin: nil, probability: nil,
                  status: status, toolBodyStarted: toolBodyStarted, cancelled: cancelled)
        }
    }

    private nonisolated struct Export: Encodable {
        let schemaVersion: String
        let coverageScope: String
        let events: [Event]
    }

    // An outcome has no variable text. Even with Int.max seq, its canonical line is
    // below 256 bytes. Reserve 512 bytes and one event until it is completed.
    private nonisolated static let outcomeReservationBytes = 512
    private let directoryURL: URL?
    private let maxBytes: Int
    private let maxEvents: Int
    private var loaded = false
    private var poisoned = false
    private var events: [Event] = []
    private var predictions: [UUID: Event] = [:]
    private var completed: Set<UUID> = []
    private var storedBytes = 0

    init(directoryURL: URL? = nil, maxBytes: Int = 2 * 1024 * 1024, maxEvents: Int = 4096) {
        self.directoryURL = directoryURL ?? FileManager.default.urls(
            for: .applicationSupportDirectory, in: .userDomainMask
        ).first?.appendingPathComponent("SelfModel", isDirectory: true)
        self.maxBytes = maxBytes
        self.maxEvents = maxEvents
    }

    /// No cancellation check here: this tiny durable operation must also be usable
    /// when recording the outcome of a cancelled tool. The caller rechecks cancellation
    /// after begin and before crossing its policy/side-effect boundary.
    func begin(context: Context, id: UUID) async throws {
        try loadIfNeeded()
        guard Self.valid(context) else { throw Failure.invalidPrediction }
        guard predictions[id] == nil else { throw Failure.duplicatePrediction }
        let event = Event.prediction(context: context, id: id, seq: events.count + 1, timestamp: Date())
        let line = try Self.line(for: event)
        let pendingAfter = predictions.count - completed.count + 1
        // Subtraction avoids overflow even for caller-supplied extreme limits.
        guard events.count <= maxEvents - 2,
              pendingAfter <= maxEvents - events.count - 1,
              line.count <= maxBytes - storedBytes,
              pendingAfter <= (maxBytes - storedBytes - line.count) / Self.outcomeReservationBytes else {
            throw Failure.capacityExceeded
        }
        try append(line)
        events.append(event)
        predictions[id] = event
    }

    func complete(id: UUID, status: OutcomeStatus, toolBodyStarted: Bool, cancelled: Bool) async throws {
        try loadIfNeeded()
        guard predictions[id] != nil else { throw Failure.unknownPrediction }
        guard !completed.contains(id) else { throw Failure.duplicateOutcome }
        let event = Event.outcome(id: id, seq: events.count + 1, timestamp: Date(),
                                  status: status, toolBodyStarted: toolBodyStarted, cancelled: cancelled)
        let line = try Self.line(for: event)
        guard line.count <= Self.outcomeReservationBytes,
              events.count < maxEvents, line.count <= maxBytes - storedBytes else {
            throw Failure.capacityExceeded
        }
        try append(line)
        events.append(event)
        completed.insert(id)
    }

    func exportJSON() async throws -> Data {
        try loadIfNeeded()
        return try exportData()
    }

    /// Counts and bytes describe the same actor-serialized point in append order.
    func exportSnapshot() async throws -> ExportSnapshot {
        try loadIfNeeded()
        return ExportSnapshot(data: try exportData(), summary: currentSummary())
    }

    func summary() async throws -> Summary {
        try loadIfNeeded()
        return currentSummary()
    }

    private func exportData() throws -> Data {
        do {
            return try Self.encoder().encode(Export(schemaVersion: Self.schemaVersion,
                                                   coverageScope: Self.coverageScope, events: events))
        } catch { throw Failure.storageFailure }
    }

    private func currentSummary() -> Summary {
        let pending = predictions.count - completed.count
        let byteRoom = maxBytes - storedBytes
        return Summary(eventCount: events.count, predictionCount: predictions.count,
                       outcomeCount: completed.count, pendingCount: pending,
                       isAtCapacity: events.count > maxEvents - 2 || pending > maxEvents - events.count - 2
                           || byteRoom / Self.outcomeReservationBytes <= pending)
    }

    private func loadIfNeeded() throws {
        guard maxBytes > 0, maxEvents > 0 else { throw Failure.invalidConfiguration }
        guard !poisoned else { throw Failure.storageFailure }
        guard !loaded else { return }
        guard let directoryURL else { throw Failure.storageFailure }
        let fileURL = directoryURL.appendingPathComponent(Self.fileName)
        do {
            if FileManager.default.fileExists(atPath: directoryURL.path) {
                let values = try directoryURL.resourceValues(forKeys: [.isDirectoryKey, .isSymbolicLinkKey])
                guard values.isDirectory == true, values.isSymbolicLink != true else { throw Failure.storageFailure }
            }
            guard FileManager.default.fileExists(atPath: fileURL.path) else {
                loaded = true
                return
            }
            try validateRegularFile(fileURL)
            let attributes = try FileManager.default.attributesOfItem(atPath: fileURL.path)
            guard let size = (attributes[.size] as? NSNumber)?.intValue, size >= 0, size <= maxBytes else {
                throw Failure.capacityExceeded
            }
            let reader = try FileHandle(forReadingFrom: fileURL)
            var data = Data()
            do {
                while true {
                    let remaining = maxBytes - data.count
                    let readLimit = min(64 * 1024, remaining == Int.max ? remaining : remaining + 1)
                    let chunk = try reader.read(upToCount: readLimit) ?? Data()
                    if chunk.isEmpty { break }
                    data.append(chunk)
                    guard data.count <= maxBytes else { throw Failure.capacityExceeded }
                }
                try reader.close()
            } catch {
                try? reader.close()
                throw error
            }
            guard data.count == size else { throw Failure.corruptFile }
            guard data.isEmpty || data.last == 0x0A else { throw Failure.corruptFile }
            var restored: [Event] = []
            var restoredPredictions: [UUID: Event] = [:]
            var restoredCompleted: Set<UUID> = []
            let decoder = JSONDecoder()
            decoder.dateDecodingStrategy = .iso8601
            let lines = data.split(separator: 0x0A, omittingEmptySubsequences: false)
            for bytes in lines.dropLast() {
                guard !bytes.isEmpty else { throw Failure.corruptFile }
                guard restored.count < maxEvents else { throw Failure.capacityExceeded }
                let line = Data(bytes)
                let event: Event
                do { event = try decoder.decode(Event.self, from: line) }
                catch { throw Failure.corruptFile }
                // This is an internal canonical JSONL format, not an import API.
                // Roundtrip equality also rejects duplicate/unknown keys, explicit null,
                // noncanonical numeric forms and ambiguous UUID/date representations.
                guard try Self.encoder().encode(event) == line,
                      event.seq == restored.count + 1,
                      Self.validTimestamp(event.timestamp) else {
                    throw Failure.corruptFile
                }
                switch event.kind {
                case .prediction:
                    guard let runID = event.runID, let toolID = event.toolID, let runtime = event.runtime,
                          let origin = event.probabilityOrigin,
                          Self.valid(Context(runID: runID, toolID: toolID, runtime: runtime,
                                             probabilityOrigin: origin, probability: event.probability)),
                          event.status == nil, event.toolBodyStarted == nil, event.cancelled == nil,
                          restoredPredictions[event.id] == nil else { throw Failure.corruptFile }
                    restoredPredictions[event.id] = event
                case .outcome:
                    guard event.runID == nil, event.toolID == nil, event.runtime == nil,
                          event.probabilityOrigin == nil, event.probability == nil,
                          event.status != nil, event.toolBodyStarted != nil, event.cancelled != nil,
                          restoredPredictions[event.id] != nil,
                          !restoredCompleted.contains(event.id) else { throw Failure.corruptFile }
                    restoredCompleted.insert(event.id)
                }
                restored.append(event)
            }
            events = restored
            predictions = restoredPredictions
            completed = restoredCompleted
            storedBytes = data.count
            loaded = true
        } catch let failure as Failure { throw failure }
        catch { throw Failure.storageFailure }
    }

    private func append(_ data: Data) throws {
        guard let directoryURL else { throw Failure.storageFailure }
        let fileURL = directoryURL.appendingPathComponent(Self.fileName)
        var handle: FileHandle?
        do {
            try FileManager.default.createDirectory(at: directoryURL, withIntermediateDirectories: true,
                                                    attributes: [.posixPermissions: 0o700])
            let directoryValues = try directoryURL.resourceValues(forKeys: [.isDirectoryKey, .isSymbolicLinkKey])
            guard directoryValues.isDirectory == true, directoryValues.isSymbolicLink != true else {
                throw Failure.storageFailure
            }
            try FileManager.default.setAttributes([.posixPermissions: 0o700], ofItemAtPath: directoryURL.path)
            if !FileManager.default.fileExists(atPath: fileURL.path) {
                guard storedBytes == 0 else { throw Failure.corruptFile }
                try Data().write(to: fileURL, options: .withoutOverwriting)
            }
            try validateRegularFile(fileURL)
            try FileManager.default.setAttributes([.posixPermissions: 0o600], ofItemAtPath: fileURL.path)
            #if os(iOS)
            try FileManager.default.setAttributes([.protectionKey: FileProtectionType.complete], ofItemAtPath: directoryURL.path)
            try FileManager.default.setAttributes([.protectionKey: FileProtectionType.complete], ofItemAtPath: fileURL.path)
            #endif
            let opened = try FileHandle(forWritingTo: fileURL)
            handle = opened
            guard try opened.seekToEnd() == UInt64(storedBytes) else { throw Failure.corruptFile }
            try opened.write(contentsOf: data)
            try opened.synchronize()
            try opened.close()
            handle = nil
            storedBytes += data.count
        } catch {
            try? handle?.close()
            // Never retry after an uncertain append. Preserve all bytes for inspection;
            // a fresh actor must strictly validate the complete file before reuse.
            poisoned = true
            if let failure = error as? Failure { throw failure }
            throw Failure.storageFailure
        }
    }

    private func validateRegularFile(_ url: URL) throws {
        let values = try url.resourceValues(forKeys: [.isRegularFileKey, .isSymbolicLinkKey])
        guard values.isRegularFile == true, values.isSymbolicLink != true else { throw Failure.corruptFile }
    }

    private nonisolated static func valid(_ context: Context) -> Bool {
        guard validToken(context.toolID, limit: 128), context.toolID == context.toolID.lowercased(),
              validToken(context.runtime, limit: 48) else { return false }
        switch context.probabilityOrigin {
        case .model:
            guard let probability = context.probability else { return false }
            return probability.isFinite && (0...1).contains(probability)
        case .missing, .invalid, .hostRepair:
            return context.probability == nil
        }
    }

    private nonisolated static func validToken(_ value: String, limit: Int) -> Bool {
        let bytes = value.utf8
        return !bytes.isEmpty && bytes.count <= limit && bytes.allSatisfy {
            (48...57).contains($0) || (65...90).contains($0) || (97...122).contains($0)
                || $0 == 46 || $0 == 45 || $0 == 95
        }
    }

    private nonisolated static func validTimestamp(_ date: Date) -> Bool {
        // Canonical UTC timestamps use fixed-width years, keeping reservations bounded.
        let seconds = date.timeIntervalSince1970
        return seconds.isFinite && seconds >= -62_135_596_800 && seconds < 253_402_300_800
    }

    private nonisolated static func encoder() -> JSONEncoder {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        encoder.dateEncodingStrategy = .iso8601
        return encoder
    }

    private nonisolated static func line(for event: Event) throws -> Data {
        guard validTimestamp(event.timestamp) else { throw Failure.invalidPrediction }
        do {
            var data = try encoder().encode(event)
            data.append(0x0A)
            return data
        } catch { throw Failure.invalidPrediction }
    }
}
