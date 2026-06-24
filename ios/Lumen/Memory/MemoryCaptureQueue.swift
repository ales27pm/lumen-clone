import Foundation
import SwiftData

nonisolated struct PendingMemoryCapture: Codable, Equatable, Identifiable, Sendable {
    let id: UUID
    let content: String
    let kind: MemoryKind
    let source: String
    let topic: String?
    let createdAt: Date
    let retryCount: Int
    let lastError: String?

    init(
        id: UUID = UUID(),
        content: String,
        kind: MemoryKind,
        source: String,
        topic: String?,
        createdAt: Date = Date(),
        retryCount: Int = 0,
        lastError: String? = nil
    ) {
        self.id = id
        self.content = content
        self.kind = kind
        self.source = source
        self.topic = topic
        self.createdAt = createdAt
        self.retryCount = retryCount
        self.lastError = lastError
    }

    func recordingFailure(_ error: Error) -> PendingMemoryCapture {
        PendingMemoryCapture(
            id: id,
            content: content,
            kind: kind,
            source: source,
            topic: topic,
            createdAt: createdAt,
            retryCount: retryCount + 1,
            lastError: MemoryCaptureQueue.sanitizedErrorCode(for: error)
        )
    }
}

nonisolated struct MemoryCaptureDrainResult: Equatable, Sendable {
    let attempted: Int
    let promoted: Int
    let remaining: Int
    let skippedReason: String?
    let lastError: String?

    static func skipped(remaining: Int, reason: String) -> MemoryCaptureDrainResult {
        MemoryCaptureDrainResult(
            attempted: 0,
            promoted: 0,
            remaining: remaining,
            skippedReason: reason,
            lastError: nil
        )
    }
}

nonisolated struct MemoryCaptureQueueDiagnostics: Equatable, Sendable {
    let pendingCount: Int
    let oldestCreatedAt: Date?
    let maxRetryCount: Int
    let lastError: String?

    init(
        pendingCount: Int,
        oldestCreatedAt: Date? = nil,
        maxRetryCount: Int = 0,
        lastError: String? = nil
    ) {
        self.pendingCount = pendingCount
        self.oldestCreatedAt = oldestCreatedAt
        self.maxRetryCount = max(0, maxRetryCount)
        self.lastError = lastError
    }
}

enum MemoryCaptureQueueError: LocalizedError, Equatable {
    case diskWriteDeferred

    var errorDescription: String? {
        switch self {
        case .diskWriteDeferred:
            return "memory capture queue write deferred by disk budget"
        }
    }
}

@MainActor
enum MemoryCaptureQueue {
    nonisolated static let maxPendingCaptures = 200
    nonisolated static let defaultDrainLimit = 8

    private static let fileName = "pending-memory-captures.json"

    static func enqueue(
        content: String,
        kind: MemoryKind = .fact,
        source: String = "app-intent-pending",
        topic: String? = nil,
        createdAt: Date = Date(),
        fileURL: URL? = nil
    ) throws -> PendingMemoryCapture {
        let trimmed = content.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            throw CocoaError(.fileWriteUnknown)
        }

        let url = try resolvedFileURL(fileURL)
        var captures = try loadPending(fileURL: url)
        let dedupeKey = trimmed.lowercased()

        if let existing = captures.first(where: { $0.content.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() == dedupeKey }) {
            return existing
        }

        let capture = PendingMemoryCapture(
            content: trimmed,
            kind: kind,
            source: source,
            topic: topic,
            createdAt: createdAt
        )
        captures.append(capture)
        if captures.count > maxPendingCaptures {
            captures.removeFirst(captures.count - maxPendingCaptures)
        }
        try savePending(captures, fileURL: url)
        return capture
    }

    static func pendingCount(fileURL: URL? = nil) throws -> Int {
        try loadPending(fileURL: resolvedFileURL(fileURL)).count
    }

    static func diagnosticsSnapshot(fileURL: URL? = nil) throws -> MemoryCaptureQueueDiagnostics {
        let captures = try loadPending(fileURL: resolvedFileURL(fileURL))
        return MemoryCaptureQueueDiagnostics(
            pendingCount: captures.count,
            oldestCreatedAt: captures.map(\.createdAt).min(),
            maxRetryCount: captures.map(\.retryCount).max() ?? 0,
            lastError: captures.first(where: { ($0.lastError?.isEmpty == false) })?.lastError
        )
    }

    static func loadPending(fileURL: URL? = nil) throws -> [PendingMemoryCapture] {
        let url = try resolvedFileURL(fileURL)
        let path = url.path(percentEncoded: false)
        guard FileManager.default.fileExists(atPath: path) else { return [] }
        let data = try Data(contentsOf: url)
        guard !data.isEmpty else { return [] }
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return try decoder.decode([PendingMemoryCapture].self, from: data)
    }

    static func drain(
        context: ModelContext,
        maxItems: Int = defaultDrainLimit,
        allowPromotion: Bool,
        fileURL: URL? = nil
    ) async -> MemoryCaptureDrainResult {
        guard allowPromotion else {
            let remaining = (try? pendingCount(fileURL: fileURL)) ?? 0
            return .skipped(remaining: remaining, reason: "promotion_not_allowed")
        }

        let hasEmbeddingRuntime = await AppLlamaService.shared.hasSemanticEmbeddingRuntime
        guard hasEmbeddingRuntime else {
            let remaining = (try? pendingCount(fileURL: fileURL)) ?? 0
            return .skipped(remaining: remaining, reason: "embedding_runtime_unavailable")
        }

        return await drain(maxItems: maxItems, fileURL: fileURL) { capture in
            try await MemoryStore.remember(
                capture.content,
                kind: capture.kind,
                source: capture.source,
                topic: capture.topic,
                context: context
            )
        }
    }

    static func drain(
        maxItems: Int = defaultDrainLimit,
        fileURL: URL? = nil,
        promote: (PendingMemoryCapture) async throws -> Void
    ) async -> MemoryCaptureDrainResult {
        let boundedLimit = max(0, maxItems)
        guard boundedLimit > 0 else {
            let remaining = (try? pendingCount(fileURL: fileURL)) ?? 0
            return .skipped(remaining: remaining, reason: "empty_drain_limit")
        }

        do {
            let url = try resolvedFileURL(fileURL)
            let captures = try loadPending(fileURL: url)
            guard !captures.isEmpty else {
                return MemoryCaptureDrainResult(attempted: 0, promoted: 0, remaining: 0, skippedReason: nil, lastError: nil)
            }

            let candidates = Array(captures.prefix(boundedLimit))
            var remainder = Array(captures.dropFirst(candidates.count))
            var promoted = 0
            var attempted = 0
            var lastError: String?

            for capture in candidates {
                attempted += 1
                do {
                    try await promote(capture)
                    promoted += 1
                } catch {
                    let failed = capture.recordingFailure(error)
                    lastError = failed.lastError
                    remainder.insert(failed, at: 0)
                    let unattempted = candidates.dropFirst(attempted)
                    remainder.insert(contentsOf: unattempted, at: 1)
                    break
                }
            }

            try savePending(remainder, fileURL: url)
            return MemoryCaptureDrainResult(
                attempted: attempted,
                promoted: promoted,
                remaining: remainder.count,
                skippedReason: nil,
                lastError: lastError
            )
        } catch {
            let remaining = (try? pendingCount(fileURL: fileURL)) ?? 0
            return MemoryCaptureDrainResult(
                attempted: 0,
                promoted: 0,
                remaining: remaining,
                skippedReason: "queue_io_failed",
                lastError: sanitizedErrorCode(for: error)
            )
        }
    }

    nonisolated static func sanitizedErrorCode(for error: Error) -> String {
        let raw = String(describing: error)
            .lowercased()
            .replacingOccurrences(of: #"[^a-z0-9_.-]+"#, with: "_", options: .regularExpression)
            .trimmingCharacters(in: CharacterSet(charactersIn: "_"))
        return raw.isEmpty ? "unknown" : String(raw.prefix(80))
    }

    private static func resolvedFileURL(_ fileURL: URL?) throws -> URL {
        if let fileURL { return fileURL }
        let base = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first
            ?? FileManager.default.temporaryDirectory
        let directory = base
            .appendingPathComponent("Lumen", isDirectory: true)
            .appendingPathComponent("Memory", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        return directory.appendingPathComponent(fileName, isDirectory: false)
    }

    private static func savePending(_ captures: [PendingMemoryCapture], fileURL: URL) throws {
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.sortedKeys]
        let data = try encoder.encode(captures)
        guard DiskWriteBudget.shared.canWrite(bytes: data.count, category: .memory) else {
            throw MemoryCaptureQueueError.diskWriteDeferred
        }
        try FileManager.default.createDirectory(at: fileURL.deletingLastPathComponent(), withIntermediateDirectories: true)
        try data.write(to: fileURL, options: [.atomic])
        DiskWriteBudget.shared.recordWrite(bytes: data.count, category: .memory)
    }
}
