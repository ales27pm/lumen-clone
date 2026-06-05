import Foundation
#if canImport(MetricKit)
import MetricKit
#endif

#if canImport(MetricKit)
final class MetricKitDiagnosticsSubscriber: NSObject, MXMetricManagerSubscriber {
    static let shared = MetricKitDiagnosticsSubscriber()

    private let store = MetricKitDiagnosticsStore.shared
    private var isRegistered = false

    private override init() { super.init() }

    func register() {
        guard !isRegistered else { return }
        isRegistered = true
        MXMetricManager.shared.add(self)
    }

    func didReceive(_ payloads: [MXMetricPayload]) {
        for payload in payloads {
            let data = payload.jsonRepresentation()
            Task { await store.persistMetricPayload(data) }
        }
    }

    func didReceive(_ payloads: [MXDiagnosticPayload]) {
        for payload in payloads {
            let data = payload.jsonRepresentation()
            Task { await store.persistDiagnosticPayload(data) }
        }
    }
}
#else
final class MetricKitDiagnosticsSubscriber {
    static let shared = MetricKitDiagnosticsSubscriber()
    private init() {}
    func register() {}
}
#endif

actor MetricKitDiagnosticsStore {
    static let shared = MetricKitDiagnosticsStore()

    private let directoryURL: URL
    private let fileManager: FileManager
    private let decoder = JSONDecoder()
    private let encoder: JSONEncoder
    private let maxPayloads = 50

    init(directoryURL: URL? = nil, fileManager: FileManager = .default) {
        self.fileManager = fileManager
        self.directoryURL = directoryURL ?? fileManager.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
            .appendingPathComponent("Diagnostics", isDirectory: true)
            .appendingPathComponent("MetricKit", isDirectory: true)
        self.encoder = JSONEncoder()
        self.encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        self.encoder.dateEncodingStrategy = .iso8601
    }

    func persistMetricPayload(_ data: Data) async {
        await persist(data, prefix: "mxmetric", extracted: extractMetricPayload(from: data))
    }

    func persistDiagnosticPayload(_ data: Data) async {
        await persist(data, prefix: "mxdiagnostic", extracted: extractDiagnosticPayload(from: data))
    }

    func exportSummaryPayloadURLs() async -> [URL] {
        guard let urls = try? fileManager.contentsOfDirectory(at: directoryURL, includingPropertiesForKeys: [.creationDateKey], options: [.skipsHiddenFiles]) else { return [] }
        return urls
            .filter { Self.isSummaryPayloadURL($0) }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
    }

    private func persist(_ data: Data, prefix: String, extracted: MetricKitExtractedDiagnostics) async {
        do {
            try fileManager.createDirectory(at: directoryURL, withIntermediateDirectories: true)
            let stamp = ISO8601DateFormatter().string(from: Date()).replacingOccurrences(of: ":", with: "-")
            let rawURL = directoryURL.appendingPathComponent("\(prefix)-\(stamp)-\(UUID().uuidString).json")
            guard DiskWriteBudget.shared.canWrite(bytes: data.count, category: .diagnostics) else { return }
            try data.write(to: rawURL, options: [.atomic])
            DiskWriteBudget.shared.recordWrite(bytes: data.count, category: .diagnostics)

            let summaryURL = Self.summaryURL(for: rawURL)
            let summaryData = try encoder.encode(extracted)
            if DiskWriteBudget.shared.canWrite(bytes: summaryData.count, category: .diagnostics) {
                try summaryData.write(to: summaryURL, options: [.atomic])
                DiskWriteBudget.shared.recordWrite(bytes: summaryData.count, category: .diagnostics)
            }
            try trimOldPayloads(prefix: prefix)
        } catch {
            PersistentRuntimeDiagnosticsObserver.shared.emit(.init(kind: .metricKitPersistFailure, values: ["errorCode": RuntimeMetricErrorSanitizer.code(for: error)]))
        }
    }

    private func trimOldPayloads(prefix: String) throws {
        let urls = (try? fileManager.contentsOfDirectory(at: directoryURL, includingPropertiesForKeys: [.creationDateKey], options: [.skipsHiddenFiles])) ?? []
        let rawPayloads = urls.filter { url in
            url.lastPathComponent.hasPrefix(prefix) && !Self.isSummaryPayloadURL(url)
        }.sorted { lhs, rhs in
            let ld = (try? lhs.resourceValues(forKeys: [.creationDateKey]).creationDate) ?? .distantPast
            let rd = (try? rhs.resourceValues(forKeys: [.creationDateKey]).creationDate) ?? .distantPast
            return ld < rd
        }
        guard rawPayloads.count > maxPayloads else { return }
        for rawURL in rawPayloads.prefix(rawPayloads.count - maxPayloads) {
            try? fileManager.removeItem(at: rawURL)
            try? fileManager.removeItem(at: Self.summaryURL(for: rawURL))
        }
    }

    private static func isSummaryPayloadURL(_ url: URL) -> Bool {
        url.lastPathComponent.hasSuffix(".summary.json")
    }

    private static func summaryURL(for rawURL: URL) -> URL {
        rawURL.deletingPathExtension().appendingPathExtension("summary.json")
    }

    private func extractMetricPayload(from data: Data) -> MetricKitExtractedDiagnostics {
        let object = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
        return MetricKitExtractedDiagnostics(
            appExitMetrics: values(at: ["applicationExitMetrics"], in: object),
            crashDiagnostics: nil,
            cpuDiagnostics: values(at: ["cpuMetrics"], in: object),
            diskWriteDiagnostics: values(at: ["diskIOMetrics"], in: object),
            foregroundAbnormalExits: values(at: ["applicationExitMetrics", "foregroundExitData", "cumulativeAbnormalExitCount"], in: object),
            backgroundAbnormalExits: values(at: ["applicationExitMetrics", "backgroundExitData", "cumulativeAbnormalExitCount"], in: object),
            watchdogExits: firstValue(containing: "watchdog", in: object),
            memoryPressureExits: firstValue(containing: "memory", in: object)
        )
    }

    private func extractDiagnosticPayload(from data: Data) -> MetricKitExtractedDiagnostics {
        let object = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
        return MetricKitExtractedDiagnostics(
            appExitMetrics: values(at: ["applicationExitMetrics"], in: object),
            crashDiagnostics: values(at: ["crashDiagnostics"], in: object) ?? values(at: ["diagnostics", "crashDiagnostics"], in: object),
            cpuDiagnostics: values(at: ["cpuExceptionDiagnostics"], in: object) ?? values(at: ["diagnostics", "cpuExceptionDiagnostics"], in: object),
            diskWriteDiagnostics: values(at: ["diskWriteExceptionDiagnostics"], in: object) ?? values(at: ["diagnostics", "diskWriteExceptionDiagnostics"], in: object),
            foregroundAbnormalExits: firstValue(containing: "foreground", in: object),
            backgroundAbnormalExits: firstValue(containing: "background", in: object),
            watchdogExits: firstValue(containing: "watchdog", in: object),
            memoryPressureExits: firstValue(containing: "memory", in: object)
        )
    }

    private func values(at path: [String], in object: [String: Any]?) -> String? {
        var current: Any? = object
        for key in path { current = (current as? [String: Any])?[key] }
        guard let current else { return nil }
        if JSONSerialization.isValidJSONObject(current), let data = try? JSONSerialization.data(withJSONObject: current), let text = String(data: data, encoding: .utf8) { return String(text.prefix(500)) }
        return String(describing: current).prefixString(500)
    }

    private func firstValue(containing needle: String, in object: Any?) -> String? {
        if let dict = object as? [String: Any] {
            for (key, value) in dict {
                if key.lowercased().contains(needle.lowercased()) { return values(at: [key], in: dict) }
                if let nested = firstValue(containing: needle, in: value) { return nested }
            }
        } else if let array = object as? [Any] {
            for value in array { if let nested = firstValue(containing: needle, in: value) { return nested } }
        }
        return nil
    }
}

struct MetricKitExtractedDiagnostics: Codable, Sendable, Equatable {
    var appExitMetrics: String?
    var crashDiagnostics: String?
    var cpuDiagnostics: String?
    var diskWriteDiagnostics: String?
    var foregroundAbnormalExits: String?
    var backgroundAbnormalExits: String?
    var watchdogExits: String?
    var memoryPressureExits: String?
}

private extension String.SubSequence {
    func prefixString(_ maxLength: Int) -> String { String(prefix(maxLength)) }
}

private extension String {
    func prefixString(_ maxLength: Int) -> String { String(prefix(maxLength)) }
}
