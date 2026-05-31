import Foundation

actor RuntimeMetricsStore {
    static let shared = RuntimeMetricsStore()

    private let fileURL: URL
    private let encoder = JSONEncoder()
    private let decoder = JSONDecoder()

    init(fileURL: URL? = nil) {
        if let fileURL {
            self.fileURL = fileURL
        } else {
            let base = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first ?? URL(fileURLWithPath: NSTemporaryDirectory())
            let dir = base.appendingPathComponent("Lumen", isDirectory: true)
            try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
            self.fileURL = dir.appendingPathComponent("runtime-metrics.jsonl")
        }
        encoder.dateEncodingStrategy = .iso8601
        decoder.dateDecodingStrategy = .iso8601
    }

    func appendMetric(_ metric: RuntimeMetric) async throws {
        let line = try encoder.encode(metric) + Data([0x0A])
        if !FileManager.default.fileExists(atPath: fileURL.path) {
            let created = FileManager.default.createFile(atPath: fileURL.path, contents: line)
            if !created {
                try line.write(to: fileURL, options: .atomic)
            }
            return
        }
        let handle = try FileHandle(forWritingTo: fileURL)
        defer { try? handle.close() }
        try handle.seekToEnd()
        try handle.write(contentsOf: line)
    }

    func recentMetrics(limit: Int) async throws -> [RuntimeMetric] {
        guard FileManager.default.fileExists(atPath: fileURL.path) else { return [RuntimeMetric]() }
        let data = try Data(contentsOf: fileURL)
        let lines = data.split(separator: 0x0A)
        let decoded: [RuntimeMetric] = lines.compactMap { try? decoder.decode(RuntimeMetric.self, from: Data($0)) }
        return Array(decoded.suffix(max(0, limit)))
    }

    func compact(maxEntries: Int) async throws {
        let recent = try await recentMetrics(limit: maxEntries)
        let newData = try recent.map { try encoder.encode($0) + Data([0x0A]) }.reduce(Data(), +)
        try newData.write(to: fileURL, options: .atomic)
    }
}
