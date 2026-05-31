import XCTest
@testable import Lumen

final class RuntimeMetricsStoreTests: XCTestCase {
    func testAppendReadAndCompact() async throws {
        let fileURL = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent("metrics-\(UUID().uuidString).jsonl")
        let store = RuntimeMetricsStore(fileURL: fileURL)
        for i in 0..<4 {
            try await store.appendMetric(.init(timestamp: Date(), runtimeName: "r", taskKind: "t", modelIDHash: nil, policySummary: "p", latencyMs: i, success: true, errorCode: nil, thermalState: .nominal, lowPowerMode: false, memoryWarningCount: 0))
        }
        let recent = try await store.recentMetrics(limit: 2)
        XCTAssertEqual(recent.count, 2)
        try await store.compact(maxEntries: 1)
        let compacted = try await store.recentMetrics(limit: 10)
        XCTAssertEqual(compacted.count, 1)
        try? FileManager.default.removeItem(at: fileURL)
    }
}
