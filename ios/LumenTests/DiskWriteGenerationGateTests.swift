import XCTest
@testable import Lumen

final class DiskWriteGenerationGateTests: XCTestCase {
    override func tearDown() {
        DiskWriteBudget.shared.setGenerationActive(false)
        super.tearDown()
    }

    func testDiagnosticsAndMetricsWritesAreDeferredDuringGeneration() {
        DiskWriteBudget.shared.setGenerationActive(true)

        XCTAssertTrue(DiskWriteBudget.shared.shouldDefer(bytes: 512, category: .diagnostics))
        XCTAssertTrue(DiskWriteBudget.shared.shouldDefer(bytes: 512, category: .logs))
        XCTAssertTrue(DiskWriteBudget.shared.shouldDefer(bytes: 512, category: .memory))
        XCTAssertFalse(DiskWriteBudget.shared.shouldDefer(bytes: 512, category: .conversation))
    }
}
