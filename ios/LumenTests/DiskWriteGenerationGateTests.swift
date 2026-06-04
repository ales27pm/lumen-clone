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

    func testGenerationLeaseEndClearsGateSynchronously() {
        let lease = DiskWriteBudget.shared.beginGeneration()
        XCTAssertTrue(DiskWriteBudget.shared.shouldDefer(bytes: 512, category: .memory))

        lease.end()

        XCTAssertFalse(DiskWriteBudget.shared.shouldDefer(bytes: 512, category: .memory))
    }

    func testGenerationLeaseEndIsIdempotentAndKeepsOtherLeasesActive() {
        let first = DiskWriteBudget.shared.beginGeneration()
        let second = DiskWriteBudget.shared.beginGeneration()

        first.end()
        first.end()
        XCTAssertTrue(DiskWriteBudget.shared.shouldDefer(bytes: 512, category: .logs))

        second.end()
        XCTAssertFalse(DiskWriteBudget.shared.shouldDefer(bytes: 512, category: .logs))
    }
}
