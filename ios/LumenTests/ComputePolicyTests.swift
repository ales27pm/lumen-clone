import XCTest
@testable import Lumen

final class ComputePolicyTests: XCTestCase {
    func testBackgroundConstrained() {
        let decision = ComputePolicy.decide(for: .init(task: .chat, isForeground: false, lowPowerMode: false, thermalState: .nominal))
        XCTAssertEqual(decision.maxTokens, 256)
        XCTAssertFalse(decision.allowHeavyRuntime)
    }

    func testForegroundLowPowerConstrained() {
        let decision = ComputePolicy.decide(for: .init(task: .chat, isForeground: true, lowPowerMode: true, thermalState: .nominal))
        XCTAssertEqual(decision.maxTokens, 512)
        XCTAssertFalse(decision.allowHeavyRuntime)
    }
}
