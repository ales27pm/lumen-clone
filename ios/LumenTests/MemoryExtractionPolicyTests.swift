import XCTest
@testable import Lumen

final class MemoryExtractionPolicyTests: XCTestCase {
    func testExplicitRememberAllowed() { XCTAssertTrue(MemoryExtractionPolicy.shouldExtract(trigger: .explicitRemember, lowPower: true, isBackground: true, containsSensitiveToolOutput: false)) }
}
