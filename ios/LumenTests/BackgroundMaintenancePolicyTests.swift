import XCTest
@testable import Lumen

final class BackgroundMaintenancePolicyTests: XCTestCase {
    func testRagMaintenancePolicy() {
        XCTAssertFalse(RAGMaintenancePolicy.allowEmbeddings(isBackground: true, lowPower: true, thermal: .critical))
        XCTAssertTrue(RAGMaintenancePolicy.allowEmbeddings(isBackground: false, lowPower: false, thermal: .nominal))
    }
}
