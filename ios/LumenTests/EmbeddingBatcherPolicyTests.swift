import XCTest
@testable import Lumen

final class EmbeddingBatcherPolicyTests: XCTestCase {
    func testPolicyNoCrash() {
        XCTAssertFalse(RAGMaintenancePolicy.allowEmbeddings(isBackground: true, lowPower: true, thermal: .critical))
    }
}
