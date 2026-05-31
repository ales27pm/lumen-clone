import XCTest
@testable import Lumen

final class MemoryScorerTests: XCTestCase {
    func testRejectCredentialLike() {
        let c = MemoryCandidate(text: "password 123", kind: "fact", topics: [], conversationID: nil, messageID: nil, createdAt: Date(), confidence: 1, extractionReason: "x", userExplicitness: .explicitPreference, sensitivity: .credentialLike)
        XCTAssertEqual(MemoryScorer.score(candidate: c).decision, .rejectSensitive)
    }
}
