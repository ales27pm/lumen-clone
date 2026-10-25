import XCTest
@testable import Lumen

final class AgentPredictionKeyIntegrityTests: XCTestCase {
    func testDuplicateAndEscapedKeysCannotBecomeValidForecasts() {
        for raw in [
            #"{"action":{"tool":"weather","args":{}},"prediction":{"successProbability":0.2,"successProbability":0.9}}"#,
            #"{"action":{"tool":"weather","args":{}},"prediction":{"successProbability":0.2},"prediction":{"successProbability":0.9}}"#,
            #"{"action":{"tool":"weather","args":{}},"prediction":{"successProbability":0.2,"success\u0050robability":0.9}}"#,
            #"{"action":{"tool":"weather","args":{}},"prediction":{},"\u0070rediction":{"successProbability":0.9}}"#,
        ] {
            XCTAssertTrue(AgentJSONKeyIntegrity.hasAmbiguousKeys(in: raw))
            let turn = AgentTurnParser.parse(raw)
            XCTAssertNil(turn.parseError)
            XCTAssertNotNil(turn.action)
            XCTAssertEqual(turn.prediction, .invalid)
        }
    }

    func testKeyLikeTextAndSameKeysInDifferentObjectsAreNotDuplicates() {
        let raw = #"{"thought":"quoted \"prediction\": { } [ ] \\ text","action":{"tool":"weather","args":{"items":[{"name":"a"},{"name":"b"}]}},"prediction":{"successProbability":0.7}}"#
        XCTAssertFalse(AgentJSONKeyIntegrity.hasAmbiguousKeys(in: raw))
        XCTAssertEqual(AgentTurnParser.parse(raw).prediction.successProbability, 0.7)
    }
}
