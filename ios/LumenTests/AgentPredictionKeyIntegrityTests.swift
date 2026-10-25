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

    func testDuplicateKeysInDecodedCompatibilityInputInvalidateOnlyForecast() {
        for raw in [
            #"{"action":{"tool":"web.search","input":"{\"query\":\"first\",\"query\":\"second\"}"},"prediction":{"successProbability":0.9}}"#,
            #"{"tool":"web.search","input":"{\"query\":\"first\",\"qu\\u0065ry\":\"second\"}","prediction":{"successProbability":0.9}}"#,
            #"{"action":{"tool":"web.search","args":"{\"query\":\"first\",\"query\":\"second\"}","input":"unused"},"prediction":{"successProbability":0.9}}"#
        ] {
            XCTAssertFalse(AgentJSONKeyIntegrity.hasAmbiguousKeys(in: raw))
            let turn = AgentTurnParser.parse(raw)
            XCTAssertNil(turn.parseError)
            XCTAssertEqual(turn.action?.tool, "web.search")
            XCTAssertNotNil(turn.action?.args["query"])
            XCTAssertEqual(turn.prediction, .invalid)
        }
    }

    func testUnambiguousCompatibilityInputAndUnusedInputTextKeepForecast() {
        for raw in [
            #"{"action":{"tool":"web.search","input":"{\"query\":\"swift\"}"},"prediction":{"successProbability":0.7}}"#,
            #"{"action":{"tool":"web.search","input":"a literal query"},"prediction":{"successProbability":0.7}}"#,
            #"{"action":{"tool":"web.search","args":{"query":"swift"},"input":"{\"query\":\"first\",\"query\":\"second\"}"},"prediction":{"successProbability":0.7}}"#
        ] {
            let turn = AgentTurnParser.parse(raw)
            XCTAssertNil(turn.parseError)
            XCTAssertEqual(turn.prediction.validity, .valid)
            XCTAssertEqual(turn.prediction.successProbability, 0.7)
        }
    }
}
