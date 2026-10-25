import Foundation
import XCTest
@testable import Lumen

final class AgentTurnPredictionTests: XCTestCase {
    func testRuntimeSchemasKeepForecastOptionalAndBoundedOnActionsOnly() throws {
        for schema in [StructuredAgentKernelExecutor.structuredAgentResponseSchema,
                       StructuredAgentKernelExecutor.structuredAgentActionResponseSchema,
                       StructuredAgentKernelExecutor.structuredAgentFinalResponseSchema] {
            _ = try JSONSerialization.jsonObject(with: Data(schema.utf8))
        }
        let object = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(
            StructuredAgentKernelExecutor.structuredAgentActionResponseSchema.utf8
        )) as? [String: Any])
        XCTAssertEqual(object["required"] as? [String], ["action"])
        let properties = try XCTUnwrap(object["properties"] as? [String: Any])
        let prediction = try XCTUnwrap(properties["prediction"] as? [String: Any])
        XCTAssertEqual(prediction["required"] as? [String], ["successProbability"])
        XCTAssertEqual(prediction["additionalProperties"] as? Bool, false)
        let predictionFields = try XCTUnwrap(prediction["properties"] as? [String: Any])
        let probability = try XCTUnwrap(predictionFields["successProbability"] as? [String: Any])
        XCTAssertEqual(probability["type"] as? String, "number")
        XCTAssertEqual(probability["minimum"] as? Double, 0)
        XCTAssertEqual(probability["maximum"] as? Double, 1)
        let final = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(
            StructuredAgentKernelExecutor.structuredAgentFinalResponseSchema.utf8
        )) as? [String: Any])
        let finalProperties = try XCTUnwrap(final["properties"] as? [String: Any])
        XCTAssertNil(finalProperties["prediction"])
    }

    func testOldActionJSONHasNoInventedPrediction() {
        let turn = AgentTurnParser.parse(#"{"action":{"tool":"weather","args":{}}}"#)
        XCTAssertNil(turn.parseError)
        XCTAssertEqual(turn.action, AgentAction(tool: "weather", args: [:]))
        XCTAssertEqual(turn.prediction, .missing)
    }

    func testActionPredictionAcceptsProbabilitiesIncludingNumericEndpoints() {
        for literal in ["0", "1", "0.72"] {
            let turn = AgentTurnParser.parse("{\"action\":{\"tool\":\"weather\",\"args\":{}},\"prediction\":{\"successProbability\":\(literal)}}")
            XCTAssertNil(turn.parseError)
            XCTAssertEqual(turn.prediction.validity, .valid)
            XCTAssertEqual(turn.prediction.successProbability, Double(literal))
        }
    }

    func testInvalidMetadataDoesNotAlterOrAuthorizeTheAction() throws {
        for metadata in ["true", "null", "[]", "{}", #"{"successProbability":true}"#,
                         #"{"successProbability":"0.8"}"#, #"{"successProbability":-0.01}"#,
                         #"{"successProbability":1.01}"#, #"{"successProbability":0.8,"extra":1}"#] {
            let turn = AgentTurnParser.parse("{\"action\":{\"tool\":\"weather\",\"args\":{}},\"prediction\":\(metadata)}")
            XCTAssertNil(turn.parseError)
            XCTAssertEqual(turn.action, AgentAction(tool: "weather", args: [:]))
            XCTAssertEqual(turn.prediction, .invalid)
        }
        let turn = AgentTurnParser.parse(#"{"action":{"tool":"web.search","args":{"query":"swift","extra":true}},"prediction":{"successProbability":0.99}}"#)
        let action = try XCTUnwrap(turn.action)
        guard case .failure(let error) = StructuredToolCallValidator.validate(action: action, availableTools: ToolRegistry.all) else {
            return XCTFail("Forecast metadata must not relax tool argument validation")
        }
        XCTAssertEqual(error, .extraArguments(tool: "web.search", arguments: ["extra"]))
    }

    func testNonFiniteProbabilityMetadataIsRejected() {
        for value in [Double.nan, .infinity, -.infinity] {
            XCTAssertEqual(AgentTurnPrediction.parse(["successProbability": value]), .invalid)
        }
        // Non-JSON NaN is still malformed action output, never an executable action.
        let turn = AgentTurnParser.parse(#"{"action":{"tool":"weather","args":{}},"prediction":{"successProbability":NaN}}"#)
        XCTAssertNotNil(turn.parseError)
        XCTAssertNil(turn.action)
    }

    func testFinalTextOrThoughtDoesNotBecomeAPrediction() {
        let final = AgentTurnParser.parse(#"{"final":"successProbability=0.99","prediction":{"successProbability":0.99}}"#)
        XCTAssertNil(final.parseError)
        XCTAssertEqual(final.prediction, .missing)
        let thought = AgentTurnParser.parse(#"{"thought":"I am 99% certain","action":{"tool":"weather","args":{}}}"#)
        XCTAssertEqual(thought.prediction, .missing)
    }

    func testHostRepairCannotInheritPredictionEvenWithCollidingDedupeKey() throws {
        let turn = AgentTurnParser.parse(#"{"action":{"tool":"rag.search","args":{"query":"swift","limit":3}},"prediction":{"successProbability":0.9}}"#)
        let original = try XCTUnwrap(turn.action)
        let repaired = AgentAction(tool: "rag.search", args: ["query": .string("swift"), "limit": .string("3")])
        XCTAssertEqual(original.dedupeKey, repaired.dedupeKey)
        XCTAssertNotEqual(original, repaired)
        let context = StructuredAgentKernelExecutor.predictionContext(for: turn, executing: repaired, runID: UUID(), runtime: "coreAI")
        XCTAssertEqual(context.probabilityOrigin, .hostRepair)
        XCTAssertNil(context.probability)
    }

    func testUnchangedModelActionRetainsOnlyItsTypedForecast() throws {
        for (metadata, expectedOrigin) in [("", SelfPredictionLedger.ProbabilityOrigin.missing),
                                           (#","prediction":{}"#, .invalid),
                                           (#","prediction":{"successProbability":0.7}"#, .model)] {
            let turn = AgentTurnParser.parse("{\"action\":{\"tool\":\"weather\",\"args\":{}}\(metadata)}")
            let action = try XCTUnwrap(turn.action)
            let context = StructuredAgentKernelExecutor.predictionContext(for: turn, executing: action, runID: UUID(), runtime: "coreAI")
            XCTAssertEqual(context.probabilityOrigin, expectedOrigin)
            XCTAssertEqual(context.probability, expectedOrigin == .model ? 0.7 : nil)
        }
    }
}
