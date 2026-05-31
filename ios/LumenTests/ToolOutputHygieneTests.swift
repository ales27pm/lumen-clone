import Testing
@testable import Lumen

struct ToolOutputHygieneTests {
    @Test func hygieneRejectsForbiddenArtifacts() {
        let scenario = E2ETestScenario(id: "h", title: "h", kind: .chat, prompt: "p", expectedIntent: .chat, forbiddenToolIDs: [], requiredTextHints: [], forbiddenTextHints: [], requiresAgentRun: false)
        let forbidden = ["<think hidden", "</think>", "<lumen_web_payload>", "{\"kind\":\"searchresults\",\"results\":[{\"mediakind\":\"page\"}]}"]
        for text in forbidden {
            let failures = E2ETestRunner.hygieneFailures(lowerRawFinal: text.lowercased(), lowerFinal: text.lowercased(), removedArtifacts: [], scenario: scenario, observations: "")
            #expect(!failures.isEmpty)
        }
    }
}
