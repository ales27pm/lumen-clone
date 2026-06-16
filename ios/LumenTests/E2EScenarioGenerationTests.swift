import Foundation
import Testing
@testable import Lumen

struct E2EScenarioGenerationTests {
    @Test func toolCoverageScenarioIDsAreUnique() {
        let ids = E2ETestScenario.allToolCoverage.map(\.id)
        #expect(Set(ids).count == ids.count)
    }

    @Test func liveToolCoverageExhaustivelyCoversRegisteredTools() {
        let registered = Set(ToolRegistry.all.map { ToolRouteGuard.canonicalToolID($0.id) })
        let grouped = Dictionary(grouping: E2ETestScenario.allToolCoverage) { scenario in
            scenario.requiredAllowedToolIDs.first.map(ToolRouteGuard.canonicalToolID) ?? ""
        }

        #expect(ToolRegistry.all.count == 53)
        #expect(Set(grouped.keys) == registered)
        #expect(E2ETestScenario.allToolCoverage.count == registered.count * ToolScenarioBank.minimumScenariosPerTool)

        for toolID in registered {
            #expect((grouped[toolID]?.count ?? 0) >= ToolScenarioBank.minimumScenariosPerTool)
        }
    }

    @Test func everyToolCoverageScenarioIsLiveAndRequiresItsCoveredTool() {
        for scenario in E2ETestScenario.allToolCoverage {
            #expect(scenario.requiresAgentRun)
            #expect(scenario.kind == .toolGuard)
            #expect(!scenario.prompt.isEmpty)
            #expect(scenario.requiredAllowedToolIDs.count == 1)
            if let toolID = scenario.requiredAllowedToolIDs.first {
                #expect(!scenario.forbiddenToolIDs.contains(toolID))
            }
        }
    }

    @Test func liveToolCoverageMatchesScenarioBankCoverageSummary() {
        let summary = ToolScenarioBank.coverageSummary()
        #expect(summary.toolCount == 53)
        #expect(summary.missingToolIDs.isEmpty)
        #expect(summary.toolsWithAtLeastThreeScenarios == summary.toolCount)
        #expect(summary.scenarioCount == E2ETestScenario.allToolCoverage.count)
        #expect(summary.isProductionReady)
    }

    @Test func liveToolCoveragePromptsRouteToExpectedIntentAndToolScope() async {
        for scenario in E2ETestScenario.allToolCoverage {
            let routing = await IntentClassifierService.shared.route(scenario.prompt)
            #expect(routing.intent == scenario.expectedIntent, "Prompt \(scenario.id) routed as \(routing.intent.rawValue), expected \(scenario.expectedIntent.rawValue)")
            for toolID in scenario.requiredAllowedToolIDs {
                #expect(IntentRouter.isToolAllowed(toolID, for: routing), "Prompt \(scenario.id) did not allow required tool \(toolID)")
            }
        }
    }

    @Test func standardSuiteIncludesLiveToolCoverage() {
        let standardIDs = Set(E2ETestScenario.standard.map(\.id))
        for scenario in E2ETestScenario.allToolCoverage {
            #expect(standardIDs.contains(scenario.id))
        }
    }
}

struct E2EBackwardCompatibilityTests {
    @Test func e2eResultDecodesWithoutPerformanceMatrix() throws {
        let json = """
        {
          "id": "00000000-0000-0000-0000-000000000001",
          "scenarioID": "s",
          "title": "t",
          "prompt": "p",
          "expectedIntent": "chat",
          "actualIntent": "chat",
          "passed": true,
          "failures": [],
          "finalText": "ok",
          "missingHints": [],
          "rewriteAttempted": false,
          "rewriteSuccess": false,
          "events": [],
          "startedAt": "2026-01-01T00:00:00Z",
          "finishedAt": "2026-01-01T00:00:01Z",
          "rawFinalPrefix": "",
          "sanitizedFinalPrefix": "",
          "rawFinalHadUnsafeLeakage": false,
          "sanitizedFinalRemovedArtifacts": [],
          "outputHygieneFailures": []
        }
        """
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        let result = try decoder.decode(E2ETestResult.self, from: Data(json.utf8))
        #expect(result.performanceMatrix == nil)
    }
}
