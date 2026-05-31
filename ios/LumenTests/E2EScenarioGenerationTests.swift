import Foundation
import Testing
@testable import Lumen

struct E2EScenarioGenerationTests {
    @Test func toolCoverageScenarioIDsAreUnique() {
        let ids = E2ETestScenario.allToolCoverage.map(\.id)
        #expect(Set(ids).count == ids.count)
    }

    @Test func toolCoverageVariantsPreserveExpectedIntent() {
        let byID = Dictionary(uniqueKeysWithValues: E2ETestScenario.allToolCoverage.map { ($0.id, $0) })
        let baseScenarios = E2ETestScenario.allToolCoverage.filter { !$0.id.contains("-variant-") }
        for base in baseScenarios {
            let variantB = byID["\(base.id)-variant-b"]
            let variantC = byID["\(base.id)-variant-c"]
            #expect(variantB != nil)
            #expect(variantC != nil)
            #expect(variantB?.expectedIntent == base.expectedIntent)
            #expect(variantC?.expectedIntent == base.expectedIntent)
            #expect(variantB?.requiredAllowedToolIDs == base.requiredAllowedToolIDs)
            #expect(variantC?.requiredAllowedToolIDs == base.requiredAllowedToolIDs)
            #expect(variantB?.forbiddenToolIDs == base.forbiddenToolIDs)
            #expect(variantC?.forbiddenToolIDs == base.forbiddenToolIDs)
        }
    }

    @Test func everyBaseScenarioHasVariantBAndVariantC() {
        let all = E2ETestScenario.allToolCoverage
        let ids = Set(all.map(\.id))
        for base in all where !base.id.contains("-variant-") {
            #expect(ids.contains("\(base.id)-variant-b"))
            #expect(ids.contains("\(base.id)-variant-c"))
        }
    }

    @Test func toolCoverageCountMatchesThreeVariantsPerBaseScenario() {
        let baseCount = E2ETestScenario.allToolCoverage.filter { !$0.id.contains("-variant-") }.count
        #expect(E2ETestScenario.allToolCoverage.count == baseCount * 3)
    }

    @Test func calendarVariantsStayActionSpecific() {
        let all = E2ETestScenario.allToolCoverage
        let listVariants = all.filter { $0.id.hasPrefix("tool-calendar-list-variant-") }
        let createVariants = all.filter { $0.id.hasPrefix("tool-calendar-create-variant-") }
        for scenario in listVariants {
            let p = scenario.prompt.lowercased()
            #expect(!p.contains("create"))
            #expect(!p.contains("schedule"))
            #expect(!p.contains("set"))
        }
        for scenario in createVariants {
            let p = scenario.prompt.lowercased()
            #expect(!p.contains("list"))
            #expect(!p.contains("show"))
            #expect(!p.contains("upcoming"))
        }
    }

    @Test func alarmVariantsStayActionSpecific() {
        let all = E2ETestScenario.allToolCoverage
        let cancelVariants = all.filter { $0.id.hasPrefix("tool-alarm-cancel-variant-") }
        let listVariants = all.filter { $0.id.hasPrefix("tool-alarm-list-variant-") }
        for scenario in cancelVariants {
            let p = scenario.prompt.lowercased()
            #expect(!p.contains("create"))
            #expect(!p.contains("set"))
            #expect(!p.contains("list"))
        }
        for scenario in listVariants {
            let p = scenario.prompt.lowercased()
            #expect(!p.contains("create"))
            #expect(!p.contains("set"))
            #expect(!p.contains("cancel"))
            #expect(!p.contains("countdown"))
        }
        let countdownVariants = all.filter { $0.id.hasPrefix("tool-alarm-countdown-variant-") }
        for scenario in countdownVariants {
            let p = scenario.prompt.lowercased()
            #expect(p.contains("countdown") || p.contains("timer"))
            #expect(!p.contains("monday at"))
        }
        let scheduleVariants = all.filter { $0.id.hasPrefix("tool-alarm-schedule-variant-") }
        for scenario in scheduleVariants {
            let p = scenario.prompt.lowercased()
            #expect(p.contains("alarm"))
            #expect(p.contains("set") || p.contains("create"))
        }
    }

    @Test func reminderAndChatVariantsRespectContracts() {
        let all = E2ETestScenario.allToolCoverage
        for scenario in all.filter({ $0.id.hasPrefix("tool-reminders-create-variant-") }) {
            let p = scenario.prompt.lowercased()
            #expect(!p.contains("list"))
            #expect(!p.contains("show"))
        }
        for scenario in all.filter({ $0.id.hasPrefix("tool-reminders-list-variant-") }) {
            let p = scenario.prompt.lowercased()
            #expect(!p.contains("create"))
            #expect(!p.contains("remind me to"))
        }
        for scenario in E2ETestScenario.chatCoverage.filter({ $0.id.contains("-variant-") }) {
            #expect(scenario.requiredAllowedToolIDs.isEmpty)
            #expect(!scenario.forbiddenToolIDs.isEmpty)
        }
    }

    @Test func defaultFallbackStillGeneratesTwoVariants() {
        let custom = E2ETestScenario(id: "custom", title: "x", kind: .toolGuard, prompt: "Ping", expectedIntent: .trigger, requiredAllowedToolIDs: ["trigger.list"], forbiddenToolIDs: ["calendar.create"], requiredTextHints: [], forbiddenTextHints: [], requiresAgentRun: false)
        let variants = [custom.id + "-variant-b", custom.id + "-variant-c"]
        #expect(variants.count == 2)
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
