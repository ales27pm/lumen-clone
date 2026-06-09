import Foundation
import Testing
@testable import Lumen

struct E2ETestRunnerHygieneTests {
    @Test func recoveredRawThinkLeakPassesWhenSanitizedFinalIsClean() {
        let scenario = E2ETestScenario(id: "s", title: "t", kind: .chat, prompt: "p", expectedIntent: .chat, forbiddenToolIDs: [], requiredTextHints: [], forbiddenTextHints: [], requiresAgentRun: false)
        let failures = E2ETestRunner.hygieneFailures(
            lowerRawFinal: "<think>secret</think> clean",
            lowerFinal: "clean",
            removedArtifacts: [.thinkBlock],
            scenario: scenario,
            observations: ""
        )
        #expect(failures.isEmpty)
    }

    @Test func postRewriteThinkLeakFails() {
        let scenario = E2ETestScenario(id: "s", title: "t", kind: .chat, prompt: "p", expectedIntent: .chat, forbiddenToolIDs: [], requiredTextHints: [], forbiddenTextHints: [], requiresAgentRun: false)
        let failures = E2ETestRunner.hygieneFailures(lowerRawFinal: "clean", lowerFinal: "<think>x</think>", removedArtifacts: [], scenario: scenario, observations: "")
        #expect(failures.contains("Sanitized output still contains hidden reasoning"))
    }

    @Test func recoveredWebPayloadPassesWhenSanitizedFinalIsClean() {
        let scenario = E2ETestScenario(id: "s", title: "t", kind: .chat, prompt: "p", expectedIntent: .chat, forbiddenToolIDs: [], requiredTextHints: [], forbiddenTextHints: [], requiresAgentRun: false)
        let failures = E2ETestRunner.hygieneFailures(
            lowerRawFinal: "<lumen_web_payload>{\"kind\":\"searchresults\",\"results\":[{\"mediakind\":\"page\"}]}</lumen_web_payload>",
            lowerFinal: "clean",
            removedArtifacts: [.lumenWebPayload, .rawToolPayload],
            scenario: scenario,
            observations: ""
        )
        #expect(failures.isEmpty)
    }

    @Test func visibleWebPayloadStillFails() {
        let scenario = E2ETestScenario(id: "s", title: "t", kind: .chat, prompt: "p", expectedIntent: .chat, forbiddenToolIDs: [], requiredTextHints: [], forbiddenTextHints: [], requiresAgentRun: false)
        let failures = E2ETestRunner.hygieneFailures(
            lowerRawFinal: "clean",
            lowerFinal: "<lumen_web_payload>{\"kind\":\"searchresults\",\"results\":[{\"mediakind\":\"page\"}]}</lumen_web_payload>",
            removedArtifacts: [],
            scenario: scenario,
            observations: ""
        )
        #expect(failures.contains("Sanitized output still contains lumen_web_payload markers"))
        #expect(failures.contains("Sanitized output still contains search-results JSON"))
    }

    @Test func weatherUmbrellaOverreachStillFailsWithoutPrecipSignals() {
        let scenario = E2ETestScenario(id: "w", title: "w", kind: .chat, prompt: "p", expectedIntent: .weather, forbiddenToolIDs: [], requiredTextHints: [], forbiddenTextHints: [], requiresAgentRun: false)
        let failures = E2ETestRunner.hygieneFailures(lowerRawFinal: "bring umbrella", lowerFinal: "you should bring an umbrella", removedArtifacts: [], scenario: scenario, observations: "temperature 70 and sunny")
        #expect(failures.contains("Weather precipitation recommendation not grounded"))
    }

    @Test func cleanMarkdownLinkPassesHygieneChecks() {
        let scenario = E2ETestScenario(id: "c", title: "c", kind: .chat, prompt: "p", expectedIntent: .chat, forbiddenToolIDs: [], requiredTextHints: [], forbiddenTextHints: [], requiresAgentRun: false)
        let failures = E2ETestRunner.hygieneFailures(lowerRawFinal: "use [link](https://example.com)", lowerFinal: "use [link](https://example.com)", removedArtifacts: [], scenario: scenario, observations: "")
        #expect(failures.isEmpty)
    }

    @Test func liveAgentUnavailableFallbackCannotPassAfterEvalRewrite() {
        let scenario = E2ETestScenario(
            id: "web",
            title: "web",
            kind: .training,
            prompt: "Search the web for Swift concurrency best practices.",
            expectedIntent: .webSearch,
            requiredAllowedToolIDs: ["web.search"],
            forbiddenToolIDs: [],
            requiredTextHints: ["swift"],
            forbiddenTextHints: [],
            requiresAgentRun: true
        )
        let failures = E2ETestRunner.liveAgentQualityFailures(
            rawFinalText: "Web search is not available in this build yet.",
            finalText: "The model produced only internal reasoning and no final answer. Try again with thinking disabled.\n\nswift",
            scenario: scenario
        )
        #expect(failures.contains("Live agent returned fallback/error text instead of completing the scenario"))
        #expect(!failures.contains(where: { $0.contains("required hint") }))
    }

    @Test func routingOnlyScenarioDoesNotApplyLiveAgentQualityGate() {
        let scenario = E2ETestScenario(
            id: "routing",
            title: "routing",
            kind: .toolGuard,
            prompt: "Search the web.",
            expectedIntent: .webSearch,
            requiredAllowedToolIDs: ["web.search"],
            forbiddenToolIDs: [],
            requiredTextHints: ["swift"],
            forbiddenTextHints: [],
            requiresAgentRun: false
        )
        let failures = E2ETestRunner.liveAgentQualityFailures(
            rawFinalText: "Web search is not available in this build yet.",
            finalText: "swift",
            scenario: scenario
        )
        #expect(failures.isEmpty)
    }

    @Test func deterministicCompatibilityToolTraceCountsAsPolicyFirstEvidenceOnlyWhenAllowed() {
        #if DEBUG
        AgentBehaviorTraceRecorder.clear()
        let startedAt = Date().addingTimeInterval(-1)
        AgentBehaviorTraceRecorder.record(
            AgentBehaviorTrace(
                id: UUID(),
                createdAt: Date(),
                event: .toolAction,
                slot: "executor",
                stage: "compatibility-tool-action",
                intent: "weather",
                promptPrefix: "What is the weather here?",
                rawOutputPrefix: "weather()",
                selectedToolID: "weather",
                toolArguments: [:],
                allowedToolIDs: ["weather"],
                requiresApproval: false,
                approvalMode: nil,
                parseError: nil,
                emittedFinalInActionTurn: false,
                modelFamily: "qwen3",
                runtimePath: "deterministic-compatibility",
                activeAdapterSlot: nil
            )
        )

        #expect(E2ETestRunner.modelRuntimeEvidenceForTests(
            since: startedAt,
            prompt: "What is the weather here?",
            acceptsPolicyFirstEvidence: true
        ))
        #expect(!E2ETestRunner.modelRuntimeEvidenceForTests(
            since: startedAt,
            prompt: "What is the weather here?",
            acceptsPolicyFirstEvidence: false
        ))
        AgentBehaviorTraceRecorder.clear()
        #else
        #expect(true)
        #endif
    }

    @Test func runStandardScenarioLoopRunsOffMainThreadWhenDetached() async {
        #if DEBUG
        let scenario = E2ETestScenario(
            id: "detached-loop-thread",
            title: "Detached loop thread",
            kind: .chat,
            prompt: "Hello there.",
            expectedIntent: .chat,
            forbiddenToolIDs: [],
            requiredTextHints: [],
            forbiddenTextHints: [],
            requiresAgentRun: false
        )
        let config = E2ERunConfig(
            systemPrompt: "",
            temperature: 0.1,
            topP: 1.0,
            repetitionPenalty: 1.0,
            maxTokens: 64,
            maxAgentSteps: 1,
            enabledToolIDs: []
        )
        let recorder = ScenarioLoopThreadRecorder()
        let recordThread: @Sendable (Bool) -> Void = { isMainThread in
            recorder.record(isMainThread: isMainThread)
        }

        let report = await Task.detached {
            await E2ETestRunner.$debugStandardScenariosOverride.withValue([scenario]) {
                await E2ETestRunner.$debugAssertScenarioLoopOffMainThread.withValue(true) {
                    await E2ETestRunner.$debugScenarioLoopThreadRecorder.withValue(recordThread) {
                        await E2ETestRunner.runStandard(config: config)
                    }
                }
            }
        }.value

        #expect(report.results.count == 1)
        #expect(recorder.values == [false])
        #else
        #expect(true)
        #endif
    }
}

struct E2ETestResultExplicitInitializerTests {
    @Test func preservesPassedAndFailuresWithoutMutationFromCache() {
        _ = FinalOutputSanitizer.sanitizeUserVisibleText("<think>x</think>safe")
        let result = E2ETestResult(
            id: UUID(),
            scenarioID: "s",
            title: "t",
            prompt: "p",
            expectedIntent: "chat",
            actualIntent: "chat",
            passed: true,
            failures: ["A", "A"],
            finalText: "safe",
            missingHints: [],
            rewriteAttempted: false,
            rewriteSuccess: true,
            events: [],
            startedAt: Date(),
            finishedAt: Date(),
            rawFinalPrefix: "r",
            sanitizedFinalPrefix: "s",
            rawFinalHadUnsafeLeakage: false,
            sanitizedFinalRemovedArtifacts: ["x", "x"],
            outputHygieneFailures: ["H", "H"]
        )
        #expect(result.passed)
        #expect(result.failures == ["A"])
        #expect(result.sanitizedFinalRemovedArtifacts == ["x"])
        #expect(result.outputHygieneFailures == ["H"])
    }
}

private final class ScenarioLoopThreadRecorder: @unchecked Sendable {
    private let lock = NSLock()
    private var recordedValues: [Bool] = []

    var values: [Bool] {
        lock.lock()
        defer { lock.unlock() }
        return recordedValues
    }

    func record(isMainThread: Bool) {
        lock.lock()
        recordedValues.append(isMainThread)
        lock.unlock()
    }
}
