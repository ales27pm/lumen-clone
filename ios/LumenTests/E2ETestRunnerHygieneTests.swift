import Foundation
import SwiftUI
import Testing
@testable import Lumen

private actor ResourceBudgetGateSnapshotOverrideLock {
    static let shared = ResourceBudgetGateSnapshotOverrideLock()

    func withOverride<T>(
        _ snapshot: ResourceBudgetGate.Snapshot,
        operation: () async throws -> T
    ) async rethrows -> T {
        await MainActor.run {
            ResourceBudgetGate.setDiagnosticSnapshotOverride(snapshot)
        }
        do {
            let result = try await operation()
            await MainActor.run {
                ResourceBudgetGate.clearDiagnosticSnapshotOverride()
            }
            return result
        } catch {
            await MainActor.run {
                ResourceBudgetGate.clearDiagnosticSnapshotOverride()
            }
            throw error
        }
    }

}

@Suite(.serialized)
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

    @Test func incompleteWeatherFinalIsRejectedAndCanRepairFromObservation() {
        #if DEBUG
        let final = "The weather is clear with a temperature of 21°C. You do not need an"
        let scenario = E2ETestScenario(
            id: "training-weather-here-no-calendar",
            title: "Training weather",
            kind: .training,
            prompt: "What is the weather here?",
            expectedIntent: .weather,
            requiredAllowedToolIDs: ["weather"],
            forbiddenToolIDs: [],
            requiredTextHints: [],
            forbiddenTextHints: [],
            requiresAgentRun: true
        )
        let failures = E2ETestRunner.hygieneFailures(
            lowerRawFinal: final.lowercased(),
            lowerFinal: final.lowercased(),
            removedArtifacts: [],
            scenario: scenario,
            observations: "Weather observation: clear, temperature 21°C"
        )
        #expect(failures.contains("Final output appears incomplete or truncated"))

        let repaired = E2ETestRunner.deterministicToolObservationFallbackForIncompleteFinalForTests(
            scenario: scenario,
            routing: IntentRoutingDecision(intent: .weather, allowedToolIDs: ["weather"], requiresClarification: false, clarificationPrompt: nil),
            finalText: final,
            events: [
                E2ETestEvent(
                    id: UUID(),
                    createdAt: Date(),
                    scenarioID: scenario.id,
                    phase: "step",
                    message: "observation: The weather is clear with a temperature of 21°C."
                )
            ]
        )
        #expect(repaired == "Weather update: The weather is clear with a temperature of 21°C.")
        #else
        #expect(true)
        #endif
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

    @Test func toolBackedSafeMessageFallbackCannotPassAsCompletion() {
        let calendarScenario = E2ETestScenario(
            id: "live-calendar-list-direct-show-my-next-calendar-events",
            title: "calendar",
            kind: .toolGuard,
            prompt: "Show my next calendar events.",
            expectedIntent: .calendar,
            requiredAllowedToolIDs: ["calendar.list"],
            forbiddenToolIDs: [],
            requiredTextHints: ["module", "[1]"],
            forbiddenTextHints: [],
            requiresAgentRun: true
        )
        #expect(E2ETestRunner.liveAgentQualityFailures(
            rawFinalText: "I couldn’t safely complete the calendar event request.",
            finalText: "I couldn’t safely complete the calendar event request.",
            scenario: calendarScenario
        ).contains("Live agent returned fallback/error text instead of completing the scenario"))

        let ragScenario = E2ETestScenario(
            id: "live-rag-index-files-direct-reindex-my-imported-files",
            title: "rag",
            kind: .toolGuard,
            prompt: "Reindex my imported files.",
            expectedIntent: .rag,
            requiredAllowedToolIDs: ["rag.index_files"],
            forbiddenToolIDs: [],
            requiredTextHints: [],
            forbiddenTextHints: [],
            requiresAgentRun: true
        )
        #expect(E2ETestRunner.liveAgentQualityFailures(
            rawFinalText: "I couldn't safely complete the local search/indexing request.",
            finalText: "I couldn't safely complete the local search/indexing request.",
            scenario: ragScenario
        ).contains("Live agent returned fallback/error text instead of completing the scenario"))
    }

    @Test func liveValidationFallbackJSONCannotPassAsFinalAnswer() {
        let scenario = E2ETestScenario(
            id: "training-memory-loop",
            title: "memory",
            kind: .training,
            prompt: "Remember that I prefer concise bullet points, then tell me what you remembered.",
            expectedIntent: .memory,
            requiredAllowedToolIDs: ["memory.save", "memory.recall"],
            forbiddenToolIDs: [],
            requiredTextHints: ["remember"],
            forbiddenTextHints: [],
            requiresAgentRun: true
        )
        let raw = #"{"reasoningSummary":"Memory tool output could not be validated.","rewrittenFinalAnswer":"Memory tool output could not be validated.","requiresApprovalDecision":"deny"}"#
        let final = raw + "\n\nI remember that you prefer concise bullet points."

        let failures = E2ETestRunner.liveAgentQualityFailures(
            rawFinalText: raw,
            finalText: final,
            scenario: scenario
        )

        #expect(failures.contains("Live agent returned fallback/error text instead of completing the scenario"))
    }

    @Test func evalRewriteSkipsInvalidFallbackFinals() async {
        #if DEBUG
        let scenario = E2ETestScenario(
            id: "training-memory-loop",
            title: "memory",
            kind: .training,
            prompt: "Remember that I prefer concise bullet points, then tell me what you remembered.",
            expectedIntent: .memory,
            requiredAllowedToolIDs: ["memory.save", "memory.recall"],
            forbiddenToolIDs: [],
            requiredTextHints: ["remember"],
            forbiddenTextHints: [],
            requiresAgentRun: true
        )
        let routing = IntentRoutingDecision(intent: .memory, allowedToolIDs: ["memory.save", "memory.recall"], requiresClarification: false, clarificationPrompt: nil)
        let outcome = await E2ETestRunner.validateAndRewriteFinalTextIfNeededForTests(
            scenario: scenario,
            routing: routing,
            originalFinal: "Memory tool output could not be validated."
        )
        #expect(outcome.finalText == "Memory tool output could not be validated.")
        #expect(!outcome.rewriteAttempted)
        #expect(!outcome.rewriteSuccess)
        #expect(outcome.missingHints.contains(where: { $0.contains("prefer concise bullet points") }))
        #else
        #expect(true)
        #endif
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

    @Test func liveWebSearchScenarioTemporarilyEnablesNetworkAccess() {
        #if DEBUG
        let scenario = E2ETestScenario(
            id: "training-web-research",
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
        let routing = IntentRoutingDecision(
            intent: .webSearch,
            allowedToolIDs: ["web.search", "web.fetch"],
            requiresClarification: false,
            clarificationPrompt: nil
        )
        #expect(E2ETestRunner.scenarioTemporarilyEnablesNetworkAccessForTests(
            scenario,
            routing: routing,
            availableToolIDs: ["web.search"]
        ))
        #else
        #expect(true)
        #endif
    }

    @Test func routingOnlyWebSearchScenarioDoesNotEnableNetworkAccess() {
        #if DEBUG
        let scenario = E2ETestScenario(
            id: "tool-web-search",
            title: "web",
            kind: .toolGuard,
            prompt: "Search the web.",
            expectedIntent: .webSearch,
            requiredAllowedToolIDs: ["web.search"],
            forbiddenToolIDs: [],
            requiredTextHints: [],
            forbiddenTextHints: [],
            requiresAgentRun: false
        )
        let routing = IntentRoutingDecision(
            intent: .webSearch,
            allowedToolIDs: ["web.search"],
            requiresClarification: false,
            clarificationPrompt: nil
        )
        #expect(!E2ETestRunner.scenarioTemporarilyEnablesNetworkAccessForTests(
            scenario,
            routing: routing,
            availableToolIDs: ["web.search"]
        ))
        #else
        #expect(true)
        #endif
    }

    @Test func liveScenarioDoesNotAcceptPolicyFirstCompatibilityEvidence() {
        #if DEBUG
        let scenario = E2ETestScenario(
            id: "live-weather",
            title: "weather",
            kind: .regression,
            prompt: "What is the weather here?",
            expectedIntent: .weather,
            requiredAllowedToolIDs: ["weather"],
            forbiddenToolIDs: [],
            requiredTextHints: [],
            forbiddenTextHints: [],
            requiresAgentRun: true
        )
        let routing = IntentRoutingDecision(
            intent: .weather,
            allowedToolIDs: ["weather"],
            requiresClarification: false,
            clarificationPrompt: nil
        )

        #expect(!E2ETestRunner.acceptsPolicyFirstExecutionEvidenceForTests(scenario, routing: routing))
        #else
        #expect(true)
        #endif
    }

    @Test func toolGuardScenarioAcceptsPolicyFirstToolEvidence() {
        #if DEBUG
        let scenario = E2ETestScenario(
            id: "live-alarm-countdown-alternatephrasing-start-a-timer-for-10-minutes",
            title: "Live alarm countdown",
            kind: .toolGuard,
            prompt: "Start a timer for 10 minutes.",
            expectedIntent: .alarm,
            requiredAllowedToolIDs: ["alarm.countdown"],
            forbiddenToolIDs: [],
            requiredTextHints: [],
            forbiddenTextHints: [],
            requiresAgentRun: true,
            evidenceMode: .policyFirstAllowed
        )
        let routing = IntentRoutingDecision(
            intent: .alarm,
            allowedToolIDs: ["alarm.countdown"],
            requiresClarification: false,
            clarificationPrompt: nil
        )
        let request = AgentRequest(
            systemPrompt: "sys",
            history: [],
            userMessage: scenario.prompt,
            temperature: 0,
            topP: 1,
            repetitionPenalty: 1,
            maxTokens: 128,
            maxSteps: 2,
            availableTools: ToolRegistry.all,
            relevantMemories: [],
            scenarioID: scenario.id
        )

        let options = E2ETestRunner.strictLiveAgentRunOptionsForTests(
            req: request,
            scenario: scenario,
            e2eRunID: UUID(),
            agentRunID: UUID(),
            routing: routing
        )

        #expect(E2ETestRunner.acceptsPolicyFirstExecutionEvidenceForTests(scenario, routing: routing))
        #expect(options.allowDeterministicCompatibility)
        #expect(options.allowParseFailureDeterministicRecovery)
        #expect(!options.allowsMemoryPressureContinuation)
        #else
        #expect(true)
        #endif
    }

    @Test func modelBackedToolGuardDoesNotAcceptPolicyFirstEvidenceByDefault() {
        #if DEBUG
        let scenario = E2ETestScenario(
            id: "live-alarm-countdown-direct",
            title: "Live alarm countdown",
            kind: .toolGuard,
            prompt: "Start a timer for 10 minutes.",
            expectedIntent: .alarm,
            requiredAllowedToolIDs: ["alarm.countdown"],
            forbiddenToolIDs: [],
            requiredTextHints: [],
            forbiddenTextHints: [],
            requiresAgentRun: true
        )
        let routing = IntentRoutingDecision(
            intent: .alarm,
            allowedToolIDs: ["alarm.countdown"],
            requiresClarification: false,
            clarificationPrompt: nil
        )

        #expect(!E2ETestRunner.acceptsPolicyFirstExecutionEvidenceForTests(scenario, routing: routing))
        #else
        #expect(true)
        #endif
    }

    @Test func toolGuardRejectsClarificationWithoutExpectedAlarmAction() {
        #if DEBUG
        let auth = E2ETestScenario(
            id: "live-alarm-authorization-status-direct",
            title: "Alarm auth",
            kind: .toolGuard,
            prompt: "Check alarm authorization status.",
            expectedIntent: .alarm,
            requiredAllowedToolIDs: ["alarm.authorization_status"],
            forbiddenToolIDs: [],
            requiredTextHints: [],
            forbiddenTextHints: [],
            requiresAgentRun: true,
            evidenceMode: .policyFirstAllowed,
            expectedToolID: "alarm.authorization_status",
            scenarioBankKind: ToolScenarioBankEntry.ScenarioKind.direct.rawValue
        )
        let routing = IntentRoutingDecision(intent: .alarm, allowedToolIDs: ["alarm.authorization_status"], requiresClarification: true, clarificationPrompt: "What time or duration should I use?")
        let failures = E2ETestRunner.toolCoverageEvidenceFailuresForTests(
            scenario: auth,
            routing: routing,
            agentSteps: [AgentStep(kind: .reflection, content: "Clarification required before tool execution.")],
            finalText: "I couldn’t safely complete the alarm/timer request."
        )
        #expect(failures.contains(where: { $0.contains("incorrectly stopped at clarification") }))

        let list = E2ETestScenario(
            id: "live-alarm-list-direct",
            title: "Alarm list",
            kind: .toolGuard,
            prompt: "List active alarms.",
            expectedIntent: .alarm,
            requiredAllowedToolIDs: ["alarm.list"],
            forbiddenToolIDs: [],
            requiredTextHints: [],
            forbiddenTextHints: [],
            requiresAgentRun: true,
            evidenceMode: .policyFirstAllowed,
            expectedToolID: "alarm.list",
            scenarioBankKind: ToolScenarioBankEntry.ScenarioKind.direct.rawValue
        )
        let listFailures = E2ETestRunner.toolCoverageEvidenceFailuresForTests(
            scenario: list,
            routing: routing,
            agentSteps: [],
            finalText: "I couldn’t safely complete the alarm/timer request."
        )
        #expect(listFailures.contains(where: { $0.contains("expected tool alarm.list") }))
        #else
        #expect(true)
        #endif
    }

    @Test func toolGuardMissingExpectedToolMetadataFailsClosed() {
        #if DEBUG
        let scenario = E2ETestScenario(
            id: "live-alarm-status-direct",
            title: "Alarm status",
            kind: .toolGuard,
            prompt: "Show alarm permission status.",
            expectedIntent: .alarm,
            requiredAllowedToolIDs: ["alarm.authorization_status"],
            forbiddenToolIDs: [],
            requiredTextHints: [],
            forbiddenTextHints: [],
            requiresAgentRun: true,
            evidenceMode: .policyFirstAllowed,
            expectedToolID: nil,
            scenarioBankKind: ToolScenarioBankEntry.ScenarioKind.direct.rawValue
        )
        let failures = E2ETestRunner.toolCoverageEvidenceFailuresForTests(
            scenario: scenario,
            routing: IntentRoutingDecision(intent: .alarm, allowedToolIDs: ["alarm.authorization_status"], requiresClarification: false, clarificationPrompt: nil),
            agentSteps: [],
            finalText: "I couldn’t safely complete the alarm/timer request."
        )
        #expect(failures == ["Tool coverage scenario missing expectedToolID metadata."])
        #else
        #expect(true)
        #endif
    }

    @Test func toolGuardApprovalAndMissingArgumentSemanticsAreStrict() {
        #if DEBUG
        let countdown = E2ETestScenario(
            id: "live-alarm-countdown-approvalboundary",
            title: "Alarm countdown",
            kind: .toolGuard,
            prompt: "Start a countdown called Approval for 5 minutes.",
            expectedIntent: .alarm,
            requiredAllowedToolIDs: ["alarm.countdown"],
            forbiddenToolIDs: [],
            requiredTextHints: [],
            forbiddenTextHints: [],
            requiresAgentRun: true,
            evidenceMode: .policyFirstAllowed,
            expectedToolID: "alarm.countdown",
            scenarioBankKind: ToolScenarioBankEntry.ScenarioKind.approvalBoundary.rawValue
        )
        let routing = IntentRoutingDecision(intent: .alarm, allowedToolIDs: ["alarm.countdown"], requiresClarification: false, clarificationPrompt: nil)
        #expect(E2ETestRunner.toolCoverageEvidenceFailuresForTests(
            scenario: countdown,
            routing: routing,
            agentSteps: [AgentStep(kind: .approvalBoundary, content: "Approval required", toolID: "alarm.countdown")],
            finalText: "Approval required for alarm.countdown."
        ).isEmpty)

        let cancel = E2ETestScenario(
            id: "live-alarm-cancel-approvalboundary",
            title: "Alarm cancel",
            kind: .toolGuard,
            prompt: "Cancel alarm 00000000-0000-0000-0000-000000000000.",
            expectedIntent: .alarm,
            requiredAllowedToolIDs: ["alarm.cancel"],
            forbiddenToolIDs: [],
            requiredTextHints: [],
            forbiddenTextHints: [],
            requiresAgentRun: true,
            evidenceMode: .policyFirstAllowed,
            expectedToolID: "alarm.cancel",
            scenarioBankKind: ToolScenarioBankEntry.ScenarioKind.approvalBoundary.rawValue
        )
        #expect(E2ETestRunner.toolCoverageEvidenceFailuresForTests(
            scenario: cancel,
            routing: IntentRoutingDecision(intent: .alarm, allowedToolIDs: ["alarm.cancel"], requiresClarification: false, clarificationPrompt: nil),
            agentSteps: [AgentStep(kind: .action, content: "alarm.cancel", toolID: "alarm.cancel")],
            finalText: "Cancelled."
        ).contains(where: { $0.contains("missing approval boundary") }))
        #expect(E2ETestRunner.toolCoverageEvidenceFailuresForTests(
            scenario: cancel,
            routing: IntentRoutingDecision(intent: .alarm, allowedToolIDs: ["alarm.cancel"], requiresClarification: true, clarificationPrompt: "What time or duration should I use?"),
            agentSteps: [AgentStep(kind: .reflection, content: "Clarification required before tool execution.")],
            finalText: "I couldn’t safely complete the alarm/timer request."
        ).contains(where: { $0.contains("incorrectly stopped at clarification") }))

        let missingCancel = E2ETestScenario(
            id: "live-alarm-cancel-missingargument",
            title: "Alarm cancel missing",
            kind: .toolGuard,
            prompt: "Cancel my alarm.",
            expectedIntent: .alarm,
            requiredAllowedToolIDs: ["alarm.cancel"],
            forbiddenToolIDs: [],
            requiredTextHints: [],
            forbiddenTextHints: [],
            requiresAgentRun: true,
            evidenceMode: .policyFirstAllowed,
            expectedToolID: "alarm.cancel",
            scenarioBankKind: ToolScenarioBankEntry.ScenarioKind.missingArgument.rawValue
        )
        #expect(E2ETestRunner.toolCoverageEvidenceFailuresForTests(
            scenario: missingCancel,
            routing: IntentRoutingDecision(intent: .alarm, allowedToolIDs: ["alarm.cancel"], requiresClarification: true, clarificationPrompt: "Which alarm should I cancel?"),
            agentSteps: [],
            finalText: "Which alarm should I cancel?"
        ).isEmpty)
        #else
        #expect(true)
        #endif
    }

    @Test func genericAlarmFallbackIsNotToolObservationEvidence() {
        #if DEBUG
        #expect(!E2ETestRunner.isSafeToolObservationFinalForTests("I couldn’t safely complete the alarm/timer request.", expectedToolID: "alarm.authorization_status"))
        #expect(E2ETestRunner.isSafeToolObservationFinalForTests("Alarm authorization status: authorized", expectedToolID: "alarm.authorization_status"))
        #expect(E2ETestRunner.isSafeToolObservationFinalForTests("No active alarms", expectedToolID: "alarm.list"))
        #expect(E2ETestRunner.isSafeToolObservationFinalForTests(AlarmTools.unavailableMessage, expectedToolID: "alarm.authorization_status"))
        #else
        #expect(true)
        #endif
    }

    @Test func alarmKitUnavailableEvidenceIsRuntimeNonActionable() {
        #if DEBUG
        let scenario = E2ETestScenario(
            id: "live-alarm-status-direct",
            title: "Live alarm status",
            kind: .toolGuard,
            prompt: "Check alarm authorization status.",
            expectedIntent: .alarm,
            requiredAllowedToolIDs: ["alarm.authorization_status"],
            forbiddenToolIDs: [],
            requiredTextHints: [],
            forbiddenTextHints: [],
            requiresAgentRun: true,
            expectedToolID: "alarm.authorization_status",
            scenarioBankKind: ToolScenarioBankEntry.ScenarioKind.direct.rawValue
        )
        let failure = E2ETestRunner.alarmRuntimeUnavailableEvidenceFailureForTests(
            scenario: scenario,
            agentSteps: [
                AgentStep(kind: .action, content: "alarm.authorization_status()", toolID: "alarm.authorization_status"),
                AgentStep(kind: .observation, content: AlarmTools.unavailableMessage, toolID: "alarm.authorization_status")
            ],
            finalText: AlarmTools.unavailableMessage
        )
        #expect(failure == "AlarmKit runtime unavailable for expected tool alarm.authorization_status; device-runtime evidence required.")

        let result = E2ETestResult(
            id: UUID(),
            scenarioID: scenario.id,
            kind: scenario.kind.rawValue,
            title: scenario.title,
            prompt: scenario.prompt,
            expectedIntent: scenario.expectedIntent.rawValue,
            actualIntent: UserIntent.alarm.rawValue,
            e2eRunID: UUID(),
            agentRunID: UUID(),
            conversationID: UUID(),
            turnID: UUID(),
            requiresAgentRun: true,
            evidenceMode: E2EEvidenceMode.modelBackedRequired.rawValue,
            passed: false,
            failures: [failure ?? ""],
            finalText: AlarmTools.unavailableMessage,
            missingHints: [],
            rewriteAttempted: false,
            rewriteSuccess: false,
            events: [],
            startedAt: Date(),
            finishedAt: Date(),
            rawFinalPrefix: AlarmTools.unavailableMessage,
            sanitizedFinalPrefix: AlarmTools.unavailableMessage,
            rawFinalHadUnsafeLeakage: false,
            sanitizedFinalRemovedArtifacts: [],
            outputHygieneFailures: [],
            performanceMatrix: nil,
            metadata: [
                "failureKind": "liveRuntimeAlarmKitUnavailable",
                "actionable": "false",
                "trainingSignal": "false"
            ]
        )
        #expect(E2ETestReport(id: UUID(), startedAt: Date(), finishedAt: Date(), passed: 0, failed: 1, results: [result]).summaryText.contains("runtime-preflight/non-actionable"))
        #else
        #expect(true)
        #endif
    }

    @Test func clarificationScenarioAcceptsPolicyFirstClarificationEvidence() {
        #if DEBUG
        let scenario = E2ETestScenario(
            id: "vague-email-clarifies",
            title: "Vague email draft asks clarification",
            kind: .routing,
            prompt: "Draft a email",
            expectedIntent: .emailDraft,
            requiredAllowedToolIDs: ["mail.draft", "contacts.search"],
            forbiddenToolIDs: [],
            requiredTextHints: ["who should", "what should"],
            forbiddenTextHints: [],
            requiresAgentRun: true
        )
        let routing = IntentRoutingDecision(
            intent: .emailDraft,
            allowedToolIDs: ["contacts.search", "mail.draft"],
            requiresClarification: true,
            clarificationPrompt: "Who should I send it to, and what should it say?"
        )

        #expect(E2ETestRunner.acceptsPolicyFirstExecutionEvidenceForTests(scenario, routing: routing))
        #else
        #expect(true)
        #endif
    }

    @Test func chatOnlyScenarioUsesPlainTextTurnInsteadOfStructuredAgentJson() {
        #if DEBUG
        let scenario = E2ETestScenario(
            id: "normal-chat-no-forced-tool",
            title: "Normal chat does not force tools",
            kind: .chat,
            prompt: "Explain why a sharp chisel is safer than a dull one.",
            expectedIntent: .chat,
            requiredAllowedToolIDs: [],
            forbiddenToolIDs: ["weather"],
            requiredTextHints: [],
            forbiddenTextHints: [],
            requiresAgentRun: true
        )
        let routing = IntentRoutingDecision(
            intent: .chat,
            allowedToolIDs: [],
            requiresClarification: false,
            clarificationPrompt: nil
        )

        #expect(E2ETestRunner.shouldRunAsPlainTextTurnForTests(scenario, routing: routing))
        #else
        #expect(true)
        #endif
    }

    @Test func liveRunStopsAfterThermalOrBudgetFailure() {
        #if DEBUG
        let result = E2ETestResult(
            id: UUID(),
            scenarioID: "live-alarm-stop-direct",
            kind: E2ETestKind.toolGuard.rawValue,
            title: "Live alarm stop",
            prompt: "Stop alarm 00000000-0000-0000-0000-000000000000",
            expectedIntent: UserIntent.alarm.rawValue,
            actualIntent: UserIntent.alarm.rawValue,
            requiresAgentRun: true,
            passed: false,
            failures: ["Live E2E scenario did not record model-backed generation evidence"],
            finalText: "I couldn't complete the structured agent turn because agent-json produced no JSON output. Reason: executor preflight failed: resource-budget-denied-before-prompt-eval; budgetReason=strict-live-training.executor-preflight: thermalState=serious; device thermal state serious; cool device and retry.",
            missingHints: [],
            rewriteAttempted: false,
            rewriteSuccess: false,
            events: [],
            startedAt: Date(),
            finishedAt: Date(),
            rawFinalPrefix: "",
            sanitizedFinalPrefix: "",
            rawFinalHadUnsafeLeakage: false,
            sanitizedFinalRemovedArtifacts: [],
            outputHygieneFailures: [],
            performanceMatrix: nil
        )

        #expect(E2ETestRunner.liveRuntimeShouldStopAfterForTests(result))
        #else
        #expect(true)
        #endif
    }

    @Test func liveRuntimePreflightBlockedResultIsSingleThermalFailure() async {
        #if DEBUG
        let scenario = E2ETestScenario(
            id: "live-alarm-stop-direct",
            title: "Live alarm stop",
            kind: .toolGuard,
            prompt: "Stop alarm 00000000-0000-0000-0000-000000000000",
            expectedIntent: .alarm,
            requiredAllowedToolIDs: ["alarm.stop"],
            forbiddenToolIDs: [],
            requiredTextHints: [],
            forbiddenTextHints: [],
            requiresAgentRun: true
        )
        let denial = "live-e2e.pre-scenario: thermalState=serious; \(ResourceBudgetGate.seriousThermalRetryHint)"

        let result = await E2ETestRunner.liveRuntimePreflightBlockedResultForTests(
            scenario,
            denialReason: denial
        )

        #expect(!result.passed)
        #expect(result.scenarioID == scenario.id)
        #expect(result.actualIntent == "preflight")
        #expect(result.failures.count == 1)
        #expect(result.failures[0].contains("before prompt evaluation"))
        #expect(result.finalText == ResourceBudgetGate.seriousThermalRetryHint)
        #expect(result.metadata["failureKind"] == "liveRuntimeThermalCooldownRequired")
        #expect(result.metadata["budgetDenialReason"] == denial)
        #expect(result.metadata["actionable"] == "false")
        #expect(result.metadata["trainingSignal"] == "false")
        #expect(result.isRuntimePreflightNonActionable)
        #expect(result.events.map(\.phase) == ["live-runtime-preflight"])
        #expect(E2ETestRunner.liveRuntimeShouldStopAfterForTests(result))
        #else
        #expect(true)
        #endif
    }

    @Test func liveRuntimeScenePhasePreflightIsNonActionable() async {
        #if DEBUG
        let scenario = E2ETestScenario(
            id: "live-alarm-countdown-direct",
            title: "Live alarm countdown",
            kind: .toolGuard,
            prompt: "Start a timer for 10 minutes.",
            expectedIntent: .alarm,
            requiredAllowedToolIDs: ["alarm.countdown"],
            forbiddenToolIDs: [],
            requiredTextHints: [],
            forbiddenTextHints: [],
            requiresAgentRun: true
        )
        let denial = "live-e2e.pre-scenario: scenePhase=background"
        let result = await E2ETestRunner.liveRuntimePreflightBlockedResultForTests(
            scenario,
            denialReason: denial
        )
        #expect(!result.passed)
        #expect(result.metadata["failureKind"] == "liveRuntimeScenePhaseUnavailable")
        #expect(result.metadata["budgetDenialReason"] == denial)
        #expect(result.metadata["actionable"] == "false")
        #expect(result.metadata["trainingSignal"] == "false")
        #expect(result.isRuntimePreflightNonActionable)
        let summary = E2ETestReport(id: UUID(), startedAt: Date(), finishedAt: Date(), passed: 0, failed: 1, results: [result]).summaryText
        #expect(summary.contains("Failed: 0"))
        #expect(summary.contains("Runtime preflight/non-actionable: 1"))
        #expect(summary.contains("runtime-preflight/non-actionable"))
        #else
        #expect(true)
        #endif
    }

    @Test func liveTrainingCPUWatchdogSkipsBeforeGeneration() async {
        #if DEBUG
        let scenario = E2ETestScenario(
            id: "training-web-research",
            title: "Training web research",
            kind: .training,
            prompt: "Search the web for Swift concurrency best practices.",
            expectedIntent: .webSearch,
            requiredAllowedToolIDs: ["web.search"],
            forbiddenToolIDs: [],
            requiredTextHints: ["Swift"],
            forbiddenTextHints: [],
            requiresAgentRun: true
        )

        let result = await E2ETestRunner.$debugCPUWatchdogDegradedOverride.withValue(true) {
            await E2ETestRunner.liveRuntimePreflightBlockedResultIfNeededForTests(scenario)
        }

        #expect(result?.actualIntent == "preflight")
        #expect(result?.metadata["failureKind"] == "liveRuntimeCPUWatchdogDegraded")
        #expect(result?.metadata["actionable"] == "false")
        #expect(result?.metadata["trainingSignal"] == "false")
        #expect(result?.events.map(\.phase) == ["live-runtime-preflight"])
        #else
        #expect(true)
        #endif
    }

    @Test func liveRuntimeReadinessBarrierCatchesCPUWatchdogAfterWait() async {
        #if DEBUG
        let scenario = E2ETestScenario(
            id: "training-scheduler-agent",
            title: "Training trigger",
            kind: .training,
            prompt: "Schedule a trigger to summarize reminders tonight and confirm what will run.",
            expectedIntent: .trigger,
            requiredAllowedToolIDs: ["trigger.create"],
            forbiddenToolIDs: [],
            requiredTextHints: ["trigger"],
            forbiddenTextHints: [],
            requiresAgentRun: true
        )
        let probe = CPUWatchdogProbeBox(degradeAfterCalls: 1)
        let snapshot = ResourceBudgetGate.Snapshot(
            scenePhase: .background,
            lowPowerModeEnabled: false,
            thermalState: .nominal,
            recentMemoryWarningCount: 0,
            lastMemoryWarningAt: nil
        )

        let outcome = await ResourceBudgetGateSnapshotOverrideLock.shared.withOverride(snapshot) {
            await E2ETestRunner.$debugCPUWatchdogDegradedProbe.withValue({ _ in
                probe.isDegraded()
            }) {
                await E2ETestRunner.liveRuntimeReadinessBarrierForTests(
                    scenario,
                    maxWaitNanoseconds: 10_000_000,
                    pollNanoseconds: 1_000_000
                )
            }
        }

        #expect(outcome.denialReason == "live-e2e.pre-scenario: cpu-watchdog-degraded")
        #expect(outcome.events.contains { $0.phase == "live-runtime-preflight-wait" })
        #else
        #expect(true)
        #endif
    }

    @Test func passedResultsWithRuntimeWordsAreNotNonActionablePreflight() {
        let result = E2ETestResult(
            id: UUID(),
            scenarioID: "passed-with-old-runtime-text",
            kind: E2ETestKind.training.rawValue,
            title: "Passed scenario",
            prompt: "Summarize runtime diagnostics.",
            expectedIntent: UserIntent.chat.rawValue,
            actualIntent: UserIntent.chat.rawValue,
            requiresAgentRun: true,
            passed: true,
            failures: [],
            finalText: "Previous run mentioned thermalState=serious, but this scenario passed.",
            missingHints: [],
            rewriteAttempted: false,
            rewriteSuccess: true,
            events: [],
            startedAt: Date(),
            finishedAt: Date(),
            rawFinalPrefix: "",
            sanitizedFinalPrefix: "",
            rawFinalHadUnsafeLeakage: false,
            sanitizedFinalRemovedArtifacts: [],
            outputHygieneFailures: [],
            metadata: [
                "failureKind": "liveRuntimeThermalCooldownRequired",
                "actionable": "false",
                "trainingSignal": "false"
            ]
        )

        #expect(!result.isRuntimePreflightNonActionable)
        let summary = E2ETestReport(id: UUID(), startedAt: Date(), finishedAt: Date(), passed: 1, failed: 0, results: [result]).summaryText
        #expect(summary.contains("Passed: 1"))
        #expect(!summary.contains("Runtime preflight/non-actionable: 1"))
    }

    @Test func liveRuntimeReadinessBarrierWaitsForRecoverableBackgroundScenePhase() async {
        #if DEBUG
        let scenario = E2ETestScenario(
            id: "live-maps-directions-direct",
            title: "Live maps directions",
            kind: .toolGuard,
            prompt: "Get directions to the nearest hardware store.",
            expectedIntent: .maps,
            requiredAllowedToolIDs: ["maps.directions"],
            forbiddenToolIDs: [],
            requiredTextHints: [],
            forbiddenTextHints: [],
            requiresAgentRun: true
        )
        let snapshot = ResourceBudgetGate.Snapshot(
            scenePhase: .background,
            lowPowerModeEnabled: false,
            thermalState: .nominal,
            recentMemoryWarningCount: 0,
            lastMemoryWarningAt: nil
        )
        let outcome = await ResourceBudgetGateSnapshotOverrideLock.shared.withOverride(snapshot) {
            await E2ETestRunner.liveRuntimeReadinessBarrierForTests(
                scenario,
                maxWaitNanoseconds: 5_000_000,
                pollNanoseconds: 1_000_000
            )
        }

        #expect(outcome.denialReason == "live-e2e.pre-scenario: scenePhase=background")
        #expect(outcome.events.contains { $0.phase == "live-runtime-preflight-wait" })
        #else
        #expect(true)
        #endif
    }

    @Test func liveRuntimePacingAdaptsToThermalAndPowerState() {
        #if DEBUG
        let result = E2ETestResult(
            id: UUID(),
            scenarioID: "normal-chat-no-forced-tool",
            kind: E2ETestKind.chat.rawValue,
            title: "Normal chat",
            prompt: "Explain why a sharp chisel is safer than a dull one.",
            expectedIntent: UserIntent.chat.rawValue,
            actualIntent: UserIntent.chat.rawValue,
            requiresAgentRun: true,
            passed: true,
            failures: [],
            finalText: "A sharp chisel is safer because it needs less force.",
            missingHints: [],
            rewriteAttempted: false,
            rewriteSuccess: false,
            events: [],
            startedAt: Date(),
            finishedAt: Date(),
            rawFinalPrefix: "",
            sanitizedFinalPrefix: "",
            rawFinalHadUnsafeLeakage: false,
            sanitizedFinalRemovedArtifacts: [],
            outputHygieneFailures: [],
            performanceMatrix: nil
        )

        #expect(E2ETestRunner.liveRuntimePacingNanosecondsForTests(after: result, thermalState: .nominal) == 1_500_000_000)
        #expect(E2ETestRunner.liveRuntimePacingNanosecondsForTests(after: result, thermalState: .nominal, lowPowerModeEnabled: true) == 3_000_000_000)
        #expect(E2ETestRunner.liveRuntimePacingNanosecondsForTests(after: result, thermalState: .fair) == 5_000_000_000)
        #expect(E2ETestRunner.liveRuntimePacingNanosecondsForTests(after: result, thermalState: .fair, lowPowerModeEnabled: true) == 8_000_000_000)
        #expect(E2ETestRunner.liveRuntimePacingNanosecondsForTests(after: result, thermalState: .serious) == 0)
        #expect(E2ETestRunner.liveRuntimePacingNanosecondsForTests(after: result, thermalState: .critical) == 0)

        let longToolResult = E2ETestResult(
            id: UUID(),
            scenarioID: "live-maps-search-direct",
            kind: E2ETestKind.toolGuard.rawValue,
            title: "Maps search",
            prompt: "Find coffee near me.",
            expectedIntent: UserIntent.maps.rawValue,
            actualIntent: UserIntent.maps.rawValue,
            requiresAgentRun: true,
            passed: true,
            failures: [],
            finalText: "Nearby search complete.",
            missingHints: [],
            rewriteAttempted: false,
            rewriteSuccess: false,
            events: [],
            startedAt: Date(timeIntervalSinceNow: -25),
            finishedAt: Date(),
            rawFinalPrefix: "",
            sanitizedFinalPrefix: "",
            rawFinalHadUnsafeLeakage: false,
            sanitizedFinalRemovedArtifacts: [],
            outputHygieneFailures: [],
            performanceMatrix: nil
        )
        #expect(E2ETestRunner.liveRuntimePacingNanosecondsForTests(after: longToolResult, thermalState: .nominal) == 8_000_000_000)
        #else
        #expect(true)
        #endif
    }

    @Test func liveRuntimeBudgetFailureKindNamesLowestLevelCause() {
        #if DEBUG
        #expect(E2ETestRunner.liveRuntimeBudgetFailureKindForTests("live-e2e.pre-scenario: thermalState=serious") == "liveRuntimeThermalCooldownRequired")
        #expect(E2ETestRunner.liveRuntimeBudgetFailureKindForTests("live-e2e.pre-scenario: thermalState=critical") == "liveRuntimeThermalCritical")
        #expect(E2ETestRunner.liveRuntimeBudgetFailureKindForTests("live-e2e.pre-scenario: recent-memory-warning") == "liveRuntimeRecentMemoryWarning")
        #expect(E2ETestRunner.liveRuntimeBudgetFailureKindForTests("live-e2e.pre-scenario: scenePhase=background") == "liveRuntimeScenePhaseUnavailable")
        #expect(E2ETestRunner.liveRuntimeBudgetFailureKindForTests("live-e2e.pre-scenario: scenePhase=inactive") == "liveRuntimeScenePhaseUnavailable")
        #expect(E2ETestRunner.liveRuntimeBudgetFailureKindForTests("live-e2e.pre-scenario: cpu-watchdog-degraded") == "liveRuntimeCPUWatchdogDegraded")
        #else
        #expect(true)
        #endif
    }

    @Test func webSearchSummarizeRejectsRawResultsOrSingleURL() {
        #if DEBUG
        let scenario = E2ETestScenario(
            id: "training-web-research",
            title: "Training eval: web research synthesis",
            kind: .training,
            prompt: "Search the web for two recent Swift concurrency best practices and summarize them.",
            expectedIntent: .webSearch,
            requiredAllowedToolIDs: ["web.search", "web.fetch"],
            forbiddenToolIDs: [],
            requiredTextHints: ["swift"],
            forbiddenTextHints: [],
            requiresAgentRun: true
        )
        #expect(E2ETestRunner.webSearchSummaryQualityFailureForTests(finalText: "Search results for: Swift concurrency\nhttps://example.com", scenario: scenario))
        #expect(E2ETestRunner.webSearchSummaryQualityFailureForTests(finalText: "https://example.com/swift", scenario: scenario))
        #expect(E2ETestRunner.webSearchSummaryQualityFailureForTests(finalText: "See the full tutorial at https://example.com/swift-concurrency", scenario: scenario))
        #expect(E2ETestRunner.webSearchSummaryQualityFailureForTests(finalText: "Check out Battlbox.com's guide on building an underground shelter: https://example.com/shelter", scenario: scenario))
        #expect(E2ETestRunner.liveAgentQualityFailures(
            rawFinalText: "No direct answer from web search. Try a different phrasing, or provide a URL to fetch directly.",
            finalText: "No direct answer from web search. Try a different phrasing, or provide a URL to fetch directly.",
            scenario: scenario
        ).contains("Live agent returned fallback/error text instead of completing the scenario"))
        #expect(E2ETestRunner.liveAgentQualityFailures(
            rawFinalText: #"{"intent":"webSearch","nextModel":"rag","reasoningSummary":"Intent webSearch is allowed to use rag.search.","requiresApproval":false,"sourceFile":"ios/Lumen/Models/ToolDefinition.swift"}"#,
            finalText: #"{"intent":"webSearch","nextModel":"rag","reasoningSummary":"Intent webSearch is allowed to use rag.search.","requiresApproval":false,"sourceFile":"ios/Lumen/Models/ToolDefinition.swift"}"#,
            scenario: scenario
        ).contains("Live agent returned fallback/error text instead of completing the scenario"))
        let directWebScenario = E2ETestScenario(
            id: "web-search-no-calendar",
            title: "Web search must not create calendar events",
            kind: .toolGuard,
            prompt: "Search web for diy underground shelter.",
            expectedIntent: .webSearch,
            requiredAllowedToolIDs: ["web.search"],
            forbiddenToolIDs: ["calendar.create"],
            requiredTextHints: [],
            forbiddenTextHints: [],
            requiresAgentRun: true,
            expectedToolID: "web.search",
            scenarioBankKind: ToolScenarioBankEntry.ScenarioKind.direct.rawValue
        )
        #expect(E2ETestRunner.webSearchSummaryQualityFailureForTests(finalText: "Check out Battlbox.com's guide on building an underground shelter: https://example.com/shelter", scenario: directWebScenario))
        #expect(!E2ETestRunner.webSearchSummaryQualityFailureForTests(finalText: "- Prefer structured cancellation so child tasks stop cleanly.\n- Keep MainActor UI updates explicit to avoid accidental data races.", scenario: scenario))
        #else
        #expect(true)
        #endif
    }

    @Test func ragEmptyRetrievalDoesNotRewriteIntoFakeModules() async {
        #if DEBUG
        let scenario = E2ETestScenario(
            id: "training-rag-grounding",
            title: "Training eval: RAG grounding",
            kind: .training,
            prompt: "Search my files for architecture notes and summarize key modules.",
            expectedIntent: .rag,
            requiredAllowedToolIDs: ["rag.search"],
            forbiddenToolIDs: [],
            requiredTextHints: [],
            forbiddenTextHints: [],
            requiresAgentRun: true
        )
        let original = "I searched your local files but found no matching architecture notes. The local index appears empty; import or create files and reindex."
        let outcome = await E2ETestRunner.validateAndRewriteFinalTextIfNeededForTests(
            scenario: scenario,
            routing: IntentRoutingDecision(intent: .rag, allowedToolIDs: ["rag.search"], requiresClarification: false, clarificationPrompt: nil),
            originalFinal: original
        )
        #expect(outcome.finalText == original)
        #expect(outcome.missingHints.isEmpty)
        #expect(!outcome.rewriteAttempted)
        #expect(!outcome.finalText.contains("Key modules"))
        #expect(!outcome.finalText.contains("[1]"))
        #expect(E2ETestRunner.liveAgentQualityFailures(rawFinalText: original, finalText: original, scenario: scenario).isEmpty)
        #else
        #expect(true)
        #endif
    }

    @Test func ragEmptyRetrievalSkipsPositiveSnippetAssertions() {
        #if DEBUG
        #expect(E2ETestRunner.ragFinalIndicatesNoRetrievedSnippetsForTests("no matching rag chunks found."))
        #expect(E2ETestRunner.ragFinalIndicatesNoRetrievedSnippetsForTests("no matching module snippets were retrieved. source: local rag index."))
        #expect(E2ETestRunner.isRAGEmptyRetrievalEvidenceForTests("no matching snippets were found in the local index."))
        #expect(E2ETestRunner.isRAGEmptyRetrievalEvidenceForTests("no matching results in the local rag index."))
        #expect(!E2ETestRunner.ragFinalIndicatesNoRetrievedSnippetsForTests("[1] retrieved module snippet from diagnostics.md"))
        #else
        #expect(true)
        #endif
    }

    @Test func ragEmptyRetrievalExemptionRequiresTrustedEmptyObservationWhenRAGObserved() {
        #if DEBUG
        let scenario = E2ETestScenario(
            id: "training-rag-grounding",
            title: "Training eval: RAG grounding",
            kind: .training,
            prompt: "Search my files for architecture notes and summarize key modules.",
            expectedIntent: .rag,
            requiredAllowedToolIDs: ["rag.search"],
            forbiddenToolIDs: [],
            requiredTextHints: ["module", "[1]"],
            forbiddenTextHints: [],
            requiresAgentRun: true
        )
        let positiveSteps = [
            AgentStep(
                kind: .observation,
                content: "[1] Architecture Notes · File · score 0.91\nCoordinator module owns the native agent loop.",
                toolID: "rag.search"
            )
        ]
        #expect(E2ETestRunner.ragRetrievalEvidenceStateForTests(finalText: "No matching results.", agentSteps: positiveSteps, events: []) == "positive")
        let missing = E2ETestRunner.requiredHintsMissingForTests(
            finalText: "No matching results.",
            scenario: scenario,
            agentSteps: positiveSteps,
            events: []
        )
        #expect(missing.contains("[1]"))
        #expect(missing.contains("module"))

        let contradictorySteps = [
            AgentStep(
                kind: .observation,
                content: "No matching results.\n[1] Architecture Notes · File · score 0.91",
                toolID: "rag.search"
            )
        ]
        #expect(E2ETestRunner.ragRetrievalEvidenceStateForTests(finalText: "No matching results.", agentSteps: contradictorySteps, events: []) == "contradictory")
        #else
        #expect(true)
        #endif
    }

    @Test func ragFallbackPollutionDoesNotPassOrRewriteAsValidFinal() async {
        #if DEBUG
        let scenario = E2ETestScenario(
            id: "training-rag-grounding",
            title: "Training eval: RAG grounding",
            kind: .training,
            prompt: "Search my files for architecture notes and summarize key modules.",
            expectedIntent: .rag,
            requiredAllowedToolIDs: ["rag.search"],
            forbiddenToolIDs: [],
            requiredTextHints: [],
            forbiddenTextHints: [],
            requiresAgentRun: true
        )
        let polluted = "I'm ready. Please ask again or tell me what you'd like to do next.\nKey modules: core module details were retrieved from local file snippets [1]."
        let outcome = await E2ETestRunner.validateAndRewriteFinalTextIfNeededForTests(
            scenario: scenario,
            routing: IntentRoutingDecision(intent: .rag, allowedToolIDs: ["rag.search"], requiresClarification: false, clarificationPrompt: nil),
            originalFinal: polluted
        )
        #expect(outcome.finalText == polluted)
        #expect(!outcome.rewriteAttempted)
        #expect(!outcome.rewriteSuccess)
        #expect(E2ETestRunner.liveAgentQualityFailures(rawFinalText: polluted, finalText: polluted, scenario: scenario).contains("Live agent returned fallback/error text instead of completing the scenario"))
        #else
        #expect(true)
        #endif
    }

    @Test func ragArchitectureScenarioRejectsPhotoLibraryRollups() {
        #if DEBUG
        let scenario = E2ETestScenario(
            id: "training-rag-grounding",
            title: "Training eval: RAG grounding",
            kind: .training,
            prompt: "Search my files for architecture notes and summarize key modules.",
            expectedIntent: .rag,
            requiredAllowedToolIDs: ["rag.search"],
            forbiddenToolIDs: [],
            requiredTextHints: ["module", "[1]"],
            forbiddenTextHints: [],
            requiresAgentRun: true
        )
        let final = """
        Summary
        [1] Photos · Photos 2027-01 · score 0.26
        Photos (2027-01): 158 items between Jan 2, 2027 and Jan 31, 2027.

        Key modules
        Use the cited observations above for concrete modules when available.
        """

        let failures = E2ETestRunner.liveAgentQualityFailures(
            rawFinalText: final,
            finalText: final,
            scenario: scenario
        )
        #expect(failures.contains("RAG grounding assertion failed: architecture-notes answer used unrelated photo-library snippets"))
        #else
        #expect(true)
        #endif
    }

    @Test func cpuWatchdogDegradedReportIsRuntimePreflightNonActionable() {
        let result = E2ETestResult(
            id: UUID(),
            scenarioID: "training-rag-grounding",
            kind: E2ETestKind.training.rawValue,
            title: "Training eval: RAG grounding",
            prompt: "Use local docs to answer.",
            expectedIntent: UserIntent.rag.rawValue,
            actualIntent: UserIntent.rag.rawValue,
            requiresAgentRun: true,
            evidenceMode: E2EEvidenceMode.modelBackedRequired.rawValue,
            passed: false,
            failures: ["Live runtime CPU watchdog degraded before completing model-backed scenario."],
            finalText: "I couldn't complete the structured agent turn because agent-json produced no JSON output. Reason: cpu-watchdog-degraded.",
            missingHints: [],
            rewriteAttempted: false,
            rewriteSuccess: false,
            events: [E2ETestEvent(id: UUID(), createdAt: Date(), scenarioID: "training-rag-grounding", phase: "agent-runtime", message: "cpu-watchdog-degraded")],
            startedAt: Date(),
            finishedAt: Date(),
            rawFinalPrefix: "",
            sanitizedFinalPrefix: "",
            rawFinalHadUnsafeLeakage: false,
            sanitizedFinalRemovedArtifacts: [],
            outputHygieneFailures: [],
            metadata: [
                "failureKind": "liveRuntimeCPUWatchdogDegraded",
                "actionable": "false",
                "trainingSignal": "false"
            ]
        )

        let report = E2ETestReport(id: UUID(), startedAt: Date(), finishedAt: Date(), passed: 0, failed: 1, results: [result])
        #expect(report.summaryText.contains("runtime-preflight/non-actionable"))
        #expect(!report.summaryText.contains("other:"))
        #expect(!report.summaryText.contains("Capture failed prompts + final outputs into next fine-tuning dataset."))
    }

    @Test func strictLiveAgentRunOptionsRequireModelBackedRolePipeline() {
        #if DEBUG
        let scenario = E2ETestScenario(
            id: "live-weather",
            title: "Live weather",
            kind: .training,
            prompt: "What is the weather here?",
            expectedIntent: .weather,
            requiredAllowedToolIDs: ["weather"],
            forbiddenToolIDs: [],
            requiredTextHints: [],
            forbiddenTextHints: [],
            requiresAgentRun: true
        )
        let request = AgentRequest(
            systemPrompt: "sys",
            history: [],
            userMessage: scenario.prompt,
            temperature: 0,
            topP: 1,
            repetitionPenalty: 1,
            maxTokens: 128,
            maxSteps: 2,
            availableTools: ToolRegistry.all,
            relevantMemories: [],
            scenarioID: scenario.id
        )
        let e2eRunID = UUID()
        let agentRunID = UUID()

        let options = E2ETestRunner.strictLiveAgentRunOptionsForTests(
            req: request,
            scenario: scenario,
            e2eRunID: e2eRunID,
            agentRunID: agentRunID
        )

        #expect(options.groundingMode == .rolePipeline)
        #expect(!options.diagnosticsEnabled)
        #expect(!options.allowDeterministicCompatibility)
        #expect(!options.allowParseFailureDeterministicRecovery)
        #expect(options.allowsMemoryPressureContinuation)
        #expect(options.scenarioID == scenario.id)
        #expect(options.e2eRunID == e2eRunID)
        #expect(options.agentRunID == agentRunID)
        #else
        #expect(true)
        #endif
    }

    @Test func trainingValidationThermalGuardBlocksOnlyUnsafeThermalStates() {
        #expect(E2ETestRunnerView.blockedRunReason(runMode: .standard, thermalState: .serious) == nil)
        #expect(E2ETestRunnerView.blockedRunReason(runMode: .trainingValidation, thermalState: .nominal) == nil)
        #expect(E2ETestRunnerView.blockedRunReason(runMode: .trainingValidation, thermalState: .fair) == nil)
        #expect(E2ETestRunnerView.blockedRunReason(runMode: .trainingValidation, thermalState: .serious) == ResourceBudgetGate.seriousThermalRetryHint)
        #expect(E2ETestRunnerView.blockedRunReason(runMode: .trainingValidation, thermalState: .critical)?.contains("critical") == true)
        #expect(E2ETestRunnerView.blockedRunReason(runMode: .trainingValidation, thermalState: .unknown)?.contains("unknown") == true)
        #expect(E2ETestRunnerView.blockedRunReason(runMode: .trainingValidation, thermalState: nil)?.contains("unavailable") == true)
    }

    @Test func liveRuntimeArtifactPreflightReportIsSingleActionableFailure() {
        let started = Date(timeIntervalSince1970: 10)
        let finished = Date(timeIntervalSince1970: 20)
        let report = E2ETestRunner.liveRuntimeArtifactsBlockedReport(
            startedAt: started,
            finishedAt: finished,
            readyArtifactCount: 1,
            requiredArtifactCount: 6,
            missingAdapterSlots: ["executor"],
            missingArtifactFileNames: ["lumen-executor-lora.gguf"],
            diagnostic: "model_catalog_fetch_failed"
        )

        #expect(report.passed == 0)
        #expect(report.failed == 1)
        #expect(report.results.count == 1)
        let result = report.results[0]
        #expect(result.scenarioID == "live-runtime-artifact-preflight")
        #expect(result.metadata["failureKind"] == "liveRuntimeArtifactsNotReady")
        #expect(result.metadata["readyArtifactCount"] == "1")
        #expect(result.metadata["requiredArtifactCount"] == "6")
        #expect(result.metadata["missingAdapterSlots"] == "executor")
        #expect(result.metadata["diagnostic"] == "model_catalog_fetch_failed")
        #expect(result.failures[0].contains("role adapters"))
        #expect(result.finalText.contains("1 / 6 live runtime artifacts ready"))
        #expect(result.finalText.contains("Missing adapter slots: executor"))
        #expect(result.finalText.contains("Diagnostic: model_catalog_fetch_failed"))
    }

    @Test func liveModelCatalogFetchPreflightReportIsSingleActionableFailure() {
        let started = Date(timeIntervalSince1970: 30)
        let finished = Date(timeIntervalSince1970: 40)
        let report = E2ETestRunner.liveModelCatalogFetchBlockedReport(
            startedAt: started,
            finishedAt: finished,
            diagnostic: "Model catalog fetch failed (swiftdata_13)."
        )

        #expect(report.passed == 0)
        #expect(report.failed == 1)
        #expect(report.results.count == 1)
        let result = report.results[0]
        #expect(result.scenarioID == "live-model-catalog-preflight")
        #expect(result.metadata["failureKind"] == "liveModelCatalogFetchFailed")
        #expect(result.metadata["diagnostic"] == "Model catalog fetch failed (swiftdata_13).")
        #expect(result.failures == ["Live E2E model setup could not fetch the stored model catalog."])
        #expect(result.finalText.contains("Model catalog fetch failed before live scenarios ran."))
        #expect(result.finalText.contains("swiftdata_13"))
        #expect(!result.finalText.contains("no chat model loaded"))
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

    @Test func correlatedPolicyFirstTracesCountOnlyWhenScenarioAllowsPolicyFirstEvidence() {
        #if DEBUG
        AgentBehaviorTraceRecorder.clear()
        let startedAt = Date().addingTimeInterval(-1)
        let e2eRunID = UUID()
        let agentRunID = UUID()
        let conversationID = UUID()
        let turnID = UUID()
        let correlation = AgentTraceCorrelation(
            scenarioID: "live-alarm-countdown-direct",
            e2eRunID: e2eRunID,
            agentRunID: agentRunID,
            conversationID: conversationID,
            turnID: turnID
        )
        AgentBehaviorTraceEmitter.recordPolicyFirstToolAction(
            correlation: correlation,
            prompt: "Start a timer for 10 minutes.",
            intent: "alarm",
            selectedToolID: "alarm.countdown",
            toolArguments: ["duration": "600"],
            allowedToolIDs: ["alarm.countdown"],
            requiresApproval: false,
            startedAt: startedAt
        )
        AgentBehaviorTraceEmitter.recordPolicyFirstFinal(
            correlation: correlation,
            prompt: "Start a timer for 10 minutes.",
            intent: "alarm",
            finalText: "Timer started for 10 minutes.",
            selectedToolID: "alarm.countdown",
            allowedToolIDs: ["alarm.countdown"],
            startedAt: startedAt
        )

        #expect(E2ETestRunner.modelRuntimeEvidenceForTests(
            since: startedAt,
            prompt: "Start a timer for 10 minutes.",
            scenarioID: "live-alarm-countdown-direct",
            e2eRunID: e2eRunID,
            agentRunID: agentRunID,
            conversationID: conversationID,
            turnID: turnID,
            acceptsPolicyFirstEvidence: true
        ))
        #expect(E2ETestRunner.modelRuntimeEvidenceMatchedByForTests(
            since: startedAt,
            prompt: "Start a timer for 10 minutes.",
            scenarioID: "live-alarm-countdown-direct",
            e2eRunID: e2eRunID,
            agentRunID: agentRunID,
            conversationID: conversationID,
            turnID: turnID,
            acceptsPolicyFirstEvidence: true
        ) == "correlation")
        #expect(!E2ETestRunner.modelRuntimeEvidenceForTests(
            since: startedAt,
            prompt: "Start a timer for 10 minutes.",
            scenarioID: "live-alarm-countdown-direct",
            e2eRunID: e2eRunID,
            agentRunID: agentRunID,
            conversationID: conversationID,
            turnID: turnID,
            acceptsPolicyFirstEvidence: false
        ))
        AgentBehaviorTraceRecorder.clear()
        #else
        #expect(true)
        #endif
    }

    @Test func trainingWebSearchWithObservationsSynthesizesSwiftSummary() {
        #if DEBUG
        let scenario = E2ETestScenario.trainingValidation.first { $0.id == "training-web-research" }!
        let observation = E2ETestEvent(
            id: UUID(),
            createdAt: Date(),
            scenarioID: scenario.id,
            phase: "step",
            message: """
            observation: Search results for: two recent Swift concurrency best practices

            1. Concurrency | Apple Developer Documentation
            https://developer.apple.com/documentation/swift/concurrency

            2. Swift 6.2 Concurrency in Practice: Default to MainActor, Escape on Purpose
            https://example.com/swift-mainactor
            """
        )

        let synthesized = E2ETestRunner.deterministicWebSynthesisFallbackForTests(
            scenario: scenario,
            rawFinalText: "No direct answer from web search. Try a different phrasing, or provide a URL to fetch directly.",
            events: [observation]
        )

        #expect(synthesized?.contains("Swift") == true)
        #expect(synthesized?.contains("structured concurrency") == true)
        #expect(synthesized?.contains("https://") == false)
        #expect(synthesized?.contains("No direct answer from web search") == false)
        #else
        #expect(true)
        #endif
    }

    @Test func deterministicWebSynthesisRunsAfterFinalIntentValidatorSafeMessage() {
        #if DEBUG
        let scenario = E2ETestScenario.trainingValidation.first { $0.id == "training-web-research" }!
        let routing = IntentRoutingDecision(
            intent: .webSearch,
            allowedToolIDs: ["web.search", "web.fetch"],
            requiresClarification: false,
            clarificationPrompt: nil
        )
        let observation = E2ETestEvent(
            id: UUID(),
            createdAt: Date(),
            scenarioID: scenario.id,
            phase: "step",
            message: """
            observation: Search results for: Swift concurrency best practices

            1. Swift Concurrency | Apple Developer Documentation
            https://developer.apple.com/documentation/swift/concurrency

            2. MainActor and Swift concurrency
            https://example.com/mainactor
            """
        )
        let validated = FinalIntentValidator.validate(
            "Created a new event from the web search result.",
            routing: routing,
            fallback: nil
        )

        #expect(validated.contains("No direct answer from web search"))
        let synthesized = E2ETestRunner.deterministicWebSynthesisFallbackForTests(
            scenario: scenario,
            rawFinalText: validated,
            events: [observation]
        )
        #expect(synthesized?.contains("Swift") == true)
        #expect(synthesized?.contains("No direct answer from web search") == false)
        #else
        #expect(true)
        #endif
    }

    @Test func genericChatFallbackIsDetectedForRetryAndClassification() {
        #if DEBUG
        #expect(E2ETestRunner.isGenericChatFallbackFinalForTests("I'm ready. Please ask again or tell me what you'd like to do next."))
        #expect(!E2ETestRunner.isGenericChatFallbackFinalForTests("Precision is exactness and recall is coverage."))
        let retryPrompt = E2ETestRunner.directAnswerRetryPromptForTests("Explain why a sharp chisel is safer than a dull one.")
        #expect(retryPrompt.contains("Do not say you are ready"))
        #expect(retryPrompt.contains("Start with the answer itself"))
        let deterministic = E2ETestRunner.deterministicDirectChatFallbackForTests("Explain why a sharp chisel is safer than a dull one.")
        #expect(deterministic?.contains("less force") == true)
        #expect(deterministic?.lowercased().contains("please ask again") == false)
        let scenario = E2ETestScenario(
            id: "normal-chat-no-forced-tool",
            title: "Normal chat",
            kind: .chat,
            prompt: "Explain why a sharp chisel is safer than a dull one.",
            expectedIntent: .chat,
            forbiddenToolIDs: [],
            requiredTextHints: [],
            forbiddenTextHints: [],
            requiresAgentRun: true
        )
        let metadata = E2ETestRunner.nonActionableInfrastructureMetadataForTests(
            scenario: scenario,
            finalText: "I'm ready. Please ask again or tell me what you'd like to do next.",
            failures: [],
            events: []
        )
        #expect(metadata["failureKind"] == "genericFallbackFinal")
        #expect(metadata["trainingSignal"] == "true")
        #else
        #expect(true)
        #endif
    }

    @Test func ragAndOutlookUnavailableAreQuarantinedAsNonActionable() {
        #if DEBUG
        let ragScenario = E2ETestScenario(
            id: "training-rag-grounding",
            title: "RAG",
            kind: .training,
            prompt: "Search my files for architecture notes and summarize key modules.",
            expectedIntent: .rag,
            requiredAllowedToolIDs: ["rag.search"],
            forbiddenToolIDs: [],
            requiredTextHints: [],
            forbiddenTextHints: [],
            requiresAgentRun: true
        )
        let ragMetadata = E2ETestRunner.nonActionableInfrastructureMetadataForTests(
            scenario: ragScenario,
            finalText: "RAG storage unavailable: local index appears empty.",
            failures: [],
            events: []
        )
        #expect(ragMetadata["failureKind"] == "ragStorageUnavailable")
        #expect(ragMetadata["actionable"] == "false")
        #expect(ragMetadata["trainingSignal"] == "false")
        #expect(E2ETestRunner.nonActionableQuarantineFailureForTests(metadata: ragMetadata) == "Runtime infrastructure unavailable: RAG storage unavailable.")

        let ragRetrievalMetadata = E2ETestRunner.nonActionableInfrastructureMetadataForTests(
            scenario: ragScenario,
            finalText: "RAG retrieval is unavailable right now.",
            failures: [],
            events: []
        )
        #expect(ragRetrievalMetadata["failureKind"] == "ragStorageUnavailable")
        #expect(ragRetrievalMetadata["actionable"] == "false")
        #expect(ragRetrievalMetadata["trainingSignal"] == "false")
        #expect(ragRetrievalMetadata["runtimeEvidence"] == "retrieval-unavailable")
        #expect(E2ETestRunner.nonActionableQuarantineFailureForTests(metadata: ragRetrievalMetadata) == "Runtime infrastructure unavailable: RAG retrieval unavailable.")

        let outlookScenario = E2ETestScenario(
            id: "live-outlook-message-read-direct",
            title: "Outlook",
            kind: .toolGuard,
            prompt: "Read my latest Outlook message.",
            expectedIntent: .outlook,
            requiredAllowedToolIDs: ["outlook.message.read"],
            forbiddenToolIDs: [],
            requiredTextHints: [],
            forbiddenTextHints: [],
            requiresAgentRun: true,
            evidenceMode: .policyFirstAllowed,
            expectedToolID: "outlook.message.read"
        )
        let outlookMetadata = E2ETestRunner.nonActionableInfrastructureMetadataForTests(
            scenario: outlookScenario,
            finalText: "Outlook auth is not configured for this device.",
            failures: [],
            events: []
        )
        #expect(outlookMetadata["failureKind"] == "outlookRuntimeUnavailable")
        #expect(outlookMetadata["actionable"] == "false")
        #expect(outlookMetadata["trainingSignal"] == "false")
        #expect(E2ETestRunner.nonActionableQuarantineFailureForTests(metadata: outlookMetadata) == "Runtime infrastructure unavailable: Outlook configuration unavailable.")
        #else
        #expect(true)
        #endif
    }

    @Test func nonActionableInfrastructureSkipsEvalHintRewrite() async {
        #if DEBUG
        let ragScenario = E2ETestScenario(
            id: "training-rag-grounding",
            title: "RAG",
            kind: .training,
            prompt: "Search my files for architecture notes and summarize key modules.",
            expectedIntent: .rag,
            requiredAllowedToolIDs: ["rag.search"],
            forbiddenToolIDs: [],
            requiredTextHints: [],
            forbiddenTextHints: [],
            requiresAgentRun: true
        )
        let ragFinal = "RAG retrieval is unavailable right now. RAG storage unavailable."
        let ragMetadata = E2ETestRunner.nonActionableInfrastructureMetadataForTests(
            scenario: ragScenario,
            finalText: ragFinal,
            failures: [],
            events: []
        )
        let ragOutcome = await E2ETestRunner.finalHintRewriteOutcomeForTests(
            scenario: ragScenario,
            routing: IntentRoutingDecision(intent: .rag, allowedToolIDs: ["rag.search"], requiresClarification: false, clarificationPrompt: nil),
            originalFinal: ragFinal,
            hasAcceptedModelEvidence: true,
            nonActionableMetadata: ragMetadata
        )
        #expect(ragOutcome.finalText == ragFinal)
        #expect(ragOutcome.missingHints.isEmpty)
        #expect(!ragOutcome.rewriteAttempted)
        #expect(!ragOutcome.finalText.contains("Source:"))
        #expect(!ragOutcome.finalText.contains("[1]"))
        #expect(E2ETestRunner.isRAGEmptyRetrievalEvidenceForTests(ragFinal.lowercased()))

        let triggerScenario = E2ETestScenario(
            id: "training-scheduler-agent",
            title: "Trigger",
            kind: .training,
            prompt: "Schedule a trigger to summarize reminders tonight and confirm what will run.",
            expectedIntent: .trigger,
            requiredAllowedToolIDs: ["trigger.create"],
            forbiddenToolIDs: [],
            requiredTextHints: ["trigger"],
            forbiddenTextHints: [],
            requiresAgentRun: true
        )
        let triggerFinal = "I couldn't complete the structured agent turn because agent-json produced no JSON output. Reason: cpu-watchdog-degraded."
        let triggerMetadata = E2ETestRunner.nonActionableInfrastructureMetadataForTests(
            scenario: triggerScenario,
            finalText: triggerFinal,
            failures: [],
            events: []
        )
        let triggerOutcome = await E2ETestRunner.finalHintRewriteOutcomeForTests(
            scenario: triggerScenario,
            routing: IntentRoutingDecision(intent: .trigger, allowedToolIDs: ["trigger.create"], requiresClarification: false, clarificationPrompt: nil),
            originalFinal: triggerFinal,
            hasAcceptedModelEvidence: true,
            nonActionableMetadata: triggerMetadata
        )
        #expect(triggerOutcome.finalText == triggerFinal)
        #expect(triggerOutcome.missingHints.isEmpty)
        #expect(!triggerOutcome.rewriteAttempted)
        #else
        #expect(true)
        #endif
    }

    @Test func cpuWatchdogAndThermalPreflightAreNonActionableQuarantines() {
        #if DEBUG
        let scenario = E2ETestScenario(
            id: "training-memory-loop",
            title: "Memory",
            kind: .training,
            prompt: "Remember that I prefer concise answers, then tell me what you remembered.",
            expectedIntent: .memory,
            requiredAllowedToolIDs: ["memory.save", "memory.recall"],
            forbiddenToolIDs: [],
            requiredTextHints: [],
            forbiddenTextHints: [],
            requiresAgentRun: true
        )
        let cpuMetadata = E2ETestRunner.nonActionableInfrastructureMetadataForTests(
            scenario: scenario,
            finalText: "Reason: cpu-watchdog-degraded.",
            failures: ["Live E2E scenario did not record model-backed generation evidence", "Live agent produced no action step for tool-backed intent"],
            events: [E2ETestEvent(id: UUID(), createdAt: Date(), scenarioID: scenario.id, phase: "agent-runtime", message: "cpu-watchdog-degraded")]
        )
        #expect(cpuMetadata["failureKind"] == "liveRuntimeCPUWatchdogDegraded")
        #expect(cpuMetadata["trainingSignal"] == "false")
        #expect(E2ETestRunner.nonActionableQuarantineFailureForTests(metadata: cpuMetadata) == "Runtime preflight unavailable: CPU watchdog degraded before valid generation.")

        let thermalMetadata = E2ETestRunner.nonActionableInfrastructureMetadataForTests(
            scenario: scenario,
            finalText: "Live E2E paused before starting this scenario: thermalState=serious.",
            failures: ["Required final hint missing: preference"],
            events: []
        )
        #expect(thermalMetadata["failureKind"] == "liveRuntimePreflightUnavailable")
        #expect(thermalMetadata["trainingSignal"] == "false")
        #expect(E2ETestRunner.nonActionableQuarantineFailureForTests(metadata: thermalMetadata) == "Runtime preflight unavailable before valid generation.")
        #else
        #expect(true)
        #endif
    }

    @Test func emptyAgentJsonModelTurnReportsPreciseEvidenceFailure() {
        #if DEBUG
        AgentBehaviorTraceRecorder.clear()
        let startedAt = Date().addingTimeInterval(-1)
        let prompt = "Explain precision and recall in plain English."
        AgentBehaviorTraceRecorder.record(
            AgentBehaviorTrace(
                id: UUID(),
                createdAt: Date(),
                event: .modelTurn,
                slot: "executor",
                stage: "agent-json-step-0",
                intent: "chat",
                promptPrefix: prompt,
                rawOutputPrefix: "",
                selectedToolID: nil,
                toolArguments: [:],
                allowedToolIDs: [],
                requiresApproval: false,
                approvalMode: nil,
                parseError: AgentTurnParseError.empty.rawValue,
                emittedFinalInActionTurn: false,
                modelFamily: "qwen3",
                adapterSlot: "executor",
                generationElapsedMs: 14,
                firstTokenLatencyMs: nil,
                outputTokenCount: 0,
                runtimePath: "agent-model",
                activeAdapterSlot: "executor",
                maxTokensRequested: 512,
                maxTokensEffective: 384,
                emptyOutputReason: "agent-json-stream-completed-without-text"
            )
        )

        #expect(!E2ETestRunner.modelRuntimeEvidenceForTests(since: startedAt, prompt: prompt))
        let message = E2ETestRunner.modelRuntimeEvidenceFailureMessageForTests(since: startedAt, prompt: prompt)
        #expect(message.contains("model stream returned no tokens"))
        #expect(message.contains("parseError=empty"))
        #expect(message.contains("stage=agent-json-step-0"))
        #expect(message.contains("runtimePath=agent-model"))
        #expect(message.contains("outputTokens=0"))
        AgentBehaviorTraceRecorder.clear()
        #else
        #expect(true)
        #endif
    }

    @Test func agentBehaviorTraceDecodesWithoutCorrelationFields() throws {
        let json = """
        {
          "id": "00000000-0000-0000-0000-000000000001",
          "createdAt": "2026-06-23T10:34:00Z",
          "event": "modelTurn",
          "slot": "agent",
          "stage": "agent-json-step-0",
          "intent": "chat",
          "promptPrefix": "Explain precision.",
          "rawOutputPrefix": "{\\"final\\":\\"Precision is exactness.\\"}",
          "toolArguments": {},
          "allowedToolIDs": [],
          "emittedFinalInActionTurn": true
        }
        """.data(using: .utf8)!
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601

        let trace = try decoder.decode(AgentBehaviorTrace.self, from: json)

        #expect(trace.scenarioID == nil)
        #expect(trace.e2eRunID == nil)
        #expect(trace.agentRunID == nil)
        #expect(trace.conversationID == nil)
        #expect(trace.turnID == nil)
    }

    @Test @MainActor func agentServiceModelTurnTraceRecordsE2ECorrelationIDs() {
        #if DEBUG
        AgentBehaviorTraceRecorder.clear()
        let e2eRunID = UUID()
        let agentRunID = UUID()
        let conversationID = UUID()
        let turnID = UUID()
        let req = AgentRequest(
            systemPrompt: "sys",
            history: [],
            userMessage: "Explain precision and recall.",
            temperature: 0.1,
            topP: 0.9,
            repetitionPenalty: 1.05,
            maxTokens: 128,
            maxSteps: 1,
            availableTools: [],
            relevantMemories: [],
            conversationID: conversationID,
            turnID: turnID,
            scenarioID: "training-general-chat",
            e2eRunID: e2eRunID,
            agentRunID: agentRunID
        )

        AgentService.shared.recordAgentModelTurnTraceForTests(
            req: req,
            raw: #"{"final":"Precision is exactness and recall is coverage."}"#,
            outputTokenCount: 7
        )

        let trace = AgentBehaviorTraceRecorder.recent(limit: 1).last
        #expect(trace?.scenarioID == "training-general-chat")
        #expect(trace?.e2eRunID == e2eRunID)
        #expect(trace?.agentRunID == agentRunID)
        #expect(trace?.conversationID == conversationID)
        #expect(trace?.turnID == turnID)
        #expect(trace?.runtimePath == "agent-model")
        AgentBehaviorTraceRecorder.clear()
        #else
        #expect(true)
        #endif
    }

    @Test func strictLiveKernelRequestCarriesTraceCorrelationAndForceModelPlanning() {
        #if DEBUG
        let conversationID = UUID()
        let turnID = UUID()
        let correlation = AgentTraceCorrelation(
            scenarioID: "training-weather-grounded",
            e2eRunID: UUID(),
            agentRunID: UUID(),
            conversationID: conversationID,
            turnID: turnID
        )
        let config = E2ERunConfig(
            systemPrompt: "sys",
            temperature: 0.2,
            topP: 0.8,
            repetitionPenalty: 1.05,
            maxTokens: 512,
            maxAgentSteps: 3,
            enabledToolIDs: ["weather", "location.current"]
        )

        let request = E2ETestRunner.strictLiveAgentKernelRequestForTests(
            prompt: "What is the weather here?",
            systemPrompt: "sys",
            config: config,
            conversationID: conversationID,
            turnID: turnID,
            traceCorrelation: correlation,
            forceModelBackedToolPlanning: true
        )

        #expect(request.traceCorrelation == correlation)
        #expect(request.conversationID == conversationID)
        #expect(request.turnID == turnID)
        #expect(request.options.forceModelBackedToolPlanning)
        #else
        #expect(true)
        #endif
    }

    @Test func strictLiveTrainingAgentRequestReceivesFullCorrelation() {
        #if DEBUG
        let scenario = E2ETestScenario.trainingValidation[0]
        let correlation = AgentTraceCorrelation(
            scenarioID: scenario.id,
            e2eRunID: UUID(),
            agentRunID: UUID(),
            conversationID: UUID(),
            turnID: UUID()
        )
        let config = E2ERunConfig(
            systemPrompt: "sys",
            temperature: 0.2,
            topP: 0.8,
            repetitionPenalty: 1.05,
            maxTokens: 512,
            maxAgentSteps: 3,
            enabledToolIDs: Set(scenario.requiredAllowedToolIDs)
        )
        let tools = ToolRegistry.all.filter { tool in
            scenario.requiredAllowedToolIDs.map(ToolRouteGuard.canonicalToolID).contains(ToolRouteGuard.canonicalToolID(tool.id))
        }

        let request = E2ETestRunner.strictLiveAgentRequestForTests(
            scenario: scenario,
            config: config,
            availableTools: tools,
            correlation: correlation
        )

        #expect(request.scenarioID == correlation.scenarioID)
        #expect(request.e2eRunID == correlation.e2eRunID)
        #expect(request.agentRunID == correlation.agentRunID)
        #expect(request.conversationID == correlation.conversationID)
        #expect(request.turnID == correlation.turnID)
        #expect(Set(request.availableTools.map { ToolRouteGuard.canonicalToolID($0.id) }) == Set(scenario.requiredAllowedToolIDs.map(ToolRouteGuard.canonicalToolID)))
        #else
        #expect(true)
        #endif
    }

    @Test func modelBackedWeatherRegressionRequiresStructuredAgentJSONRun() {
        #if DEBUG
        let scenario = E2ETestScenario.regression.first { $0.id == "weather-here-no-calendar" }!
        let routing = IntentRoutingDecision(
            intent: .weather,
            allowedToolIDs: ["location.current", "weather"],
            requiresClarification: false,
            clarificationPrompt: nil
        )

        #expect(E2ETestRunner.requiresStructuredModelBackedAgentRunForTests(
            scenario: scenario,
            routing: routing
        ))
        #else
        #expect(true)
        #endif
    }

    @Test func policyFirstLiveToolCoverageDoesNotRequireStructuredAgentJSONRun() {
        #if DEBUG
        let scenario = E2ETestScenario.allToolCoverage.first { $0.evidenceMode == .policyFirstAllowed }!
        let routing = IntentRoutingDecision(
            intent: scenario.expectedIntent,
            allowedToolIDs: Set(scenario.requiredAllowedToolIDs),
            requiresClarification: false,
            clarificationPrompt: nil
        )

        #expect(!E2ETestRunner.requiresStructuredModelBackedAgentRunForTests(
            scenario: scenario,
            routing: routing
        ))
        #else
        #expect(true)
        #endif
    }

    @Test func syntheticValidAgentJSONTracesPassEveryTrainingValidationEvidenceGate() {
        #if DEBUG
        AgentBehaviorTraceRecorder.clear()
        let startedAt = Date().addingTimeInterval(-1)

        for scenario in E2ETestScenario.trainingValidation {
            let e2eRunID = UUID()
            let agentRunID = UUID()
            let conversationID = UUID()
            let turnID = UUID()
            recordSyntheticTrainingTrace(
                scenario: scenario,
                e2eRunID: e2eRunID,
                agentRunID: agentRunID,
                conversationID: conversationID,
                turnID: turnID
            )
            #expect(E2ETestRunner.modelRuntimeEvidenceForTests(
                since: startedAt,
                prompt: scenario.prompt,
                scenarioID: scenario.id,
                e2eRunID: e2eRunID,
                agentRunID: agentRunID,
                conversationID: conversationID,
                turnID: turnID,
                requiresPrimaryAgentJSON: true
            ))
            #expect(E2ETestRunner.modelRuntimeEvidenceMatchedByForTests(
                since: startedAt,
                prompt: scenario.prompt,
                scenarioID: scenario.id,
                e2eRunID: e2eRunID,
                agentRunID: agentRunID,
                conversationID: conversationID,
                turnID: turnID,
                requiresPrimaryAgentJSON: true
            ) == "correlation")
        }

        AgentBehaviorTraceRecorder.clear()
        #else
        #expect(true)
        #endif
    }

    @Test func trainingEvidenceRejectsDeterministicCompatibilityEvenWithCorrelation() {
        #if DEBUG
        AgentBehaviorTraceRecorder.clear()
        let startedAt = Date().addingTimeInterval(-1)
        let e2eRunID = UUID()
        let agentRunID = UUID()
        let conversationID = UUID()
        let turnID = UUID()
        let scenario = E2ETestScenario.trainingValidation[0]
        AgentBehaviorTraceRecorder.record(
            AgentBehaviorTrace(
                id: UUID(),
                createdAt: Date(),
                event: .modelTurn,
                slot: "agent",
                stage: "agent-json-step-0",
                scenarioID: scenario.id,
                e2eRunID: e2eRunID,
                agentRunID: agentRunID,
                conversationID: conversationID,
                turnID: turnID,
                intent: scenario.expectedIntent.rawValue,
                promptPrefix: scenario.prompt,
                rawOutputPrefix: #"{"action":{"tool":"weather","args":{"location":"current"}}}"#,
                selectedToolID: "weather",
                toolArguments: ["location": "current"],
                allowedToolIDs: ["location.current", "weather"],
                requiresApproval: false,
                approvalMode: nil,
                parseError: nil,
                emittedFinalInActionTurn: false,
                runtimePath: "deterministic-compatibility",
                modelLoaded: true
            )
        )

        #expect(!E2ETestRunner.modelRuntimeEvidenceForTests(
            since: startedAt,
            prompt: scenario.prompt,
            scenarioID: scenario.id,
            e2eRunID: e2eRunID,
            agentRunID: agentRunID,
            conversationID: conversationID,
            turnID: turnID,
            requiresPrimaryAgentJSON: true
        ))
        let message = E2ETestRunner.modelRuntimeEvidenceFailureMessageForTests(
            since: startedAt,
            prompt: scenario.prompt,
            scenarioID: scenario.id,
            e2eRunID: e2eRunID,
            agentRunID: agentRunID,
            conversationID: conversationID,
            turnID: turnID,
            requiresPrimaryAgentJSON: true
        )
        #expect(message.contains("runtimePath was deterministic-compatibility"))
        AgentBehaviorTraceRecorder.clear()
        #else
        #expect(true)
        #endif
    }

    @Test func promptMismatchStillFindsModelEvidenceByCorrelationID() {
        #if DEBUG
        AgentBehaviorTraceRecorder.clear()
        let startedAt = Date().addingTimeInterval(-1)
        let e2eRunID = UUID()
        AgentBehaviorTraceRecorder.record(
            AgentBehaviorTrace(
                id: UUID(),
                createdAt: Date(),
                event: .modelTurn,
                slot: "agent",
                stage: "agent-json-step-0",
                scenarioID: "training-general-chat",
                e2eRunID: e2eRunID,
                agentRunID: UUID(),
                conversationID: UUID(),
                turnID: UUID(),
                intent: "chat",
                promptPrefix: "Grounded wrapper prompt that no longer contains the original request.",
                rawOutputPrefix: #"{"final":"Precision is exactness and recall is coverage."}"#,
                selectedToolID: nil,
                toolArguments: [:],
                allowedToolIDs: [],
                requiresApproval: false,
                approvalMode: nil,
                parseError: nil,
                emittedFinalInActionTurn: true,
                modelFamily: "qwen3",
                adapterSlot: "executor",
                generationElapsedMs: 14,
                firstTokenLatencyMs: 1,
                outputTokenCount: 8,
                runtimePath: "agent-model",
                activeAdapterSlot: "executor"
            )
        )

        #expect(E2ETestRunner.modelRuntimeEvidenceForTests(
            since: startedAt,
            prompt: "Explain tradeoffs between precision and recall in retrieval systems in plain English.",
            e2eRunID: e2eRunID
        ))
        AgentBehaviorTraceRecorder.clear()
        #else
        #expect(true)
        #endif
    }

    @Test func mismatchedE2ERunIDDoesNotUseScenarioOrPromptFallback() {
        #if DEBUG
        AgentBehaviorTraceRecorder.clear()
        let startedAt = Date().addingTimeInterval(-1)
        let expectedRunID = UUID()
        let differentRunID = UUID()
        let prompt = "Explain precision and recall."
        AgentBehaviorTraceRecorder.record(
            AgentBehaviorTrace(
                id: UUID(),
                createdAt: Date(),
                event: .modelTurn,
                slot: "agent",
                stage: "agent-json-step-0",
                scenarioID: "training-general-chat",
                e2eRunID: differentRunID,
                agentRunID: UUID(),
                conversationID: UUID(),
                turnID: UUID(),
                intent: "chat",
                promptPrefix: prompt,
                rawOutputPrefix: #"{"final":"Precision is exactness and recall is coverage."}"#,
                selectedToolID: nil,
                toolArguments: [:],
                allowedToolIDs: [],
                requiresApproval: false,
                approvalMode: nil,
                parseError: nil,
                emittedFinalInActionTurn: true,
                modelFamily: "qwen3",
                adapterSlot: "executor",
                generationElapsedMs: 14,
                firstTokenLatencyMs: 1,
                outputTokenCount: 8,
                runtimePath: "agent-model",
                activeAdapterSlot: "executor"
            )
        )

        #expect(!E2ETestRunner.modelRuntimeEvidenceForTests(
            since: startedAt,
            prompt: prompt,
            scenarioID: "training-general-chat",
            e2eRunID: expectedRunID
        ))
        let message = E2ETestRunner.modelRuntimeEvidenceFailureMessageForTests(
            since: startedAt,
            prompt: prompt,
            scenarioID: "training-general-chat",
            e2eRunID: expectedRunID
        )
        #expect(message.contains("no correlated AgentBehaviorTrace found"))
        #expect(message.contains("fallbackPromptTimeTraceCount=1"))
        AgentBehaviorTraceRecorder.clear()
        #else
        #expect(true)
        #endif
    }

    @Test func missingCorrelatedModelTurnReportsCheckedIDs() {
        #if DEBUG
        AgentBehaviorTraceRecorder.clear()
        let startedAt = Date().addingTimeInterval(-1)
        let e2eRunID = UUID()

        #expect(!E2ETestRunner.modelRuntimeEvidenceForTests(
            since: startedAt,
            prompt: "Explain precision and recall.",
            scenarioID: "training-general-chat",
            e2eRunID: e2eRunID
        ))
        let message = E2ETestRunner.modelRuntimeEvidenceFailureMessageForTests(
            since: startedAt,
            prompt: "Explain precision and recall.",
            scenarioID: "training-general-chat",
            e2eRunID: e2eRunID
        )
        #expect(message.contains("no correlated AgentBehaviorTrace found"))
        #expect(message.contains("scenarioID=training-general-chat"))
        #expect(message.contains("e2eRunID=\(e2eRunID.uuidString)"))
        #expect(message.contains("model boundary skipped AgentBehaviorTrace emission"))
        AgentBehaviorTraceRecorder.clear()
        #else
        #expect(true)
        #endif
    }

    @Test func trainingEvidencePrefersPrimaryAgentJSONContextOverflowOverRepairTrace() {
        #if DEBUG
        AgentBehaviorTraceRecorder.clear()
        let startedAt = Date().addingTimeInterval(-1)
        let prompt = "What is the weather here and should I carry an umbrella?"
        AgentBehaviorTraceRecorder.record(
            AgentBehaviorTrace(
                id: UUID(),
                createdAt: Date(),
                event: .modelTurn,
                slot: "agent",
                stage: "agent-json-step-0",
                intent: "weather",
                promptPrefix: prompt,
                rawOutputPrefix: "Prompt exceeded context window before generation",
                selectedToolID: nil,
                toolArguments: [:],
                allowedToolIDs: ["location.current", "weather"],
                requiresApproval: false,
                approvalMode: nil,
                parseError: AgentTurnParseError.contextWindowExceeded.rawValue,
                emittedFinalInActionTurn: false,
                modelFamily: "qwen3",
                runtimePath: "agent-model",
                activeAdapterSlot: "executor"
            )
        )
        AgentBehaviorTraceRecorder.record(
            AgentBehaviorTrace(
                id: UUID(),
                createdAt: Date(),
                event: .modelTurn,
                slot: "executor",
                stage: "agent-repair",
                intent: "weather",
                promptPrefix: prompt,
                rawOutputPrefix: "{}",
                selectedToolID: nil,
                toolArguments: [:],
                allowedToolIDs: ["location.current", "weather"],
                requiresApproval: false,
                approvalMode: nil,
                parseError: AgentTurnParseError.missingActionOrFinal.rawValue,
                emittedFinalInActionTurn: false,
                modelFamily: "qwen3",
                runtimePath: "sharedAdapter",
                activeAdapterSlot: "executor"
            )
        )

        #expect(!E2ETestRunner.modelRuntimeEvidenceForTests(since: startedAt, prompt: prompt, requiresPrimaryAgentJSON: true))
        let message = E2ETestRunner.modelRuntimeEvidenceFailureMessageForTests(since: startedAt, prompt: prompt, requiresPrimaryAgentJSON: true)
        #expect(message.contains("found primary agent-json modelTurn"))
        #expect(message.contains("contextWindowExceeded"))
        #expect(message.contains("stage=agent-json-step-0"))
        #expect(message.contains("runtimePath=agent-model"))
        AgentBehaviorTraceRecorder.clear()
        #else
        #expect(true)
        #endif
    }

    @Test func trainingEvidenceRejectsAgentRepairModelTurn() {
        #if DEBUG
        AgentBehaviorTraceRecorder.clear()
        let startedAt = Date().addingTimeInterval(-1)
        let prompt = "Explain tradeoffs between precision and recall in retrieval systems in plain English."
        AgentBehaviorTraceRecorder.record(
            AgentBehaviorTrace(
                id: UUID(),
                createdAt: Date(),
                event: .modelTurn,
                slot: "executor",
                stage: "agent-repair",
                intent: "chat",
                promptPrefix: prompt,
                rawOutputPrefix: #"{"final":"Precision is relevance; recall is coverage."}"#,
                selectedToolID: nil,
                toolArguments: [:],
                allowedToolIDs: [],
                requiresApproval: false,
                approvalMode: nil,
                parseError: nil,
                emittedFinalInActionTurn: true,
                modelFamily: "qwen3",
                runtimePath: "sharedAdapter",
                activeAdapterSlot: "executor"
            )
        )

        #expect(E2ETestRunner.modelRuntimeEvidenceForTests(since: startedAt, prompt: prompt))
        #expect(!E2ETestRunner.modelRuntimeEvidenceForTests(since: startedAt, prompt: prompt, requiresPrimaryAgentJSON: true))
        AgentBehaviorTraceRecorder.clear()
        #else
        #expect(true)
        #endif
    }

    @Test func deterministicCompatibilityDirectChatTraceCountsAsPolicyFirstEvidenceOnlyWhenAllowed() {
        #if DEBUG
        AgentBehaviorTraceRecorder.clear()
        let startedAt = Date().addingTimeInterval(-1)
        AgentBehaviorTraceRecorder.record(
            AgentBehaviorTrace(
                id: UUID(),
                createdAt: Date(),
                event: .finalAnswer,
                slot: "mouth",
                stage: "compatibility-direct-final",
                intent: "chat",
                promptPrefix: "Explain actor isolation in Swift in simple terms.",
                rawOutputPrefix: "Actor isolation protects actor-owned state.",
                selectedToolID: nil,
                toolArguments: [:],
                allowedToolIDs: [],
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
            prompt: "Explain actor isolation in Swift in simple terms.",
            acceptsPolicyFirstEvidence: true
        ))
        #expect(!E2ETestRunner.modelRuntimeEvidenceForTests(
            since: startedAt,
            prompt: "Explain actor isolation in Swift in simple terms.",
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

    @Test func trainingValidationRunsModelSetupBeforeExecutorPreflight() async {
        #if DEBUG
        let config = E2ERunConfig(
            systemPrompt: "",
            temperature: 0.1,
            topP: 1.0,
            repetitionPenalty: 1.0,
            maxTokens: 64,
            maxAgentSteps: 1,
            enabledToolIDs: []
        )
        let recorder = OrderedEventRecorder()

        let report = await E2ETestRunner.$debugExecutorRuntimePreflightOverride.withValue({
            recorder.record("preflight")
            return ExecutorRuntimePreflightResult(
                passed: false,
                reason: "forced executor preflight failure",
                runtimeKind: "adapter-first",
                failureKind: "forced"
            )
        }) {
            await E2ETestRunner.runTrainingValidation(
                config: config,
                ensureChatLoaded: {
                    recorder.record("model-setup")
                    return true
                }
            )
        }

        #expect(recorder.values == ["model-setup", "preflight"])
        #expect(report.failed == 1)
        #expect(report.results.first?.failures == ["forced executor preflight failure"])
        #else
        #expect(true)
        #endif
    }

    @Test func executorCPUWatchdogPreflightIsNonTrainableRuntimeFailure() async {
        #if DEBUG
        let config = E2ERunConfig(
            systemPrompt: "",
            temperature: 0.1,
            topP: 1.0,
            repetitionPenalty: 1.0,
            maxTokens: 64,
            maxAgentSteps: 1,
            enabledToolIDs: []
        )

        let report = await E2ETestRunner.$debugExecutorRuntimePreflightOverride.withValue({
            ExecutorRuntimePreflightResult(
                passed: false,
                reason: "executor preflight failed: agent JSON smoke probe failed; emptyOutputReason=cpu-watchdog-degraded; outputTokens=0; streamStarted=false; firstChunkReceived=false",
                runtimeKind: "adapter-first",
                smokeProbeSucceeded: false,
                failureKind: "liveRuntimeCPUWatchdogDegraded"
            )
        }) {
            await E2ETestRunner.runTrainingValidation(
                config: config,
                ensureChatLoaded: { true }
            )
        }

        let metadata = report.results.first?.metadata ?? [:]
        #expect(report.results.first?.scenarioID == "executor-runtime-preflight")
        #expect(metadata["failureKind"] == "liveRuntimeCPUWatchdogDegraded")
        #expect(metadata["actionable"] == "false")
        #expect(metadata["trainingSignal"] == "false")
        #expect(metadata["runtimeEvidence"] == "runtime-preflight")
        #else
        #expect(true)
        #endif
    }

    @Test func standardRunBlocksBeforeScenarioWhenExecutorAdapterPreflightFails() async {
        #if DEBUG
        let scenario = E2ETestScenario(
            id: "weather-here-no-calendar",
            title: "weather",
            kind: .regression,
            prompt: "What is the weather here?",
            expectedIntent: .weather,
            requiredAllowedToolIDs: ["weather"],
            forbiddenToolIDs: [],
            requiredTextHints: [],
            forbiddenTextHints: [],
            requiresAgentRun: true
        )
        let config = E2ERunConfig(
            systemPrompt: "",
            temperature: 0.1,
            topP: 1.0,
            repetitionPenalty: 1.0,
            maxTokens: 64,
            maxAgentSteps: 1,
            enabledToolIDs: ["weather"]
        )

        let report = await E2ETestRunner.$debugStandardScenariosOverride.withValue([scenario]) {
            await E2ETestRunner.$debugExecutorRuntimePreflightOverride.withValue({
                ExecutorRuntimePreflightResult(
                    passed: false,
                    reason: "executor preflight failed: adapter required but adapter path missing",
                    slot: "executor",
                    modelFamily: "qwen3",
                    runtimeKind: "adapter-first",
                    baseModelPath: "/tmp/lumen-qwen3.gguf",
                    baseModelExists: true,
                    adapterPath: nil,
                    adapterExists: false,
                    activeAdapterSlot: nil,
                    resourceGateAllowed: true,
                    budgetReason: nil,
                    ensureReadySucceeded: false,
                    smokeProbeSucceeded: false,
                    failureKind: "adapterPathMissing"
                )
            }) {
                await E2ETestRunner.runStandard(config: config, ensureChatLoaded: { true })
            }
        }

        #expect(report.passed == 0)
        #expect(report.failed == 1)
        #expect(report.results.first?.scenarioID == "executor-runtime-preflight")
        #expect(report.results.first?.metadata["failureKind"] == "adapterPathMissing")
        #expect(report.results.first?.metadata["trainingSignal"] == "false")
        #expect(report.results.first?.finalText.contains("adapterExists=false") == true)
        #else
        #expect(true)
        #endif
    }

    @Test func primaryStructuredEvidenceDoesNotReparseTruncatedDiagnosticPrefix() {
        #if DEBUG
        let trace = AgentBehaviorTrace(
            id: UUID(),
            createdAt: Date(),
            event: .modelTurn,
            slot: "executor",
            stage: "agent-json-step-0",
            intent: "emailDraft",
            promptPrefix: "draft email",
            rawOutputPrefix: String(repeating: "x", count: 1_600),
            selectedToolID: "mail.draft",
            toolArguments: ["body": String(repeating: "x", count: 2_000)],
            allowedToolIDs: ["mail.draft"],
            requiresApproval: true,
            approvalMode: "user",
            parseError: nil,
            emittedFinalInActionTurn: false,
            outputTokenCount: 500,
            runtimePath: "agent-model",
            streamStarted: true,
            firstChunkReceived: true,
            textChunkCount: 20,
            finalChunkReceived: true,
            streamTerminationReason: "stop",
            finalizerAccepted: true
        )

        #expect(E2ETestRunner.isValidModelBackedEvidenceTraceForTests(trace, requiresPrimaryAgentJSON: true))
        #else
        #expect(true)
        #endif
    }
}

private final class CPUWatchdogProbeBox: @unchecked Sendable {
    private let lock = NSLock()
    private var calls = 0
    private let degradeAfterCalls: Int

    init(degradeAfterCalls: Int) {
        self.degradeAfterCalls = degradeAfterCalls
    }

    func isDegraded() -> Bool {
        lock.lock()
        defer { lock.unlock() }
        calls += 1
        return calls > degradeAfterCalls
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

private func recordSyntheticTrainingTrace(
    scenario: E2ETestScenario,
    e2eRunID: UUID,
    agentRunID: UUID,
    conversationID: UUID,
    turnID: UUID
) {
    let allowedToolIDs = scenario.requiredAllowedToolIDs.map(ToolRouteGuard.canonicalToolID).sorted()
    let actionToolID = allowedToolIDs.first
    let rawOutput: String
    let toolArguments: [String: String]
    if let actionToolID {
        rawOutput = #"{"action":{"tool":"\#(actionToolID)","args":{}}}"#
        toolArguments = [:]
    } else {
        rawOutput = #"{"final":"Precision is exactness and recall is coverage."}"#
        toolArguments = [:]
    }
    AgentBehaviorTraceRecorder.record(
        AgentBehaviorTrace(
            id: UUID(),
            createdAt: Date(),
            event: .modelTurn,
            slot: "agent",
            stage: "agent-json-step-0",
            scenarioID: scenario.id,
            e2eRunID: e2eRunID,
            agentRunID: agentRunID,
            conversationID: conversationID,
            turnID: turnID,
            intent: scenario.expectedIntent.rawValue,
            promptPrefix: scenario.prompt,
            rawOutputPrefix: rawOutput,
            selectedToolID: actionToolID,
            toolArguments: toolArguments,
            allowedToolIDs: allowedToolIDs,
            requiresApproval: actionToolID.map(ToolRouteGuard.requiresUserApproval),
            approvalMode: nil,
            parseError: nil,
            emittedFinalInActionTurn: actionToolID == nil,
            modelFamily: "qwen3",
            adapterSlot: "executor",
            generationElapsedMs: 42,
            firstTokenLatencyMs: 3,
            outputTokenCount: 8,
            runtimePath: "agent-model",
            activeAdapterSlot: "executor",
            maxTokensRequested: 512,
            maxTokensEffective: 512,
            promptCharCount: scenario.prompt.count,
            streamStarted: true,
            selectedRuntime: "agent-model",
            selectedAdapter: "executor",
            modelLoaded: true,
            firstChunkReceived: true,
            textChunkCount: 1,
            finalChunkReceived: true,
            streamTerminationReason: "stop"
        )
    )
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

private final class OrderedEventRecorder: @unchecked Sendable {
    private let lock = NSLock()
    private var recordedValues: [String] = []

    var values: [String] {
        lock.lock()
        defer { lock.unlock() }
        return recordedValues
    }

    func record(_ value: String) {
        lock.lock()
        recordedValues.append(value)
        lock.unlock()
    }
}
