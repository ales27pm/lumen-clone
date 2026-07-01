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
        #expect(!options.allowParseFailureDeterministicRecovery)
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
        let denial = "live-e2e.pre-scenario: scenePhase=inactive"
        let result = await E2ETestRunner.liveRuntimePreflightBlockedResultForTests(
            scenario,
            denialReason: denial
        )
        #expect(!result.passed)
        #expect(result.metadata["failureKind"] == "liveRuntimeScenePhaseUnavailable")
        #expect(result.metadata["budgetDenialReason"] == denial)
        #expect(result.metadata["actionable"] == "false")
        #expect(result.metadata["trainingSignal"] == "false")
        #expect(E2ETestReport(id: UUID(), startedAt: Date(), finishedAt: Date(), passed: 0, failed: 1, results: [result]).summaryText.contains("runtime-preflight/non-actionable"))
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
        #expect(E2ETestRunner.liveAgentQualityFailures(
            rawFinalText: #"{"intent":"webSearch","nextModel":"rag","reasoningSummary":"Intent webSearch is allowed to use rag.search.","requiresApproval":false,"sourceFile":"ios/Lumen/Models/ToolDefinition.swift"}"#,
            finalText: #"{"intent":"webSearch","nextModel":"rag","reasoningSummary":"Intent webSearch is allowed to use rag.search.","requiresApproval":false,"sourceFile":"ios/Lumen/Models/ToolDefinition.swift"}"#,
            scenario: scenario
        ).contains("Live agent returned fallback/error text instead of completing the scenario"))
        #expect(!E2ETestRunner.webSearchSummaryQualityFailureForTests(finalText: "- Prefer structured cancellation so child tasks stop cleanly.\n- Keep MainActor UI updates explicit to avoid accidental data races.", scenario: scenario))
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
            missingArtifactFileNames: ["lumen-executor-lora.gguf"]
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
        #expect(result.failures[0].contains("role adapters"))
        #expect(result.finalText.contains("1 / 6 live runtime artifacts ready"))
        #expect(result.finalText.contains("Missing adapter slots: executor"))
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
        #expect(message.contains("agent-json emitted empty output"))
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
        #expect(message.contains("AgentService model path was not entered"))
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
        #expect(report.results.first?.finalText.contains("adapterExists=false") == true)
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
