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
            requiredArtifactCount: 6
        )

        #expect(report.passed == 0)
        #expect(report.failed == 1)
        #expect(report.results.count == 1)
        let result = report.results[0]
        #expect(result.scenarioID == "live-runtime-artifact-preflight")
        #expect(result.metadata["failureKind"] == "liveRuntimeArtifactsNotReady")
        #expect(result.metadata["readyArtifactCount"] == "1")
        #expect(result.metadata["requiredArtifactCount"] == "6")
        #expect(result.failures[0].contains("role adapters"))
        #expect(result.finalText == "1 / 6 live runtime artifacts ready")
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
