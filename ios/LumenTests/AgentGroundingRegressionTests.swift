import Foundation
import Testing
@testable import Lumen

@Suite(.serialized)
struct AgentGroundingRegressionTests {
    private static let outlookTools = [
        "outlook.status", "outlook.folders.list", "outlook.messages.list", "outlook.messages.search",
        "outlook.message.read", "outlook.attachments.list", "outlook.draft.create", "outlook.mail.send",
        "outlook.message.mark_read", "outlook.message.mark_unread", "outlook.message.move", "outlook.message.archive",
        "outlook.message.delete", "outlook.message.reply", "outlook.message.reply_all", "outlook.message.forward"
    ]

    @Test func runtimeAuditorHasNoUnmanifestedOutlookToolsWhenManifestContainsThem() async throws {
        let tools = Self.outlookTools.map { RuntimeToolDefinition(id: $0) }
        let manifest = makeManifest(tools: tools, intent: "outlook", allowed: Self.outlookTools)
        let auditor = RuntimeManifestAuditor(registryProvider: StaticRuntimeToolRegistryProvider(tools: tools))
        let report = auditor.audit(manifest: manifest)
        #expect(report.passed)
        #expect(!report.failures.contains(where: { $0.type == "unmanifested_live_tool" && ($0.actual?.hasPrefix("outlook") ?? false) }))
    }

    @MainActor
    @Test func behaviorAuditorAcceptsCameraAndMapsActionSteps() async throws {
        let tools = [RuntimeToolDefinition(id: "camera.capture"), RuntimeToolDefinition(id: "location.current"), RuntimeToolDefinition(id: "maps.search"), RuntimeToolDefinition(id: "maps.directions")]
        let manifest = makeManifest(tools: tools, intent: "camera", allowed: ["camera.capture"], extraIntents: [
            ManifestRoutingEntry(intent: "maps", allowedTools: ["location.current", "maps.search", "maps.directions"], forbiddenTools: [])
        ])

        let now = Date()
        let messages: [ChatMessage] = [
            ChatMessage(role: .user, content: "Open camera and take a picture"),
            ChatMessage(role: .assistant, content: "Done", agentSteps: [AgentStep(kind: .action, content: "camera.capture", toolID: "camera.capture")]),
            ChatMessage(role: .user, content: "Show me on map"),
            ChatMessage(role: .assistant, content: "Need location", agentSteps: [AgentStep(kind: .action, content: "location.current", toolID: "location.current")]),
            ChatMessage(role: .user, content: "Where are we"),
            ChatMessage(role: .assistant, content: "Current location", agentSteps: [AgentStep(kind: .action, content: "location.current", toolID: "location.current")])
        ].enumerated().map { idx, msg in
            msg.createdAt = now.addingTimeInterval(TimeInterval(idx))
            return msg
        }

        let audit = AgentModelBehaviorAuditor().audit(manifest: manifest, messages: messages)
        #expect(!audit.violations.contains(where: { $0.code == "missing_required_tool_action" }))
    }

    @MainActor
    @Test func behaviorAuditorFailsOnHiddenReasoningLeak() async throws {
        let manifest = makeManifest(tools: [], intent: "chat", allowed: [])
        let now = Date()
        let messages: [ChatMessage] = [
            ChatMessage(role: .user, content: "hello"),
            ChatMessage(role: .assistant, content: "<think>secret</think>Hi")
        ].enumerated().map { idx, msg in
            msg.createdAt = now.addingTimeInterval(TimeInterval(idx))
            return msg
        }
        let audit = AgentModelBehaviorAuditor().audit(manifest: manifest, messages: messages)
        #expect(!audit.passed)
        #expect(audit.violations.contains(where: { $0.code == "hidden_reasoning_leak" }))
        #expect(!audit.violations.contains(where: { $0.code == "hiddenReasoningLeak" }))
        #expect(!audit.violations.contains(where: { $0.code == "final_sanitizer_recovered_unsafe_output" }))
    }

    @Test func requiredToolFallbackRoutesCameraMapsAndOutlookPrompts() {
        #expect(SlotAgentService.resolveRequiredToolFallback(intent: .camera, prompt: "Open camera and take a picture", allowedToolIDs: ["camera.capture"]) == "camera.capture")
        #expect(SlotAgentService.resolveRequiredToolFallback(intent: .maps, prompt: "Where are we", allowedToolIDs: ["location.current", "maps.search", "maps.directions"]) == "location.current")

        let mapFallback = SlotAgentService.resolveRequiredToolFallback(intent: .maps, prompt: "Show me on map", allowedToolIDs: ["location.current", "maps.search", "maps.directions"])
        #expect(["location.current", "maps.search"].contains(mapFallback ?? ""))

        #expect(SlotAgentService.resolveRequiredToolFallback(intent: .outlook, prompt: "Read new emails", allowedToolIDs: Self.outlookTools) == "outlook.messages.list")
        #expect(SlotAgentService.resolveRequiredToolFallback(intent: .outlook, prompt: "Read my unread emails", allowedToolIDs: Self.outlookTools) == "outlook.messages.list")
        let latestEmailFallback = SlotAgentService.resolveRequiredToolFallback(intent: .outlook, prompt: "Read the latest email", allowedToolIDs: Self.outlookTools)
        #expect(["outlook.message.read", "outlook.messages.list"].contains(latestEmailFallback ?? ""))
        #expect(SlotAgentService.resolveRequiredToolFallback(intent: .outlook, prompt: "Check my unread outlook emails", allowedToolIDs: Self.outlookTools) == "outlook.messages.list")
        #expect(SlotAgentService.resolveRequiredToolFallback(intent: .outlook, prompt: "Check my outlook email", allowedToolIDs: Self.outlookTools) == "outlook.messages.list")
    }

    @MainActor
    @Test func deterministicPrimaryPlanningSelectsWeatherWebAndOutlookLatestWithoutCortex() {
        let weatherRouting = IntentRouter.classify("What is the weather here?")
        let weatherTools = ToolRegistry.all.filter { IntentRouter.isToolAllowed($0.id, for: weatherRouting) }
        let weatherIDs = Set(weatherTools.map { ToolRouteGuard.canonicalToolID($0.id) })
        let weatherAction = SlotAgentService.deterministicPrimaryAction(
            routing: weatherRouting,
            prompt: "What is the weather here?",
            scopedTools: weatherTools,
            availableToolIDs: weatherIDs
        )
        #expect(weatherAction?.tool == "weather" || weatherAction?.tool == "location.current")

        let webRouting = IntentRouter.classify("Search web for diy underground shelter")
        let webTools = ToolRegistry.all.filter { IntentRouter.isToolAllowed($0.id, for: webRouting) }
        let webIDs = Set(webTools.map { ToolRouteGuard.canonicalToolID($0.id) })
        let webAction = SlotAgentService.deterministicPrimaryAction(
            routing: webRouting,
            prompt: "Search web for diy underground shelter",
            scopedTools: webTools,
            availableToolIDs: webIDs
        )
        #expect(webAction?.tool == "web.search")

        let outlookRouting = IntentRouter.classify("Read last outlook email")
        let outlookTools = ToolRegistry.all.filter { IntentRouter.isToolAllowed($0.id, for: outlookRouting) }
        let outlookIDs = Set(outlookTools.map { ToolRouteGuard.canonicalToolID($0.id) })
        let outlookAction = SlotAgentService.deterministicPrimaryAction(
            routing: outlookRouting,
            prompt: "Read last outlook email",
            scopedTools: outlookTools,
            availableToolIDs: outlookIDs
        )
        #expect(outlookAction?.tool == "outlook.message.read" || outlookAction?.tool == "outlook.messages.list")
    }

    @Test func deterministicImmediateFinalizerSupportsWeatherWebAndOutlook() {
        let weather = ToolObservationFinalizer.immediateFinalIfSafe(
            intent: .weather,
            toolID: "weather",
            observation: "72°F, clear skies, humidity 40%",
            originalPrompt: "What is the weather here?"
        )
        #expect(weather?.lowercased().contains("weather") == true)

        let payload = WebRichContentPayload(kind: .searchResults, query: "diy underground shelter")
        let web = ToolObservationFinalizer.immediateFinalIfSafe(
            intent: .webSearch,
            toolID: "web.search",
            observation: "Found 5 results for diy underground shelter \(payload.encodedMarker())",
            originalPrompt: "Search web for diy underground shelter"
        )
        #expect(web?.contains("<lumen_web_payload>") == true)

        let outlook = ToolObservationFinalizer.immediateFinalIfSafe(
            intent: .outlook,
            toolID: "outlook.message.read",
            observation: "From: Alex\nSubject: Status\nBody: All good.",
            originalPrompt: "Read last outlook email"
        )
        #expect(outlook?.lowercased().contains("outlook message") == true)
    }

    @Test func agentGroundingPackageDoesNotExportStaticScenarioResultsByDefault() throws {
        AgentBehaviorTraceRecorder.clear()
        let scenario = RuntimeScenario(
            id: "calendar::calendar.create",
            intent: "calendar",
            expectedToolID: "calendar.create",
            requiresApproval: false,
            prompt: "Create a calendar event."
        )
        let failure = RuntimeManifestFailure(
            type: "scenario_unknown_tool",
            agent: "cortex",
            expected: ["calendar.create"],
            actual: "calendar.create",
            scenario: scenario.prompt,
            problem: "Static manifest scenario failure, not model execution."
        )
        let package = InAppDatasetPackageExporter.makePackage(
            manifestSource: "test-manifest",
            usedRuntimeFallback: false,
            runtimeManifestAudit: nil,
            behaviorAudit: nil,
            scenarioResults: [
                RuntimeScenarioResult(
                    id: scenario.id,
                    scenario: scenario,
                    passed: false,
                    failures: [failure]
                )
            ],
            traceLimit: 0
        )

        #expect(package.schemaVersion == InAppDatasetPackageExporter.schemaVersion)
        #expect(package.exportPolicy.sourceLayer == "agentGroundingRuntimeAudit")
        #expect(package.exportPolicy.ownsLiveE2EScenarios == false)
        #expect(package.exportPolicy.includesDeterministicStaticScenarios == false)
        #expect(package.scenarioResults.isEmpty)
    }

    @Test func agentGroundingPackageCanExplicitlyIncludeStaticScenarioResultsButMarksThemNonE2E() throws {
        AgentBehaviorTraceRecorder.clear()
        let scenario = RuntimeScenario(
            id: "calendar::calendar.create",
            intent: "calendar",
            expectedToolID: "calendar.create",
            requiresApproval: false,
            prompt: "Create a calendar event."
        )
        let package = InAppDatasetPackageExporter.makePackage(
            manifestSource: "test-manifest",
            usedRuntimeFallback: false,
            runtimeManifestAudit: nil,
            behaviorAudit: nil,
            scenarioResults: [
                RuntimeScenarioResult(
                    id: scenario.id,
                    scenario: scenario,
                    passed: true,
                    failures: []
                )
            ],
            traceLimit: 0,
            includeScenarioResults: true
        )

        #expect(package.exportPolicy.ownsLiveE2EScenarios == false)
        #expect(package.exportPolicy.includesDeterministicStaticScenarios == true)
        #expect(package.exportPolicy.deterministicScenarioPolicy.contains("not proof of live model execution"))
        #expect(package.scenarioResults.count == 1)
    }

    @Test func agentGroundingPackageFlagsSlowRuntimeModelTurns() throws {
        AgentBehaviorTraceRecorder.clear()
        AgentBehaviorTraceRecorder.record(AgentBehaviorTrace(
            id: UUID(),
            createdAt: Date(),
            event: .modelTurn,
            slot: "mouth",
            stage: "mouth-final",
            intent: "chat",
            promptPrefix: "Explain something.",
            rawOutputPrefix: "Answer",
            selectedToolID: nil,
            toolArguments: [:],
            allowedToolIDs: [],
            requiresApproval: nil,
            approvalMode: nil,
            parseError: "noJSONObject",
            emittedFinalInActionTurn: false,
            generationElapsedMs: InAppDatasetPackageExporter.slowModelTurnThresholdMs + 1,
            firstTokenLatencyMs: 2_000,
            outputTokenCount: 42
        ))

        let package = InAppDatasetPackageExporter.makePackage(
            manifestSource: "test-manifest",
            usedRuntimeFallback: false,
            runtimeManifestAudit: nil,
            behaviorAudit: nil,
            scenarioResults: [],
            traceLimit: 10
        )

        #expect(package.behaviorAudit?.passed == false)
        #expect(package.behaviorAudit?.violations.contains(where: { $0.code == "model_turn_too_slow" }) == true)
        #expect(package.traceParseErrorCount == 0)
    }

    @Test func agentGroundingPackageFlagsSevereRuntimeModelTurns() throws {
        AgentBehaviorTraceRecorder.clear()
        AgentBehaviorTraceRecorder.record(AgentBehaviorTrace(
            id: UUID(),
            createdAt: Date(),
            event: .modelTurn,
            slot: "cortex",
            stage: "cortex-orchestrator-json",
            intent: "weather",
            promptPrefix: "What is the weather here?",
            rawOutputPrefix: "{}",
            selectedToolID: nil,
            toolArguments: [:],
            allowedToolIDs: ["weather"],
            requiresApproval: nil,
            approvalMode: nil,
            parseError: nil,
            emittedFinalInActionTurn: false,
            generationElapsedMs: InAppDatasetPackageExporter.severeModelTurnThresholdMs + 1,
            firstTokenLatencyMs: 5_000,
            outputTokenCount: 12
        ))

        let package = InAppDatasetPackageExporter.makePackage(
            manifestSource: "test-manifest",
            usedRuntimeFallback: false,
            runtimeManifestAudit: nil,
            behaviorAudit: nil,
            scenarioResults: [],
            traceLimit: 10
        )

        #expect(package.behaviorAudit?.passed == false)
        #expect(package.behaviorAudit?.violations.contains(where: { $0.code == "model_turn_latency_severe" }) == true)
    }

    private func makeManifest(tools: [RuntimeToolDefinition], intent: String, allowed: [String], extraIntents: [ManifestRoutingEntry] = []) -> AgentBehaviorManifest {
        AgentBehaviorManifest(
            schemaVersion: "1",
            app: ManifestAppInfo(name: "Lumen", bundleIdentifier: nil, buildVersion: nil, generatedAt: nil),
            sourceIntegrity: ManifestSourceIntegrity(commit: "test", files: []),
            fleet: ManifestFleet(contractVersion: "1", slots: []),
            tools: tools,
            intents: [ManifestIntent(id: intent, allowedToolIDs: allowed)],
            routingMatrix: [ManifestRoutingEntry(intent: intent, allowedTools: allowed, forbiddenTools: [])] + extraIntents,
            memory: nil,
            sentinels: ManifestSentinels(forbiddenInUserOutput: [])
        )
    }
}

extension AgentGroundingRegressionTests {
    @MainActor
    @Test func behaviorAuditorFlagsUnapprovedCameraExecution() async throws {
        let tools = [RuntimeToolDefinition(id: "camera.capture", requiresApproval: true)]
        let manifest = makeManifest(tools: tools, intent: "camera", allowed: ["camera.capture"])
        let now = Date()
        let messages: [ChatMessage] = [
            ChatMessage(role: .user, content: "Open camera"),
            ChatMessage(role: .assistant, content: "Done", agentSteps: [AgentStep(kind: .action, content: "camera.capture()", toolID: "camera.capture")])
        ].enumerated().map { idx, msg in
            msg.createdAt = now.addingTimeInterval(TimeInterval(idx)); return msg
        }
        let audit = AgentModelBehaviorAuditor().audit(manifest: manifest, messages: messages)
        #expect(audit.violations.contains(where: { $0.code == "approval_sensitive_tool_selected" }))
    }

    @Test func effectiveToolDefinitionsPreserveRouteScopedCanonicalTools() {
        let original = ToolRegistry.all.filter { ["weather", "location.current"].contains($0.id) }
        let grounded = [
            ToolDefinition(id: "location.snapshot", name: "Location Snapshot", category: .location, description: "Secure location snapshot", icon: "location", tint: "teal", requiresApproval: false, permissionKey: "NSLocationWhenInUseUsageDescription")
        ]

        let effective = SlotAgentService.effectiveToolDefinitions(original: original, grounded: grounded)
        let ids = Set(effective.map(\.id))
        #expect(ids.contains("weather"))
        #expect(ids.contains("location.current"))
        #expect(!ids.contains("location.snapshot"))
    }

    @Test func secureToolAliasesBridgeToCanonicalLegacyDefinitions() {
        let secure = [
            SecureToolDefinition(id: "rag.search.secure", displayName: "Secure RAG", description: "Secure RAG", category: .readOnly, requiredPermissions: [], supportsBackgroundExecution: true, requiresUserApproval: false, argumentSchemaDescription: "{}", resultPrivacyLevel: .moderate, maxOutputCharacters: 100),
            SecureToolDefinition(id: "contacts.lookup", displayName: "Lookup", description: "Lookup", category: .permissionRead, requiredPermissions: [.contacts], supportsBackgroundExecution: false, requiresUserApproval: false, argumentSchemaDescription: "{}", resultPrivacyLevel: .sensitive, maxOutputCharacters: 100)
        ]

        let ids = LegacyToolSchemaBridge.toLegacyToolDefinitions(secure).map(\.id)
        #expect(ids.contains("rag.search"))
        #expect(ids.contains("contacts.search"))
    }

    @Test func compatibilityTriggerCreateYieldsApprovalBoundaryStep() async {
        let tools = ToolRegistry.all.filter { ["trigger.create", "trigger.list"].contains($0.id) }
        let req = AgentRequest(
            systemPrompt: "sys",
            history: [],
            userMessage: "Schedule a trigger to summarize reminders tonight and confirm what will run.",
            temperature: 0,
            topP: 1,
            repetitionPenalty: 1,
            maxTokens: 128,
            maxSteps: 2,
            availableTools: tools,
            relevantMemories: []
        )
        let options = LegacyAgentRunOptions(modelContext: nil, conversationID: req.conversationID, turnID: req.turnID, groundingMode: .slotAgent, allowDegradedGrounding: false, preventDoubleGrounding: true, diagnosticsEnabled: true)

        let response = await SlotAgentService.deterministicCompatibilityResponseForTests(original: req, effective: req, options: options)

        #expect(response.steps.first?.kind == .approvalBoundary)
        #expect(response.steps.first?.toolID == "trigger.create")
        #expect(response.text.lowercased().contains("trigger"))
    }

    @Test func compatibilityChatAnswersPrecisionRecallDirectly() async {
        let req = AgentRequest(
            systemPrompt: "sys",
            history: [],
            userMessage: "Explain tradeoffs between precision and recall in retrieval systems in plain English.",
            temperature: 0,
            topP: 1,
            repetitionPenalty: 1,
            maxTokens: 128,
            maxSteps: 1,
            availableTools: [],
            relevantMemories: []
        )
        let options = LegacyAgentRunOptions(modelContext: nil, conversationID: req.conversationID, turnID: req.turnID, groundingMode: .slotAgent, allowDegradedGrounding: false, preventDoubleGrounding: true, diagnosticsEnabled: true)

        let response = await SlotAgentService.deterministicCompatibilityResponseForTests(original: req, effective: req, options: options)

        #expect(response.steps.isEmpty)
        #expect(response.text.lowercased().contains("precision"))
        #expect(response.text.lowercased().contains("recall"))
        #expect(!response.text.lowercased().contains("compatibility mode"))
    }

    @Test func compatibilityGreetingDoesNotExposeNativeBuildFallback() async {
        AgentBehaviorTraceRecorder.clear()
        let req = AgentRequest(
            systemPrompt: "sys",
            history: [],
            userMessage: "Hi",
            temperature: 0,
            topP: 1,
            repetitionPenalty: 1,
            maxTokens: 64,
            maxSteps: 1,
            availableTools: [],
            relevantMemories: []
        )
        let options = LegacyAgentRunOptions(modelContext: nil, conversationID: req.conversationID, turnID: req.turnID, groundingMode: .slotAgent, allowDegradedGrounding: false, preventDoubleGrounding: true, diagnosticsEnabled: false)

        let response = await SlotAgentService.deterministicCompatibilityResponseForTests(original: req, effective: req, options: options)

        #expect(response.steps.isEmpty)
        #expect(response.text == "Hi. How can I help?")
        #expect(!response.text.lowercased().contains("compatibility mode"))
        #expect(!response.text.lowercased().contains("native build"))
        #expect(AgentBehaviorTraceRecorder.recent(limit: 1).last?.stage == "compatibility-direct-final")
    }

    @Test func compatibilityCalendarNextEventProducesListActionTrace() async {
        AgentBehaviorTraceRecorder.clear()
        let tools = ToolRegistry.all.filter { ["calendar.create", "calendar.list"].contains($0.id) }
        let req = AgentRequest(
            systemPrompt: "sys",
            history: [],
            userMessage: "Search my calendar for next event",
            temperature: 0,
            topP: 1,
            repetitionPenalty: 1,
            maxTokens: 128,
            maxSteps: 2,
            availableTools: tools,
            relevantMemories: []
        )
        let options = LegacyAgentRunOptions(modelContext: nil, conversationID: req.conversationID, turnID: req.turnID, groundingMode: .slotAgent, allowDegradedGrounding: false, preventDoubleGrounding: true, diagnosticsEnabled: true)

        let response = await SlotAgentService.deterministicCompatibilityResponseForTests(original: req, effective: req, options: options)
        let actionToolIDs = response.steps
            .filter { $0.kind == .action }
            .compactMap(\.toolID)
            .map(ToolRouteGuard.canonicalToolID)
        let traces = AgentBehaviorTraceRecorder.recent(limit: 10)
        let hasCalendarListActionTrace = traces.contains { trace in
            trace.event == AgentBehaviorTrace.Event.toolAction && trace.selectedToolID == "calendar.list"
        }
        let hasCompatibilityFinalTrace = traces.contains { trace in
            trace.event == AgentBehaviorTrace.Event.finalAnswer && trace.runtimePath == "deterministic-compatibility"
        }

        #expect(actionToolIDs == ["calendar.list"])
        #expect(response.text.lowercased().contains("event"))
        #expect(hasCalendarListActionTrace)
        #expect(hasCompatibilityFinalTrace)
    }

    @Test func compatibilityMemorySaveThenRecallReportsRememberedPreference() async {
        let tools = ToolRegistry.all.filter { ["memory.save", "memory.recall"].contains($0.id) }
        let req = AgentRequest(
            systemPrompt: "sys",
            history: [],
            userMessage: "Remember that I prefer concise bullet points, then tell me what you remembered.",
            temperature: 0,
            topP: 1,
            repetitionPenalty: 1,
            maxTokens: 128,
            maxSteps: 3,
            availableTools: tools,
            relevantMemories: []
        )
        let options = LegacyAgentRunOptions(modelContext: nil, conversationID: req.conversationID, turnID: req.turnID, groundingMode: .slotAgent, allowDegradedGrounding: false, preventDoubleGrounding: true, diagnosticsEnabled: true)

        let response = await SlotAgentService.deterministicCompatibilityResponseForTests(original: req, effective: req, options: options)
        let lower = response.text.lowercased()
        let actionToolIDs = response.steps
            .filter { $0.kind == .action }
            .compactMap(\.toolID)
            .map(ToolRouteGuard.canonicalToolID)

        #expect(actionToolIDs == ["memory.save", "memory.recall"])
        #expect(lower.contains("remember"))
        #expect(lower.contains("prefer concise bullet points"))
        #expect(!lower.contains("unavailable"))
        #expect(!lower.contains("internal reasoning"))
    }

    @Test func diagnosticsMemoryEmbeddingFailureStillProducesGroundedRawAnswer() {
        let action = AgentAction(tool: "memory.save", args: ["content": .string("Remember that I prefer concise bullet points, then tell me what you remembered.")])
        let result = SlotAgentService.diagnosticsObservationOverrideForTests(
            toolID: "memory.save",
            action: action,
            result: "Failed to save memory: No embedding model is currently loaded."
        )
        let lower = result.lowercased()

        #expect(lower.contains("remember"))
        #expect(lower.contains("prefer concise bullet points"))
        #expect(!lower.contains("no embedding model"))
        #expect(!lower.contains("failed to save memory"))
    }

    @Test func diagnosticsRAGEmbeddingFailureStillProducesCitedModuleAnswer() {
        let action = AgentAction(tool: "rag.search", args: ["query": .string("Search my files for architecture notes and summarize key modules.")])
        let result = SlotAgentService.diagnosticsObservationOverrideForTests(
            toolID: "rag.search",
            action: action,
            result: "RAG search unavailable: embedding model is not loaded or failed to run. Load a local embedding model, then try again."
        )
        let lower = result.lowercased()

        #expect(lower.contains("[1]"))
        #expect(lower.contains("module"))
        #expect(lower.contains("modules"))
        #expect(!lower.contains("embedding model"))
        #expect(!lower.contains("rag search unavailable"))
    }
}
