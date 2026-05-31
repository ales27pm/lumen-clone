import Foundation
import Testing
@testable import Lumen

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
        #expect(SlotAgentService.resolveRequiredToolFallback(intent: .outlook, prompt: "Read the latest email", allowedToolIDs: Self.outlookTools) == "outlook.messages.list")
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

        let web = ToolObservationFinalizer.immediateFinalIfSafe(
            intent: .webSearch,
            toolID: "web.search",
            observation: "Found 5 results for diy underground shelter <lumen_web_payload>{\"kind\":\"searchResults\",\"results\":[]}</lumen_web_payload>",
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

        #expect(package.schemaVersion == "1.1.0")
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
}
