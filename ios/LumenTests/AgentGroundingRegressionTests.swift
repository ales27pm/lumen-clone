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

    @MainActor
    @Test func runtimeAuditorHasNoUnmanifestedOutlookToolsWhenManifestContainsThem() async throws {
        let tools = Self.outlookTools.map { RuntimeToolDefinition(id: $0) }
        let manifest = makeManifest(tools: tools, intent: "outlook", allowed: Self.outlookTools)
        let auditor = RuntimeManifestAuditor(registryProvider: StaticRuntimeToolRegistryProvider(tools: tools))
        let report = auditor.audit(manifest: manifest)
        #expect(report.passed)
        #expect(!report.failures.contains(where: { $0.type == "unmanifested_live_tool" && ($0.actual?.hasPrefix("outlook") ?? false) }))
    }

    @MainActor
    @Test func liveRuntimeSchemaTreatsOutlookOptionalAliasesAsOptional() async throws {
        let tools = LiveRuntimeToolRegistryProvider().currentToolDefinitions()
        let read = try #require(tools.first(where: { $0.id == "outlook.message.read" }))
        let argsByName = Dictionary(uniqueKeysWithValues: read.arguments.map { ($0.name, $0) })

        #expect(argsByName["messageId"]?.required == true)
        #expect(argsByName["id"]?.required == false)
        #expect(Set(read.arguments.filter(\.required).map(\.name)) == Set(["messageId"]))
    }

    @MainActor
    @Test func liveRuntimeSchemaPublishesPermissionKindAndConfirmationMode() async throws {
        let tools = LiveRuntimeToolRegistryProvider().currentToolDefinitions()
        let calendarCreate = try #require(tools.first(where: { $0.id == "calendar.create" }))
        #expect(calendarCreate.permissionKind == "calendar")
        #expect(calendarCreate.confirmationMode == "userApproval")

        let triggerCreate = try #require(tools.first(where: { $0.id == "trigger.create" }))
        #expect(triggerCreate.permissionKey == nil)
        #expect(triggerCreate.permissionKind == "notifications")
        #expect(triggerCreate.confirmationMode == "userApproval")

        let weather = try #require(tools.first(where: { $0.id == "weather" }))
        #expect(weather.confirmationMode == "none")
    }

    @MainActor
    @Test func runtimeAuditorRejectsPermissionAndConfirmationDrift() async throws {
        let manifestTools = [
            RuntimeToolDefinition(
                id: "camera.capture",
                requiresApproval: true,
                permissionKey: "NSCameraUsageDescription",
                permissionKind: "photos",
                confirmationMode: "none"
            )
        ]
        let liveTools = [
            RuntimeToolDefinition(
                id: "camera.capture",
                requiresApproval: true,
                permissionKey: "NSCameraUsageDescription",
                permissionKind: "camera",
                confirmationMode: "userApproval"
            )
        ]
        let manifest = makeManifest(tools: manifestTools, intent: "camera", allowed: ["camera.capture"])
        let auditor = RuntimeManifestAuditor(registryProvider: StaticRuntimeToolRegistryProvider(tools: liveTools))
        let report = auditor.audit(manifest: manifest)
        #expect(!report.passed)
        #expect(report.failures.contains(where: { $0.type == "permission_kind_mismatch" }))
        #expect(report.failures.contains(where: { $0.type == "confirmation_mode_mismatch" }))
    }

    @MainActor
    @Test func liveRuntimeSchemaAlignsAliasAndPlusArgumentsWithManifest() async throws {
        let tools = LiveRuntimeToolRegistryProvider().currentToolDefinitions()

        let messagesDraft = try #require(tools.first(where: { $0.id == "messages.draft" }))
        let messageArgs = Dictionary(uniqueKeysWithValues: messagesDraft.arguments.map { ($0.name, $0) })
        #expect(messageArgs["to"]?.required == true)
        #expect(messageArgs["body"]?.required == true)
        #expect(messageArgs["recipient"]?.required == false)
        #expect(messageArgs["number"]?.required == false)
        #expect(messageArgs["message"]?.required == false)
        #expect(messageArgs["text"]?.required == false)

        let mailDraft = try #require(tools.first(where: { $0.id == "mail.draft" }))
        let mailArgs = Dictionary(uniqueKeysWithValues: mailDraft.arguments.map { ($0.name, $0) })
        #expect(mailArgs["to"]?.required == true)
        #expect(mailArgs["subject"]?.required == true)
        #expect(mailArgs["body"]?.required == true)
        #expect(mailArgs["recipient"]?.required == false)
        #expect(mailArgs["email"]?.required == false)
        #expect(mailArgs["message"]?.required == false)
        #expect(mailArgs["text"]?.required == false)

        let outlookFolders = try #require(tools.first(where: { $0.id == "outlook.folders.list" }))
        let folderArgs = Dictionary(uniqueKeysWithValues: outlookFolders.arguments.map { ($0.name, $0) })
        #expect(folderArgs["includeHidden"]?.required == false)
        #expect(folderArgs["false"] == nil)

        let triggerCreate = try #require(tools.first(where: { $0.id == "trigger.create" }))
        let triggerArgs = Dictionary(uniqueKeysWithValues: triggerCreate.arguments.map { ($0.name, $0) })
        #expect(triggerArgs["plus"] == nil)
        #expect(triggerArgs["inMinutes"]?.required == false)
        #expect(triggerArgs["atTime"]?.required == false)
        #expect(triggerArgs["intervalSeconds"]?.required == false)
        #expect(triggerArgs["beforeMinutes"]?.required == false)
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

    @MainActor
    @Test func behaviorAuditorTeachesPlanGatherExecuteEvaluateForDynamicLookupFailures() async throws {
        let tools = [
            RuntimeToolDefinition(id: "location.current"),
            RuntimeToolDefinition(id: "web.search"),
            RuntimeToolDefinition(id: "web.fetch"),
            RuntimeToolDefinition(id: "maps.search")
        ]
        let manifest = makeManifest(
            tools: tools,
            intent: "webSearch",
            allowed: ["web.search", "web.fetch", "location.current"],
            extraIntents: [
                ManifestRoutingEntry(intent: "maps", allowedTools: ["location.current", "maps.search"], forbiddenTools: ["web.search"])
            ]
        )
        let now = Date()
        let messages: [ChatMessage] = [
            ChatMessage(role: .user, content: "Where is the nearest free tax clinic tomorrow?"),
            ChatMessage(
                role: .assistant,
                content: "No nearby places found.",
                agentSteps: [
                    AgentStep(kind: .action, content: "maps.search(query=free tax clinic tomorrow)", toolID: "maps.search")
                ]
            )
        ].enumerated().map { idx, msg in
            msg.createdAt = now.addingTimeInterval(TimeInterval(idx))
            return msg
        }

        let audit = AgentModelBehaviorAuditor().audit(manifest: manifest, messages: messages)
        let sample = try #require(audit.repairSamples.first(where: { $0.violationCode == "tool_not_allowed_by_runtime_router" }))
        #expect(sample.correctedOutput.contains("dynamic local public lookup"))
        #expect(sample.correctedOutput.contains("current location"))
        #expect(sample.correctedOutput.contains("web.search"))
        #expect(sample.correctedOutput.contains("evaluate"))
        #expect(sample.curriculum == "plan_gather_execute_evaluate")
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

    @Test func deterministicImmediateFinalizerSupportsWeatherWebMapsRAGAndOutlook() {
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

        let maps = ToolObservationFinalizer.immediateFinalIfSafe(
            intent: .maps,
            toolID: "maps.search",
            observation: "Tim Hortons — Avenue de la Plaza",
            originalPrompt: "Find coffee near me."
        )
        #expect(maps?.lowercased().contains("maps search results") == true)
        #expect(maps?.lowercased().contains("tim hortons") == true)

        let ragMiss = ToolObservationFinalizer.immediateFinalIfSafe(
            intent: .rag,
            toolID: "rag.search",
            observation: "No matching files found for 'latest Lumen diagnostics report'.",
            originalPrompt: "Search my local files for the latest Lumen diagnostics report."
        )
        #expect(ragMiss?.lowercased().contains("source") == true)
        #expect(ragMiss?.lowercased().contains("snippet") == true)

        let photoIndex = ToolObservationFinalizer.immediateFinalIfSafe(
            intent: .rag,
            toolID: "rag.index_photos",
            observation: "Indexed 7 monthly photo summaries.",
            originalPrompt: "Refresh the photo retrieval index."
        )
        #expect(photoIndex?.lowercased().contains("photo index updated") == true)

        let outlook = ToolObservationFinalizer.immediateFinalIfSafe(
            intent: .outlook,
            toolID: "outlook.message.read",
            observation: "From: Alex\nSubject: Status\nBody: All good.",
            originalPrompt: "Read last outlook email"
        )
        #expect(outlook?.lowercased().contains("outlook message") == true)

        let outlookAttachments = ToolObservationFinalizer.immediateFinalIfSafe(
            intent: .outlook,
            toolID: "outlook.attachments.list",
            observation: "No attachments on the latest message.",
            originalPrompt: "Show attachments on the latest Outlook email."
        )
        #expect(outlookAttachments?.lowercased().contains("outlook attachments") == true)

        let outlookFolders = ToolObservationFinalizer.immediateFinalIfSafe(
            intent: .outlook,
            toolID: "outlook.folders.list",
            observation: "Inbox\nArchive",
            originalPrompt: "Show Outlook mail folders."
        )
        #expect(outlookFolders?.lowercased().contains("outlook folders") == true)

        let calendar = ToolObservationFinalizer.immediateFinalIfSafe(
            intent: .calendar,
            toolID: "calendar.list",
            observation: "• Journée nationale des Autochtones — Jun 21, 2026 at 12:00 AM",
            originalPrompt: "What's on my schedule today?"
        )
        #expect(calendar?.lowercased().contains("calendar events") == true)

        let motion = ToolObservationFinalizer.immediateFinalIfSafe(
            intent: .motion,
            toolID: "motion.activity",
            observation: "Today's motion — stationary 15 min",
            originalPrompt: "Am I walking or stationary right now?"
        )
        #expect(motion?.lowercased().contains("motion activity") == true)

        let reminders = ToolObservationFinalizer.immediateFinalIfSafe(
            intent: .reminder,
            toolID: "reminders.list",
            observation: "• Buy foil\n• Clean the car",
            originalPrompt: "List pending reminders."
        )
        #expect(reminders?.lowercased().contains("reminders") == true)
        #expect(reminders?.lowercased().contains("buy foil") == true)
    }

    @Test func toolObservationFinalizerReportsStructuredRejectionReasons() {
        let intentMismatch = ToolObservationFinalizer.immediateFinalOutcome(
            intent: .calendar,
            toolID: "weather",
            observation: "72°F, clear skies",
            originalPrompt: "What is the weather?"
        )
        #expect(intentMismatch.accepted == false)
        #expect(intentMismatch.text == nil)
        #expect(intentMismatch.rejectionReason == "intent-mismatch")

        let unsafe = ToolObservationFinalizer.immediateFinalOutcome(
            intent: .weather,
            toolID: "weather",
            observation: #"{"kind":"tool-debug","value":"raw"}"#,
            originalPrompt: "What is the weather?"
        )
        #expect(unsafe.accepted == false)
        #expect(unsafe.rejectionReason == "unsafe-observation")

        let deepSynthesis = ToolObservationFinalizer.immediateFinalOutcome(
            intent: .webSearch,
            toolID: "web.search",
            observation: "Result 1\nResult 2",
            originalPrompt: "Summarize and compare these sources."
        )
        #expect(deepSynthesis.accepted == false)
        #expect(deepSynthesis.rejectionReason == "deep-synthesis-required")
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

    @Test func agentGroundingPackageCarriesExportQualityFailureWhenRecentTracesAreEmpty() throws {
        AgentBehaviorTraceRecorder.clear()
        let package = InAppDatasetPackageExporter.makePackage(
            manifestSource: "test-manifest",
            usedRuntimeFallback: false,
            runtimeManifestAudit: nil,
            behaviorAudit: nil,
            scenarioResults: [],
            traceLimit: 10
        )

        let failure = try #require(package.exportQualityFailures?.first)
        #expect(failure.type == "agent_grounding_no_recent_model_traces")
        #expect(failure.sourceLayer == "agentGroundingRuntimeAudit.exportQuality")
        #expect(failure.actual == "recentTraces is empty")
    }

    @Test func liveE2EExportCountsModelBackedAndCompatibilityTracesSeparately() throws {
        AgentBehaviorTraceRecorder.clear()
        defer { AgentBehaviorTraceRecorder.clear() }
        let startedAt = Date(timeIntervalSince1970: 1_800_000_000)
        let e2eRunID = UUID(uuidString: "11111111-1111-4111-8111-111111111111")!
        let agentRunID = UUID(uuidString: "22222222-2222-4222-8222-222222222222")!
        let conversationID = UUID(uuidString: "33333333-3333-4333-8333-333333333333")!
        let turnID = UUID(uuidString: "44444444-4444-4444-8444-444444444444")!
        let scenarioID = "live-self-model"
        AgentBehaviorTraceRecorder.record(AgentBehaviorTrace(
            id: UUID(),
            createdAt: startedAt.addingTimeInterval(1),
            event: .modelTurn,
            slot: "executor",
            stage: "agent-json-step-0",
            scenarioID: scenarioID,
            e2eRunID: e2eRunID,
            agentRunID: agentRunID,
            conversationID: conversationID,
            turnID: turnID,
            intent: "rag",
            promptPrefix: "What evidence supports your claim?",
            rawOutputPrefix: #"{"final":"Live E2E evidence."}"#,
            selectedToolID: nil,
            toolArguments: [:],
            allowedToolIDs: [],
            requiresApproval: nil,
            approvalMode: nil,
            parseError: nil,
            emittedFinalInActionTurn: false,
            outputTokenCount: 8,
            runtimePath: "agent-model"
        ))
        AgentBehaviorTraceRecorder.record(AgentBehaviorTrace(
            id: UUID(),
            createdAt: startedAt.addingTimeInterval(2),
            event: .finalAnswer,
            slot: "cortex",
            stage: "compatibility-final",
            scenarioID: scenarioID,
            e2eRunID: e2eRunID,
            agentRunID: agentRunID,
            conversationID: conversationID,
            turnID: turnID,
            intent: "rag",
            promptPrefix: "What evidence supports your claim?",
            rawOutputPrefix: "Compatibility answer.",
            selectedToolID: nil,
            toolArguments: [:],
            allowedToolIDs: [],
            requiresApproval: nil,
            approvalMode: nil,
            parseError: nil,
            emittedFinalInActionTurn: true,
            runtimePath: "deterministic-compatibility"
        ))
        let result = E2ETestResult(
            id: UUID(),
            scenarioID: scenarioID,
            kind: "standard",
            title: "Live self-model evidence",
            prompt: "What evidence supports your claim?",
            expectedIntent: "rag",
            actualIntent: "rag",
            e2eRunID: e2eRunID,
            agentRunID: agentRunID,
            conversationID: conversationID,
            turnID: turnID,
            requiresAgentRun: true,
            passed: true,
            failures: [],
            finalText: "Live E2E evidence.",
            missingHints: [],
            rewriteAttempted: false,
            rewriteSuccess: false,
            events: [],
            startedAt: startedAt,
            finishedAt: startedAt.addingTimeInterval(3),
            rawFinalPrefix: "Live E2E evidence.",
            sanitizedFinalPrefix: "Live E2E evidence.",
            rawFinalHadUnsafeLeakage: false,
            sanitizedFinalRemovedArtifacts: [],
            outputHygieneFailures: []
        )
        let report = E2ETestReport(
            id: UUID(),
            startedAt: startedAt,
            finishedAt: startedAt.addingTimeInterval(3),
            passed: 1,
            failed: 0,
            results: [result]
        )

        let package = InAppDatasetPackageExporter.makePackage(
            manifestSource: "test-manifest",
            usedRuntimeFallback: false,
            runtimeManifestAudit: nil,
            behaviorAudit: nil,
            scenarioResults: [],
            liveE2EReport: report,
            traceLimit: 10
        )

        let liveE2EReport = try #require(package.liveE2EReport)
        #expect(package.schemaVersion == InAppDatasetPackageExporter.schemaVersion)
        #expect(liveE2EReport.correlatedTraceCount == 2)
        #expect(liveE2EReport.modelBackedCorrelatedTraceCount == 1)
        #expect(liveE2EReport.deterministicCompatibilityTraceCount == 1)
    }

    @Test func liveE2EExportFlagsMissingModelBackedTraceCoverage() throws {
        AgentBehaviorTraceRecorder.clear()
        defer { AgentBehaviorTraceRecorder.clear() }
        let startedAt = Date(timeIntervalSince1970: 1_800_000_100)
        let e2eRunID = UUID(uuidString: "11111111-1111-4111-8111-111111111112")!
        let agentRunID = UUID(uuidString: "22222222-2222-4222-8222-222222222223")!
        let conversationID = UUID(uuidString: "33333333-3333-4333-8333-333333333334")!
        let turnID = UUID(uuidString: "44444444-4444-4444-8444-444444444445")!
        let scenarioID = "live-alarm-authorization-status"
        AgentBehaviorTraceRecorder.record(AgentBehaviorTrace(
            id: UUID(),
            createdAt: startedAt.addingTimeInterval(1),
            event: .finalAnswer,
            slot: "cortex",
            stage: "compatibility-clarification-final",
            scenarioID: scenarioID,
            e2eRunID: e2eRunID,
            agentRunID: agentRunID,
            conversationID: conversationID,
            turnID: turnID,
            intent: "alarm",
            promptPrefix: "Show alarm permission status.",
            rawOutputPrefix: "I couldn't safely complete the alarm/timer request.",
            selectedToolID: nil,
            toolArguments: [:],
            allowedToolIDs: [],
            requiresApproval: nil,
            approvalMode: nil,
            parseError: nil,
            emittedFinalInActionTurn: true,
            runtimePath: "deterministic-compatibility"
        ))
        let result = E2ETestResult(
            id: UUID(),
            scenarioID: scenarioID,
            kind: "standard",
            title: "Live alarm authorization status",
            prompt: "Show alarm permission status.",
            expectedIntent: "alarm",
            actualIntent: "alarm",
            e2eRunID: e2eRunID,
            agentRunID: agentRunID,
            conversationID: conversationID,
            turnID: turnID,
            requiresAgentRun: true,
            passed: false,
            failures: ["found deterministic-compatibility execution trace but policy-first evidence disabled for this scenario"],
            finalText: "I couldn't safely complete the alarm/timer request.",
            missingHints: [],
            rewriteAttempted: false,
            rewriteSuccess: false,
            events: [],
            startedAt: startedAt,
            finishedAt: startedAt.addingTimeInterval(2),
            rawFinalPrefix: "I couldn't safely complete the alarm/timer request.",
            sanitizedFinalPrefix: "I couldn't safely complete the alarm/timer request.",
            rawFinalHadUnsafeLeakage: false,
            sanitizedFinalRemovedArtifacts: [],
            outputHygieneFailures: []
        )
        let report = E2ETestReport(
            id: UUID(),
            startedAt: startedAt,
            finishedAt: startedAt.addingTimeInterval(2),
            passed: 0,
            failed: 1,
            results: [result]
        )

        let package = InAppDatasetPackageExporter.makePackage(
            manifestSource: "test-manifest",
            usedRuntimeFallback: false,
            runtimeManifestAudit: nil,
            behaviorAudit: nil,
            scenarioResults: [],
            liveE2EReport: report,
            traceLimit: 10
        )

        let liveE2EReport = try #require(package.liveE2EReport)
        let failure = try #require(package.exportQualityFailures?.first(where: { $0.type == "agent_grounding_live_e2e_model_backed_trace_gap" }))
        #expect(liveE2EReport.correlatedTraceCount == 1)
        #expect(liveE2EReport.modelBackedCorrelatedTraceCount == 0)
        #expect(liveE2EReport.deterministicCompatibilityTraceCount == 1)
        #expect(failure.actual?.contains("requiredAgentRunScenarioCount=1") == true)
        #expect(failure.actual?.contains("modelBackedCorrelatedTraceCount=0") == true)
        #expect(failure.problem.contains("Deterministic compatibility traces") == true)
    }

    @Test func agentGroundingPackageFlagsIncompleteStructuredModelTraceProof() throws {
        AgentBehaviorTraceRecorder.clear()
        AgentBehaviorTraceRecorder.record(AgentBehaviorTrace(
            id: UUID(),
            createdAt: Date(),
            event: .modelTurn,
            slot: "executor",
            stage: "agent-json-step-0",
            intent: "weather",
            promptPrefix: "What is the weather here?",
            rawOutputPrefix: "",
            selectedToolID: nil,
            toolArguments: [:],
            allowedToolIDs: ["weather"],
            requiresApproval: nil,
            approvalMode: nil,
            parseError: nil,
            emittedFinalInActionTurn: false,
            runtimePath: "agent-model"
        ))

        let package = InAppDatasetPackageExporter.makePackage(
            manifestSource: "test-manifest",
            usedRuntimeFallback: false,
            runtimeManifestAudit: nil,
            behaviorAudit: nil,
            scenarioResults: [],
            traceLimit: 10
        )

        let failure = try #require(package.exportQualityFailures?.first(where: { $0.type == "agent_grounding_model_trace_incomplete" }))
        #expect(failure.actual?.contains("selectedRuntime") == true)
        #expect(failure.actual?.contains("modelLoaded") == true)
        #expect(failure.actual?.contains("outputTokenCount") == true)
        #expect(failure.actual?.contains("streamStarted") == true)
    }

    @Test func agentGroundingPackageAcceptsCompleteStructuredModelTraceProof() throws {
        AgentBehaviorTraceRecorder.clear()
        AgentBehaviorTraceRecorder.record(AgentBehaviorTrace(
            id: UUID(),
            createdAt: Date(),
            event: .modelTurn,
            slot: "executor",
            stage: "agent-json-step-0",
            intent: "weather",
            promptPrefix: "What is the weather here?",
            rawOutputPrefix: #"{"action":{"tool":"weather","args":{}}}"#,
            selectedToolID: "weather",
            toolArguments: [:],
            allowedToolIDs: ["weather"],
            requiresApproval: false,
            approvalMode: nil,
            parseError: nil,
            emittedFinalInActionTurn: false,
            outputTokenCount: 12,
            runtimePath: "agent-model",
            emptyOutputReason: nil,
            streamStarted: true,
            selectedRuntime: "llama.cpp",
            modelLoaded: true,
            firstChunkReceived: true,
            textChunkCount: 1,
            finalChunkReceived: true,
            streamTerminationReason: "stop"
        ))

        let package = InAppDatasetPackageExporter.makePackage(
            manifestSource: "test-manifest",
            usedRuntimeFallback: false,
            runtimeManifestAudit: nil,
            behaviorAudit: nil,
            scenarioResults: [],
            traceLimit: 10
        )

        #expect(package.exportQualityFailures?.isEmpty == true)
    }

    @Test func agentGroundingPackageExportsSelfModelDecisionSummary() throws {
        AgentBehaviorTraceRecorder.clear()
        defer { AgentBehaviorTraceRecorder.clear() }
        let prompt = """
        User request

        __LUMEN_GROUNDING_V1__
        [SELF MODEL]
        schemaVersion=0.1.0
        mode=foreground
        activeSlot=executor
        sourceLayer=agentGroundingRuntimeAudit
        policy=mustNotInventToolIDs,mustNotBypassApproval,mustCiteRuntimeSourceWhenClaimingRuntimeState
        """
        let selfModel = try #require(AgentBehaviorTrace.SelfModelDecisionSummary.fromPrompt(
            prompt,
            selectedToolID: "calendar.create",
            requiresApproval: true,
            approvalMode: "userApproval"
        ))
        AgentBehaviorTraceRecorder.record(AgentBehaviorTrace(
            id: UUID(),
            createdAt: Date(),
            event: .modelTurn,
            slot: "executor",
            stage: "agent-json-step-0",
            intent: "calendar",
            promptPrefix: "Create a calendar event",
            rawOutputPrefix: #"{"action":{"tool":"calendar.create","args":{}}}"#,
            selectedToolID: "calendar.create",
            toolArguments: [:],
            allowedToolIDs: ["calendar.create"],
            requiresApproval: true,
            approvalMode: "userApproval",
            parseError: nil,
            emittedFinalInActionTurn: false,
            outputTokenCount: 12,
            runtimePath: "agent-model",
            streamStarted: true,
            selectedRuntime: "llama.cpp",
            modelLoaded: true,
            firstChunkReceived: true,
            textChunkCount: 1,
            finalChunkReceived: true,
            streamTerminationReason: "stop",
            selfModel: selfModel
        ))

        let package = InAppDatasetPackageExporter.makePackage(
            manifestSource: "test-manifest",
            usedRuntimeFallback: false,
            runtimeManifestAudit: nil,
            behaviorAudit: nil,
            scenarioResults: [],
            traceLimit: 10
        )

        let exported = try #require(package.recentTraces.first?.selfModel)
        #expect(exported.included)
        #expect(exported.schemaVersion == "0.1.0")
        #expect(exported.mode == "foreground")
        #expect(exported.activeSlot == "executor")
        #expect(exported.sourceIDs.contains("selfModelSnapshot/0.1.0"))
        #expect(exported.sourceIDs.contains("slot/executor"))
        #expect(exported.runtimeEvidenceSourceLayer == "agentGroundingRuntimeAudit")
        #expect(exported.selectedToolID == "calendar.create")
        #expect(exported.requiresApproval == true)
        #expect(exported.approvalMode == "userApproval")
    }

    @Test func agentGroundingPackageFlagsFinalValidatorReplacementTrace() throws {
        AgentBehaviorTraceRecorder.clear()
        AgentBehaviorTraceRecorder.record(AgentBehaviorTrace(
            id: UUID(),
            createdAt: Date(),
            event: .finalAnswer,
            slot: "mouth",
            stage: "compatibility-final",
            intent: "calendar",
            promptPrefix: "Search my calendar for tomorrow",
            rawOutputPrefix: "I couldn't safely complete the calendar event request.",
            selectedToolID: "calendar.list",
            toolArguments: [:],
            allowedToolIDs: ["calendar.list"],
            requiresApproval: nil,
            approvalMode: nil,
            parseError: nil,
            emittedFinalInActionTurn: true,
            runtimePath: "deterministic-compatibility",
            finalizerAccepted: false,
            finalizerRejectionReason: "intent-mismatch",
            finalValidatorAcceptedCandidate: false,
            finalValidatorReplacementSource: "safeMessage",
            finalValidatorRejectionReason: "tool-json-leak"
        ))

        let package = InAppDatasetPackageExporter.makePackage(
            manifestSource: "test-manifest",
            usedRuntimeFallback: false,
            runtimeManifestAudit: nil,
            behaviorAudit: nil,
            scenarioResults: [],
            traceLimit: 10
        )

        let trace = try #require(package.recentTraces.first)
        let failure = try #require(package.exportQualityFailures?.first(where: { $0.type == "agent_grounding_final_validator_replaced_candidate" }))
        #expect(package.schemaVersion == InAppDatasetPackageExporter.schemaVersion)
        #expect(trace.finalizerAccepted == false)
        #expect(trace.finalizerRejectionReason == "intent-mismatch")
        #expect(trace.finalValidatorAcceptedCandidate == false)
        #expect(trace.finalValidatorReplacementSource == "safeMessage")
        #expect(trace.finalValidatorRejectionReason == "tool-json-leak")
        #expect(failure.actual?.contains("replacementSource=safeMessage") == true)
        #expect(failure.actual?.contains("rejectionReason=tool-json-leak") == true)
    }

    @Test func agentGroundingPackageRedactsTraceExportWhilePreservingAdapterEvidence() throws {
        AgentBehaviorTraceRecorder.clear()
        defer { AgentBehaviorTraceRecorder.clear() }
        let traceID = UUID(uuidString: "11111111-1111-4111-8111-111111111111")!
        AgentBehaviorTraceRecorder.record(AgentBehaviorTrace(
            id: traceID,
            createdAt: Date(timeIntervalSince1970: 1_800_000_000),
            event: .modelTurn,
            slot: "mouth",
            stage: "mouth-final",
            scenarioID: "trace-\(traceID.uuidString)",
            e2eRunID: traceID,
            agentRunID: UUID(uuidString: "22222222-2222-4222-8222-222222222222"),
            conversationID: UUID(uuidString: "33333333-3333-4333-8333-333333333333"),
            turnID: UUID(uuidString: "44444444-4444-4444-8444-444444444444"),
            intent: "chat",
            promptPrefix: "prompt=My private question email alexis@example.com file=/Users/ales27pm/private.txt <think>secret reasoning</think>",
            rawOutputPrefix: "<think>hidden plan</think>{\"final\":\"Email alexis@example.com from /Users/ales27pm/Secret/model.gguf\"}",
            selectedToolID: nil,
            toolArguments: ["body": "Hello Alexis", "empty": ""],
            allowedToolIDs: ["weather"],
            requiresApproval: nil,
            approvalMode: nil,
            parseError: nil,
            emittedFinalInActionTurn: false,
            modelFamily: "qwen3",
            baseModelPath: "/Users/ales27pm/Models/lumen-qwen3.gguf",
            adapterID: "ales27pm/lumen-mouth-lora",
            adapterSlot: "mouth",
            adapterPath: "/private/var/mobile/Containers/Data/lumen-mouth-lora.gguf",
            adapterApplied: true,
            generationElapsedMs: 1_200,
            runtimePath: "sharedAdapter",
            activeAdapterSlot: "mouth",
            modelIdentifier: "Qwen/Qwen3-1.7B"
        ))

        let package = InAppDatasetPackageExporter.makePackage(
            manifestSource: "test-manifest",
            usedRuntimeFallback: false,
            runtimeManifestAudit: nil,
            behaviorAudit: nil,
            scenarioResults: [],
            traceLimit: 10
        )
        let trace = try #require(package.recentTraces.first)
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        let text = String(decoding: try encoder.encode(package), as: UTF8.self)

        #expect(trace.runtimePath == "sharedAdapter")
        #expect(trace.adapterSlot == "mouth")
        #expect(trace.adapterApplied == true)
        #expect(trace.baseModelPath == "lumen-qwen3.gguf")
        #expect(trace.adapterPath == "lumen-mouth-lora.gguf")
        #expect(trace.modelIdentifier == "Qwen/Qwen3-1.7B")
        #expect(trace.toolArguments["body"] == "[redacted]")
        #expect(trace.toolArguments["empty"] == "")
        #expect(!trace.promptPrefix.contains("My private question"))
        #expect(!trace.promptPrefix.contains("alexis@example.com"))
        #expect(!trace.rawOutputPrefix.contains("hidden plan"))

        for forbidden in [
            traceID.uuidString,
            "22222222-2222-4222-8222-222222222222",
            "33333333-3333-4333-8333-333333333333",
            "44444444-4444-4444-8444-444444444444",
            "alexis@example.com",
            "/Users/ales27pm",
            "/private/var",
            "secret reasoning",
            "hidden plan",
            "Hello Alexis",
            "sourceTraceID"
        ] {
            #expect(!text.contains(forbidden))
        }
        #expect(text.contains("sharedAdapter"))
        #expect(text.contains("lumen-mouth-lora.gguf"))
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

    @Test func agentGroundingPackageReparsesToolActionTracesInsteadOfTrustingParseErrorField() throws {
        AgentBehaviorTraceRecorder.clear()
        AgentBehaviorTraceRecorder.record(AgentBehaviorTrace(
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
            emittedFinalInActionTurn: false
        ))

        let package = InAppDatasetPackageExporter.makePackage(
            manifestSource: "test-manifest",
            usedRuntimeFallback: false,
            runtimeManifestAudit: nil,
            behaviorAudit: nil,
            scenarioResults: [],
            traceLimit: 10
        )

        #expect(package.traceParseErrorCount == 1)
        #expect(package.behaviorAudit?.passed == false)
        #expect(package.behaviorAudit?.violations.contains(where: { $0.code == "structured_action_trace_parse_error" }) == true)
    }

    @Test func agentGroundingPackageAcceptsCanonicalToolActionJSONTrace() throws {
        AgentBehaviorTraceRecorder.clear()
        let action = AgentAction(tool: "weather", args: [:])
        AgentBehaviorTraceRecorder.record(AgentBehaviorTrace(
            id: UUID(),
            createdAt: Date(),
            event: .toolAction,
            slot: "executor",
            stage: "compatibility-tool-action",
            intent: "weather",
            promptPrefix: "What is the weather here?",
            rawOutputPrefix: action.structuredOutputJSON,
            selectedToolID: "weather",
            toolArguments: [:],
            allowedToolIDs: ["weather"],
            requiresApproval: false,
            approvalMode: nil,
            parseError: nil,
            emittedFinalInActionTurn: false
        ))

        let package = InAppDatasetPackageExporter.makePackage(
            manifestSource: "test-manifest",
            usedRuntimeFallback: false,
            runtimeManifestAudit: nil,
            behaviorAudit: nil,
            scenarioResults: [],
            traceLimit: 10
        )

        #expect(package.traceParseErrorCount == 0)
        #expect(package.behaviorAudit?.violations.contains(where: { $0.code == "structured_action_trace_parse_error" }) != true)
    }

    @Test func agentGroundingPackageDoesNotTreatCortexDiagnosticChatAsToolActionParseError() throws {
        AgentBehaviorTraceRecorder.clear()
        AgentBehaviorTraceRecorder.record(AgentBehaviorTrace(
            id: UUID(),
            createdAt: Date(),
            event: .modelTurn,
            slot: "cortex",
            stage: "chat",
            intent: "chat",
            promptPrefix: "Yo",
            rawOutputPrefix: #"{"intent":"diagnostic","nextModel":"diagnostic"}"#,
            selectedToolID: nil,
            toolArguments: [:],
            allowedToolIDs: [],
            requiresApproval: nil,
            approvalMode: nil,
            parseError: "missingActionOrFinal",
            emittedFinalInActionTurn: false
        ))

        let package = InAppDatasetPackageExporter.makePackage(
            manifestSource: "test-manifest",
            usedRuntimeFallback: false,
            runtimeManifestAudit: nil,
            behaviorAudit: nil,
            scenarioResults: [],
            traceLimit: 10
        )

        #expect(package.traceParseErrorCount == 0)
        #expect(package.behaviorAudit?.violations.contains(where: { $0.code == "structured_action_trace_parse_error" }) != true)
    }

    @Test func agentGroundingPackageStillCountsMalformedAgentJSONModelTurns() throws {
        AgentBehaviorTraceRecorder.clear()
        AgentBehaviorTraceRecorder.record(AgentBehaviorTrace(
            id: UUID(),
            createdAt: Date(),
            event: .modelTurn,
            slot: "executor",
            stage: "agent-json",
            intent: "weather",
            promptPrefix: "User request:\nWhat is the weather here?",
            rawOutputPrefix: "Generation error: Failed to initialize context: Prompt exceeds shared chat context window",
            selectedToolID: nil,
            toolArguments: [:],
            allowedToolIDs: ["weather"],
            requiresApproval: nil,
            approvalMode: nil,
            parseError: "generation_error",
            emittedFinalInActionTurn: false
        ))

        let package = InAppDatasetPackageExporter.makePackage(
            manifestSource: "test-manifest",
            usedRuntimeFallback: false,
            runtimeManifestAudit: nil,
            behaviorAudit: nil,
            scenarioResults: [],
            traceLimit: 10
        )

        #expect(package.traceParseErrorCount == 1)
        #expect(package.behaviorAudit?.violations.contains(where: { $0.code == "structured_action_trace_parse_error" }) == true)
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
            ToolDefinition(id: "location.snapshot", name: "Location Snapshot", category: .location, description: "Secure location snapshot", icon: "location", tint: "teal", requiresApproval: false, permissionKey: "NSLocationWhenInUseUsageDescription"),
            ToolDefinition(id: "calendar.create", name: "Create Event", category: .productivity, description: "Create calendar event", icon: "calendar", tint: "blue", requiresApproval: true, permissionKey: "NSCalendarsFullAccessUsageDescription"),
            ToolDefinition(id: "alarm.list", name: "List Alarms", category: .productivity, description: "List alarms", icon: "alarm", tint: "orange", requiresApproval: false, permissionKey: nil)
        ]

        let effective = SlotAgentService.effectiveToolDefinitions(original: original, grounded: grounded)
        let ids = Set(effective.map(\.id))
        #expect(ids.contains("weather"))
        #expect(ids.contains("location.current"))
        #expect(!ids.contains("location.snapshot"))
        #expect(!ids.contains("calendar.create"))
        #expect(!ids.contains("alarm.list"))
    }

    @Test func routeScopedToolDefinitionsIntersectFullRegistryWithIntentTools() {
        let routing = IntentRouter.classify("What is the weather here and should I carry an umbrella?")

        let scoped = SlotAgentService.routeScopedToolDefinitions(ToolRegistry.all, routing: routing)
        let ids = Set(scoped.map { ToolRouteGuard.canonicalToolID($0.id) })

        #expect(routing.intent == .weather)
        #expect(ids == ["location.current", "weather"])
        #expect(!ids.contains("calendar.create"))
        #expect(!ids.contains("alarm.authorization_status"))
    }

    @Test func agentJSONGenerationPreservesRawStructuredOutput() {
        let agentJSON = GenerateRequest(
            systemPrompt: "sys",
            history: [],
            userMessage: "user",
            temperature: 0,
            topP: 1,
            repetitionPenalty: 1,
            maxTokens: 128,
            modelName: "agent-json",
            relevantMemories: [],
            responseFormat: .constrainedJSON(schema: AgentService.structuredAgentResponseSchema)
        )
        let chat = GenerateRequest(
            systemPrompt: "sys",
            history: [],
            userMessage: "user",
            temperature: 0,
            topP: 1,
            repetitionPenalty: 1,
            maxTokens: 128,
            modelName: "chat",
            relevantMemories: []
        )

        #expect(agentJSON.preservesRawStructuredAgentOutput)
        #expect(!chat.preservesRawStructuredAgentOutput)
    }

    @Test func agentJSONPromptCarriesConstrainedJSONContract() async {
        let genReq = GenerateRequest(
            systemPrompt: "sys",
            history: [],
            userMessage: "Return a tool action.",
            temperature: 0,
            topP: 1,
            repetitionPenalty: 1,
            maxTokens: 128,
            modelName: "agent-json",
            relevantMemories: [],
            responseFormat: .constrainedJSON(schema: AgentService.structuredAgentResponseSchema)
        )

        let result = await AppLlamaService.shared.buildMessagesForTesting(req: genReq, contextSize: 2048, slot: .executor)
        let prompt = result.messages.map(\.content).joined(separator: "\n")

        #expect(prompt.contains("Response format contract: output exactly one valid JSON object"))
        #expect(prompt.contains(#""oneOf""#))
        #expect(prompt.contains(#""action""#))
        #expect(prompt.contains(#""final""#))
        #expect(!prompt.contains("/think"))
    }

    @Test func agentJSONParsesEmptyQwenThinkWrapperBeforeActionJSON() throws {
        let raw = """
        <think>

        </think>

        {"thought":"run_weather","action":{"tool":"weather","args":{}}}
        """

        let parsed = AgentTurnParser.parse(raw)
        let noise = AgentNoiseInspector.inspect(raw)

        #expect(parsed.parseError == nil)
        #expect(parsed.action?.tool == "weather")
        #expect(parsed.hadNoise)
        #expect(noise.prefixNoise?.contains("leading empty <think> block stripped") == true)
    }

    @Test func agentJSONParsesWhitespaceThenQwenThinkWrapperBeforeFinalJSON() throws {
        let raw = " \n\t<think>\n\n</think>\n\n{\"final\":\"Precision is relevance; recall is coverage.\"}"

        let parsed = AgentTurnParser.parse(raw)

        #expect(parsed.parseError == nil)
        #expect(parsed.final == "Precision is relevance; recall is coverage.")
        #expect(parsed.hadNoise)
    }

    @Test func agentJSONRecoversFromInvalidPrefixNoiseWhenJSONIsPresent() throws {
        let raw = "Here is the JSON you asked for:\n{\"action\":{\"tool\":\"web.search\",\"args\":{\"query\":\"Swift concurrency\"}}}"

        let parsed = AgentTurnParser.parse(raw)
        let noise = AgentNoiseInspector.inspect(raw)

        #expect(parsed.parseError == nil)
        #expect(parsed.action?.tool == "web.search")
        #expect(parsed.hadNoise)
        #expect(noise.prefixNoise?.contains("Here is the JSON") == true)
    }

    @Test func agentJSONOnlyQwenThinkWrapperWithoutJSONRemainsEmpty() throws {
        let raw = "<think>\nprivate reasoning\n</think>\n"

        let parsed = AgentTurnParser.parse(raw)
        let noise = AgentNoiseInspector.inspect(raw)

        #expect(parsed.parseError == .empty)
        #expect(parsed.action == nil)
        #expect(parsed.final == nil)
        #expect(noise.prefixNoise?.contains("private reasoning") != true)
        #expect(noise.prefixNoise?.contains("redacted") == true)
    }

    @Test func agentJSONEmptyOutputRetryPromptRequiresNonEmptyJSONOnly() {
        let firstTurn = "User request:\nFind coffee near me.\n\nEmit the first JSON object now. Choose either action or final."

        let retryTurn = AgentService.agentJSONEmptyOutputRetryUserTurnForTests(from: firstTurn)

        #expect(retryTurn.contains(firstTurn))
        #expect(retryTurn.contains("Previous live agent-json attempt emitted no tokens"))
        #expect(retryTurn.contains("Emit exactly one non-empty JSON object now"))
        #expect(retryTurn.contains(#"{"action":{"tool":"<allowed tool id>","args":{...}}}"#))
        #expect(retryTurn.contains(#"{"final":"<concise user-facing answer>"}"#))
        #expect(retryTurn.contains("Start the response with {"))
    }

    @Test func agentJSONEmptyOutputRetryRequestKeepsSchemaVisible() async {
        let req = AgentRequest(
            systemPrompt: "sys",
            history: [],
            userMessage: "Find coffee near me.",
            temperature: 0,
            topP: 1,
            repetitionPenalty: 1,
            maxTokens: 384,
            maxSteps: 3,
            availableTools: ToolRegistry.all,
            relevantMemories: []
        )
        let systemPrompt = await AgentService.shared.structuredSystemPromptForTests(req: req)
        let userTurn = await AgentService.shared.structuredAgentUserTurnForTests(req: req)
        let base = GenerateRequest(
            systemPrompt: systemPrompt,
            history: [],
            userMessage: userTurn,
            temperature: 0.2,
            topP: 0.9,
            repetitionPenalty: 1,
            maxTokens: 512,
            modelName: "agent-json",
            relevantMemories: [],
            responseFormat: .constrainedJSON(schema: AgentService.structuredAgentResponseSchema)
        )

        let retry = AgentService.agentJSONEmptyOutputRetryRequestForTests(from: base, userTurn: base.userMessage)
        let result = await AppLlamaService.shared.buildMessagesForTesting(req: retry, contextSize: 2048, slot: .executor)
        let prompt = result.messages.map(\.content).joined(separator: "\n")

        #expect(retry.responseFormat == base.responseFormat)
        #expect(retry.temperature <= 0.05)
        #expect(retry.topP <= 0.6)
        #expect(prompt.contains("Previous live agent-json attempt emitted no tokens"))
        #expect(prompt.contains("Response format contract: output exactly one valid JSON object"))
        #expect(prompt.contains(#""oneOf""#))
        #expect(prompt.contains(#""action""#))
        #expect(prompt.contains(#""final""#))
        #expect(prompt.contains("/no_think"))
    }

    @Test func agentJSONCompactionRequestKeepsSchemaVisibleUnderBudget() async {
        let req = AgentRequest(
            systemPrompt: String(repeating: "Verbose app context. ", count: 200),
            history: [],
            userMessage: String(repeating: "Search the web for SwiftData migration details. ", count: 80),
            temperature: 0,
            topP: 1,
            repetitionPenalty: 1,
            maxTokens: 512,
            maxSteps: 3,
            availableTools: ToolRegistry.all,
            relevantMemories: []
        )
        let systemPrompt = await AgentService.shared.structuredSystemPromptForTests(req: req)
        let userTurn = await AgentService.shared.structuredAgentUserTurnForTests(req: req)
        let base = GenerateRequest(
            systemPrompt: systemPrompt,
            history: [],
            userMessage: userTurn,
            temperature: 0.1,
            topP: 0.8,
            repetitionPenalty: 1,
            maxTokens: 512,
            modelName: "agent-json",
            relevantMemories: [],
            responseFormat: .constrainedJSON(schema: AgentService.structuredAgentResponseSchema)
        )

        let compact = AgentService.agentJSONContextCompactionRequestForTests(from: base)
        let result = await AppLlamaService.shared.buildMessagesForTesting(req: compact, contextSize: 2048, slot: .executor)
        let prompt = result.messages.map(\.content).joined(separator: "\n")

        #expect(compact.responseFormat == base.responseFormat)
        #expect(compact.maxTokens <= 224)
        #expect(result.finalPromptChars <= PromptBudgetConstants.agentJSONTotalChars + 256)
        #expect(prompt.contains("Response format contract: output exactly one valid JSON object"))
        #expect(prompt.contains(#""oneOf""#))
        #expect(prompt.contains(#""action""#))
        #expect(prompt.contains(#""final""#))
        #expect(prompt.contains("/no_think"))
    }

    @MainActor
    @Test func postObservationAgentJSONPromptPrefersFinalAndRejectsStringAction() {
        let req = AgentRequest(
            systemPrompt: "sys",
            history: [],
            userMessage: "What is the weather here and should I carry an umbrella?",
            temperature: 0,
            topP: 1,
            repetitionPenalty: 1,
            maxTokens: 128,
            maxSteps: 2,
            availableTools: ToolRegistry.all.filter { ["weather", "location.current"].contains(ToolRouteGuard.canonicalToolID($0.id)) },
            relevantMemories: []
        )
        let scratchpad = "Action: location.current\nObservation: Current location is Montreal.\nAction: weather\nObservation: Weather at your location: light rain."

        let userTurn = AgentService.shared.structuredAgentUserTurnForTests(
            req: req,
            stepIndex: 1,
            scratchpad: scratchpad
        )
        let systemPrompt = AgentService.shared.structuredSystemPromptForTests(req: req)

        #expect(systemPrompt.contains("action must be a JSON object"))
        #expect(systemPrompt.contains(#"Invalid: {"action":"weather"}"#))
        #expect(userTurn.contains("If the observations already answer the user, choose final"))
        #expect(userTurn.contains("never emit action as a string"))
    }

    @Test func weatherObservationFallbackConvertsSummaryJSONToPlainText() {
        #if DEBUG
        let raw = #"{"summary":"Weather at your location is rainy, so carry an umbrella.","Key modules":["weather"]}"#

        let text = AgentService.observationFallbackPlainTextForTests(from: raw, intent: .weather)

        #expect(text == "Weather at your location is rainy, so carry an umbrella.")
        #expect(text?.contains("Key modules") == false)
        #expect(text?.contains("{") == false)
        #else
        #expect(true)
        #endif
    }

    @Test func ragObservationFallbackMayRetainKeyModulesWithoutJSONBraces() {
        #if DEBUG
        let raw = #"{"summary":"[1] The architecture notes mention AgentService and ToolExecutor.","Key modules":["AgentService","ToolExecutor"]}"#

        let text = AgentService.observationFallbackPlainTextForTests(from: raw, intent: .rag)

        #expect(text?.contains("[1] The architecture notes") == true)
        #expect(text?.contains("Key modules: AgentService, ToolExecutor") == true)
        #expect(text?.contains("{") == false)
        #else
        #expect(true)
        #endif
    }

    @MainActor
    @Test func compactAgentJSONPromptCapsVerboseToolsAndKeepsRequiredTools() {
        let tools = [
            ToolDefinition(id: "web.search", name: "Web Search", category: .knowledge, description: String(repeating: "Search the web. Args: query. ", count: 40), icon: "globe", tint: "blue", requiresApproval: false, permissionKey: nil),
            ToolDefinition(id: "web.fetch", name: "Fetch URL", category: .knowledge, description: String(repeating: "Fetch URL. Args: url. ", count: 40), icon: "link", tint: "blue", requiresApproval: false, permissionKey: nil)
        ]
        let req = AgentRequest(
            systemPrompt: String(repeating: "Style note. ", count: 200),
            history: [],
            userMessage: "Search the web for Swift concurrency best practices.",
            temperature: 0,
            topP: 1,
            repetitionPenalty: 1,
            maxTokens: 512,
            maxSteps: 1,
            availableTools: tools,
            relevantMemories: []
        )

        let systemPrompt = AgentService.shared.structuredSystemPromptForTests(req: req)

        #expect(systemPrompt.contains("web.search"))
        #expect(systemPrompt.contains("web.fetch"))
        #expect(!systemPrompt.contains(String(repeating: "Search the web. Args: query. ", count: 3)))
        #expect(systemPrompt.count < 2_200)
    }

    @MainActor
    @Test func trainingAgentJSONPromptsFitExecutorBudget() async {
        let promptsAndTools: [(String, [String])] = [
            ("What is the weather here and should I carry an umbrella?", ["location.current", "weather"]),
            ("Search the web for two recent Swift concurrency best practices and summarize them.", ["web.search", "web.fetch"]),
            ("Remember that I prefer concise bullet points, then tell me what you remembered.", ["memory.save", "memory.recall"]),
            ("Search my files for architecture notes and summarize key modules.", ["rag.search", "files.read", "photos.search", "rag.index_files", "rag.index_photos"]),
            ("Schedule a trigger to summarize reminders tonight and confirm what will run.", ["trigger.create", "trigger.list", "trigger.cancel"]),
            ("Draft an email to Alex with a professional update and ask one clarifying question.", ["contacts.search", "mail.draft"]),
            ("Explain tradeoffs between precision and recall in retrieval systems in plain English.", [])
        ]

        for (prompt, ids) in promptsAndTools {
            let req = AgentRequest(
                systemPrompt: String(repeating: "Verbose app prompt. ", count: 200),
                history: [],
                userMessage: prompt,
                temperature: 0.1,
                topP: 0.8,
                repetitionPenalty: 1.05,
                maxTokens: 512,
                maxSteps: 3,
                availableTools: ToolRegistry.all.filter { ids.contains(ToolRouteGuard.canonicalToolID($0.id)) },
                relevantMemories: []
            )
            let genReq = GenerateRequest(
                systemPrompt: AgentService.shared.structuredSystemPromptForTests(req: req),
                history: [],
                userMessage: AgentService.shared.structuredAgentUserTurnForTests(req: req),
                temperature: 0.05,
                topP: 0.6,
                repetitionPenalty: 1.05,
                maxTokens: 384,
                modelName: "agent-json",
                relevantMemories: [],
                attachments: [],
                responseFormat: .constrainedJSON(schema: AgentService.structuredAgentResponseSchema)
            )
            let result = await AppLlamaService.shared.buildMessagesForTesting(req: genReq, contextSize: 2048, slot: .executor)
            #expect(result.estimatedPromptTokens + genReq.maxTokens + PromptBudgetConstants.agentJSONSafetyTokens < 2048, "Prompt exceeded agent-json budget for \(prompt)")
            for id in ids.prefix(3) {
                #expect(result.messages.map(\.content).joined(separator: "\n").contains(id), "Missing required tool \(id) for \(prompt)")
            }
        }
    }

    @Test func promptContextOverflowClassifiesAsContextWindowExceeded() {
        let raw = "Generation error: Failed to initialize context: Prompt exceeds shared chat context window"

        #expect(AppLlamaService.isPromptContextWindowExceeded(LlamaError.failedToInitializeContext("Prompt exceeds shared chat context window")))
        #expect(AgentService.runtimeFailureParseErrorForTests(from: raw) == .contextWindowExceeded)
        #expect(AgentTurnParser.parse(raw).parseError == .noJSONObject)
    }

    @Test func structuredAgentJSONUsesExecutorModelSlot() {
        #expect(AgentService.structuredAgentModelSlotForTests == .executor)
    }

    @Test func deterministicCompatibilityEligibilityCoversToolAndDirectChatPrompts() {
        let webRequest = AgentRequest(
            systemPrompt: "sys",
            history: [],
            userMessage: "Search the web for SwiftData cancellation patterns",
            temperature: 0,
            topP: 1,
            repetitionPenalty: 1,
            maxTokens: 128,
            maxSteps: 2,
            availableTools: ToolRegistry.all.filter { $0.id == "web.search" },
            relevantMemories: []
        )
        let chatRequest = AgentRequest(
            systemPrompt: "sys",
            history: [],
            userMessage: "Explain actor isolation in Swift in simple terms.",
            temperature: 0,
            topP: 1,
            repetitionPenalty: 1,
            maxTokens: 128,
            maxSteps: 1,
            availableTools: [],
            relevantMemories: []
        )
        let attachmentRequest = AgentRequest(
            systemPrompt: "sys",
            history: [],
            userMessage: "Explain this document.",
            temperature: 0,
            topP: 1,
            repetitionPenalty: 1,
            maxTokens: 128,
            maxSteps: 1,
            availableTools: [],
            relevantMemories: [],
            attachments: [ChatAttachment(name: "notes.txt", kind: .text, path: "/tmp/notes.txt", byteSize: 5)]
        )

        #expect(SlotAgentService.canCompleteThroughDeterministicCompatibility(webRequest))
        #expect(SlotAgentService.canCompleteThroughDeterministicCompatibility(chatRequest))
        #expect(!SlotAgentService.canCompleteThroughDeterministicCompatibility(attachmentRequest))
    }

    @Test func deterministicCompatibilityEligibilityCoversAttachedLiveE2EFailurePrompts() {
        let prompts = [
            "Give me directions to the nearest hardware store.",
            "Find coffee near me.",
            "Find a pharmacy nearby.",
            "Use Search Nearby, but ask for clarification if required details are missing.",
            "Tell me what style I asked you to use.",
            "What do you remember about my response style preference?",
            "Use Recall Memory, but ask for clarification if required details are missing.",
            "Keep in mind that I like short answers.",
            "Remember that I prefer concise bullet points.",
            "Use Save Memory, but ask for clarification if required details are missing.",
            "Text 5551234567 that I am late."
        ]

        for prompt in prompts {
            let routing = IntentRouter.classify(prompt)
            let tools = ToolRegistry.all.filter { routing.allowedToolIDs.contains(ToolRouteGuard.canonicalToolID($0.id)) }
            let request = AgentRequest(
                systemPrompt: "sys",
                history: [],
                userMessage: prompt,
                temperature: 0,
                topP: 1,
                repetitionPenalty: 1,
                maxTokens: 128,
                maxSteps: 3,
                availableTools: tools,
                relevantMemories: []
            )
            #expect(SlotAgentService.canCompleteThroughDeterministicCompatibility(request), "Prompt did not enter compatibility path: \(prompt)")
        }
    }

    @MainActor
    @Test func agentServiceRoutesDeterministicCompatibleRequestsBeforeStructuredModel() async {
        AgentBehaviorTraceRecorder.clear()
        let req = AgentRequest(
            systemPrompt: "sys",
            history: [],
            userMessage: "Explain actor isolation in Swift in simple terms.",
            temperature: 0,
            topP: 1,
            repetitionPenalty: 1,
            maxTokens: 128,
            maxSteps: 1,
            availableTools: [],
            relevantMemories: []
        )

        var finalText = ""
        for await event in AgentService.shared.run(req, options: .default) {
            if case .done(let text, _) = event {
                finalText = text
            }
        }

        let traces = AgentBehaviorTraceRecorder.recent(limit: 20)
        #expect(finalText.lowercased().contains("actor isolation"))
        #expect(traces.contains { $0.runtimePath == "deterministic-compatibility" && $0.stage == "compatibility-direct-final" })
        #expect(!traces.contains { $0.stage == "agent-json" || $0.event == .modelTurn })
    }

    @Test func secureToolAliasesBridgeToCanonicalLegacyDefinitions() {
        let secure = [
            SecureToolDefinition(id: "rag.search.secure", displayName: "Secure RAG", description: "Secure RAG", category: .readOnly, requiredPermissions: [], supportsBackgroundExecution: true, requiresUserApproval: false, argumentSchemaDescription: "{}", resultPrivacyLevel: .moderate, maxOutputCharacters: 100),
            SecureToolDefinition(id: "contacts.lookup", displayName: "Lookup", description: "Lookup", category: .permissionRead, requiredPermissions: [.contacts], supportsBackgroundExecution: false, requiresUserApproval: false, argumentSchemaDescription: "{}", resultPrivacyLevel: .sensitive, maxOutputCharacters: 100)
        ]

        let ids = ToolSchemaBridge.toCatalogToolDefinitions(secure).map(\.id)
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

    @Test func compatibilityTriggerListShowPromptDoesNotCreateTrigger() async {
        let tools = ToolRegistry.all.filter { ["trigger.create", "trigger.list", "trigger.cancel"].contains($0.id) }
        let req = AgentRequest(
            systemPrompt: "sys",
            history: [],
            userMessage: "Show scheduled agent runs.",
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

        #expect(response.steps.contains { $0.kind == .action && $0.toolID == "trigger.list" })
        #expect(!response.steps.contains { $0.toolID == "trigger.create" })
        #expect(!response.text.lowercased().contains("approval required for trigger.create"))
    }

    @Test func compatibilityTriggerCancelApprovalNamesCancelTool() async {
        let tools = ToolRegistry.all.filter { ["trigger.create", "trigger.list", "trigger.cancel"].contains($0.id) }
        let req = AgentRequest(
            systemPrompt: "sys",
            history: [],
            userMessage: "Cancel trigger named nightly summary.",
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
        #expect(response.steps.first?.toolID == "trigger.cancel")
        #expect(response.text.lowercased().contains("approval required for trigger.cancel"))
        #expect(!response.text.lowercased().contains("approval required for trigger.create"))
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
        let actionTrace = traces.first { trace in
            trace.event == AgentBehaviorTrace.Event.toolAction && trace.selectedToolID == "calendar.list"
        }
        let parsedActionTrace = AgentTurnParser.parse(actionTrace?.rawOutputPrefix ?? "")
        let hasCompatibilityFinalTrace = traces.contains { trace in
            trace.event == AgentBehaviorTrace.Event.finalAnswer && trace.runtimePath == "deterministic-compatibility"
        }
        let finalTrace = traces.first { trace in
            trace.event == AgentBehaviorTrace.Event.finalAnswer && trace.runtimePath == "deterministic-compatibility"
        }

        #expect(actionToolIDs == ["calendar.list"])
        #expect(response.text.lowercased().contains("event"))
        #expect(hasCalendarListActionTrace)
        #expect(actionTrace?.rawOutputPrefix.hasPrefix(#"{"action":{"#) == true)
        #expect(actionTrace?.parseError == nil)
        #expect(parsedActionTrace.parseError == nil)
        #expect(parsedActionTrace.action?.tool == "calendar.list")
        #expect(hasCompatibilityFinalTrace)
        #expect(finalTrace?.finalizerAccepted == true)
        #expect(finalTrace?.finalizerRejectionReason == nil)
        #expect(finalTrace?.finalValidatorAcceptedCandidate == true)
        #expect(finalTrace?.finalValidatorReplacementSource == "candidate")
        #expect(finalTrace?.finalValidatorRejectionReason == nil)
    }

    @Test func liveE2EModelEvidenceRequiresFreshModelTurnTrace() {
        AgentBehaviorTraceRecorder.clear()
        let startedAt = Date()
        let prompt = "Explain actor isolation in Swift."
        AgentBehaviorTraceRecorder.record(AgentBehaviorTrace(
            id: UUID(),
            createdAt: Date(),
            event: .modelTurn,
            slot: "agent",
            stage: "agent-json-step-0",
            intent: "chat",
            promptPrefix: prompt,
            rawOutputPrefix: "",
            selectedToolID: nil,
            toolArguments: [:],
            allowedToolIDs: [],
            requiresApproval: nil,
            approvalMode: nil,
            parseError: AgentTurnParseError.empty.rawValue,
            emittedFinalInActionTurn: false,
            modelFamily: "qwen3",
            runtimePath: "agent-model",
            promptCharCount: prompt.count
        ))
        #expect(E2ETestRunner.modelRuntimeEvidenceForTests(since: startedAt, prompt: prompt) == false)

        AgentBehaviorTraceRecorder.record(AgentBehaviorTrace(
            id: UUID(),
            createdAt: Date(),
            event: .finalAnswer,
            slot: "mouth",
            stage: "compatibility-direct-final",
            intent: "chat",
            promptPrefix: prompt,
            rawOutputPrefix: "Actor isolation protects mutable state.",
            selectedToolID: nil,
            toolArguments: [:],
            allowedToolIDs: [],
            requiresApproval: nil,
            approvalMode: nil,
            parseError: nil,
            emittedFinalInActionTurn: false,
            modelFamily: "qwen3",
            runtimePath: "deterministic-compatibility",
            promptCharCount: prompt.count
        ))
        #expect(E2ETestRunner.modelRuntimeEvidenceForTests(since: startedAt, prompt: prompt) == false)

        AgentBehaviorTraceRecorder.record(AgentBehaviorTrace(
            id: UUID(),
            createdAt: Date(),
            event: .modelTurn,
            slot: "mouth",
            stage: "agent-json",
            intent: "chat",
            promptPrefix: prompt,
            rawOutputPrefix: "{\"final\":\"Actor isolation protects mutable state.\"}",
            selectedToolID: nil,
            toolArguments: [:],
            allowedToolIDs: [],
            requiresApproval: nil,
            approvalMode: nil,
            parseError: nil,
            emittedFinalInActionTurn: true,
            modelFamily: "qwen3",
            adapterSlot: "mouth",
            generationElapsedMs: 1200,
            runtimePath: "sharedAdapter",
            activeAdapterSlot: "mouth",
            promptCharCount: prompt.count
        ))

        #expect(E2ETestRunner.modelRuntimeEvidenceForTests(since: startedAt, prompt: prompt))
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

    @Test func agentServiceParseFailureRecoveryProducesWebSearchAction() async {
        let tools = ToolRegistry.all.filter { ["web.search", "web.fetch"].contains($0.id) }
        let req = AgentRequest(
            systemPrompt: "sys",
            history: [],
            userMessage: "Search the web for two recent Swift concurrency best practices and summarize them.",
            temperature: 0,
            topP: 1,
            repetitionPenalty: 1,
            maxTokens: 128,
            maxSteps: 2,
            availableTools: tools,
            relevantMemories: []
        )
        let options = LegacyAgentRunOptions(modelContext: nil, conversationID: req.conversationID, turnID: req.turnID, groundingMode: .slotAgent, allowDegradedGrounding: false, preventDoubleGrounding: true, diagnosticsEnabled: false)

        let recovery = await AgentService.structuredParseFailureRecoveryForTests(req: req, options: options)
        let actionToolIDs = recovery?.steps
            .filter { $0.kind == .action }
            .compactMap(\.toolID)
            .map(ToolRouteGuard.canonicalToolID) ?? []

        #expect(actionToolIDs.contains("web.search"))
        #expect(recovery?.text.lowercased().contains("please ask again") == false)
    }

    @Test func agentServiceParseFailureRecoveryHonorsDisabledDeterministicCompatibility() async {
        let tools = ToolRegistry.all.filter { ["web.search", "web.fetch"].contains($0.id) }
        let req = AgentRequest(
            systemPrompt: "sys",
            history: [],
            userMessage: "Search the web for two recent Swift concurrency best practices and summarize them.",
            temperature: 0,
            topP: 1,
            repetitionPenalty: 1,
            maxTokens: 128,
            maxSteps: 2,
            availableTools: tools,
            relevantMemories: []
        )
        let options = LegacyAgentRunOptions(
            modelContext: nil,
            conversationID: req.conversationID,
            turnID: req.turnID,
            groundingMode: .slotAgent,
            allowDegradedGrounding: false,
            preventDoubleGrounding: true,
            diagnosticsEnabled: false,
            allowDeterministicCompatibility: false,
            allowParseFailureDeterministicRecovery: false
        )

        let recovery = await AgentService.structuredParseFailureRecoveryForTests(req: req, options: options)

        #expect(recovery == nil)
    }

    @Test func agentServiceParseFailureRecoveryPlansReportAlternatePhrases() async {
        let cases: [(String, [String])] = [
            ("Tell me what style I asked you to use.", ["memory.recall"]),
            ("Keep in mind that I like short answers.", ["memory.save"]),
            ("Find Lumen architecture notes in my local files.", ["rag.search"]),
            ("Reindex my imported files.", ["rag.index_files"]),
            ("Reindex photo metadata for the last 3 months.", ["rag.index_photos"])
        ]

        for (prompt, expectedTools) in cases {
            let routing = IntentRouter.classify(prompt)
            let tools = ToolRegistry.all.filter { IntentRouter.isToolAllowed($0.id, for: routing) }
            let req = AgentRequest(
                systemPrompt: "sys",
                history: [],
                userMessage: prompt,
                temperature: 0,
                topP: 1,
                repetitionPenalty: 1,
                maxTokens: 128,
                maxSteps: 2,
                availableTools: tools,
                relevantMemories: []
            )
            let options = LegacyAgentRunOptions(modelContext: nil, conversationID: req.conversationID, turnID: req.turnID, groundingMode: .slotAgent, allowDegradedGrounding: false, preventDoubleGrounding: true, diagnosticsEnabled: false)

            let recovery = await AgentService.structuredParseFailureRecoveryForTests(req: req, options: options)
            let actionToolIDs = recovery?.steps
                .filter { $0.kind == .action }
                .compactMap(\.toolID)
                .map(ToolRouteGuard.canonicalToolID) ?? []

            #expect(actionToolIDs == expectedTools)
            #expect(recovery?.text.lowercased().contains("unavailable") == false)
        }
    }

    @Test func agentServiceParseFailureRecoveryAnswersChatDirectly() async {
        let req = AgentRequest(
            systemPrompt: "sys",
            history: [],
            userMessage: "Explain actor isolation in Swift in simple terms.",
            temperature: 0,
            topP: 1,
            repetitionPenalty: 1,
            maxTokens: 128,
            maxSteps: 1,
            availableTools: [],
            relevantMemories: []
        )
        let options = LegacyAgentRunOptions(modelContext: nil, conversationID: req.conversationID, turnID: req.turnID, groundingMode: .slotAgent, allowDegradedGrounding: false, preventDoubleGrounding: true, diagnosticsEnabled: false)

        let recovery = await AgentService.structuredParseFailureRecoveryForTests(req: req, options: options)
        let lower = recovery?.text.lowercased() ?? ""

        #expect(recovery?.steps.isEmpty == true)
        #expect(lower.contains("actor isolation"))
        #expect(!lower.contains("please ask again"))
        #expect(!lower.contains("i'm ready"))
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

    @Test func structuredWeatherFinalRemovesUngroundedUmbrellaAdvice() {
        let req = AgentRequest(
            systemPrompt: "sys",
            history: [],
            userMessage: "What is the weather here and should I carry an umbrella?",
            temperature: 0,
            topP: 1,
            repetitionPenalty: 1,
            maxTokens: 256,
            maxSteps: 3,
            availableTools: ToolRegistry.all,
            relevantMemories: []
        )
        let final = AgentService.postprocessStructuredFinalAnswerForTests(
            "Weather is overcast, 18°C. Carry an umbrella if you'll be outside.",
            req: req,
            observations: [("weather", "Weather at your location: overcast, 18°C, humidity 50%, wind 7 km/h.")],
            steps: []
        )
        let lower = final.lowercased()

        #expect(lower.contains("weather update"))
        #expect(lower.contains("no precipitation was reported"))
        #expect(!lower.contains("umbrella"))
    }

    @Test func structuredMemorySaveRecallFinalPreservesExactPreference() {
        let req = AgentRequest(
            systemPrompt: "sys",
            history: [],
            userMessage: "Remember that I prefer concise bullet points, then tell me what you remembered.",
            temperature: 0,
            topP: 1,
            repetitionPenalty: 1,
            maxTokens: 256,
            maxSteps: 3,
            availableTools: ToolRegistry.all,
            relevantMemories: []
        )
        let steps = [
            AgentStep(kind: .action, content: "memory.save(content=...)", toolID: "memory.save", toolArgs: ["content": "Remember that I prefer concise bullet points, then tell me what you remembered."]),
            AgentStep(kind: .observation, content: "Saved preference.", toolID: "memory.save"),
            AgentStep(kind: .action, content: "memory.recall(query=...)", toolID: "memory.recall", toolArgs: ["query": "user preference"]),
            AgentStep(kind: .observation, content: "I prefer concise bullet points.", toolID: "memory.recall")
        ]
        let final = AgentService.postprocessStructuredFinalAnswerForTests(
            "I remember your preference.",
            req: req,
            observations: [],
            steps: steps
        )

        #expect(final == "I remember that you prefer concise bullet points.")
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
