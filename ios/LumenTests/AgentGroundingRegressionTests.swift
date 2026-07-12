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

    @Test func persistentAgentBehaviorTraceRedactsRawPromptOutputArgumentsAndPaths() {
        let trace = AgentBehaviorTrace(
            id: UUID(),
            createdAt: Date(timeIntervalSince1970: 1_800_000_000),
            event: .toolAction,
            slot: "executor",
            stage: "agent-json-step-0",
            intent: "mail",
            promptPrefix: "Email Sarah the launch password is swordfish.",
            rawOutputPrefix: #"{"tool":"outlook.mail.send","arguments":{"body":"swordfish"}}"#,
            selectedToolID: "outlook.mail.send",
            toolArguments: ["body": "launch password is swordfish", "to": "sarah@example.com"],
            allowedToolIDs: ["outlook.mail.send"],
            requiresApproval: true,
            approvalMode: "foreground",
            parseError: nil,
            emittedFinalInActionTurn: false,
            baseModelPath: "/private/var/mobile/Containers/Data/model.gguf",
            adapterPath: "/private/var/mobile/Containers/Data/adapter.gguf",
            runtimePath: "/private/var/mobile/Containers/Data/model.gguf",
            promptCharCount: 44
        )

        let redacted = trace.redactedForPersistentDiagnostics()

        #expect(!redacted.promptPrefix.contains("swordfish"))
        #expect(!redacted.rawOutputPrefix.contains("outlook.mail.send"))
        #expect(!redacted.toolArguments.description.contains("sarah@example.com"))
        #expect(!(redacted.baseModelPath ?? "").contains("/private"))
        #expect(!(redacted.adapterPath ?? "").contains("/private"))
        #expect(!(redacted.runtimePath ?? "").contains("/private"))
        #expect(redacted.promptPrefix.contains("sha256="))
        #expect(redacted.toolArguments["body"]?.contains("sha256=") == true)
        #expect(redacted.selectedToolID == "outlook.mail.send")
        #expect(redacted.promptCharCount == 44)
    }

    @Test func persistentAgentParseTracesRedactRawPromptAndModelOutput() {
        let failure = AgentParseFailureTrace(
            id: UUID(),
            createdAt: Date(timeIntervalSince1970: 1_800_000_000),
            parseError: "malformed_json",
            modelName: "agent-json",
            temperature: 0,
            topP: 1,
            maxTokens: 128,
            stepIndex: 1,
            systemPromptPrefix: "System prompt with private policy",
            userTurnPrefix: "User asks about secret account 1234",
            rawOutputPrefix: "Raw model output with secret account 1234",
            streamedThoughtPrefix: "Hidden reasoning",
            streamedFinalPrefix: "Visible final",
            selectedJSONPrefix: #"{"arguments":{"body":"secret account 1234"}}"#,
            prefixNoise: "prefix secret",
            suffixNoise: "suffix secret"
        )
        let noise = AgentParseNoiseTrace(
            id: UUID(),
            createdAt: Date(timeIntervalSince1970: 1_800_000_000),
            modelName: "agent-json",
            temperature: 0,
            topP: 1,
            maxTokens: 128,
            stepIndex: 1,
            systemPromptPrefix: "System prompt with private policy",
            userTurnPrefix: "User asks about secret account 1234",
            rawOutputPrefix: "Raw model output with secret account 1234",
            selectedJSONPrefix: #"{"arguments":{"body":"secret account 1234"}}"#,
            prefixNoise: "prefix secret",
            suffixNoise: "suffix secret"
        )

        let redactedFailure = failure.redactedForPersistentDiagnostics()
        let redactedNoise = noise.redactedForPersistentDiagnostics()

        #expect(!redactedFailure.systemPromptPrefix.contains("private policy"))
        #expect(!redactedFailure.userTurnPrefix.contains("1234"))
        #expect(!redactedFailure.rawOutputPrefix.contains("Raw model output"))
        #expect(!(redactedFailure.selectedJSONPrefix ?? "").contains("secret account"))
        #expect(redactedFailure.parseError == "malformed_json")
        #expect(redactedFailure.rawOutputPrefix.contains("sha256="))
        #expect(!redactedNoise.userTurnPrefix.contains("1234"))
        #expect((redactedNoise.selectedJSONPrefix ?? "").contains("sha256="))
    }

    @Test func developerTraceCodecRedactsPromptsMemoryToolArgumentsAndAttachmentPathsBeforePersistence() throws {
        let rawPath = "/private/var/mobile/Containers/Data/Application/secret-folder/launch-plan.txt"
        let trace = DeveloperTrace(
            conversationID: UUID(),
            messageID: UUID(),
            modelName: "chat",
            systemPrompt: "System prompt with internal launch password",
            developerPrompt: "Developer prompt with private policy",
            userPrompt: "Summarize Sarah's launch plan and password.",
            resolvedContext: [
                TraceContextItem(
                    role: "attachment",
                    title: "launch-plan.txt",
                    content: "Attachment contains the launch password.",
                    source: rawPath,
                    metadata: ["name": "launch-plan.txt", "path": rawPath]
                )
            ],
            retrievedMemory: [
                TraceMemoryItem(
                    content: "Sarah password memory",
                    scope: "conversation",
                    authority: "referenceOnly",
                    createdAt: nil,
                    expiresAt: nil,
                    source: rawPath,
                    topic: "Sarah"
                )
            ],
            toolPlan: [
                TraceToolPlanItem(
                    toolID: "outlook.mail.send",
                    reason: "Email Sarah the launch password",
                    requiresApproval: true,
                    arguments: ["body": "launch password"]
                )
            ],
            toolCalls: [
                TraceToolCall(
                    toolID: "outlook.mail.send",
                    arguments: ["body": "launch password"],
                    status: "success",
                    result: "Sent to sarah@example.com"
                )
            ],
            agentMessages: [
                TraceAgentMessage(
                    role: "assistant",
                    content: "The launch password is swordfish.",
                    toolID: nil,
                    metadata: ["raw": "swordfish"]
                )
            ],
            rawModelOutput: "Raw model output with swordfish",
            reasoningText: "Hidden reasoning about swordfish",
            visibleAnswer: "Visible answer with swordfish",
            parserWarnings: ["warning mentions swordfish"],
            tokenUsage: nil,
            finishReason: "stop",
            error: "error mentions swordfish"
        )

        let encoded = try #require(DeveloperTraceCodec.encode(trace))

        #expect(!encoded.contains("swordfish"))
        #expect(!encoded.contains("launch password"))
        #expect(!encoded.contains("launch-plan.txt"))
        #expect(!encoded.contains(rawPath))
        #expect(encoded.contains("sha256="))
        #expect(encoded.contains("outlook.mail.send"))
        let decoded = try #require(DeveloperTraceCodec.decode(encoded))
        #expect(decoded.userPrompt.contains("sha256="))
        #expect(decoded.resolvedContext.first?.source?.contains("sha256=") == true)
        #expect(decoded.toolCalls.first?.arguments["body"]?.contains("sha256=") == true)
    }

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
        #expect(mailArgs["subject"]?.required == false)
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

        let payload = WebRichContentPayload(
            kind: .searchResults,
            query: "diy underground shelter",
            results: [
                WebSearchResultPayload(
                    title: "Underground Shelter Planning",
                    url: "https://example.com/shelter-planning",
                    snippet: "Start with drainage, ventilation, and local permit constraints before choosing a shelter design.",
                    source: "Example Preparedness",
                    mediaKind: nil
                ),
                WebSearchResultPayload(
                    title: "Safe Room Ventilation Basics",
                    url: "https://example.com/ventilation",
                    snippet: "Ventilation and emergency exits are core safety requirements for enclosed shelter spaces.",
                    source: "Example Safety",
                    mediaKind: nil
                )
            ]
        )
        let web = ToolObservationFinalizer.immediateFinalIfSafe(
            intent: .webSearch,
            toolID: "web.search",
            observation: "Found 5 results for diy underground shelter \(payload.encodedMarker())",
            originalPrompt: "Search web for diy underground shelter and summarize the findings."
        )
        #expect(web?.contains("<lumen_web_payload>") == true)
        let visibleWeb = WebRichContentPayload.removingMarkers(from: web ?? "")
        #expect(visibleWeb.contains("Summary:") == true)
        #expect(visibleWeb.lowercased().contains("drainage") == true)
        #expect(visibleWeb.lowercased().contains("web search results") == false)
        #expect(visibleWeb.contains("https://") == false)

        let maps = ToolObservationFinalizer.immediateFinalIfSafe(
            intent: .maps,
            toolID: "maps.search",
            observation: "Tim Hortons — Avenue de la Plaza",
            originalPrompt: "Find coffee near me."
        )
        #expect(maps?.lowercased().contains("maps search results") == true)
        #expect(maps?.lowercased().contains("tim hortons") == true)

        let memory = ToolObservationFinalizer.immediateFinalIfSafe(
            intent: .memory,
            toolID: "memory.recall",
            observation: "- [E7A33820] I prefer concise bullet points | kind=fact | score=0.00 | source=agent",
            originalPrompt: "What do you remember about my response style preference?"
        )
        #expect(memory == "I remember that you prefer concise bullet points.")
        #expect(memory?.contains("E7A33820") == false)
        #expect(memory?.contains("score=") == false)

        let memoryUnavailable = ToolObservationFinalizer.immediateFinalIfSafe(
            intent: .memory,
            toolID: "memory.recall",
            observation: "Memory unavailable. Diagnostic: swiftdata_shared_container_unavailable.",
            originalPrompt: "What do you remember?"
        )
        #expect(memoryUnavailable == "Memory unavailable.")
        #expect(memoryUnavailable?.lowercased().contains("diagnostic") == false)

        let multiFactMemory = ToolObservationFinalizer.immediateFinalIfSafe(
            intent: .memory,
            toolID: "memory.recall",
            observation: """
            - [A1] I prefer concise bullet points | kind=fact | score=0.90 | source=agent
            - [A2] User's name is Alexis | kind=fact | score=0.88 | source=agent
            """,
            originalPrompt: "What do you remember?"
        )
        #expect(multiFactMemory == "I remember that you prefer concise bullet points; your name is Alexis.")

        let negatedPreference = ToolObservationFinalizer.immediateFinalIfSafe(
            intent: .memory,
            toolID: "memory.recall",
            observation: "- [A1] I do not prefer verbose answers | kind=fact | score=0.00 | source=agent",
            originalPrompt: "What do you remember about my response style?"
        )
        #expect(negatedPreference == "I remember that I do not prefer verbose answers.")
        #expect(negatedPreference?.contains("you prefer verbose answers") == false)

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

    @Test func nativeKernelPolicyFirstContinuesMapsSearchAfterDegradedLocation() {
        let routing = IntentRoutingDecision(
            intent: .maps,
            allowedToolIDs: ["location.current", "maps.search"],
            requiresClarification: false,
            clarificationPrompt: nil
        )

        #expect(AssistantKernel.shouldContinueNativeToolChainAfterNonSuccess(
            after: "Position snapshot is disabled in this build.",
            currentToolID: "location.current",
            nextToolID: "maps.search",
            routing: routing
        ))
        #expect(AssistantKernel.shouldContinueNativeToolChainAfterNonSuccess(
            after: "Location permission denied.",
            currentToolID: "location.current",
            nextToolID: "maps.search",
            routing: routing
        ))
        #expect(!AssistantKernel.shouldContinueNativeToolChainAfterNonSuccess(
            after: "Position snapshot is disabled in this build.",
            currentToolID: "location.current",
            nextToolID: nil,
            routing: routing
        ))
    }

    @Test func agentServiceRepairsMemoryRecallBeforeSaveInvariant() {
        #if DEBUG
        let prompt = "Remember that I prefer concise bullet points, then tell me what you remembered."
        let recallFirst = AgentAction(tool: "memory.recall", args: ["query": .string("user preference")])
        let repaired = AgentService.repairedMemoryActionForTests(
            modelAction: recallFirst,
            prompt: prompt,
            steps: []
        )
        #expect(repaired.action.tool == "memory.save")
        #expect(repaired.action.args["content"]?.stringValue == "I prefer concise bullet points")
        #expect(repaired.action.args["kind"]?.stringValue == "fact")
        #expect(repaired.reflection?.content.contains("memory.recall into memory.save") == true)

        let contaminatedSave = AgentAction(tool: "memory.save", args: [
            "content": .string("I prefer concise bullet points, then tell me what you remembered."),
            "kind": .string("note")
        ])
        let normalized = AgentService.repairedMemoryActionForTests(
            modelAction: contaminatedSave,
            prompt: prompt,
            steps: []
        )
        #expect(normalized.action.tool == "memory.save")
        #expect(normalized.action.args["content"]?.stringValue == "I prefer concise bullet points")
        #expect(normalized.action.args["kind"]?.stringValue == "fact")
        #expect(normalized.reflection != nil)

        let savedStep = AgentStep(kind: .action, content: "memory.save", toolID: "memory.save", toolArgs: [
            "content": "I prefer concise bullet points",
            "kind": "fact"
        ])
        let next = AgentService.nextRequiredMemoryActionForTests(prompt: prompt, steps: [savedStep])
        #expect(next?.tool == "memory.recall")
        #expect(next?.args["query"]?.stringValue == "prefer concise bullet points")

        let noSaveRepair = AgentService.repairedMemoryActionForTests(
            modelAction: recallFirst,
            prompt: prompt,
            steps: [],
            availableToolIDs: ["memory.recall"]
        )
        #expect(noSaveRepair.action.tool == "memory.recall")
        #expect(noSaveRepair.reflection == nil)

        let noRecallNext = AgentService.nextRequiredMemoryActionForTests(
            prompt: prompt,
            steps: [savedStep],
            availableToolIDs: ["memory.save"]
        )
        #expect(noRecallNext == nil)

        let req = AgentRequest(
            systemPrompt: "sys",
            history: [],
            userMessage: prompt,
            temperature: 0,
            topP: 1,
            repetitionPenalty: 1,
            maxTokens: 128,
            maxSteps: 2,
            availableTools: ToolRegistry.all.filter { ["memory.save", "memory.recall"].contains(ToolRouteGuard.canonicalToolID($0.id)) },
            relevantMemories: []
        )
        let final = AgentService.postprocessStructuredFinalAnswerForTests(
            "Memory tool output could not be validated.",
            req: req,
            observations: [
                ("memory.save", "Saved."),
                ("memory.recall", "I prefer concise bullet points")
            ],
            steps: [
                savedStep,
                AgentStep(kind: .action, content: "memory.recall", toolID: "memory.recall", toolArgs: ["query": "prefer concise bullet points"])
            ]
        )
        #expect(final == "I remember that you prefer concise bullet points.")
        #expect(!final.contains("I'm ready"))
        #expect(!final.contains("Please ask again"))
        #expect(!final.contains("Memory tool output could not be validated"))
        #else
        #expect(true)
        #endif
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
        #expect(deepSynthesis.accepted == true)
        #expect(deepSynthesis.text?.contains("Summary:") == true)
        #expect(deepSynthesis.rejectionReason == nil)
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
        #expect(package.exportKind == InAppDatasetPackageExporter.exportKind)
        #expect(package.testFlight.sourceAction == InAppDatasetPackageExporter.sourceAction)
        #expect(package.testFlight.filePrefix == InAppDatasetPackageExporter.filePrefix)
        #expect(package.exportPolicy.sourceLayer == "agentGroundingRuntimeAudit")
        #expect(package.exportPolicy.format == "testflight-agent-grounding-runtime-json-package")
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
            rawOutputPrefix: #"{"action":{"tool":"rag.search","args":{"query":"evidence"}}}"#,
            selectedToolID: "rag.search",
            toolArguments: [:],
            allowedToolIDs: ["rag.search"],
            requiresApproval: nil,
            approvalMode: nil,
            parseError: nil,
            emittedFinalInActionTurn: false,
            outputTokenCount: 8,
            runtimePath: "agent-model",
            streamStarted: true,
            modelLoaded: true,
            firstChunkReceived: true,
            textChunkCount: 1,
            finalChunkReceived: true,
            streamTerminationReason: "stop"
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
        #expect(package.testFlight.liveE2EReportIncluded == true)
        #expect(liveE2EReport.correlatedTraceCount == 2)
        #expect(liveE2EReport.modelBackedCorrelatedTraceCount == 1)
        #expect(liveE2EReport.modelBackedCorrelatedScenarioCount == 1)
        #expect(liveE2EReport.deterministicCompatibilityTraceCount == 1)
    }

    @Test func liveE2EExportRequiresDistinctModelBackedScenarioCoverage() throws {
        AgentBehaviorTraceRecorder.clear()
        defer { AgentBehaviorTraceRecorder.clear() }
        let startedAt = Date(timeIntervalSince1970: 1_800_000_050)
        let coveredE2ERunID = UUID(uuidString: "11111111-1111-4111-8111-111111111113")!
        let coveredAgentRunID = UUID(uuidString: "22222222-2222-4222-8222-222222222224")!
        let coveredConversationID = UUID(uuidString: "33333333-3333-4333-8333-333333333335")!
        let coveredTurnID = UUID(uuidString: "44444444-4444-4444-8444-444444444446")!
        let missingE2ERunID = UUID(uuidString: "11111111-1111-4111-8111-111111111114")!
        let missingAgentRunID = UUID(uuidString: "22222222-2222-4222-8222-222222222225")!
        let missingConversationID = UUID(uuidString: "33333333-3333-4333-8333-333333333336")!
        let missingTurnID = UUID(uuidString: "44444444-4444-4444-8444-444444444447")!

        for offset in 1...2 {
            AgentBehaviorTraceRecorder.record(AgentBehaviorTrace(
                id: UUID(),
                createdAt: startedAt.addingTimeInterval(TimeInterval(offset)),
                event: .modelTurn,
                slot: "executor",
                stage: "agent-json-step-\(offset)",
                scenarioID: "covered-live-scenario",
                e2eRunID: coveredE2ERunID,
                agentRunID: coveredAgentRunID,
                conversationID: coveredConversationID,
                turnID: coveredTurnID,
                intent: "rag",
                promptPrefix: "What evidence supports your claim?",
                rawOutputPrefix: #"{"action":{"tool":"rag.search","args":{"query":"evidence"}}}"#,
                selectedToolID: "rag.search",
                toolArguments: [:],
                allowedToolIDs: ["rag.search"],
                requiresApproval: nil,
                approvalMode: nil,
                parseError: nil,
                emittedFinalInActionTurn: false,
                outputTokenCount: 8,
                runtimePath: "agent-model",
                streamStarted: true,
                modelLoaded: true,
                firstChunkReceived: true,
                textChunkCount: 1,
                finalChunkReceived: true,
                streamTerminationReason: "stop"
            ))
        }

        func result(
            scenarioID: String,
            title: String,
            prompt: String,
            e2eRunID: UUID,
            agentRunID: UUID,
            conversationID: UUID,
            turnID: UUID
        ) -> E2ETestResult {
            E2ETestResult(
                id: UUID(),
                scenarioID: scenarioID,
                kind: "standard",
                title: title,
                prompt: prompt,
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
        }

        let report = E2ETestReport(
            id: UUID(),
            startedAt: startedAt,
            finishedAt: startedAt.addingTimeInterval(3),
            passed: 2,
            failed: 0,
            results: [
                result(
                    scenarioID: "covered-live-scenario",
                    title: "Covered live scenario",
                    prompt: "What evidence supports your claim?",
                    e2eRunID: coveredE2ERunID,
                    agentRunID: coveredAgentRunID,
                    conversationID: coveredConversationID,
                    turnID: coveredTurnID
                ),
                result(
                    scenarioID: "missing-live-scenario",
                    title: "Missing live scenario",
                    prompt: "What source proves the runtime state?",
                    e2eRunID: missingE2ERunID,
                    agentRunID: missingAgentRunID,
                    conversationID: missingConversationID,
                    turnID: missingTurnID
                )
            ]
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
        #expect(liveE2EReport.modelBackedCorrelatedTraceCount == 2)
        #expect(liveE2EReport.modelBackedCorrelatedScenarioCount == 1)
        #expect(failure.actual?.contains("evidenceRequiredScenarioCount=2") == true)
        #expect(failure.actual?.contains("missingEvidenceScenarioCount=1") == true)
        #expect(failure.actual?.contains("modelBackedCorrelatedTraceCount=2") == true)
        #expect(failure.actual?.contains("modelBackedCorrelatedScenarioCount=1") == true)
    }

    @Test func liveE2EExportRejectsTraceThatAmbiguouslyMatchesMultipleResults() throws {
        AgentBehaviorTraceRecorder.clear()
        defer { AgentBehaviorTraceRecorder.clear() }
        let startedAt = Date(timeIntervalSince1970: 1_800_000_060)
        let scenarioID = "ambiguous-live-scenario"
        let e2eRunID = UUID()
        let agentRunID = UUID()
        let conversationID = UUID()
        let turnID = UUID()

        AgentBehaviorTraceRecorder.record(AgentBehaviorTrace(
            id: UUID(),
            createdAt: startedAt,
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
            rawOutputPrefix: #"{"action":{"tool":"rag.search","args":{"query":"evidence"}}}"#,
            selectedToolID: "rag.search",
            toolArguments: [:],
            allowedToolIDs: ["rag.search"],
            requiresApproval: nil,
            approvalMode: nil,
            parseError: nil,
            emittedFinalInActionTurn: false,
            outputTokenCount: 8,
            runtimePath: "agent-model",
            streamStarted: true,
            modelLoaded: true,
            firstChunkReceived: true,
            textChunkCount: 1,
            finalChunkReceived: true,
            streamTerminationReason: "stop"
        ))

        func result(
            title: String,
            agentRunID: UUID? = nil,
            conversationID: UUID? = nil,
            turnID: UUID? = nil
        ) -> E2ETestResult {
            E2ETestResult(
                id: UUID(),
                scenarioID: scenarioID,
                kind: "standard",
                title: title,
                prompt: "What evidence supports your claim?",
                expectedIntent: "rag",
                actualIntent: "rag",
                e2eRunID: e2eRunID,
                agentRunID: agentRunID,
                conversationID: conversationID,
                turnID: turnID,
                requiresAgentRun: true,
                evidenceMode: E2EEvidenceMode.modelBackedRequired.rawValue,
                passed: true,
                failures: [],
                finalText: "Live E2E evidence.",
                missingHints: [],
                rewriteAttempted: false,
                rewriteSuccess: false,
                events: [],
                startedAt: startedAt,
                finishedAt: startedAt.addingTimeInterval(1),
                rawFinalPrefix: "Live E2E evidence.",
                sanitizedFinalPrefix: "Live E2E evidence.",
                rawFinalHadUnsafeLeakage: false,
                sanitizedFinalRemovedArtifacts: [],
                outputHygieneFailures: []
            )
        }

        let report = E2ETestReport(
            id: UUID(),
            startedAt: startedAt,
            finishedAt: startedAt.addingTimeInterval(1),
            passed: 2,
            failed: 0,
            results: [
                result(title: "Partial correlation"),
                result(
                    title: "Full correlation",
                    agentRunID: agentRunID,
                    conversationID: conversationID,
                    turnID: turnID
                )
            ]
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
        let failure = try #require(package.exportQualityFailures?.first(where: {
            $0.type == "agent_grounding_live_e2e_model_backed_trace_gap"
        }))
        #expect(liveE2EReport.correlatedTraceCount == 0)
        #expect(liveE2EReport.modelBackedCorrelatedTraceCount == 0)
        #expect(liveE2EReport.modelBackedCorrelatedScenarioCount == 0)
        #expect(liveE2EReport.payload.results.allSatisfy { $0.correlationToken == nil })
        #expect(package.recentTraces.first?.correlationToken == nil)
        #expect(failure.actual?.contains("missingEvidenceScenarioCount=2") == true)
    }

    @Test func policyFirstLiveE2EExportAcceptsCorrelatedDeterministicEvidence() throws {
        AgentBehaviorTraceRecorder.clear()
        defer { AgentBehaviorTraceRecorder.clear() }
        let startedAt = Date(timeIntervalSince1970: 1_800_000_075)
        let scenarioID = "policy-first-tool-coverage"
        let e2eRunID = UUID()
        let agentRunID = UUID()
        let conversationID = UUID()
        let turnID = UUID()
        let rawCorrelationDiagnostic = AgentTraceCorrelation(
            scenarioID: scenarioID,
            e2eRunID: e2eRunID,
            agentRunID: agentRunID,
            conversationID: conversationID,
            turnID: turnID
        ).diagnosticText
        AgentBehaviorTraceRecorder.record(AgentBehaviorTrace(
            id: UUID(),
            createdAt: startedAt,
            event: .toolAction,
            slot: "policy",
            stage: "deterministic-compatibility-tool",
            scenarioID: scenarioID,
            e2eRunID: e2eRunID,
            agentRunID: agentRunID,
            conversationID: conversationID,
            turnID: turnID,
            intent: "weather",
            promptPrefix: "weather",
            rawOutputPrefix: "",
            selectedToolID: "weather",
            toolArguments: [:],
            allowedToolIDs: ["weather"],
            requiresApproval: false,
            approvalMode: nil,
            parseError: nil,
            emittedFinalInActionTurn: false,
            runtimePath: "deterministic-compatibility"
        ))
        AgentBehaviorTraceRecorder.record(AgentBehaviorTrace(
            id: UUID(),
            createdAt: startedAt.addingTimeInterval(0.1),
            event: .toolAction,
            slot: "policy",
            stage: "deterministic-compatibility-tool-partial-correlation",
            scenarioID: scenarioID,
            e2eRunID: e2eRunID,
            agentRunID: agentRunID,
            conversationID: conversationID,
            turnID: nil,
            intent: "weather",
            promptPrefix: "weather",
            rawOutputPrefix: "",
            selectedToolID: "weather",
            toolArguments: [:],
            allowedToolIDs: ["weather"],
            requiresApproval: false,
            approvalMode: nil,
            parseError: nil,
            emittedFinalInActionTurn: false,
            runtimePath: "deterministic-compatibility"
        ))
        let result = E2ETestResult(
            id: UUID(),
            scenarioID: scenarioID,
            kind: "toolCoverage",
            title: "Policy-first weather",
            prompt: "What's the weather?",
            expectedIntent: "weather",
            actualIntent: "weather",
            e2eRunID: e2eRunID,
            agentRunID: agentRunID,
            conversationID: conversationID,
            turnID: turnID,
            requiresAgentRun: true,
            evidenceMode: E2EEvidenceMode.policyFirstAllowed.rawValue,
            passed: true,
            failures: [],
            finalText: "Weather observation.",
            missingHints: [],
            rewriteAttempted: false,
            rewriteSuccess: false,
            events: [E2ETestEvent(
                id: UUID(),
                createdAt: startedAt,
                scenarioID: scenarioID,
                phase: "correlation",
                message: rawCorrelationDiagnostic
            )],
            startedAt: startedAt,
            finishedAt: startedAt.addingTimeInterval(1),
            rawFinalPrefix: "Weather observation.",
            sanitizedFinalPrefix: "Weather observation.",
            rawFinalHadUnsafeLeakage: false,
            sanitizedFinalRemovedArtifacts: [],
            outputHygieneFailures: [],
            metadata: ["traceCorrelation": rawCorrelationDiagnostic]
        )
        let report = E2ETestReport(
            id: UUID(),
            startedAt: startedAt,
            finishedAt: startedAt.addingTimeInterval(1),
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

        #expect(package.liveE2EReport?.modelBackedCorrelatedScenarioCount == 0)
        #expect(package.liveE2EReport?.deterministicCompatibilityTraceCount == 1)
        #expect(package.liveE2EReport?.correlatedTraceCount == 1)
        #expect(package.exportQualityFailures?.contains(where: { $0.type == "agent_grounding_live_e2e_model_backed_trace_gap" }) != true)
        let exportedResult = try #require(package.liveE2EReport?.payload.results.first)
        let correlatedTrace = try #require(package.recentTraces.first { $0.stage == "deterministic-compatibility-tool" })
        let partialTrace = try #require(package.recentTraces.first { $0.stage == "deterministic-compatibility-tool-partial-correlation" })
        #expect(exportedResult.e2eRunID == nil)
        #expect(exportedResult.agentRunID == nil)
        #expect(exportedResult.conversationID == nil)
        #expect(exportedResult.turnID == nil)
        #expect(exportedResult.correlationToken?.hasPrefix("corr_v1_") == true)
        #expect(exportedResult.events.first?.message.contains("[redacted-correlation]") == true)
        #expect(exportedResult.metadata["traceCorrelation"]?.contains("[redacted-correlation]") == true)
        #expect(correlatedTrace.correlationToken == exportedResult.correlationToken)
        #expect(partialTrace.correlationToken == nil)

        let encoded = try JSONEncoder().encode(package)
        let json = try #require(String(data: encoded, encoding: .utf8))
        #expect(!json.contains(e2eRunID.uuidString))
        #expect(!json.contains(agentRunID.uuidString))
        #expect(!json.contains(conversationID.uuidString))
        #expect(!json.contains(turnID.uuidString))
    }

    @Test func liveE2EExportDoesNotCreditScenarioOnlyStaleTrace() throws {
        AgentBehaviorTraceRecorder.clear()
        defer { AgentBehaviorTraceRecorder.clear() }
        let startedAt = Date(timeIntervalSince1970: 1_800_000_090)
        let scenarioID = "policy-first-stale-scenario"
        AgentBehaviorTraceRecorder.record(AgentBehaviorTrace(
            id: UUID(),
            createdAt: startedAt.addingTimeInterval(-60),
            event: .toolAction,
            slot: "policy",
            stage: "deterministic-compatibility-stale-tool",
            scenarioID: scenarioID,
            intent: "weather",
            promptPrefix: "weather",
            rawOutputPrefix: "weather(validated)",
            selectedToolID: "weather",
            toolArguments: [:],
            allowedToolIDs: ["weather"],
            requiresApproval: false,
            approvalMode: nil,
            parseError: nil,
            emittedFinalInActionTurn: false,
            runtimePath: "deterministic-compatibility"
        ))
        let result = E2ETestResult(
            id: UUID(),
            scenarioID: scenarioID,
            kind: "toolCoverage",
            title: "Unidentified current run",
            prompt: "What's the weather?",
            expectedIntent: "weather",
            actualIntent: "weather",
            requiresAgentRun: true,
            evidenceMode: E2EEvidenceMode.policyFirstAllowed.rawValue,
            passed: false,
            failures: ["Current run did not record correlated evidence."],
            finalText: "Weather unavailable.",
            missingHints: [],
            rewriteAttempted: false,
            rewriteSuccess: false,
            events: [],
            startedAt: startedAt,
            finishedAt: startedAt.addingTimeInterval(1),
            rawFinalPrefix: "Weather unavailable.",
            sanitizedFinalPrefix: "Weather unavailable.",
            rawFinalHadUnsafeLeakage: false,
            sanitizedFinalRemovedArtifacts: [],
            outputHygieneFailures: []
        )
        let report = E2ETestReport(
            id: UUID(),
            startedAt: startedAt,
            finishedAt: startedAt.addingTimeInterval(1),
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

        #expect(package.liveE2EReport?.correlatedTraceCount == 0)
        #expect(package.liveE2EReport?.deterministicCompatibilityTraceCount == 0)
        #expect(package.liveE2EReport?.payload.results.first?.correlationToken == nil)
        #expect(package.recentTraces.first?.correlationToken == nil)
        #expect(package.exportQualityFailures?.contains(where: {
            $0.type == "agent_grounding_live_e2e_model_backed_trace_gap"
        }) == true)
    }

    @Test func liveE2EExportRejectsStructuredFinalWithoutExplicitRuntimeAndFinalizerProof() throws {
        AgentBehaviorTraceRecorder.clear()
        defer { AgentBehaviorTraceRecorder.clear() }
        let startedAt = Date(timeIntervalSince1970: 1_800_000_095)
        let scenarioID = "model-final-missing-proof"
        let e2eRunID = UUID()
        let agentRunID = UUID()
        let conversationID = UUID()
        let turnID = UUID()
        AgentBehaviorTraceRecorder.record(AgentBehaviorTrace(
            id: UUID(),
            createdAt: startedAt,
            event: .modelTurn,
            slot: "executor",
            stage: "agent-json-step-1",
            scenarioID: scenarioID,
            e2eRunID: e2eRunID,
            agentRunID: agentRunID,
            conversationID: conversationID,
            turnID: turnID,
            intent: "weather",
            promptPrefix: "weather",
            rawOutputPrefix: #"{"final":"It is clear."}"#,
            selectedToolID: nil,
            toolArguments: [:],
            allowedToolIDs: ["weather"],
            requiresApproval: false,
            approvalMode: nil,
            parseError: nil,
            emittedFinalInActionTurn: true,
            runtimePath: "agent-model"
        ))
        let result = E2ETestResult(
            id: UUID(),
            scenarioID: scenarioID,
            kind: "toolCoverage",
            title: "Weather final missing proof",
            prompt: "What's the weather?",
            expectedIntent: "weather",
            actualIntent: "weather",
            e2eRunID: e2eRunID,
            agentRunID: agentRunID,
            conversationID: conversationID,
            turnID: turnID,
            requiresAgentRun: true,
            evidenceMode: E2EEvidenceMode.modelBackedRequired.rawValue,
            passed: true,
            failures: [],
            finalText: "It is clear.",
            missingHints: [],
            rewriteAttempted: false,
            rewriteSuccess: false,
            events: [],
            startedAt: startedAt,
            finishedAt: startedAt.addingTimeInterval(1),
            rawFinalPrefix: "It is clear.",
            sanitizedFinalPrefix: "It is clear.",
            rawFinalHadUnsafeLeakage: false,
            sanitizedFinalRemovedArtifacts: [],
            outputHygieneFailures: []
        )
        let report = E2ETestReport(
            id: UUID(),
            startedAt: startedAt,
            finishedAt: startedAt.addingTimeInterval(1),
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

        #expect(package.liveE2EReport?.correlatedTraceCount == 1)
        #expect(package.liveE2EReport?.modelBackedCorrelatedTraceCount == 0)
        #expect(package.liveE2EReport?.modelBackedCorrelatedScenarioCount == 0)
        #expect(package.liveE2EReport?.payload.results.first?.correlationToken?.hasPrefix("corr_v1_") == true)
        #expect(package.exportQualityFailures?.contains(where: {
            $0.type == "agent_grounding_live_e2e_model_backed_trace_gap"
        }) == true)
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
        #expect(liveE2EReport.modelBackedCorrelatedScenarioCount == 0)
        #expect(liveE2EReport.deterministicCompatibilityTraceCount == 1)
        #expect(failure.actual?.contains("evidenceRequiredScenarioCount=1") == true)
        #expect(failure.actual?.contains("missingEvidenceScenarioCount=1") == true)
        #expect(failure.actual?.contains("modelBackedCorrelatedTraceCount=0") == true)
        #expect(failure.actual?.contains("modelBackedCorrelatedScenarioCount=0") == true)
        #expect(failure.problem.contains("evidenceMode") == true)
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

    @Test func agentGroundingPackageUsesParserDerivedToolActionParseResult() throws {
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

        #expect(package.traceParseErrorCount == 0)
        #expect(package.behaviorAudit?.violations.contains(where: { $0.code == "structured_action_trace_parse_error" }) != true)
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

    @Test func deterministicWebSummaryFallbackSynthesizesSwiftSearchObservations() throws {
        let observation = """
        Search results for: Swift concurrency best practices
        {"title":"Swift.org - Concurrency","url":"https://swift.org/documentation/concurrency/","snippet":"Swift concurrency uses async/await and structured concurrency so tasks can be cancelled and scoped cleanly."}
        {"title":"Apple Developer - MainActor","url":"https://developer.apple.com/documentation/swift/mainactor","snippet":"Use MainActor isolation for UI state updates and keep work that can suspend out of synchronous UI paths."}
        """

        let summary = try #require(AgentService.deterministicWebSummaryFallbackForTests(observations: [("web.search", observation)]))

        #expect(summary.contains("Swift") || summary.contains("swift"))
        #expect(summary.split(separator: "\n").filter { $0.trimmingCharacters(in: .whitespaces).hasPrefix("-") }.count >= 2)
        #expect(!summary.lowercased().contains("no direct answer from web search"))
        #expect(!summary.lowercased().hasPrefix("search results for:"))
    }

    @Test func deterministicWebSummaryFallbackSynthesizesPlainNumberedSearchResults() throws {
        let observation = """
        Search results for: Swift concurrency best practices

        1. The Essential Guide to Concurrency in Swift: Avoiding Common ... - Medium
        https://medium.com/@rashadsh/the-essential-guide-to-concurrency-in-swift-avoiding-common-pitfalls-9ac58ada1367

        2. Swift Concurrency Best Practices - beefed.ai
        https://beefed.ai/en/swift-concurrency-best-practices

        3. Concurrency | Apple Developer Documentation
        https://developer.apple.com/documentation/swift/concurrency
        """

        let summary = try #require(AgentService.deterministicWebSummaryFallbackForTests(observations: [("web.search", observation)]))

        #expect(summary.contains("Swift"))
        #expect(summary.contains("Concurrency") || summary.contains("concurrency"))
        #expect(summary.split(separator: "\n").filter { $0.trimmingCharacters(in: .whitespaces).hasPrefix("-") }.count >= 2)
        #expect(!summary.lowercased().contains("no direct answer from web search"))
        #expect(!summary.lowercased().hasPrefix("search results for:"))
    }

    @Test func deterministicWebSummaryFallbackSynthesizesLivePayload() throws {
        let observation = """
        Search results for: Swift concurrency best practices
        <lumen_web_payload>{"results":[{"title":"Swift.org - Concurrency","url":"https://swift.org/documentation/concurrency/"},{"title":"Apple Developer - MainActor and Swift concurrency","url":"https://developer.apple.com/documentation/swift/mainactor"}]}</lumen_web_payload>
        """

        let summary = try #require(AgentService.deterministicWebSummaryFallbackForTests(observations: [("web.search", observation)]))

        #expect(summary.contains("Swift"))
        #expect(summary.contains("MainActor") || summary.contains("Concurrency"))
        #expect(summary.split(separator: "\n").filter { $0.trimmingCharacters(in: .whitespaces).hasPrefix("-") }.count >= 2)
        #expect(!summary.lowercased().contains("no direct answer from web search"))
        #expect(!summary.lowercased().contains("<lumen_web_payload"))
        #expect(summary.components(separatedBy: "Swift.org - Concurrency").count == 2)
    }

    @Test func missingActionToolRepairsToOnlyAllowedToolWhenSafe() throws {
        let raw = #"{"thought":"search","action":{"args":{"query":"Swift concurrency best practices"}}}"#
        let req = AgentRequest(
            systemPrompt: "sys",
            history: [],
            userMessage: "Search the web for Swift concurrency best practices.",
            temperature: 0,
            topP: 1,
            repetitionPenalty: 1,
            maxTokens: 128,
            maxSteps: 2,
            availableTools: ToolRegistry.all.filter { ToolRouteGuard.canonicalToolID($0.id) == "web.search" },
            relevantMemories: []
        )

        let repaired = try #require(AgentService.repairMissingToolActionForTests(raw: raw, req: req))

        #expect(repaired.action.tool == "web.search")
        #expect(repaired.action.args["query"]?.stringValue == "Swift concurrency best practices")
    }

    @Test func noisyActionTurnIsNotExecutableEvenWhenParserFindsAction() throws {
        let raw = "Here is the JSON:\n{\"action\":{\"tool\":\"web.search\",\"args\":{\"query\":\"Swift concurrency\"}}}"
        let parsed = AgentTurnParser.parse(raw)
        let executable = AgentService.strictToolExecutableTurnForTests(parsed)

        #expect(parsed.parseError == nil)
        #expect(parsed.action?.tool == "web.search")
        #expect(parsed.hadNoise)
        #expect(executable.parseError == .noisyOutput)
        #expect(executable.action == nil)
    }

    @Test func noisyMissingActionToolDoesNotRepairToAllowedTool() throws {
        let raw = #"Here is the JSON: {"thought":"search","action":{"args":{"query":"Swift concurrency best practices"}}}"#
        let req = AgentRequest(
            systemPrompt: "sys",
            history: [],
            userMessage: "Search the web for Swift concurrency best practices.",
            temperature: 0,
            topP: 1,
            repetitionPenalty: 1,
            maxTokens: 128,
            maxSteps: 2,
            availableTools: ToolRegistry.all.filter { ToolRouteGuard.canonicalToolID($0.id) == "web.search" },
            relevantMemories: []
        )

        #expect(AgentService.repairMissingToolActionForTests(raw: raw, req: req) == nil)
    }

    @Test func missingActionToolDoesNotRepairWhenMultipleToolsAreAllowed() throws {
        let raw = #"{"thought":"search","action":{"args":{"query":"Swift concurrency best practices"}}}"#
        let req = AgentRequest(
            systemPrompt: "sys",
            history: [],
            userMessage: "Search the web for Swift concurrency best practices.",
            temperature: 0,
            topP: 1,
            repetitionPenalty: 1,
            maxTokens: 128,
            maxSteps: 2,
            availableTools: ToolRegistry.all.filter {
                let id = ToolRouteGuard.canonicalToolID($0.id)
                return id == "web.search" || id == "web.fetch"
            },
            relevantMemories: []
        )

        #expect(AgentService.repairMissingToolActionForTests(raw: raw, req: req) == nil)
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

    @Test func firstToolRequiredAgentJSONTurnUsesActionOnlySchema() {
        let req = AgentRequest(
            systemPrompt: "sys",
            history: [],
            userMessage: "What is the weather here and should I carry an umbrella?",
            temperature: 0,
            topP: 1,
            repetitionPenalty: 1,
            maxTokens: 384,
            maxSteps: 3,
            availableTools: ToolRegistry.all,
            relevantMemories: []
        )

        let schema = AgentService.structuredAgentResponseFormatSchemaForTests(req: req)

        #expect(schema == AgentService.structuredAgentActionResponseSchema)
        #expect(schema.contains(#""required":["action"]"#))
        #expect(!schema.contains(#""oneOf""#))
        #expect(!schema.contains(#""final""#))
    }

    @Test func directChatAgentJSONTurnWithoutToolsUsesFinalOnlySchema() {
        let req = AgentRequest(
            systemPrompt: "sys",
            history: [],
            userMessage: "Explain precision and recall in plain English.",
            temperature: 0,
            topP: 1,
            repetitionPenalty: 1,
            maxTokens: 384,
            maxSteps: 3,
            availableTools: [],
            relevantMemories: []
        )

        let schema = AgentService.structuredAgentResponseFormatSchemaForTests(req: req)

        #expect(schema == AgentService.structuredAgentFinalResponseSchema)
        #expect(schema.contains(#""required":["final"]"#))
        #expect(!schema.contains(#""oneOf""#))
        #expect(!schema.contains(#""action""#))
    }

    @Test func agentJSONIncompleteOutputRetryPromptRequiresFreshValidJSONOnly() {
        let firstTurn = "User request:\nSet a reminder for 6 PM.\n\nEmit the first JSON object now. Choose either action or final."
        let raw = "<think>\n</think>\n{\""

        let retryTurn = AgentService.agentJSONIncompleteOutputRetryUserTurnForTests(
            from: firstTurn,
            rawOutput: raw
        )

        #expect(retryTurn.contains("User request:"))
        #expect(retryTurn.contains("Set a reminder for 6 PM."))
        #expect(retryTurn.contains("Previous live agent-json attempt stopped after an incomplete JSON object"))
        #expect(retryTurn.contains("Ignore the incomplete object"))
        #expect(retryTurn.contains("Do not include schema, required, status, approvalPrompt"))
        #expect(retryTurn.contains(#"{"action":{"tool":"<allowed tool id>","args":{}}}"#))
        #expect(retryTurn.contains(#"{"final":"<concise user-facing answer>"}"#))
        #expect(retryTurn.contains("Output JSON only"))
    }

    @Test func agentJSONIncompleteOutputRetryRequestKeepsSchemaVisibleAndUsesFreshID() async {
        let req = AgentRequest(
            systemPrompt: "sys",
            history: [],
            userMessage: "Set a reminder for 6 PM.",
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

        let retry = AgentService.agentJSONIncompleteOutputRetryRequestForTests(
            from: base,
            userTurn: base.userMessage,
            rawOutput: "<think>\n</think>\n{\""
        )
        let result = await AppLlamaService.shared.buildMessagesForTesting(req: retry, contextSize: 2048, slot: .executor)
        let prompt = result.messages.map(\.content).joined(separator: "\n")

        #expect(retry.id != base.id)
        #expect(retry.responseFormat == base.responseFormat)
        #expect(retry.temperature <= 0.02)
        #expect(retry.topP <= 0.4)
        #expect(prompt.contains("Previous live agent-json attempt stopped after an incomplete JSON object"))
        #expect(prompt.contains("Response format contract: output exactly one valid JSON object"))
        #expect(prompt.contains(#""oneOf""#))
        #expect(prompt.contains(#""action""#))
        #expect(prompt.contains(#""final""#))
        #expect(prompt.contains("/no_think"))
    }

    @Test func agentJSONMissingDecisionRetryPromptForcesActionOnly() {
        let firstTurn = "User request:\nWhat is the weather here?\n\nEmit the first JSON object now. Choose either action or final."
        let raw = "{\n}\n"

        let retryTurn = AgentService.agentJSONMissingDecisionRetryUserTurnForTests(
            from: firstTurn,
            rawOutput: raw
        )

        #expect(retryTurn.contains("User request:"))
        #expect(retryTurn.contains("What is the weather here?"))
        #expect(retryTurn.contains("no action or final"))
        #expect(retryTurn.contains("requires a tool action before any final answer"))
        #expect(retryTurn.contains("Allowed tool IDs: none"))
        #expect(retryTurn.contains(#"{"action":{"tool":"<allowed tool id>","args":{}}}"#))
        #expect(retryTurn.contains("Do not emit {}, final, prose"))
        #expect(retryTurn.contains("Output JSON only"))
    }

    @Test func agentJSONMissingDecisionRetryRequestUsesActionOnlySchemaAndFreshID() async {
        let req = AgentRequest(
            systemPrompt: "sys",
            history: [],
            userMessage: "What is the weather here and should I carry an umbrella?",
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
            responseFormat: .constrainedJSON(schema: AgentService.structuredAgentActionResponseSchema)
        )

        let retry = AgentService.agentJSONMissingDecisionRetryRequestForTests(
            from: base,
            userTurn: base.userMessage,
            rawOutput: "{\n}\n",
            allowedToolIDs: ["weather", "location.current"]
        )
        let result = await AppLlamaService.shared.buildMessagesForTesting(req: retry, contextSize: 2048, slot: .executor)
        let prompt = result.messages.map(\.content).joined(separator: "\n")

        #expect(retry.id != base.id)
        #expect(retry.responseFormat == .constrainedJSON(schema: AgentService.structuredAgentActionResponseSchema))
        #expect(retry.temperature <= 0.02)
        #expect(retry.topP <= 0.35)
        #expect(prompt.contains("Previous live agent-json attempt emitted a JSON object with no action or final"))
        #expect(prompt.contains("This turn requires a tool action"))
        #expect(prompt.contains("Allowed tool IDs: location.current, weather"))
        #expect(prompt.contains(#""required":["action"]"#))
        #expect(!prompt.contains(#""oneOf""#))
        #expect(prompt.contains("/no_think"))
    }

    @Test func firstWebSearchObservationStopsE2ETrainingLoopBeforeSecondSearch() {
        let tools = ToolRegistry.all.filter { ["web.search", "web.fetch"].contains(ToolRouteGuard.canonicalToolID($0.id)) }
        let req = AgentRequest(
            systemPrompt: "sys",
            history: [],
            userMessage: "Search the web for two recent Swift concurrency best practices and summarize them.",
            temperature: 0,
            topP: 1,
            repetitionPenalty: 1,
            maxTokens: 384,
            maxSteps: 3,
            availableTools: tools,
            relevantMemories: [],
            scenarioID: "training-web-research",
            e2eRunID: UUID()
        )
        let observations = [
            ("web.search", """
            Search results for: Swift concurrency best practices

            1. Concurrency | Apple Developer Documentation
            https://developer.apple.com/documentation/swift/concurrency

            2. Swift MainActor guidance
            https://example.com/mainactor
            """)
        ]

        #expect(AgentService.shouldStopAfterFirstWebObservationForTests(req: req, actionTool: "web.search", observations: observations))
    }

    @Test func placeholderWeatherFinalRequiresToolActionBeforeObservation() {
        let req = AgentRequest(
            systemPrompt: "sys",
            history: [],
            userMessage: "What is the weather here?",
            temperature: 0,
            topP: 1,
            repetitionPenalty: 1,
            maxTokens: 384,
            maxSteps: 3,
            availableTools: ToolRegistry.all.filter { ["weather", "location.current"].contains(ToolRouteGuard.canonicalToolID($0.id)) },
            relevantMemories: []
        )

        #expect(AgentService.toolRequiredFinalNeedsActionForTests("[insert local weather information]", req: req))
        #expect(!AgentService.toolRequiredFinalNeedsActionForTests("Weather update: 18 C and cloudy.", req: req, observations: [("weather", "18 C and cloudy")]))
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
        let hasCompatibilityFinalTrace = traces.contains { trace in
            trace.event == AgentBehaviorTrace.Event.finalAnswer && trace.runtimePath == "deterministic-compatibility"
        }
        let finalTrace = traces.first { trace in
            trace.event == AgentBehaviorTrace.Event.finalAnswer && trace.runtimePath == "deterministic-compatibility"
        }

        #expect(actionToolIDs == ["calendar.list"])
        #expect(response.text.lowercased().contains("event"))
        #expect(hasCalendarListActionTrace)
        #expect(actionTrace?.parseError == nil)
        #expect(actionTrace?.allowedToolIDs.contains("calendar.list") == true)
        #expect(actionTrace?.streamTerminationReason == "validated-tool-action")
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

    @Test func malformedTurnErrorObservationDoesNotShortCircuitParseRecovery() async {
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
        let errorOnlyObservations = [
            (tool: "made.up.tool", result: "Unknown tool: made.up.tool. Emit a final turn instead."),
            (tool: "web.search", result: "Tool web.search is disabled. Enable it in Tools.")
        ]

        #expect(!AgentService.hasUsableObservationForTests(intent: .webSearch, observations: errorOnlyObservations))

        let recovery = await AgentService.structuredParseFailureRecoveryForTests(req: req, options: options)
        let actionToolIDs = recovery?.steps
            .filter { $0.kind == .action }
            .compactMap(\.toolID)
            .map(ToolRouteGuard.canonicalToolID) ?? []

        #expect(actionToolIDs.contains("web.search"))
        #expect(recovery?.text.lowercased().contains("unknown tool") == false)
        #expect(recovery?.text.lowercased().contains("disabled") == false)
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

    @Test func agentServiceParseFailureRecoveryProducesAlarmApprovalBoundary() async {
        let prompt = "Set an alarm for tomorrow at 7."
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

        #expect(recovery?.steps.first?.kind == .approvalBoundary)
        #expect(recovery?.steps.first?.toolID == "alarm.schedule")
        #expect(recovery?.text.lowercased().contains("approval required for alarm.schedule") == true)
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

    @Test func structuredWebSummaryInvalidFinalFallsBackToSearchObservations() {
        let req = AgentRequest(
            systemPrompt: "sys",
            history: [],
            userMessage: "Search the web for two recent Swift concurrency best practices and summarize them.",
            temperature: 0,
            topP: 1,
            repetitionPenalty: 1,
            maxTokens: 256,
            maxSteps: 3,
            availableTools: ToolRegistry.all,
            relevantMemories: []
        )
        let final = AgentService.postprocessStructuredFinalAnswerForTests(
            #"{"intent":"webSearch","nextModel":"rag","reasoningSummary":"Intent webSearch is allowed to use rag.search.","requiresApproval":false,"sourceFile":"ios/Lumen/Models/ToolDefinition.swift"}"#,
            req: req,
            observations: [
                ("web.search", """
                Search results for: Swift concurrency best practices
                Prefer structured concurrency with TaskGroup or async let so cancellation and errors propagate through child tasks.
                Keep UI mutations isolated to MainActor and move long-running work off the main actor to avoid responsiveness and data-race issues.
                """)
            ],
            steps: [AgentStep(kind: .observation, content: "Search results", toolID: "web.search")]
        )

        #expect(final.contains("Summary:"))
        #expect(final.contains("structured concurrency"))
        #expect(final.contains("MainActor"))
        #expect(!final.contains("\"intent\""))
        #expect(!final.contains("sourceFile"))
    }

    @Test func structuredWebPartialRoutingJSONFallsBackToSearchObservations() {
        let req = AgentRequest(
            systemPrompt: "sys",
            history: [],
            userMessage: "Search the web for two recent Swift concurrency best practices and summarize them.",
            temperature: 0,
            topP: 1,
            repetitionPenalty: 1,
            maxTokens: 256,
            maxSteps: 3,
            availableTools: ToolRegistry.all,
            relevantMemories: []
        )
        let final = AgentService.postprocessStructuredFinalAnswerForTests(
            #"{"intent":"webSearch","nextModel":"rag","reasoningSummary":"Intent webSearch is allowed to use rag.search."}"#,
            req: req,
            observations: [
                ("web.search", """
                Search results for: Swift concurrency best practices
                Prefer structured concurrency so cancellation and errors propagate through child tasks.
                Keep UI state updates isolated to MainActor and move long-running work off the main actor.
                """)
            ],
            steps: [AgentStep(kind: .observation, content: "Search results", toolID: "web.search")]
        )

        #expect(final.contains("Summary:"))
        #expect(final.contains("structured concurrency"))
        #expect(final.contains("MainActor"))
        #expect(!final.contains("\"intent\""))
        #expect(!final.contains("nextModel"))
        #expect(!final.contains("reasoningSummary"))
    }

    @Test func structuredWebSummaryExtractsLumenPayloadTitles() {
        let req = AgentRequest(
            systemPrompt: "sys",
            history: [],
            userMessage: "Search web for diy underground shelter",
            temperature: 0,
            topP: 1,
            repetitionPenalty: 1,
            maxTokens: 256,
            maxSteps: 3,
            availableTools: ToolRegistry.all,
            relevantMemories: []
        )
        let final = AgentService.postprocessStructuredFinalAnswerForTests(
            "Check out BattlBox's guide on building an underground shelter: https://www.battlbox.com/blogs/outdoors/how-to-build-an-underground-shelter-a-comprehensive-guide",
            req: req,
            observations: [
                ("web.search", """
                Search results for: diy underground shelter

                <lumen_web_payload>{"results":[{"source":"www.bushcraftbasecamp.com","title":"10 Steps to Build a DIY Underground Bushcraft Survival Shelter","mediaKind":"page","url":"https://www.bushcraftbasecamp.com/10-steps-to-build-a-diy-underground-bushcraft-survival-shelter/"},{"source":"www.battlbox.com","title":"How To Build An Underground Shelter - Battlbox.com","mediaKind":"page","url":"https://www.battlbox.com/blogs/outdoors/how-to-build-an-underground-shelter-a-comprehensive-guide"}],"kind":"searchResults"}</lumen_web_payload>
                """)
            ],
            steps: [AgentStep(kind: .observation, content: "Search results", toolID: "web.search")]
        )

        #expect(final.contains("Summary:"))
        #expect(final.contains("DIY Underground Bushcraft Survival Shelter"))
        #expect(final.contains("Battlbox.com"))
        #expect(!final.contains("<lumen_web_payload"))
        #expect(!final.contains("No direct answer from web search"))
    }

    @Test func structuredWebNoDirectAnswerFallsBackToSearchObservations() {
        let req = AgentRequest(
            systemPrompt: "sys",
            history: [],
            userMessage: "Search the web for two recent Swift concurrency best practices and summarize them.",
            temperature: 0,
            topP: 1,
            repetitionPenalty: 1,
            maxTokens: 256,
            maxSteps: 3,
            availableTools: ToolRegistry.all,
            relevantMemories: []
        )
        let final = AgentService.postprocessStructuredFinalAnswerForTests(
            "No direct answer from web search. Try a different phrasing, or provide a URL to fetch directly.",
            req: req,
            observations: [
                ("web.search", """
                Search results for: Swift concurrency best practices
                {"title":"Swift.org - Concurrency","url":"https://swift.org/documentation/concurrency/","snippet":"Prefer structured concurrency so cancellation and errors propagate through child tasks."}
                {"title":"Apple Developer - MainActor","url":"https://developer.apple.com/documentation/swift/mainactor","snippet":"Keep UI state updates isolated to MainActor and move long-running work off the main actor."}
                """)
            ],
            steps: [AgentStep(kind: .observation, content: "Search results", toolID: "web.search")]
        )

        #expect(final.contains("Summary:"))
        #expect(final.contains("structured concurrency"))
        #expect(final.contains("MainActor"))
        #expect(!final.lowercased().contains("no direct answer from web search"))
    }

    @Test func structuredRAGEmptyRetrievalOverridesPollutedFallbackFinal() {
        let req = AgentRequest(
            systemPrompt: "sys",
            history: [],
            userMessage: "Search my files for architecture notes and summarize key modules.",
            temperature: 0,
            topP: 1,
            repetitionPenalty: 1,
            maxTokens: 256,
            maxSteps: 3,
            availableTools: ToolRegistry.all,
            relevantMemories: []
        )
        let final = AgentService.postprocessStructuredFinalAnswerForTests(
            "I'm ready. Please ask again or tell me what you'd like to do next. Key modules: core module details were retrieved from local file snippets [1].",
            req: req,
            observations: [
                ("rag.search", "No matching files found for 'architecture notes'. Your local index appears empty. Import or create local files, then reindex.")
            ],
            steps: [AgentStep(kind: .observation, content: "No matching files found", toolID: "rag.search")]
        )

        #expect(final == "I searched your local files but found no matching architecture notes. The local index appears empty; import or create files and reindex.")
        #expect(!final.contains("Key modules"))
        #expect(!final.contains("[1]"))
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
