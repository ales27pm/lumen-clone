import XCTest
@testable import Lumen

final class AgentModeGroundingHardeningTests: XCTestCase {
    func testAgentSimplePromptUsesFastPathAndSkipsFullGrounding() {
        let req = makeRequest(userMessage: "Yo", availableTools: ToolRegistry.all, memories: makeMemories(count: 5))
        let result = SlotAgentService.fastGroundingResult(for: req, options: .default)

        XCTAssertTrue(SlotAgentService.shouldUseFastAgentPath(req))
        XCTAssertEqual(result.metricsSummary, "fast-agent")
        XCTAssertTrue(result.bridgedTools.isEmpty)
        XCTAssertLessThanOrEqual(result.sections.count, 2)
        XCTAssertFalse(result.sections.contains { $0.title.lowercased().contains("source") || $0.title.lowercased().contains("tool") })
    }

    func testAgentSimplePromptStaysUnderFastGroundingCharCap() {
        let req = makeRequest(
            userMessage: "Yo",
            availableTools: ToolRegistry.all,
            memories: makeMemories(count: 8, content: String(repeating: "long memory ", count: 50))
        )
        let result = SlotAgentService.fastGroundingResult(for: req, options: .default)

        XCTAssertLessThanOrEqual(result.userMessage.count + result.systemPrompt.count, PromptBudgetConstants.fastInteractiveTotalChars)
        XCTAssertLessThanOrEqual(result.sections.reduce(0) { $0 + $1.estimatedChars }, 220)
    }

    func testDeveloperTraceAndReasoningCaptureIntentionallyBypassFastMode() {
        let traceSelection = PromptLatencyClassifier.classify(
            userMessage: "Yo",
            attachments: [],
            developerTraceModeEnabled: true,
            reasoningCaptureEnabled: false,
            modelName: "chat"
        )
        let reasoningSelection = PromptLatencyClassifier.classify(
            userMessage: "Yo",
            attachments: [],
            developerTraceModeEnabled: false,
            reasoningCaptureEnabled: true,
            modelName: "chat"
        )

        XCTAssertEqual(traceSelection.latencyClass, .developerTrace)
        XCTAssertEqual(reasoningSelection.latencyClass, .developerTrace)
    }

    @MainActor func testGroundingCancellationExitsCleanly() async {
        let token = AgentGroundingCancellationToken()
        token.cancel()
        let request = LegacyGroundingRequest(
            userMessage: "Use tools please",
            conversationID: nil,
            turnID: UUID(),
            history: [],
            mode: .foreground,
            task: .chat,
            roleOrSlot: "slotAgent",
            externalRelevantMemories: makeMemories(count: 2),
            externalAvailableTools: Array(ToolRegistry.all.prefix(5)),
            policy: .slotAgent,
            baseSystemPrompt: "system",
            preventDoubleGrounding: true
        )

        let result = await LegacyTurnGroundingCoordinator.shared.prepareGroundedRequest(
            request,
            provider: LegacyGroundingContextProvider(directContext: nil, allowSharedFallback: false),
            cancellationToken: token
        )

        XCTAssertEqual(result.metricsSummary, "cancelled")
        XCTAssertTrue(result.degradedReasons.contains("cancelled"))
        XCTAssertTrue(result.sections.isEmpty)
        XCTAssertTrue(result.bridgedTools.isEmpty)
    }

    func testToolHeavyAndFilePromptsDoNotUseFastPath() {
        let toolReq = makeRequest(userMessage: "Search the web for SwiftData examples", availableTools: ToolRegistry.all, memories: [])
        let fileReq = makeRequest(
            userMessage: "Summarize this file",
            availableTools: ToolRegistry.all,
            memories: [],
            attachments: [ChatAttachment(name: "doc.txt", kind: .text, path: "/tmp/doc.txt", byteSize: 128)]
        )

        XCTAssertFalse(SlotAgentService.shouldUseFastAgentPath(toolReq))
        XCTAssertFalse(SlotAgentService.shouldUseFastAgentPath(fileReq))
    }

    private func makeRequest(
        userMessage: String,
        availableTools: [ToolDefinition],
        memories: [MemoryContextItem],
        attachments: [ChatAttachment] = []
    ) -> AgentRequest {
        AgentRequest(
            systemPrompt: "system",
            history: [],
            userMessage: userMessage,
            temperature: 0.7,
            topP: 0.9,
            repetitionPenalty: 1.05,
            maxTokens: 128,
            maxSteps: 3,
            availableTools: availableTools,
            relevantMemories: memories,
            attachments: attachments
        )
    }

    private func makeMemories(count: Int, content: String = "tiny memory") -> [MemoryContextItem] {
        (0..<count).map { index in
            MemoryContextItem(
                content: "\(content) \(index)",
                scope: .conversation,
                authority: .referenceOnly,
                createdAt: nil,
                expiresAt: nil,
                source: "test",
                topic: nil
            )
        }
    }
}
