import XCTest
@testable import Lumen

final class PromptFastInteractiveBudgetTests: XCTestCase {
    func testShortSimplePromptUsesFastInteractiveBudget() {
        let history = (0..<8).flatMap { index in
            [
                (role: MessageRole.user, content: String(repeating: "older user turn \(index) ", count: 40)),
                (role: MessageRole.assistant, content: String(repeating: "older assistant turn \(index) ", count: 40))
            ]
        }
        let memories = (0..<6).map { index in
            MemoryContextItem(
                content: String(repeating: "memory \(index) ", count: 40),
                scope: .conversation,
                authority: .referenceOnly,
                createdAt: nil,
                expiresAt: nil,
                source: "test",
                topic: nil
            )
        }
        let selection = PromptLatencyClassifier.classify(
            userMessage: "Hi",
            attachments: [],
            developerTraceModeEnabled: false,
            reasoningCaptureEnabled: false,
            modelName: "chat"
        )
        let assembly = PromptAssembler.assemble(
            systemPrompt: String(repeating: "Verbose developer/system instructions. ", count: 200),
            history: history,
            userMessage: "Hi",
            memories: memories,
            attachments: [],
            budget: .fastInteractive(),
            latencyClass: selection.latencyClass
        )

        XCTAssertEqual(selection.latencyClass, .fastInteractive)
        XCTAssertLessThanOrEqual(assembly.usedChars, PromptBudgetConstants.fastInteractiveTotalChars)
        XCTAssertLessThanOrEqual(assembly.history.count, 2)
        XCTAssertTrue(assembly.attachmentStates.isEmpty)
    }

    func testAttachmentPromptKeepsDocumentGroundedBudget() throws {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString).appendingPathExtension("txt")
        try String(repeating: "attached document context ", count: 200).write(to: url, atomically: true, encoding: .utf8)
        defer { try? FileManager.default.removeItem(at: url) }
        let attachment = ChatAttachment(name: "doc.txt", kind: .text, path: url.path, byteSize: 1024)
        let selection = PromptLatencyClassifier.classify(
            userMessage: "Summarize this file",
            attachments: [attachment],
            developerTraceModeEnabled: false,
            reasoningCaptureEnabled: false,
            modelName: "chat"
        )
        let budget = PromptBudget.make(
            contextSize: 4096,
            maxTokens: 512,
            systemPromptChars: 64,
            userMessageChars: 19,
            hasAttachments: true,
            hasMemories: false
        )
        let assembly = PromptAssembler.assemble(
            systemPrompt: "system",
            history: [],
            userMessage: "Summarize this file",
            memories: [],
            attachments: [attachment],
            budget: budget,
            latencyClass: selection.latencyClass
        )

        XCTAssertEqual(selection.latencyClass, .documentGrounded)
        XCTAssertEqual(assembly.attachmentStates.count, 1)
        XCTAssertGreaterThan(assembly.attachmentStates[0].includedChars, 0)
        XCTAssertTrue(assembly.systemPrompt.contains("Attachment 1"))
    }

    func testDeveloperTraceDoesNotUseFastSlimming() {
        let selection = PromptLatencyClassifier.classify(
            userMessage: "Hi",
            attachments: [],
            developerTraceModeEnabled: true,
            reasoningCaptureEnabled: true,
            modelName: "chat"
        )

        XCTAssertEqual(selection.latencyClass, .developerTrace)
    }

    func testFastSystemPromptPreservesOriginalConstraintsWhenSlimmed() {
        let systemPrompt = "MUST_KEEP_TOOL_POLICY_START "
            + String(repeating: "verbose operational guidance ", count: 120)
            + " MUST_KEEP_PRIVACY_RULE_END"
        let assembly = PromptAssembler.assemble(
            systemPrompt: systemPrompt,
            history: [],
            userMessage: "Hi",
            memories: [],
            attachments: [],
            budget: .fastInteractive(),
            latencyClass: .fastInteractive
        )

        XCTAssertLessThanOrEqual(assembly.systemPrompt.count, PromptBudgetConstants.fastInteractiveSystemChars)
        XCTAssertTrue(assembly.systemPrompt.contains("MUST_KEEP_TOOL_POLICY_START"))
        XCTAssertTrue(assembly.systemPrompt.contains("MUST_KEEP_PRIVACY_RULE_END"))
        XCTAssertTrue(assembly.systemPrompt.contains("Fast interactive mode"))
    }

    func testNormalInteractiveSmallMemoryShareDoesNotUseFastMemoryCaps() {
        let memoryContent = "normal memory " + String(repeating: "detail ", count: 16) + "normal-memory-tail"
        let memory = MemoryContextItem(
            content: memoryContent,
            scope: .conversation,
            authority: .referenceOnly,
            createdAt: nil,
            expiresAt: nil,
            source: "test",
            topic: nil
        )
        let budget = PromptBudget(totalChars: 1_000, attachmentsShare: 0, memoriesShare: PromptBudgetConstants.fastInteractiveMemoriesChars, historyShare: 0)

        let normal = PromptAssembler.assemble(
            systemPrompt: "system",
            history: [],
            userMessage: "Tell me more",
            memories: [memory],
            attachments: [],
            budget: budget,
            latencyClass: .normalInteractive
        )
        let fast = PromptAssembler.assemble(
            systemPrompt: "system",
            history: [],
            userMessage: "Hi",
            memories: [memory],
            attachments: [],
            budget: budget,
            latencyClass: .fastInteractive
        )

        XCTAssertTrue(normal.systemPrompt.contains(memoryContent))
        XCTAssertFalse(normal.systemPrompt.contains("[... truncated ...]"))
        XCTAssertTrue(fast.systemPrompt.contains("[... truncated ...]"))
    }

    func testFinalPromptBuildResultStaysUnderFastCapAfterModelWrappers() async {
        let request = GenerateRequest(
            systemPrompt: String(repeating: "Verbose developer/system instructions. ", count: 240),
            history: (0..<10).flatMap { index in
                [
                    (role: MessageRole.user, content: String(repeating: "older user turn \(index) ", count: 35)),
                    (role: MessageRole.assistant, content: String(repeating: "older assistant turn \(index) ", count: 35))
                ]
            },
            userMessage: "Hi",
            temperature: 0.7,
            topP: 0.9,
            repetitionPenalty: 1.05,
            maxTokens: 128,
            modelName: "chat",
            relevantMemories: (0..<6).map { index in
                MemoryContextItem(
                    content: String(repeating: "memory \(index) ", count: 35),
                    scope: .conversation,
                    authority: .referenceOnly,
                    createdAt: nil,
                    expiresAt: nil,
                    source: "test",
                    topic: nil
                )
            },
            attachments: [],
            developerTraceModeEnabled: false,
            reasoningCaptureEnabled: false
        )

        let result = await AppLlamaService.shared.buildMessagesForTesting(req: request, contextSize: 4096, slot: .cortex)

        XCTAssertEqual(result.latencySelection.latencyClass, .fastInteractive)
        XCTAssertLessThanOrEqual(result.finalPromptChars, PromptBudgetConstants.fastInteractiveTotalChars)
        XCTAssertEqual(result.estimatedPromptTokens, max(1, result.finalPromptChars / 4))
    }
}
