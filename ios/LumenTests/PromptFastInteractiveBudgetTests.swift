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
}
