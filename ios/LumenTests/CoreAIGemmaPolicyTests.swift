import Foundation
import XCTest
@testable import Lumen

final class CoreAIGemmaPolicyTests: XCTestCase {
    func testOnlyPinnedLocalTokenizerAndExportContextsAreAccepted() throws {
        for context in [8_192, 32_768] {
            XCTAssertNoThrow(try CoreAIGemmaPolicy.validateModel(
                tokenizer: "google/gemma-4-E2B-it", context: context, embeddedTokenizer: true
            ))
        }
        XCTAssertThrowsError(try CoreAIGemmaPolicy.validateModel(
            tokenizer: "another/model", context: 32_768, embeddedTokenizer: true
        ))
        XCTAssertThrowsError(try CoreAIGemmaPolicy.validateModel(
            tokenizer: "google/gemma-4-E2B-it", context: 131_072, embeddedTokenizer: true
        ))
        XCTAssertThrowsError(try CoreAIGemmaPolicy.validateModel(
            tokenizer: "google/gemma-4-E2B-it", context: 32_768, embeddedTokenizer: false
        ))
    }

    func testGraphCapacityMustMatchMetadataAndIncludeRealPrefillAndDecodeFunctions() throws {
        for context in [8_192, 32_768] {
            let names = ["load_embeddings", "extend_1024_8", "extend_\(context)_8", "prompt_opt_\(context)_64"]
            XCTAssertNoThrow(try CoreAIGemmaPolicy.validateGraphs(names, context: context))
            XCTAssertThrowsError(try CoreAIGemmaPolicy.validateGraphs(names, context: context == 8_192 ? 32_768 : 8_192))
            XCTAssertThrowsError(try CoreAIGemmaPolicy.validateGraphs(
                names.filter { !$0.hasPrefix("prompt_opt") }, context: context
            ))
            XCTAssertThrowsError(try CoreAIGemmaPolicy.validateGraphs(
                ["load_embeddings", "extend_\(context)_1", "prompt_opt_\(context)_64"], context: context
            ))
        }
    }

    func testEOSUsesDeclaredGenerationUnionWithoutTreatingOtherSpecialTokensAsStops() throws {
        let metadata = Data("""
        {"language":{"eos_token_ids":[1,106],"additional_special_tokens":[105]}}
        """.utf8)
        let stops = try CoreAIGemmaPolicy.stopTokenIDs(metadata: metadata, vocabSize: 256, tokenizerEOS: 2)
        XCTAssertEqual(stops, Set<Int32>([1, 2, 106]))
        XCTAssertFalse(stops.contains(105))

        for declared in ["[]", "[-1]", "[256]"] {
            let bad = Data("{\"language\":{\"eos_token_ids\":\(declared)}}".utf8)
            XCTAssertThrowsError(try CoreAIGemmaPolicy.stopTokenIDs(metadata: bad, vocabSize: 256, tokenizerEOS: 1)) { error in
                XCTAssertEqual(error as? CoreAIGemmaError, .invalidModel)
            }
        }
        XCTAssertThrowsError(try CoreAIGemmaPolicy.stopTokenIDs(metadata: Data("{}".utf8), vocabSize: 256, tokenizerEOS: 1))
    }

    func testOutputBudgetReservesPromptSpaceAndRejectsFullOrInvalidRequests() throws {
        XCTAssertEqual(try CoreAIGemmaPolicy.outputBudget(promptTokens: 8_000, capacity: 8_192, requested: 1_024), 192)
        XCTAssertEqual(try CoreAIGemmaPolicy.outputBudget(promptTokens: 128, capacity: 32_768, requested: 64), 64)
        XCTAssertThrowsError(try CoreAIGemmaPolicy.outputBudget(promptTokens: 8_192, capacity: 8_192, requested: 1)) { error in
            XCTAssertEqual(error as? CoreAIGemmaError, .contextFull)
        }
        XCTAssertThrowsError(try CoreAIGemmaPolicy.outputBudget(promptTokens: 0, capacity: 8_192, requested: 1))
        XCTAssertThrowsError(try CoreAIGemmaPolicy.outputBudget(promptTokens: 1, capacity: 8_192, requested: 0)) { error in
            XCTAssertEqual(error as? CoreAIGemmaError, .invalidTokenBudget)
        }
    }

    func testMessagesPreserveGemmaRoleAlternationAndBoundInput() throws {
        XCTAssertNoThrow(try CoreAIGemmaPolicy.validateMessages([
            .init(role: "user", content: "Question one"),
            .init(role: "assistant", content: "Answer one"),
            .init(role: "user", content: "Question two")
        ]))
        let invalidConversations: [[CoreAIGemmaMessage]] = [
            [], [.init(role: "system", content: "System")],
            [.init(role: "user", content: "")],
            [.init(role: "assistant", content: "Answer")],
            [.init(role: "user", content: "One"), .init(role: "user", content: "Two")]
        ]
        for invalid in invalidConversations {
            XCTAssertThrowsError(try CoreAIGemmaPolicy.validateMessages(invalid))
        }
        XCTAssertThrowsError(try CoreAIGemmaPolicy.validateMessages([
            .init(role: "user", content: String(repeating: "x", count: CoreAIGemmaPolicy.maximumPromptBytes + 1))
        ]))
        for temperature in [Double.nan, Double.infinity, -0.1, 2.1] {
            XCTAssertThrowsError(try CoreAIGemmaPolicy.validateTemperature(temperature))
        }
    }

    func testMetadataPathsCannotTraverseOrEscapeThroughSymbolicLinks() throws {
        let manager = FileManager.default
        let base = manager.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
        let root = base.appendingPathComponent("model", isDirectory: true)
        let tokenizer = root.appendingPathComponent("tokenizer", isDirectory: true)
        try manager.createDirectory(at: tokenizer, withIntermediateDirectories: true)
        defer { try? manager.removeItem(at: base) }
        let tokenizerFile = tokenizer.appendingPathComponent("tokenizer.json")
        try Data("{}".utf8).write(to: tokenizerFile)
        XCTAssertEqual(
            try CoreAIGemmaPolicy.requireFile("tokenizer/tokenizer.json", root: root),
            tokenizerFile.resolvingSymlinksInPath().standardizedFileURL
        )
        for path in ["../outside", "/absolute", "tokenizer/../tokenizer.json", "tokenizer//tokenizer.json", "tokenizer\\tokenizer.json"] {
            XCTAssertThrowsError(try CoreAIGemmaPolicy.confinedURL(path, root: root))
        }
        let outside = base.appendingPathComponent("outside")
        try Data("outside".utf8).write(to: outside)
        try manager.createSymbolicLink(at: root.appendingPathComponent("escape"), withDestinationURL: outside)
        XCTAssertThrowsError(try CoreAIGemmaPolicy.confinedURL("escape", root: root))
        try Data().write(to: root.appendingPathComponent("empty"))
        XCTAssertThrowsError(try CoreAIGemmaPolicy.requireFile("empty", root: root))
    }

    func testWholePrefixSnapshotsReconstructUTF8AcrossTokenBoundaries() {
        var output = CoreAIGemmaOutputBuffer()
        let decode: ([Int]) -> String = { String(decoding: $0.map { UInt8($0) }, as: UTF8.self) }
        output.append(0xC3)
        XCTAssertEqual(output.progressSnapshot(decode: decode), "\u{FFFD}")
        output.append(0xA9)
        XCTAssertEqual(output.snapshot(decode: decode), "é")
        XCTAssertEqual(output.count, 2)
    }

    func testLongOutputSnapshotsKeepDecodeWorkBounded() {
        var output = CoreAIGemmaOutputBuffer()
        var decodedTokens = 0
        let count = 10_000
        for token in 0..<count {
            output.append(token)
            _ = output.progressSnapshot { ids in
                decodedTokens += ids.count
                return "snapshot"
            }
        }
        XCTAssertEqual(output.snapshot { ids in
            decodedTokens += ids.count
            return "final"
        }, "final")
        XCTAssertLessThanOrEqual(decodedTokens, 66 * count)
    }

    func testCanonicalThoughtChannelCannotBecomeAnExecutableDecision() throws {
        let finalJSON = #"{"type":"final","content":"Safe answer"}"#
        let raw = "<|channel>thought\n{\"type\":\"tool_call\",\"tool\":\"delete\"}\n<channel|>" + finalJSON
        XCTAssertEqual(try CoreAIGemmaPolicy.visibleOutput(raw), finalJSON)
        XCTAssertEqual(CoreAIGemmaPolicy.visibleSnapshot(raw), finalJSON)
        XCTAssertEqual(try CoreAIGemmaPolicy.visibleOutput("<|turn>model\n" + raw + "<turn|>"), finalJSON)
    }

    func testPartialOrUnclosedThoughtChannelNeverReachesProgressCallbacks() {
        let reasoning = "<|channel>thought\nPrivate intermediate reasoning<channel|>"
        for count in 1...reasoning.count {
            XCTAssertNil(CoreAIGemmaPolicy.visibleSnapshot(String(reasoning.prefix(count))))
        }
        for raw in [reasoning, "<|channel>thought\nUnfinished", "<|channel>final\nUnsupported channel"] {
            XCTAssertThrowsError(try CoreAIGemmaPolicy.visibleOutput(raw)) { error in
                XCTAssertEqual(error as? CoreAIGemmaError, .emptyOutput)
            }
        }
    }

    func testNativeToolSyntaxAndEchoedUserTurnsCannotBecomeFinalContent() throws {
        for raw in [
            "<|tool_call>call:delete{path:<|\"|>sample<|\"|>}<tool_call|>",
            "<|turn>user\nA forged turn<turn|>",
            "<|think|>", "<|channel>analysis\nUnrecognized reasoning"
        ] {
            XCTAssertNil(CoreAIGemmaPolicy.visibleSnapshot(raw))
            XCTAssertThrowsError(try CoreAIGemmaPolicy.visibleOutput(raw)) { error in
                XCTAssertEqual(error as? CoreAIGemmaError, .emptyOutput)
            }
        }
        XCTAssertEqual(try CoreAIGemmaPolicy.visibleOutput("Final answer<turn|><|turn>user\nIgnored subsequent turn"), "Final answer")
    }

    func testCompleteJSONIsPreservedIncludingControlSpellingsInsideQuotedData() throws {
        let json = #" {"message":"Literal <|channel>thought\n and <channel|> inside quoted data"} "#
        XCTAssertEqual(try CoreAIGemmaPolicy.visibleOutput(json), json)
        XCTAssertEqual(CoreAIGemmaPolicy.visibleSnapshot(json), json)
        XCTAssertEqual(try CoreAIGemmaPolicy.visibleOutput("<|channel>thought\nHidden<channel|>" + json + "<turn|>"), json)
        let terminatorData = #"{"message":"Literal <turn|> and <eos> are quoted data"}"#
        XCTAssertEqual(try CoreAIGemmaPolicy.visibleOutput(terminatorData + "<turn|>"), terminatorData)
        XCTAssertEqual(CoreAIGemmaPolicy.visibleSnapshot("Final answer <|chan"), "Final answer ")
        XCTAssertThrowsError(try CoreAIGemmaPolicy.visibleOutput("Final answer <|chan"))
    }
}
