import Foundation
import XCTest
@testable import Lumen

@MainActor
final class CoreAITextEmbeddingRoutingTests: XCTestCase {
    private actor Calls {
        private(set) var coreAI: [EmbeddingRequest] = []
        private(set) var legacy = 0
        func record(_ request: EmbeddingRequest) -> EmbeddingRuntimeResult {
            coreAI.append(request)
            let count = request.dimensions ?? 256
            return .init(vector: Array(repeating: 1 / sqrt(Double(count)), count: count), modelIdentifier: "test:eg2:128", formatVersion: 4)
        }
        func legacyVector() -> [Float] { legacy += 1; return [1] }
    }

    func testExplicitCoreAISelectionReceivesPurposeAndDimensionWithoutUsingLegacy() async throws {
        let calls = Calls()
        let kernel = AssistantKernel(router: AssistantRuntimeRouter(
            llama: .init(embedHandler: { _ in await calls.legacyVector() }, embeddingModelIdentifier: "test:legacy"),
            coreAI: .disabled,
            coreAIEmbedding: .init(isAvailable: true, embedHandler: { await calls.record($0) })
        ))
        let result = try await kernel.runEmbeddingWithIdentity(context(), purpose: .retrievalQuery, dimensions: 128)
        let requests = await calls.coreAI
        let legacy = await calls.legacy
        XCTAssertEqual(requests, [.init(text: "raw query", purpose: .retrievalQuery, dimensions: 128)])
        XCTAssertEqual(legacy, 0)
        XCTAssertEqual(result.vector.count, 128)
        XCTAssertEqual(result.formatVersion, 4)
        XCTAssertEqual(result.modelIdentifier, "test:eg2:128")
    }

    func testMissingSelectedPackageNeverFallsBackToLoadedLegacy() async {
        let calls = Calls()
        let router = AssistantRuntimeRouter(
            llama: .init(embedHandler: { _ in await calls.legacyVector() }, embeddingModelIdentifier: "test:legacy"),
            coreAIEmbedding: .init(isAvailable: false, isPreferred: true),
            allowDiagnosticFallbackSelection: true
        )
        XCTAssertEqual(router.selection(for: context()).runtime, .unavailable)
        do {
            _ = try await AssistantKernel(router: router).runEmbeddingWithIdentity(context(), purpose: .retrievalQuery)
            XCTFail("The selected missing package must fail")
        } catch { }
        let legacy = await calls.legacy
        XCTAssertEqual(legacy, 0)
    }

    func testEmbeddingSelectionDoesNotChangeTheGemmaGenerator() {
        let router = AssistantRuntimeRouter(
            coreAI: .init(isAvailable: true),
            coreAIEmbedding: .init(isAvailable: false, isPreferred: true)
        )
        let chat = AssistantTurnContext(task: .chat, input: "hello", isForeground: true, lowPowerMode: false, thermalState: .nominal)
        XCTAssertEqual(router.selection(for: chat).runtime, .coreAI)
        XCTAssertEqual(router.selection(for: context()).runtime, .unavailable)
        let matrix = AssistantRuntimeCapabilityMatrix.current(
            llama: .init(embedHandler: { _ in [1] }),
            coreAI: .init(isAvailable: true),
            coreAIEmbedding: .init(isAvailable: true)
        )
        XCTAssertEqual(matrix.selectableGenerationRuntimes, [.coreAI])
        XCTAssertEqual(matrix.selectableEmbeddingRuntimes, [.coreAI])
    }

    func testResourceDenialCannotSelectAnAlternativeEmbeddingModel() {
        let router = AssistantRuntimeRouter(
            llama: .init(embedHandler: { _ in [1] }),
            coreAIEmbedding: .init(isAvailable: true)
        )
        let denied = AssistantTurnContext(task: .embedding, input: "query", isForeground: false, lowPowerMode: true, thermalState: .critical, allowHeavyRuntime: false)
        XCTAssertEqual(router.selection(for: denied).runtime, .unavailable)
    }

    func testInvalidVectorOrMetadataIsRejectedWithoutFallback() async {
        let valid = Array(repeating: 1 / sqrt(256.0), count: 256)
        let invalid: [EmbeddingRuntimeResult] = [
            .init(vector: [1], modelIdentifier: "test:wrong-size", formatVersion: 4),
            .init(vector: Array(repeating: 0, count: 256), modelIdentifier: "test:zero", formatVersion: 4),
            .init(vector: Array(repeating: .nan, count: 256), modelIdentifier: "test:nan", formatVersion: 4),
            .init(vector: valid, modelIdentifier: "test:old-format", formatVersion: 3),
            .init(vector: valid, modelIdentifier: "", formatVersion: 4)
        ]
        for result in invalid {
            let adapter = CoreAITextEmbeddingRuntimeAdapter(isAvailable: true, embedHandler: { _ in result })
            do {
                _ = try await adapter.embedWithIdentity(request: .init(text: "query", purpose: .retrievalQuery))
                XCTFail("Invalid vectors must fail")
            } catch { }
        }
    }

    func testLegacyInputFormattingPreservesTaskPrefixesAndCallerText() {
        XCTAssertEqual(LlamaRuntimeAdapter.legacyEmbeddingText(.init(text: "raw query", purpose: .retrievalQuery)), "search_query: raw query")
        XCTAssertEqual(LlamaRuntimeAdapter.legacyEmbeddingText(.init(text: "Source type: note\nContent:\ntext", purpose: .retrievalDocument(title: "already in body"))), "search_document:\nSource type: note\nContent:\ntext")
        XCTAssertEqual(LlamaRuntimeAdapter.legacyEmbeddingText(.init(text: "search_query: literal user text")), "search_query: literal user text")
    }

    func testCoreAIConsumerBodiesKeepTheTailForTokenizerOverflowValidation() {
        let content = String(repeating: "x", count: 5_000) + " TAIL"
        XCTAssertTrue(SemanticEmbeddingText.queryBody(content, legacyBounds: false).hasSuffix("TAIL"))
        XCTAssertTrue(SemanticEmbeddingText.memoryDocumentBody(content: content, kind: .conversation, source: "test", topic: nil, legacyBounds: false).hasSuffix("TAIL"))
        XCTAssertTrue(SemanticEmbeddingText.documentBody(content: content, sourceType: "note", legacyBounds: false).hasSuffix("TAIL"))
        XCTAssertEqual(SemanticEmbeddingText.queryBody(content, legacyBounds: true).count, 4_096)
        XCTAssertEqual("search_query: " + SemanticEmbeddingText.queryBody(content, legacyBounds: true), SemanticEmbeddingText.query(content))
    }

    func testEmbeddingErrorsUseStablePayloadFreeMetricCategories() {
        XCTAssertEqual(RuntimeMetricErrorSanitizer.code(for: CoreAITextEmbeddingError.notPrepared), "coreai_embedding_not_prepared")
        XCTAssertEqual(RuntimeMetricErrorSanitizer.code(for: CoreAITextEmbeddingError.integrityFailure), "coreai_embedding_integrity_failure")
        XCTAssertEqual(RuntimeMetricErrorSanitizer.code(for: CoreAITextEmbeddingError.inputTooLong), "coreai_embedding_input_too_long")
    }

    private func context() -> AssistantTurnContext {
        .init(task: .embedding, input: "raw query", isForeground: true, lowPowerMode: false, thermalState: .nominal)
    }
}
