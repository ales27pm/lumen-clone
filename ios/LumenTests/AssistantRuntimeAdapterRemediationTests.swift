import XCTest
@testable import Lumen

final class AssistantRuntimeAdapterRemediationTests: XCTestCase {
    func testLlamaAndFoundationDoNotEchoPrompt() async {
        let request = TextGenerationRequest(prompt: "private prompt", systemPrompt: "", maxTokens: 16)
        do {
            _ = try await LlamaRuntimeAdapter(isAvailable: true, unavailableReason: nil).generate(request: request)
            XCTFail("Llama adapter should not produce prompt echo")
        } catch {}
        do {
            _ = try await FoundationModelsRuntimeAdapter().generate(request: request)
            XCTFail("FoundationModels adapter should not produce prompt echo")
        } catch {}
    }

    func testCoreMLMissingFileUnavailable() {
        let url = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent(UUID().uuidString)
        let runtime = CoreMLRuntimeAdapter(modelURL: url)
        XCTAssertFalse(runtime.isAvailable)
        XCTAssertEqual(runtime.unavailableReason, "CoreML embedding runtime staged: implementation missing")
        XCTAssertEqual(runtime.availabilityStatus, "staged: implementation missing")
    }


    func testCoreMLNilModelUnavailableReason() {
        let runtime = CoreMLRuntimeAdapter(modelURL: nil)
        XCTAssertFalse(runtime.isAvailable)
        XCTAssertEqual(runtime.unavailableReason, "CoreML embedding runtime staged: implementation missing")
        XCTAssertEqual(runtime.availabilityStatus, "staged: implementation missing")
    }

    func testCoreMLStagedEmbeddingDoesNotLookLikeEmptyEmbeddingSuccess() async {
        let url = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent("fake-\(UUID().uuidString).mlmodelc")
        FileManager.default.createFile(atPath: url.path, contents: Data(), attributes: nil)
        defer { try? FileManager.default.removeItem(at: url) }
        let runtime = CoreMLRuntimeAdapter(modelURL: url)

        XCTAssertFalse(runtime.isAvailable)
        XCTAssertFalse(runtime.supportsEmbeddings)

        do {
            let vector = try await runtime.embed(request: EmbeddingRequest(text: "hello", dimensions: nil))
            XCTFail("CoreML embed should not return empty success vector: \(vector)")
        } catch CoreMLRuntimeError.embeddingExtractionNotImplemented {
            // Expected while the adapter is staged.
        } catch {
            XCTFail("Unexpected error: \(error)")
        }
    }

    func testCoreMLEmbedThrowsWhenModelMissing() async {
        let url = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent(UUID().uuidString)
        let runtime = CoreMLRuntimeAdapter(modelURL: url)
        do {
            _ = try await runtime.embed(request: EmbeddingRequest(text: "hello", dimensions: nil))
            XCTFail("CoreML embed should not return an empty success vector for missing model")
        } catch CoreMLRuntimeError.embeddingExtractionNotImplemented {
            // Expected while the adapter is staged.
        } catch {
            XCTFail("Unexpected error: \(error)")
        }
    }

    func testMetricErrorSanitizerDoesNotExposeDescription() {
        let code = RuntimeMetricErrorSanitizer.code(for: LocalRuntimeError.unavailable("raw private text"))
        XCTAssertEqual(code, "runtime_unavailable")
        XCTAssertFalse(code.contains("raw private text"))
    }
}
