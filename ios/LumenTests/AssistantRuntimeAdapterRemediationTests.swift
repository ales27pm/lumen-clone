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
        XCTAssertEqual(runtime.unavailableReason, "Configured Core ML model file is missing")
    }


    func testCoreMLNilModelUnavailableReason() {
        let runtime = CoreMLRuntimeAdapter(modelURL: nil)
        XCTAssertFalse(runtime.isAvailable)
        XCTAssertEqual(runtime.unavailableReason, "No Core ML embedding model configured")
    }

    func testCoreMLEmbedThrowsWhenModelMissing() async {
        let url = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent(UUID().uuidString)
        let runtime = CoreMLRuntimeAdapter(modelURL: url)
        do {
            _ = try await runtime.embed(request: EmbeddingRequest(text: "hello", dimensions: nil))
            XCTFail("CoreML embed should not return an empty success vector for missing model")
        } catch CoreMLRuntimeError.modelNotFound {
            // Expected.
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
