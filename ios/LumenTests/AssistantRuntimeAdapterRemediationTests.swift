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
        XCTAssertEqual(runtime.unavailableReason, "CoreML embedding runtime is experimental and is excluded from Release routing.")
        XCTAssertEqual(runtime.availabilityStatus, "experimental runtime excluded from Release routing")
    }


    func testCoreMLNilModelUnavailableReason() {
        let runtime = CoreMLRuntimeAdapter(modelURL: nil)
        XCTAssertFalse(runtime.isAvailable)
        XCTAssertEqual(runtime.unavailableReason, "CoreML embedding runtime is experimental and is excluded from Release routing.")
        XCTAssertEqual(runtime.availabilityStatus, "experimental runtime excluded from Release routing")
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
        } catch CoreMLRuntimeError.experimentalRuntimeDisabled {
            // Expected while the adapter is experimental.
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
        } catch CoreMLRuntimeError.experimentalRuntimeDisabled {
            // Expected while the adapter is experimental.
        } catch {
            XCTFail("Unexpected error: \(error)")
        }
    }

    func testCapabilityMatrixMarksExperimentalAdaptersNonSelectable() {
        let matrix = AssistantRuntimeCapabilityMatrix.current(
            foundation: FoundationModelsRuntimeAdapter(),
            llama: LlamaRuntimeAdapter(isAvailable: false),
            fallback: DeterministicFallbackRuntime(),
            coreML: CoreMLRuntimeAdapter(modelURL: nil)
        )

        let foundation = matrix.row(for: .foundationModels)
        XCTAssertEqual(foundation?.generationSupported, false)
        XCTAssertEqual(foundation?.generationSelectable, false)
        XCTAssertEqual(foundation?.status, "experimental runtime excluded from Release routing")

        let coreML = matrix.row(for: .coreML)
        XCTAssertEqual(coreML?.embeddingSupported, false)
        XCTAssertEqual(coreML?.embeddingSelectable, false)
        XCTAssertEqual(coreML?.status, "experimental runtime excluded from Release routing")

        XCTAssertEqual(matrix.selectableGenerationRuntimes, [.deterministicFallback])
        XCTAssertEqual(matrix.selectableEmbeddingRuntimes, [])
    }

    func testCapabilityMatrixDoesNotExposeBareLlamaAvailabilityStatus() {
        let matrix = AssistantRuntimeCapabilityMatrix.current()
        let llama = matrix.row(for: .llama)

        XCTAssertNotEqual(llama?.status, "available")
        XCTAssertTrue(llama?.status.contains("generation") == true)
        XCTAssertTrue(llama?.status.contains("embeddings") == true)
        XCTAssertEqual(llama?.embeddingSupported, true)
    }

    func testMetricErrorSanitizerDoesNotExposeDescription() {
        let code = RuntimeMetricErrorSanitizer.code(for: LocalRuntimeError.unavailable("raw private text"))
        XCTAssertEqual(code, "runtime_unavailable")
        XCTAssertFalse(code.contains("raw private text"))
    }
}
