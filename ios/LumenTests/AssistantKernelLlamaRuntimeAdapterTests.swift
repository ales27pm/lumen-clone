import XCTest
@testable import Lumen

@MainActor
final class AssistantKernelLlamaRuntimeAdapterTests: XCTestCase {
    private actor StubLlamaStreamingService: LlamaRuntimeStreamingService {
        let isChatLoaded: Bool
        private let tokens: [GenerationToken]

        init(isChatLoaded: Bool, tokens: [GenerationToken] = []) {
            self.isChatLoaded = isChatLoaded
            self.tokens = tokens
        }

        func stream(_ req: GenerateRequest, slot: LumenModelSlot) -> AsyncStream<GenerationToken> {
            AsyncStream { continuation in
                for token in tokens {
                    continuation.yield(token)
                }
                continuation.finish()
            }
        }
    }

    func testDefaultRouterSelectsLiveLlamaWhenHeavyRuntimeAllowed() {
        let router = AssistantRuntimeRouter()
        let context = AssistantTurnContext(
            task: .chat,
            input: "hello",
            isForeground: true,
            lowPowerMode: false,
            thermalState: .nominal
        )

        XCTAssertEqual(router.runtime(for: context), .llama)
    }

    func testRunTextTurnFallsBackWhenSelectedLlamaBecomesUnavailable() async throws {
        let adapter = LlamaRuntimeAdapter(isAvailable: true, unavailableReason: nil) { _ in
            throw LocalRuntimeError.unavailable("controlled no loaded chat model")
        }
        let router = AssistantRuntimeRouter(llama: adapter)
        let kernel = AssistantKernel(router: router)
        let context = AssistantTurnContext(
            task: .chat,
            input: "hello",
            isForeground: true,
            lowPowerMode: false,
            thermalState: .nominal
        )

        let output = try await kernel.runTextTurn(context)

        XCTAssertEqual(output, "Lumen is running in limited local mode.")
    }

    func testRunTextTurnMetersFallbackFailureAfterLlamaUnavailable() async throws {
        let metricsURL = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("assistant-kernel-fallback-failure-\(UUID().uuidString).jsonl")
        defer { try? FileManager.default.removeItem(at: metricsURL) }

        let adapter = LlamaRuntimeAdapter(isAvailable: true, unavailableReason: nil) { _ in
            throw LocalRuntimeError.unavailable("controlled no loaded chat model")
        }
        let fallback = DeterministicFallbackRuntime { _ in
            throw LocalRuntimeError.unavailable("controlled fallback unavailable")
        }
        let router = AssistantRuntimeRouter(llama: adapter, fallback: fallback)
        let kernel = AssistantKernel(router: router, metricsStore: RuntimeMetricsStore(fileURL: metricsURL))
        let context = AssistantTurnContext(
            task: .chat,
            input: "hello",
            isForeground: true,
            lowPowerMode: false,
            thermalState: .nominal
        )

        do {
            _ = try await kernel.runTextTurn(context)
            XCTFail("Expected deterministic fallback failure to propagate")
        } catch LocalRuntimeError.unavailable(let reason) {
            XCTAssertEqual(reason, "controlled fallback unavailable")
        } catch {
            XCTFail("Unexpected error: \(error)")
        }

        let metrics = try await RuntimeMetricsStore(fileURL: metricsURL).recentMetrics(limit: 1)
        XCTAssertEqual(metrics.last?.runtimeName, AssistantRuntimeKind.deterministicFallback.rawValue)
        XCTAssertEqual(metrics.last?.policySummary, "fallback_after_llama_unavailable")
        XCTAssertEqual(metrics.last?.success, false)
        XCTAssertEqual(metrics.last?.errorCode, "runtime_unavailable")
    }

    func testLiveLlamaAdapterReportsAvailableToRouterButThrowsUntilModelLoaded() async {
        let service = StubLlamaStreamingService(isChatLoaded: false)
        let adapter = LlamaRuntimeAdapter.live(service: service)
        XCTAssertTrue(adapter.isAvailable)

        do {
            _ = try await adapter.generate(request: TextGenerationRequest(prompt: "hello", systemPrompt: "", maxTokens: 8))
            XCTFail("Expected live llama adapter to throw until a chat model is loaded")
        } catch LocalRuntimeError.unavailable(let reason) {
            XCTAssertTrue(reason.contains("no loaded chat model") || reason.contains("no visible output") || reason.contains("stream failed"))
        } catch {
            XCTFail("Unexpected error: \(error)")
        }
    }

    func testLiveLlamaAdapterTreatsStreamErrorTextAsUnavailable() async {
        let service = StubLlamaStreamingService(
            isChatLoaded: true,
            tokens: [.text("Generation error: model failed"), .done]
        )
        let adapter = LlamaRuntimeAdapter.live(service: service)

        do {
            _ = try await adapter.generate(request: TextGenerationRequest(prompt: "hello", systemPrompt: "", maxTokens: 8))
            XCTFail("Expected live llama stream error text to become an unavailable runtime error")
        } catch LocalRuntimeError.unavailable(let reason) {
            XCTAssertTrue(reason.contains("stream failed"))
        } catch {
            XCTFail("Unexpected error: \(error)")
        }
    }
}
