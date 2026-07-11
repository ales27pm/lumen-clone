import XCTest
@testable import Lumen

@MainActor
final class AssistantKernelLlamaRuntimeAdapterTests: XCTestCase {
    private actor StubLlamaStreamingService: LlamaRuntimeStreamingService {
        let isChatLoaded: Bool
        let isEmbedLoaded: Bool
        private let tokens: [GenerationToken]
        private let embedding: [Double]

        init(
            isChatLoaded: Bool,
            isEmbedLoaded: Bool = false,
            tokens: [GenerationToken] = [],
            embedding: [Double] = []
        ) {
            self.isChatLoaded = isChatLoaded
            self.isEmbedLoaded = isEmbedLoaded
            self.tokens = tokens
            self.embedding = embedding
        }

        func stream(_ req: GenerateRequest, slot: LumenModelSlot) -> AsyncStream<GenerationToken> {
            AsyncStream { continuation in
                for token in tokens {
                    continuation.yield(token)
                }
                continuation.finish()
            }
        }

        func embed(_ text: String) async throws -> [Double] {
            embedding
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

    func testStructuredStreamingUsesInjectedLlamaService() async throws {
        let service = StubLlamaStreamingService(
            isChatLoaded: true,
            tokens: [.text(#"{"action":{"tool":"device.status","args":{}}}"#), .done]
        )
        let kernel = AssistantKernel(router: AssistantRuntimeRouter(llamaService: service))
        let request = GenerateRequest(
            systemPrompt: "Return JSON.",
            history: [],
            userMessage: "Check device status.",
            temperature: 0,
            topP: 1,
            repetitionPenalty: 1.05,
            maxTokens: 64,
            modelName: "agent-json",
            relevantMemories: []
        )

        let stream = try await kernel.streamStructuredLlama(request, slot: .executor)
        var output = ""
        for await token in stream {
            if case .text(let text) = token { output += text }
        }

        XCTAssertEqual(output, #"{"action":{"tool":"device.status","args":{}}}"#)
        XCTAssertEqual(AgentTurnParser.parse(output).action?.tool, "device.status")
    }

    func testRunTextTurnPropagatesSelectedLlamaUnavailable() async throws {
        let adapter = LlamaRuntimeAdapter(isAvailable: true, unavailableReason: nil, generateHandler: { _ in
            throw LocalRuntimeError.unavailable("controlled no loaded chat model")
        })
        let router = AssistantRuntimeRouter(llama: adapter)
        let kernel = AssistantKernel(router: router)
        let context = AssistantTurnContext(
            task: .chat,
            input: "hello",
            isForeground: true,
            lowPowerMode: false,
            thermalState: .nominal
        )

        do {
            _ = try await kernel.runTextTurn(context)
            XCTFail("Expected selected runtime failure to propagate")
        } catch LocalRuntimeError.unavailable(let reason) {
            XCTAssertEqual(reason, "controlled no loaded chat model")
        } catch {
            XCTFail("Unexpected error: \(error)")
        }
    }

    func testRunTextTurnMetersSelectedRuntimeFailure() async throws {
        let metricsURL = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("assistant-kernel-fallback-failure-\(UUID().uuidString).jsonl")
        defer { try? FileManager.default.removeItem(at: metricsURL) }

        let adapter = LlamaRuntimeAdapter(isAvailable: true, unavailableReason: nil, generateHandler: { _ in
            throw LocalRuntimeError.unavailable("controlled no loaded chat model")
        })
        let router = AssistantRuntimeRouter(llama: adapter)
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
            XCTFail("Expected runtime failure to propagate")
        } catch LocalRuntimeError.unavailable(let reason) {
            XCTAssertEqual(reason, "controlled no loaded chat model")
        } catch {
            XCTFail("Unexpected error: \(error)")
        }

        let metrics = try await RuntimeMetricsStore(fileURL: metricsURL).recentMetrics(limit: 1)
        XCTAssertEqual(metrics.last?.runtimeName, AssistantRuntimeKind.llama.rawValue)
        XCTAssertEqual(metrics.last?.policySummary, "llama available")
        XCTAssertEqual(metrics.last?.success, false)
        XCTAssertEqual(metrics.last?.errorCode, "runtime_unavailable")
    }

    func testRunTextTurnMetersUnavailableRuntimeSelection() async throws {
        let metricsURL = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("assistant-kernel-unavailable-selection-\(UUID().uuidString).jsonl")
        defer { try? FileManager.default.removeItem(at: metricsURL) }

        let router = AssistantRuntimeRouter(
            llama: .init(generateHandler: { _ in "should not run" }),
            allowDiagnosticFallbackSelection: false
        )
        let kernel = AssistantKernel(router: router, metricsStore: RuntimeMetricsStore(fileURL: metricsURL))
        let context = AssistantTurnContext(
            task: .chat,
            input: "hello",
            isForeground: true,
            lowPowerMode: false,
            thermalState: .nominal,
            allowHeavyRuntime: false
        )

        do {
            _ = try await kernel.runTextTurn(context)
            XCTFail("Expected unavailable runtime selection to throw")
        } catch AssistantKernel.KernelError.unsupportedRuntimeForTextTurn(let runtime) {
            XCTAssertEqual(runtime, .unavailable)
        } catch {
            XCTFail("Unexpected error: \(error)")
        }

        let metrics = try await RuntimeMetricsStore(fileURL: metricsURL).recentMetrics(limit: 1)
        XCTAssertEqual(metrics.last?.runtimeName, AssistantRuntimeKind.unavailable.rawValue)
        XCTAssertEqual(metrics.last?.policySummary, "foregroundInteractive: heavyRuntime=false")
        XCTAssertEqual(metrics.last?.success, false)
        XCTAssertEqual(metrics.last?.errorCode, "unsupported_runtime_for_text_turn")
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

    func testLiveLlamaAdapterEmbedsWhenEmbeddingModelLoaded() async throws {
        let service = StubLlamaStreamingService(
            isChatLoaded: false,
            isEmbedLoaded: true,
            embedding: [0.25, 0.5, 0.75]
        )
        let adapter = LlamaRuntimeAdapter.live(service: service)

        let embeddingSelectable = await adapter.isEmbeddingSelectable()
        XCTAssertTrue(embeddingSelectable)
        let vector = try await adapter.embed(request: EmbeddingRequest(text: "hello"))

        XCTAssertEqual(vector, [Float(0.25), Float(0.5), Float(0.75)])
        XCTAssertTrue(adapter.supportsEmbeddings)
    }

    func testCapabilityMatrixMarksLoadedLiveLlamaEmbeddingsSelectable() async {
        let service = StubLlamaStreamingService(
            isChatLoaded: true,
            isEmbedLoaded: true,
            embedding: [1.0]
        )
        let adapter = LlamaRuntimeAdapter.live(service: service)
        let matrix = await AssistantRuntimeCapabilityMatrix.currentIncludingRuntimeState(llama: adapter)
        let llama = matrix.row(for: .llama)

        XCTAssertEqual(llama?.embeddingSupported, true)
        XCTAssertEqual(llama?.embeddingSelectable, true)
        XCTAssertEqual(llama?.status, "generation available; embeddings available")
        XCTAssertEqual(matrix.selectableEmbeddingRuntimes, [.llama])
    }

    func testKernelRunEmbeddingUsesLiveLlamaEmbeddingRuntime() async throws {
        let metricsURL = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("assistant-kernel-embedding-\(UUID().uuidString).jsonl")
        defer { try? FileManager.default.removeItem(at: metricsURL) }
        let service = StubLlamaStreamingService(
            isChatLoaded: false,
            isEmbedLoaded: true,
            embedding: [0.1, 0.2]
        )
        let router = AssistantRuntimeRouter(llamaService: service, allowDiagnosticFallbackSelection: false)
        let kernel = AssistantKernel(router: router, metricsStore: RuntimeMetricsStore(fileURL: metricsURL))
        let context = AssistantTurnContext(
            task: .embedding,
            input: "semantic text",
            isForeground: true,
            lowPowerMode: false,
            thermalState: .nominal
        )

        let vector = try await kernel.runEmbedding(context)

        XCTAssertEqual(vector.count, 2)
        XCTAssertEqual(vector[0], 0.1, accuracy: 0.000001)
        XCTAssertEqual(vector[1], 0.2, accuracy: 0.000001)
        let metrics = try await RuntimeMetricsStore(fileURL: metricsURL).recentMetrics(limit: 1)
        XCTAssertEqual(metrics.last?.runtimeName, AssistantRuntimeKind.llama.rawValue)
        XCTAssertEqual(metrics.last?.policySummary, "llama embedding available")
        XCTAssertEqual(metrics.last?.success, true)
    }
}
