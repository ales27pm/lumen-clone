import Foundation
import Testing
@testable import Lumen

@MainActor
struct GGUFEngineScaffoldTests {
    @Test func promptBuilderIncludesSystemContextToolsAndRoleMarkers() throws {
        let request = LLMRequest(
            messages: [
                LLMChatMessage(role: .user, content: "What did I save?"),
                LLMChatMessage(role: .assistant, content: "You saved a note."),
                LLMChatMessage(role: .tool, content: #"{"result":"note"}"#)
            ],
            systemPrompt: "Be concise.",
            context: [
                LLMContextItem(title: "Memory", content: "Saved note content.", source: "local")
            ],
            tools: [
                LLMToolDefinition(
                    name: "searchNotes",
                    description: "Search local notes.",
                    jsonSchema: #"{"type":"object"}"#,
                    isDestructive: false,
                    requiresUserApproval: false
                )
            ]
        )

        let prompt = try GGUFPromptBuilder.buildPrompt(from: request)

        #expect(prompt.contains("<|system|>\nBe concise."))
        #expect(prompt.contains("<|system|>\nContext:"))
        #expect(prompt.contains("Title: Memory"))
        #expect(prompt.contains("Source: local"))
        #expect(prompt.contains("<|tool|>\nAvailable tools:"))
        #expect(prompt.contains("Tool: searchNotes"))
        #expect(prompt.contains("<|user|>\nWhat did I save?"))
        #expect(prompt.contains("<|assistant|>\nYou saved a note."))
        #expect(prompt.contains("<|tool|>\n{\"result\":\"note\"}"))
    }

    @Test func promptBuilderThrowsInvalidRequestForEmptyPrompt() {
        let request = LLMRequest(messages: [])

        do {
            _ = try GGUFPromptBuilder.buildPrompt(from: request)
            #expect(Bool(false))
        } catch let error as LLMEngineError {
            #expect(error == .invalidRequest("Prompt is empty."))
        } catch {
            #expect(Bool(false))
        }
    }

    @Test func ggufEngineRejectsNonGGUFModelsOnLoad() async {
        let engine = GGUFEngine()
        let model = LocalLLMModel(
            id: "tiny.intent.not.gguf",
            displayName: "Tiny Intent",
            backend: .tinyIntent,
            contextLength: 512
        )

        do {
            try await engine.load(model: model, profile: .simulatorSafe)
            #expect(Bool(false))
        } catch let error as LLMEngineError {
            #expect(error == .backendUnavailable("tinyIntent"))
        } catch {
            #expect(Bool(false))
        }
    }

    @Test func ggufEngineRejectsMissingLocalFile() async {
        let engine = GGUFEngine()
        let missingURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("missing-\(UUID().uuidString).gguf")
        let model = makeGGUFModel(id: "missing.gguf", localURL: missingURL)

        do {
            try await engine.load(model: model, profile: .simulatorSafe)
            #expect(Bool(false))
        } catch let error as LLMEngineError {
            #expect(error == .modelNotFound)
        } catch {
            #expect(Bool(false))
        }
    }

    @Test func ggufEngineWithUnavailableBridgeFailsLoadClearly() async throws {
        let tempURL = try makeTemporaryGGUFFile()
        defer { try? FileManager.default.removeItem(at: tempURL.deletingLastPathComponent()) }

        let engine = GGUFEngine(nativeBridge: UnavailableGGUFNativeBridge())
        let model = makeGGUFModel(id: "unavailable.gguf", localURL: tempURL)

        do {
            try await engine.load(model: model, profile: .simulatorSafe)
            #expect(Bool(false))
        } catch let error as LLMEngineError {
            #expect(error == .backendUnavailable("GGUF native backend is not compiled."))
            #expect(error.errorDescription?.contains("GGUF native backend is not compiled") == true)
        } catch {
            #expect(Bool(false))
        }
    }

    @Test func ggufEngineIsLoadedReturnsFalseBeforeSuccessfulLoad() async {
        let engine = GGUFEngine()

        #expect(await engine.isLoaded(modelID: nil) == false)
        #expect(await engine.isLoaded(modelID: "any.gguf") == false)
    }

    @Test func ggufEngineGenerateWithoutLoadedModelFailsWithModelNotLoaded() async {
        let engine = GGUFEngine()
        let request = LLMRequest(messages: [
            LLMChatMessage(role: .user, content: "Hello")
        ])

        do {
            for try await _ in engine.generate(request) {
            }
            #expect(Bool(false))
        } catch let error as LLMEngineError {
            #expect(error == .modelNotLoaded)
        } catch {
            #expect(Bool(false))
        }
    }

    @Test func ggufEngineCapsGenerationMaxTokensToRemainingContext() async throws {
        let bridge = CapturingGGUFNativeBridge()
        let tempURL = try makeTemporaryGGUFFile()
        defer { try? FileManager.default.removeItem(at: tempURL.deletingLastPathComponent()) }

        let profile = InferenceProfile(
            name: "Tiny Context",
            contextTokens: 16,
            batchSize: 1,
            threadCount: 1,
            gpuLayerCount: 0,
            useMetal: false,
            useMemoryMapping: true,
            lowPowerMode: true
        )
        let engine = GGUFEngine(nativeBridge: bridge)
        try await engine.load(model: makeGGUFModel(id: "capped.gguf", localURL: tempURL), profile: profile)

        let request = LLMRequest(
            messages: [
                LLMChatMessage(role: .user, content: "Use this prompt to leave only a small response window.")
            ],
            sampling: LLMSamplingConfig(
                temperature: 0,
                topP: 1,
                topK: 1,
                repeatPenalty: 1,
                maxTokens: 128
            ),
            budget: InferenceBudget(
                maxPromptTokens: 128,
                maxCompletionTokens: 128,
                maxWallClockSeconds: 10,
                allowGPU: false,
                allowBackgroundExecution: false
            )
        )
        let prompt = try GGUFPromptBuilder.buildPrompt(from: request)
        let promptTokens = Int(ceil(Double(prompt.count) / 4.0))
        let expectedMaxTokens = max(0, min(profile.contextTokens, LLMEngineCapabilities.localGGUF.maximumContextTokens) - promptTokens)

        for try await _ in engine.generate(request) {
        }

        #expect(bridge.generateConfigs.first?.sampling.maxTokens == expectedMaxTokens)
    }

    @Test func engineFactoryRegistersTinyIntentEngine() async throws {
        let router = await LLMEngineFactory.makeDefaultRouter()

        #expect(await router.hasBackend(.tinyIntent))
        #expect(await router.hasBackend(.gguf) == false)

        let engine = try await router.engine(for: .tinyIntent)
        #expect(engine.id == "tiny-intent")
    }

    @Test func engineFactoryRegistersGGUFBackendWhenIncluded() async {
        let router = await LLMEngineFactory.makeDefaultRouter(includeUnavailableGGUF: true)

        #expect(await router.hasBackend(.gguf))
    }

    @Test func engineFactoryOmitsGGUFBackendWhenExcluded() async {
        let router = await LLMEngineFactory.makeDefaultRouter(includeUnavailableGGUF: false)

        #expect(await router.hasBackend(.tinyIntent))
        #expect(await router.hasBackend(.gguf) == false)
    }

    @Test func unavailableBridgeStatusIsUnavailable() async {
        let bridge = UnavailableGGUFNativeBridge()

        #expect(await bridge.status() == .unavailable)
    }

    @Test func unavailableBridgeGenerateFailsWithBackendNotCompiled() async {
        let bridge = UnavailableGGUFNativeBridge()
        let config = GGUFBridgeGenerateConfig(
            prompt: "Hello",
            sampling: GGUFBridgeSamplingConfig(
                temperature: 0,
                topP: 1,
                topK: 1,
                repeatPenalty: 1,
                seed: nil,
                maxTokens: 1,
                stopSequences: []
            )
        )

        do {
            for try await _ in bridge.generate(config: config) {
            }
            #expect(Bool(false))
        } catch let error as GGUFBridgeError {
            #expect(error == .backendNotCompiled)
        } catch {
            #expect(Bool(false))
        }
    }

    private func makeGGUFModel(id: String, localURL: URL) -> LocalLLMModel {
        LocalLLMModel(
            id: id,
            displayName: "Test GGUF",
            backend: .gguf,
            localURL: localURL,
            contextLength: 2_048
        )
    }

    private func makeTemporaryGGUFFile() throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("LumenGGUFEngineScaffoldTests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let url = directory.appendingPathComponent("test.gguf")
        try Data("fake gguf scaffold".utf8).write(to: url)
        return url
    }
}

nonisolated private final class CapturingGGUFNativeBridge: GGUFNativeBridge, @unchecked Sendable {
    private let lock = NSLock()
    private var capturedGenerateConfigs: [GGUFBridgeGenerateConfig] = []

    var generateConfigs: [GGUFBridgeGenerateConfig] {
        lock.withLock {
            capturedGenerateConfigs
        }
    }

    func status() async -> GGUFBridgeStatus {
        .loaded
    }

    func load(config: GGUFBridgeLoadConfig) async throws -> GGUFBridgeModelInfo {
        GGUFBridgeModelInfo(
            modelPath: config.modelPath,
            contextTokens: config.contextTokens,
            backendDescription: "Capturing test bridge",
            isMetalEnabled: config.useMetal
        )
    }

    func unload() async {
    }

    func generate(config: GGUFBridgeGenerateConfig) -> AsyncThrowingStream<String, Error> {
        lock.withLock {
            capturedGenerateConfigs.append(config)
        }

        return AsyncThrowingStream { continuation in
            continuation.finish()
        }
    }

    func cancel() async {
    }
}
