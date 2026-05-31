import Foundation
import Testing
@testable import Lumen

struct LLMEngineFoundationTests {
    @Test func samplingConfigClampsInvalidValues() {
        let config = LLMSamplingConfig(
            temperature: -1,
            topP: 2,
            topK: -20,
            repeatPenalty: -0.5,
            maxTokens: 0,
            stopSequences: ["", "\n"]
        )

        #expect(config.temperature == 0)
        #expect(config.topP == 1)
        #expect(config.topK == 0)
        #expect(config.repeatPenalty == 1.0)
        #expect(config.maxTokens == 1)
        #expect(config.stopSequences == ["\n"])
    }

    @Test func routerRegistersAndReturnsTinyIntentEngine() async throws {
        let router = LLMEngineRouter()
        let engine = TinyIntentEngine()

        await router.register(engine, for: .tinyIntent)

        #expect(await router.hasBackend(.tinyIntent))
        #expect(await router.availableBackends() == [.tinyIntent])

        let routed = try await router.engine(for: .tinyIntent)
        #expect(routed.id == "tiny-intent")

        let model = makeTinyIntentModel(id: "tiny.intent.router")
        let modelRouted = try await router.engine(for: model)
        #expect(modelRouted.id == "tiny-intent")
    }

    @Test func tinyIntentEngineExactModelMatchingWorks() async throws {
        let engine = TinyIntentEngine()
        let model = makeTinyIntentModel(id: "tiny.intent.loaded")

        try await engine.load(model: model, profile: .simulatorSafe)

        #expect(await engine.isLoaded(modelID: nil))
        #expect(await engine.isLoaded(modelID: "tiny.intent.loaded"))
        #expect(await engine.isLoaded(modelID: "tiny.intent.other") == false)
    }

    @Test func tinyIntentEngineStreamsStartedTokenCompleted() async throws {
        let engine = TinyIntentEngine()
        let model = makeTinyIntentModel(id: "tiny.intent.stream")
        try await engine.load(model: model, profile: .simulatorSafe)

        let requestID = UUID()
        let request = LLMRequest(
            id: requestID,
            messages: [
                LLMChatMessage(role: .user, content: "Can you look up my notes?")
            ],
            sampling: .deterministic,
            budget: .fast
        )

        var events: [LLMTokenEvent] = []
        for try await event in engine.generate(request) {
            events.append(event)
        }

        #expect(events.count == 3)
        #expect(events.first == .started(requestID: requestID))
        #expect(events.contains(.token("This looks like a search or retrieval request.")))

        guard case .completed(let summary) = events.last else {
            #expect(Bool(false))
            return
        }

        #expect(summary.requestID == requestID)
        #expect(summary.modelID == "tiny.intent.stream")
        #expect(summary.finishReason == .stop)
    }

    @Test func wrongModelLoadedErrorKeepsExpectedAndActualIDs() {
        let error = LLMEngineError.wrongModelLoaded(expected: "expected.model", actual: "actual.model")

        #expect(error == .wrongModelLoaded(expected: "expected.model", actual: "actual.model"))
        #expect(error.errorDescription?.contains("expected.model") == true)
        #expect(error.errorDescription?.contains("actual.model") == true)
    }

    private func makeTinyIntentModel(id: String) -> LocalLLMModel {
        LocalLLMModel(
            id: id,
            displayName: "Tiny Intent",
            backend: .tinyIntent,
            contextLength: 512
        )
    }
}
