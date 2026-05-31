import Foundation

protocol LLMEngine: Sendable {
    var id: String { get }
    var displayName: String { get }
    var capabilities: LLMEngineCapabilities { get }

    func load(model: LocalLLMModel, profile: InferenceProfile) async throws
    func unload() async
    func isLoaded(modelID: String?) async -> Bool
    func generate(_ request: LLMRequest) -> AsyncThrowingStream<LLMTokenEvent, Error>
    func cancelCurrentGeneration() async
}
