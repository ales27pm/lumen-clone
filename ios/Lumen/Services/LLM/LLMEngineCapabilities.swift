import Foundation

nonisolated struct LLMEngineCapabilities: Sendable, Codable, Equatable {
    let supportsStreaming: Bool
    let supportsToolCalling: Bool
    let supportsStructuredOutput: Bool
    let supportsEmbeddings: Bool
    let supportsVision: Bool
    let supportsMetalAcceleration: Bool
    let supportsBackgroundExecution: Bool
    let maximumContextTokens: Int
    let maximumOutputTokens: Int

    init(
        supportsStreaming: Bool,
        supportsToolCalling: Bool,
        supportsStructuredOutput: Bool,
        supportsEmbeddings: Bool,
        supportsVision: Bool,
        supportsMetalAcceleration: Bool,
        supportsBackgroundExecution: Bool,
        maximumContextTokens: Int,
        maximumOutputTokens: Int
    ) {
        self.supportsStreaming = supportsStreaming
        self.supportsToolCalling = supportsToolCalling
        self.supportsStructuredOutput = supportsStructuredOutput
        self.supportsEmbeddings = supportsEmbeddings
        self.supportsVision = supportsVision
        self.supportsMetalAcceleration = supportsMetalAcceleration
        self.supportsBackgroundExecution = supportsBackgroundExecution
        self.maximumContextTokens = max(1, maximumContextTokens)
        self.maximumOutputTokens = max(1, maximumOutputTokens)
    }

    static let localGGUF = LLMEngineCapabilities(
        supportsStreaming: true,
        supportsToolCalling: true,
        supportsStructuredOutput: true,
        supportsEmbeddings: false,
        supportsVision: false,
        supportsMetalAcceleration: true,
        supportsBackgroundExecution: false,
        maximumContextTokens: 8_192,
        maximumOutputTokens: 2_048
    )

    static let coreML = LLMEngineCapabilities(
        supportsStreaming: true,
        supportsToolCalling: false,
        supportsStructuredOutput: true,
        supportsEmbeddings: false,
        supportsVision: false,
        supportsMetalAcceleration: true,
        supportsBackgroundExecution: true,
        maximumContextTokens: 4_096,
        maximumOutputTokens: 1_024
    )

    static let tinyIntent = LLMEngineCapabilities(
        supportsStreaming: true,
        supportsToolCalling: false,
        supportsStructuredOutput: false,
        supportsEmbeddings: false,
        supportsVision: false,
        supportsMetalAcceleration: false,
        supportsBackgroundExecution: true,
        maximumContextTokens: 512,
        maximumOutputTokens: 128
    )

    static let mockTesting = LLMEngineCapabilities(
        supportsStreaming: true,
        supportsToolCalling: true,
        supportsStructuredOutput: true,
        supportsEmbeddings: true,
        supportsVision: true,
        supportsMetalAcceleration: false,
        supportsBackgroundExecution: true,
        maximumContextTokens: 16_384,
        maximumOutputTokens: 4_096
    )
}
