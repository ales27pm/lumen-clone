import Foundation

nonisolated struct GGUFBridgeLoadConfig: Sendable, Codable, Equatable {
    let modelPath: String
    let contextTokens: Int
    let batchSize: Int
    let threadCount: Int
    let gpuLayerCount: Int
    let useMetal: Bool
    let useMemoryMapping: Bool
}

nonisolated struct GGUFBridgeSamplingConfig: Sendable, Codable, Equatable {
    let temperature: Double
    let topP: Double
    let topK: Int
    let repeatPenalty: Double
    let seed: UInt64?
    let maxTokens: Int
    let stopSequences: [String]
}

nonisolated struct GGUFBridgeGenerateConfig: Sendable, Codable, Equatable {
    let prompt: String
    let sampling: GGUFBridgeSamplingConfig
}

nonisolated struct GGUFBridgeModelInfo: Sendable, Codable, Equatable {
    let modelPath: String
    let contextTokens: Int
    let backendDescription: String
    let isMetalEnabled: Bool
}

nonisolated enum GGUFBridgeStatus: String, Sendable, Codable, Equatable {
    case unavailable
    case unloaded
    case loaded
    case generating
}
